# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""The served temperature: one temperature per question type (choice, noul, score), applied to the option-code logits
before the softmax, p = softmax(z / T[type]).

A temperature never changes which option is chosen; it changes how spread out the probabilities are, and so `noul`,
`confidence`, the `probabilities` and a score's expected level.

  default   the map fitted for Manchego v2.1 on the project's own held-out development data. It ships with this package as
            temperature_map_v2.1.json, and its SHA-256 is pinned below. It applies only to the published v2.1 builds
            listed in the file, and only under contract "auto" (0.2.0): every other model gets T = 1.0.
  off       T = 1.0 for every type: the v0.1.0 policy, byte for byte.
  <path>    a JSON file: schema 1 (the default map's) or schema 2 (0.2.0).

Schema 2 ("manchego-temperature-map/2") is a typed map bound to ONE set of weights:

    {"schema": "manchego-temperature-map/2",
     "temperatures": {"choice": T, "noul": T, "score": T},     # each in [0.05, 20]; below 1 sharpens, above 1 flattens
     "model_sha256": "<weights_sha256 of the weights it was fitted on (weights.identify)>",
     "fitted_on": "<text: which data, which split, which prompt contract>",
     "rule": "<text: the objective and the decision rule that chose these values>",
     "contract": "auto" | "semif",                             # optional: the served contract it was fitted under
     "model": "<text>", ...}                                   # optional; everything else is kept as provenance

A schema-2 map is refused (the server does not start) when the loaded weights' SHA-256 is not `model_sha256`, when the
weights were not hashed (--no-hash), or when its `contract` is not the served one. It is never applied "as asked".

The server's default (0.2.0, `resolve`): a map applies only to weights it is bound to by hash. With --no-hash the
packaged v2.1 map applies only when a listed v2.1 build is declared (hub id or revision), as in 0.1.2; a bare
directory with nothing declared gets T = 1.0.

Nothing here imports an ML library.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

TYPES = ("choice", "noul", "score")
SCHEMA = "manchego-temperature-map/1"
SCHEMA_V2 = "manchego-temperature-map/2"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
DEFAULT_FILE = Path(__file__).with_name("temperature_map_v2.1.json")
DEFAULT_SHA256 = "2458d21771bbddc9c23ae0e63c880f2ebe5a6aa997e8c9de6b5e67b7210cd07e"
OFF_SPECS = ("off", "none", "t1", "1", "1.0")
T_MIN, T_MAX = 0.05, 20.0


class TemperatureMapError(ValueError):
    """A map that cannot be used: bad schema, bad values, or a packaged default that is not the published file."""


@dataclass(frozen=True)
class TemperatureMap:
    temperatures: dict
    source: str                       # "default" | "file" | "off"
    file: str | None = None
    sha256: str | None = None
    meta: dict = field(default_factory=dict)
    binding: str | None = None        # how the map relates to the loaded weights
    note: str | None = None           # why the default map is not in force, when it is not
    schema: str = SCHEMA
    model_sha256: str | None = None   # schema 2: the weights it is bound to
    contract: str | None = None       # schema 2: the served contract it was fitted under (optional)

    def T(self, kind: str) -> float:
        return self.temperatures.get(kind, 1.0)

    @property
    def is_identity(self) -> bool:
        return all(self.temperatures[t] == 1.0 for t in TYPES)

    def wire_temperature(self) -> Any:
        """The `temperature` field of the `manchego` block: 1.0 when every type is at 1.0 (as in v0.1.0), else the map."""
        return 1.0 if self.is_identity else {t: self.temperatures[t] for t in TYPES}

    def describe(self) -> dict:
        d = {"source": self.source, "temperatures": {t: self.temperatures[t] for t in TYPES}, "file": self.file,
             "sha256": self.sha256, "fitted_for": self.meta.get("model"), "binding": self.binding}
        if self.schema == SCHEMA_V2:
            d.update(schema=self.schema, model_sha256=self.model_sha256, contract=self.contract)
        if self.note:
            d["note"] = self.note
        return d

    def provenance(self) -> dict | None:
        """The file's own record of what it was fitted on and what it changes (for /healthz)."""
        if not self.meta:
            return None
        keys = ("model", "applies_to", "fitted_on", "held_out_result", "known_costs", "disclosure")
        if self.schema == SCHEMA_V2:
            keys = ("schema", "model", "model_sha256", "contract", "fitted_on", "rule") + keys[1:]
        return {k: self.meta[k] for k in dict.fromkeys(keys) if k in self.meta}


OFF = TemperatureMap({t: 1.0 for t in TYPES}, "off", binding="none: T = 1.0 for every type (the v0.1.0 policy)")


def _temperatures(data: Any, where: str) -> dict:
    if not isinstance(data, dict) or data.get("schema") not in (SCHEMA, SCHEMA_V2):
        raise TemperatureMapError(f"{where}: not a temperature map (expected \"schema\": \"{SCHEMA}\" or \"{SCHEMA_V2}\")")
    temps = data.get("temperatures")
    if not isinstance(temps, dict) or set(temps) != set(TYPES):
        raise TemperatureMapError(f"{where}: \"temperatures\" must give exactly {', '.join(TYPES)}")
    out = {}
    for t in TYPES:
        v = temps[t]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not T_MIN <= v <= T_MAX:
            raise TemperatureMapError(f"{where}: temperature for {t} must be a number in [{T_MIN}, {T_MAX}], got {v!r}")
        out[t] = float(v)
    return out


def load(spec: str | None = None) -> TemperatureMap:
    """`default` (also when empty or None), `off`, or a path to a JSON map."""
    s = (spec or "default").strip()
    if s.lower() in OFF_SPECS:
        return OFF
    if s.lower() == "default":
        raw = DEFAULT_FILE.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != DEFAULT_SHA256:
            raise TemperatureMapError(f"the packaged {DEFAULT_FILE.name} has sha256 {sha}, not the published {DEFAULT_SHA256}; "
                                      "reinstall the package, or pass --temperature-map off / a file explicitly")
        data = json.loads(raw)
        return TemperatureMap(_temperatures(data, DEFAULT_FILE.name), "default", DEFAULT_FILE.name, sha, data)
    p = Path(s)
    if not p.is_file():
        raise TemperatureMapError(f"temperature map {s!r}: expected 'default', 'off' or a JSON file")
    raw = p.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise TemperatureMapError(f"{p}: not JSON ({e})") from None
    temps = _temperatures(data, str(p))
    if data["schema"] == SCHEMA_V2:
        return TemperatureMap(temps, "file", str(p), hashlib.sha256(raw).hexdigest(), data, schema=SCHEMA_V2,
                              **_v2_binding(data, str(p)))
    return TemperatureMap(temps, "file", str(p), hashlib.sha256(raw).hexdigest(), data)


def _v2_binding(data: dict, where: str) -> dict:
    """The fields a schema-2 map must carry: what it is bound to and how it was made."""
    from .contract import POLICIES
    sha = data.get("model_sha256")
    if not isinstance(sha, str) or not SHA256_HEX.fullmatch(sha):
        raise TemperatureMapError(f"{where}: \"model_sha256\" must be the 64-hex-digit weights_sha256 of the weights it was fitted on")
    for key in ("fitted_on", "rule"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise TemperatureMapError(f"{where}: \"{key}\" must be a nonempty text (schema 2 records how the map was made)")
    contract = data.get("contract")
    if contract is not None and contract not in POLICIES:
        raise TemperatureMapError(f"{where}: \"contract\" must be one of {', '.join(POLICIES)}, got {contract!r}")
    return {"model_sha256": sha, "contract": contract}


def bind(tmap: TemperatureMap, weights_sha256: str | None, log: Callable[[str], None] = print,
         declared_repo: str | None = None, declared_revision: str | None = None) -> TemperatureMap:
    """Relate a map to the loaded weights (the `weights_sha256` of weights.identify, or None when they were not hashed).

    A map is fitted for one model. When the weights were hashed and are not a build the map lists, the default map
    is not applied: T = 1.0 for every type, with a warning. An explicit file is applied, and the warning says the weights
    are not a listed build.

    Unhashed weights (--no-hash) cannot be checked, so the declaration decides: when a hub id or revision is declared
    (`declared_repo`, `declared_revision`) and it is not a build the default map lists (the MLX 4-bit build, for
    example), the default map is not applied (T = 1.0, with a warning). With a listed declaration, or with nothing
    declared (a bare directory), the map is kept and reported as not checked.
    """
    if tmap.source == "off":
        return tmap
    if tmap.schema == SCHEMA_V2:
        return _bind_v2(tmap, weights_sha256)
    listed = {a.get("weights_sha256"): a for a in tmap.meta.get("applies_to") or [] if isinstance(a, dict)}
    if weights_sha256 is None:
        if tmap.source == "default" and (declared_repo or declared_revision):
            if not any((not declared_repo or a.get("repo") == declared_repo)
                       and (not declared_revision or a.get("revision") == declared_revision) for a in listed.values()):
                why = (f"the weights were not hashed (--no-hash) and are declared as {declared_repo or '(a directory)'}"
                       f"@{declared_revision or '(no revision)'}, which is not a build the default map (fitted for "
                       f"{tmap.meta.get('model')}) lists, so T = 1.0 for every type")
                log(f"warning: {why}")
                return replace(OFF, note=why)
            return replace(tmap, binding="not checked: the weights were not hashed (--no-hash); declared as a build this map lists")
        return replace(tmap, binding="not checked: the weights were not hashed (--no-hash)")
    if weights_sha256 in listed:
        a = listed[weights_sha256]
        return replace(tmap, binding=f"a build this map lists: {a.get('repo')}@{a.get('revision')}")
    if tmap.source == "default":
        why = (f"the default map is fitted for {tmap.meta.get('model')} and lists {len(listed)} published build(s); "
               f"the loaded weights ({weights_sha256}) are not one of them, so T = 1.0 for every type")
        log(f"warning: {why}")
        return replace(OFF, note=why)
    log(f"warning: temperature map {tmap.file} does not list the loaded weights ({weights_sha256}); applying it as asked")
    return replace(tmap, binding="NOT a build this map lists; applied because it was passed explicitly")


def _bind_v2(tmap: TemperatureMap, weights_sha256: str | None) -> TemperatureMap:
    """Schema 2: applied only to the weights it is bound to. Anything else refuses to start."""
    if weights_sha256 is None:
        raise TemperatureMapError(f"temperature map {tmap.file} is bound to weights {tmap.model_sha256}, and the weights were not "
                                  "hashed (--no-hash), so the binding cannot be checked. Hash the weights, or pass --temperature-map off")
    if weights_sha256 != tmap.model_sha256:
        raise TemperatureMapError(f"temperature map {tmap.file} is bound to weights {tmap.model_sha256}; the loaded weights are "
                                  f"{weights_sha256}. A temperature map is fitted for one model: refusing to apply it to another")
    return replace(tmap, binding=f"bound: model_sha256 is the loaded weights ({weights_sha256})")


def resolve(tmap: TemperatureMap, weights_sha256: str | None, log: Callable[[str], None] = print,
            declared_repo: str | None = None, declared_revision: str | None = None, contract: str = "auto") -> TemperatureMap:
    """The server's temperature for the loaded model (0.2.0): `bind`, plus two rules that make T = 1.0 the default for any
    model the map is not bound to by hash.

      * The packaged v2.1 map was fitted under contract "auto"; under any other contract it is not applied (T = 1.0).
      * With --no-hash and nothing declared (a bare directory, no hub id, no revision) the packaged map is not applied
        (T = 1.0): the weights could be any model. 0.1.2 kept it, reported as not checked.
      * A schema-2 map must match the served contract when it names one, and the weights by hash (`bind`), or the server
        does not start.
    An explicit schema-1 file keeps 0.1.2's behaviour (applied, with a warning when the weights are not a listed build).
    """
    if tmap.source == "off":
        return tmap
    if tmap.schema == SCHEMA_V2:
        if tmap.contract is not None and tmap.contract != contract:
            raise TemperatureMapError(f"temperature map {tmap.file} was fitted under contract {tmap.contract!r}; this server "
                                      f"serves contract {contract!r}. Refusing to apply it")
        return bind(tmap, weights_sha256, log=log)
    if tmap.source == "default" and contract != "auto":
        why = (f"the default map ({tmap.meta.get('model')}) was fitted under contract auto; this server serves contract "
               f"{contract!r}, so T = 1.0 for every type")
        log(f"warning: {why}")
        return replace(OFF, note=why)
    if tmap.source == "default" and weights_sha256 is None and not (declared_repo or declared_revision):
        why = ("the weights were not hashed (--no-hash) and no build is declared (--model is a directory, no --revision), so "
               "the default map cannot be bound to them: T = 1.0 for every type")
        log(f"warning: {why}")
        return replace(OFF, note=why)
    return bind(tmap, weights_sha256, log=log, declared_repo=declared_repo, declared_revision=declared_revision)
