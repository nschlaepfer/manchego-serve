# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""What the model folder says about how to serve it: `manchego_config.json`, next to the weights (0.2.0).

Fields read (everything else in the file is documentation and is ignored):

  contract      "auto" (the default: the short prompt up to 26 options, the state-first prompt beyond; Manchego v2.1) or
                "semif" (the SemIf prompt up to 16 options, the state-first prompt for the rest; for models trained under
                the SemIf contract, such as Manchego v3). contract.py has both policies.
  served_name   the name this server gives the model in responses and in /v1/models (default "manchego-2.1").
  release_date  reported by /v1/models with `served_name` (default: v2.1's).
  serving       the model's default fast path (0.2.0), e.g. {"cuda_graphs": true, "fast_host": true}; optional
                "graph_buckets": [256, 512, 1024, 2048]. Applied only by the torch backend on CUDA (elsewhere it is not
                applied, and /healthz says so); every piece can be overridden on the command line or in the environment
                (backends/fast_path.py, `settle`). The linear-attention kernels are not a serving default: they are chosen
                by --gdn-kernels only.

The published Manchego v2.1 folder has a manchego_config.json with no `contract` and no `serving` field, and a folder may
have no file at all: both mean "auto" and no fast path, so v2.1 is served exactly as by 0.1.x. A file that is not JSON,
names an unknown contract, or has a `serving` field this server cannot read stops the server at start-up rather than
guessing. Nothing here imports an ML library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .contract import AUTO, POLICIES

FILE = "manchego_config.json"
DEFAULT_NAME = "manchego-2.1"
SERVING_KEYS = ("cuda_graphs", "fast_host", "graph_buckets")


class ModelConfigError(ValueError):
    """A manchego_config.json the server will not guess about."""


@dataclass(frozen=True)
class ModelConfig:
    contract: str = AUTO
    contract_declared: bool = False     # the file names a contract (else the default "auto")
    served_name: str | None = None
    release_date: str | None = None
    model: str | None = None            # the file's own "model" and "version", for the /v1/models description
    version: str | None = None
    file: str | None = None
    sha256: str | None = None
    serving: dict | None = None         # the `serving` field (None when the file has none)

    def describe(self) -> dict:
        d = {"file": self.file, "sha256": self.sha256, "contract": self.contract,
             "contract_source": FILE if self.contract_declared else f"default ({FILE} names no contract)" if self.file
             else f"default (no {FILE})", "served_name": self.served_name or DEFAULT_NAME}
        if self.serving is not None:        # only when declared, so a folder without it reports what 0.1.x reported
            d["serving"] = dict(self.serving)
        return d


def _serving(value, where) -> dict:
    """The `serving` field, checked: an object with only SERVING_KEYS; booleans; distinct positive token counts."""
    if not isinstance(value, dict):
        raise ModelConfigError(f"{where}: serving must be a JSON object, e.g. {{\"cuda_graphs\": true, \"fast_host\": true}}")
    unknown = sorted(set(value) - set(SERVING_KEYS))
    if unknown:
        raise ModelConfigError(f"{where}: serving has {', '.join(map(repr, unknown))}; this server reads only "
                               f"{', '.join(SERVING_KEYS)} (the linear-attention kernels are chosen by --gdn-kernels only)")
    for key in ("cuda_graphs", "fast_host"):
        if key in value and not isinstance(value[key], bool):
            raise ModelConfigError(f"{where}: serving.{key} must be true or false")
    if "graph_buckets" in value:
        b = value["graph_buckets"]
        if (not isinstance(b, list) or not b or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in b)
                or len(set(b)) != len(b)):
            raise ModelConfigError(f"{where}: serving.graph_buckets must be a list of distinct positive token counts")
    return dict(value)


def read(model_dir: str | Path) -> ModelConfig:
    p = Path(model_dir) / FILE
    if not p.is_file():
        return ModelConfig()
    raw = p.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ModelConfigError(f"{p}: not JSON ({e})") from None
    if not isinstance(data, dict):
        raise ModelConfigError(f"{p}: expected a JSON object")
    contract = data.get("contract", AUTO)
    if contract not in POLICIES:
        raise ModelConfigError(f"{p}: contract {contract!r} is not one this server has ({', '.join(POLICIES)})")
    for key in ("served_name", "release_date"):
        if key in data and not (isinstance(data[key], str) and data[key].strip()):
            raise ModelConfigError(f"{p}: {key} must be a nonempty string")
    text = {k: data.get(k) if isinstance(data.get(k), str) else None for k in ("model", "version")}
    serving = _serving(data["serving"], p) if "serving" in data else None
    return ModelConfig(contract=contract, contract_declared="contract" in data, served_name=data.get("served_name"),
                       release_date=data.get("release_date"), file=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                       serving=serving, **text)
