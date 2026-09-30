# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""What the model folder says about how to serve it: `manchego_config.json`, next to the weights (0.2.0).

Fields read (everything else in the file is documentation and is ignored):

  contract      "auto" (the default: the short prompt up to 26 options, the state-first prompt beyond; Manchego v2.1) or
                "semif" (the SemIf prompt up to 16 options, the state-first prompt for the rest; for models trained under
                the SemIf contract, such as Manchego v3). contract.py has both policies.
  served_name   the name this server gives the model in responses and in /v1/models (default "manchego-2.1").
  release_date  reported by /v1/models with `served_name` (default: v2.1's).

The published Manchego v2.1 folder has a manchego_config.json with no `contract` field, and a folder may have no file at
all: both mean "auto", so v2.1 is served exactly as by 0.1.x. A file that is not JSON, or names an unknown contract, stops
the server at start-up rather than guessing. Nothing here imports an ML library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .contract import AUTO, POLICIES

FILE = "manchego_config.json"
DEFAULT_NAME = "manchego-2.1"


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

    def describe(self) -> dict:
        return {"file": self.file, "sha256": self.sha256, "contract": self.contract,
                "contract_source": FILE if self.contract_declared else f"default ({FILE} names no contract)" if self.file
                else f"default (no {FILE})", "served_name": self.served_name or DEFAULT_NAME}


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
    return ModelConfig(contract=contract, contract_declared="contract" in data, served_name=data.get("served_name"),
                       release_date=data.get("release_date"), file=str(p), sha256=hashlib.sha256(raw).hexdigest(), **text)
