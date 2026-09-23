# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Where the weights are, and proof that they are the published bytes.

The pinned revisions below are the public Manchego v2.1 repositories on the Hugging Face Hub (tag `v2.1` in each). The
SHA-256 of every weight file is the Hub's own LFS object id for that file at that revision. At start-up the server hashes
the weight files it loaded and reports whether they match a pin. Nothing here opens a network connection: resolving a
hub id reads the local cache only (`manchego-serve-download` fills it once, at setup time).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PINS = {
    "oraculumai/Manchego": {
        "revision": "77403228b7dfdf823af99a5f562bdcf80b708d4c", "tag": "v2.1", "format": "bf16 Transformers", "backend": "torch",
        "files": {
            "model.safetensors-00001-of-00002.safetensors": "1d5df0ff579a8de08c078df2423e3111185e126e33614fcc981254687389efe3",
            "model.safetensors-00002-of-00002.safetensors": "b12ea4894982c0f1e2982eb6841e1a307d5a3b2bcb51987d6844bcd102e5aa0a",
        },
    },
    "oraculumai/Manchego-MLX-8bit": {
        "revision": "79e55e2d0c4446abfe0d55d829e8854de9177c1b", "tag": "v2.1", "format": "MLX 8-bit", "backend": "mlx",
        "files": {"model.safetensors": "ffa9e0c31db78604a7d32fd8bfdfc5c0938c97d53afc10d1f9c0a7f64ed5ef43"},
    },
    "oraculumai/Manchego-MLX-4bit": {
        "revision": "184016ce35c3a400880634352360b39cfdbb901e", "tag": "v2.1", "format": "MLX 4-bit", "backend": "mlx",
        "files": {"model.safetensors": "301da6413d517ef33d62fbe8d996e9e48d1750eaae9ed6297ceb7fb3a54e09e2"},
    },
}
DEFAULT_REPO = {"torch": "oraculumai/Manchego", "mlx": "oraculumai/Manchego-MLX-8bit"}


class WeightsNotFound(RuntimeError):
    pass


def resolve(model: str, revision: str | None) -> tuple[str, str | None]:
    """(local directory, revision). A directory is used as is. A hub id is looked up in the local cache ONLY."""
    if Path(model).is_dir():
        return str(Path(model).resolve()), revision
    if revision is None and model in PINS:
        revision = PINS[model]["revision"]
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(model, revision=revision, local_files_only=True)
    except Exception as e:  # LocalEntryNotFoundError and friends
        raise WeightsNotFound(f"{model}@{revision} is not in the local Hugging Face cache and this server never downloads. "
                              f"Fetch it once with: manchego-serve-download --repo {model} --revision {revision}") from e
    return path, revision


def weight_files(model_dir: str) -> list[Path]:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise WeightsNotFound(f"no *.safetensors files in {model_dir}")
    return files


def sha256_file(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(os.path.realpath(path), "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def identify(model_dir: str) -> dict:
    """SHA-256 of every weight file, a combined digest, and the pin they match (if any)."""
    files = {p.name: sha256_file(p) for p in weight_files(model_dir)}
    combined = hashlib.sha256("".join(f"{files[n]}  {n}\n" for n in sorted(files)).encode()).hexdigest()
    match = next(({"repo": repo, "revision": pin["revision"], "tag": pin["tag"]} for repo, pin in PINS.items() if pin["files"] == files), None)
    return {"files": files, "weights_sha256": combined, "weights_match_pin": match}
