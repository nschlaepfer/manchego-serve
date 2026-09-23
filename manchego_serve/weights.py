# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Where the weights are, and proof that they are the published bytes.

The pinned revisions below are the public Manchego v2.1 repositories on the Hugging Face Hub (tag `v2.1` in each). The
SHA-256 of every weight file is the Hub's own LFS object id for that file at that revision. The support files that shape
the prompt and the model (chat template, tokenizer, configs) are pinned too: their SHA-256 were computed from copies whose
git blob ids (or LFS ids) equal the Hub's at that revision. At start-up the server hashes the files it loads and reports
whether they match a pin. Nothing here opens a network connection: resolving a hub id reads the local cache only
(`manchego-serve-download` fills it once, at setup time).
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
        "support": {
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
            "model.safetensors.index.json": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        },
    },
    "oraculumai/Manchego-MLX-8bit": {
        "revision": "79e55e2d0c4446abfe0d55d829e8854de9177c1b", "tag": "v2.1", "format": "MLX 8-bit", "backend": "mlx",
        "files": {"model.safetensors": "ffa9e0c31db78604a7d32fd8bfdfc5c0938c97d53afc10d1f9c0a7f64ed5ef43"},
        "support": {
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "18c3c39e696f9311fc6665cf8e0357f31d6db68b888df8f23acee86550c59a2c",
            "model.safetensors.index.json": "06ad7fcba6631617927dd87d50e03a47d2201fe8a94feba65850e87333629e22",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679",
        },
    },
    "oraculumai/Manchego-MLX-4bit": {
        "revision": "184016ce35c3a400880634352360b39cfdbb901e", "tag": "v2.1", "format": "MLX 4-bit", "backend": "mlx",
        "files": {"model.safetensors": "301da6413d517ef33d62fbe8d996e9e48d1750eaae9ed6297ceb7fb3a54e09e2"},
        "support": {
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "8591d683d8d132234766399897415b9427967c05e77c1c169a40de3e81177e62",
            "model.safetensors.index.json": "2e97890b24a47b9c290489efeb2b68b7e4dd4101360de80122165ebae4e0d6cf",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679",
        },
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


SUPPORT_FILES = ("chat_template.jinja", "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")


def identify(model_dir: str) -> dict:
    """SHA-256 of every weight file, a combined digest, and the pin they match (if any); SHA-256 of the support files
    (None when absent) and whether they are the pinned ones of that same revision."""
    files = {p.name: sha256_file(p) for p in weight_files(model_dir)}
    combined = hashlib.sha256("".join(f"{files[n]}  {n}\n" for n in sorted(files)).encode()).hexdigest()
    match = next(({"repo": repo, "revision": pin["revision"], "tag": pin["tag"]} for repo, pin in PINS.items() if pin["files"] == files), None)
    support = {n: (sha256_file(Path(model_dir) / n) if (Path(model_dir) / n).is_file() else None) for n in SUPPORT_FILES}
    support_ok = bool(match) and all(support.get(n) == h for n, h in PINS[match["repo"]]["support"].items())
    return {"files": files, "weights_sha256": combined, "weights_match_pin": match, "support_files": support,
            "support_files_match_pin": support_ok}
