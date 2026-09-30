# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Where the weights are, and proof that they are the published bytes.

Two Manchego versions are pinned, each in three repositories on the Hugging Face Hub (bf16 Transformers, MLX 8-bit,
MLX 4-bit):

  v2.1  tag `v2.1`. The SHA-256 of every weight file is the Hub's own LFS object id for that file at that revision; the
        support files that shape the prompt and the model (chat template, tokenizer, configs) were hashed from copies
        whose git blob ids (or LFS ids) equal the Hub's at that revision.
  v3    the default (tag `v3`, uploaded 2026-09-30). Every SHA-256 was computed from the release builds that were uploaded,
        and each upload was verified against them (every LFS object id equal). The revisions (V3_REVISION,
        V3_MLX8_REVISION, V3_MLX4_REVISION) are the commits of that upload.

A version name (`v3`, `v2.1`) may stand for a revision: it means the commit this package pins, never the Hub's tag of
that name. At start-up the server hashes the files it loads and reports whether they match a pinned revision. Nothing
here opens a network connection: resolving a hub id reads the local cache only (`manchego-serve-download` fills it
once, at setup time).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

# Manchego v3's commits on the Hub. PLACEHOLDERS: each is replaced by the full 40-character sha of the commit that holds
# the v3 files, after the upload (the release plan's commit A), and before this package is tagged.
V3_REVISION = "f82e029d0ad4d1bdd1ca12f5f7b548b84fabc9a9"       # oraculumai/Manchego, tag v3 (2026-09-30)
V3_MLX8_REVISION = "4ebdc0dd50481cfa9e40f83f05571ee59e53041d"  # oraculumai/Manchego-MLX-8bit, tag v3
V3_MLX4_REVISION = "0c17e076e8e358e41a9f8332dca062e4942ace0b"  # oraculumai/Manchego-MLX-4bit, tag v3

FULL_SHA = re.compile(r"[0-9a-f]{40}")
BF16_SUPPORT = {   # identical in v2.1 and v3
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "model.safetensors.index.json": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
}

PINS = {   # Manchego v2.1 (the name 0.1.x used for them)
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
V21_PINS = PINS
V3_PINS = {
    "oraculumai/Manchego": {
        "revision": V3_REVISION, "tag": "v3", "format": "bf16 Transformers", "backend": "torch",
        "files": {
            "model.safetensors-00001-of-00002.safetensors": "036f8c81ba3436f89da99dd149e04b4640c7560df3126a9c31668e30db86b89d",
            "model.safetensors-00002-of-00002.safetensors": "44fb326ecba28d864b762d5b70cd08746f7d5b07fade1e92885b7c2600428af6",
        },
        "support": dict(BF16_SUPPORT),
    },
    "oraculumai/Manchego-MLX-8bit": {
        "revision": V3_MLX8_REVISION, "tag": "v3", "format": "MLX 8-bit", "backend": "mlx",
        "files": {"model.safetensors": "02546297da8c59e9545d1600cb25ce5b6f3237104a19f9dd9674b961a9248e85"},
        "support": dict(PINS["oraculumai/Manchego-MLX-8bit"]["support"]),        # identical to v2.1's
    },
    "oraculumai/Manchego-MLX-4bit": {
        "revision": V3_MLX4_REVISION, "tag": "v3", "format": "MLX 4-bit", "backend": "mlx",
        "files": {"model.safetensors": "a6e8ea1e4d549ac951252d20684ff12884cd2d7b0c417aaabc6acffca98428be"},
        "support": dict(PINS["oraculumai/Manchego-MLX-4bit"]["support"]),        # identical to v2.1's
    },
}
RELEASES = {"v3": V3_PINS, "v2.1": V21_PINS}          # newest first
VERSIONS = tuple(RELEASES)
DEFAULT_REPO = {"torch": "oraculumai/Manchego", "mlx": "oraculumai/Manchego-MLX-8bit"}


class WeightsNotFound(RuntimeError):
    pass


class RevisionError(ValueError):
    """A version name this package cannot turn into a pinned commit."""


def is_pinned(revision: str | None) -> bool:
    """A full commit sha (a placeholder is not)."""
    return bool(revision) and bool(FULL_SHA.fullmatch(revision))


def published_pins():
    """(version, repo, pin) for every pin whose revision is a full commit sha, newest version first."""
    return [(v, repo, pin) for v, pins in RELEASES.items() for repo, pin in pins.items() if is_pinned(pin["revision"])]


def default_version(repo: str) -> str | None:
    """The newest version this package pins a revision of `repo` for (v3 once its revision is filled in, else v2.1)."""
    return next((v for v, r, _ in published_pins() if r == repo), None)


def pin_for(repo: str, revision: str | None) -> tuple[str, dict] | tuple[None, None]:
    """(version, pin) when `revision` is a pinned revision of `repo`."""
    return next(((v, pin) for v, r, pin in published_pins() if r == repo and pin["revision"] == revision), (None, None))


def pinned_revision(repo: str, version: str | None = None) -> str:
    """The commit this package pins for `repo` at `version` (None: the default version)."""
    if version is None:
        version = default_version(repo)
        if version is None:
            raise RevisionError(f"this package pins no revision of {repo}; pass a full commit sha")
    pin = RELEASES.get(version, {}).get(repo)
    if pin is None:
        raise RevisionError(f"{version!r} is not a Manchego version this package pins for {repo} "
                            f"(known: {', '.join(v for v in VERSIONS if repo in RELEASES[v]) or 'none'})")
    if not is_pinned(pin["revision"]):
        raise RevisionError(f"Manchego {version}'s revision of {repo} is not pinned in this build of manchego-serve yet "
                            f"(weights.py holds the placeholder {pin['revision']!r}, filled after the Hub upload); pass the "
                            f"full commit sha, or v2.1")
    return pin["revision"]


def named_revision(repo: str, revision: str | None) -> str | None:
    """A version name (v3, v2.1) as the commit this package pins for `repo`; anything else unchanged."""
    return pinned_revision(repo, revision) if revision in RELEASES else revision


def resolve(model: str, revision: str | None, default_repo: str | None = None) -> tuple[str, str | None]:
    """(local directory, revision). A directory is used as is (a version name as `revision` is read for `default_repo`).
    A hub id is looked up in the local cache ONLY, at `revision` (a sha or a version name; None: the default version's
    pinned commit)."""
    if Path(model).is_dir():
        return str(Path(model).resolve()), named_revision(default_repo, revision) if default_repo else revision
    revision = named_revision(model, revision or None)
    if revision is None and any(model in pins for pins in RELEASES.values()):
        revision = pinned_revision(model)
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
    """SHA-256 of every weight file, a combined digest, and the pinned revision they match (if any: a v3 build matches
    none until its revision is pinned); SHA-256 of the support files (None when absent) and whether they are the pinned
    ones of that same revision."""
    files = {p.name: sha256_file(p) for p in weight_files(model_dir)}
    combined = hashlib.sha256("".join(f"{files[n]}  {n}\n" for n in sorted(files)).encode()).hexdigest()
    found = next(((repo, pin) for _, repo, pin in published_pins() if pin["files"] == files), None)
    match = {"repo": found[0], "revision": found[1]["revision"], "tag": found[1]["tag"]} if found else None
    support = {n: (sha256_file(Path(model_dir) / n) if (Path(model_dir) / n).is_file() else None) for n in SUPPORT_FILES}
    support_ok = bool(found) and all(support.get(n) == h for n, h in found[1]["support"].items())
    return {"files": files, "weights_sha256": combined, "weights_match_pin": match, "support_files": support,
            "support_files_match_pin": support_ok}
