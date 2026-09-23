"""Pins, weight identification, the download check and offline mode. No network, no weights."""
import hashlib
import os

import pytest

from manchego_serve import download, weights
from manchego_serve.weights import PINS, identify

EXPECTED_DIGESTS = {  # sha256 over "<file sha256>  <file name>\n" lines, sorted by name (see weights.identify)
    "oraculumai/Manchego": "1130745e2a9506ece5a35a1d3da5ad1ce17e8a05052d8fc45d1971986bd8de61",
    "oraculumai/Manchego-MLX-8bit": "6b0cb89600ffcc0941c557ac60a8445709d5f996a6baec24d9a92213ddccf84b",
    "oraculumai/Manchego-MLX-4bit": "0fb734ac0221f4fcc314daae3ac2391f1eb41dc2f381ef9c2fe319907e49a98a",
}


def test_pins_are_full_shas_and_digests_are_stable():
    for repo, pin in PINS.items():
        assert len(pin["revision"]) == 40 and int(pin["revision"], 16) >= 0
        assert all(len(h) == 64 for h in pin["files"].values())
        f = pin["files"]
        assert hashlib.sha256("".join(f"{f[n]}  {n}\n" for n in sorted(f)).encode()).hexdigest() == EXPECTED_DIGESTS[repo]


def test_identify_unknown_bytes_match_no_pin(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not the weights")
    ident = identify(str(tmp_path))
    assert ident["weights_match_pin"] is None
    assert ident["files"]["model.safetensors"] == hashlib.sha256(b"not the weights").hexdigest()


def test_identify_matches_a_pin_when_hashes_agree(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(weights, "sha256_file", lambda p: PINS["oraculumai/Manchego-MLX-4bit"]["files"]["model.safetensors"])
    assert identify(str(tmp_path))["weights_match_pin"]["repo"] == "oraculumai/Manchego-MLX-4bit"


def _pinned_hash(repo):
    """sha256_file stand-in: the pinned hash of whichever file of `repo` is asked for (tampered files are handled by
    writing the name 'tampered' into the file)."""
    pin = PINS[repo]
    def f(p):
        if open(p, "rb").read() == b"tampered":
            return hashlib.sha256(b"tampered").hexdigest()
        return {**pin["files"], **pin["support"]}[os.path.basename(p)]
    return f


def test_download_refuses_bytes_that_do_not_match_the_pin(tmp_path, monkeypatch, capsys):
    repo = "oraculumai/Manchego-MLX-4bit"
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    for n in weights.SUPPORT_FILES:
        (tmp_path / n).write_bytes(b"x")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(weights, "sha256_file", _pinned_hash(repo))
    assert download.main(["--repo", repo]) == 1                       # weights tampered
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert download.main(["--repo", repo]) == 0                       # everything as pinned
    (tmp_path / "chat_template.jinja").write_bytes(b"tampered")
    assert download.main(["--repo", repo]) == 1                       # chat template tampered
    (tmp_path / "chat_template.jinja").unlink()
    assert download.main(["--repo", repo]) == 1                       # chat template missing


def test_support_files_are_pinned_per_repo_and_checked(tmp_path, monkeypatch):
    for repo, pin in PINS.items():
        assert set(pin["support"]) == set(weights.SUPPORT_FILES) and all(len(h) == 64 for h in pin["support"].values())
    repo = "oraculumai/Manchego-MLX-8bit"
    for n in ("model.safetensors",) + weights.SUPPORT_FILES:
        (tmp_path / n).write_bytes(b"x")
    monkeypatch.setattr(weights, "sha256_file", _pinned_hash(repo))
    ident = identify(str(tmp_path))
    assert ident["weights_match_pin"]["repo"] == repo and ident["support_files_match_pin"]
    (tmp_path / "tokenizer.json").write_bytes(b"tampered")
    ident = identify(str(tmp_path))
    assert ident["weights_match_pin"]["repo"] == repo and not ident["support_files_match_pin"]


def test_resolve_reads_the_local_cache_only():
    with pytest.raises(weights.WeightsNotFound) as e:
        weights.resolve("oraculumai/not-a-published-repo", "0" * 40)
    assert "manchego-serve-download" in str(e.value)


def test_server_forces_offline(monkeypatch):
    from manchego_serve.server import force_offline
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        monkeypatch.setenv(k, "0")
    force_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
