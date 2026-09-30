"""Pins, weight identification, the download check and offline mode. No network, no weights."""
import hashlib
import os
from pathlib import Path

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
    assert download.main(["--repo", repo, "--revision", "v2.1"]) == 1          # weights tampered
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert download.main(["--repo", repo, "--revision", "v2.1"]) == 0          # everything as pinned
    (tmp_path / "chat_template.jinja").write_bytes(b"tampered")
    assert download.main(["--repo", repo, "--revision", "v2.1"]) == 1          # chat template tampered
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


# ------------------------------------------------------------------ Manchego v3 pins (0.2.0)
V3_DIGESTS = {  # weights_sha256 of the v3 release builds (the temperature map's bindings, temperature.V3_BUILDS)
    "oraculumai/Manchego": "2ee838433bfe278a226dc644667ad4a99ece82cc47325c7645a7dae723c1863b",
    "oraculumai/Manchego-MLX-8bit": "358b025b04001e50a065f8c87929175211264bd6182af74b67bd6caa2f639657",
    "oraculumai/Manchego-MLX-4bit": "e1bc5538b8dced2a857b4980dba045c2ca01db1c369aa416fc19f0f5c593e782",
}
V3_REVISIONS = {"oraculumai/Manchego": "3" * 40, "oraculumai/Manchego-MLX-8bit": "4" * 40, "oraculumai/Manchego-MLX-4bit": "5" * 40}


def digest(files):
    return hashlib.sha256("".join(f"{files[n]}  {n}\n" for n in sorted(files)).encode()).hexdigest()


def pin_v3(monkeypatch, repos=tuple(V3_REVISIONS)):
    """As after the upload: the v3 revisions filled in."""
    for repo in repos:
        monkeypatch.setitem(weights.V3_PINS[repo], "revision", V3_REVISIONS[repo])


def test_v3_pins_are_the_release_builds():
    from manchego_serve import temperature as TM
    assert weights.RELEASES == {"v3": weights.V3_PINS, "v2.1": PINS} and weights.VERSIONS == ("v3", "v2.1")
    for repo, pin in weights.V3_PINS.items():
        assert pin["tag"] == "v3" and pin["backend"] == PINS[repo]["backend"] and pin["format"] == PINS[repo]["format"]
        assert digest(pin["files"]) == V3_DIGESTS[repo] and all(len(h) == 64 for h in pin["files"].values())
        assert pin["support"] == PINS[repo]["support"]                     # v3 ships v2.1's tokenizer, template and configs
    assert weights.BF16_SUPPORT == PINS["oraculumai/Manchego"]["support"]
    assert set(V3_DIGESTS.values()) == set(TM.V3_BUILDS)
    assert TM.default_v3().model_sha256 == V3_DIGESTS["oraculumai/Manchego"]


def test_v3_revisions_are_the_uploaded_commits():
    """The v3 revisions are the verified upload commits (tag v3, 2026-09-30); v3 is every repository's default and v2.1 stays
    available by name."""
    got = {"oraculumai/Manchego": weights.V3_REVISION, "oraculumai/Manchego-MLX-8bit": weights.V3_MLX8_REVISION,
           "oraculumai/Manchego-MLX-4bit": weights.V3_MLX4_REVISION}
    assert got == {"oraculumai/Manchego": "53251b0c118d28bfe7908eac1b3a02c08a877edc",
                   "oraculumai/Manchego-MLX-8bit": "778cc7b870ac74c2676efbf803c8baf28c13922e",
                   "oraculumai/Manchego-MLX-4bit": "ade390a4644d73e288f7ea442a10f980495816ec"}
    assert sorted(v for v, _, _ in weights.published_pins()) == ["v2.1"] * 3 + ["v3"] * 3
    for repo in V3_DIGESTS:
        assert weights.default_version(repo) == "v3" and weights.pinned_revision(repo) == got[repo]
        assert weights.pinned_revision(repo, "v2.1") == PINS[repo]["revision"]
    assert weights.default_version("someone/else") is None


def test_version_names_are_pinned_commits(monkeypatch, tmp_path):
    repo = "oraculumai/Manchego"
    assert weights.named_revision(repo, "v2.1") == PINS[repo]["revision"] and weights.named_revision(repo, "a" * 40) == "a" * 40
    assert weights.resolve(str(tmp_path), "v2.1", default_repo=repo) == (str(tmp_path.resolve()), PINS[repo]["revision"])
    assert weights.resolve(str(tmp_path), "v2.1") == (str(tmp_path.resolve()), "v2.1")      # no repository to read it for
    assert weights.resolve(str(tmp_path), "v3", default_repo=repo)[1] == weights.V3_REVISION
    with pytest.raises(weights.RevisionError, match="not a Manchego version"):
        weights.pinned_revision("someone/else", "v2.1")
    pin_v3(monkeypatch)
    assert weights.default_version(repo) == "v3" and weights.pinned_revision(repo) == V3_REVISIONS[repo]
    assert weights.resolve(str(tmp_path), "v3", default_repo=repo)[1] == V3_REVISIONS[repo]
    assert weights.pinned_revision(repo, "v2.1") == PINS[repo]["revision"]                  # v2.1 stays available by name


def _bytes_of(pin):
    def f(p):
        if open(p, "rb").read() == b"tampered":
            return hashlib.sha256(b"tampered").hexdigest()
        return {**pin["files"], **pin["support"]}[os.path.basename(p)]
    return f


def _folder(tmp_path, pin):
    for n in list(pin["files"]) + list(weights.SUPPORT_FILES):
        (tmp_path / n).write_bytes(b"x")
    return tmp_path


def test_identify_v3_bytes(tmp_path, monkeypatch):
    repo = "oraculumai/Manchego"
    pin = weights.V3_PINS[repo]
    _folder(tmp_path, pin)
    monkeypatch.setattr(weights, "sha256_file", _bytes_of(pin))
    ident = identify(str(tmp_path))
    assert ident["weights_sha256"] == V3_DIGESTS[repo]
    assert ident["weights_match_pin"] == {"repo": repo, "revision": weights.V3_REVISION, "tag": "v3"}   # the published v3 commit
    pin_v3(monkeypatch)
    ident = identify(str(tmp_path))
    assert ident["weights_match_pin"] == {"repo": repo, "revision": V3_REVISIONS[repo], "tag": "v3"}
    assert ident["support_files_match_pin"]


def _download(monkeypatch, tmp_path, pin, capsys=None):
    import huggingface_hub
    asked = {}

    def snapshot(repo, revision=None, local_dir=None):
        asked.update(repo=repo, revision=revision)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(weights, "sha256_file", _bytes_of(pin))
    return asked


def test_download_default_is_v3_and_v21_stays_available_by_name(tmp_path, monkeypatch, capsys):
    repo = "oraculumai/Manchego"
    pin = weights.V3_PINS[repo]
    asked = _download(monkeypatch, _folder(tmp_path, pin), pin)
    assert download.main([]) == 0 and asked == {"repo": repo, "revision": weights.V3_REVISION}
    assert '"version": "v3"' in capsys.readouterr().out
    asked = _download(monkeypatch, _folder(tmp_path, PINS[repo]), PINS[repo])
    assert download.main(["--revision", "v2.1"]) == 0 and asked["revision"] == PINS[repo]["revision"]


def test_download_fetches_v3_once_pinned(tmp_path, monkeypatch, capsys):
    repo = "oraculumai/Manchego"
    pin_v3(monkeypatch)
    pin = weights.V3_PINS[repo]
    asked = _download(monkeypatch, _folder(tmp_path, pin), pin)
    assert download.main([]) == 0 and asked["revision"] == V3_REVISIONS[repo]
    assert '"version": "v3"' in capsys.readouterr().out
    (tmp_path / "model.safetensors-00002-of-00002.safetensors").write_bytes(b"tampered")
    assert download.main(["--revision", "v3"]) == 1
    (tmp_path / "model.safetensors-00002-of-00002.safetensors").write_bytes(b"x")
    monkeypatch.setattr(weights, "sha256_file", _bytes_of(PINS[repo]))       # v2.1 bytes where v3 was asked
    for n in PINS[repo]["files"]:
        (tmp_path / n).write_bytes(b"x")
    assert download.main(["--revision", "v3"]) == 1
    asked = _download(monkeypatch, tmp_path, PINS[repo])
    assert download.main(["--revision", "v2.1"]) == 0 and asked["revision"] == PINS[repo]["revision"]


def test_dockerfile_default_follows_the_pins():
    """The image's MANCHEGO_VERSION default is the default version of oraculumai/Manchego: v2.1 while V3_REVISION is a
    placeholder, v3 once it is filled in (this test fails until the Dockerfile follows)."""
    import re
    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    default = re.search(r"^ARG MANCHEGO_VERSION=(\S+)$", text, re.M).group(1)
    assert default == weights.default_version("oraculumai/Manchego")
    assert re.search(r"^ARG MODEL_REVISION=$", text, re.M)
    assert text.count("${MODEL_REVISION:-${MANCHEGO_VERSION}}") == 2        # the download and the run-time declaration
