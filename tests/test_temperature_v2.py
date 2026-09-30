"""Temperature map schema 2 (0.2.0): typed temperatures that may sharpen (T < 1), bound to one model's weights_sha256,
refused for any other; and the server default: T = 1.0 for any model the map is not bound to by hash."""
import json
import math
import random

import pytest

from manchego_serve import contract as C
from manchego_serve import temperature as TM
from manchego_serve.decider import Decider
from test_server import CHOICE, NOUL, SCORE, STATE, FakeBackend
from test_temperature import questions, random_logits, top

DEFAULT = TM.load("default")
V21_HASHES = [a["weights_sha256"] for a in DEFAULT.meta["applies_to"]]
V3_HASH = "ab" * 32                              # any weights the packaged map does not list


def v2_map(tmp_path, **over):
    data = {"schema": TM.SCHEMA_V2, "model": "Manchego v3 (test)", "temperatures": {"choice": 0.8, "noul": 0.6, "score": 1.0},
            "model_sha256": V3_HASH, "fitted_on": "held-out development rows of the model, contract semif, 50/50 split by instance",
            "rule": "minimise NLL on the fit half per type; serve a type's T only if held-out NLL and ECE both improve", "contract": "semif"}
    data.update(over)
    data = {k: v for k, v in data.items() if v is not ...}
    p = tmp_path / "tmap_v3.json"
    p.write_text(json.dumps(data))
    return p


# ------------------------------------------------------------------ loading
def test_v2_map_loads_and_may_sharpen(tmp_path):
    m = TM.load(str(v2_map(tmp_path)))
    assert m.schema == TM.SCHEMA_V2 and m.source == "file" and m.model_sha256 == V3_HASH and m.contract == "semif"
    assert m.temperatures == {"choice": 0.8, "noul": 0.6, "score": 1.0} and not m.is_identity
    d = m.describe()
    assert d["schema"] == TM.SCHEMA_V2 and d["model_sha256"] == V3_HASH and d["contract"] == "semif"
    prov = m.provenance()
    assert prov["fitted_on"].startswith("held-out") and prov["rule"].startswith("minimise") and prov["model_sha256"] == V3_HASH


@pytest.mark.parametrize("over", [
    {"model_sha256": ...}, {"model_sha256": "ABCD"}, {"model_sha256": "AB" * 32}, {"model_sha256": 7},
    {"fitted_on": ...}, {"fitted_on": ""}, {"fitted_on": "  "}, {"rule": ...}, {"rule": 3},
    {"contract": "short"}, {"temperatures": {"choice": 0.01, "noul": 0.6, "score": 1.0}},
    {"temperatures": {"choice": 0.8, "noul": 0.6}}, {"schema": "manchego-temperature-map/3"},
])
def test_bad_v2_maps_are_refused(tmp_path, over):
    with pytest.raises(TM.TemperatureMapError):
        TM.load(str(v2_map(tmp_path, **over)))


def test_v1_describe_is_unchanged():
    """Schema-1 maps (the packaged v2.1 map) report exactly the 0.1.2 fields."""
    assert set(DEFAULT.describe()) == {"source", "temperatures", "file", "sha256", "fitted_for", "binding"}
    assert DEFAULT.schema == TM.SCHEMA and DEFAULT.model_sha256 is None


# ------------------------------------------------------------------ binding: refused unless the hash matches
def test_v2_binds_only_to_its_weights(tmp_path):
    m = TM.load(str(v2_map(tmp_path)))
    ok = TM.bind(m, V3_HASH)
    assert ok.temperatures == m.temperatures and ok.binding.startswith("bound")
    with pytest.raises(TM.TemperatureMapError, match="refusing"):
        TM.bind(m, V21_HASHES[0])
    with pytest.raises(TM.TemperatureMapError, match="not hashed"):
        TM.bind(m, None)
    for declared in ({}, {"declared_repo": "x/y", "declared_revision": "0" * 40}):   # a declaration never replaces the hash
        with pytest.raises(TM.TemperatureMapError):
            TM.bind(m, None, **declared)


def test_v2_resolve_checks_the_contract(tmp_path):
    m = TM.load(str(v2_map(tmp_path)))
    assert TM.resolve(m, V3_HASH, contract="semif").binding.startswith("bound")
    with pytest.raises(TM.TemperatureMapError, match="contract"):
        TM.resolve(m, V3_HASH, contract="auto")
    free = TM.load(str(v2_map(tmp_path, contract=...)))            # no contract named: any contract, the hash still binds
    assert TM.resolve(free, V3_HASH, contract="auto").temperatures == free.temperatures
    with pytest.raises(TM.TemperatureMapError):
        TM.resolve(free, V21_HASHES[0], contract="auto")


# ------------------------------------------------------------------ the default: off unless bound by hash
def test_default_map_is_off_for_any_other_model():
    logs = []
    for sha in (V3_HASH, "0" * 64):
        r = TM.resolve(DEFAULT, sha, log=logs.append)
        assert r.is_identity and r.source == "off" and "not one of them" in r.note
    assert logs


def test_default_map_applies_to_v21_under_auto_only():
    for sha in V21_HASHES:
        r = TM.resolve(DEFAULT, sha, contract="auto")
        assert r.temperatures == DEFAULT.temperatures and r.binding.startswith("a build this map lists")
        logs = []
        s = TM.resolve(DEFAULT, sha, contract="semif", log=logs.append)
        assert s.is_identity and "contract auto" in s.note and logs


def test_default_map_unhashed():
    """--no-hash: a bare directory with nothing declared is off (0.2.0); a declared listed build keeps the 0.1.2 behaviour."""
    logs = []
    bare = TM.resolve(DEFAULT, None, log=logs.append)
    assert bare.is_identity and "cannot be bound" in bare.note and logs
    for a in DEFAULT.meta["applies_to"]:
        kept = TM.resolve(DEFAULT, None, declared_repo=a["repo"], declared_revision=a["revision"])
        assert kept.temperatures == DEFAULT.temperatures and "not checked" in kept.binding
        by_rev = TM.resolve(DEFAULT, None, declared_revision=a["revision"])
        assert by_rev.temperatures == DEFAULT.temperatures
    assert TM.resolve(DEFAULT, None, declared_repo="oraculumai/Manchego-MLX-4bit").is_identity
    assert TM.resolve(TM.OFF, None) is TM.OFF


def test_explicit_v1_file_keeps_its_0_1_2_behaviour():
    explicit = TM.TemperatureMap(DEFAULT.temperatures, "file", "m.json", "x", DEFAULT.meta)
    kept = TM.resolve(explicit, V3_HASH, log=lambda m: None, contract="semif")
    assert kept.temperatures == DEFAULT.temperatures and kept.binding.startswith("NOT")


# ------------------------------------------------------------------ sharpening in the readout
@pytest.mark.parametrize("temps", [{"choice": 0.5, "noul": 0.3, "score": 0.7}, {"choice": 0.05, "noul": 0.05, "score": 0.05}])
def test_sharpening_never_changes_the_choice(temps):
    rng = random.Random(3)
    for _ in range(2000):
        k = rng.choice([2, 3, 5, 16, 27, 255])
        z = random_logits(rng, k)
        for q, opts in questions(rng, k):
            a1, aT = C.answer(q, opts, z, 1.0), C.answer(q, opts, z, temps[q["type"]])
            assert aT.get("choice") == a1.get("choice") and top(aT) == top(a1)
            assert max(aT["probabilities"].values()) >= max(a1["probabilities"].values()) - 1e-12   # T < 1 sharpens


def test_v2_map_in_a_response(tmp_path):
    m = TM.resolve(TM.load(str(v2_map(tmp_path))), V3_HASH, contract="semif")
    out = Decider(FakeBackend(), temperature_map=m, contract="semif").handle({"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE}})
    mm = out["manchego"]
    assert mm["temperature"] == {"choice": 0.8, "noul": 0.6, "score": 1.0}
    assert mm["temperature_by_question"] == {"c": 0.8, "n": 0.6, "s": 1.0}
    assert mm["temperature_map"]["model_sha256"] == V3_HASH and mm["temperature_map"]["binding"].startswith("bound")


def test_noul_abstention_band_arithmetic():
    """README warning: JevBench v1.5 counts P(yes) strictly between 0.20 and 0.80 as an abstention (wrong). P(yes) >= 0.8
    needs a yes-no logit margin of T * ln 4: 1.386 at T = 1, 2.398 at the v2.1 map's noul T = 1.73. Every noul answer with a
    margin in between is a Yes at T = 1 and an abstention under the map (and symmetrically for No)."""
    T = DEFAULT.T("noul")
    assert T == 1.73
    q, opts = {"type": "noul", "instructions": "x"}, [("true", None), ("false", None)]
    lo, hi = math.log(4), T * math.log(4)
    for margin in (lo + 0.01, 2.0, hi - 0.01):
        assert C.answer(q, opts, [margin, 0.0], 1.0)["noul"] >= 0.8
        assert 0.2 < C.answer(q, opts, [margin, 0.0], T)["noul"] < 0.8
        assert 0.2 < C.answer(q, opts, [0.0, margin], T)["noul"] < 0.8 and C.answer(q, opts, [0.0, margin], 1.0)["noul"] <= 0.2
    assert C.answer(q, opts, [hi + 0.01, 0.0], T)["noul"] >= 0.8


# ------------------------------------------------------------------ the packaged Manchego v3 map (0.2.0)
V3_BF16 = "2ee838433bfe278a226dc644667ad4a99ece82cc47325c7645a7dae723c1863b"
V3_MLX8 = "358b025b04001e50a065f8c87929175211264bd6182af74b67bd6caa2f639657"
V3_MLX4 = "e1bc5538b8dced2a857b4980dba045c2ca01db1c369aa416fc19f0f5c593e782"
V3_T = {"choice": 1.5, "noul": 0.2, "score": 1.0}


def test_packaged_v3_map_is_the_fitted_file():
    import hashlib
    assert hashlib.sha256(TM.V3_FILE.read_bytes()).hexdigest() == TM.V3_SHA256
    m = TM.default_v3()
    assert m.schema == TM.SCHEMA_V2 and m.source == "default" and m.file == "temperature_map_v3.json"
    assert m.temperatures == V3_T and m.model_sha256 == V3_BF16 and m.contract == "semif"
    assert "TMAP-V15" in m.meta["fitted_on"] and m.meta["rule"]
    assert set(TM.V3_BUILDS) == {V3_BF16, V3_MLX8, V3_MLX4}
    assert TM.V3_BUILDS[V3_BF16]["note"] is None
    assert TM.V3_BUILDS[V3_MLX8]["note"] == TM.V3_BUILDS[V3_MLX4]["note"] == TM.V3_MLX_NOTE
    assert "applied to the MLX builds as-is" in TM.V3_MLX_NOTE and "never changed by a temperature" in TM.V3_MLX_NOTE


def test_packaged_v3_map_is_shipped():
    import tomllib
    from pathlib import Path
    data = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert "temperature_map_v3.json" in data["tool"]["setuptools"]["package-data"]["manchego_serve"]


def test_default_applies_the_v3_map_to_the_v3_builds_under_semif():
    for sha in (V3_BF16, V3_MLX8, V3_MLX4):
        r = TM.resolve(DEFAULT, sha, contract="semif")
        assert r.temperatures == V3_T and r.schema == TM.SCHEMA_V2 and r.source == "default" and r.sha256 == TM.V3_SHA256
        d = r.describe()
        assert d["model_sha256"] == V3_BF16 and d["contract"] == "semif" and d["file"] == "temperature_map_v3.json"
        assert (TM.V3_MLX_NOTE in r.binding) is (sha != V3_BF16) and sha in r.binding
    assert TM.resolve(DEFAULT, V3_BF16, contract="semif").binding.startswith("bound: model_sha256 is the loaded weights")


def test_default_v3_map_needs_contract_semif():
    for sha in (V3_BF16, V3_MLX8):
        logs = []
        r = TM.resolve(DEFAULT, sha, contract="auto", log=logs.append)
        assert r.is_identity and "fitted under contract semif" in r.note and logs


def test_default_under_semif_is_off_for_any_other_model():
    for sha in (V21_HASHES + [V3_HASH, "0" * 64]):
        r = TM.resolve(DEFAULT, sha, contract="semif", log=lambda m: None)
        assert r.is_identity and r.source == "off"


def test_default_v3_map_never_applies_unhashed():
    logs = []
    r = TM.resolve(DEFAULT, None, contract="semif", log=logs.append)
    assert r.is_identity and "not hashed" in r.note and logs
    for declared in ({"declared_repo": "oraculumai/Manchego"}, {"declared_revision": "2" * 40},
                     {"declared_repo": "oraculumai/Manchego-MLX-8bit", "declared_revision": "1" * 40}):
        assert TM.resolve(DEFAULT, None, contract="semif", log=lambda m: None, **declared).is_identity


def test_explicit_v3_file_stays_bound_to_the_bf16_weights_only():
    """Passed as a file, the schema-2 map binds by its own model_sha256 only: the MLX builds get it through the
    default, never through the file."""
    m = TM.load(str(TM.V3_FILE))
    assert m.source == "file" and TM.resolve(m, V3_BF16, contract="semif").temperatures == V3_T
    for sha in (V3_MLX8, V3_MLX4):
        with pytest.raises(TM.TemperatureMapError, match="refusing"):
            TM.resolve(m, sha, contract="semif")


def test_a_damaged_packaged_v3_map_stops_the_start(tmp_path, monkeypatch):
    bad = tmp_path / "temperature_map_v3.json"
    bad.write_bytes(TM.V3_FILE.read_bytes().replace(b'"noul": 0.2', b'"noul": 0.3'))
    monkeypatch.setattr(TM, "V3_FILE", bad)
    with pytest.raises(TM.TemperatureMapError, match="not the published"):
        TM.load("default")
    assert TM.load("off") is TM.OFF


def test_v3_map_in_a_response():
    m = TM.resolve(DEFAULT, V3_MLX8, contract="semif")
    out = Decider(FakeBackend(), temperature_map=m, contract="semif").handle({"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE}})
    mm = out["manchego"]
    assert mm["temperature"] == V3_T and mm["temperature_by_question"] == {"c": 1.5, "n": 0.2, "s": 1.0}
    assert mm["temperature_map"]["source"] == "default" and TM.V3_MLX_NOTE in mm["temperature_map"]["binding"]


def test_v3_noul_temperature_and_the_abstention_band():
    """README: under JevBench v1.5 a noul answer is an abstention (scored wrong) when 0.20 < P(yes) < 0.80. At T = 1 that
    is every yes/no logit margin below ln 4 = 1.386; the v3 map's noul T = 0.2 narrows it to margins below 0.2 * ln 4 =
    0.277. Sharpening never changes which side of 0.5 an answer is on."""
    T = V3_T["noul"]
    q, opts = {"type": "noul", "instructions": "x"}, [("true", None), ("false", None)]
    lo, hi = T * math.log(4), math.log(4)
    for margin in (lo + 0.01, 0.8, hi - 0.01):
        assert 0.2 < C.answer(q, opts, [margin, 0.0], 1.0)["noul"] < 0.8
        assert C.answer(q, opts, [margin, 0.0], T)["noul"] >= 0.8 and C.answer(q, opts, [0.0, margin], T)["noul"] <= 0.2
    for margin in (0.0, 0.1, lo - 0.01):
        assert 0.2 < C.answer(q, opts, [margin, 0.0], T)["noul"] < 0.8
    rng = random.Random(5)
    for _ in range(2000):
        z = [rng.uniform(-6, 6), rng.uniform(-6, 6)]
        assert (C.answer(q, opts, z, T)["noul"] > 0.5) == (C.answer(q, opts, z, 1.0)["noul"] > 0.5)
