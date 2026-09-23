"""The per-type temperature map (0.1.1). It loads, applies per question type and never changes the chosen option. With
`off`, the readout is v0.1.0's, byte for byte, including on the golden fixtures. No weights needed."""
import hashlib
import json
import math
import random

import pytest

import v010_reference as V010
from conftest import BY_ID, FIX, load
from manchego_serve import contract as C
from manchego_serve import temperature as TM
from manchego_serve.decider import Decider
from test_server import CHOICE, NOUL, SCORE, STATE, FakeBackend, FakeTok, menu

DEFAULT = TM.load("default")
MAPS = {"default": DEFAULT,
        "flat": TM.TemperatureMap({"choice": 20.0, "noul": 20.0, "score": 20.0}, "file"),
        "sharp": TM.TemperatureMap({"choice": 0.05, "noul": 0.05, "score": 0.05}, "file"),
        "mixed": TM.TemperatureMap({"choice": 1.79, "noul": 0.6, "score": 3.3}, "file")}


# ------------------------------------------------------------------ the packaged map
def test_default_map_is_the_pinned_file():
    raw = TM.DEFAULT_FILE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TM.DEFAULT_SHA256 == DEFAULT.sha256
    data = json.loads(raw)
    assert data["schema"] == TM.SCHEMA and set(data["temperatures"]) == set(TM.TYPES)
    assert DEFAULT.source == "default" and DEFAULT.temperatures == {t: float(data["temperatures"][t]) for t in TM.TYPES}
    assert all(TM.T_MIN <= v <= TM.T_MAX for v in DEFAULT.temperatures.values())
    for key in ("model", "applies_to", "fitted_on", "held_out_result", "known_costs", "disclosure"):
        assert key in data, key
    assert {a["repo"] for a in data["applies_to"]} <= {"oraculumai/Manchego", "oraculumai/Manchego-MLX-8bit", "oraculumai/Manchego-MLX-4bit"}


def test_a_tampered_default_is_refused(tmp_path, monkeypatch):
    bad = tmp_path / TM.DEFAULT_FILE.name
    data = json.loads(TM.DEFAULT_FILE.read_text())
    data["temperatures"]["choice"] = 9.0
    bad.write_text(json.dumps(data))
    monkeypatch.setattr(TM, "DEFAULT_FILE", bad)
    with pytest.raises(TM.TemperatureMapError, match="not the published"):
        TM.load("default")


@pytest.mark.parametrize("spec", ["off", "OFF", "none", "1", "1.0", "t1"])
def test_off_is_identity(spec):
    m = TM.load(spec)
    assert m is TM.OFF and m.is_identity and m.wire_temperature() == 1.0
    assert all(m.T(t) == 1.0 for t in TM.TYPES)


def test_load_empty_or_none_is_default():
    assert TM.load(None).sha256 == TM.load("").sha256 == TM.load("default").sha256 == TM.DEFAULT_SHA256


def test_custom_file_loads(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema": TM.SCHEMA, "temperatures": {"choice": 1.5, "noul": 2, "score": 1.0}}))
    m = TM.load(str(p))
    assert m.source == "file" and m.file == str(p) and m.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert m.temperatures == {"choice": 1.5, "noul": 2.0, "score": 1.0}
    assert m.wire_temperature() == {"choice": 1.5, "noul": 2.0, "score": 1.0}


@pytest.mark.parametrize("temps", [
    {"choice": 1.5, "noul": 2.0},                          # a type missing
    {"choice": 1.5, "noul": 2.0, "score": 1.0, "rank": 1},  # an unknown type
    {"choice": 0.0, "noul": 2.0, "score": 1.0},             # not positive
    {"choice": -1, "noul": 2.0, "score": 1.0},
    {"choice": float("nan"), "noul": 2.0, "score": 1.0},
    {"choice": True, "noul": 2.0, "score": 1.0},
    {"choice": "1.5", "noul": 2.0, "score": 1.0},
    {"choice": 1e6, "noul": 2.0, "score": 1.0},
])
def test_bad_maps_are_refused(tmp_path, temps):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"schema": TM.SCHEMA, "temperatures": temps}, allow_nan=True))
    with pytest.raises(TM.TemperatureMapError):
        TM.load(str(p))


def test_bad_specs_are_refused(tmp_path):
    with pytest.raises(TM.TemperatureMapError):
        TM.load(str(tmp_path / "missing.json"))
    p = tmp_path / "x.json"
    p.write_text("not json")
    with pytest.raises(TM.TemperatureMapError):
        TM.load(str(p))
    p.write_text(json.dumps({"temperatures": {"choice": 1, "noul": 1, "score": 1}}))
    with pytest.raises(TM.TemperatureMapError, match="schema"):
        TM.load(str(p))


# ------------------------------------------------------------------ binding to the loaded weights
def test_binding():
    listed = [a["weights_sha256"] for a in DEFAULT.meta["applies_to"]]
    assert listed
    logs = []
    ok = TM.bind(DEFAULT, listed[0], log=logs.append)
    assert ok.temperatures == DEFAULT.temperatures and ok.binding.startswith("a build this map lists") and not logs
    unchecked = TM.bind(DEFAULT, None, log=logs.append)
    assert unchecked.temperatures == DEFAULT.temperatures and "not checked" in unchecked.binding
    other = TM.bind(DEFAULT, "0" * 64, log=logs.append)
    assert other.is_identity and other.source == "off" and "not one of them" in other.note and logs
    explicit = TM.TemperatureMap(DEFAULT.temperatures, "file", "m.json", "x", DEFAULT.meta)
    kept = TM.bind(explicit, "0" * 64, log=logs.append)
    assert kept.temperatures == DEFAULT.temperatures and kept.binding.startswith("NOT")
    assert TM.bind(TM.OFF, "0" * 64) is TM.OFF


def test_every_published_v21_build_listed_is_pinned():
    from manchego_serve.weights import PINS
    pinned = {p["revision"]: repo for repo, p in PINS.items()}
    for a in DEFAULT.meta["applies_to"]:
        assert pinned.get(a["revision"]) == a["repo"], a


# ------------------------------------------------------------------ T = 1 is v0.1.0, byte for byte
def random_logits(rng, k):
    style = rng.randrange(4)
    if style == 0:
        return [rng.gauss(0, 3) for _ in range(k)]
    if style == 1:                                      # near-ties
        base = rng.uniform(-5, 20)
        return [base + rng.choice([0.0, 1e-6, -1e-6, 1e-3, 0.0]) for _ in range(k)]
    if style == 2:                                      # one confident option
        z = [rng.uniform(0, 8) for _ in range(k)]
        z[rng.randrange(k)] += rng.uniform(5, 30)
        return z
    return [float(rng.randrange(-3, 4)) for _ in range(k)]   # exact ties


def questions(rng, k):
    return [({"type": "choice", "instructions": "x", "criteria": {f"v{i}": None for i in range(k)}},
             [(f"v{i}", None) for i in range(k)]),
            ({"type": "score", "instructions": "x", "criteria": [f"l{i}" for i in range(k)]},
             [(str(i), f"l{i}") for i in range(k)])]


def test_off_answer_matches_v010_bytes_on_random_logits():
    rng = random.Random(11)
    for _ in range(3000):
        k = rng.choice([2, 3, 5, 10, 26, 27, 100, 255])
        z = random_logits(rng, k)
        for q, opts in questions(rng, k):
            assert json.dumps(C.answer(q, opts, z, TM.OFF.T(q["type"]))) == json.dumps(V010.answer(q, opts, z))
        zn = random_logits(rng, 2)
        qn = {"type": "noul", "instructions": "x"}
        on = [("true", None), ("false", None)]
        assert json.dumps(C.answer(qn, on, zn, 1.0)) == json.dumps(V010.answer(qn, on, zn))


class Replay(FakeBackend):
    """Returns recorded logits in call order (sequential execution scores questions in request order)."""

    def __init__(self, logits_in_order):
        super().__init__()
        self.queue = list(logits_in_order)

    def logits_batch(self, items):
        return [self.queue.pop(0) for _ in items]


GOLDEN = load("golden_mlx8bit.json")


def golden_logits():
    path = FIX / "golden_logits_mlx8bit.json"
    if not path.exists():
        pytest.skip("fixtures/golden_logits_mlx8bit.json not recorded (tests/record_golden_logits.py)")
    return json.loads(path.read_text())


@pytest.mark.parametrize("execution", ["sequential", "batched"])
def test_off_reproduces_the_v010_golden_fixtures_byte_for_byte(execution):
    """The recorded MLX 8-bit logits through today's readout with the map off give golden_mlx8bit.json's answers exactly."""
    rec = golden_logits()
    assert rec["weights"] == GOLDEN["weights"]
    ref = GOLDEN["results"][execution]
    assert len(ref) == 20 and set(rec["results"][execution]) == set(ref)
    for rid, r in ref.items():
        body = BY_ID[rid]["body"]
        logits = rec["results"][execution][rid]["logits"]
        d = Decider(Replay([logits[n] for n in body["questions"]]), temperature_map=TM.OFF)
        out = d.handle(body)
        assert json.dumps(out["answers"]) == json.dumps(r["answers"]), rid
        assert out["manchego"]["temperature"] == 1.0
        assert set(out["manchego"]["temperature_by_question"].values()) == {1.0}


def test_default_map_changes_numbers_but_never_the_choice_on_the_golden_requests():
    rec = golden_logits()
    changed = 0
    for rid, r in GOLDEN["results"]["sequential"].items():
        body = BY_ID[rid]["body"]
        logits = rec["results"]["sequential"][rid]["logits"]
        out = Decider(Replay([logits[n] for n in body["questions"]]), temperature_map=DEFAULT).handle(body)
        for n, a in out["answers"].items():
            g = r["answers"][n]
            assert a["type"] == g["type"] and a.get("choice") == g.get("choice") and a.get("legend") == g.get("legend")
            assert list(a.get("probabilities", {})) == list(g.get("probabilities", {}))
            if a["type"] == "noul":
                assert (a["noul"] > 0.5) == (g["noul"] > 0.5) and (a["noul"] < 0.5) == (g["noul"] < 0.5)
                changed += a["noul"] != g["noul"] and DEFAULT.T("noul") != 1.0
            else:
                pa, pg = list(a["probabilities"].values()), list(g["probabilities"].values())
                assert pa.index(max(pa)) == pg.index(max(pg))
                changed += pa != pg and DEFAULT.T(a["type"]) != 1.0
    assert changed > 0


# ------------------------------------------------------------------ the map never changes the chosen option
def top(a: dict) -> int:
    p = list(a["probabilities"].values())
    return p.index(max(p))


@pytest.mark.parametrize("name", list(MAPS))
def test_argmax_never_changes(name):
    tmap = MAPS[name]
    rng = random.Random(7)
    for _ in range(3000):
        k = rng.choice([2, 3, 5, 10, 26, 27, 100, 255])
        z = random_logits(rng, k)
        for q, opts in questions(rng, k):
            a1, aT = C.answer(q, opts, z, 1.0), C.answer(q, opts, z, tmap.T(q["type"]))
            assert aT.get("choice") == a1.get("choice")
            if q["type"] == "choice":
                assert aT["choice"] == opts[C.argmax_first(z)][0]
            assert top(aT) == top(a1)
        zn = random_logits(rng, 2)
        qn, on = {"type": "noul", "instructions": "x"}, [("true", None), ("false", None)]
        n1, nT = C.answer(qn, on, zn, 1.0)["noul"], C.answer(qn, on, zn, tmap.T("noul"))["noul"]
        assert (n1 > 0.5) == (nT > 0.5) and (n1 < 0.5) == (nT < 0.5)


def test_choice_follows_the_logits_on_near_and_exact_ties():
    q = {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None, "c": None}}
    opts = [("a", None), ("b", None), ("c", None)]
    for z, want in (([10.0, 10.0 + 1e-9, -50.0], "b"), ([3.0, 3.0, 1.0], "a"), ([1.0, 3.0, 3.0], "b")):
        for T in (0.05, 0.5, 1.0, 1.79, 20.0):
            assert C.answer(q, opts, z, T)["choice"] == want == V010.answer(q, opts, z)["choice"]


# ------------------------------------------------------------------ the map applies per type
def test_map_applies_per_question_type():
    tmap = MAPS["mixed"]
    body = {"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE}}
    off = Decider(FakeBackend()).handle(body)
    on = Decider(FakeBackend(), temperature_map=tmap).handle(body)
    raw = {n: FakeBackend().logits_batch([(None, [0] * k)])[0] for n, k in (("c", 3), ("n", 2), ("s", 3))}
    for n, kind in (("c", "choice"), ("n", "noul"), ("s", "score")):
        p = C.softmax(raw[n], tmap.T(kind))
        if kind == "noul":
            assert on["answers"][n]["noul"] == p[0]
        else:
            assert list(on["answers"][n]["probabilities"].values()) == p
            assert on["answers"][n]["confidence"] == C.confidence(p)
    assert on["answers"]["c"]["choice"] == off["answers"]["c"]["choice"]
    assert on["answers"]["s"]["score"] == pytest.approx(sum(i * p for i, p in enumerate(on["answers"]["s"]["probabilities"].values())))
    m = on["manchego"]
    assert m["temperature"] == tmap.temperatures
    assert m["temperature_by_question"] == {"c": 1.79, "n": 0.6, "s": 3.3}
    assert m["temperature_map"]["source"] == "file"
    assert off["manchego"]["temperature"] == 1.0 and off["manchego"]["temperature_map"]["source"] == "off"


def test_default_map_in_a_response():
    out = Decider(FakeBackend(), temperature_map=DEFAULT).handle({"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE}})
    m = out["manchego"]
    assert m["temperature_map"]["sha256"] == TM.DEFAULT_SHA256 and m["temperature_map"]["source"] == "default"
    assert m["temperature_by_question"] == {"c": DEFAULT.T("choice"), "n": DEFAULT.T("noul"), "s": DEFAULT.T("score")}
    assert m["temperature"] == (1.0 if DEFAULT.is_identity else DEFAULT.temperatures)


def test_off_decider_matches_v010_on_the_fake_backend():
    body = {"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE, "m": {"type": "choice", "instructions": "x", "criteria": menu(40)}}}
    out = Decider(FakeBackend()).handle(body)
    b = FakeBackend()
    for name, q in body["questions"].items():
        opts = C.options_of(q)
        z = b.logits_batch([(None, [0] * len(opts))])[0]
        assert json.dumps(out["answers"][name]) == json.dumps(V010.answer(q, opts, z))


# ------------------------------------------------------------------ command line and /healthz
def test_command_line_and_environment(monkeypatch):
    from manchego_serve.server import parse_args
    monkeypatch.delenv("MANCHEGO_TEMPERATURE_MAP", raising=False)
    assert parse_args([]).temperature_map == "default"
    assert parse_args(["--temperature-map", "off"]).temperature_map == "off"
    assert parse_args(["--no-temperature-map"]).temperature_map == "off"
    assert parse_args(["--temperature-map", "/x/m.json"]).temperature_map == "/x/m.json"
    monkeypatch.setenv("MANCHEGO_TEMPERATURE_MAP", "off")
    assert parse_args([]).temperature_map == "off"
    assert parse_args(["--temperature-map", "default"]).temperature_map == "default"
    monkeypatch.setenv("MANCHEGO_TEMPERATURE_MAP", "")
    assert parse_args([]).temperature_map == "default"


def test_healthz_reports_the_map():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from manchego_serve.server import create_app
    h = TestClient(create_app(Decider(FakeBackend(), temperature_map=DEFAULT))).get("/healthz").json()
    assert h["temperature_map"]["sha256"] == TM.DEFAULT_SHA256
    assert h["temperature_map"]["temperatures"] == DEFAULT.temperatures
    assert h["temperature_map_provenance"]["model"] == DEFAULT.meta["model"]
    h0 = TestClient(create_app(Decider(FakeBackend()))).get("/healthz").json()
    assert h0["temperature"] == 1.0 and h0["temperature_map"]["source"] == "off" and h0["temperature_map_provenance"] is None


def test_confidence_definition_is_unchanged_under_the_map():
    out = Decider(FakeBackend(), temperature_map=MAPS["mixed"]).handle({"state": STATE, "questions": {"c": CHOICE}})["answers"]["c"]
    p = list(out["probabilities"].values())
    assert out["confidence"] == pytest.approx((len(p) * max(p) - 1) / (len(p) - 1))
    assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
