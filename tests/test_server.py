"""The wire contract, limits and refusals, with a fake backend: no model, no tokenizer, no network."""
import math

import pytest

from manchego_serve import contract as C
from manchego_serve.decider import Decider

# Capacity markers the Decision Index HTTP engine recognises in a 422 body
# (github.com/apolinario/decision-index, decision_index/engines/http.py, CAPACITY_MARKERS).
MARKER_TOO_MANY_OPTIONS = "options per choice"
MARKER_TOO_LONG = "maximum context length"
MARKER_TOO_FEW_OPTIONS = "a choice needs at least two options"


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [sum(map(ord, text)) % 1000] if len(text.split()) == 1 else list(range(len(text.split())))

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "\n".join(m["role"] + ": " + m["content"] for m in msgs)


class FakeBackend:
    name, precision = "fake", "none"

    def __init__(self):
        self.tok, self.calls = FakeTok(), []

    def logits_batch(self, items):
        self.calls.append(len(items))
        return [[float(len(c) - i) * 0.5 for i in range(len(c))] for _, c in items]


def decider(**kw):
    return Decider(FakeBackend(), info={"model": "fake", "revision": "0" * 40}, **kw)


STATE = "Order 7: the kettle arrived with a cracked lid."
CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "refunds", "shipping": None, "billing": None}}
NOUL = {"type": "noul", "instructions": "Is the item damaged?"}
SCORE = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}


def menu(n):
    return {f"opt{i}": None for i in range(n)}


# ------------------------------------------------------------------ response shapes
def test_response_shapes():
    out = decider().handle({"model": "anything", "state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE}})
    c, n, s = out["answers"]["c"], out["answers"]["n"], out["answers"]["s"]
    assert set(c) == {"type", "choice", "confidence", "probabilities"} and c["type"] == "choice"
    assert list(c["probabilities"]) == ["returns", "shipping", "billing"]
    assert c["choice"] == max(c["probabilities"], key=c["probabilities"].get)
    assert set(n) == {"type", "noul"} and 0 < n["noul"] < 1
    assert set(s) == {"type", "score", "confidence", "legend", "probabilities"}
    assert list(s["probabilities"]) == ["0", "1", "2"] and s["legend"] == {"0": "low", "1": "medium", "2": "high"}
    assert s["score"] == pytest.approx(sum(i * s["probabilities"][str(i)] for i in range(3)))
    for a in (c, s):
        assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)
    assert out["model"] == "manchego-2.1"
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0
    m = out["manchego"]
    assert m["contract"] == "auto" and m["temperature"] == 1.0 and m["permute"] == 1 and m["execution"] == "sequential"
    assert m["contract_by_question"] == {"c": "short", "n": "short", "s": "short"}
    assert m["forward_passes"] == 3


def test_confidence_is_peak():
    p = [0.88, 0.12, 0.0]
    assert C.confidence(p) == pytest.approx((3 * 0.88 - 1) / 2)
    assert C.confidence([0.5, 0.5]) == 0.0 and C.confidence([1.0, 0.0]) == 1.0
    out = decider().handle({"state": STATE, "questions": {"c": CHOICE}})["answers"]["c"]
    p = list(out["probabilities"].values())
    assert out["confidence"] == pytest.approx((len(p) * max(p) - 1) / (len(p) - 1))


def test_contract_auto_switches_at_27():
    d = decider()
    out = d.handle({"state": STATE, "questions": {"a": {"type": "choice", "instructions": "x", "criteria": menu(26)},
                                                  "b": {"type": "choice", "instructions": "x", "criteria": menu(27)},
                                                  "c": {"type": "choice", "instructions": "x", "criteria": menu(255)}}})
    assert out["manchego"]["contract_by_question"] == {"a": "short", "b": "state_first", "c": "state_first"}
    assert len(out["answers"]["c"]["probabilities"]) == 255


def test_sequential_is_one_row_per_pass_and_batched_pads():
    d = decider()
    d.handle({"state": STATE, "questions": {"a": CHOICE, "b": NOUL, "c": SCORE}})
    assert d.b.calls == [1, 1, 1]
    d = decider(execution="batched")
    out = d.handle({"state": STATE, "questions": {"a": CHOICE, "b": NOUL, "c": SCORE}})
    assert d.b.calls == [3] and out["manchego"]["forward_passes"] == 1


def test_identical_questions_get_identical_answers():
    out = decider().handle({"state": STATE, "questions": {"x": CHOICE, "y": CHOICE}})["answers"]
    assert out["x"] == out["y"]


# ------------------------------------------------------------------ limits and refusals
@pytest.mark.parametrize("q,marker", [
    ({"type": "choice", "instructions": "x", "criteria": menu(256)}, MARKER_TOO_MANY_OPTIONS),
    ({"type": "score", "instructions": "x", "criteria": [str(i) for i in range(256)]}, MARKER_TOO_MANY_OPTIONS),
    ({"type": "choice", "instructions": "x", "criteria": {"only": None}}, MARKER_TOO_FEW_OPTIONS),
])
def test_capacity_refusals_carry_the_markers(q, marker):
    with pytest.raises(C.BadRequest) as e:
        decider().handle({"state": STATE, "questions": {"q": q}})
    assert marker in str(e.value)


def test_too_long_prompt_is_refused_with_the_context_marker():
    with pytest.raises(C.BadRequest) as e:
        decider(max_prompt_tokens=50).handle({"state": "word " * 100, "questions": {"q": NOUL}})
    assert MARKER_TOO_LONG in str(e.value)


def test_other_refusals():
    d = decider(max_questions=2)
    for body in ({"questions": {"q": NOUL}}, {"state": STATE, "questions": {}}, {"state": STATE, "questions": []},
                 {"state": STATE, "questions": {"a": NOUL, "b": NOUL, "c": NOUL}},
                 {"state": STATE, "questions": {"q": {"type": "rank", "instructions": "x"}}},
                 {"state": STATE, "questions": {"q": {"type": "noul"}}},
                 {"state": STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}}):
        with pytest.raises(C.BadRequest):
            d.handle(body)


# ------------------------------------------------------------------ HTTP
@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from manchego_serve.server import create_app
    return TestClient(create_app(decider()))


def test_http_roundtrip(client):
    r = client.post("/v1/systemone", json={"model": "jev-latest", "state": STATE, "questions": {"n": NOUL}, "extra_option": True})
    assert r.status_code == 200
    assert r.json()["answers"]["n"]["type"] == "noul"
    h = client.get("/healthz").json()
    assert h["ok"] and h["contract"] == "auto" and h["limits"]["options_per_question"] == 255
    m = client.get("/v1/models").json()
    assert m["models"][0]["name"] == "manchego-2.1" and {"name", "description", "release_date"} <= set(m["models"][0])


def test_http_422_bodies_carry_markers(client):
    r = client.post("/v1/systemone", json={"state": STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": menu(300)}}})
    assert r.status_code == 422 and MARKER_TOO_MANY_OPTIONS in r.text
    assert r.json()["detail"]["error_type"] == "invalid_request"
    r = client.post("/v1/systemone", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_http_api_keys():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from manchego_serve.server import create_app
    c = TestClient(create_app(decider(), api_keys={"k1"}))
    body = {"state": STATE, "questions": {"n": NOUL}}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"authorization": "Bearer k1"}).status_code == 200


def test_probabilities_finite_and_normalised_on_extreme_logits():
    class Extreme(FakeBackend):
        def logits_batch(self, items):
            return [[1e4 if i == 0 else -1e4 for i in range(len(c))] for _, c in items]
    out = Decider(Extreme()).handle({"state": STATE, "questions": {"c": CHOICE}})["answers"]["c"]
    assert all(math.isfinite(p) for p in out["probabilities"].values())
    assert sum(out["probabilities"].values()) == 1.0 and out["confidence"] == 1.0
