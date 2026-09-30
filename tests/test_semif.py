"""Contract "semif" (0.2.0): the SemIf prompt, exactly as the research repository trains and reads it, with the contract v2
overflow; chosen per model by manchego_config.json. golden_semif.json was recorded from the training code alone
(tests/record_semif_golden.py); nothing here imports the training repository."""
import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import TokOnly, load
from manchego_serve import contract as C
from manchego_serve import contract_semif as SR
from manchego_serve import model_config as MC
from manchego_serve.decider import Decider, in_client_order
from test_parity import _template
from test_server import FakeBackend, menu

GOLDEN = load("golden_semif.json")
REQS = {r["id"]: r for r in load("semif_requests.json")["requests"]}
CASES = sorted(GOLDEN["cases"])
# sha256 of code_of() of the training code's renderer, the file recorded in golden_semif.json["renderer"] (sha256
# 86740b8d..., commit 0da1e038). contract_semif.py has its own documentation; its code must be exactly this.
RENDERER_CODE_SHA256 = "84908917ad4c300b891516d5c7d116c5273c21642d7915d7df1476bc5337f235"


def shown_as(g: dict) -> str:
    return C.SEMIF if g["shown"] == "semif" else C.STATE_FIRST


# ------------------------------------------------------------------ provenance
def code_of(source: str) -> str:
    """A module's code: its source without its docstrings (the module's and every function's and class's, located by
    the parser), comments and blank lines. Every other line is kept byte for byte, so two modules with the same code_of
    run the same code whatever their documentation says."""
    tree = ast.parse(source)
    drop = set()
    for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            drop.update(range(first.lineno, first.end_lineno + 1))
    lines = (line.rstrip() for i, line in enumerate(source.splitlines(), 1) if i not in drop)
    return "".join(line + "\n" for line in lines if line.strip() and not line.lstrip().startswith("#"))


def test_renderer_code_is_the_training_code():
    code = code_of(Path(SR.__file__).read_text(encoding="utf-8"))
    assert hashlib.sha256(code.encode()).hexdigest() == RENDERER_CODE_SHA256
    assert SR.PROMPT_VERSION == GOLDEN["renderer"]["prompt_version"] == "direct-options-v1"
    assert [n for n in vars(SR) if callable(vars(SR)[n]) and getattr(vars(SR)[n], "__module__", "") == SR.__name__] == [
        "_text", "option_keys_and_texts", "render", "prompt_text"]


def test_code_of_ignores_documentation_only():
    base = '# header\n"""doc"""\nimport json\n\n\ndef f(x):\n    """doc\n    more"""\n    # note\n    return json.dumps(x)  # kept\n'
    assert code_of(base) == "import json\ndef f(x):\n    return json.dumps(x)  # kept\n"
    assert code_of(base.replace('"""doc"""', '"""other text\n\nover lines"""')) == code_of(base)
    assert code_of(base.replace("json.dumps(x)", "json.dumps(x, indent=1)")) != code_of(base)


def test_renderer_documentation_is_for_publication():
    """The file ships next to the v3 weights: its documentation names no research-repository path and does not call
    itself a draft."""
    doc = " ".join(d for d in [SR.__doc__] + [f.__doc__ for f in (SR.option_keys_and_texts, SR.render, SR.prompt_text)] if d)
    for banned in ("NOT the served contract", "serve_systemone", "scripts/", "runs/", "qwen_decisions", "draft"):
        assert banned not in doc
    assert "https://github.com/TheoLeeCJ/SemIf" in SR.__doc__ and "MIT" in SR.__doc__


def test_notice_credits_semif():
    notice = (Path(__file__).resolve().parents[1] / "NOTICE").read_text(encoding="utf-8")
    assert "SemIf (formerly OpenJev)" in notice and "https://github.com/TheoLeeCJ/SemIf," in notice
    assert "MIT License" in notice and "Copyright (c) 2026 TheoLeeCJ" in notice
    assert "The above copyright notice and this permission notice shall be included in all" in notice
    assert "SemIf-OpenJev" not in notice


def test_renderer_code_against_the_training_file():
    """Optional, when the training repository's renderer is at hand (MANCHEGO_TEST_SEMIF_RENDERER=<its path>): the recorded
    code hash is that file's, and that file is the one golden_semif.json was recorded from."""
    path = os.environ.get("MANCHEGO_TEST_SEMIF_RENDERER")
    if not path:
        pytest.skip("set MANCHEGO_TEST_SEMIF_RENDERER to the training code's semif_contract.py")
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == GOLDEN["renderer"]["sha256"]
    assert hashlib.sha256(code_of(raw.decode("utf-8")).encode()).hexdigest() == RENDERER_CODE_SHA256


def test_fixture_coverage():
    assert set(CASES) == set(REQS)
    shown = [GOLDEN["cases"][c]["shown"] for c in CASES]
    assert shown.count("semif") >= 25 and shown.count("v2") >= 8
    kinds = {(REQS[c]["question"]["type"], GOLDEN["cases"][c]["shown"]) for c in CASES}
    assert {(k, s) for k in ("noul", "choice", "score") for s in ("semif", "v2")} <= kinds
    sizes = {len(GOLDEN["cases"][c]["codes"]) for c in CASES}
    assert {2, 16, 17, 26, 40} <= sizes
    assert {GOLDEN["cases"][c]["semif_refusal"] for c in CASES if GOLDEN["cases"][c]["shown"] == "v2"} >= {
        "state must be nonempty", "17 options; SemIf's contract supports 2..16"}


# ------------------------------------------------------------------ the rendering, byte for byte
@pytest.mark.parametrize("cid", CASES)
def test_prompt_matches_training_without_tokenizer(cid):
    """No weights, no tokenizer: the messages, codes and slot order are the training code's, and wrapped in the (fixed)
    Qwen3.5 chat template they are its prompt byte for byte."""
    r, g = REQS[cid], GOLDEN["cases"][cid]
    q = r["question"]
    p = C.prompt_for(C.SEMIF, r["state"], q, C.options_of(q))
    assert p["contract"] == shown_as(g)
    assert p["messages"] == g["messages"]
    assert p["codes"] == g["codes"]
    assert p["order"] == g["order"]
    assert _template(p["messages"]) == g["prompt"]


@pytest.mark.parametrize("cid", CASES)
def test_ids_and_code_ids_match_training(tokenizer, cid):
    r, g = REQS[cid], GOLDEN["cases"][cid]
    dec = Decider(TokOnly(tokenizer), contract=C.SEMIF)
    q = r["question"]
    assert dec.render(r["state"], q, C.options_of(q)) == (shown_as(g), g["prompt"])
    plan = dec.plan(r["state"], q)
    assert plan["ids"] == g["ids"]
    assert plan["cands"] == g["code_ids"]
    assert plan["contract"] == shown_as(g) and plan["order"] == g["order"]


def test_semif_letters_are_single_distinct_tokens(tokenizer):
    dec = Decider(TokOnly(tokenizer), contract=C.SEMIF)
    ids = [dec._code_id(L) for L in SR.LETTERS]
    assert len(set(ids)) == len(SR.LETTERS) == C.SEMIF_MAX_OPTIONS == 16


# ------------------------------------------------------------------ the overflow rule
@pytest.mark.parametrize("kind,state,n,shows", [
    ("choice", "s", 2, True), ("choice", "s", 16, True), ("choice", "s", 17, False), ("choice", "s", 26, False),
    ("score", "s", 16, True), ("score", "s", 17, False), ("noul", "s", 2, True),
    ("noul", "", 2, False), ("noul", {}, 2, False), ("noul", [], 2, False), ("noul", None, 2, False),
    ("noul", " ", 2, True), ("noul", 0, 2, True), ("noul", False, 2, True), ("noul", [None], 2, True), ("noul", {"a": None}, 2, True),
    ("rank", "s", 3, False),
])
def test_semif_can_show(kind, state, n, shows):
    opts = [("true", None), ("false", None)] if kind == "noul" else [(str(i), None) for i in range(n)]
    assert C.semif_can_show(kind, state, opts) is shows


def test_noul_needs_true_and_false():
    assert not C.semif_can_show("noul", "s", [("yes", None), ("no", None)])
    assert C.semif_can_show("noul", "s", [("false", None), ("true", None)])


def test_overflow_never_uses_the_short_prompt():
    """Under semif, 17 to 26 options go to the state-first prompt (as in the reads), never to the short prompt."""
    for n in (17, 20, 26, 27, 255):
        q = {"type": "choice", "instructions": "x", "criteria": menu(n)}
        p = C.prompt_for(C.SEMIF, "s", q, C.options_of(q))
        assert p["contract"] == C.STATE_FIRST and p["codes"] == C.codes_for(C.STATE_FIRST, n)
        assert p["messages"] == C.render_v2("s", q, C.options_of(q))


def test_auto_prompt_for_is_the_v012_rendering():
    for n in (2, 26, 27, 255):
        q = {"type": "choice", "instructions": "x", "criteria": menu(n)}
        opts = C.options_of(q)
        p = C.prompt_for(C.AUTO, "s", q, opts)
        c = C.contract_for(n)
        assert p == {"contract": c, "messages": C.RENDER[c]("s", q, opts), "codes": C.codes_for(c, n), "order": list(range(n))}
    with pytest.raises(ValueError):
        C.prompt_for("short", "s", {"type": "noul", "instructions": "x"}, [("true", None), ("false", None)])


def test_in_client_order():
    z = [0.5, 1.5, -2.0]
    assert in_client_order(z, [0, 1, 2]) is z
    assert in_client_order(z, [2, 0, 1]) == [1.5, -2.0, 0.5]


# ------------------------------------------------------------------ served end to end (fake backend)
def test_semif_decider_end_to_end():
    body = {"state": "Order 7: the kettle arrived with a cracked lid.",
            "questions": {"n": {"type": "noul", "instructions": "Damaged?", "criteria": {"false": "intact", "true": "damaged"}},
                          "c": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "refunds", "shipping": None}},
                          "s": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
                          "big": {"type": "choice", "instructions": "x", "criteria": menu(17)}}}
    out = Decider(FakeBackend(), contract=C.SEMIF).handle(body)
    m = out["manchego"]
    assert m["contract"] == "semif"
    assert m["contract_by_question"] == {"n": "semif", "c": "semif", "s": "semif", "big": "state_first"}
    # FakeBackend's logits fall with the slot: slot A is always the largest, i.e. true for noul and the first option otherwise
    assert out["answers"]["n"]["noul"] > 0.5
    assert out["answers"]["c"]["choice"] == "returns" and list(out["answers"]["c"]["probabilities"]) == ["returns", "shipping"]
    assert list(out["answers"]["s"]["probabilities"]) == ["0", "1", "2"]
    assert len(out["answers"]["big"]["probabilities"]) == 17
    empty = Decider(FakeBackend(), contract=C.SEMIF).handle({"state": "", "questions": {"n": body["questions"]["n"]}})
    assert empty["manchego"]["contract_by_question"] == {"n": "state_first"}


def test_semif_keeps_the_limits_and_refusals():
    d = Decider(FakeBackend(), contract=C.SEMIF)
    d.handle({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": menu(255)}}})
    with pytest.raises(C.BadRequest, match="options per choice"):
        d.handle({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": menu(256)}}})
    with pytest.raises(C.BadRequest, match="a choice needs at least two options"):
        d.handle({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}}})


def test_unknown_contract_is_refused():
    with pytest.raises(ValueError):
        Decider(FakeBackend(), contract="short")


# ------------------------------------------------------------------ manchego_config.json
V21_CONFIG = {"model": "Manchego", "version": "v2.1", "base_model": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
              "served_policy": "short prompt up to 26 options, state_first beyond; one prompt per forward pass", "temperature": 1.0}


def write(tmp_path, data) -> Path:
    (tmp_path / MC.FILE).write_text(data if isinstance(data, str) else json.dumps(data))
    return tmp_path


def test_model_config_defaults_to_auto(tmp_path):
    none = MC.read(tmp_path)
    assert none.contract == C.AUTO and not none.contract_declared and none.file is None
    assert "no manchego_config.json" in none.describe()["contract_source"]
    v21 = MC.read(write(tmp_path, V21_CONFIG))            # the published v2.1 file has no `contract` field
    assert v21.contract == C.AUTO and not v21.contract_declared and v21.sha256 and v21.served_name is None
    assert v21.describe()["served_name"] == "manchego-2.1"


def test_model_config_semif(tmp_path):
    cfg = MC.read(write(tmp_path, {**V21_CONFIG, "version": "v3", "contract": "semif", "served_name": "manchego-3"}))
    assert cfg.contract == C.SEMIF and cfg.contract_declared and cfg.served_name == "manchego-3"
    assert cfg.describe()["contract_source"] == MC.FILE


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", json.dumps({"contract": "short"}), json.dumps({"contract": None}),
                                 json.dumps({"served_name": ""}), json.dumps({"served_name": 3})])
def test_model_config_refuses_what_it_cannot_read(tmp_path, bad):
    with pytest.raises(MC.ModelConfigError):
        MC.read(write(tmp_path, bad))


def test_auto_config_serves_the_v012_prompts(tmp_path):
    """A folder whose manchego_config.json names no contract is served with the v0.1.x prompts (golden_prompts.json)."""
    from conftest import BY_ID
    cfg = MC.read(write(tmp_path, V21_CONFIG))
    golden = load("golden_prompts.json")["golden"]
    for rid, qs in golden.items():
        body = BY_ID[rid]["body"]
        for name, g in qs.items():
            q = body["questions"][name]
            p = C.prompt_for(cfg.contract, body["state"], q, C.options_of(q))
            assert p["contract"] == g["contract"] and p["codes"] == g["codes"] and _template(p["messages"]) == g["prompt"]


# ------------------------------------------------------------------ command line and HTTP
def test_contract_command_line(monkeypatch):
    from manchego_serve.server import parse_args
    monkeypatch.delenv("MANCHEGO_CONTRACT", raising=False)
    assert parse_args([]).contract == "model"
    assert parse_args(["--contract", "semif"]).contract == "semif"
    monkeypatch.setenv("MANCHEGO_CONTRACT", "auto")
    assert parse_args([]).contract == "auto"
    with pytest.raises(SystemExit):
        parse_args(["--contract", "short"])


def test_http_under_semif():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from manchego_serve.server import create_app
    d = Decider(FakeBackend(), contract=C.SEMIF, model_name="manchego-3",
                info={"model_label": "Manchego v3", "release_date": "2026-10-01", "model_config": {"contract": "semif"}})
    c = TestClient(create_app(d))
    h = c.get("/healthz").json()
    assert h["contract"] == "semif" and h["model"] == "manchego-3" and h["model_config"]["contract"] == "semif"
    assert h["limits"]["semif_prompt_options"] == 16 and "short_prompt_options" not in h["limits"]
    m = c.get("/v1/models").json()
    assert m["models"][0]["name"] == "manchego-3" and m["models"][0]["release_date"] == "2026-10-01"
    assert m["models"][0]["description"].startswith("Manchego v3 ")
    r = c.post("/v1/systemone", json={"state": "s", "questions": {"n": {"type": "noul", "instructions": "x"}}}).json()
    assert r["model"] == "manchego-3" and r["manchego"]["contract_by_question"] == {"n": "semif"}
