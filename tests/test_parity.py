"""Prompt parity with the reference implementation (the server that produced the published numbers).

golden_prompts.json holds, for every question of the invented requests in requests.json, the reference's chat-templated
prompt, its token ids, the contract it chose and the option-code token ids. This package must reproduce all of them.
"""
from pathlib import Path

import pytest

from conftest import BY_ID, TokOnly, load
from manchego_serve import contract as C
from manchego_serve.decider import Decider

GOLDEN = load("golden_prompts.json")["golden"]
CASES = [(rid, name) for rid, qs in GOLDEN.items() for name in qs]
PUBLIC_CONTRACT_V2_BLOB = "a6c586acbf861ce55e4a4ed6dfd78d7d4515f5d5"  # git blob id of contract_v2.py at oraculumai/Manchego@77403228 (tag v2.1)


def test_fixture_coverage():
    kinds = {BY_ID[rid]["body"]["questions"][n]["type"] for rid, n in CASES}
    sizes = {len(GOLDEN[rid][n]["codes"]) for rid, n in CASES}
    assert kinds == {"noul", "choice", "score"}
    assert {2, 5, 26, 27, 100, 255} <= sizes
    assert {GOLDEN[rid][n]["contract"] for rid, n in CASES} == {"short", "state_first"}
    assert len(BY_ID) >= 40


def test_vendored_contract_v2_is_the_published_file():
    """manchego_serve/contract_v2.py is byte-identical to the file published with the v2.1 weights."""
    path = Path(C.V2.__file__)
    data = path.read_bytes()
    import hashlib
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == PUBLIC_CONTRACT_V2_BLOB


def _template(msgs):
    """Qwen3.5's chat template for plain-text messages, generation prompt added, thinking disabled."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.mark.parametrize("rid,name", CASES)
def test_messages_match_reference_without_tokenizer(rid, name):
    """Runs with no weights and no tokenizer: the rendered messages, wrapped in the (fixed) chat template, are the
    reference prompt byte for byte, and the contract and codes are the reference's."""
    body, g = BY_ID[rid]["body"], GOLDEN[rid][name]
    q = body["questions"][name]
    opts = C.options_of(q)
    c = C.contract_for(len(opts))
    assert c == g["contract"]
    assert C.codes_for(c, len(opts)) == g["codes"]
    assert _template(C.RENDER[c](body["state"], q, opts)) == g["prompt"]


@pytest.mark.parametrize("rid,name", CASES)
def test_prompt_ids_and_code_ids_match_reference(tokenizer, rid, name):
    body, g = BY_ID[rid]["body"], GOLDEN[rid][name]
    dec = Decider(TokOnly(tokenizer))
    q = body["questions"][name]
    c, text = dec.render(body["state"], q, C.options_of(q))
    assert (c, text) == (g["contract"], g["prompt"])
    plan = dec.plan(body["state"], q)
    assert plan["ids"] == g["ids"]
    assert plan["cands"] == g["code_ids"]


def test_every_code_is_one_distinct_token(tokenizer):
    dec = Decider(TokOnly(tokenizer))
    for contract, n in ((C.SHORT, 26), (C.STATE_FIRST, 255)):
        ids = [dec._code_id(code) for code in C.codes_for(contract, n)]
        assert len(set(ids)) == n
