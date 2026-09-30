# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Contract "semif": the prompt Manchego v3 was trained with, SemIf's `direct-options-v1` prompt, with JevBench's
`semif_direct` mapping of a System One question onto it. SemIf (formerly OpenJev, https://github.com/TheoLeeCJ/SemIf) is
MIT-licensed; the system message below is its text (see NOTICE).

Pure functions, no ML import. In manchego-serve this is the renderer of the served contract "semif": contract.py
(`prompt_for`) renders a question with it when SemIf can show the question (2 to 16 options and a nonempty state), and
with the state-first prompt of contract_v2.py otherwise. The code of this module is the training code's renderer,
unchanged; only the documentation was written for publication. manchego-serve's tests/test_semif.py checks both: the
code, and every prompt against fixtures recorded from the training code.

    system  "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only
             its uppercase letter, with no explanation or reasoning."
    user    json.dumps({"evidence": state, "criterion": question,
                        "options": [{"letter": "A", "description": "<key>: <text>"}, ...]}, ensure_ascii=False)

    noul    keys "true" then "false" (fixed order; A = true), text = the criterion's text or "The proposition is <key>."
    choice  keys in the client's order, text = description or the key itself
    score   keys "0".."n-1" in level order, text = the level description
    letters A..P (SemIf validates 2..16 options)

The state is embedded as JSON: a string stays a JSON string, an object or array stays structured (SemIf's direct mode).
Rendered with the chat template, add_generation_prompt=True, enable_thinking=False; the readout is the letters' logits
at the last prompt position.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "direct-options-v1"
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
KINDS = ("noul", "choice", "score")


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def option_keys_and_texts(kind: str, options: list[tuple[str, Any]]) -> list[tuple[str, str]]:
    """(key, description) in rendering order. `options` is [(value, description-or-None)], as contract.options_of returns
    them; for noul the order is forced to true, false whatever order the caller used."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    if kind == "noul":
        given = {str(v): d for v, d in options}
        if set(given) != {"true", "false"}:
            raise ValueError("noul options must be exactly 'true' and 'false'")
        return [(k, f"{k}: " + (_text(given[k]) if given[k] else f"The proposition is {k}.")) for k in ("true", "false")]
    out = [(str(v), f"{v}: " + (_text(d) if d is not None and d != "" else str(v))) for v, d in options]
    if len({k for k, _ in out}) != len(out):
        raise ValueError("duplicate option keys")
    return out


def render(state: Any, question: Any, options: list[tuple[str, Any]], kind: str) -> dict:
    """-> {"messages", "letters", "keys", "prompt_version"}. `letters[i]` is the answer slot for `keys[i]`."""
    if state is None or state == "" or state == {} or state == []:
        raise ValueError("state must be nonempty")
    pairs = option_keys_and_texts(kind, options)
    if not 2 <= len(pairs) <= len(LETTERS):
        raise ValueError(f"{len(pairs)} options; SemIf's contract supports 2..{len(LETTERS)}")
    letters = list(LETTERS[: len(pairs)])
    payload = {"evidence": state, "criterion": _text(question),
               "options": [{"letter": L, "description": d} for L, (_, d) in zip(letters, pairs)]}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    return {"messages": messages, "letters": letters, "keys": [k for k, _ in pairs], "prompt_version": PROMPT_VERSION}


def prompt_text(tok, state: Any, question: Any, options: list[tuple[str, Any]], kind: str) -> str:
    """The full prompt string through the tokenizer's chat template, thinking off (SemIf's encode_prompt)."""
    r = render(state, question, options, kind)
    return tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
