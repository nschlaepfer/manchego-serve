# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""The served prompt policy: which prompt a question is rendered with, and how the answer is read out.

Policy (contract "auto"), fixed:
  * up to 26 options: the short prompt (question first; called "ours" in the training code), codes A-Z;
  * 27 to 255 options: the state-first prompt of `contract_v2.py` (vendored, unchanged, from oraculumai/Manchego);
  * one prompt per question, read once: a softmax over the offered option-code tokens at the last prompt position, at
    the temperature of the question's type (temperature.py: the fitted v2.1 map by default; 1.0 for every type under
    `--temperature-map off`, as in v0.1.0); one option order (the client's); the chosen option is the largest logit;
    confidence = (K * max p - 1) / (K - 1), clamped to [0, 1].

The short prompt is the one Manchego was trained on, byte for byte.

Policy "semif" (0.2.0, for models trained under the SemIf prompt contract; chosen per model by the `contract` field of the
model folder's manchego_config.json, see model_config.py):
  * a question SemIf can show (2 to 16 options, a nonempty state, noul as true/false) is rendered by `contract_semif.py`
    (SemIf `direct-options-v1`: a system message and one JSON user message, letters A-P, noul as true then false), exactly
    as the research repository's training and development-read code renders it;
  * every other question (17 to 255 options, or an empty state: "", {}, [] or null) falls back to the state-first prompt of
    `contract_v2.py`, codes A-Z then two-letter codes: the v2 overflow rule of those development reads. The short prompt
    is never used under "semif", not even for 17 to 26 options.
  The readout, the temperature, the choice and the confidence are the same as under "auto".

Nothing here imports an ML library.
"""

from __future__ import annotations

import json
import math
from typing import Any

from . import contract_semif as SEMIF_R
from . import contract_v2 as V2

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OURS_TAIL = "Reply with only the letter of the best option."
NOUL_DEFAULT = {"true": "the condition holds", "false": "the condition does not hold"}

SHORT, STATE_FIRST, AUTO = "short", "state_first", "auto"
SEMIF = "semif"                        # both a policy (SemIf, with the v2 overflow) and the name of the SemIf prompt itself
POLICIES = (AUTO, SEMIF)
LIMIT = {SHORT: 26, STATE_FIRST: 255, AUTO: 255}
SEMIF_MAX_OPTIONS = len(SEMIF_R.LETTERS)   # 16: what the SemIf prompt itself shows; the policy takes up to 255 through the overflow
MAX_OPTIONS = LIMIT[AUTO]
TEMPERATURE = 1.0
PERMUTE = 1
CONFIDENCE_DEFINITION = "(K * max(p) - 1) / (K - 1), clamped to [0, 1]; K = number of offered options"


class BadRequest(ValueError):
    """A request the server will not score. Returned as HTTP 422 with this message."""


def _text(x: Any) -> str:
    """Instructions and descriptions may arrive as JSON values; they are rendered deterministically."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


def options_of(q: dict) -> list[tuple[str, str | None]]:
    """(value, description) in the order the client gave them."""
    kind, crit = q.get("type"), q.get("criteria")
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        return [("true", crit.get("true")), ("false", crit.get("false"))]
    if kind == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            raise BadRequest("a choice needs at least two options (criteria: an object of option value -> description or null)")
        return [(str(k), None if v is None else _text(v)) for k, v in crit.items()]
    if kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise BadRequest("a score needs at least two levels (criteria: a list of level descriptions). The System One contract says "
                             f"a score takes 2 to 10 levels; this server accepts 2 to {LIMIT[AUTO]}.")
        return [(str(i), _text(d)) for i, d in enumerate(crit)]
    raise BadRequest(f"unknown question type {kind!r}; expected noul, choice or score")


def render_ours(state: Any, q: dict, opts: list[tuple[str, str | None]]) -> list[dict]:
    """The short prompt, as Manchego was trained on it."""
    kind = q["type"]
    lines = [_text(q["instructions"]).strip(), "", "State:", state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2), "", "Options:"]
    for L, (v, d) in zip(LETTERS, opts):
        if kind == "noul":
            text = f"{'yes' if v == 'true' else 'no'} — {d or NOUL_DEFAULT[v]}"
        elif kind == "score":
            text = f"level {v} — {d}"
        else:
            text = f"{v} — {d}" if d else v
        lines.append(f"{L} = {text}")
    lines += ["", OURS_TAIL]
    return [{"role": "user", "content": "\n".join(lines)}]


def render_v2(state: Any, q: dict, opts: list[tuple[str, str | None]]) -> list[dict]:
    """The state-first prompt, rendered by the vendored contract_v2.py."""
    return V2.messages(state, q["type"], q["instructions"], opts, codes_for(STATE_FIRST, len(opts)))


RENDER = {SHORT: render_ours, STATE_FIRST: render_v2}


def contract_for(n_options: int) -> str:
    """Contract "auto": the short prompt up to 26 options, the state-first prompt for 27..255."""
    return SHORT if n_options <= LIMIT[SHORT] else STATE_FIRST


def codes_for(contract: str, n: int) -> list[str]:
    if contract == STATE_FIRST:
        return V2.codebook(n)
    return list(LETTERS[:n])


def semif_can_show(kind: str, state: Any, opts: list[tuple[str, Any]]) -> bool:
    """Whether the SemIf prompt shows this question, exactly as the development reads decide it
    (scripts/lean_reads_score.py `semif_can_show`): 2..16 options of a known kind, a nonempty state, and for noul exactly
    the values true and false. Everything else goes to contract v2."""
    n = len(opts)
    if kind not in SEMIF_R.KINDS or not 2 <= n <= SEMIF_MAX_OPTIONS:
        return False
    if state is None or state == "" or state == {} or state == []:
        return False
    return kind != "noul" or sorted(str(v) for v, _ in opts) == ["false", "true"]


def prompt_for(policy: str, state: Any, q: dict, opts: list[tuple[str, str | None]]) -> dict:
    """The prompt one question is read with under a policy: {contract, messages, codes, order}.

    `codes[j]` is the option-code token read for slot j, and `order[j]` the index (in the client's option order) of the
    option in slot j. Under "auto" the slots are the client's order. Under "semif" they are SemIf's keys, which are also the
    client's order: noul is always true then false (options_of gives it that way), choice and score keep their order.
    """
    if policy == AUTO:
        c = contract_for(len(opts))
        return {"contract": c, "messages": RENDER[c](state, q, opts), "codes": codes_for(c, len(opts)), "order": list(range(len(opts)))}
    if policy != SEMIF:
        raise ValueError(f"unknown contract policy {policy!r}; expected one of {POLICIES}")
    if not semif_can_show(q["type"], state, opts):
        return {"contract": STATE_FIRST, "messages": render_v2(state, q, opts), "codes": codes_for(STATE_FIRST, len(opts)),
                "order": list(range(len(opts)))}
    r = SEMIF_R.render(state, q["instructions"], opts, q["type"])
    index = {str(v): i for i, (v, _) in enumerate(opts)}
    return {"contract": SEMIF, "messages": r["messages"], "codes": r["letters"], "order": [index[k] for k in r["keys"]]}


def softmax(z: list[float], T: float = TEMPERATURE) -> list[float]:
    m = max(z)
    e = [math.exp((x - m) / max(T, 1e-4)) for x in z]
    s = sum(e)
    return [x / s for x in e]


def confidence(p: list[float]) -> float:
    """(K * max p - 1) / (K - 1), clamped to [0, 1]: 0 when uniform, 1 when certain."""
    if len(p) < 2:
        return 1.0
    return max(0.0, min(1.0, (len(p) * max(p) - 1.0) / (len(p) - 1.0)))


def argmax_first(z: list[float]) -> int:
    """Index of the largest logit; the first one in the client's order on an exact tie. It does not depend on T."""
    best = 0
    for i in range(1, len(z)):
        if z[i] > z[best]:
            best = i
    return best


def answer(q: dict, opts: list[tuple[str, str | None]], logits: list[float], T: float = TEMPERATURE) -> dict:
    """The wire answer for one question from its option-code logits (in the client's option order).

    `T` is the temperature of this question's type (temperature.py). At T = 1.0 this is the v0.1.0 readout, byte for byte.
    The chosen option is read from the logits, so no temperature can change it.
    """
    values = [v for v, _ in opts]
    probs = {v: 0.0 for v in values}
    for v, pi in zip(values, softmax(logits, T)):
        probs[v] += pi / PERMUTE
    p_list = [probs[v] for v in values]
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if q["type"] == "choice":
        return {"type": "choice", "choice": values[argmax_first(logits)], "confidence": confidence(p_list), "probabilities": probs}
    levels = q["criteria"]
    return {"type": "score", "score": sum(i * probs[str(i)] for i in range(len(levels))), "confidence": confidence(p_list),
            "legend": {str(i): _text(d) for i, d in enumerate(levels)}, "probabilities": probs}
