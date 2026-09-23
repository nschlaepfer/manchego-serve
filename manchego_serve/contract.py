# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""The served prompt policy: which prompt a question is rendered with, and how the answer is read out.

Policy (contract "auto"), fixed:
  * up to 26 options: the short prompt (question first; called "ours" in the training code), codes A-Z;
  * 27 to 255 options: the state-first prompt of `contract_v2.py` (vendored, unchanged, from oraculumai/Manchego);
  * one prompt per question, read once: a softmax at temperature 1.0 over the offered option-code tokens at the last
    prompt position; one option order (the client's); confidence = (K * max p - 1) / (K - 1), clamped to [0, 1].

The short prompt is the one Manchego was trained on, byte for byte. Nothing here imports an ML library.
"""

from __future__ import annotations

import json
import math
from typing import Any

from . import contract_v2 as V2

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OURS_TAIL = "Reply with only the letter of the best option."
NOUL_DEFAULT = {"true": "the condition holds", "false": "the condition does not hold"}

SHORT, STATE_FIRST, AUTO = "short", "state_first", "auto"
LIMIT = {SHORT: 26, STATE_FIRST: 255, AUTO: 255}
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
            raise BadRequest("a score needs at least two levels (criteria: a list of level descriptions)")
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


def answer(q: dict, opts: list[tuple[str, str | None]], logits: list[float]) -> dict:
    """The wire answer for one question from its option-code logits (in the client's option order)."""
    values = [v for v, _ in opts]
    probs = {v: 0.0 for v in values}
    for v, pi in zip(values, softmax(logits, TEMPERATURE)):
        probs[v] += pi / PERMUTE
    p_list = [probs[v] for v in values]
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if q["type"] == "choice":
        return {"type": "choice", "choice": max(values, key=probs.__getitem__), "confidence": confidence(p_list), "probabilities": probs}
    levels = q["criteria"]
    return {"type": "score", "score": sum(i * probs[str(i)] for i in range(len(levels))), "confidence": confidence(p_list),
            "legend": {str(i): _text(d) for i, d in enumerate(levels)}, "probabilities": probs}
