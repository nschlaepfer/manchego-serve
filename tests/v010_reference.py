"""The v0.1.0 readout, frozen: `softmax`, `confidence` and `answer` copied verbatim from manchego_serve/contract.py at tag
v0.1.0 (commit 4c7332f), with its constants inlined (TEMPERATURE = 1.0, PERMUTE = 1). Tests compare the current code
under `--temperature-map off` against this, byte for byte."""
import json
import math
from typing import Any

TEMPERATURE = 1.0
PERMUTE = 1


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1)


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
