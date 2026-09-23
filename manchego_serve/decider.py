# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""One System One request in, one response out. The model is reached only through a backend's `logits_batch`."""

from __future__ import annotations

from typing import Any

from . import __version__
from .contract import (AUTO, CONFIDENCE_DEFINITION, MAX_OPTIONS, PERMUTE, RENDER, TEMPERATURE, BadRequest, answer,
                       codes_for, contract_for, options_of)

MODEL_NAME = "manchego-2.1"
DEFAULT_MAX_PROMPT_TOKENS = 32768     # per question: state + question + options, after the chat template
DEFAULT_MAX_QUESTIONS = 1024          # per request; every question re-reads the state
DEFAULT_BATCH_TOKENS = 16384          # padded-token budget of one microbatch (execution "batched" only)
EXECUTIONS = ("sequential", "batched")


class Decider:
    """Plans, scores and answers the questions of a request.

    execution "sequential" (default): one prompt per forward pass, no padding. A question's probabilities then do not
    depend on the other questions of the request or on how many there are. This is how the published numbers were read.
    execution "batched": all prompts of a request in right-padded microbatches, longest first, at most `batch_tokens`
    padded tokens each. Faster; at bf16 or 8 bits the batch shape moves probabilities slightly.
    Either way every question is its own sequence (its own row): no question can attend to another.
    """

    def __init__(self, backend, execution: str = "sequential", batch_tokens: int = DEFAULT_BATCH_TOKENS,
                 max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS, max_questions: int = DEFAULT_MAX_QUESTIONS,
                 info: dict | None = None):
        if execution not in EXECUTIONS:
            raise ValueError(f"unknown execution {execution!r}; expected one of {EXECUTIONS}")
        self.b, self.execution, self.batch_tokens = backend, execution, int(batch_tokens)
        self.max_prompt_tokens, self.max_questions = int(max_prompt_tokens), int(max_questions)
        self.info = dict(info or {})
        self._code_ids: dict[str, int] = {}

    # ------------------------------------------------------------------ prompts
    def _code_id(self, code: str) -> int:
        if code not in self._code_ids:
            enc = self.b.tok.encode(code, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"option code {code!r} is not a single token for this tokenizer")
            self._code_ids[code] = enc[0]
        return self._code_ids[code]

    def render(self, state: Any, q: dict, opts: list[tuple[str, str | None]]) -> tuple[str, str]:
        """(contract, chat-templated prompt text)."""
        c = contract_for(len(opts))
        msgs = RENDER[c](state, q, opts)
        return c, self.b.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def plan(self, state: Any, q: dict) -> dict:
        """Everything about one question that does not need the model."""
        if not isinstance(q, dict) or "instructions" not in q:
            raise BadRequest("a question needs type and instructions")
        opts = options_of(q)
        if len(opts) > MAX_OPTIONS:
            raise BadRequest(f"{len(opts)} options; this server supports at most {MAX_OPTIONS} options per choice "
                             "(score levels count as options). Larger menus are refused, not truncated.")
        if len({v for v, _ in opts}) != len(opts):
            raise BadRequest("duplicate option values")
        c, text = self.render(state, q, opts)
        ids = self.b.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_prompt_tokens:
            raise BadRequest(f"prompt is {len(ids)} tokens; the maximum context length is {self.max_prompt_tokens} tokens per question "
                             "(state + question + options). Nothing is truncated: send less state.")
        cands = [self._code_id(code) for code in codes_for(c, len(opts))]
        return {"q": q, "opts": opts, "contract": c, "ids": list(ids), "cands": cands}

    # ------------------------------------------------------------------ scoring
    def score_many(self, encoded: list[tuple[list[int], list[int]]]) -> tuple[list[list[float]], int]:
        """Option-code logits for every prompt, and how many forward passes it took."""
        if self.execution == "sequential":
            return [self.b.logits_batch([e])[0] for e in encoded], len(encoded)
        order = sorted(range(len(encoded)), key=lambda k: -len(encoded[k][0]))
        out: list = [None] * len(encoded)
        calls, k = 0, 0
        while k < len(order):
            width = len(encoded[order[k]][0])
            take = max(1, min(len(order) - k, self.batch_tokens // max(width, 1)))
            idx = order[k: k + take]
            for j, z in zip(idx, self.b.logits_batch([encoded[j] for j in idx])):
                out[j] = z
            calls, k = calls + 1, k + take
        return out, calls

    # ------------------------------------------------------------------ the request
    def handle(self, body: Any) -> dict:
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict) or not body["questions"]:
            raise BadRequest("expected {state, model, questions: {name: question}}")
        if len(body["questions"]) > self.max_questions:
            raise BadRequest(f"{len(body['questions'])} questions is too many tokens for one request: every question re-reads the state, "
                             f"and this server accepts at most {self.max_questions} questions per request. Nothing is truncated: split the request.")
        plans = {}
        for name, q in body["questions"].items():            # question names are for the caller; they never reach the model
            try:
                plans[name] = self.plan(body["state"], q)
            except BadRequest as e:
                raise BadRequest(f"question {name!r}: {e}") from None
        names = list(plans)
        logits, calls = self.score_many([(plans[n]["ids"], plans[n]["cands"]) for n in names])
        answers = {n: answer(plans[n]["q"], plans[n]["opts"], z) for n, z in zip(names, logits)}
        return {"model": MODEL_NAME, "answers": answers,
                "usage": {"input_tokens": sum(len(plans[n]["ids"]) for n in names), "output_tokens": 0},
                "manchego": {**self.describe(), "contract_by_question": {n: plans[n]["contract"] for n in names}, "forward_passes": calls}}

    def describe(self) -> dict:
        """What produced the numbers. Reported in every response and by /healthz."""
        return {"server": f"manchego-serve {__version__}", "backend": self.b.name, "precision": getattr(self.b, "precision", None),
                "contract": AUTO, "temperature": TEMPERATURE, "permute": PERMUTE, "confidence_definition": CONFIDENCE_DEFINITION,
                "execution": self.execution, "model_repo": self.info.get("repo"), "model_revision": self.info.get("revision"),
                "weights_sha256": self.info.get("weights_sha256"), "weights_verified": self.info.get("weights_verified"),
                "support_files_verified": self.info.get("support_files_verified"),
                "isolation": "one sequence per question; no question can attend to another"}
