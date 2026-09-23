# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Manchego behind the System One wire contract.

    POST /v1/systemone   {"state": <str | JSON>, "model": <any str>, "questions": {<name>: {"type": "noul"|"choice"|"score",
                          "instructions": <str | JSON>, "criteria": <object | list | absent>}}}
    GET  /v1/models
    GET  /healthz

The server never opens an outbound connection: HF_HUB_OFFLINE=1 is forced before any model code is imported, and the
weights must already be on disk (a directory, or a hub id already in the local cache).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:  # module level: FastAPI resolves the `Request` annotation from this module's globals
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - the contract and backends work without the web stack
    FastAPI = Request = JSONResponse = None

from .contract import BadRequest, LIMIT, MAX_OPTIONS
from .decider import DEFAULT_BATCH_TOKENS, DEFAULT_MAX_PROMPT_TOKENS, DEFAULT_MAX_QUESTIONS, EXECUTIONS, MODEL_NAME, Decider

RELEASE_DATE = "2026-09-21"
ALIASES = ("manchego-latest",)
WARMUP_REQUEST = {"model": MODEL_NAME, "state": "Order 1182: two mugs, one arrived broken. The customer attached a photo.",
                  "questions": {"refund": {"type": "noul", "instructions": "Should the broken mug be refunded?"},
                                "route": {"type": "choice", "instructions": "Which team should handle this?",
                                          "criteria": {"returns": "refunds and replacements", "shipping": None, "billing": None}}}}


def force_offline() -> None:
    """Called before transformers / huggingface_hub / mlx_lm are imported, so their constants see it."""
    for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                 "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"}.items():
        os.environ[k] = v
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def models_payload() -> dict:
    """`models` is the shape TypeSafe's SDKs validate; `data` is an OpenAI-style list. The request's `model` field may be
    any string: this server serves one model and names it in every response."""
    names = (MODEL_NAME,) + ALIASES
    return {"models": [{"name": n, "description": "Manchego v2.1 decision model (Qwen3.5-4B fine-tune), System One wire contract"
                        + ("" if n == MODEL_NAME else f"; alias of {MODEL_NAME}"), "release_date": RELEASE_DATE} for n in names],
            "data": [{"id": MODEL_NAME, "aliases": list(ALIASES)}]}


class Overloaded(Exception):
    def __init__(self, message: str, retry_after_s: float):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class Gate:
    """One model worker; at most `max_queue` requests may wait for it, none longer than `queue_timeout_s` (then 529 with
    Retry-After). The model runs in a worker thread so /healthz answers while it is busy."""

    def __init__(self, max_queue: int = 32, queue_timeout_s: float = 600.0):
        self.max_queue, self.queue_timeout_s = max_queue, queue_timeout_s
        self.slot, self.pool = asyncio.Semaphore(1), ThreadPoolExecutor(max_workers=1, thread_name_prefix="manchego-model")
        self.waiting, self.recent_ms = 0, []

    def retry_after_s(self) -> float:
        per = (sum(self.recent_ms) / len(self.recent_ms) / 1000) if self.recent_ms else 0.5
        return round(min(30.0, max(0.2, per * (self.waiting + 1))), 2)

    async def run(self, fn):
        if self.waiting >= self.max_queue:
            raise Overloaded(f"{self.waiting} requests are already waiting (limit {self.max_queue})", self.retry_after_s())
        self.waiting += 1
        try:
            await asyncio.wait_for(self.slot.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError:
            raise Overloaded(f"waited {self.queue_timeout_s:g} s for the model without reaching it", self.retry_after_s()) from None
        finally:
            self.waiting -= 1
        try:
            t0 = time.perf_counter()
            out = await asyncio.get_running_loop().run_in_executor(self.pool, fn)
            self.recent_ms = (self.recent_ms + [(time.perf_counter() - t0) * 1000])[-50:]
            return out
        finally:
            self.slot.release()


def create_app(decider: Decider, api_keys: set[str] | None = None, max_queue: int = 32, queue_timeout_s: float = 600.0,
               health_extra: dict | None = None):
    if FastAPI is None:
        raise RuntimeError("install fastapi and uvicorn to serve")
    app = FastAPI(title="Manchego System One", docs_url=None, redoc_url=None, openapi_url=None)
    box: dict[str, Any] = {"gate": None}
    keys = set(api_keys or ())

    def gate() -> Gate:  # created inside the running event loop
        if box["gate"] is None:
            box["gate"] = Gate(max_queue, queue_timeout_s)
        return box["gate"]

    def err(code: int, kind: str, msg: str, headers: dict | None = None):
        return JSONResponse(status_code=code, content={"detail": {"error_type": kind, "message": msg}}, headers=headers)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, **decider.describe(), "model": MODEL_NAME,
                "limits": {"options_per_question": MAX_OPTIONS, "short_prompt_options": LIMIT["short"],
                           "max_prompt_tokens_per_question": decider.max_prompt_tokens, "max_questions_per_request": decider.max_questions},
                "weight_files": decider.info.get("weight_files"), "support_files": decider.info.get("support_files"), **(health_extra or {})}

    @app.get("/v1/models")
    async def models():
        return models_payload()

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        if keys:
            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("bearer ") or auth[7:].strip() not in keys:
                return err(401, "authentication_error", "Missing or invalid API key")
        try:
            body = await request.json()
        except Exception:
            return err(422, "invalid_request", "body is not JSON")
        try:
            out = await gate().run(lambda: decider.handle(body))
        except Overloaded as e:
            return err(529, "overloaded_error", f"{e}. Retry after {e.retry_after_s:g} s.",
                       {"retry-after": str(max(1, math.ceil(e.retry_after_s))), "retry-after-ms": str(int(e.retry_after_s * 1000))})
        except BadRequest as e:
            return err(422, "invalid_request", str(e))
        except Exception as e:  # the model worker failed; say so, keep serving
            return err(500, "internal_error", f"{type(e).__name__}: {e}")
        return JSONResponse(content=out)

    return app


def warm_up(decider: Decider) -> dict:
    """Score a fixed invented request twice before accepting traffic: kernels are compiled / selected on the first pass,
    and the second shows whether this runtime repeats itself bit for bit."""
    t0 = time.perf_counter()
    a = decider.handle(WARMUP_REQUEST)["answers"]
    t1 = time.perf_counter()
    b = decider.handle(WARMUP_REQUEST)["answers"]
    t2 = time.perf_counter()
    return {"first_ms": round((t1 - t0) * 1000, 1), "second_ms": round((t2 - t1) * 1000, 1), "repeat_identical": a == b}


def build(backend: str, model: str | None, revision: str | None, execution: str = "sequential", dtype: str = "auto",
          device: str = "auto", batch_tokens: int = DEFAULT_BATCH_TOKENS, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
          max_questions: int = DEFAULT_MAX_QUESTIONS, hash_weights: bool = True, log=print) -> Decider:
    force_offline()
    from .backends import load_backend
    from .weights import DEFAULT_REPO, resolve, identify
    model = model or DEFAULT_REPO[backend]
    model_dir, revision = resolve(model, revision)
    is_dir = Path(model).is_dir()
    info: dict[str, Any] = {"repo": None if is_dir else model, "revision": revision, "weights_sha256": None, "weights_verified": None,
                            "support_files_verified": None}
    if hash_weights:
        t0 = time.perf_counter()
        ident = identify(model_dir)
        pin = ident["weights_match_pin"]
        if is_dir and pin:
            info["repo"] = pin["repo"]
            info["revision"] = info["revision"] or pin["revision"]
        verified = bool(pin) and pin["repo"] == info["repo"] and pin["revision"] == info["revision"]
        info.update(weights_sha256=ident["weights_sha256"], weights_verified=verified, weight_files=ident["files"],
                    support_files_verified=verified and ident["support_files_match_pin"], support_files=ident["support_files"])
        log(f"weights: sha256 {ident['weights_sha256']} ({time.perf_counter() - t0:.1f} s); "
            + (f"these are the published bytes of {pin['repo']}@{pin['revision']} ({pin['tag']})" if pin else "NOT the bytes of any published v2.1 revision"))
        if pin and not verified:
            log(f"warning: declared {info['repo']}@{info['revision']}, but the bytes are {pin['repo']}@{pin['revision']}")
        if pin and not ident["support_files_match_pin"]:
            log("warning: the chat template, tokenizer or config files are NOT the published ones of that revision; "
                "prompts or numbers may differ from the published policy")
    t0 = time.perf_counter()
    b = load_backend(backend, model_dir, dtype=dtype, device=device)
    log(f"loaded {backend} ({getattr(b, 'precision', '?')}) from {model_dir} in {time.perf_counter() - t0:.1f} s")
    return Decider(b, execution=execution, batch_tokens=batch_tokens, max_prompt_tokens=max_prompt_tokens,
                   max_questions=max_questions, info=info)


def main(argv: list[str] | None = None) -> int:
    force_offline()
    ap = argparse.ArgumentParser(prog="manchego-serve", description="Serve Manchego v2.1 on the System One wire contract, offline.")
    ap.add_argument("--backend", choices=("torch", "mlx"), default=os.environ.get("MANCHEGO_BACKEND", "torch"))
    ap.add_argument("--model", default=os.environ.get("MANCHEGO_MODEL"), help="weights directory, or a hub id already in the "
                    "local cache (default: $MANCHEGO_MODEL, else oraculumai/Manchego for torch, oraculumai/Manchego-MLX-8bit for mlx)")
    ap.add_argument("--revision", default=os.environ.get("MANCHEGO_REVISION"),
                    help="full commit sha of --model (default: $MANCHEGO_REVISION, else the pinned v2.1 revision)")
    ap.add_argument("--host", default=os.environ.get("MANCHEGO_HOST", "127.0.0.1"), help="default: $MANCHEGO_HOST, else 127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MANCHEGO_PORT", "8000")), help="default: $MANCHEGO_PORT, else 8000")
    ap.add_argument("--execution", choices=EXECUTIONS, default="sequential",
                    help="sequential (default): one prompt per forward pass, as the published numbers were read; "
                         "batched: padded microbatches, faster, moves probabilities slightly at bf16/8-bit")
    ap.add_argument("--batch-tokens", type=int, default=DEFAULT_BATCH_TOKENS, help="padded-token budget per microbatch (batched only)")
    ap.add_argument("--dtype", default="auto", help="torch only: auto (bfloat16 on CUDA, float32 on CPU), bfloat16, float32")
    ap.add_argument("--device", default="auto", help="torch only: auto, cuda:N or cpu")
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    ap.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    ap.add_argument("--max-queue", type=int, default=32, help="requests allowed to wait for the model; the next gets 529")
    ap.add_argument("--queue-timeout", type=float, default=600.0, help="seconds a request may wait before 529")
    ap.add_argument("--no-hash", action="store_true", help="skip hashing the weight files at start-up")
    ap.add_argument("--no-warmup", action="store_true", help="skip the two warm-up requests at start-up")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(f"[manchego-serve] {msg}", file=sys.stderr, flush=True)

    decider = build(args.backend, args.model, args.revision, args.execution, args.dtype, args.device, args.batch_tokens,
                    args.max_prompt_tokens, args.max_questions, hash_weights=not args.no_hash, log=log)
    extra: dict[str, Any] = {"runtime": getattr(decider.b, "runtime", None)}
    if not args.no_warmup:
        extra["warmup"] = warm_up(decider)
        log(f"warm-up: {extra['warmup']}")
    keys = {k for k in os.environ.get("MANCHEGO_API_KEYS", "").split(",") if k}
    app = create_app(decider, api_keys=keys, max_queue=args.max_queue, queue_timeout_s=args.queue_timeout, health_extra=extra)
    import uvicorn
    log(f"serving {MODEL_NAME} on http://{args.host}:{args.port} (execution {args.execution}, contract auto)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", server_header=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
