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

from . import model_config as MC
from . import temperature as TM
from .contract import POLICIES, SEMIF, SEMIF_MAX_OPTIONS, BadRequest, LIMIT, MAX_OPTIONS
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


def models_payload(name: str = MODEL_NAME, label: str = "Manchego v2.1", release_date: str = RELEASE_DATE) -> dict:
    """`models` is the shape TypeSafe's SDKs validate; `data` is an OpenAI-style list. The request's `model` field may be
    any string: this server serves one model and names it in every response."""
    names = (name,) + ALIASES
    return {"models": [{"name": n, "description": f"{label} decision model (Qwen3.5-4B fine-tune), System One wire contract"
                        + ("" if n == name else f"; alias of {name}"), "release_date": release_date} for n in names],
            "data": [{"id": name, "aliases": list(ALIASES)}]}


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

    prompt_limit = ({"semif_prompt_options": SEMIF_MAX_OPTIONS} if decider.contract == SEMIF
                    else {"short_prompt_options": LIMIT["short"]})
    listing = models_payload(decider.model_name, decider.info.get("model_label") or "Manchego v2.1",
                             decider.info.get("release_date") or RELEASE_DATE)

    @app.get("/healthz")
    async def healthz():
        out = {"ok": True, **decider.describe(), "model": decider.model_name,
               "limits": {"options_per_question": MAX_OPTIONS, **prompt_limit,
                          "max_prompt_tokens_per_question": decider.max_prompt_tokens, "max_questions_per_request": decider.max_questions},
               "weight_files": decider.info.get("weight_files"), "support_files": decider.info.get("support_files"),
               "temperature_map_provenance": decider.tmap.provenance(), **(health_extra or {})}
        if decider.info.get("model_config"):
            out["model_config"] = decider.info["model_config"]
        return out

    @app.get("/v1/models")
    async def models():
        return listing

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
          max_questions: int = DEFAULT_MAX_QUESTIONS, hash_weights: bool = True, log=print,
          temperature_map: str | None = "default", contract: str | None = "model") -> Decider:
    force_offline()
    tmap = TM.load(temperature_map)          # fail fast on a bad map, before the weights are read
    from .backends import load_backend
    from .weights import DEFAULT_REPO, resolve, identify
    model = model or DEFAULT_REPO[backend]
    model_dir, revision = resolve(model, revision)
    cfg = MC.read(model_dir)            # fail fast on a bad manchego_config.json, before the weights are read
    from_cli = contract not in (None, "", "model")
    policy = contract if from_cli else cfg.contract
    if policy not in POLICIES:
        raise ValueError(f"unknown contract {contract!r}; expected model, {', '.join(POLICIES)}")
    source = "--contract / $MANCHEGO_CONTRACT" if from_cli else cfg.describe()["contract_source"]
    log(f"contract: {policy} ({source})")
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
    tmap = TM.bind(tmap, info["weights_sha256"], log=log, declared_repo=info["repo"], declared_revision=info["revision"])
    log(f"temperature map: {tmap.source}, T = {tmap.temperatures} ({tmap.binding or tmap.note})")
    t0 = time.perf_counter()
    b = load_backend(backend, model_dir, dtype=dtype, device=device)
    log(f"loaded {backend} ({getattr(b, 'precision', '?')}) from {model_dir} in {time.perf_counter() - t0:.1f} s")
    info["model_config"] = {**cfg.describe(), "contract": policy, "contract_source": source}
    if cfg.served_name:
        info["model_label"] = " ".join(x for x in (cfg.model, cfg.version) if x) or cfg.served_name
        info["release_date"] = cfg.release_date
    return Decider(b, execution=execution, batch_tokens=batch_tokens, max_prompt_tokens=max_prompt_tokens,
                   max_questions=max_questions, info=info, temperature_map=tmap, contract=policy,
                   model_name=cfg.served_name or MODEL_NAME)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    ap.add_argument("--temperature-map", default=os.environ.get("MANCHEGO_TEMPERATURE_MAP") or "default",
                    help="one temperature per question type, applied to the option-code logits: 'default' (the map fitted "
                         "for v2.1, shipped with the package), 'off' (T = 1.0 for every type, the v0.1.0 policy) or a JSON "
                         "file (default: $MANCHEGO_TEMPERATURE_MAP, else 'default'). It never changes the chosen option")
    ap.add_argument("--no-temperature-map", action="store_true", help="same as --temperature-map off")
    ap.add_argument("--contract", choices=("model",) + POLICIES, default=os.environ.get("MANCHEGO_CONTRACT") or "model",
                    help="prompt policy: 'model' (default: the `contract` field of the model folder's manchego_config.json, "
                         "'auto' when absent, as for Manchego v2.1), 'auto' (short prompt up to 26 options, state-first beyond) or "
                         "'semif' (SemIf prompt up to 16 options, state-first beyond; for models trained under SemIf). "
                         "Default: $MANCHEGO_CONTRACT, else 'model'")
    args = ap.parse_args(argv)
    if args.no_temperature_map:
        args.temperature_map = "off"
    return args


def main(argv: list[str] | None = None) -> int:
    force_offline()
    args = parse_args(argv)

    def log(msg: str) -> None:
        print(f"[manchego-serve] {msg}", file=sys.stderr, flush=True)

    decider = build(args.backend, args.model, args.revision, args.execution, args.dtype, args.device, args.batch_tokens,
                    args.max_prompt_tokens, args.max_questions, hash_weights=not args.no_hash, log=log,
                    temperature_map=args.temperature_map, contract=args.contract)
    extra: dict[str, Any] = {"runtime": getattr(decider.b, "runtime", None)}
    if not args.no_warmup:
        extra["warmup"] = warm_up(decider)
        log(f"warm-up: {extra['warmup']}")
    keys = {k for k in os.environ.get("MANCHEGO_API_KEYS", "").split(",") if k}
    app = create_app(decider, api_keys=keys, max_queue=args.max_queue, queue_timeout_s=args.queue_timeout, health_extra=extra)
    import uvicorn
    log(f"serving {decider.model_name} on http://{args.host}:{args.port} (execution {args.execution}, contract {decider.contract}, "
        f"temperature {decider.tmap.wire_temperature()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", server_header=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
