#!/usr/bin/env python3
# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Validate the opt-in fast path (docs/FAST_PATH.md). Two checks:

  agree    Loads the model ONCE with the reference path, scores every golden question (tests/fixtures/requests.json and
           semif_requests.json, each question alone, one forward pass, under the model's contract), then switches the same
           loaded model to each fast-path configuration and scores them again. Per configuration: the largest
           probability difference from the reference over every option of every question (T = 1), and every question whose
           chosen option changed. Exit status 1 when a configuration fails its threshold.

             python tools/fast_path_check.py agree --model /models/manchego-v3 --out agree.json

  latency  Serial single-decision requests against a running server (as JevBench v1.5 measures speed): p50, p95, their
           mean, and the mean input tokens. Nothing else may use the GPU meanwhile.

             python tools/fast_path_check.py latency --url http://127.0.0.1:8000 --n 450 --out latency.json

Configurations (agree): fast_host (must be bit for bit: max |dp| = 0), kernels (--gdn-kernels fast), graphs
(--cuda-graphs), all (--fast-path). kernels, graphs and all change the arithmetic: they pass when max |dp| <= --max-dprob
and no chosen option changes (--near-tie exempts questions whose two best reference logits are closer than it; default
0, so every change fails). The fixtures are invented requests; the check uses no benchmark data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"


def golden_questions() -> list[dict]:
    """[{id, state, question}]: every question of the fixture requests, each to be scored alone."""
    out = []
    for r in json.loads((FIXTURES / "requests.json").read_text(encoding="utf-8"))["requests"]:
        for name, q in r["body"]["questions"].items():
            out.append({"id": f"{r['id']}/{name}", "state": r["body"]["state"], "question": q})
    for r in json.loads((FIXTURES / "semif_requests.json").read_text(encoding="utf-8"))["requests"]:
        out.append({"id": f"semif/{r['id']}", "state": r["state"], "question": r["question"]})
    return out


def pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))]


def summary_ms(xs: list[float]) -> dict:
    p50, p95 = pct(xs, 50), pct(xs, 95)
    return {"n": len(xs), "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "mean_of_p50_p95_ms": round((p50 + p95) / 2, 2),
            "mean_ms": round(statistics.fmean(xs), 2), "min_ms": round(min(xs), 2), "max_ms": round(max(xs), 2)}


# ------------------------------------------------------------------ agree
CONFIGS = {
    "fast_host": {"fast_host": True},
    "kernels": {"gdn_kernels": "fast"},
    "graphs": {"cuda_graphs": True},
    "all": {"gdn_kernels": "fast", "cuda_graphs": True, "fast_host": True},
}


def score_all(backend, decider, plans: list[dict], sync) -> tuple[list[list[float]], list[float]]:
    from manchego_serve.decider import in_client_order
    logits, ms = [], []
    for p in plans:
        sync()
        t0 = time.perf_counter()
        z = backend.logits_batch([(p["ids"], p["cands"])])[0]
        sync()
        ms.append((time.perf_counter() - t0) * 1000)
        logits.append(in_client_order(z, p["order"]))
    return logits, ms


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def compare(qs, plans, ref: list[list[float]], got: list[list[float]], near_tie: float) -> dict:
    worst, worst_id, flips, ties = 0.0, None, [], []
    for q, p, zr, zg in zip(qs, plans, ref, got):
        pr, pg = softmax(zr), softmax(zg)
        d = max(abs(a - b) for a, b in zip(pr, pg))
        if d > worst:
            worst, worst_id = d, q["id"]
        ar, ag = zr.index(max(zr)), zg.index(max(zg))
        if ar != ag:
            top = sorted(zr, reverse=True)
            gap = top[0] - top[1]
            (ties if gap < near_tie else flips).append({"id": q["id"], "reference_top2_gap": gap, "contract": p["contract"]})
    return {"max_dprob": worst, "max_dprob_question": worst_id, "argmax_changes": flips, "argmax_changes_near_ties": ties}


def agree(a) -> int:
    from manchego_serve.server import force_offline
    force_offline()
    import torch
    from manchego_serve import model_config as MC
    from manchego_serve.backends import fast_path as FP
    from manchego_serve.backends.torch_backend import TorchBackend
    from manchego_serve.decider import Decider
    backend = TorchBackend(a.model, dtype=a.dtype, device=a.device)
    contract = MC.read(a.model).contract if a.contract == "model" else a.contract
    decider = Decider(backend, contract=contract, max_prompt_tokens=10 ** 9)
    if backend.device.startswith("cuda"):
        sync = lambda: torch.cuda.synchronize(backend.device)  # noqa: E731
    elif backend.device.startswith("mps"):
        sync = torch.mps.synchronize
    else:
        sync = lambda: None  # noqa: E731
    if a.emulate_graphs:
        if backend.device.startswith("cuda"):
            raise SystemExit("--emulate-graphs is for machines without CUDA; on CUDA the real graphs are checked")
        import dataclasses
        check = FP.FastPathConfig.check                                   # graphs only: the kernels still need CUDA
        FP.FastPathConfig.check = lambda self, device: check(dataclasses.replace(self, cuda_graphs=False), device)
        import manchego_serve.backends.torch_backend as TB
        TB.GraphBuckets = lambda *x, **k: FP.GraphBuckets(*x, **{**k, "capture": FP.eager_capture, "pin": False})
    qs = golden_questions()
    plans = [decider.plan(q["state"], q["question"]) for q in qs]
    buckets = FP.parse_buckets(a.buckets)
    report = {"model": a.model, "contract": contract, "device": backend.device, "precision": backend.precision,
              "runtime": backend.runtime, "questions": len(qs), "prompt_tokens": {"min": min(len(p["ids"]) for p in plans),
              "max": max(len(p["ids"]) for p in plans), "beyond_largest_bucket": sum(len(p["ids"]) > max(buckets) for p in plans)},
              "thresholds": {"max_dprob": a.max_dprob, "near_tie": a.near_tie, "fast_host": "bit for bit (max |dp| = 0)"},
              "emulated_graphs": bool(a.emulate_graphs), "configs": {}}
    score_all(backend, decider, plans[:3], sync)                         # warm-up
    ref, ref_ms = score_all(backend, decider, plans, sync)
    report["reference"] = {"gdn_kernels": backend.runtime["gdn_kernels"], "latency": summary_ms(ref_ms)}
    print(f"reference: {len(qs)} questions, p50 {pct(ref_ms, 50):.1f} ms", file=sys.stderr)
    failed = []
    for name in a.configs.split(","):
        kw = dict(CONFIGS[name], graph_buckets=buckets) if name in ("graphs", "all") else dict(CONFIGS[name])
        try:
            in_force = backend.configure_fast(FP.FastPathConfig(**kw))
        except Exception as e:  # e.g. graph capture failed: that is a result, reported, not a crash
            report["configs"][name] = {"error": f"{type(e).__name__}: {e}", "pass": False}
            failed.append(name)
            print(f"{name}: FAILED to configure: {e}", file=sys.stderr)
            backend.configure_fast(FP.FastPathConfig())
            continue
        score_all(backend, decider, plans[:3], sync)
        got, ms = score_all(backend, decider, plans, sync)
        cmp = compare(qs, plans, ref, got, a.near_tie)
        ok = (cmp["max_dprob"] == 0.0 and not cmp["argmax_changes_near_ties"]) if name == "fast_host" else (
            cmp["max_dprob"] <= a.max_dprob)
        ok = ok and not cmp["argmax_changes"]
        notes = []
        if kw.get("gdn_kernels") == "fast" and all(str(v).startswith("reference") for v in in_force["gdn_kernels_in_use"].values()):
            ok = False                                                    # nothing fast ran: a vacuous pass is a failure
            notes.append("no fast kernel is importable: this configuration ran the reference kernels")
        counts = dict(backend._single.counts) if backend._single is not None else None
        report["configs"][name] = {"fast_path": in_force, "calls": counts, **cmp, "latency": summary_ms(ms), "notes": notes,
                                   "pass": ok}
        failed += [] if ok else [name]
        print(f"{name}: max |dp| {cmp['max_dprob']:.3g} ({cmp['max_dprob_question']}), argmax changes "
              f"{len(cmp['argmax_changes'])} (+{len(cmp['argmax_changes_near_ties'])} near ties), p50 {pct(ms, 50):.1f} ms: "
              f"{'PASS' if ok else 'FAIL'}", file=sys.stderr)
        backend.configure_fast(FP.FastPathConfig())
    report["pass"] = not failed
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if not failed else 1


# ------------------------------------------------------------------ latency
def request_bodies(n: int, path: str | None) -> list[dict]:
    if path:
        bodies = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        bodies = [{"model": "manchego", "state": q["state"], "questions": {"q": q["question"]}} for q in golden_questions()]
    return [bodies[i % len(bodies)] for i in range(n)]


def post(url: str, body: dict, key: str | None, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **({"authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:500]}


def latency(a) -> int:
    base = a.url.rstrip("/")
    health = json.loads(urllib.request.urlopen(base + "/healthz", timeout=30).read())
    bodies = request_bodies(a.n, a.requests)
    for body in request_bodies(a.warmup, a.requests):
        post(base + "/v1/systemone", body, a.api_key, a.timeout)
    ms, tokens, errors, by_len = [], [], [], {}
    for i, body in enumerate(bodies):
        t0 = time.perf_counter()
        code, out = post(base + "/v1/systemone", body, a.api_key, a.timeout)
        dt = (time.perf_counter() - t0) * 1000
        if code != 200:
            errors.append({"i": i, "status": code, "body": out})
            continue
        ms.append(dt)
        tok = out["usage"]["input_tokens"]
        tokens.append(tok)
        by_len.setdefault(next(b for b in (256, 512, 1024, 2048, 10 ** 9) if tok <= b), []).append(dt)
    report = {"url": base, "requests": len(bodies), "ok": len(ms), "errors": errors[:10], "warmup": a.warmup,
              "latency": summary_ms(ms) if ms else None, "mean_input_tokens": statistics.fmean(tokens) if tokens else None,
              "by_prompt_tokens": {("<=" + str(k) if k < 10 ** 9 else ">2048"): summary_ms(v) for k, v in sorted(by_len.items())},
              "server": {k: health.get(k) for k in ("server", "backend", "precision", "contract", "execution", "temperature",
                                                   "weights_sha256", "weights_verified", "fast_path", "runtime", "warmup")}}
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if not errors else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree", help="fast path vs reference on the golden requests, one loaded model")
    g.add_argument("--model", required=True, help="a local model folder")
    g.add_argument("--contract", default="model", choices=("model", "auto", "semif"))
    g.add_argument("--dtype", default="auto")
    g.add_argument("--device", default="auto")
    g.add_argument("--configs", default="fast_host,kernels,graphs,all")
    g.add_argument("--buckets", default="256,512,1024,2048")
    g.add_argument("--max-dprob", type=float, default=0.03)
    g.add_argument("--near-tie", type=float, default=0.0, help="exempt argmax changes whose reference top-2 logit gap is below this")
    g.add_argument("--emulate-graphs", action="store_true", help="no CUDA: an eager stand-in for the graphs (padding effect only)")
    g.add_argument("--out")
    lat = sub.add_parser("latency", help="serial single-decision latency against a running server")
    lat.add_argument("--url", default="http://127.0.0.1:8000")
    lat.add_argument("--n", type=int, default=450)
    lat.add_argument("--warmup", type=int, default=20)
    lat.add_argument("--requests", help="a JSON list of request bodies (default: every golden question, one per request, cycled)")
    lat.add_argument("--api-key", default=os.environ.get("MANCHEGO_API_KEY"))
    lat.add_argument("--timeout", type=float, default=120.0)
    lat.add_argument("--out")
    a = ap.parse_args(argv)
    return agree(a) if a.cmd == "agree" else latency(a)


if __name__ == "__main__":
    sys.exit(main())
