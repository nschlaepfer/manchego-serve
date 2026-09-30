# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""The torch backend's fast path (0.2.0.dev). Every piece is opt-in and OFF by default; with all of them off the torch
backend runs the 0.1.x code, and the published numbers' arithmetic.

  gdn_kernels   Qwen3.5's gated-delta-net (linear-attention) layers. "reference" (default): Transformers' reference PyTorch
                functions, EVEN when flash-linear-attention or causal-conv1d is installed (0.1.x used them silently when
                they were importable when Transformers was imported). "fast": fla's chunk_gated_delta_rule /
                fused_recurrent_gated_delta_rule and causal-conv1d's causal_conv1d_fn / causal_conv1d_update, each where
                importable (reported one by one); the others stay reference. CUDA only. Changes the arithmetic.
  cuda_graphs   one-prompt forward passes replayed as CUDA graphs captured at start-up at fixed padded lengths (the
                buckets, default 256, 512, 1024, 2048): the prompt is right-padded to the smallest bucket that holds it,
                with no attention mask and no KV cache, and the hidden state is read at the prompt's own last position.
                Every layer is causal and the padding comes after the read position, so the padding cannot reach the
                answer; the kernels see another shape, so at bf16 the numbers move slightly. Longer prompts and padded
                batches run eagerly. CUDA only.
  fast_host     the eager one-prompt path without the host work it does not need: the prompt and the option-code ids go
                to the device in one non-blocking copy each from pinned memory; no all-ones attention mask (Transformers
                checks such a mask on the host, twice per forward, and then drops it: the same kernels run either way);
                the option-code rows of the output matrix are cached per code set; one device-to-host copy per question.
                Bit for bit the reference readout. Any device.

The mask: the text model accepts a precomputed {layer type: mask} mapping; {type: None} is exactly what Transformers
builds from an all-ones mask or from none (SDPA with is_causal=True, no padding mask in the recurrent layers), and it
also keeps Transformers from building an explicit causal mask while a CUDA graph is being captured (its `is_tracing`
is true during stream capture, which would otherwise switch the attention kernel inside the graph).

Nothing here runs at import time; torch and Transformers are imported by the functions that need them.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

DEFAULT_BUCKETS = (256, 512, 1024, 2048)
GDN_KERNELS = ("reference", "fast")
QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
# module-level name in Transformers' Qwen3.5 modelling -> (package, module, function) of the fast kernel
GDN_FUNCTIONS = {
    "torch_chunk_gated_delta_rule": ("flash-linear-attention", "fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
    "torch_recurrent_gated_delta_rule": ("flash-linear-attention", "fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule"),
    "causal_conv1d_fn": ("causal-conv1d", "causal_conv1d", "causal_conv1d_fn"),
    "causal_conv1d_update": ("causal-conv1d", "causal_conv1d", "causal_conv1d_update"),
}
ROWS_CACHE_ENTRIES = 32


class FastPathError(RuntimeError):
    """A fast-path setting this runtime cannot honour. The server refuses to start rather than run something else."""


@dataclass(frozen=True)
class FastPathConfig:
    gdn_kernels: str = "reference"
    cuda_graphs: bool = False
    graph_buckets: tuple = DEFAULT_BUCKETS
    fast_host: bool = False
    graph_warmup: int = 2

    def __post_init__(self):
        if self.gdn_kernels not in GDN_KERNELS:
            raise FastPathError(f"gdn_kernels {self.gdn_kernels!r}: expected one of {GDN_KERNELS}")
        b = tuple(self.graph_buckets)
        if not b or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in b) or len(set(b)) != len(b):
            raise FastPathError(f"graph buckets {b!r}: expected distinct positive token counts")
        object.__setattr__(self, "graph_buckets", tuple(sorted(b)))

    @property
    def is_off(self) -> bool:
        return self.gdn_kernels == "reference" and not self.cuda_graphs and not self.fast_host

    @property
    def single_path(self) -> bool:
        """Whether one-prompt calls leave the reference code."""
        return self.cuda_graphs or self.fast_host

    def check(self, device: str) -> None:
        """Refuse the CUDA-only pieces on any other device."""
        if not str(device).startswith("cuda"):
            asked = [n for n, on in (("--gdn-kernels fast", self.gdn_kernels == "fast"), ("--cuda-graphs", self.cuda_graphs)) if on]
            if asked:
                raise FastPathError(f"{' and '.join(asked)} need a CUDA device; this backend runs on {device}")

    def describe(self) -> dict:
        return {"gdn_kernels": self.gdn_kernels, "cuda_graphs": list(self.graph_buckets) if self.cuda_graphs else False,
                "fast_host": self.fast_host}


def parse_buckets(text: str) -> tuple:
    try:
        return tuple(int(x) for x in str(text).replace(" ", "").split(",") if x)
    except ValueError:
        raise FastPathError(f"graph buckets {text!r}: expected comma-separated token counts, e.g. 256,512,1024,2048") from None


# ------------------------------------------------------------------ (a) gated-delta-net kernels
def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def _filtered(fn: Callable, origin: str, reference: Callable) -> Callable:
    """Call `fn` with only the keyword arguments it names, exactly as Transformers' own wrapper does
    (`use_kernel_func_from_hub_with_fallback`: the layers pass keyword arguments a kernel does not take). `reference` is
    remembered so a later selection can go back."""
    accept = frozenset(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items() if k in accept})

    call.manchego_origin, call.manchego_reference = origin, reference
    return call


def _reference_of(obj: Callable) -> Callable:
    """The reference PyTorch function behind one of our wrappers, or behind a Transformers wrapper (functools.wraps keeps
    it as __wrapped__)."""
    if hasattr(obj, "manchego_reference"):
        return obj.manchego_reference
    seen = obj
    while hasattr(seen, "__wrapped__"):
        seen = seen.__wrapped__
    return seen


def select_gdn_kernels(mode: str, modeling: Any = None, importer: Callable[[str], Any] = importlib.import_module) -> dict:
    """Point the Qwen3.5 gated-delta-net functions at the reference or the fast kernels. Returns {name: what runs}.

    `modeling` is Transformers' Qwen3.5 modelling module (imported when None); `importer` imports a kernel module (tests
    pass fakes). The layers look these names up in the module at call time, so this applies to models already loaded."""
    if mode not in GDN_KERNELS:
        raise FastPathError(f"gdn_kernels {mode!r}: expected one of {GDN_KERNELS}")
    if modeling is None:
        modeling = importlib.import_module(QWEN35_MODELING)
    report = {}
    for name, (dist, module, func) in GDN_FUNCTIONS.items():
        current = getattr(modeling, name, None)
        if current is None:
            report[name] = "absent from this Transformers version"
            continue
        reference = _reference_of(current)
        if mode == "reference":
            if _importable(module) or getattr(current, "manchego_origin", None):
                setattr(modeling, name, _filtered(reference, "reference", reference))
                report[name] = f"reference (forced; {dist} is installed but not used)" if _importable(module) else "reference"
            else:
                report[name] = f"reference ({dist} not installed)"
            continue
        try:
            fast = getattr(importer(module), func)
            if not callable(fast):
                raise ImportError(f"{module}.{func} is not callable")
        except Exception as e:  # ImportError, a Triton version guard, a missing symbol...
            setattr(modeling, name, _filtered(reference, "reference", reference))
            report[name] = f"reference ({dist} not importable: {type(e).__name__}: {e})"[:300]
            continue
        setattr(modeling, name, _filtered(fast, f"{module}.{func}", reference))
        report[name] = f"{module}.{func}"
    return report


# ------------------------------------------------------------------ (b) CUDA graphs at fixed padded lengths
def no_masks(model) -> dict:
    """{layer type: None} for the text model's layer types: the mask mapping that means "no padding, causal"."""
    cfg = getattr(model.config, "text_config", None) or model.config
    return {t: None for t in set(getattr(cfg, "layer_types", None) or ("full_attention",))}


def text_body(model):
    """The decoder stack the readout runs (model.model) and the output matrix (lm_head.weight)."""
    return model.model, model.lm_head.weight


def cuda_capture(step: Callable, pool: Any, warmup: int, torch: Any):
    """Capture `step()` as a CUDA graph. -> (replay, static output, pool). Warm-up runs on a side stream first, as
    torch.cuda.graphs requires; `pool` is shared by every bucket so the graphs reuse one memory pool."""
    cuda = torch.cuda
    if pool is None:
        pool = cuda.graph_pool_handle()
    side = cuda.Stream()
    side.wait_stream(cuda.current_stream())
    with cuda.stream(side):
        for _ in range(warmup):
            step()
    cuda.current_stream().wait_stream(side)
    graph = cuda.CUDAGraph()
    with cuda.graph(graph, pool=pool):
        out = step()
    cuda.synchronize()
    return graph.replay, out, pool


def eager_capture(step: Callable, pool: Any, warmup: int, torch: Any):
    """Stand-in for cuda_capture where there is no CUDA (tests): replay recomputes into the same static output."""
    out = step()
    return (lambda: out.copy_(step())), out, pool


class GraphBuckets:
    """One-prompt forwards through captured graphs. Static buffers per bucket: the padded ids (1, L), the read position
    (1,), and the float32 hidden state at that position (D,)."""

    def __init__(self, torch: Any, model, device: str, buckets: tuple, pad_id: int, capture: Callable = cuda_capture,
                 warmup: int = 2, pin: bool = True):
        self.torch, self.model, self.device, self.buckets = torch, model, device, tuple(sorted(buckets))
        self.pad_id, self.capture, self.warmup = int(pad_id), capture, warmup
        self.body, _ = text_body(model)
        self.masks = no_masks(model)
        self.ids, self.last, self.host, self.replay, self.out = {}, {}, {}, {}, {}
        self.pin = pin

    def bucket_for(self, n: int) -> int | None:
        return next((L for L in self.buckets if n <= L), None)

    def capture_all(self) -> dict:
        t = self.torch
        pool, report = None, {}
        with t.inference_mode():
            for L in reversed(self.buckets):                       # largest first: the shared pool is sized once
                ids = t.full((1, L), self.pad_id, dtype=t.long, device=self.device)
                last = t.full((1,), L - 1, dtype=t.long, device=self.device)
                self.host[L] = t.full((L + 1,), self.pad_id, dtype=t.long, pin_memory=self.pin)

                def step(ids=ids, last=last):
                    h = self.body(input_ids=ids, attention_mask=self.masks, use_cache=False).last_hidden_state
                    return h.index_select(1, last).reshape(-1).float()

                self.replay[L], self.out[L], pool = self.capture(step, pool, self.warmup, t)
                self.ids[L], self.last[L] = ids, last
                report[str(L)] = "captured"
        return report

    def hidden_last(self, prompt: list[int]):
        """Float32 hidden state at the prompt's last position, or None when no bucket holds the prompt."""
        n = len(prompt)
        L = self.bucket_for(n)
        if L is None or L not in self.replay:
            return None
        host = self.host[L]
        buf = host.numpy()
        buf[:n] = prompt
        buf[n:L] = self.pad_id
        buf[L] = n - 1
        with self.torch.inference_mode():             # the static buffers were made in inference mode
            self.ids[L].copy_(host[:L].view(1, L), non_blocking=True)
            self.last[L].copy_(host[L:], non_blocking=True)
            self.replay[L]()
        return self.out[L]

    def release(self) -> None:
        self.ids, self.last, self.host, self.replay, self.out = {}, {}, {}, {}, {}


# ------------------------------------------------------------------ (c) the lean one-prompt path
@dataclass
class SinglePath:
    """One prompt at a time: through a graph when one holds it, else eagerly without the mask and with one copy each way."""
    torch: Any
    model: Any
    device: str
    graphs: GraphBuckets | None = None
    pin: bool = True
    rows: dict = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: {"graph": 0, "eager": 0})

    def __post_init__(self):
        self.body, self.W = text_body(self.model)
        self.masks = no_masks(self.model)

    def _to_device(self, values: list[int]):
        t = self.torch
        host = t.tensor(values, dtype=t.long)
        if self.pin:
            host = host.pin_memory()
        return host.to(self.device, non_blocking=True)

    def candidate_rows(self, cands: list[int]):
        """float32 output-matrix rows of these option codes, (K, D); the same values as W[cands].float()."""
        key = tuple(cands)
        rows = self.rows.get(key)
        if rows is None:
            rows = self.W.index_select(0, self._to_device(cands)).float()
            if len(self.rows) >= ROWS_CACHE_ENTRIES:
                self.rows.pop(next(iter(self.rows)))
            self.rows[key] = rows
        return rows

    def hidden_last(self, prompt: list[int]):
        if self.graphs is not None:
            h = self.graphs.hidden_last(prompt)
            if h is not None:
                self.counts["graph"] += 1
                return h
        self.counts["eager"] += 1
        ids = self._to_device(prompt).view(1, -1)
        h = self.body(input_ids=ids, attention_mask=self.masks, use_cache=False).last_hidden_state
        return h[0, len(prompt) - 1].float()

    def logits(self, prompt: list[int], cands: list[int]) -> list[float]:
        with self.torch.inference_mode():
            h_last = self.hidden_last(prompt)
            return (h_last @ self.candidate_rows(cands).T).cpu().tolist()
