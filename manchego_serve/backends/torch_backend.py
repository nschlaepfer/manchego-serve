# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""PyTorch + Transformers backend: bfloat16 on CUDA, float32 on CPU (overridable).

With the fast path off (the default) this is the 0.1.x backend: `logits_batch` below is its code, unchanged. The opt-in
fast path (fast_path.py: gated-delta-net kernels, CUDA graphs, the lean one-prompt path) only ever replaces one-prompt
calls and the kernel functions; padded batches always run the reference code.
"""

from __future__ import annotations

from .fast_path import FastPathConfig, FastPathError, GraphBuckets, SinglePath, select_gdn_kernels


class TorchBackend:
    name = "torch"

    def __init__(self, model_dir: str, dtype: str = "auto", device: str = "auto", fast: FastPathConfig | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        if dtype not in ("bfloat16", "float32", "float16"):
            raise ValueError(f"unsupported dtype {dtype!r}; expected bfloat16, float32 or float16")
        fast = fast or FastPathConfig()
        fast.check(device)                     # refuse before the weights are read
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(device)
        self._setup(torch, model, tok, device, dtype, fast)

    @classmethod
    def from_model(cls, model, tok, device: str = "cpu", precision: str | None = None, fast: FastPathConfig | None = None):
        """A backend around an already-loaded model and tokenizer (tests, tools/fast_path_check.py)."""
        import torch
        self = cls.__new__(cls)
        precision = precision or str(next(model.parameters()).dtype).replace("torch.", "")
        self._setup(torch, model, tok, device, precision, fast or FastPathConfig())
        return self

    def _setup(self, torch, model, tok, device: str, precision: str, fast: FastPathConfig) -> None:
        self.torch, self.tok, self.device, self.precision = torch, tok, device, precision
        self.model = model.eval()
        self.runtime = {"torch": torch.__version__, "device": self.device,
                        "cuda_device": torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else None}
        self.fast, self.fast_path, self._single = FastPathConfig(), None, None
        self.configure_fast(fast)

    # ------------------------------------------------------------------ the fast path (opt-in)
    def _is_qwen35(self) -> bool:
        cfg = self.model.config
        return str(getattr(cfg, "model_type", "")).startswith("qwen3_5")

    def configure_fast(self, fast: FastPathConfig) -> dict | None:
        """Apply a fast-path configuration to the loaded model (at start-up; tools/fast_path_check.py switches between
        configurations on one loaded model). Returns what is in force, or None when every piece is off."""
        fast.check(self.device)
        if self._single is not None and self._single.graphs is not None:
            self._single.graphs.release()
        self._single = None
        if self._is_qwen35():
            kernels = select_gdn_kernels(fast.gdn_kernels)
        elif fast.gdn_kernels == "fast":
            raise FastPathError(f"--gdn-kernels fast is for Qwen3.5 models; this is {self.model.config.model_type!r}")
        else:
            kernels = {"note": f"not a Qwen3.5 model ({getattr(self.model.config, 'model_type', None)}): nothing selected"}
        self.runtime["gdn_kernels"] = kernels
        self.fast, self.fast_path = fast, None
        if fast.is_off:
            return None
        pin = self.device.startswith("cuda")
        graphs, captured = None, False
        if fast.cuda_graphs:
            pad = self.tok.pad_token_id if getattr(self.tok, "pad_token_id", None) is not None else (
                getattr(self.tok, "eos_token_id", None) or 0)
            graphs = GraphBuckets(self.torch, self.model, self.device, fast.graph_buckets, pad, warmup=fast.graph_warmup, pin=pin)
            captured = graphs.capture_all()
        if fast.single_path:
            self._single = SinglePath(self.torch, self.model, self.device, graphs, pin=pin)
        self.fast_path = {**fast.describe(), "gdn_kernels_in_use": kernels, "graphs": captured}
        return self.fast_path

    # ------------------------------------------------------------------ scoring
    def logits(self, ids: list[int], cands: list[int]) -> list[float]:
        return self.logits_batch([(ids, cands)])[0]

    def logits_batch(self, items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        """Right-padded batch; hidden state gathered at each row's last REAL position; float32 dot with the candidate
        rows of the output matrix. Right padding puts pad tokens after the read position, so neither causal attention
        nor the recurrent layers carry them into it. With one row there is no padding at all."""
        if self._single is not None and len(items) == 1:
            ids, cands = items[0]
            return [self._single.logits(list(ids), list(cands))]
        t = self.torch
        n, width = len(items), max(len(i) for i, _ in items)
        ids = t.zeros((n, width), dtype=t.long, device=self.device)
        att = t.zeros((n, width), dtype=t.long, device=self.device)
        for r, (row, _) in enumerate(items):
            ids[r, : len(row)] = t.tensor(row, device=self.device)
            att[r, : len(row)] = 1
        body, W = self.model.model, self.model.lm_head.weight
        with t.inference_mode():
            h = body(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
            last = t.tensor([len(row) - 1 for row, _ in items], device=self.device)
            h_last = h[t.arange(n, device=self.device), last].float()
            out = []
            for r, (_, cands) in enumerate(items):
                out.append((h_last[r] @ W[t.tensor(cands, device=self.device)].float().T).cpu().tolist())
        return out
