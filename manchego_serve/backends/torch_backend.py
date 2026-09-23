# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""PyTorch + Transformers backend: bfloat16 on CUDA, float32 on CPU (overridable)."""

from __future__ import annotations


class TorchBackend:
    name = "torch"

    def __init__(self, model_dir: str, dtype: str = "auto", device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        if dtype not in ("bfloat16", "float32", "float16"):
            raise ValueError(f"unsupported dtype {dtype!r}; expected bfloat16, float32 or float16")
        self.device, self.precision = device, dtype
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=getattr(torch, dtype)).to(self.device)
        self.model = model.eval()
        self.runtime = {"torch": torch.__version__, "device": self.device,
                        "cuda_device": torch.cuda.get_device_name(self.device) if self.device.startswith("cuda") else None}

    def logits(self, ids: list[int], cands: list[int]) -> list[float]:
        return self.logits_batch([(ids, cands)])[0]

    def logits_batch(self, items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        """Right-padded batch; hidden state gathered at each row's last REAL position; float32 dot with the candidate
        rows of the output matrix. Right padding puts pad tokens after the read position, so neither causal attention
        nor the recurrent layers carry them into it. With one row there is no padding at all."""
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
