# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""MLX backend (Apple silicon; optional extra `mlx`). For the MLX conversions oraculumai/Manchego-MLX-8bit / -4bit."""

from __future__ import annotations

import json
from pathlib import Path

MASKED_LOGIT = -1e9  # padding columns of the candidate matrix; never returned


class MLXBackend:
    name = "mlx"

    def __init__(self, model_dir: str):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tok = load(model_dir)
        self.model.eval()
        try:
            q = json.loads((Path(model_dir) / "config.json").read_text()).get("quantization") or {}
        except OSError:
            q = {}
        self.precision = f"q{q['bits']}g{q.get('group_size', '')}" if q.get("bits") else "unquantized"
        import mlx_lm
        self.runtime = {"mlx_lm": getattr(mlx_lm, "__version__", None), "device": str(mx.default_device())}

    def logits(self, ids: list[int], cands: list[int]) -> list[float]:
        return self.logits_batch([(ids, cands)])[0]

    def _collate(self, batch: list[tuple[list[int], list[int]]], pad_id: int = 0):
        """Right-pad prompts with `pad_id` and menus with each row's first code; mask the padded codes."""
        mx = self.mx
        T = max(len(ids) for ids, _ in batch)
        K = max(len(c) for _, c in batch)
        tokens, last, cands, mask = [], [], [], []
        for ids, c in batch:
            n, k = len(ids), len(c)
            tokens.append(list(ids) + [pad_id] * (T - n))
            last.append(n - 1)
            cands.append(list(c) + [c[0]] * (K - k))
            mask.append([True] * k + [False] * (K - k))
        return mx.array(tokens, dtype=mx.int32), mx.array(last, dtype=mx.int32), mx.array(cands, dtype=mx.int32), mx.array(mask)

    def _candidate_logits(self, tokens, last_pos, cand_ids, cand_mask):
        """(B, K) float32 logits of the offered codes at each row's last real position, projected in float32."""
        mx = self.mx
        text = getattr(self.model, "language_model", self.model)
        hidden = text.model(tokens)
        h_last = hidden[mx.arange(tokens.shape[0]), last_pos].astype(mx.float32)
        if text.args.tie_word_embeddings:
            rows = text.model.embed_tokens(cand_ids).astype(mx.float32)  # (B, K, D), dequantised
            picked = (rows * h_last[:, None, :]).sum(axis=-1)
        else:
            logits = text.lm_head(h_last.astype(hidden.dtype)).astype(mx.float32)
            picked = mx.take_along_axis(logits, cand_ids, axis=-1)
        return mx.where(cand_mask, picked, MASKED_LOGIT)

    def logits_batch(self, items: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        """One padded batch through the backbone (rows in ascending length); each row read at its own last position."""
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        tokens, last, cands, mask = self._collate([items[i] for i in order])
        z = self._candidate_logits(tokens, last, cands, mask)
        self.mx.eval(z)
        out: list = [None] * len(items)
        for row, i in zip(z.tolist(), order):
            out[i] = row[: len(items[i][1])]
        return out
