# Copyright 2026 oraculumai
# SPDX-License-Identifier: Apache-2.0
"""Backends: a tokenizer (`tok`) and `logits_batch([(prompt_ids, code_ids), ...]) -> [[float, ...], ...]`.

Each row's hidden state is read at its own last prompt position and projected, in float32, onto the output-embedding
rows of that row's option codes only: the readout Manchego was trained with.
"""

from __future__ import annotations


def load_backend(kind: str, model_dir: str, dtype: str = "auto", device: str = "auto", fast=None):
    """`fast`: a fast_path.FastPathConfig (torch only; None or all-off for the reference path)."""
    if kind == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(model_dir, dtype=dtype, device=device, fast=fast)
    if kind == "mlx":
        if fast is not None and not fast.is_off:
            from .fast_path import FastPathError
            raise FastPathError("the fast path (--gdn-kernels fast, --cuda-graphs, --fast-host) is for the torch backend")
        from .mlx_backend import MLXBackend
        return MLXBackend(model_dir)
    raise ValueError(f"unknown backend {kind!r}; expected torch or mlx")
