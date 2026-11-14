"""A two-stage ``lm_head``: a 4-bit NVFP4 copy finds the candidates, NVIDIA's bf16 weights score them exactly.

At decode the bf16 head reads 0.67 GB of 3.56 GB per token. The 4-bit copy reads 0.19 GB; re-scoring the ``top_m``
candidates from the bf16 rows reads a few MB. Top candidates keep the bf16 logit, so greedy decoding and top-p sampling
see the served head's values on every token that matters; the rest of the vocabulary carries 4-bit logits. The 4-bit
copy is built at load time from the bf16 weights, so the weights file is unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

GROUP_SIZE = 16
BITS = 4
TOP_M = 1024
FORCE_BLOCK = 64  # quality gates only: 64 positions of 1024 bf16 rows is 320 MB
MAX_ROWS = 4  # above this the 4-bit head loses to bf16 on MLX 0.32.4: its fast route needs at most 4 rows


def _quantized_head(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(x, weight, scales=scales, transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="nvfp4")


class FilteredHead(nn.Module):
    """Decode steps of up to ``MAX_ROWS`` rows; anything else takes the full bf16 path unless ``force_all`` (quality gates)."""

    def __init__(self, linear: nn.Linear, logit_scale: mx.array, top_m: int = TOP_M):
        super().__init__()
        self.weight = linear.weight
        self.top_m = top_m
        self.logit_scale = logit_scale
        self.force_all = False
        self.q_weight, self.q_scales, *_ = mx.quantize(linear.weight, group_size=GROUP_SIZE, bits=BITS, mode="nvfp4")
        mx.eval(self.q_weight, self.q_scales)
        self.freeze()

    def _filter(self, x: mx.array) -> mx.array:
        y = _quantized_head(x, self.q_weight, self.q_scales)
        score = y.astype(mx.float32) * self.logit_scale
        idx = mx.argsort(-score, axis=-1)[..., : self.top_m]  # argpartition runs on the CPU; argsort is GPU and faster
        exact = x[..., None, :] @ self.weight[idx].swapaxes(-1, -2)  # (B, T, 1, D) @ (B, T, D, M)
        return mx.put_along_axis(y, idx, exact.squeeze(-2).astype(y.dtype), axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.force_all:
            blocks = []
            for s in range(0, x.shape[1], FORCE_BLOCK):
                blocks.append(self._filter(x[:, s:s + FORCE_BLOCK]))
                mx.eval(blocks[-1])
            return mx.concatenate(blocks, axis=1)
        if x.shape[1] != 1 or x.shape[0] > MAX_ROWS:
            return x @ self.weight.T
        return self._filter(x)


def install_head(model, mode: str, top_m: int = TOP_M) -> None:
    """``mode``: ``filter`` (two-stage, bf16 weights kept) or ``nvfp4`` (4-bit head only, a different weights file)."""
    if mode == "filter":
        scale = model.logit_scale.astype(mx.float32) / model.args.initializer_range
        mx.eval(scale)
        model.lm_head = FilteredHead(model.lm_head, scale, top_m)
    elif mode == "nvfp4":
        q, s, *_ = mx.quantize(model.lm_head.weight, group_size=GROUP_SIZE, bits=BITS, mode="nvfp4")
        mx.eval(q, s)
        model.lm_head = _Nvfp4Head(q, s)
    else:
        raise ValueError(f"unknown head mode {mode}")


class _Nvfp4Head(nn.Module):
    def __init__(self, weight: mx.array, scales: mx.array):
        super().__init__()
        self.weight, self.scales = weight, scales
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        return _quantized_head(x, self.weight, self.scales)
