"""The Metal kernels against mlx-lm's own Dory functions, on the strided layouts attention passes in."""

import platform

import pytest

if platform.machine() != "arm64" or platform.system() != "Darwin":
    pytest.skip("Metal kernels need an Apple Silicon Mac", allow_module_level=True)

import mlx.core as mx
from mlx_lm.models import dory

from brainspa_mlx import fastops


def _close(a, b, tol=2e-2):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item() < tol


@pytest.mark.parametrize("batch, steps", [(1, 1), (8, 1), (3, 5)])
def test_qk_prep_matches_stock_on_transposed_input(batch, steps):
    mx.random.seed(0)
    heads, d = 24, 256
    qk = mx.random.normal((batch, steps, heads, d)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    scale = mx.random.uniform(0.5, 2.0, (1, heads, 1, d)).astype(mx.float32)
    assert _close(fastops.qk_prep(qk, scale), dory.qk_prep(qk, scale))


@pytest.mark.parametrize("rows", [1, 8])
def test_slerp_norm_matches_stock(rows):
    mx.random.seed(1)
    res = mx.random.normal((rows, 1, 2560)).astype(mx.bfloat16)
    out = mx.random.normal((rows, 1, 2560)).astype(mx.bfloat16)
    alpha = mx.random.uniform(0.0, 0.1, (2560,)).astype(mx.float32)
    assert _close(fastops.slerp_norm(res, out, alpha), dory.slerp_norm(res, out, alpha))
