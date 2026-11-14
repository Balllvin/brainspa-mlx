"""Hand-written Metal kernels for the nGPT norm steps of Dory, the part of a decode step that is not weight reads.

``mlx_lm``'s ``slerp_norm``, ``gate_out``, ``mlp_act``, ``qk_prep``, ``justnorm`` and ``loop_inject`` are shapeless
compiled graphs, but MLX cannot fuse across ``Reduce``, so one call is several dispatches and each row reduction walks
2560 elements serially on one GPU thread (about 21 microseconds for ``slerp_norm``). Each kernel here does the whole fp32
computation in one dispatch of 256-thread groups, one group per row, which measured about 5 microseconds. The arithmetic
is the compiled function's: rounding to bf16 happens only at the end. Only the order of the additions in a sum differs.

``install`` patches the functions in ``mlx_lm.models.dory``, where the model code reads them as module globals.
"""

from __future__ import annotations

import mlx.core as mx

TG = 256  # threads per group; the helpers work for any SIMD width

_HEADER = f"""
constant float NORM_EPS = 1e-6f;
constant uint TG = {TG};
inline float sigmoid_mx(float x) {{
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}}
inline float tg_sum(float lane_sum, threadgroup float* partial, uint sg, uint n_sg) {{
  const float s = simd_sum(lane_sum);
  if (simd_is_first()) partial[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0;
  for (uint i = 0; i < n_sg; i++) total += partial[i];
  return total;
}}
"""

_SETUP = """
    const uint lane = thread_index_in_threadgroup;
    const uint sg = lane / threads_per_simdgroup;
    const uint n_sg = (TG + threads_per_simdgroup - 1) / threads_per_simdgroup;
    const uint row = threadgroup_position_in_grid.x;
"""

_ROW3 = """
    const uint T = {x}_shape[1];
    const uint D = {x}_shape[2];
    const uint r_b = row / T, r_t = row % T;
    const uint base_y = row * D;
"""


def _base(x: str) -> str:
    return f"    const uint base_{x} = r_b * {x}_strides[0] + r_t * {x}_strides[1];\n"


def _kernel(name: str, inputs: list[str], body: str):
    return mx.fast.metal_kernel(name=f"brainspa_mlx_{name}", input_names=inputs, output_names=["y"],
                                source=_SETUP + body, header=_HEADER, ensure_row_contiguous=False)


_SLERP = _kernel("slerp_norm", ["res", "out", "alpha"], _ROW3.format(x="res") + _base("res") + _base("out") + """
    threadgroup float pa[32];
    threadgroup float pb[32];
    threadgroup float py[32];
    float aa = 0, bb = 0;
    for (uint k = lane; k < D; k += TG) {
      float a = res[base_res + k], b = out[base_out + k];
      aa += a * a;
      bb += b * b;
    }
    const float ra = metal::precise::rsqrt(tg_sum(aa, pa, sg, n_sg) + NORM_EPS);
    const float rb = metal::precise::rsqrt(tg_sum(bb, pb, sg, n_sg) + NORM_EPS);
    float yy = 0;
    for (uint k = lane; k < D; k += TG) {
      float a = float(res[base_res + k]) * ra, b = float(out[base_out + k]) * rb;
      float v = a + alpha[k] * (b - a);
      yy += v * v;
    }
    const float ry = metal::precise::rsqrt(tg_sum(yy, py, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      float a = float(res[base_res + k]) * ra, b = float(out[base_out + k]) * rb;
      y[base_y + k] = bfloat16_t((a + alpha[k] * (b - a)) * ry);
    }
""")

_GATE = _kernel("gate_out", ["o", "g", "sg2"], _ROW3.format(x="o") + _base("o") + _base("g") + """
    threadgroup float py[32];
    float yy = 0;
    for (uint k = lane; k < D; k += TG) {
      float v = float(o[base_o + k]) * sigmoid_mx(float(g[base_g + k]) * sg2[k]);
      yy += v * v;
    }
    const float ry = metal::precise::rsqrt(tg_sum(yy, py, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      float v = float(o[base_o + k]) * sigmoid_mx(float(g[base_g + k]) * sg2[k]);
      y[base_y + k] = bfloat16_t(v * ry);
    }
""")

_MLP = _kernel("mlp_act", ["g", "u", "sg_", "su"], _ROW3.format(x="g") + _base("g") + _base("u") + """
    threadgroup float py[32];
    float yy = 0;
    for (uint k = lane; k < D; k += TG) {
      float gf = float(g[base_g + k]) * sg_[k];
      float v = gf * sigmoid_mx(gf) * (float(u[base_u + k]) * su[k]);
      yy += v * v;
    }
    const float ry = metal::precise::rsqrt(tg_sum(yy, py, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      float gf = float(g[base_g + k]) * sg_[k];
      float v = gf * sigmoid_mx(gf) * (float(u[base_u + k]) * su[k]);
      y[base_y + k] = bfloat16_t(v * ry);
    }
""")

# qk arrives as the transpose of (B, T, H, d), so its rows are not contiguous in memory: index it by strides. (Reading it
# as contiguous rows was right at batch 1 only, by layout luck, and wrong at batch 8: max error about 1.0 on the test.)
_QK = _kernel("qk_prep", ["qk", "scale"], """
    const uint D = qk_shape[3], T = qk_shape[2], heads = qk_shape[1];
    const uint t = row % T;
    const uint h = (row / T) % heads;
    const uint b = row / (T * heads);
    const uint base = b * qk_strides[0] + h * qk_strides[1] + t * qk_strides[2];
    const uint sbase = h * scale_strides[1];
    threadgroup float px[32];
    float xx = 0;
    for (uint k = lane; k < D; k += TG) {
      float x = qk[base + k * qk_strides[3]];
      xx += x * x;
    }
    const float r = metal::precise::rsqrt(tg_sum(xx, px, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      y[row * D + k] = bfloat16_t(float(qk[base + k * qk_strides[3]]) * r * scale[sbase + k * scale_strides[3]]);
    }
""")

_NORM = _kernel("justnorm", ["x"], _ROW3.format(x="x") + _base("x") + """
    threadgroup float px[32];
    float xx = 0;
    for (uint k = lane; k < D; k += TG) {
      float v = x[base_x + k];
      xx += v * v;
    }
    const float r = metal::precise::rsqrt(tg_sum(xx, px, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      y[base_y + k] = bfloat16_t(float(x[base_x + k]) * r);
    }
""")

_INJECT = _kernel("loop_inject", ["state", "input_h"], _ROW3.format(x="state") + _base("state") + _base("input_h") + """
    threadgroup float px[32];
    float xx = 0;
    for (uint k = lane; k < D; k += TG) {
      float v = float(bfloat16_t(float(state[base_state + k]) + float(input_h[base_input_h + k])));
      xx += v * v;
    }
    const float r = metal::precise::rsqrt(tg_sum(xx, px, sg, n_sg) + NORM_EPS);
    for (uint k = lane; k < D; k += TG) {
      float v = float(bfloat16_t(float(state[base_state + k]) + float(input_h[base_input_h + k])));
      y[base_y + k] = bfloat16_t(v * r);
    }
""")


def _run(kernel, inputs: list[mx.array], template: mx.array, rows: int) -> mx.array:
    return kernel(inputs=inputs, output_shapes=[template.shape], output_dtypes=[template.dtype],
                  grid=(rows * TG, 1, 1), threadgroup=(TG, 1, 1))[0]


def slerp_norm(res: mx.array, out: mx.array, alpha: mx.array) -> mx.array:
    return _run(_SLERP, [res, out, alpha], res, res.shape[0] * res.shape[1])


def gate_out(o: mx.array, g: mx.array, sg2: mx.array) -> mx.array:
    return _run(_GATE, [o, g, sg2], o, o.shape[0] * o.shape[1])


def mlp_act(g: mx.array, u: mx.array, sg: mx.array, su: mx.array) -> mx.array:
    return _run(_MLP, [g, u, sg, su], g, g.shape[0] * g.shape[1])


def qk_prep(qk: mx.array, scale: mx.array) -> mx.array:
    return _run(_QK, [qk, scale], qk, qk.shape[0] * qk.shape[1] * qk.shape[2])


def justnorm(x: mx.array) -> mx.array:
    return _run(_NORM, [x], x, x.shape[0] * x.shape[1])


def loop_inject(state: mx.array, input_h: mx.array) -> mx.array:
    return _run(_INJECT, [state, input_h], state, state.shape[0] * state.shape[1])


ALWAYS = ("slerp_norm", "qk_prep")  # faster at every row count
WIDE_ONLY = ("gate_out", "mlp_act", "justnorm", "loop_inject")  # at 1 to 4 rows these lose to the stock dispatches
FUNCS = ALWAYS + WIDE_ONLY
MIN_WIDE_ROWS = 5


def _wide(fast, stock):
    def route(x: mx.array, *rest: mx.array) -> mx.array:
        return fast(x, *rest) if x.shape[0] * x.shape[1] >= MIN_WIDE_ROWS else stock(x, *rest)

    return route


def install() -> None:
    """Route ``mlx_lm.models.dory`` to the kernels for this process."""
    from mlx_lm.models import dory

    for name in ALWAYS:
        setattr(dory, name, globals()[name])
    for name in WIDE_ONLY:
        setattr(dory, name, _wide(globals()[name], getattr(dory, name)))
