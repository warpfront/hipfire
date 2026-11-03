// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Generic N-stream residual ops: broadcast, route, aggregate, postmix, mean.
//!
//! One [`StreamMixOp`] is one stage over `rows` rows (T = 1 decode is
//! `rows == 1`); [`execute_stream_mix`] validates it and launches. Operands are
//! borrowed persistent buffers; nothing is allocated here and no device value
//! is read back, so every stage is graph-capture safe.
//!
//! A mixing site reads the OLD streams, so the stage order for one site is
//!
//! 1. `Route`     old streams -> `pre`, `post`, `res` controls
//! 2. `Aggregate` old streams, `pre` -> branch input (optionally normed)
//! 3. (branch runs outside this module)
//! 4. `Postmix`   old streams, branch output, `post`, `res` -> NEW streams in a
//!    separate ping-pong buffer
//!
//! and the model tail is `Mean` over the final streams (optionally normed).
//! `Broadcast` seeds the streams from one embedding row.
//!
//! Streams are F32 planes carrying model-dtype values; [`StreamRounding`]
//! selects bfloat16 rounding at every stage output (or none, for the F32
//! oracle mode). Chart policy is typed ([`StreamChart`]); the existing
//! DeepSeek4 Sinkhorn controls stay in `pipeline::mla::hc` and are not routed
//! through this module.

use crate::types::DispatchError;
pub use rdna_compute::stream_mix::StreamRounding;
use rdna_compute::stream_mix::{
    StreamAggregate, StreamGamma, StreamMean, StreamPostmix, StreamRouteAggregate,
    StreamRouteHelmert, STREAM_MIX_MAX_STREAMS, STREAM_MIX_MAX_WIDTH,
};
use rdna_compute::{DType, Gpu, GpuTensor};

fn hip_err(what: impl std::fmt::Display, e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("{what}: {e:?}"))
}

fn geometry(msg: impl Into<String>) -> DispatchError {
    DispatchError::Hip(format!("stream_mix: {}", msg.into()))
}

/// Plain RMSNorm folded into an aggregate/mean stage: `x * rsqrt(mean(x^2) +
/// eps) * gamma` (no `1 + gamma`), rounded per the op's [`StreamRounding`].
#[derive(Clone, Copy)]
pub struct StreamNorm<'a> {
    /// `[width]`, F32 or BF16.
    pub gamma: &'a GpuTensor,
    pub eps: f32,
}

/// Chart policy that turns the selector output into stream controls.
#[non_exhaustive]
pub enum StreamChart<'a> {
    /// `pre`/`post` sigmoid gates plus a stream-to-stream map from the bounded
    /// Helmert chart: `E = Q Z Q^T`, `rho = max(0, max(-E/B))`,
    /// `Phi = B + E/(1+rho)`, `res = (1-g) I + g Phi`, `g = sigmoid(eta)`,
    /// `B = (I + 11^T/n)/2`. Works for any `2 <= n <= 8`.
    HelmertBounded {
        /// Selector matrix `[2n + (n-1)^2, n * width]`, F32 (BF16-representable
        /// values are fine): rows `pre[n]`, `post[n]`, chart `[(n-1)^2]`.
        selector: &'a GpuTensor,
        /// F32 `[alpha_pre, alpha_post, alpha_z, eta, b_pre[n], b_post[n],
        /// b_z[(n-1)^2]]`.
        params: &'a GpuTensor,
        /// Multiplier of the sigmoid `post` gate.
        post_factor: f32,
        /// Epsilon of the weightless RMS scale applied after the selector GEMM.
        selector_eps: f32,
    },
}

/// Per-row controls, F32.
#[derive(Clone, Copy)]
pub struct StreamControls<'a> {
    /// `[rows, n]` aggregation weights.
    pub pre: &'a GpuTensor,
    /// `[rows, n]` branch write-back weights.
    pub post: &'a GpuTensor,
    /// `[rows, n, n]` residual map, destination/source (`res[i][j]` weights
    /// source stream `j` into destination stream `i`).
    pub res: &'a GpuTensor,
}

/// One stage of the stream mixing site chain.
pub enum StreamMixStage<'a> {
    /// `streams_out[r, s, :] = round(input[r, :])` for every stream `s`.
    Broadcast {
        /// `[rows, input_stride]` F32.
        input: &'a GpuTensor,
        input_stride: usize,
        /// `[rows, n, width]` (written).
        streams_out: &'a GpuTensor,
    },
    /// Selector GEMM (existing F32 projection GEMM) + chart finalize.
    Route {
        /// Old streams `[rows, n, width]` (read).
        streams: &'a GpuTensor,
        chart: StreamChart<'a>,
        /// Scratch `[rows, selector rows]` F32 for the raw selector output.
        selector_out: &'a GpuTensor,
        /// Controls (all three written).
        controls: StreamControls<'a>,
    },
    /// [`Route`](Self::Route) followed by [`Aggregate`](Self::Aggregate) of the same
    /// site, outputs bit-identical to running the two stages in order. With
    /// `HIPFIRE_RAON_MHC_FUSED=1` (gfx-independent, prefill rows >=
    /// [`MHC_FUSED_MIN_ROWS`], supported geometry) the old streams are read once for
    /// both the weightless RMS and the aggregate instead of twice; otherwise it
    /// launches the two stages.
    RouteAggregate {
        /// Old streams `[rows, n, width]` (read).
        streams: &'a GpuTensor,
        chart: StreamChart<'a>,
        /// Scratch `[rows, selector rows]` F32 for the raw selector output.
        selector_out: &'a GpuTensor,
        /// Controls (all three written).
        controls: StreamControls<'a>,
        norm: Option<StreamNorm<'a>>,
        /// `[rows, width]` (written).
        out: &'a GpuTensor,
    },
    /// `out = [norm](round(sum_s pre[s] * streams[s]))`.
    Aggregate {
        streams: &'a GpuTensor,
        /// Only `pre` is read.
        controls: StreamControls<'a>,
        norm: Option<StreamNorm<'a>>,
        /// `[rows, width]` (written).
        out: &'a GpuTensor,
    },
    /// `streams_out[i] = round(sum_j res[i,j] * streams[j] + post[i] * branch)`.
    Postmix {
        /// Old streams (read).
        streams: &'a GpuTensor,
        /// Branch output `[rows, width]` F32.
        branch: &'a GpuTensor,
        /// Only `post` and `res` are read.
        controls: StreamControls<'a>,
        /// New streams; must not alias `streams` (written).
        streams_out: &'a GpuTensor,
    },
    /// `out = [norm](round(mean_s streams[s]))`.
    Mean {
        streams: &'a GpuTensor,
        norm: Option<StreamNorm<'a>>,
        /// `[rows, width]` (written).
        out: &'a GpuTensor,
    },
    /// Copy the `rows` stream rows (`n * width` wide) into an F32 plane at row
    /// stride `dst_row_stride`, column `dst_col` (a speculative drafter's
    /// carrier capture, or an observer's F32 view of BF16 streams). BF16
    /// streams are widened exactly; F32 streams are copied.
    Capture {
        streams: &'a GpuTensor,
        /// F32, written at `[r * dst_row_stride + dst_col, +n * width)`.
        dst: &'a GpuTensor,
        dst_row_stride: usize,
        dst_col: usize,
    },
}

/// One stream-mix stage over `rows` rows of `streams` streams of `width`.
pub struct StreamMixOp<'a> {
    pub rows: usize,
    pub streams: usize,
    pub width: usize,
    pub rounding: StreamRounding,
    pub stage: StreamMixStage<'a>,
}

impl StreamMixOp<'_> {
    /// Checks geometry and capability only; launches nothing.
    pub fn validate(&self) -> Result<(), DispatchError> {
        if self.rows == 0 || self.width == 0 || self.streams == 0 {
            return Err(geometry("empty rows/streams/width"));
        }
        if self.streams > STREAM_MIX_MAX_STREAMS {
            return Err(DispatchError::UnsupportedVariant {
                family: "stream_mix",
                variant: "stream count above kernel limit",
                arch: "any",
                quant: "f32-plane",
            });
        }
        if self.width > STREAM_MIX_MAX_WIDTH {
            return Err(DispatchError::UnsupportedVariant {
                family: "stream_mix",
                variant: "row width above LDS-resident limit",
                arch: "any",
                quant: "f32-plane",
            });
        }
        if let StreamMixStage::Route { chart, .. } | StreamMixStage::RouteAggregate { chart, .. } =
            &self.stage
        {
            match chart {
                StreamChart::HelmertBounded { .. } if self.streams < 2 => {
                    return Err(geometry("HelmertBounded chart needs at least 2 streams"));
                }
                StreamChart::HelmertBounded { .. } => {}
            }
        }
        if let StreamMixStage::Capture {
            dst,
            dst_row_stride,
            dst_col,
            ..
        } = &self.stage
        {
            let row = self.streams * self.width;
            let need = (self.rows - 1)
                .checked_mul(*dst_row_stride)
                .and_then(|n| n.checked_add(*dst_col + row))
                .ok_or_else(|| geometry("capture extent overflows"))?;
            if dst.dtype != DType::F32 || *dst_row_stride < row || dst.numel() < need {
                return Err(geometry(format!(
                    "capture of {} rows x {row} needs an F32 destination of {need} elements at \
                     row stride >= {row} (got {:?} x {}, stride {dst_row_stride})",
                    self.rows,
                    dst.dtype,
                    dst.numel()
                )));
            }
        }
        Ok(())
    }

    /// Selector rows `2n + (n-1)^2` of a Helmert chart.
    pub fn helmert_selector_rows(streams: usize) -> usize {
        2 * streams + (streams - 1) * (streams - 1)
    }
}

fn gamma<'a>(norm: Option<StreamNorm<'a>>) -> Option<StreamGamma<'a>> {
    norm.map(|n| StreamGamma {
        weight: n.gamma,
        eps: n.eps,
    })
}

/// Batched GEMM `y[n, m] = a[m, :] . x[n, :]` (F32 activation `x`, F32 output) for
/// the small-M, large-K shapes of the mHC selector and the MoE router.
///
/// The weight `a` is BF16 (the checkpoint's own dtype; widening to F32 is exact,
/// so the kernels read half the bytes and widen in registers). Every kernel keeps
/// the per-output arithmetic of `gemm_f32_batched` (lane-strided FMA sum, width-32
/// shuffle tree), so the output is bit-identical to that kernel run on the widened
/// weight at every `n`.
///
/// Router-shaped calls (`m > SMALL_ROWS_SELECTOR_SHAPE_M`, F32 activation, `k` a
/// multiple of 512) on an arch with the register-tiled kernels (gfx1201, gfx1100:
/// [`small_rows_thresholds`]) run [`Gpu::gemm_bf16_small_rows`] at every row count, its
/// profile picked by `n` and the arch's thresholds ([`small_rows_profile`]);
/// `HIPFIRE_BF16W_SMALL_ROWS=0` restores the routes below.
///
/// Otherwise `n < BF16_TILED_ROWS_MIN_N` (decode, speculative verify) and every arch
/// but gfx1201 run `gemm_bf16_xf32_batched`. gfx1201 with `n >= BF16_TILED_ROWS_MIN_N`
/// (prefill, Raon pp512..16384) runs the x-stationary `gemm_bf16_xf32_tiled_rows`,
/// which reads each activation row once per 8*MPW outputs instead of once per
/// output; its few-block grid is far slower at n = 1..16 and it measured slower on
/// gfx1100. `HIPFIRE_BF16W_TILED_ROWS=0` runs the batched kernel everywhere.
///
/// A tensor the loader kept F32 (its source values were not all exactly
/// BF16-representable) runs `gemm_f32_batched`.
pub(crate) fn rows_gemm(
    gpu: &mut Gpu,
    a: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) -> rdna_compute::HipResult<()> {
    match a.dtype {
        DType::BF16 => {
            if let Some(profile) = small_rows_route(&gpu.arch, x.dtype == DType::BF16, m, k, n) {
                gpu.gemm_bf16_small_rows(a, x, y, m, k, n, profile)
            } else if n >= BF16_TILED_ROWS_MIN_N
                && gpu.arch == "gfx1201"
                && hipfire_config::developer_bool("HIPFIRE_BF16W_TILED_ROWS", true)
            {
                gpu.gemm_bf16_xf32_tiled_rows(a, x, y, m, k, n)
            } else {
                gpu.gemm_bf16_xf32_batched(a, x, y, m, k, n)
            }
        }
        _ => gpu.gemm_f32_batched(a, x, y, m, k, n),
    }
}

/// Output rows up to which a call is selector-shaped (Raon mHC selector, M = 17: BF16
/// activation rows, `_s*` small-rows kernels); larger M is router-shaped (MoE router,
/// M = 128: F32 activation rows, `_r*` kernels). A shape/kernel-family split, not a row
/// threshold: any other pairing keeps the batched/tiled routes.
const SMALL_ROWS_SELECTOR_SHAPE_M: usize = 32;

/// Row-count thresholds of the register-tiled small-rows profiles for one arch. The
/// kernels are bit-identical in every profile, so a threshold only moves speed: each
/// arch carries the crossovers measured on its own card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SmallRowsThresholds {
    /// Router shape (F32 activation): rows from which the 4-rows-per-CTA profile runs.
    f32_rows4: usize,
    /// Router shape: rows from which the 8-rows-per-CTA profile runs.
    f32_rows8: usize,
    /// Selector shape (BF16 activation): rows from which the 4-rows-per-CTA profile runs.
    bf16_rows4: usize,
    /// Selector shape: rows from which the 8-rows-per-CTA profile runs.
    bf16_rows8: usize,
}

// selector-receipt: receipts/selectors/small_rows_f32_rows4_min_rows.json
/// gfx1201 router-shaped (F32 activation) calls with `n` at least this many rows use the
/// 4-rows-per-CTA small-rows profile; fewer rows (decode, speculative verify) the
/// 1-row profile.
pub(crate) const SMALL_ROWS_F32_ROWS4_MIN_ROWS: usize = 16;
// selector-receipt: receipts/selectors/small_rows_f32_rows8_min_rows.json
/// gfx1201 router-shaped calls with `n` at least this many rows use the 8-rows-per-CTA profile.
pub(crate) const SMALL_ROWS_F32_ROWS8_MIN_ROWS: usize = 256;
// selector-receipt: receipts/selectors/small_rows_bf16_rows4_min_rows.json
/// gfx1201 selector-shaped (BF16 activation) calls with `n` at least this many rows use the
/// 4-rows-per-CTA profile; fewer rows the 1-row profile.
pub(crate) const SMALL_ROWS_BF16_ROWS4_MIN_ROWS: usize = 80;
// selector-receipt: receipts/selectors/small_rows_bf16_rows8_min_rows.json
/// gfx1201 selector-shaped calls with `n` at least this many rows use the 8-rows-per-CTA profile.
pub(crate) const SMALL_ROWS_BF16_ROWS8_MIN_ROWS: usize = 512;

// selector-receipt: receipts/selectors/small_rows_gfx1100_f32_rows4_min_rows.json
/// gfx1100 router-shaped (F32 activation) calls with `n` at least this many rows use the
/// 4-rows-per-CTA small-rows profile; fewer rows the 1-row profile (RX 7900 XTX: p1 wins to
/// N = 22, p4 from N = 24).
pub(crate) const SMALL_ROWS_GFX1100_F32_ROWS4_MIN_ROWS: usize = 24;
// selector-receipt: receipts/selectors/small_rows_gfx1100_f32_rows8_min_rows.json
/// gfx1100 router-shaped calls with `n` at least this many rows use the 8-rows-per-CTA profile.
pub(crate) const SMALL_ROWS_GFX1100_F32_ROWS8_MIN_ROWS: usize = 384;
// selector-receipt: receipts/selectors/small_rows_gfx1100_bf16_rows4_min_rows.json
/// gfx1100 selector-shaped (BF16 activation) calls with `n` at least this many rows use the
/// 4-rows-per-CTA profile; fewer rows the 1-row profile.
pub(crate) const SMALL_ROWS_GFX1100_BF16_ROWS4_MIN_ROWS: usize = 160;
// selector-receipt: receipts/selectors/small_rows_gfx1100_bf16_rows8_min_rows.json
/// gfx1100 selector-shaped calls with `n` at least this many rows use the 8-rows-per-CTA profile.
pub(crate) const SMALL_ROWS_GFX1100_BF16_ROWS8_MIN_ROWS: usize = 1024;

/// Thresholds of the archs the register-tiled small-rows kernels are admitted on, each
/// measured on its own card: gfx1201 in `receipts/selectors/evidence/small_rows_node_gfx1201.txt`
/// (R9700), gfx1100 in `receipts/selectors/evidence/small_rows_node_gfx1100.txt` (RX 7900 XTX,
/// bit-identical at every row count 1..4096). `None` for every other arch: a new arch needs
/// its own node measurement before it is listed here.
fn small_rows_thresholds(arch: &str) -> Option<SmallRowsThresholds> {
    match arch {
        "gfx1201" => Some(SmallRowsThresholds {
            f32_rows4: SMALL_ROWS_F32_ROWS4_MIN_ROWS,
            f32_rows8: SMALL_ROWS_F32_ROWS8_MIN_ROWS,
            bf16_rows4: SMALL_ROWS_BF16_ROWS4_MIN_ROWS,
            bf16_rows8: SMALL_ROWS_BF16_ROWS8_MIN_ROWS,
        }),
        "gfx1100" => Some(SmallRowsThresholds {
            f32_rows4: SMALL_ROWS_GFX1100_F32_ROWS4_MIN_ROWS,
            f32_rows8: SMALL_ROWS_GFX1100_F32_ROWS8_MIN_ROWS,
            bf16_rows4: SMALL_ROWS_GFX1100_BF16_ROWS4_MIN_ROWS,
            bf16_rows8: SMALL_ROWS_GFX1100_BF16_ROWS8_MIN_ROWS,
        }),
        _ => None,
    }
}

/// Register-tiled small-rows profile for one call, by arch, x dtype (router-shaped F32 /
/// selector-shaped BF16, see [`SMALL_ROWS_SELECTOR_SHAPE_M`]) and row count; `None` when
/// the arch has no measured thresholds or the shape is not one the kernel family is tuned for.
pub(crate) fn small_rows_profile(
    arch: &str,
    x_bf16: bool,
    m: usize,
    n: usize,
) -> Option<rdna_compute::gemm::Bf16SmallRowsProfile> {
    use rdna_compute::gemm::Bf16SmallRowsProfile as P;
    let t = small_rows_thresholds(arch)?;
    let (rows4, rows8) = if x_bf16 {
        if m > SMALL_ROWS_SELECTOR_SHAPE_M {
            return None;
        }
        (t.bf16_rows4, t.bf16_rows8)
    } else {
        if m <= SMALL_ROWS_SELECTOR_SHAPE_M {
            return None;
        }
        (t.f32_rows4, t.f32_rows8)
    };
    Some(if n >= rows8 {
        P::Rows8
    } else if n >= rows4 {
        P::Rows4
    } else {
        P::Rows1
    })
}

/// [`small_rows_profile`] gated by K (the kernels need `k` a multiple of
/// [`rdna_compute::gemm::BF16_SMALL_ROWS_K_MULTIPLE`]) and by `HIPFIRE_BF16W_SMALL_ROWS`
/// (developer flag, default ON: bit-identical to the batched/tiled routes; `0` restores them).
fn small_rows_route(
    arch: &str,
    x_bf16: bool,
    m: usize,
    k: usize,
    n: usize,
) -> Option<rdna_compute::gemm::Bf16SmallRowsProfile> {
    if !k.is_multiple_of(rdna_compute::gemm::BF16_SMALL_ROWS_K_MULTIPLE)
        || !hipfire_config::developer_bool("HIPFIRE_BF16W_SMALL_ROWS", true)
    {
        return None;
    }
    small_rows_profile(arch, x_bf16, m, n)
}

/// Smallest row count that takes the tiled kernel: the measured prefill region
/// (pp512 and up). Decode (n = 1) and speculative verify (n <= 16) stay on
/// `gemm_bf16_xf32_batched`.
// selector-receipt: receipts/selectors/bf16_tiled_rows_min_n.json
pub(crate) const BF16_TILED_ROWS_MIN_N: usize = 512;

/// True for a device tensor the selector / router GEMMs accept as the weight:
/// BF16, or F32 when the loader kept the F32 values.
pub(crate) fn rows_gemm_weight_ok(a: &GpuTensor) -> bool {
    matches!(a.dtype, DType::BF16 | DType::F32)
}

/// JITs every kernel [`execute_stream_mix`] can launch (stream-mix kernels and
/// the selector GEMMs) so a later graph capture compiles nothing.
pub fn preload_stream_mix(gpu: &mut Gpu) -> Result<(), DispatchError> {
    gpu.stream_mix_preload()
        .map_err(|e| hip_err("stream_mix preload", e))?;
    gpu.ensure_kernel_public(
        "gemm_bf16_xf32_batched",
        rdna_compute::kernels::GEMM_BF16_XF32_BATCHED_SRC,
        "gemm_bf16_xf32_batched",
    )
    .map_err(|e| hip_err("stream_mix selector gemm preload", e))?;
    gpu.ensure_kernel_public(
        "gemm_bf16_xf32_tiled_rows",
        rdna_compute::kernels::GEMM_BF16_XF32_TILED_ROWS_SRC,
        "gemm_bf16_xf32_tiled_rows",
    )
    .map_err(|e| hip_err("stream_mix selector tiled gemm preload", e))?;
    gpu.ensure_kernel_public(
        "gemm_bf16_xf32_tiled_rows",
        rdna_compute::kernels::GEMM_BF16_XF32_TILED_ROWS_SRC,
        "gemm_bf16_xf32_tiled_rows_mp4",
    )
    .map_err(|e| hip_err("stream_mix selector tiled gemm mp4 preload", e))?;
    for sym in [
        "gemm_bf16_xf32_small_rows_r1",
        "gemm_bf16_xf32_small_rows_r4",
        "gemm_bf16_xf32_small_rows_r8",
        "gemm_bf16_xbf16_small_rows_s1",
        "gemm_bf16_xbf16_small_rows_s4",
        "gemm_bf16_xbf16_small_rows_s8",
    ] {
        gpu.ensure_kernel_public(
            "gemm_bf16_small_rows",
            rdna_compute::kernels::GEMM_BF16_SMALL_ROWS_SRC,
            sym,
        )
        .map_err(|e| hip_err("stream_mix small-rows gemm preload", e))?;
    }
    gpu.ensure_kernel_public(
        "gemm_f32_batched",
        rdna_compute::kernels::GEMM_F32_SRC,
        "gemm_f32_batched",
    )
    .map_err(|e| hip_err("stream_mix selector f32 gemm preload", e))
}

// selector-receipt: receipts/selectors/mhc_fused_min_rows.json
/// Smallest row count that takes the fused route+aggregate kernel. Decode and
/// speculative verify (rows <= 16) keep the two-stage chain.
pub const MHC_FUSED_MIN_ROWS: usize = 64;

/// `HIPFIRE_RAON_MHC_FUSED`: default ON (bit-identical); `0` kills; `1` fuses the Helmert route finalize into
/// the aggregate (one pass over the old streams), `0` is the kill switch.
fn mhc_fused_admitted(n: usize, width: usize, rows: usize) -> bool {
    rows >= MHC_FUSED_MIN_ROWS
        && Gpu::stream_mix_route_aggregate_supported(n, width)
        && hipfire_config::developer_bool("HIPFIRE_RAON_MHC_FUSED", true)
}

/// Selector projection `selector_out[r, :] = selector . streams[r, :]` of a
/// Helmert chart; the finalize kernel applies the model-dtype round and the
/// inverse-RMS scale.
fn helmert_selector_gemm(
    gpu: &mut Gpu,
    selector: &GpuTensor,
    streams: &GpuTensor,
    selector_out: &GpuTensor,
    n: usize,
    width: usize,
    rows: usize,
) -> Result<(), DispatchError> {
    let zdim = StreamMixOp::helmert_selector_rows(n);
    let x_dim = n * width;
    if !rows_gemm_weight_ok(selector) || selector.numel() < zdim * x_dim {
        return Err(geometry(format!(
            "HelmertBounded selector must be BF16 (or F32) [{zdim}, {x_dim}]"
        )));
    }
    if streams.dtype == DType::BF16 {
        // BF16 streams: the selector is read by the BF16-activation small-rows
        // kernels (or their batched/tiled twins under the kill switches), all with the
        // lane-strided/shuffle-tree arithmetic of gemm_f32_batched (bit-identical outputs).
        if selector.dtype != DType::BF16 {
            return Err(geometry("BF16 streams need a BF16 selector weight"));
        }
        if let Some(profile) = small_rows_route(&gpu.arch, true, zdim, x_dim, rows) {
            return gpu
                .gemm_bf16_small_rows(selector, streams, selector_out, zdim, x_dim, rows, profile)
                .map_err(|e| hip_err("stream_mix selector gemm (BF16 streams)", e));
        }
        let tiled = rows >= BF16_TILED_ROWS_MIN_N
            && gpu.arch == "gfx1201"
            && hipfire_config::developer_bool("HIPFIRE_BF16W_TILED_ROWS", true);
        return gpu
            .gemm_bf16_xbf16_rows(selector, streams, selector_out, zdim, x_dim, rows, tiled)
            .map_err(|e| hip_err("stream_mix selector gemm (BF16 streams)", e));
    }
    rows_gemm(gpu, selector, streams, selector_out, zdim, x_dim, rows)
        .map_err(|e| hip_err("stream_mix selector gemm", e))
}

/// Validates and launches one stream-mix stage.
pub fn execute_stream_mix(gpu: &mut Gpu, op: &StreamMixOp<'_>) -> Result<(), DispatchError> {
    op.validate()?;
    let (rows, n, width, rounding) = (op.rows, op.streams, op.width, op.rounding);
    match &op.stage {
        StreamMixStage::Broadcast {
            input,
            input_stride,
            streams_out,
        } => gpu
            .stream_mix_broadcast(input, streams_out, rows, n, width, *input_stride, rounding)
            .map_err(|e| hip_err("stream_mix broadcast", e)),
        StreamMixStage::Route {
            streams,
            chart,
            selector_out,
            controls,
        } => match chart {
            StreamChart::HelmertBounded {
                selector,
                params,
                post_factor,
                selector_eps,
            } => {
                helmert_selector_gemm(gpu, selector, streams, selector_out, n, width, rows)?;
                gpu.stream_mix_route_helmert(&StreamRouteHelmert {
                    streams,
                    z_raw: selector_out,
                    params,
                    pre: controls.pre,
                    post: controls.post,
                    res: controls.res,
                    rows,
                    n,
                    width,
                    post_factor: *post_factor,
                    selector_eps: *selector_eps,
                    rounding,
                })
                .map_err(|e| hip_err("stream_mix route_helmert", e))
            }
        },
        StreamMixStage::RouteAggregate {
            streams,
            chart,
            selector_out,
            controls,
            norm,
            out,
        } => match chart {
            StreamChart::HelmertBounded {
                selector,
                params,
                post_factor,
                selector_eps,
            } => {
                helmert_selector_gemm(gpu, selector, streams, selector_out, n, width, rows)?;
                if streams.dtype == DType::BF16 || mhc_fused_admitted(n, width, rows) {
                    return gpu
                        .stream_mix_route_aggregate(&StreamRouteAggregate {
                            streams,
                            z_raw: selector_out,
                            params,
                            pre: controls.pre,
                            post: controls.post,
                            res: controls.res,
                            norm: gamma(*norm),
                            out,
                            rows,
                            n,
                            width,
                            post_factor: *post_factor,
                            selector_eps: *selector_eps,
                            rounding,
                        })
                        .map_err(|e| hip_err("stream_mix route_aggregate", e));
                }
                gpu.stream_mix_route_helmert(&StreamRouteHelmert {
                    streams,
                    z_raw: selector_out,
                    params,
                    pre: controls.pre,
                    post: controls.post,
                    res: controls.res,
                    rows,
                    n,
                    width,
                    post_factor: *post_factor,
                    selector_eps: *selector_eps,
                    rounding,
                })
                .map_err(|e| hip_err("stream_mix route_helmert", e))?;
                gpu.stream_mix_aggregate(&StreamAggregate {
                    streams,
                    pre: controls.pre,
                    norm: gamma(*norm),
                    out,
                    rows,
                    n,
                    width,
                    rounding,
                })
                .map_err(|e| hip_err("stream_mix aggregate", e))
            }
        },
        StreamMixStage::Aggregate {
            streams,
            controls,
            norm,
            out,
        } => gpu
            .stream_mix_aggregate(&StreamAggregate {
                streams,
                pre: controls.pre,
                norm: gamma(*norm),
                out,
                rows,
                n,
                width,
                rounding,
            })
            .map_err(|e| hip_err("stream_mix aggregate", e)),
        StreamMixStage::Postmix {
            streams,
            branch,
            controls,
            streams_out,
        } => gpu
            .stream_mix_postmix(&StreamPostmix {
                streams,
                branch,
                post: controls.post,
                res: controls.res,
                out: streams_out,
                rows,
                n,
                width,
                rounding,
            })
            .map_err(|e| hip_err("stream_mix postmix", e)),
        StreamMixStage::Mean { streams, norm, out } => gpu
            .stream_mix_mean(&StreamMean {
                streams,
                norm: gamma(*norm),
                out,
                rows,
                n,
                width,
                rounding,
            })
            .map_err(|e| hip_err("stream_mix mean", e)),
        StreamMixStage::Capture {
            streams,
            dst,
            dst_row_stride,
            dst_col,
        } => {
            let row = n * width;
            if streams.dtype == DType::BF16 {
                gpu.stream_mix_widen_rows(streams, dst, rows, row, *dst_row_stride, *dst_col)
            } else {
                gpu.copy_rows_strided_f32(
                    streams,
                    dst,
                    rows,
                    row,
                    row,
                    *dst_row_stride,
                    *dst_col,
                    None,
                )
            }
            .map_err(|e| hip_err("stream_mix capture", e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_rows_match_helmert_geometry() {
        assert_eq!(StreamMixOp::helmert_selector_rows(4), 17);
        assert_eq!(StreamMixOp::helmert_selector_rows(3), 10);
        assert_eq!(StreamMixOp::helmert_selector_rows(2), 5);
    }

    /// Expected profile of `n` rows given the `[rows4, rows8)` floors: every n in 1..=16384 is
    /// checked, so the floors are exact, not spot-checked.
    fn assert_profile_thresholds(arch: &str, x_bf16: bool, m: usize, rows4: usize, rows8: usize) {
        use rdna_compute::gemm::Bf16SmallRowsProfile as P;
        for n in 1..=16384 {
            let want = if n >= rows8 {
                P::Rows8
            } else if n >= rows4 {
                P::Rows4
            } else {
                P::Rows1
            };
            assert_eq!(
                small_rows_profile(arch, x_bf16, m, n),
                Some(want),
                "{arch} x_bf16 = {x_bf16} m = {m} n = {n}"
            );
        }
    }

    #[test]
    fn small_rows_profile_follows_the_activation_dtype_and_row_count() {
        use rdna_compute::gemm::Bf16SmallRowsProfile as P;
        // gfx1201. Router shape: F32 activation, M = 128; selector shape: BF16 activation, M = 17.
        assert_profile_thresholds("gfx1201", false, 128, 16, 256);
        assert_profile_thresholds("gfx1201", true, 17, 80, 512);
        for (n, want) in [
            (15, P::Rows1),
            (16, P::Rows4),
            (255, P::Rows4),
            (256, P::Rows8),
        ] {
            assert_eq!(small_rows_profile("gfx1201", false, 128, n), Some(want));
        }
        for (n, want) in [
            (79, P::Rows1),
            (80, P::Rows4),
            (511, P::Rows4),
            (512, P::Rows8),
        ] {
            assert_eq!(small_rows_profile("gfx1201", true, 17, n), Some(want));
        }
        // gfx1100 (RX 7900 XTX): router 1..=23 Rows1, 24..=383 Rows4, 384+ Rows8; selector
        // 1..=159 Rows1, 160..=1023 Rows4, 1024+ Rows8.
        assert_profile_thresholds("gfx1100", false, 128, 24, 384);
        assert_profile_thresholds("gfx1100", true, 17, 160, 1024);
        for (n, want) in [
            (1, P::Rows1),
            (23, P::Rows1),
            (24, P::Rows4),
            (383, P::Rows4),
            (384, P::Rows8),
            (4096, P::Rows8),
        ] {
            assert_eq!(small_rows_profile("gfx1100", false, 128, n), Some(want));
        }
        for (n, want) in [
            (1, P::Rows1),
            (159, P::Rows1),
            (160, P::Rows4),
            (1023, P::Rows4),
            (1024, P::Rows8),
            (4096, P::Rows8),
        ] {
            assert_eq!(small_rows_profile("gfx1100", true, 17, n), Some(want));
        }
        // The thresholds are the shipped constants.
        assert_eq!(
            small_rows_thresholds("gfx1100"),
            Some(SmallRowsThresholds {
                f32_rows4: SMALL_ROWS_GFX1100_F32_ROWS4_MIN_ROWS,
                f32_rows8: SMALL_ROWS_GFX1100_F32_ROWS8_MIN_ROWS,
                bf16_rows4: SMALL_ROWS_GFX1100_BF16_ROWS4_MIN_ROWS,
                bf16_rows8: SMALL_ROWS_GFX1100_BF16_ROWS8_MIN_ROWS,
            })
        );
        // The other activation/M pairings keep the batched/tiled routes, on both archs.
        for arch in ["gfx1201", "gfx1100"] {
            for n in [1, 15, 16, 24, 255, 256, 384, 512, 1024, 4096] {
                assert_eq!(
                    small_rows_profile(arch, false, 17, n),
                    None,
                    "{arch} F32 x, M = 17, n = {n}"
                );
                assert_eq!(
                    small_rows_profile(arch, true, 128, n),
                    None,
                    "{arch} BF16 x, M = 128, n = {n}"
                );
            }
        }
        // An arch without a node measurement keeps the batched/tiled routes.
        for arch in ["gfx1151", "gfx942", "gfx1030", ""] {
            for n in [1, 16, 512, 4096] {
                assert_eq!(small_rows_profile(arch, false, 128, n), None, "{arch}");
                assert_eq!(small_rows_profile(arch, true, 17, n), None, "{arch}");
                assert_eq!(small_rows_route(arch, false, 128, 2560, n), None, "{arch}");
            }
        }
    }
}
