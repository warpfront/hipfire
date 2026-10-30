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
    StreamAggregate, StreamGamma, StreamMean, StreamPostmix, StreamRouteHelmert,
    STREAM_MIX_MAX_STREAMS, STREAM_MIX_MAX_WIDTH,
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
        if let StreamMixStage::Route { chart, .. } = &self.stage {
            match chart {
                StreamChart::HelmertBounded { .. } if self.streams < 2 => {
                    return Err(geometry("HelmertBounded chart needs at least 2 streams"));
                }
                StreamChart::HelmertBounded { .. } => {}
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

/// JITs every kernel [`execute_stream_mix`] can launch (stream-mix kernels and
/// the batched F32 selector GEMM) so a later graph capture compiles nothing.
pub fn preload_stream_mix(gpu: &mut Gpu) -> Result<(), DispatchError> {
    gpu.stream_mix_preload()
        .map_err(|e| hip_err("stream_mix preload", e))?;
    gpu.ensure_kernel_public(
        "gemm_f32_batched",
        rdna_compute::kernels::GEMM_F32_SRC,
        "gemm_f32_batched",
    )
    .map_err(|e| hip_err("stream_mix selector gemm preload", e))
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
                let zdim = StreamMixOp::helmert_selector_rows(n);
                let x_dim = n * width;
                if selector.dtype != DType::F32 || selector.numel() < zdim * x_dim {
                    return Err(geometry(format!(
                        "HelmertBounded selector must be F32 [{zdim}, {x_dim}]"
                    )));
                }
                // Selector projection through the existing batched F32 GEMM:
                // selector_out[r, :] = selector . streams[r, :]. The kernel below
                // applies the model-dtype round and the inverse-RMS scale.
                gpu.gemm_f32_batched(selector, streams, selector_out, zdim, x_dim, rows)
                    .map_err(|e| hip_err("stream_mix selector gemm", e))?;
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
}
