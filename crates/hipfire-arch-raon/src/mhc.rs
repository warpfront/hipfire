// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon four-stream mHC binding onto generic StreamMixOp. Owner: slice H.
//!
//! The numerical work is the generic `hipfire_dispatch::pipeline::stream_mix`
//! op with the `HelmertBounded` chart. This file only binds the checkpoint's
//! per-site parameters (`self_attn_raon_mhc.*` / `mlp_raon_mhc.*`, 17 selector
//! controls) to it and fixes the Raon constants. It contains no kernels.
//!
//! Checkpoint tensors of one site (all BF16 except `res.eta`, F32):
//!
//! | name                       | shape       | role                               |
//! |----------------------------|-------------|------------------------------------|
//! | `phi_pre`                  | `[4, 10240]`| selector rows 0..4                 |
//! | `phi_post`                 | `[4, 10240]`| selector rows 4..8                 |
//! | `res.phi_z`                | `[9, 10240]`| selector rows 8..17 (chart)        |
//! | `alpha_pre/alpha_post`     | scalar      | gate scales                        |
//! | `res.alpha_z`              | scalar      | chart scale                        |
//! | `b_pre`, `b_post`          | `[4]`       | gate biases                        |
//! | `res.b_z`                  | `[9]`       | chart bias                         |
//! | `res.eta`                  | scalar F32  | identity/chart blend `sigmoid(eta)`|
//!
//! The selector is uploaded as one `[17, 10240]` F32 matrix (BF16 values
//! widened losslessly, rows `pre|post|z`); alphas, biases and `eta` as one F32
//! vector `[alpha_pre, alpha_post, alpha_z, eta, b_pre, b_post, b_z]`.

use hipfire_dispatch::pipeline::stream_mix::{
    execute_stream_mix, StreamChart, StreamControls, StreamMixOp, StreamMixStage, StreamNorm,
    StreamRounding,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Raon-1-mini stream count.
pub const RAON_STREAMS: usize = 4;
/// Raon-1-mini hidden width.
pub const RAON_HIDDEN: usize = 2560;
/// `post` gate multiplier (`raon_mhc_post_factor`).
pub const RAON_POST_FACTOR: f32 = 2.0;
/// Weightless selector RMS epsilon (`raon_mhc_selector_eps`).
pub const RAON_SELECTOR_EPS: f32 = 1e-5;

/// Geometry and numerics of every mHC site of one model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaonMhcShape {
    pub streams: usize,
    pub width: usize,
    pub post_factor: f32,
    pub selector_eps: f32,
    /// `Bf16` for the model-dtype path; `F32` only for the labelled F32 oracle.
    pub rounding: StreamRounding,
}

impl RaonMhcShape {
    /// Official geometry: 4 streams, hidden 2560, BF16 boundaries.
    pub const fn raon_mini() -> Self {
        Self {
            streams: RAON_STREAMS,
            width: RAON_HIDDEN,
            post_factor: RAON_POST_FACTOR,
            selector_eps: RAON_SELECTOR_EPS,
            rounding: StreamRounding::Bf16,
        }
    }

    /// Selector rows `2n + (n-1)^2` (17 for n = 4).
    pub fn selector_rows(&self) -> usize {
        StreamMixOp::helmert_selector_rows(self.streams)
    }

    /// Selector columns `n * width` (10240).
    pub fn selector_cols(&self) -> usize {
        self.streams * self.width
    }

    fn check(&self) -> Result<(), String> {
        if self.streams < 2 || self.width == 0 {
            return Err(format!("mhc: invalid shape {self:?}"));
        }
        Ok(())
    }
}

/// Host values of one site, already widened to F32 (use [`bf16_bits_to_f32`]
/// for checkpoint BF16 data).
pub struct RaonMhcHost<'a> {
    pub phi_pre: &'a [f32],
    pub phi_post: &'a [f32],
    pub phi_z: &'a [f32],
    pub alpha_pre: f32,
    pub alpha_post: f32,
    pub alpha_z: f32,
    pub b_pre: &'a [f32],
    pub b_post: &'a [f32],
    pub b_z: &'a [f32],
    pub eta: f32,
}

/// Lossless BF16 -> F32 widening of checkpoint bits.
pub fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

/// Device parameters of one mHC site (attention or MLP).
pub struct RaonMhcSite {
    shape: RaonMhcShape,
    selector: GpuTensor,
    params: GpuTensor,
}

/// Per-launch scratch shared by all sites of a forward: the raw selector
/// output and the controls. Sized for `max_rows`; reused by every site because
/// controls are consumed (aggregate, postmix) before the next site routes.
pub struct RaonMhcWork {
    max_rows: usize,
    selector_out: GpuTensor,
    pre: GpuTensor,
    post: GpuTensor,
    res: GpuTensor,
}

fn herr(what: &str, e: impl std::fmt::Debug) -> String {
    format!("{what}: {e:?}")
}

impl RaonMhcWork {
    pub fn new(gpu: &mut Gpu, shape: &RaonMhcShape, max_rows: usize) -> Result<Self, String> {
        shape.check()?;
        if max_rows == 0 {
            return Err("mhc work: max_rows == 0".into());
        }
        let (n, z) = (shape.streams, shape.selector_rows());
        let mut alloc = |dims: &[usize]| {
            gpu.alloc_tensor(dims, DType::F32)
                .map_err(|e| herr("mhc work alloc", e))
        };
        Ok(Self {
            max_rows,
            selector_out: alloc(&[max_rows, z])?,
            pre: alloc(&[max_rows, n])?,
            post: alloc(&[max_rows, n])?,
            res: alloc(&[max_rows, n, n])?,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Controls of the last routed rows (`pre`, `post`, `res`; F32).
    pub fn controls(&self) -> StreamControls<'_> {
        StreamControls {
            pre: &self.pre,
            post: &self.post,
            res: &self.res,
        }
    }

    /// Raw selector GEMM output `[rows, 17]` of the last route.
    pub fn selector_out(&self) -> &GpuTensor {
        &self.selector_out
    }

    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        for t in [self.selector_out, self.pre, self.post, self.res] {
            gpu.free_tensor(t).map_err(|e| herr("mhc work free", e))?;
        }
        Ok(())
    }
}

impl RaonMhcSite {
    /// Uploads one site's parameters. Refuses wrong lengths and non-finite values.
    pub fn upload(
        gpu: &mut Gpu,
        shape: &RaonMhcShape,
        host: &RaonMhcHost<'_>,
    ) -> Result<Self, String> {
        shape.check()?;
        let (n, w) = (shape.streams, shape.width);
        let nm2 = (n - 1) * (n - 1);
        let cols = n * w;
        let expect = [
            ("phi_pre", host.phi_pre.len(), n * cols),
            ("phi_post", host.phi_post.len(), n * cols),
            ("res.phi_z", host.phi_z.len(), nm2 * cols),
            ("b_pre", host.b_pre.len(), n),
            ("b_post", host.b_post.len(), n),
            ("res.b_z", host.b_z.len(), nm2),
        ];
        for (name, got, want) in expect {
            if got != want {
                return Err(format!("mhc: {name} has {got} values, expected {want}"));
            }
        }
        let mut selector = Vec::with_capacity(shape.selector_rows() * cols);
        selector.extend_from_slice(host.phi_pre);
        selector.extend_from_slice(host.phi_post);
        selector.extend_from_slice(host.phi_z);
        let mut params = vec![host.alpha_pre, host.alpha_post, host.alpha_z, host.eta];
        params.extend_from_slice(host.b_pre);
        params.extend_from_slice(host.b_post);
        params.extend_from_slice(host.b_z);
        if selector.iter().chain(params.iter()).any(|v| !v.is_finite()) {
            return Err("mhc: non-finite site parameter".into());
        }
        let selector_t = gpu
            .upload_f32(&selector, &[shape.selector_rows(), cols])
            .map_err(|e| herr("mhc selector upload", e))?;
        let params_t = match gpu.upload_f32(&params, &[params.len()]) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(selector_t);
                return Err(herr("mhc params upload", e));
            }
        };
        Ok(Self {
            shape: *shape,
            selector: selector_t,
            params: params_t,
        })
    }

    pub fn shape(&self) -> &RaonMhcShape {
        &self.shape
    }

    pub fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        gpu.free_tensor(self.selector)
            .map_err(|e| herr("mhc free", e))?;
        gpu.free_tensor(self.params)
            .map_err(|e| herr("mhc free", e))
    }

    fn op<'a>(&self, rows: usize, stage: StreamMixStage<'a>) -> StreamMixOp<'a> {
        StreamMixOp {
            rows,
            streams: self.shape.streams,
            width: self.shape.width,
            rounding: self.shape.rounding,
            stage,
        }
    }

    fn check_rows(&self, rows: usize, work: &RaonMhcWork) -> Result<(), String> {
        if rows == 0 || rows > work.max_rows {
            return Err(format!("mhc: rows {rows} outside 1..={}", work.max_rows));
        }
        Ok(())
    }

    /// Controls `pre/post/res` for `rows` rows from the OLD streams
    /// `[rows, n, width]`, into `work`. Call before the branch runs.
    pub fn route(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        streams: &GpuTensor,
        work: &RaonMhcWork,
    ) -> Result<(), String> {
        self.check_rows(rows, work)?;
        let op = self.op(
            rows,
            StreamMixStage::Route {
                streams,
                chart: StreamChart::HelmertBounded {
                    selector: &self.selector,
                    params: &self.params,
                    post_factor: self.shape.post_factor,
                    selector_eps: self.shape.selector_eps,
                },
                selector_out: &work.selector_out,
                controls: work.controls(),
            },
        );
        execute_stream_mix(gpu, &op).map_err(|e| e.to_string())
    }

    /// `out[rows, width] = RMSNorm(round(sum_s pre_s * X_s))` (`norm` is the
    /// layer's plain-gamma input/post-attention norm).
    pub fn aggregate(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        streams: &GpuTensor,
        work: &RaonMhcWork,
        norm: StreamNorm<'_>,
        out: &GpuTensor,
    ) -> Result<(), String> {
        self.check_rows(rows, work)?;
        let op = self.op(
            rows,
            StreamMixStage::Aggregate {
                streams,
                controls: work.controls(),
                norm: Some(norm),
                out,
            },
        );
        execute_stream_mix(gpu, &op).map_err(|e| e.to_string())
    }

    /// New streams `X'_i = round(sum_j res_ij X_j + post_i * branch)` into the
    /// ping-pong buffer `streams_out`, using the controls routed from the old
    /// streams.
    pub fn postmix(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        streams: &GpuTensor,
        branch: &GpuTensor,
        work: &RaonMhcWork,
        streams_out: &GpuTensor,
    ) -> Result<(), String> {
        self.check_rows(rows, work)?;
        let op = self.op(
            rows,
            StreamMixStage::Postmix {
                streams,
                branch,
                controls: work.controls(),
                streams_out,
            },
        );
        execute_stream_mix(gpu, &op).map_err(|e| e.to_string())
    }
}

/// Replicates each embedding row (`[rows, emb_stride]`, F32 holding BF16
/// values) into all streams `[rows, n, width]`.
pub fn broadcast_embedding(
    gpu: &mut Gpu,
    shape: &RaonMhcShape,
    rows: usize,
    embedding: &GpuTensor,
    emb_stride: usize,
    streams_out: &GpuTensor,
) -> Result<(), String> {
    shape.check()?;
    let op = StreamMixOp {
        rows,
        streams: shape.streams,
        width: shape.width,
        rounding: shape.rounding,
        stage: StreamMixStage::Broadcast {
            input: embedding,
            input_stride: emb_stride,
            streams_out,
        },
    };
    execute_stream_mix(gpu, &op).map_err(|e| e.to_string())
}

/// Final hidden state: `RMSNorm(round(mean_s X_s))` with plain gamma (`model.norm`).
pub fn final_mean_norm(
    gpu: &mut Gpu,
    shape: &RaonMhcShape,
    rows: usize,
    streams: &GpuTensor,
    norm: StreamNorm<'_>,
    out: &GpuTensor,
) -> Result<(), String> {
    shape.check()?;
    let op = StreamMixOp {
        rows,
        streams: shape.streams,
        width: shape.width,
        rounding: shape.rounding,
        stage: StreamMixStage::Mean {
            streams,
            norm: Some(norm),
            out,
        },
    };
    execute_stream_mix(gpu, &op).map_err(|e| e.to_string())
}

/// JITs all mHC kernels before graph capture / warmup.
pub fn preload(gpu: &mut Gpu) -> Result<(), String> {
    hipfire_dispatch::pipeline::stream_mix::preload_stream_mix(gpu).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raon_shape_has_17_controls_over_10240_inputs() {
        let s = RaonMhcShape::raon_mini();
        assert_eq!(s.selector_rows(), 17);
        assert_eq!(s.selector_cols(), 10240);
    }

    #[test]
    fn bf16_widening_is_exact() {
        let bits = [0x3f80u16, 0xc000, 0x0000, 0x3e80];
        assert_eq!(bf16_bits_to_f32(&bits), vec![1.0, -2.0, 0.0, 0.25]);
    }
}
