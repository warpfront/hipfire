// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon RDA binding (`raon_rda_prepare`) onto generic GatedDeltaOp. Owner: slice R.
//!
//! Composition per RDA layer (PLAN §1.5), N rows in one pass, no per-token
//! launches:
//!
//! ```text
//! (caller, projection API)  q_raw=Wq x  k_raw=Wk x  v_raw=Wv x  z=Wg x
//!                           b=Wb x  delta=Wrda_w x  f=Wf_b(Wf_a x)
//! 3x causal_conv_depthwise_silu_rows_f32   (K=4, no bias, SiLU, BF16 round, history update)
//! raon_rda_prepare_f32                     (q/k norm, log_alpha, beta, vprime)
//! execute_gated_delta                      (GatedDeltaOp, PerKeyChannel, y BF16)
//! gated_rmsnorm_sigmoid_rows_f32           (per-head RMSNorm * sigmoid(z), one BF16 cast)
//! (caller)                  branch_output = Wo gated
//! ```
//!
//! All activations are FP32 planes holding BF16-rounded values (the projection
//! API must deliver BF16-rounded projections). Projections and the output
//! projection stay with the caller (slice A projection API) so this module has
//! no weight-codec dependency.
//!
//! Conv history is stored raw/pre-conv, `[K-1, P]`, oldest -> newest.

use hipfire_dispatch::ops::delta_net::{
    execute_gated_delta, DecayAxis, GatedDeltaOp, GatedDeltaStrides,
};
use rdna_compute::gated_delta_vector::{
    CausalConvRowsGeom, GatedRmsnormRowsGeom, RdaPrepareLaunch,
};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Fixed RDA geometry and constants of the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RdaGeometry {
    pub n_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// Per-head output RMSNorm epsilon.
    pub rms_eps: f32,
    /// q/k L2 normalization clamp.
    pub norm_eps: f32,
    /// `log_alpha = lower_bound * sigmoid(u)`.
    pub gate_lower_bound: f32,
    /// Model dtype boundaries: true = BF16 rounding (checkpoint), false = FP32
    /// model dtype (F32-mode oracle).
    pub round_bf16: bool,
}

impl RdaGeometry {
    /// Raon-1-mini-preview: 40 heads x 128, conv K=4, lower bound -5.
    pub fn raon() -> Self {
        Self {
            n_heads: 40,
            head_dim: 128,
            conv_kernel: 4,
            rms_eps: 1e-5,
            norm_eps: 1e-6,
            gate_lower_bound: -5.0,
            round_bf16: true,
        }
    }

    /// Projection width `P = n_heads * head_dim`.
    pub fn proj(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Elements of one slot's recurrent state `[H, V, K]`.
    pub fn state_elems(&self) -> usize {
        self.n_heads * self.head_dim * self.head_dim
    }
}

/// Immutable per-layer weights (F32 device tensors; BF16-origin values widened).
pub struct RaonRdaLayer<'a> {
    /// Depthwise conv weights `[P, K]` for q, k, v.
    pub conv_w_q: &'a GpuTensor,
    pub conv_w_k: &'a GpuTensor,
    pub conv_w_v: &'a GpuTensor,
    /// `[H]` F32.
    pub a_log: &'a GpuTensor,
    /// `[H*K]` F32 (`[H, K]` row-major).
    pub dt_bias: &'a GpuTensor,
    /// Shared per-head RMSNorm gamma `[K]` F32.
    pub o_norm_gamma: &'a GpuTensor,
}

/// Mutable per-slot layer state: FP32 `[H,V,K]` recurrent matrix and the three
/// raw conv histories `[K-1, P]` (oldest -> newest).
pub struct RdaLayerState<'a> {
    pub state: &'a GpuTensor,
    pub hist_q: &'a GpuTensor,
    pub hist_k: &'a GpuTensor,
    pub hist_v: &'a GpuTensor,
    /// Slot index into `state` (multiples of `state_slot_stride`); 0 for a
    /// per-slot view.
    pub slot: usize,
    /// Elements between slots in `state`; `geom.state_elems()` for dense.
    pub state_slot_stride: usize,
}

/// Projected activations, row-major, BF16-rounded FP32 planes.
pub struct RdaProjected<'a> {
    pub q_raw: &'a GpuTensor,
    pub k_raw: &'a GpuTensor,
    pub v_raw: &'a GpuTensor,
    /// Output gate projection `z = Wg x` `[N, P]`.
    pub z: &'a GpuTensor,
    /// `b = Wb x` `[N, H]`.
    pub b: &'a GpuTensor,
    /// `delta = Wrda_w x` `[N, H]`.
    pub delta: &'a GpuTensor,
    /// Decay `f = Wf_b(Wf_a x)` `[N, P]`.
    pub f: &'a GpuTensor,
}

/// Bounded scratch for up to `max_rows` rows.
pub struct RdaScratch {
    pub max_rows: usize,
    pub qconv: GpuTensor,
    pub kconv: GpuTensor,
    pub vconv: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub vprime: GpuTensor,
    pub log_alpha: GpuTensor,
    pub beta: GpuTensor,
    /// BF16-rounded recurrence output (`out = cast_model(y)`) `[N, P]`.
    pub y: GpuTensor,
}

impl RdaScratch {
    pub fn new(gpu: &mut Gpu, geom: &RdaGeometry, max_rows: usize) -> Result<Self, String> {
        let p = geom.proj();
        let plane = |gpu: &mut Gpu, cols: usize| {
            gpu.alloc_tensor(&[max_rows, cols], DType::F32)
                .map_err(|e| format!("rda scratch alloc: {e:?}"))
        };
        Ok(Self {
            max_rows,
            qconv: plane(gpu, p)?,
            kconv: plane(gpu, p)?,
            vconv: plane(gpu, p)?,
            q: plane(gpu, p)?,
            k: plane(gpu, p)?,
            vprime: plane(gpu, p)?,
            log_alpha: plane(gpu, p)?,
            beta: plane(gpu, geom.n_heads)?,
            y: plane(gpu, p)?,
        })
    }
}

/// Run one RDA layer over `n` rows. Writes `gated [N, P]` (input of the output
/// projection) and updates recurrent state + conv histories in place.
/// A zero-row call is a no-op (history/state unchanged).
#[allow(clippy::too_many_arguments)]
pub fn rda_forward_rows(
    gpu: &mut Gpu,
    geom: &RdaGeometry,
    layer: &RaonRdaLayer<'_>,
    st: &RdaLayerState<'_>,
    x: &RdaProjected<'_>,
    scratch: &RdaScratch,
    gated: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if n > scratch.max_rows {
        return Err(format!(
            "rda_forward_rows: {n} rows exceed scratch capacity {}",
            scratch.max_rows
        ));
    }
    let p = geom.proj();
    let h = geom.n_heads;
    let kd = geom.head_dim;
    let err = |what: &str, e: String| format!("rda_forward_rows {what}: {e}");

    // Three independent depthwise convs: raw -> SiLU -> BF16, history in place.
    for (input, weight, hist, out, what) in [
        (x.q_raw, layer.conv_w_q, st.hist_q, &scratch.qconv, "conv q"),
        (x.k_raw, layer.conv_w_k, st.hist_k, &scratch.kconv, "conv k"),
        (x.v_raw, layer.conv_w_v, st.hist_v, &scratch.vconv, "conv v"),
    ] {
        let g = CausalConvRowsGeom {
            n_rows: n,
            n_channels: p,
            ksize: geom.conv_kernel,
            in_row_stride: p,
            out_row_stride: p,
            w_ch_stride: geom.conv_kernel,
            w_tap_stride: 1,
            hist_tap_stride: p,
            apply_silu: true,
            round_bf16: geom.round_bf16,
        };
        gpu.causal_conv_depthwise_silu_rows_f32(out, input, weight, hist, hist, &g)
            .map_err(|e| err(what, format!("{e:?}")))?;
    }

    gpu.raon_rda_prepare_f32(&RdaPrepareLaunch {
        q_out: &scratch.q,
        k_out: &scratch.k,
        vprime_out: &scratch.vprime,
        log_alpha_out: &scratch.log_alpha,
        beta_out: &scratch.beta,
        qconv: &scratch.qconv,
        kconv: &scratch.kconv,
        vconv: &scratch.vconv,
        f: x.f,
        b: x.b,
        delta: x.delta,
        a_log: layer.a_log,
        dt_bias: layer.dt_bias,
        n_rows: n,
        n_heads: h,
        head_dim: kd,
        conv_row_stride: p,
        f_row_stride: p,
        bd_row_stride: h,
        out_row_stride: p,
        beta_row_stride: h,
        q_scale: (kd as f64).powf(-0.5) as f32,
        norm_eps: geom.norm_eps,
        lower_bound: geom.gate_lower_bound,
        round_bf16: geom.round_bf16,
    })
    .map_err(|e| err("prepare", format!("{e:?}")))?;

    let mut strides = GatedDeltaStrides::contiguous(h, 1, kd, kd, DecayAxis::PerKeyChannel);
    strides.state_slot = st.state_slot_stride;
    execute_gated_delta(
        gpu,
        &GatedDeltaOp {
            query: &scratch.q,
            key: &scratch.k,
            value: &scratch.vprime,
            log_decay: &scratch.log_alpha,
            beta: &scratch.beta,
            state: st.state,
            output: &scratch.y,
            slot_idx: None,
            n_dev: None,
            n_tokens: n,
            n_heads: h,
            qk_group: 1,
            key_dim: kd,
            value_dim: kd,
            decay_axis: DecayAxis::PerKeyChannel,
            round_output_bf16: geom.round_bf16,
            slot: st.slot,
            strides,
        },
    )
    .map_err(|e| err("recurrence", e.to_string()))?;

    gpu.gated_rmsnorm_sigmoid_rows_f32(
        gated,
        &scratch.y,
        x.z,
        layer.o_norm_gamma,
        &GatedRmsnormRowsGeom {
            n_rows: n,
            n_heads: h,
            head_dim: kd,
            x_row_stride: p,
            gate_row_stride: p,
            out_row_stride: p,
            eps: geom.rms_eps,
            round_bf16: geom.round_bf16,
        },
    )
    .map_err(|e| err("output gate", format!("{e:?}")))
}
