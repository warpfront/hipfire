// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Qwen3.5 embedding-encoder program (#666 G19).
//!
//! Each layer binds resident weights, the GDN state and one encode's scratch
//! to typed `hipfire_dispatch` steps; the step executors own every launch.
//! There is no decode: an encode is one bidirectional pass over a row block of
//! back-to-back sequences ([`EncoderSegment`]s).
//!
//! ```text
//! full attn  RmsnormAutomatic -> Encoder(Project q, gate, k, v) -> QkNorm(q, k) -> Encoder(Mrope)
//!            -> Encoder(Attention) -> Encoder(SigmoidGate) -> Encoder(Project o, +residual)
//! GDN        RmsnormAutomatic -> Encoder(Project qkv, z, a, b) -> Encoder(AlphaGate)
//!            -> Encoder(ConvSplit) -> Encoder(QkL2) -> Encoder(Scan) -> Encoder(GatedNorm)
//!            -> Encoder(Project out, +residual)
//! MLP        RmsnormAutomatic -> Encoder(Project gate, up) -> SiluMul -> Encoder(Project down, +residual)
//! head       RmsnormAutomatic -> Encoder(Project dense)
//! ```

use hipfire_dispatch::pipeline::encoder::{EncoderHeads, EncoderOp, EncoderSegment};
use hipfire_dispatch::pipeline::Step;
use hipfire_dispatch::types::RotationPlan;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::EmbedConfig;
use crate::state::GdnEncodeState;
use crate::weights::{EmbedWeights, LayerWeights, MixerWeights};

/// One encode's activations, sized for its `rows`; reused by every layer.
pub(crate) struct EncodeScratch {
    rows: usize,
    pub hidden: GpuTensor,
    pub positions: GpuTensor,
    normed: GpuTensor,
    proj_out: GpuTensor,
    q: GpuTensor,
    gate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    ctx: GpuTensor,
    qkv: GpuTensor,
    z: GpuTensor,
    alpha: GpuTensor,
    beta: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v_raw: GpuTensor,
    recurrent: GpuTensor,
    gdn_normed: GpuTensor,
    mlp_gate: GpuTensor,
    mlp_up: GpuTensor,
    pub dense: GpuTensor,
}

impl EncodeScratch {
    /// `hidden` is the uploaded `[rows, hidden]` embedding block, `positions`
    /// the `[rows][3]` i32 mRoPE positions; both are owned from here on.
    pub fn new(
        gpu: &mut Gpu,
        cfg: &EmbedConfig,
        rows: usize,
        hidden: GpuTensor,
        positions: GpuTensor,
    ) -> Result<Self, String> {
        let mut f = |n: usize| {
            gpu.alloc_tensor(&[rows * n], DType::F32)
                .map_err(|e| format!("qwen35-embed: alloc: {e:?}"))
        };
        let (h, qd, kvd) = (cfg.hidden_size, cfg.q_dim(), cfg.kv_dim());
        let (kd, vd, nv) = (
            cfg.gdn_key_dim(),
            cfg.gdn_value_dim(),
            cfg.linear_num_value_heads,
        );
        Ok(Self {
            rows,
            hidden,
            positions,
            normed: f(h)?,
            proj_out: f(h)?,
            q: f(qd)?,
            gate: f(qd)?,
            k: f(kvd)?,
            v: f(kvd)?,
            ctx: f(qd)?,
            qkv: f(2 * kd + vd)?,
            z: f(vd)?,
            alpha: f(nv)?,
            beta: f(nv)?,
            q_raw: f(kd)?,
            k_raw: f(kd)?,
            v_raw: f(vd)?,
            recurrent: f(vd)?,
            gdn_normed: f(vd)?,
            mlp_gate: f(cfg.intermediate_size)?,
            mlp_up: f(cfg.intermediate_size)?,
            dense: f(cfg.dense_out_features)?,
        })
    }

    /// Free everything but the residual stream and return it.
    pub fn into_hidden(mut self, gpu: &mut Gpu) -> Result<GpuTensor, String> {
        let empty = gpu
            .alloc_tensor(&[1], DType::F32)
            .map_err(|e| format!("qwen35-embed: alloc: {e:?}"))?;
        let hidden = std::mem::replace(&mut self.hidden, empty);
        self.free_gpu(gpu)?;
        Ok(hidden)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        for t in [
            self.hidden,
            self.positions,
            self.normed,
            self.proj_out,
            self.q,
            self.gate,
            self.k,
            self.v,
            self.ctx,
            self.qkv,
            self.z,
            self.alpha,
            self.beta,
            self.q_raw,
            self.k_raw,
            self.v_raw,
            self.recurrent,
            self.gdn_normed,
            self.mlp_gate,
            self.mlp_up,
            self.dense,
        ] {
            gpu.free_tensor(t)
                .map_err(|e| format!("qwen35-embed: free: {e:?}"))?;
        }
        Ok(())
    }
}

fn project<'a>(
    w: &'a GpuTensor,
    x: &'a GpuTensor,
    out: &'a GpuTensor,
    residual: Option<&'a GpuTensor>,
    rows: usize,
    out_dim: usize,
    k: usize,
) -> Step<'a> {
    Step::Encoder(EncoderOp::Project {
        w,
        x,
        out,
        residual,
        rows,
        out_dim,
        k,
    })
}

fn row_norm<'a>(
    x: &'a GpuTensor,
    w: &'a GpuTensor,
    out: &'a GpuTensor,
    rows: usize,
    cfg: &EmbedConfig,
) -> Step<'a> {
    Step::RmsnormAutomatic {
        x,
        norm_weight: w,
        x_plain: out,
        out,
        awq_scale: None,
        k: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
        rotation: RotationPlan::None,
        rows,
    }
}

/// Layer `layer`'s mixer: pre-norm through the residual add. `delta` is the
/// GDN layer index (its state slot) for a GDN layer.
pub(crate) fn mixer_steps<'a>(
    cfg: &EmbedConfig,
    layer: &'a LayerWeights,
    delta: usize,
    state: &'a GdnEncodeState,
    s: &'a EncodeScratch,
    segments: &'a [EncoderSegment],
) -> Vec<Step<'a>> {
    let (n, h, eps) = (s.rows, cfg.hidden_size, cfg.rms_norm_eps);
    let mut steps = vec![row_norm(&s.hidden, &layer.input_ln, &s.normed, n, cfg)];
    match &layer.mixer {
        MixerWeights::FullAttn {
            q_proj,
            gate_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        } => {
            let (qd, kvd) = (cfg.q_dim(), cfg.kv_dim());
            let heads = EncoderHeads {
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
            };
            steps.extend([
                project(q_proj, &s.normed, &s.q, None, n, qd, h),
                project(gate_proj, &s.normed, &s.gate, None, n, qd, h),
                project(k_proj, &s.normed, &s.k, None, n, kvd, h),
                project(v_proj, &s.normed, &s.v, None, n, kvd, h),
                Step::QkNorm {
                    x: &s.q,
                    weight: q_norm,
                    n_groups: n * cfg.n_heads,
                    head_dim: cfg.head_dim,
                    eps,
                },
                Step::QkNorm {
                    x: &s.k,
                    weight: k_norm,
                    n_groups: n * cfg.n_kv_heads,
                    head_dim: cfg.head_dim,
                    eps,
                },
                Step::Encoder(EncoderOp::Mrope {
                    q: &s.q,
                    k: &s.k,
                    positions: &s.positions,
                    heads,
                    n_rot: cfg.n_rot(),
                    theta: cfg.rope_theta,
                    section: cfg.mrope_section,
                    rows: n,
                }),
                Step::Encoder(EncoderOp::Attention {
                    q: &s.q,
                    k: &s.k,
                    v: &s.v,
                    ctx: &s.ctx,
                    segments,
                    heads,
                }),
                Step::Encoder(EncoderOp::SigmoidGate {
                    out: &s.ctx,
                    gate: &s.gate,
                }),
                project(o_proj, &s.ctx, &s.proj_out, Some(&s.hidden), n, h, qd),
            ]);
        }
        MixerWeights::Gdn {
            qkv,
            z,
            a,
            b,
            a_log,
            dt_bias,
            conv,
            gdn_norm,
            out_proj,
        } => {
            let (kd, vd) = (cfg.gdn_key_dim(), cfg.gdn_value_dim());
            let nv = cfg.linear_num_value_heads;
            let vhd = cfg.linear_value_head_dim;
            steps.extend([
                project(qkv, &s.normed, &s.qkv, None, n, 2 * kd + vd, h),
                project(z, &s.normed, &s.z, None, n, vd, h),
                project(a, &s.normed, &s.alpha, None, n, nv, h),
                project(b, &s.normed, &s.beta, None, n, nv, h),
                Step::Encoder(EncoderOp::AlphaGate {
                    beta: &s.beta,
                    alpha: &s.alpha,
                    dt_bias,
                    a_log,
                    heads: nv,
                    rows: n,
                }),
                Step::Encoder(EncoderOp::ConvSplit {
                    qkv: &s.qkv,
                    weight: conv,
                    ring: &state.conv_states[delta],
                    q: &s.q_raw,
                    k: &s.k_raw,
                    v: &s.v_raw,
                    key_dim: kd,
                    value_dim: vd,
                    segments,
                }),
                Step::Encoder(EncoderOp::QkL2 {
                    q: &s.q_raw,
                    k: &s.k_raw,
                    heads: cfg.linear_num_key_heads,
                    head_dim: cfg.linear_key_head_dim,
                    eps,
                    rows: n,
                }),
                Step::Encoder(EncoderOp::Scan {
                    q: &s.q_raw,
                    k: &s.k_raw,
                    v: &s.v_raw,
                    gate: &s.alpha,
                    beta: &s.beta,
                    state: &state.s_matrices[delta],
                    out: &s.recurrent,
                    key_dim: kd,
                    heads: nv,
                    head_dim: vhd,
                    segments,
                }),
                Step::Encoder(EncoderOp::GatedNorm {
                    x: &s.recurrent,
                    z: &s.z,
                    w: gdn_norm,
                    out: &s.gdn_normed,
                    heads: nv,
                    head_dim: vhd,
                    eps,
                    rows: n,
                }),
                project(
                    out_proj,
                    &s.gdn_normed,
                    &s.proj_out,
                    Some(&s.hidden),
                    n,
                    h,
                    vd,
                ),
            ]);
        }
    }
    steps
}

/// The SwiGLU MLP after every mixer.
pub(crate) fn mlp_steps<'a>(
    cfg: &EmbedConfig,
    layer: &'a LayerWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 5] {
    let (n, h, inter) = (s.rows, cfg.hidden_size, cfg.intermediate_size);
    [
        row_norm(&s.hidden, &layer.post_ln, &s.normed, n, cfg),
        project(&layer.gate_proj, &s.normed, &s.mlp_gate, None, n, inter, h),
        project(&layer.up_proj, &s.normed, &s.mlp_up, None, n, inter, h),
        Step::SiluMul {
            gate: &s.mlp_gate,
            up: &s.mlp_up,
            out: &s.mlp_gate,
        },
        project(
            &layer.down_proj,
            &s.mlp_gate,
            &s.proj_out,
            Some(&s.hidden),
            n,
            h,
            inter,
        ),
    ]
}

/// Final norm and the 128-d Dense head into `s.dense`.
pub(crate) fn head_steps<'a>(
    cfg: &EmbedConfig,
    w: &'a EmbedWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 2] {
    let n = s.rows;
    [
        row_norm(&s.hidden, &w.final_norm, &s.normed, n, cfg),
        project(
            &w.dense,
            &s.normed,
            &s.dense,
            None,
            n,
            cfg.dense_out_features,
            cfg.hidden_size,
        ),
    ]
}
