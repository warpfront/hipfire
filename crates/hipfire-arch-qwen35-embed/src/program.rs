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
//! full attn  RmsnormAutomatic -> Project(q, gate, k, v) -> QkNorm(q, k) -> Rope(Mrope)
//!            -> Encoder(Attention) -> Encoder(SigmoidGate) -> Project(o, +residual)
//! GDN        RmsnormAutomatic -> Project(qkv, z, a, b) -> Encoder(AlphaGate)
//!            -> Encoder(ConvSplit) -> Encoder(QkL2) -> Encoder(Scan) -> Encoder(GatedNorm)
//!            -> Project(out, +residual)
//! MLP        RmsnormAutomatic -> Project(gate, up) -> SiluMul -> Project(down, +residual)
//! head       RmsnormAutomatic -> Project(dense)
//! ```
//!
//! Every projection runs on `ProjectLadder::EncoderF16` (F16 weight, F32
//! activation) with the shared F16 activation copy in [`EncodeScratch`].

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::encoder::{EncoderHeads, EncoderOp, EncoderSegment};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind};
use hipfire_dispatch::pipeline::{ProjectLadder, ProjectOp, Step};
use hipfire_dispatch::types::RotationPlan;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::EmbedConfig;
use crate::state::GdnEncodeState;
use crate::weights::{EmbedWeights, LayerWeights, MixerWeights};

/// Carve offsets round up to 64 floats (256 B), so odd row counts keep every
/// role's base vector-aligned.
const CARVE_ALIGN: usize = 64;
/// Gap after every carve: 4 KiB + 256 B. Padded row counts are multiples of
/// 64, so without it every role base shares its low address bits with the
/// others (1-2% slower at 1-2K tokens on gfx1151; power-of-two gaps do not help).
const CARVE_STAGGER: usize = 1088;

/// One encode's activations, sized for its `rows`; reused by every layer.
///
/// The attention, GDN and MLP roles are never live at once, so all three
/// sets are views into one `arena` (its width is the widest set): the live
/// set stays at one sublayer's, as when every op allocated its own output.
pub(crate) struct EncodeScratch {
    rows: usize,
    pub hidden: GpuTensor,
    pub positions: GpuTensor,
    normed: GpuTensor,
    proj_out: GpuTensor,
    /// F16 activation copy of the LDS projections, `rows * max k`.
    x16: GpuTensor,
    pub dense: GpuTensor,
    arena: GpuTensor,
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
}

/// Views of `widths` (floats per row) laid back to back in `arena`.
fn carve<const N: usize>(arena: &GpuTensor, rows: usize, widths: [usize; N]) -> [GpuTensor; N] {
    let mut off = 0;
    widths.map(|w| {
        let t = arena.sub_offset(off, w * rows);
        off += (w * rows).next_multiple_of(CARVE_ALIGN) + CARVE_STAGGER;
        t
    })
}

fn carved_len(rows: usize, widths: &[usize]) -> usize {
    widths
        .iter()
        .map(|w| (w * rows).next_multiple_of(CARVE_ALIGN) + CARVE_STAGGER)
        .sum()
}

impl EncodeScratch {
    /// `hidden` is the uploaded `[rows, hidden]` embedding block, `positions`
    /// the `[rows][3]` i32 mRoPE positions; both are owned from here on (and
    /// freed on error).
    pub fn new(
        gpu: &mut Gpu,
        cfg: &EmbedConfig,
        rows: usize,
        hidden: GpuTensor,
        positions: GpuTensor,
    ) -> Result<Self, String> {
        let (h, qd, kvd) = (cfg.hidden_size, cfg.q_dim(), cfg.kv_dim());
        let (kd, vd, nv) = (
            cfg.gdn_key_dim(),
            cfg.gdn_value_dim(),
            cfg.linear_num_value_heads,
        );
        let inter = cfg.intermediate_size;
        let attn = [qd, qd, kvd, kvd, qd];
        let gdn = [2 * kd + vd, vd, nv, nv, kd, kd, vd, vd, vd];
        let mlp = [inter, inter];
        let arena_len = carved_len(rows, &attn)
            .max(carved_len(rows, &gdn))
            .max(carved_len(rows, &mlp));
        let k_max = h.max(qd).max(vd).max(inter);
        let sizes = [
            (rows * h, DType::F32),
            (rows * h, DType::F32),
            (rows * k_max, DType::F16),
            (rows * cfg.dense_out_features, DType::F32),
            (arena_len, DType::F32),
        ];
        let mut owned = vec![hidden, positions];
        for (n, dtype) in sizes {
            match gpu.alloc_tensor(&[n], dtype) {
                Ok(t) => owned.push(t),
                Err(e) => {
                    for t in owned {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(format!("qwen35-embed: alloc: {e:?}"));
                }
            }
        }
        let Ok([hidden, positions, normed, proj_out, x16, dense, arena]) =
            <[GpuTensor; 7]>::try_from(owned)
        else {
            unreachable!("two owned tensors plus five allocations");
        };
        let [q, gate, k, v, ctx] = carve(&arena, rows, attn);
        let [qkv, z, alpha, beta, q_raw, k_raw, v_raw, recurrent, gdn_normed] =
            carve(&arena, rows, gdn);
        let [mlp_gate, mlp_up] = carve(&arena, rows, mlp);
        Ok(Self {
            rows,
            hidden,
            positions,
            normed,
            proj_out,
            x16,
            dense,
            arena,
            q,
            gate,
            k,
            v,
            ctx,
            qkv,
            z,
            alpha,
            beta,
            q_raw,
            k_raw,
            v_raw,
            recurrent,
            gdn_normed,
            mlp_gate,
            mlp_up,
        })
    }

    /// The GDN intermediates the `HIPFIRE_EMBED_DEBUG` trace reports after a
    /// GDN mixer: `(label, tensor)`.
    pub fn gdn_trace_planes(&self) -> [(&'static str, &GpuTensor); 4] {
        [
            ("a_t(after gate)", &self.alpha),
            ("qkv_t", &self.qkv),
            ("scan out", &self.recurrent),
            ("gated_norm", &self.gdn_normed),
        ]
    }

    /// Free everything but the residual stream and return it.
    pub fn into_hidden(self, gpu: &mut Gpu) -> Result<GpuTensor, String> {
        let hidden = self.hidden;
        free_all(
            gpu,
            [
                self.positions,
                self.normed,
                self.proj_out,
                self.x16,
                self.dense,
                self.arena,
            ],
        )?;
        Ok(hidden)
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        free_all(
            gpu,
            [
                self.hidden,
                self.positions,
                self.normed,
                self.proj_out,
                self.x16,
                self.dense,
                self.arena,
            ],
        )
    }
}

/// Free every tensor; the first failure is reported after the rest are freed.
fn free_all<const N: usize>(gpu: &mut Gpu, tensors: [GpuTensor; N]) -> Result<(), String> {
    let mut first = Ok(());
    for t in tensors {
        if let (Err(e), Ok(())) = (gpu.free_tensor(t), &first) {
            first = Err(format!("qwen35-embed: free: {e:?}"));
        }
    }
    first
}

#[allow(clippy::too_many_arguments)]
fn project<'a>(
    w: &'a GpuTensor,
    x: &'a GpuTensor,
    out: &'a GpuTensor,
    residual: Option<&'a GpuTensor>,
    rows: usize,
    out_dim: usize,
    k: usize,
    x16: &'a GpuTensor,
) -> Step<'a> {
    Step::Project(ProjectOp {
        weight: WeightRef {
            buf: w,
            dtype: w.dtype,
            m: out_dim,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        },
        input: x,
        output: out,
        rows,
        rotation: Some(x16),
        ladder: ProjectLadder::EncoderF16,
        residual,
        grouped: None,
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
    let x16 = &s.x16;
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
                project(q_proj, &s.normed, &s.q, None, n, qd, h, x16),
                project(gate_proj, &s.normed, &s.gate, None, n, qd, h, x16),
                project(k_proj, &s.normed, &s.k, None, n, kvd, h, x16),
                project(v_proj, &s.normed, &s.v, None, n, kvd, h, x16),
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
                Step::Rope {
                    q: &s.q,
                    k: &s.k,
                    pos_buf: &s.positions.buf,
                    positions: Some(&s.positions),
                    n_heads: cfg.n_heads,
                    n_kv_heads: cfg.n_kv_heads,
                    head_dim: cfg.head_dim,
                    rope: Rope {
                        kind: RopeKind::Mrope {
                            rot_dim: cfg.n_rot(),
                            section: cfg.mrope_section,
                        },
                        theta: cfg.rope_theta,
                    },
                },
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
                project(o_proj, &s.ctx, &s.proj_out, Some(&s.hidden), n, h, qd, x16),
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
                project(qkv, &s.normed, &s.qkv, None, n, 2 * kd + vd, h, x16),
                project(z, &s.normed, &s.z, None, n, vd, h, x16),
                project(a, &s.normed, &s.alpha, None, n, nv, h, x16),
                project(b, &s.normed, &s.beta, None, n, nv, h, x16),
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
                    x16,
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
    let (n, h, inter, x16) = (s.rows, cfg.hidden_size, cfg.intermediate_size, &s.x16);
    [
        row_norm(&s.hidden, &layer.post_ln, &s.normed, n, cfg),
        project(
            &layer.gate_proj,
            &s.normed,
            &s.mlp_gate,
            None,
            n,
            inter,
            h,
            x16,
        ),
        project(&layer.up_proj, &s.normed, &s.mlp_up, None, n, inter, h, x16),
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
            x16,
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
            &s.x16,
        ),
    ]
}
