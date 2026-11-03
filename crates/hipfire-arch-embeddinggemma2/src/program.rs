// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative EmbeddingGemma 2 encoder-trunk program (#666 G20).
//!
//! Each layer binds resident weights and one encode's scratch to typed
//! `hipfire_dispatch` steps; the step executors own every launch. There is no
//! decode: an encode is one bidirectional pass over a row block of
//! back-to-back sequences ([`EncoderSegment`]s).
//!
//! ```text
//! PLE        Project(ple_projection) -> Scale(hidden^-0.5) -> RmsnormAutomatic (per slice)
//! attention  RmsnormAutomatic -> Project(q, k, v) -> QkNorm(q, k, v) -> Rope (half-split)
//!            -> Encoder(Attention, scale 1.0, sliding band) -> Project(o)
//!            -> Encoder(PostNormResidual)
//! MLP        RmsnormAutomatic -> Project(gate, up) -> Encoder(GeluMul) -> Project(down)
//!            -> Encoder(PostNormResidual)
//! PLE block  Project(ple_gate) -> Encoder(PerLayerGate) -> Project(ple_proj)
//!            -> Encoder(PostNormResidual) -> Scale(layer_scalar)
//! head       RmsnormAutomatic -> Project(embedding_projection)
//! ```
//!
//! Every projection runs on `ProjectLadder::EncoderF16` (F16 weight, F32
//! activation) with the shared F16 activation copy in [`EncodeScratch`].

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::encoder::{AttnScale, EncoderHeads, EncoderOp, EncoderSegment};
use hipfire_dispatch::pipeline::sandwich::{Rope, RopeKind, ScaleOp};
use hipfire_dispatch::pipeline::{ProjectLadder, ProjectOp, Step};
use hipfire_dispatch::types::RotationPlan;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::config::{EmbedConfig, LayerType};
use crate::weights::{EmbedWeights, LayerWeights};

/// One encode's activations, sized for its `rows`; reused by every layer.
///
/// The attention (`q`/`k`/`v`/`ctx`), MLP (`mlp_gate`/`mlp_up`), PLE
/// (`ple_act`) and head (`embedding`) temporaries are never live at the same
/// time, so they are views into one `arena`: the trunk then touches about
/// the memory the per-op allocations of the hand-written forward did, which
/// keeps short encodes inside gfx1151's MALL. Q/K/V/ctx hold the widest
/// layer's geometry (head_dim varies per layer).
pub(crate) struct EncodeScratch {
    rows: usize,
    pub hidden: GpuTensor,
    /// One I32 position per row (bits in an F32-typed buffer).
    positions: GpuTensor,
    /// `[rows, layers * ple]`: every layer's normed per-layer input.
    ple_all: GpuTensor,
    normed: GpuTensor,
    proj_out: GpuTensor,
    /// F16 activation copy of the LDS projections, `rows * max k`.
    x16: GpuTensor,
    arena: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    ctx: GpuTensor,
    mlp_gate: GpuTensor,
    mlp_up: GpuTensor,
    ple_act: GpuTensor,
    pub embedding: GpuTensor,
}

impl EncodeScratch {
    /// `hidden` is the uploaded `[rows, hidden]` embedding block, `positions`
    /// the per-row I32 positions; both are owned from here on (and freed on
    /// error).
    pub fn new(
        gpu: &mut Gpu,
        cfg: &EmbedConfig,
        rows: usize,
        hidden: GpuTensor,
        positions: GpuTensor,
    ) -> Result<Self, String> {
        let layers = 0..cfg.num_layers;
        let qd = layers.clone().map(|l| cfg.q_dim(l)).max().unwrap_or(0);
        let kvd = layers.map(|l| cfg.kv_dim(l)).max().unwrap_or(0);
        let (h, inter, ple) = (cfg.hidden_size, cfg.intermediate_size, cfg.ple_dim);
        let width = (2 * qd + 2 * kvd)
            .max(2 * inter)
            .max(h)
            .max(cfg.embedding_dim);
        let k_max = h.max(qd).max(inter).max(ple);
        let sizes = [
            (rows * cfg.num_layers * ple, DType::F32),
            (rows * h, DType::F32),
            (rows * h, DType::F32),
            (rows * k_max, DType::F16),
            (rows * width, DType::F32),
        ];
        let mut owned = vec![hidden, positions];
        for (n, dtype) in sizes {
            match gpu.alloc_tensor(&[n], dtype) {
                Ok(t) => owned.push(t),
                Err(e) => {
                    for t in owned {
                        let _ = gpu.free_tensor(t);
                    }
                    return Err(format!("embeddinggemma2: alloc: {e:?}"));
                }
            }
        }
        let Ok([hidden, positions, ple_all, normed, proj_out, x16, arena]) =
            <[GpuTensor; 7]>::try_from(owned)
        else {
            unreachable!("two owned tensors plus five allocations");
        };
        // Views: (offset, width) in units of `rows` elements.
        let view = |off: usize, w: usize| arena.sub_offset(rows * off, rows * w);
        Ok(Self {
            rows,
            q: view(0, qd),
            k: view(qd, kvd),
            v: view(qd + kvd, kvd),
            ctx: view(qd + 2 * kvd, qd),
            mlp_gate: view(0, inter),
            mlp_up: view(inter, inter),
            ple_act: view(0, h),
            embedding: view(0, cfg.embedding_dim),
            hidden,
            positions,
            ple_all,
            normed,
            proj_out,
            x16,
            arena,
        })
    }

    /// Free everything but the residual stream and return it.
    pub fn into_hidden(self, gpu: &mut Gpu) -> Result<GpuTensor, String> {
        let hidden = self.hidden;
        free_all(
            gpu,
            [
                self.positions,
                self.ple_all,
                self.normed,
                self.proj_out,
                self.x16,
                self.arena,
            ],
        )?;
        Ok(hidden)
    }

    /// Frees the owned buffers; the views die with `arena`.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        free_all(
            gpu,
            [
                self.hidden,
                self.positions,
                self.ple_all,
                self.normed,
                self.proj_out,
                self.x16,
                self.arena,
            ],
        )
    }
}

fn free_all<const N: usize>(gpu: &mut Gpu, tensors: [GpuTensor; N]) -> Result<(), String> {
    for t in tensors {
        gpu.free_tensor(t)
            .map_err(|e| format!("embeddinggemma2: free: {e:?}"))?;
    }
    Ok(())
}

/// `out = x · wᵀ` on the F16 encoder ladder.
fn project<'a>(
    s: &'a EncodeScratch,
    w: &'a GpuTensor,
    x: &'a GpuTensor,
    out: &'a GpuTensor,
    out_dim: usize,
    k: usize,
) -> Step<'a> {
    Step::Project(ProjectOp {
        weight: WeightRef {
            buf: w,
            dtype: DType::F16,
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
        rows: s.rows,
        rotation: Some(&s.x16),
        ladder: ProjectLadder::EncoderF16,
        residual: None,
        grouped: None,
    })
}

/// `hidden += rmsnorm(proj_out)·w`: the sandwich norm on a branch output.
fn post_norm_residual<'a>(cfg: &EmbedConfig, s: &'a EncodeScratch, w: &'a GpuTensor) -> Step<'a> {
    Step::Encoder(EncoderOp::PostNormResidual {
        x: &s.proj_out,
        w,
        residual: &s.hidden,
        rows: s.rows,
        width: cfg.hidden_size,
        eps: cfg.rms_norm_eps,
    })
}

fn row_norm<'a>(
    x: &'a GpuTensor,
    w: &'a GpuTensor,
    out: &'a GpuTensor,
    rows: usize,
    k: usize,
    eps: f32,
) -> Step<'a> {
    Step::RmsnormAutomatic {
        x,
        norm_weight: w,
        x_plain: out,
        out,
        awq_scale: None,
        k,
        eps,
        rotation: RotationPlan::None,
        rows,
    }
}

fn head_norm<'a>(
    x: &'a GpuTensor,
    weight: &'a GpuTensor,
    n_groups: usize,
    head_dim: usize,
    eps: f32,
) -> Step<'a> {
    Step::QkNorm {
        x,
        weight,
        n_groups,
        head_dim,
        eps,
    }
}

/// Once per encode: `ple_all = rmsnorm((ple_projection · x) · hidden^-0.5)`
/// over each `[ple]` slice of every row.
pub(crate) fn ple_steps<'a>(
    cfg: &EmbedConfig,
    w: &'a EmbedWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 3] {
    let (n, h, ple, layers) = (s.rows, cfg.hidden_size, cfg.ple_dim, cfg.num_layers);
    [
        project(s, &w.ple_projection, &s.hidden, &s.ple_all, layers * ple, h),
        Step::Scale(ScaleOp {
            x: &s.ple_all,
            factor: cfg.ple_scale(),
        }),
        row_norm(
            &s.ple_all,
            &w.ple_norm,
            &s.ple_all,
            n * layers,
            ple,
            cfg.rms_norm_eps,
        ),
    ]
}

/// Layer `li`'s attention sublayer: pre-norm through the post-normed residual add.
pub(crate) fn attention_steps<'a>(
    cfg: &EmbedConfig,
    li: usize,
    layer: &'a LayerWeights,
    s: &'a EncodeScratch,
    segments: &'a [EncoderSegment],
) -> [Step<'a>; 11] {
    let (n, h, eps) = (s.rows, cfg.hidden_size, cfg.rms_norm_eps);
    let attn = cfg.attn(li);
    let heads = EncoderHeads {
        n_heads: cfg.n_heads,
        n_kv_heads: attn.n_kv_heads,
        head_dim: attn.head_dim,
    };
    let (qd, kvd, hd) = (cfg.q_dim(li), cfg.kv_dim(li), attn.head_dim);
    [
        row_norm(&s.hidden, &layer.input_ln, &s.normed, n, h, eps),
        project(s, &layer.q_proj, &s.normed, &s.q, qd, h),
        project(s, &layer.k_proj, &s.normed, &s.k, kvd, h),
        project(s, &layer.v_proj, &s.normed, &s.v, kvd, h),
        // Learned per-head Q/K norms and the scale-less V norm.
        head_norm(&s.q, &layer.q_norm, n * cfg.n_heads, hd, eps),
        head_norm(&s.k, &layer.k_norm, n * attn.n_kv_heads, hd, eps),
        head_norm(&s.v, &layer.v_ones, n * attn.n_kv_heads, hd, eps),
        Step::Rope {
            q: &s.q,
            k: &s.k,
            pos_buf: &s.positions.buf,
            positions: Some(&s.positions),
            n_heads: cfg.n_heads,
            n_kv_heads: attn.n_kv_heads,
            head_dim: hd,
            rope: Rope {
                kind: RopeKind::PartialHalved { rot_pairs: hd / 2 },
                theta: cfg.rope_theta(li),
            },
        },
        // The QK norms replace 1/sqrt(head_dim): this family scales by 1.0.
        Step::Encoder(EncoderOp::Attention {
            q: &s.q,
            k: &s.k,
            v: &s.v,
            ctx: &s.ctx,
            segments,
            heads,
            scale: AttnScale::Fixed(1.0),
            window: (cfg.layer_types[li] == LayerType::Sliding).then_some(cfg.sliding_window),
        }),
        project(s, &layer.o_proj, &s.ctx, &s.proj_out, h, qd),
        post_norm_residual(cfg, s, &layer.post_attn_ln),
    ]
}

/// The gated gelu-tanh MLP after every attention sublayer.
pub(crate) fn mlp_steps<'a>(
    cfg: &EmbedConfig,
    layer: &'a LayerWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 6] {
    let (n, h, inter) = (s.rows, cfg.hidden_size, cfg.intermediate_size);
    [
        row_norm(
            &s.hidden,
            &layer.pre_ff_ln,
            &s.normed,
            n,
            h,
            cfg.rms_norm_eps,
        ),
        project(s, &layer.gate_proj, &s.normed, &s.mlp_gate, inter, h),
        project(s, &layer.up_proj, &s.normed, &s.mlp_up, inter, h),
        Step::Encoder(EncoderOp::GeluMul {
            gate: &s.mlp_gate,
            up: &s.mlp_up,
            len: n * inter,
        }),
        project(s, &layer.down_proj, &s.mlp_gate, &s.proj_out, h, inter),
        post_norm_residual(cfg, s, &layer.post_ff_ln),
    ]
}

/// Layer `li`'s per-layer-input block, then `× layer_scalar` on the stream.
pub(crate) fn ple_block_steps<'a>(
    cfg: &EmbedConfig,
    li: usize,
    layer: &'a LayerWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 5] {
    let (n, h, ple) = (s.rows, cfg.hidden_size, cfg.ple_dim);
    [
        project(s, &layer.ple_gate, &s.hidden, &s.ple_act, h, h),
        Step::Encoder(EncoderOp::PerLayerGate {
            gate: &s.ple_act,
            inputs: &s.ple_all,
            rows: n,
            width: ple,
            n_layers: cfg.num_layers,
            layer: li,
        }),
        project(s, &layer.ple_proj, &s.ple_act, &s.proj_out, h, ple),
        post_norm_residual(cfg, s, &layer.ple_post_norm),
        Step::Scale(ScaleOp {
            x: &s.hidden,
            factor: layer.layer_scalar,
        }),
    ]
}

/// Final norm and the Matryoshka embedding projection into `s.embedding`.
pub(crate) fn head_steps<'a>(
    cfg: &EmbedConfig,
    w: &'a EmbedWeights,
    s: &'a EncodeScratch,
) -> [Step<'a>; 2] {
    let (n, h) = (s.rows, cfg.hidden_size);
    [
        row_norm(&s.hidden, &w.final_norm, &s.normed, n, h, cfg.rms_norm_eps),
        project(
            s,
            &w.embedding_projection,
            &s.normed,
            &s.embedding,
            cfg.embedding_dim,
            h,
        ),
    ]
}
