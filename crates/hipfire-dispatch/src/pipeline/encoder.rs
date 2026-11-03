// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `EncoderOp` / `execute_encoder`: the stages of a bidirectional hybrid
//! encoder trunk over a row block of independent, back-to-back sequences
//! (Qwen3.5 multi-vector embedding, #666 G19). F16 weights, F32 activations.
//!
//! ```text
//! full attn  [RmsnormAutomatic] -> Project(q, gate, k, v) -> QkNorm(q, k) -> Mrope
//!            -> Attention (bidirectional, per segment) -> SigmoidGate -> Project(o, +residual)
//! GDN        [RmsnormAutomatic] -> Project(qkv, z, a, b) -> AlphaGate -> ConvSplit (per segment)
//!            -> QkL2 -> Scan (per segment) -> GatedNorm -> Project(out, +residual)
//! ```
//!
//! Row-independent stages run once over the whole block. The two stateful
//! ones run per [`EncoderSegment`] with their state zeroed at each segment's
//! start, and attention never reads keys outside its own segment, so a block
//! of sequences is equivalent to encoding each alone. Stages borrow every
//! operand; the only allocations are each stage's own transient scratch.

use rdna_compute::gemm::{LdsTile, LdsTileSplitK};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::types::DispatchError;

fn hip(e: impl std::fmt::Debug) -> DispatchError {
    DispatchError::Hip(format!("{e:?}"))
}

/// One sequence's row block inside the concatenated batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderSegment {
    /// First row of this segment in the `[T, hidden]` row block.
    pub row_offset: usize,
    /// Rows it occupies, its own padding included.
    pub rows: usize,
    /// Real token count: the attention key bound (padding keys are masked).
    pub n_valid: usize,
}

/// Full-attention head geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderHeads {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl EncoderHeads {
    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

pub enum EncoderOp<'a> {
    /// `out[rows, out_dim] = x[rows, k] · w_f16[out_dim, k]ᵀ`, then
    /// `residual += out` when given. No bias.
    Project {
        w: &'a GpuTensor,
        x: &'a GpuTensor,
        out: &'a GpuTensor,
        residual: Option<&'a GpuTensor>,
        rows: usize,
        out_dim: usize,
        k: usize,
    },
    /// Interleaved (t, h, w) mRoPE, rotate-half over the first `n_rot`
    /// channels of each head; `positions` is `[rows][3]` i32.
    Mrope {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        positions: &'a GpuTensor,
        heads: EncoderHeads,
        n_rot: usize,
        theta: f32,
        section: [usize; 3],
        rows: usize,
    },
    /// Bidirectional attention within each segment. Scales `q` in place on
    /// the eager-GEMM route.
    Attention {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        v: &'a GpuTensor,
        ctx: &'a GpuTensor,
        segments: &'a [EncoderSegment],
        heads: EncoderHeads,
    },
    /// `out *= σ(gate)`.
    SigmoidGate {
        out: &'a GpuTensor,
        gate: &'a GpuTensor,
    },
    /// In place: `beta = σ(beta)`, `alpha = −exp(a_log)·softplus(alpha + dt_bias)`.
    AlphaGate {
        beta: &'a GpuTensor,
        alpha: &'a GpuTensor,
        dt_bias: &'a GpuTensor,
        a_log: &'a GpuTensor,
        heads: usize,
        rows: usize,
    },
    /// Causal conv + SiLU split of `qkv` into Q|K|V, per segment from a zeroed ring.
    ConvSplit {
        qkv: &'a GpuTensor,
        weight: &'a GpuTensor,
        ring: &'a GpuTensor,
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        v: &'a GpuTensor,
        key_dim: usize,
        value_dim: usize,
        segments: &'a [EncoderSegment],
    },
    /// Per-head L2 norm of Q (times `1/sqrt(head_dim)`) and K, in place.
    QkL2 {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        heads: usize,
        head_dim: usize,
        eps: f32,
        rows: usize,
    },
    /// FP32 delta-rule scan, per segment from a zeroed state. Key heads equal
    /// value heads.
    Scan {
        q: &'a GpuTensor,
        k: &'a GpuTensor,
        v: &'a GpuTensor,
        gate: &'a GpuTensor,
        beta: &'a GpuTensor,
        state: &'a GpuTensor,
        out: &'a GpuTensor,
        key_dim: usize,
        heads: usize,
        head_dim: usize,
        segments: &'a [EncoderSegment],
    },
    /// `out = rmsnorm(x)·w·silu(z)` per value head.
    GatedNorm {
        x: &'a GpuTensor,
        z: &'a GpuTensor,
        w: &'a GpuTensor,
        out: &'a GpuTensor,
        heads: usize,
        head_dim: usize,
        eps: f32,
        rows: usize,
    },
}

fn fits(t: &GpuTensor, elems: usize, what: &str) -> Result<(), DispatchError> {
    if t.numel() < elems {
        return Err(DispatchError::Hip(format!(
            "encoder: {what} holds {} elements, needs {elems}",
            t.numel()
        )));
    }
    Ok(())
}

fn segments_rows(segments: &[EncoderSegment]) -> usize {
    segments.last().map_or(0, |s| s.row_offset + s.rows)
}

impl EncoderOp<'_> {
    /// Geometry and operand sizes; launch-free.
    pub fn validate_for_gpu(&self, _gpu: &Gpu) -> Result<(), DispatchError> {
        match self {
            EncoderOp::Project {
                w,
                x,
                out,
                residual,
                rows,
                out_dim,
                k,
            } => {
                if *k == 0 || k % 16 != 0 {
                    return Err(DispatchError::Hip(format!(
                        "encoder: GEMM k={k} is not a multiple of 16"
                    )));
                }
                fits(w, out_dim * k, "projection weight")?;
                fits(x, rows * k, "projection input")?;
                fits(out, rows * out_dim, "projection output")?;
                if let Some(r) = residual {
                    fits(r, rows * out_dim, "projection residual")?;
                }
            }
            EncoderOp::Mrope {
                q,
                k,
                positions,
                heads,
                n_rot,
                rows,
                ..
            } => {
                if *n_rot > heads.head_dim {
                    return Err(DispatchError::Hip("encoder: mrope n_rot > head_dim".into()));
                }
                fits(q, rows * heads.q_dim(), "mrope q")?;
                fits(k, rows * heads.kv_dim(), "mrope k")?;
                fits(positions, rows * 3, "mrope positions")?;
            }
            EncoderOp::Attention {
                q,
                k,
                v,
                ctx,
                segments,
                heads,
            } => {
                if heads.n_kv_heads == 0 || heads.n_heads % heads.n_kv_heads != 0 {
                    return Err(DispatchError::Hip(
                        "encoder: heads not a GQA multiple".into(),
                    ));
                }
                check_segments(segments)?;
                let n = segments_rows(segments);
                fits(q, n * heads.q_dim(), "attention q")?;
                fits(k, n * heads.kv_dim(), "attention k")?;
                fits(v, n * heads.kv_dim(), "attention v")?;
                fits(ctx, n * heads.q_dim(), "attention ctx")?;
            }
            EncoderOp::SigmoidGate { out, gate } => fits(gate, out.numel(), "sigmoid gate")?,
            EncoderOp::AlphaGate {
                beta,
                alpha,
                dt_bias,
                a_log,
                heads,
                rows,
            } => {
                fits(beta, rows * heads, "beta")?;
                fits(alpha, rows * heads, "alpha")?;
                fits(dt_bias, *heads, "dt_bias")?;
                fits(a_log, *heads, "a_log")?;
            }
            EncoderOp::ConvSplit {
                qkv,
                q,
                k,
                v,
                key_dim,
                value_dim,
                segments,
                ..
            } => {
                check_segments(segments)?;
                let n = segments_rows(segments);
                fits(qkv, n * (2 * key_dim + value_dim), "conv input")?;
                fits(q, n * key_dim, "conv q")?;
                fits(k, n * key_dim, "conv k")?;
                fits(v, n * value_dim, "conv v")?;
            }
            EncoderOp::QkL2 {
                q,
                k,
                heads,
                head_dim,
                rows,
                ..
            } => {
                fits(q, rows * heads * head_dim, "l2 q")?;
                fits(k, rows * heads * head_dim, "l2 k")?;
            }
            EncoderOp::Scan {
                q,
                k,
                v,
                gate,
                beta,
                state,
                out,
                key_dim,
                heads,
                head_dim,
                segments,
            } => {
                check_segments(segments)?;
                let n = segments_rows(segments);
                fits(q, n * key_dim, "scan q")?;
                fits(k, n * key_dim, "scan k")?;
                fits(v, n * heads * head_dim, "scan v")?;
                fits(gate, n * heads, "scan gate")?;
                fits(beta, n * heads, "scan beta")?;
                fits(state, heads * head_dim * head_dim, "scan state")?;
                fits(out, n * heads * head_dim, "scan out")?;
            }
            EncoderOp::GatedNorm {
                x,
                z,
                w,
                out,
                heads,
                head_dim,
                rows,
                ..
            } => {
                fits(x, rows * heads * head_dim, "gated norm x")?;
                fits(z, rows * heads * head_dim, "gated norm z")?;
                fits(w, *head_dim, "gated norm weight")?;
                fits(out, rows * heads * head_dim, "gated norm out")?;
            }
        }
        Ok(())
    }
}

/// Segments must tile the block contiguously from row 0 with `n_valid <= rows`.
fn check_segments(segments: &[EncoderSegment]) -> Result<(), DispatchError> {
    let mut next = 0;
    for s in segments {
        if s.row_offset != next || s.rows == 0 || s.n_valid == 0 || s.n_valid > s.rows {
            return Err(DispatchError::Hip(format!("encoder: bad segment {s:?}")));
        }
        next += s.rows;
    }
    if segments.is_empty() {
        return Err(DispatchError::Hip("encoder: no segments".into()));
    }
    Ok(())
}

pub fn execute_encoder(gpu: &mut Gpu, op: &EncoderOp<'_>) -> Result<(), DispatchError> {
    match op {
        EncoderOp::Project {
            w,
            x,
            out,
            residual,
            rows,
            out_dim,
            k,
        } => {
            project(gpu, w, x, out, *out_dim, *k, *rows)?;
            if let Some(r) = residual {
                gpu.add_inplace_f32(r, out).map_err(hip)?;
            }
            Ok(())
        }
        EncoderOp::Mrope {
            q,
            k,
            positions,
            heads,
            n_rot,
            theta,
            section,
            rows,
        } => gpu
            .rope_mrope_halfsplit_f32_batched(
                q,
                k,
                &positions.buf,
                heads.n_heads,
                heads.n_kv_heads,
                heads.head_dim,
                *n_rot,
                *theta,
                *rows,
                0,
                *section,
            )
            .map_err(hip),
        EncoderOp::Attention {
            q,
            k,
            v,
            ctx,
            segments,
            heads,
        } => {
            let (qd, kvd) = (heads.q_dim(), heads.kv_dim());
            for seg in segments.iter() {
                let q_s = q.sub_offset(seg.row_offset * qd, seg.rows * qd);
                let k_s = k.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
                let v_s = v.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
                let c_s = ctx.sub_offset(seg.row_offset * qd, seg.rows * qd);
                attention(gpu, &q_s, &k_s, &v_s, &c_s, seg.rows, seg.n_valid, heads)?;
            }
            Ok(())
        }
        EncoderOp::SigmoidGate { out, gate } => gpu.sigmoid_mul_f32(out, gate).map_err(hip),
        EncoderOp::AlphaGate {
            beta,
            alpha,
            dt_bias,
            a_log,
            heads,
            rows,
        } => gpu
            .fused_sigmoid_alpha_gate_f32_batched(beta, alpha, dt_bias, a_log, *heads, *rows)
            .map_err(hip),
        EncoderOp::ConvSplit {
            qkv,
            weight,
            ring,
            q,
            k,
            v,
            key_dim,
            value_dim,
            segments,
        } => {
            let (kd, vd) = (*key_dim, *value_dim);
            let ch = 2 * kd + vd;
            for seg in segments.iter() {
                gpu.zero_f32(ring).map_err(hip)?;
                let (o, r) = (seg.row_offset, seg.rows);
                gpu.conv1d_silu_split_f32_n(
                    &q.sub_offset(o * kd, r * kd),
                    &k.sub_offset(o * kd, r * kd),
                    &v.sub_offset(o * vd, r * vd),
                    &qkv.sub_offset(o * ch, r * ch),
                    weight,
                    ring,
                    kd,
                    vd,
                    r,
                )
                .map_err(hip)?;
            }
            Ok(())
        }
        EncoderOp::QkL2 {
            q,
            k,
            heads,
            head_dim,
            eps,
            rows,
        } => gpu
            .fused_qk_l2_norm_scale_f32_batched(
                q,
                k,
                *heads,
                *head_dim,
                1.0 / (*head_dim as f32).sqrt(),
                *eps,
                *rows,
            )
            .map_err(hip),
        EncoderOp::Scan {
            q,
            k,
            v,
            gate,
            beta,
            state,
            out,
            key_dim,
            heads,
            head_dim,
            segments,
        } => {
            let (kd, vd, nh) = (*key_dim, heads * head_dim, *heads);
            for seg in segments.iter() {
                gpu.zero_f32(state).map_err(hip)?;
                let (o, n) = (seg.row_offset, seg.rows);
                let (q, k, v) = (
                    q.sub_offset(o * kd, n * kd),
                    k.sub_offset(o * kd, n * kd),
                    v.sub_offset(o * vd, n * vd),
                );
                let (g, b) = (
                    gate.sub_offset(o * nh, n * nh),
                    beta.sub_offset(o * nh, n * nh),
                );
                let y = out.sub_offset(o * vd, n * vd);
                if rdna_compute::norm::gdn_chunked() && n > 1 {
                    gpu.gated_delta_net_f32_chunked(
                        &q,
                        &k,
                        &v,
                        &g,
                        &b,
                        state,
                        &y,
                        n,
                        nh,
                        *head_dim,
                        rdna_compute::norm::gdn_chunk_size(),
                    )
                } else {
                    gpu.gated_delta_net_f32_batch_seq(
                        &q, &k, &v, &g, &b, state, &y, n, nh, *head_dim,
                    )
                }
                .map_err(hip)?;
            }
            Ok(())
        }
        EncoderOp::GatedNorm {
            x,
            z,
            w,
            out,
            heads,
            head_dim,
            eps,
            rows,
        } => gpu
            .gated_norm_f32_batched(x, z, w, out, *heads, *head_dim, *eps, *rows)
            .map_err(hip),
    }
}

// ── projection ladder ───────────────────────────────────────────────────

fn projection_gfx1201_tile(m: usize) -> LdsTileSplitK {
    match m {
        0..512 => LdsTileSplitK::new(64, 128, 32, 64, true, 1),
        512..1024 => LdsTileSplitK::new(128, 128, 32, 64, false, 1),
        _ => LdsTileSplitK::new(128, 256, 32, 64, false, 1),
    }
}

fn cast_f16(gpu: &mut Gpu, x: &GpuTensor, n: usize) -> Result<GpuTensor, DispatchError> {
    let dst = gpu.alloc_tensor(&[n], DType::F16).map_err(hip)?;
    gpu.cast_f32_to_f16(x, &dst).map_err(hip)?;
    Ok(dst)
}

/// `y[n, out] = x[n, k] · w_f16[out, k]ᵀ`, F32 `x` and `y`.
///
/// Large shapes take the LDS-tiled macro-tile kernel; small ones keep
/// `gemm_f16_wmma_mb8`. Measured on gfx1101 at n=4096: mb8 runs the trunk's
/// GEMMs at ~1.6 TFLOP/s (2% of the f16 peak) because it is a register-blocked
/// 32-thread block with no LDS staging. The tiled kernel's X operand is f16,
/// so activations narrow (~1e-3 relative against the f32 reference, two
/// orders below what 4-bit weights cost). `HIPFIRE_EMBED_GEMM=f32` forces mb8.
fn project(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    out: usize,
    k: usize,
    n: usize,
) -> Result<(), DispatchError> {
    let wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch_caps.has_wmma_w32_gfx12();
    if !wmma {
        let yt = gpu.alloc_tensor(&[out * n], DType::F32).map_err(hip)?;
        let r = gpu
            .gemm_f16(w, x, &yt, out, k, n)
            .and_then(|_| gpu.transpose_f32(&yt, y, out, n));
        gpu.free_tensor(yt).map_err(hip)?;
        return r.map_err(hip);
    }
    let use_tiled = k % 64 == 0
        && out >= 256
        && !embed_gemm_f32()
        && (gpu.arch_caps.has_wmma_w32() || gpu.arch == "gfx1201");
    if use_tiled {
        let x16 = cast_f16(gpu, x, n * k)?;
        let r = if gpu.arch == "gfx1201" {
            gpu.gemm_f16_x_f16_wmma_lds_splitk(w, &x16, y, out, k, n, projection_gfx1201_tile(out))
        } else {
            gpu.gemm_f16_x_f16_wmma_lds_auto(w, &x16, y, None, out, k, n)
        };
        gpu.free_tensor(x16).map_err(hip)?;
        r.map_err(hip)
    } else {
        gpu.gemm_f16_wmma_mb8(w, x, y, out, k, n).map_err(hip)
    }
}

// ── attention ───────────────────────────────────────────────────────────

/// gfx11 LDS-tile selection for eager attention — `cands` is the caller's
/// preference order, `m`/`batch` the GEMM's M/N. Retained for tile-sweep labs;
/// production and parity use [`attention_gemm_ld`] for architecture dispatch.
pub fn attention_lds_tile(cands: &[LdsTile], m: usize, batch: usize) -> Option<LdsTile> {
    cands
        .iter()
        .copied()
        .find(|t| t.bm <= m && t.bn <= batch && Gpu::LDS_TILE_VARIANTS.contains(t))
}

/// The QK^T tile preference list (contract dim is small head_dim).
pub fn attention_qk_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 32, 64, 64, false).pipelined(),
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

/// The PV tile preference list (contract dim is the row count).
pub fn attention_pv_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(128, 128, 32, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

#[derive(Debug, PartialEq, Eq)]
enum AttentionRoute {
    Gfx1201(LdsTileSplitK),
    Gfx11(Option<LdsTile>),
}

fn attention_route(arch: &str, m: usize, batch: usize, qk: bool) -> AttentionRoute {
    if arch == "gfx1201" {
        let tile = if qk && m >= 128 && batch >= 128 {
            LdsTileSplitK::new(128, 128, 32, 64, false, 1)
        } else if !qk && m >= 64 && batch >= 128 {
            LdsTileSplitK::new(64, 128, 32, 64, true, 1)
        } else {
            LdsTileSplitK::new(64, 64, 32, 64, true, 1)
        };
        AttentionRoute::Gfx1201(tile)
    } else {
        let cands = if qk {
            attention_qk_tiles()
        } else {
            attention_pv_tiles()
        };
        AttentionRoute::Gfx11(attention_lds_tile(&cands, m, batch))
    }
}

/// Production eager-attention GEMM: F16 A/X (possibly pitched), F32 `[batch, m]`
/// output. gfx1201 uses optimized split1 LDS with ascending K; gfx11 preserves
/// its measured QK/PV candidate order and automatic fallback.
#[allow(clippy::too_many_arguments)]
pub fn attention_gemm_ld(
    gpu: &mut Gpu,
    a: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch: usize,
    lda: usize,
    ldx: usize,
    qk: bool,
) -> rdna_compute::HipResult<()> {
    if !gpu.arch_caps.has_wmma_w32() && gpu.arch != "gfx1201" {
        return Err(rdna_compute::HipError::new(
            0,
            "encoder: attention LDS requires gfx11 WMMA or exact gfx1201",
        ));
    }
    match attention_route(&gpu.arch, m, batch, qk) {
        AttentionRoute::Gfx1201(tile) => {
            gpu.gemm_f16_x_f16_wmma_lds_splitk_ld(a, x, y, m, k, batch, tile, lda, ldx)
        }
        AttentionRoute::Gfx11(Some(tile)) => {
            gpu.gemm_f16_x_f16_wmma_lds_tiled_ld(a, x, y, None, m, k, batch, tile, lda, ldx)
        }
        AttentionRoute::Gfx11(None) => {
            gpu.gemm_f16_x_f16_wmma_lds_auto_ld(a, x, y, None, m, k, batch, lda, ldx)
        }
    }
}

/// One segment's bidirectional attention. The Q-tiled kernel by default;
/// `HIPFIRE_EMBED_ATTN=naive` takes the per-(query, head) reference. Where the
/// eager decomposition applies (n % 64 == 0, n >= 256, head_dim % 64 == 0,
/// gfx11 WMMA or exact gfx1201) it runs instead, unless
/// `HIPFIRE_EMBED_ATTN=no-gemm`. Padding keys (rows past `n_valid`) are masked
/// on every route.
#[allow(clippy::too_many_arguments)]
fn attention(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    ctx: &GpuTensor,
    n: usize,
    n_valid: usize,
    h: &EncoderHeads,
) -> Result<(), DispatchError> {
    let scale = 1.0 / (h.head_dim as f32).sqrt();
    if embed_attn_naive() {
        if n == n_valid {
            return gpu
                .attention_text_f32(
                    q,
                    k,
                    v,
                    None,
                    None,
                    ctx,
                    n,
                    h.n_heads,
                    h.n_kv_heads,
                    h.head_dim,
                    scale,
                    false,
                )
                .map_err(hip);
        }
        // f32 key mask: 1.0 over the real tokens, 0.0 over the padding rows.
        let mut host = vec![0f32; n];
        for x in host.iter_mut().take(n_valid.min(n)) {
            *x = 1.0;
        }
        let mask = gpu.upload_f32(&host, &[n]).map_err(hip)?;
        let r = gpu.attention_text_f32(
            q,
            k,
            v,
            None,
            Some(&mask),
            ctx,
            n,
            h.n_heads,
            h.n_kv_heads,
            h.head_dim,
            scale,
            false,
        );
        gpu.free_tensor(mask).map_err(hip)?;
        return r.map_err(hip);
    }
    let wmma = gpu.arch_caps.has_wmma_w32() || gpu.arch == "gfx1201";
    if !embed_attn_no_gemm() && n % 64 == 0 && n >= 256 && h.head_dim % 64 == 0 && wmma {
        return attention_eager(gpu, q, k, v, ctx, n, n_valid, h);
    }
    gpu.attention_encoder_qtiled_f32(
        q,
        k,
        v,
        ctx,
        n,
        n_valid,
        h.n_heads,
        h.n_kv_heads,
        h.head_dim,
    )
    .map_err(hip)
}

/// Bidirectional attention as two batched GEMMs plus a row softmax, per head
/// (the WMMA GEMM runs this shape ~2.6x faster than the scalar tiled kernel).
///
/// The q/gate split at load puts each head's columns contiguously, so head
/// `h` is a `sub_offset` view with an explicit row pitch (the `_ld` GEMMs);
/// only V needs a real transpose. The PV GEMM contracts over `n`, hence the
/// `n % 64 == 0` contract. `1/sqrt(head_dim)` is folded into Q in f32 before
/// the f16 cast (n·q_dim elements, not n² scores).
#[allow(clippy::too_many_arguments)]
fn attention_eager(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    ctx: &GpuTensor,
    n: usize,
    n_valid: usize,
    h: &EncoderHeads,
) -> Result<(), DispatchError> {
    let hd = h.head_dim;
    let (qd, kvd) = (h.q_dim(), h.kv_dim());
    let rep = h.n_heads / h.n_kv_heads;

    gpu.scale_f32(q, 1.0 / (hd as f32).sqrt()).map_err(hip)?;
    let q16 = cast_f16(gpu, q, n * qd)?;
    let k16 = cast_f16(gpu, k, n * kvd)?;
    // 2-D: `softmax` takes rows from shape[0] and width from shape.last().
    let s = gpu.alloc_tensor(&[n, n], DType::F32).map_err(hip)?;
    let p16 = gpu.alloc_tensor(&[n * n], DType::F16).map_err(hip)?;
    let vh = gpu.alloc_tensor(&[n * hd], DType::F32).map_err(hip)?;
    let vht = gpu.alloc_tensor(&[hd * n], DType::F32).map_err(hip)?;
    let vht16 = gpu.alloc_tensor(&[hd * n], DType::F16).map_err(hip)?;
    let o = gpu.alloc_tensor(&[n * hd], DType::F32).map_err(hip)?;

    let mut result = Ok(());
    for head in 0..h.n_heads {
        let kh = head / rep;
        // Head views: (rows - 1) * pitch + k elements, as the GEMM bounds want.
        let q_h = q16.sub_offset(head * hd, (n - 1) * qd + hd);
        let k_h = k16.sub_offset(kh * hd, (n - 1) * kvd + hd);
        let v_h = v.sub_offset(kh * hd, (n - 1) * kvd + hd);
        let step = (|| -> rdna_compute::HipResult<()> {
            // S = Q_h · K_hᵀ
            attention_gemm_ld(gpu, &k_h, &q_h, &s, n, hd, n, kvd, qd, true)?;
            // Masked softmax to f16; columns past n_valid are padding.
            gpu.attention_rows_softmax_f16(&s, &p16, n, n, 0, n_valid)?;
            // V_h -> [hd, n] f16 (the source view carries the column offset).
            gpu.copy_rows_strided_f32(&v_h, &vh, n, hd, kvd, hd, 0, None)?;
            gpu.transpose_f32(&vh, &vht, n, hd)?;
            gpu.cast_f32_to_f16(&vht, &vht16)?;
            // O_h = P · V_h, scattered into the head-interleaved ctx columns.
            attention_gemm_ld(gpu, &vht16, &p16, &o, hd, n, n, n, n, false)?;
            gpu.copy_rows_strided_f32(&o, ctx, n, hd, hd, qd, head * hd, None)
        })();
        if let Err(e) = step {
            result = Err(DispatchError::Hip(format!(
                "encoder: attention head {head}: {e:?}"
            )));
            break;
        }
    }
    for t in [q16, k16, s, p16, vh, vht, vht16, o] {
        gpu.free_tensor(t).map_err(hip)?;
    }
    result
}

// ── developer knobs (read once) ─────────────────────────────────────────

fn knob(name: &str, values: &[&str]) -> bool {
    hipfire_config::developer_var(name)
        .ok()
        .is_some_and(|v| values.contains(&v.as_str()))
}

use std::sync::LazyLock;

/// `HIPFIRE_EMBED_GEMM=f32|mb8`: every projection on `gemm_f16_wmma_mb8`.
static EMBED_GEMM_F32: LazyLock<bool> =
    LazyLock::new(|| knob("HIPFIRE_EMBED_GEMM", &["f32", "mb8"]));
/// `HIPFIRE_EMBED_ATTN=no-gemm|tiled`: keep the tiled kernel where eager applies.
static EMBED_ATTN_NO_GEMM: LazyLock<bool> =
    LazyLock::new(|| knob("HIPFIRE_EMBED_ATTN", &["no-gemm", "tiled"]));
/// `HIPFIRE_EMBED_ATTN=naive`: the per-(query, head) reference kernel.
static EMBED_ATTN_NAIVE: LazyLock<bool> = LazyLock::new(|| knob("HIPFIRE_EMBED_ATTN", &["naive"]));

fn embed_gemm_f32() -> bool {
    *EMBED_GEMM_F32
}
fn embed_attn_no_gemm() -> bool {
    *EMBED_ATTN_NO_GEMM
}
fn embed_attn_naive() -> bool {
    *EMBED_ATTN_NAIVE
}

#[cfg(test)]
mod tests {
    use super::{
        attention_route, check_segments, projection_gfx1201_tile, AttentionRoute, EncoderSegment,
        LdsTile, LdsTileSplitK,
    };

    #[test]
    fn gfx1201_projection_tiles_are_supported_unsplit_entries() {
        for (m, entry) in [
            (256, "qwen4_wmma_lds_64_128_32_64_k64_p"),
            (511, "qwen4_wmma_lds_64_128_32_64_k64_p"),
            (512, "qwen4_wmma_lds_128_128_32_64_k64"),
            (1023, "qwen4_wmma_lds_128_128_32_64_k64"),
            (1024, "qwen4_wmma_lds_128_256_32_64_k64"),
            (4096, "qwen4_wmma_lds_128_256_32_64_k64"),
        ] {
            let tile = projection_gfx1201_tile(m);
            assert_eq!(tile.split, 1);
            assert_eq!(tile.ks, 64);
            assert_eq!(tile.entry(), entry);
        }
    }

    #[test]
    fn gfx1201_attention_tiles_are_supported_unsplit_entries() {
        for (m, batch, qk, tile, entry) in [
            (
                128,
                128,
                true,
                LdsTileSplitK::new(128, 128, 32, 64, false, 1),
                "qwen4_wmma_lds_128_128_32_64_k64",
            ),
            (
                127,
                128,
                true,
                LdsTileSplitK::new(64, 64, 32, 64, true, 1),
                "qwen4_wmma_lds_64_64_32_64_k64_p",
            ),
            (
                128,
                127,
                true,
                LdsTileSplitK::new(64, 64, 32, 64, true, 1),
                "qwen4_wmma_lds_64_64_32_64_k64_p",
            ),
            (
                64,
                128,
                false,
                LdsTileSplitK::new(64, 128, 32, 64, true, 1),
                "qwen4_wmma_lds_64_128_32_64_k64_p",
            ),
            (
                63,
                128,
                false,
                LdsTileSplitK::new(64, 64, 32, 64, true, 1),
                "qwen4_wmma_lds_64_64_32_64_k64_p",
            ),
            (
                64,
                127,
                false,
                LdsTileSplitK::new(64, 64, 32, 64, true, 1),
                "qwen4_wmma_lds_64_64_32_64_k64_p",
            ),
        ] {
            assert_eq!(
                attention_route("gfx1201", m, batch, qk),
                AttentionRoute::Gfx1201(tile)
            );
            assert_eq!(tile.split, 1);
            assert_eq!(tile.ks, 64);
            assert_eq!(tile.entry(), entry);
        }
    }

    #[test]
    fn gfx11_attention_preserves_candidate_order_and_auto_fallback() {
        for arch in ["gfx1100", "gfx1151"] {
            for qk in [true, false] {
                let large = if qk {
                    LdsTile::new(128, 128, 32, 64, 64, false).pipelined()
                } else {
                    LdsTile::new(128, 128, 64, 64, 64, false)
                };
                assert_eq!(
                    attention_route(arch, 256, 256, qk),
                    AttentionRoute::Gfx11(Some(large))
                );
                let small = LdsTile::new(64, 64, 32, 64, 64, false);
                assert_eq!(
                    attention_route(arch, 64, 128, qk),
                    AttentionRoute::Gfx11(Some(small))
                );
                assert_eq!(
                    attention_route(arch, 128, 64, qk),
                    AttentionRoute::Gfx11(Some(small))
                );
                assert_eq!(
                    attention_route(arch, 63, 128, qk),
                    AttentionRoute::Gfx11(None)
                );
                assert_eq!(
                    attention_route(arch, 128, 63, qk),
                    AttentionRoute::Gfx11(None)
                );
            }
        }
    }

    #[test]
    fn segments_must_tile_the_block_from_row_zero() {
        let s = |row_offset, rows, n_valid| EncoderSegment {
            row_offset,
            rows,
            n_valid,
        };
        assert!(check_segments(&[s(0, 7, 7), s(7, 320, 302)]).is_ok());
        // A gap would leave a stale row block; an overlap a cross-sequence read.
        assert!(check_segments(&[s(0, 7, 7), s(8, 320, 302)]).is_err());
        assert!(check_segments(&[s(0, 7, 7), s(6, 320, 302)]).is_err());
        assert!(check_segments(&[s(1, 7, 7)]).is_err());
        assert!(check_segments(&[s(0, 7, 8)]).is_err());
        assert!(check_segments(&[]).is_err());
    }
}
