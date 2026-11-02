// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The GPU encoder forward: tokens (+ optional merged image embeds) →
//! 128-d per-token embeddings.
//!
//! Layer loop mirrors HF `Qwen3_5Model` with `is_causal = false`:
//!
//! - **Full-attention layers** run **bidirectional**: pre-norm → fused
//!   q/gate + k + v GEMMs → per-head Q/K RMSNorm → interleaved mRoPE →
//!   [`Gpu::attention_text_f32`] with `causal = false` (the T5/Qwen3-CLIP
//!   encoder kernel; no KV cache, no padding mask at one sequence per
//!   encode) → `ctx · σ(gate)` → o_proj → residual.
//! - **GDN layers** keep their causal recurrence (kernel-inherent; HF
//!   ignores `is_causal` for linear attention): pre-norm → qkv/z/a/b GEMMs
//!   → `fused_sigmoid_alpha_gate` (β = σ(b), g = −exp(A_log)·softplus(a +
//!   dt_bias)) → causal conv + SiLU split → per-head L2 norm + scale →
//!   `gated_delta_net_f32_chunked` (FP32 state, oracle-equal to the
//!   sequential scan) → gated RMSNorm × silu(z) → out_proj → residual.
//! - Shared SwiGLU MLP after every mixer, final RMSNorm, then the 128-d
//!   Dense head.
//!
//! Kernel-call shapes are lifted from the production qwen35 prefill path
//! (`hipfire-arch-qwen35/src/qwen35/prefill.rs`). Eligible projections and
//! eager attention use optimized per-architecture LDS WMMA (gfx11 incumbent,
//! gfx1201 ascending-K split1). Small projections retain WMMA `mb8` with
//! naive fallback. Projection inputs and all norms are F32; LDS operands
//! narrow to F16 and GEMM outputs remain F32.

use crate::config::EmbedConfig;
use crate::state::GdnEncodeState;
use crate::weights::{EmbedWeights, ImageSpan};
use rdna_compute::gemm::{LdsTile, LdsTileSplitK};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult};

/// Input assembled by the caller (bundle / generate body): token ids, the
/// host hidden-state rows for image positions, and per-token (t, h, w)
/// mRoPE positions.
pub struct EncodeInput<'a> {
    pub ids: &'a [u32],
    /// Merged image embeds `[k, hidden]` (empty for text-only).
    pub image_embeds: &'a [f32],
    pub spans: &'a [ImageSpan],
    /// `[n][3]` (t, h, w) positions — text rows carry `[p, p, p]` (bit-equal
    /// to the 1D rope angles), image rows the 2D grid phases.
    pub pos3: &'a [[i32; 3]],
}

/// gfx11 LDS-tile selection for eager attention — `cands` is the caller's
/// preference order, `m`/`batch` the GEMM's M/N. Retained for tile-sweep labs;
/// production and parity use [`attention_gemm_ld`] for architecture dispatch.
pub fn attention_lds_tile(cands: &[LdsTile], m: usize, batch: usize) -> Option<LdsTile> {
    cands
        .iter()
        .copied()
        .find(|t| t.bm <= m && t.bn <= batch && Gpu::LDS_TILE_VARIANTS.contains(t))
}

/// The QK^T tile preference list (contract dim is small head_dim) —
/// mirrors `Enc::gemm_ld(qk=true)`.
pub fn attention_qk_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 32, 64, 64, false).pipelined(),
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

/// The PV tile preference list (contract dim is the row count) —
/// mirrors `Enc::gemm_ld(qk=false)`.
pub fn attention_pv_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(128, 128, 32, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

fn projection_gfx1201_tile(m: usize) -> LdsTileSplitK {
    match m {
        0..512 => LdsTileSplitK::new(64, 128, 32, 64, true, 1),
        512..1024 => LdsTileSplitK::new(128, 128, 32, 64, false, 1),
        _ => LdsTileSplitK::new(128, 256, 32, 64, false, 1),
    }
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
) -> HipResult<()> {
    if !gpu.arch_caps.has_wmma_w32() && gpu.arch != "gfx1201" {
        return Err(rdna_compute::HipError::new(
            0,
            "qwen35-embed: attention LDS requires gfx11 WMMA or exact gfx1201",
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

/// Small scratch wrapper: alloc/upload/free + the f16 GEMM dispatch. A
/// stripped `TextGpu` (that one is `pub(crate)` in hipfire-arch-diffusion),
/// with the WMMA/naive split copied from qwen35-vl's `linear_f16`.
struct Enc<'a> {
    gpu: &'a mut Gpu,
}

impl<'a> Enc<'a> {
    fn new(gpu: &'a mut Gpu) -> Self {
        Self { gpu }
    }

    fn alloc(&mut self, n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("qwen35-embed: alloc: {e:?}"))
    }

    fn upload(&mut self, data: &[f32], n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .upload_f32(data, &[n])
            .map_err(|e| format!("qwen35-embed: upload: {e:?}"))
    }

    fn free(&mut self, t: GpuTensor) -> Result<(), String> {
        self.gpu
            .free_tensor(t)
            .map_err(|e| format!("qwen35-embed: free: {e:?}"))
    }

    /// `y[n, out] = x[n, k] · w_f16[out, k]ᵀ` (no bias — every text
    /// projection here is bias-free).
    ///
    /// **`x` is F32.** `gemm_f16_wmma_mb8` and naive `gemm_f16` read X as
    /// F32; eligible LDS projections explicitly cast it to F16 scratch.
    /// Weights are F16 and the `[n, out]` output is always F32.
    fn gemm(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        out: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuTensor, String> {
        if k == 0 || k % 16 != 0 {
            return Err(format!("qwen35-embed: GEMM k={k} is not a multiple of 16"));
        }
        let y = self.alloc(n * out)?;
        let wmma = self.gpu.arch_caps.has_wmma_w32() || self.gpu.arch_caps.has_wmma_w32_gfx12();
        if !wmma {
            let yt = self.alloc(out * n)?;
            self.gpu
                .gemm_f16(w, x, &yt, out, k, n)
                .map_err(|e| format!("qwen35-embed: gemm (naive): {e:?}"))?;
            self.gpu
                .transpose_f32(&yt, &y, out, n)
                .map_err(|e| format!("qwen35-embed: transpose: {e:?}"))?;
            self.free(yt)?;
            return Ok(y);
        }
        // Large shapes take the LDS-tiled macro-tile kernel; small ones keep
        // `gemm_f16_wmma_mb8`. Measured on gfx1101 at n=4096: mb8 runs the
        // trunk's GEMMs at ~1.6 TFLOP/s (2% of the f16 peak) because it is a
        // register-blocked 32-thread block with no LDS staging, which is the
        // right choice for the small batches it was written for (the vision
        // tower) and the wrong one here. The tiled kernel is the same one the
        // FLUX text encoders use for their prefill.
        //
        // Cost of the switch: the tiled kernel's X operand is f16 (mb8's is
        // f32), so activations narrow. That is ~1e-3 relative against the f32
        // reference — two orders below what 4-bit weights already cost
        // (6e-2, see scripts/check_embed_quant.py) and validated by the same
        // oracle. `HIPFIRE_EMBED_GEMM=f32` forces mb8 to A/B it or to keep
        // activations full precision.
        let use_tiled = k % 64 == 0
            && out >= 256
            && !embed_gemm_f32()
            && (self.gpu.arch_caps.has_wmma_w32() || self.gpu.arch == "gfx1201");
        if use_tiled {
            let x16 = self.cast_f16(x, n * k)?;
            let r = if self.gpu.arch == "gfx1201" {
                self.gpu.gemm_f16_x_f16_wmma_lds_splitk(
                    w,
                    &x16,
                    &y,
                    out,
                    k,
                    n,
                    projection_gfx1201_tile(out),
                )
            } else {
                self.gpu
                    .gemm_f16_x_f16_wmma_lds_auto(w, &x16, &y, None, out, k, n)
            };
            self.free(x16)?;
            r.map_err(|e| format!("qwen35-embed: gemm (lds-tiled): {e:?}"))?;
        } else {
            self.gpu
                .gemm_f16_wmma_mb8(w, x, &y, out, k, n)
                .map_err(|e| format!("qwen35-embed: gemm: {e:?}"))?;
        }
        Ok(y)
    }

    /// One attention GEMM with a measured tile where the shape allows one.
    /// `qk` selects the candidate list: QK^T contracts over head_dim (small K),
    /// PV over n (large K), and they want different tiles.
    #[allow(clippy::too_many_arguments)]
    fn gemm_ld(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
        lda: usize,
        ldx: usize,
        qk: bool,
    ) -> Result<(), rdna_compute::HipError> {
        attention_gemm_ld(self.gpu, a, x, y, m, k, batch, lda, ldx, qk)
    }

    fn alloc_shaped(&mut self, shape: &[usize], dtype: DType) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, dtype)
            .map_err(|e| format!("qwen35-embed: alloc: {e:?}"))
    }

    fn alloc_f16(&mut self, n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(&[n], DType::F16)
            .map_err(|e| format!("qwen35-embed: alloc f16: {e:?}"))
    }

    /// f32 `[n]` → f16 `[n]` scratch (the LDS-tiled GEMM's activation operand).
    fn cast_f16(&mut self, x: &GpuTensor, n: usize) -> Result<GpuTensor, String> {
        let dst = self
            .gpu
            .alloc_tensor(&[n], DType::F16)
            .map_err(|e| format!("qwen35-embed: alloc f16: {e:?}"))?;
        self.gpu
            .cast_f32_to_f16(x, &dst)
            .map_err(|e| format!("qwen35-embed: cast f16: {e:?}"))?;
        Ok(dst)
    }

    fn rmsnorm(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
        batch: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.gpu
            .rmsnorm_batched(x, w, out, batch, d, eps)
            .map_err(|e| format!("qwen35-embed: rmsnorm: {e:?}"))
    }

    fn download(&mut self, t: &GpuTensor) -> Result<Vec<f32>, String> {
        self.gpu
            .download_f32(t)
            .map_err(|e| format!("qwen35-embed: download: {e:?}"))
    }

    fn add_inplace(&mut self, a: &GpuTensor, b: &GpuTensor, ctx: &str) -> Result<(), String> {
        self.gpu
            .add_inplace_f32(a, b)
            .map_err(|e| format!("qwen35-embed: {ctx}: {e:?}"))
    }

    fn silu_mul(&mut self, gate: &GpuTensor, up: &GpuTensor) -> Result<(), String> {
        self.gpu
            .silu_mul_f32(gate, up, gate)
            .map_err(|e| format!("qwen35-embed: silu: {e:?}"))
    }

    fn memcpy_htod(&mut self, dst: &GpuTensor, bytes: &[u8]) -> Result<(), String> {
        self.gpu
            .hip
            .memcpy_htod(&dst.buf, bytes)
            .map_err(|e| format!("qwen35-embed: memcpy_htod: {e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn rope_mrope(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        pos3: &GpuTensor,
        cfg: &EmbedConfig,
        n: usize,
    ) -> Result<(), rdna_compute::HipError> {
        self.gpu.rope_mrope_halfsplit_f32_batched(
            q,
            k,
            &pos3.buf,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.n_rot(),
            cfg.rope_theta,
            n,
            0,
            cfg.mrope_section,
        )
    }

    /// Bidirectional attention as two batched GEMMs plus a row softmax, per
    /// head — the "eager attention" decomposition, using the primitives that
    /// already exist rather than a hand-written kernel.
    ///
    /// Why: the tiled SIMT kernel runs the attention arithmetic at ~2.05
    /// TFLOP/s (6.9% of f32 peak) because it is scalar f32 FMA. The trunk's
    /// WMMA GEMM measured **5.5-6.25 TFLOP/s at exactly this shape**
    /// (`[M=n_k, K=head_dim, B=n_q]`, see `examples/bench_gemm_tiles`), so
    /// decomposing attention into `S = Q·Kᵀ`, a row softmax, and `O = P·V`
    /// buys ~2.6x on the same math with a kernel path that is already tuned
    /// and measured.
    ///
    /// Layout trick that avoids any gather: the q/gate weight split at load
    /// puts each head's columns contiguously (`upload_q_gate_split`), so head
    /// `h` is a `sub_offset` view with an explicit row pitch — which is what
    /// the `_ld` GEMM variants take. Only V needs a real transpose (its `[n,
    /// hd]` must become `[hd, n]` for the second GEMM's A operand), and that
    /// is a strided copy + `transpose_f32` + cast.
    ///
    /// Constraint: the tiled GEMM asserts `k % 64 == 0`, and the second GEMM's
    /// contraction is `n`, so this path needs `n % 64 == 0`. Shorter inputs
    /// keep the tiled kernel (cheap at small n) — see `attention`.
    #[allow(clippy::too_many_arguments)]
    fn attention_gemm(
        &mut self,
        q: &mut GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        ctx: &GpuTensor,
        n: usize,
        n_valid: usize,
        cfg: &EmbedConfig,
    ) -> Result<(), String> {
        let hd = cfg.head_dim;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let heads = cfg.n_heads;
        let rep = cfg.n_heads / cfg.n_kv_heads;

        // The softmax needs QK^T / sqrt(head_dim). The GEMM has no scale, so
        // the factor is folded into Q once per layer (n * q_heads * head_dim
        // elements) rather than into the n x n scores (16x more work at
        // n = 4096) — and in f32 before the f16 cast, so it costs no
        // precision. The tiled kernel applies its own scale, which is why this
        // lives here and not in the caller.
        self.gpu
            .scale_f32(q, 1.0 / (hd as f32).sqrt())
            .map_err(|e| format!("qwen35-embed: attn Q scale: {e:?}"))?;
        let q16 = self.cast_f16(q, n * qd)?;
        let k16 = self.cast_f16(k, n * kvd)?;
        // Scores are [n_q, n_k]; both are n here (one sequence, self-attention).
        // NOTE the shape must be 2-D: `softmax_f32` takes its row count from
        // `shape[0]` and its width from `shape.last()`, so a flat [n*n]
        // allocation makes it softmax the whole matrix as one row of n^2
        // elements (measured: O(1) wrong embeddings).
        let s = self.alloc_shaped(&[n, n], DType::F32)?;
        let p16 = self.alloc_f16(n * n)?;
        let vh = self.alloc(n * hd)?;
        let vht = self.alloc(hd * n)?;
        let vht16 = self.alloc_f16(hd * n)?;
        let o = self.alloc(n * hd)?;

        let mut result = Ok(());
        for h in 0..heads {
            let kh = h / rep;
            // `sub_offset` views: head `h` of a [n, qd] buffer, first `hd`
            // columns of each row, row pitch qd. Length is exactly what the
            // GEMM's bounds assert wants: (rows - 1) * pitch + k.
            let q_h = q16.sub_offset(h * hd, (n - 1) * qd + hd);
            let k_h = k16.sub_offset(kh * hd, (n - 1) * kvd + hd);
            let step = (|| -> Result<(), String> {
                // S = Q_h · K_hᵀ   ([n, hd] · [hd, n] -> [n, n])
                self.gemm_ld(&k_h, &q_h, &s, n, hd, n, kvd, qd, true)
                    .map_err(|e| format!("qwen35-embed: attn QK^T (head {h}): {e:?}"))?;
                // Masked softmax writing f16 in one pass. Columns past
                // `n_valid` are the sequence padding: their K rows are zeros,
                // so an unmasked softmax would hand them a real share of every
                // row and dilute the attention.
                self.gpu
                    .attention_rows_softmax_f16(&s, &p16, n, n, 0, n_valid)
                    .map_err(|e| format!("qwen35-embed: attn softmax (head {h}): {e:?}"))?;
                // V_h -> [hd, n] f16 for the second GEMM's A operand. The
                // head's column offset goes on the SOURCE view: this helper
                // has a dst column offset but no src one, so shifting the
                // source is how a strided column slice is expressed.
                let v_h = v.sub_offset(kh * hd, (n - 1) * kvd + hd);
                self.gpu
                    .copy_rows_strided_f32(&v_h, &vh, n, hd, kvd, hd, 0, None)
                    .map_err(|e| format!("qwen35-embed: attn V gather (head {h}): {e:?}"))?;
                self.gpu
                    .transpose_f32(&vh, &vht, n, hd)
                    .map_err(|e| format!("qwen35-embed: attn V transpose (head {h}): {e:?}"))?;
                self.gpu
                    .cast_f32_to_f16(&vht, &vht16)
                    .map_err(|e| format!("qwen35-embed: attn V cast (head {h}): {e:?}"))?;
                // O_h = P · V_h
                self.gemm_ld(&vht16, &p16, &o, hd, n, n, n, n, false)
                    .map_err(|e| format!("qwen35-embed: attn PV (head {h}): {e:?}"))?;
                // Scatter the head back into the head-interleaved ctx columns.
                self.gpu
                    .copy_rows_strided_f32(&o, ctx, n, hd, hd, qd, h * hd, None)
                    .map_err(|e| format!("qwen35-embed: attn O scatter (head {h}): {e:?}"))?;
                Ok(())
            })();
            if step.is_err() {
                result = step;
                break;
            }
        }

        for t in [q16, k16, s, p16, vh, vht, vht16, o] {
            self.free(t)?;
        }
        result
    }

    fn attention(
        &mut self,
        q: &mut GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        ctx: &GpuTensor,
        n: usize,
        n_valid: usize,
        cfg: &EmbedConfig,
    ) -> Result<(), rdna_compute::HipError> {
        // The Q-tiled bidirectional kernel (K shared across QB query rows)
        // unless `HIPFIRE_EMBED_ATTN=naive` asks for the one-block-per-row
        // reference — that kernel is correct for any shape but re-reads all of
        // K/V per query row, which is ~82% of the encode at N=4096.
        //
        // Padding: since the eager-path pad landed, a segment with
        // `n_valid >= 256` has `rows` padded up to a 64 multiple and its
        // K/V rows past `n_valid` are zeros. Any fallback kernel must still
        // mask them — `attention_text_f32` gets an f32 key mask, the tiled
        // kernel takes `n_valid` as its key bound.
        if embed_attn_naive() {
            if n == n_valid {
                return self.gpu.attention_text_f32(
                    q,
                    k,
                    v,
                    None,
                    None,
                    ctx,
                    n,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    1.0 / (cfg.head_dim as f32).sqrt(),
                    false,
                );
            }
            let mask = self.key_mask(n, n_valid)?;
            let r = self.gpu.attention_text_f32(
                q,
                k,
                v,
                None,
                Some(&mask),
                ctx,
                n,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                1.0 / (cfg.head_dim as f32).sqrt(),
                false,
            );
            self.free(mask)
                .map_err(|e| rdna_compute::HipError::new(0, &e))?;
            return r;
        }
        // The eager-attention decomposition needs n % 64 == 0 (the PV GEMM
        // contracts over n) and head_dim % 64 == 0 (the LDS-tiled WMMA GEMM's
        // K), plus gfx11 WMMA or exact gfx1201. Short inputs, unsupported
        // shapes and other architectures (including other gfx12) retain the
        // tiled SIMT kernel, which handles any n and is cheap at small n
        // precisely because its cost is O(n^2).
        let wmma = self.gpu.arch_caps.has_wmma_w32() || self.gpu.arch == "gfx1201";
        if !embed_attn_no_gemm() && n % 64 == 0 && n >= 256 && cfg.head_dim % 64 == 0 && wmma {
            return self
                .attention_gemm(q, k, v, ctx, n, n_valid, cfg)
                .map_err(|e| rdna_compute::HipError::new(0, &e));
        }
        self.gpu.attention_encoder_qtiled_f32(
            q,
            k,
            v,
            ctx,
            n,
            n_valid,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        )
    }

    /// f32 `[n]` key mask for the naive attention path: 1.0 over the real
    /// tokens, 0.0 over the padding rows. Built on host and uploaded — the
    /// knob it serves is a debug A/B path, not a hot one.
    fn key_mask(&mut self, n: usize, n_valid: usize) -> Result<GpuTensor, rdna_compute::HipError> {
        let mut host = vec![0f32; n];
        for v in host.iter_mut().take(n_valid.min(n)) {
            *v = 1.0;
        }
        self.gpu.upload_f32(&host, &[n])
    }

    fn sigmoid_mul(&mut self, out: &GpuTensor, gate: &GpuTensor) -> Result<(), String> {
        self.gpu
            .sigmoid_mul_f32(out, gate)
            .map_err(|e| format!("qwen35-embed: sigmoid gate: {e:?}"))
    }

    fn alpha_gate(
        &mut self,
        beta: &GpuTensor,
        alpha: &GpuTensor,
        dt_bias: &GpuTensor,
        a_log: &GpuTensor,
        n_heads: usize,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .fused_sigmoid_alpha_gate_f32_batched(beta, alpha, dt_bias, a_log, n_heads, n)
            .map_err(|e| format!("qwen35-embed: alpha gate: {e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_split(
        &mut self,
        q_out: &GpuTensor,
        k_out: &GpuTensor,
        v_out: &GpuTensor,
        input: &GpuTensor,
        weight: &GpuTensor,
        state: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .conv1d_silu_split_f32_n(q_out, k_out, v_out, input, weight, state, k_dim, v_dim, n)
            .map_err(|e| format!("qwen35-embed: conv: {e:?}"))
    }

    fn qk_l2_norm(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        cfg: &EmbedConfig,
        khd: usize,
        eps: f32,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .fused_qk_l2_norm_scale_f32_batched(
                q,
                k,
                cfg.linear_num_key_heads,
                khd,
                1.0 / (khd as f32).sqrt(),
                eps,
                n,
            )
            .map_err(|e| format!("qwen35-embed: qk norm: {e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_scan(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        s: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), String> {
        let res = if rdna_compute::norm::gdn_chunked() && n > 1 {
            self.gpu.gated_delta_net_f32_chunked(
                q,
                k,
                v,
                gate,
                beta,
                s,
                out,
                n,
                n_heads,
                head_dim,
                rdna_compute::norm::gdn_chunk_size(),
            )
        } else {
            self.gpu
                .gated_delta_net_f32_batch_seq(q, k, v, gate, beta, s, out, n, n_heads, head_dim)
        };
        res.map_err(|e| format!("qwen35-embed: gdn: {e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_norm(
        &mut self,
        x: &GpuTensor,
        z: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .gated_norm_f32_batched(x, z, w, out, n_heads, head_dim, eps, n)
            .map_err(|e| format!("qwen35-embed: gated norm: {e:?}"))
    }
}

/// Env-gated (`HIPFIRE_EMBED_DEBUG=1`) finite-value trace of the residual
/// stream: prints absmax and the non-finite count at each stage so a NaN can
/// be localized to a layer and a mixer kind. Downloads one tensor per call,
/// so this is strictly a debugging path.
fn trace(
    enc: &mut Enc<'_>,
    label: impl FnOnce() -> String,
    t: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    if !debug_enabled() {
        return Ok(());
    }
    let label = label();
    let v = enc.download(t)?;
    let mut absmax = 0f32;
    let mut bad = 0usize;
    for &x in &v[..n.min(v.len())] {
        if !x.is_finite() {
            bad += 1;
        } else if x.abs() > absmax {
            absmax = x.abs();
        }
    }
    eprintln!(
        "  [embed-debug] {label}: absmax={absmax:.4} nonfinite={bad}/{}",
        v.len()
    );
    Ok(())
}

/// `HIPFIRE_EMBED_GEMM=f32` forces the f32-activation `gemm_f16_wmma_mb8` on
/// every projection instead of the LDS-tiled f16-activation kernel used for
/// the large shapes. Read once; the tiled kernel is the production path.
fn embed_gemm_f32() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_EMBED_GEMM")
                .ok()
                .as_deref(),
            Some("f32") | Some("mb8")
        )
    })
}

/// `HIPFIRE_EMBED_ATTN=no-gemm` keeps the tiled SIMT attention even where the
/// eager-attention decomposition applies. Read once.
fn embed_attn_no_gemm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_EMBED_ATTN")
                .ok()
                .as_deref(),
            Some("no-gemm") | Some("tiled")
        )
    })
}

/// `HIPFIRE_EMBED_ATTN=naive` selects the reference per-(query, head)
/// attention kernel instead of the Q-tiled one. Read once; the tiled kernel is
/// the production path.
fn embed_attn_naive() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            hipfire_config::developer_var("HIPFIRE_EMBED_ATTN")
                .ok()
                .as_deref(),
            Some("naive")
        )
    })
}

/// `HIPFIRE_EMBED_DEBUG=1`, read once (the trace calls are on the hot path;
/// the label closure keeps the `format!` off it when disabled).
fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_EMBED_DEBUG")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Run the full encoder and the Dense head; returns `[n, dense_out]`
/// row-major **unmasked, unnormalized** token embeddings (the
/// sentence-transformers modules 2/3 run host-side in [`crate::post`]).
///
/// Resets the GDN state on entry (every encode starts from zero recurrence).
pub fn encode_embeddings(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<Vec<f32>, String> {
    let mut out = encode_embeddings_batch(gpu, w, state, cfg, std::slice::from_ref(input))?;
    Ok(out.pop().expect("one input in, one embedding out"))
}

/// Several sequences in one forward, one `[n_i, dense_out]` row-major block
/// per input (same per-input shape and content as [`encode_embeddings`]).
///
/// The batch is *row-concatenation*, not padding to a common length: each
/// sequence keeps its own real length `n_i` and is padded only to its own
/// [`padded_rows`] multiple. That works because nothing in this architecture
/// couples rows across sequences except the two stateful mixers, and both are
/// run per sequence over a row slice:
///
/// - GatedDeltaNet's conv ring and delta rule carry state along rows, so each
///   segment runs with the layer state reset at its start;
/// - full attention is bidirectional *within* a sequence and must not see
///   another sequence's keys, so each segment gets its own score matrix,
///   softmax over its own rows, and its own `n_valid` bound.
///
/// Everything else (all projections, both norms, the gates, mRoPE, the MLP,
/// the Dense head) is row-independent and runs once over the whole row block.
/// Concretely this means a batch needs neither an attention mask nor a common
/// padded length — the equivalent of a `[B·L, B·L]` score matrix never exists.
///
/// The cost is that the per-sequence parts do not amortize: a batch of 8 short
/// sequences costs about as much attention and GDN work as 8 separate encodes,
/// and saves on the row-independent work (the GEMMs, dominated by the
/// weights). `T` should therefore be capped near where the weight-bound work
/// stops dominating; see the slot engine's batch window.
pub fn encode_embeddings_batch(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
) -> Result<Vec<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Err("qwen35-embed: empty batch".into());
    }
    let segs = segment_layout(inputs)?;
    let t_rows: usize = segs.iter().map(|s| s.rows).sum();
    state.reset(gpu)?;

    let hidden = encode_hidden_segments(gpu, w, state, cfg, inputs, &segs, t_rows)?;
    // Dense head: norm → f16 → [t_rows, dense_out]. Row-independent, so it runs
    // over the whole row block in one call.
    let dense_out = cfg.dense_out_features;
    let mut e = Enc::new(gpu);
    let dense = {
        let fin = e.alloc(t_rows * cfg.hidden_size)?;
        e.rmsnorm(
            &hidden,
            &w.final_norm,
            &fin,
            t_rows,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let emb = e.gemm(&w.dense, &fin, dense_out, cfg.hidden_size, t_rows)?;
        e.free(fin)?;
        emb
    };
    let all = e.download(&dense)?;
    e.free(dense)?;
    e.free(hidden)?;

    // Drop each segment's padding rows: the caller's mask/normalize expects
    // exactly one row per real token.
    let mut out = Vec::with_capacity(inputs.len());
    for (input, seg) in inputs.iter().zip(&segs) {
        let n = input.ids.len();
        let start = seg.row_offset * dense_out;
        out.push(all[start..start + n * dense_out].to_vec());
    }
    Ok(out)
}

/// The backbone only: returns the final-normed hidden states
/// `[n, hidden]` as a device tensor (freed by the caller). Split out so a
/// future pooled/mean variant can reuse everything above the Dense head.
///
/// Unpadded and without a state reset — this is the debug/inspection entry
/// point, not the serving one; use [`encode_embeddings`] for correctness.
pub fn encode_hidden(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<GpuTensor, String> {
    let n = input.ids.len();
    if n == 0 {
        return Err("qwen35-embed: empty input".into());
    }
    let segs = [EncodeSegment {
        row_offset: 0,
        rows: n,
        n_valid: n,
    }];
    encode_hidden_segments(gpu, w, state, cfg, std::slice::from_ref(input), &segs, n)
}

/// One sequence's row block inside the concatenated batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeSegment {
    /// First row of this segment in the `[T, hidden]` row block.
    pub row_offset: usize,
    /// Rows it occupies: its own `padded_rows(n)`, so at most one 64-row block
    /// of tail padding beyond its real tokens.
    pub rows: usize,
    /// Real token count — the attention key bound (padding rows are masked)
    /// and the row count the caller gets back.
    pub n_valid: usize,
}

/// Validate the inputs and lay their row blocks out back to back.
fn segment_layout(inputs: &[EncodeInput<'_>]) -> Result<Vec<EncodeSegment>, String> {
    let mut segs = Vec::with_capacity(inputs.len());
    let mut row_offset = 0usize;
    for input in inputs {
        let n = input.ids.len();
        if n == 0 {
            return Err("qwen35-embed: empty input".into());
        }
        if input.pos3.len() != n {
            return Err(format!(
                "qwen35-embed: {} positions for {n} tokens",
                input.pos3.len()
            ));
        }
        let rows = padded_rows(n);
        segs.push(EncodeSegment {
            row_offset,
            rows,
            n_valid: n,
        });
        row_offset += rows;
    }
    Ok(segs)
}

/// Rows the encoder actually runs, after padding the sequence up to a multiple
/// of 64 so the eager attention's PV GEMM meets its `k % 64 == 0` contract.
///
/// Padding is at the END, which the architecture tolerates: GDN is causal, so
/// the padded rows cannot influence a real row's state, and the full-attention
/// layers mask the padded keys in the softmax. The padded rows' own outputs are
/// garbage and are dropped by the caller. Below 256 tokens no padding is taken
/// and the Q-tiled kernel runs instead — its cost is O(n^2), so it is cheap
/// there and it needs no mask.
pub(crate) fn padded_rows(n: usize) -> usize {
    if n < 256 {
        n
    } else {
        n.div_ceil(64) * 64
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_hidden_segments(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    state: &mut GdnEncodeState,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
    segs: &[EncodeSegment],
    n: usize,
) -> Result<GpuTensor, String> {
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let inter = cfg.intermediate_size;

    // ── prologue: host gather + image splice, one upload ──────────────
    // `gather_hidden` FILLS the buffer it is handed (it clears first), so each
    // segment is gathered into its own vector and padded out to its row block
    // before being appended. The segments then land back to back in
    // `row_offset` order, which is what every slice below assumes — gathering
    // straight into `host_hidden` would let each segment overwrite the
    // previous one's rows. Trailing pos3 rows get position 0: their rope phase
    // is never read (their queries are dropped and their keys are masked).
    let mut host_hidden: Vec<f32> = Vec::with_capacity(n * h);
    let mut pos3_host: Vec<[i32; 3]> = Vec::with_capacity(n);
    for (input, seg) in inputs.iter().zip(segs) {
        let mut rows: Vec<f32> = Vec::new();
        w.gather_hidden(input.ids, input.image_embeds, input.spans, h, &mut rows)?;
        rows.resize(seg.rows * h, 0.0);
        host_hidden.extend_from_slice(&rows);
        pos3_host.extend_from_slice(input.pos3);
        pos3_host.resize(seg.row_offset + seg.rows, [0, 0, 0]);
    }
    let mut e = Enc::new(gpu);
    let hidden = e.upload(&host_hidden, n * h)?;

    // mRoPE positions: [n][3] i32 uploaded as raw bits into an F32-typed
    // buffer (the rope kernel's DeviceBuffer contract — see
    // forward_slots.rs' pos3 upload).
    let pos_bytes: Vec<u8> = pos3_host
        .iter()
        .flat_map(|t| t.iter().flat_map(|v| v.to_ne_bytes()))
        .collect();
    let pos3 = e.alloc(n * 3)?;
    e.memcpy_htod(&pos3, &pos_bytes)?;

    trace(&mut e, || "embed".into(), &hidden, n * h)?;
    let mut delta_layer_idx = 0usize;
    for (li, layer) in w.layers.iter().enumerate() {
        // ── pre-norm ──────────────────────────────────────────────────
        let normed = e.alloc(n * h)?;
        e.rmsnorm(&hidden, &layer.input_ln, &normed, n, h, eps)?;

        match &layer.mixer {
            crate::weights::MixerWeights::FullAttn {
                q_proj,
                gate_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
            } => {
                let qd = cfg.q_dim();
                let kvd = cfg.kv_dim();
                let q = e.gemm(q_proj, &normed, qd, h, n)?;
                let gate = e.gemm(gate_proj, &normed, qd, h, n)?;
                let k = e.gemm(k_proj, &normed, kvd, h, n)?;
                let v = e.gemm(v_proj, &normed, kvd, h, n)?;
                e.free(normed)?;

                // Per-head Q/K RMSNorm (pre-rope), in place.
                e.rmsnorm(&q, q_norm, &q, n * cfg.n_heads, cfg.head_dim, eps)?;
                e.rmsnorm(&k, k_norm, &k, n * cfg.n_kv_heads, cfg.head_dim, eps)?;

                // Interleaved mRoPE over the partial rotary dim.
                e.rope_mrope(&q, &k, &pos3, cfg, n)
                    .map_err(|err| format!("qwen35-embed: rope (layer {li}): {err:?}"))?;

                // Bidirectional full attention — the whole point of this
                // architecture. No bias, no causal mask: within a segment the
                // whole sequence attends to the whole sequence. Across
                // segments nothing may be shared, so each one runs its own
                // attention over its own row slice with its own key bound.
                let ctx = e.alloc(n * qd)?;
                for seg in segs {
                    let mut q_s = q.sub_offset(seg.row_offset * qd, seg.rows * qd);
                    let k_s = k.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
                    let v_s = v.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
                    let c_s = ctx.sub_offset(seg.row_offset * qd, seg.rows * qd);
                    e.attention(&mut q_s, &k_s, &v_s, &c_s, seg.rows, seg.n_valid, cfg)
                        .map_err(|err| format!("qwen35-embed: attention (layer {li}): {err:?}"))?;
                }
                e.free(q)?;
                e.free(k)?;
                e.free(v)?;

                // Gated output: ctx *= σ(gate), then o_proj.
                e.sigmoid_mul(&ctx, &gate)
                    .map_err(|err| format!("qwen35-embed: attn gate (layer {li}): {err:?}"))?;
                e.free(gate)?;
                let attn = e.gemm(o_proj, &ctx, h, qd, n)?;
                e.free(ctx)?;
                e.add_inplace(&hidden, &attn, "residual")?;
                e.free(attn)?;
            }
            crate::weights::MixerWeights::Gdn {
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
                let k_dim = cfg.gdn_key_dim();
                let v_dim = cfg.gdn_value_dim();
                let n_v = cfg.linear_num_value_heads;
                let khd = cfg.linear_key_head_dim;
                let vhd = cfg.linear_value_head_dim;

                let qkv_t = e.gemm(qkv, &normed, k_dim * 2 + v_dim, h, n)?;
                let z_t = e.gemm(z, &normed, v_dim, h, n)?;
                let a_t = e.gemm(a, &normed, n_v, h, n)?;
                let b_t = e.gemm(b, &normed, n_v, h, n)?;
                e.free(normed)?;

                // β = σ(b); g = −exp(A_log)·softplus(a + dt_bias), in place.
                e.alpha_gate(&b_t, &a_t, dt_bias, a_log, n_v, n)
                    .map_err(|err| format!("qwen35-embed: alpha gate (layer {li}): {err:?}"))?;
                trace(
                    &mut e,
                    || format!("L{li} gdn: a_t(after gate)"),
                    &a_t,
                    n * n_v,
                )?;
                trace(
                    &mut e,
                    || format!("L{li} gdn: qkv_t"),
                    &qkv_t,
                    n * (k_dim * 2 + v_dim),
                )?;

                // Causal conv + SiLU split (Q|K|V channels), ring state. The
                // ring carries state along rows, so it runs per segment with
                // the ring zeroed at each segment's start — a batch is several
                // independent sequences and each starts from zero recurrence,
                // exactly as it would alone.
                let ch = k_dim * 2 + v_dim;
                let q_raw = e.alloc(n * k_dim)?;
                let k_raw = e.alloc(n * k_dim)?;
                let v_t2 = e.alloc(n * v_dim)?;
                for seg in segs {
                    state.reset_conv(&mut *e.gpu, delta_layer_idx)?;
                    let qkv_s = qkv_t.sub_offset(seg.row_offset * ch, seg.rows * ch);
                    let qr = q_raw.sub_offset(seg.row_offset * k_dim, seg.rows * k_dim);
                    let kr = k_raw.sub_offset(seg.row_offset * k_dim, seg.rows * k_dim);
                    let vt = v_t2.sub_offset(seg.row_offset * v_dim, seg.rows * v_dim);
                    e.conv_split(
                        &qr,
                        &kr,
                        &vt,
                        &qkv_s,
                        conv,
                        &state.conv_states[delta_layer_idx],
                        k_dim,
                        v_dim,
                        seg.rows,
                    )
                    .map_err(|err| format!("qwen35-embed: conv (layer {li}): {err:?}"))?;
                }
                e.free(qkv_t)?;

                // L2-norm(Q)·scale + L2-norm(K) per head. Row-independent, so
                // one call over the whole row block.
                trace(
                    &mut e,
                    || format!("L{li} gdn: conv q_raw"),
                    &q_raw,
                    n * k_dim,
                )?;
                e.qk_l2_norm(&q_raw, &k_raw, cfg, khd, eps, n)
                    .map_err(|err| format!("qwen35-embed: qk norm (layer {li}): {err:?}"))?;

                // FP32 delta-rule scan over the sequence — state-carrying like
                // the conv, so per segment and over the same slices. Key heads
                // equal value heads (16/16) on this family, so no GQA
                // repeat-interleave is needed.
                let attn_out = e.alloc(n * v_dim)?;
                for seg in segs {
                    // S is zeroed here, not with the conv above: this is a
                    // second pass over the segments (the L2 norm in between is
                    // row-independent), so the matrix still holds the state of
                    // the segment scanned last if it is not cleared first.
                    state.reset_s(&mut *e.gpu, delta_layer_idx)?;
                    let qr = q_raw.sub_offset(seg.row_offset * k_dim, seg.rows * k_dim);
                    let kr = k_raw.sub_offset(seg.row_offset * k_dim, seg.rows * k_dim);
                    let vt = v_t2.sub_offset(seg.row_offset * v_dim, seg.rows * v_dim);
                    let a_s = a_t.sub_offset(seg.row_offset * n_v, seg.rows * n_v);
                    let b_s = b_t.sub_offset(seg.row_offset * n_v, seg.rows * n_v);
                    let ao = attn_out.sub_offset(seg.row_offset * v_dim, seg.rows * v_dim);
                    e.gdn_scan(
                        &qr,
                        &kr,
                        &vt,
                        &a_s,
                        &b_s,
                        &state.s_matrices[delta_layer_idx],
                        &ao,
                        seg.rows,
                        n_v,
                        vhd,
                    )
                    .map_err(|err| format!("qwen35-embed: gdn scan (layer {li}): {err:?}"))?;
                }
                trace(
                    &mut e,
                    || format!("L{li} gdn: scan out"),
                    &attn_out,
                    n * v_dim,
                )?;
                e.free(q_raw)?;
                e.free(k_raw)?;
                e.free(v_t2)?;
                e.free(a_t)?;
                e.free(b_t)?;

                // rmsnorm(x)·w·silu(z) per value head, then out_proj.
                let normed_gdn = e.alloc(n * v_dim)?;
                e.gated_norm(&attn_out, &z_t, gdn_norm, &normed_gdn, n_v, vhd, eps, n)
                    .map_err(|err| format!("qwen35-embed: gated norm (layer {li}): {err:?}"))?;
                trace(
                    &mut e,
                    || format!("L{li} gdn: gated_norm"),
                    &normed_gdn,
                    n * v_dim,
                )?;
                e.free(attn_out)?;
                e.free(z_t)?;
                let out = e.gemm(out_proj, &normed_gdn, h, v_dim, n)?;
                e.free(normed_gdn)?;
                e.add_inplace(&hidden, &out, "residual")?;
                e.free(out)?;
                delta_layer_idx += 1;
            }
        }

        trace(&mut e, || format!("layer {li} post-mixer"), &hidden, n * h)?;
        // ── SwiGLU MLP (both mixer kinds) ─────────────────────────────
        let normed2 = e.alloc(n * h)?;
        e.rmsnorm(&hidden, &layer.post_ln, &normed2, n, h, eps)?;
        let g = e.gemm(&layer.gate_proj, &normed2, inter, h, n)?;
        let u = e.gemm(&layer.up_proj, &normed2, inter, h, n)?;
        e.free(normed2)?;
        e.silu_mul(&g, &u)
            .map_err(|err| format!("qwen35-embed: silu (layer {li}): {err:?}"))?;
        e.free(u)?;
        let d = e.gemm(&layer.down_proj, &g, h, inter, n)?;
        e.free(g)?;
        e.add_inplace(&hidden, &d, "residual mlp")
            .map_err(|err| format!("qwen35-embed: residual mlp (layer {li}): {err:?}"))?;
        e.free(d)?;
    }

    e.free(pos3)?;
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    // GPU forwards are exercised end-to-end by the daemon `embed` verb and the
    // `batch_parity` example (lab feature); the host-side math this module
    // shares (masks, normalization, scoring) is covered in crate::post. What is
    // testable without a device is the segment layout every slice in the
    // batched forward is derived from — an off-by-one there is a silent
    // cross-sequence read, not a crash.
    use super::{
        attention_route, padded_rows, projection_gfx1201_tile, segment_layout, AttentionRoute,
        EncodeInput, LdsTile, LdsTileSplitK,
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

    fn input(len: usize) -> (Vec<u32>, Vec<[i32; 3]>) {
        (vec![0; len], (0..len as i32).map(|i| [i, i, i]).collect())
    }

    #[test]
    fn padded_rows_skips_padding_below_the_eager_threshold() {
        // Below 256 the Q-tiled kernel runs and takes any length; at and above
        // it the eager PV GEMM needs a multiple of 64 in its contraction.
        assert_eq!(padded_rows(0), 0);
        assert_eq!(padded_rows(1), 1);
        assert_eq!(padded_rows(255), 255);
        assert_eq!(padded_rows(256), 256);
        assert_eq!(padded_rows(257), 320);
        assert_eq!(padded_rows(302), 320);
        assert_eq!(padded_rows(320), 320);
        assert_eq!(padded_rows(321), 384);
    }

    #[test]
    fn segments_tile_the_row_block_in_order_with_their_own_padding() {
        let owned: Vec<_> = [7usize, 302, 10, 258].iter().map(|n| input(*n)).collect();
        let inputs: Vec<EncodeInput<'_>> = owned
            .iter()
            .map(|(ids, pos3)| EncodeInput {
                ids,
                image_embeds: &[],
                spans: &[],
                pos3,
            })
            .collect();
        let segs = segment_layout(&inputs).expect("layout");

        assert_eq!(segs.len(), 4);
        // Contiguous, in input order, each padded to its own multiple of 64 —
        // never to a common length, which is what keeps a batch from paying for
        // the longest sequence in it.
        assert_eq!(
            segs.iter().map(|s| s.row_offset).collect::<Vec<_>>(),
            vec![0, 7, 327, 337]
        );
        assert_eq!(
            segs.iter().map(|s| s.rows).collect::<Vec<_>>(),
            vec![7, 320, 10, 320]
        );
        assert_eq!(
            segs.iter().map(|s| s.n_valid).collect::<Vec<_>>(),
            vec![7, 302, 10, 258]
        );
        // The last segment's block end is the batch's row count.
        let last = segs.last().expect("non-empty");
        assert_eq!(last.row_offset + last.rows, 657);
    }

    #[test]
    fn an_empty_or_mismatched_input_is_refused() {
        let empty = input(0);
        let empty_in = EncodeInput {
            ids: &empty.0,
            image_embeds: &[],
            spans: &[],
            pos3: &empty.1,
        };
        assert!(segment_layout(&[empty_in]).is_err());

        let four = input(4);
        let pos3: Vec<[i32; 3]> = four.1[..2].to_vec();
        let mismatched = EncodeInput {
            ids: &four.0,
            image_embeds: &[],
            spans: &[],
            pos3: &pos3,
        };
        let err = segment_layout(&[mismatched]).expect_err("refused");
        assert!(err.contains("2 positions for 4 tokens"), "{err}");
    }
}
