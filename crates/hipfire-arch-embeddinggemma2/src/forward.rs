// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The GPU encoder forward: tokens → `[n, 768]` per-token projected
//! embeddings (pooling/normalization are host-side in [`crate::post`]).
//!
//! Layer loop mirrors HF `EmbeddingGemma2TextModel`:
//!
//! - PLE (once): `ple_all = rmsnorm((ple_proj · x_scaled) · hidden^-0.5)`
//!   reshaped `[n, layers, ple]`, normed over the slice width.
//! - Per layer: pre-norm → q/k/v GEMMs → per-head Q/K RMSNorm (learned) and
//!   scale-less V RMSNorm → **half-split RoPE over the full head_dim** with
//!   the layer type's θ → **bidirectional attention with scaling 1.0**
//!   (the QK norms replace the usual 1/√d) → o_proj → post-attention norm on
//!   the branch output → residual → pre-FFN norm → gated gelu-tanh MLP →
//!   post-FFN norm on the branch output → residual → PLE block
//!   (`residual + post_norm(proj(gelu_tanh(gate(x)) ⊙ ple_slice))`) →
//!   `× layer_scalar`.
//! - Final norm → `embedding_projection` (768).
//!
//! Sliding window: a sliding layer attends to `|i-j| <= 512`. For
//! `n <= window+1` that is vacuous — no mask anywhere. Longer sequences run
//! the eager-attention path (they are ≥ 256 rows, so `padded_rows` has
//! already made `n % 64 == 0`), where the materialized score matrix takes an
//! additive band mask before the softmax. The small-`n` fallback kernel
//! (`attention_text_f32`, one block per query row) is only reached below 256
//! rows where no window can bite, and it takes the scale explicitly (1.0)
//! — the Q-tiled encoder kernel bakes in 1/√d and cannot serve this arch.
//!
//! Weights are f16 GEMM operands, activations and norms f32 — the
//! qwen35-embed numerics contract.

use crate::config::EmbedConfig;
use crate::weights::EmbedWeights;
use rdna_compute::gemm::LdsTile;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Input assembled by the caller: token ids plus the projected rows a media
/// span replaces.
///
/// `ids` is the EXPANDED sequence (media markers already spliced, see
/// [`crate::carrier::assemble_input`]): every placeholder row is a real row
/// whose embedding is overwritten by `media`. Nothing about positions or
/// RoPE changes — this is HF's `masked_scatter` on `inputs_embeds`, so the
/// media rows participate in attention and pooling like any other row.
pub struct EncodeInput<'a> {
    pub ids: &'a [u32],
    /// One entry per media span, in `AssembledInput.media_spans` order.
    /// Empty for text-only inputs.
    pub media: &'a [MediaRows<'a>],
}

/// The projected encoder rows for one media span.
pub struct MediaRows<'a> {
    /// First row of the span inside `EncodeInput.ids`.
    pub start_row: usize,
    /// `[len, hidden]` row-major — `len` == the span's placeholder count.
    pub rows: &'a [f32],
}

impl EncodeInput<'_> {
    /// A text-only input (no media spans).
    pub fn text(ids: &[u32]) -> EncodeInput<'_> {
        EncodeInput { ids, media: &[] }
    }
}

// ─── LDS tile tables (mirrors the qwen35-embed attention GEMMs) ─────────────

pub fn attention_qk_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 32, 64, 64, false).pipelined(),
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

pub fn attention_pv_tiles() -> [LdsTile; 3] {
    [
        LdsTile::new(128, 128, 64, 64, 64, false),
        LdsTile::new(128, 128, 32, 64, 64, false),
        LdsTile::new(64, 64, 32, 64, 64, false),
    ]
}

fn attention_lds_tile(cands: &[LdsTile], m: usize, batch: usize) -> Option<LdsTile> {
    cands
        .iter()
        .copied()
        .find(|t| t.bm <= m && t.bn <= batch && Gpu::LDS_TILE_VARIANTS.contains(t))
}

/// Small scratch wrapper: alloc/upload/free + the f16 GEMM dispatch (the
/// qwen35-embed `Enc`, minus that arch's GDN hooks). `pub(crate)` so the
/// vision tower ([`crate::vision`]) drives the same GEMM dispatch and
/// attention decomposition rather than a second copy of either.
pub(crate) struct Enc<'a> {
    gpu: &'a mut Gpu,
}

impl<'a> Enc<'a> {
    pub(crate) fn new(gpu: &'a mut Gpu) -> Self {
        Self { gpu }
    }

    /// The raw device handle, for the audio tower's own JIT kernels (clip
    /// clamp, GLU, causal depthwise conv) — the vision tower needs no escape
    /// hatch because every kernel it uses is on the shared dispatch.
    pub(crate) fn gpu_mut(&mut self) -> &mut Gpu {
        self.gpu
    }

    pub(crate) fn alloc(&mut self, n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(&[n], DType::F32)
            .map_err(|e| format!("embeddinggemma2: alloc: {e:?}"))
    }

    pub(crate) fn upload(&mut self, data: &[f32], n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .upload_f32(data, &[n])
            .map_err(|e| format!("embeddinggemma2: upload: {e:?}"))
    }

    pub(crate) fn free(&mut self, t: GpuTensor) -> Result<(), String> {
        self.gpu
            .free_tensor(t)
            .map_err(|e| format!("embeddinggemma2: free: {e:?}"))
    }

    /// `y[n, out] = x[n, k] · w_f16[out, k]ᵀ` — X operand f32 (mb8) or f16
    /// (LDS-tiled), weight always f16. Same dispatch contract as the
    /// qwen35-embed encoder; `HIPFIRE_EMBED_GEMM=f32` keeps f32 activations.
    pub(crate) fn gemm(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        out: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuTensor, String> {
        if k == 0 || k % 16 != 0 {
            return Err(format!(
                "embeddinggemma2: GEMM k={k} is not a multiple of 16"
            ));
        }
        let y = self.alloc(n * out)?;
        let wmma = self.gpu.arch_caps.has_wmma_w32() || self.gpu.arch_caps.has_wmma_w32_gfx12();
        if !wmma {
            let yt = self.alloc(out * n)?;
            self.gpu
                .gemm_f16(w, x, &yt, out, k, n)
                .map_err(|e| format!("embeddinggemma2: gemm (naive): {e:?}"))?;
            self.gpu
                .transpose_f32(&yt, &y, out, n)
                .map_err(|e| format!("embeddinggemma2: transpose: {e:?}"))?;
            self.free(yt)?;
            return Ok(y);
        }
        let use_tiled = k % 64 == 0 && out >= 256 && !embed_gemm_f32();
        if use_tiled {
            let x16 = self.cast_f16(x, n * k)?;
            let r = self
                .gpu
                .gemm_f16_x_f16_wmma_lds_auto(w, &x16, &y, None, out, k, n);
            self.free(x16)?;
            r.map_err(|e| format!("embeddinggemma2: gemm (lds-tiled): {e:?}"))?;
        } else {
            self.gpu
                .gemm_f16_wmma_mb8(w, x, &y, out, k, n)
                .map_err(|e| format!("embeddinggemma2: gemm: {e:?}"))?;
        }
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_ld(
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
        let cands = if qk {
            attention_qk_tiles()
        } else {
            attention_pv_tiles()
        };
        if let Some(tile) = attention_lds_tile(&cands, m, batch) {
            return self
                .gpu
                .gemm_f16_x_f16_wmma_lds_tiled_ld(a, x, y, None, m, k, batch, tile, lda, ldx);
        }
        self.gpu
            .gemm_f16_x_f16_wmma_lds_auto_ld(a, x, y, None, m, k, batch, lda, ldx)
    }

    pub(crate) fn alloc_shaped(
        &mut self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, dtype)
            .map_err(|e| format!("embeddinggemma2: alloc: {e:?}"))
    }

    pub(crate) fn alloc_f16(&mut self, n: usize) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(&[n], DType::F16)
            .map_err(|e| format!("embeddinggemma2: alloc f16: {e:?}"))
    }

    pub(crate) fn cast_f16(&mut self, x: &GpuTensor, n: usize) -> Result<GpuTensor, String> {
        let dst = self
            .gpu
            .alloc_tensor(&[n], DType::F16)
            .map_err(|e| format!("embeddinggemma2: alloc f16: {e:?}"))?;
        self.gpu
            .cast_f32_to_f16(x, &dst)
            .map_err(|e| format!("embeddinggemma2: cast f16: {e:?}"))?;
        Ok(dst)
    }

    pub(crate) fn rmsnorm(
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
            .map_err(|e| format!("embeddinggemma2: rmsnorm: {e:?}"))
    }

    pub(crate) fn download(&mut self, t: &GpuTensor) -> Result<Vec<f32>, String> {
        self.gpu
            .download_f32(t)
            .map_err(|e| format!("embeddinggemma2: download: {e:?}"))
    }

    pub(crate) fn add_inplace(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        ctx: &str,
    ) -> Result<(), String> {
        self.gpu
            .add_inplace_f32(a, b)
            .map_err(|e| format!("embeddinggemma2: {ctx}: {e:?}"))
    }

    // ── kernels the vision tower shares with the trunk path ─────────────

    /// `HIPFIRE_EMBED_DEBUG=1` finite-value trace of the tower's residual
    /// stream (the same knob the trunk path uses).
    pub(crate) fn trace(
        &mut self,
        label: impl FnOnce() -> String,
        t: &GpuTensor,
        n: usize,
    ) -> Result<(), String> {
        trace(self, label, t, n)
    }

    /// `true` when this GPU has the WMMA path the eager attention needs.
    pub(crate) fn has_wmma(&self) -> bool {
        self.gpu.arch_caps.has_wmma_w32() || self.gpu.arch_caps.has_wmma_w32_gfx12()
    }

    /// Bidirectional attention, scale 1.0, keys past `n_valid` masked. The
    /// eager decomposition when it is available at this shape, the
    /// one-block-per-query-row reference kernel otherwise (which is exact
    /// only when `n == n_valid` — pass no padding there).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &mut self,
        q: &mut GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        ctx: &GpuTensor,
        n: usize,
        n_valid: usize,
        heads: usize,
        kv_heads: usize,
        hd: usize,
        qd: usize,
        kvd: usize,
    ) -> Result<(), String> {
        let eager =
            n >= 256 && n % 64 == 0 && hd % 64 == 0 && self.has_wmma() && !embed_attn_naive();
        if eager {
            return attention_gemm(
                self,
                q,
                k,
                v,
                ctx,
                n,
                n_valid,
                heads,
                heads / kv_heads,
                hd,
                qd,
                kvd,
                None,
            );
        }
        if n != n_valid {
            return Err(format!(
                "embeddinggemma2: attention at {n} rows with {n_valid} valid keys has no \
                 masked path here (the padding fallback needs an unmasked row count)"
            ));
        }
        self.gpu
            .attention_text_f32(q, k, v, None, None, ctx, n, heads, kv_heads, hd, 1.0, false)
            .map_err(|e| format!("embeddinggemma2: attention: {e:?}"))
    }

    /// Axial 2-D RoPE over Q and K in place (`crate::vision` builds the
    /// per-patch tables in the permuted head-dim order).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_2d(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        cos: &GpuTensor,
        sin: &GpuTensor,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Result<(), String> {
        self.gpu
            .rope_2d_halfsplit_f32(q, k, cos, sin, rows, heads, kv_heads, hd)
            .map_err(|e| format!("embeddinggemma2: rope_2d: {e:?}"))
    }

    /// `x = gelu_tanh(x)` in place.
    pub(crate) fn gelu_tanh(&mut self, x: &GpuTensor, n: usize) -> Result<(), String> {
        self.gpu
            .gelu_tanh_f32(x, x, n)
            .map_err(|e| format!("embeddinggemma2: gelu: {e:?}"))
    }

    /// `c = a ⊙ b`.
    pub(crate) fn mul(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), String> {
        self.gpu
            .mul_f32(a, b, c)
            .map_err(|e| format!("embeddinggemma2: mul: {e:?}"))
    }
}

/// `HIPFIRE_EMBED_GEMM=f32` escape hatch, read once (same knob as qwen35-embed).
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

/// Env-gated (`HIPFIRE_EMBED_DEBUG=1`) finite-value trace of the residual
/// stream (same debugging contract as the qwen35-embed encoder).
fn trace(
    enc: &mut Enc<'_>,
    label: impl FnOnce() -> String,
    t: &GpuTensor,
    n: usize,
) -> Result<(), String> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_EMBED_DEBUG")
            .ok()
            .as_deref()
            == Some("1")
    }) {
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

/// One sequence's row block inside the concatenated batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeSegment {
    /// First row of this segment in the `[T, hidden]` row block.
    pub row_offset: usize,
    /// Rows it occupies: its own `padded_rows(n)`.
    pub rows: usize,
    /// Real token count — the attention key bound.
    pub n_valid: usize,
}

/// Validate the inputs and lay their row blocks out back to back.
fn segment_layout(inputs: &[EncodeInput<'_>]) -> Result<Vec<EncodeSegment>, String> {
    let mut segs = Vec::with_capacity(inputs.len());
    let mut row_offset = 0usize;
    for input in inputs {
        let n = input.ids.len();
        if n == 0 {
            return Err("embeddinggemma2: empty input".into());
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

/// Write one media span's projected rows over its placeholder rows.
///
/// `rows` is the already-gathered, already-scaled `[seg.rows, hidden]` block;
/// `n` is the input's real row count. The span must fit inside the real rows
/// and carry exactly `len × hidden` finite floats.
fn splice_media(
    rows: &mut [f32],
    media: &MediaRows<'_>,
    hidden: usize,
    n: usize,
) -> Result<(), String> {
    if hidden == 0 || media.rows.len() % hidden != 0 {
        return Err(format!(
            "embeddinggemma2: media rows are {} floats, not a whole number of {hidden}-wide \
             hidden rows",
            media.rows.len()
        ));
    }
    let len = media.rows.len() / hidden;
    if len == 0 {
        return Err("embeddinggemma2: media span carries no rows".into());
    }
    if media.start_row + len > n {
        return Err(format!(
            "embeddinggemma2: media span [{}, {}) runs past the input's {n} rows",
            media.start_row,
            media.start_row + len
        ));
    }
    if rows.len() < (media.start_row + len) * hidden {
        return Err(format!(
            "embeddinggemma2: media span [{}, {}) does not fit the gathered block",
            media.start_row,
            media.start_row + len
        ));
    }
    if let Some(bad) = media.rows.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "embeddinggemma2: media rows contain a non-finite value at element {bad}"
        ));
    }
    let start = media.start_row * hidden;
    rows[start..start + media.rows.len()].copy_from_slice(media.rows);
    Ok(())
}

/// Rows the encoder actually runs: ≥ 256 sequences are padded up to a
/// multiple of 64 so the eager attention's PV GEMM meets its `k % 64`
/// contract. Padding is at the END and every real query row masks the
/// padded keys in the softmax (`n_valid` bound); the padded rows' own
/// outputs are garbage and dropped by the caller.
pub(crate) fn padded_rows(n: usize) -> usize {
    if n < 256 {
        n
    } else {
        n.div_ceil(64) * 64
    }
}

/// Run the full encoder and the 768-d projection; returns `[n, embedding_dim]`
/// row-major per-token projected embeddings, unpool­ed and unnormalized
/// (mean pooling + L2 run host-side in [`crate::bundle`]).
pub fn encode_embeddings(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<Vec<f32>, String> {
    let mut out = encode_embeddings_batch(gpu, w, cfg, std::slice::from_ref(input))?;
    Ok(out.pop().expect("one input in, one embedding out"))
}

/// Several sequences in one forward, one `[n_i, embedding_dim]` row-major
/// block per input. Row-concatenation batching, exactly the qwen35-embed
/// scheme: nothing in this architecture couples rows across sequences
/// except attention, which runs per segment over its own row slice with its
/// own key bound (and its own position restart for RoPE).
pub fn encode_embeddings_batch(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
) -> Result<Vec<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Err("embeddinggemma2: empty batch".into());
    }
    let segs = segment_layout(inputs)?;
    let t_rows: usize = segs.iter().map(|s| s.rows).sum();
    let hidden = encode_hidden_segments(gpu, w, cfg, inputs, &segs, t_rows)?;

    // Final norm + embedding projection, row-independent over the block.
    let dim = cfg.embedding_dim;
    let mut e = Enc::new(gpu);
    let projected = {
        let fin = e.alloc(t_rows * cfg.hidden_size)?;
        e.rmsnorm(
            &hidden,
            &w.final_norm,
            &fin,
            t_rows,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let emb = e.gemm(&w.embedding_projection, &fin, dim, cfg.hidden_size, t_rows)?;
        e.free(fin)?;
        emb
    };
    let all = e.download(&projected)?;
    e.free(projected)?;
    e.free(hidden)?;

    let mut out = Vec::with_capacity(inputs.len());
    for (input, seg) in inputs.iter().zip(&segs) {
        let n = input.ids.len();
        let start = seg.row_offset * dim;
        out.push(all[start..start + n * dim].to_vec());
    }
    Ok(out)
}

/// The trunk only: final-normed hidden states `[T, hidden]` as a device
/// tensor (freed by the caller). Inspection entry point, not the serving one.
pub fn encode_hidden(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    input: &EncodeInput<'_>,
) -> Result<GpuTensor, String> {
    let n = input.ids.len();
    if n == 0 {
        return Err("embeddinggemma2: empty input".into());
    }
    let segs = [EncodeSegment {
        row_offset: 0,
        rows: n,
        n_valid: n,
    }];
    encode_hidden_segments(gpu, w, cfg, std::slice::from_ref(input), &segs, n)
}

#[allow(clippy::too_many_arguments)]
fn encode_hidden_segments(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
    segs: &[EncodeSegment],
    n: usize,
) -> Result<GpuTensor, String> {
    let h = cfg.hidden_size;
    let eps = cfg.rms_norm_eps;
    let inter = cfg.intermediate_size;

    // ── prologue: host gather (× embed scale), one upload ──────────────
    let scale = cfg.embed_scale();
    let mut host_hidden: Vec<f32> = Vec::with_capacity(n * h);
    for (input, seg) in inputs.iter().zip(segs) {
        let mut rows: Vec<f32> = Vec::new();
        w.gather_hidden(input.ids, h, &mut rows)?;
        rows.resize(seg.rows * h, 0.0);
        for v in rows.iter_mut() {
            *v *= scale;
        }
        // Media rows are written over the placeholder rows AFTER the
        // embedding scale: HF replaces already-scaled embedding rows with
        // the projectors' output, which is not scaled (masked_scatter on
        // `inputs_embeds`).
        for media in input.media {
            splice_media(&mut rows, media, h, input.ids.len())?;
        }
        host_hidden.extend_from_slice(&rows);
    }
    let mut e = Enc::new(gpu);
    let hidden = e.upload(&host_hidden, n * h)?;
    trace(&mut e, || "embed (×√512)".into(), &hidden, n * h)?;

    // Positions: one per row, restarting per segment — padding rows continue
    // the count (their queries are dropped and their keys masked, so the
    // value never reaches an output). The rope kernel reads this buffer as
    // i32 (see gemma4's forward_batch positions upload); the tensor stays
    // F32-typed with the i32 bits memcpy'd in, the same DeviceBuffer
    // contract the arch-17 encoder's pos3 upload uses.
    let mut pos_host: Vec<i32> = Vec::with_capacity(n);
    for seg in segs {
        for p in 0..seg.rows {
            pos_host.push(p as i32);
        }
    }
    let pos_bytes: Vec<u8> = pos_host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let positions = e.alloc(n)?;
    e.gpu
        .hip
        .memcpy_htod(&positions.buf, &pos_bytes)
        .map_err(|err| format!("embeddinggemma2: positions upload: {err:?}"))?;

    // ── PLE: one projection, normed slices for every layer ────────────
    // ple_all [n, layers*ple]: gemm → ×hidden^-0.5 → RMSNorm over each
    // [layers, ple] slice row (batched over n*layers rows of width ple).
    let ple_all = if cfg.ple_dim > 0 {
        let raw = e.gemm(
            &w.ple_projection,
            &hidden,
            cfg.num_layers * cfg.ple_dim,
            h,
            n,
        )?;
        e.gpu
            .scale_f32(&raw, cfg.ple_scale())
            .map_err(|err| format!("embeddinggemma2: ple scale: {err:?}"))?;
        e.rmsnorm(
            &raw,
            &w.ple_norm,
            &raw,
            n * cfg.num_layers,
            cfg.ple_dim,
            eps,
        )?;
        Some(raw)
    } else {
        None
    };

    for (li, layer) in w.layers.iter().enumerate() {
        let attn_cfg = cfg.attn(li);
        let hd = attn_cfg.head_dim;
        let heads = cfg.n_heads;
        let kv_heads = attn_cfg.n_kv_heads;
        let rep = heads / kv_heads;
        let qd = cfg.q_dim(li);
        let kvd = cfg.kv_dim(li);

        // ── attention block ────────────────────────────────────────────
        let normed = e.alloc(n * h)?;
        e.rmsnorm(&hidden, &layer.input_ln, &normed, n, h, eps)?;
        let q = e.gemm(&layer.q_proj, &normed, qd, h, n)?;
        let k = e.gemm(&layer.k_proj, &normed, kvd, h, n)?;
        let v = e.gemm(&layer.v_proj, &normed, kvd, h, n)?;
        e.free(normed)?;

        // Per-head Q/K norm (learned weight) and scale-less V norm, in place.
        e.rmsnorm(&q, &layer.q_norm, &q, n * heads, hd, eps)?;
        e.rmsnorm(&k, &layer.k_norm, &k, n * kv_heads, hd, eps)?;
        e.rmsnorm(&v, &layer.v_ones, &v, n * kv_heads, hd, eps)?;

        // Half-split RoPE over the full head_dim, this layer type's θ.
        e.gpu
            .rope_partial_halved_f32_batched(
                &q,
                &k,
                &positions,
                heads,
                kv_heads,
                hd,
                hd / 2,
                cfg.rope_theta(li),
                n,
            )
            .map_err(|err| format!("embeddinggemma2: rope (layer {li}): {err:?}"))?;

        let ctx = e.alloc(n * qd)?;
        for seg in segs {
            let mut q_s = q.sub_offset(seg.row_offset * qd, seg.rows * qd);
            let k_s = k.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
            let v_s = v.sub_offset(seg.row_offset * kvd, seg.rows * kvd);
            let c_s = ctx.sub_offset(seg.row_offset * qd, seg.rows * qd);
            attention(
                &mut e,
                &mut q_s,
                &k_s,
                &v_s,
                &c_s,
                seg.rows,
                seg.n_valid,
                heads,
                kv_heads,
                rep,
                hd,
                qd,
                kvd,
                if cfg.window_bites(li, seg.rows) {
                    Some(cfg.sliding_window)
                } else {
                    None
                },
            )
            .map_err(|err| format!("embeddinggemma2: attention (layer {li}): {err:?}"))?;
        }
        e.free(q)?;
        e.free(k)?;
        e.free(v)?;

        let o = e.gemm(&layer.o_proj, &ctx, h, qd, n)?;
        e.free(ctx)?;
        // Sandwich norm on the branch output, then residual.
        e.rmsnorm(&o, &layer.post_attn_ln, &o, n, h, eps)?;
        e.add_inplace(&hidden, &o, "residual attn")?;
        e.free(o)?;
        trace(
            &mut e,
            || format!("layer {li} post-attn").into(),
            &hidden,
            n * h,
        )?;

        // ── gated gelu-tanh MLP ────────────────────────────────────────
        let normed2 = e.alloc(n * h)?;
        e.rmsnorm(&hidden, &layer.pre_ff_ln, &normed2, n, h, eps)?;
        let g = e.gemm(&layer.gate_proj, &normed2, inter, h, n)?;
        let u = e.gemm(&layer.up_proj, &normed2, inter, h, n)?;
        e.free(normed2)?;
        e.gpu
            .gelu_tanh_f32(&g, &g, n * inter)
            .map_err(|err| format!("embeddinggemma2: gelu (layer {li}): {err:?}"))?;
        e.gpu
            .mul_f32(&g, &u, &g)
            .map_err(|err| format!("embeddinggemma2: gelu·up (layer {li}): {err:?}"))?;
        e.free(u)?;
        let d = e.gemm(&layer.down_proj, &g, h, inter, n)?;
        e.free(g)?;
        e.rmsnorm(&d, &layer.post_ff_ln, &d, n, h, eps)?;
        e.add_inplace(&hidden, &d, "residual mlp")?;
        e.free(d)?;
        trace(
            &mut e,
            || format!("layer {li} post-mlp").into(),
            &hidden,
            n * h,
        )?;

        // ── PLE block ──────────────────────────────────────────────────
        let ple_all = ple_all
            .as_ref()
            .ok_or_else(|| "embeddinggemma2: PLE block but no PLE projection".to_string())?;
        let t = e.gemm(&layer.ple_gate, &hidden, h, h, n)?;
        // t = gelu_tanh(t) ⊙ ple_all[:, li] (the gemma4 fused strided kernel:
        // identical op — gelu-tanh of the gate times the layer's packed PLE
        // slice; here ple_dim == hidden == 512).
        e.gpu
            .gemma4_ple_gelu_mul_strided_f32(
                &t,
                ple_all,
                &t,
                n,
                cfg.ple_dim,
                cfg.num_layers * cfg.ple_dim,
                li,
            )
            .map_err(|err| format!("embeddinggemma2: ple gate·mul (layer {li}): {err:?}"))?;
        let p = e.gemm(&layer.ple_proj, &t, h, cfg.ple_dim, n)?;
        e.free(t)?;
        e.rmsnorm(&p, &layer.ple_post_norm, &p, n, h, eps)?;
        e.add_inplace(&hidden, &p, "residual ple")?;
        e.free(p)?;

        // ── layer_scalar on the whole residual stream ──────────────────
        e.gpu
            .scale_f32(&hidden, layer.layer_scalar)
            .map_err(|err| format!("embeddinggemma2: layer_scalar (layer {li}): {err:?}"))?;
        trace(&mut e, || format!("layer {li} end").into(), &hidden, n * h)?;
    }

    if let Some(p) = ple_all {
        e.free(p)?;
    }
    e.free(positions)?;
    Ok(hidden)
}

/// One layer's bidirectional attention over one segment.
///
/// Eager path (WMMA arch, `n >= 256` — `padded_rows` guarantees `n % 64 == 0`
/// there — and `hd % 64 == 0`): `S = Q·Kᵀ` per head → additive band mask on
/// sliding layers whose window actually restricts at this length → row
/// softmax over `[0, n_valid)` with a direct f16 store → `O = P·V`, scatter
/// back per head. **No 1/√d**: this family scales attention by 1.0.
///
/// Fallback (`attention_text_f32`, one block per query row): any `n < 256`,
/// non-WMMA arches, or `HIPFIRE_EMBED_ATTN=naive`. It takes the scale
/// explicitly and needs no window there: below 256 rows a 512-radius window
/// can never bite (a config with a window smaller than 255 whose sequence
/// outruns it on the fallback is refused instead of mis-masked).
#[allow(clippy::too_many_arguments)]
fn attention(
    e: &mut Enc<'_>,
    q: &mut GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    ctx: &GpuTensor,
    n: usize,
    n_valid: usize,
    heads: usize,
    kv_heads: usize,
    rep: usize,
    hd: usize,
    qd: usize,
    kvd: usize,
    window: Option<usize>,
) -> Result<(), String> {
    if let Some(w) = window {
        // The eager path is masked below; every other path must refuse a
        // window that actually restricts at this length.
        let eager = n >= 256
            && n % 64 == 0
            && hd % 64 == 0
            && (e.gpu.arch_caps.has_wmma_w32() || e.gpu.arch_caps.has_wmma_w32_gfx12())
            && !embed_attn_naive();
        if !eager {
            return Err(format!(
                "embeddinggemma2: sliding window {w} restricts at {n} rows but this \
                 path (small-n / non-WMMA) has no band mask — refusing rather than \
                 mis-masking"
            ));
        }
    }
    let wmma = e.gpu.arch_caps.has_wmma_w32() || e.gpu.arch_caps.has_wmma_w32_gfx12();
    if !embed_attn_naive() && n >= 256 && n % 64 == 0 && hd % 64 == 0 && wmma {
        return attention_gemm(e, q, k, v, ctx, n, n_valid, heads, rep, hd, qd, kvd, window);
    }
    let r = e.gpu.attention_text_f32(
        q, k, v, None, None, ctx, n, heads, kv_heads, hd,
        1.0, // this family scales attention by 1.0 (QK norms replace 1/√d)
        false,
    );
    r.map_err(|err| format!("attention_text_f32: {err:?}"))
}

/// `HIPFIRE_EMBED_ATTN=naive` keeps the reference kernel everywhere it is
/// correct (it has no band mask — see [`attention`]). Read once.
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

/// The eager decomposition (see [`attention`]). Layout identical to the
/// qwen35-embed encoder: per-head `sub_offset` views with row pitch, V
/// transposed once per head for the second GEMM's A operand.
#[allow(clippy::too_many_arguments)]
fn attention_gemm(
    e: &mut Enc<'_>,
    q: &mut GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    ctx: &GpuTensor,
    n: usize,
    n_valid: usize,
    heads: usize,
    rep: usize,
    hd: usize,
    qd: usize,
    kvd: usize,
    window: Option<usize>,
) -> Result<(), String> {
    let q16 = e.cast_f16(q, n * qd)?;
    let k16 = e.cast_f16(k, n * kvd)?;
    let s = e.alloc_shaped(&[n, n], DType::F32)?;
    let p16 = e.alloc_f16(n * n)?;
    let vh = e.alloc(n * hd)?;
    let vht = e.alloc(hd * n)?;
    let vht16 = e.alloc_f16(hd * n)?;
    let o = e.alloc(n * hd)?;

    let mut result = Ok(());
    for h in 0..heads {
        let kh = h / rep;
        let q_h = q16.sub_offset(h * hd, (n - 1) * qd + hd);
        let k_h = k16.sub_offset(kh * hd, (n - 1) * kvd + hd);
        let step = (|| -> Result<(), String> {
            // S = Q_h · K_hᵀ   ([n, hd] · [hd, n] -> [n, n]) — no 1/√d fold.
            e.gemm_ld(&k_h, &q_h, &s, n, hd, n, kvd, qd, true)
                .map_err(|err| format!("attn QK^T (head {h}): {err:?}"))?;
            if let Some(w) = window {
                e.gpu
                    .attention_scores_band_mask_f32(&s, n, n, w)
                    .map_err(|err| format!("attn band mask (head {h}): {err:?}"))?;
            }
            // Softmax over [0, n_valid): padding keys contribute nothing.
            e.gpu
                .attention_rows_softmax_f16(&s, &p16, n, n, 0, n_valid)
                .map_err(|err| format!("attn softmax (head {h}): {err:?}"))?;
            let v_h = v.sub_offset(kh * hd, (n - 1) * kvd + hd);
            e.gpu
                .copy_rows_strided_f32(&v_h, &vh, n, hd, kvd, hd, 0, None)
                .map_err(|err| format!("attn V gather (head {h}): {err:?}"))?;
            e.gpu
                .transpose_f32(&vh, &vht, n, hd)
                .map_err(|err| format!("attn V transpose (head {h}): {err:?}"))?;
            e.gpu
                .cast_f32_to_f16(&vht, &vht16)
                .map_err(|err| format!("attn V cast (head {h}): {err:?}"))?;
            // O_h = P · V_h
            e.gemm_ld(&vht16, &p16, &o, hd, n, n, n, n, false)
                .map_err(|err| format!("attn PV (head {h}): {err:?}"))?;
            e.gpu
                .copy_rows_strided_f32(&o, ctx, n, hd, hd, qd, h * hd, None)
                .map_err(|err| format!("attn O scatter (head {h}): {err:?}"))?;
            Ok(())
        })();
        if step.is_err() {
            result = step;
            break;
        }
    }

    for t in [q16, k16, s, p16, vh, vht, vht16, o] {
        e.free(t)?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{padded_rows, segment_layout, splice_media, EncodeInput, MediaRows};

    fn input(len: usize) -> Vec<u32> {
        vec![0; len]
    }

    #[test]
    fn padded_rows_skips_padding_below_the_eager_threshold() {
        assert_eq!(padded_rows(0), 0);
        assert_eq!(padded_rows(1), 1);
        assert_eq!(padded_rows(255), 255);
        assert_eq!(padded_rows(256), 256);
        assert_eq!(padded_rows(257), 320);
        assert_eq!(padded_rows(302), 320);
        assert_eq!(padded_rows(321), 384);
    }

    #[test]
    fn segments_tile_the_row_block_in_order() {
        let owned = [input(7), input(302), input(10), input(258)];
        let inputs: Vec<EncodeInput<'_>> = owned.iter().map(|ids| EncodeInput::text(ids)).collect();
        let segs = segment_layout(&inputs).expect("layout");
        assert_eq!(segs.len(), 4);
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
    }

    #[test]
    fn an_empty_input_is_refused() {
        let empty = input(0);
        let inp = EncodeInput::text(&empty);
        assert!(segment_layout(&[inp]).is_err());
    }

    #[test]
    fn a_media_span_overwrites_its_placeholder_rows() {
        // 4 rows of 2, span = rows 1..3 → those two rows are replaced, the
        // others are untouched.
        let mut rows = vec![9.0f32; 4 * 2];
        let media = MediaRows {
            start_row: 1,
            rows: &[1.0, 2.0, 3.0, 4.0],
        };
        splice_media(&mut rows, &media, 2, 4).expect("splice");
        assert_eq!(rows, vec![9.0, 9.0, 1.0, 2.0, 3.0, 4.0, 9.0, 9.0]);
    }

    #[test]
    fn a_media_span_is_validated_before_it_is_written() {
        let mut rows = vec![0.0f32; 4 * 2];
        // Ragged row count.
        let ragged = MediaRows {
            start_row: 0,
            rows: &[1.0, 2.0, 3.0],
        };
        assert!(splice_media(&mut rows, &ragged, 2, 4)
            .unwrap_err()
            .contains("whole number"));
        // Past the end of the real rows.
        let over = MediaRows {
            start_row: 3,
            rows: &[1.0, 2.0, 3.0, 4.0],
        };
        assert!(splice_media(&mut rows, &over, 2, 4)
            .unwrap_err()
            .contains("past the input"));
        // Non-finite projected value.
        let nan = MediaRows {
            start_row: 0,
            rows: &[1.0, f32::NAN],
        };
        assert!(splice_media(&mut rows, &nan, 2, 4)
            .unwrap_err()
            .contains("non-finite"));
        // Empty span.
        let empty = MediaRows {
            start_row: 0,
            rows: &[],
        };
        assert!(splice_media(&mut rows, &empty, 2, 4)
            .unwrap_err()
            .contains("no rows"));
    }
}
