// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The encoder forward: tokens → `[n, 768]` per-token projected embeddings
//! (pooling/normalization are host-side in [`crate::post`]).
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
//! The layers are the typed step program in [`crate::program`]; this module
//! stages host input, sequences the programs and drops the padding rows.
//!
//! Sliding window: a sliding layer attends to `|i-j| <= 512`. For
//! `n <= window+1` that is vacuous — no mask anywhere. Longer sequences run
//! the eager-attention route (they are ≥ 256 rows, so `padded_rows` has
//! already made `n % 64 == 0`), which takes an additive band mask before the
//! softmax; a route without one refuses rather than mis-masks.
//!
//! Weights are f16 GEMM operands, activations and norms f32 — the
//! qwen35-embed numerics contract. [`Enc`] is the media towers' launch
//! wrapper; the text trunk does not use it.

use crate::config::EmbedConfig;
use crate::program::{
    attention_steps, head_steps, mlp_steps, ple_block_steps, ple_steps, EncodeScratch,
};
use crate::weights::EmbedWeights;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::encoder::{padded_rows, EncoderSegment};
use hipfire_dispatch::pipeline::{execute_steps, Step};
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

/// Validate the inputs and lay their row blocks out back to back, each its
/// own `padded_rows(n)` with the real token count as the attention key bound.
fn segment_layout(inputs: &[EncodeInput<'_>]) -> Result<Vec<EncoderSegment>, String> {
    let mut segs = Vec::with_capacity(inputs.len());
    let mut row_offset = 0usize;
    for input in inputs {
        let n = input.ids.len();
        if n == 0 {
            return Err("embeddinggemma2: empty input".into());
        }
        let rows = padded_rows(n);
        segs.push(EncoderSegment {
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
    let s = encode_trunk(gpu, w, cfg, inputs, &segs, t_rows)?;

    // Final norm + embedding projection, row-independent over the block.
    let all = run(gpu, &head_steps(cfg, w, &s)).and_then(|()| {
        gpu.download_f32(&s.embedding)
            .map_err(|e| format!("embeddinggemma2: download: {e:?}"))
    });
    s.free_gpu(gpu)?;
    let all = all?;

    let dim = cfg.embedding_dim;
    let mut out = Vec::with_capacity(inputs.len());
    for (input, seg) in inputs.iter().zip(&segs) {
        let n = input.ids.len();
        let start = seg.row_offset * dim;
        out.push(all[start..start + n * dim].to_vec());
    }
    Ok(out)
}

/// The trunk only: the residual stream `[T, hidden]` after the last layer, as
/// a device tensor (freed by the caller). Inspection entry point, not the
/// serving one.
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
    let segs = [EncoderSegment {
        row_offset: 0,
        rows: n,
        n_valid: n,
    }];
    encode_trunk(gpu, w, cfg, std::slice::from_ref(input), &segs, n)?.into_hidden(gpu)
}

fn run(gpu: &mut Gpu, steps: &[Step<'_>]) -> Result<(), String> {
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, steps).map_err(|e| format!("embeddinggemma2: {e}"))
}

/// Stage the input rows and positions, then run every layer's program. The
/// returned scratch holds the trunk output in `hidden`; it is freed on error.
fn encode_trunk(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    inputs: &[EncodeInput<'_>],
    segs: &[EncoderSegment],
    n: usize,
) -> Result<EncodeScratch, String> {
    let h = cfg.hidden_size;
    if cfg.ple_dim == 0 {
        return Err("embeddinggemma2: PLE block but no PLE projection".into());
    }

    // Host gather (× embed scale), one upload.
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
    // Positions: one per row, restarting per segment — padding rows continue
    // the count (their queries are dropped and their keys masked, so the
    // value never reaches an output). The rope kernel reads this buffer as
    // i32; the tensor stays F32-typed with the i32 bits memcpy'd in.
    let pos_bytes: Vec<u8> = segs
        .iter()
        .flat_map(|seg| 0..seg.rows as i32)
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    let hidden = gpu
        .upload_f32(&host_hidden, &[n * h])
        .map_err(|e| format!("embeddinggemma2: upload: {e:?}"))?;
    let positions = gpu
        .alloc_tensor(&[n], DType::F32)
        .map_err(|e| format!("embeddinggemma2: alloc: {e:?}"))?;
    gpu.hip
        .memcpy_htod(&positions.buf, &pos_bytes)
        .map_err(|e| format!("embeddinggemma2: positions upload: {e:?}"))?;

    let s = EncodeScratch::new(gpu, cfg, n, hidden, positions)?;
    match run_layers(gpu, w, cfg, segs, &s) {
        Ok(()) => Ok(s),
        Err(e) => {
            s.free_gpu(gpu)?;
            Err(e)
        }
    }
}

fn run_layers(
    gpu: &mut Gpu,
    w: &EmbedWeights,
    cfg: &EmbedConfig,
    segs: &[EncoderSegment],
    s: &EncodeScratch,
) -> Result<(), String> {
    let len = s.hidden.numel();
    let tr = |gpu: &mut Gpu, label: &dyn Fn() -> String| {
        trace(&mut Enc::new(gpu), label, &s.hidden, len)
    };
    tr(gpu, &|| "embed (×√512)".into())?;
    run(gpu, &ple_steps(cfg, w, s)).map_err(|e| format!("{e} (ple)"))?;
    for (li, layer) in w.layers.iter().enumerate() {
        run(gpu, &attention_steps(cfg, li, layer, s, segs))
            .map_err(|e| format!("{e} (layer {li} attention)"))?;
        tr(gpu, &|| format!("layer {li} post-attn"))?;
        run(gpu, &mlp_steps(cfg, layer, s)).map_err(|e| format!("{e} (layer {li} mlp)"))?;
        tr(gpu, &|| format!("layer {li} post-mlp"))?;
        run(gpu, &ple_block_steps(cfg, li, layer, s))
            .map_err(|e| format!("{e} (layer {li} ple)"))?;
        tr(gpu, &|| format!("layer {li} end"))?;
    }
    Ok(())
}

/// `HIPFIRE_EMBED_ATTN=naive` keeps the towers on the reference kernel
/// (see [`Enc::attention`]). Read once.
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

/// The towers' eager decomposition (see [`Enc::attention`]). Layout identical
/// to the qwen35-embed encoder: per-head `sub_offset` views with row pitch, V
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
    use super::{segment_layout, splice_media, EncodeInput, MediaRows};

    fn input(len: usize) -> Vec<u32> {
        vec![0; len]
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
