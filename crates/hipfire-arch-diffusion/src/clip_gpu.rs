// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU CLIP-L text encoder — a layer-for-layer mirror of the CPU reference in
//! [`crate::clip`], so the two files diff side by side.
//!
//! CLIP is the cheap half of FLUX conditioning (12 layers, `d` 768, 77 tokens
//! — 0.24 s on the host against T5's 54.9 s), so this exists for symmetry and
//! for the conditioning cache, not because it was the bottleneck. It shares
//! the [`crate::t5_gpu::TextGpu`] scratch helper and every kernel with
//! [`crate::t5_gpu`]; the differences from T5 are exactly the four the model
//! architectures differ by:
//!
//! | | T5-XXL | CLIP-L |
//! |---|---|---|
//! | norm | RMSNorm, scale only | LayerNorm, affine (gamma+beta) |
//! | linears | bias-free | every linear carries a bias |
//! | attention | additive relative bias, `scale = 1` | causal mask, `scale = 1/sqrt(hd)` |
//! | MLP | gated GELU-tanh | quick-GELU (`x·sigmoid(1.702x)`) |
//!
//! Embeddings stay on the host, as in [`crate::t5_gpu`]: token + learned
//! position embedding is a 77-row gather out of a `vocab × 768` table.
//!
//! Masking is CAUSAL ONLY. The diffusers FLUX pipeline calls
//! `CLIPTextModel(input_ids)` with no `attention_mask`, so padded positions
//! attend freely — see [`crate::clip::encode`] for the full note. Adding a
//! key-padding mask here diverges from the golden at every padded row.
//!
//! Pooling: `pooled = last_hidden_state[argmax(input_ids)]`, taken AFTER the
//! final LayerNorm, exactly as the CPU reference does (transformers 5 EOT
//! semantics, and there is no `text_projection` in this model class).

use crate::clip::{ClipConfig, ClipWeights};
use crate::flux::Tensor;
use crate::flux_gpu::upload_flux_tensor;
use crate::t5_gpu::TextGpu;
use rdna_compute::{Gpu, GpuTensor};

/// GPU-resident CLIP text-encoder weights. Linear `.weight`s are f16 (the
/// WMMA GEMM's operand); biases and LayerNorm gamma/beta stay f32.
///
/// Token/position embeddings are NOT here — they stay on the host.
/// `ClipWeights` remains the source of truth; [`Self::free_gpu`] returns every
/// device buffer without touching it.
pub struct GpuClipWeights {
    pub config: ClipConfig,
    ln1_w: Vec<GpuTensor>,
    ln1_b: Vec<GpuTensor>,
    q_w: Vec<GpuTensor>,
    q_b: Vec<GpuTensor>,
    k_w: Vec<GpuTensor>,
    k_b: Vec<GpuTensor>,
    v_w: Vec<GpuTensor>,
    v_b: Vec<GpuTensor>,
    out_w: Vec<GpuTensor>,
    out_b: Vec<GpuTensor>,
    ln2_w: Vec<GpuTensor>,
    ln2_b: Vec<GpuTensor>,
    fc1_w: Vec<GpuTensor>,
    fc1_b: Vec<GpuTensor>,
    fc2_w: Vec<GpuTensor>,
    fc2_b: Vec<GpuTensor>,
    final_ln_w: GpuTensor,
    final_ln_b: GpuTensor,
}

impl GpuClipWeights {
    /// `Some(reason)` if this checkpoint has no GPU path, `None` if it does.
    /// Checked before upload so `ensure_gpu` can fall back to the host
    /// encoder rather than stranding a loadable model — see
    /// [`crate::t5_gpu::GpuT5Weights::unsupported`].
    pub fn unsupported(host: &ClipWeights) -> Option<&'static str> {
        let cfg = &host.config;
        if cfg.hidden_act != "quick_gelu" {
            // The FLUX clip_l config ships `quick_gelu`. Substituting the
            // exact-erf GELU does not fail, it silently moves the pooled
            // vector — so the erf variant stays host-only rather than being
            // quietly approximated on the GPU.
            return Some("hidden_act is not `quick_gelu` — the GPU path implements only that");
        }
        if cfg.num_attention_heads == 0 || cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Some("hidden_size is not divisible by num_attention_heads");
        }
        None
    }

    /// Upload the encoder weights once.
    pub fn from_host(gpu: &mut Gpu, host: &ClipWeights) -> Result<Self, String> {
        if let Some(why) = Self::unsupported(host) {
            return Err(format!("clip gpu: unsupported checkpoint: {why}"));
        }
        let cfg = host.config.clone();
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        // `.weight` routes through the f16 branch of `upload_flux_tensor`;
        // anything else (biases, norm affines) stays f32.
        let lin = |gpu: &mut Gpu, t: &Tensor, tag: &str, rows, cols| {
            upload_flux_tensor(gpu, &format!("{tag}.weight"), &t.data, [rows, cols])
        };
        let f32v = |gpu: &mut Gpu, t: &Tensor, tag: &str, n: usize| {
            upload_flux_tensor(gpu, tag, &t.data, [n, 1])
        };
        let mut w = GpuClipWeights {
            final_ln_w: f32v(gpu, &host.final_ln_w, "clip.final_ln.gamma", h)?,
            final_ln_b: f32v(gpu, &host.final_ln_b, "clip.final_ln.beta", h)?,
            ln1_w: vec![],
            ln1_b: vec![],
            q_w: vec![],
            q_b: vec![],
            k_w: vec![],
            k_b: vec![],
            v_w: vec![],
            v_b: vec![],
            out_w: vec![],
            out_b: vec![],
            ln2_w: vec![],
            ln2_b: vec![],
            fc1_w: vec![],
            fc1_b: vec![],
            fc2_w: vec![],
            fc2_b: vec![],
            config: cfg,
        };
        for i in 0..w.config.num_hidden_layers {
            w.ln1_w
                .push(f32v(gpu, &host.ln1_w[i], "clip.ln1.gamma", h)?);
            w.ln1_b.push(f32v(gpu, &host.ln1_b[i], "clip.ln1.beta", h)?);
            w.q_w
                .push(lin(gpu, &host.q_w[i], &format!("clip.{i}.q"), h, h)?);
            w.q_b.push(f32v(gpu, &host.q_b[i], "clip.q.bias", h)?);
            w.k_w
                .push(lin(gpu, &host.k_w[i], &format!("clip.{i}.k"), h, h)?);
            w.k_b.push(f32v(gpu, &host.k_b[i], "clip.k.bias", h)?);
            w.v_w
                .push(lin(gpu, &host.v_w[i], &format!("clip.{i}.v"), h, h)?);
            w.v_b.push(f32v(gpu, &host.v_b[i], "clip.v.bias", h)?);
            w.out_w
                .push(lin(gpu, &host.out_w[i], &format!("clip.{i}.out"), h, h)?);
            w.out_b.push(f32v(gpu, &host.out_b[i], "clip.out.bias", h)?);
            w.ln2_w
                .push(f32v(gpu, &host.ln2_w[i], "clip.ln2.gamma", h)?);
            w.ln2_b.push(f32v(gpu, &host.ln2_b[i], "clip.ln2.beta", h)?);
            w.fc1_w.push(lin(
                gpu,
                &host.fc1_w[i],
                &format!("clip.{i}.fc1"),
                inter,
                h,
            )?);
            w.fc1_b
                .push(f32v(gpu, &host.fc1_b[i], "clip.fc1.bias", inter)?);
            w.fc2_w.push(lin(
                gpu,
                &host.fc2_w[i],
                &format!("clip.{i}.fc2"),
                h,
                inter,
            )?);
            w.fc2_b.push(f32v(gpu, &host.fc2_b[i], "clip.fc2.bias", h)?);
        }
        Ok(w)
    }

    /// Return every device buffer to the pool. Consumes self, so a field that
    /// forgets to free fails to compile here. Returns the number freed.
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let GpuClipWeights {
            config: _,
            ln1_w,
            ln1_b,
            q_w,
            q_b,
            k_w,
            k_b,
            v_w,
            v_b,
            out_w,
            out_b,
            ln2_w,
            ln2_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            final_ln_w,
            final_ln_b,
        } = self;
        let mut freed = 0usize;
        let drop_all = |gpu: &mut Gpu, ts: Vec<GpuTensor>, freed: &mut usize| {
            for t in ts {
                gpu.free_tensor(t).expect("clip gpu: free weight");
                *freed += 1;
            }
        };
        for group in [
            ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, ln2_w, ln2_b, fc1_w, fc1_b,
            fc2_w, fc2_b,
        ] {
            drop_all(gpu, group, &mut freed);
        }
        for t in [final_ln_w, final_ln_b] {
            gpu.free_tensor(t).expect("clip gpu: free final ln");
            freed += 1;
        }
        freed
    }
}

/// CLIP text forward on the GPU. Structural mirror of [`crate::clip::encode`].
///
/// Returns `(last_hidden_state [len, hidden], pooled [hidden])` on the HOST.
/// Unlike T5's, this output does not stay on the device: `pooled` is the only
/// thing FLUX consumes from CLIP, it is `hidden`-wide (768 floats), and the
/// MMDiT's `vector_in` embedder takes it as a host `Vec<f32>`.
///
/// `attention_mask` is accepted for signature parity with the CPU reference
/// and, exactly as there, intentionally unused.
pub fn encode(
    gpu: &mut Gpu,
    gw: &GpuClipWeights,
    host: &ClipWeights,
    input_ids: &[u32],
    attention_mask: &[u8],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let _ = attention_mask;
    let cfg = &gw.config;
    let len = input_ids.len();
    let h = cfg.hidden_size;
    let heads = cfg.num_attention_heads;
    let hd = h / heads;
    let eps = cfg.layer_norm_eps;
    if len == 0 {
        return Err("clip gpu: empty input_ids".into());
    }
    if heads * hd != h {
        return Err(format!(
            "clip gpu: hidden_size {h} is not divisible by num_attention_heads {heads}"
        ));
    }
    if len > cfg.max_position_embeddings {
        return Err(format!(
            "clip gpu: {len} tokens exceeds max_position_embeddings {}",
            cfg.max_position_embeddings
        ));
    }

    // ── token + learned position embeddings (host gather, one upload) ──
    let mut emb = vec![0f32; len * h];
    for (r, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        if (tok + 1) * h > host.token_embed.data.len() {
            return Err(format!("clip gpu: token id {tok} out of embedding range"));
        }
        for c in 0..h {
            emb[r * h + c] = host.token_embed.data[tok * h + c] + host.pos_embed.data[r * h + c];
        }
    }

    // CLIP's scale is the conventional 1/sqrt(head_dim) — unlike T5, which
    // folds its scaling into the relative bias and passes 1.0.
    let scale = 1.0 / (hd as f32).sqrt();

    let mut t = TextGpu::new(gpu);
    let hidden = t.upload(&emb, &[len, h])?;

    for i in 0..cfg.num_hidden_layers {
        // ── pre-norm self-attention ──────────────────────────────────
        let n1 = t.alloc(&[len, h])?;
        t.layernorm(&hidden, &gw.ln1_w[i], &gw.ln1_b[i], &n1, len, h, eps)?;
        let n1_f16 = t.cast_act(&n1, len, h)?;
        let q = t.gemm(&n1_f16, &gw.q_w[i], Some(&gw.q_b[i]), len, h, h)?;
        let k = t.gemm(&n1_f16, &gw.k_w[i], Some(&gw.k_b[i]), len, h, h)?;
        let v = t.gemm(&n1_f16, &gw.v_w[i], Some(&gw.v_b[i]), len, h, h)?;
        t.free(n1_f16)?;
        t.free(n1)?;

        let ctx = t.alloc(&[len, h])?;
        // Causal mask, no bias, no key-padding mask — the diffusers quirk.
        t.attn(
            &q, &k, &v, None, None, &ctx, len, heads, heads, hd, scale, true,
        )?;
        t.free(q)?;
        t.free(k)?;
        t.free(v)?;

        let ctx_f16 = t.cast_act(&ctx, len, h)?;
        let attn = t.gemm(&ctx_f16, &gw.out_w[i], Some(&gw.out_b[i]), len, h, h)?;
        t.free(ctx_f16)?;
        t.free(ctx)?;
        t.add_inplace(&hidden, &attn)?;
        t.free(attn)?;

        // ── pre-norm MLP (fc1 → quick-GELU → fc2) ────────────────────
        let n2 = t.alloc(&[len, h])?;
        t.layernorm(&hidden, &gw.ln2_w[i], &gw.ln2_b[i], &n2, len, h, eps)?;
        let n2_f16 = t.cast_act(&n2, len, h)?;
        let fc1 = t.gemm(
            &n2_f16,
            &gw.fc1_w[i],
            Some(&gw.fc1_b[i]),
            len,
            cfg.intermediate_size,
            h,
        )?;
        t.free(n2_f16)?;
        t.free(n2)?;
        t.quick_gelu(&fc1, &fc1, len * cfg.intermediate_size)?;
        let act_f16 = t.cast_act(&fc1, len, cfg.intermediate_size)?;
        let fc2 = t.gemm(
            &act_f16,
            &gw.fc2_w[i],
            Some(&gw.fc2_b[i]),
            len,
            h,
            cfg.intermediate_size,
        )?;
        t.free(act_f16)?;
        t.free(fc1)?;
        t.add_inplace(&hidden, &fc2)?;
        t.free(fc2)?;
    }

    // final_layer_norm → last_hidden_state
    let last_dev = t.alloc(&[len, h])?;
    t.layernorm(
        &hidden,
        &gw.final_ln_w,
        &gw.final_ln_b,
        &last_dev,
        len,
        h,
        eps,
    )?;
    t.free(hidden)?;
    let last = t.download(&last_dev)?;
    t.free(last_dev)?;

    // pooled = row at argmax(input_ids), i.e. the EOT position under the
    // transformers-5 `eos_token_id == 2` rule the CPU reference pins.
    // Ties resolve to the FIRST maximum, matching `>` in the CPU loop.
    let mut eot = 0usize;
    let mut best = input_ids[0];
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok > best {
            best = tok;
            eot = i;
        }
    }
    let pooled = last[eot * h..(eot + 1) * h].to_vec();
    Ok((last, pooled))
}
