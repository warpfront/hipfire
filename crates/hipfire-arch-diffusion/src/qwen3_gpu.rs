// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU Qwen3 encoder with hidden-state taps — a layer-for-layer mirror of the
//! CPU reference in [`crate::qwen3`], so the two files diff side by side. Same
//! shape as [`crate::t5_gpu`] is to [`crate::t5`], and it reuses that file's
//! [`TextGpu`] scratch helper rather than growing a third dispatch wrapper.
//!
//! Why it exists: FLUX.2 Klein conditions on a **4B** (or 9B) Qwen3 causal-LM
//! tower, and `qwen3::encode_taps` runs every linear through `nn::linear`, a
//! scalar rayon loop. At the Klein 4B geometry (hidden 2560, 36 layers,
//! intermediate 9728, 512-token frame) one prompt is ~3.7 TFLOP of host work
//! — minutes per prompt, against a denoise loop measured in seconds. The
//! conditioning, not the trunk, is the cost.
//!
//! What is on the GPU and what is not:
//! - **On the GPU**: every linear (f16 weights through the WMMA GEMM, f32
//!   accumulate), all four RMSNorms per layer (pre-attention, per-head Q/K,
//!   post-attention), the half-split RoPE, the GQA causal attention, and the
//!   SwiGLU MLP.
//! - **On the host**: the token-embedding gather only. Qwen3's table is
//!   `vocab × hidden` (151936 × 2560 = 778 MB in f16) and a prompt touches at
//!   most a few hundred of its rows, so uploading it would cost more than the
//!   gather saves. Same split [`crate::t5_gpu`] makes.
//!
//! Numerics: weights are f16, accumulators f32 — the same tradeoff
//! [`crate::flux_gpu`] and [`crate::t5_gpu`] already make. Parity against the
//! f32 host reference is therefore ~1e-3 relative, not bit-exact; the gate is
//! `examples/gpu_qwen3_parity.rs` at rel_l2 ≤ 2e-3 per tap.
//!
//! Masking: unlike T5 (whose golden attends over the pad rows), Qwen3 IS
//! masked — the causal window plus a key-padding mask built from the Klein
//! right-padding. Both are the same masks the CPU reference applies, and both
//! are pushed into `attention_text_f32` rather than materialised as a bias.
//!
//! Positions are `0..len` always: Klein right-pads, so a real token never
//! shifts, and the pad tail's positions are never read by a real query.

use crate::f16_stage::F16Stage;
use crate::flux_gpu::upload_flux_tensor;
use crate::qwen3::{Qwen3Config, Qwen3Plan, Qwen3Weights};
use crate::t5_gpu::TextGpu;
use hipfire_runtime::model_source::ModelSource;
use rdna_compute::{Gpu, GpuTensor};

/// Stage one 2-D linear as f16 words and upload it, then release its pages.
fn upload_lin_f16(
    gpu: &mut Gpu,
    src: &dyn ModelSource,
    plan: &Qwen3Plan,
    name: &str,
    rows: usize,
    cols: usize,
    stage: &mut F16Stage,
) -> Result<GpuTensor, String> {
    let words = plan.stage_f16(src, name, rows, cols, stage)?;
    let t = gpu
        .upload_f16_bits(words, &[rows, cols])
        .map_err(|e| format!("qwen3 gpu: upload `{name}` f16: {e:?}"))?;
    plan.release(src, name);
    Ok(t)
}

/// Upload one 1-D norm vector as f32 (what `rmsnorm_batched` reads), then
/// release its pages. These are `hidden` or `head_dim` wide — a few KB each.
fn upload_vec_f32(
    gpu: &mut Gpu,
    src: &dyn ModelSource,
    plan: &Qwen3Plan,
    name: &str,
    n: usize,
) -> Result<GpuTensor, String> {
    let t = plan.tensor(src, name, n, 1)?;
    let g = gpu
        .upload_f32(&t.data, &[n, 1])
        .map_err(|e| format!("qwen3 gpu: upload `{name}` f32: {e:?}"))?;
    plan.release(src, name);
    Ok(g)
}

/// GPU-resident Qwen3 encoder weights. Linears are f16 (the WMMA GEMM's
/// weight operand); the four norm vectors per layer are f32.
///
/// The token embedding is NOT here — it stays on the host (see the module
/// docs), so every `encode_taps` call also takes a host [`Qwen3Weights`] whose
/// `embed` is real (a [`Qwen3Plan::materialize_light`] set is enough).
pub struct GpuQwen3Weights {
    pub config: Qwen3Config,
    input_norm: Vec<GpuTensor>,
    q: Vec<GpuTensor>,
    k: Vec<GpuTensor>,
    v: Vec<GpuTensor>,
    o: Vec<GpuTensor>,
    q_norm: Vec<GpuTensor>,
    k_norm: Vec<GpuTensor>,
    post_norm: Vec<GpuTensor>,
    gate: Vec<GpuTensor>,
    up: Vec<GpuTensor>,
    down: Vec<GpuTensor>,
}

impl GpuQwen3Weights {
    /// `Some(reason)` if a checkpoint with this config has no GPU path,
    /// `None` if it does.
    ///
    /// Checked BEFORE upload so an unsupported checkpoint costs nothing and,
    /// more importantly, so the bundle's `ensure_gpu` can leave the GPU
    /// encoder unset and let the host encoder serve it — the host path
    /// handles every variant. Same contract as
    /// [`crate::t5_gpu::GpuT5Weights::unsupported_plan`].
    pub fn unsupported(cfg: &Qwen3Config) -> Option<&'static str> {
        if cfg.kv_heads == 0 || cfg.heads % cfg.kv_heads != 0 {
            return Some("heads is not a multiple of num_key_value_heads — GQA head mapping");
        }
        if cfg.head_dim % 2 != 0 {
            return Some("odd head_dim — half-split RoPE needs an even head_dim");
        }
        // Every GEMM's K comes from one of these three widths. The WMMA GEMM
        // needs K % 16; the real Klein geometry (2560 / 4096 / 9728) is
        // K % 64 and takes the LDS route.
        if cfg.hidden % 16 != 0
            || cfg.intermediate % 16 != 0
            || (cfg.heads * cfg.head_dim) % 16 != 0
        {
            return Some("hidden / intermediate / heads*head_dim must be multiples of 16 (WMMA K)");
        }
        None
    }

    /// Upload the decoder weights STRAIGHT FROM THE CHECKPOINT.
    ///
    /// [`from_host`](Self::from_host) requires a fully decoded
    /// [`Qwen3Weights`], which for Klein 4B is ~16 GB of host f32 that the
    /// bundle would then keep forever even though the GPU encoder only reads
    /// the embedding table back from the host. This streams each layer's
    /// linears out of the mmap through a reusable f16 staging buffer (peak
    /// ~50 MB, the `intermediate × hidden` projections) and releases the pages
    /// behind it.
    ///
    /// Numerically identical to `from_host`: the host RNE conversion matches
    /// the device `(_Float16)` cast it replaces, and the norms stay f32.
    pub fn from_stream(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &Qwen3Plan,
    ) -> Result<Self, String> {
        if let Some(why) = Self::unsupported(&plan.config) {
            return Err(format!("qwen3 gpu: unsupported checkpoint: {why}"));
        }
        let mut out = Self::empty(plan.config.clone());
        match Self::stream_layers(gpu, src, plan, &mut out) {
            Ok(()) => Ok(out),
            Err(e) => {
                // `GpuTensor` has no `Drop`, and this is a multi-GB upload on
                // a device that is already holding the FLUX.2 transformer. A
                // half-finished set would hold several GB nothing would ever
                // reclaim, for the rest of the process.
                let freed = out.free_partial(gpu);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    fn empty(config: Qwen3Config) -> Self {
        Self {
            config,
            input_norm: vec![],
            q: vec![],
            k: vec![],
            v: vec![],
            o: vec![],
            q_norm: vec![],
            k_norm: vec![],
            post_norm: vec![],
            gate: vec![],
            up: vec![],
            down: vec![],
        }
    }

    /// The per-layer upload loop, split out so `from_stream` owns the
    /// partially-filled `out` on the error path and can return it to the pool.
    ///
    /// Key order is [`Qwen3Plan::layer_keys`]' — the seven linears (q, k, v,
    /// o, gate, up, down) f16, then the four vectors (input_norm, q_norm,
    /// k_norm, post_norm) f32.
    fn stream_layers(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &Qwen3Plan,
        out: &mut GpuQwen3Weights,
    ) -> Result<(), String> {
        let mut stage = F16Stage::new();
        for i in 0..out.config.layers {
            let [input_norm, q, k, v, o, q_norm, k_norm, post_norm, gate, up, down] =
                plan.layer_keys(i);
            let mut lin = |gpu: &mut Gpu, key: &(String, usize, usize)| {
                upload_lin_f16(gpu, src, plan, &key.0, key.1, key.2, &mut stage)
            };
            out.q.push(lin(gpu, &q)?);
            out.k.push(lin(gpu, &k)?);
            out.v.push(lin(gpu, &v)?);
            out.o.push(lin(gpu, &o)?);
            out.gate.push(lin(gpu, &gate)?);
            out.up.push(lin(gpu, &up)?);
            out.down.push(lin(gpu, &down)?);
            for (slot, key) in [
                (&mut out.input_norm, &input_norm),
                (&mut out.q_norm, &q_norm),
                (&mut out.k_norm, &k_norm),
                (&mut out.post_norm, &post_norm),
            ] {
                slot.push(upload_vec_f32(gpu, src, plan, &key.0, key.1)?);
            }
        }
        stage.clear();
        Ok(())
    }

    /// Upload from already-decoded host tables. The eager twin of
    /// [`from_stream`](Self::from_stream) — only worth it when the caller
    /// already holds the full f32 set (a parity harness), since building one
    /// just to upload it costs ~16 GB of host RAM at Klein 4B.
    pub fn from_host(gpu: &mut Gpu, host: &Qwen3Weights) -> Result<Self, String> {
        if let Some(why) = Self::unsupported(&host.config) {
            return Err(format!("qwen3 gpu: unsupported checkpoint: {why}"));
        }
        if host.is_light() {
            return Err(
                "qwen3 gpu: from_host on a LIGHT weight set (embedding only) — materialise the \
                 layers first, or use from_stream"
                    .into(),
            );
        }
        let mut out = Self::empty(host.config.clone());
        match Self::upload_host_layers(gpu, host, &mut out) {
            Ok(()) => Ok(out),
            Err(e) => {
                let freed = out.free_partial(gpu);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    fn upload_host_layers(
        gpu: &mut Gpu,
        host: &Qwen3Weights,
        out: &mut GpuQwen3Weights,
    ) -> Result<(), String> {
        // `.weight` routes through the f16 branch of `upload_flux_tensor`;
        // anything else stays f32. Same dtype split the MMDiT upload makes.
        for (i, l) in host.layers.iter().enumerate() {
            for (slot, t, name) in [
                (&mut out.q, &l.q, "q"),
                (&mut out.k, &l.k, "k"),
                (&mut out.v, &l.v, "v"),
                (&mut out.o, &l.o, "o"),
                (&mut out.gate, &l.gate, "gate"),
                (&mut out.up, &l.up, "up"),
                (&mut out.down, &l.down, "down"),
            ] {
                slot.push(upload_flux_tensor(
                    gpu,
                    &format!("qwen3.{i}.{name}.weight"),
                    &t.data,
                    [t.rows, t.cols],
                )?);
            }
            for (slot, t, name) in [
                (&mut out.input_norm, &l.input_norm, "input_norm"),
                (&mut out.q_norm, &l.q_norm, "q_norm"),
                (&mut out.k_norm, &l.k_norm, "k_norm"),
                (&mut out.post_norm, &l.post_norm, "post_norm"),
            ] {
                slot.push(upload_flux_tensor(
                    gpu,
                    &format!("qwen3.{i}.{name}"),
                    &t.data,
                    [t.rows, 1],
                )?);
            }
        }
        Ok(())
    }

    /// Best-effort release of whatever this (possibly partially built) weight
    /// set holds. Error-path twin of [`free_gpu`](Self::free_gpu), which
    /// panics on a failed free — wrong while unwinding a different failure,
    /// where a cleanup panic would destroy the diagnostic that mattered.
    fn free_partial(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        for group in self.into_groups() {
            for t in group {
                if gpu.free_tensor(t).is_ok() {
                    freed += 1;
                }
            }
        }
        freed
    }

    /// Return every device buffer to the pool. Consumes self, so a field that
    /// forgets to free fails to compile in [`Self::into_groups`] (the same
    /// contract `GpuT5Weights::free_gpu` holds). Returns the number of buffers
    /// freed.
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let mut freed = 0usize;
        for group in self.into_groups() {
            for t in group {
                gpu.free_tensor(t).expect("qwen3 gpu: free weight");
                freed += 1;
            }
        }
        freed
    }

    /// Every device-owning field, destructured — so adding a field without
    /// listing it here is a compile error, not a leak.
    fn into_groups(self) -> [Vec<GpuTensor>; 11] {
        let GpuQwen3Weights {
            config: _,
            input_norm,
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            post_norm,
            gate,
            up,
            down,
        } = self;
        [
            input_norm, q, k, v, o, q_norm, k_norm, post_norm, gate, up, down,
        ]
    }
}

/// Qwen3 causal-LM forward on the GPU with hidden-state taps. Structural
/// mirror of [`crate::qwen3::encode_taps`].
///
/// `host` supplies the token-embedding table (and nothing else); `gw`
/// supplies every layer from the device. Returns a **device** f32 tensor
/// `[len, taps.len() * hidden]` — the residual stream after each 1-based
/// layer index in `taps`, concatenated per token in `taps` order, exactly the
/// layout the CPU reference returns. The caller owns it and must free it.
/// Keeping it on the device is the point: it feeds the FLUX.2 transformer's
/// `txt_in` with no host round trip.
pub fn encode_taps(
    gpu: &mut Gpu,
    gw: &GpuQwen3Weights,
    host: &Qwen3Weights,
    input_ids: &[u32],
    key_mask: &[u8],
    taps: &[usize],
) -> Result<GpuTensor, String> {
    let cfg = gw.config.clone();
    let len = input_ids.len();
    let d = cfg.hidden;
    let (heads, kvh, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
    let qd = heads * hd;
    let kvd = kvh * hd;
    let eps = cfg.eps;
    let scale = 1.0 / (hd as f32).sqrt();
    if len == 0 {
        return Err("qwen3 gpu: empty input_ids".into());
    }
    if key_mask.len() != len {
        return Err(format!(
            "qwen3 gpu: key_mask has {} entries for {len} tokens",
            key_mask.len()
        ));
    }
    if taps.is_empty() {
        return Err("qwen3 gpu: no taps requested".into());
    }
    if let Some(&bad) = taps.iter().find(|&&t| t == 0 || t > cfg.layers) {
        return Err(format!(
            "qwen3 gpu: tap {bad} is out of range for a {}-layer tower (taps are 1-based)",
            cfg.layers
        ));
    }
    // A repeated tap would leave its second output slab UNWRITTEN — the
    // per-layer copy is keyed by `position`, which finds only the first
    // match — and this output buffer is uninitialised by construction. The
    // CPU reference has the same first-match rule but a zeroed buffer, so a
    // duplicate is a silent divergence rather than a crash. Refuse it.
    if let Some(&dup) = taps
        .iter()
        .enumerate()
        .find_map(|(i, t)| taps[..i].contains(t).then_some(t))
    {
        return Err(format!("qwen3 gpu: tap {dup} is listed more than once"));
    }
    if host.embed.cols != d {
        return Err(format!(
            "qwen3 gpu: host embedding is {} wide but the device weights are for hidden {d}",
            host.embed.cols
        ));
    }
    if gw.q.len() != cfg.layers {
        return Err(format!(
            "qwen3 gpu: {} uploaded layers for a {}-layer config",
            gw.q.len(),
            cfg.layers
        ));
    }
    if let Some(why) = GpuQwen3Weights::unsupported(&cfg) {
        return Err(format!("qwen3 gpu: unsupported checkpoint: {why}"));
    }

    // ── token embeddings (host gather, one upload) ───────────────────
    let mut emb = vec![0f32; len * d];
    for (r, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        if (tok + 1) * d > host.embed.data.len() {
            return Err(format!("qwen3 gpu: token id {tok} out of embedding range"));
        }
        emb[r * d..(r + 1) * d].copy_from_slice(&host.embed.data[tok * d..(tok + 1) * d]);
    }

    // Key-padding mask as f32 (the kernel's contract: 1.0 visible, 0.0
    // masked) and RoPE positions as i32 BITS in an F32-typed tensor — the
    // `rope_batched_f32` slot convention, mirrored from `llama.rs`.
    let mask_f32: Vec<f32> = key_mask.iter().map(|&m| f32::from(m != 0)).collect();
    let positions: Vec<i32> = (0..len as i32).collect();

    let mut t = TextGpu::new(gpu);
    // Straight-line, like the CPU reference; an error leaks the intermediates
    // allocated so far, which is the same contract `t5_gpu::encode` and
    // `flux_gpu::forward_parts` have (a failed forward is not a recoverable
    // state for the pool anyway).
    let mask_dev = t.upload(&mask_f32, &[len])?;
    let pos_dev = t.upload_i32_bits(&positions)?;
    let hidden = t.upload(&emb, &[len, d])?;
    let out = t.alloc(&[len, taps.len() * d])?;

    for li in 0..cfg.layers {
        // ── self-attention (pre-norm) ────────────────────────────────
        let normed = t.alloc(&[len, d])?;
        t.rmsnorm(&hidden, &gw.input_norm[li], &normed, len, d, eps)?;
        let normed_f16 = t.cast_act(&normed, len, d)?;
        let q = t.gemm(&normed_f16, &gw.q[li], None, len, qd, d)?;
        let k = t.gemm(&normed_f16, &gw.k[li], None, len, kvd, d)?;
        let v = t.gemm(&normed_f16, &gw.v[li], None, len, kvd, d)?;
        t.free(normed_f16)?;
        t.free(normed)?;

        // Per-head Q/K RMSNorm, BEFORE RoPE (Qwen3), in place: one row per
        // (token, head) over `head_dim`.
        t.rmsnorm(&q, &gw.q_norm[li], &q, len * heads, hd, eps)?;
        t.rmsnorm(&k, &gw.k_norm[li], &k, len * kvh, hd, eps)?;
        t.rope(&q, &k, &pos_dev, heads, kvh, hd, cfg.rope_theta as f32, len)?;

        let ctx = t.alloc(&[len, qd])?;
        t.attn(
            &q,
            &k,
            &v,
            None, /* additive bias */
            Some(&mask_dev),
            &ctx,
            len,
            heads,
            kvh,
            hd,
            scale,
            true, /* causal */
        )?;
        t.free(q)?;
        t.free(k)?;
        t.free(v)?;

        let ctx_f16 = t.cast_act(&ctx, len, qd)?;
        let attn_out = t.gemm(&ctx_f16, &gw.o[li], None, len, d, qd)?;
        t.free(ctx_f16)?;
        t.free(ctx)?;
        t.add_inplace(&hidden, &attn_out)?;
        t.free(attn_out)?;

        // ── SwiGLU MLP (pre-norm) ────────────────────────────────────
        let ffn_in = t.alloc(&[len, d])?;
        t.rmsnorm(&hidden, &gw.post_norm[li], &ffn_in, len, d, eps)?;
        let ffn_f16 = t.cast_act(&ffn_in, len, d)?;
        let gate = t.gemm(&ffn_f16, &gw.gate[li], None, len, cfg.intermediate, d)?;
        let up = t.gemm(&ffn_f16, &gw.up[li], None, len, cfg.intermediate, d)?;
        t.free(ffn_f16)?;
        t.free(ffn_in)?;
        // act = silu(gate) * up, written in place over `gate`.
        t.silu_mul(&gate, &up, &gate)?;
        t.free(up)?;
        let act_f16 = t.cast_act(&gate, len, cfg.intermediate)?;
        t.free(gate)?;
        let down = t.gemm(&act_f16, &gw.down[li], None, len, d, cfg.intermediate)?;
        t.free(act_f16)?;
        t.add_inplace(&hidden, &down)?;
        t.free(down)?;

        // ── tap ──────────────────────────────────────────────────────
        if let Some(ti) = taps.iter().position(|&tap| tap == li + 1) {
            t.copy_rows(&hidden, &out, len, d, d, taps.len() * d, ti * d)?;
        }
    }

    t.free(hidden)?;
    t.free(pos_dev)?;
    t.free(mask_dev)?;
    Ok(out)
}

/// [`encode_taps`] followed by a download — the shape the CPU reference
/// returns. Only for parity harnesses; the serving path keeps the tensor
/// on-device.
pub fn encode_taps_host(
    gpu: &mut Gpu,
    gw: &GpuQwen3Weights,
    host: &Qwen3Weights,
    input_ids: &[u32],
    key_mask: &[u8],
    taps: &[usize],
) -> Result<Vec<f32>, String> {
    let t = encode_taps(gpu, gw, host, input_ids, key_mask, taps)?;
    let out = gpu
        .download_f32(&t)
        .map_err(|e| format!("qwen3 gpu: download taps: {e:?}"));
    gpu.free_tensor(t)
        .map_err(|e| format!("qwen3 gpu: free taps: {e:?}"))?;
    out
}
