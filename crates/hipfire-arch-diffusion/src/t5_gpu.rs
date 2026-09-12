// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU T5 encoder — a layer-for-layer mirror of the CPU reference in
//! [`crate::t5`], so the two files diff side by side.
//!
//! Why it exists: `t5::encode` runs every linear through
//! `nn::linear`, a scalar rayon loop. At the real FLUX geometry (T5-XXL:
//! `d_model` 4096, `d_ff` 10240, 24 layers, 256 tokens) that is ~2.4 TFLOP of
//! host work — roughly 55 s per prompt on the target machine, against a
//! ComfyUI fixed cost of 12.7 s for the whole conditioning + VAE stack. The
//! denoise loop was never the problem; this was.
//!
//! What is on the GPU and what is not:
//! - **On the GPU**: every linear (f16 weights through the WMMA GEMM, f32
//!   accumulate), the T5 RMSNorms, the attention, and the gated-GELU FFN.
//! - **On the host**: the token-embedding gather and the relative-position
//!   bucket table. The embedding table is `vocab × d_model`
//!   (32128 × 4096 = 263 MB in f16) and every prompt touches at most 256 of
//!   its rows, so uploading it would cost more than the gather saves. The
//!   bucket table is data-INDEPENDENT — a pure function of `(len, buckets,
//!   max_distance)` — so the `[heads, len, len]` bias is built once on the
//!   host and cached device-side per sequence length.
//!
//! Numerics: weights are f16, accumulators f32 — the same tradeoff
//! [`crate::flux_gpu`] already makes for the MMDiT. Parity against the f32
//! host reference is therefore ~1e-3 relative, not bit-exact; the gate is
//! `examples/gpu_t5_parity.rs` at rel ≤ 5e-3.
//!
//! Masking: the CPU reference deliberately ignores `attention_mask` (ComfyUI
//! constructs T5-XXL with `enable_attention_masks=False`, so the golden
//! attends over the pad rows too). This mirror does the same — see
//! [`crate::t5::encode`] for the full note. Passing a mask here would diverge
//! from the golden at layer 0.

use crate::f16_stage::F16Stage;
use crate::flux::Tensor;
use crate::flux_gpu::upload_flux_tensor;
use crate::t5::{compute_rel_bias, T5Config, T5Plan, T5Weights};
use hipfire_runtime::model_source::ModelSource;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Stage one 2-D linear as f16 words and upload it, then release its pages.
fn upload_lin_f16(
    gpu: &mut Gpu,
    src: &dyn ModelSource,
    plan: &T5Plan,
    name: &str,
    rows: usize,
    cols: usize,
    stage: &mut F16Stage,
) -> Result<GpuTensor, String> {
    let words = plan.stage_f16(src, name, rows, cols, stage)?;
    let t = gpu
        .upload_f16_bits(words, &[rows, cols])
        .map_err(|e| format!("t5 gpu: upload `{name}` f16: {e:?}"))?;
    plan.release(src, name);
    Ok(t)
}

/// Upload one 1-D norm vector as f32 (what `rmsnorm_batched` reads), then
/// release its pages. These are `d_model` wide — a few KB each.
fn upload_vec_f32(
    gpu: &mut Gpu,
    src: &dyn ModelSource,
    plan: &T5Plan,
    name: &str,
    n: usize,
) -> Result<GpuTensor, String> {
    let t = plan.tensor(src, name, n, 1)?;
    let g = gpu
        .upload_f32(&t.data, &[n, 1])
        .map_err(|e| format!("t5 gpu: upload `{name}` f32: {e:?}"))?;
    plan.release(src, name);
    Ok(g)
}

/// GPU-resident T5 encoder weights. Linears are f16 (the WMMA GEMM's weight
/// operand); norms and the relative-attention-bias are f32.
///
/// The token embedding is NOT here — it stays on the host (see the module
/// docs). `T5Weights` remains the source of truth; [`Self::free_gpu`] returns
/// every device buffer without touching it.
pub struct GpuT5Weights {
    pub config: T5Config,
    q: Vec<GpuTensor>,
    k: Vec<GpuTensor>,
    v: Vec<GpuTensor>,
    o: Vec<GpuTensor>,
    attn_norm: Vec<GpuTensor>,
    wi: Vec<GpuTensor>,
    /// Gate projection — populated only for the gated (v1.1) FFN, exactly as
    /// [`T5Weights::wi_gate`]. FLUX conditions on T5-XXL v1.1, which is gated.
    wi_gate: Vec<GpuTensor>,
    wo: Vec<GpuTensor>,
    ffn_norm: Vec<GpuTensor>,
    final_norm: GpuTensor,
    /// Device-resident `[heads, len, len]` relative bias, cached by `len`.
    /// Data-independent, so one upload serves every prompt of that length.
    rel_bias: Option<(usize, GpuTensor)>,
}

impl GpuT5Weights {
    /// `Some(reason)` if this checkpoint has no GPU path, `None` if it does.
    ///
    /// Checked BEFORE upload so an unsupported checkpoint costs nothing and,
    /// more importantly, so `ensure_gpu` can leave `gpu_t5 = None` and let the
    /// host encoder serve it. Failing the upload instead would strand a
    /// perfectly loadable model: the host path handles every variant.
    pub fn unsupported(host: &T5Weights) -> Option<&'static str> {
        if host.wi_gate.is_empty() {
            // T5 v1.0's ungated `wo(relu(wi x))`. FLUX conditions on T5-XXL
            // v1.1 (gated GELU); running the wrong FFN form does not error,
            // it yields a wrong text embedding and a plausible-but-wrong
            // image, so the GPU path implements only the gated one.
            return Some("ungated (T5 v1.0) FFN — the GPU path implements only v1.1 gated-GELU");
        }
        let cfg = &host.config;
        if cfg.num_heads * cfg.d_kv != cfg.d_model {
            return Some("num_heads * d_kv != d_model — q/k/v are indexed head-interleaved");
        }
        None
    }

    /// Upload the encoder weights once. Idempotent at the call site: the
    /// caller (`FluxPipeBundle::ensure_gpu`) holds the result for the session.
    pub fn from_host(gpu: &mut Gpu, host: &T5Weights) -> Result<Self, String> {
        if let Some(why) = Self::unsupported(host) {
            return Err(format!("t5 gpu: unsupported checkpoint: {why}"));
        }
        let cfg = host.config.clone();
        let d = cfg.d_model;
        let d_ff = cfg.d_ff;
        // `.weight` routes through the f16 branch of `upload_flux_tensor`;
        // anything else stays f32. Same dtype split as the MMDiT upload.
        let lin = |gpu: &mut Gpu, t: &Tensor, name: &str, rows, cols| {
            upload_flux_tensor(gpu, &format!("{name}.weight"), &t.data, [rows, cols])
        };
        let vecf32 = |gpu: &mut Gpu, t: &Tensor, name: &str, n: usize| {
            upload_flux_tensor(gpu, name, &t.data, [n, 1])
        };
        let mut out = GpuT5Weights {
            final_norm: vecf32(gpu, &host.final_norm, "t5.final_norm", d)?,
            q: vec![],
            k: vec![],
            v: vec![],
            o: vec![],
            attn_norm: vec![],
            wi: vec![],
            wi_gate: vec![],
            wo: vec![],
            ffn_norm: vec![],
            rel_bias: None,
            config: cfg,
        };
        for i in 0..out.config.num_layers {
            out.q
                .push(lin(gpu, &host.q[i], &format!("t5.{i}.q"), d, d)?);
            out.k
                .push(lin(gpu, &host.k[i], &format!("t5.{i}.k"), d, d)?);
            out.v
                .push(lin(gpu, &host.v[i], &format!("t5.{i}.v"), d, d)?);
            out.o
                .push(lin(gpu, &host.o[i], &format!("t5.{i}.o"), d, d)?);
            out.attn_norm
                .push(vecf32(gpu, &host.attn_norm[i], "t5.attn_norm", d)?);
            out.wi
                .push(lin(gpu, &host.wi[i], &format!("t5.{i}.wi"), d_ff, d)?);
            if !host.wi_gate.is_empty() {
                out.wi_gate.push(lin(
                    gpu,
                    &host.wi_gate[i],
                    &format!("t5.{i}.wi_gate"),
                    d_ff,
                    d,
                )?);
            }
            out.wo
                .push(lin(gpu, &host.wo[i], &format!("t5.{i}.wo"), d, d_ff)?);
            out.ffn_norm
                .push(vecf32(gpu, &host.ffn_norm[i], "t5.ffn_norm", d)?);
        }
        Ok(out)
    }

    /// `Some(reason)` if a checkpoint with this config and FFN form has no GPU
    /// path. The plan-shaped twin of [`unsupported`](Self::unsupported), for
    /// the streaming loader, which never builds the host tables that version
    /// inspects.
    pub fn unsupported_plan(cfg: &T5Config, gated: bool) -> Option<&'static str> {
        if !gated {
            return Some("ungated (T5 v1.0) FFN — the GPU path implements only v1.1 gated-GELU");
        }
        if cfg.num_heads * cfg.d_kv != cfg.d_model {
            return Some("num_heads * d_kv != d_model — q/k/v are indexed head-interleaved");
        }
        None
    }

    /// Upload the encoder weights STRAIGHT FROM THE CHECKPOINT.
    ///
    /// [`from_host`](Self::from_host) requires a fully decoded `T5Weights`,
    /// which for T5-XXL is ~18.5 GB of host f32 that the bundle then keeps
    /// forever even though the GPU encoder only reads the embedding table and
    /// the relative-attention bias back from the host. This streams each of
    /// the 24 layers' linears out of the mmap through a reusable f16 staging
    /// buffer (peak ~80 MB, the `d_ff × d_model` projections) and releases the
    /// pages behind it.
    ///
    /// Numerically identical to `from_host`: the host RNE conversion matches
    /// the device `(_Float16)` cast it replaces, and the norms stay f32.
    pub fn from_stream(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &T5Plan,
    ) -> Result<Self, String> {
        if let Some(why) = Self::unsupported_plan(&plan.config, plan.gated) {
            return Err(format!("t5 gpu: unsupported checkpoint: {why}"));
        }
        let cfg = plan.config.clone();
        let d = cfg.d_model;
        // `final_norm` is uploaded first and alone: if IT fails, nothing is
        // resident yet and there is nothing to clean up.
        let final_norm = upload_vec_f32(gpu, src, plan, "encoder.final_layer_norm.weight", d)?;
        let mut out = GpuT5Weights {
            final_norm,
            q: vec![],
            k: vec![],
            v: vec![],
            o: vec![],
            attn_norm: vec![],
            wi: vec![],
            wi_gate: vec![],
            wo: vec![],
            ffn_norm: vec![],
            rel_bias: None,
            config: cfg,
        };
        match Self::stream_layers(gpu, src, plan, &mut out) {
            Ok(()) => Ok(out),
            Err(e) => {
                // `GpuTensor` has no `Drop`, and T5 is the upload most likely
                // to fail: it asks for ~9.3 GB on a device that is already
                // holding the 24 GB transformer. Worse, `ensure_gpu` LATCHES
                // the failure (`gpu_t5_declined`) so it never retries — so
                // without this, a half-finished T5 upload would hold several
                // GB of device memory that nothing would ever reclaim, for the
                // rest of the daemon session.
                let freed = out.free_partial(gpu);
                Err(format!("{e} [freed {freed} partially-uploaded tensors]"))
            }
        }
    }

    /// The per-layer upload loop, split out so `from_stream` owns the
    /// partially-filled `out` on the error path and can return it to the pool.
    fn stream_layers(
        gpu: &mut Gpu,
        src: &dyn ModelSource,
        plan: &T5Plan,
        out: &mut GpuT5Weights,
    ) -> Result<(), String> {
        let d = out.config.d_model;
        let d_ff = out.config.d_ff;
        let mut stage = F16Stage::new();
        for i in 0..out.config.num_layers {
            let b = format!("encoder.block.{i}");
            // `.weight` semantics of `upload_flux_tensor`, without the host
            // table: 2-D linears go f16-resident, 1-D norms stay f32.
            let mut lin = |gpu: &mut Gpu, name: String, rows: usize, cols: usize| {
                upload_lin_f16(gpu, src, plan, &name, rows, cols, &mut stage)
            };
            out.q.push(lin(
                gpu,
                format!("{b}.layer.0.SelfAttention.q.weight"),
                d,
                d,
            )?);
            out.k.push(lin(
                gpu,
                format!("{b}.layer.0.SelfAttention.k.weight"),
                d,
                d,
            )?);
            out.v.push(lin(
                gpu,
                format!("{b}.layer.0.SelfAttention.v.weight"),
                d,
                d,
            )?);
            out.o.push(lin(
                gpu,
                format!("{b}.layer.0.SelfAttention.o.weight"),
                d,
                d,
            )?);
            let ffn_in = plan.ffn_in(i);
            out.wi.push(lin(gpu, ffn_in[0].clone(), d_ff, d)?);
            out.wi_gate.push(lin(gpu, ffn_in[1].clone(), d_ff, d)?);
            out.wo.push(lin(
                gpu,
                format!("{b}.layer.1.DenseReluDense.wo.weight"),
                d,
                d_ff,
            )?);
            out.attn_norm.push(upload_vec_f32(
                gpu,
                src,
                plan,
                &format!("{b}.layer.0.layer_norm.weight"),
                d,
            )?);
            out.ffn_norm.push(upload_vec_f32(
                gpu,
                src,
                plan,
                &format!("{b}.layer.1.layer_norm.weight"),
                d,
            )?);
        }
        stage.clear();
        Ok(())
    }

    /// Best-effort release of whatever this (possibly partially built) weight
    /// set holds. Error-path twin of [`free_gpu`](Self::free_gpu), which
    /// panics on a failed free — wrong while unwinding a different failure,
    /// where a cleanup panic would destroy the diagnostic that mattered.
    fn free_partial(self, gpu: &mut Gpu) -> usize {
        let GpuT5Weights {
            config: _,
            q,
            k,
            v,
            o,
            attn_norm,
            wi,
            wi_gate,
            wo,
            ffn_norm,
            final_norm,
            rel_bias,
        } = self;
        let mut freed = 0usize;
        let drop_one = |gpu: &mut Gpu, t: GpuTensor, freed: &mut usize| {
            if gpu.free_tensor(t).is_ok() {
                *freed += 1;
            }
        };
        for group in [q, k, v, o, attn_norm, wi, wi_gate, wo, ffn_norm] {
            for t in group {
                drop_one(gpu, t, &mut freed);
            }
        }
        drop_one(gpu, final_norm, &mut freed);
        if let Some((_, t)) = rel_bias {
            drop_one(gpu, t, &mut freed);
        }
        freed
    }

    /// Return every device buffer to the pool. Consumes self, so a field that
    /// forgets to free fails to compile here (the same contract
    /// `GpuFluxWeights::free_gpu` holds). Returns the number of buffers freed.
    pub fn free_gpu(self, gpu: &mut Gpu) -> usize {
        let GpuT5Weights {
            config: _,
            q,
            k,
            v,
            o,
            attn_norm,
            wi,
            wi_gate,
            wo,
            ffn_norm,
            final_norm,
            rel_bias,
        } = self;
        let mut freed = 0usize;
        let drop_all = |gpu: &mut Gpu, ts: Vec<GpuTensor>, freed: &mut usize| {
            for t in ts {
                gpu.free_tensor(t).expect("t5 gpu: free weight");
                *freed += 1;
            }
        };
        drop_all(gpu, q, &mut freed);
        drop_all(gpu, k, &mut freed);
        drop_all(gpu, v, &mut freed);
        drop_all(gpu, o, &mut freed);
        drop_all(gpu, attn_norm, &mut freed);
        drop_all(gpu, wi, &mut freed);
        drop_all(gpu, wi_gate, &mut freed);
        drop_all(gpu, wo, &mut freed);
        drop_all(gpu, ffn_norm, &mut freed);
        gpu.free_tensor(final_norm)
            .expect("t5 gpu: free final_norm");
        freed += 1;
        if let Some((_, b)) = rel_bias {
            gpu.free_tensor(b).expect("t5 gpu: free rel_bias");
            freed += 1;
        }
        freed
    }

    /// Build and upload the `[heads, len, len]` relative bias for this length
    /// if it is not already resident. The bucket math is the host
    /// [`compute_rel_bias`] — one implementation, no GPU copy of it to drift.
    ///
    /// Deliberately returns `()` rather than `&GpuTensor`: a returned
    /// reference would borrow `self` mutably for the whole forward and lock
    /// out every `&self.q[i]` read after it. The caller reads the tensor back
    /// through [`Self::rel_bias`] immediately afterwards.
    fn ensure_rel_bias(
        &mut self,
        gpu: &mut Gpu,
        host: &T5Weights,
        len: usize,
    ) -> Result<(), String> {
        // A different prompt length invalidates the whole table (it is
        // `[heads, len, len]`), so the stale one is freed, not kept.
        if matches!(self.rel_bias, Some((l, _)) if l != len) {
            if let Some((_, t)) = self.rel_bias.take() {
                gpu.free_tensor(t)
                    .map_err(|e| format!("t5 gpu: free stale rel_bias: {e:?}"))?;
            }
        }
        if self.rel_bias.is_none() {
            let heads = self.config.num_heads;
            let bias = compute_rel_bias(
                len,
                len,
                &host.rel_bias,
                self.config.relative_attention_max_distance,
            );
            let t = gpu
                .upload_f32(&bias, &[heads, len, len])
                .map_err(|e| format!("t5 gpu: upload rel_bias: {e:?}"))?;
            self.rel_bias = Some((len, t));
        }
        Ok(())
    }

    /// The resident relative bias, after [`Self::ensure_rel_bias`].
    fn rel_bias(&self) -> &GpuTensor {
        &self
            .rel_bias
            .as_ref()
            .expect("t5 gpu: ensure_rel_bias must run before rel_bias")
            .1
    }
}

/// T5 encoder forward on the GPU. Structural mirror of [`crate::t5::encode`].
///
/// `host` supplies the token-embedding table (and the relative-bias bucket
/// weights); `gw` supplies everything else from the device. Returns the
/// `last_hidden_state` as a **device** f32 tensor `[len, d_model]` — the
/// caller owns it and must free it. Keeping it on the device is the point:
/// it feeds the transformer's `txt_in` with no host round trip.
///
/// `attention_mask` is accepted for signature parity with the CPU reference
/// and, exactly as there, intentionally unused.
pub fn encode(
    gpu: &mut Gpu,
    gw: &mut GpuT5Weights,
    host: &T5Weights,
    input_ids: &[u32],
    attention_mask: &[u8],
) -> Result<GpuTensor, String> {
    let _ = attention_mask;
    let cfg = gw.config.clone();
    let len = input_ids.len();
    let d = cfg.d_model;
    let heads = cfg.num_heads;
    let hd = cfg.d_kv;
    let eps = cfg.layer_norm_epsilon;
    if len == 0 {
        return Err("t5 gpu: empty input_ids".into());
    }
    if heads * hd != d {
        return Err(format!(
            "t5 gpu: heads*d_kv ({}) != d_model ({d}); the CPU reference indexes \
             q/k/v as [pos*d_model + h*d_kv + t] and assumes they match",
            heads * hd
        ));
    }
    // T5 v1.0's ungated `wo(relu(wi x))` FFN has no GPU path — FLUX conditions
    // on T5-XXL **v1.1**, which is gated. Fail before allocating anything
    // rather than silently running the wrong FFN form (which does not error,
    // it just produces a wrong text embedding and a plausible-but-wrong image).
    if gw.wi_gate.is_empty() {
        return Err(
            "t5 gpu: ungated (v1.0) FFN is not implemented on the GPU path — FLUX \
             conditions on T5-XXL v1.1 (gated-GELU). Set HIPFIRE_T5_GPU=0 to run \
             the host encoder for a v1.0 checkpoint."
                .into(),
        );
    }

    // ── token embeddings (host gather, one upload) ───────────────────
    let mut emb = vec![0f32; len * d];
    for (r, &tok) in input_ids.iter().enumerate() {
        let tok = tok as usize;
        if (tok + 1) * d > host.embed.data.len() {
            return Err(format!("t5 gpu: token id {tok} out of embedding range"));
        }
        emb[r * d..(r + 1) * d].copy_from_slice(&host.embed.data[tok * d..(tok + 1) * d]);
    }

    // Relative bias (layer 0's embedding) reused by every layer, as on CPU.
    gw.ensure_rel_bias(gpu, host, len)?;
    let gw: &GpuT5Weights = gw;
    let bias = gw.rel_bias();

    let mut t = TextGpu::new(gpu);
    // Straight-line, like the CPU reference; an error leaks the intermediates
    // allocated so far, which is the same contract `flux_gpu::forward_parts`
    // has (a failed forward is not a recoverable state for the pool anyway).
    {
        let hidden = t.upload(&emb, &[len, d])?;

        for i in 0..cfg.num_layers {
            // ── self-attention (pre-norm) ────────────────────────────
            let normed = t.alloc(&[len, d])?;
            t.rmsnorm(&hidden, &gw.attn_norm[i], &normed, len, d, eps)?;
            let normed_f16 = t.cast_act(&normed, len, d)?;
            let q = t.gemm(&normed_f16, &gw.q[i], None, len, d, d)?;
            let k = t.gemm(&normed_f16, &gw.k[i], None, len, d, d)?;
            let v = t.gemm(&normed_f16, &gw.v[i], None, len, d, d)?;
            t.free(normed_f16)?;
            t.free(normed)?;

            let context = t.alloc(&[len, d])?;
            // scale = 1.0: transformers 5 folds the scaling into the relative
            // bias and does NOT scale the dot product. See `t5::encode`.
            t.attn(
                &q,
                &k,
                &v,
                Some(bias),
                None,
                &context,
                len,
                heads,
                heads,
                hd,
                1.0,
                false,
            )?;
            t.free(q)?;
            t.free(k)?;
            t.free(v)?;

            let ctx_f16 = t.cast_act(&context, len, d)?;
            let attn_out = t.gemm(&ctx_f16, &gw.o[i], None, len, d, d)?;
            t.free(ctx_f16)?;
            t.free(context)?;
            t.add_inplace(&hidden, &attn_out)?;
            t.free(attn_out)?;

            // ── FFN (pre-norm) ───────────────────────────────────────
            let ffn_in = t.alloc(&[len, d])?;
            t.rmsnorm(&hidden, &gw.ffn_norm[i], &ffn_in, len, d, eps)?;
            let ffn_f16 = t.cast_act(&ffn_in, len, d)?;
            let wi_out = t.gemm(&ffn_f16, &gw.wi[i], None, len, cfg.d_ff, d)?;
            let gate = t.gemm(&ffn_f16, &gw.wi_gate[i], None, len, cfg.d_ff, d)?;
            // act = gelu_new(wi_0 x) * (wi_1 x), written in place over wi_out.
            t.gelu_new_mul(&wi_out, &gate, &wi_out, len * cfg.d_ff)?;
            t.free(gate)?;
            let act = wi_out;
            t.free(ffn_f16)?;
            t.free(ffn_in)?;

            let act_f16 = t.cast_act(&act, len, cfg.d_ff)?;
            let wo_out = t.gemm(&act_f16, &gw.wo[i], None, len, d, cfg.d_ff)?;
            t.free(act_f16)?;
            t.free(act)?;
            t.add_inplace(&hidden, &wo_out)?;
            t.free(wo_out)?;
        }

        // last_hidden_state = final_layer_norm(hidden)
        let last = t.alloc(&[len, d])?;
        t.rmsnorm(&hidden, &gw.final_norm, &last, len, d, eps)?;
        t.free(hidden)?;
        Ok(last)
    }
}

/// [`encode`] followed by a download — the shape the CPU reference returns.
/// Only for parity harnesses; the serving path keeps the tensor on-device.
pub fn encode_host(
    gpu: &mut Gpu,
    gw: &mut GpuT5Weights,
    host: &T5Weights,
    input_ids: &[u32],
    attention_mask: &[u8],
) -> Result<Vec<f32>, String> {
    let t = encode(gpu, gw, host, input_ids, attention_mask)?;
    let out = gpu
        .download_f32(&t)
        .map_err(|e| format!("t5 gpu: download last_hidden_state: {e:?}"));
    gpu.free_tensor(t)
        .map_err(|e| format!("t5 gpu: free last_hidden_state: {e:?}"))?;
    out
}

/// Shared scratch/dispatch helper for both text encoders — the same role
/// `Gpuf` plays in [`crate::flux_gpu`], minus the FLUX-specific block bodies.
/// Lives here rather than in a third file so the two encoder mirrors stay
/// one-to-one with their CPU references; [`crate::clip_gpu`] uses it too.
///
/// Every allocation it hands out must be freed explicitly: `GpuTensor` /
/// `DeviceBuffer` have no `Drop`, so a dropped tensor leaks its device
/// allocation for the life of the process.
pub(crate) struct TextGpu<'a> {
    gpu: &'a mut Gpu,
}

impl<'a> TextGpu<'a> {
    pub(crate) fn new(gpu: &'a mut Gpu) -> Self {
        Self { gpu }
    }

    /// Uninitialized `[shape]` f32 — every caller pure-assigns it with the
    /// next kernel, so the `zeros` memset would be pure waste.
    pub(crate) fn alloc(&mut self, shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .alloc_tensor(shape, DType::F32)
            .map_err(|e| format!("text gpu: alloc {shape:?}: {e:?}"))
    }

    pub(crate) fn upload(&mut self, data: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
        self.gpu
            .upload_f32(data, shape)
            .map_err(|e| format!("text gpu: upload {shape:?}: {e:?}"))
    }

    /// Upload `[n]` i32 values into an **F32-typed** `[n]` tensor, bits
    /// unchanged. That cosmetic-dtype slot is the `rope_batched_f32` position
    /// contract — the kernel reads `const int*` — and it is what
    /// `hipfire_runtime::llama` passes there too. Nothing ever reads these
    /// bytes as floats.
    pub(crate) fn upload_i32_bits(&mut self, data: &[i32]) -> Result<GpuTensor, String> {
        let t = self
            .gpu
            .alloc_tensor(&[data.len()], DType::F32)
            .map_err(|e| format!("text gpu: alloc i32 slot [{}]: {e:?}", data.len()))?;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        match self.gpu.hip.memcpy_htod(&t.buf, bytes) {
            Ok(()) => Ok(t),
            Err(e) => {
                let _ = self.gpu.free_tensor(t);
                Err(format!("text gpu: upload i32 slot: {e:?}"))
            }
        }
    }

    pub(crate) fn free(&mut self, t: GpuTensor) -> Result<(), String> {
        self.gpu
            .free_tensor(t)
            .map_err(|e| format!("text gpu: free: {e:?}"))
    }

    /// Cast an f32 activation `[m, k]` to the f16 scratch the WMMA GEMM
    /// wants. Split out so one activation feeding several GEMMs (q/k/v off
    /// the same norm; `wi`/`wi_gate` off the same FFN norm) is cast once.
    pub(crate) fn cast_act(
        &mut self,
        a: &GpuTensor,
        m: usize,
        k: usize,
    ) -> Result<GpuTensor, String> {
        let f16 = self
            .gpu
            .alloc_tensor(&[m, k], DType::F16)
            .map_err(|e| format!("text gpu: alloc act f16: {e:?}"))?;
        self.gpu
            .cast_f32_to_f16(a, &f16)
            .map_err(|e| format!("text gpu: cast act f16: {e:?}"))?;
        Ok(f16)
    }

    /// `y[m, out] = x_f16[m, k] · Wᵀ + bias`, `w` f16 `[out, k]`.
    ///
    /// Prefers the LDS macro-tile kernel (K % 64 == 0, bias fused) and falls
    /// back to the 16-step kernel plus a separate `bias_add` for a ragged K —
    /// the same routing `Gpuf::gemm_pre` uses. Real geometry never takes the
    /// fallback (T5-XXL 4096/10240, CLIP-L 768/3072 are all K % 64 == 0); a
    /// tiny synthetic fixture can.
    pub(crate) fn gemm(
        &mut self,
        x_f16: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        m: usize,
        out: usize,
        k: usize,
    ) -> Result<GpuTensor, String> {
        if k == 0 || k % 16 != 0 {
            return Err(format!(
                "text gpu: wmma gemm [{m}x{k}]·[{out}x{k}]: K must be a multiple of 16 (got {k})"
            ));
        }
        if w.shape.len() != 2 || w.shape[0] != out || w.shape[1] != k {
            return Err(format!(
                "text gpu: weight shape {:?} is not [{out}, {k}]",
                w.shape
            ));
        }
        let y = self.alloc(&[m, out])?;
        if k % 64 == 0 {
            self.gpu
                .gemm_f16_x_f16_wmma_lds_auto(w, x_f16, &y, bias, out, k, m)
                .map_err(|e| format!("text gpu: wmma lds gemm [{m}x{k}]·[{out}x{k}]: {e:?}"))?;
            return Ok(y);
        }
        self.gpu
            .gemm_f16_x_f16_wmma(w, x_f16, &y, out, k, m)
            .map_err(|e| format!("text gpu: wmma gemm [{m}x{k}]·[{out}x{k}]: {e:?}"))?;
        if let Some(b) = bias {
            self.gpu
                .bias_add_f32(&y, b, m, out)
                .map_err(|e| format!("text gpu: bias_add: {e:?}"))?;
        }
        Ok(y)
    }

    pub(crate) fn rmsnorm(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.gpu
            .rmsnorm_batched(x, weight, out, rows, d, eps)
            .map_err(|e| format!("text gpu: rmsnorm: {e:?}"))
    }

    pub(crate) fn layernorm(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        out: &GpuTensor,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.gpu
            .layernorm_batched(x, gamma, beta, out, rows, d, eps)
            .map_err(|e| format!("text gpu: layernorm: {e:?}"))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        bias: Option<&GpuTensor>,
        key_mask: Option<&GpuTensor>,
        out: &GpuTensor,
        n: usize,
        heads: usize,
        n_kv_heads: usize,
        hd: usize,
        scale: f32,
        causal: bool,
    ) -> Result<(), String> {
        self.gpu
            .attention_text_f32(
                q, k, v, bias, key_mask, out, n, heads, n_kv_heads, hd, scale, causal,
            )
            .map_err(|e| format!("text gpu: attention: {e:?}"))
    }

    pub(crate) fn gelu_new_mul(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .gelu_new_mul_f32(a, b, out, n)
            .map_err(|e| format!("text gpu: gelu_new_mul: {e:?}"))
    }

    pub(crate) fn quick_gelu(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .quick_gelu_f32(x, out, n)
            .map_err(|e| format!("text gpu: quick_gelu: {e:?}"))
    }

    /// In-place half-split RoPE over `q` `[n, heads*hd]` and `k`
    /// `[n, kv_heads*hd]`, GQA-native. `positions` is the F32-typed i32 slot
    /// [`Self::upload_i32_bits`] builds. Used by [`crate::qwen3_gpu`]; T5 and
    /// CLIP have no RoPE.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        positions: &GpuTensor,
        heads: usize,
        kv_heads: usize,
        hd: usize,
        theta: f32,
        n: usize,
    ) -> Result<(), String> {
        self.gpu
            .rope_batched_f32(q, k, positions, heads, kv_heads, hd, theta, n)
            .map_err(|e| format!("text gpu: rope: {e:?}"))
    }

    /// SwiGLU term `out[i] = silu(gate[i]) * up[i]`. `out` may alias `gate`.
    pub(crate) fn silu_mul(
        &mut self,
        gate: &GpuTensor,
        up: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), String> {
        self.gpu
            .silu_mul_f32(gate, up, out)
            .map_err(|e| format!("text gpu: silu_mul: {e:?}"))
    }

    /// `dst[r*dst_row_stride + dst_col + c] = src[r*src_row_stride + c]` for
    /// `n_rows × len` — one launch. The tap-assemble of
    /// [`crate::qwen3_gpu::encode_taps`], and the same helper
    /// `flux_gpu::assemble_rows` uses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy_rows(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n_rows: usize,
        len: usize,
        src_row_stride: usize,
        dst_row_stride: usize,
        dst_col_offset: usize,
    ) -> Result<(), String> {
        self.gpu
            .copy_rows_strided_f32(
                src,
                dst,
                n_rows,
                len,
                src_row_stride,
                dst_row_stride,
                dst_col_offset,
            )
            .map_err(|e| format!("text gpu: copy_rows @col {dst_col_offset}: {e:?}"))
    }

    pub(crate) fn add_inplace(&mut self, a: &GpuTensor, b: &GpuTensor) -> Result<(), String> {
        self.gpu
            .add_inplace_f32(a, b)
            .map_err(|e| format!("text gpu: add_inplace: {e:?}"))
    }

    pub(crate) fn download(&self, t: &GpuTensor) -> Result<Vec<f32>, String> {
        self.gpu
            .download_f32(t)
            .map_err(|e| format!("text gpu: download: {e:?}"))
    }
}
