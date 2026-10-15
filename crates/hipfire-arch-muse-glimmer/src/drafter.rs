// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! # STATUS: working — 1.24x over AR, byte-identical (measured 2026-08-11)
//!
//! hiptrx gfx1201 (R9700), 64 tokens greedy, 3 fresh processes per arm, target
//! `muse-glimmer-30b.mq4`, prompt md5 `2ef49ee70df1483079b1f73c1f768339`:
//!
//! | mode | tok/s (3 runs) | median | tau |
//! |---|---|---:|---:|
//! | AR | 32.96 · 32.94 · 32.94 | 32.94 | — |
//! | DFlash | 41.03 · 41.00 · 40.95 | **41.00** | **8.333** |
//!
//! 66 of 135 proposals accepted over 9 windows. Output byte-identical to AR at
//! temp 0, which is the required contract: acceptance is greedy-argmax, so any
//! divergence would be a bug rather than an acceptance-rate matter.
//!
//! Still opt-in via `HIPFIRE_DFLASH_DRAFT` (repo default is `dflash_mode=off`).
//!
//! ## What it took, so the next reader does not repeat it
//!
//! Four separate defects had to be cleared, and every one of them left the
//! engine *running and producing correct text* — only the acceptance rate
//! revealed them. In discovery order:
//!
//! 1. **Noise embedding never filled.** The drafter's whole input was a
//!    zero-filled `vec![0f32; block*hidden]`. Drafts decoded to token 0.
//! 2. **No attention.** The per-layer loop copied Q straight into `attn_out`
//!    behind a comment reading "for minimal, just do o_proj over q". Every
//!    block row was then bit-identical, capping acceptance at 1 per window.
//! 3. **Wrong block structure.** The drafter is a standard two-norm Llama block
//!    (58 tensors: no pre/post-FFN norm), not the target's four-norm sandwich.
//! 4. **Context delivered through the wrong pathway** — the big one. Upstream
//!    CONCATENATES the projected context into K/V, so K/V spans `ctx+block`
//!    while Q spans `block`. This code instead broadcast-ADDED a single context
//!    row into `x` and attended over the block alone. Fixing it moved tau from
//!    1.016 to 8.333 in one step.
//!
//! The authority for (4) is upstream `modeling_muse_glimmer_assistant.py`, whose
//! attention comment states it outright: *"The total k/v states in Dflash are the
//! concatenation of the previous `context_hidden_states` ... and the actual
//! projections on the diffusion window."* Guessing cost several rounds; reading
//! it cost one.
//!
//! Verify is a single BATCHED forward over the block, not B sequential decodes.
//! That distinction is the entire economics of speculative decode: the
//! sequential version streamed all 15.5 GB of weights 16 times per window and
//! ran at 12.0 tok/s — *slower than AR* — at the very same tau 8.333.
//!
//! Muse Glimmer DFlash drafter (`model_type = muse_glimmer_assistant`, arch_id = 23).
//!
//! A 5-layer block-diffusion draft head for the arch-14 Glimmer target.
//! It is NOT a standalone LM: every draft step reuses the TARGET's vocab table
//! and the TARGET's lm_head. Architecture (traced against the HF safetensors
//! `meta-models/Muse-Glimmer-30B-assistant` — 58 tensors, no embed/lm_head):
//!
//!   target_hidden_proj = output_norm_enc( fc · target_hidden )  // once, all ctx rows
//!   // fc: [hidden, num_extract*hidden]; target_hidden rows = previously accepted tokens
//!   x = noise_embedding (raw target.embed_tokens([seed, MASK×15]), no embed_norm)
//!   // context is NOT added into x — it is concatenated into K/V only
//!   for each drafter layer (5×, RoPE θ=500000, window 2048, GQA 32/8, hd 128):
//!     // STANDARD two-norm Llama block — NOT the target's four-norm sandwich:
//!     residual = x
//!     n1 = rmsnorm(x, input_layernorm, 1e-5) → tmp
//!     q = q_proj(n1)                         // BLOCK only            → B rows
//!     k = k_proj(cat[target_hidden_proj, n1]) // CONTEXT ++ BLOCK      → ctx+B rows
//!     v = v_proj(cat[target_hidden_proj, n1]) // CONTEXT ++ BLOCK      → ctx+B rows
//!     q = q_norm(q); k = k_norm(k)   // per-head WEIGHTED RMSNorm (real q_norm/k_norm weights)
//!     RoPE half-split on Q (block positions) and K (full ctx+block span)
//!     attn_out = attention_dflash_f32(Q[B], K/V[ctx+B])  // bidirectional, GQA, f32
//!     attn = o_proj(attn_out); x = residual + attn
//!     residual = x
//!     n2 = rmsnorm(x, post_attention_layernorm, 1e-5)  // <-- IS the pre-FFN norm
//!     ffn = down(silu(gate(n2))*up(n2)); x = residual + ffn
//!   n = norm(x) → logits = n · target.lm_head.T → argmax
//!
//! Shape (confirmed from artifact / GlimmerDrafterConfig::from_hfq):
//!   n_layers=5, hidden=6656, intermediate=19968, n_heads=32, n_kv_heads=8,
//!   head_dim=128, q_dim=4096, kv_dim=1024, GQA group=4, SWA=2048 on all layers, block=16.
//!
//! Extent decision: context is CONCATENATED into K/V (not broadcast-added into x).
//!   - Q length = block (B), K/V length = ctx_len + block, positions span ctx+B.
//!   - `target_hidden_proj` is computed once via encoder.fc + output_norm_enc over
//!     every accepted ctx row and reused by all 5 layers.
//!   - Scratch k/v are sized `(max_ctx+block)*kv_dim`; q stays `block*q_dim`.
//!
//! Helper choice: `Gpu::attention_dflash_f32` (f32 K/V, GQA, bidirectional,
//! no causal mask). Rejected `attention_q8_0_kv_swa`/`attention_q8_0_kv` — they
//! require a Q8 quantized KV cache, single-query decode shape, and a causal
//! windowed contract; the draft's K/V lives in F32 scratch as [(ctx+B)×kvd] and
//! the block-diffusion contract needs many queries in parallel. Rejected
//! `attention_f32`/`attention_flash*` single-query variants for the same reason.
//! `attention_dflash_f32` matches dtype (f32), layout ([B×q_dim], [L×kvd]),
//! GQA (32/8), and masking (non-causal, bidirectional).
//!
//! Masking / window approximation: upstream layers are all `sliding_attention`
//! with window 2048 and build a bidirectional sliding-window mask. Queries
//! attend bi-directionally within the block and (windowedly) to prior ctx K/V.
//! `attention_dflash_f32` is FULL bidirectional over L=ctx+B — exact while
//! `ctx+B <= 2048`, and over-attends (no window cutoff) beyond that. No
//! windowed bidirectional kernel exists; this is the one real approximation
//! and is stated, not hidden. No new HIP kernel is introduced.
//!
//! Critical embed_norm contract (see `forward.rs:84` and
//! `/tmp/modeling_muse_glimmer.py:439`): the DFlash block's `noise_embedding`
//! is **raw** `target.embed_tokens([seed, MASK×15])` with NO
//! `embed_norm` (scale-less RMSNorm). The AR path at `forward::embed_lookup`
//! DOES apply it; the DFlash path deliberately does not.
//!
//! ## Logit calibration: ordering is good, magnitude is not
//!
//! The drafter's logits rank tokens well but carry almost no dynamic range, so
//! their softmax is NOT a usable probability distribution. Measured 2026-08-14
//! on hiptrx gfx1201, `muse-glimmer-30b.mq4` + `muse-glimmer-30b-dflash.mq4`,
//! serve battery at registry sampling (temp 1.0), via a temporary accept-input
//! dump (branch `glimmer-rejection-debug`, `HIPFIRE_GLIMMER_ACCEPT_DIAG=1`):
//!
//! | quantity | draft row | target row |
//! |---|---|---|
//! | row sum | 1.0000 | 1.0000 |
//! | argmax token | 19669 | 19669 (same) |
//! | probability at argmax | **0.0007** | **0.9998** |
//!
//! Rows are correctly aligned (draft row `i` and target row `i` both predict
//! `drafts[i]`, and they agree on the argmax across windows: 19669, 200023,
//! 10064). Both sum to 1. The target is near-deterministic while the draft is
//! nearly flat — a uniform distribution over the 202048-token vocab would put
//! 4.95e-6 on each token, and the draft's PEAK is 7e-4.
//!
//! Mechanism: the draft path norms `scratch.x` with the DRAFTER's `norm` and
//! then applies the TARGET's `lm_head`, skipping the target's `final_norm`
//! scaling that `forward.rs`'s verify path applies before its own lm_head.
//! Right ordering with no contrast is the fingerprint of a scale mismatch, not
//! of genuine drafter uncertainty.
//!
//! This costs nothing today because every shipped consumer is scale-invariant:
//! greedy DFlash takes the argmax, and the sampled path
//! (`hipfire_runtime::ddtree::naive_sample_chain`) accepts on `p_t(draft)` drawn
//! from the TARGET's distribution. It matters if you write something that
//! treats the drafter's softmax as a distribution. Speculative REJECTION
//! sampling does exactly that, and it fails hard: drafts sampled from the flat
//! `q` are effectively random, the target assigns them 1e-14..1e-8, and
//! `u * q(draft) <= p(draft)` is never satisfied — measured `accepted 0` of 135
//! proposals, tau 1.000, 27.2 tok/s against 28.1 on plain AR, with coherent
//! output the whole time, so it fails SILENTLY.
//!
//! Two traps if you try to fix it:
//!   - A positive scalar rescale leaves argmax (and therefore greedy) untouched,
//!     but applying the target's `final_norm` is a per-channel transform and
//!     WOULD change the argmax. Re-validate greedy tau before believing any
//!     recalibration.
//!   - Calibration alone buys no speculative throughput. Where the target sits
//!     at 0.9998, naive chain-sample already accepts at ~that rate, and a
//!     calibrated `min(1, p/q)` converges to the same value. The measured gap
//!     (sampled 1.84x vs greedy 3.18x over AR) comes from temperature-1.0
//!     sampling drawing a non-argmax token and breaking the chain, which no
//!     lossless accept rule can recover — only a better-aligned drafter can.
//!
//! The forward is one typed `hipfire_dispatch` step program per layer
//! (`layer_steps`); the context-row watermarks, staging and the window
//! gather bounds stay here.

use crate::glimmer::GlimmerHiddenLog;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::sandwich::{
    gemm_rows, Activation, DraftContext, KvProjection, KvRoute, Ladder, Rope, RopeKind,
    SandwichAttentionOp, SandwichKv, SandwichMlpOp, SandwichRoute, SandwichStream, StepsRoute,
};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, RowsRoute, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::{DType, Gpu, GpuTensor};

pub const GLIMMER_DRAFTER_ARCH_ID: u32 = 23;

/// Daemon/load default for `HIPFIRE_GLIMMER_CTX_CAP` when unset.
/// Sampled once at carrier load and passed into scratch + device hidden log.
/// Demo may choose a distinct default independently.
///
/// This is a sliding suffix window, not a hard context limit: once `cur_pos`
/// exceeds it, `daemon.rs` pins `ctx_len` here and advances
/// `cur_start = cur_pos - ctx_len`, so the drafter always sees the most recent
/// rows. Rows sliding out leave the drafter's attention only — the target's KV
/// is untouched and still verifies every token, so this bounds tau, never
/// correctness.
///
/// 512 measured on hiptrx gfx1201, muse-glimmer-30b.mq4 (Q8 attention) + the
/// mq4 drafter, greedy so each point is deterministic:
///   - Long context (1024-token prompt fixture, cap is binding):
///     64 -> 29.7 tok/s (tau 2.617) | 128 -> 31.7 (2.841) | 256 -> 50.7 (4.741)
///     512 -> 55.9 (5.375) | 768 -> 48.5 (4.640) | 1024 -> 51.8 (5.160)
///     4096 -> 39.2 (5.647)
///     tau rises monotonically with the window, but per-window cost (the
///     `ctx_len*kv_dim` D2D gather x2 x5 layers, plus attention over
///     `ctx_len+block`) outruns it past ~512.
///   - Short prompts (genre battery, <=256 generated): 256/512/1024 are
///     IDENTICAL (tau 8.0/10.048/5.25/4.017/3.919, ~89 tok/s) because
///     `ctx_len = min(cur_pos, cap)` never reaches the cap.
/// So 512 is free in the short regime and ~+10% in the long one. `fc_input`
/// VRAM is `cap * num_extract * hidden * 4` = 68 MB here (34 MB at 256,
/// 545 MB at 4096). The surface is bumpy (768 lands below 256), so treat this
/// as the measured best of the candidates rather than a smooth optimum.
pub const GLIMMER_DRAFTER_CTX_CAP_DEFAULT: usize = 512;

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GlimmerDrafterConfig {
    pub n_layers: usize,              // 5
    pub hidden: usize,                // 6656
    pub intermediate: usize,          // 19968
    pub n_heads: usize,               // 32
    pub n_kv_heads: usize,            // 8
    pub head_dim: usize,              // 128
    pub norm_eps: f32,                // 1e-5
    pub rope_theta: f32,              // 500000.0
    pub sliding_window: usize,        // 2048
    pub block_size: usize,            // 16
    pub mask_token_id: u32,           // 201818
    pub target_layer_ids: Vec<usize>, // [1,13,25,37,49]
}

impl GlimmerDrafterConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("glimmer drafter: metadata_json not valid JSON: {e}"))?;
        let cfg = meta
            .get("config")
            .ok_or_else(|| "glimmer drafter: metadata_json missing `config` wrapper".to_string())?;
        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());

        let hidden =
            getu(cfg, "hidden_size").ok_or("glimmer drafter: missing hidden_size")? as usize;
        let n_layers = getu(cfg, "num_hidden_layers")
            .ok_or("glimmer drafter: missing num_hidden_layers")? as usize;
        let intermediate = getu(cfg, "intermediate_size")
            .ok_or("glimmer drafter: missing intermediate_size")?
            as usize;
        let n_heads = getu(cfg, "num_attention_heads")
            .ok_or("glimmer drafter: missing num_attention_heads")? as usize;
        let n_kv_heads = getu(cfg, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let head_dim = getu(cfg, "head_dim")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        let norm_eps = getf(cfg, "rms_norm_eps").unwrap_or(1e-5) as f32;
        let rope_theta = cfg
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(500000.0) as f32;
        let sliding_window = getu(cfg, "sliding_window").unwrap_or(2048) as usize;
        let block_size =
            getu(cfg, "block_size").ok_or("glimmer drafter: missing block_size")? as usize;
        let mask_token_id =
            getu(cfg, "mask_token_id").ok_or("glimmer drafter: missing mask_token_id")? as u32;
        let target_layer_ids: Vec<usize> = cfg
            .get("target_layer_ids")
            .and_then(|v| v.as_array())
            .ok_or("glimmer drafter: missing target_layer_ids")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        if target_layer_ids.len() != 5 {
            return Err(format!(
                "glimmer drafter: target_layer_ids len {} != 5",
                target_layer_ids.len()
            ));
        }
        if target_layer_ids != vec![1, 13, 25, 37, 49] {
            eprintln!(
                "glimmer drafter: WARNING target_layer_ids {:?} != expected [1,13,25,37,49]",
                target_layer_ids
            );
        }

        Ok(GlimmerDrafterConfig {
            n_layers,
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            norm_eps,
            rope_theta,
            sliding_window,
            block_size,
            mask_token_id,
            target_layer_ids,
        })
    }

    #[inline]
    pub fn num_extract(&self) -> usize {
        self.target_layer_ids.len()
    }
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

// ─── Load helpers ───────────────────────────────────────────────────────

fn load_f32_vec(hfq: &HfqFile, name: &str, expected: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer drafter: tensor '{name}' not found"))?;
    if info.shape.iter().fold(1usize, |a, &b| a * b as usize) != expected {
        return Err(format!(
            "glimmer drafter: tensor '{name}' shape {:?} != expected {}",
            info.shape, expected
        ));
    }
    match info.quant_type {
        1 => Ok(data
            .chunks_exact(2)
            .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        2 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        16 => Ok(data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        qt => Err(format!(
            "glimmer drafter: unsupported quant_type {qt} for F32 tensor '{name}'"
        )),
    }
}

fn load_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize) -> Result<GpuTensor, String> {
    let v = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&v, &[dim])
        .map_err(|e| format!("glimmer drafter: upload norm '{name}': {e:?}"))
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("glimmer drafter: tensor '{name}' not found"))?;
    let mut wt = match info.quant_type {
        3 => {
            let buf = gpu
                .upload_raw(data, &[data.len()])
                .map_err(|e| format!("glimmer drafter: upload Q8 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::Q8_0,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        // MQ4G256 (13) and its Lloyd-codebook sibling (19). The drafter's
        // forward already projects MQ4 through the FWHT rotation (the batched
        // `GemmRows` steps); only this loader arm was
        // missing, so an MQ4 drafter loaded fine everywhere except here and
        // DFlash silently fell back to AR with
        // "unsupported quant_type 13 for 'encoder.fc.weight'".
        //
        // Qwen's DFlash drafters have always been MQ4 (arch 20); Glimmer's was
        // the only Q8 one, at 2.59 GB against qwen35-27b's 0.88 GB for the same
        // 58-tensor / 36-weight shape.
        13 | 19 => {
            let buf = gpu
                .upload_raw(data, &[data.len()])
                .map_err(|e| format!("glimmer drafter: upload MQ4 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        1 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m * k])
                .map_err(|e| format!("glimmer drafter: upload F16->F32 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m * k])
                .map_err(|e| format!("glimmer drafter: upload F32 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        16 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let buf = gpu
                .upload_f32(&f32_data, &[m * k])
                .map_err(|e| format!("glimmer drafter: upload BF16->F32 '{name}': {e:?}"))?;
            WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            }
        }
        qt => {
            return Err(format!(
                "glimmer drafter: unsupported quant_type {qt} for '{name}'"
            ))
        }
    };
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = hipfire_runtime::hfq::load_awq_scale(hfq, gpu, name, k);
    }
    Ok(wt)
}

// ─── Weights ────────────────────────────────────────────────────────────

pub struct GlimmerDrafterLayer {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub q_proj: WeightTensor,
    pub k_proj: WeightTensor,
    pub v_proj: WeightTensor,
    pub o_proj: WeightTensor,
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,
}

pub struct GlimmerDrafterWeights {
    pub fc: WeightTensor,           // encoder.fc.weight [hidden, num_extract*hidden]
    pub output_norm_enc: GpuTensor, // encoder.output_norm_enc.weight
    pub norm: GpuTensor,            // norm.weight
    pub layers: Vec<GlimmerDrafterLayer>,
}

impl GlimmerDrafterWeights {
    pub fn load(hfq: &HfqFile, cfg: &GlimmerDrafterConfig, gpu: &mut Gpu) -> Result<Self, String> {
        if hfq.arch_id != GLIMMER_DRAFTER_ARCH_ID {
            return Err(format!(
                "glimmer drafter: expected arch_id {} (muse_glimmer_assistant), got {}",
                GLIMMER_DRAFTER_ARCH_ID, hfq.arch_id
            ));
        }
        let ne = cfg.num_extract();
        let h = cfg.hidden;
        // All-or-nothing: every GPU allocation created during this call is
        // released on Err, ownership transfers exactly once on Ok.
        // Free via immediate path so a pending async memset cannot race the pool.
        fn free_weight_immediate(gpu: &mut Gpu, w: WeightTensor) {
            if let Some(paro) = w.paro {
                if !paro.is_alias {
                    let _ = gpu.release_tensor_immediate(paro.pairs);
                    let _ = gpu.release_tensor_immediate(paro.theta);
                    let _ = gpu.release_tensor_immediate(paro.channel_scales);
                }
            }
            if let Some(awq) = w.awq_scale {
                let _ = gpu.release_tensor_immediate(awq);
            }
            let _ = gpu.release_tensor_immediate(w.buf);
        }
        struct Partial<'a> {
            gpu: &'a mut Gpu,
            fc: Option<WeightTensor>,
            output_norm_enc: Option<GpuTensor>,
            norm: Option<GpuTensor>,
            layers: Vec<GlimmerDrafterLayer>,
        }
        impl Drop for Partial<'_> {
            fn drop(&mut self) {
                if let Some(w) = self.fc.take() {
                    free_weight_immediate(self.gpu, w);
                }
                if let Some(t) = self.output_norm_enc.take() {
                    let _ = self.gpu.release_tensor_immediate(t);
                }
                if let Some(t) = self.norm.take() {
                    let _ = self.gpu.release_tensor_immediate(t);
                }
                for l in self.layers.drain(..) {
                    let _ = self.gpu.release_tensor_immediate(l.input_layernorm);
                    let _ = self
                        .gpu
                        .release_tensor_immediate(l.post_attention_layernorm);
                    let _ = self.gpu.release_tensor_immediate(l.q_norm);
                    let _ = self.gpu.release_tensor_immediate(l.k_norm);
                    free_weight_immediate(self.gpu, l.q_proj);
                    free_weight_immediate(self.gpu, l.k_proj);
                    free_weight_immediate(self.gpu, l.v_proj);
                    free_weight_immediate(self.gpu, l.o_proj);
                    free_weight_immediate(self.gpu, l.gate_proj);
                    free_weight_immediate(self.gpu, l.up_proj);
                    free_weight_immediate(self.gpu, l.down_proj);
                }
            }
        }
        let mut p = Partial {
            gpu,
            fc: None,
            output_norm_enc: None,
            norm: None,
            layers: Vec::with_capacity(cfg.n_layers),
        };
        p.fc = Some(load_wt(hfq, &mut *p.gpu, "encoder.fc.weight", h, ne * h)?);
        p.output_norm_enc = Some(load_norm(
            hfq,
            &mut *p.gpu,
            "encoder.output_norm_enc.weight",
            h,
        )?);
        p.norm = Some(load_norm(hfq, &mut *p.gpu, "norm.weight", h)?);
        for i in 0..cfg.n_layers {
            let prefix = format!("layers.{i}");
            let input_layernorm = load_norm(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.input_layernorm.weight"),
                h,
            )?;
            let post_attention_layernorm = match load_norm(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.post_attention_layernorm.weight"),
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    return Err(e);
                }
            };
            let q_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.q_proj.weight"),
                cfg.q_dim(),
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    return Err(e);
                }
            };
            let k_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.k_proj.weight"),
                cfg.kv_dim(),
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    return Err(e);
                }
            };
            let v_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.v_proj.weight"),
                cfg.kv_dim(),
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    return Err(e);
                }
            };
            let o_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.o_proj.weight"),
                h,
                cfg.q_dim(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    return Err(e);
                }
            };
            let q_norm = match load_norm(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.q_norm.weight"),
                cfg.head_dim,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    free_weight_immediate(p.gpu, o_proj);
                    return Err(e);
                }
            };
            let k_norm = match load_norm(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.self_attn.k_norm.weight"),
                cfg.head_dim,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    free_weight_immediate(p.gpu, o_proj);
                    let _ = p.gpu.release_tensor_immediate(q_norm);
                    return Err(e);
                }
            };
            let gate_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.mlp.gate_proj.weight"),
                cfg.intermediate,
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    free_weight_immediate(p.gpu, o_proj);
                    let _ = p.gpu.release_tensor_immediate(q_norm);
                    let _ = p.gpu.release_tensor_immediate(k_norm);
                    return Err(e);
                }
            };
            let up_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.mlp.up_proj.weight"),
                cfg.intermediate,
                h,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    free_weight_immediate(p.gpu, o_proj);
                    let _ = p.gpu.release_tensor_immediate(q_norm);
                    let _ = p.gpu.release_tensor_immediate(k_norm);
                    free_weight_immediate(p.gpu, gate_proj);
                    return Err(e);
                }
            };
            let down_proj = match load_wt(
                hfq,
                &mut *p.gpu,
                &format!("{prefix}.mlp.down_proj.weight"),
                h,
                cfg.intermediate,
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = p.gpu.release_tensor_immediate(input_layernorm);
                    let _ = p.gpu.release_tensor_immediate(post_attention_layernorm);
                    free_weight_immediate(p.gpu, q_proj);
                    free_weight_immediate(p.gpu, k_proj);
                    free_weight_immediate(p.gpu, v_proj);
                    free_weight_immediate(p.gpu, o_proj);
                    let _ = p.gpu.release_tensor_immediate(q_norm);
                    let _ = p.gpu.release_tensor_immediate(k_norm);
                    free_weight_immediate(p.gpu, gate_proj);
                    free_weight_immediate(p.gpu, up_proj);
                    return Err(e);
                }
            };
            p.layers.push(GlimmerDrafterLayer {
                input_layernorm,
                post_attention_layernorm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                gate_proj,
                up_proj,
                down_proj,
            });
        }
        let fc = p.fc.take().expect("fc present");
        let output_norm_enc = p.output_norm_enc.take().expect("output_norm_enc present");
        let norm = p.norm.take().expect("norm present");
        let layers = std::mem::take(&mut p.layers);
        std::mem::forget(p);
        Ok(GlimmerDrafterWeights {
            fc,
            output_norm_enc,
            norm,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.fc.free_all(gpu);
        let _ = gpu.free_tensor(self.output_norm_enc);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.input_layernorm);
            let _ = gpu.free_tensor(l.post_attention_layernorm);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            l.q_proj.free_all(gpu);
            l.k_proj.free_all(gpu);
            l.v_proj.free_all(gpu);
            l.o_proj.free_all(gpu);
            l.gate_proj.free_all(gpu);
            l.up_proj.free_all(gpu);
            l.down_proj.free_all(gpu);
        }
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────

pub struct GlimmerDrafterScratch {
    pub x: GpuTensor, // [block*hidden] — noise + evolving hidden (no ctx add)
    pub target_hidden_proj: GpuTensor, // [max_ctx * hidden] — ctx rows, reused by all layers
    pub q: GpuTensor, // [block * q_dim]  — Q is block-only
    pub k: GpuTensor, // [(max_ctx + block) * kv_dim] — ctx ++ block
    pub v: GpuTensor, // [(max_ctx + block) * kv_dim] — ctx ++ block
    pub attn_out: GpuTensor, // [block * q_dim]
    pub tmp: GpuTensor, // [block*hidden] scratch
    pub gate_ffn: GpuTensor,
    pub up_ffn: GpuTensor,
    pub ffn_hidden: GpuTensor,
    pub logits_tmp: GpuTensor, // [hidden] for final norm
    /// Batched GEMM rotation scratch.
    pub x_rot: GpuTensor, // [block * hidden] — shared rotation for q / gate/up
    /// `[ctx_capacity * num_extract * hidden]` — FWHT-rotated `fc_input` rows.
    pub fc_rot: GpuTensor,
    pub kv_input_rot: GpuTensor, // [(max_ctx+block)*hidden] — FWHT-rotated ctx rows for K/V
    pub ffn_hidden_rot: GpuTensor, // [block*intermediate] — FWHT-rotated ffn_hidden for down_proj
    /// Persistent buffer for encoder.fc input: [max_ctx * num_extract * hidden]
    /// Host `target_hidden` (ctx*ne*h) is uploaded into this buffer each forward
    /// via a sub-offset view; no per-forward alloc. Sized to the configured cap
    /// (max_ctx) to mirror `hipfire-runtime/src/dflash.rs:DflashScratch::target_hidden`
    /// which Qwen sizes once to max_ctx at construction and reuses via `target_hidden`
    /// + `thlog` rather than allocating per forward. Glimmer's daemon caps
    /// ctx_len to `sliding_window` (2048) but scratch is sized to max_seq for
    /// completeness — same trade-off Qwen makes with DEFAULT_DFLASH_CTX_CAP=8192.
    pub fc_input: GpuTensor, // [max_ctx * num_extract * hidden]
    // ── Incremental fc cache (mirrors dflash.rs::TargetHiddenLog) ───────────
    // Measured motivation (gfx1100, Muse Glimmer 30B, greedy,
    // HIPFIRE_GLIMMER_TIMING=1, per-window):
    //   ctx_len 97  -> drafter  4.4ms | draft_lm 19.9ms | verify 62.8ms
    //   ctx_len 2048 -> drafter 124.4ms | draft_lm 32.6ms | verify 151.3ms
    // The drafter term grows 28× (4.4 -> 124.4 ms) and is the single largest
    // window cost. Cost is: H2D of ctx_len*num_extract*hidden f32 (~273 MB at
    // cap 2048, ne=5, h=6656) + proj_gemm_batched(..., b=ctx_len, "fc")
    // GEMM M=2048,N=6656,K=33280 (~9e11 FLOP) per window + full K/V build over
    // ctx_len+block each window. This mirrors dflash.rs::TargetHiddenLog
    // (550-696, scratch.thlog at 790, delta-H2D 1668-1717, delta-proj
    // 1754-1775, K/V fill 1854-1893): only NEW rows are uploaded/projected.
    // Watermarks:
    //   fc_uploaded_rows  ↔ TargetHiddenLog::uploaded_rows
    //   fc_projected_rows ↔ TargetHiddenLog::proj_cached_rows
    //   fc_window_start   ↔ TargetHiddenLog::abs_positions[0] (when non-empty)
    // Invariant: cached prefix [0..fc_projected_rows) is valid iff window
    // start is unchanged and ctx_len >= fc_projected_rows. Sliding-window
    // invalidation (daemon.rs:30671-30680 `ctx_len = n_rows.min(ctx_cap)`
    // + `start = (n_rows - ctx_len)*row_elems`) detects suffix slide and
    // resets to 0 (full recompute). Correct partial win; K/V remains full.
    pub fc_uploaded_rows: usize,
    pub fc_projected_rows: usize,
    /// Absolute position of ctx row 0 in the cached prefix (positions[0] when
    /// fc_uploaded_rows>0). Tracks suffix slide once n_rows exceeds
    /// sliding_window=2048; a naive row-count watermark is WRONG in that
    /// regime (row 0 is a different token) and would silently corrupt drafts.
    pub fc_window_start: i32,
    // ── Incremental K/V cache (absolute-position-keyed) ────────────────────
    // Qwen's `dflash.rs` fills `k_ctx_cached`/`v_ctx_cached` only past a
    // watermark (1854-1893) and keys by absolute position; Glimmer now mirrors
    // that. The drafter's attention previously rebuilt K/V over the entire
    // `ctx_len + block` span every window (5 layers × 2 GEMMs over 2048 rows)
    // — the other half of the 28× blowup. This cache makes it incremental:
    // each layer keeps `k_ctx_cached`/`v_ctx_cached` sized to `max_ctx*kv_dim`
    // (absolute-indexed, direct, no ring wrap until max_seq) and `kv_abs_end`
    // (next absolute position to fill, ↔ TargetHiddenLog::proj_cached_rows
    // but absolute). On a forward-extension window only the tail
    // `[kv_abs_end .. cur_end)` is projected; the prefix `[cur_start .. cur_end)`
    // is gathered from the cache. Sliding-window suffix movement does NOT
    // invalidate — it just selects a different absolute range of an already
    // populated cache, as a normal KV cache does. Invalidate only on
    // genuine rollback (`cur_end < kv_abs_end`) or conversation reset
    // (`ctx_len==0` or `cur_start==0` with small `cur_end`). VRAM at cap
    // 2048: 2048*1024*4=8 MiB per tensor, 16 MiB per layer, 80 MiB for 5
    // layers (k+v). At max_seq 8192 direct: 8192*1024*4=32 MiB per tensor,
    // 64 MiB per layer, 320 MiB for 5 layers — still < 2% of a 24 GiB card
    // but we size to `max_ctx` direct for simplicity (no ring wrap); if
    // max_seq were 16384 the direct cost doubles to 640 MiB and a 2048-slot
    // ring (80 MiB) would be preferable.
    pub kv_abs_end: i32,
    pub k_ctx_cached: Vec<GpuTensor>,
    pub v_ctx_cached: Vec<GpuTensor>,
    /// Device positions for RoPE, sized for the full ctx+block span and
    /// allocated ONCE. Uploaded each forward; views feed `rope_batched_f32`.
    ///
    /// This was originally malloc'd and freed inside the per-layer loop, which
    /// is both a hipMalloc per layer per window and a lifetime hazard — the
    /// freed pointer is what surfaced as `hipMemcpy H2D: an illegal memory
    /// access` on the second window. The target's `GlimmerState` already keeps
    /// a persistent `pos_buf` for exactly this reason; the drafter now matches.
    /// Capacity: `(max_ctx + block) * 4` bytes (i32 positions).
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// Absolute sequence capacity used to size K/V caches and `target_hidden_proj`
    /// (constructor `max_ctx` argument).
    pub max_ctx: usize,
    /// Explicit context capacity used to size `fc_input` (and the max `ctx_len`
    /// this scratch will accept). Set by the caller; never derived here.
    pub ctx_capacity: usize,
}

impl GlimmerDrafterScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &GlimmerDrafterConfig,
        max_ctx: usize,
        ctx_capacity: usize,
    ) -> Result<Self, String> {
        if ctx_capacity == 0 {
            return Err("glimmer drafter scratch: ctx_capacity == 0".into());
        }
        if ctx_capacity > max_ctx {
            return Err(format!(
                "glimmer drafter scratch: ctx_capacity {ctx_capacity} > max_ctx {max_ctx}"
            ));
        }
        let h = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let block = cfg.block_size;
        let kv_rows = max_ctx + block;
        // `fc_input` scales with CONTEXT rather than block. Callers pass an
        // already-clamped `ctx_capacity` (daemon/load: HIPFIRE_GLIMMER_CTX_CAP
        // defaulting to GLIMMER_DRAFTER_CTX_CAP_DEFAULT, clamped to max_seq).
        // No env sampling or sliding-window default here.
        let pos_buf = gpu
            .hip
            .malloc(kv_rows * 4)
            .map_err(|e| format!("glimmer drafter: alloc pos_buf: {e:?}"))?;

        // Narrow allocation-failure cleanup: free prior tensors immediately
        // (not via the reusable pool) so a partial constructor does not leak
        // or leave stale memset'd pool entries.
        struct Partial<'a> {
            gpu: &'a mut Gpu,
            pos_buf: Option<hip_bridge::DeviceBuffer>,
            tensors: Vec<GpuTensor>,
        }
        impl Drop for Partial<'_> {
            fn drop(&mut self) {
                for t in self.tensors.drain(..) {
                    let _ = self.gpu.release_tensor_immediate(t);
                }
                if let Some(buf) = self.pos_buf.take() {
                    let _ = self.gpu.hip.free(buf);
                }
            }
        }
        let mut partial = Partial {
            gpu,
            pos_buf: Some(pos_buf),
            tensors: Vec::new(),
        };
        let mut alloc = |n: usize, label: &str| -> Result<(), String> {
            let t = partial
                .gpu
                .zeros(&[n], DType::F32)
                .map_err(|e| format!("glimmer drafter scratch {label}: {e:?}"))?;
            partial.tensors.push(t);
            Ok(())
        };

        // Per-layer K/V cache for incremental attention (absolute-position-keyed).
        // Sized to `max_ctx * kv_dim` direct (no ring) so window [cur_start..cur_end)
        // is contiguous at offset cur_start*kvd.
        for li in 0..cfg.n_layers {
            alloc(max_ctx * kvd, &format!("k_ctx_cached{li}"))?;
            alloc(max_ctx * kvd, &format!("v_ctx_cached{li}"))?;
        }
        // Fixed scratch fields in assembly order after the 2*n_layers KV tensors.
        alloc(block * h, "x")?;
        alloc(max_ctx * h, "target_hidden_proj")?;
        alloc(block * qd, "q")?;
        alloc(kv_rows * kvd, "k")?;
        alloc(kv_rows * kvd, "v")?;
        alloc(block * qd, "attn_out")?;
        alloc(block * h, "tmp")?;
        alloc(block * cfg.intermediate, "gate_ffn")?;
        alloc(block * cfg.intermediate, "up_ffn")?;
        alloc(block * cfg.intermediate, "ffn_hidden")?;
        alloc(h, "logits_tmp")?;
        alloc(block * h, "x_rot")?;
        alloc(ctx_capacity * cfg.num_extract() * h, "fc_rot")?;
        alloc(kv_rows * h, "kv_input_rot")?;
        alloc(block * cfg.intermediate, "ffn_hidden_rot")?;
        alloc(ctx_capacity * cfg.num_extract() * h, "fc_input")?;

        // Success: take ownership out of Partial so Drop is a no-op.
        let tensors = std::mem::take(&mut partial.tensors);
        let pos_buf = partial.pos_buf.take().expect("pos_buf present");
        std::mem::forget(partial);

        let mut it = tensors.into_iter();
        let mut next = |label: &str| -> GpuTensor {
            it.next()
                .unwrap_or_else(|| panic!("glimmer drafter scratch: missing {label}"))
        };
        let mut k_ctx_cached = Vec::with_capacity(cfg.n_layers);
        let mut v_ctx_cached = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k_ctx_cached.push(next("k_ctx_cached"));
            v_ctx_cached.push(next("v_ctx_cached"));
        }
        let out = GlimmerDrafterScratch {
            x: next("x"),
            target_hidden_proj: next("target_hidden_proj"),
            q: next("q"),
            k: next("k"),
            v: next("v"),
            attn_out: next("attn_out"),
            tmp: next("tmp"),
            gate_ffn: next("gate_ffn"),
            up_ffn: next("up_ffn"),
            ffn_hidden: next("ffn_hidden"),
            logits_tmp: next("logits_tmp"),
            x_rot: next("x_rot"),
            fc_rot: next("fc_rot"),
            kv_input_rot: next("kv_input_rot"),
            ffn_hidden_rot: next("ffn_hidden_rot"),
            fc_input: next("fc_input"),
            fc_uploaded_rows: 0,
            fc_projected_rows: 0,
            fc_window_start: 0,
            kv_abs_end: 0,
            k_ctx_cached,
            v_ctx_cached,
            pos_buf,
            max_ctx,
            ctx_capacity,
        };
        debug_assert!(
            it.next().is_none(),
            "glimmer drafter scratch: leftover tensors"
        );
        Ok(out)
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.x,
            self.target_hidden_proj,
            self.q,
            self.k,
            self.v,
            self.attn_out,
            self.tmp,
            self.gate_ffn,
            self.up_ffn,
            self.ffn_hidden,
            self.logits_tmp,
            self.x_rot,
            self.fc_rot,
            self.kv_input_rot,
            self.ffn_hidden_rot,
            self.fc_input,
        ] {
            let _ = gpu.free_tensor(t);
        }
        for t in self.k_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        for t in self.v_ctx_cached {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.hip.free(self.pos_buf);
    }

    #[inline]
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    #[inline]
    pub fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    /// Clear absolute K/V/fc watermarks (no GPU memset).
    pub fn reset_history(&mut self) {
        self.kv_abs_end = 0;
        self.fc_uploaded_rows = 0;
        self.fc_projected_rows = 0;
        self.fc_window_start = 0;
    }

    /// Rewind absolute K/V watermark to `abs_end` (clamped) and clear
    /// observational fc row counters. Does not clear GPU cache contents.
    pub fn rewind_history(&mut self, abs_end: usize) {
        let end = abs_end as i32;
        if self.kv_abs_end > end {
            self.kv_abs_end = end;
        }
        self.fc_uploaded_rows = 0;
        self.fc_projected_rows = 0;
        self.fc_window_start = 0;
    }
}

/// One layer's context rows: the `delta` rows new since the last window
/// (their projection input `thp`, FWHT scratch `rot`, and K/V cache slots),
/// the cached window `[cur_start, cur_end)`, and where that window lands at
/// the front of the concatenated K/V with its positions.
struct CtxViews {
    thp: GpuTensor,
    rot: GpuTensor,
    k_new: GpuTensor,
    v_new: GpuTensor,
    k_window: GpuTensor,
    v_window: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    delta: usize,
}

/// Views of one layer's forward: the concatenated K/V `[ctx ++ block]`, the
/// block rows' K/V and positions, and the context rows when there are any.
struct LayerViews {
    k_full: GpuTensor,
    v_full: GpuTensor,
    k_tail: GpuTensor,
    v_tail: GpuTensor,
    pos_tail: GpuTensor,
    ctx: Option<CtxViews>,
}

/// The drafter layers' stream over the block rows: pre-norm sublayers that
/// add into `x` in place, on the verify ladder.
fn stream<'a>(
    cfg: &GlimmerDrafterConfig,
    s: &'a GlimmerDrafterScratch,
    block: usize,
) -> SandwichStream<'a> {
    SandwichStream {
        rows: block,
        hidden: cfg.hidden,
        eps: cfg.norm_eps,
        post_eps: cfg.norm_eps,
        x: &s.x,
        residual: &s.tmp,
        normed: &s.tmp,
        out: None,
        route: RowsRoute::Verify,
        kernels: SandwichRoute::Steps(StepsRoute {
            shared_rotation: false,
            fused_silu_rotate: false,
        }),
        bf16: None,
    }
}

/// One drafter layer: a two-norm Llama block over the block rows whose K/V
/// span the context rows ahead of them.
///
/// ```text
/// n  = rmsnorm(x, input_layernorm);  q = q_norm(Wq·n)
/// K  = k_norm(Wk·[ctx ++ n]); V = Wv·[ctx ++ n]    (ctx rows cached post-k_norm, pre-RoPE)
/// x += Wo·attend(rope(q), rope(K), V)               (bidirectional over ctx ++ block)
/// x += Wd·(silu(Wg·m) ⊙ Wu·m),  m = rmsnorm(x, post_attention_layernorm)
/// ```
fn layer_steps<'a>(
    cfg: &GlimmerDrafterConfig,
    layer: &'a GlimmerDrafterLayer,
    s: &'a GlimmerDrafterScratch,
    v: &'a LayerViews,
    block: usize,
    ctx_len: usize,
) -> [Step<'a>; 2] {
    let context = v.ctx.as_ref().map(|c| DraftContext {
        delta: c.delta,
        input: &c.thp,
        rot: &c.rot,
        k_new: &c.k_new,
        v_new: &c.v_new,
        k_window: &c.k_window,
        v_window: &c.v_window,
        k: &c.k,
        v: &c.v,
        rows: ctx_len,
        positions: &c.pos,
    });
    [
        Step::SandwichAttention(SandwichAttentionOp {
            stream: stream(cfg, s, block),
            position: ctx_len,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            input_norm: &layer.input_layernorm,
            wq: layer.q_proj.dispatch_ref(),
            q_norm: &layer.q_norm,
            kv_proj: Some(KvProjection {
                wk: layer.k_proj.dispatch_ref(),
                wv: Some(layer.v_proj.dispatch_ref()),
                k_norm: &layer.k_norm,
                v_norm: None,
            }),
            q_scale: 1.0,
            rope: Rope {
                kind: RopeKind::RotateHalf,
                theta: cfg.rope_theta,
            },
            kv: SandwichKv {
                k_cache: &v.k_full,
                v_cache: &v.v_full,
                physical_cap: ctx_len + block,
                window: 0,
                route: KvRoute::Draft {
                    kv_rows: ctx_len + block,
                    context,
                },
            },
            pos_buf: &s.pos_buf,
            positions: Some(&v.pos_tail),
            gate: None,
            wo: layer.o_proj.dispatch_ref(),
            post_norm: None,
            x_rot: &s.x_rot,
            q: &s.q,
            k: &v.k_tail,
            v: &v.v_tail,
            attn_out: &s.attn_out,
        }),
        Step::SandwichMlp(SandwichMlpOp {
            stream: stream(cfg, s, block),
            pre_norm: &layer.post_attention_layernorm,
            w_gate: layer.gate_proj.dispatch_ref(),
            w_up: layer.up_proj.dispatch_ref(),
            w_down: layer.down_proj.dispatch_ref(),
            activation: Activation::Silu,
            hidden_dim: cfg.intermediate,
            post_norm: None,
            x_rot: &s.ffn_hidden_rot,
            gate: &s.gate_ffn,
            up: &s.up_ffn,
            act: &s.ffn_hidden,
            out: &s.tmp,
        }),
    ]
}

// ─── Forward ──────────────────────────────────────────────────────────
/// Muse Glimmer DFlash draft forward (host target_hidden).
///
/// `noise_embedding`: `[block_size * hidden]` raw F32 embeddings of
/// `[seed, MASK×(block-1)]` via `target.embed_tokens` (no embed_norm).
/// `target_hidden`: `[ctx_len * num_extract * hidden]` concatenated residual
/// hidden from `target_layer_ids` (1,13,25,37,49) — every previously accepted
/// token row, not a single broadcast row.
/// `positions`: `[ctx_len + block_size]` absolute i32 span; the tail
/// `positions[ctx_len..]` are the Q / block positions.
/// Caller applies target `lm_head` over rows `1..block_size` to obtain draft logits.
#[allow(clippy::too_many_arguments)]
pub fn glimmer_drafter_forward(
    gpu: &mut Gpu,
    cfg: &GlimmerDrafterConfig,
    weights: &GlimmerDrafterWeights,
    scratch: &mut GlimmerDrafterScratch,
    noise_embedding: &[f32],
    target_hidden: &[f32],
    positions: &[i32],
    block_size: usize,
    ctx_len: usize,
) -> Result<(), String> {
    glimmer_drafter_forward_inner(
        gpu,
        cfg,
        weights,
        scratch,
        noise_embedding,
        TargetHiddenSrc::Host(target_hidden),
        positions,
        block_size,
        ctx_len,
    )
}

/// Muse Glimmer DFlash draft forward with device-resident target hidden log.
///
/// Same contract as [`glimmer_drafter_forward`], except context rows are pulled
/// from `target_hidden` via ordered async D2D instead of host H2D.
#[allow(clippy::too_many_arguments)]
pub fn glimmer_drafter_forward_device(
    gpu: &mut Gpu,
    cfg: &GlimmerDrafterConfig,
    weights: &GlimmerDrafterWeights,
    scratch: &mut GlimmerDrafterScratch,
    noise_embedding: &[f32],
    target_hidden: &GlimmerHiddenLog,
    positions: &[i32],
    block_size: usize,
    ctx_len: usize,
) -> Result<(), String> {
    glimmer_drafter_forward_inner(
        gpu,
        cfg,
        weights,
        scratch,
        noise_embedding,
        TargetHiddenSrc::Device(target_hidden),
        positions,
        block_size,
        ctx_len,
    )
}

enum TargetHiddenSrc<'a> {
    Host(&'a [f32]),
    Device(&'a GlimmerHiddenLog),
}

#[allow(clippy::too_many_arguments)]
fn glimmer_drafter_forward_inner(
    gpu: &mut Gpu,
    cfg: &GlimmerDrafterConfig,
    weights: &GlimmerDrafterWeights,
    scratch: &mut GlimmerDrafterScratch,
    noise_embedding: &[f32],
    target_hidden: TargetHiddenSrc<'_>,
    positions: &[i32],
    block_size: usize,
    ctx_len: usize,
) -> Result<(), String> {
    if block_size != cfg.block_size {
        return Err(format!(
            "glimmer drafter: block_size {} != cfg.block_size {}",
            block_size, cfg.block_size
        ));
    }
    let expected_noise = block_size * cfg.hidden;
    if noise_embedding.len() != expected_noise {
        return Err(format!(
            "glimmer drafter: noise_embedding len {} != expected {}",
            noise_embedding.len(),
            expected_noise
        ));
    }
    let expected_pos = ctx_len + block_size;
    if positions.len() != expected_pos {
        return Err(format!(
            "glimmer drafter: positions len {} != expected {} (ctx_len={} block_size={})",
            positions.len(),
            expected_pos,
            ctx_len,
            block_size
        ));
    }
    if ctx_len > 0 && positions[0] < 0 {
        return Err(format!(
            "glimmer drafter: positions[0]={} must be >= 0 when ctx_len>0",
            positions[0]
        ));
    }
    if ctx_len > scratch.ctx_capacity {
        return Err(format!(
            "glimmer drafter: ctx_len {ctx_len} > scratch.ctx_capacity {}",
            scratch.ctx_capacity
        ));
    }
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let eps = cfg.norm_eps;
    let kvd = cfg.kv_dim();
    let l = ctx_len + block_size; // K/V length
                                  // Absolute-position watermark for incremental fc/K/V. Mirrors
                                  // dflash.rs::TargetHiddenLog but absolute-keyed (like a normal KV cache)
                                  // so the daemon's suffix window `ctx_len = n_rows.min(sliding_window)` and
                                  // `start = (n_rows-ctx_len)*row_elems` slide does NOT invalidate — it
                                  // just selects a different absolute range of an already-populated cache.
                                  // Invalidate only on genuine rollback (cur_end < watermark) or reset.
    let cur_start: i32 = if ctx_len > 0 { positions[0] } else { 0 };
    let cur_end: i32 = cur_start + ctx_len as i32;
    if cur_end as usize > scratch.max_ctx {
        return Err(format!(
            "glimmer drafter: cur_end {cur_end} > scratch.max_ctx {}",
            scratch.max_ctx
        ));
    }
    let expected_th = ctx_len * ne * h;
    match &target_hidden {
        TargetHiddenSrc::Host(host) => {
            if host.len() != expected_th {
                return Err(format!(
                    "glimmer drafter: target_hidden len {} != expected {} (ctx_len={} num_extract={} hidden={})",
                    host.len(),
                    expected_th,
                    ctx_len,
                    ne,
                    h
                ));
            }
        }
        TargetHiddenSrc::Device(log) => {
            if !log.stage_is_idle() {
                return Err("glimmer drafter: target_hidden log stage must be Idle".into());
            }
            if log.hidden() != h {
                return Err(format!(
                    "glimmer drafter: target_hidden log hidden {} != cfg.hidden {h}",
                    log.hidden()
                ));
            }
            if log.num_extract() != ne {
                return Err(format!(
                    "glimmer drafter: target_hidden log num_extract {} != cfg {}",
                    log.num_extract(),
                    ne
                ));
            }
            if ctx_len > 0 {
                let cs = cur_start as usize;
                let ce = cur_end as usize;
                if log.committed_abs_end() < ce {
                    return Err(format!(
                        "glimmer drafter: log committed_abs_end {} < cur_end {ce}",
                        log.committed_abs_end()
                    ));
                }
                if cs < log.valid_abs_start() || ce > log.committed_abs_end() {
                    return Err(format!(
                        "glimmer drafter: requested [{cs},{ce}) outside log valid [{},{})",
                        log.valid_abs_start(),
                        log.committed_abs_end()
                    ));
                }
            }
        }
    }
    // Rollback / reset detection (absolute, not row-index).
    if ctx_len == 0 {
        // No context — clear watermarks.
        scratch.reset_history();
    } else if scratch.kv_abs_end > cur_end {
        // cur_end went backwards: rollback after rejected spec block or new
        // shorter conversation. The tail [cur_end .. old watermark) is stale
        // (same absolute positions will be rewritten with different content).
        scratch.reset_history();
    }
    // Incremental fc: only rows [fill_start .. cur_end) need source + fc.
    if ctx_len > 0 {
        let ne_h = ne * h;
        let fc_cap = scratch.fc_input.shape.iter().product::<usize>();
        // fc_input is window-sized (2048*ne_h) and used as temp staging for
        // delta rows, so capacity check is delta*ne_h <= fc_cap, not ctx_len.
        let fill_start = scratch.kv_abs_end.max(cur_start);
        let delta = (cur_end - fill_start) as usize;
        // Full window may be larger than fc_cap if max_ctx>window and we had
        // a gap; but delta is at most ctx_len (2048) which fits window. The
        // pre-existing guard `ctx_len*ne_h <= fc_cap` is therefore still valid
        // for the full-window delta case; for absolute-gap delta>window we
        // would need to chunk, but that never occurs in steady forward
        // extension (delta <= block_size + accepted).
        if ctx_len * ne_h > fc_cap && delta == ctx_len {
            // Only the full-window case needs the original guard; delta case
            // is bounded by window and already checked via delta*ne_h.
            return Err(format!(
                "glimmer drafter: fc_input holds {fc_cap} floats but this window needs \
                 {} (ctx_len={ctx_len} num_extract={ne} hidden={h}); the drafter context \
                 cap was raised above the value present when scratch was built",
                ctx_len * ne_h
            ));
        }
        if delta * ne_h > fc_cap {
            return Err(format!(
                "glimmer drafter: fc_input holds {fc_cap} floats but delta {delta} needs {}",
                delta * ne_h
            ));
        }
        if delta > 0 {
            // Stage delta rows contiguously at fc_input[0..delta) — absolute
            // position is encoded in the *destination* (target_hidden_proj) not
            // the source staging.
            let fc_in_seg = scratch.fc_input.sub_offset(0, delta * ne_h);
            match target_hidden {
                TargetHiddenSrc::Host(host) => {
                    let host_off = (fill_start - cur_start) as usize;
                    let host_seg = &host[host_off * ne_h..(host_off + delta) * ne_h];
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            host_seg.as_ptr() as *const u8,
                            host_seg.len() * 4,
                        )
                    };
                    gpu.hip.memcpy_htod(&fc_in_seg.buf, bytes).map_err(|e| {
                        format!("drafter htod fc_input delta [{fill_start}..{cur_end}): {e:?}")
                    })?;
                }
                TargetHiddenSrc::Device(log) => {
                    log.copy_committed_rows_to(
                        gpu,
                        fill_start as usize,
                        delta,
                        &scratch.fc_input,
                        0,
                    )
                    .map_err(|e| {
                        format!("drafter d2d fc_input delta [{fill_start}..{cur_end}): {e}")
                    })?;
                }
            }
            let target_seg = scratch
                .target_hidden_proj
                .sub_offset(fill_start as usize * h, delta * h);
            let fc_rot_seg = scratch.fc_rot.sub_offset(0, delta * ne_h);
            let fc = weights.fc.dispatch_ref();
            let ctx = DispatchCtx::new(gpu);
            gemm_rows(
                gpu,
                &ctx,
                &fc,
                GemvInput::Raw(&fc_in_seg),
                &target_seg,
                &fc_rot_seg,
                None,
                delta,
                Ladder::Route(RowsRoute::Verify),
            )
            .and_then(|()| {
                let norm = Step::RmsnormAutomatic {
                    x: &target_seg,
                    norm_weight: &weights.output_norm_enc,
                    x_plain: &target_seg,
                    out: &target_seg,
                    awq_scale: None,
                    k: h,
                    eps,
                    rotation: RotationPlan::None,
                    rows: delta,
                };
                execute_steps(gpu, &ctx, &[norm])
            })
            .map_err(|e| format!("drafter fc delta [{fill_start}..{cur_end}): {e:?}"))?;
        }
        // Keep row-index watermarks in sync for observability (they are row-
        // count based and would otherwise slide-invalidate every window at cap;
        // the absolute watermark `kv_abs_end` is the source of truth now).
        scratch.fc_uploaded_rows = ctx_len;
        scratch.fc_projected_rows = ctx_len;
        scratch.fc_window_start = cur_start;
        // Note: `kv_abs_end` is advanced *after* the per-layer K/V fill below,
        // so fc and K/V stay in sync. If ctx_len==0 we already reset.
    }

    // --- 2. noise_embedding into scratch.x (context is NOT added into x) ---
    {
        let host_bytes = unsafe {
            std::slice::from_raw_parts(
                noise_embedding.as_ptr() as *const u8,
                noise_embedding.len() * 4,
            )
        };
        gpu.hip
            .memcpy_htod(&scratch.x.buf, host_bytes)
            .map_err(|e| format!("drafter htod x: {e:?}"))?;
    }

    // Upload full position span once; reused by every layer's RoPE.
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 4)
        };
        let pos_view =
            unsafe { hip_bridge::DeviceBuffer::from_raw(scratch.pos_buf.as_ptr(), l * 4) };
        gpu.hip
            .memcpy_htod(&pos_view, bytes)
            .map_err(|e| format!("drafter htod positions: {e:?}"))?;
    }

    // --- 3. Per-layer transformer — ctx-concatenated K/V, block Q ---
    let ctx = DispatchCtx::new(gpu);
    let pos_all = GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(scratch.pos_buf.as_ptr(), l * 4) },
        shape: vec![l],
        dtype: DType::F32,
    };
    let fill_start = scratch.kv_abs_end.max(cur_start) as usize;
    let delta = (cur_end as usize).saturating_sub(fill_start);
    for (li, layer) in weights.layers.iter().enumerate() {
        let v = LayerViews {
            k_full: scratch.k.sub_offset(0, l * kvd),
            v_full: scratch.v.sub_offset(0, l * kvd),
            k_tail: scratch.k.sub_offset(ctx_len * kvd, block_size * kvd),
            v_tail: scratch.v.sub_offset(ctx_len * kvd, block_size * kvd),
            pos_tail: pos_all.sub_offset(ctx_len, block_size),
            ctx: (ctx_len > 0).then(|| CtxViews {
                thp: scratch
                    .target_hidden_proj
                    .sub_offset(fill_start * h, delta * h),
                rot: scratch.kv_input_rot.sub_offset(fill_start * h, delta * h),
                k_new: scratch.k_ctx_cached[li].sub_offset(fill_start * kvd, delta * kvd),
                v_new: scratch.v_ctx_cached[li].sub_offset(fill_start * kvd, delta * kvd),
                k_window: scratch.k_ctx_cached[li]
                    .sub_offset(cur_start as usize * kvd, ctx_len * kvd),
                v_window: scratch.v_ctx_cached[li]
                    .sub_offset(cur_start as usize * kvd, ctx_len * kvd),
                k: scratch.k.sub_offset(0, ctx_len * kvd),
                v: scratch.v.sub_offset(0, ctx_len * kvd),
                pos: pos_all.sub_offset(0, ctx_len),
                delta,
            }),
        };
        let steps = layer_steps(cfg, layer, scratch, &v, block_size, ctx_len);
        execute_steps(gpu, &ctx, &steps).map_err(|e| format!("drafter L{li}: {e:?}"))?;
    }
    // Advance absolute watermark — prefix [0..cur_end) is now valid in both
    // fc (target_hidden_proj) and per-layer K/V caches. Next window with the
    // same or larger cur_end will be delta-only; a rollback (cur_end <
    // watermark) will be caught at the top of the next call and reset.
    if ctx_len > 0 {
        scratch.kv_abs_end = cur_end;
    }
    let norm = Step::RmsnormAutomatic {
        x: &scratch.x,
        norm_weight: &weights.norm,
        x_plain: &scratch.x,
        out: &scratch.x,
        awq_scale: None,
        k: h,
        eps,
        rotation: RotationPlan::None,
        rows: block_size,
    };
    execute_steps(gpu, &ctx, &[norm]).map_err(|e| format!("drafter final norm: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qkv_dims_match_config() {
        // The glimmer_drafter_forward gate above must trip when mask_token_id is perturbed.
        let cfg = GlimmerDrafterConfig {
            n_layers: 5,
            hidden: 6656,
            intermediate: 19968,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            norm_eps: 1e-5,
            rope_theta: 500000.0,
            sliding_window: 2048,
            block_size: 16,
            mask_token_id: 201819, // perturbed
            target_layer_ids: vec![1, 13, 25, 37, 49],
        };
        assert_ne!(cfg.mask_token_id, 201818);
        assert_eq!(cfg.q_dim(), 32 * 128);
        assert_eq!(cfg.kv_dim(), 8 * 128);
        assert_eq!(cfg.num_extract(), 5);
    }
}
