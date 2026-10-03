// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt, Kate, Kevin Read
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4 model: hybrid sliding-window + full attention, dense FFN (SwiGLU + gelu_pytorch_tanh).
//!
//! Architectural features vs. Qwen3.5:
//!   • Sliding-window attention on 5 of every 6 layers (window=1024).
//!   • Full attention layers use head_dim=512 (global_head_dim) with
//!     attention_k_eq_v: V is the pre-k_norm output of k_proj (no v_proj).
//!   • Partial proportional RoPE on full layers (first 64 of 512 dims rotate,
//!     rope_theta=1e6; sliding uses default RoPE with theta=10000).
//!   • Sandwich RMSNorm: input + post-attn + pre-FFN + post-FFN per layer,
//!     plus a learned per-layer `layer_scalar [1]` at layer end.
//!   • Attention scale = 1.0 (not 1/√d); Q/K norms absorb scaling.
//!   • Final logit softcap: `tanh(logits/30) * 30` before sampling.
//!   • MLP: SwiGLU with `gelu_pytorch_tanh` activation.
//!   • Tied LM head (embed_tokens.weight aliased).
//!   • Embed scale: sqrt(hidden_size) multiplied onto every embedding row lookup.

use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::execute_steps;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, EmbeddingFormat, WeightTensor};
use hipfire_runtime::weight_store::upload_pooled_bytes;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Sliding-window causal attention (window=1024 on 31B).
    Sliding,
    /// Full causal attention (global).
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    /// Standard RoPE: all head_dim positions rotate.
    Default,
    /// Proportional RoPE (Gemma 4 full layers): only the first
    /// `partial_rotary_factor × head_dim` positions rotate; rest are NoPE.
    Proportional,
}

#[derive(Debug, Clone)]
pub struct Gemma4Config {
    // Common
    pub dim: usize,        // hidden_size, e.g. 5376 on 31B
    pub n_layers: usize,   // 60 on 31B
    pub vocab_size: usize, // 262144 on Gemma 4
    pub norm_eps: f32,     // 1e-6
    pub bos_token: u32,    // 2
    pub eos_token: u32,    // 1
    pub pad_token: u32,    // 0

    // Attention heads (same count for sliding + full)
    pub n_heads: usize, // 32 on 31B

    // Sliding-window attention
    pub sliding_head_dim: usize,   // 256 on 31B
    pub sliding_n_kv_heads: usize, // 16 on 31B
    pub sliding_rope_theta: f32,   // 10000.0
    pub sliding_window: usize,     // 1024

    // Full attention (global)
    pub full_head_dim: usize,            // 512 on 31B (= global_head_dim)
    pub full_n_kv_heads: usize,          // 4 on 31B
    pub full_rope_theta: f32,            // 1_000_000.0
    pub full_rope_type: RopeType,        // Proportional on 31B
    pub full_partial_rotary_factor: f32, // 0.25
    pub attention_k_eq_v: bool,          // true on 31B — V = pre-k_norm output

    // FFN (SwiGLU, gelu_pytorch_tanh)
    pub hidden_dim: usize, // intermediate_size = 21504 on 31B

    // MoE (26B-A4B). enable_moe_block=true → every layer carries a parallel
    // MoE branch whose output sums with the standard SwiGLU output before
    // the post_feedforward_layernorm. Zero on dense models (31B).
    pub enable_moe_block: bool,       // true on 26B-A4B
    pub moe_intermediate_size: usize, // 704 on 26B-A4B (per-expert FFN hidden)
    pub num_experts: usize,           // 128 on 26B-A4B
    pub top_k_experts: usize,         // 8 on 26B-A4B (kernel hardcoded to 8)

    // Output
    pub final_logit_softcapping: f32, // 30.0 — tanh(x/30)*30
    pub tie_word_embeddings: bool,    // true — lm_head aliases embed_tokens
    pub embed_scale: f32,             // sqrt(dim), applied at embed lookup

    // Per-layer dispatch (len == n_layers)
    pub layer_types: Vec<LayerType>,

    // Vision integration (present even on text-only 31B since config ships it)
    pub has_vision: bool,
    pub image_token_id: u32, // 258880
    pub boi_token_id: u32,   // 255999
    pub eoi_token_id: u32,   // 258882
    pub audio_token_id: u32, // 258881 (reserved, unused on dense 31B)
    pub video_token_id: u32, // 258884 (reserved)
}

pub fn config_from_hfq(hfq: &HfqFile) -> Option<Gemma4Config> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config")?;
    let tc = config.get("text_config").unwrap_or(config);

    let dim = tc.get("hidden_size")?.as_u64()? as usize;
    let n_layers = tc.get("num_hidden_layers")?.as_u64()? as usize;
    let vocab_size = tc.get("vocab_size")?.as_u64()? as usize;
    let norm_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let bos_token = tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32;
    let eos_token = tc.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let pad_token = tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let n_heads = tc.get("num_attention_heads")?.as_u64()? as usize;

    // Sliding attention params
    let sliding_head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(dim / n_heads);
    let sliding_n_kv_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let sliding_window = tc
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as usize;

    // Full attention params (may differ from sliding)
    let full_head_dim = tc
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(sliding_head_dim);
    let full_n_kv_heads = tc
        .get("num_global_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(sliding_n_kv_heads as u64) as usize;
    let attention_k_eq_v = tc
        .get("attention_k_eq_v")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // rope_parameters is a dict with "sliding_attention" and "full_attention" sub-dicts
    // per the Gemma 4 config schema. Parse both independently.
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));

    let sliding_rope_theta = sliding_rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0) as f32;
    let full_rope_theta = full_rope
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let full_rope_type = match full_rope
        .and_then(|r| r.get("rope_type"))
        .and_then(|v| v.as_str())
    {
        Some("proportional") => RopeType::Proportional,
        _ => RopeType::Default,
    };
    let full_partial_rotary_factor = full_rope
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let hidden_dim = tc.get("intermediate_size")?.as_u64()? as usize;

    // MoE config (26B-A4B). Absent / false on dense models (31B).
    let enable_moe_block = tc
        .get("enable_moe_block")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc
        .get("top_k_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let final_logit_softcapping = tc
        .get("final_logit_softcapping")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let tie_word_embeddings = tc
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| config.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    let embed_scale = (dim as f32).sqrt();

    let layer_types: Vec<LayerType> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                    "full_attention" => LayerType::Full,
                    _ => LayerType::Sliding,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![LayerType::Sliding; n_layers]);

    // Multimodal token IDs (top-level in config, not under text_config)
    let has_vision = config
        .get("vision_config")
        .map(|v| !v.is_null())
        .unwrap_or(false);
    let image_token_id = config
        .get("image_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258880) as u32;
    let boi_token_id = config
        .get("boi_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(255999) as u32;
    let eoi_token_id = config
        .get("eoi_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258882) as u32;
    let audio_token_id = config
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258881) as u32;
    let video_token_id = config
        .get("video_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(258884) as u32;

    Some(Gemma4Config {
        dim,
        n_layers,
        vocab_size,
        norm_eps,
        bos_token,
        eos_token,
        pad_token,
        n_heads,
        sliding_head_dim,
        sliding_n_kv_heads,
        sliding_rope_theta,
        sliding_window,
        full_head_dim,
        full_n_kv_heads,
        full_rope_theta,
        full_rope_type,
        full_partial_rotary_factor,
        attention_k_eq_v,
        hidden_dim,
        enable_moe_block,
        moe_intermediate_size,
        num_experts,
        top_k_experts,
        final_logit_softcapping,
        tie_word_embeddings,
        embed_scale,
        layer_types,
        has_vision,
        image_token_id,
        boi_token_id,
        eoi_token_id,
        audio_token_id,
        video_token_id,
    })
}

// ─── Weights ────────────────────────────────────────────────────────────

/// Per-layer weights for a SLIDING layer (head_dim=256, 16 KV heads, full RoPE).
pub struct SlidingLayerWeights {
    pub input_layernorm: GpuTensor,            // [dim]
    pub post_attention_layernorm: GpuTensor,   // [dim]
    pub pre_feedforward_layernorm: GpuTensor,  // [dim]
    pub post_feedforward_layernorm: GpuTensor, // [dim]
    pub layer_scalar: GpuTensor,               // [1]
    /// Host-side mirror of layer_scalar. Populated at load time so decode can
    /// call `gpu.scale_f32(x, layer_scalar_host)` without a D2H round-trip.
    pub layer_scalar_host: f32,

    // Attention (sliding — head_dim=256)
    pub q_proj: WeightTensor, // [n_heads * 256, dim]
    pub k_proj: WeightTensor, // [16 * 256, dim]
    pub v_proj: WeightTensor, // [16 * 256, dim]
    pub o_proj: WeightTensor, // [dim, n_heads * 256]
    pub q_norm: GpuTensor,    // [256]
    pub k_norm: GpuTensor,    // [256]

    // MLP (SwiGLU)
    pub gate_proj: WeightTensor, // [hidden_dim, dim]
    pub up_proj: WeightTensor,   // [hidden_dim, dim]
    pub down_proj: WeightTensor, // [dim, hidden_dim]

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-layer weights for a FULL layer (head_dim=512, 4 KV heads, K=V shared).
///
/// Note: no `v_proj` — V is the pre-k_norm output of k_proj, renormed by
/// weight-less `v_norm`. No `v_norm` tensor either (no_scale — the `with_scale=False`
/// RMSNorm applies only the divide, no learned gain). We reuse the existing
/// rmsnorm kernel with a ones-filled `v_norm_ones` buffer (shared across
/// full-attn layers) to preserve the no-scale semantics.
pub struct FullLayerWeights {
    pub input_layernorm: GpuTensor,
    pub post_attention_layernorm: GpuTensor,
    pub pre_feedforward_layernorm: GpuTensor,
    pub post_feedforward_layernorm: GpuTensor,
    pub layer_scalar: GpuTensor,
    /// Host-side mirror of layer_scalar. See SlidingLayerWeights for rationale.
    pub layer_scalar_host: f32,

    // Attention (full — head_dim=512, K=V)
    pub q_proj: WeightTensor, // [n_heads * 512, dim]
    pub k_proj: WeightTensor, // [4 * 512, dim]
    // no v_proj — V = pre-k_norm output of k_proj
    pub o_proj: WeightTensor, // [dim, n_heads * 512]
    pub q_norm: GpuTensor,    // [512]
    pub k_norm: GpuTensor,    // [512]
    // no v_norm weight — v_norm is no-scale (divide only)

    // MLP (SwiGLU, same shape as sliding)
    pub gate_proj: WeightTensor,
    pub up_proj: WeightTensor,
    pub down_proj: WeightTensor,

    // MoE branch — Some on 26B-A4B (every layer is MoE), None on dense models.
    pub moe: Option<MoeLayerExtras>,
}

/// Per-expert FFN weights for a single MoE expert. 128 of these per layer
/// on 26B-A4B; views into the per-layer pool allocation (so `free_gpu`
/// doesn't free these — the pool owns the bytes).
pub struct MoeExpertWeights {
    /// `[2 * moe_intermediate, dim]` — gate + up fused. Rows [0, mi) are
    /// gate; rows [mi, 2*mi) are up. Quantized as MQ4G256 / MG4G256 when
    /// dim is 256-aligned (it is on 26B-A4B: dim=2816).
    pub gate_up_proj: WeightTensor,
    /// `[dim, moe_intermediate]` — projects per-expert FFN hidden back to dim.
    /// On 26B-A4B, mi=704 isn't 256-aligned so this drops to Q8_0 via the
    /// quantizer fallback chain.
    pub down_proj: WeightTensor,
}

/// MoE branch weights for a Gemma 4 MoE layer (26B-A4B). Present on every
/// layer when `config.enable_moe_block` is set. The branch adds a parallel
/// FFN computation alongside the standard SwiGLU; outputs are summed via
/// sandwich norms then a final post_feedforward_layernorm closes the layer.
pub struct MoeLayerExtras {
    /// `[n_experts, dim]` — projects router input to expert logits.
    pub router_proj: WeightTensor,
    /// `[dim]` — multiplicative scale on router input (`router.scale` in HF).
    pub router_scale: GpuTensor,
    /// `[n_experts]` — per-expert post-`down_proj` scale (`router.per_expert_scale`).
    pub per_expert_scale: GpuTensor,
    /// Host mirror of `per_expert_scale` for fast top-K weight composition.
    pub per_expert_scale_host: Vec<f32>,
    /// `[dim]` — RMSNorm applied to attn_out before the MoE branch.
    pub pre_feedforward_layernorm_2: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_mlp (standard SwiGLU output) BEFORE summing.
    pub post_feedforward_layernorm_1: GpuTensor,
    /// `[dim]` — RMSNorm applied to cur_moe (MoE branch output) BEFORE summing.
    pub post_feedforward_layernorm_2: GpuTensor,
    /// Pool allocation for all gate_up tensors. Per-expert WeightTensors
    /// alias into this; `free_gpu` frees the pool, not each WeightTensor.
    pub experts_gate_up_pool: GpuTensor,
    /// Pool allocation for all down tensors. Same aliasing.
    pub experts_down_pool: GpuTensor,
    /// Per-expert views into the pools above.
    pub experts: Vec<MoeExpertWeights>,
    /// `[n_exp]` u64 device pointers — one per expert's gate_up weight
    /// base. Built once at load by reading each expert's pool sub-view
    /// pointer. The indexed MoE kernels read
    /// `expert_ptrs[topk_indices[krank]]` to locate the active expert
    /// weight WITHOUT a D2H sync.
    pub experts_gate_up_ptrs: GpuTensor,
    /// `[n_exp]` u64 device pointers for each expert's down weight base.
    pub experts_down_ptrs: GpuTensor,
}

pub enum LayerWeights {
    Sliding(SlidingLayerWeights),
    Full(FullLayerWeights),
}

pub struct Gemma4Weights {
    /// Token embedding [vocab_size, dim], Q8F16 to keep the 262144×5376 table manageable.
    /// Aliased as lm_head when tie_word_embeddings is true.
    pub embed_tokens: GpuTensor,
    /// Embed/LM-head format tag for dispatch.
    pub embd_format: EmbeddingFormat,
    /// LM-head projection (shares bytes with embed_tokens when tied).
    pub lm_head: WeightTensor,
    /// Model-final RMSNorm scale [dim].
    pub final_norm: GpuTensor,
    /// Per-layer weights indexed by layer ordinal.
    pub layers: Vec<LayerWeights>,
}

impl Gemma4Weights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed_tokens);
        let _ = gpu.free_tensor(self.final_norm);
        // lm_head aliases embed_tokens (tied weights) — a Borrowed view,
        // dropped here, never freed.
        for l in self.layers {
            Self::free_layer(gpu, l);
        }
    }

    /// Reclaim every GPU owner of one completed layer. Norm/scalar tensors go
    /// through `free_tensor`; projection weights go through
    /// `WeightTensor::free_all` so attached AWQ sidecars are reclaimed too.
    /// MoE expert views (`sub_offset` into the pools) are dropped, never
    /// freed — the pools own those bytes. Shared with load-time rollback so a
    /// late load failure reclaims completed layers exactly like unload does.
    fn free_layer(gpu: &mut Gpu, layer: LayerWeights) {
        match layer {
            LayerWeights::Sliding(s) => {
                for t in [
                    s.input_layernorm,
                    s.post_attention_layernorm,
                    s.pre_feedforward_layernorm,
                    s.post_feedforward_layernorm,
                    s.layer_scalar,
                    s.q_norm,
                    s.k_norm,
                ] {
                    let _ = gpu.free_tensor(t);
                }
                for w in [
                    s.q_proj,
                    s.k_proj,
                    s.v_proj,
                    s.o_proj,
                    s.gate_proj,
                    s.up_proj,
                    s.down_proj,
                ] {
                    w.free_all(gpu);
                }
                if let Some(moe) = s.moe {
                    Self::free_moe(gpu, moe);
                }
            }
            LayerWeights::Full(f) => {
                for t in [
                    f.input_layernorm,
                    f.post_attention_layernorm,
                    f.pre_feedforward_layernorm,
                    f.post_feedforward_layernorm,
                    f.layer_scalar,
                    f.q_norm,
                    f.k_norm,
                ] {
                    let _ = gpu.free_tensor(t);
                }
                for w in [
                    f.q_proj,
                    f.k_proj,
                    f.o_proj,
                    f.gate_proj,
                    f.up_proj,
                    f.down_proj,
                ] {
                    w.free_all(gpu);
                }
                if let Some(moe) = f.moe {
                    Self::free_moe(gpu, moe);
                }
            }
        }
    }

    fn free_moe(gpu: &mut Gpu, moe: MoeLayerExtras) {
        // router_proj is a real owner (buffer + optional AWQ sidecar).
        moe.router_proj.free_all(gpu);
        for t in [
            moe.router_scale,
            moe.per_expert_scale,
            moe.pre_feedforward_layernorm_2,
            moe.post_feedforward_layernorm_1,
            moe.post_feedforward_layernorm_2,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // Pool + pointer-table owners. Per-expert WeightTensors alias into the
        // pools via sub_offset — Borrowed views, dropped (never freed) with
        // `moe.experts` at function end.
        for t in [
            moe.experts_gate_up_pool,
            moe.experts_down_pool,
            moe.experts_gate_up_ptrs,
            moe.experts_down_ptrs,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}
// ─── Load-time fault seam ───────────────────────────────────────────────
// Private injectable failure points for GPU failure/retry regressions.
// Production passes `None` (one untaken branch per stage, zero behavior
// change); in-file tests pass `Some` to fail the n-th checked GPU upload so
// rollback of every staged owner is exercised. This is the lowered analogue
// of qwen35's `new_opt_with_alloc` counting-allocator seam, shaped as a
// check-hook because gemma4 leaf uploads are heterogeneous (upload_f32 /
// pooled-bytes) and don't funnel through one allocator closure. No env knob,
// no global allocation sweep: rollback frees only slot-staged owners plus
// completed layers, never aliases (lm_head, expert views) or GPU-global
// caches (mq signs/rotation scratch owned by `Gpu`).
struct AllocFaults {
    calls: usize,
    fail_at: Option<usize>,
}

impl AllocFaults {
    fn check(&mut self, stage: &'static str) -> HipResult<()> {
        self.calls += 1;
        if self.fail_at == Some(self.calls) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("injected gemma4 load fault at {stage} (op {})", self.calls),
            ));
        }
        Ok(())
    }
}

/// Run one staged fault check: fail this GPU-owning step when a test asked
/// for it. `None` (production) is a no-op.
fn fault_check(fault: &mut Option<&mut AllocFaults>, stage: &'static str) -> HipResult<()> {
    if let Some(f) = fault {
        f.check(stage)?;
    }
    Ok(())
}

// ─── Loading helpers ───────────────────────────────────────────────────

/// Decode a shape-[n] F16 or F32 tensor from HFQ into an F32 host Vec.
fn load_f32_vec(hfq: &HfqFile, name: &str, expected_n: usize) -> HipResult<Vec<f32>> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("tensor not found: {name}")))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    if n != expected_n {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("shape mismatch for {name}: expected {expected_n}, got {n}"),
        ));
    }
    if hipfire_config::developer_var("HIPFIRE_GEMMA4_DUMP")
        .ok()
        .as_deref()
        == Some("1")
        && data.len() <= 4
        && name.contains("layer_scalar")
    {
        eprintln!(
            "[gemma4] load_f32_vec({name}): qt={}, shape={:?}, raw_bytes={:02x?}",
            info.quant_type,
            info.shape,
            &data[..data.len().min(4)]
        );
    }
    let f32_data = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("expected F16/F32 for {name}, got qt={qt}"),
            ));
        }
    };
    Ok(f32_data)
}

/// Load a Gemma 4 RMSNorm weight — `x * weight` form, NO +1 shift.
///
/// Distinct from qwen35::load_norm_weight which shifts by +1 for HF Gemma
/// 2/3-style `x * (1 + weight)`. Gemma 4 uses plain `x * weight` with weights
/// initialized to 1.0 (see modeling_gemma4.py::Gemma4RMSNorm line 157).
fn load_gemma4_norm(hfq: &HfqFile, gpu: &mut Gpu, name: &str, dim: usize) -> HipResult<GpuTensor> {
    let f32_data = load_f32_vec(hfq, name, dim)?;
    gpu.upload_f32(&f32_data, &[dim])
}

/// Load a 256-element head-dim Q/K RMSNorm weight. Same semantics as
/// `load_gemma4_norm` but scoped to the attention head_dim (256 on sliding,
/// 512 on full).
fn load_gemma4_head_norm(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    head_dim: usize,
) -> HipResult<GpuTensor> {
    load_gemma4_norm(hfq, gpu, name, head_dim)
}

/// Load the per-layer `layer_scalar` — shape-[1] BF16/F16 tensor — returning
/// both a GPU-resident [1]-tensor (for potential batched use) and its host-side
/// f32 value (used by the decode path to call `scale_f32(x, cpu_scalar)`).
fn load_layer_scalar(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<(GpuTensor, f32)> {
    let data = load_f32_vec(hfq, name, 1)?;
    let host_val = data[0];
    let gpu_tensor = gpu.upload_f32(&data, &[1])?;
    Ok((gpu_tensor, host_val))
}
/// Load an AWQ per-channel scale sidecar (`<weight>.awq_scale.weight`, F16
/// [k]) through the pooled uploader. Absence or malformed sidecars return
/// None exactly like `hfq::load_awq_scale`; a present, well-formed sidecar
/// whose allocation/copy fails propagates the error into staged rollback
/// instead of silently dropping the scale (which would compute `(W·s)·x`).
fn load_gemma4_awq_scale(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    weight_name: &str,
    k: usize,
) -> HipResult<Option<GpuTensor>> {
    let sidecar_name = match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{weight_name}.awq_scale.weight"),
    };
    let Some((sc_info, sc_data)) = hfq.tensor_data_vec(&sidecar_name) else {
        return Ok(None);
    };
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return Ok(None);
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return Ok(None);
    }
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = upload_pooled_bytes(gpu, &f32_bytes, &[f32_bytes.len()])?;
    Ok(Some(t))
}

/// HFQ4-G128 rows each start a fresh 72-byte group; a trailing partial group
/// is padded. Files quantized before the 2-D packer (`b4846285e`) packed
/// groups across rows whenever `K % 128 != 0` (Gemma 4 expert `down_proj`,
/// K = 704), which every per-row kernel reads as garbage. Refuse them.
fn check_hfq4g128_rows(name: &str, m: usize, k: usize, bytes: usize) -> HipResult<()> {
    let rows = m * k.div_ceil(128) * 72;
    if bytes == rows {
        return Ok(());
    }
    let reason = if bytes == (m * k).div_ceil(128) * 72 {
        "packs HFQ4-G128 groups across rows (quantized before hipfire-quantize \
         b4846285e); requantize the model"
    } else {
        "has an unexpected byte size"
    };
    Err(hip_bridge::HipError::new(
        0,
        &format!("{name} [{m} x {k}] {reason} ({bytes} bytes, expected {rows})"),
    ))
}

/// Load a quantized projection weight. Mirrors qwen35::load_weight_tensor_raw
/// but uses the Gemma 4 tensor-name convention (`model.language_model.<name>`).
fn load_gemma4_weight(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    load_gemma4_weight_impl(hfq, gpu, name, m, k, None)
}

/// Injectable-fault twin of [`load_gemma4_weight`]: tests fail the sidecar
/// stage after the primary buffer is owned, proving the primary is reclaimed
/// even though the outer staged rollback never sees a half-built WeightTensor.
fn load_gemma4_weight_impl(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
    mut fault: Option<&mut AllocFaults>,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| hip_bridge::HipError::new(0, &format!("tensor not found: {name}")))?;
    let dtype = match info.quant_type {
        1 => {
            // F16 → upload as f32
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = upload_pooled_bytes(gpu, bytes, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                paro: None,
            });
        }
        16 => {
            if rdna_compute::calib_force_bf16() {
                // Native BF16 teacher — keep raw 2-byte payload as BF16 for MFMA.
                // Otherwise the batched GEMM would land on the scalar F32 kernel.
                let buf = upload_pooled_bytes(gpu, data, &[m, k])?;
                return Ok(WeightTensor {
                    buf,
                    gpu_dtype: DType::BF16,
                    m,
                    k,
                    row_stride: 0,
                    awq_scale: None,
                    lloyd_lut_e4m3: None,
                    lloyd_lut_f16: None,
                    lloyd_lut_c16: None,
                    paro: None,
                });
            }
            // Default: BF16 → widen to F32 (shift, not f16 decode)
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = upload_pooled_bytes(gpu, bytes, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                paro: None,
            });
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq) — upload as-is.
            let buf = upload_pooled_bytes(gpu, data, &[m, k])?;
            return Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                paro: None,
            });
        }
        // Q8F16/Q8_0 projections are emitted when a matrix is not eligible
        // for the requested grouped format.  The lowered forward path already
        // dispatches DType::Q8_0; keep its loader aligned with the eager and
        // drafter loaders instead of rejecting a valid quantizer fallback.
        3 => DType::Q8_0,
        4 => DType::Q4K,
        6 => DType::HFQ4G256,
        7 => DType::HFQ4G128,
        8 => DType::HFQ6G256,
        9 => DType::HFQ2G256,
        10 => DType::HFQ2G128,
        11 => DType::HFQ3G256,
        12 => DType::HFQ3G128,
        13 => DType::MQ4G256,
        44 => DType::MQ4G256V2,
        14 => DType::MQ8G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        // MG4-G256 — Magnum-Gemma 4-bit. Same binary layout as MQ4G256 (136 B/group),
        // differs only in calibration policy at quant time. Alias to MQ4G256 so the
        // existing GEMV path handles it without a kernel change. ID was 19 on
        // origin/gemma4 pre-rebase; reassigned to 30 because master shipped
        // MQ2G256Lloyd at 19.
        30 => DType::MQ4G256,
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported quant_type {qt} for {name}"),
            ));
        }
    };
    if dtype == DType::HFQ4G128 {
        check_hfq4g128_rows(name, m, k, data.len())?;
    }
    let buf = upload_pooled_bytes(gpu, data, &[data.len()])?;
    // Fault seam: fail after the primary owns its buffer but before the
    // sidecar attaches. Rollback here frees the primary directly — the outer
    // transaction never received it.
    if let Err(e) = fault_check(&mut fault, "awq_sidecar") {
        let _ = gpu.free_tensor(buf);
        return Err(e);
    }
    let awq_scale = if dtype.supports_awq_sidecar() {
        match load_gemma4_awq_scale(hfq, gpu, name, k) {
            Ok(s) => s,
            Err(e) => {
                let _ = gpu.free_tensor(buf);
                return Err(e);
            }
        }
    } else {
        None
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        awq_scale,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
        paro: None,
    })
}

/// Load the MoE branch weights for a single Gemma 4 MoE layer (26B-A4B).
/// Builds 128 expert WeightTensors aliased into per-layer gate_up / down
/// pool allocations. Pooling avoids the small-allocation HIP fragmentation
/// that OOM'd at layer 25 on the pre-pool path (origin/gemma4 commit log).
fn load_moe_layer_extras(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    p: &str,
    config: &Gemma4Config,
    mut fault: Option<&mut AllocFaults>,
) -> HipResult<MoeLayerExtras> {
    let n_exp = config.num_experts;
    let dim = config.dim;
    let mi = config.moe_intermediate_size;

    // Staged owners: every successful GPU allocation lands in a slot the
    // moment it succeeds. Per-expert WeightTensors are sub_offset views into
    // the pools — never staged, never freed. On any error the fail! arm frees
    // every staged owner (weights via free_all so AWQ sidecars go too).
    let mut router_proj_opt: Option<WeightTensor> = None;
    let mut router_scale_opt: Option<GpuTensor> = None;
    let mut per_expert_scale_opt: Option<GpuTensor> = None;
    let mut per_expert_scale_host: Vec<f32> = Vec::new();
    let mut pre2_opt: Option<GpuTensor> = None;
    let mut post1_opt: Option<GpuTensor> = None;
    let mut post2_opt: Option<GpuTensor> = None;
    let mut gate_up_pool_opt: Option<GpuTensor> = None;
    let mut down_pool_opt: Option<GpuTensor> = None;
    let mut gate_ptrs_opt: Option<GpuTensor> = None;
    let mut down_ptrs_opt: Option<GpuTensor> = None;
    macro_rules! fail {
        ($e:expr) => {{
            if let Some(t) = down_ptrs_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            if let Some(t) = gate_ptrs_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            if let Some(t) = down_pool_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            if let Some(t) = gate_up_pool_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            for t in [
                post2_opt.take(),
                post1_opt.take(),
                pre2_opt.take(),
                per_expert_scale_opt.take(),
                router_scale_opt.take(),
            ]
            .into_iter()
            .flatten()
            {
                let _ = gpu.free_tensor(t);
            }
            if let Some(w) = router_proj_opt.take() {
                w.free_all(gpu);
            }
            return Err($e);
        }};
    }
    macro_rules! check {
        ($stage:expr) => {
            if let Err(e) = fault_check(&mut fault, $stage) {
                fail!(e);
            }
        };
    }
    macro_rules! stage {
        ($slot:ident, $stage:expr, $val:expr) => {{
            check!($stage);
            match $val {
                Ok(v) => {
                    $slot = Some(v);
                }
                Err(e) => fail!(e),
            }
        }};
    }

    stage!(
        router_proj_opt,
        "moe_router_proj",
        load_gemma4_weight(hfq, gpu, &format!("{p}.router.proj.weight"), n_exp, dim)
    );
    // NOTE: `router.scale` and `router.per_expert_scale` ship WITHOUT the
    // `.weight` suffix in HF's 26B-A4B safetensors (so `should_quantize`
    // returns false → stored as F16). Loader uses bare paths.
    stage!(
        router_scale_opt,
        "moe_router_scale",
        load_gemma4_norm(hfq, gpu, &format!("{p}.router.scale"), dim)
    );
    check!("moe_per_expert_scale_host");
    match load_f32_vec(hfq, &format!("{p}.router.per_expert_scale"), n_exp) {
        Ok(v) => {
            per_expert_scale_host = v;
        }
        Err(e) => fail!(e),
    }
    check!("moe_per_expert_scale");
    match (|| -> HipResult<GpuTensor> {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                per_expert_scale_host.as_ptr() as *const u8,
                per_expert_scale_host.len() * 4,
            )
        };
        upload_pooled_bytes(gpu, bytes, &[n_exp])
    })() {
        Ok(t) => {
            per_expert_scale_opt = Some(t);
        }
        Err(e) => fail!(e),
    }
    stage!(
        pre2_opt,
        "moe_pre_norm2",
        load_gemma4_norm(
            hfq,
            gpu,
            &format!("{p}.pre_feedforward_layernorm_2.weight"),
            dim,
        )
    );
    stage!(
        post1_opt,
        "moe_post_norm1",
        load_gemma4_norm(
            hfq,
            gpu,
            &format!("{p}.post_feedforward_layernorm_1.weight"),
            dim,
        )
    );
    stage!(
        post2_opt,
        "moe_post_norm2",
        load_gemma4_norm(
            hfq,
            gpu,
            &format!("{p}.post_feedforward_layernorm_2.weight"),
            dim,
        )
    );

    // Pool all `n_experts` weights of one kind into a single GPU allocation.
    // 128 experts × 2 kinds × 30 layers = 7680 separate hipMalloc on the
    // unpooled path fragmented the HIP heap and OOM'd at layer ~25 even
    // when total memory fit. Pool collapses that to 60 allocs.
    let load_pool = |gpu: &mut Gpu, base: &str| -> HipResult<(GpuTensor, DType, usize)> {
        // First pass: read first expert to learn quant_type + bytes-per-expert.
        let first_name = format!("{p}.experts.0.{base}.weight");
        let (first_info, first_data) = hfq.tensor_data(&first_name).ok_or_else(|| {
            hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {first_name}"))
        })?;
        let bytes_per_expert = first_data.len();
        let dtype = match first_info.quant_type {
            3 => DType::Q8_0,
            4 => DType::Q4K,
            6 => DType::HFQ4G256,
            7 => DType::HFQ4G128,
            8 => DType::HFQ6G256,
            9 => DType::HFQ2G256,
            10 => DType::HFQ2G128,
            11 => DType::HFQ3G256,
            12 => DType::HFQ3G128,
            // MQ4G256 (13) and MG4G256 (30) share dispatch.
            13 | 30 => DType::MQ4G256,
            44 => DType::MQ4G256V2,
            14 => DType::MQ8G256,
            15 => DType::MQ6G256,
            17 => DType::MQ3G256,
            18 => DType::MQ2G256,
            qt => {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("unsupported MoE expert quant_type {qt} for {first_name}"),
                ));
            }
        };
        if dtype == DType::HFQ4G128 {
            let (m, k) = match first_info.shape[..] {
                [m, k] => (m as usize, k as usize),
                _ => {
                    return Err(hip_bridge::HipError::new(
                        0,
                        &format!(
                            "{first_name}: expected a 2-D shape, got {:?}",
                            first_info.shape
                        ),
                    ))
                }
            };
            check_hfq4g128_rows(&first_name, m, k, bytes_per_expert)?;
        }
        // Concat all experts' bytes into one CPU buffer, upload once.
        let mut concat = Vec::with_capacity(bytes_per_expert * n_exp);
        concat.extend_from_slice(first_data);
        for x in 1..n_exp {
            let name = format!("{p}.experts.{x}.{base}.weight");
            let (info, data) = hfq.tensor_data(&name).ok_or_else(|| {
                hip_bridge::HipError::new(0, &format!("MoE expert tensor not found: {name}"))
            })?;
            if data.len() != bytes_per_expert {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "MoE expert {name} byte size mismatch ({} vs {bytes_per_expert})",
                        data.len()
                    ),
                ));
            }
            if info.quant_type != first_info.quant_type {
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!(
                        "MoE expert {name} quant_type mismatch ({} vs {})",
                        info.quant_type, first_info.quant_type
                    ),
                ));
            }
            concat.extend_from_slice(data);
        }
        let pool = upload_pooled_bytes(gpu, &concat, &[concat.len()])?;
        Ok((pool, dtype, bytes_per_expert))
    };

    check!("moe_gate_up_pool");
    let (gate_up_pool, gate_up_dtype, gate_up_bpe) = match load_pool(gpu, "gate_up_proj") {
        Ok(v) => v,
        Err(e) => fail!(e),
    };
    gate_up_pool_opt = Some(gate_up_pool);
    check!("moe_down_pool");
    let (down_pool, down_dtype, down_bpe) = match load_pool(gpu, "down_proj") {
        Ok(v) => v,
        Err(e) => fail!(e),
    };
    down_pool_opt = Some(down_pool);

    let mut experts = Vec::with_capacity(n_exp);
    for x in 0..n_exp {
        let gu_view = gate_up_pool_opt
            .as_ref()
            .expect("gemma4 load: gate-up pool staged before views")
            .sub_offset(x * gate_up_bpe, gate_up_bpe);
        let dn_view = down_pool_opt
            .as_ref()
            .expect("gemma4 load: down pool staged before views")
            .sub_offset(x * down_bpe, down_bpe);
        experts.push(MoeExpertWeights {
            gate_up_proj: WeightTensor {
                buf: gu_view,
                gpu_dtype: gate_up_dtype,
                m: 2 * mi,
                k: dim,
                row_stride: 0,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                paro: None,
            },
            down_proj: WeightTensor {
                buf: dn_view,
                gpu_dtype: down_dtype,
                m: dim,
                k: mi,
                row_stride: 0,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                paro: None,
            },
        });
    }

    // Build [n_exp] device tensors of u64 weight-base pointers — one for
    // each pool. The indexed MoE kernels read these as
    // `expert_ptrs[topk_indices[krank]]`, eliminating the per-token D2H
    // sync of the legacy CPU per-expert loop. Pointers are stable for the
    // model's lifetime (pool allocations don't move).
    let gate_up_ptr_u64: Vec<u64> = experts
        .iter()
        .map(|e| e.gate_up_proj.buf.buf.as_ptr() as u64)
        .collect();
    let down_ptr_u64: Vec<u64> = experts
        .iter()
        .map(|e| e.down_proj.buf.buf.as_ptr() as u64)
        .collect();
    let gate_up_ptr_bytes: Vec<u8> = gate_up_ptr_u64
        .iter()
        .flat_map(|p| p.to_ne_bytes())
        .collect();
    let down_ptr_bytes: Vec<u8> = down_ptr_u64.iter().flat_map(|p| p.to_ne_bytes()).collect();
    // Each u64 = 8 bytes = 2 f32 slots. The tensor sees [n_exp * 2]
    // f32 entries; the kernel casts the backing buffer to u64* itself.
    check!("moe_gate_up_ptrs");
    match upload_pooled_bytes(gpu, &gate_up_ptr_bytes, &[n_exp * 2]) {
        Ok(t) => {
            gate_ptrs_opt = Some(t);
        }
        Err(e) => fail!(e),
    }
    check!("moe_down_ptrs");
    match upload_pooled_bytes(gpu, &down_ptr_bytes, &[n_exp * 2]) {
        Ok(t) => {
            down_ptrs_opt = Some(t);
        }
        Err(e) => fail!(e),
    }

    Ok(MoeLayerExtras {
        router_proj: router_proj_opt
            .take()
            .expect("gemma4 load: router_proj staged once"),
        router_scale: router_scale_opt
            .take()
            .expect("gemma4 load: router_scale staged once"),
        per_expert_scale: per_expert_scale_opt
            .take()
            .expect("gemma4 load: per_expert_scale staged once"),
        per_expert_scale_host,
        pre_feedforward_layernorm_2: pre2_opt.take().expect("gemma4 load: pre_norm2 staged once"),
        post_feedforward_layernorm_1: post1_opt
            .take()
            .expect("gemma4 load: post_norm1 staged once"),
        post_feedforward_layernorm_2: post2_opt
            .take()
            .expect("gemma4 load: post_norm2 staged once"),
        experts_gate_up_pool: gate_up_pool_opt
            .take()
            .expect("gemma4 load: gate-up pool staged once"),
        experts_down_pool: down_pool_opt
            .take()
            .expect("gemma4 load: down pool staged once"),
        experts,
        experts_gate_up_ptrs: gate_ptrs_opt
            .take()
            .expect("gemma4 load: gate-up ptrs staged once"),
        experts_down_ptrs: down_ptrs_opt
            .take()
            .expect("gemma4 load: down ptrs staged once"),
    })
}

/// Load Gemma 4 text model weights from an HFQ file.
///
/// Design notes:
///   - `lm_head` aliases the `embed_tokens` GPU bytes (tied weights). We upload
///     the embed data once and create a second WeightTensor whose DeviceBuffer
///     points at the same allocation via `buf.alias()`. `Gemma4Weights::free_gpu`
///     skips freeing the LM head to avoid a double-free.
///   - Vision tensors are skipped here — Phase 7 `gemma4_vision::load_weights`
///     picks those up from the same HFQ file in a separate pass.
///   - The `v_norm_ones_full` ones-filled scratch buffer is populated here so
///     the forward pass never has to manage one-time init state.
///   - Transactional construction: every successful GPU allocation lands in a
///     slot the moment it succeeds (embed/final-norm at top level, each leaf
///     inside [`load_single_layer`], pools/ptr tables inside
///     `load_moe_layer_extras`). Any failure frees all staged owners — weights
///     via `WeightTensor::free_all` so AWQ sidecars go too — plus every
///     completed layer. Aliases (`lm_head`, expert views) are dropped, never
///     freed. Success takes each slot exactly once into the returned structs.
pub fn load_weights(
    hfq: &mut HfqFile,
    config: &Gemma4Config,
    gpu: &mut Gpu,
) -> HipResult<Gemma4Weights> {
    load_weights_impl(hfq, config, gpu, None)
}

/// Injectable-fault twin of [`load_weights`] for GPU failure/retry
/// regressions. Production passes `None`; in-file tests pass `Some` to fail
/// the n-th staged step and prove every nested owner rolls back.
fn load_weights_impl(
    hfq: &mut HfqFile,
    config: &Gemma4Config,
    gpu: &mut Gpu,
    mut fault: Option<&mut AllocFaults>,
) -> HipResult<Gemma4Weights> {
    let mut embed_opt: Option<GpuTensor> = None;
    let mut embd_format_opt: Option<EmbeddingFormat> = None;
    // Borrowed alias of embed_tokens — staged for take-once publish, dropped
    // (never freed) on rollback.
    let mut lm_head_opt: Option<WeightTensor> = None;
    let mut final_norm_opt: Option<GpuTensor> = None;
    let mut layers: Vec<LayerWeights> = Vec::with_capacity(config.n_layers);
    macro_rules! fail {
        ($e:expr) => {{
            for layer in layers.drain(..) {
                Gemma4Weights::free_layer(gpu, layer);
            }
            if let Some(t) = final_norm_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            let _ = lm_head_opt.take();
            if let Some(t) = embed_opt.take() {
                let _ = gpu.free_tensor(t);
            }
            return Err($e);
        }};
    }
    macro_rules! check {
        ($stage:expr) => {
            if let Err(e) = fault_check(&mut fault, $stage) {
                fail!(e);
            }
        };
    }
    macro_rules! stage {
        ($slot:ident, $stage:expr, $val:expr) => {{
            check!($stage);
            match $val {
                Ok(v) => {
                    $slot = Some(v);
                }
                Err(e) => fail!(e),
            }
        }};
    }

    eprintln!("gemma4: loading embed_tokens...");
    // Nothing staged yet: plain `?` cannot leak here.
    fault_check(&mut fault, "embed")?;
    let embed_name = "model.language_model.embed_tokens.weight";
    let (embed_info, embed_data) = hfq
        .tensor_data(embed_name)
        .ok_or_else(|| hip_bridge::HipError::new(0, "embed_tokens not found in HFQ"))?;
    let (embed_tokens, embd_format) = match embed_info.quant_type {
        3 => {
            eprintln!("  (Q8_0 / Q8F16, {} MB)", embed_data.len() / 1_000_000);
            (
                upload_pooled_bytes(gpu, embed_data, &[embed_data.len()])?,
                EmbeddingFormat::Q8_0,
            )
        }
        6 => {
            eprintln!("  (HFQ4-G256, {} MB)", embed_data.len() / 1_000_000);
            (
                upload_pooled_bytes(gpu, embed_data, &[embed_data.len()])?,
                EmbeddingFormat::HFQ4G256,
            )
        }
        7 => {
            eprintln!("  (HFQ4-G128, {} MB)", embed_data.len() / 1_000_000);
            (
                upload_pooled_bytes(gpu, embed_data, &[embed_data.len()])?,
                EmbeddingFormat::HFQ4G128,
            )
        }
        1 => {
            eprintln!("  (F16 → F32)");
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            (
                gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
                EmbeddingFormat::F32,
            )
        }
        16 => {
            eprintln!("  (BF16 → F32)");
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            (
                gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
                EmbeddingFormat::F32,
            )
        }
        2 => {
            // F32 raw (oracle / --format f32 passthrough .hfq).
            eprintln!("  (F32 raw, {} MB)", embed_data.len() / 1_000_000);
            let f32_data: Vec<f32> = embed_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            (
                gpu.upload_f32(&f32_data, &[config.vocab_size, config.dim])?,
                EmbeddingFormat::F32,
            )
        }
        qt => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported embed quant_type {qt}"),
            ));
        }
    };
    embed_opt = Some(embed_tokens);
    embd_format_opt = Some(embd_format);

    // Tied LM head: WeightTensor whose buffer aliases the embed allocation.
    // Rollback and free_gpu skip freeing this — embed_tokens owns the bytes.
    check!("lm_head");
    let lm_head = {
        let alias_buf = unsafe {
            embed_opt
                .as_ref()
                .expect("gemma4 load: embed staged")
                .buf
                .alias()
        };
        let dtype = match embd_format_opt
            .as_ref()
            .expect("gemma4 load: embed format staged")
        {
            EmbeddingFormat::Q8_0 => DType::Q8_0,
            EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
            EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
            EmbeddingFormat::F32 => DType::F32,
            EmbeddingFormat::Q4K => DType::Q4K,
        };
        let alias_tensor = GpuTensor {
            buf: alias_buf,
            shape: embed_opt
                .as_ref()
                .expect("gemma4 load: embed staged")
                .shape
                .clone(),
            dtype,
        };
        WeightTensor {
            buf: alias_tensor,
            gpu_dtype: dtype,
            m: config.vocab_size,
            k: config.dim,
            row_stride: 0,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
            paro: None,
        }
    };
    lm_head_opt = Some(lm_head);

    eprintln!("gemma4: loading final norm...");
    stage!(
        final_norm_opt,
        "final_norm",
        load_gemma4_norm(hfq, gpu, "model.language_model.norm.weight", config.dim)
    );

    eprintln!("gemma4: loading {} layers...", config.n_layers);
    for i in 0..config.n_layers {
        check!("layer");
        match load_single_layer(hfq, gpu, config, i, fault.as_deref_mut()) {
            Ok(layer) => layers.push(layer),
            Err(e) => fail!(e),
        }
    }
    eprintln!("gemma4: loaded all {} layers", config.n_layers);

    Ok(Gemma4Weights {
        embed_tokens: embed_opt
            .take()
            .expect("gemma4 load: embed_tokens staged once"),
        embd_format: embd_format_opt
            .take()
            .expect("gemma4 load: embed format staged once"),
        lm_head: lm_head_opt
            .take()
            .expect("gemma4 load: lm_head staged once"),
        final_norm: final_norm_opt
            .take()
            .expect("gemma4 load: final_norm staged once"),
        layers,
    })
}

/// Build one decoder layer with staged-owner rollback: every successful leaf
/// (norms, scalar, projections with their AWQ sidecars, MoE pools/ptr
/// tables) lands in a slot immediately; any later failure frees all staged
/// owners and the caller frees nothing further for this layer. Expert views
/// stay inside a successfully built `MoeLayerExtras` and are never freed.
fn load_single_layer(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    config: &Gemma4Config,
    i: usize,
    mut fault: Option<&mut AllocFaults>,
) -> HipResult<LayerWeights> {
    let p = format!("model.language_model.layers.{i}");
    // Staged-owner slots, declared ahead of the macros so the fail!/check!/stage!
    // bodies resolve them lexically (macro_rules hygiene: bare identifiers in a
    // macro body resolve at the macro definition site). Both layer arms share
    // these bindings — only one arm executes per call. v_opt stays None on full
    // layers, which reuse k_proj's pre-norm output as V.
    let mut scalar_opt: Option<GpuTensor> = None;
    let mut scalar_host_opt: Option<f32> = None;
    let mut moe_opt: Option<MoeLayerExtras> = None;
    let mut input_opt: Option<GpuTensor> = None;
    let mut post_attn_opt: Option<GpuTensor> = None;
    let mut pre_ffn_opt: Option<GpuTensor> = None;
    let mut post_ffn_opt: Option<GpuTensor> = None;
    let mut q_opt: Option<WeightTensor> = None;
    let mut k_opt: Option<WeightTensor> = None;
    let mut v_opt: Option<WeightTensor> = None;
    let mut o_opt: Option<WeightTensor> = None;
    let mut qn_opt: Option<GpuTensor> = None;
    let mut kn_opt: Option<GpuTensor> = None;
    let mut gate_opt: Option<WeightTensor> = None;
    let mut up_opt: Option<WeightTensor> = None;
    let mut down_opt: Option<WeightTensor> = None;
    macro_rules! fail {
        ($e:expr) => {{
            if let Some(m) = moe_opt.take() {
                Gemma4Weights::free_moe(gpu, m);
            }
            for w in [
                q_opt.take(),
                k_opt.take(),
                v_opt.take(),
                o_opt.take(),
                gate_opt.take(),
                up_opt.take(),
                down_opt.take(),
            ]
            .into_iter()
            .flatten()
            {
                w.free_all(gpu);
            }
            for t in [
                input_opt.take(),
                post_attn_opt.take(),
                pre_ffn_opt.take(),
                post_ffn_opt.take(),
                scalar_opt.take(),
                qn_opt.take(),
                kn_opt.take(),
            ]
            .into_iter()
            .flatten()
            {
                let _ = gpu.free_tensor(t);
            }
            return Err($e);
        }};
    }
    macro_rules! check {
        ($stage:expr) => {
            if let Err(e) = fault_check(&mut fault, $stage) {
                fail!(e);
            }
        };
    }
    macro_rules! stage {
        ($slot:ident, $stage:expr, $val:expr) => {{
            check!($stage);
            match $val {
                Ok(v) => {
                    $slot = Some(v);
                }
                Err(e) => fail!(e),
            }
        }};
    }
    match config.layer_types[i] {
        LayerType::Sliding => {
            let hd = config.sliding_head_dim;
            let kv_dim = config.sliding_n_kv_heads * hd;
            let q_dim = config.n_heads * hd;
            check!("layer_scalar");
            match load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar")) {
                Ok((t, h)) => {
                    scalar_opt = Some(t);
                    scalar_host_opt = Some(h);
                }
                Err(e) => fail!(e),
            }
            if i == 0 {
                eprintln!(
                    "[gemma4] L0 sliding layer_scalar = {}",
                    scalar_host_opt
                        .as_ref()
                        .expect("gemma4 load: scalar staged")
                );
            }
            check!("moe");
            moe_opt = if config.enable_moe_block {
                match load_moe_layer_extras(hfq, gpu, &p, config, fault.as_deref_mut()) {
                    Ok(m) => Some(m),
                    Err(e) => fail!(e),
                }
            } else {
                None
            };
            stage!(
                input_opt,
                "input_layernorm",
                load_gemma4_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), config.dim,)
            );
            stage!(
                post_attn_opt,
                "post_attention_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                pre_ffn_opt,
                "pre_feedforward_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                post_ffn_opt,
                "post_feedforward_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                q_opt,
                "q_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    config.dim,
                )
            );
            stage!(
                k_opt,
                "k_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )
            );
            stage!(
                v_opt,
                "v_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    config.dim,
                )
            );
            stage!(
                o_opt,
                "o_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    q_dim,
                )
            );
            stage!(
                qn_opt,
                "q_norm",
                load_gemma4_head_norm(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), hd,)
            );
            stage!(
                kn_opt,
                "k_norm",
                load_gemma4_head_norm(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), hd,)
            );
            stage!(
                gate_opt,
                "gate_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )
            );
            stage!(
                up_opt,
                "up_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )
            );
            stage!(
                down_opt,
                "down_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )
            );
            Ok(LayerWeights::Sliding(SlidingLayerWeights {
                input_layernorm: input_opt.take().expect("gemma4 load: input staged once"),
                post_attention_layernorm: post_attn_opt
                    .take()
                    .expect("gemma4 load: post-attn staged once"),
                pre_feedforward_layernorm: pre_ffn_opt
                    .take()
                    .expect("gemma4 load: pre-ffn staged once"),
                post_feedforward_layernorm: post_ffn_opt
                    .take()
                    .expect("gemma4 load: post-ffn staged once"),
                layer_scalar: scalar_opt.take().expect("gemma4 load: scalar staged once"),
                layer_scalar_host: scalar_host_opt
                    .take()
                    .expect("gemma4 load: scalar host staged"),
                q_proj: q_opt.take().expect("gemma4 load: q_proj staged once"),
                k_proj: k_opt.take().expect("gemma4 load: k_proj staged once"),
                v_proj: v_opt.take().expect("gemma4 load: v_proj staged once"),
                o_proj: o_opt.take().expect("gemma4 load: o_proj staged once"),
                q_norm: qn_opt.take().expect("gemma4 load: q_norm staged once"),
                k_norm: kn_opt.take().expect("gemma4 load: k_norm staged once"),
                gate_proj: gate_opt.take().expect("gemma4 load: gate staged once"),
                up_proj: up_opt.take().expect("gemma4 load: up staged once"),
                down_proj: down_opt.take().expect("gemma4 load: down staged once"),
                moe: moe_opt.take(),
            }))
        }
        LayerType::Full => {
            let hd = config.full_head_dim;
            let kv_dim = config.full_n_kv_heads * hd;
            let q_dim = config.n_heads * hd;
            // v_opt (declared above) stays None here: full layers reuse k_proj's
            // pre-norm output as V, so the slot stages and frees nothing.
            check!("layer_scalar");
            match load_layer_scalar(hfq, gpu, &format!("{p}.layer_scalar")) {
                Ok((t, h)) => {
                    scalar_opt = Some(t);
                    scalar_host_opt = Some(h);
                }
                Err(e) => fail!(e),
            }
            if i <= 6 {
                eprintln!(
                    "[gemma4] L{i} full layer_scalar = {}",
                    scalar_host_opt
                        .as_ref()
                        .expect("gemma4 load: scalar staged")
                );
            }
            check!("moe");
            moe_opt = if config.enable_moe_block {
                match load_moe_layer_extras(hfq, gpu, &p, config, fault.as_deref_mut()) {
                    Ok(m) => Some(m),
                    Err(e) => fail!(e),
                }
            } else {
                None
            };
            stage!(
                input_opt,
                "input_layernorm",
                load_gemma4_norm(hfq, gpu, &format!("{p}.input_layernorm.weight"), config.dim,)
            );
            stage!(
                post_attn_opt,
                "post_attention_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_attention_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                pre_ffn_opt,
                "pre_feedforward_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                post_ffn_opt,
                "post_feedforward_layernorm",
                load_gemma4_norm(
                    hfq,
                    gpu,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    config.dim,
                )
            );
            stage!(
                q_opt,
                "q_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    config.dim,
                )
            );
            stage!(
                k_opt,
                "k_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    config.dim,
                )
            );
            // no v_proj on full layers — V reuses k_proj's pre-norm output.
            stage!(
                o_opt,
                "o_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    config.dim,
                    q_dim,
                )
            );
            stage!(
                qn_opt,
                "q_norm",
                load_gemma4_head_norm(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), hd,)
            );
            stage!(
                kn_opt,
                "k_norm",
                load_gemma4_head_norm(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), hd,)
            );
            // no v_norm weight — v_norm is no-scale (ones buffer passed at decode time).
            stage!(
                gate_opt,
                "gate_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.gate_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )
            );
            stage!(
                up_opt,
                "up_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.up_proj.weight"),
                    config.hidden_dim,
                    config.dim,
                )
            );
            stage!(
                down_opt,
                "down_proj",
                load_gemma4_weight(
                    hfq,
                    gpu,
                    &format!("{p}.mlp.down_proj.weight"),
                    config.dim,
                    config.hidden_dim,
                )
            );
            // Full layers never load v_proj: the slot stays None and frees nothing.
            debug_assert!(v_opt.is_none(), "gemma4 load: full layers stage no v_proj");
            Ok(LayerWeights::Full(FullLayerWeights {
                input_layernorm: input_opt.take().expect("gemma4 load: input staged once"),
                post_attention_layernorm: post_attn_opt
                    .take()
                    .expect("gemma4 load: post-attn staged once"),
                pre_feedforward_layernorm: pre_ffn_opt
                    .take()
                    .expect("gemma4 load: pre-ffn staged once"),
                post_feedforward_layernorm: post_ffn_opt
                    .take()
                    .expect("gemma4 load: post-ffn staged once"),
                layer_scalar: scalar_opt.take().expect("gemma4 load: scalar staged once"),
                layer_scalar_host: scalar_host_opt
                    .take()
                    .expect("gemma4 load: scalar host staged"),
                q_proj: q_opt.take().expect("gemma4 load: q_proj staged once"),
                k_proj: k_opt.take().expect("gemma4 load: k_proj staged once"),
                // no v_proj — V = pre-k_norm output of k_proj
                o_proj: o_opt.take().expect("gemma4 load: o_proj staged once"),
                q_norm: qn_opt.take().expect("gemma4 load: q_norm staged once"),
                k_norm: kn_opt.take().expect("gemma4 load: k_norm staged once"),
                gate_proj: gate_opt.take().expect("gemma4 load: gate staged once"),
                up_proj: up_opt.take().expect("gemma4 load: up staged once"),
                down_proj: down_opt.take().expect("gemma4 load: down staged once"),
                moe: moe_opt.take(),
            }))
        }
    }
}

/// One-time init for the scratch buffers that must hold a constant value
/// across forward passes (notably the ones-filled `v_norm_ones_full`).
/// Call once after `Gemma4Scratch::new` before the first forward pass.
pub fn init_scratch_constants(
    gpu: &mut Gpu,
    scratch: &Gemma4Scratch,
    full_head_dim: usize,
) -> HipResult<()> {
    let ones: Vec<f32> = vec![1.0; full_head_dim];
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
    gpu.hip.memcpy_htod(&scratch.v_norm_ones_full.buf, bytes)?;
    Ok(())
}

// ─── Scratch ────────────────────────────────────────────────────────────

use hip_bridge::DeviceBuffer;
/// Flash tile size for gemma4 lowered path. Matches the HIP partition kernel's
/// `TILE_SIZE = 128`.
pub const GEMMA4_FLASH_TILE: usize = 128;

/// Pure geometry: single-query flash partial length for `max_seq`.
/// `n_heads * ceil(max_seq / TILE) * (2 + head_dim)` floats.
#[inline]
pub fn gemma4_flash_partials_len(max_seq: usize, n_heads: usize, full_head_dim: usize) -> usize {
    let tiles = max_seq.div_ceil(GEMMA4_FLASH_TILE);
    n_heads * tiles * (2 + full_head_dim)
}

/// Convenience for `Gemma4Config`.
#[inline]
pub fn gemma4_flash_partials_len_for_config(max_seq: usize, config: &Gemma4Config) -> usize {
    gemma4_flash_partials_len(max_seq, config.n_heads, config.full_head_dim)
}

/// Single-query flash partial length that covers every decode attend tier the
/// lowered path can run on `arch`: the asym3 hd512 tile ([`GEMMA4_FLASH_TILE`])
/// and the Q8 decode tile (`q8_flash_tile_size`) for the full tier at `max_seq`
/// and the sliding ring at `min(sliding_window, max_seq)`. The Q8 tile can be
/// smaller than 128 (tile32 on gfx1100 up to 8K), which needs more partials.
pub fn gemma4_decode_flash_partials_len(
    arch: &str,
    config: &Gemma4Config,
    max_seq: usize,
) -> usize {
    let q8_len = |n_kv_heads: usize, head_dim: usize, cap: usize| {
        let tile = rdna_compute::attention::q8_flash_tile_size(
            arch,
            config.n_heads,
            n_kv_heads,
            head_dim,
            cap,
        );
        config.n_heads * cap.div_ceil(tile) * (2 + head_dim)
    };
    let sliding_cap = config.sliding_window.min(max_seq);
    gemma4_flash_partials_len_for_config(max_seq, config)
        .max(q8_len(
            config.full_n_kv_heads,
            config.full_head_dim,
            max_seq,
        ))
        .max(q8_len(
            config.sliding_n_kv_heads,
            config.sliding_head_dim,
            sliding_cap,
        ))
}

/// Per-decode scratch, sized once at model-load time against the MAX of
/// sliding and full attention dimensions so a single buffer works across
/// layer types. 31B target shapes: sliding Q=[32*256]=8192, full Q=[32*512]=16384
/// → size Q at 16384. Sliding KV=[16*256]=4096, full KV=[4*512]=2048 → size at 4096.
pub struct Gemma4Scratch {
    pub x: GpuTensor,        // [dim] — hidden state
    pub residual: GpuTensor, // [dim] — saved for sandwich residual
    pub tmp: GpuTensor,      // [dim] — norm output scratch
    /// `[dim]` FWHT-rotated normed activation shared by MagnumQuant projections.
    pub x_rot: GpuTensor,

    /// Position buffer (single i32 on device, updated per decode step).
    pub pos_buf: DeviceBuffer,

    // Attention scratch — sized for max(sliding, full)
    pub q: GpuTensor, // [max(n_heads*head_dim_sliding, n_heads*head_dim_full)]
    pub k: GpuTensor, // [max(n_kv_heads*head_dim for each layer type)]
    pub v: GpuTensor, // [same as k]
    pub attn_out: GpuTensor, // [same as q]

    // MLP scratch
    pub gate_ffn: GpuTensor,   // [hidden_dim]
    pub up_ffn: GpuTensor,     // [hidden_dim]
    pub ffn_hidden: GpuTensor, // [hidden_dim]
    pub ffn_out: GpuTensor,    // [dim]

    // Output
    pub logits: GpuTensor,     // [vocab_size]
    pub sample_buf: GpuTensor, // [2] — (token_id, new_rng_state) for GPU sampling
    pub repeat_buf: GpuTensor, // [1024] — rolling window for repeat penalty

    // Flash attention tile partials. Sized by `gemma4_decode_flash_partials_len`
    // for the LARGER of the cache shapes: full-attn head_dim=512 at the asym3
    // tile (128) or the arch's Q8 decode tile, sliding head_dim=256 over the ring.
    pub flash_partials: GpuTensor,

    // No-scale v_norm ones buffer (full-attn layers compute v_norm without
    // a learned weight — we pass this ones-filled tensor to the existing
    // rmsnorm kernel to get no-scale RMS semantics).
    pub v_norm_ones_full: GpuTensor, // [full_head_dim]

    // ── MoE scratch (26B-A4B only). Zero-sized on dense models. ─────────
    pub moe_cur_mlp: GpuTensor,   // [dim] — rmsnorm(ffn_out, post_norm_1)
    pub moe_pre2: GpuTensor,      // [dim] — rmsnorm(attn_out, pre_norm_2)
    pub moe_router_in: GpuTensor, // [dim] — router input (post-rmsnorm + scale)
    pub moe_router_logits: GpuTensor, // [n_experts]
    pub moe_topk_indices: GpuTensor, // [top_k_experts] — i32 packed in f32 slots
    pub moe_topk_weights: GpuTensor, // [top_k_experts]
    pub moe_cur_moe: GpuTensor,   // [dim] — accumulator across top-K experts
    pub moe_expert_gate_up: GpuTensor, // [2 * moe_intermediate_size]
    pub moe_expert_hidden: GpuTensor, // [moe_intermediate_size] — gelu(gate) * up
    pub moe_expert_out: GpuTensor, // [dim] — single expert's down_proj output

    // ── Indexed MoE batched scratch (k_top=8 hardcoded by kernel). ──────
    // These back the device-side fused path that replaces the 8-iteration
    // per-expert CPU loop. Only allocated when the MoE branch is enabled.
    /// `[dim]` — moe_pre2 after one FWHT pass (MQ4 gate_up expects pre-rotated x).
    pub moe_pre2_rot: GpuTensor,
    /// `[k_top × 2 × mi]` — fused gate+up output, one row per top-K rank.
    pub moe_expert_gate_batch: GpuTensor, // [k_top × mi]
    pub moe_expert_up_batch: GpuTensor, // [k_top × mi]

    /// `[k_top × mi]` — gelu_tanh(gate)*up batched over k_top experts.
    pub moe_expert_hidden_batch: GpuTensor,
}

impl Gemma4Scratch {
    /// `max_seq` is the sole allocation authority — MUST equal `LoadCtx::max_seq`
    /// and the `KvCache::max_seq` / `physical_cap` for the paired caches.
    /// `flash_partials` is sized from this single value; the `HIPFIRE_KV_SEQ`
    /// env var is no longer consulted.
    pub fn new(gpu: &mut Gpu, config: &Gemma4Config, max_seq: usize) -> HipResult<Self> {
        Self::new_with_alloc(
            gpu,
            config,
            max_seq,
            |gpu, shape, dtype| gpu.zeros(shape, dtype),
            |gpu| gpu.hip.malloc(4),
        )
    }

    /// Injectable-allocator twin of [`new`] for GPU failure/retry regressions:
    /// tests fail the n-th allocation and prove every staged owner (tensors +
    /// `pos_buf`) rolls back into the pool for immediate retry. Mirrors
    /// qwen35's `new_opt_with_alloc` seam. No env knob, no global sweep.
    fn new_with_alloc(
        gpu: &mut Gpu,
        config: &Gemma4Config,
        max_seq: usize,
        mut alloc: impl FnMut(&mut Gpu, &[usize], DType) -> HipResult<GpuTensor>,
        mut alloc_pos: impl FnMut(&mut Gpu) -> HipResult<DeviceBuffer>,
    ) -> HipResult<Self> {
        // Library code must not abort hosts: reject tiny contexts as an error
        // BEFORE any allocation (admission refuses these up front; direct
        // `new` callers in examples/tools get a clean Err, not a panic).
        if max_seq < GEMMA4_FLASH_TILE {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "gemma4 scratch: max_seq {max_seq} too small (minimum one flash tile = 128)"
                ),
            ));
        }
        let dim = config.dim;
        let q_dim =
            (config.n_heads * config.sliding_head_dim).max(config.n_heads * config.full_head_dim);
        let kv_dim = (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim);

        // Transactional construction: slots own every tensor until take-once
        // publish, pos_slot owns the raw position buffer. Any failure
        // reverse-drains all staged owners (GpuTensor/DeviceBuffer have no
        // Drop that could release device memory for us).
        let mut slots: Vec<Option<GpuTensor>> = Vec::with_capacity(64);
        let mut pos_slot: Option<DeviceBuffer> = None;
        macro_rules! rollback {
            () => {{
                while let Some(slot) = slots.pop() {
                    if let Some(t) = slot {
                        let _ = gpu.free_tensor(t);
                    }
                }
                if let Some(pos) = pos_slot.take() {
                    let _ = gpu.hip.free(pos);
                }
            }};
        }
        macro_rules! alloc {
            ($shape:expr, $dt:expr) => {{
                match alloc(gpu, $shape, $dt) {
                    Ok(t) => {
                        slots.push(Some(t));
                        slots.len() - 1
                    }
                    Err(e) => {
                        rollback!();
                        return Err(e);
                    }
                }
            }};
        }
        macro_rules! take {
            ($i:expr) => {
                slots[$i].take().expect("gemma4 scratch slot taken twice")
            };
        }

        let i_x = alloc!(&[dim], DType::F32);
        let i_residual = alloc!(&[dim], DType::F32);
        let i_tmp = alloc!(&[dim], DType::F32);
        let i_x_rot = alloc!(&[dim], DType::F32);

        match alloc_pos(gpu) {
            Ok(pos) => {
                pos_slot = Some(pos);
            }
            Err(e) => {
                rollback!();
                return Err(e);
            }
        }

        let i_q = alloc!(&[q_dim], DType::F32);
        let i_k = alloc!(&[kv_dim], DType::F32);
        let i_v = alloc!(&[kv_dim], DType::F32);
        let i_attn_out = alloc!(&[q_dim], DType::F32);

        let i_gate_ffn = alloc!(&[config.hidden_dim], DType::F32);
        let i_up_ffn = alloc!(&[config.hidden_dim], DType::F32);
        let i_ffn_hidden = alloc!(&[config.hidden_dim], DType::F32);
        let i_ffn_out = alloc!(&[dim], DType::F32);
        let i_logits = alloc!(&[config.vocab_size], DType::F32);
        let i_sample_buf = alloc!(&[2], DType::F32);
        let i_repeat_buf = alloc!(&[1024], DType::F32);

        // Flash partials sizing. Per-head × max_tiles × (2 + head_dim) floats,
        // covering the full tier (head_dim=512, stride 514) in either KV format
        // and the sliding q8 ring (256, stride 258) at this arch's Q8 decode
        // tile. `max_seq` is the single authority shared with both KV caches —
        // no independent `HIPFIRE_KV_SEQ` env var.
        let flash_partials_sz = gemma4_decode_flash_partials_len(&gpu.arch, config, max_seq);
        let i_flash_partials = alloc!(&[flash_partials_sz], DType::F32);

        // (Note 2026-05-19): removed the precomputed sliding/full cos+sin
        // tables that were allocated here but never read by any kernel.
        // `rope_f32` and `rope_partial_halved_f32` compute cos/sin inline
        // from `pos_buf[0]` + `freq_base`; the lookup-table path was wired
        // but never finished. The tables ate 4 * max_kv_seq * head_dim
        // floats — 768 MB at max_kv_seq=131072 — which pushed the full KV
        // cache (~970 MB at 128k) out of VRAM into GTT (PCIe-paged system
        // RAM), causing attention reads to take a slow path.

        // v_norm ones — populated on first use in the forward pass.
        // Allocated up to the LARGER of sliding_head_dim and full_head_dim
        // since both sliding and full apply no-scale v_norm (the post-rebase
        // fix added v_norm to sliding_layer_decode; sliding head_dim=256,
        // full head_dim=512 → max=512 covers both).
        let v_norm_max = config.sliding_head_dim.max(config.full_head_dim);
        let i_v_norm_ones_full = alloc!(&[v_norm_max], DType::F32);

        // MoE scratch. Allocated unconditionally because the buffers are tiny
        // relative to the model; zero-sized on dense models would just complicate
        // the dispatch path. Sized for 26B-A4B: n_experts=128, top_k=8, mi=704.
        let n_exp = config.num_experts.max(1);
        let mi = config.moe_intermediate_size.max(1);
        let k_top = config.top_k_experts.max(1);
        let i_moe_cur_mlp = alloc!(&[dim], DType::F32);
        let i_moe_pre2 = alloc!(&[dim], DType::F32);
        let i_moe_router_in = alloc!(&[dim], DType::F32);
        let i_moe_router_logits = alloc!(&[n_exp], DType::F32);
        let i_moe_topk_indices = alloc!(&[k_top], DType::F32);
        let i_moe_topk_weights = alloc!(&[k_top], DType::F32);
        let i_moe_cur_moe = alloc!(&[dim], DType::F32);
        let i_moe_expert_gate_up = alloc!(&[2 * mi], DType::F32);
        let i_moe_expert_hidden = alloc!(&[mi], DType::F32);
        let i_moe_expert_out = alloc!(&[dim], DType::F32);

        // Indexed-MoE scratch (k_top fixed at 8 by the kernel).
        let i_moe_pre2_rot = alloc!(&[dim], DType::F32);
        let i_moe_expert_gate_batch = alloc!(&[k_top * mi], DType::F32);
        let i_moe_expert_up_batch = alloc!(&[k_top * mi], DType::F32);
        let i_moe_expert_hidden_batch = alloc!(&[k_top * mi], DType::F32);

        let scratch = Gemma4Scratch {
            x: take!(i_x),
            residual: take!(i_residual),
            tmp: take!(i_tmp),
            x_rot: take!(i_x_rot),
            pos_buf: pos_slot.take().expect("gemma4 scratch pos_buf staged once"),
            q: take!(i_q),
            k: take!(i_k),
            v: take!(i_v),
            attn_out: take!(i_attn_out),
            gate_ffn: take!(i_gate_ffn),
            up_ffn: take!(i_up_ffn),
            ffn_hidden: take!(i_ffn_hidden),
            ffn_out: take!(i_ffn_out),
            logits: take!(i_logits),
            sample_buf: take!(i_sample_buf),
            repeat_buf: take!(i_repeat_buf),
            flash_partials: take!(i_flash_partials),
            v_norm_ones_full: take!(i_v_norm_ones_full),
            moe_cur_mlp: take!(i_moe_cur_mlp),
            moe_pre2: take!(i_moe_pre2),
            moe_router_in: take!(i_moe_router_in),
            moe_router_logits: take!(i_moe_router_logits),
            moe_topk_indices: take!(i_moe_topk_indices),
            moe_topk_weights: take!(i_moe_topk_weights),
            moe_cur_moe: take!(i_moe_cur_moe),
            moe_expert_gate_up: take!(i_moe_expert_gate_up),
            moe_expert_hidden: take!(i_moe_expert_hidden),
            moe_expert_out: take!(i_moe_expert_out),
            moe_pre2_rot: take!(i_moe_pre2_rot),
            moe_expert_gate_batch: take!(i_moe_expert_gate_batch),
            moe_expert_up_batch: take!(i_moe_expert_up_batch),
            moe_expert_hidden_batch: take!(i_moe_expert_hidden_batch),
        };
        debug_assert!(
            slots.iter().all(|s| s.is_none()) && pos_slot.is_none(),
            "gemma4 scratch: every staged owner published exactly once"
        );
        Ok(scratch)
    }

    /// Release every GPU allocation owned by this scratch. Mirrors the
    /// Qwen35Scratch / LlamaScratch pattern so `unload_model` in the daemon
    /// can reclaim VRAM on idle eviction.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.x_rot);
        // pos_buf is a raw DeviceBuffer (not a GpuTensor): free it explicitly —
        // DeviceBuffer has no Drop-side free. Eager precedent: gemma4.rs free_gpu.
        let _ = gpu.hip.free(self.pos_buf);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up_ffn);
        let _ = gpu.free_tensor(self.ffn_hidden);
        let _ = gpu.free_tensor(self.ffn_out);
        let _ = gpu.free_tensor(self.logits);
        let _ = gpu.free_tensor(self.sample_buf);
        let _ = gpu.free_tensor(self.repeat_buf);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.v_norm_ones_full);
        let _ = gpu.free_tensor(self.moe_cur_mlp);
        let _ = gpu.free_tensor(self.moe_pre2);
        let _ = gpu.free_tensor(self.moe_router_in);
        let _ = gpu.free_tensor(self.moe_router_logits);
        let _ = gpu.free_tensor(self.moe_topk_indices);
        let _ = gpu.free_tensor(self.moe_topk_weights);
        let _ = gpu.free_tensor(self.moe_cur_moe);
        let _ = gpu.free_tensor(self.moe_expert_gate_up);
        let _ = gpu.free_tensor(self.moe_expert_hidden);
        let _ = gpu.free_tensor(self.moe_expert_out);
        let _ = gpu.free_tensor(self.moe_pre2_rot);
        let _ = gpu.free_tensor(self.moe_expert_gate_batch);
        let _ = gpu.free_tensor(self.moe_expert_up_batch);
        let _ = gpu.free_tensor(self.moe_expert_hidden_batch);
    }
}
#[cfg(test)]
mod scratch_geometry_tests {
    use super::*;

    #[test]
    fn flash_partials_geometry_matches_formula() {
        // Single tile edge
        assert_eq!(gemma4_flash_partials_len(128, 32, 512), 32 * 1 * 514);
        assert_eq!(gemma4_flash_partials_len(129, 32, 512), 32 * 2 * 514);
        assert_eq!(gemma4_flash_partials_len(256, 32, 512), 32 * 2 * 514);
        // 32k baseline (FALLBACK_KV_SEQ before fix)
        assert_eq!(gemma4_flash_partials_len(32768, 32, 512), 32 * 256 * 514);
        assert_eq!(gemma4_flash_partials_len(32768, 32, 512), 4_210_688);
        // 131072 must be exactly 4× the 32768 geometry (no overflow, no env var)
        assert_eq!(gemma4_flash_partials_len(131072, 32, 512), 32 * 1024 * 514);
        assert_eq!(gemma4_flash_partials_len(131072, 32, 512), 16_842_752);
        assert_eq!(
            gemma4_flash_partials_len(131072, 32, 512),
            4 * gemma4_flash_partials_len(32768, 32, 512)
        );
    }

    #[test]
    fn flash_scales_invariant_to_tile_rounding() {
        // Non-multiple of 128 must ceil
        let tiles_32769 = 32769usize.div_ceil(GEMMA4_FLASH_TILE);
        assert_eq!(tiles_32769, 257);
        assert_eq!(gemma4_flash_partials_len(32769, 32, 512), 32 * 257 * 514);
        // 131071 is one short of 131072 -> still 1024 tiles (ceil)
        assert_eq!(131071usize.div_ceil(GEMMA4_FLASH_TILE), 1024);
        assert_eq!(
            gemma4_flash_partials_len(131071, 32, 512),
            gemma4_flash_partials_len(131072, 32, 512)
        );
    }

    fn dummy_cfg_31b() -> Gemma4Config {
        // Minimal config mirroring 31B/26B shapes: n_heads=32, full_head_dim=512
        Gemma4Config {
            dim: 5376,
            n_layers: 40,
            vocab_size: 262144,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 32,
            sliding_head_dim: 256,
            sliding_n_kv_heads: 16,
            sliding_rope_theta: 10000.0,
            sliding_window: 1024,
            full_head_dim: 512,
            full_n_kv_heads: 4,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            hidden_dim: 21504,
            enable_moe_block: false,
            moe_intermediate_size: 704,
            num_experts: 128,
            top_k_experts: 8,
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            embed_scale: (5376 as f32).sqrt(),
            layer_types: vec![LayerType::Sliding; 40],
            has_vision: false,
            image_token_id: 258880,
            boi_token_id: 255999,
            eoi_token_id: 258882,
            audio_token_id: 258881,
            video_token_id: 258884,
        }
    }

    #[test]
    fn hfq4g128_rows_must_start_fresh_groups() {
        // K = 704: 5.5 groups per row. The 2-D packer pads each row to 6 groups.
        assert!(check_hfq4g128_rows("w", 2816, 704, 2816 * 6 * 72).is_ok());
        // The pre-b4846285e flat packer straddles rows: refused with its cause.
        let err = check_hfq4g128_rows("w", 2816, 704, 15488 * 72).unwrap_err();
        assert!(err.to_string().contains("across rows"), "{err}");
        assert!(check_hfq4g128_rows("w", 2816, 704, 2816 * 6 * 72 - 72).is_err());
        // Group-aligned K: both packers agree.
        assert!(check_hfq4g128_rows("w", 4, 256, 4 * 2 * 72).is_ok());
    }

    #[test]
    fn decode_partials_cover_q8_decode_tile() {
        let cfg = dummy_cfg_31b();
        // gfx1100 decodes Q8 with tile32 up to 8K: 4x the tile-128 partials.
        assert_eq!(
            gemma4_decode_flash_partials_len("gfx1100", &cfg, 8192),
            32 * (8192 / 32) * 514
        );
        assert_eq!(
            gemma4_decode_flash_partials_len("gfx1100", &cfg, 8192),
            4 * gemma4_flash_partials_len_for_config(8192, &cfg)
        );
        // Tile128 arches keep the tile-128 geometry.
        for arch in ["gfx1201", "gfx1151"] {
            for max_seq in [128usize, 8192, 32768] {
                assert_eq!(
                    gemma4_decode_flash_partials_len(arch, &cfg, max_seq),
                    gemma4_flash_partials_len_for_config(max_seq, &cfg)
                );
            }
        }
        // Past 8K gfx1100 is tile128 too.
        assert_eq!(
            gemma4_decode_flash_partials_len("gfx1100", &cfg, 32768),
            gemma4_flash_partials_len_for_config(32768, &cfg)
        );
        // Small contexts: the sliding q8 ring (tile32) must fit as well.
        let small = gemma4_decode_flash_partials_len("gfx1100", &cfg, 1000);
        assert!(small >= 32 * 1000usize.div_ceil(32) * 258);
        assert!(small >= 32 * 1000usize.div_ceil(32) * 514);
    }
}
#[cfg(test)]
mod load_rollback_tests {
    use super::*;
    use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
    use std::cell::Cell;

    fn try_gpu() -> Option<Gpu> {
        match Gpu::init() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("skip: no GPU ({e:?})");
                None
            }
        }
    }

    fn tiny_config(layers: Vec<LayerType>, moe: bool) -> Gemma4Config {
        Gemma4Config {
            dim: 16,
            n_layers: layers.len(),
            vocab_size: 16,
            norm_eps: 1e-6,
            bos_token: 2,
            eos_token: 1,
            pad_token: 0,
            n_heads: 2,
            sliding_head_dim: 8,
            sliding_n_kv_heads: 2,
            sliding_rope_theta: 10_000.0,
            sliding_window: 32,
            full_head_dim: 8,
            full_n_kv_heads: 2,
            full_rope_theta: 1_000_000.0,
            full_rope_type: RopeType::Proportional,
            full_partial_rotary_factor: 0.25,
            attention_k_eq_v: true,
            hidden_dim: 32,
            enable_moe_block: moe,
            moe_intermediate_size: 8,
            num_experts: if moe { 2 } else { 0 },
            top_k_experts: if moe { 2 } else { 0 },
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            embed_scale: 4.0,
            layer_types: layers,
            has_vision: false,
            image_token_id: 0,
            boi_token_id: 0,
            eoi_token_id: 0,
            audio_token_id: 0,
            video_token_id: 0,
        }
    }

    fn mem(name: String, quant_type: u8, shape: Vec<u32>, data: Vec<u8>) -> HfqMemTensor {
        HfqMemTensor {
            name,
            quant_type,
            shape,
            group_size: 0,
            data,
        }
    }

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn ones_f32(n: usize) -> Vec<u8> {
        f32_bytes(&vec![1.0f32; n])
    }

    /// Dense projections as MQ4G256 blobs (quant_type 13) with F16 AWQ sidecars
    /// so rollback/unload must reclaim sidecars too — the pre-fix free_gpu
    /// leaked them via free_tensor-on-buf.
    fn proj_entries(p: &str, rel: &str, m: usize, k: usize) -> Vec<HfqMemTensor> {
        let name = format!("{p}.{rel}");
        let awq_name = match name.strip_suffix(".weight") {
            Some(stem) => format!("{stem}.awq_scale.weight"),
            None => format!("{name}.awq_scale.weight"),
        };
        vec![
            mem(name, 13, vec![m as u32, k as u32], vec![0u8; 256]),
            mem(awq_name, 1, vec![k as u32], vec![0u8; k * 2]),
        ]
    }

    fn dense_layer_entries(cfg: &Gemma4Config, i: usize) -> Vec<HfqMemTensor> {
        let p = format!("model.language_model.layers.{i}");
        let mut v = vec![
            mem(format!("{p}.layer_scalar"), 2, vec![1], f32_bytes(&[1.0])),
            mem(
                format!("{p}.input_layernorm.weight"),
                2,
                vec![cfg.dim as u32],
                ones_f32(cfg.dim),
            ),
            mem(
                format!("{p}.post_attention_layernorm.weight"),
                2,
                vec![cfg.dim as u32],
                ones_f32(cfg.dim),
            ),
            mem(
                format!("{p}.pre_feedforward_layernorm.weight"),
                2,
                vec![cfg.dim as u32],
                ones_f32(cfg.dim),
            ),
            mem(
                format!("{p}.post_feedforward_layernorm.weight"),
                2,
                vec![cfg.dim as u32],
                ones_f32(cfg.dim),
            ),
        ];
        let (hd, n_kv, has_v) = match cfg.layer_types[i] {
            LayerType::Sliding => (cfg.sliding_head_dim, cfg.sliding_n_kv_heads, true),
            LayerType::Full => (cfg.full_head_dim, cfg.full_n_kv_heads, false),
        };
        let kv_dim = n_kv * hd;
        let q_dim = cfg.n_heads * hd;
        let mut projs = vec![
            ("self_attn.q_proj.weight", q_dim, cfg.dim),
            ("self_attn.k_proj.weight", kv_dim, cfg.dim),
        ];
        if has_v {
            projs.push(("self_attn.v_proj.weight", kv_dim, cfg.dim));
        }
        projs.extend([
            ("self_attn.o_proj.weight", cfg.dim, q_dim),
            ("mlp.gate_proj.weight", cfg.hidden_dim, cfg.dim),
            ("mlp.up_proj.weight", cfg.hidden_dim, cfg.dim),
            ("mlp.down_proj.weight", cfg.dim, cfg.hidden_dim),
        ]);
        for (rel, m, k) in projs {
            v.extend(proj_entries(&p, rel, m, k));
        }
        v.push(mem(
            format!("{p}.self_attn.q_norm.weight"),
            2,
            vec![hd as u32],
            ones_f32(hd),
        ));
        v.push(mem(
            format!("{p}.self_attn.k_norm.weight"),
            2,
            vec![hd as u32],
            ones_f32(hd),
        ));
        v
    }

    fn moe_entries(cfg: &Gemma4Config, i: usize) -> Vec<HfqMemTensor> {
        let p = format!("model.language_model.layers.{i}");
        let n_exp = cfg.num_experts;
        let dim = cfg.dim;
        let mut v = proj_entries(&p, "router.proj.weight", n_exp, dim);
        v.push(mem(
            format!("{p}.router.scale"),
            2,
            vec![dim as u32],
            ones_f32(dim),
        ));
        v.push(mem(
            format!("{p}.router.per_expert_scale"),
            2,
            vec![n_exp as u32],
            ones_f32(n_exp),
        ));
        for rel in [
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ] {
            v.push(mem(
                format!("{p}.{rel}"),
                2,
                vec![dim as u32],
                ones_f32(dim),
            ));
        }
        // Q8_0 expert blobs: content is never executed, only uploaded into
        // the pools; equal sizes keep the pool concat path happy.
        for x in 0..n_exp {
            v.push(mem(
                format!("{p}.experts.{x}.gate_up_proj.weight"),
                3,
                vec![64],
                vec![0u8; 64],
            ));
            v.push(mem(
                format!("{p}.experts.{x}.down_proj.weight"),
                3,
                vec![64],
                vec![0u8; 64],
            ));
        }
        v
    }

    fn fixture_tensors(cfg: &Gemma4Config) -> Vec<HfqMemTensor> {
        let mut v = vec![
            mem(
                "model.language_model.embed_tokens.weight".to_string(),
                2,
                vec![cfg.vocab_size as u32, cfg.dim as u32],
                ones_f32(cfg.vocab_size * cfg.dim),
            ),
            mem(
                "model.language_model.norm.weight".to_string(),
                2,
                vec![cfg.dim as u32],
                ones_f32(cfg.dim),
            ),
        ];
        for i in 0..cfg.n_layers {
            v.extend(dense_layer_entries(cfg, i));
            if cfg.enable_moe_block {
                v.extend(moe_entries(cfg, i));
            }
        }
        v
    }

    fn open_fixture(tensors: Vec<HfqMemTensor>, tag: &str) -> (HfqFile, std::path::PathBuf) {
        let path =
            std::env::temp_dir().join(format!("gemma4_rollback_{tag}_{}.hfq", std::process::id()));
        write_hfqm_package_mem(&path, 13, "{}", &tensors).expect("write fixture hfq");
        let hfq = HfqFile::open(&path).expect("open fixture hfq");
        (hfq, path)
    }

    fn expect_injected(err: hip_bridge::HipError, fail_at: usize) {
        let msg = format!("{err:?}");
        assert!(
            msg.contains("injected gemma4"),
            "fail_at={fail_at}: expected the injected fault, got {msg}"
        );
    }

    #[test]
    #[ignore = "requires an AMD GPU; proves tiny max_seq is a rollback-safe Err, not a host abort"]
    fn scratch_rejects_small_max_seq_before_allocating() {
        let Some(mut gpu) = try_gpu() else { return };
        let cfg = tiny_config(vec![LayerType::Sliding], false);
        let before = gpu.pool_stats();
        for max_seq in [0usize, 1, 64, 127] {
            match Gemma4Scratch::new(&mut gpu, &cfg, max_seq) {
                Ok(_) => panic!("max_seq={max_seq} must fail"),
                Err(e) => assert!(
                    format!("{e:?}").contains("too small"),
                    "unexpected error for max_seq={max_seq}: {e:?}"
                ),
            }
        }
        assert_eq!(
            gpu.pool_stats(),
            before,
            "rejected scratch must not allocate"
        );
        Gemma4Scratch::new(&mut gpu, &cfg, 128)
            .expect("floor value loads")
            .free_gpu(&mut gpu);
        gpu.drain_pool();
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises real allocation rollback and retry"]
    fn scratch_alloc_failure_at_any_point_reclaims_all_and_retry_succeeds() {
        let Some(mut gpu) = try_gpu() else { return };
        let cfg = tiny_config(vec![LayerType::Sliding], false);
        // Two warm create/free cycles settle the pool: on the first reuse pass
        // a larger pooled block can satisfy a smaller same-bucket request
        // (LIFO pop), orphaning the smaller block to hip.free plus one fresh
        // alloc. After that the size distribution is stable, so the baseline
        // below proves rollback reclaims rather than warmup noise.
        for _ in 0..2 {
            Gemma4Scratch::new(&mut gpu, &cfg, 128)
                .expect("warm scratch")
                .free_gpu(&mut gpu);
        }
        let fresh = gpu.pool_stats().0;
        // Count staged allocations in one clean pass; the sweep below fails
        // each observed allocation in turn, so no count is pinned here.
        let total = {
            let n = Cell::new(0usize);
            let s = Gemma4Scratch::new_with_alloc(
                &mut gpu,
                &cfg,
                128,
                |g, shape, dt| {
                    n.set(n.get() + 1);
                    g.alloc_tensor(shape, dt)
                },
                |g| {
                    n.set(n.get() + 1);
                    g.hip.malloc(4)
                },
            )
            .expect("counting pass");
            let total = n.get();
            s.free_gpu(&mut gpu);
            total
        };
        assert_eq!(gpu.pool_stats().0, fresh);
        // Fail every staged allocation in turn — including the pos_buf slot
        // and the last pb_bf16 tail. Pool-stat plateau observes pooled
        // owners exactly; the 4-byte pos_buf is covered by the explicit
        // hip.free arm plus retry success.
        for fail_at in 1..=total {
            let n = Cell::new(0usize);
            let fail = |n: &Cell<usize>| {
                n.set(n.get() + 1);
                n.get() == fail_at
            };
            let r = Gemma4Scratch::new_with_alloc(
                &mut gpu,
                &cfg,
                128,
                |g, shape, dt| {
                    if fail(&n) {
                        Err(hip_bridge::HipError::new(0, "injected scratch alloc fault"))
                    } else {
                        g.alloc_tensor(shape, dt)
                    }
                },
                |g| {
                    if fail(&n) {
                        Err(hip_bridge::HipError::new(0, "injected scratch alloc fault"))
                    } else {
                        g.hip.malloc(4)
                    }
                },
            );
            match r {
                Ok(_) => panic!("fail_at={fail_at} must fail"),
                Err(_) => assert_eq!(
                    gpu.pool_stats().0,
                    fresh,
                    "fail_at={fail_at} leaked pooled owners"
                ),
            }
        }
        Gemma4Scratch::new(&mut gpu, &cfg, 128)
            .expect("immediate retry after allocation failure")
            .free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh,
            "retry leaked instead of reusing the warm pool"
        );
        gpu.drain_pool();
    }

    /// Exhaustive fail-point sweep over one fixture: fail every staged step,
    /// prove each failure is the injected one and the pool plateaus, then
    /// prove the public path retries clean and unloads (sidecars included).
    fn sweep_fixture(tag: &str, cfg: &Gemma4Config) {
        let Some(mut gpu) = try_gpu() else { return };
        let (mut hfq, path) = open_fixture(fixture_tensors(cfg), tag);
        load_weights(&mut hfq, cfg, &mut gpu)
            .expect("warm load")
            .free_gpu(&mut gpu);
        let fresh = gpu.pool_stats().0;
        let mut probe = AllocFaults {
            calls: 0,
            fail_at: None,
        };
        match load_weights_impl(&mut hfq, cfg, &mut gpu, Some(&mut probe)) {
            Ok(w) => w.free_gpu(&mut gpu),
            Err(e) => panic!("counting pass must succeed: {e:?}"),
        }
        let total = probe.calls;
        assert_eq!(gpu.pool_stats().0, fresh);
        for fail_at in 1..=total {
            let mut f = AllocFaults {
                calls: 0,
                fail_at: Some(fail_at),
            };
            match load_weights_impl(&mut hfq, cfg, &mut gpu, Some(&mut f)) {
                Ok(_) => panic!("fail_at={fail_at} must fail"),
                Err(e) => {
                    expect_injected(e, fail_at);
                    assert_eq!(
                        gpu.pool_stats().0,
                        fresh,
                        "fail_at={fail_at} leaked pooled owners"
                    );
                }
            }
        }
        // An unfired hook loads clean: success takes every slot exactly once
        // (any double-take panics on the take().expect publish).
        let mut never = AllocFaults {
            calls: 0,
            fail_at: Some(total + 100),
        };
        match load_weights_impl(&mut hfq, cfg, &mut gpu, Some(&mut never)) {
            Ok(w) => {
                assert_eq!(w.layers.len(), cfg.n_layers);
                w.free_gpu(&mut gpu);
            }
            Err(e) => panic!("unfired hook must load clean: {e:?}"),
        }
        // Public-path retry reuses the warm pool, then unloads everything.
        load_weights(&mut hfq, cfg, &mut gpu)
            .expect("immediate retry after load failure")
            .free_gpu(&mut gpu);
        assert_eq!(
            gpu.pool_stats().0,
            fresh,
            "retry leaked instead of reusing the warm pool"
        );
        gpu.drain_pool();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises staged-owner rollback across layers and AWQ sidecars"]
    fn load_weights_mid_layer_failure_reclaims_completed_layers_and_retry_succeeds() {
        let cfg = tiny_config(vec![LayerType::Sliding, LayerType::Full], false);
        sweep_fixture("dense", &cfg);
    }

    #[test]
    #[ignore = "requires an AMD GPU; exercises MoE pool/pointer-table rollback and retry"]
    fn load_weights_moe_pool_failure_reclaims_and_retry_succeeds() {
        let cfg = tiny_config(vec![LayerType::Sliding], true);
        sweep_fixture("moe", &cfg);
    }

    /// Bounded sidecar-stage regression: fail after the primary buffer is
    /// owned but before a present, valid AWQ sidecar attaches. The outer
    /// staged sweep never covers this nested seam (the WeightTensor is never
    /// built), so the leaf itself must free the primary. Retry must attach
    /// the sidecar and reclaim everything.
    #[test]
    #[ignore = "requires an AMD GPU; exercises sidecar-stage primary reclaim and retry"]
    fn load_weight_sidecar_failure_reclaims_primary_and_retry_succeeds() {
        let Some(mut gpu) = try_gpu() else { return };
        let tensors = vec![
            mem("test.weight".to_string(), 13, vec![8, 16], vec![0u8; 256]),
            mem(
                "test.awq_scale.weight".to_string(),
                1,
                vec![16],
                vec![0u8; 32],
            ),
        ];
        let (hfq, path) = open_fixture(tensors, "sidecar");
        // Warm through the production wrapper: sidecar attaches, then unload.
        match load_gemma4_weight(&hfq, &mut gpu, "test.weight", 8, 16) {
            Ok(w) => {
                assert!(
                    w.awq_scale.is_some(),
                    "fixture sidecar must attach on the clean path"
                );
                w.free_all(&mut gpu);
            }
            Err(e) => panic!("warm sidecar load must succeed: {e:?}"),
        }
        let fresh = gpu.pool_stats().0;
        // Fail at the sidecar stage: primary owned, sidecar never attached.
        let mut f = AllocFaults {
            calls: 0,
            fail_at: Some(1),
        };
        match load_gemma4_weight_impl(&hfq, &mut gpu, "test.weight", 8, 16, Some(&mut f)) {
            Ok(_) => panic!("sidecar-stage fault must fail"),
            Err(e) => {
                expect_injected(e, 1);
                assert_eq!(
                    gpu.pool_stats().0,
                    fresh,
                    "sidecar-stage failure leaked the primary buffer"
                );
            }
        }
        // Retry through the production wrapper reuses the warm pool.
        match load_gemma4_weight(&hfq, &mut gpu, "test.weight", 8, 16) {
            Ok(w) => {
                assert!(w.awq_scale.is_some(), "retry must attach the sidecar");
                w.free_all(&mut gpu);
            }
            Err(e) => panic!("retry after sidecar failure must succeed: {e:?}"),
        }
        assert_eq!(
            gpu.pool_stats().0,
            fresh,
            "retry leaked instead of reusing the warm pool"
        );
        gpu.drain_pool();
        let _ = std::fs::remove_file(&path);
    }
}

// ─── Forward pass ───────────────────────────────────────────────────────

/// Fingerprint the allocations and constants baked into the lowered AR graph.
/// Position is deliberately absent: RoPE, KV writes and attention read pos_buf.
fn lowered_graph_binding(
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    kv_sliding: &hipfire_runtime::llama::KvCache,
    kv_full: &hipfire_runtime::llama::KvCache,
    s: &Gemma4Scratch,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x100_0000_01b3);
    macro_rules! tensors {
        ($($t:expr),* $(,)?) => {
            $(
                mix($t.buf.as_ptr() as u64);
                mix($t.numel() as u64);
                mix($t.dtype as u64);
            )*
        };
    }
    macro_rules! projection {
        ($w:expr) => {{
            let w = &$w;
            tensors!(w.buf);
            for v in [w.m, w.k, w.row_stride, w.gpu_dtype as usize] {
                mix(v as u64);
            }
            if let Some(p) = &w.paro {
                tensors!(p.pairs, p.theta, p.channel_scales);
                mix(p.krot as u64);
                mix(p.group_size as u64);
            }
            if let Some(t) = &w.awq_scale {
                tensors!(t);
            }
            for v in w.lloyd_lut_f16.iter().flatten() {
                mix(*v as u64);
            }
        }};
    }
    tensors!(weights.embed_tokens, weights.final_norm);
    projection!(weights.lm_head);
    mix(weights.embd_format as u64);
    for v in [
        config.dim,
        config.n_layers,
        config.vocab_size,
        config.n_heads,
        config.sliding_head_dim,
        config.sliding_n_kv_heads,
        config.sliding_window,
        config.full_head_dim,
        config.full_n_kv_heads,
        config.hidden_dim,
        config.enable_moe_block as usize,
        config.moe_intermediate_size,
        config.num_experts,
        config.top_k_experts,
        config.attention_k_eq_v as usize,
    ] {
        mix(v as u64);
    }
    for v in [
        config.norm_eps,
        config.sliding_rope_theta,
        config.full_rope_theta,
        config.full_partial_rotary_factor,
        config.final_logit_softcapping,
    ] {
        mix(v.to_bits() as u64);
    }
    for ty in &config.layer_types {
        mix(matches!(ty, LayerType::Full) as u64);
    }
    macro_rules! layer {
        ($lw:expr) => {{
            let lw = $lw;
            tensors!(
                lw.input_layernorm,
                lw.post_attention_layernorm,
                lw.pre_feedforward_layernorm,
                lw.post_feedforward_layernorm,
                lw.layer_scalar,
                lw.q_norm,
                lw.k_norm,
            );
            mix(lw.layer_scalar_host.to_bits() as u64);
            projection!(lw.q_proj);
            projection!(lw.k_proj);
            projection!(lw.o_proj);
            projection!(lw.gate_proj);
            projection!(lw.up_proj);
            projection!(lw.down_proj);
            mix(lw.moe.is_some() as u64);
            if let Some(m) = &lw.moe {
                projection!(m.router_proj);
                tensors!(
                    m.router_scale,
                    m.per_expert_scale,
                    m.pre_feedforward_layernorm_2,
                    m.post_feedforward_layernorm_1,
                    m.post_feedforward_layernorm_2,
                    m.experts_gate_up_pool,
                    m.experts_down_pool,
                    m.experts_gate_up_ptrs,
                    m.experts_down_ptrs,
                );
                if let Some(first) = m.experts.first() {
                    projection!(first.gate_up_proj);
                    projection!(first.down_proj);
                }
            }
        }};
    }
    for lw in &weights.layers {
        match lw {
            LayerWeights::Sliding(lw) => {
                mix(0);
                layer!(lw);
                projection!(lw.v_proj);
            }
            LayerWeights::Full(lw) => {
                mix(1);
                layer!(lw);
            }
        }
    }
    for kv in [kv_sliding, kv_full] {
        for v in [
            kv.max_seq,
            kv.physical_cap,
            kv.compact_offset,
            kv.quant_q8 as usize,
            kv.quant_asym3 as usize,
            kv.quant_fwht as usize,
            kv.quant_asym4 as usize,
            kv.quant_asym2 as usize,
            kv.v_mode_bits() as usize,
        ] {
            mix(v as u64);
        }
        for t in kv.k_gpu.iter().chain(&kv.v_gpu) {
            tensors!(t);
        }
        for t in kv.givens_cos.iter().chain(&kv.givens_sin) {
            tensors!(t);
        }
    }
    mix(s.pos_buf.as_ptr() as u64);
    tensors!(
        s.x,
        s.residual,
        s.tmp,
        s.q,
        s.k,
        s.v,
        s.attn_out,
        s.gate_ffn,
        s.up_ffn,
        s.ffn_hidden,
        s.ffn_out,
        s.logits,
        s.flash_partials,
        s.v_norm_ones_full,
        s.moe_cur_mlp,
        s.moe_pre2,
        s.moe_router_in,
        s.moe_router_logits,
        s.moe_topk_indices,
        s.moe_topk_weights,
        s.moe_cur_moe,
        s.moe_expert_gate_up,
        s.moe_expert_hidden,
        s.moe_expert_out,
        s.moe_pre2_rot,
        s.moe_expert_gate_batch,
        s.moe_expert_up_batch,
        s.moe_expert_hidden_batch,
    );
    h.max(1)
}

/// Single-token decode. Phase 3 implementation.
///
/// Precondition: the loader initializes `scratch.v_norm_ones_full` once before
/// the first forward call.
#[allow(clippy::too_many_arguments)]
pub fn forward_scratch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    token: u32,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let dim = config.dim;

    // 1) Embedding lookup + sqrt(dim) scale.
    // ALWAYS direct — the captured graph can't bake in `token` (varies per
    // call). The embed lookup fills scratch.x; the rest of the forward
    // (which is graph-capturable now that the MoE branch has no D2H syncs)
    // reads from scratch.x. Same split Qwen35 uses for its captured path.
    embed(gpu, weights, &scratch.x, token, dim)?;
    gpu.scale_f32(&scratch.x, config.embed_scale)?;

    // Same explicit per-model override as the hand path; HIPFIRE_GRAPH also
    // controls this carrier. With both switches unset, capture/replay is the
    // measured default on exact gfx1201 only; every other arch stays opt-in.
    static GRAPH_ENV: std::sync::LazyLock<Option<bool>> = std::sync::LazyLock::new(|| {
        let parse = |name| match hipfire_config::developer_var(name).ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        };
        parse("HIPFIRE_GEMMA4_GRAPH").or_else(|| parse("HIPFIRE_GRAPH"))
    });
    let graph_on = (*GRAPH_ENV).unwrap_or(gpu.arch == "gfx1201");
    let eligible = std::mem::replace(&mut gpu.graphs.ar_graph_eligible, true);
    let use_graph = graph_on
        && eligible
        && !gpu.replay.is_recording()
        && kv_sliding.compact_offset == 0
        && kv_full.compact_offset == 0
        // The CPU expert fallback downloads top-K inside the body. Q8 expert
        // down uses atomicAdd, whose scheduling-dependent sum is not exact.
        && weights.layers.iter().all(|layer| {
            let moe = match layer {
                LayerWeights::Sliding(lw) => &lw.moe,
                LayerWeights::Full(lw) => &lw.moe,
            };
            moe.as_ref().map_or(true, |m| m.experts.first().is_some_and(|e| {
                e.down_proj.gpu_dtype == DType::HFQ4G128
                    && hipfire_dispatch::pipeline::sandwich::RoutedExperts::supports(
                        e.gate_up_proj.gpu_dtype,
                        e.down_proj.gpu_dtype,
                    )
            }))
        })
        && hipfire_config::developer_var("HIPFIRE_GEMMA4_DUMP").as_deref() != Ok("1");
    let binding = if use_graph {
        lowered_graph_binding(weights, config, kv_sliding, kv_full, scratch)
    } else {
        0
    };
    if use_graph && gpu.graphs.ar_forward_binding != binding {
        gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
        gpu.graphs.ar_forward_binding = binding;
    }
    if use_graph && gpu.active_stream.is_none() {
        // Embedding above may have run on the default stream. Complete it
        // before switching streams for capture/replay.
        gpu.hip.device_synchronize()?;
        gpu.active_stream = Some(gpu.hip.stream_create()?);
    }
    if use_graph
        && gpu.graphs.ar_forward_replay_enabled
        && !gpu.graphs.ar_forward_kernel_dirty
        && gpu.graphs.graph_exec.is_some()
    {
        gpu.hip.stream_write_value32(
            gpu.active_stream.as_ref().unwrap(),
            &scratch.pos_buf,
            pos as u32,
            0,
        )?;
        if let Err(e) =
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
        {
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
            return Err(e);
        }
    } else if use_graph && !gpu.graphs.ar_forward_kernel_dirty {
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        if let Err(e) = gpu.graphs.begin_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        ) {
            gpu.graphs.capture_mode = false;
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
            return Err(e);
        }
        let body = forward_scratch_inner(gpu, weights, config, pos, kv_sliding, kv_full, scratch);
        // Always close capture, including a failing body, before dropping its
        // blobs. Captured work is never launched when either phase fails.
        let end = gpu.graphs.end_graph_capture(
            &gpu.hip,
            gpu.device_id,
            gpu.active_stream.as_ref().unwrap(),
        );
        if let Err(e) = body.and(end) {
            gpu.graphs.capture_mode = false;
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
            return Err(e);
        }
        gpu.graphs.ar_forward_binding = binding;
        if let Err(e) =
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, gpu.active_stream.as_ref().unwrap())
        {
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
            return Err(e);
        }
        gpu.graphs.ar_forward_replay_enabled = true;
        eprintln!(
            "[gemma4 lowered hipGraph] captured {} blobs, instantiated",
            gpu.graphs.ar_forward_blobs.len(),
        );
    } else {
        if !use_graph && gpu.graphs.graph_exec.is_some() {
            gpu.graphs.graph_destroy(&gpu.hip, gpu.device_id);
        }
        gpu.hip
            .memcpy_htod(&scratch.pos_buf, &(pos as i32).to_ne_bytes())?;
        forward_scratch_inner(gpu, weights, config, pos, kv_sliding, kv_full, scratch)?;
        // A successful direct call JITs the same body before capture.
        if use_graph {
            gpu.graphs.ar_forward_kernel_dirty = false;
        }
    }
    Ok(())
}

/// Graph-capturable body of `forward_scratch`. Caller is responsible for:
///   - embed lookup + embed scale (fills scratch.x — varies per call)
///   - pos_buf update (htod for warmup/capture; stream_write_value32 for replay)
fn forward_scratch_inner(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    use crate::program::{head, Geometry, LayerScratch, ProgramBinding, Resident};
    let hip = |e: String| hip_bridge::HipError::new(0, &e);
    let geo = Geometry::lowered(config);
    let binding = ProgramBinding {
        geo,
        resident: Resident {
            kv_sliding,
            kv_full,
            pos_buf: &scratch.pos_buf,
            v_norm_ones: &scratch.v_norm_ones_full,
            flash_partials: &scratch.flash_partials,
        },
        rows: 1,
        position: pos,
        positions: None,
        scratch: LayerScratch::lowered(scratch, config.enable_moe_block),
    };
    let mut steps = Vec::with_capacity(3 * config.n_layers + 3);
    layer_steps(&binding, weights, config, &mut steps).map_err(hip)?;
    let lm_head = weights.lm_head.dispatch_ref();
    head(
        &geo,
        &weights.final_norm,
        &lm_head,
        &scratch.x,
        &scratch.tmp,
        &scratch.logits,
        &mut steps,
    );
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &steps).map_err(|e| hip(format!("gemma4: {e}")))
}

/// `dst = embed(token)`, unscaled.
fn embed(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    dst: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    let table = &weights.embed_tokens;
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => gpu.embedding_lookup_hfq4g256(table, dst, token, dim),
        EmbeddingFormat::HFQ4G128 => gpu.embedding_lookup_hfq4g128(table, dst, token, dim),
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(table, dst, token, dim),
        EmbeddingFormat::F32 => gpu.embedding_lookup(table, dst, token, dim),
        _ => Err(hip_bridge::HipError::new(
            0,
            "unsupported Gemma 4 embed format",
        )),
    }
}

/// Every layer's steps, one cache slot per layer of each type in layer order.
fn layer_steps<'a>(
    binding: &crate::program::ProgramBinding<'a>,
    weights: &'a Gemma4Weights,
    config: &Gemma4Config,
    steps: &mut Vec<hipfire_dispatch::pipeline::Step<'a>>,
) -> Result<(), String> {
    use crate::program::{LayerKv, LayerRefs};
    let mut slots = [0usize; 2];
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let class = matches!(layer, LayerWeights::Full(_)) as usize;
        let kv = LayerKv {
            slot: slots[class],
            writes: true,
        };
        slots[class] += 1;
        binding.layer(
            layer_idx,
            LayerRefs::lowered(layer, config.hidden_dim),
            kv,
            steps,
        )?;
    }
    Ok(())
}

/// Batched prefill: runs `tokens` from `start_pos` through the declarative
/// layer program, `rows > 1` at a time, writing both KV caches (which must be
/// Q8). Computes no logits; follow with [`forward_scratch`] on the last token
/// when they are needed. The calibration collector, when armed, sees every
/// projection input.
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill_batch(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &mut hipfire_runtime::llama::KvCache,
    kv_full: &mut hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> HipResult<()> {
    let hip = |e: String| hip_bridge::HipError::new(0, &e);
    let rows_max = crate::gemma4::GEMMA4_FORWARD_BATCH_MAX;
    for (i, chunk) in tokens.chunks(rows_max).enumerate() {
        let pos = start_pos + i * rows_max;
        if let [token] = chunk {
            forward_scratch(
                gpu, weights, config, *token, pos, kv_sliding, kv_full, scratch,
            )?;
        } else {
            prefill_rows(
                gpu, weights, config, chunk, pos, kv_sliding, kv_full, scratch,
            )
            .map_err(hip)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prefill_rows(
    gpu: &mut Gpu,
    weights: &Gemma4Weights,
    config: &Gemma4Config,
    tokens: &[u32],
    start_pos: usize,
    kv_sliding: &hipfire_runtime::llama::KvCache,
    kv_full: &hipfire_runtime::llama::KvCache,
    scratch: &Gemma4Scratch,
) -> Result<(), String> {
    use crate::program::{
        routed_scratch, Geometry, ProgramBinding, Resident, RowActivations, RowBuffers, RowWidths,
    };
    let rows = tokens.len();
    let dim = config.dim;
    let end = start_pos + rows;
    if end > kv_sliding.physical_cap || end > kv_full.physical_cap {
        return Err(format!(
            "gemma4 prefill: positions [{start_pos}, {end}) exceed the KV capacity"
        ));
    }
    let geo = Geometry::lowered(config);
    let head_dim = config.sliding_head_dim.max(config.full_head_dim);
    let widths = RowWidths {
        dim,
        max_q: config.n_heads * head_dim,
        max_kv: (config.sliding_n_kv_heads * config.sliding_head_dim)
            .max(config.full_n_kv_heads * config.full_head_dim),
        ffn: config.hidden_dim,
    };
    let mut bufs = RowBuffers::new(gpu);
    let acts = RowActivations::new(&mut bufs, gpu, rows, start_pos, widths)?;
    let cap = kv_sliding.physical_cap.max(kv_full.physical_cap);
    let flash_partials = bufs.alloc(
        crate::program::q8_flash_partials_len(gpu, &geo, cap, rows),
        "flash_partials",
    )?;
    for (row, &token) in tokens.iter().enumerate() {
        embed(gpu, weights, &acts.x.sub_offset(row * dim, dim), token, dim)
            .map_err(|e| format!("gemma4 prefill embed: {e:?}"))?;
    }
    gpu.scale_f32(&acts.x, config.embed_scale)
        .map_err(|e| format!("gemma4 prefill embed scale: {e:?}"))?;
    let binding = ProgramBinding {
        geo,
        resident: Resident {
            kv_sliding,
            kv_full,
            pos_buf: &scratch.pos_buf,
            v_norm_ones: &scratch.v_norm_ones_full,
            flash_partials: &flash_partials,
        },
        rows,
        position: start_pos,
        positions: Some(&acts.positions),
        scratch: acts.scratch(
            None,
            config.enable_moe_block.then(|| routed_scratch(scratch)),
        ),
    };
    let mut steps = Vec::with_capacity(3 * config.n_layers);
    layer_steps(&binding, weights, config, &mut steps)?;
    let ctx = DispatchCtx::new(gpu);
    execute_steps(gpu, &ctx, &steps).map_err(|e| format!("gemma4 prefill: {e}"))
}
