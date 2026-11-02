//! Qwen3.5 native MTP (Multi-Token Prediction / NextN) head.
//!
//! This module loads the single transformer-decoder block + NextN overlay
//! that ships in Qwen3.5/3.6 dense checkpoints (`mtp.*` tensor namespace),
//! quantized into a `.mtp` file by `crates/hipfire-quantize/src/bin/mtp_extract.rs`
//! (arch_id = 21, `QWEN35_MTP_HEAD`). Loading + forward path are independent
//! of the trunk's full forward — the head consumes a single `prev_hidden`
//! activation produced by the trunk + the next committed token, and emits
//! logits over the full vocab via the trunk's shared `lm_head`.
//!
//! ## Forward (per llama.cpp `qwen35_mtp.cpp` and HF reference):
//!
//! ```text
//! tok_embd = embed[next_token]                 # via trunk's tok_embd
//! e_norm   = RMSNorm(tok_embd, enorm, eps)     # F32 norm weight
//! h_norm   = RMSNorm(prev_hidden, hnorm, eps)
//! cur      = eh_proj @ concat(e_norm, h_norm)  # 2d → d
//! inpSA    = cur                               # save for residual
//! cur      = RMSNorm(cur, attn_norm)
//! Q_full   = wq @ cur                          # 2 * head_dim * n_head
//! Q, gate  = deinterleave(Q_full)              # split per-head
//! Q        = RMSNorm(Q, attn_q_norm)           # per-head
//! K, V     = wk @ cur, wv @ cur
//! K        = RMSNorm(K, attn_k_norm)
//! Q, K     = rope_partial_interleaved(Q, K, pos)  # default RoPE for Qwen3.5
//! kv[pos]  = K, V                              # MTP-private KV cache
//! attn     = attention(Q, kv[..=pos], V_cache, scale = 1/sqrt(head_dim))
//! attn     = sigmoid_mul(attn, gate)           # gated-Q output
//! cur      = wo @ attn + inpSA                 # residual
//! ffn_in   = cur
//! cur      = RMSNorm(cur, attn_post_norm)      # POST-attn norm (NOT pre-FFN)
//! ffn      = ffn_down(silu(ffn_gate @ cur) * (ffn_up @ cur))
//! cur      = ffn + ffn_in
//! cur      = RMSNorm(cur, shared_head_norm)    # pre-LM-head norm
//! logits   = lm_head_weights @ cur             # caller supplies trunk's lm_head
//! ```
//!
//! ## Per-call alloc: zero
//!
//! All intermediates live in [`Qwen35MtpHeadScratch`], which is allocated
//! once per slot. The forward writes logits into `scratch.logits` (caller
//! reads via `gpu.download_f32`). KV cache is in [`Qwen35MtpHeadKvCache`]
//! (single-layer F32, separate from the trunk).
//!
//! ## RoPE choice
//!
//! Qwen3.5 spec says M-RoPE multi-section, but the trunk's full-attention
//! layer also uses `rope_partial_interleaved_f32` (qwen35.rs:2431,2611) —
//! the M-RoPE sections + partial-rotary-factor=0.25 reduce to the same
//! single-section partial RoPE for text-only tokens. We mirror trunk
//! behavior so the MTP-head numerics stay in sync with trunk-trained
//! distillation targets.
//!
//! ## What this module does NOT do
//!
//! - tree-decode multi-step recursion: caller composes step N+1 by feeding
//!   `scratch.t_mtp_out` back as `prev_hidden`.
//! - sampling: caller takes argmax / does temperature / top-k.
//! - KV rollback / batched verify: this is a single-token forward, the
//!   verify-loop equivalent for MTP would be Task 11.

use crate::qwen35::{MoeFfnWeights, Qwen35Config, Qwen35Weights};
use hip_bridge::{DeviceBuffer, HipError, HipResult};
use hipfire_dispatch::pipeline::gated_mlp::{
    Activation, GatedMlpOp, GatedMlpScratch, MlpLadder, MlpOutput, MlpStorage, RmsNorm,
};
use hipfire_dispatch::pipeline::hybrid::{AttentionKv, AttentionRow, GatedAttentionOp, HybridDims};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_dispatch::types::DispatchError;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{self, f16_to_f32, weight_gemv, EmbeddingFormat, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::path::{Path, PathBuf};

/// Kill switch for the kv_only MTP fill concat acceleration
/// (`HIPFIRE_MTP_FILL_FAST=0` restores the per-row D2D concat loop).
pub(crate) fn mtp_fill_fast_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_MTP_FILL_FAST", true));
    *ON
}

/// Restore the incumbent v1 GEMM with `HIPFIRE_MTP_FILL_PM_GEMM=0`.
fn mtp_fill_pm_gemm_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_MTP_FILL_PM_GEMM", true)
    });
    *ON
}

// ─── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35MtpFfnKind {
    Dense,
    Moe,
}

/// All dimensions and hyperparams the MTP head needs. Loaded from the
/// `.mtp` file's metadata JSON; nothing is hardcoded per model size.
#[derive(Debug, Clone)]
pub struct Qwen35MtpHeadConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub ffn_kind: Qwen35MtpFfnKind,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub vocab_size: usize,
    pub rope_theta: f32,
    /// Partial-rotary factor; defaults to 0.25 to mirror Qwen3.5 trunk.
    /// Stored as the absolute n_rot (head_dim * factor).
    pub n_rot: usize,
    pub rms_norm_eps: f32,
    /// Maximum positions the head's KV cache can store. Caller picks at
    /// allocation time.
    pub max_seq: usize,
    /// True iff the source model's `tie_word_embeddings` is true. Lets the
    /// caller know whether trunk's embed_tokens + lm_head are aliases.
    pub tie_word_embeddings: bool,
}

/// Layer prefix the MTP MoE layer loads under through the trunk MoE loader
/// (see `qwen35::load::mtp_sidecar_tensor_name`).
pub(crate) const MTP_MOE_PREFIX: &str = "mtp_head";

impl Qwen35MtpHeadConfig {
    /// Trunk MoE-layer config for the MTP layer's routed FFN: one
    /// full-attention layer with this head's MoE geometry and no REAP map.
    fn moe_config(&self) -> Qwen35Config {
        Qwen35Config {
            dim: self.n_embd,
            n_layers: 1,
            vocab_size: self.vocab_size,
            norm_eps: self.rms_norm_eps,
            eos_token: 0,
            n_heads: self.n_head,
            n_kv_heads: self.n_head_kv,
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
            partial_rotary_factor: self.n_rot as f32 / self.head_dim as f32,
            is_vl_text: false,
            mrope_interleaved: false,
            mrope_section: [0; 3],
            linear_num_key_heads: 0,
            linear_num_value_heads: 0,
            linear_key_head_dim: 0,
            linear_value_head_dim: 0,
            conv_kernel_dim: 0,
            hidden_dim: 0,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            has_shared_expert: true,
            norm_topk_prob: self.norm_topk_prob,
            layer_types: vec![crate::qwen35::LayerType::FullAttention],
            paged_experts: false,
            vram_budget_bytes: u64::MAX,
            reap_keep: None,
        }
    }

    /// Decode geometry of the MTP layer for the shared hybrid steps.
    fn hybrid_dims(&self) -> HybridDims {
        HybridDims {
            dim: self.n_embd,
            n_layers: 1,
            n_heads: self.n_head,
            n_kv_heads: self.n_head_kv,
            head_dim: self.head_dim,
            n_rot: self.n_rot,
            rope_theta: self.rope_theta,
            norm_eps: self.rms_norm_eps,
            linear_key_heads: 0,
            linear_value_heads: 0,
            linear_key_dim: 0,
            linear_value_dim: 0,
            conv_kernel: 0,
            num_experts: self.num_experts,
        }
    }

    /// Parse from a `.mtp` file's metadata JSON. Defaults match the Task 8
    /// extractor's canonical layout.
    fn from_metadata(meta: &serde_json::Value, max_seq: usize) -> Self {
        let g = |k: &str, default: f64| -> f64 {
            meta.get(k).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let gu = |k: &str| -> usize {
            meta.get(k)
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!(".mtp metadata missing required key '{k}'"))
                as usize
        };
        let n_embd = gu("n_embd");
        let n_head = gu("n_head");
        let n_head_kv = gu("n_head_kv");
        let head_dim = gu("n_embd_head");
        let n_ff = gu("n_ff");
        let ffn_kind = match meta
            .get("ffn_kind")
            .and_then(|v| v.as_str())
            .unwrap_or("dense")
        {
            "dense" => Qwen35MtpFfnKind::Dense,
            "moe" => Qwen35MtpFfnKind::Moe,
            other => panic!(".mtp metadata has unsupported ffn_kind='{other}'"),
        };
        let num_experts = meta
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_experts_per_tok = meta
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let moe_intermediate_size = meta
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let shared_expert_intermediate_size = meta
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let norm_topk_prob = meta
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let vocab_size = gu("vocab_size");
        // partial_rotary_factor lives nested under config_text_config; fall
        // back to 0.25 (Qwen3.5 default) when absent.
        let prf = meta
            .get("config_text_config")
            .and_then(|c| c.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);
        let n_rot = (head_dim as f64 * prf) as usize;
        let rope_theta = g("rope_theta", 10_000_000.0) as f32;
        let rms_norm_eps = g("rms_norm_eps", 1e-6) as f32;
        let tie_word_embeddings = meta
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        Self {
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            ffn_kind,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_topk_prob,
            vocab_size,
            rope_theta,
            n_rot,
            rms_norm_eps,
            max_seq,
            tie_word_embeddings,
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────

pub struct Qwen35MtpDenseFfnWeights {
    pub gate: WeightTensor, // [n_ff, n_embd]
    pub up: WeightTensor,   // [n_ff, n_embd]
    pub down: WeightTensor, // [n_embd, n_ff]
}

pub enum Qwen35MtpFfnWeights {
    Dense(Qwen35MtpDenseFfnWeights),
    /// Routed experts bound like a trunk MoE layer; `config` carries the
    /// MTP layer's MoE geometry for the shared decode recipe.
    Moe {
        ffn: MoeFfnWeights,
        config: Qwen35Config,
    },
}

/// All 15 GPU-resident MTP head tensors (+2 optional FastMTP-style
/// vocab-compression sidecar tensors). Ownership is tied to the head
/// instance; [`Qwen35MtpHeadWeights::free_gpu`] releases them at unload.
pub struct Qwen35MtpHeadWeights {
    // Norms (F32, 1D)
    pub shared_head_norm: GpuTensor,
    pub enorm: GpuTensor,
    pub hnorm: GpuTensor,
    pub attn_norm: GpuTensor,
    pub attn_post_norm: GpuTensor,
    pub attn_q_norm: GpuTensor,
    pub attn_k_norm: GpuTensor,
    // 2D weights (MQ4 / Q8)
    pub eh_proj: WeightTensor, // [n_embd, 2 * n_embd]
    pub wq: WeightTensor,      // [2 * head_dim * n_head, n_embd]
    pub wk: WeightTensor,      // [head_dim * n_head_kv, n_embd]
    pub wv: WeightTensor,      // [head_dim * n_head_kv, n_embd]
    pub wo: WeightTensor,      // [n_embd, head_dim * n_head]
    pub ffn: Qwen35MtpFfnWeights,

    // Optional FastMTP-style vocab-compression sidecar (present iff the
    // .mtp metadata says `has_compressed_lm_head_draft: true`).
    //
    // - `lm_head_draft`: top-K rows of trunk lm_head (MQ4 / Q8), used by
    //   `mtp_head_forward_compressed` in place of the trunk's full lm_head.
    //   Shape [compressed_vocab_size, n_embd]. ~7.7x BW reduction at K=32K
    //   on a 248K-vocab Qwen3.5/3.6.
    // - `lm_head_draft_vocab_map`: host-side u32 array mapping draft idx
    //   -> full vocab idx. Kept for host fallback paths and diagnostics.
    // - `lm_head_draft_vocab_map_gpu`: same map on-device for the greedy
    //   MTP token-chain path, where draft idx -> full token must happen
    //   without a per-step D2H sync.
    // - `compressed_vocab_size`: K. Cached for fast access in forward.
    pub lm_head_draft: Option<WeightTensor>,
    pub lm_head_draft_vocab_map: Option<Vec<u32>>,
    pub lm_head_draft_vocab_map_gpu: Option<GpuTensor>,
    pub compressed_vocab_size: Option<usize>,
}

impl Qwen35MtpHeadWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.shared_head_norm);
        let _ = gpu.free_tensor(self.enorm);
        let _ = gpu.free_tensor(self.hnorm);
        let _ = gpu.free_tensor(self.attn_norm);
        let _ = gpu.free_tensor(self.attn_post_norm);
        let _ = gpu.free_tensor(self.attn_q_norm);
        let _ = gpu.free_tensor(self.attn_k_norm);
        let _ = gpu.free_tensor(self.eh_proj.buf);
        let _ = gpu.free_tensor(self.wq.buf);
        let _ = gpu.free_tensor(self.wk.buf);
        let _ = gpu.free_tensor(self.wv.buf);
        let _ = gpu.free_tensor(self.wo.buf);
        match self.ffn {
            Qwen35MtpFfnWeights::Dense(ffn) => {
                let _ = gpu.free_tensor(ffn.gate.buf);
                let _ = gpu.free_tensor(ffn.up.buf);
                let _ = gpu.free_tensor(ffn.down.buf);
            }
            Qwen35MtpFfnWeights::Moe { ffn, .. } => crate::qwen35::weights::free_moe_ffn(gpu, ffn),
        }
        if let Some(lm_d) = self.lm_head_draft {
            let _ = gpu.free_tensor(lm_d.buf);
        }
        if let Some(vmap) = self.lm_head_draft_vocab_map_gpu {
            let _ = gpu.free_tensor(vmap);
        }
    }
}

// ─── Scratch ─────────────────────────────────────────────────────────────

/// Per-call GPU scratch for the MTP head forward — allocated once via
/// [`Qwen35MtpHeadScratch::new`], reused on every call. Mirrors
/// `Qwen35Scratch` but sized for the MTP head's single block + LM head.
pub struct Qwen35MtpHeadScratch {
    // Activation stages
    pub tok_embd: GpuTensor, // [n_embd]
    pub e_norm: GpuTensor,   // [n_embd]
    pub h_norm: GpuTensor,   // [n_embd]
    pub concat: GpuTensor,   // [2 * n_embd]
    pub cur: GpuTensor,      // [n_embd] — primary residual stream
    pub residual: GpuTensor, // [n_embd] — normalized (rotated) projection input
    pub tmp: GpuTensor,      // [n_embd] — RMSNorm output scratch

    // Attention sub-block
    pub q_full: GpuTensor,   // [2 * head_dim * n_head]
    pub q: GpuTensor,        // [head_dim * n_head]
    pub gate: GpuTensor,     // [head_dim * n_head]
    pub k: GpuTensor,        // [head_dim * n_head_kv]
    pub v: GpuTensor,        // [head_dim * n_head_kv]
    pub attn_out: GpuTensor, // [head_dim * n_head]

    // FFN sub-block
    pub gate_ffn: GpuTensor,   // [n_ff]
    pub up: GpuTensor,         // [n_ff]
    pub ffn_hidden: GpuTensor, // [n_ff]

    // MoE FFN scratch, allocated only for ffn_kind=moe.
    pub moe_router_logits: Option<GpuTensor>, // [num_experts]
    pub moe_scalar_buf: Option<GpuTensor>,    // [1]
    pub moe_x_rot: Option<GpuTensor>,         // [n_embd]
    pub moe_gate_up_buf: Option<GpuTensor>,   // [2 * max(moe_intermediate, shared_intermediate)]
    pub moe_gate_buf: Option<GpuTensor>,      // [max(moe_intermediate, shared_intermediate)]
    pub moe_up_buf: Option<GpuTensor>,        // [max(moe_intermediate, shared_intermediate)]
    pub moe_ffn_hidden: Option<GpuTensor>,    // [max(moe_intermediate, shared_intermediate)]
    pub moe_ffn_out: Option<GpuTensor>,       // [n_embd]
    pub moe_gate_batch: Option<GpuTensor>,    // [top_k * moe_intermediate]
    pub moe_up_batch: Option<GpuTensor>,      // [top_k * moe_intermediate]
    pub moe_rot_batch: Option<GpuTensor>,     // [top_k * moe_intermediate]
    pub moe_topk_indices: Option<GpuTensor>,  // [top_k]
    pub moe_topk_weights: Option<GpuTensor>,  // [top_k]
    pub moe_down_expanded: Option<GpuTensor>, // [top_k * n_embd]

    // Snapshot of the post-FFN, pre-LM-head-norm hidden — caller can
    // capture this and feed back as `prev_hidden` for an n+2 prediction.
    pub t_mtp_out: GpuTensor, // [n_embd]

    // LM head output
    pub logits: GpuTensor, // [vocab_size]

    // Optional FastMTP-style compressed-logits scratch — sized
    // [compressed_vocab_size] when the head was loaded with a sidecar.
    // Populated by `mtp_head_forward_compressed` instead of `logits`.
    pub logits_compressed: Option<GpuTensor>,

    // FlashAttention partials buffer for the asym3 attention kernel.
    // Sized [n_heads * max_tiles * (2 + head_dim)] F32 where max_tiles =
    // ceil(max_seq / 128). Required by `attention_flash_asym3`. Trunk
    // sizes this with batch_mult=16 for batched prefill; MTP single-step
    // forward uses batch_mult=1.
    pub flash_partials: GpuTensor,

    // Position scalar — uploaded each forward into a 4-byte device buffer.
    pub pos_buf: DeviceBuffer,
}

impl Qwen35MtpHeadScratch {
    /// View of the MoE scratch in the trunk decode recipe's layout. Panics
    /// if the head was built without MoE scratch.
    fn moe_refs(&self) -> crate::qwen35::forward::MoeScratchRef<'_> {
        fn moe(t: &Option<GpuTensor>) -> &GpuTensor {
            t.as_ref().expect("MoE MTP scratch not allocated")
        }
        crate::qwen35::forward::MoeScratchRef {
            router_logits: moe(&self.moe_router_logits),
            scalar_buf: moe(&self.moe_scalar_buf),
            x_rot_local: moe(&self.moe_x_rot),
            gate_up_buf: moe(&self.moe_gate_up_buf),
            gate_buf: moe(&self.moe_gate_buf),
            up_buf: moe(&self.moe_up_buf),
            ffn_hidden: moe(&self.moe_ffn_hidden),
            ffn_out: moe(&self.moe_ffn_out),
            gate_batch: moe(&self.moe_gate_batch),
            up_batch: moe(&self.moe_up_batch),
            rot_batch: moe(&self.moe_rot_batch),
            topk_indices: moe(&self.moe_topk_indices),
            topk_weights: moe(&self.moe_topk_weights),
            down_expanded: moe(&self.moe_down_expanded),
        }
    }

    /// Device bytes [`Self::new`] allocates for `config` on `arch` (the flash
    /// partials depend on the arch's flash tile size), from shape arithmetic
    /// alone. `compressed_vocab` adds the `ensure_compressed_logits` tensor.
    pub fn requested_bytes(
        arch: &str,
        config: &Qwen35MtpHeadConfig,
        compressed_vocab: Option<usize>,
    ) -> usize {
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        let moe_max_inter = config
            .moe_intermediate_size
            .max(config.shared_expert_intermediate_size);
        // tok_embd, e_norm, h_norm, cur, residual, tmp, t_mtp_out
        let mut elems = 7 * dim
            + 2 * dim // concat
            + 2 * q_dim // q_full
            + 3 * q_dim // q, gate, attn_out
            + 2 * kv_dim // k, v
            + 3 * config.n_ff // gate_ffn, up, ffn_hidden
            + config.vocab_size; // logits
        if config.ffn_kind == Qwen35MtpFfnKind::Moe {
            let batch = config.num_experts_per_tok * config.moe_intermediate_size;
            elems += config.num_experts // router logits
                + 1 // scalar
                + dim // x_rot
                + 2 * moe_max_inter // gate_up
                + 3 * moe_max_inter // gate, up, ffn_hidden
                + dim // ffn_out
                + 3 * batch // gate/up/rot batch
                + 2 * config.num_experts_per_tok // topk idx/weights
                + config.num_experts_per_tok * dim
                + dim.div_ceil(4); // down_expanded
        }
        let tile_size = rdna_compute::attention::q8_flash_tile_size(
            arch,
            config.n_head,
            config.n_head_kv,
            config.head_dim,
            config.max_seq,
        );
        let max_tiles = (config.max_seq + tile_size - 1) / tile_size;
        elems += config.n_head * max_tiles * (2 + config.head_dim); // flash_partials
        elems += compressed_vocab.unwrap_or(0);
        elems * 4 + 4 // + pos_buf
    }

    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        let moe_max_inter = config
            .moe_intermediate_size
            .max(config.shared_expert_intermediate_size);
        let alloc_moe = config.ffn_kind == Qwen35MtpFfnKind::Moe;
        Ok(Self {
            tok_embd: gpu.alloc_tensor(&[dim], DType::F32)?,
            e_norm: gpu.alloc_tensor(&[dim], DType::F32)?,
            h_norm: gpu.alloc_tensor(&[dim], DType::F32)?,
            concat: gpu.alloc_tensor(&[2 * dim], DType::F32)?,
            cur: gpu.alloc_tensor(&[dim], DType::F32)?,
            residual: gpu.alloc_tensor(&[dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[dim], DType::F32)?,
            q_full: gpu.alloc_tensor(&[2 * q_dim], DType::F32)?,
            q: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[q_dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            up: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[config.n_ff], DType::F32)?,
            moe_router_logits: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts], DType::F32)?)
            } else {
                None
            },
            moe_scalar_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[1], DType::F32)?)
            } else {
                None
            },
            moe_x_rot: if alloc_moe {
                Some(gpu.alloc_tensor(&[dim], DType::F32)?)
            } else {
                None
            },
            moe_gate_up_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[2 * moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_gate_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_up_buf: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_ffn_hidden: if alloc_moe {
                Some(gpu.alloc_tensor(&[moe_max_inter], DType::F32)?)
            } else {
                None
            },
            moe_ffn_out: if alloc_moe {
                Some(gpu.alloc_tensor(&[dim], DType::F32)?)
            } else {
                None
            },
            moe_gate_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_up_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_rot_batch: if alloc_moe {
                Some(gpu.alloc_tensor(
                    &[config.num_experts_per_tok * config.moe_intermediate_size],
                    DType::F32,
                )?)
            } else {
                None
            },
            moe_topk_indices: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_topk_weights: if alloc_moe {
                Some(gpu.alloc_tensor(&[config.num_experts_per_tok], DType::F32)?)
            } else {
                None
            },
            moe_down_expanded: if alloc_moe {
                // The trunk decode layout: payload plus the zeroed counter tail.
                Some(gpu.zeros(
                    &[config.num_experts_per_tok * dim + dim.div_ceil(4)],
                    DType::F32,
                )?)
            } else {
                None
            },
            t_mtp_out: gpu.alloc_tensor(&[dim], DType::F32)?,
            logits: gpu.alloc_tensor(&[config.vocab_size], DType::F32)?,
            logits_compressed: None,
            flash_partials: {
                // Sized with the tile the flash launch picks for this shape: a
                // fixed 128 under-allocates 4x where the arch picks 32 (gfx1100
                // at max_seq <= 8192) and the tile kernel page-faults.
                let tile_size = rdna_compute::attention::q8_flash_tile_size(
                    &gpu.arch,
                    config.n_head,
                    config.n_head_kv,
                    config.head_dim,
                    config.max_seq,
                );
                let max_tiles = (config.max_seq + tile_size - 1) / tile_size;
                gpu.alloc_tensor(
                    &[config.n_head * max_tiles * (2 + config.head_dim)],
                    DType::F32,
                )?
            },
            pos_buf: gpu.hip.malloc(4)?,
        })
    }

    /// Allocate the compressed-logits scratch for FastMTP-style forwards.
    /// Idempotent — a no-op if already allocated to the right size. Call
    /// once after head load when the head reports a compressed_vocab_size.
    pub fn ensure_compressed_logits(
        &mut self,
        gpu: &mut Gpu,
        compressed_vocab_size: usize,
    ) -> HipResult<()> {
        if let Some(existing) = self.logits_compressed.as_ref() {
            if existing.numel() == compressed_vocab_size {
                return Ok(());
            }
            // Size changed — drop and reallocate. This shouldn't happen in
            // normal use (sidecar K is fixed at extract time) but we guard
            // anyway.
            if let Some(old) = self.logits_compressed.take() {
                let _ = gpu.free_tensor(old);
            }
        }
        self.logits_compressed = Some(gpu.alloc_tensor(&[compressed_vocab_size], DType::F32)?);
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.tok_embd);
        let _ = gpu.free_tensor(self.e_norm);
        let _ = gpu.free_tensor(self.h_norm);
        let _ = gpu.free_tensor(self.concat);
        let _ = gpu.free_tensor(self.cur);
        let _ = gpu.free_tensor(self.residual);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.q_full);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.gate);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.gate_ffn);
        let _ = gpu.free_tensor(self.up);
        let _ = gpu.free_tensor(self.ffn_hidden);
        if let Some(t) = self.moe_router_logits {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_scalar_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_x_rot {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_up_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_up_buf {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_ffn_hidden {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_ffn_out {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_gate_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_up_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_rot_batch {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_topk_indices {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_topk_weights {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.moe_down_expanded {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.t_mtp_out);
        let _ = gpu.free_tensor(self.logits);
        if let Some(lc) = self.logits_compressed {
            let _ = gpu.free_tensor(lc);
        }
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── KV cache (single-layer, MTP-private) ────────────────────────────────

/// The MTP head has a single attention block, so its KV cache is one
/// per-layer K + V buffer. Separate from the trunk's `KvCache` since the
/// MTP head writes the SAME absolute position the trunk just emitted —
/// reusing the trunk's cache would mean either double-write or
/// snapshot/restore on every cycle.
///
/// **Format:** Q8_0 (8-bit per-block quantized K and V, 34 B/block of 32
/// elements). Picked over F32 (4× larger, no flash-attn kernel) and
/// asym3 (3-bit, ~9% perf regression on canonical due to lower attention
/// fidelity → lower τ at K=5; benchmarked 2026-05-15). Q8 runs the
/// `attention_flash_q8_0` flash-attn tile kernel, preserves attention
/// quality close to F32 (much higher precision than 3-bit), and is 4×
/// smaller than F32 (2 KB / token vs 8 KB at hd=256, n_kv_heads=4).
///
/// Internally wraps a single-layer [`llama::KvCache`] built via
/// `new_gpu_q8` so we share buffer-sizing semantics with the trunk's
/// `--kv-mode q8` path. Single-layer-ness means `inner.k_gpu[0]` /
/// `inner.v_gpu[0]` are the only slots used.
///
/// History (this branch):
/// - v1: hardcoded F32 + naive `attention_f32` (canonical 49 tok/s K=5)
/// - v2: tried asym3 — coherent but -9% perf at K=5 (44 tok/s) due to
///       3-bit K quant lowering attention quality → lower τ ceiling
/// - v3 (current): Q8 — flash-attn enabled, attention quality preserved
/// KV format for the MTP head's per-token decode path. Default is Q8.
///
/// - **Q8**: 8-bit per-block K and V, `attention_q8_0_kv` (non-flash).
///   Reference: -3% perf vs F32 at K=5 per [[mtp-session-state-2026-05-15-compaction]],
///   externally validated by Unsloth's 76.45% Q8 hit-rate claim.
/// - **Asym3**: 3-bit Givens-rotated K + Q8 V, `attention_flash_asym3`.
///   Prior session v2 attempt: -9% perf vs F32 (lower attention fidelity → lower τ).
///   Reverted in favor of Q8. Re-running with the new ±1-3% methodology
///   to confirm/refute that delta.
/// - **Fwht4**: 4-bit signed-FWHT-rotated K + Q8 V, `attention_flash_fwht4`.
///   Master's feat/fwht Phase 2 bench (3.5-27B prose, dflash): Fwht3 fixes
///   asym3 collapse (τ 1.27→3.60) and Fwht2 EXCEEDS Q8 (τ 4.08 vs 3.91).
///   Wired here as the MTP-head's matching test.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MtpKvMode {
    Q8,
    Asym3,
    Fwht4,
}

impl MtpKvMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "q8" => Ok(Self::Q8),
            "asym3" | "turbo3" => Ok(Self::Asym3),
            "fwht4" => Ok(Self::Fwht4),
            other => Err(format!(
                "unknown MTP kv-mode '{other}' (expected: q8, asym3, fwht4)"
            )),
        }
    }
}

pub struct Qwen35MtpHeadKvCache {
    pub inner: hipfire_runtime::llama::KvCache,
    pub max_seq: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub kv_mode: MtpKvMode,
}

impl Qwen35MtpHeadKvCache {
    /// Device bytes [`Self::new_with_kv_mode`] allocates for `MtpKvMode::Q8`:
    /// one layer, K and V, each `max_seq * n_head_kv * (head_dim / 32) * 34`
    /// bytes rounded up to whole F32 words. Pure shape arithmetic.
    pub fn q8_requested_bytes(config: &Qwen35MtpHeadConfig) -> usize {
        let total_blocks = config.n_head_kv * (config.head_dim / 32);
        let cache_bytes = config.max_seq * total_blocks * 34;
        2 * (cache_bytes.div_ceil(4) * 4)
    }

    /// Default constructor: Q8 KV (current ship).
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig) -> HipResult<Self> {
        Self::new_with_kv_mode(gpu, config, MtpKvMode::Q8)
    }

    /// Allocate the MTP head's single-layer KV cache in the requested format.
    pub fn new_with_kv_mode(
        gpu: &mut Gpu,
        config: &Qwen35MtpHeadConfig,
        kv_mode: MtpKvMode,
    ) -> HipResult<Self> {
        let inner = match kv_mode {
            MtpKvMode::Q8 => hipfire_runtime::llama::KvCache::new_gpu_q8(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
            MtpKvMode::Asym3 => hipfire_runtime::llama::KvCache::new_gpu_asym3(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
            MtpKvMode::Fwht4 => hipfire_runtime::llama::KvCache::new_gpu_fwht4(
                gpu,
                /* n_layers */ 1,
                config.n_head_kv,
                config.head_dim,
                config.max_seq,
            )?,
        };
        Ok(Self {
            inner,
            max_seq: config.max_seq,
            n_head_kv: config.n_head_kv,
            head_dim: config.head_dim,
            kv_mode,
        })
    }

    /// Reset positions to all-zeros. Q8 K/V slots are 34 B per 32-element
    /// block stored inside an F32-typed buffer; zeroing the underlying
    /// buffer is equivalent to "no positions written" at the kernel level.
    /// With `HIPFIRE_MTP_FILL_FAST` enabled, allocation-free device zeroing
    /// is ordered on the active stream and writes the exact same positive-zero
    /// bytes across the full K/V buffers. Setting the flag to `0` preserves
    /// the original host-zero allocation and H2D-copy path.
    pub fn reset(&mut self, gpu: &mut Gpu) -> HipResult<()> {
        if mtp_fill_fast_enabled() {
            gpu.bind_thread()?;
            for layer in 0..self.inner.k_gpu.len() {
                let k_n = self.inner.k_gpu[layer].numel();
                let v_n = self.inner.v_gpu[layer].numel();
                gpu.hip.memset_d32_async(
                    &self.inner.k_gpu[layer].buf,
                    0,
                    k_n,
                    gpu.active_stream.as_ref(),
                )?;
                gpu.hip.memset_d32_async(
                    &self.inner.v_gpu[layer].buf,
                    0,
                    v_n,
                    gpu.active_stream.as_ref(),
                )?;
            }
            return Ok(());
        }
        for layer in 0..self.inner.k_gpu.len() {
            let k_n = self.inner.k_gpu[layer].numel();
            let v_n = self.inner.v_gpu[layer].numel();
            let zeros_k = vec![0.0f32; k_n];
            let zeros_v = vec![0.0f32; v_n];
            let kb: &[u8] =
                unsafe { std::slice::from_raw_parts(zeros_k.as_ptr() as *const u8, k_n * 4) };
            let vb: &[u8] =
                unsafe { std::slice::from_raw_parts(zeros_v.as_ptr() as *const u8, v_n * 4) };
            gpu.hip.memcpy_htod(&self.inner.k_gpu[layer].buf, kb)?;
            gpu.hip.memcpy_htod(&self.inner.v_gpu[layer].buf, vb)?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        // llama::KvCache holds GpuTensors with NO Drop impl — dropping it
        // leaks the k_gpu/v_gpu buffers. Free them explicitly. (The old
        // "they free on Drop" comment was false; see mtp_spec/mtp_compose
        // which already bypass this wrapper for the same reason.)
        let _ = self.inner.free_gpu(gpu);
    }
}

// ─── Top-level handle ────────────────────────────────────────────────────

/// Loaded MTP head: config + weights, ready for `mtp_head_forward`.
/// Caller separately allocates [`Qwen35MtpHeadScratch`] (per inference slot)
/// and [`Qwen35MtpHeadKvCache`] (per generation).
pub struct Qwen35MtpHead {
    pub config: Qwen35MtpHeadConfig,
    pub weights: Qwen35MtpHeadWeights,
}

impl Qwen35MtpHead {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.weights.free_gpu(gpu);
    }
}

// ─── Bundled .mq4-mtp loader (trunk + MTP head in one file) ───────────────

/// Trailer magic written by `mq4_merge_mtp`. Indicates the file is a bundle
/// of a trunk `.mq4` followed by an MTP `.mtp` section.
pub const BUNDLE_TRAILER_MAGIC: &[u8; 8] = b"HFBNDMTP";
/// Trailer is 8 bytes magic + 8 bytes u64 mtp-section offset = 16 bytes.
pub const BUNDLE_TRAILER_LEN: u64 = 16;

/// Inspect a file's trailing 16 bytes for the `mq4_merge_mtp` trailer. If
/// present, returns the byte offset where the embedded MTP `.mtp` section
/// starts. Returns `None` for plain `.mq4` trunk files (no MTP bundled).
///
/// Cheap operation — single 16-byte read from end of file. Safe to call
/// on any path: returns `Ok(None)` rather than erroring when the file is
/// too small or the magic doesn't match.
pub fn detect_bundled_mtp_offset(path: &Path) -> std::io::Result<Option<u64>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let file_size = f.metadata()?.len();
    if file_size < BUNDLE_TRAILER_LEN {
        return Ok(None);
    }
    f.seek(SeekFrom::End(-(BUNDLE_TRAILER_LEN as i64)))?;
    let mut trailer = [0u8; BUNDLE_TRAILER_LEN as usize];
    f.read_exact(&mut trailer)?;
    if &trailer[..8] != BUNDLE_TRAILER_MAGIC {
        return Ok(None);
    }
    let mtp_offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
    if mtp_offset >= file_size - BUNDLE_TRAILER_LEN {
        // Corrupt trailer (offset past EOF). Treat as not-bundled rather
        // than crashing — the trunk loader's HFQM parse will succeed
        // independent of trailer state.
        return Ok(None);
    }
    Ok(Some(mtp_offset))
}

/// Load the MTP head section embedded inside a bundled `.mq4-mtp` file.
/// `path` is the bundle file path; the trailer is read to find the embedded
/// MTP section offset, then [`load_mtp_head_at_offset`] parses it.
///
/// Returns `Ok(None)` if `path` is a plain trunk `.mq4` (no bundle trailer).
pub fn load_mtp_head_bundled(
    path: &Path,
    gpu: &mut Gpu,
    max_seq: usize,
) -> HipResult<Option<Qwen35MtpHead>> {
    let mtp_offset = match detect_bundled_mtp_offset(path) {
        Ok(Some(off)) => off,
        Ok(None) => return Ok(None),
        Err(e) => panic!("read bundle trailer from {}: {e}", path.display()),
    };
    let head = load_mtp_head_at_offset(path, gpu, max_seq, mtp_offset)?;
    Ok(Some(head))
}

// ─── Trunk compatibility ─────────────────────────────────────────────────

/// Refuse a `.mtp` head that cannot attach to the trunk it sits beside,
/// before any GPU upload.
///
/// The head reads the trunk's final hidden state and reuses the trunk's
/// `embed_tokens` and `lm_head`, so its `n_embd` must equal the trunk's
/// hidden size and its `vocab_size` the trunk's vocab. A mismatched head
/// would otherwise load and draft from out-of-range rows. `base_offset`
/// is `0` for a standalone `.mtp` and the trailer offset for a bundle.
///
/// This checks shape only: a head from another checkpoint with the same
/// dims (e.g. a Qwen3.5-27B head on a Qwen3.8-27B trunk) passes, and the
/// verify then rejects its drafts rather than committing them.
pub fn check_mtp_head_for_trunk(
    path: &Path,
    base_offset: u64,
    trunk_dim: usize,
    trunk_vocab: usize,
) -> Result<(), String> {
    let hfq = HfqFile::open_at_offset(path, base_offset)
        .map_err(|e| format!("cannot open {} as an HFQM .mtp head: {e}", path.display()))?;
    if hfq.arch_id != 21 {
        return Err(format!(
            "{} has arch_id={} (expected 21 = qwen35_mtp_head)",
            path.display(),
            hfq.arch_id
        ));
    }
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|e| format!("{}: metadata JSON is invalid: {e}", path.display()))?;
    let dim = |key: &str| {
        meta.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| format!("{}: metadata has no '{key}'", path.display()))
    };
    let (n_embd, vocab_size) = (dim("n_embd")?, dim("vocab_size")?);
    if n_embd != trunk_dim || vocab_size != trunk_vocab {
        let source = meta
            .get("source_model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        return Err(format!(
            "{} does not match this trunk: head n_embd={n_embd} vocab={vocab_size} \
             (source_model={source}), trunk hidden={trunk_dim} vocab={trunk_vocab}; \
             use the head extracted from this model's own checkpoint",
            path.display()
        ));
    }
    Ok(())
}

/// Candidate `.mtp` sidecar paths for a trunk weight file, most specific
/// first.
///
/// `Path::with_extension("mtp")` only replaces the last extension, so
/// `qwen3.5-4b.mq4v2.hfq` becomes `qwen3.5-4b.mq4v2.mtp`. Product extracts
/// sit next to the trunk as `qwen3.5-4b.mtp`. Both are probed, plus the
/// no-`.hfq`/no-quant sibling (`qwen3.5-4b.mq4` → `qwen3.5-4b.mtp`).
///
/// The list itself lives in [`hipfire_runtime::sidecar`], shared with the
/// `.vl` probe and the CLI's capability advertisement so the three cannot
/// drift apart.
pub fn mtp_sidecar_candidates(trunk: &Path) -> Vec<PathBuf> {
    hipfire_runtime::sidecar::sidecar_candidates(trunk, "mtp")
}

/// First candidate that exists on disk, if any.
pub fn find_mtp_sidecar(trunk: &Path) -> Option<PathBuf> {
    mtp_sidecar_candidates(trunk)
        .into_iter()
        .find(|p| p.exists())
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load a `.mtp` file (arch_id = 21) created by `mtp_extract` (Task 8).
/// Returns the head ready for `mtp_head_forward`.
///
/// `max_seq` bounds the per-position KV cache later allocated by
/// [`Qwen35MtpHeadKvCache::new`]; pick to match your decode budget.
pub fn load_mtp_head(path: &Path, gpu: &mut Gpu, max_seq: usize) -> HipResult<Qwen35MtpHead> {
    load_mtp_head_at_offset(path, gpu, max_seq, 0)
}

/// Like [`load_mtp_head`] but opens the HFQM container at `base_offset`
/// inside `path`. Pass `0` for a standalone `.mtp` file; for a bundled
/// `.mq4-mtp` file pass the offset returned by [`detect_bundled_mtp_offset`].
pub fn load_mtp_head_at_offset(
    path: &Path,
    gpu: &mut Gpu,
    max_seq: usize,
    base_offset: u64,
) -> HipResult<Qwen35MtpHead> {
    let hfq = HfqFile::open_at_offset(path, base_offset).unwrap_or_else(|e| {
        panic!(
            "open .mtp file {} @ offset {base_offset}: {e}",
            path.display()
        )
    });
    assert_eq!(
        hfq.arch_id,
        21,
        ".mtp file at {} has arch_id={} (expected 21 = QWEN35_MTP_HEAD); \
         is this actually an MTP head extracted by mtp_extract?",
        path.display(),
        hfq.arch_id
    );
    let meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).expect(".mtp metadata JSON parse failed");
    let config = Qwen35MtpHeadConfig::from_metadata(&meta, max_seq);
    // Pure metadata prevalidation of every qt53 tensor (roles, uniformity,
    // shapes, extents, AWQ) before the first upload below. No qt53 => no-op.
    validate_mtp_qt53_layout(
        hfq.tensor_infos(),
        config.ffn_kind == Qwen35MtpFfnKind::Moe,
        mtp_meta_declares_awq(&meta),
        config.num_experts,
        config.n_embd,
        config.moe_intermediate_size,
    )
    .map_err(|error| HipError::new(0, &error))?;

    // ── Norms (F32, 1D) ─────────────────────────────────────────────────
    //
    // The .mtp file uses bare tensor names ("enorm", "wq", ...) — no
    // "model.language_model." prefix. We read raw bytes and upload
    // directly; `load_weight_tensor` from qwen35.rs unconditionally
    // prepends that prefix and so cannot be reused here.
    let n_embd = config.n_embd;
    let head_dim = config.head_dim;

    // shared_head_norm gets +1.0 like the per-layer norms — empirically
    // verified 2026-05-15 A/B: removing +1.0 regressed K=3 from τ=3.08 to
    // τ=2.00 on 27B-3.5 LRU bench. The MTP head trains its `mtp.norm` with
    // the trunk per-layer convention, NOT the trunk final-norm convention.
    let shared_head_norm = load_norm_raw(&hfq, gpu, "shared_head_norm", n_embd)?;
    let enorm = load_norm_raw(&hfq, gpu, "enorm", n_embd)?;
    let hnorm = load_norm_raw(&hfq, gpu, "hnorm", n_embd)?;
    let attn_norm = load_norm_raw(&hfq, gpu, "attn_norm", n_embd)?;
    let attn_post_norm = load_norm_raw(&hfq, gpu, "attn_post_norm", n_embd)?;
    let attn_q_norm = load_norm_raw(&hfq, gpu, "attn_q_norm", head_dim)?;
    let attn_k_norm = load_norm_raw(&hfq, gpu, "attn_k_norm", head_dim)?;

    // ── 2D weights ──────────────────────────────────────────────────────
    let q_full_dim = 2 * head_dim * config.n_head;
    let kv_dim = head_dim * config.n_head_kv;
    let q_dim = head_dim * config.n_head;

    let eh_proj = load_weight_raw(&hfq, gpu, "eh_proj", n_embd, 2 * n_embd)?;
    let wq = load_weight_raw(&hfq, gpu, "wq", q_full_dim, n_embd)?;
    let wk = load_weight_raw(&hfq, gpu, "wk", kv_dim, n_embd)?;
    let wv = load_weight_raw(&hfq, gpu, "wv", kv_dim, n_embd)?;
    let wo = load_weight_raw(&hfq, gpu, "wo", n_embd, q_dim)?;
    let ffn = match config.ffn_kind {
        Qwen35MtpFfnKind::Dense => Qwen35MtpFfnWeights::Dense(Qwen35MtpDenseFfnWeights {
            gate: load_weight_raw(&hfq, gpu, "ffn_gate", config.n_ff, n_embd)?,
            up: load_weight_raw(&hfq, gpu, "ffn_up", config.n_ff, n_embd)?,
            down: load_weight_raw(&hfq, gpu, "ffn_down", n_embd, config.n_ff)?,
        }),
        Qwen35MtpFfnKind::Moe => {
            assert_eq!(
                config.num_experts_per_tok, 8,
                "MoE MTP runtime currently supports top_k=8, got {}",
                config.num_experts_per_tok
            );
            let moe_config = config.moe_config();
            Qwen35MtpFfnWeights::Moe {
                ffn: crate::qwen35::load::load_moe_ffn(
                    &hfq,
                    gpu,
                    MTP_MOE_PREFIX,
                    &moe_config,
                    0,
                    hipfire_runtime::offload::LayerResidency::DEVICE,
                )?,
                config: moe_config,
            }
        }
    };

    // ── Optional FastMTP-style compressed lm_head_draft + vocab map ────
    let has_compressed = meta
        .get("has_compressed_lm_head_draft")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let (
        lm_head_draft,
        lm_head_draft_vocab_map,
        lm_head_draft_vocab_map_gpu,
        compressed_vocab_size,
    ) = if has_compressed {
        let cvs = meta
            .get("compressed_vocab_size")
            .and_then(|v| v.as_u64())
            .expect("metadata claims has_compressed_lm_head_draft but lacks compressed_vocab_size")
            as usize;
        assert!(cvs > 0, "compressed_vocab_size must be positive");
        let lm_d = load_weight_raw(&hfq, gpu, "lm_head_draft.weight", cvs, n_embd)?;
        let (vmap_info, vmap_bytes) = hfq
            .tensor_data_vec("lm_head_draft.vocab_map")
            .expect("compressed sidecar missing vocab_map tensor");
        assert_eq!(
            vmap_info.shape,
            vec![cvs as u32],
            "lm_head_draft.vocab_map shape != [compressed_vocab_size]"
        );
        assert_eq!(
            vmap_bytes.len(),
            cvs * 4,
            "lm_head_draft.vocab_map byte count != {} (got {})",
            cvs * 4,
            vmap_bytes.len()
        );
        let vmap: Vec<u32> = vmap_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vmap_gpu = gpu.upload_raw(&vmap_bytes, &[vmap_bytes.len()])?;
        (Some(lm_d), Some(vmap), Some(vmap_gpu), Some(cvs))
    } else {
        (None, None, None, None)
    };

    let weights = Qwen35MtpHeadWeights {
        shared_head_norm,
        enorm,
        hnorm,
        attn_norm,
        attn_post_norm,
        attn_q_norm,
        attn_k_norm,
        eh_proj,
        wq,
        wk,
        wv,
        wo,
        ffn,
        lm_head_draft,
        lm_head_draft_vocab_map,
        lm_head_draft_vocab_map_gpu,
        compressed_vocab_size,
    };

    Ok(Qwen35MtpHead { config, weights })
}

/// Load a 1D F32 norm tensor from a .mtp file. Mirrors the +1.0 offset
/// convention used by the trunk (Qwen3.5 RMSNorm: `out = x · rsqrt(var+eps)
/// · (1 + weight)`). All `.mtp` norms ship as quant_type=2 (F32) per the
/// `mtp_extract` packing rule.
fn load_norm_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected_n: usize,
) -> HipResult<GpuTensor> {
    load_norm_raw_with_offset(hfq, gpu, name, expected_n, true)
}

/// Like load_norm_raw but allows callers to control whether the +1.0 trunk
/// offset is applied. Trunk's per-layer norms get +1; trunk's FINAL norm
/// (`model.norm.weight` → `output_norm`) is RAW. The MTP head's
/// `mtp.norm.weight` (shared_head_norm) is the equivalent of the trunk's
/// final norm — should also be RAW. Pass apply_plus_one=false for that one.
fn load_norm_raw_with_offset(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    expected_n: usize,
    apply_plus_one: bool,
) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    let mut f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("norm '{name}' has unexpected qt={other} (expected F16 or F32)"),
    };
    assert_eq!(
        f32_data.len(),
        expected_n,
        "norm '{name}': loaded {} elems but expected {expected_n}",
        f32_data.len(),
    );
    // Qwen3.5 / 3.6 RMSNorm convention: weight is stored as the offset from
    // 1.0. Trunk's `load_norm_weight` does the same `+= 1.0` pre-upload
    // step for every per-layer norm. The MTP head's `mtp.norm.weight`
    // (mapped to `shared_head_norm`) is the equivalent of the trunk's
    // `model.norm.weight` — but the trunk treats the FINAL norm as raw
    // (no +1 offset, see `load_norm_weight_raw`). We follow the trunk's
    // per-layer convention here for ALL norms because the safetensors
    // shipped values for `mtp.norm.weight` ≈ 0 (consistent with offset
    // representation). Off-by-one risk is small: the trunk does the same
    // +1 for `shared_expert_intermediate.norm` etc.
    if apply_plus_one {
        for v in &mut f32_data {
            *v += 1.0;
        }
    }
    gpu.upload_f32(&f32_data, &[expected_n])
}

/// Load a 2D quantized weight tensor from a .mtp file. Resolves any of
/// the supported quant types into a [`WeightTensor`]; m and k are passed
/// in (the mtp container stores shape but we trust caller-supplied dims).
fn load_weight_raw(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!(".mtp tensor '{name}' missing"));
    sanity_check_2d_shape(name, info, m, k);
    weight_tensor_from_raw(gpu, info.quant_type, &data, m, k, name)
}

/// Cross-check the on-disk shape against the caller's expected (m, k).
/// Catches silent dim mismatches (e.g. tied vocab, head split done wrong).
fn sanity_check_2d_shape(name: &str, info: &HfqTensorInfo, m: usize, k: usize) {
    if info.shape.len() != 2 {
        panic!(
            ".mtp tensor '{name}': expected 2D shape, got {}D = {:?}",
            info.shape.len(),
            info.shape,
        );
    }
    let on_disk_m = info.shape[0] as usize;
    let on_disk_k = info.shape[1] as usize;
    assert_eq!(
        on_disk_m, m,
        ".mtp tensor '{name}': shape[0]={on_disk_m} but expected m={m}"
    );
    assert_eq!(
        on_disk_k, k,
        ".mtp tensor '{name}': shape[1]={on_disk_k} but expected k={k}"
    );
}

/// MQ4G128V2 (qt53): admitted in `.mtp` heads ONLY as the routed-expert
/// `moe_experts.{id}.down` tensor (see [`mtp_qt53_down_expert_id`]).
const MTP_QT_MQ4G128V2: u8 = 53;
/// MQ4G256V2 (qt44): the routed gate_up partner of a qt53 down.
const MTP_QT_MQ4G256V2: u8 = 44;

fn mtp_packed_dtype(quant_type: u8) -> Option<DType> {
    match quant_type {
        13 => Some(DType::MQ4G256),
        15 => Some(DType::MQ6G256),
        44 => Some(DType::MQ4G256V2),
        47 => Some(DType::MQ6G256V2),
        MTP_QT_MQ4G128V2 => Some(DType::MQ4G128V2),
        _ => None,
    }
}

/// Routed expert id of an exact canonical `moe_experts.{id}.down` name
/// (decimal digits, no sign / leading zeros / extra segments); `None` for
/// every other tensor name.
fn mtp_qt53_down_expert_id(name: &str) -> Option<usize> {
    let id = name.strip_prefix("moe_experts.")?.strip_suffix(".down")?;
    if id.is_empty()
        || !id.bytes().all(|b| b.is_ascii_digit())
        || (id.len() > 1 && id.starts_with('0'))
    {
        return None;
    }
    id.parse().ok()
}

fn mtp_qt53_role_error(name: &str) -> String {
    format!(
        ".mtp tensor '{name}' is MQ4G128V2 (qt=53), which an MTP head admits only for the \
         routed-expert down projection 'moe_experts.<id>.down'; every other MTP tensor role \
         (attention, eh_proj, router, shared expert, routed gate_up, dense FFN, lm_head_draft) \
         must not be qt=53"
    )
}

/// Pure per-tensor qt53 admission run BEFORE upload: exact down name, nonzero
/// geometry, and the shared qt53 blob validator.
fn mtp_qt53_admit_tensor(name: &str, m: usize, k: usize, data: &[u8]) -> HipResult<()> {
    if mtp_qt53_down_expert_id(name).is_none() {
        return Err(HipError::new(0, &mtp_qt53_role_error(name)));
    }
    crate::qwen35::load::validate_mq4g128v2_blob(data, m, k)
        .map_err(|error| HipError::new(0, &format!(".mtp tensor '{name}' qt=53: {error}")))
}

/// Pure: refusal for the batched MTP head forward when ANY head weight is
/// MQ4G128V2 (qt53). qt53 routed down has only the serial decode route.
fn mtp_batched_qt53_refusal(dtypes: impl IntoIterator<Item = DType>) -> Result<(), DispatchError> {
    if dtypes.into_iter().any(|dtype| dtype == DType::MQ4G128V2) {
        return Err(DispatchError::UnsupportedVariant {
            family: "mtp-head",
            variant: "qt53-batched-mtp-unsupported-use-serial",
            arch: "",
            quant: "MQ4G128V2",
        });
    }
    Ok(())
}

/// Every weight dtype of the head (attention, eh_proj, FFN incl. every routed
/// expert, lm_head_draft), for [`mtp_batched_qt53_refusal`].
fn mtp_head_weight_dtypes(w: &Qwen35MtpHeadWeights) -> impl Iterator<Item = DType> + '_ {
    let fixed = [&w.eh_proj, &w.wq, &w.wk, &w.wv, &w.wo]
        .into_iter()
        .chain(w.lm_head_draft.as_ref())
        .map(|t| t.gpu_dtype);
    let ffn: Box<dyn Iterator<Item = DType> + '_> = match &w.ffn {
        Qwen35MtpFfnWeights::Dense(f) => {
            Box::new([f.gate.gpu_dtype, f.up.gpu_dtype, f.down.gpu_dtype].into_iter())
        }
        Qwen35MtpFfnWeights::Moe { ffn: f, .. } => Box::new(
            [
                f.router.gpu_dtype,
                f.shared_expert_gate.gpu_dtype,
                f.shared_expert.gate.gpu_dtype,
                f.shared_expert.up.gpu_dtype,
                f.shared_expert.down.gpu_dtype,
            ]
            .into_iter()
            .chain(
                f.experts
                    .iter()
                    .flat_map(|e| [e.gate_up.gpu_dtype, e.down.gpu_dtype]),
            ),
        ),
    };
    fixed.chain(ffn)
}

/// Serial-entry route-or-refusal, run before any launch for weights that did
/// not come through the loader: qt53 anywhere except the uniform routed down
/// (qt44 gate_up + qt53 down) is refused. Non-qt53 heads pass untouched.
fn mtp_head_serial_route_check(
    w: &Qwen35MtpHeadWeights,
    cfg: &Qwen35MtpHeadConfig,
) -> HipResult<()> {
    let outside_routed_down = {
        let fixed = [&w.eh_proj, &w.wq, &w.wk, &w.wv, &w.wo]
            .into_iter()
            .chain(w.lm_head_draft.as_ref())
            .any(|t| t.gpu_dtype == DType::MQ4G128V2);
        fixed
            || match &w.ffn {
                Qwen35MtpFfnWeights::Dense(f) => [&f.gate, &f.up, &f.down]
                    .into_iter()
                    .any(|t| t.gpu_dtype == DType::MQ4G128V2),
                Qwen35MtpFfnWeights::Moe { ffn: f, .. } => [
                    &f.router,
                    &f.shared_expert_gate,
                    &f.shared_expert.gate,
                    &f.shared_expert.up,
                    &f.shared_expert.down,
                ]
                .into_iter()
                .any(|t| t.gpu_dtype == DType::MQ4G128V2),
            }
    };
    if outside_routed_down {
        return Err(HipError::new(
            0,
            "MTP head carries MQ4G128V2 (qt=53) outside the routed-expert down projection; \
             qt=53 is admitted only as moe_experts.<id>.down",
        ));
    }
    if let Qwen35MtpFfnWeights::Moe { ffn: f, .. } = &w.ffn {
        mtp_moe_route_check(
            f.experts
                .iter()
                .map(|e| (e.gate_up.gpu_dtype, e.down.gpu_dtype)),
            cfg.moe_intermediate_size,
            cfg.num_experts_per_tok,
        )
        .map_err(|error| HipError::new(0, &error))?;
    }
    Ok(())
}

/// True when the head metadata declares an AWQ sidecar (any top-level key
/// containing `awq` whose value is not null/false/0/empty).
fn mtp_meta_declares_awq(meta: &serde_json::Value) -> bool {
    meta.as_object().is_some_and(|object| {
        object.iter().any(|(key, value)| {
            key.to_ascii_lowercase().contains("awq")
                && match value {
                    serde_json::Value::Null => false,
                    serde_json::Value::Bool(flag) => *flag,
                    serde_json::Value::Number(number) => number.as_f64() != Some(0.0),
                    serde_json::Value::String(text) => !text.is_empty(),
                    serde_json::Value::Array(items) => !items.is_empty(),
                    serde_json::Value::Object(map) => !map.is_empty(),
                }
        })
    })
}

/// Whole-index qt53 admission, metadata only (no payload, no upload).
///
/// No qt53 tensor => `Ok` (every existing non-qt53 head is untouched).
/// Otherwise the head must be MoE and, for EVERY routed expert `0..n_exp`:
/// `moe_experts.{x}.down` is qt53 (uniform; mixed rejected), shape
/// `[dim, mi]` (K = mi), group_size 128, exact extent `dim*(mi/128)*68`;
/// `moe_experts.{x}.gate_up` is qt44, shape `[2*mi, dim]`, exact extent
/// `2*mi*(dim/256)*136`. No qt53 tensor may carry any other name, and no AWQ
/// metadata/sidecar may exist.
fn validate_mtp_qt53_layout(
    infos: &[HfqTensorInfo],
    ffn_is_moe: bool,
    meta_awq: bool,
    n_exp: usize,
    dim: usize,
    mi: usize,
) -> Result<(), String> {
    let qt53: Vec<&HfqTensorInfo> = infos
        .iter()
        .filter(|t| t.quant_type == MTP_QT_MQ4G128V2)
        .collect();
    if qt53.is_empty() {
        return Ok(());
    }
    for t in &qt53 {
        match mtp_qt53_down_expert_id(&t.name) {
            None => return Err(mtp_qt53_role_error(&t.name)),
            Some(id) if !ffn_is_moe || id >= n_exp => {
                return Err(format!(
                    ".mtp tensor '{}' is qt=53 but expert id {id} is outside this head's \
                     {n_exp} routed experts (MoE head required)",
                    t.name
                ));
            }
            Some(_) => {}
        }
    }
    if meta_awq || infos.iter().any(|t| t.name.contains(".awq_scale")) {
        return Err(
            ".mtp qt=53 routed down has no AWQ route: the head declares AWQ metadata or an \
             .awq_scale sidecar"
                .to_string(),
        );
    }
    if n_exp == 0 || dim == 0 || dim % 256 != 0 || mi == 0 || mi % 128 != 0 {
        return Err(format!(
            ".mtp qt=53 routed down requires experts>0, hidden % 256 == 0 (qt44 gate_up K) and \
             moe_intermediate % 128 == 0 (qt53 down K); got experts={n_exp} hidden={dim} \
             moe_intermediate={mi}"
        ));
    }
    let by_name: std::collections::HashMap<&str, &HfqTensorInfo> =
        infos.iter().map(|t| (t.name.as_str(), t)).collect();
    let down_bytes = dim * (mi / 128) * 68;
    let gate_up_bytes = 2 * mi * (dim / 256) * 136;
    for x in 0..n_exp {
        let down_name = format!("moe_experts.{x}.down");
        let down = by_name
            .get(down_name.as_str())
            .ok_or_else(|| format!(".mtp qt=53 head is missing '{down_name}'"))?;
        if down.quant_type != MTP_QT_MQ4G128V2 {
            return Err(format!(
                ".mtp routed down must be uniformly qt=53 when any is: '{down_name}' has qt={}",
                down.quant_type
            ));
        }
        if down.shape != [dim as u32, mi as u32] || down.group_size != 128 {
            return Err(format!(
                ".mtp tensor '{down_name}' qt=53: shape {:?} group_size {} != [{dim}, {mi}] \
                 (K = moe_intermediate) group_size 128",
                down.shape, down.group_size
            ));
        }
        if down.data_size != down_bytes {
            return Err(format!(
                ".mtp tensor '{down_name}' qt=53: {} bytes != exact {down_bytes}",
                down.data_size
            ));
        }
        let gate_up_name = format!("moe_experts.{x}.gate_up");
        let gate_up = by_name
            .get(gate_up_name.as_str())
            .ok_or_else(|| format!(".mtp qt=53 head is missing '{gate_up_name}'"))?;
        if gate_up.quant_type != MTP_QT_MQ4G256V2 {
            return Err(format!(
                ".mtp qt=53 routed down requires qt=44 (MQ4G256V2) gate_up on every expert: \
                 '{gate_up_name}' has qt={}",
                gate_up.quant_type
            ));
        }
        if gate_up.shape != [(2 * mi) as u32, dim as u32] {
            return Err(format!(
                ".mtp tensor '{gate_up_name}': shape {:?} != [{}, {dim}]",
                gate_up.shape,
                2 * mi
            ));
        }
        if gate_up.data_size != gate_up_bytes {
            return Err(format!(
                ".mtp tensor '{gate_up_name}': {} bytes != exact {gate_up_bytes}",
                gate_up.data_size
            ));
        }
    }
    Ok(())
}

/// Typed route-or-refusal for the MTP MoE routed experts, checked before any
/// launch. Non-qt53 pairs pass untouched; a qt53 down requires every expert to
/// be exactly (qt44 gate_up, qt53 down), K = mi a multiple of 128, top_k 8.
fn mtp_moe_route_check(
    pairs: impl IntoIterator<Item = (DType, DType)>,
    mi: usize,
    k_top: usize,
) -> Result<(), String> {
    let mut any_down53 = false;
    let mut any_gate53 = false;
    let mut all_qt44_qt53 = true;
    for (gate, down) in pairs {
        any_down53 |= down == DType::MQ4G128V2;
        any_gate53 |= gate == DType::MQ4G128V2;
        all_qt44_qt53 &= gate == DType::MQ4G256V2 && down == DType::MQ4G128V2;
    }
    if any_gate53 {
        return Err("MTP MoE routed gate_up cannot be MQ4G128V2 (qt=53 is down-only)".to_string());
    }
    if !any_down53 {
        return Ok(());
    }
    if !all_qt44_qt53 {
        return Err(
            "MTP MoE qt=53 routed down requires uniform (MQ4G256V2 gate_up, MQ4G128V2 down) on \
             every expert"
                .to_string(),
        );
    }
    if mi == 0 || mi % 128 != 0 || k_top != 8 {
        return Err(format!(
            "MTP MoE qt=53 routed down requires top_k=8 and moe_intermediate % 128 == 0; got \
             top_k={k_top} moe_intermediate={mi}"
        ));
    }
    Ok(())
}

/// Wrap raw quantized bytes into a [`WeightTensor`]. Local copy of the
/// dispatch table from `qwen35::load_weight_tensor_raw`, restricted to the
/// quant types emitted by the MTP extractors and published sidecars.
fn weight_tensor_from_raw(
    gpu: &Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
) -> HipResult<WeightTensor> {
    match quant_type {
        qt @ (13 | 15 | 44 | 47) => {
            let dtype = mtp_packed_dtype(qt).expect("packed quant arm must map");
            assert!(
                k % 256 == 0,
                ".mtp tensor '{name}' is {dtype:?} with K={k} not divisible by 256"
            );
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: dtype,
                m,
                k,
                row_stride: 0,
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        MTP_QT_MQ4G128V2 => {
            // Own arm: row-local K=128 groups (no K % 256 rule). Role + blob
            // validation run before the upload; only `moe_experts.{id}.down`.
            mtp_qt53_admit_tensor(name, m, k, data)?;
            let dtype = mtp_packed_dtype(MTP_QT_MQ4G128V2).expect("qt53 must map");
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: dtype,
                m,
                k,
                row_stride: dtype.row_stride(k),
                paro: None,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            })
        }
        3 => {
            // Q8_F16 (group_size=32, 34 bytes/group) — same byte layout as
            // GGML Q8_0; existing gemv_q8_0 dispatch works directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor {
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
            })
        }
        1 => {
            // F16 → dequantize on host, upload as F32 (no GPU-native F16
            // GEMV; the trunk does the same conversion in load_weight_tensor_raw).
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = gpu.upload_raw(bytes, &[m, k])?;
            Ok(WeightTensor {
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
            })
        }
        2 => {
            // F32 raw.
            let buf = gpu.upload_raw(data, &[m, k])?;
            Ok(WeightTensor {
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
            })
        }
        other => panic!(
            ".mtp tensor '{name}': unsupported quant_type={other} \
             (supported: MQ4/MQ6 G256 V1=13/15, V2=44/47, routed-down-only MQ4G128V2=53, \
              Q8_F16=3, F16=1, F32=2)"
        ),
    }
}

#[cfg(test)]
mod packed_dtype_tests {
    use super::*;

    #[test]
    fn published_mtp_packed_formats_map_to_exact_runtime_dtypes() {
        assert_eq!(mtp_packed_dtype(13), Some(DType::MQ4G256));
        assert_eq!(mtp_packed_dtype(15), Some(DType::MQ6G256));
        assert_eq!(mtp_packed_dtype(44), Some(DType::MQ4G256V2));
        assert_eq!(mtp_packed_dtype(47), Some(DType::MQ6G256V2));
        assert_eq!(mtp_packed_dtype(53), Some(DType::MQ4G128V2));
        assert_eq!(mtp_packed_dtype(48), None);
    }

    fn info(
        name: &str,
        qt: u8,
        shape: [u32; 2],
        group_size: u32,
        data_size: usize,
    ) -> HfqTensorInfo {
        HfqTensorInfo {
            name: name.to_string(),
            quant_type: qt,
            shape: shape.to_vec(),
            group_size,
            data_offset: 0,
            data_size,
        }
    }

    // A3B-shaped geometry scaled down: hidden 256 (qt44 K), mi 128 (qt53 K).
    const DIM: usize = 256;
    const MI: usize = 128;
    const N_EXP: usize = 3;

    fn good_head() -> Vec<HfqTensorInfo> {
        let mut infos = vec![
            info("wq", 44, [64, DIM as u32], 256, 64 * (DIM / 256) * 136),
            info("moe_router", 1, [N_EXP as u32, DIM as u32], 0, 1),
        ];
        for x in 0..N_EXP {
            infos.push(info(
                &format!("moe_experts.{x}.gate_up"),
                44,
                [2 * MI as u32, DIM as u32],
                256,
                2 * MI * (DIM / 256) * 136,
            ));
            infos.push(info(
                &format!("moe_experts.{x}.down"),
                53,
                [DIM as u32, MI as u32],
                128,
                DIM * (MI / 128) * 68,
            ));
        }
        infos
    }

    fn layout(infos: &[HfqTensorInfo]) -> Result<(), String> {
        validate_mtp_qt53_layout(infos, true, false, N_EXP, DIM, MI)
    }

    #[test]
    fn qt53_admits_only_exact_routed_down_names() {
        assert_eq!(mtp_qt53_down_expert_id("moe_experts.0.down"), Some(0));
        assert_eq!(mtp_qt53_down_expert_id("moe_experts.255.down"), Some(255));
        for bad in [
            "moe_experts.00.down",
            "moe_experts.01.down",
            "moe_experts..down",
            "moe_experts.+1.down",
            "moe_experts.-1.down",
            "moe_experts.1.gate_up",
            "moe_experts.1.down.awq_scale.weight",
            "moe_experts.1.2.down",
            "moe_experts.x.down",
            "moe_experts.1.down ",
            "moe_shared_down",
            "ffn_down",
            "moe_router",
            "moe_shared_gate",
            "moe_shared_up",
            "moe_shared_expert_gate",
            "eh_proj",
            "wq",
            "wk",
            "wv",
            "wo",
            "ffn_gate",
            "ffn_up",
            "lm_head_draft.weight",
            "prefix.moe_experts.1.down",
        ] {
            assert_eq!(mtp_qt53_down_expert_id(bad), None, "{bad}");
            let error = mtp_qt53_admit_tensor(bad, 4, 128, &[]).unwrap_err();
            assert!(error.to_string().contains("qt=53"), "{bad}: {error}");
            assert!(error.to_string().contains(bad), "{bad}: {error}");
        }
    }

    #[test]
    fn qt53_admission_accepts_exact_symmetric_blob_and_rejects_bad_extent() {
        let (m, k) = (4usize, 128usize);
        let mut blob = Vec::new();
        for _ in 0..m {
            blob.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale 1.0
            blob.extend_from_slice(&0xC800u16.to_le_bytes()); // zero -8.0
            blob.extend_from_slice(&[0x88u8; 64]);
        }
        mtp_qt53_admit_tensor("moe_experts.2.down", m, k, &blob).unwrap();
        let short = &blob[..blob.len() - 1];
        let error = mtp_qt53_admit_tensor("moe_experts.2.down", m, k, short).unwrap_err();
        assert!(error.to_string().contains("moe_experts.2.down"), "{error}");
        assert!(mtp_qt53_admit_tensor("moe_experts.2.gate_up", m, k, &blob).is_err());
    }

    #[test]
    fn qt53_layout_accepts_uniform_qt44_gate_up_qt53_down_and_ignores_non_qt53_heads() {
        layout(&good_head()).unwrap();
        // No qt53 anywhere: the validator is a no-op regardless of arguments.
        let mut plain = good_head();
        for t in &mut plain {
            if t.quant_type == 53 {
                t.quant_type = 44;
                t.shape = vec![DIM as u32, MI as u32];
            }
        }
        assert_eq!(
            validate_mtp_qt53_layout(&plain, false, true, 0, 0, 0),
            Ok(())
        );
    }

    #[test]
    fn qt53_layout_rejects_mixed_down_wrong_gate_up_shapes_extents_awq_and_roles() {
        let reject = |mutate: &dyn Fn(&mut Vec<HfqTensorInfo>), needle: &str| {
            let mut infos = good_head();
            mutate(&mut infos);
            let error = layout(&infos).unwrap_err();
            assert!(error.contains(needle), "{needle}: {error}");
        };
        let at = |infos: &mut Vec<HfqTensorInfo>, name: &str| {
            infos.iter().position(|t| t.name == name).unwrap()
        };
        // Mixed down: one expert stays qt44.
        reject(
            &|i| {
                let p = at(i, "moe_experts.1.down");
                i[p].quant_type = 44;
            },
            "uniformly qt=53",
        );
        // gate_up must be qt44 on every expert.
        reject(
            &|i| {
                let p = at(i, "moe_experts.0.gate_up");
                i[p].quant_type = 47;
            },
            "requires qt=44",
        );
        // qt53 gate_up is a role violation (down-only).
        reject(
            &|i| {
                let p = at(i, "moe_experts.0.gate_up");
                i[p].quant_type = 53;
            },
            "routed-expert down projection",
        );
        // Down K != mi, wrong rows, wrong group size, wrong extent.
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.down");
                i[p].shape = vec![DIM as u32, 2 * MI as u32];
            },
            "shape",
        );
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.down");
                i[p].shape = vec![MI as u32, DIM as u32];
            },
            "shape",
        );
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.down");
                i[p].group_size = 256;
            },
            "group_size",
        );
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.down");
                i[p].data_size += 68;
            },
            "bytes",
        );
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.gate_up");
                i[p].shape = vec![MI as u32, DIM as u32];
            },
            "shape",
        );
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.gate_up");
                i[p].data_size -= 136;
            },
            "bytes",
        );
        // Missing expert / out-of-range qt53 id.
        reject(
            &|i| {
                let p = at(i, "moe_experts.2.gate_up");
                i.remove(p);
            },
            "missing",
        );
        reject(
            &|i| {
                i.push(info(
                    "moe_experts.3.down",
                    53,
                    [DIM as u32, MI as u32],
                    128,
                    1,
                ))
            },
            "outside",
        );
        // AWQ sidecar.
        reject(
            &|i| {
                i.push(info(
                    "moe_experts.0.down.awq_scale.weight",
                    1,
                    [MI as u32, 0],
                    0,
                    2,
                ))
            },
            "AWQ",
        );
        // Every other role refuses qt53 by name.
        for role in [
            "wq",
            "moe_router",
            "moe_shared_down",
            "moe_shared_gate",
            "moe_shared_up",
            "moe_shared_expert_gate",
            "eh_proj",
            "ffn_down",
            "lm_head_draft.weight",
        ] {
            let mut infos = good_head();
            infos.push(info(role, 53, [DIM as u32, MI as u32], 128, 68));
            let error = layout(&infos).unwrap_err();
            assert!(error.contains(role), "{role}: {error}");
        }
        // AWQ metadata and a non-MoE head.
        let infos = good_head();
        assert!(validate_mtp_qt53_layout(&infos, true, true, N_EXP, DIM, MI)
            .unwrap_err()
            .contains("AWQ"));
        assert!(validate_mtp_qt53_layout(&infos, false, false, N_EXP, DIM, MI).is_err());
        // K = mi must be a whole number of 128-groups; hidden of 256-groups.
        assert!(validate_mtp_qt53_layout(&infos, true, false, N_EXP, DIM, MI + 64).is_err());
        assert!(validate_mtp_qt53_layout(&infos, true, false, N_EXP, DIM + 128, MI).is_err());
    }

    #[test]
    fn mtp_metadata_awq_detection() {
        use serde_json::json;
        assert!(!mtp_meta_declares_awq(&json!({"n_embd": 2048})));
        assert!(!mtp_meta_declares_awq(
            &json!({"awq": false, "awq_scale": 0, "x_awq": null})
        ));
        assert!(mtp_meta_declares_awq(&json!({"awq": true})));
        assert!(mtp_meta_declares_awq(&json!({"quant_AWQ_group": 128})));
        assert!(mtp_meta_declares_awq(&json!({"awq_recipe": "s2"})));
    }

    #[test]
    fn mtp_moe_route_check_is_route_or_refusal() {
        use DType::{MQ4G128V2 as Q53, MQ4G256V2 as Q44, MQ6G256V2 as Q6};
        // Existing non-qt53 pairs are untouched.
        assert_eq!(mtp_moe_route_check([(Q44, Q44); 4], 512, 8), Ok(()));
        assert_eq!(mtp_moe_route_check([(Q44, Q6), (Q6, Q44)], 100, 4), Ok(()));
        assert_eq!(mtp_moe_route_check([(Q44, Q53); 4], 512, 8), Ok(()));
        // Mixed down, qt53 gate_up, non-qt44 gate_up, bad K, bad top_k.
        assert!(mtp_moe_route_check([(Q44, Q53), (Q44, Q44)], 512, 8).is_err());
        assert!(mtp_moe_route_check([(Q53, Q53)], 512, 8).is_err());
        assert!(mtp_moe_route_check([(Q53, Q44)], 512, 8).is_err());
        assert!(mtp_moe_route_check([(Q6, Q53)], 512, 8).is_err());
        assert!(mtp_moe_route_check([(Q44, Q53)], 576, 8).is_err());
        assert!(mtp_moe_route_check([(Q44, Q53)], 0, 8).is_err());
        assert!(mtp_moe_route_check([(Q44, Q53)], 512, 10).is_err());
    }

    #[test]
    fn batched_mtp_head_refuses_any_qt53_with_typed_serial_only_variant() {
        use DType::{MQ4G128V2 as Q53, MQ4G256V2 as Q44, MQ6G256V2 as Q6, Q8_0};
        assert!(mtp_batched_qt53_refusal([Q44, Q6, Q8_0, DType::MQ4G256]).is_ok());
        assert!(mtp_batched_qt53_refusal(std::iter::empty()).is_ok());
        for dtypes in [vec![Q44, Q53], vec![Q53], vec![Q8_0, Q6, Q53, Q44]] {
            match mtp_batched_qt53_refusal(dtypes).unwrap_err() {
                DispatchError::UnsupportedVariant {
                    family,
                    variant,
                    quant,
                    ..
                } => {
                    assert_eq!(family, "mtp-head");
                    assert_eq!(variant, "qt53-batched-mtp-unsupported-use-serial");
                    assert_eq!(quant, "MQ4G128V2");
                }
                other => panic!("expected UnsupportedVariant, got {other:?}"),
            }
        }
    }

    #[test]
    fn batched_prompt_fill_keeps_q8_mqv2_and_falls_back_elsewhere() {
        assert!(weight_gemm_batched_supported(DType::MQ4G256V2));
        assert!(weight_gemm_batched_supported(DType::MQ6G256V2));
        assert!(weight_gemm_batched_supported(DType::MQ4G256));
        assert!(weight_gemm_batched_supported(DType::Q8_0));
        assert!(!weight_gemm_batched_supported(DType::MQ6G256));

        let mqv2 = [
            DType::MQ4G256V2,
            DType::MQ4G256V2,
            DType::MQ6G256V2,
            DType::MQ6G256V2,
        ];
        assert!(mtp_prompt_fill_uses_batched(MtpKvMode::Q8, &mqv2));
        assert!(!mtp_prompt_fill_uses_batched(MtpKvMode::Asym3, &mqv2));
        assert!(!mtp_prompt_fill_uses_batched(MtpKvMode::Fwht4, &mqv2));
        assert!(!mtp_prompt_fill_uses_batched(
            MtpKvMode::Q8,
            &[
                DType::MQ6G256,
                DType::MQ4G256V2,
                DType::MQ4G256V2,
                DType::MQ4G256V2
            ]
        ));
    }
}

#[cfg(test)]
mod trunk_compat_tests {
    use super::check_mtp_head_for_trunk;
    use hipfire_runtime::hfq::write_hfqm_package_mem;
    use std::path::PathBuf;

    fn write_head(name: &str, arch_id: u32, meta: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("mtp-compat-{}-{name}", std::process::id()));
        write_hfqm_package_mem(&path, arch_id, meta, &[]).expect("write test head");
        path
    }

    #[test]
    fn head_matching_trunk_dims_is_accepted() {
        let path = write_head(
            "ok.mtp",
            21,
            r#"{"n_embd":5120,"vocab_size":248320,"source_model":"qwen3.8-27b"}"#,
        );
        assert_eq!(check_mtp_head_for_trunk(&path, 0, 5120, 248320), Ok(()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn head_with_other_hidden_size_or_vocab_is_refused() {
        // A Qwen3.6-35B-A3B head (hidden 2048) beside a 27B trunk (hidden 5120).
        let path = write_head(
            "a3b.mtp",
            21,
            r#"{"n_embd":2048,"vocab_size":248320,"source_model":"qwen3.6-35b-a3b"}"#,
        );
        let err = check_mtp_head_for_trunk(&path, 0, 5120, 248320).unwrap_err();
        assert!(err.contains("head n_embd=2048"), "{err}");
        assert!(err.contains("trunk hidden=5120"), "{err}");
        assert!(err.contains("source_model=qwen3.6-35b-a3b"), "{err}");
        // Same hidden size, different vocabulary.
        let err = check_mtp_head_for_trunk(&path, 0, 2048, 151936).unwrap_err();
        assert!(err.contains("vocab=248320"), "{err}");
        assert!(err.contains("trunk hidden=2048 vocab=151936"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn non_mtp_container_is_refused_without_panicking() {
        let path = write_head("trunk.mtp", 5, r#"{"n_embd":5120,"vocab_size":248320}"#);
        let err = check_mtp_head_for_trunk(&path, 0, 5120, 248320).unwrap_err();
        assert!(err.contains("arch_id=5"), "{err}");
        let _ = std::fs::remove_file(path);

        let path = write_head("nodims.mtp", 21, r#"{"n_embd":5120}"#);
        let err = check_mtp_head_for_trunk(&path, 0, 5120, 248320).unwrap_err();
        assert!(err.contains("no 'vocab_size'"), "{err}");
        let _ = std::fs::remove_file(path);

        let missing = std::env::temp_dir().join("mtp-compat-does-not-exist.mtp");
        assert!(check_mtp_head_for_trunk(&missing, 0, 5120, 248320).is_err());
    }
}

// ─── Forward pass ────────────────────────────────────────────────────────

/// Single MTP-head forward. Produces logits over the full vocab in
/// `scratch.logits` for position `pos + 1`. Caller does sampling.
///
/// - `next_token`: the most-recently committed token id at position `pos`.
///   Embedded via the trunk's `weights.token_embd` and used as the
///   prediction-target signal.
/// - `prev_hidden`: the trunk's post-final-norm hidden state at position
///   `pos` (or any other contextually-equivalent activation that's
///   distillation-aligned with the MTP block's training input).
/// - `pos`: current position (the slot the MTP block writes its K/V into,
///   then attends to all positions 0..=pos).
/// - `lm_head_weights`: the trunk's `weights.output`. The MTP file
///   intentionally does NOT pack a separate LM head — Qwen3.5/3.6 share
///   the trunk's lm_head with the MTP head ("shared_lm_head_with_trunk":
///   true in the mtp metadata).
///
/// Side effects:
/// - Writes K/V at slot `pos` into `kv` (overwriting any prior data there).
/// - Writes logits into `scratch.logits` (caller `download_f32` to read).
/// - Writes the post-FFN, pre-LM-head-norm hidden into `scratch.t_mtp_out`
///   so callers wanting an n+2 prediction can feed it back as `prev_hidden`.
pub fn mtp_head_forward(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    pos: usize,
    trunk_weights: &Qwen35Weights,
    lm_head_weights: &WeightTensor,
) -> HipResult<()> {
    let cfg = &head.config;
    let n_embd = cfg.n_embd;

    assert_eq!(
        lm_head_weights.k, n_embd,
        "mtp_head_forward: lm_head_weights.k={} but n_embd={n_embd}; \
         lm_head must accept the MTP head's hidden",
        lm_head_weights.k,
    );

    // Run the block (NextN concat + eh_proj + attn + FFN). Writes
    // `scratch.t_mtp_out`.
    mtp_head_forward_block_only(
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        None,
        pos,
        trunk_weights,
    )?;

    // Standard single-step lm_head: shared_head_norm + GEMV over t_mtp_out.
    let w = &head.weights;
    gpu.rmsnorm_f32(
        &scratch.t_mtp_out,
        &w.shared_head_norm,
        &scratch.tmp,
        cfg.rms_norm_eps,
    )?;
    weight_gemv(gpu, lm_head_weights, &scratch.tmp, &scratch.logits)?;

    Ok(())
}

/// FastMTP-style compressed forward. Identical to [`mtp_head_forward`]
/// but dispatches the LM head against the head's bundled compressed
/// `lm_head_draft` (shape `[K, n_embd]`) instead of the trunk's full
/// `[V, n_embd]` head. Writes logits into `scratch.logits_compressed`
/// (must be pre-allocated via [`Qwen35MtpHeadScratch::ensure_compressed_logits`]).
///
/// Caller does argmax over the K-element compressed logit vector and
/// remaps the index to a full-vocab token id via `weights.lm_head_draft_vocab_map`
/// (a tiny CPU-side `Vec<u32>`).
///
/// Lossless greedy guarantee: trunk verification is unchanged (still
/// uses the full vocab head), so any out-of-K-vocab token the trunk
/// would have emitted is automatically rejected by the argmax mismatch.
/// Only effect of an unrepresentative sidecar is degraded τ — the engine
/// emits the same tokens it would have emitted in plain AR.
///
/// Panics if the head was loaded without a compressed sidecar (check
/// `head.weights.lm_head_draft.is_some()` before calling).
pub fn mtp_head_forward_compressed(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    pos: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    // Block forward — identical to mtp_head_forward; produces t_mtp_out.
    mtp_head_forward_block_only(
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        None,
        pos,
        trunk_weights,
    )?;

    mtp_head_apply_lm_head_draft(gpu, head, scratch)?;

    Ok(())
}

/// Applies the compressed draft LM head to `scratch.t_mtp_out`.
///
/// This is the shared tail of the host-mediated compressed path and the
/// device-token-chain path. Keeping it in one helper prevents those paths from
/// drifting if the compressed head gains another post-block transform.
pub fn mtp_head_apply_lm_head_draft(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let lm_head_draft = w
        .lm_head_draft
        .as_ref()
        .expect("mtp_head_apply_lm_head_draft called but head has no lm_head_draft sidecar");
    let logits_c = scratch.logits_compressed.as_ref().expect(
        "mtp_head_apply_lm_head_draft: scratch.logits_compressed not allocated; \
                 call Qwen35MtpHeadScratch::ensure_compressed_logits first",
    );

    assert_eq!(
        lm_head_draft.k, cfg.n_embd,
        "mtp_head_apply_lm_head_draft: lm_head_draft.k={} but n_embd={}",
        lm_head_draft.k, cfg.n_embd,
    );

    gpu.rmsnorm_f32(
        &scratch.t_mtp_out,
        &w.shared_head_norm,
        &scratch.tmp,
        cfg.rms_norm_eps,
    )?;
    weight_gemv(gpu, lm_head_draft, &scratch.tmp, logits_c)
}

/// Block-only variant of [`mtp_head_forward`]: runs the NextN concat +
/// eh_proj + attention + FFN, but **stops before** `shared_head_norm` and
/// `lm_head`. Writes the post-FFN, pre-shared-head-norm hidden into
/// `scratch.t_mtp_out`. Caller is responsible for running
/// [`mtp_head_apply_lm_head_batched`] on a stack of N `t_mtp_out`s to
/// recover the predicted-token logits in one batched GEMM.
///
/// ## Optional embedding override (lossy K-step chaining)
///
/// In the standard path (`next_token_embed = None`), `next_token` is
/// embedded via the trunk's `token_embd` table — same as
/// `mtp_head_forward`. The K-step batched-lm_head optimization in
/// `spec_step_mtp` chains forwards WITHOUT yet knowing each step's
/// predicted token (we postpone all K argmaxes to a single end-of-chain
/// batched lm_head). To allow step k+1 to proceed before step k's
/// `lm_head` runs, the caller passes `next_token_embed = Some(prev_step_t_mtp_out)`,
/// which BYPASSES the embedding lookup and feeds the previous step's
/// `t_mtp_out` directly as the "embedding of the predicted token."
///
/// This is **architecturally lossy** — the MTP head was trained with
/// discrete-token round-trips through `token_embd`, so feeding the
/// continuous post-FFN hidden as a substitute for `embed[token]` is OOD
/// for the head. Acceptance rate (τ) may degrade. Lossless guarantee is
/// preserved at the trunk-verify level: any incorrect MTP candidate is
/// rejected by the trunk's argmax check and the cycle just re-AR-decodes
/// from the bonus token.
///
/// `next_token` is IGNORED when `next_token_embed` is `Some(_)`. The
/// caller may pass any sentinel (e.g. 0).
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_only(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    next_token_embed: Option<&GpuTensor>,
    pos: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    let cfg = &head.config;
    let n_embd = cfg.n_embd;

    assert_eq!(
        prev_hidden.numel(),
        n_embd,
        "mtp_head_forward_block_only: prev_hidden has {} elems but expected n_embd={n_embd}",
        prev_hidden.numel(),
    );
    assert!(
        pos < kv.max_seq,
        "mtp_head_forward_block_only: pos={pos} >= kv.max_seq={}",
        kv.max_seq,
    );

    // Upload position scalar for the RoPE / attention kernels.
    let pos_i32 = pos as i32;
    gpu.hip
        .memcpy_htod(&scratch.pos_buf, &pos_i32.to_ne_bytes())?;

    mtp_head_forward_block_only_with_pos_buf(
        gpu,
        head,
        scratch,
        kv,
        next_token,
        prev_hidden,
        next_token_embed,
        &scratch.pos_buf,
        pos,
        pos + 1,
        trunk_weights,
    )
}

/// Graph-capturable variant of [`mtp_head_forward_block_only`].
///
/// `pos_buf` must point at an i32 device slot containing `pos`. Captured
/// proposal graphs pass a stable per-step slot here and update the slot bytes
/// before each graph launch, avoiding scalar position arguments baked into
/// captured nodes. `seq_len_hint` controls the attention launch shape and
/// shared-memory reservation; graph callers pass a tier cap that is >= every
/// replayed `pos + 1`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_only_with_pos_buf(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &Qwen35MtpHeadScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_token: u32,
    prev_hidden: &GpuTensor,
    next_token_embed: Option<&GpuTensor>,
    pos_buf: &DeviceBuffer,
    pos: usize,
    seq_len_hint: usize,
    trunk_weights: &Qwen35Weights,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let n_embd = cfg.n_embd;
    mtp_head_serial_route_check(w, cfg)?;

    assert_eq!(
        prev_hidden.numel(),
        n_embd,
        "mtp_head_forward_block_only_with_pos_buf: prev_hidden has {} elems but expected n_embd={n_embd}",
        prev_hidden.numel(),
    );
    assert!(
        pos < kv.max_seq,
        "mtp_head_forward_block_only_with_pos_buf: pos={pos} >= kv.max_seq={}",
        kv.max_seq,
    );

    // ── 1. Token embedding (table lookup OR caller-supplied override) ────
    let dim_bytes = n_embd * 4;
    if let Some(embed) = next_token_embed {
        assert_eq!(
            embed.numel(), n_embd,
            "mtp_head_forward_block_only: next_token_embed has {} elems but expected n_embd={n_embd}",
            embed.numel(),
        );
        // Lossy-recursion path: feed the caller's pre-computed activation
        // directly into the e_norm input slot. Aliasing-safe (separate
        // backing buffers — caller passes the previous step's t_mtp_out).
        gpu.memcpy_dtod_at_auto(&scratch.tok_embd.buf, 0, &embed.buf, 0, dim_bytes)?;
    } else {
        embed_lookup_into(gpu, trunk_weights, &scratch.tok_embd, next_token, n_embd)?;
    }

    // ── 2. RMSNorm both inputs to the NextN projection ───────────────────
    gpu.rmsnorm_f32(
        &scratch.tok_embd,
        &w.enorm,
        &scratch.e_norm,
        cfg.rms_norm_eps,
    )?;
    gpu.rmsnorm_f32(prev_hidden, &w.hnorm, &scratch.h_norm, cfg.rms_norm_eps)?;

    // ── 3. concat = [e_norm | h_norm], then cur = eh_proj @ concat ───────
    gpu.memcpy_dtod_at_auto(&scratch.concat.buf, 0, &scratch.e_norm.buf, 0, dim_bytes)?;
    gpu.memcpy_dtod_at_auto(
        &scratch.concat.buf,
        dim_bytes,
        &scratch.h_norm.buf,
        0,
        dim_bytes,
    )?;
    weight_gemv(gpu, &w.eh_proj, &scratch.concat, &scratch.cur)?;

    // ── 4-9. Gated full attention (the trunk's decode step), residual in
    // place on `cur`. The KV slot and attention launch shape come from
    // `seq_len_hint - 1`; the RoPE and KV-write position flows via `pos_buf`.
    // `tier_inputs()` reports flash_mode 0, which would pin the Q8 tier to the
    // non-flash AttnQ8_0Kv at any context (10 ms per draft step at 33k against
    // 0.3 ms on the flash tile), so the head takes the trunk's flash policy.
    // flash_partials is sized with the same tile the launch picks (see
    // `Qwen35MtpHeadScratch::new`).
    let ctx = hipfire_dispatch::context::DispatchCtx::new(gpu);
    let attention = GatedAttentionOp {
        dims: cfg.hybrid_dims(),
        x: &scratch.cur,
        prerotated_input: false,
        attn_norm: &w.attn_norm,
        wq: w.wq.dispatch_ref(),
        wk: w.wk.dispatch_ref(),
        wv: w.wv.dispatch_ref(),
        q_norm: &w.attn_q_norm,
        k_norm: &w.attn_k_norm,
        wo: w.wo.dispatch_ref(),
        kv: AttentionKv {
            tier: hipfire_dispatch::families::kv_tier::KvTierInputs {
                flash_mode: llama::attention_flash_mode(&gpu.arch),
                ..kv.inner.tier_inputs()
            },
            k_cache: &kv.inner.k_gpu[0],
            v_cache: &kv.inner.v_gpu[0],
            physical_cap: kv.inner.physical_cap,
            givens_cos: kv.inner.givens_cos.as_ref(),
            givens_sin: kv.inner.givens_sin.as_ref(),
            compact_offset: 0,
        },
        row: AttentionRow {
            position: seq_len_hint - 1,
            pos_buf,
            q: &scratch.q,
            k: &scratch.k,
            v: &scratch.v,
            gate: &scratch.gate,
            flash_partials: &scratch.flash_partials,
            out: &scratch.attn_out,
        },
        plain: &scratch.tmp,
        x_rot: &scratch.residual,
        q_gate: &scratch.q_full,
        tap: None,
        mrope: None,
        output: hipfire_dispatch::pipeline::hybrid::MixerOutput::Residual,
    };
    execute_steps(gpu, &ctx, &[Step::GatedAttention(attention)]).map_err(HipError::from)?;

    // ── 10. FFN (dense SwiGLU or the trunk's sealed MoE), residual in place
    // on `cur`. `attn_post_norm` is the pre-FFN norm; its name reflects its
    // source position in the .mtp metadata.
    match &w.ffn {
        Qwen35MtpFfnWeights::Dense(ffn) => {
            let gate = ffn.gate.dispatch_ref();
            let ffn = GatedMlpOp {
                rows: 1,
                hidden: n_embd,
                intermediate: gate.m,
                x: &scratch.cur,
                pre_norm: Some(RmsNorm {
                    weight: &w.attn_post_norm,
                    eps: cfg.rms_norm_eps,
                    normed: &scratch.tmp,
                }),
                gate,
                up: ffn.up.dispatch_ref(),
                down: ffn.down.dispatch_ref(),
                activation: Activation::Silu,
                storage: MlpStorage::F32,
                output: MlpOutput::Residual,
                ladder: MlpLadder::Product,
                scratch: GatedMlpScratch {
                    gate: &scratch.gate_ffn,
                    up: &scratch.up,
                    act: &scratch.ffn_hidden,
                    x_rot: &scratch.residual,
                },
            };
            execute_steps(gpu, &ctx, &[Step::GatedMlp(ffn)]).map_err(HipError::from)?;
        }
        Qwen35MtpFfnWeights::Moe { ffn, config } => {
            let refs = scratch.moe_refs();
            crate::qwen35::forward::moe_ffn_decode_rmsnorm(
                gpu,
                ffn,
                &scratch.cur,
                &w.attn_post_norm,
                &scratch.tmp,
                config,
                &refs,
            )?;
        }
    }

    // Snapshot the post-FFN, pre-LM-head-norm hidden for callers that chain
    // into n+2 prediction OR feed the batched end-of-chain LM head.
    gpu.memcpy_dtod_at_auto(&scratch.t_mtp_out.buf, 0, &scratch.cur.buf, 0, dim_bytes)?;

    Ok(())
}

/// Batched end-of-chain LM head: applies `shared_head_norm` to `n` stacked
/// `t_mtp_out` rows and runs the trunk's lm_head as a single batched GEMM.
///
/// `t_mtp_outs_stacked` has shape `[n, n_embd]` (row-major, contiguous).
/// `logits_batched` is the caller-allocated output of shape `[n, vocab]`.
/// `tmp_batched` is `[n, n_embd]` scratch for the rmsnorm output (caller
/// owns; reused across cycles).
/// `rot_batched` is `[n, n_embd]` scratch used for FWHT-rotated x for
/// MagnumQuant lm_heads (MQ4/MQ3/MQ6); ignored for non-MQ dtypes.
///
/// Mirrors the per-dtype dispatch in `mtp_probe::probe_one_step` and
/// `speculative::verify_dflash_block_inner`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_apply_lm_head_batched(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    lm_head_weights: &WeightTensor,
    t_mtp_outs_stacked: &GpuTensor,
    tmp_batched: &GpuTensor,
    rot_batched: &GpuTensor,
    logits_batched: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    let cfg = &head.config;
    let w = &head.weights;
    let n_embd = cfg.n_embd;
    // Output dim = the lm_head weight's row count. For the FULL trunk lm_head
    // this matches cfg.vocab_size; for FastMTP-style compressed lm_head_draft
    // this is the (smaller) compressed_vocab_size — the function is parametric
    // on whichever WeightTensor the caller passes in.
    let vocab = lm_head_weights.m;
    assert_eq!(
        lm_head_weights.k, n_embd,
        "mtp_head_apply_lm_head_batched: lm_head_weights.k={} but n_embd={n_embd}",
        lm_head_weights.k,
    );
    assert!(
        t_mtp_outs_stacked.numel() >= n * n_embd,
        "t_mtp_outs_stacked too small: {} < n*n_embd ({})",
        t_mtp_outs_stacked.numel(),
        n * n_embd,
    );
    assert!(
        tmp_batched.numel() >= n * n_embd,
        "tmp_batched too small: {} < n*n_embd ({})",
        tmp_batched.numel(),
        n * n_embd,
    );
    assert!(
        logits_batched.numel() >= n * vocab,
        "logits_batched too small: {} < n*vocab ({})",
        logits_batched.numel(),
        n * vocab,
    );

    // Per-row shared_head_norm.
    gpu.rmsnorm_batched(
        t_mtp_outs_stacked,
        &w.shared_head_norm,
        tmp_batched,
        n,
        n_embd,
        cfg.rms_norm_eps,
    )?;

    // Per-dtype batched LM head dispatch (mirrors mtp_probe.rs:278+).
    let logits_view = logits_batched.sub_offset(0, n * vocab);
    match lm_head_weights.gpu_dtype {
        DType::Q8_0 => {
            // Route the full-vocab MTP-head lm_head to the WMMA Q8 kernel
            // (gemm_q8_0_wmma: gfx12 gemm_q8_0_wmma_gfx12 ~0.3ms / gfx11 RDNA3
            // sibling) instead of the scalar 1-wave-per-row kernel — which
            // rocprofv3 (2026-06-16, A3B K=1 gfx1201) measured at ~9.2ms/cycle
            // = 40% of all MTP GPU time (one full-vocab B=2 GEMM per proposal).
            // Mirrors the trunk verify lm_head (mtp_spec.rs:2704), but calls the
            // WMMA kernel DIRECTLY rather than via gemm_q8_0_batched_chunked
            // (whose WMMA arm is gated to is_rdna4()) so RDNA3 (gfx11) — where
            // the head lm_head is ALSO scalar today — gets the fix too. WMMA
            // needs wave32 (gfx11+) and K%32==0; else fall back to scalar.
            // Opt out with HIPFIRE_MTP_HEAD_LMHEAD_WMMA=0.
            let use_wmma = hipfire_config::developer_var("HIPFIRE_MTP_HEAD_LMHEAD_WMMA")
                .ok()
                .as_deref()
                != Some("0")
                && gpu.arch_caps.has_wmma()
                && lm_head_weights.k % 32 == 0;
            if use_wmma {
                gpu.gemm_q8_0_wmma(
                    &lm_head_weights.buf,
                    tmp_batched,
                    &logits_view,
                    lm_head_weights.m,
                    lm_head_weights.k,
                    n,
                )?;
            } else {
                gpu.gemm_q8_0_batched_f32_chunked(
                    &lm_head_weights.buf,
                    tmp_batched,
                    &logits_view,
                    lm_head_weights.m,
                    lm_head_weights.k,
                    n,
                )?;
            }
        }
        DType::HFQ4G256 => {
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf,
                tmp_batched,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ4G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq4g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ3G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq3g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::HFQ6G256 => {
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf,
                tmp_batched,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        DType::MQ6G256 => {
            let rot_view = rot_batched.sub_offset(0, n * lm_head_weights.k);
            llama::rotate_x_mq_batched_for(
                gpu,
                lm_head_weights,
                tmp_batched,
                &rot_view,
                lm_head_weights.k,
                n,
            )?;
            gpu.gemm_hfq6g256_batched_lmhead(
                &lm_head_weights.buf,
                &rot_view,
                &logits_view,
                lm_head_weights.m,
                lm_head_weights.k,
                n,
            )?;
        }
        _ => {
            // Fallback: per-row weight_gemv. Same path mtp_probe uses for
            // unrecognized dtypes. Defeats the K-amortization but keeps
            // correctness for less-common lm_head formats.
            for i in 0..n {
                let row = tmp_batched.sub_offset(i * n_embd, n_embd);
                let logits_row = logits_view.sub_offset(i * vocab, vocab);
                weight_gemv(gpu, lm_head_weights, &row, &logits_row)?;
            }
        }
    }
    Ok(())
}

/// Per-format embedding-lookup dispatch. Mirrors `mtp_probe::embed_lookup_to_scratch`.
fn embed_lookup_into(
    gpu: &mut Gpu,
    weights: &Qwen35Weights,
    out: &GpuTensor,
    token: u32,
    dim: usize,
) -> HipResult<()> {
    match weights.embd_format {
        EmbeddingFormat::HFQ4G256 => {
            gpu.embedding_lookup_hfq4g256(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::HFQ4G128 => {
            gpu.embedding_lookup_hfq4g128(&weights.token_embd, out, token, dim)
        }
        EmbeddingFormat::Q8_0 => gpu.embedding_lookup_q8(&weights.token_embd, out, token, dim),
        EmbeddingFormat::Q4K => gpu.embedding_lookup_q4k(&weights.token_embd, out, token, dim),
        EmbeddingFormat::F32 => gpu.embedding_lookup(&weights.token_embd, out, token, dim),
        EmbeddingFormat::Ggml(_) => Err(hip_bridge::HipError::new(
            0,
            "GGUF-native embedding tables are served by the dense qwen35 decode/prefill route only",
        )),
    }
}

// ─── Batched MTP head forward (Task 11b) ─────────────────────────────────

/// Per-call GPU scratch for the batched MTP head forward — sized for up to
/// `max_n` MTP forwards processed in a single per-layer GEMM batch. Mirrors
/// [`Qwen35MtpHeadScratch`] but with each tensor sized `max_n × <dim>`.
///
/// All buffers are row-major contiguous: row `i` for slot i lives at byte
/// offset `i * <dim> * 4`, exactly as the trunk's batched-prefill scratch
/// expects (and matches the layout the existing batched GEMM /
/// rmsnorm_batched / rope_batched / deinterleave_batched kernels read).
pub struct Qwen35MtpHeadBatchedScratch {
    pub max_n: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    // Activations
    pub tok_embd: GpuTensor, // [max_n × n_embd]
    pub e_norm: GpuTensor,   // [max_n × n_embd]
    pub h_norm: GpuTensor,   // [max_n × n_embd]
    pub concat: GpuTensor,   // [max_n × 2 × n_embd]
    pub cur: GpuTensor,      // [max_n × n_embd]
    pub residual: GpuTensor, // [max_n × n_embd]
    pub tmp: GpuTensor,      // [max_n × n_embd]
    // Attention sub-block
    pub q_full: GpuTensor,   // [max_n × 2 × q_dim]
    pub q: GpuTensor,        // [max_n × q_dim]
    pub gate: GpuTensor,     // [max_n × q_dim]
    pub k: GpuTensor,        // [max_n × kv_dim]
    pub v: GpuTensor,        // [max_n × kv_dim]
    pub attn_out: GpuTensor, // [max_n × q_dim]
    pub o: GpuTensor,        // [max_n × n_embd]
    // FFN sub-block
    pub gate_ffn: GpuTensor,   // [max_n × n_ff]
    pub up: GpuTensor,         // [max_n × n_ff]
    pub ffn_hidden: GpuTensor, // [max_n × n_ff]
    pub ffn_out: GpuTensor,    // [max_n × n_embd]
    /// Stacked post-FFN, pre-LM-head-norm hidden — feeds
    /// `mtp_head_apply_lm_head_batched` directly. [max_n × n_embd]
    pub t_mtp_outs: GpuTensor,
    /// Per-slot positions device buffer. [max_n] i32, uploaded fresh each call.
    pub positions: GpuTensor,
    /// True only for scratch from [`Self::new_for_prompt_fill`]: the eleven
    /// inactive fields are borrowed views and `free_gpu` releases nine owners.
    prompt_fill_only: bool,
}

impl Qwen35MtpHeadBatchedScratch {
    pub fn new(gpu: &mut Gpu, config: &Qwen35MtpHeadConfig, max_n: usize) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "Qwen35MtpHeadBatchedScratch: max_n must be >= 1"
        );
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        Ok(Self {
            max_n,
            n_embd: dim,
            n_ff: config.n_ff,
            q_dim,
            kv_dim,
            tok_embd: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            e_norm: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            h_norm: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            concat: gpu.alloc_tensor(&[max_n * 2 * dim], DType::F32)?,
            cur: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            residual: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            tmp: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            q_full: gpu.alloc_tensor(&[max_n * 2 * q_dim], DType::F32)?,
            q: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            gate: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            k: gpu.alloc_tensor(&[max_n * kv_dim], DType::F32)?,
            v: gpu.alloc_tensor(&[max_n * kv_dim], DType::F32)?,
            attn_out: gpu.alloc_tensor(&[max_n * q_dim], DType::F32)?,
            o: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            gate_ffn: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            up: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            ffn_hidden: gpu.alloc_tensor(&[max_n * config.n_ff], DType::F32)?,
            ffn_out: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            t_mtp_outs: gpu.alloc_tensor(&[max_n * dim], DType::F32)?,
            positions: gpu.alloc_tensor(&[max_n], DType::F32)?, // F32 alias for i32 storage
            prompt_fill_only: false,
        })
    }

    /// Lean scratch for the fast kv_only prompt fill: allocates only
    /// `tok_embd, e_norm, h_norm, concat, cur, tmp, k, v, positions`. The
    /// other eleven fields are one-element borrowed views of `positions`
    /// (no extra HIP allocations). It is valid only for
    /// [`mtp_head_forward_block_batched`] with `kv_only && HIPFIRE_MTP_FILL_FAST`;
    /// any other use is refused by an assertion. A failed allocation releases
    /// every buffer allocated so far before the error is returned.
    pub(crate) fn new_for_prompt_fill(
        gpu: &mut Gpu,
        config: &Qwen35MtpHeadConfig,
        max_n: usize,
    ) -> HipResult<Self> {
        assert!(
            max_n >= 1,
            "Qwen35MtpHeadBatchedScratch: max_n must be >= 1"
        );
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;

        // Transactional construction: `owned` holds every successful
        // allocation until the struct is built; on error, reverse-release.
        let numels = [
            max_n * dim,     // tok_embd
            max_n * dim,     // e_norm
            max_n * dim,     // h_norm
            max_n * 2 * dim, // concat
            max_n * dim,     // cur
            max_n * dim,     // tmp
            max_n * kv_dim,  // k
            max_n * kv_dim,  // v
            max_n,           // positions
        ];
        let mut owned: [Option<GpuTensor>; 9] = Default::default();
        for (i, numel) in numels.into_iter().enumerate() {
            match gpu.alloc_tensor(&[numel], DType::F32) {
                Ok(t) => owned[i] = Some(t),
                Err(e) => {
                    for slot in owned.iter_mut().rev() {
                        if let Some(t) = slot.take() {
                            let _ = gpu.free_tensor(t);
                        }
                    }
                    return Err(e);
                }
            }
        }
        let mut take = |i: usize| owned[i].take().expect("lean MTP scratch slot taken twice");
        let tok_embd = take(0);
        let e_norm = take(1);
        let h_norm = take(2);
        let concat = take(3);
        let cur = take(4);
        let tmp = take(5);
        let k = take(6);
        let v = take(7);
        let positions = take(8);
        // One-element non-owning views of the owned positions buffer; never
        // freed, never dereferenced beyond one element.
        let sentinel = || positions.sub_offset(0, 1);
        Ok(Self {
            max_n,
            n_embd: dim,
            n_ff: config.n_ff,
            q_dim,
            kv_dim,
            tok_embd,
            e_norm,
            h_norm,
            concat,
            cur,
            residual: sentinel(),
            tmp,
            q_full: sentinel(),
            q: sentinel(),
            gate: sentinel(),
            k,
            v,
            attn_out: sentinel(),
            o: sentinel(),
            gate_ffn: sentinel(),
            up: sentinel(),
            ffn_hidden: sentinel(),
            ffn_out: sentinel(),
            t_mtp_outs: sentinel(),
            positions,
            prompt_fill_only: true,
        })
    }

    /// Allocation-size model for [`Self::new`] / [`Self::new_for_prompt_fill`]:
    /// one pool allocation of `numel` F32 elements, with the pool's 256 B floor.
    fn requested_bytes(numel: usize) -> usize {
        (numel * 4).max(256)
    }

    /// Requested device bytes of the full [`Self::new`] layout (20 buffers).
    fn full_requested_bytes(config: &Qwen35MtpHeadConfig, rows: usize) -> usize {
        let dim = config.n_embd;
        let q_dim = config.head_dim * config.n_head;
        let kv_dim = config.head_dim * config.n_head_kv;
        let r = Self::requested_bytes;
        // tok_embd, e_norm, h_norm, cur, residual, tmp, o, ffn_out, t_mtp_outs
        r(rows * dim) * 9
            + r(rows * 2 * dim)
            + r(rows * 2 * q_dim)
            + r(rows * q_dim) * 3 // q, gate, attn_out
            + r(rows * kv_dim) * 2
            + r(rows * config.n_ff) * 3
            + r(rows)
    }

    /// Requested device bytes of the lean [`Self::new_for_prompt_fill`] layout
    /// (9 owners; the 11 inactive fields are borrowed views of `positions`).
    fn lean_requested_bytes(config: &Qwen35MtpHeadConfig, rows: usize) -> usize {
        let dim = config.n_embd;
        let kv_dim = config.head_dim * config.n_head_kv;
        let r = Self::requested_bytes;
        // tok_embd, e_norm, h_norm, cur, tmp
        r(rows * dim) * 5 + r(rows * 2 * dim) + r(rows * kv_dim) * 2 + r(rows)
    }

    /// Requested bytes of one rotated-input scratch of `k` columns by `rows`.
    fn rotation_requested_bytes(rows: usize, k: usize) -> usize {
        Self::requested_bytes(rows * k)
    }

    /// Requested device bytes of the legacy prompt fill at `rows` rows:
    /// [`Self::new`]'s full layout plus the `rows x rotation_k` rotation
    /// scratch. Upper bound of the lean fill layout at the same rows.
    pub fn legacy_prompt_fill_bytes(
        config: &Qwen35MtpHeadConfig,
        rows: usize,
        rotation_k: usize,
    ) -> usize {
        Self::full_requested_bytes(config, rows)
            .saturating_add(Self::rotation_requested_bytes(rows, rotation_k))
    }

    /// Largest prompt-fill head capacity whose lean scratch + fill rotation
    /// requested bytes do not exceed the legacy full scratch + legacy rotation
    /// requested bytes at `legacy_rows`. The budget capacity is aligned down to
    /// a multiple of 8 when it is at least 8, and only then clamped by
    /// `prompt_len` and `trunk_chunk_max`, so a short prompt keeps a single
    /// call of exactly its own length. Pure metadata: no GPU query or
    /// allocation. Returns 0 for an empty prompt; otherwise at least 1.
    pub(crate) fn prompt_fill_rows_for_budget(
        cfg: &Qwen35MtpHeadConfig,
        legacy_rows: usize,
        trunk_chunk_max: usize,
        prompt_len: usize,
        legacy_rotation_k: usize,
        fill_rotation_k: usize,
    ) -> usize {
        if prompt_len == 0 || legacy_rows == 0 || trunk_chunk_max == 0 {
            return 0;
        }
        let budget = Self::full_requested_bytes(cfg, legacy_rows).saturating_add(
            Self::rotation_requested_bytes(legacy_rows, legacy_rotation_k),
        );
        let fits = |rows: usize| {
            Self::lean_requested_bytes(cfg, rows)
                .saturating_add(Self::rotation_requested_bytes(rows, fill_rotation_k))
                <= budget
        };
        // Requested bytes are monotonically non-decreasing in rows. Double
        // from the legacy row count to the first non-fitting bound (lean +
        // fill rotation is never wider than legacy per row, so this ends within
        // a few doublings), then binary-search the largest fitting row count.
        let mut lo = 0usize; // largest known-fitting rows (0 = none known)
        let mut bad = legacy_rows;
        while fits(bad) {
            lo = bad;
            bad *= 2;
        }
        while bad - lo > 1 {
            let mid = lo + (bad - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                bad = mid;
            }
        }
        let aligned = if lo >= 8 { lo & !7 } else { lo };
        aligned.min(prompt_len).min(trunk_chunk_max).max(1)
    }

    /// Frees owned allocations. Lean scratch owns only the nine active
    /// buffers; its eleven inactive fields are borrowed views of `positions`
    /// and must never reach `free_tensor`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.tok_embd);
        let _ = gpu.free_tensor(self.e_norm);
        let _ = gpu.free_tensor(self.h_norm);
        let _ = gpu.free_tensor(self.concat);
        let _ = gpu.free_tensor(self.cur);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.k);
        let _ = gpu.free_tensor(self.v);
        if !self.prompt_fill_only {
            let _ = gpu.free_tensor(self.residual);
            let _ = gpu.free_tensor(self.q_full);
            let _ = gpu.free_tensor(self.q);
            let _ = gpu.free_tensor(self.gate);
            let _ = gpu.free_tensor(self.attn_out);
            let _ = gpu.free_tensor(self.o);
            let _ = gpu.free_tensor(self.gate_ffn);
            let _ = gpu.free_tensor(self.up);
            let _ = gpu.free_tensor(self.ffn_hidden);
            let _ = gpu.free_tensor(self.ffn_out);
            let _ = gpu.free_tensor(self.t_mtp_outs);
        }
        let _ = gpu.free_tensor(self.positions);
    }
}

/// Per-format BATCHED weight GEMM dispatch for the MTP head's inner-layer
/// projections. M is the OUTPUT row count, K is the input dim. Each batch
/// row `i` reads `x[i * k..(i+1) * k]` and writes `y[i * m..(i+1) * m]`.
///
/// Incumbent routes are unchanged unless this is an eh/wk/wv fast-fill
/// projection: eligible MQ4-v1 on gfx1201 then uses the exact PM GEMM.
fn weight_gemm_batched(
    gpu: &mut Gpu,
    w: &WeightTensor,
    x_batched: &GpuTensor,
    y_batched: &GpuTensor,
    n: usize,
    rotated_x_scratch: Option<&GpuTensor>,
    fast_fill: bool,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::Q8_0 => gpu.gemm_q8_0_batched_f32_chunked(&w.buf, x_batched, y_batched, w.m, w.k, n),
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x_batched, y_batched, w.m, w.k, n),
        DType::MQ4G256 => {
            // MQ4 needs an FWHT-rotated x first (matches trunk lm_head + dflash patterns).
            let rot = rotated_x_scratch.expect("MQ4 batched gemm requires rotated_x_scratch");
            llama::rotate_x_mq_batched_for(gpu, w, x_batched, rot, w.k, n)?;
            hipfire_dispatch::families::gemm::run_mtp_fill_mq4(
                gpu,
                &w.buf,
                rot,
                y_batched,
                w.m,
                w.k,
                n,
                fast_fill && mtp_fill_pm_gemm_enabled(),
            )
        }
        DType::MQ4G256V2 => {
            let rot = rotated_x_scratch.expect("MQ4V2 batched gemm requires rotated_x_scratch");
            llama::rotate_x_mq_batched_for(gpu, w, x_batched, rot, w.k, n)?;
            hipfire_dispatch::pipeline::batched::run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq4G256V2,
                &w.buf,
                w.gpu_dtype,
                rot,
                y_batched,
                w.m,
                w.k,
                n,
            )
        }
        // qt=52: no batched-LUT kernel exists — per-row LUT GEMV via
        // weight_gemv (rotates internally). Slow but correct.
        DType::MQ4G256V2Lloyd => {
            for i in 0..n {
                let x_row = x_batched.sub_offset(i * w.k, w.k);
                let y_row = y_batched.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)?;
            }
            Ok(())
        }
        DType::MQ6G256V2 => {
            let rot = rotated_x_scratch.expect("MQ6V2 batched gemm requires rotated_x_scratch");
            llama::rotate_x_mq_batched_for(gpu, w, x_batched, rot, w.k, n)?;
            hipfire_dispatch::pipeline::batched::run_plain_gemm_key(
                gpu,
                hipfire_dispatch::types::KernelKey::GemmMq6G256V2,
                &w.buf,
                w.gpu_dtype,
                rot,
                y_batched,
                w.m,
                w.k,
                n,
            )
        }
        DType::F32 => {
            // Fallback: per-row gemv (slow but correct). MTP head loaded via
            // load_weight_raw can ship F32 for rare formats — keep functional.
            for i in 0..n {
                let x_row = x_batched.sub_offset(i * w.k, w.k);
                let y_row = y_batched.sub_offset(i * w.m, w.m);
                weight_gemv(gpu, w, &x_row, &y_row)?;
            }
            Ok(())
        }
        other => panic!("weight_gemm_batched: unsupported dtype {:?}", other),
    }
}

/// True when [`weight_gemm_batched`] has a non-panicking arm for `dtype`.
/// Published MQ6G256-v1 sidecars have GEMV but no batched GEMM.
fn weight_gemm_batched_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Q8_0
            | DType::HFQ4G256
            | DType::MQ4G256
            | DType::MQ4G256V2
            // qt=52 served by the per-row LUT-GEMV arm (slow but correct).
            | DType::MQ4G256V2Lloyd
            | DType::MQ6G256V2
            | DType::F32
    )
}

/// Prompt-fill may use the chunked batched path only for Q8 MTP KV and
/// projections [`weight_gemm_batched`] can represent. Asym3/Fwht4 and
/// MQ6G256-v1 keep the tokenwise GEMV / KV-write path.
pub(crate) fn mtp_prompt_fill_uses_batched(
    kv_mode: MtpKvMode,
    projection_dtypes: &[DType],
) -> bool {
    kv_mode == MtpKvMode::Q8
        && projection_dtypes
            .iter()
            .copied()
            .all(weight_gemm_batched_supported)
}

/// Batched MTP head block forward (Task 11b).
///
/// Runs the NextN concat + eh_proj + gated-Q attention + SwiGLU FFN over
/// `n` parallel slots in a single per-layer GEMM batch. Each slot has its
/// OWN `prev_hidden` and `next_token`; the K/V for slot `i` are written to
/// the MTP head's KV cache at `positions[i]`.
///
/// ## v1 simplification: per-slot independent attention
///
/// Each slot's attention reads ONLY its own freshly-written K/V at the
/// caller-supplied position. There's no cross-slot attention, no historical
/// reads from previous cycles' MTP slots. This is the v1 simplification —
/// the MTP head was trained with full historical attention, so τ_mtp may
/// drop, but the trunk verify is the correctness gate: bad MTP candidates
/// just get rejected.
///
/// Concretely, attention reduces to: `out[h, d] = V[h, d]` (single-key
/// attention with self-only key collapses softmax to 1). The gate
/// (sigmoid_mul) then modulates V before `wo`. We still WRITE K/V to the
/// per-slot KV slot so future cycles' historical reads (when wired) would
/// see the correct cache state — slot writes are cheap and keep the cache
/// snapshot invariant simple.
///
/// ## Inputs
///
/// - `prev_hiddens_stacked`: `[n × n_embd]` row-major. Slot i's drafter /
///   trunk hidden that gets fed into `hnorm`.
/// - `next_tokens`: length-`n` slice of u32 token ids — embedded via the
///   trunk's `token_embd` table, then RMSNorm'd via `enorm`.
/// - `positions`: length-`n` slice of i32 absolute MTP-cache slots. Each
///   `positions[i]` MUST be < `kv.max_seq` and unique (or the K/V writes
///   collide). Caller is responsible for slot allocation.
///
/// ## Output
///
/// - `scratch.t_mtp_outs[i]` holds the post-FFN hidden for slot i. Pass the
///   full `t_mtp_outs` view to [`mtp_head_apply_lm_head_batched`] for the
///   batched lm_head step.
///
/// ## Side effects
///
/// - K/V for slot i written into `kv` at slot `positions[i]`. Caller may
///   call `kv.reset(gpu)` after the cycle if the writes shouldn't persist.
#[allow(clippy::too_many_arguments)]
pub fn mtp_head_forward_block_batched(
    gpu: &mut Gpu,
    head: &Qwen35MtpHead,
    scratch: &mut Qwen35MtpHeadBatchedScratch,
    kv: &mut Qwen35MtpHeadKvCache,
    next_tokens: &[u32],
    prev_hiddens_stacked: &GpuTensor,
    positions: &[i32],
    n: usize,
    trunk_weights: &Qwen35Weights,
    rotated_x_scratch: Option<&GpuTensor>,
    kv_only: bool,
) -> HipResult<()> {
    // qt53 has only the serial decode route: typed refusal before ANY launch
    // (attention included), never the generic batched-MoE panic below.
    mtp_batched_qt53_refusal(mtp_head_weight_dtypes(&head.weights))
        .map_err(|error| HipError::new(0, &error.to_string()))?;
    let cfg = &head.config;
    let w = &head.weights;
    let dim = cfg.n_embd;
    let q_dim = cfg.head_dim * cfg.n_head;
    let kv_dim = cfg.head_dim * cfg.n_head_kv;
    let dim_bytes = dim * 4;

    assert_eq!(next_tokens.len(), n, "next_tokens.len() != n");
    assert_eq!(positions.len(), n, "positions.len() != n");
    assert!(
        n <= scratch.max_n,
        "n={n} > scratch.max_n={}",
        scratch.max_n
    );
    assert!(
        prev_hiddens_stacked.numel() >= n * dim,
        "prev_hiddens_stacked too small: {} < n*dim ({})",
        prev_hiddens_stacked.numel(),
        n * dim
    );
    for &p in positions {
        assert!(
            (p as usize) < kv.max_seq,
            "position {p} >= kv.max_seq {}",
            kv.max_seq,
        );
    }

    // Lean prompt-fill scratch owns only the buffers the fast kv_only path
    // touches; refuse any other use before a sentinel view can be dereferenced.
    let fast_fill = kv_only && mtp_fill_fast_enabled();
    assert!(
        !scratch.prompt_fill_only || fast_fill,
        "lean prompt-fill scratch requires kv_only && HIPFIRE_MTP_FILL_FAST"
    );

    // Reuse the raw i32 positions buffer for token ids during embedding, then
    // overwrite it with absolute positions before RoPE/KV writes.
    let token_bytes =
        unsafe { std::slice::from_raw_parts(next_tokens.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&scratch.positions.buf, token_bytes)?;

    let tok_embd_view = scratch.tok_embd.sub_offset(0, n * dim);
    if trunk_weights.embd_format == EmbeddingFormat::Q8_0 {
        gpu.embedding_lookup_q8_batched(
            &trunk_weights.token_embd,
            &tok_embd_view,
            &scratch.positions,
            n,
            dim,
        )?;
    } else {
        for (i, &tok) in next_tokens.iter().enumerate() {
            let dst = scratch.tok_embd.sub_offset(i * dim, dim);
            embed_lookup_into(gpu, trunk_weights, &dst, tok, dim)?;
        }
    }

    let position_bytes =
        unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, n * 4) };
    gpu.hip
        .memcpy_htod(&scratch.positions.buf, position_bytes)?;

    // ── 2. Batched RMSNorm both inputs to NextN projection ───────────────
    let e_norm_view = scratch.e_norm.sub_offset(0, n * dim);
    let prev_view = prev_hiddens_stacked.sub_offset(0, n * dim);
    let h_norm_view = scratch.h_norm.sub_offset(0, n * dim);
    gpu.rmsnorm_batched(
        &tok_embd_view,
        &w.enorm,
        &e_norm_view,
        n,
        dim,
        cfg.rms_norm_eps,
    )?;
    gpu.rmsnorm_batched(&prev_view, &w.hnorm, &h_norm_view, n, dim, cfg.rms_norm_eps)?;

    // ── 3. Build concat = [e_norm | h_norm] per slot ─────────────────────
    // Layout: concat[i] = [e_norm[i, 0..dim], h_norm[i, 0..dim]] (length 2*dim).
    if fast_fill {
        // Two pitched D2D copies replace the 2n per-row copies: e_norm rows go
        // to concat[:, 0..dim], h_norm rows to concat[:, dim..2*dim].
        gpu.memcpy_dtod_2d_at_ordered_async(
            &scratch.concat.buf,
            0,
            2 * dim_bytes,
            &scratch.e_norm.buf,
            0,
            dim_bytes,
            dim_bytes,
            n,
        )?;
        gpu.memcpy_dtod_2d_at_ordered_async(
            &scratch.concat.buf,
            dim_bytes,
            2 * dim_bytes,
            &scratch.h_norm.buf,
            0,
            dim_bytes,
            dim_bytes,
            n,
        )?;
    } else {
        for i in 0..n {
            gpu.hip.memcpy_dtod_at(
                &scratch.concat.buf,
                i * 2 * dim_bytes,
                &scratch.e_norm.buf,
                i * dim_bytes,
                dim_bytes,
            )?;
            gpu.hip.memcpy_dtod_at(
                &scratch.concat.buf,
                i * 2 * dim_bytes + dim_bytes,
                &scratch.h_norm.buf,
                i * dim_bytes,
                dim_bytes,
            )?;
        }
    }

    // cur = eh_proj @ concat (per-slot). w.eh_proj has m=n_embd, k=2*n_embd.
    let concat_view = scratch.concat.sub_offset(0, n * 2 * dim);
    let cur_view = scratch.cur.sub_offset(0, n * dim);
    weight_gemm_batched(
        gpu,
        &w.eh_proj,
        &concat_view,
        &cur_view,
        n,
        rotated_x_scratch,
        fast_fill,
    )?;

    // Save residual (inpSA) for the post-attn add. kv_only returns before the
    // post-attn add, so fast_fill skips it.
    if !fast_fill {
        gpu.hip
            .memcpy_dtod_at(&scratch.residual.buf, 0, &scratch.cur.buf, 0, n * dim_bytes)?;
    }

    // ── 4. Pre-attn norm + Q/K/V projections ─────────────────────────────
    let tmp_view = scratch.tmp.sub_offset(0, n * dim);
    gpu.rmsnorm_batched(&cur_view, &w.attn_norm, &tmp_view, n, dim, cfg.rms_norm_eps)?;

    // wq emits 2 * q_dim per slot. fast_fill (kv_only) never consumes Q/gate,
    // so the wq GEMM, deinterleave and per-head Q norm are skipped. Lean
    // prompt-fill scratch borrows one-element sentinels for q, so no full
    // n*q_dim view may be formed from it: fast_fill uses a one-element valid
    // pointer, and the full q_full/q/gate views exist only inside !fast_fill.
    let q_view = if fast_fill {
        scratch.q.sub_offset(0, 1)
    } else {
        scratch.q.sub_offset(0, n * q_dim)
    };
    if !fast_fill {
        let q_full_view = scratch.q_full.sub_offset(0, n * 2 * q_dim);
        let gate_view = scratch.gate.sub_offset(0, n * q_dim);
        weight_gemm_batched(
            gpu,
            &w.wq,
            &tmp_view,
            &q_full_view,
            n,
            rotated_x_scratch,
            false,
        )?;

        // Deinterleave per-head into Q + Gate. Output: [n × n_head × head_dim] each.
        gpu.deinterleave_f32_batched(
            &q_full_view,
            &q_view,
            &gate_view,
            cfg.n_head,
            cfg.head_dim,
            n,
        )?;

        // attn_q_norm: per-head normalization of Q.
        // rmsnorm_batched treats `batch` as "rows of length n", and gates the
        // single norm-weight broadcast across all rows. For per-head norm here,
        // each Q row is `n_head` heads of length `head_dim`, so we want
        // `batch = n * n_head`, `n = head_dim`. Same convention as the trunk's
        // per-head q_norm at qwen35.rs.
        gpu.rmsnorm_batched(
            &q_view,
            &w.attn_q_norm,
            &q_view,
            n * cfg.n_head,
            cfg.head_dim,
            cfg.rms_norm_eps,
        )?;
    }

    // K, V projections.
    let k_view = scratch.k.sub_offset(0, n * kv_dim);
    let v_view = scratch.v.sub_offset(0, n * kv_dim);
    weight_gemm_batched(
        gpu,
        &w.wk,
        &tmp_view,
        &k_view,
        n,
        rotated_x_scratch,
        fast_fill,
    )?;
    // fast_fill: wk and wv share input tmp_view; when both are plain MQ4G256
    // (no AWQ sidecar, same k) the rotation wk just wrote into rotated_x_scratch
    // is bit-identical to what wv would recompute, so reuse it.
    match rotated_x_scratch {
        Some(rot)
            if fast_fill
                && w.wk.gpu_dtype == DType::MQ4G256
                && w.wv.gpu_dtype == DType::MQ4G256
                && w.wk.awq_scale.is_none()
                && w.wv.awq_scale.is_none()
                && w.wk.k == w.wv.k =>
        {
            if mtp_fill_pm_gemm_enabled() && gpu.arch == "gfx1201" {
                hipfire_dispatch::families::gemm::run_mtp_fill_mq4(
                    gpu, &w.wv.buf, rot, &v_view, w.wv.m, w.wv.k, n, true,
                )?;
            } else {
                hipfire_dispatch::pipeline::batched::run_plain_gemm_key(
                    gpu,
                    hipfire_dispatch::types::KernelKey::GemmHfq4G256,
                    &w.wv.buf,
                    w.wv.gpu_dtype,
                    rot,
                    &v_view,
                    w.wv.m,
                    w.wv.k,
                    n,
                )?;
            }
        }
        _ => weight_gemm_batched(
            gpu,
            &w.wv,
            &tmp_view,
            &v_view,
            n,
            rotated_x_scratch,
            fast_fill,
        )?,
    }
    gpu.rmsnorm_batched(
        &k_view,
        &w.attn_k_norm,
        &k_view,
        n * cfg.n_head_kv,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;

    // ── 5. Batched RoPE on Q + K ─────────────────────────────────────────
    // fast_fill (kv_only): Q is never consumed (return follows the K/V cache
    // writes), so rope runs with zero Q heads over the same valid q_view.
    gpu.rope_partial_interleaved_f32_batched(
        &q_view,
        &k_view,
        &scratch.positions,
        if fast_fill { 0 } else { cfg.n_head },
        cfg.n_head_kv,
        cfg.head_dim,
        cfg.n_rot,
        cfg.rope_theta,
        n,
        // pos_offset=0: MTP head has its own non-compacted KV. Unchanged behavior.
        0,
    )?;

    // ── 6. v1 simplification: per-slot K/V writes + attention = V
    // ─────────────────────────────────────────────────────────────────────
    //
    // Each slot's attention reads its own KV slot only, so attn_out = V
    // (single-token softmax = 1). We still write K/V to make the KV cache
    // available for future cycles that wire historical reads.
    //
    gpu.kv_cache_write_q8_0_batched(
        &kv.inner.k_gpu[0],
        &k_view,
        &scratch.positions,
        cfg.n_head_kv,
        cfg.head_dim,
        n,
    )?;
    gpu.kv_cache_write_q8_0_batched(
        &kv.inner.v_gpu[0],
        &v_view,
        &scratch.positions,
        cfg.n_head_kv,
        cfg.head_dim,
        n,
    )?;
    if kv_only {
        return Ok(());
    }

    // Only the !fast_fill path reaches here (fast_fill implies kv_only).
    // Lean scratch never gets here: it is guarded to fast kv_only above.
    let gate_view = scratch.gate.sub_offset(0, n * q_dim);

    // attn_out = V (self-only attention with 1 key collapses softmax to 1).
    // V layout = [n × n_head_kv × head_dim]; attn_out layout = [n × n_head × head_dim].
    // For GQA (n_head > n_head_kv), repeat each KV head `n_head/n_head_kv` times.
    let attn_out_view = scratch.attn_out.sub_offset(0, n * q_dim);
    if cfg.n_head == cfg.n_head_kv {
        // No GQA expansion: attn_out := V row-by-row.
        gpu.hip
            .memcpy_dtod_at(&scratch.attn_out.buf, 0, &scratch.v.buf, 0, n * kv_dim * 4)?;
    } else {
        let ratio = cfg.n_head / cfg.n_head_kv;
        assert!(
            cfg.n_head % cfg.n_head_kv == 0,
            "n_head ({}) must be divisible by n_head_kv ({})",
            cfg.n_head,
            cfg.n_head_kv,
        );
        // Per-slot per-head expand: attn_out[i, h, d] = V[i, h/ratio, d].
        for i in 0..n {
            for h in 0..cfg.n_head {
                let kv_h = h / ratio;
                let src_off = (i * kv_dim + kv_h * cfg.head_dim) * 4;
                let dst_off = (i * q_dim + h * cfg.head_dim) * 4;
                gpu.hip.memcpy_dtod_at(
                    &scratch.attn_out.buf,
                    dst_off,
                    &scratch.v.buf,
                    src_off,
                    cfg.head_dim * 4,
                )?;
            }
        }
    }

    // ── 7. Apply gate (sigmoid * attn_out, in-place) ─────────────────────
    // sigmoid_mul_f32 is shape-agnostic (operates on flat buffer); pass the
    // full `n * q_dim` views.
    gpu.sigmoid_mul_f32(&attn_out_view, &gate_view)?;

    // ── 8. Output projection + residual ──────────────────────────────────
    let o_view = scratch.o.sub_offset(0, n * dim);
    weight_gemm_batched(
        gpu,
        &w.wo,
        &attn_out_view,
        &o_view,
        n,
        rotated_x_scratch,
        false,
    )?;
    let res_view = scratch.residual.sub_offset(0, n * dim);
    gpu.add_inplace_f32(&o_view, &res_view)?;

    // ── 9. POST-attn norm + SwiGLU FFN + residual ────────────────────────
    gpu.rmsnorm_batched(
        &o_view,
        &w.attn_post_norm,
        &tmp_view,
        n,
        dim,
        cfg.rms_norm_eps,
    )?;

    let gate_ffn_view = scratch.gate_ffn.sub_offset(0, n * cfg.n_ff);
    let up_view = scratch.up.sub_offset(0, n * cfg.n_ff);
    let ffn_hidden_view = scratch.ffn_hidden.sub_offset(0, n * cfg.n_ff);
    let ffn_out_view = scratch.ffn_out.sub_offset(0, n * dim);
    match &w.ffn {
        Qwen35MtpFfnWeights::Dense(ffn) => {
            weight_gemm_batched(
                gpu,
                &ffn.gate,
                &tmp_view,
                &gate_ffn_view,
                n,
                rotated_x_scratch,
                false,
            )?;
            weight_gemm_batched(
                gpu,
                &ffn.up,
                &tmp_view,
                &up_view,
                n,
                rotated_x_scratch,
                false,
            )?;
            gpu.silu_mul_f32(&gate_ffn_view, &up_view, &ffn_hidden_view)?;
            weight_gemm_batched(
                gpu,
                &ffn.down,
                &ffn_hidden_view,
                &ffn_out_view,
                n,
                rotated_x_scratch,
                false,
            )?;
            gpu.add_inplace_f32(&ffn_out_view, &o_view)?;
        }
        Qwen35MtpFfnWeights::Moe { .. } => {
            panic!("batched MTP head forward does not support MoE FFN yet; use serial MTP");
        }
    }

    // ── 10. Snapshot post-FFN hidden into t_mtp_outs ─────────────────────
    gpu.hip.memcpy_dtod_at(
        &scratch.t_mtp_outs.buf,
        0,
        &scratch.ffn_out.buf,
        0,
        n * dim_bytes,
    )?;

    Ok(())
}

#[cfg(test)]
mod sidecar_probe_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn mq4v2_hfq_also_probes_stem_mtp() {
        let c = mtp_sidecar_candidates(Path::new("/models/qwen3.5-4b.mq4v2.hfq"));
        assert!(
            c.iter().any(|p| p.ends_with("qwen3.5-4b.mq4v2.mtp")),
            "{c:?}"
        );
        assert!(c.iter().any(|p| p.ends_with("qwen3.5-4b.mtp")), "{c:?}");
    }

    #[test]
    fn mq4_probes_stem_mtp() {
        let c = mtp_sidecar_candidates(Path::new("/models/qwen3.5-4b.mq4"));
        assert!(c.iter().any(|p| p.ends_with("qwen3.5-4b.mtp")), "{c:?}");
    }

    #[test]
    fn find_mtp_sidecar_uses_stem_when_last_extension_missing() {
        let dir = std::env::temp_dir().join(format!("hipfire-mtp-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let trunk = dir.join("qwen3.5-4b.mq4v2.hfq");
        let stem = dir.join("qwen3.5-4b.mtp");
        std::fs::write(&trunk, b"trunk").expect("trunk");
        std::fs::write(&stem, b"mtp").expect("stem sidecar");
        let found = find_mtp_sidecar(&trunk).expect("stem sidecar exists");
        assert_eq!(found, stem);
        assert!(
            find_mtp_sidecar(&dir.join("missing.mq4v2.hfq")).is_none(),
            "absent sidecars must miss"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod prompt_fill_budget_tests {
    use super::*;

    /// Fixture-backed 27B MTP head dimensions (D=5120, Q=6144, KV=1024, F=17408).
    fn cfg_27b() -> Qwen35MtpHeadConfig {
        Qwen35MtpHeadConfig {
            n_embd: 5120,
            n_head: 24,
            n_head_kv: 4,
            head_dim: 256,
            n_ff: 17408,
            ffn_kind: Qwen35MtpFfnKind::Dense,
            num_experts: 0,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: false,
            vocab_size: 248320,
            rope_theta: 10_000_000.0,
            n_rot: 64,
            rms_norm_eps: 1e-6,
            max_seq: 16384,
            tie_word_embeddings: false,
        }
    }

    const LEGACY_K: usize = 17408; // max(2*dim, n_ff, dim, q_dim)
    const FILL_K: usize = 10240; // 2 * dim

    fn rows(legacy: usize, trunk_max: usize, prompt: usize) -> usize {
        Qwen35MtpHeadBatchedScratch::prompt_fill_rows_for_budget(
            &cfg_27b(),
            legacy,
            trunk_max,
            prompt,
            LEGACY_K,
            FILL_K,
        )
    }

    #[test]
    fn kv_and_scratch_bytes_match_shape_arithmetic() {
        let c = cfg_27b();
        // K and V, each 16384 * (4 heads * 256/32 blocks) * 34 B.
        assert_eq!(
            Qwen35MtpHeadKvCache::q8_requested_bytes(&c),
            2 * 16384 * 32 * 34
        );
        let tile = rdna_compute::attention::q8_flash_tile_size("gfx1201", 24, 4, 256, 16384);
        let flash = 24 * 16384usize.div_ceil(tile) * (2 + 256);
        // 7 dim rows + concat + q_full + 3 q_dim + 2 kv_dim + 3 n_ff + logits.
        let elems = 7 * 5120 + 2 * 5120 + 2 * 6144 + 3 * 6144 + 2 * 1024 + 3 * 17408 + 248320;
        assert_eq!(
            Qwen35MtpHeadScratch::requested_bytes("gfx1201", &c, None),
            (elems + flash) * 4 + 4
        );
        assert_eq!(
            Qwen35MtpHeadScratch::requested_bytes("gfx1201", &c, Some(32768)),
            (elems + flash + 32768) * 4 + 4
        );
        assert_eq!(
            Qwen35MtpHeadBatchedScratch::legacy_prompt_fill_bytes(&c, 512, 17408),
            512 * 565_252 + 512 * 17408 * 4
        );
    }

    #[test]
    fn byte_models_match_per_row_widths() {
        let c = cfg_27b();
        assert_eq!(
            Qwen35MtpHeadBatchedScratch::full_requested_bytes(&c, 512),
            512 * 565_252
        );
        assert_eq!(
            Qwen35MtpHeadBatchedScratch::lean_requested_bytes(&c, 512)
                + Qwen35MtpHeadBatchedScratch::rotation_requested_bytes(512, FILL_K),
            512 * 192_516
        );
    }

    #[test]
    fn default_caps_are_1688_and_1264() {
        assert_eq!(rows(512, 8192, 16_059), 1688);
        assert_eq!(rows(384, 8192, 16_059), 1264);
    }

    #[test]
    fn cap_is_clamped_by_trunk_chunk_and_prompt() {
        assert_eq!(rows(512, 1000, 16_059), 1000);
        assert_eq!(rows(512, 8192, 1500), 1500);
        assert_eq!(rows(384, 8192, 1264), 1264);
        assert_eq!(rows(384, 8192, 1265), 1264);
    }

    #[test]
    fn short_prompts_keep_their_own_length() {
        // Alignment applies to the budget capacity before the prompt clamp.
        assert_eq!(rows(512, 8192, 0), 0);
        assert_eq!(rows(512, 8192, 1), 1);
        assert_eq!(rows(512, 8192, 2), 2);
        assert_eq!(rows(512, 8192, 9), 9);
        assert_eq!(rows(384, 8192, 9), 9);
    }

    #[test]
    fn requested_bytes_never_exceed_legacy_budget() {
        let c = cfg_27b();
        for legacy in [1usize, 7, 8, 64, 384, 512, 1024] {
            let budget = Qwen35MtpHeadBatchedScratch::full_requested_bytes(&c, legacy)
                + Qwen35MtpHeadBatchedScratch::rotation_requested_bytes(legacy, LEGACY_K);
            for prompt in [1usize, 2, 9, 383, 1264, 1688, 4096, 16_059] {
                let r = rows(legacy, 8192, prompt);
                assert!(
                    r >= 1 && r <= prompt.min(8192),
                    "legacy={legacy} prompt={prompt} r={r}"
                );
                let used = Qwen35MtpHeadBatchedScratch::lean_requested_bytes(&c, r)
                    + Qwen35MtpHeadBatchedScratch::rotation_requested_bytes(r, FILL_K);
                assert!(
                    used <= budget,
                    "legacy={legacy} prompt={prompt} rows={r}: {used} > {budget}"
                );
            }
        }
    }
}
