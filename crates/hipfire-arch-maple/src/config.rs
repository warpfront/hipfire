// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Maple-Preview config, parsed from the HFQ `metadata_json` envelope (which
//! carries the source `config.json` wholesale under the `config` key).
//!
//! Ground truth: `deepgrove/maple-preview` `config.json` +
//! `modeling_maple.py` (transformers 4.57.1). 24 layers, hidden 2048,
//! GQA 16 q / 4 kv, head_dim 128, 256 experts top-8 on EVERY layer
//! (`moe_intermediate_size` 512, no shared experts, no expert bias),
//! vocab 151936, untied lm_head, rope_theta 10000, sliding_window 512.
//!
//! Four things that are easy to get silently wrong:
//!   1. **`intermediate_size` (4096) is DEAD.** Every layer is MoE; the expert
//!      FFN width is `moe_intermediate_size` (512). Wiring the dense value
//!      builds the wrong shapes.
//!   2. **NoPE on the global layers.** `nope_on_global_attention` — RoPE runs
//!      only where `layer_types[l] == sliding_attention`. In the reference this
//!      is expressed as `self.sliding_window is not None` gating
//!      `apply_rotary_pos_emb`; getting it backwards is a silent quality bug.
//!   3. **Partial rotary.** `partial_rotary_factor` 0.5 ⇒ only the first 64 of
//!      128 head dims rotate, the rest pass through untouched.
//!   4. **QK-norm runs BEFORE RoPE** (`modeling_maple.py:300-303`), and the
//!      norms are per-head over `head_dim`, not over hidden.
//!
//! `quantize: true` and `preaffine` are DEAD KEYS — no quantization code exists
//! in `modeling_maple.py`. They are deliberately not parsed, so nothing can
//! branch on them.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::ModelSource;
use serde::Deserialize;

/// Per-layer attention kind, decoded from `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapleLayerType {
    /// Global attention over the full context. **NoPE** — no rotary applied.
    Full,
    /// Local sliding-window attention (window = `sliding_window` = 512),
    /// **with** partial RoPE.
    Sliding,
}

/// The SwiGLU clamp is hardcoded in `modeling_maple.py`, not a config key:
/// `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)`.
pub const MAPLE_SWIGLU_CLAMP: f32 = 7.0;

/// Typed Maple-Preview shape constants.
#[derive(Clone, Debug)]
pub struct MapleConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Explicit per-head dim (128). NOT hidden/n_heads — 16*128 = 2048 happens
    /// to match here, but the checkpoint states it and we honour the statement.
    pub head_dim: usize,
    /// Routed-expert FFN width (`moe_intermediate_size` = 512). The top-level
    /// `intermediate_size` (4096) is unused — every layer is MoE.
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// 0 for Maple — there are no shared experts.
    pub num_shared_experts: usize,
    /// Renormalize the gathered top-k router weights to sum 1 (`norm_topk_prob`
    /// = true): softmax over ALL experts, take top-8, THEN renormalise.
    pub norm_topk_prob: bool,
    pub rope_theta: f32,
    /// Fraction of `head_dim` that rotates (0.5 ⇒ 64 of 128).
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// Local-attention window for `sliding_attention` layers (512).
    pub sliding_window: usize,
    /// false for Maple — `lm_head.weight` is a distinct tensor.
    pub tie_word_embeddings: bool,
    /// Per-layer attention kind (length == num_hidden_layers).
    pub layer_types: Vec<MapleLayerType>,
    /// The SwiGLU clamp bound. Not a config key; carried here so the forward
    /// path reads a named constant rather than a bare 7.0.
    pub swiglu_clamp: f32,
}

#[derive(Deserialize)]
struct RawMapleConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    moe_intermediate_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default)]
    num_shared_experts: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f32,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    /// "full_attention" | "sliding_attention" per layer.
    layer_types: Vec<String>,
}

fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_partial_rotary() -> f32 {
    0.5
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_max_pos() -> usize {
    131_072
}
fn default_sliding_window() -> usize {
    512
}

impl MapleConfig {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// Parse from a `metadata_json` string (HFQ metadata OR a SafetensorsSource's
    /// config.json, both of which embed the HF config under the `config` key).
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("maple: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "maple: metadata_json missing `config` wrapper".to_string())?;
        Self::from_config_value(inner)
    }

    pub fn from_safetensors(source: &dyn ModelSource) -> Result<Self, String> {
        Self::from_metadata_json(source.metadata_json())
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob).
    pub fn from_config_value(inner: &serde_json::Value) -> Result<Self, String> {
        let raw: RawMapleConfig = serde_json::from_value(inner.clone())
            .map_err(|e| format!("maple: parsing config failed: {e}"))?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(format!(
                "maple: layer_types len {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            ));
        }
        let layer_types = raw
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "full" | "global" => Ok(MapleLayerType::Full),
                "sliding_attention" | "sliding" | "swa" => Ok(MapleLayerType::Sliding),
                other => Err(format!("maple: unknown layer_type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MapleConfig {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            moe_intermediate_size: raw.moe_intermediate_size,
            num_experts: raw.num_experts,
            num_experts_per_tok: raw.num_experts_per_tok,
            num_shared_experts: raw.num_shared_experts,
            norm_topk_prob: raw.norm_topk_prob,
            rope_theta: raw.rope_theta,
            partial_rotary_factor: raw.partial_rotary_factor,
            rms_norm_eps: raw.rms_norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            sliding_window: raw.sliding_window,
            tie_word_embeddings: raw.tie_word_embeddings,
            layer_types,
            swiglu_clamp: MAPLE_SWIGLU_CLAMP,
        })
    }

    /// q projection output width (n_heads * head_dim).
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    /// k/v projection output width (n_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn layer_type(&self, layer: usize) -> MapleLayerType {
        self.layer_types[layer]
    }
    /// RoPE is applied on sliding layers ONLY (`nope_on_global_attention`).
    pub fn applies_rope(&self, layer: usize) -> bool {
        self.layer_types[layer] == MapleLayerType::Sliding
    }
    /// Number of head dims that rotate: `head_dim * partial_rotary_factor`
    /// (64 of 128). The remaining dims pass through unrotated.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }
    /// The key window a query at `pos` may attend to, as a half-open range.
    /// Full layers see everything; sliding layers see the last
    /// `sliding_window` keys inclusive of `pos`.
    pub fn attention_span(&self, layer: usize, pos: usize) -> std::ops::Range<usize> {
        match self.layer_types[layer] {
            MapleLayerType::Full => 0..pos + 1,
            MapleLayerType::Sliding => pos.saturating_sub(self.sliding_window - 1)..pos + 1,
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    /// The published config.json, verbatim (deepgrove/maple-preview,
    /// transformers 4.57.1). Wrapped in the `config` envelope the runtime uses.
    const PUBLISHED_CONFIG_JSON: &str = r#"{"config":{
  "architectures": ["MapleForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "embedding_dropout": 0.0,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_types": [
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention"
  ],
  "max_position_embeddings": 131072,
  "max_window_layers": 24,
  "moe_intermediate_size": 512,
  "moe_router_enable_expert_bias": false,
  "nope_on_global_attention": true,
  "norm_topk_prob": true,
  "num_attention_heads": 16,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 24,
  "num_key_value_heads": 4,
  "num_shared_experts": 0,
  "output_dropout": 0.0,
  "output_router_logits": false,
  "pad_token_id": null,
  "partial_rotary_factor": 0.5,
  "preaffine": false,
  "quantize": true,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000,
  "router_dtype": "fp32",
  "sliding_window": 512,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "use_qk_norm": true,
  "use_rmsnorm": true,
  "vocab_size": 151936
}}"#;

    fn cfg() -> MapleConfig {
        MapleConfig::from_metadata_json(PUBLISHED_CONFIG_JSON).expect("published config parses")
    }

    #[test]
    fn parses_published_maple_config() {
        let c = cfg();
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 4);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 512);
        assert_eq!(c.sliding_window, 512);
        assert_eq!(c.vocab_size, 151936);
        assert_eq!(c.partial_rotary_factor, 0.5);
        assert_eq!(c.rope_theta, 10_000.0);
        assert!(!c.tie_word_embeddings, "lm_head is a separate tensor");
        assert!(c.norm_topk_prob);
        assert_eq!(c.num_shared_experts, 0);
        assert_eq!(c.swiglu_clamp, 7.0);
    }

    #[test]
    fn layer_types_are_three_sliding_then_one_full() {
        let c = cfg();
        assert_eq!(c.layer_types.len(), 24);
        for (i, lt) in c.layer_types.iter().enumerate() {
            let expect = if i % 4 == 3 {
                MapleLayerType::Full
            } else {
                MapleLayerType::Sliding
            };
            assert_eq!(*lt, expect, "layer {i}");
        }
    }

    #[test]
    fn full_layers_are_nope_and_sliding_layers_rope() {
        // `nope_on_global_attention`: RoPE is applied ONLY on sliding layers.
        // Getting this backwards is a silent quality bug, not a crash.
        let c = cfg();
        assert!(!c.applies_rope(3), "layer 3 is full ⇒ NoPE");
        assert!(c.applies_rope(0), "layer 0 is sliding ⇒ RoPE");
        assert_eq!(c.rotary_dim(), 64, "partial_rotary 0.5 of head_dim 128");
    }

    #[test]
    fn sliding_layers_see_at_most_the_window_and_full_layers_see_everything() {
        let c = cfg();
        let sliding = c.attention_span(0, 2000);
        assert_eq!(sliding.start, 2000 - 512 + 1);
        assert_eq!(sliding.end, 2001);
        assert_eq!(sliding.len(), 512);

        let full = c.attention_span(3, 2000);
        assert_eq!(full.start, 0);
        assert_eq!(full.end, 2001);
    }

    #[test]
    fn a_short_context_does_not_underflow_the_sliding_window() {
        // pos < window: the span must clamp at 0, not wrap around usize.
        let c = cfg();
        assert_eq!(c.attention_span(0, 3), 0..4);
    }

    #[test]
    fn expert_ffn_width_is_moe_intermediate_not_intermediate_size() {
        // `intermediate_size` (4096) is a dead key — every layer is MoE and the
        // experts are 512 wide. Reading the wrong one builds 8x-too-big FFNs.
        let c = cfg();
        assert_eq!(c.moe_intermediate_size, 512);
    }

    #[test]
    fn layer_types_length_mismatch_is_rejected() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "moe_intermediate_size": 32, "num_experts": 8, "num_experts_per_tok": 2,
            "layer_types": ["sliding_attention","full_attention"]
        }}"#;
        let err = MapleConfig::from_metadata_json(json).unwrap_err();
        assert!(err.contains("layer_types len 2"), "got: {err}");
    }

    #[test]
    fn missing_config_wrapper_errs() {
        assert!(MapleConfig::from_metadata_json(r#"{"hidden_size":2048}"#).is_err());
    }

    #[test]
    fn unknown_layer_type_fails_closed() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "moe_intermediate_size": 32, "num_experts": 8, "num_experts_per_tok": 2,
            "layer_types": ["linear_attention"]
        }}"#;
        let err = MapleConfig::from_metadata_json(json).unwrap_err();
        assert!(err.contains("unknown layer_type"), "got: {err}");
    }
}
