// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 config, parsed from the HFQ `metadata_json` envelope (which carries
//! the source `config.json` wholesale under the `config` key) or directly
//! from a safetensors `ModelSource`.
//!
//! Ground truth: LiquidAI/LFM3-1.7B config.json + transformers
//! `Lfm3Config`/`modeling_lfm3.py` (read 2026-10-31):
//!   hidden 2048, 24 layers, 16 q-heads / 4 kv-heads, head_dim 128,
//!   layer_types interleave conv / full_attention / sliding_attention
//!   (full every 4th from layer 2, sliding between, conv elsewhere),
//!   conv_L_cache 7 (depthwise causal short-conv), sliding_window 256,
//!   use_sandwich_norm true, dense SwiGLU FFN (intermediate 7680, no MoE),
//!   rope_parameters: full_attention {theta 1e6, partial_rotary 0.5},
//!   sliding_attention {theta 1e4, full rotary}, eps 1e-5, tied embeddings.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::model_source::ModelSource;
use serde::Deserialize;

/// Per-layer mixer kind, decoded from `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    /// Double-gated LIV short-convolution (depthwise causal, kernel = conv_L_cache).
    Conv,
    /// GQA attention with per-head QK-norm + partial-rotary RoPE (theta 1e6).
    FullAttention,
    /// GQA attention with full-dim RoPE (theta 1e4) + sliding window.
    SlidingAttention,
}

impl MixerKind {
    /// True for both attention variants (they share KV-cache slots).
    pub fn is_attention(self) -> bool {
        matches!(self, MixerKind::FullAttention | MixerKind::SlidingAttention)
    }
}

/// Per-attention-type RoPE parameters (LFM3 keys RoPE by layer type, like Gemma 3).
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    pub theta: f32,
    /// Fraction of head_dim rotated. Defaults to 1.0 (full rotary) when absent
    /// (sliding_attention has no `partial_rotary_factor` in the config).
    pub partial_rotary_factor: f32,
}

impl RopeParams {
    /// n_rot = partial_rotary_factor * head_dim (dims actually rotated).
    pub fn n_rot(&self, head_dim: usize) -> usize {
        (self.partial_rotary_factor * head_dim as f32) as usize
    }
}

/// Typed LFM3 shape constants.
#[derive(Clone, Debug)]
pub struct Lfm3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Short-conv kernel size (HF `conv_L_cache`); decode conv-state holds K-1.
    pub conv_kernel_size: usize,
    /// Dense SwiGLU FFN intermediate size (HF `intermediate_size`).
    pub intermediate_size: usize,
    /// RoPE for `full_attention` layers (theta 1e6, partial 0.5).
    pub rope_full: RopeParams,
    /// RoPE for `sliding_attention` layers (theta 1e4, full rotary).
    pub rope_sliding: RopeParams,
    /// Sliding-window size for `sliding_attention` layers (HF `sliding_window`).
    pub sliding_window: usize,
    /// RMSNorm the mixer/FFN outputs before the residual stream.
    pub use_sandwich_norm: bool,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    /// lm_head shares embed_tokens.
    pub tie_word_embeddings: bool,
    /// Per-layer mixer choice (length == num_hidden_layers).
    pub layer_types: Vec<MixerKind>,
}

#[derive(Deserialize)]
struct RawRope {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_full_rotary")]
    partial_rotary_factor: f32,
}

#[derive(Deserialize)]
struct RawLfm3Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_conv_l")]
    conv_L_cache: usize,
    intermediate_size: usize,
    #[serde(default)]
    rope_parameters: Option<std::collections::HashMap<String, RawRope>>,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    #[serde(default = "default_true")]
    use_sandwich_norm: bool,
    #[serde(default = "default_eps")]
    norm_eps: f32,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    layer_types: Vec<String>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}
fn default_full_rotary() -> f32 {
    1.0
}
fn default_conv_l() -> usize {
    7
}
fn default_sliding_window() -> usize {
    256
}
fn default_eps() -> f32 {
    1e-5
}
fn default_max_pos() -> usize {
    128_000
}
fn default_true() -> bool {
    true
}

impl Lfm3Config {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let wrapper: serde_json::Value = serde_json::from_str(&hfq.metadata_json)
            .map_err(|e| format!("lfm3: metadata_json not valid JSON: {e}"))?;
        let inner = wrapper
            .get("config")
            .ok_or_else(|| "lfm3: metadata_json missing `config` wrapper".to_string())?;
        // LFM3-VL nests the text backbone config under `text_config`; the
        // text-only LFM3-1.7B has the fields at the top level of `config`.
        let text = inner.get("text_config").unwrap_or(inner);
        Self::from_config_value(text)
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob, or a
    /// bare config.json for the text-only `LiquidAI/LFM3-1.7B` dir source).
    pub fn from_config_value(inner: &serde_json::Value) -> Result<Self, String> {
        let raw: RawLfm3Config = serde_json::from_value(inner.clone())
            .map_err(|e| format!("lfm3: parsing config failed: {e}"))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawLfm3Config) -> Result<Self, String> {
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(format!(
                "lfm3: layer_types len {} != num_hidden_layers {}",
                raw.layer_types.len(),
                raw.num_hidden_layers
            ));
        }
        let layer_types = raw
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" | "attention" | "attn" => Ok(MixerKind::FullAttention),
                "sliding_attention" | "sliding" | "swa" => Ok(MixerKind::SlidingAttention),
                "conv" | "short_conv" | "conv1d" => Ok(MixerKind::Conv),
                other => Err(format!("lfm3: unknown layer_type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Per-type RoPE. LFM3 keys rope_parameters by attention layer type
        // (full_attention / sliding_attention). Defaults match Lfm3Config's
        // __post_init__ when the config omits them.
        let rope_full = raw
            .rope_parameters
            .as_ref()
            .and_then(|m| m.get("full_attention"))
            .map(|r| RopeParams {
                theta: r.rope_theta,
                partial_rotary_factor: r.partial_rotary_factor,
            })
            .unwrap_or(RopeParams {
                theta: 1_000_000.0,
                partial_rotary_factor: 0.5,
            });
        let rope_sliding = raw
            .rope_parameters
            .as_ref()
            .and_then(|m| m.get("sliding_attention"))
            .map(|r| RopeParams {
                theta: r.rope_theta,
                partial_rotary_factor: r.partial_rotary_factor,
            })
            .unwrap_or(RopeParams {
                theta: 10_000.0,
                partial_rotary_factor: 1.0,
            });

        // A config with sliding_attention layers MUST declare a window.
        if layer_types.iter().any(|t| *t == MixerKind::SlidingAttention)
            && raw.sliding_window == 0
        {
            return Err(
                "lfm3: layer_types has sliding_attention but sliding_window is 0/None".to_string(),
            );
        }

        Ok(Lfm3Config {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim,
            conv_kernel_size: raw.conv_L_cache,
            intermediate_size: raw.intermediate_size,
            rope_full,
            rope_sliding,
            sliding_window: raw.sliding_window,
            use_sandwich_norm: raw.use_sandwich_norm,
            rms_norm_eps: raw.norm_eps,
            max_position_embeddings: raw.max_position_embeddings,
            tie_word_embeddings: raw.tie_word_embeddings,
            layer_types,
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
    pub fn mixer(&self, layer: usize) -> MixerKind {
        self.layer_types[layer]
    }
    /// RoPE params for a given attention layer's type.
    pub fn rope_for(&self, layer: usize) -> RopeParams {
        match self.mixer(layer) {
            MixerKind::FullAttention => self.rope_full,
            MixerKind::SlidingAttention => self.rope_sliding,
            MixerKind::Conv => self.rope_full, // unused (conv has no RoPE)
        }
    }
    /// Number of attention layers (== KV-cache slots needed). Both full and
    /// sliding attention layers carry KV.
    pub fn num_attention_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.is_attention()).count()
    }
    /// Number of conv layers (== conv-state cache slots needed).
    pub fn num_conv_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == MixerKind::Conv)
            .count()
    }
}

/// Parse Lfm3Config from a `&dyn ModelSource` (safetensors or HFQ wrapper).
/// The metadata JSON should contain the `{"architecture":..., "config":{...}}`
/// envelope as produced by `SafetensorsSource::build_metadata_json`, OR a bare
/// `config.json` object (the text-only `LiquidAI/LFM3-1.7B` dir layout).
pub fn config_from_source(source: &dyn ModelSource) -> Option<Lfm3Config> {
    let meta: serde_json::Value = serde_json::from_str(source.metadata_json()).ok()?;
    // Envelope form: { "config": { ... } }.
    if let Some(inner) = meta.get("config") {
        // LFM3-VL nests the text backbone config under `text_config`.
        let text = inner.get("text_config").unwrap_or(inner);
        return Lfm3Config::from_config_value(text).ok();
    }
    // Bare config.json form (no wrapper): the metadata_json IS the config.
    Lfm3Config::from_config_value(&meta).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm3_1_7b_config_parses() {
        // Mirrors LiquidAI/LFM3-1.7B config.json (text-only).
        let cfg = serde_json::json!({
            "model_type": "lfm3",
            "vocab_size": 128000,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "conv_L_cache": 7,
            "intermediate_size": 7680,
            "sliding_window": 256,
            "use_sandwich_norm": true,
            "norm_eps": 1e-5,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "layer_types": ["conv","conv","full_attention","conv","sliding_attention",
                            "conv","full_attention","conv","sliding_attention","conv",
                            "full_attention","conv","sliding_attention","conv","full_attention",
                            "conv","sliding_attention","conv","full_attention","conv",
                            "sliding_attention","conv","full_attention","conv"]
        });
        let c = Lfm3Config::from_config_value(&cfg).unwrap();
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.conv_kernel_size, 7);
        assert_eq!(c.sliding_window, 256);
        assert!(c.use_sandwich_norm);
        assert_eq!(c.rope_full.theta, 1_000_000.0);
        assert_eq!(c.rope_full.n_rot(128), 64);
        assert_eq!(c.rope_sliding.theta, 10_000.0);
        assert_eq!(c.rope_sliding.n_rot(128), 128); // full rotary
        assert_eq!(c.num_attention_layers(), 11); // 6 full + 5 sliding
        assert_eq!(c.num_conv_layers(), 13);
        assert_eq!(c.mixer(0), MixerKind::Conv);
        assert_eq!(c.mixer(2), MixerKind::FullAttention);
        assert_eq!(c.mixer(4), MixerKind::SlidingAttention);
    }

    #[test]
    fn rejects_sliding_without_window() {
        let cfg = serde_json::json!({
            "model_type": "lfm3", "vocab_size": 2, "hidden_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
            "intermediate_size": 16, "sliding_window": 0,
            "layer_types": ["sliding_attention"]
        });
        let err = Lfm3Config::from_config_value(&cfg).unwrap_err();
        assert!(err.contains("sliding_window"));
    }
}
