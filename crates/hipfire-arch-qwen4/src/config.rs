// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Qwen4 (Qwen3.8-Flash-Next) configuration parsing and admission.
//!
//! The offline source/quantizer parser accepts two JSON envelopes: an HF
//! metadata envelope (`{"config": ...}`) and a raw Transformers `config.json`.
//! Runtime serving admission uses the strict nested conditional-generation
//! parser, which additionally requires the outer model type and canonical
//! architecture entry before allocation.

use serde_json::Value;
use std::fmt;

/// Primary HFQ architecture id reserved for Qwen4.
pub const ARCH_ID: u32 = 16;
/// Stable source model type used by the outer Qwen4 conditional config.
pub const MODEL_TYPE: &str = "qwen4_exp";
/// Stable source model type used by the text config.
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";
/// The only architecture entry admitted by this crate.
pub const ARCHITECTURE_NAME: &str = "Qwen4ExpForConditionalGeneration";

/// Decoder layer kind in the Qwen4 hybrid trunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Gated DeltaNet / linear-attention layer.
    LinearAttention,
    /// Qwen Sparse Attention layer (the full-attention branch).
    FullAttention,
}

impl LayerType {
    /// Parse the exact Transformers spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "linear_attention" => Some(Self::LinearAttention),
            "full_attention" => Some(Self::FullAttention),
            _ => None,
        }
    }

    /// Return the exact Transformers spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearAttention => "linear_attention",
            Self::FullAttention => "full_attention",
        }
    }
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dtype of the recurrent Gated DeltaNet state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecurrentStateDType {
    F32,
}

impl RecurrentStateDType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "float32" | "f32" => Some(Self::F32),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
        }
    }
}

/// Source/default weight dtype named by the Qwen4 checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceDType {
    BF16,
}

impl SourceDType {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "bfloat16" | "bf16" => Some(Self::BF16),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BF16 => "bfloat16",
        }
    }
}

/// Nested native MTP shape.  This is intentionally data-only: generation and
/// acceptance remain owned by the runtime engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4MtpConfig {
    /// Number of decoder layers in the MTP module.
    pub num_hidden_layers: usize,
    /// MTP's per-layer attention kinds.
    pub layer_types: Vec<LayerType>,
    /// Whether the MTP module uses its hybrid implementation path.
    pub hybrid: bool,
    /// MTP RoPE base, retained for the later typed executor.
    pub rope_theta: f64,
}

impl Qwen4MtpConfig {
    /// Validate the native one-full-attention-layer MTP contract.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_hidden_layers != 1 {
            return Err(format!(
                "qwen4: MTP num_hidden_layers={} (expected 1)",
                self.num_hidden_layers
            ));
        }
        if self.layer_types != [LayerType::FullAttention] {
            return Err(format!(
                "qwen4: MTP layer_types={:?} (expected [full_attention])",
                self.layer_types
            ));
        }
        if !self.hybrid {
            return Err("qwen4: MTP hybrid must be true".into());
        }
        if !approximately_equal(self.rope_theta, 10_000_000.0) {
            return Err(format!(
                "qwen4: MTP rope_theta={} (expected 10000000)",
                self.rope_theta
            ));
        }
        Ok(())
    }
}

/// Fully typed Qwen4 text configuration.
///
/// The fields retain the checkpoint's names where doing so avoids ambiguity.
/// `validate` is strict by design: a parseable but different Qwen-family
/// configuration must not reach a Qwen4 allocator or executor.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4Config {
    /// Canonical architecture id for the future registry entry.
    pub architecture_id: u32,
    /// Outer source model type (`qwen4_exp`).
    pub model_type: String,
    /// Inner text model type (`qwen4_exp_text`).
    pub text_model_type: String,
    /// Source dtype (`bfloat16` in the pinned checkpoint).
    pub dtype: SourceDType,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub full_attention_interval: usize,

    // Gated DeltaNet dimensions.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub recurrent_state_dtype: RecurrentStateDType,

    // QSA indexer dimensions.
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,

    // Routed and shared experts.
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub output_gate_type: String,

    // Hyper-connections.
    pub hc_count: usize,
    pub hc_lowrank: usize,

    // Per-layer n-gram embedding (PLE).
    pub ple_layer_ids: Vec<usize>,
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    pub split_ngram_parts: usize,
    pub heads_per_ngram: usize,
    pub ngram_size: usize,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: usize,

    // Token/generation metadata used by PLE history and MTP.
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub mtp: Qwen4MtpConfig,
}

impl Qwen4Config {
    /// Parse an offline source/quantizer JSON input.
    ///
    /// This deliberately accepts either the HF metadata envelope or a raw
    /// Transformers `config.json`; runtime serving admission must use the
    /// strict [`Self::from_metadata_json`] contract instead.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("qwen4: configuration is not valid JSON: {e}"))?;
        Self::from_value(&value)
    }

    /// Parse the strict nested conditional-generation envelope used by a
    /// runtime serving artifact.
    pub fn from_metadata_json(json: &str) -> Result<Self, String> {
        Self::from_nested_metadata_json(json)
    }

    /// Parse only the nested conditional-generation envelope used by a
    /// Qwen4 serving artifact.
    ///
    /// `from_value` is the explicit source/quantizer parser and remains
    /// compatible with raw text `config.json` inputs. Runtime admission is
    /// stricter: a source must identify the outer conditional-generation
    /// model, its nested text model, and the one canonical architecture entry.
    /// This keeps an arbitrary text-shaped config from being classified as the
    /// reserved arch 16 route.
    pub fn from_nested_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("qwen4: configuration is not valid JSON: {e}"))?;
        Self::from_nested_value(&value)
    }

    /// Strict runtime classifier for the nested text+MTP Qwen4 envelope.
    pub fn from_nested_value(value: &Value) -> Result<Self, String> {
        let config = value.get("config").unwrap_or(value);
        let object = config
            .as_object()
            .ok_or_else(|| "qwen4: nested `config` must be a JSON object".to_string())?;
        if object.get("model_type").and_then(Value::as_str) != Some(MODEL_TYPE) {
            return Err(format!(
                "qwen4: nested runtime admission requires outer model_type={MODEL_TYPE:?}"
            ));
        }
        let text = object
            .get("text_config")
            .ok_or_else(|| "qwen4: nested runtime admission requires `text_config`".to_string())?
            .as_object()
            .ok_or_else(|| "qwen4: `text_config` must be a JSON object".to_string())?;
        if text.get("model_type").and_then(Value::as_str) != Some(TEXT_MODEL_TYPE) {
            return Err(format!(
                "qwen4: nested runtime admission requires text_config.model_type={TEXT_MODEL_TYPE:?}"
            ));
        }
        let architectures = object
            .get("architectures")
            .ok_or_else(|| "qwen4: nested runtime admission requires `architectures`".to_string())?
            .as_array()
            .ok_or_else(|| "qwen4: `architectures` must be an array".to_string())?;
        if architectures.len() != 1
            || architectures.first().and_then(Value::as_str) != Some(ARCHITECTURE_NAME)
        {
            return Err(format!(
                "qwen4: nested runtime admission requires architectures=[{ARCHITECTURE_NAME:?}]"
            ));
        }
        Self::from_value(value)
    }

    /// Parse the strict nested runtime envelope from HFQ metadata.
    pub fn from_nested_metadata_json(json: &str) -> Result<Self, String> {
        Self::from_nested_json(json)
    }

    /// Parse from an already decoded metadata/config value.
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let config = match value.get("config") {
            Some(config) => config,
            None => value,
        };
        if !config.is_object() {
            return Err("qwen4: `config` must be a JSON object".into());
        }

        let text = match config.get("text_config") {
            Some(text) if text.is_object() => text,
            Some(_) => return Err("qwen4: `text_config` must be a JSON object".into()),
            None => config,
        };

        let outer_model_type = config.get("model_type").and_then(Value::as_str);
        if let Some(model_type) = outer_model_type {
            if model_type != MODEL_TYPE && model_type != TEXT_MODEL_TYPE {
                return Err(format!(
                    "qwen4: unsupported outer model_type={model_type:?}"
                ));
            }
            if config.get("text_config").is_some() && model_type != MODEL_TYPE {
                return Err(format!(
                    "qwen4: nested config requires outer model_type={MODEL_TYPE:?}, got {model_type:?}"
                ));
            }
        }
        let text_model_type = required_string(text, "model_type")?;
        if text_model_type != TEXT_MODEL_TYPE {
            return Err(format!(
                "qwen4: expected text_config.model_type={TEXT_MODEL_TYPE:?}, got {text_model_type:?}"
            ));
        }
        if let Some(architectures) = config.get("architectures") {
            let values = architectures
                .as_array()
                .ok_or_else(|| "qwen4: `architectures` must be an array".to_string())?;
            if !values
                .iter()
                .filter_map(Value::as_str)
                .any(|name| name == ARCHITECTURE_NAME)
            {
                return Err(format!(
                    "qwen4: architectures does not contain {ARCHITECTURE_NAME:?}"
                ));
            }
        }

        // A raw text config has the inner model type at its root; retain the
        // canonical outer family id in the typed config.
        let model_type = MODEL_TYPE.to_string();
        let dtype = SourceDType::parse(
            &optional_string(text, "dtype").unwrap_or_else(|| "bfloat16".into()),
        )
        .ok_or_else(|| "qwen4: dtype must be bfloat16".to_string())?;
        let rope_parameters = text.get("rope_parameters");
        let rope_theta = rope_parameters
            .and_then(|v| v.get("rope_theta"))
            .and_then(Value::as_f64)
            .or_else(|| text.get("rope_theta").and_then(Value::as_f64))
            .unwrap_or(10_000_000.0);
        let partial_rotary_factor = text
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .or_else(|| {
                rope_parameters
                    .and_then(|v| v.get("partial_rotary_factor"))
                    .and_then(Value::as_f64)
            })
            .unwrap_or(0.25);

        let layer_types = parse_layers(required_value(text, "layer_types")?, "layer_types")?;
        let mtp_value = required_value(text, "mtp")?;
        if !mtp_value.is_object() {
            return Err("qwen4: `text_config.mtp` must be an object".into());
        }
        let mtp = Qwen4MtpConfig {
            num_hidden_layers: required_usize(mtp_value, "num_hidden_layers")?,
            layer_types: parse_layers(
                required_value(mtp_value, "layer_types")?,
                "mtp.layer_types",
            )?,
            hybrid: required_bool(mtp_value, "hybrid")?,
            rope_theta: required_f64(mtp_value, "rope_theta")?,
        };

        let cfg = Self {
            architecture_id: ARCH_ID,
            model_type,
            text_model_type: text_model_type.to_string(),
            dtype,
            hidden_size: required_usize(text, "hidden_size")?,
            vocab_size: required_usize(text, "vocab_size")?,
            num_hidden_layers: required_usize(text, "num_hidden_layers")?,
            max_position_embeddings: required_usize(text, "max_position_embeddings")?,
            num_attention_heads: required_usize(text, "num_attention_heads")?,
            num_key_value_heads: required_usize(text, "num_key_value_heads")?,
            head_dim: required_usize(text, "head_dim")?,
            partial_rotary_factor,
            rope_theta,
            attention_bias: optional_bool(text, "attention_bias").unwrap_or(false),
            layer_types,
            full_attention_interval: required_usize(text, "full_attention_interval")?,
            linear_num_key_heads: required_usize(text, "linear_num_key_heads")?,
            linear_num_value_heads: required_usize(text, "linear_num_value_heads")?,
            linear_key_head_dim: required_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: required_usize(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: required_usize(text, "linear_conv_kernel_dim")?,
            recurrent_state_dtype: RecurrentStateDType::parse(&required_string(
                text,
                "mamba_ssm_dtype",
            )?)
            .ok_or_else(|| "qwen4: mamba_ssm_dtype must be float32".to_string())?,
            indexer_n_heads: required_usize(text, "indexer_n_heads")?,
            indexer_kv_heads: required_usize(text, "indexer_kv_heads")?,
            indexer_head_dim: required_usize(text, "indexer_head_dim")?,
            indexer_budget: required_usize(text, "indexer_budget")?,
            indexer_compress_ratio: required_usize(text, "indexer_compress_ratio")?,
            num_experts: required_usize(text, "num_experts")?,
            num_experts_per_tok: required_usize(text, "num_experts_per_tok")?,
            moe_intermediate_size: required_usize(text, "moe_intermediate_size")?,
            shared_expert_intermediate_size: required_usize(
                text,
                "shared_expert_intermediate_size",
            )?,
            norm_topk_prob: optional_bool(text, "norm_topk_prob").unwrap_or(true),
            output_gate_type: optional_string(text, "output_gate_type")
                .unwrap_or_else(|| "sigmoid".into()),
            hc_count: required_usize(text, "hc_count")?,
            hc_lowrank: required_usize(text, "hc_lowrank")?,
            ple_layer_ids: parse_usize_array(
                required_value(text, "ple_layer_ids")?,
                "ple_layer_ids",
            )?,
            ple_conv_kernel_size: required_usize(text, "ple_conv_kernel_size")?,
            ple_embed_dim: required_usize(text, "ple_embed_dim")?,
            split_ngram_parts: required_usize(text, "split_ngram_parts")?,
            heads_per_ngram: required_usize(text, "heads_per_ngram")?,
            ngram_size: required_usize(text, "ngram_size")?,
            ngram_vocab_size_base: required_u64(text, "ngram_vocab_size_base")?,
            make_ngram_vocab_size_divisible_by: required_usize(
                text,
                "make_ngram_vocab_size_divisible_by",
            )?,
            eos_token_id: parse_eos(text.get("eos_token_id"), 248_044)?,
            tie_word_embeddings: optional_bool(text, "tie_word_embeddings")
                .or_else(|| optional_bool(config, "tie_word_embeddings"))
                .unwrap_or(false),
            mtp_num_hidden_layers: required_usize(text, "mtp_num_hidden_layers")?,
            mtp_use_dedicated_embeddings: required_bool(text, "mtp_use_dedicated_embeddings")?,
            mtp,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the exact pinned Qwen3.8-Flash-Next geometry.
    pub fn validate(&self) -> Result<(), String> {
        expect(self.architecture_id, ARCH_ID, "architecture_id")?;
        expect_str(&self.model_type, MODEL_TYPE, "model_type")?;
        expect_str(&self.text_model_type, TEXT_MODEL_TYPE, "text_model_type")?;
        expect_eq(self.dtype, SourceDType::BF16, "dtype")?;
        expect(self.hidden_size, 2560, "hidden_size")?;
        expect(self.vocab_size, 248_320, "vocab_size")?;
        expect(self.num_hidden_layers, 48, "num_hidden_layers")?;
        expect(
            self.max_position_embeddings,
            262_144,
            "max_position_embeddings",
        )?;
        expect(self.num_attention_heads, 24, "num_attention_heads")?;
        expect(self.num_key_value_heads, 2, "num_key_value_heads")?;
        expect(self.head_dim, 256, "head_dim")?;
        expect_float(self.partial_rotary_factor, 0.25, "partial_rotary_factor")?;
        expect_float(self.rope_theta, 10_000_000.0, "rope_theta")?;
        expect(self.attention_bias, false, "attention_bias")?;
        expect(self.full_attention_interval, 4, "full_attention_interval")?;
        expect(self.linear_num_key_heads, 16, "linear_num_key_heads")?;
        expect(self.linear_num_value_heads, 48, "linear_num_value_heads")?;
        expect(self.linear_key_head_dim, 128, "linear_key_head_dim")?;
        expect(self.linear_value_head_dim, 128, "linear_value_head_dim")?;
        expect(self.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim")?;
        expect_eq(
            self.recurrent_state_dtype,
            RecurrentStateDType::F32,
            "mamba_ssm_dtype",
        )?;
        expect(self.indexer_n_heads, 4, "indexer_n_heads")?;
        expect(self.indexer_kv_heads, 1, "indexer_kv_heads")?;
        expect(self.indexer_head_dim, 128, "indexer_head_dim")?;
        expect(self.indexer_budget, 2048, "indexer_budget")?;
        expect(self.indexer_compress_ratio, 4, "indexer_compress_ratio")?;
        expect(self.num_experts, 512, "num_experts")?;
        expect(self.num_experts_per_tok, 10, "num_experts_per_tok")?;
        expect(self.moe_intermediate_size, 640, "moe_intermediate_size")?;
        expect(
            self.shared_expert_intermediate_size,
            640,
            "shared_expert_intermediate_size",
        )?;
        expect(self.norm_topk_prob, true, "norm_topk_prob")?;
        expect_str(&self.output_gate_type, "sigmoid", "output_gate_type")?;
        expect(self.hc_count, 4, "hc_count")?;
        expect(self.hc_lowrank, 320, "hc_lowrank")?;
        expect(
            self.ple_layer_ids.as_slice(),
            [2usize].as_slice(),
            "ple_layer_ids",
        )?;
        expect(self.ple_conv_kernel_size, 4, "ple_conv_kernel_size")?;
        expect(self.ple_embed_dim, 2560, "ple_embed_dim")?;
        expect(self.split_ngram_parts, 128, "split_ngram_parts")?;
        expect(self.heads_per_ngram, 8, "heads_per_ngram")?;
        expect(self.ngram_size, 3, "ngram_size")?;
        expect(
            self.ngram_vocab_size_base,
            20_000_000,
            "ngram_vocab_size_base",
        )?;
        expect(
            self.make_ngram_vocab_size_divisible_by,
            128,
            "make_ngram_vocab_size_divisible_by",
        )?;
        if self.eos_token_id as usize >= self.vocab_size {
            return Err(format!(
                "qwen4: eos_token_id={} is outside vocab_size={}",
                self.eos_token_id, self.vocab_size
            ));
        }
        expect(self.tie_word_embeddings, false, "tie_word_embeddings")?;
        expect(self.mtp_num_hidden_layers, 1, "mtp_num_hidden_layers")?;
        expect(
            self.mtp_use_dedicated_embeddings,
            false,
            "mtp_use_dedicated_embeddings",
        )?;
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "qwen4: layer_types length={} (expected {})",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        let expected_layers: Vec<_> = (0..self.num_hidden_layers)
            .map(|layer| {
                if layer % self.full_attention_interval == self.full_attention_interval - 1 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        if self.layer_types != expected_layers {
            return Err(format!(
                "qwen4: layer_types do not follow 3 linear + 1 full order: {:?}",
                self.layer_types
            ));
        }
        if self.n_linear_layers() != 36 || self.n_full_layers() != 12 {
            return Err(format!(
                "qwen4: layer counts linear={} full={} (expected 36/12)",
                self.n_linear_layers(),
                self.n_full_layers()
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
            || self.ple_embed_dim % (2 * self.heads_per_ngram) != 0
        {
            return Err("qwen4: head or PLE dimensions are not divisible".into());
        }
        self.mtp.validate()
    }

    /// Number of Gated DeltaNet layers.
    pub fn n_linear_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::LinearAttention)
            .count()
    }

    /// Number of QSA/full-attention layers.
    pub fn n_full_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::FullAttention)
            .count()
    }

    /// Number of values in each sharded n-gram row.
    pub fn ple_row_width(&self) -> usize {
        self.ple_embed_dim / (2 * self.heads_per_ngram)
    }

    /// Dilation of the PLE short convolution: upstream spaces its taps one
    /// n-gram apart.
    pub fn ple_conv_dilation(&self) -> usize {
        self.ngram_size
    }

    /// Rows of PLE short-convolution history, `(kernel - 1) * dilation`.
    pub fn ple_conv_history_rows(&self) -> usize {
        (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation()
    }

    /// The capacity selected by QSA: indexer budget plus the incomplete tail.
    pub fn qsa_selected_capacity(&self) -> usize {
        self.indexer_budget + self.indexer_compress_ratio.saturating_sub(1)
    }
}

pub(crate) fn compact_test_config() -> Qwen4Config {
    let layers: Vec<_> = (0..48)
        .map(|idx| {
            if idx % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let value = serde_json::json!({
        "architectures": [ARCHITECTURE_NAME],
        "model_type": MODEL_TYPE,
        "text_config": {
            "model_type": TEXT_MODEL_TYPE,
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000.0},
            "attention_bias": false,
            "layer_types": layers,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "norm_topk_prob": true,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "split_ngram_parts": 128,
            "heads_per_ngram": 8,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "eos_token_id": 248044,
            "tie_word_embeddings": false,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "output_gate_type": "sigmoid",
            "mtp": {
                "hybrid": true,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "rope_theta": 10000000.0
            }
        }
    });
    Qwen4Config::from_value(&value).expect("compact Qwen4 GPU fixture config")
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = right.abs().max(1.0);
    (left - right).abs() <= scale * 1e-9
}

fn required_value<'a>(object: &'a Value, key: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("qwen4: missing {key}"))
}

fn required_usize(object: &Value, key: &str) -> Result<usize, String> {
    let value = required_value(object, key)?;
    let value = value
        .as_u64()
        .ok_or_else(|| format!("qwen4: {key} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("qwen4: {key} does not fit usize"))
}

fn required_u64(object: &Value, key: &str) -> Result<u64, String> {
    required_value(object, key)?
        .as_u64()
        .ok_or_else(|| format!("qwen4: {key} must be a non-negative integer"))
}

fn required_f64(object: &Value, key: &str) -> Result<f64, String> {
    required_value(object, key)?
        .as_f64()
        .ok_or_else(|| format!("qwen4: {key} must be a number"))
}

fn required_bool(object: &Value, key: &str) -> Result<bool, String> {
    required_value(object, key)?
        .as_bool()
        .ok_or_else(|| format!("qwen4: {key} must be boolean"))
}

fn required_string<'a>(object: &'a Value, key: &str) -> Result<&'a str, String> {
    required_value(object, key)?
        .as_str()
        .ok_or_else(|| format!("qwen4: {key} must be a string"))
}

fn optional_bool(object: &Value, key: &str) -> Option<bool> {
    object.get(key).and_then(Value::as_bool)
}

fn optional_string(object: &Value, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_string)
}

fn parse_layers(value: &Value, key: &str) -> Result<Vec<LayerType>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("qwen4: {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value
                .as_str()
                .ok_or_else(|| format!("qwen4: {key}[{index}] must be a string"))?;
            LayerType::parse(name)
                .ok_or_else(|| format!("qwen4: unsupported {key}[{index}]={name:?}"))
        })
        .collect()
}

fn parse_usize_array(value: &Value, key: &str) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("qwen4: {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let number = value
                .as_u64()
                .ok_or_else(|| format!("qwen4: {key}[{index}] must be a non-negative integer"))?;
            usize::try_from(number).map_err(|_| format!("qwen4: {key}[{index}] does not fit usize"))
        })
        .collect()
}

fn parse_eos(value: Option<&Value>, default: u32) -> Result<u32, String> {
    let value = match value {
        None | Some(Value::Null) => return Ok(default),
        Some(value) => value,
    };
    let number = match value {
        Value::Number(number) => number
            .as_u64()
            .ok_or_else(|| "qwen4: eos_token_id must be a non-negative integer".to_string())?,
        Value::Array(values) => values
            .first()
            .and_then(Value::as_u64)
            .ok_or_else(|| "qwen4: eos_token_id array must contain an integer".to_string())?,
        _ => return Err("qwen4: eos_token_id must be an integer or array".into()),
    };
    u32::try_from(number).map_err(|_| "qwen4: eos_token_id does not fit u32".into())
}

fn expect<T: PartialEq + fmt::Debug>(actual: T, wanted: T, field: &str) -> Result<(), String> {
    if actual == wanted {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual:?} (expected {wanted:?})"))
    }
}

fn expect_eq<T: PartialEq + fmt::Debug>(actual: T, wanted: T, field: &str) -> Result<(), String> {
    expect(actual, wanted, field)
}

fn expect_str(actual: &str, wanted: &str, field: &str) -> Result<(), String> {
    if actual == wanted {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual:?} (expected {wanted:?})"))
    }
}

fn expect_float(actual: f64, wanted: f64, field: &str) -> Result<(), String> {
    if approximately_equal(actual, wanted) {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual} (expected {wanted})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        let layers: Vec<_> = (0..48)
            .map(|idx| {
                if idx % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        json!({
            "architectures": [ARCHITECTURE_NAME],
            "model_type": MODEL_TYPE,
            "text_config": {
                "model_type": TEXT_MODEL_TYPE,
                "dtype": "bfloat16",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "max_position_embeddings": 262144,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0},
                "attention_bias": false,
                "layer_types": layers,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "split_ngram_parts": 128,
                "heads_per_ngram": 8,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044,
                "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {
                    "hybrid": true,
                    "layer_types": ["full_attention"],
                    "num_hidden_layers": 1,
                    "rope_theta": 10000000.0
                }
            }
        })
    }

    #[test]
    fn parses_nested_metadata_and_exact_geometry() {
        let metadata = json!({"config": fixture()});
        let cfg = Qwen4Config::from_value(&metadata).expect("fixture should parse");
        assert_eq!(cfg.architecture_id, 16);
        assert_eq!(cfg.n_linear_layers(), 36);
        assert_eq!(cfg.n_full_layers(), 12);
        assert_eq!(cfg.layer_types[2], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
        assert_eq!(cfg.ple_row_width(), 160);
        assert_eq!(cfg.ple_conv_dilation(), 3);
        assert_eq!(cfg.ple_conv_history_rows(), 9);
        assert_eq!(cfg.qsa_selected_capacity(), 2051);
    }

    #[test]
    fn accepts_raw_text_config_without_metadata_wrapper() {
        let mut raw = fixture();
        let text = raw.as_object_mut().unwrap().remove("text_config").unwrap();
        let mut text = text;
        text.as_object_mut()
            .unwrap()
            .insert("model_type".into(), Value::String(TEXT_MODEL_TYPE.into()));
        let cfg = Qwen4Config::from_value(&text).expect("raw text config should parse");
        assert_eq!(cfg.text_model_type, TEXT_MODEL_TYPE);
    }
    #[test]
    fn strict_runtime_classifier_requires_nested_conditional_envelope() {
        let metadata = json!({"config": fixture()});
        let metadata_json = metadata.to_string();
        let cfg = Qwen4Config::from_metadata_json(&metadata_json).expect("nested config");
        assert_eq!(cfg.model_type, MODEL_TYPE);
        assert_eq!(cfg.text_model_type, TEXT_MODEL_TYPE);

        let mut raw = fixture();
        let text = raw.as_object_mut().unwrap().remove("text_config").unwrap();
        assert!(Qwen4Config::from_metadata_json(&text.to_string()).is_err());

        let mut missing_architecture = fixture();
        missing_architecture
            .as_object_mut()
            .unwrap()
            .remove("architectures");
        assert!(Qwen4Config::from_metadata_json(&missing_architecture.to_string()).is_err());
    }

    #[test]
    fn strict_runtime_classifier_accepts_conditional_config_with_vision_marker() {
        let mut metadata = fixture();
        metadata["vision_config"] = json!({"model_type": "vision"});
        let cfg = Qwen4Config::from_nested_value(&metadata).expect("text config");
        assert_eq!(cfg.architecture_id, ARCH_ID);
    }

    #[test]
    fn rejects_wrong_layer_order_before_allocation() {
        let mut value = fixture();
        value["text_config"]["layer_types"][0] = Value::String("full_attention".into());
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("layer_types"));
    }

    #[test]
    fn rejects_wrong_mtp_embedding_policy() {
        let mut value = fixture();
        value["text_config"]["mtp_use_dedicated_embeddings"] = Value::Bool(true);
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("mtp_use_dedicated_embeddings"));
    }

    #[test]
    fn rejects_wrong_recurrent_dtype() {
        let mut value = fixture();
        value["text_config"]["mamba_ssm_dtype"] = Value::String("bfloat16".into());
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("mamba_ssm_dtype"));
    }
}
