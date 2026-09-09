// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Spark-X2.5-4B config, parsed from HFQ `metadata_json` envelope.
//!
//! Ground truth: XHToken/Spark-X2.5-4B `config.json` +
//! `modeling_spark.py` + `configuration_spark.py` (transformers 4.57.1).
//! 36 layers, hidden 2560, intermediate 10240, GQA 16q/4kv head_dim 256,
//! vocab 131072, 3:1 sliding/full hybrid (window 512), dual RoPE,
//! tie_word_embeddings true, hidden_act gelu, headwise sigmoid gate.
//!
//! Critical shape rules (easy to get silently wrong):
//!   1. **`head_dim` is EXPLICIT (256).** Never fall back to `hidden/n_heads`
//!      (2560/16 = 160). Q dim = 16×256 = **4096 ≠ hidden 2560**.
//!   2. **Dual `rope_parameters`.** Full layers: θ=5e6, partial_rotary=0.25
//!      (64 dims → 32 pairs). Sliding: θ=1e4, partial_rotary=1.0
//!      (256 dims → 128 pairs). Both use Llama half-split RoPE.
//!   3. **`hidden_act` must be exact-erf `gelu`** (not tanh approx, not silu).
//!   4. **Headwise sigmoid attn gate** (`g_proj` → n_heads scalars, broadcast
//!      over head_dim) — not elementwise over q_dim.

use hipfire_runtime::hfq::HfqFile;

/// Per-layer attention kind, decoded from `layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spark25LayerType {
    SlidingAttention,
    FullAttention,
}

impl Spark25LayerType {
    pub fn is_sliding(self) -> bool {
        matches!(self, Self::SlidingAttention)
    }
    pub fn is_full(self) -> bool {
        matches!(self, Self::FullAttention)
    }
}

/// Typed Spark-X2.5 shape constants.
#[derive(Clone, Debug)]
pub struct Spark25Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Explicit per-head dim from HF `head_dim`. NEVER derived as dim/n_heads
    /// (for 4B that would yield 160 instead of the required 256).
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// Local-attention window for sliding layers (HF `sliding_window` = 512).
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// Must be `"gelu"` (exact-erf). Validated at parse time.
    pub hidden_act: String,
    /// Per-layer attention kind; length == `n_layers`.
    pub layer_types: Vec<Spark25LayerType>,
    pub rope_theta_sliding: f32,
    pub rope_theta_full: f32,
    pub partial_rotary_factor_sliding: f32,
    pub partial_rotary_factor_full: f32,
    pub headwise_attn_output_gate: bool,
    /// Expected `"sigmoid"` for the published checkpoint.
    pub gate_attn_act_mode: String,
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
}

impl Spark25Config {
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// Parse from a `metadata_json` string (HFQ metadata OR a SafetensorsSource's
    /// config.json, both of which embed the HF config under the `config` key).
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("spark25: metadata_json not valid JSON: {e}"))?;
        let config = meta
            .get("config")
            .ok_or_else(|| "spark25: metadata_json missing `config` wrapper".to_string())?;
        // Spark is flat; still accept nested `text_config` like gemma4 for
        // forward-compat with multimodal packaging.
        let tc = config.get("text_config").unwrap_or(config);
        Self::from_config_value(tc)
    }

    /// Parse from a raw `config.json` Value (the inner `config` blob, or
    /// `text_config` when nested).
    pub fn from_config_value(tc: &serde_json::Value) -> Result<Self, String> {
        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let getb = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());
        let gets = |v: &serde_json::Value, k: &str| {
            v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string())
        };

        let dim = getu(tc, "hidden_size").ok_or("spark25: missing hidden_size")? as usize;
        let hidden_dim =
            getu(tc, "intermediate_size").ok_or("spark25: missing intermediate_size")? as usize;
        let n_layers =
            getu(tc, "num_hidden_layers").ok_or("spark25: missing num_hidden_layers")? as usize;
        let vocab_size = getu(tc, "vocab_size").ok_or("spark25: missing vocab_size")? as usize;
        let n_heads =
            getu(tc, "num_attention_heads").ok_or("spark25: missing num_attention_heads")? as usize;
        let n_kv_heads = getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;

        if n_heads == 0 {
            return Err("spark25: num_attention_heads must be nonzero".into());
        }
        if n_kv_heads == 0 {
            return Err("spark25: num_key_value_heads must be nonzero".into());
        }
        if n_heads % n_kv_heads != 0 {
            return Err(format!(
                "spark25: num_attention_heads ({n_heads}) must be divisible by \
                 num_key_value_heads ({n_kv_heads})"
            ));
        }

        // head_dim is REQUIRED and EXPLICIT. Falling back to dim/n_heads is a
        // silent quality bug (4B: 2560/16 = 160 ≠ 256; q_dim 4096 ≠ hidden).
        let head_dim = match getu(tc, "head_dim") {
            Some(v) if v > 0 => v as usize,
            Some(_) => return Err("spark25: head_dim must be nonzero".into()),
            None => {
                return Err(
                    "spark25: missing required explicit head_dim (must not derive from \
                     hidden_size/num_attention_heads)"
                        .into(),
                );
            }
        };

        let max_position_embeddings =
            getu(tc, "max_position_embeddings").unwrap_or(1_048_576) as usize;
        let sliding_window = getu(tc, "sliding_window").unwrap_or(512) as usize;
        if sliding_window == 0 {
            return Err("spark25: sliding_window must be nonzero".into());
        }
        let rms_norm_eps = getf(tc, "rms_norm_eps").unwrap_or(1e-6) as f32;
        let tie_word_embeddings = getb(tc, "tie_word_embeddings").unwrap_or(true);

        let hidden_act = gets(tc, "hidden_act").unwrap_or_else(|| "gelu".to_string());
        // modeling_spark.py raises unless hidden_act == "gelu" (exact-erf).
        if hidden_act != "gelu" {
            return Err(format!(
                "spark25: hidden_act must be \"gelu\" (exact-erf); got {hidden_act:?}"
            ));
        }

        // Bias paths are unsupported by the forward kernels (no bias GEMV).
        // Fail closed: silently treating them as false would execute the wrong model.
        if getb(tc, "attention_bias") == Some(true) {
            return Err(
                "spark25: attention_bias=true is unsupported by this backend (fail closed)".into(),
            );
        }
        if getb(tc, "mlp_bias") == Some(true) {
            return Err(
                "spark25: mlp_bias=true is unsupported by this backend (fail closed)".into(),
            );
        }

        let layer_types_raw = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "spark25: missing layer_types array".to_string())?;
        if layer_types_raw.len() != n_layers {
            return Err(format!(
                "spark25: layer_types len {} != num_hidden_layers {n_layers}",
                layer_types_raw.len()
            ));
        }
        let layer_types = layer_types_raw
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("spark25: layer_types[{i}] is not a string"))?;
                match s {
                    "sliding_attention" | "sliding" | "swa" => {
                        Ok(Spark25LayerType::SlidingAttention)
                    }
                    "full_attention" | "full" | "global" => Ok(Spark25LayerType::FullAttention),
                    other => Err(format!(
                        "spark25: unknown layer_type {other:?} at index {i}"
                    )),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Dual rope_parameters: {full_attention:{...}, sliding_attention:{...}}.
        let rope_params = tc.get("rope_parameters");
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));

        let rope_theta_sliding = sliding_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(10_000.0) as f32;
        let rope_theta_full = full_rope
            .and_then(|r| getf(r, "rope_theta"))
            .unwrap_or(5_000_000.0) as f32;
        let partial_rotary_factor_sliding = sliding_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(1.0) as f32;
        let partial_rotary_factor_full = full_rope
            .and_then(|r| getf(r, "partial_rotary_factor"))
            .unwrap_or(0.25) as f32;

        if !(0.0 < partial_rotary_factor_sliding && partial_rotary_factor_sliding <= 1.0) {
            return Err(format!(
                "spark25: sliding partial_rotary_factor out of (0,1]: \
                 {partial_rotary_factor_sliding}"
            ));
        }
        if !(0.0 < partial_rotary_factor_full && partial_rotary_factor_full <= 1.0) {
            return Err(format!(
                "spark25: full partial_rotary_factor out of (0,1]: {partial_rotary_factor_full}"
            ));
        }

        let headwise_attn_output_gate = getb(tc, "headwise_attn_output_gate").unwrap_or(true);
        let gate_attn_act_mode =
            gets(tc, "gate_attn_act_mode").unwrap_or_else(|| "sigmoid".to_string());
        if headwise_attn_output_gate && gate_attn_act_mode != "sigmoid" {
            return Err(format!(
                "spark25: unsupported gate_attn_act_mode={gate_attn_act_mode:?} \
                 (expected \"sigmoid\")"
            ));
        }

        let bos_token = parse_token_id(tc.get("bos_token_id"), 0);
        let eos_token = parse_token_id(tc.get("eos_token_id"), 1);
        let pad_token = parse_token_id(tc.get("pad_token_id"), 2);

        let cfg = Spark25Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            sliding_window,
            rms_norm_eps,
            tie_word_embeddings,
            hidden_act,
            layer_types,
            rope_theta_sliding,
            rope_theta_full,
            partial_rotary_factor_sliding,
            partial_rotary_factor_full,
            headwise_attn_output_gate,
            gate_attn_act_mode,
            bos_token,
            eos_token,
            pad_token,
        };
        cfg.validate_shapes()?;
        Ok(cfg)
    }

    /// Sanity-check derived dims. `q_dim` is n_heads*head_dim and need NOT
    /// equal `dim` (4B: 4096 vs 2560).
    fn validate_shapes(&self) -> Result<(), String> {
        if self.dim == 0 || self.hidden_dim == 0 || self.n_layers == 0 || self.vocab_size == 0 {
            return Err("spark25: zero-valued core dimension".into());
        }
        if self.layer_types.len() != self.n_layers {
            return Err(format!(
                "spark25: layer_types len {} != n_layers {}",
                self.layer_types.len(),
                self.n_layers
            ));
        }
        let q = self.q_dim();
        let kv = self.kv_dim();
        if q == 0 || kv == 0 {
            return Err("spark25: q_dim/kv_dim must be nonzero".into());
        }
        // Document the intentional mismatch: q_dim may exceed dim.
        let _ = (q, self.dim);
        Ok(())
    }

    /// Q projection width = `n_heads * head_dim` (4096 for 4B; NOT `dim`).
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// K/V projection width = `n_kv_heads * head_dim` (1024 for 4B).
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// Fused `q_k_v_proj` rows = q_dim + 2*kv_dim (6144 for 4B).
    pub fn qkv_rows(&self) -> usize {
        self.q_dim() + 2 * self.kv_dim()
    }
    /// Attention window for a layer: sliding_window for SWA, 0 (= unbounded) for full.
    pub fn window_for(&self, layer_idx: usize) -> usize {
        match self.layer_types[layer_idx] {
            Spark25LayerType::SlidingAttention => self.sliding_window,
            Spark25LayerType::FullAttention => 0,
        }
    }
    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_types[layer_idx].is_sliding()
    }
    pub fn layer_type(&self, layer_idx: usize) -> Spark25LayerType {
        self.layer_types[layer_idx]
    }
    /// Partial-rotary factor for a layer (full: 0.25, sliding: 1.0 on 4B).
    pub fn rope_partial_factor(&self, layer_idx: usize) -> f32 {
        match self.layer_types[layer_idx] {
            Spark25LayerType::FullAttention => self.partial_rotary_factor_full,
            Spark25LayerType::SlidingAttention => self.partial_rotary_factor_sliding,
        }
    }
    /// Number of RoPE pairs for a layer (full: 32, sliding: 128 on 4B).
    /// `rope_head_dim = head_dim * partial_rotary_factor`; pairs = half of that
    /// (Llama rotate_half convention).
    pub fn n_rot_pairs_for(&self, layer_idx: usize) -> usize {
        let prf = self.rope_partial_factor(layer_idx);
        let rope_head_dim = (self.head_dim as f32 * prf) as usize;
        rope_head_dim / 2
    }
    pub fn rope_theta_for(&self, layer_idx: usize) -> f32 {
        match self.layer_types[layer_idx] {
            Spark25LayerType::FullAttention => self.rope_theta_full,
            Spark25LayerType::SlidingAttention => self.rope_theta_sliding,
        }
    }
    /// Attention scale = 1/sqrt(head_dim). For head_dim=256 this is 0.0625.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// bos/eos/pad may be a bare number or a non-empty array (take first).
fn parse_token_id(v: Option<&serde_json::Value>, default: u32) -> u32 {
    match v {
        Some(serde_json::Value::Array(a)) => a
            .first()
            .and_then(|x| x.as_u64())
            .map(|n| n as u32)
            .unwrap_or(default),
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|n| n as u32).unwrap_or(default),
        _ => default,
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    /// Published XHToken/Spark-X2.5-4B config.json, wrapped in the HFQ
    /// `config` envelope the runtime uses.
    const PUBLISHED_CONFIG_JSON: &str = r#"{"config":{
  "architectures": ["Spark2_5ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0,
  "bos_token_id": 0,
  "dtype": "bfloat16",
  "eos_token_id": 1,
  "gate_attn_act_mode": "sigmoid",
  "head_dim": 256,
  "headwise_attn_output_gate": true,
  "hidden_act": "gelu",
  "hidden_size": 2560,
  "initializer_range": 0.01976,
  "intermediate_size": 10240,
  "layer_types": [
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention",
    "sliding_attention","sliding_attention","sliding_attention","full_attention"
  ],
  "max_position_embeddings": 1048576,
  "mlp_bias": false,
  "model_type": "spark2_5",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 4,
  "pad_token_id": 2,
  "rms_norm_eps": 0.000001,
  "rope_parameters": {
    "full_attention": {
      "partial_rotary_factor": 0.25,
      "rope_theta": 5000000
    },
    "sliding_attention": {
      "partial_rotary_factor": 1,
      "rope_theta": 10000
    }
  },
  "sliding_window": 512,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.1",
  "use_cache": true,
  "vocab_size": 131072
}}"#;

    fn cfg() -> Spark25Config {
        Spark25Config::from_metadata_json(PUBLISHED_CONFIG_JSON).expect("published config parses")
    }

    #[test]
    fn parses_published_spark_config() {
        let c = cfg();
        assert_eq!(c.n_layers, 36);
        assert_eq!(c.dim, 2560);
        assert_eq!(c.hidden_dim, 10240);
        assert_eq!(c.n_heads, 16);
        assert_eq!(c.n_kv_heads, 4);
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.vocab_size, 131_072);
        assert_eq!(c.max_position_embeddings, 1_048_576);
        assert_eq!(c.sliding_window, 512);
        assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(c.tie_word_embeddings);
        assert_eq!(c.hidden_act, "gelu");
        assert!(c.headwise_attn_output_gate);
        assert_eq!(c.gate_attn_act_mode, "sigmoid");
        assert_eq!(c.bos_token, 0);
        assert_eq!(c.eos_token, 1);
        assert_eq!(c.pad_token, 2);
        assert_eq!(c.rope_theta_sliding, 10_000.0);
        assert_eq!(c.rope_theta_full, 5_000_000.0);
        assert_eq!(c.partial_rotary_factor_sliding, 1.0);
        assert_eq!(c.partial_rotary_factor_full, 0.25);
    }

    #[test]
    fn explicit_head_dim_not_hidden_over_heads() {
        // The silent-quality-bug trap: dim/n_heads = 2560/16 = 160.
        // q_dim must be 16*256 = 4096, which exceeds hidden.
        let c = cfg();
        assert_ne!(c.head_dim, c.dim / c.n_heads, "must not derive head_dim");
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.q_dim(), 4096);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.qkv_rows(), 6144);
        assert_ne!(c.q_dim(), c.dim, "q_dim (4096) ≠ hidden (2560)");
    }

    #[test]
    fn layer_types_are_three_sliding_then_one_full() {
        let c = cfg();
        assert_eq!(c.layer_types.len(), 36);
        let mut n_sliding = 0usize;
        let mut n_full = 0usize;
        for (i, lt) in c.layer_types.iter().enumerate() {
            let expect = if i % 4 == 3 {
                Spark25LayerType::FullAttention
            } else {
                Spark25LayerType::SlidingAttention
            };
            assert_eq!(*lt, expect, "layer {i}");
            match lt {
                Spark25LayerType::SlidingAttention => n_sliding += 1,
                Spark25LayerType::FullAttention => n_full += 1,
            }
        }
        assert_eq!(n_sliding, 27);
        assert_eq!(n_full, 9);
    }

    #[test]
    fn rope_type_per_layer_full_vs_sliding() {
        let c = cfg();
        // Sliding layer 0: full rotary (prf=1.0), θ=1e4, 128 pairs.
        assert!(c.is_sliding(0));
        assert_eq!(c.rope_theta_for(0), 10_000.0);
        assert_eq!(c.rope_partial_factor(0), 1.0);
        assert_eq!(c.n_rot_pairs_for(0), 128);

        // Full layer 3: partial rotary (prf=0.25 → 64 dims → 32 pairs), θ=5e6.
        assert!(!c.is_sliding(3));
        assert_eq!(c.layer_type(3), Spark25LayerType::FullAttention);
        assert_eq!(c.rope_theta_for(3), 5_000_000.0);
        assert_eq!(c.rope_partial_factor(3), 0.25);
        assert_eq!(c.n_rot_pairs_for(3), 32);

        // Last full layer (35).
        assert_eq!(c.n_rot_pairs_for(35), 32);
        assert_eq!(c.rope_theta_for(35), 5_000_000.0);
    }

    #[test]
    fn swa_vs_full_window() {
        let c = cfg();
        assert_eq!(c.window_for(0), 512, "sliding → SWA 512");
        assert_eq!(c.window_for(1), 512);
        assert_eq!(c.window_for(2), 512);
        assert_eq!(c.window_for(3), 0, "full → unbounded (0)");
        assert_eq!(c.window_for(7), 0);
        assert_eq!(c.window_for(35), 0);
        assert!((c.attn_scale() - 0.0625).abs() < 1e-7);
    }

    #[test]
    fn layer_types_length_mismatch_is_rejected() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "gelu",
            "layer_types": ["sliding_attention","full_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(
            err.contains("layer_types len 2") && err.contains("num_hidden_layers 4"),
            "got: {err}"
        );
    }

    #[test]
    fn missing_head_dim_is_rejected() {
        // Must NOT silently fall back to hidden/n_heads.
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "hidden_act": "gelu",
            "layer_types": ["sliding_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(
            err.contains("missing required explicit head_dim"),
            "got: {err}"
        );
    }

    #[test]
    fn non_gelu_hidden_act_is_rejected() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "silu",
            "layer_types": ["sliding_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(err.contains("hidden_act must be \"gelu\""), "got: {err}");
    }

    #[test]
    fn gelu_pytorch_tanh_is_rejected() {
        // Spark uses exact-erf gelu, not the gemma tanh approximation.
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "gelu_pytorch_tanh",
            "layer_types": ["full_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(err.contains("hidden_act must be \"gelu\""), "got: {err}");
    }

    #[test]
    fn generic_dims_validate_not_hardcoded_4b() {
        // A tiny non-4B shape must still parse: proves dims are not hardcoded.
        let json = r#"{"config":{
            "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "hidden_act": "gelu", "sliding_window": 64,
            "tie_word_embeddings": true,
            "headwise_attn_output_gate": true, "gate_attn_act_mode": "sigmoid",
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention","full_attention"
            ],
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000},
                "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}
            }
        }}"#;
        let c = Spark25Config::from_metadata_json(json).expect("generic dims parse");
        assert_eq!(c.dim, 128);
        assert_eq!(c.hidden_dim, 256);
        assert_eq!(c.n_layers, 4);
        assert_eq!(c.head_dim, 32);
        assert_eq!(c.q_dim(), 128); // 4*32 — happens to equal dim here
        assert_eq!(c.kv_dim(), 64);
        assert_eq!(c.qkv_rows(), 256);
        assert_eq!(c.n_rot_pairs_for(0), 16); // sliding: 32/2
        assert_eq!(c.n_rot_pairs_for(3), 4); // full: 0.25*32/2 = 4
        assert_eq!(c.window_for(0), 64);
        assert_eq!(c.window_for(3), 0);
        assert_eq!(c.vocab_size, 512);
    }

    #[test]
    fn q_dim_can_exceed_hidden_on_generic_shape() {
        // head_dim larger than hidden/n_heads — mirrors the 4B asymmetry.
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 48, "hidden_act": "gelu",
            "layer_types": ["full_attention"],
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000},
                "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}
            }
        }}"#;
        let c = Spark25Config::from_metadata_json(json).expect("asymmetric q_dim");
        assert_eq!(c.q_dim(), 192);
        assert!(c.q_dim() > c.dim);
        assert_eq!(c.n_rot_pairs_for(0), 12); // 0.5 * 48 / 2
    }

    #[test]
    fn gqa_heads_must_divide() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 5, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "gelu",
            "layer_types": ["sliding_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(err.contains("divisible"), "got: {err}");
    }

    #[test]
    fn missing_config_wrapper_errs() {
        assert!(Spark25Config::from_metadata_json(r#"{"hidden_size":2560}"#).is_err());
    }

    #[test]
    fn text_config_nested_envelope_is_accepted() {
        // Multimodal packaging may nest under text_config; Spark is flat but
        // we accept both.
        let json = r#"{"config":{"text_config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "gelu",
            "layer_types": ["sliding_attention"]
        }}}"#;
        let c = Spark25Config::from_metadata_json(json).expect("text_config nest");
        assert_eq!(c.dim, 64);
        assert_eq!(c.head_dim, 16);
    }

    #[test]
    fn unknown_layer_type_fails_closed() {
        let json = r#"{"config":{
            "vocab_size": 100, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "hidden_act": "gelu",
            "layer_types": ["linear_attention"]
        }}"#;
        let err = Spark25Config::from_metadata_json(json).unwrap_err();
        assert!(err.contains("unknown layer_type"), "got: {err}");
    }
}
