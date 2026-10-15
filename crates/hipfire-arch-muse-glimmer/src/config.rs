// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Config parsing for `muse_glimmer` (arch 14).
//!
//! Parsed from the HFQ `metadata_json.config.text_config` envelope.
//! Tensors live under `model.language_model.*` (see `lib.rs`).

use hipfire_runtime::hfq::HfqFile;

// ─── Layer type ────────────────────────────────────────────────────────

/// Per-layer attention type. Glimmer has 39 sliding + 13 full (3:1, [L,L,L,G]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlimmerLayerType {
    Sliding,
    Full,
}

// ─── Config ────────────────────────────────────────────────────────────

/// Typed Muse Glimmer dense-text shape constants.
#[derive(Debug, Clone)]
pub struct GlimmerConfig {
    pub dim: usize,                // hidden_size = 6656
    pub n_layers: usize,           // num_hidden_layers = 52
    pub vocab_size: usize,         // 202048
    pub n_heads: usize,            // num_attention_heads = 32
    pub n_kv_heads: usize,         // num_key_value_heads = 2
    pub head_dim: usize,           // 128 (uniform)
    pub sliding_window: usize,     // 2048
    pub max_position_embeddings: usize, // 131072
    pub hidden_dim: usize,         // intermediate_size = 19968
    pub rms_norm_eps: f32,         // 1e-5 for pre-norms
    pub post_norm_eps: f32,        // 1e-8 for post-norms — SEPARATE value
    pub qk_scale_factor: f32,      // 3.87 (scale-less QK-norm, no weight tensors)
    pub output_multiplier: f32,    // 0.196116135 == 1/sqrt(6656/256)
    pub final_logit_softcapping: f32, // 20.0
    pub hidden_activation: String, // "silu"
    pub attention_bias: bool,      // false
    pub tie_word_embeddings: bool, // false (untied lm_head)
    pub bos_token: u32,            // 200000
    pub eos_token: u32,            // 200001
    pub pad_token: Option<u32>,
    pub layer_types: Vec<GlimmerLayerType>,
    /// Per-layer RoPE theta. 500000.0 on sliding layers, 0.0 on full (NoPE).
    /// Key off THIS array, not the `layer_types` string (`lib.rs:11`).
    pub layer_rope_theta: Vec<f32>,
}

impl GlimmerConfig {
    /// Parse from the HFQ metadata envelope.
    /// Mirrors `Gemma4Config::from_hfq` but with Glimmer-specific fields.
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        Self::from_metadata_json(&hfq.metadata_json)
    }

    /// Parse the HFQ `metadata_json` text. Refuses any declared semantics the
    /// decoder does not implement instead of computing them silently wrong.
    pub fn from_metadata_json(metadata_json: &str) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(metadata_json)
            .map_err(|e| format!("glimmer: metadata_json not valid JSON: {e}"))?;
        let config = meta
            .get("config")
            .ok_or_else(|| "glimmer: metadata_json missing `config` wrapper".to_string())?;
        let tc = config.get("text_config").unwrap_or(config);

        let getu = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64());
        let getf = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        let getb = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());

        let dim = getu(tc, "hidden_size").ok_or("glimmer: missing hidden_size")? as usize;
        let n_layers = getu(tc, "num_hidden_layers")
            .ok_or("glimmer: missing num_hidden_layers")? as usize;
        let vocab_size = getu(tc, "vocab_size").ok_or("glimmer: missing vocab_size")? as usize;
        // split eps — using one value for both is silently wrong (brief RESOLVED)
        let rms_norm_eps = getf(tc, "rms_norm_eps").unwrap_or(1e-5) as f32;
        let post_norm_eps = getf(tc, "post_norm_eps").unwrap_or(1e-8) as f32;

        let bos_token = getu(tc, "bos_token_id").unwrap_or(200_000) as u32;
        let eos_token = match tc.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                a.first().and_then(|x| x.as_u64()).unwrap_or(200_001) as u32
            }
            Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(200_001) as u32,
            _ => 200_001,
        };
        let pad_token = getu(tc, "pad_token_id").map(|v| v as u32);

        let n_heads =
            getu(tc, "num_attention_heads").ok_or("glimmer: missing num_attention_heads")? as usize;
        let n_kv_heads =
            getu(tc, "num_key_value_heads").unwrap_or(n_heads as u64) as usize;
        let head_dim = getu(tc, "head_dim").map(|v| v as usize).unwrap_or(dim / n_heads);

        let sliding_window = getu(tc, "sliding_window").unwrap_or(2048) as usize;
        let max_position_embeddings =
            getu(tc, "max_position_embeddings").unwrap_or(131_072) as usize;
        let hidden_dim =
            getu(tc, "intermediate_size").ok_or("glimmer: missing intermediate_size")? as usize;

        let final_logit_softcapping = getf(tc, "final_logit_softcapping").unwrap_or(20.0) as f32;
        let output_multiplier = getf(tc, "output_multiplier").unwrap_or(0.196116135) as f32;
        let qk_scale_factor = getf(tc, "qk_scale_factor").unwrap_or(3.87) as f32;

        let hidden_activation = tc
            .get("hidden_activation")
            .and_then(|v| v.as_str())
            .unwrap_or("silu")
            .to_string();

        let attention_bias = getb(tc, "attention_bias").unwrap_or(false);
        // tie_word_embeddings may live on config or text_config; check both
        let tie_word_embeddings = getb(tc, "tie_word_embeddings")
            .or_else(|| getb(config, "tie_word_embeddings"))
            .unwrap_or(false);

        // layer_types: array of "sliding_attention" / "full_attention". READ it.
        let layer_types: Vec<GlimmerLayerType> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                        "full_attention" => GlimmerLayerType::Full,
                        _ => GlimmerLayerType::Sliding,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![GlimmerLayerType::Sliding; n_layers]);

        if layer_types.len() != n_layers {
            return Err(format!(
                "glimmer: layer_types len {} != n_layers {}",
                layer_types.len(),
                n_layers
            ));
        }

        // layer_rope_theta: 500000.0 on sliding, 0.0 on full (NoPE). Key off THIS.
        let layer_rope_theta: Vec<f32> = tc
            .get("layer_rope_theta")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .unwrap_or_else(|| {
                // Fallback: derive from layer_types if array missing (should not happen)
                layer_types
                    .iter()
                    .map(|t| match t {
                        GlimmerLayerType::Full => 0.0,
                        GlimmerLayerType::Sliding => 500_000.0,
                    })
                    .collect()
            });

        if layer_rope_theta.len() != n_layers {
            return Err(format!(
                "glimmer: layer_rope_theta len {} != n_layers {}",
                layer_rope_theta.len(),
                n_layers
            ));
        }
        // Semantic checkpoint: the layer program implements SiLU SwiGLU, bias-free
        // projections, and exactly two layer kinds — sliding layers rotate (theta
        // != 0) and read the sliding cache through a window; full layers are NoPE
        // (theta == 0) and read the full cache causally. The window keys off the
        // theta and the cache off the type, so a disagreement would attend the
        // wrong span.
        if hidden_activation != "silu" {
            return Err(format!(
                "glimmer: hidden_activation {hidden_activation:?} unsupported (only \"silu\")"
            ));
        }
        if attention_bias {
            return Err(
                "glimmer: attention_bias=true unsupported (bias-free projections only)".into(),
            );
        }
        for (i, (&kind, &theta)) in layer_types.iter().zip(&layer_rope_theta).enumerate() {
            if (kind == GlimmerLayerType::Full) != (theta == 0.0) {
                return Err(format!(
                    "glimmer: layer {i} is {kind:?} with rope theta {theta}; full layers must be \
                     NoPE (theta 0) and sliding layers must rotate"
                ));
            }
        }

        Ok(GlimmerConfig {
            dim,
            n_layers,
            vocab_size,
            n_heads,
            n_kv_heads,
            head_dim,
            sliding_window,
            max_position_embeddings,
            hidden_dim,
            rms_norm_eps,
            post_norm_eps,
            qk_scale_factor,
            output_multiplier,
            final_logit_softcapping,
            hidden_activation,
            attention_bias,
            tie_word_embeddings,
            bos_token,
            eos_token,
            pad_token,
            layer_types,
            layer_rope_theta,
        })
    }

    /// True if this layer has RoPE (theta != 0). Key off `layer_rope_theta`
    /// not `layer_types` (`lib.rs:11`).
    #[inline]
    pub fn has_rope(&self, layer_idx: usize) -> bool {
        self.layer_rope_theta[layer_idx] != 0.0
    }

    /// RoPE theta for this layer (0 means NoPE).
    #[inline]
    pub fn rope_theta_for(&self, layer_idx: usize) -> f32 {
        self.layer_rope_theta[layer_idx]
    }

    /// Window for attention_q8_0_kv_swa: sliding layers use sliding_window,
    /// full (NoPE) layers use 0 (full causal). Derived from rope theta.
    #[inline]
    pub fn window_for(&self, layer_idx: usize) -> usize {
        if self.has_rope(layer_idx) {
            self.sliding_window
        } else {
            0
        }
    }

    pub fn n_sliding_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == GlimmerLayerType::Sliding)
            .count()
    }

    pub fn n_full_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == GlimmerLayerType::Full)
            .count()
    }

    /// Uniform q projection width (32*128 = 4096).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Uniform kv projection width (2*128 = 256).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-layer `[L, L, L, G]` text config with `patch` merged in.
    fn parse(patch: serde_json::Value) -> Result<GlimmerConfig, String> {
        let mut tc = serde_json::json!({
            "hidden_size": 512, "num_hidden_layers": 4, "vocab_size": 1000,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 128,
            "intermediate_size": 1024, "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
        });
        for (k, v) in patch.as_object().unwrap() {
            tc[k] = v.clone();
        }
        GlimmerConfig::from_metadata_json(
            &serde_json::json!({ "config": { "text_config": tc } }).to_string(),
        )
    }

    #[test]
    fn layer_kinds_and_declared_semantics_fail_closed() {
        let cfg = parse(serde_json::json!({})).unwrap();
        assert_eq!(cfg.window_for(0), 2048);
        assert_eq!(cfg.window_for(3), 0);
        assert!(cfg.has_rope(2) && !cfg.has_rope(3));

        let theta = |t: [f64; 4]| serde_json::json!({ "layer_rope_theta": t });
        let gelu = serde_json::json!({ "hidden_activation": "gelu_pytorch_tanh" });
        let bias = serde_json::json!({ "attention_bias": true });
        for (patch, needle) in [
            // A rotating full layer, and a NoPE sliding layer.
            (theta([5e5, 5e5, 5e5, 1e4]), "layer 3"),
            (theta([5e5, 0.0, 5e5, 0.0]), "layer 1"),
            (gelu, "hidden_activation"),
            (bias, "attention_bias"),
        ] {
            let err = parse(patch.clone()).unwrap_err();
            assert!(err.contains(needle), "{patch}: {err}");
        }
    }
}
