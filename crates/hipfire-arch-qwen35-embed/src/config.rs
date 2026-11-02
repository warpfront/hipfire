// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Composite config for the Qwen3.5 embedding architecture.
//!
//! One `EmbedConfig` is parsed from up to five files in the model dir:
//!
//! - `config.json` — the composite HF config: `text_config` (the Qwen3.5
//!   hybrid backbone), `vision_config` (the SigLIP-style tower), and the
//!   `image_token_id` / `vision_start_token_id` / `vision_end_token_id`
//!   placeholder ids.
//! - `sentence_bert_config.json` — the ST ≤ v3 sidecar; on the pplx release
//!   this is the file that carries `query_length` / `document_length`.
//! - `config_sentence_transformers.json` — `prompts` (the `"[Q] "` /
//!   `"[D] "` marker strings by kind) and `query_length` / `document_length`
//!   truncation limits (wins per key where both sidecars define one).
//! - `2_MultiVectorMask/config.json` — the ColBERT skiplist words and which
//!   kinds they apply to (`skiplist_tasks`).
//! - `1_Dense/config.json` — `in_features` / `out_features` of the Dense
//!   projection (checked against the loaded `linear.weight` shape).
//!
//! Every sidecar has a hardcoded fallback matching the pplx-embed-v2
//! release, so a dir missing them (or a hand-quantized `.hfq` built from
//! just `model.safetensors`) still loads; the sidecar values win when
//! present.

use hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig;

/// Which sentence-transformers prompt/truncation profile an encode uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Query,
    Document,
}

/// Layer mixer kinds, parsed from `text_config.layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

/// The parsed composite config.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    // ── text backbone (Qwen3.5 family) ─────────────────────────────────
    pub hidden_size: usize,
    pub num_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub rms_norm_eps: f32,
    pub hidden_act_silu: bool, // always true today; kept for the error message
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,

    // full attention
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: [usize; 3],

    // linear attention (GatedDeltaNet)
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub conv_kernel_dim: usize,

    // ── multimodal splice ids (top-level config.json) ──────────────────
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,

    // ── sentence-transformers profile ──────────────────────────────────
    /// Marker token strings prepended per kind (`"[Q] "` / `"[D] "`).
    pub query_prompt: String,
    pub document_prompt: String,
    /// Max token count (marker included) per kind.
    pub query_length: usize,
    pub document_length: usize,
    /// ColBERT skiplist: token strings whose embeddings are dropped.
    pub skiplist_words: Vec<String>,
    /// Which kinds the skiplist applies to (`skiplist_tasks`).
    pub skiplist_query: bool,
    pub skiplist_document: bool,

    // ── Dense head ─────────────────────────────────────────────────────
    pub dense_out_features: usize,

    // ── vision tower ───────────────────────────────────────────────────
    pub vision: Option<VisionConfig>,
}

/// pplx-embed-v2 release defaults for every sidecar value (see module docs).
pub const DEFAULT_QUERY_PROMPT: &str = "[Q] ";
pub const DEFAULT_DOCUMENT_PROMPT: &str = "[D] ";
pub const DEFAULT_QUERY_LENGTH: usize = 1024;
pub const DEFAULT_DOCUMENT_LENGTH: usize = 4096;
pub const DEFAULT_DENSE_OUT: usize = 128;

pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 248_056;
pub const DEFAULT_VISION_START_TOKEN_ID: u32 = 248_053;
pub const DEFAULT_VISION_END_TOKEN_ID: u32 = 248_054;

fn u64_of(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn f64_of(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

fn parse_layer_types(
    text: &serde_json::Value,
    num_layers: usize,
) -> Result<Vec<LayerType>, String> {
    let raw = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "text_config.layer_types is required (hybrid backbone)".to_string())?;
    if raw.len() != num_layers {
        return Err(format!(
            "text_config.layer_types has {} entries for num_hidden_layers {num_layers}",
            raw.len()
        ));
    }
    raw.iter()
        .enumerate()
        .map(|(i, v)| {
            let s = v.as_str().unwrap_or("");
            match s {
                "linear_attention" => Ok(LayerType::LinearAttention),
                "full_attention" => Ok(LayerType::FullAttention),
                other => Err(format!(
                    "layer_types[{i}] = {other:?}: qwen35-embed supports linear_attention / full_attention"
                )),
            }
        })
        .collect()
}

/// Unwrap a checkpoint-metadata blob to the composite config object.
///
/// Both source kinds wrap the checkpoint config under a `"config"` key
/// (`SafetensorsSource::build_metadata_json` and the quantizer's HFQ
/// metadata); a raw `config.json` value is unwrapped. Returning a reference
/// into the input keeps this allocation-free on the load path.
pub fn config_from_metadata(meta: &serde_json::Value) -> &serde_json::Value {
    meta.get("config").unwrap_or(meta)
}

/// Parse the composite `config.json` (a `serde_json::Value` of its contents).
pub fn parse_config(config: &serde_json::Value) -> Result<EmbedConfig, String> {
    let text = config
        .get("text_config")
        .ok_or("config.json has no text_config (qwen35-embed is a composite model)")?;

    let hidden_size = u64_of(text, "hidden_size").ok_or("text_config.hidden_size")? as usize;
    let num_layers =
        u64_of(text, "num_hidden_layers").ok_or("text_config.num_hidden_layers")? as usize;
    let n_heads = u64_of(text, "num_attention_heads").ok_or("num_attention_heads")? as usize;
    let n_kv_heads = u64_of(text, "num_key_value_heads").ok_or("num_key_value_heads")? as usize;
    // Validate before the first divide: a hand-edited config with 0 heads
    // would panic here (or in preflight_slot, which the daemon calls on the
    // slot path before any GPU work).
    if n_heads == 0 || n_kv_heads == 0 {
        return Err(format!(
            "num_attention_heads={n_heads} / num_key_value_heads={n_kv_heads}: both must be > 0"
        ));
    }
    let head_dim = match u64_of(text, "head_dim") {
        Some(d) => d as usize,
        None => hidden_size / n_heads,
    };
    let intermediate_size = u64_of(text, "intermediate_size").ok_or("intermediate_size")? as usize;
    let vocab_size = u64_of(text, "vocab_size").ok_or("vocab_size")? as usize;
    let rms_norm_eps = f64_of(text, "rms_norm_eps").unwrap_or(1e-6) as f32;
    let hidden_act = text
        .get("hidden_act")
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
        .to_string();

    let linear_num_key_heads = u64_of(text, "linear_num_key_heads").unwrap_or(32) as usize;
    let linear_num_value_heads = u64_of(text, "linear_num_value_heads").unwrap_or(32) as usize;
    let linear_key_head_dim = u64_of(text, "linear_key_head_dim").unwrap_or(128) as usize;
    let linear_value_head_dim = u64_of(text, "linear_value_head_dim").unwrap_or(128) as usize;
    let conv_kernel_dim = u64_of(text, "linear_conv_kernel_dim").unwrap_or(4) as usize;

    // rope_parameters nests the theta / partial factor / mrope section on this
    // family; flat keys are the fallback.
    let rope = text.get("rope_parameters").cloned().unwrap_or_default();
    let rope_theta = f64_of(&rope, "rope_theta")
        .or_else(|| f64_of(text, "rope_theta"))
        .unwrap_or(10_000_000.0) as f32;
    let partial_rotary_factor = f64_of(&rope, "partial_rotary_factor")
        .or_else(|| f64_of(text, "partial_rotary_factor"))
        .unwrap_or(0.25) as f32;
    let mrope_section: [usize; 3] = rope
        .get("mrope_section")
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let vals: Vec<usize> = a
                .iter()
                .filter_map(|x| x.as_u64().map(|v| v as usize))
                .collect();
            (vals.len() == 3).then_some([vals[0], vals[1], vals[2]])
        })
        .unwrap_or([11, 11, 10]);

    let layer_types = parse_layer_types(text, num_layers)?;
    let n_rot_pairs_total: usize = mrope_section.iter().sum();
    let n_rot = (head_dim as f32 * partial_rotary_factor) as usize;
    if n_rot % 2 != 0 {
        return Err(format!(
            "partial rope dim {n_rot} is odd (head_dim {head_dim} × factor {partial_rotary_factor})"
        ));
    }
    if n_rot_pairs_total * 2 != n_rot {
        return Err(format!(
            "mrope_section {n_rot_pairs_total} pairs ≠ rotary dim {n_rot}/2 — HF applies \
             interleaved mRoPE over exactly the section pairs"
        ));
    }
    if n_heads % n_kv_heads != 0 {
        return Err(format!(
            "{n_heads} heads not divisible by {n_kv_heads} kv heads"
        ));
    }
    // The forward's GDN block reads k rows keyed by the VALUE-head count —
    // no GQA repeat-interleave between key and value heads exists here. The
    // whole family is 16/16; accept anything else only when a real interleave
    // lands, not by silently serving corrupted output.
    if linear_num_value_heads != linear_num_key_heads {
        return Err(format!(
            "linear_num_value_heads {linear_num_value_heads} != linear_num_key_heads \
             {linear_num_key_heads}: the encoder's delta-rule scan has no kv GQA interleave"
        ));
    }
    if hidden_act != "silu" {
        return Err(format!("hidden_act {hidden_act}: only silu is supported"));
    }
    // `attn_output_gate` ships as a JSON bool — read it as one (u64_of would
    // miss false and accept). Absent defaults to gated, matching HF.
    if text.get("attn_output_gate").and_then(|v| v.as_bool()) == Some(false) {
        return Err("attn_output_gate=false is not supported (fused q+gate projection)".into());
    }
    // The encoder's full attention is bidirectional; serving a checkpoint
    // that declares causal attention would produce wrong vectors without an
    // error, so refuse instead of silently flipping the mask.
    if text.get("is_causal").and_then(|v| v.as_bool()) == Some(true) {
        return Err("is_causal=true is not the bidirectional encoder this arch implements".into());
    }

    let image_token_id =
        u64_of(config, "image_token_id").unwrap_or(DEFAULT_IMAGE_TOKEN_ID as u64) as u32;
    let vision_start_token_id = u64_of(config, "vision_start_token_id")
        .unwrap_or(DEFAULT_VISION_START_TOKEN_ID as u64) as u32;
    let vision_end_token_id =
        u64_of(config, "vision_end_token_id").unwrap_or(DEFAULT_VISION_END_TOKEN_ID as u64) as u32;

    // ── vision tower (optional: text-only exports omit vision_config) ──
    let vision = config.get("vision_config").and_then(parse_vision_config);

    Ok(EmbedConfig {
        hidden_size,
        num_layers,
        layer_types,
        rms_norm_eps,
        hidden_act_silu: true,
        intermediate_size,
        vocab_size,
        tie_word_embeddings: true,
        n_heads,
        n_kv_heads,
        head_dim,
        rope_theta,
        partial_rotary_factor,
        mrope_section,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        conv_kernel_dim,
        image_token_id,
        vision_start_token_id,
        vision_end_token_id,
        query_prompt: DEFAULT_QUERY_PROMPT.to_string(),
        document_prompt: DEFAULT_DOCUMENT_PROMPT.to_string(),
        query_length: DEFAULT_QUERY_LENGTH,
        document_length: DEFAULT_DOCUMENT_LENGTH,
        skiplist_words: crate::post::default_skiplist_words(),
        skiplist_query: false,
        skiplist_document: true,
        dense_out_features: DEFAULT_DENSE_OUT,
        vision,
    })
}

/// Build the qwen35-vl [`VisionConfig`] out of `vision_config` (its HFQ
/// parser reads the same keys out of embedded metadata; this is the dir /
/// composite form).
fn parse_vision_config(vc: &serde_json::Value) -> Option<VisionConfig> {
    let hidden_size = u64_of(vc, "hidden_size")? as usize;
    let num_heads = u64_of(vc, "num_heads").unwrap_or(16) as usize;
    // A declared-but-zero head count would divide by zero below — treat it
    // the same as a missing tower: no vision config rather than a panic.
    if num_heads == 0 {
        return None;
    }
    Some(VisionConfig {
        hidden_size,
        num_heads,
        head_dim: hidden_size / num_heads,
        num_layers: u64_of(vc, "depth").unwrap_or(12) as usize,
        mlp_dim: u64_of(vc, "intermediate_size").unwrap_or(3072) as usize,
        patch_size: u64_of(vc, "patch_size").unwrap_or(16) as usize,
        temporal_patch_size: u64_of(vc, "temporal_patch_size").unwrap_or(2) as usize,
        out_hidden_size: u64_of(vc, "out_hidden_size").unwrap_or(1024) as usize,
        spatial_merge_size: u64_of(vc, "spatial_merge_size").unwrap_or(2) as usize,
        num_position_embeddings: u64_of(vc, "num_position_embeddings").unwrap_or(2304) as usize,
        rope_theta: f64_of(vc, "rope_theta").unwrap_or(10_000.0) as f32,
        norm_eps: 1e-6,
    })
}

impl EmbedConfig {
    /// Overlay `sentence_bert_config.json` (the ST ≤ v3 sidecar). The
    /// pplx-embed-v2 release carries its `query_length` / `document_length`
    /// HERE — `config_sentence_transformers.json` has only the prompts — so
    /// reading this file is what makes a checkpoint with different limits
    /// truncate at its own lengths instead of this crate's defaults. Applied
    /// before [`Self::apply_st_config`], whose keys win where both define
    /// one.
    pub fn apply_sbert_config(&mut self, sb: &serde_json::Value) {
        if let Some(prompts) = sb.get("prompts").and_then(|v| v.as_object()) {
            if let Some(q) = prompts.get("query").and_then(|v| v.as_str()) {
                self.query_prompt = q.to_string();
            }
            if let Some(d) = prompts.get("document").and_then(|v| v.as_str()) {
                self.document_prompt = d.to_string();
            }
        }
        if let Some(q) = u64_of(sb, "query_length") {
            self.query_length = q as usize;
        }
        if let Some(d) = u64_of(sb, "document_length") {
            self.document_length = d as usize;
        }
    }

    /// Overlay the sentence-transformers sidecar values
    /// (`config_sentence_transformers.json` contents) when present.
    pub fn apply_st_config(&mut self, st: &serde_json::Value) {
        if let Some(prompts) = st.get("prompts").and_then(|v| v.as_object()) {
            if let Some(q) = prompts.get("query").and_then(|v| v.as_str()) {
                self.query_prompt = q.to_string();
            }
            if let Some(d) = prompts.get("document").and_then(|v| v.as_str()) {
                self.document_prompt = d.to_string();
            }
        }
        if let Some(q) = u64_of(st, "query_length") {
            self.query_length = q as usize;
        }
        if let Some(d) = u64_of(st, "document_length") {
            self.document_length = d as usize;
        }
    }

    /// Overlay `2_MultiVectorMask/config.json` (skiplist words + tasks).
    pub fn apply_mask_config(&mut self, mask: &serde_json::Value) {
        if let Some(words) = mask.get("skiplist_words").and_then(|v| v.as_array()) {
            self.skiplist_words = words
                .iter()
                .filter_map(|w| w.as_str().map(str::to_string))
                .collect();
        }
        let tasks = mask
            .get("skiplist_tasks")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.skiplist_query = tasks.iter().any(|t| t == "query");
        self.skiplist_document = tasks.iter().any(|t| t == "document");
    }

    /// Overlay `1_Dense/config.json` (out_features check).
    pub fn apply_dense_config(&mut self, dense: &serde_json::Value) {
        if let Some(out) = u64_of(dense, "out_features") {
            self.dense_out_features = out as usize;
        }
    }

    pub fn kind_prompt(&self, kind: InputKind) -> &str {
        match kind {
            InputKind::Query => &self.query_prompt,
            InputKind::Document => &self.document_prompt,
        }
    }

    pub fn kind_length(&self, kind: InputKind) -> usize {
        match kind {
            InputKind::Query => self.query_length,
            InputKind::Document => self.document_length,
        }
    }

    pub fn kind_skiplist(&self, kind: InputKind) -> bool {
        match kind {
            InputKind::Query => self.skiplist_query,
            InputKind::Document => self.skiplist_document,
        }
    }

    /// Rotary dim in elements (= pairs × 2) for the full-attention rope.
    pub fn n_rot(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Total GDN key/value widths (2048 each on the 0.6B).
    pub fn gdn_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn gdn_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The composite config.json of perplexity-ai/pplx-embed-v2-late-0.6b
    /// (trimmed to the fields the parser reads; full values).
    const PPLX_CONFIG: &str = r#"{
      "architectures": ["Qwen3_5Model"],
      "image_token_id": 248056,
      "model_type": "qwen3_5",
      "tie_word_embeddings": true,
      "vision_start_token_id": 248053,
      "vision_end_token_id": 248054,
      "text_config": {
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "is_causal": false,
        "layer_types": ["linear_attention","full_attention","linear_attention","full_attention",
                        "linear_attention","full_attention","linear_attention","full_attention",
                        "linear_attention","full_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "mrope_interleaved": true,
          "mrope_section": [11, 11, 10],
          "partial_rotary_factor": 0.25,
          "rope_theta": 10000000.0,
          "rope_type": "default"
        },
        "vocab_size": 248320
      },
      "vision_config": {
        "depth": 12,
        "hidden_size": 768,
        "in_channels": 3,
        "intermediate_size": 3072,
        "num_heads": 12,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
      }
    }"#;

    #[test]
    fn parses_pplx_composite_config() {
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let cfg = parse_config(&v).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.layer_types.len(), 12);
        assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.n_rot(), 64);
        assert_eq!(cfg.mrope_section, [11, 11, 10]);
        assert_eq!(cfg.rope_theta, 1e7);
        assert_eq!(cfg.gdn_key_dim(), 2048);
        assert_eq!(cfg.gdn_value_dim(), 2048);
        assert_eq!(cfg.image_token_id, 248_056);
        let vis = cfg.vision.expect("vision config");
        assert_eq!(vis.hidden_size, 768);
        assert_eq!(vis.num_layers, 12);
        assert_eq!(vis.head_dim, 64);
        assert_eq!(vis.out_hidden_size, 1024);
        assert_eq!(vis.num_position_embeddings, 2304);
        // Sidecar defaults before the overlays run.
        assert_eq!(cfg.query_prompt, "[Q] ");
        assert_eq!(cfg.document_prompt, "[D] ");
        assert!(!cfg.skiplist_query);
        assert!(cfg.skiplist_document);
    }

    #[test]
    fn sidecar_overlays_apply() {
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let mut cfg = parse_config(&v).unwrap();
        let st: serde_json::Value = serde_json::from_str(
            r#"{"prompts":{"query":"[Q] ","document":"[D] "},
                                    "query_length":32,"document_length":128}"#,
        )
        .unwrap();
        cfg.apply_st_config(&st);
        assert_eq!(cfg.query_length, 32);
        assert_eq!(cfg.document_length, 128);

        let mask: serde_json::Value = serde_json::from_str(
            r#"{"skiplist_words":["!","."],"skiplist_tasks":["document","query"]}"#,
        )
        .unwrap();
        cfg.apply_mask_config(&mask);
        assert_eq!(cfg.skiplist_words, vec!["!", "."]);
        assert!(cfg.skiplist_query);
        assert!(cfg.skiplist_document);

        let dense: serde_json::Value =
            serde_json::from_str(r#"{"in_features":1024,"out_features":128}"#).unwrap();
        cfg.apply_dense_config(&dense);
        assert_eq!(cfg.dense_out_features, 128);
    }

    #[test]
    fn sbert_sidecar_carries_the_pplx_length_limits() {
        // The pplx-embed-v2 release's `sentence_bert_config.json`, verbatim
        // in the fields this reads: `config_sentence_transformers.json` has
        // NO length keys, so this file is the only place the limits come
        // from — and they are 1024/4096, not this crate's pre-overlay
        // defaults' neighbours (512/2048) nor anything else.
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let mut cfg = parse_config(&v).unwrap();
        let sbert: serde_json::Value = serde_json::from_str(
            r#"{"modality_config":{},"module_output_name":"token_embeddings",
                "query_length":1024,"document_length":4096,"query_expansion":null}"#,
        )
        .unwrap();
        cfg.apply_sbert_config(&sbert);
        assert_eq!(cfg.query_length, 1024);
        assert_eq!(cfg.document_length, 4096);

        // A different export with different limits truncates at ITS limits,
        // not ours — the reason the file is read at all.
        let mut cfg2 = parse_config(&v).unwrap();
        let other: serde_json::Value =
            serde_json::from_str(r#"{"query_length":512,"document_length":2048}"#).unwrap();
        cfg2.apply_sbert_config(&other);
        assert_eq!(cfg2.query_length, 512);
        assert_eq!(cfg2.document_length, 2048);

        // ST precedence: where both sidecars define a key, the newer
        // config_sentence_transformers.json wins (applied after).
        let mut cfg3 = parse_config(&v).unwrap();
        cfg3.apply_sbert_config(&other);
        let st: serde_json::Value = serde_json::from_str(r#"{"query_length":8}"#).unwrap();
        cfg3.apply_st_config(&st);
        assert_eq!(cfg3.query_length, 8);
        assert_eq!(cfg3.document_length, 2048);
    }

    #[test]
    fn config_from_metadata_unwraps_both_shapes() {
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let wrapped = serde_json::json!({ "architecture": "qwen3_5", "config": v.clone() });
        let cfg = parse_config(config_from_metadata(&wrapped)).unwrap();
        assert_eq!(cfg.hidden_size, 1024);
        // A raw config.json (no wrapper) still parses.
        let cfg2 = parse_config(config_from_metadata(&v)).unwrap();
        assert_eq!(cfg2.num_layers, 12);
    }

    #[test]
    fn rejects_mrope_section_mismatch() {
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let mut bad = v.clone();
        bad["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([8, 8, 8]);
        let err = parse_config(&bad).unwrap_err();
        assert!(err.contains("mrope_section"), "{err}");
    }

    #[test]
    fn rejects_unknown_layer_type() {
        let v: serde_json::Value = serde_json::from_str(PPLX_CONFIG).unwrap();
        let mut bad = v.clone();
        bad["text_config"]["layer_types"][0] = serde_json::json!("sliding_attention");
        let err = parse_config(&bad).unwrap_err();
        assert!(err.contains("sliding_attention"), "{err}");
    }
}
