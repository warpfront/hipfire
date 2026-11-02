// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tensor-name resolution and source access for the EmbeddingGemma-2
//! checkpoint (same two-source contract as the qwen35-embed crate; see that
//! module's docs for the HFQ native-methods / owned-fallback rationale).
//!
//! The checkpoint stores the text backbone under `language_model.*`
//! (`EmbeddingGemma2Model` is the bare composite); those names are resolved
//! by [`tensor_candidates`]. BOTH towers live in the **`.vl` sidecar** — a
//! separate source opened by the loader, whose flat names carry no
//! `model.`/`language_model.` prefix: the vision tower
//! (`vision_tower.*`, `embed_vision.*`) and the audio tower
//! (`audio_tower.*`, `embed_audio.*`) are resolved by the `*_tower` helpers
//! below, which never fall back to the trunk's namespace.

use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use std::borrow::Cow;

/// One tensor's storage descriptor + bytes (borrowed from a live mmap or
/// owned when the source must copy).
pub struct TensorBytes<'a> {
    /// HFQ quant-type byte; `0xFF` for a safetensors dir (dtype-driven).
    pub quant_type: u8,
    /// Safetensors dtype string; empty for an `.hfq`.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Cow<'a, [u8]>,
}

impl TensorBytes<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Candidate source names for a backbone tensor given its bare
/// `language_model.`-stripped name, most-specific first.
pub fn tensor_candidates(name: &str) -> Vec<String> {
    vec![
        format!("language_model.{name}"),
        format!("model.language_model.{name}"),
        name.to_string(),
    ]
}

/// Read a tensor by bare name from either source kind.
///
/// The caller must decode the returned bytes BEFORE calling [`release`]
/// (the release madvise drops the mmap's resident pages).
pub fn read<'a>(src: &'a ModelSource, name: &str) -> Result<TensorBytes<'a>, String> {
    let candidates = tensor_candidates(name);
    match src {
        ModelSource::Dir(d) => {
            for cand in &candidates {
                if let Some((info, bytes)) = d.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
            }
        }
        ModelSource::Hfq(h) => {
            for cand in &candidates {
                if let Some((info, bytes)) = h.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
                if let Some((info, vec)) = h.tensor_data_vec(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Owned(vec),
                    });
                }
            }
        }
    }
    Err(format!(
        "embeddinggemma2: missing tensor `{name}` (tried {candidates:?})"
    ))
}

/// Release the pages a [`read`] pulled in, once the caller has consumed the
/// bytes. Best-effort and advisory.
pub fn release(src: &ModelSource, name: &str) {
    let source: &dyn hipfire_runtime::model_source::ModelSource = match src {
        ModelSource::Dir(d) => d,
        ModelSource::Hfq(h) => h,
    };
    for cand in tensor_candidates(name) {
        if has(src, &cand) {
            source.release_tensor_pages(&cand);
            return;
        }
    }
}

/// Presence check for one candidate name.
///
/// `ModelSource::Hfq`'s trait `tensor_info` is a documented stub — HFQ's own
/// table is `HfqTensorInfo`-typed, not the trait's `TensorInfo` — so the HFQ
/// arm answers from the native index (`tensor_infos()`, the same table
/// [`read`] and [`read_tower`] fetch through). The `Dir` arm uses the trait's
/// real implementation. Calling the trait here instead is how the arch-18
/// `.vl` pack read as tower-less while carrying all 963 tower tensors.
fn has(src: &ModelSource, name: &str) -> bool {
    match src {
        ModelSource::Dir(d) => {
            let s: &dyn hipfire_runtime::model_source::ModelSource = d;
            s.tensor_info(name).is_some()
        }
        ModelSource::Hfq(h) => h.tensor_infos().iter().any(|t| t.name == name),
    }
}

/// Bare backbone names every checkpoint must provide (layer index elided).
pub fn required_text_tensors() -> Vec<String> {
    let mut out = vec![
        "embed_tokens.weight".to_string(),
        "norm.weight".to_string(),
        "embedding_projection.weight".to_string(),
        "ple.per_layer_model_projection.weight".to_string(),
        "ple.per_layer_projection_norm.weight".to_string(),
    ];
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "layer_scalar",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "ple_block.per_layer_input_gate.weight",
        "ple_block.per_layer_projection.weight",
        "ple_block.post_per_layer_input_norm.weight",
    ] {
        out.push(format!("layers.N.{n}"));
    }
    out
}

// ─── sidecar (vision tower) tensors ────────────────────────────────────────
//
// The `.vl` sidecar is a standalone HFQM/`.hfq` container holding ONLY the
// tower groups (`vision_tower.*` 210 tensors + `embed_vision.*`), so its
// names carry no `language_model.` prefix and are read through this second
// candidate list. The text-backbone list is deliberately NOT consulted: a
// trunk-local `vision_tower.*` would be a different (HFQ-native) packaging
// and silently mixing the two is how a tower gets half-loaded.

/// Candidate source names for a tower tensor, most-specific first.
pub fn tower_tensor_candidates(name: &str) -> Vec<String> {
    vec![name.to_string(), format!("model.{name}")]
}

/// Read a tower tensor from the sidecar source (see [`read`] for the
/// ownership/release contract, which is identical).
pub fn read_tower<'a>(src: &'a ModelSource, name: &str) -> Result<TensorBytes<'a>, String> {
    let candidates = tower_tensor_candidates(name);
    match src {
        ModelSource::Dir(d) => {
            for cand in &candidates {
                if let Some((info, bytes)) = d.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: info.dtype.clone(),
                        shape: info.shape.clone(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
            }
        }
        ModelSource::Hfq(h) => {
            for cand in &candidates {
                if let Some((info, bytes)) = h.tensor_data(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Borrowed(bytes),
                    });
                }
                if let Some((info, vec)) = h.tensor_data_vec(cand) {
                    return Ok(TensorBytes {
                        quant_type: info.quant_type,
                        dtype: String::new(),
                        shape: info.shape.iter().map(|v| *v as usize).collect(),
                        bytes: Cow::Owned(vec),
                    });
                }
            }
        }
    }
    Err(format!(
        "embeddinggemma2: missing tower tensor `{name}` in the vision sidecar \
         (tried {candidates:?})"
    ))
}

/// Whether the sidecar carries a tensor by this tower name.
pub fn tower_has(src: &ModelSource, name: &str) -> bool {
    tower_tensor_candidates(name).iter().any(|c| has(src, c))
}

/// Release the pages a [`read_tower`] pulled in (same contract as [`release`]).
pub fn release_tower(src: &ModelSource, name: &str) {
    let source: &dyn hipfire_runtime::model_source::ModelSource = match src {
        ModelSource::Dir(d) => d,
        ModelSource::Hfq(h) => h,
    };
    for cand in tower_tensor_candidates(name) {
        if has(src, &cand) {
            source.release_tensor_pages(&cand);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tower_candidates_keep_the_bare_name_first() {
        assert_eq!(
            tower_tensor_candidates("vision_tower.patch_embedder.input_proj.weight"),
            vec![
                "vision_tower.patch_embedder.input_proj.weight".to_string(),
                "model.vision_tower.patch_embedder.input_proj.weight".to_string(),
            ]
        );
        assert_eq!(
            tower_tensor_candidates("embed_vision.embedding_projection.weight")[0],
            "embed_vision.embedding_projection.weight"
        );
    }

    #[test]
    fn candidates_prefer_bare_composite_prefix() {
        assert_eq!(
            tensor_candidates("layers.0.self_attn.q_proj.weight"),
            vec![
                "language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
                "layers.0.self_attn.q_proj.weight".to_string(),
            ]
        );
        assert_eq!(
            tensor_candidates("ple.per_layer_model_projection.weight")[0],
            "language_model.ple.per_layer_model_projection.weight"
        );
    }
}
