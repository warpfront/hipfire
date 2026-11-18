//! Vision-tower sidecar policy (`qwen3.8-27b-vision.hfq`).
//!
//! Pure predicates behind the vision-only pack:
//!
//! ```bash
//! hipfire-quantize <hf-dir> --include-vision --include-prefix model.visual. \
//!     --output qwen3.8-27b-vision.hfq
//! # or the shorthand:
//! hipfire-quantize <hf-dir> --vision-only --output qwen3.8-27b-vision.hfq
//! ```
//!
//! Flash Next uses this same dtype policy through `--qwen4-flash-next
//! --vision-only`. Its official source also uses `model.visual.*`, including
//! the complete merger/projector; the native pack supports bounded remote
//! ranges without ingesting the language or PLE payloads.
//!
//! Lives in the lib target (not the binary) so the contract is pinned by
//! `cargo test -p hipfire-quantize --lib vision`. The binary's ingest loop
//! (`pipeline.rs`) and dtype fallback call these; they must agree with
//! `model_filter::should_quantize`, which keeps every vision-group tensor
//! off the text-quantize path.
//!
//! EmbeddingGemma-2 (arch 18) reuses the same emit path for its own `.vl`
//! sibling, but never through `--vision-only`: the checkpoint's tower groups
//! are flat (`vision_tower.*` / `audio_tower.*` / `embed_vision.*` /
//! `embed_audio.*`, see [`is_eg2_sidecar_tensor`]), so no `--include-prefix`
//! selects them and the arch-18 pack is written unconditionally by the
//! ordinary trunk build. [`eg2_sidecar_dtype`] is the dtype half of that
//! reuse; it wins over [`vision_dtype`] for the names both recognise.

/// Tensor prefix that selects the Qwen3.5-family vision tower, and the
/// default `--include-prefix` implied by `--vision-only`.
pub const VISION_SIDECAR_PREFIX: &str = "model.visual.";

/// Tower tensors: the Qwen3.5-VL `model.visual.*` names plus the
/// `visual.*` / `vision_tower.*` / `model.vision_tower.*` /
/// `model.vision_adapter.*` / `model.vision_projection.*` aliases other
/// families use. Same set as the ingest loop's `is_vision` gate.
pub fn is_vision_tower_tensor(name: &str) -> bool {
    name.starts_with("model.visual.")
        || name.starts_with("visual.")
        || name.starts_with("vision_tower.")
        || name.starts_with("model.vision_tower.")
        || name.starts_with("model.vision_adapter.")
        || name.starts_with("model.vision_projection.")
}

/// Full vision group: tower tensors plus the LFM2/Idefics-style
/// `model.multi_modal_projector.` MLP, which rides the vision path under
/// `--include-vision` and is skipped without it. Same set as the ingest
/// loop's `vision_group` gate. Note the projector is NOT under
/// [`VISION_SIDECAR_PREFIX`], so a vision-only sidecar never contains it.
pub fn is_vision_group_tensor(name: &str) -> bool {
    is_vision_tower_tensor(name) || name.starts_with("model.multi_modal_projector.")
}

/// Emitted container for a vision-group tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionDtype {
    /// Weight matrix: F16 (qt=1), consumed by the `gemm_f16` path.
    F16Matrix,
    /// Norm weight/bias, projection bias, or learned position table:
    /// F32 (qt=2, lossless widen from BF16/F16 source). The loader's
    /// `load_f32_*` arms read these directly; `load_f16_gpu` would narrow.
    F32Vector,
}

/// Container for a vision-group tensor, or `None` when `name` is outside
/// the vision group (text tensors take the normal quant path).
///
/// Vectors are name-selected, not rank-selected: `pos_embed.weight` is a
/// 2-D `[num_positions, hidden]` table but loads through `load_f32_cpu`,
/// so it rides F32 like every other vector.
pub fn vision_dtype(name: &str, ndim: usize) -> Option<VisionDtype> {
    if !is_vision_group_tensor(name) {
        return None;
    }
    let is_vector = ndim <= 1
        || name.ends_with(".bias")
        || name.contains(".norm")
        || name.ends_with("pos_embed.weight");
    Some(if is_vector {
        VisionDtype::F32Vector
    } else {
        VisionDtype::F16Matrix
    })
}

/// EmbeddingGemma-2 (arch 18) tower groups: the gemma4 vision tower, the
/// audio conformer tower, and the two projection heads that map each tower
/// into the shared 512-wide sentence space. Flat checkpoint names — this
/// checkpoint has no `model.` prefix and no `visual.` alias, so the qwen35
/// prefix set never matches it (and vice versa).
///
/// The trunk is tower-less by construction: arch 18 skips every one of these
/// from the text artifact and ships them as the model's own `.vl` sidecar.
pub fn is_eg2_sidecar_tensor(name: &str) -> bool {
    name.starts_with("vision_tower.")
        || name.starts_with("audio_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("embed_audio.")
}

/// Container for an EmbeddingGemma-2 tower tensor, or `None` when `name` is
/// outside the arch-18 tower groups (the trunk tensors take the normal text
/// path, which for arch 18 means the skip gate).
///
/// Same convention as [`vision_dtype`] and for the same reason — tower
/// matrices and conv kernels load through the f16 GEMM/conv path
/// (`load_f16_gpu`), their norms, biases, `per_dim_scale`, clip scalars and
/// lookup tables through the f32 arms — with two arch-18 specifics:
///
/// * the audio tower's clipped-linears ship their clip bounds as **0-d**
///   tensors (`input_min`/`input_max`/`output_min`/`output_max`, 480 of
///   them): scalars, not matrices, so `ndim == 0` rides F32;
/// * the vision tower's positional lookup table is spelled
///   `position_embedding_table` (3-D `[2, 10240, 768]`), not
///   `pos_embed.weight`, so [`vision_dtype`]'s suffix list never matches it
///   and a plain rank rule would narrow it to F16 — a lookup table must ride
///   F32 exactly like the qwen35 one.
pub fn eg2_sidecar_dtype(name: &str, ndim: usize) -> Option<VisionDtype> {
    if !is_eg2_sidecar_tensor(name) {
        return None;
    }
    let is_vector = ndim <= 1 || name == "vision_tower.patch_embedder.position_embedding_table";
    Some(if is_vector {
        VisionDtype::F32Vector
    } else {
        VisionDtype::F16Matrix
    })
}

/// `--include-prefix` gate: when set, only tensors under the prefix are
/// ingested; when unset every tensor passes.
pub fn passes_include_prefix(name: &str, prefix: Option<&str>) -> bool {
    prefix.map_or(true, |p| name.starts_with(p))
}

/// Resolve the effective `--include-prefix`: an explicit prefix always
/// wins; `--vision-only` implies [`VISION_SIDECAR_PREFIX`].
pub fn resolve_vision_prefix<'a>(explicit: Option<&'a str>, vision_only: bool) -> Option<&'a str> {
    match explicit {
        Some(p) => Some(p),
        None => {
            if vision_only {
                Some(VISION_SIDECAR_PREFIX)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8 tower inventory (scope source of truth:
    /// `load_vision_weights`, `hipfire-arch-qwen35-vl/src/qwen35_vl.rs`):
    /// 3 head tensors + 12 per block x 27 + 6 merger = 333.
    fn vision_sidecar_names() -> Vec<(String, usize)> {
        let mut names: Vec<(String, usize)> = Vec::with_capacity(333);
        // Head: proj matrix (F16), proj bias (F32), pos-embed table (F32).
        names.push(("model.visual.patch_embed.proj.weight".to_string(), 2));
        names.push(("model.visual.patch_embed.proj.bias".to_string(), 1));
        names.push(("model.visual.pos_embed.weight".to_string(), 2));
        for i in 0..27 {
            let p = format!("model.visual.blocks.{i}");
            for suffix in [
                "norm1.weight",
                "norm1.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
            ] {
                let full = format!("{p}.{suffix}");
                let ndim = if suffix.ends_with(".weight") && !suffix.starts_with("norm") {
                    2
                } else {
                    1
                };
                names.push((full, ndim));
            }
        }
        for suffix in [
            "norm.weight",
            "norm.bias",
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
        ] {
            let full = format!("model.visual.merger.{suffix}");
            let ndim = if suffix.ends_with(".weight") && !suffix.starts_with("norm") {
                2
            } else {
                1
            };
            names.push((full, ndim));
        }
        names
    }

    #[test]
    fn vision_sidecar_name_set_is_333_with_f16_matrices_and_f32_vectors() {
        let names = vision_sidecar_names();
        // 3 head + 12 x 27 blocks + 6 merger.
        assert_eq!(names.len(), 3 + 12 * 27 + 6, "sidecar inventory size");
        let mut f16 = 0usize;
        let mut f32 = 0usize;
        for (name, ndim) in &names {
            // ONLY model.visual.* tensors are ingested under the sidecar filter.
            assert!(
                passes_include_prefix(name, Some(VISION_SIDECAR_PREFIX)),
                "{name} must pass the sidecar prefix filter"
            );
            assert!(
                is_vision_group_tensor(name),
                "{name} must be in the vision group"
            );
            match vision_dtype(name, *ndim) {
                Some(VisionDtype::F16Matrix) => f16 += 1,
                Some(VisionDtype::F32Vector) => f32 += 1,
                None => panic!("{name} must have a vision dtype"),
            }
        }
        // Matrices: patch_embed.proj + 4 per block (qkv, proj, fc1, fc2)
        // + merger fc1/fc2 = 1 + 108 + 2. Everything else is F32 vectors.
        assert_eq!(f16, 1 + 4 * 27 + 2, "F16 (qt=1) matrix count");
        assert_eq!(f32, 2 + 8 * 27 + 4, "F32 (qt=2) norm/bias/pos_embed count");
        assert_eq!(f16 + f32, 333);
    }

    #[test]
    fn vision_sidecar_spot_dtypes_match_loader_arms() {
        // load_f16_gpu arms.
        for (name, ndim) in [
            ("model.visual.patch_embed.proj.weight", 2),
            ("model.visual.blocks.0.attn.qkv.weight", 2),
            ("model.visual.blocks.26.mlp.linear_fc2.weight", 2),
            ("model.visual.merger.linear_fc1.weight", 2),
        ] {
            assert_eq!(vision_dtype(name, ndim), Some(VisionDtype::F16Matrix));
        }
        // load_f32_gpu / load_f32_cpu arms (biases, norms, pos-embed table).
        for (name, ndim) in [
            ("model.visual.patch_embed.proj.bias", 1),
            ("model.visual.pos_embed.weight", 2),
            ("model.visual.blocks.0.norm1.weight", 1),
            ("model.visual.blocks.0.attn.qkv.bias", 1),
            ("model.visual.merger.norm.bias", 1),
            ("model.visual.merger.linear_fc2.bias", 1),
        ] {
            assert_eq!(vision_dtype(name, ndim), Some(VisionDtype::F32Vector));
        }
    }

    #[test]
    fn vision_sidecar_filter_rejects_text_and_projector() {
        // Text tensors: outside the group, rejected by the sidecar prefix.
        let text = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(vision_dtype(text, 2), None);
        assert!(!passes_include_prefix(text, Some(VISION_SIDECAR_PREFIX)));
        assert!(!is_vision_group_tensor(text));
        // The multi-modal projector rides the vision group in full-VL builds
        // but lives outside the sidecar prefix: never in the sidecar file.
        let proj = "model.multi_modal_projector.linear.weight";
        assert!(is_vision_group_tensor(proj));
        assert!(!is_vision_tower_tensor(proj));
        assert!(!passes_include_prefix(proj, Some(VISION_SIDECAR_PREFIX)));
    }

    #[test]
    fn vision_only_flag_resolution_prefers_explicit_prefix() {
        assert_eq!(
            resolve_vision_prefix(None, true),
            Some(VISION_SIDECAR_PREFIX)
        );
        assert_eq!(resolve_vision_prefix(Some("mtp."), true), Some("mtp."));
        assert_eq!(resolve_vision_prefix(None, false), None);
        assert_eq!(
            resolve_vision_prefix(Some("model.visual."), false),
            Some("model.visual.")
        );
    }

    /// EmbeddingGemma-2 (arch 18) tower inventory. Scope source of truth: the
    /// `google/embeddinggemma-2` safetensors header (1376 tensors; 963 of them
    /// tower-side) cross-checked against `docs/plans/embeddinggemma2-tower-spec.md`.
    /// Rank is carried because the dtype policy is rank-sensitive: the audio
    /// clip bounds are 0-d and the vision positional table is 3-D — the two
    /// cases a name-only (qwen35-shaped) rule classifies wrong.
    fn eg2_sidecar_names() -> Vec<(String, usize)> {
        let mut names: Vec<(String, usize)> = Vec::with_capacity(963);
        // Vision tower: patch embedder + 16 encoder layers of 13 tensors.
        names.push((
            "vision_tower.patch_embedder.input_proj.weight".to_string(),
            2,
        ));
        names.push((
            "vision_tower.patch_embedder.position_embedding_table".to_string(),
            3,
        ));
        for i in 0..16 {
            let p = format!("vision_tower.encoder.layers.{i}");
            for suffix in [
                "input_layernorm.weight",            // rank 1
                "mlp.down_proj.linear.weight",       // rank 2
                "mlp.gate_proj.linear.weight",       // rank 2
                "mlp.up_proj.linear.weight",         // rank 2
                "post_attention_layernorm.weight",   // rank 1
                "post_feedforward_layernorm.weight", // rank 1
                "pre_feedforward_layernorm.weight",  // rank 1
                "self_attn.k_norm.weight",           // rank 1
                "self_attn.k_proj.linear.weight",    // rank 2
                "self_attn.o_proj.linear.weight",    // rank 2
                "self_attn.q_norm.weight",           // rank 1
                "self_attn.q_proj.linear.weight",    // rank 2
                "self_attn.v_proj.linear.weight",    // rank 2
            ] {
                let ndim = if suffix.ends_with(".linear.weight") {
                    2
                } else {
                    1
                };
                names.push((format!("{p}.{suffix}"), ndim));
            }
        }
        // Audio tower: conv subsample + output projection, then 12 conformer
        // layers of 62 tensors (12 matrices, 50 vectors/scalars).
        for (suffix, ndim) in [
            ("output_proj.weight", 2),
            ("output_proj.bias", 1),
            ("subsample_conv_projection.input_proj_linear.weight", 2),
            ("subsample_conv_projection.layer0.conv.weight", 4),
            ("subsample_conv_projection.layer0.norm.weight", 1),
            ("subsample_conv_projection.layer1.conv.weight", 4),
            ("subsample_conv_projection.layer1.norm.weight", 1),
        ] {
            names.push((format!("audio_tower.{suffix}"), ndim));
        }
        // Per-layer matrices: FFN pair x2, depthwise lconv, lconv linear
        // start/end, and the four attention projections + relative-k.
        const AUDIO_LAYER_MATRICES: [(&str, usize); 12] = [
            ("feed_forward1.ffw_layer_1.linear.weight", 2),
            ("feed_forward1.ffw_layer_2.linear.weight", 2),
            ("feed_forward2.ffw_layer_1.linear.weight", 2),
            ("feed_forward2.ffw_layer_2.linear.weight", 2),
            ("lconv1d.depthwise_conv1d.weight", 3),
            ("lconv1d.linear_end.linear.weight", 2),
            ("lconv1d.linear_start.linear.weight", 2),
            ("self_attn.k_proj.linear.weight", 2),
            ("self_attn.post.linear.weight", 2),
            ("self_attn.q_proj.linear.weight", 2),
            ("self_attn.relative_k_proj.weight", 2),
            ("self_attn.v_proj.linear.weight", 2),
        ];
        // Per-layer vectors: the clipped-linears' 0-d clip bounds (each of the
        // 6 clipped linears ships input_min/max + output_min/max), the norms,
        // and per_dim_scale.
        const AUDIO_LAYER_VECTORS: [&str; 50] = [
            "feed_forward1.ffw_layer_1.input_max", // 0-d
            "feed_forward1.ffw_layer_1.input_min",
            "feed_forward1.ffw_layer_1.output_max",
            "feed_forward1.ffw_layer_1.output_min",
            "feed_forward1.ffw_layer_2.input_max",
            "feed_forward1.ffw_layer_2.input_min",
            "feed_forward1.ffw_layer_2.output_max",
            "feed_forward1.ffw_layer_2.output_min",
            "feed_forward1.post_layer_norm.weight", // 1-d
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward2.ffw_layer_1.input_max",
            "feed_forward2.ffw_layer_1.input_min",
            "feed_forward2.ffw_layer_1.output_max",
            "feed_forward2.ffw_layer_1.output_min",
            "feed_forward2.ffw_layer_2.input_max",
            "feed_forward2.ffw_layer_2.input_min",
            "feed_forward2.ffw_layer_2.output_max",
            "feed_forward2.ffw_layer_2.output_min",
            "feed_forward2.post_layer_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "lconv1d.linear_end.input_max",
            "lconv1d.linear_end.input_min",
            "lconv1d.linear_end.output_max",
            "lconv1d.linear_end.output_min",
            "lconv1d.linear_start.input_max",
            "lconv1d.linear_start.input_min",
            "lconv1d.linear_start.output_max",
            "lconv1d.linear_start.output_min",
            "lconv1d.pre_layer_norm.weight",
            "norm_out.weight",
            "norm_post_attn.weight",
            "norm_pre_attn.weight",
            "self_attn.k_proj.input_max",
            "self_attn.k_proj.input_min",
            "self_attn.k_proj.output_max",
            "self_attn.k_proj.output_min",
            "self_attn.per_dim_scale",
            "self_attn.post.input_max",
            "self_attn.post.input_min",
            "self_attn.post.output_max",
            "self_attn.post.output_min",
            "self_attn.q_proj.input_max",
            "self_attn.q_proj.input_min",
            "self_attn.q_proj.output_max",
            "self_attn.q_proj.output_min",
            "self_attn.v_proj.input_max",
            "self_attn.v_proj.input_min",
            "self_attn.v_proj.output_max",
            "self_attn.v_proj.output_min",
        ];
        for i in 0..12 {
            let p = format!("audio_tower.layers.{i}");
            for (suffix, ndim) in AUDIO_LAYER_MATRICES {
                names.push((format!("{p}.{suffix}"), ndim));
            }
            for suffix in AUDIO_LAYER_VECTORS {
                let ndim = if suffix.ends_with(".weight") || suffix.ends_with("per_dim_scale") {
                    1
                } else {
                    0
                };
                names.push((format!("{p}.{suffix}"), ndim));
            }
        }
        // The two tower-to-sentence projection heads.
        names.push(("embed_vision.embedding_projection.weight".to_string(), 2));
        names.push(("embed_audio.embedding_projection.weight".to_string(), 2));
        names
    }

    #[test]
    fn eg2_sidecar_name_set_is_963_with_f16_matrices_and_f32_vectors() {
        let names = eg2_sidecar_names();
        // 210 vision (2 + 13 x 16) + 751 audio (7 + 62 x 12) + 2 projections.
        assert_eq!(names.len(), 210 + 751 + 2, "sidecar inventory size");
        let group = |prefix: &str| names.iter().filter(|(n, _)| n.starts_with(prefix)).count();
        assert_eq!(group("vision_tower."), 210, "vision tower tensors");
        assert_eq!(group("audio_tower."), 751, "audio tower tensors");
        assert_eq!(
            names.len() - group("vision_tower.") - group("audio_tower."),
            2
        );
        let mut f16 = 0usize;
        let mut f32 = 0usize;
        for (name, ndim) in &names {
            assert!(
                is_eg2_sidecar_tensor(name),
                "{name} must be in the arch-18 tower group"
            );
            match eg2_sidecar_dtype(name, *ndim) {
                Some(VisionDtype::F16Matrix) => f16 += 1,
                Some(VisionDtype::F32Vector) => f32 += 1,
                None => panic!("{name} must have an arch-18 sidecar dtype"),
            }
        }
        // Matrices: 7 per vision layer + patch_embedder.input_proj; 12 per
        // audio layer + output_proj/input_proj_linear/2 subsample convs; the
        // two projection heads. Everything else is F32: norms, biases,
        // per_dim_scale, the positional table and the 480 0-d clip scalars.
        assert_eq!(f16, 1 + 7 * 16 + 12 * 12 + 4 + 2, "F16 (qt=1) matrix count");
        assert_eq!(f32, 1 + 6 * 16 + 50 * 12 + 3, "F32 (qt=2) vector count");
        assert_eq!(f16 + f32, 963);
    }

    #[test]
    fn eg2_sidecar_spot_dtypes_match_the_rank_rule() {
        // Matrices and conv kernels ride F16.
        for (name, ndim) in [
            ("vision_tower.patch_embedder.input_proj.weight", 2),
            ("vision_tower.encoder.layers.0.mlp.up_proj.linear.weight", 2),
            (
                "vision_tower.encoder.layers.15.self_attn.o_proj.linear.weight",
                2,
            ),
            ("audio_tower.output_proj.weight", 2),
            (
                "audio_tower.subsample_conv_projection.input_proj_linear.weight",
                2,
            ),
            (
                "audio_tower.subsample_conv_projection.layer0.conv.weight",
                4,
            ),
            (
                "audio_tower.subsample_conv_projection.layer1.conv.weight",
                4,
            ),
            ("audio_tower.layers.0.lconv1d.depthwise_conv1d.weight", 3),
            ("audio_tower.layers.11.self_attn.relative_k_proj.weight", 2),
            ("audio_tower.layers.7.lconv1d.linear_start.linear.weight", 2),
            ("embed_vision.embedding_projection.weight", 2),
            ("embed_audio.embedding_projection.weight", 2),
        ] {
            assert_eq!(
                eg2_sidecar_dtype(name, ndim),
                Some(VisionDtype::F16Matrix),
                "{name}"
            );
        }
        // Vectors, the 0-d clip scalars and the positional lookup table ride F32.
        for (name, ndim) in [
            ("vision_tower.encoder.layers.0.input_layernorm.weight", 1),
            ("vision_tower.encoder.layers.3.self_attn.q_norm.weight", 1),
            // 3-D positional lookup table: rank alone would narrow it to F16.
            ("vision_tower.patch_embedder.position_embedding_table", 3),
            ("audio_tower.output_proj.bias", 1),
            (
                "audio_tower.subsample_conv_projection.layer0.norm.weight",
                1,
            ),
            ("audio_tower.layers.0.self_attn.per_dim_scale", 1),
            ("audio_tower.layers.0.norm_out.weight", 1),
            // 0-d clip bounds: scalars, not matrices.
            ("audio_tower.layers.0.self_attn.q_proj.input_min", 0),
            ("audio_tower.layers.5.self_attn.post.output_max", 0),
            ("audio_tower.layers.11.lconv1d.linear_end.input_max", 0),
            (
                "audio_tower.layers.2.feed_forward1.ffw_layer_2.output_min",
                0,
            ),
        ] {
            assert_eq!(
                eg2_sidecar_dtype(name, ndim),
                Some(VisionDtype::F32Vector),
                "{name}"
            );
        }
    }

    /// The arch-18 selector is additive: it claims none of the qwen35 family's
    /// names and changes none of their dtypes, and the trunk's own tensors stay
    /// outside it (arch 18 skips them off the text-quantize path).
    #[test]
    fn eg2_sidecar_selection_leaves_qwen35_and_trunk_names_alone() {
        for (name, ndim) in [
            ("model.visual.patch_embed.proj.weight", 2),
            ("model.visual.pos_embed.weight", 2),
            ("model.visual.blocks.0.norm1.bias", 1),
            ("model.multi_modal_projector.linear.weight", 2),
        ] {
            assert_eq!(eg2_sidecar_dtype(name, ndim), None, "{name}");
        }
        for (name, ndim) in [
            ("language_model.model.embed_tokens.weight", 2),
            ("language_model.model.layers.0.self_attn.q_proj.weight", 2),
            ("vision_tower_extra.weight", 2),
            ("embedded.weight", 2),
        ] {
            assert_eq!(eg2_sidecar_dtype(name, ndim), None, "{name}");
        }
        // The qwen35 selectors are untouched by the arch-18 additions.
        assert_eq!(
            vision_dtype("model.visual.patch_embed.proj.weight", 2),
            Some(VisionDtype::F16Matrix)
        );
        assert_eq!(
            vision_dtype("model.visual.pos_embed.weight", 2),
            Some(VisionDtype::F32Vector)
        );
        assert_eq!(
            vision_dtype("language_model.model.layers.0.self_attn.q_proj.weight", 2),
            None
        );
    }
}
