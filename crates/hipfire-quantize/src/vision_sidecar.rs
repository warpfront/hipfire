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
//! Lives in the lib target (not the binary) so the contract is pinned by
//! `cargo test -p hipfire-quantize --lib vision`. The binary's ingest loop
//! (`pipeline.rs`) and dtype fallback call these; they must agree with
//! `model_filter::should_quantize`, which keeps every vision-group tensor
//! off the text-quantize path.

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
}
