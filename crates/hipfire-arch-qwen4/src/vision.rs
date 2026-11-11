// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash Next image tower admission and the shared Qwen vision implementation.
//!
//! `Qwen/Qwen3.8-Flash-Next` at revision
//! `de4b8e4d43b917e7706784d8bb445c9af86a3540` contains 333 `model.visual.*`
//! tensors, including its merger/projector. Its ViT and image preprocessing
//! use the same equations as Qwen3.5-VL; do not duplicate those GPU kernels.
//! The sidecar retains the upstream names and complete model configuration.

use hip_bridge::{HipError, HipResult};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};
use rdna_compute::Gpu;
use serde_json::Value;
use std::collections::BTreeMap;

pub use qwen35_vl::{VisionConfig as Qwen4VisionConfig, VisionWeights as Qwen4VisionWeights};

pub const IMAGE_TOKEN_ID: u32 = 248056;
pub const VISION_START_TOKEN_ID: u32 = 248053;
pub const VISION_END_TOKEN_ID: u32 = 248054;
pub const VISION_TENSOR_COUNT: usize = 333;
/// Packed F16 matrices and F32 bias/norm vectors; the position table stays
/// on CPU. Backend-specific fc2 row padding is additional resident storage.
pub const DEVICE_WEIGHT_BYTES: u64 = 893_367_232;

/// Maximum admitted patch count, bounded by the shared attention LDS contract.
pub const MAX_IMAGE_PATCHES: usize = 16_128;

/// Conservative additional device bytes across one image request. Includes
/// free-list-retained scratch of every size used by `VisionTowerJob`, not just
/// simultaneously live tensors. We deliberately assume no cross-size reuse.
///
/// Base WMMA: patches(1536), hidden(3*1152), QKV(3456), MLP(4304),
/// rotary(2*36), projected(2560/4) floats per patch. The LDS GEMMs also retain
/// F16 cast buffers at widths 1536, 1152 and padded 4352; the merger's
/// `[patches/4, 4608]` cast reuses the same-size 1152 buffer. WMMA attention
/// adds three F16 `[patches, 16, 80]` Q/K/V packs. Charge these even when an
/// override selects mb8/naive, so the bound covers either backend selection.
/// Non-WMMA keeps its transpose destinations and fourth hidden-width buffer.
/// Layers reuse the same shapes, so the budget does not grow with tower depth.
/// Model weights and caller-owned projected prompt rows are charged separately.
pub fn device_scratch_bytes(max_patches: usize, has_wmma: bool) -> Result<u64, String> {
    if max_patches == 0 || max_patches > MAX_IMAGE_PATCHES || !max_patches.is_multiple_of(4) {
        return Err(
            "qwen4 vision: scratch patch budget must be a multiple of four in 4..=16128".into(),
        );
    }
    let bytes_per_patch = if has_wmma {
        let cast_bytes = 2 * (1536 + 1152 + 4352);
        let attention_pack_bytes = 3 * 16 * 80 * 2;
        53_856 + cast_bytes + attention_pack_bytes
    } else {
        92_064
    };
    Ok(max_patches as u64 * bytes_per_patch)
}

/// Parse only the admitted image tower profile. In particular, a 27B sidecar
/// with its 5120-wide projector must not be attached to the 2560-wide trunk.
pub fn config_from_hfq(hfq: &HfqFile) -> Result<Qwen4VisionConfig, String> {
    if hfq.arch_id != crate::config::ARCH_ID {
        return Err(format!(
            "qwen4 vision: sidecar arch_id must be 16, got {}",
            hfq.arch_id
        ));
    }
    let metadata: Value = serde_json::from_str(&hfq.metadata_json)
        .map_err(|error| format!("qwen4 vision: invalid metadata: {error}"))?;
    config_from_json(
        metadata
            .get("config")
            .ok_or("qwen4 vision: missing config")?,
    )
}

/// CPU-only source admission, also used by the sidecar exporter before download.
pub fn config_from_json(config: &Value) -> Result<Qwen4VisionConfig, String> {
    if config.get("model_type").and_then(Value::as_str) != Some("qwen4_exp") {
        return Err("qwen4 vision: expected qwen4_exp source config".into());
    }
    if config.get("language_model_only").and_then(Value::as_bool) != Some(false) {
        return Err("qwen4 vision: source must declare language_model_only=false".into());
    }
    for (key, expected) in [
        ("image_token_id", IMAGE_TOKEN_ID),
        ("vision_start_token_id", VISION_START_TOKEN_ID),
        ("vision_end_token_id", VISION_END_TOKEN_ID),
    ] {
        require_number(config, key, expected as u64)?;
    }
    let text = config
        .get("text_config")
        .ok_or("qwen4 vision: missing text_config")?;
    require_number(text, "hidden_size", 2560)?;
    let rope = text
        .get("rope_parameters")
        .ok_or("qwen4 vision: missing text mRoPE configuration")?;
    if rope.get("mrope_interleaved").and_then(Value::as_bool) != Some(true)
        || rope.get("mrope_section") != Some(&serde_json::json!([11, 11, 10]))
        || rope.get("rope_theta").and_then(Value::as_f64) != Some(10_000_000.0)
    {
        return Err("qwen4 vision: unsupported text mRoPE configuration".into());
    }
    let vc = config
        .get("vision_config")
        .ok_or("qwen4 vision: missing vision_config")?;
    for (key, expected) in [
        ("hidden_size", 1152),
        ("num_heads", 16),
        ("depth", 27),
        ("intermediate_size", 4304),
        ("patch_size", 16),
        ("temporal_patch_size", 2),
        ("out_hidden_size", 2560),
        ("spatial_merge_size", 2),
        ("num_position_embeddings", 2304),
        ("in_channels", 3),
    ] {
        require_number(vc, key, expected)?;
    }
    if vc.get("hidden_act").and_then(Value::as_str) != Some("gelu_pytorch_tanh") {
        return Err("qwen4 vision: unsupported tower activation".into());
    }
    if !vc
        .get("deepstack_visual_indexes")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        return Err("qwen4 vision: deepstack visual features are not supported".into());
    }
    let theta = vc
        .get("rope_parameters")
        .and_then(|v| v.get("rope_theta"))
        .or_else(|| vc.get("rope_theta"))
        .and_then(Value::as_f64)
        .unwrap_or(10000.0);
    if theta != 10000.0 {
        return Err(format!("qwen4 vision: unsupported rotary theta {theta}"));
    }
    Ok(Qwen4VisionConfig {
        hidden_size: 1152,
        num_heads: 16,
        head_dim: 72,
        num_layers: 27,
        mlp_dim: 4304,
        patch_size: 16,
        temporal_patch_size: 2,
        out_hidden_size: 2560,
        spatial_merge_size: 2,
        num_position_embeddings: 2304,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    })
}

fn require_number(value: &Value, key: &str, expected: u64) -> Result<(), String> {
    if value.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(format!(
            "qwen4 vision: {key} must be {expected}, got {}",
            value.get(key).unwrap_or(&Value::Null)
        ));
    }
    Ok(())
}

/// Check every tensor before any GPU allocation, including the output
/// projector, byte lengths and supported encodings. This admission accepts
/// F16/F32 sidecars only; quantized tower admission needs its own evidence.
pub fn validate_hfq(hfq: &HfqFile, config: &Qwen4VisionConfig) -> Result<(), String> {
    let source = config_from_hfq(hfq)?;
    if profile_dimensions(config) != profile_dimensions(&source)
        || config.rope_theta != source.rope_theta
        || config.norm_eps != source.norm_eps
    {
        return Err("qwen4 vision: supplied geometry disagrees with sidecar config".into());
    }
    validate_tensor_infos(hfq.tensors(), config)?;
    for info in hfq.tensors() {
        if hfq.tensor_data(&info.name).is_none() {
            return Err(format!(
                "qwen4 vision: missing or truncated mapped payload {}",
                info.name
            ));
        }
    }
    Ok(())
}

fn profile_dimensions(config: &Qwen4VisionConfig) -> [usize; 11] {
    [
        config.hidden_size,
        config.num_heads,
        config.head_dim,
        config.num_layers,
        config.mlp_dim,
        config.patch_size,
        config.temporal_patch_size,
        config.out_hidden_size,
        config.spatial_merge_size,
        config.num_position_embeddings,
        3,
    ]
}

fn expected_shapes(config: &Qwen4VisionConfig) -> BTreeMap<String, Vec<u32>> {
    let h = config.hidden_size as u32;
    let m = config.mlp_dim as u32;
    let merge = h * (config.spatial_merge_size * config.spatial_merge_size) as u32;
    let mut shapes = BTreeMap::new();
    for (name, shape) in [
        (
            "patch_embed.proj.weight",
            vec![
                h,
                3,
                config.temporal_patch_size as u32,
                config.patch_size as u32,
                config.patch_size as u32,
            ],
        ),
        ("patch_embed.proj.bias", vec![h]),
        (
            "pos_embed.weight",
            vec![config.num_position_embeddings as u32, h],
        ),
        ("merger.norm.weight", vec![h]),
        ("merger.norm.bias", vec![h]),
        ("merger.linear_fc1.weight", vec![merge, merge]),
        ("merger.linear_fc1.bias", vec![merge]),
        (
            "merger.linear_fc2.weight",
            vec![config.out_hidden_size as u32, merge],
        ),
        (
            "merger.linear_fc2.bias",
            vec![config.out_hidden_size as u32],
        ),
    ] {
        shapes.insert(format!("model.visual.{name}"), shape);
    }
    for layer in 0..config.num_layers {
        for (name, shape) in [
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("attn.qkv.weight", vec![3 * h, h]),
            ("attn.qkv.bias", vec![3 * h]),
            ("attn.proj.weight", vec![h, h]),
            ("attn.proj.bias", vec![h]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
            ("mlp.linear_fc1.weight", vec![m, h]),
            ("mlp.linear_fc1.bias", vec![m]),
            ("mlp.linear_fc2.weight", vec![h, m]),
            ("mlp.linear_fc2.bias", vec![h]),
        ] {
            shapes.insert(format!("model.visual.blocks.{layer}.{name}"), shape);
        }
    }
    shapes
}

/// Validate a planned F16/F32 sidecar inventory before fetching source payloads.
pub fn validate_tensor_infos(
    infos: &[HfqTensorInfo],
    config: &Qwen4VisionConfig,
) -> Result<(), String> {
    let mut expected = expected_shapes(config);
    if infos.len() != VISION_TENSOR_COUNT {
        return Err(format!(
            "qwen4 vision: expected {VISION_TENSOR_COUNT} tensors, got {}",
            infos.len()
        ));
    }
    for info in infos {
        let shape = expected
            .remove(&info.name)
            .ok_or_else(|| format!("qwen4 vision: unexpected or duplicate tensor {}", info.name))?;
        if info.shape != shape {
            return Err(format!(
                "qwen4 vision: {} has shape {:?}, expected {:?}",
                info.name, info.shape, shape
            ));
        }
        let bytes = match info.quant_type {
            1 => 2,
            2 => 4,
            qt => {
                return Err(format!(
                    "qwen4 vision: {} has unsupported quant type {qt}; require F16/F32",
                    info.name
                ))
            }
        };
        let size = shape
            .iter()
            .try_fold(bytes, |n: usize, &d| n.checked_mul(d as usize))
            .ok_or_else(|| format!("qwen4 vision: {} byte size overflow", info.name))?;
        if info.data_size != size {
            return Err(format!(
                "qwen4 vision: {} has {} bytes, expected {size}",
                info.name, info.data_size
            ));
        }
    }
    Ok(())
}

/// Upload the admitted tower through the shared Qwen vision loader.
/// Flash Next currently requires resident matrices, including for its
/// `auto` and `on` modes; the Qwen3.5/VL streamed route remains independent.
/// Release with `Qwen4VisionWeights::free_gpu` at model unload.
pub fn load_weights(
    hfq: &HfqFile,
    config: &Qwen4VisionConfig,
    gpu: &mut Gpu,
) -> HipResult<Qwen4VisionWeights> {
    validate_hfq(hfq, config).map_err(|error| HipError::new(1, &error))?;
    qwen35_vl::load_vision_weights(hfq, config, gpu, qwen35_vl::TowerResidency::Resident)
}

/// Encode a resized, normalized RGB CHW image. Returns row-major F32 visual
/// embeddings of shape `[height / 32 * width / 32, 2560]` on the host, ready
/// for the language carrier's prompt-row splice.
pub fn encode(
    gpu: &mut Gpu,
    weights: &Qwen4VisionWeights,
    config: &Qwen4VisionConfig,
    chw: &[f32],
    height: usize,
    width: usize,
) -> HipResult<Vec<f32>> {
    let factor = config.patch_size * config.spatial_merge_size;
    let pixels = height
        .checked_mul(width)
        .ok_or_else(|| HipError::new(1, "qwen4 vision: image dimensions overflow"))?;
    if factor == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(factor)
        || !width.is_multiple_of(factor)
        || pixels > 4_128_768
        || chw.len() != pixels * 3
    {
        return Err(HipError::new(
            1,
            "qwen4 vision: invalid resized RGB image dimensions",
        ));
    }
    let patches = crate::image::extract_patches(
        chw,
        3,
        height,
        width,
        config.patch_size,
        config.temporal_patch_size,
        config.spatial_merge_size,
    );
    qwen35_vl::vision_forward(
        gpu,
        weights,
        config,
        &patches,
        height / config.patch_size,
        width / config.patch_size,
    )
}

#[cfg(test)]
#[path = "vision/tests.rs"]
mod tests;
