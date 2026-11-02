// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vision path: image bytes → SigLIP-style tower → merged 1024-d tokens.
//!
//! The tower itself is `hipfire-arch-qwen35-vl`'s (patch-embed GEMM,
//! interpolated learned pos-embed, 2D rope, ViT blocks, LayerNorm + 2×2
//! merge + GELU MLP merger) — the pplx tower is the same architecture at
//! hidden 768 / depth 12, and nothing in that code is size-pinned. This
//! module only owns the **loading** of `visual.*` tensors out of a
//! safetensors dir / raw `.hfq` (the vl crate's loader is HFQ- and
//! `model.visual.`-prefixed) and the request-side preprocessing +
//! `vision_forward` call.
//!
//! Known numerical deviation, inherited from the shared tower: the merger
//! GELU is the tanh approximation (`gelu_tanh_f32`) where HF
//! `Qwen3_5VisionPatchMerger` uses exact-erf `nn.GELU()`. Same tradeoff
//! production Qwen3.5-VL serving makes; ~1e-3 relative on merged tokens.

use crate::config::EmbedConfig;
use crate::names::{self, TensorBytes};
use half::f16;
use hipfire_arch_qwen35_vl::qwen35_vl::{
    vision_forward, TowerMats, VisionConfig, VisionLayerMats, VisionLayerWeights, VisionWeights,
};
use hipfire_runtime::loader_api::ModelSource;
use rdna_compute::Gpu;

/// Vision tensors are kept raw by the quantizer (F16 under
/// `--include-vision`), but a compressed tower is decoded the same way the
/// trunk is, so this path costs nothing extra to be format-agnostic.
fn to_f32(t: &TensorBytes<'_>, name: &str) -> Result<Vec<f32>, String> {
    let n = t.numel();
    if t.quant_type != 0xFF {
        let v =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(t.quant_type, &t.bytes, n);
        if v.len() != n {
            return Err(format!(
                "qwen35-embed: vision tensor `{name}`: dequantized {} values for {n} elements",
                v.len()
            ));
        }
        return Ok(v);
    }
    let b = &t.bytes;
    // Shared byte-length guard: a declared shape exceeding the stored bytes
    // is a malformed-artifact error, not a panic on an unchecked slice.
    let need_bytes = |w: usize| -> Result<(), String> {
        if b.len() < n * w {
            return Err(format!(
                "qwen35-embed: vision tensor `{name}`: {} bytes < {}",
                b.len(),
                n * w
            ));
        }
        Ok(())
    };
    match t.dtype.as_str() {
        "F32" => {
            need_bytes(4)?;
            Ok(b[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F16" => {
            need_bytes(2)?;
            Ok(b[..n * 2]
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        "BF16" => {
            need_bytes(2)?;
            Ok(b[..n * 2]
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        other => Err(format!(
            "qwen35-embed: vision tensor `{name}` dtype {other} (F16/F32/BF16 only)"
        )),
    }
}

/// Whether the vision tower's weights are placed in host RAM (GPU-addressable,
/// read over PCIe) instead of device memory.
///
/// The tower is ~250 MB of the model's ~740 MB, and it is only touched by an
/// image request, so on a card where the resident desktop leaves less than the
/// full model's worth of VRAM it is the right component to spill: text-only
/// embedding then loads and runs with the whole tower's budget free. This is
/// the same tradeoff `HIPFIRE_OFFLOAD_EXEC=pcie` and `HIPFIRE_GPU_LAYER_BUDGET`
/// make for the text arches (`docs/plans/partial-gpu-offload-design.md`), scoped
/// to the one component this arch can offload whole.
///
/// Evaluated ONCE per load, by [`load_vision_weights`] before any tower tensor
/// uploads: deciding per upload would flip mid-tower as the earlier resident
/// tensors consume the very headroom being measured, and land the tower half
/// resident, half host-mapped. Correct either way (both placements are plain
/// readable memory to the kernels) but a placement nobody chose and nobody can
/// reproduce from the config.
///
/// Unset, the decision is automatic: the tower stays device-resident only
/// when it fits in the VRAM free at load (plus a 64 MB margin), and is
/// host-mapped otherwise. `HIPFIRE_EMBED_VISION_OFFLOAD=1` forces the
/// host-mapped tower, `=0` forces it resident.
fn vision_offload(gpu: &Gpu, cfg: &EmbedConfig) -> bool {
    // Explicit override wins.
    if let Ok(v) = hipfire_config::developer_var("HIPFIRE_EMBED_VISION_OFFLOAD") {
        match v.to_ascii_lowercase().as_str() {
            "1" | "on" | "true" => return true,
            "0" | "off" | "false" => return false,
            _ => {}
        }
    }
    // Auto: keep the tower device-resident only when it comfortably fits in
    // what is free right now (after the trunk is already up), so a card with a
    // busy desktop degrades to PCIe reads instead of failing the load or
    // OOMing on the first image. An unreadable VRAM figure assumes room — the
    // resident path is faster and the failure mode is a clean alloc error.
    // Capped sizing query (UMA-safe clamp, then the HIPFIRE_EMU cap).
    let Some((free, _)) = gpu.device_mem_info().ok() else {
        return false;
    };
    free < tower_resident_bytes(cfg).saturating_add(RESIDENCY_MARGIN_BYTES)
}

/// Headroom kept on top of the tower's own bytes for the forward's scratch and
/// for anything else loading concurrently.
const RESIDENCY_MARGIN_BYTES: usize = 64 << 20;

/// Bytes the vision tower occupies device-resident (f16 weights + f32 norms),
/// derived from `vision_config` — no checkpoint reads.
fn tower_resident_bytes(cfg: &EmbedConfig) -> usize {
    let Some(v) = cfg.vision.as_ref() else {
        return 0;
    };
    let h = v.hidden_size;
    let mlp = v.mlp_dim;
    let merge = h * v.spatial_merge_size * v.spatial_merge_size;
    let patch = 3 * v.temporal_patch_size * v.patch_size * v.patch_size;
    let per_layer = 3 * h * h + h * h + mlp * h + h * mlp;
    let weights = v.num_layers * per_layer + h * patch + merge * merge + v.out_hidden_size * merge;
    // Two norm/bias vectors per block plus the merger's, f32 (upper bound).
    let vectors = (v.num_layers * 2 + 1) * (2 * h + 3 * h + mlp) * 4;
    weights * 2 + vectors
}

/// Upload f16 weight words either device-resident or host-mapped (`offload`
/// is the load-time placement decision — see [`vision_offload`]).
fn put_f16(
    gpu: &mut Gpu,
    offload: bool,
    words: &[u16],
    shape: &[usize],
) -> Result<rdna_compute::GpuTensor, String> {
    if offload {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        gpu.upload_raw_host_mapped(&bytes, shape)
            .map_err(|e| format!("qwen35-embed: host-mapped f16 upload: {e:?}"))
    } else {
        gpu.upload_f16_bits(words, shape)
            .map_err(|e| format!("qwen35-embed: f16 upload: {e:?}"))
    }
}

/// Upload an f32 vector either device-resident or host-mapped.
fn put_f32(
    gpu: &mut Gpu,
    offload: bool,
    v: &[f32],
    shape: &[usize],
) -> Result<rdna_compute::GpuTensor, String> {
    if offload {
        gpu.upload_f32_host(v, shape)
            .map_err(|e| format!("qwen35-embed: host-mapped f32 upload: {e:?}"))
    } else {
        gpu.upload_f32(v, shape)
            .map_err(|e| format!("qwen35-embed: f32 upload: {e:?}"))
    }
}

fn up_lin(
    gpu: &mut Gpu,
    src: &ModelSource,
    offload: bool,
    name: &str,
) -> Result<rdna_compute::GpuTensor, String> {
    let t = names::read(src, name, true)?;
    let n = t.numel();
    let words: Vec<u16> = to_f32(&t, name)?
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect();
    let rows = t.shape.first().copied().unwrap_or(0);
    // N-D → 2-D row-major flatten: the 5-D conv-style patch-embed weight
    // `[out, in, T, h, w]` becomes `[out, in·T·h·w]`, which is exactly the
    // GEMM form for the (C, T, h, w) channel-outer patch layout.
    let cols: usize = if t.shape.len() >= 2 {
        t.shape[1..].iter().product()
    } else {
        1
    };
    debug_assert_eq!(words.len(), n);
    names::release(src, name, true);
    put_f16(gpu, offload, &words, &[rows, cols])
        .map_err(|e| format!("qwen35-embed: upload vision `{name}`: {e:?}"))
}

fn up_vec(
    gpu: &mut Gpu,
    src: &ModelSource,
    offload: bool,
    name: &str,
    n: usize,
) -> Result<rdna_compute::GpuTensor, String> {
    let t = names::read(src, name, true)?;
    let v = to_f32(&t, name)?;
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: vision `{name}` is {} elems, expected {n}",
            v.len()
        ));
    }
    drop(t);
    names::release(src, name, true);
    put_f32(gpu, offload, &v, &[n])
        .map_err(|e| format!("qwen35-embed: upload vision `{name}`: {e:?}"))
}

fn host_vec(src: &ModelSource, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let t = names::read(src, name, true)?;
    let v = to_f32(&t, name)?;
    drop(t);
    names::release(src, name, true);
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: vision `{name}` is {} elems, expected {n}",
            v.len()
        ));
    }
    Ok(v)
}

/// Load the tower into the qwen35-vl weight set (see module docs for why
/// this isn't `qwen35_vl::load_vision_weights`).
pub fn load_vision_weights(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &EmbedConfig,
) -> Result<VisionWeights, String> {
    let vis: &VisionConfig = cfg
        .vision
        .as_ref()
        .ok_or("qwen35-embed: vision weights present but no vision_config")?;
    // One placement decision for the whole tower, taken with the trunk
    // already resident and none of the tower up yet — the only moment the
    // free-VRAM reading answers the question being asked.
    let offload = vision_offload(gpu, cfg);
    let side = (vis.num_position_embeddings as f64).sqrt().round() as usize;
    if side == 0 || side * side != vis.num_position_embeddings {
        return Err(format!(
            "qwen35-embed: num_position_embeddings ({}) must be a perfect square",
            vis.num_position_embeddings
        ));
    }
    let h = vis.hidden_size;
    let mut layers = Vec::with_capacity(vis.num_layers);
    let mut layer_mats = Vec::with_capacity(vis.num_layers);
    for i in 0..vis.num_layers {
        let p = format!("blocks.{i}");
        let norm1_w = up_vec(gpu, src, offload, &format!("{p}.norm1.weight"), h)?;
        let norm1_b = up_vec(gpu, src, offload, &format!("{p}.norm1.bias"), h)?;
        let qkv_w = up_lin(gpu, src, offload, &format!("{p}.attn.qkv.weight"))?;
        let qkv_b = up_vec(gpu, src, offload, &format!("{p}.attn.qkv.bias"), 3 * h)?;
        let proj_w = up_lin(gpu, src, offload, &format!("{p}.attn.proj.weight"))?;
        let proj_b = up_vec(gpu, src, offload, &format!("{p}.attn.proj.bias"), h)?;
        let norm2_w = up_vec(gpu, src, offload, &format!("{p}.norm2.weight"), h)?;
        let norm2_b = up_vec(gpu, src, offload, &format!("{p}.norm2.bias"), h)?;
        let fc1_w = up_lin(gpu, src, offload, &format!("{p}.mlp.linear_fc1.weight"))?;
        let fc1_b = up_vec(
            gpu,
            src,
            offload,
            &format!("{p}.mlp.linear_fc1.bias"),
            vis.mlp_dim,
        )?;
        let fc2_w = up_lin(gpu, src, offload, &format!("{p}.mlp.linear_fc2.weight"))?;
        let fc2_b = up_vec(gpu, src, offload, &format!("{p}.mlp.linear_fc2.bias"), h)?;
        layers.push(VisionLayerWeights {
            norm1_w,
            norm1_b,
            qkv_b,
            proj_b,
            norm2_w,
            norm2_b,
            fc1_b,
            fc2_b,
            // Packed (not K-padded): the shared tower routes a packed fc2
            // with `mlp_dim % 64 != 0` through `gemm_f16_wmma_mb8`.
            fc2_ld: vis.mlp_dim,
        });
        layer_mats.push(VisionLayerMats {
            qkv_w,
            proj_w,
            fc1_w,
            fc2_w,
        });
    }
    let merge_dim = h * vis.spatial_merge_size * vis.spatial_merge_size;
    let patch_embed_w = up_lin(gpu, src, offload, "patch_embed.proj.weight")?;
    let patch_embed_b = up_vec(gpu, src, offload, "patch_embed.proj.bias", h)?;
    let pos_embed = host_vec(src, "pos_embed.weight", vis.num_position_embeddings * h)?;
    let merger_norm_w = up_vec(gpu, src, offload, "merger.norm.weight", h)?;
    let merger_norm_b = up_vec(gpu, src, offload, "merger.norm.bias", h)?;
    let merger_fc1_w = up_lin(gpu, src, offload, "merger.linear_fc1.weight")?;
    let merger_fc1_b = up_vec(gpu, src, offload, "merger.linear_fc1.bias", merge_dim)?;
    let merger_fc2_w = up_lin(gpu, src, offload, "merger.linear_fc2.weight")?;
    let merger_fc2_b = up_vec(
        gpu,
        src,
        offload,
        "merger.linear_fc2.bias",
        vis.out_hidden_size,
    )?;
    Ok(VisionWeights {
        patch_embed_w,
        patch_embed_b,
        pos_embed,
        layers,
        merger_norm_w,
        merger_norm_b,
        merger_fc1_b,
        merger_fc2_b,
        mats: TowerMats::Resident {
            layers: layer_mats,
            merger_fc1_w,
            merger_fc2_w,
        },
    })
}

/// One merged image token per `spatial_merge_size²` patch group.
pub fn merged_grid(cfg: &EmbedConfig, grid_h: usize, grid_w: usize) -> (usize, usize) {
    let m = cfg
        .vision
        .as_ref()
        .map(|v| v.spatial_merge_size)
        .unwrap_or(2);
    (grid_h / m, grid_w / m)
}

/// Decode + preprocess + encode an image to merged `[n_merged,
/// out_hidden]` host embeds. `patch_size`/`merge` come from the vision
/// config; preprocessing (smart-resize, `/127.5 − 1` normalize, 2×2-grouped
/// patch extraction) is `hipfire-arch-qwen35-vl::image`.
///
/// Returns `(merged_embeds, grid_h, grid_w)` where the grid is the
/// **pre-merge patch grid** (`img / patch_size`) — the contract
/// `hipfire_arch_qwen35_vl::mrope::build_mrope_positions` divides by
/// `spatial_merge_size` itself. The merged token count is
/// `merged.len() / out_hidden_size`.
pub fn encode_image(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    cfg: &EmbedConfig,
    image_bytes: &[u8],
) -> Result<(Vec<f32>, usize, usize), String> {
    let vis = cfg
        .vision
        .as_ref()
        .ok_or("qwen35-embed: image input on a text-only checkpoint")?;
    // `load_and_preprocess_from_bytes` returns (CHW, **pixel** h, pixel w) —
    // the pixel grid is a multiple of `patch_size * spatial_merge_size`. The
    // tower consumes patch vectors + a PATCH grid, so run `extract_patches`
    // and divide by `patch_size` (the qwen35-vl serve path does the same:
    // print/`hipfire-generate/src/vision.rs` computes grid = img/patch_size).
    // Passing the pixel dims as the grid was the first-cut bug here: a 64×64
    // input became a 256×256 grid (65,536 patches) and OOM'd.
    let (pixels, img_h, img_w) = hipfire_arch_qwen35_vl::image::load_and_preprocess_from_bytes(
        image_bytes,
        vis.patch_size,
        vis.spatial_merge_size,
    )?;
    let patches = hipfire_arch_qwen35_vl::image::extract_patches(
        &pixels,
        3,
        img_h,
        img_w,
        vis.patch_size,
        vis.temporal_patch_size,
        vis.spatial_merge_size,
    );
    let grid_h = img_h / vis.patch_size;
    let grid_w = img_w / vis.patch_size;
    if hipfire_config::developer_var("HIPFIRE_EMBED_DEBUG")
        .ok()
        .as_deref()
        == Some("1")
    {
        // Physical (raw hipMemGetInfo) diagnostic, never the emulated capacity.
        let (f, t) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        eprintln!(
            "  [embed-debug] vision: {}x{} px -> {}x{} grid ({} patches), weights loaded; physical VRAM {} MB free / {} MB",
            img_w, img_h, grid_w, grid_h, grid_h * grid_w, f / (1024*1024), t / (1024*1024)
        );
    }
    let merged = vision_forward(gpu, weights, vis, &patches, grid_h, grid_w)
        .map_err(|e| format!("qwen35-embed: vision forward: {e:?}"))?;
    let (mh, mw) = merged_grid(cfg, grid_h, grid_w);
    let n_merged = mh * mw;
    if merged.len() != n_merged * vis.out_hidden_size {
        return Err(format!(
            "qwen35-embed: vision tower returned {} floats, expected {n_merged}×{}",
            merged.len(),
            vis.out_hidden_size
        ));
    }
    Ok((merged, grid_h, grid_w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{config_from_metadata, parse_config};

    fn pplx_cfg() -> crate::config::EmbedConfig {
        // The composite shape both source kinds produce (config under a
        // `config` key) — the same string the config module tests.
        let v: serde_json::Value = serde_json::from_str(
            r#"{"config":{"architectures":["Qwen3_5Model"],"model_type":"qwen3_5",
                 "text_config":{"hidden_size":1024,"num_hidden_layers":2,
                   "num_attention_heads":8,"num_key_value_heads":2,"head_dim":256,
                   "intermediate_size":3584,"vocab_size":10,"rms_norm_eps":1e-6,
                   "hidden_act":"silu","attn_output_gate":true,
                   "linear_num_key_heads":16,"linear_num_value_heads":16,
                   "linear_key_head_dim":128,"linear_value_head_dim":128,
                   "linear_conv_kernel_dim":4,
                   "rope_parameters":{"rope_theta":1e7,"partial_rotary_factor":0.25,
                                      "mrope_section":[11,11,10]},
                   "layer_types":["linear_attention","full_attention"]},
                 "vision_config":{"hidden_size":768,"num_heads":12,"depth":2,
                   "intermediate_size":3072,"out_hidden_size":1024,"patch_size":16,
                   "temporal_patch_size":2,"spatial_merge_size":2,
                   "num_position_embeddings":2304}}}"#,
        )
        .unwrap();
        parse_config(config_from_metadata(&v)).unwrap()
    }

    #[test]
    fn merged_grid_divides_by_spatial_merge_size() {
        let cfg = pplx_cfg();
        // A 16x16 patch grid merges 2x2 -> 8x8 = 64 tokens (the value the
        // gfx1101 tower smoke produced for a 256x256-pixel input).
        assert_eq!(merged_grid(&cfg, 16, 16), (8, 8));
        assert_eq!(merged_grid(&cfg, 1, 1), (0, 0)); // partial group drops
        assert_eq!(merged_grid(&cfg, 32, 48), (16, 24));
    }
}
