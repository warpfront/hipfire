// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-VL vision tower + multi-modal projector.
//!
//! Ground truth: transformers `modeling_lfm3_vl.py` / `configuration_lfm3_vl.py`
//! / `image_processing_lfm3_vl.py` (read 2026-10-31, `LiquidAI/LFM3-VL-2B`).
//!
//! The vision tower is a hybrid stack of 14 `Lfm3VlVisionLayer`s. Each layer is
//! either:
//!   * `conv`           — `Lfm3VlVisionConv2d`: in_proj→(B,C,x)→gated=B*x→
//!                        depthwise 2D conv (k=3, pad=1)→out_proj(C*conv)  [NEW kernel]
//!   * `full_attention` — `Lfm3VlVisionAttention`: bidirectional GQA (16Q/8KV,
//!                        head_dim 64) + per-head QK RMSNorm + axial 2D RoPE  [NEW GQA kernel]
//!
//! Layer pattern (config.json): conv,conv,attn,conv,attn,conv,attn,conv,attn,
//! conv,attn,conv,attn,conv  (8 conv / 6 attn). Standard pre-norm residual
//! (operator_norm + ffn_norm — NO sandwich post-norm in the vision tower).
//! FFN is dense SwiGLU. Positional encoding is axial 2D RoPE (theta 1e6).
//!
//! The projector (`Lfm3VlMultiModalProjector`): LayerNorm(1024) → reshape
//! [n_merged, 4*1024] → Linear(4096→4096)+GELU(erf) → Linear(4096→2048).
//! Each 2×2 merge block of vision tokens → one LM token (2048-dim).
//!
//! GPU primitives: `gemm_f16`+`transpose_f32` (linears), `rmsnorm_batched`
//! (vision norms + QK norm), `rope_2d_halfsplit_f32` (axial RoPE, reused from
//! dots.ocr — same [h,w,h,w] quarter-repeat layout), `vit_attention_gqa_f32`
//! (NEW bidirectional GQA), `conv2d_gated_f32` (NEW gated depthwise 2D conv),
//! `layernorm_batched` (projector LayerNorm), `gelu_erf_f32` (NEW erf GELU),
//! `silu_mul_f32` (SwiGLU), `add_inplace_f32` (residuals).

use hip_bridge::{HipError, HipResult};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Vision layer kind (LFM3-VL vision has only conv + full_attention; no sliding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionLayerKind {
    Conv,
    FullAttention,
}

/// Multi-modal projector config (top-level `Lfm3VlConfig` fields).
#[derive(Clone, Debug)]
pub struct ProjectorConfig {
    pub hidden_size: usize,       // projector_hidden_size (4096)
    pub hidden_act: String,       // "gelu" (erf)
    pub layer_norm_eps: f32,      // 1e-6
    pub downsample_factor: usize, // == vision spatial_merge_size (2)
    pub text_hidden_size: usize,  // text_config.hidden_size (2048) — projector output
}

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,         // 1024
    pub intermediate_size: usize,   // 2560
    pub num_hidden_layers: usize,   // 14
    pub num_attention_heads: usize, // 16
    pub num_key_value_heads: usize, // 8
    pub head_dim: usize,            // 64
    pub patch_size: usize,          // 16
    pub temporal_patch_size: usize, // 2
    pub in_channels: usize,         // 3
    pub spatial_merge_size: usize,  // 2
    pub norm_eps: f32,              // 1e-5
    pub rope_theta: f32,            // 1e6 (axial)
    pub layer_types: Vec<VisionLayerKind>,
    pub projector: ProjectorConfig,
    // Special-token ids (top-level Lfm3VlConfig fields) for prompt building.
    // Resolved from config.json rather than the tokenizer because the image
    // placeholder token (``) is a 9-byte special string that is awkward to
    // look up by name.
    pub image_token_id: u32,        // 125030
    pub vision_start_token_id: u32, // 125009
    pub vision_end_token_id: u32,   // 125010
}

impl VisionConfig {
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn is_attention(&self, layer: usize) -> bool {
        self.layer_types[layer] == VisionLayerKind::FullAttention
    }
}

/// Parse the LFM3-VL vision + projector config from the HFQ metadata envelope.
pub fn vision_config_from_hfq(hfq: &HfqFile) -> Option<VisionConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config").or(Some(&meta))?;
    let vc = config.get("vision_config")?;
    let tc = config.get("text_config")?;

    let hidden_size = vc.get("hidden_size")?.as_u64()? as usize;
    let num_attention_heads = vc.get("num_attention_heads")?.as_u64()? as usize;
    let num_key_value_heads = vc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads as u64) as usize;
    let head_dim = vc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden_size / num_attention_heads) as u64) as usize;
    let num_hidden_layers = vc.get("num_hidden_layers")?.as_u64()? as usize;

    // layer_types: default ["conv" if i<2 or i%2 else "full_attention" ...].
    let layer_types: Vec<VisionLayerKind> = vc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|s| match s.as_str().unwrap_or("conv") {
                    "full_attention" | "attention" | "attn" => VisionLayerKind::FullAttention,
                    _ => VisionLayerKind::Conv,
                })
                .collect()
        })
        .unwrap_or_else(|| {
            (0..num_hidden_layers)
                .map(|i| {
                    if i < 2 || i % 2 != 0 {
                        VisionLayerKind::Conv
                    } else {
                        VisionLayerKind::FullAttention
                    }
                })
                .collect()
        });

    let rope_theta = vc
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;

    let projector = ProjectorConfig {
        hidden_size: config
            .get("projector_hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize,
        hidden_act: config
            .get("projector_hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
        layer_norm_eps: config
            .get("projector_layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
        downsample_factor: config
            .get("downsample_factor")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize,
        text_hidden_size: tc.get("hidden_size")?.as_u64()? as usize,
    };

    Some(VisionConfig {
        hidden_size,
        intermediate_size: vc.get("intermediate_size")?.as_u64()? as usize,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        patch_size: vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
        temporal_patch_size: vc
            .get("temporal_patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize,
        in_channels: vc.get("in_channels").and_then(|v| v.as_u64()).unwrap_or(3) as usize,
        spatial_merge_size: vc
            .get("spatial_merge_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize,
        norm_eps: vc.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
        rope_theta,
        layer_types,
        projector,
        image_token_id: config
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(125_030) as u32,
        vision_start_token_id: config
            .get("vision_start_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(125_009) as u32,
        vision_end_token_id: config
            .get("vision_end_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(125_010) as u32,
    })
}

// ─── GPU-side weights ────────────────────────────────────────────────────────

/// Per-layer vision weights. Conv layers carry the gated depthwise conv tensors;
/// attention layers carry the GQA projections + QK RMSNorm. Both share the
/// operator/ffn RMSNorm + dense SwiGLU FFN.
pub struct VisionLayerWeights {
    pub operator_norm: GpuTensor,   // [hidden] F32 RMSNorm
    pub ffn_norm: GpuTensor,        // [hidden] F32 RMSNorm
    pub feed_forward_w1: GpuTensor, // [intermediate, hidden] F16
    pub feed_forward_w2: GpuTensor, // [hidden, intermediate] F16
    pub feed_forward_w3: GpuTensor, // [intermediate, hidden] F16
    // Conv-only:
    pub conv_in_proj: Option<GpuTensor>, // [3*hidden, hidden] F16
    pub conv_out_proj: Option<GpuTensor>, // [hidden, hidden] F16
    pub conv_spatial: Option<GpuTensor>, // [hidden*9] F32 (3x3 depthwise, flattened)
    // Attention-only:
    pub q_proj: Option<GpuTensor>,   // [n_heads*head_dim, hidden] F16
    pub k_proj: Option<GpuTensor>,   // [n_kv*head_dim, hidden] F16
    pub v_proj: Option<GpuTensor>,   // [n_kv*head_dim, hidden] F16
    pub out_proj: Option<GpuTensor>, // [hidden, hidden] F16
    pub q_layernorm: Option<GpuTensor>, // [head_dim] F32 RMSNorm
    pub k_layernorm: Option<GpuTensor>, // [head_dim] F32 RMSNorm
}

pub struct ProjectorWeights {
    pub layer_norm_w: GpuTensor, // [hidden] F32
    pub layer_norm_b: GpuTensor, // [hidden] F32
    pub linear_1_w: GpuTensor,   // [proj_hidden, merge_dim] F16
    pub linear_1_b: GpuTensor,   // [proj_hidden] F32
    pub linear_2_w: GpuTensor,   // [text_hidden, proj_hidden] F16
    pub linear_2_b: GpuTensor,   // [text_hidden] F32
}

pub struct VisionWeights {
    pub patchifier: GpuTensor,     // [hidden, patch_dim] F16 (no bias)
    pub embedding_norm: GpuTensor, // [hidden] F32 RMSNorm
    pub layers: Vec<VisionLayerWeights>,
    pub projector: ProjectorWeights,
}

impl VisionWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.patchifier);
        let _ = gpu.free_tensor(self.embedding_norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.operator_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.feed_forward_w1);
            let _ = gpu.free_tensor(l.feed_forward_w2);
            let _ = gpu.free_tensor(l.feed_forward_w3);
            for t in [
                l.conv_in_proj,
                l.conv_out_proj,
                l.conv_spatial,
                l.q_proj,
                l.k_proj,
                l.v_proj,
                l.out_proj,
                l.q_layernorm,
                l.k_layernorm,
            ]
            .into_iter()
            .flatten()
            {
                let _ = gpu.free_tensor(t);
            }
        }
        let p = self.projector;
        let _ = gpu.free_tensor(p.layer_norm_w);
        let _ = gpu.free_tensor(p.layer_norm_b);
        let _ = gpu.free_tensor(p.linear_1_w);
        let _ = gpu.free_tensor(p.linear_1_b);
        let _ = gpu.free_tensor(p.linear_2_w);
        let _ = gpu.free_tensor(p.linear_2_b);
    }
}

// ─── Weight loading (HFQ) ────────────────────────────────────────────────────
// Mirrors hipfire-arch-qwen35-vl's load_f32_cpu / load_f16_gpu / dequant_hfq4:
// vision 2D weights ship as F16 (qt=1) for gemm_f16; norms / conv-filter /
// biases ship as F16 and are dequantized to F32 at load for the F32 kernels.

// PANIC-1: a missing / wrong-dtype tensor is a loader error, not a bug in the
// caller — a partially-quantized HFQ must fail the load cleanly instead of
// crashing the daemon, so these helpers return Result.
fn load_f32_cpu(hfq: &HfqFile, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| format!("lfm3-vl: vision tensor not found: {name}"))?;
    let mut vals: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        6 | 7 => dequant_hfq4(data, n, info.group_size as usize),
        other => {
            return Err(format!(
                "lfm3-vl: expected F16/F32/HFQ4 for {name}, got qt={other}"
            ))
        }
    };
    vals.truncate(n);
    Ok(vals)
}

fn load_f32_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str, n: usize) -> HipResult<GpuTensor> {
    let vals = load_f32_cpu(hfq, name, n).map_err(|e| HipError::new(999, &e))?;
    gpu.upload_f32(&vals, &[n])
}

fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .ok_or_else(|| HipError::new(999, &format!("lfm3-vl: vision tensor not found: {name}")))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        1 => gpu.upload_raw(data, &[n]),
        6 | 7 => {
            let f32_data = dequant_hfq4(data, n, info.group_size as usize);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        other => Err(HipError::new(
            999,
            &format!(
                "lfm3-vl: {name}: unsupported vision quant_type={other} (expected F16=1, HFQ4=6/7)"
            ),
        )),
    }
}

/// Dequantize HFQ4 blocks to F32 (G256 or G128). Same as qwen35-vl's helper.
fn dequant_hfq4(data: &[u8], n: usize, group_size: usize) -> Vec<f32> {
    let nibble_bytes = group_size / 2;
    let block_size = 8 + nibble_bytes; // 4+4 scale/zero + nibbles
    let mut out = Vec::with_capacity(n);
    let n_groups = n.div_ceil(group_size);
    for g in 0..n_groups {
        let off = g * block_size;
        if off + 8 > data.len() {
            break;
        }
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let zero = f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let nibbles = &data[off + 8..(off + block_size).min(data.len())];
        let base = g * group_size;
        for i in 0..group_size.min(n.saturating_sub(base)) {
            let byte_idx = i / 2;
            if byte_idx >= nibbles.len() {
                break;
            }
            let nibble = if i % 2 == 0 {
                nibbles[byte_idx] & 0xF
            } else {
                nibbles[byte_idx] >> 4
            };
            out.push(scale * nibble as f32 + zero);
        }
    }
    out.truncate(n);
    out
}

pub fn load_vision_weights(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
) -> HipResult<VisionWeights> {
    let h = config.hidden_size;
    let pfx = "model.vision_tower";

    eprintln!("  loading LFM3-VL vision weights (GPU)...");
    let patchifier = load_f16_gpu(hfq, gpu, &format!("{pfx}.patchifier.weight"))?;
    let embedding_norm = load_f32_gpu(hfq, gpu, &format!("{pfx}.embedding_norm.weight"), h)?;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        if i % 4 == 0 {
            eprintln!("  loading vision block {i}/{}...", config.num_hidden_layers);
        }
        let lp = format!("{pfx}.layers.{i}");
        let operator_norm = load_f32_gpu(hfq, gpu, &format!("{lp}.operator_norm.weight"), h)?;
        let ffn_norm = load_f32_gpu(hfq, gpu, &format!("{lp}.ffn_norm.weight"), h)?;
        let feed_forward_w1 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w1.weight"))?;
        let feed_forward_w2 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w2.weight"))?;
        let feed_forward_w3 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w3.weight"))?;

        let (
            conv_in_proj,
            conv_out_proj,
            conv_spatial,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
        ) = if config.is_attention(i) {
            // Attention layer: q/k/v/out_proj + QK RMSNorm.
            (
                None,
                None,
                None,
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.q_proj.weight"),
                )?),
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.k_proj.weight"),
                )?),
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.v_proj.weight"),
                )?),
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.out_proj.weight"),
                )?),
                Some(load_f32_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.q_layernorm.weight"),
                    config.head_dim,
                )?),
                Some(load_f32_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.self_attn.k_layernorm.weight"),
                    config.head_dim,
                )?),
            )
        } else {
            // Conv layer: in_proj / out_proj / spatial_conv (3x3 depthwise).
            // spatial_conv weight is [hidden, 1, 3, 3] → flatten to [hidden*9] F32.
            let conv_spatial = {
                let vals = load_f32_cpu(hfq, &format!("{lp}.conv.spatial_conv.weight"), h * 9)
                    .map_err(|e| HipError::new(999, &e))?;
                gpu.upload_f32(&vals, &[h * 9])?
            };
            (
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.conv.in_proj.weight"),
                )?),
                Some(load_f16_gpu(
                    hfq,
                    gpu,
                    &format!("{lp}.conv.out_proj.weight"),
                )?),
                Some(conv_spatial),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        };

        layers.push(VisionLayerWeights {
            operator_norm,
            ffn_norm,
            feed_forward_w1,
            feed_forward_w2,
            feed_forward_w3,
            conv_in_proj,
            conv_out_proj,
            conv_spatial,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_layernorm,
            k_layernorm,
        });
    }

    eprintln!("  loading LFM3-VL projector...");
    let proj = &config.projector;
    let merge_dim = h * proj.downsample_factor * proj.downsample_factor;
    let projector = ProjectorWeights {
        layer_norm_w: load_f32_gpu(hfq, gpu, "model.multi_modal_projector.layer_norm.weight", h)?,
        layer_norm_b: load_f32_gpu(hfq, gpu, "model.multi_modal_projector.layer_norm.bias", h)?,
        linear_1_w: load_f16_gpu(hfq, gpu, "model.multi_modal_projector.linear_1.weight")?,
        linear_1_b: load_f32_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_1.bias",
            proj.hidden_size,
        )?,
        linear_2_w: load_f16_gpu(hfq, gpu, "model.multi_modal_projector.linear_2.weight")?,
        linear_2_b: load_f32_gpu(
            hfq,
            gpu,
            "model.multi_modal_projector.linear_2.bias",
            proj.text_hidden_size,
        )?,
    };
    let _ = merge_dim; // asserted by linear_1_w shape at gemm time

    Ok(VisionWeights {
        patchifier,
        embedding_norm,
        layers,
        projector,
    })
}

// ─── Axial 2D RoPE cos/sin tables (CPU) ──────────────────────────────────────
// Identical layout to dots.ocr's `build_rope_2d_tables` (and LFM3-VL's HF
// `Lfm3VlVisionRotaryEmbedding`): per-patch [h, w, h, w] quarter-repeat,
// inv_freq[k] = theta^(-2k/(head_dim/2)). LFM3-VL uses theta=1e6 (vs dots.ocr
// 1e4). Consumed by `rope_2d_halfsplit_f32`.

pub fn build_axial_rope_tables(
    grid_h: usize,
    grid_w: usize,
    head_dim: usize,
    spatial_merge_size: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        head_dim % 4,
        0,
        "lfm3-vl: axial RoPE needs head_dim % 4 == 0 (got {head_dim})"
    );
    assert_eq!(grid_h % spatial_merge_size, 0);
    assert_eq!(grid_w % spatial_merge_size, 0);

    let n = grid_h * grid_w;
    let quarter = head_dim / 4;
    let half = head_dim / 2;
    let denom = half as f32;
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|k| theta.powf(-(2.0 * k as f32) / denom))
        .collect();

    let gw = grid_w / spatial_merge_size;
    let sm = spatial_merge_size;
    let mut cos = vec![0.0f32; n * head_dim];
    let mut sin = vec![0.0f32; n * head_dim];

    let mut idx = 0usize;
    for gy in 0..(grid_h / sm) {
        for gx in 0..gw {
            for sy in 0..sm {
                for sx in 0..sm {
                    let hpos = (gy * sm + sy) as f32;
                    let wpos = (gx * sm + sx) as f32;
                    let base = idx * head_dim;
                    for k in 0..quarter {
                        let (hc, hs) = ((hpos * inv_freq[k]).cos(), (hpos * inv_freq[k]).sin());
                        let (wc, ws) = ((wpos * inv_freq[k]).cos(), (wpos * inv_freq[k]).sin());
                        cos[base + k] = hc;
                        cos[base + quarter + k] = wc;
                        cos[base + half + k] = hc;
                        cos[base + half + quarter + k] = wc;
                        sin[base + k] = hs;
                        sin[base + quarter + k] = ws;
                        sin[base + half + k] = hs;
                        sin[base + half + quarter + k] = ws;
                    }
                    idx += 1;
                }
            }
        }
    }
    (cos, sin)
}

// ─── GPU linear helpers ─────────────────────────────────────────────────────
// gemm_f16 produces Y[out_dim, n]; we transpose to [n, out_dim] (row-major).

fn linear_f16_no_bias(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32)?;
    gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n)?;
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32)?;
    gpu.transpose_f32(&yt, &y, out_dim, n)?;
    gpu.free_tensor(yt)?;
    Ok(y)
}

fn linear_f16_bias(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    bias: &GpuTensor,
    out_dim: usize,
    in_dim: usize,
    n: usize,
) -> HipResult<GpuTensor> {
    let y = linear_f16_no_bias(gpu, w, x, out_dim, in_dim, n)?;
    gpu.bias_add_f32(&y, bias, n, out_dim)?;
    Ok(y)
}

// ─── GPU vision forward ──────────────────────────────────────────────────────

/// Run the LFM3-VL vision tower + projector over one image's patches.
///
/// `patches`: `[n, patch_dim]` row-major F32 (n = grid_h*grid_w,
/// patch_dim = in_channels * temporal_patch_size * patch_size² = 1536), in the
/// `(gy, gx, sy, sx)` merge-grouped order produced by `extract_patches`.
///
/// Returns `[n_merged, text_hidden_size]` row-major F32 — one projected visual
/// token per 2×2 merge block, ready to splice into the LM context at
/// `image_token_id` positions.
pub fn vision_forward(
    gpu: &mut Gpu,
    weights: &VisionWeights,
    config: &VisionConfig,
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
) -> HipResult<Vec<f32>> {
    let h = config.hidden_size;
    let n = grid_h * grid_w;
    let head_dim = config.head_dim;
    let n_heads = config.num_attention_heads;
    let n_kv = config.num_key_value_heads;
    let sms = config.spatial_merge_size;
    let eps = config.norm_eps;
    let patch_dim =
        config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
    let t0 = std::time::Instant::now();

    eprintln!("  LFM3-VL vision forward (GPU): {n} patches, {grid_h}x{grid_w} grid");

    // Patchifier: Linear(patch_dim -> h, no bias) → [n, h].
    let x_patches = gpu.upload_f32(patches, &[n * patch_dim])?;
    let mut x = linear_f16_no_bias(gpu, &weights.patchifier, &x_patches, h, patch_dim, n)?;
    gpu.free_tensor(x_patches)?;

    // embedding_norm (RMSNorm over [n, h]).
    let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.rmsnorm_batched(&x, &weights.embedding_norm, &tmp, n, h, eps)?;
    gpu.free_tensor(x)?;
    x = tmp;

    // Axial 2D RoPE cos/sin tables (uploaded once).
    let (cos, sin) = build_axial_rope_tables(grid_h, grid_w, head_dim, sms, config.rope_theta);
    let cos_gpu = gpu.upload_f32(&cos, &[n * head_dim])?;
    let sin_gpu = gpu.upload_f32(&sin, &[n * head_dim])?;

    for li in 0..config.num_hidden_layers {
        let lw = &weights.layers[li];

        // operator_norm (RMSNorm) → tmp
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.rmsnorm_batched(&x, &lw.operator_norm, &tmp, n, h, eps)?;

        let proj_out = if config.is_attention(li) {
            // q/k/v proj (no bias).
            let q =
                linear_f16_no_bias(gpu, lw.q_proj.as_ref().unwrap(), &tmp, config.q_dim(), h, n)?;
            let k = linear_f16_no_bias(
                gpu,
                lw.k_proj.as_ref().unwrap(),
                &tmp,
                config.kv_dim(),
                h,
                n,
            )?;
            let v = linear_f16_no_bias(
                gpu,
                lw.v_proj.as_ref().unwrap(),
                &tmp,
                config.kv_dim(),
                h,
                n,
            )?;
            gpu.free_tensor(tmp)?;

            // Per-head QK RMSNorm (head_dim slices).
            gpu.rmsnorm_batched(
                &q,
                lw.q_layernorm.as_ref().unwrap(),
                &q,
                n * n_heads,
                head_dim,
                eps,
            )?;
            gpu.rmsnorm_batched(
                &k,
                lw.k_layernorm.as_ref().unwrap(),
                &k,
                n * n_kv,
                head_dim,
                eps,
            )?;

            // Axial 2D RoPE on q/k (separate GQA buffers).
            gpu.rope_2d_halfsplit_f32(&q, &k, &cos_gpu, &sin_gpu, n, n_heads, n_kv, head_dim)?;

            // Bidirectional GQA attention.
            let attn_out = gpu.alloc_tensor(&[n * h], DType::F32)?;
            gpu.vit_attention_gqa_f32(&q, &k, &v, &attn_out, n, n_heads, n_kv, head_dim)?;
            gpu.free_tensor(q)?;
            gpu.free_tensor(k)?;
            gpu.free_tensor(v)?;

            // out_proj (no bias).
            let po = linear_f16_no_bias(gpu, lw.out_proj.as_ref().unwrap(), &attn_out, h, h, n)?;
            gpu.free_tensor(attn_out)?;
            po
        } else {
            // in_proj → bcx [n, 3h] (B|C|x).
            let bcx =
                linear_f16_no_bias(gpu, lw.conv_in_proj.as_ref().unwrap(), &tmp, 3 * h, h, n)?;
            gpu.free_tensor(tmp)?;
            // Gated depthwise 2D conv → proj_in [n, h].
            let proj_in = gpu.alloc_tensor(&[n * h], DType::F32)?;
            gpu.conv2d_gated_f32(
                &bcx,
                lw.conv_spatial.as_ref().unwrap(),
                &proj_in,
                n,
                h,
                grid_h,
                grid_w,
                sms,
            )?;
            gpu.free_tensor(bcx)?;
            // out_proj (no bias).
            let po =
                linear_f16_no_bias(gpu, lw.conv_out_proj.as_ref().unwrap(), &proj_in, h, h, n)?;
            gpu.free_tensor(proj_in)?;
            po
        };

        // Residual: x += proj_out.
        gpu.add_inplace_f32(&x, &proj_out)?;
        gpu.free_tensor(proj_out)?;

        // FFN: ffn_norm → SwiGLU (w1 gate, w3 up, silu_mul, w2 down).
        let tmp = gpu.alloc_tensor(&[n * h], DType::F32)?;
        gpu.rmsnorm_batched(&x, &lw.ffn_norm, &tmp, n, h, eps)?;
        let gate = linear_f16_no_bias(
            gpu,
            &lw.feed_forward_w1,
            &tmp,
            config.intermediate_size,
            h,
            n,
        )?;
        let up = linear_f16_no_bias(
            gpu,
            &lw.feed_forward_w3,
            &tmp,
            config.intermediate_size,
            h,
            n,
        )?;
        gpu.free_tensor(tmp)?;
        let act = gpu.alloc_tensor(&[n * config.intermediate_size], DType::F32)?;
        gpu.silu_mul_f32(&gate, &up, &act)?;
        gpu.free_tensor(gate)?;
        gpu.free_tensor(up)?;
        let po = linear_f16_no_bias(
            gpu,
            &lw.feed_forward_w2,
            &act,
            h,
            config.intermediate_size,
            n,
        )?;
        gpu.free_tensor(act)?;

        gpu.add_inplace_f32(&x, &po)?;
        gpu.free_tensor(po)?;
    }

    gpu.hip.device_synchronize()?;
    gpu.free_tensor(cos_gpu)?;
    gpu.free_tensor(sin_gpu)?;
    eprintln!(
        "  LFM3-VL vision tower complete ({:.2}s)",
        t0.elapsed().as_secs_f32()
    );

    // ── Projector: LayerNorm → 2x2 merge rearrange → GELU MLP ──
    let pcfg = &config.projector;
    let pw = &weights.projector;
    let merged_h = grid_h / sms;
    let merged_w = grid_w / sms;
    let n_merged = merged_h * merged_w;
    let merge_dim = h * sms * sms; // 4096

    // LayerNorm all patches (with mean + bias).
    let normed = gpu.alloc_tensor(&[n * h], DType::F32)?;
    gpu.layernorm_batched(
        &x,
        &pw.layer_norm_w,
        &pw.layer_norm_b,
        &normed,
        n,
        h,
        pcfg.layer_norm_eps,
    )?;
    gpu.free_tensor(x)?;

    // 2x2 merge: patches arrive in (gy, gx, sy, sx) merge-grouped order (see
    // extract_patches / conv2d_gated_f32), so each 2x2 block's four patches
    // are already contiguous in raster order. The old "rearrange" copied
    // block i's patches to output row i unchanged — an identity — and paid a
    // download → CPU copy → upload round-trip for it (NITS-1). HF's pixel
    // merger is a plain reshape of the same buffer; feed `normed` to the
    // gemm directly.
    debug_assert_eq!(n, n_merged * sms * sms);
    debug_assert_eq!(merge_dim, h * sms * sms);
    let m1 = linear_f16_bias(
        gpu,
        &pw.linear_1_w,
        &normed,
        &pw.linear_1_b,
        pcfg.hidden_size,
        merge_dim,
        n_merged,
    )?;
    gpu.free_tensor(normed)?;
    gpu.gelu_erf_f32(&m1, &m1, n_merged * pcfg.hidden_size)?;

    let m2 = linear_f16_bias(
        gpu,
        &pw.linear_2_w,
        &m1,
        &pw.linear_2_b,
        pcfg.text_hidden_size,
        pcfg.hidden_size,
        n_merged,
    )?;
    gpu.free_tensor(m1)?;

    let result = gpu.download_f32(&m2)?;
    gpu.free_tensor(m2)?;

    eprintln!(
        "  LFM3-VL vision done: {n_merged} tokens x {} dims ({:.2}s)",
        pcfg.text_hidden_size,
        t0.elapsed().as_secs_f32()
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LFM3-VL vision shape constants (HF `Lfm3VlConfig.vision_config` for
    /// LiquidAI/LFM3-VL-2B). Pin the derived helpers + the projector merge
    /// dim so a config-shape regression (e.g. a wrong head_dim default) is
    /// caught without a GPU/HFQ round-trip.
    fn lfm3_vl_vision_config() -> VisionConfig {
        // 14 layers, alternating conv/full_attention per the HF default
        // (i < 2 or i % 2 != 0 → conv).
        let layer_types = (0..14)
            .map(|i| {
                if i < 2 || i % 2 != 0 {
                    VisionLayerKind::Conv
                } else {
                    VisionLayerKind::FullAttention
                }
            })
            .collect();
        VisionConfig {
            hidden_size: 1024,
            intermediate_size: 2560,
            num_hidden_layers: 14,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 64,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            layer_types,
            projector: ProjectorConfig {
                hidden_size: 4096,
                hidden_act: "gelu".to_string(),
                layer_norm_eps: 1e-6,
                downsample_factor: 2,
                text_hidden_size: 2048,
            },
            image_token_id: 125_030,
            vision_start_token_id: 125_009,
            vision_end_token_id: 125_010,
        }
    }

    #[test]
    fn lfm3_vl_derived_dims_match_hf() {
        let vc = lfm3_vl_vision_config();
        // GQA: 16Q / 8KV, head_dim 64 → q_dim 1024, kv_dim 512.
        assert_eq!(vc.q_dim(), 1024);
        assert_eq!(vc.kv_dim(), 512);
        assert_eq!(vc.num_attention_heads / vc.num_key_value_heads, 2); // groups = 2
    }

    #[test]
    fn lfm3_vl_layer_pattern_is_conv_first_two_then_alternating() {
        let vc = lfm3_vl_vision_config();
        // HF default: layers 0,1 conv; then full_attention on even indices ≥2.
        assert!(!vc.is_attention(0));
        assert!(!vc.is_attention(1));
        assert!(vc.is_attention(2));
        assert!(!vc.is_attention(3));
        assert!(vc.is_attention(4));
        // 14 layers → 6 full_attention (2,4,6,8,10,12) + 8 conv.
        let n_attn = (0..14).filter(|&i| vc.is_attention(i)).count();
        assert_eq!(n_attn, 6);
    }

    #[test]
    fn lfm3_vl_projector_merge_dim_is_4x_hidden() {
        let vc = lfm3_vl_vision_config();
        let sms = vc.spatial_merge_size;
        let merge_dim = vc.hidden_size * sms * sms;
        // 1024 * 2 * 2 = 4096 → matches projector.hidden_size (the 2x2 merge
        // feeds linear_1(4096→4096)). VL-RESIZE-1 / GRID-1 depend on this.
        assert_eq!(merge_dim, 4096);
        assert_eq!(merge_dim, vc.projector.hidden_size);
    }
}
