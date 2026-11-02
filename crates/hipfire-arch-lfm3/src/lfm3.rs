// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 weights + decode state.
//!
//! Mirrors `hipfire-arch-lfm2moe`'s loader (shared `WeightTensor`, `KvCache`,
//! conv-state ring buffer) with the LFM3 deltas:
//!   * a per-layer mixer split into conv / full_attention / sliding_attention
//!     (full and sliding share the same `AttnWeights` shape; the forward
//!     consults `cfg.mixer(l)` for the window + RoPE-theta choice),
//!   * a `gate_proj` per attention layer (sigmoid output gate, `[n_heads, hidden]`),
//!   * optional `post_operator_norm` / `post_ffn_norm` (sandwich norm),
//!   * dense SwiGLU FFN only (no MoE in the text backbone),
//!   * LFM3 tensor names: `self_attn.q_norm` / `k_norm` (NOT LFM2's
//!     `q_layernorm` / `k_layernorm`).
//!
//! Tensor-name prefix is `model.` for the text-only `LiquidAI/LFM3-1.7B`.
//! The Omni text backbone uses `model.language_model.` — handled when the
//! multimodal load path lands (Phase 4). lm_head is tied to embed_tokens.

use crate::config::{Lfm3Config, MixerKind};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, KvCache, WeightTensor};
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::safetensors_source::{bf16_bytes_to_f16, source_bytes_to_f32_vec};
use hipfire_runtime::{screen_weight_tensor, MmqScreenable};
use rdna_compute::{DType, Gpu, GpuTensor};

/// Batched-prefill flash sub-batch size at full context — the trailing factor of
/// the `flash_partials` allocation (mirrors cohere2moe). LFM3 decode is
/// single-token, so this only bounds the prefill flash sub-batch.
const FLASH_PREFILL_SUBBATCH: usize = 64;

// ───────────────────────── HFQ load helpers ─────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    if let Some((info, data)) = hfq.tensor_data_vec(name) {
        return Ok((info.quant_type, data));
    }
    // LFM3-VL nests the text backbone under `model.language_model.*` (the VL
    // checkpoint's `Lfm3VlModel.language_model`). Text-only LFM3-1.7B uses
    // `model.*` directly, so this fallback is a no-op there; for VL it resolves
    // `model.embed_tokens.weight` → `model.language_model.embed_tokens.weight`.
    // LFM3-Audio nests the text backbone under `lfm.model.*` (the audio
    // checkpoint's `Lfm3AudioModel.lfm.model`) — second fallback.
    if let Some(stripped) = name.strip_prefix("model.") {
        let prefixed = format!("model.language_model.{stripped}");
        if let Some((info, data)) = hfq.tensor_data_vec(&prefixed) {
            return Ok((info.quant_type, data));
        }
        let prefixed = format!("lfm.model.{stripped}");
        if let Some((info, data)) = hfq.tensor_data_vec(&prefixed) {
            return Ok((info.quant_type, data));
        }
    }
    Err(format!("lfm3: tensor not found in HFQ: {name}"))
}

/// Load a 1D/raw F16/F32/Q8 vector → F32 GpuTensor with the given shape.
/// LFM3 uses STANDARD RMSNorm (`weight * x̂`, no +1 offset — verified against
/// `Lfm3RMSNorm`), so no offset is baked in. Also used for the depthwise conv
/// filter ([hidden, K]) and the attention gate (loaded as a WeightTensor).
fn load_f32(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => dequant_q8_0(&data),
        _ => {
            return Err(format!("lfm3: expected F16/F32/Q8 for {name}, got qt={qt}"))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("lfm3: upload {name}: {e:?}"))
}

fn load_wt(
    hfq: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    wt_from_raw(gpu, qt, &data, m, k).map_err(|e| format!("lfm3: load_wt {name}: {e}"))
}

/// Minimal Q8_0 dequant (32-elem blocks: little-endian f16 scale + 32 int8).
fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 34 * 32);
    for blk in data.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push((q as i8) as f32 * scale);
        }
    }
    out
}

/// quant_type → DType mapping (mirrors lfm2moe::wt_from_raw).
fn wt_from_raw(
    gpu: &mut Gpu,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        1 => DType::F16,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

// ──────────────────────────── Weights ────────────────────────────

/// LIV short-conv mixer weights (identical shape to LFM2; K=7 for LFM3).
pub struct ConvWeights {
    /// in_proj: [3*hidden, hidden] — produces B | C_gate | x.
    pub in_proj: WeightTensor,
    /// Depthwise causal filter, squeezed [hidden, K] F32.
    pub conv_weight: GpuTensor,
    /// out_proj: [hidden, hidden].
    pub out_proj: WeightTensor,
    /// Index into the conv-state ring-buffer cache.
    pub conv_state_idx: usize,
}

/// GQA attention mixer weights (shared by full_attention and sliding_attention
/// layers; the forward picks the window + RoPE-theta from `cfg.mixer(l)`).
pub struct AttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    /// Per-head QK-norm weight, [head_dim].
    pub q_norm: GpuTensor,
    pub k_norm: GpuTensor,
    /// Sigmoid output gate: [n_heads, hidden] → per-head scalar gate.
    /// LFM3-specific (absent in LFM2).
    pub gate_proj: WeightTensor,
    /// Index into the KV cache.
    pub kv_idx: usize,
}

pub enum Mixer {
    Conv(ConvWeights),
    Attention(AttnWeights),
}

/// Dense SwiGLU MLP (LFM3 text backbone is dense-only — no MoE).
pub struct DenseFfn {
    pub w1: WeightTensor, // gate [inter, hidden]
    pub w3: WeightTensor, // up   [inter, hidden]
    pub w2: WeightTensor, // down [hidden, inter]
}

pub struct Lfm3LayerWeights {
    pub operator_norm: GpuTensor, // pre-mixer RMSNorm
    pub ffn_norm: GpuTensor,      // pre-FFN RMSNorm
    /// Sandwich norm on the mixer output (present iff `use_sandwich_norm`).
    pub post_operator_norm: Option<GpuTensor>,
    /// Sandwich norm on the FFN output (present iff `use_sandwich_norm`).
    pub post_ffn_norm: Option<GpuTensor>,
    pub mixer: Mixer,
    pub ffn: DenseFfn,
}

pub struct Lfm3Weights {
    pub embed: GpuTensor, // model.embed_tokens.weight (raw, for embedding_lookup)
    pub embedding_norm: GpuTensor, // model.embedding_norm.weight (final norm)
    pub lm_head: WeightTensor, // tied = embed_tokens (loaded as a weight tensor)
    pub layers: Vec<Lfm3LayerWeights>,
}

impl MmqScreenable for Lfm3Weights {
    fn screen_mmq_weights(&self, gpu: &mut Gpu) -> (usize, usize) {
        let (mut safe, mut unsafe_count) = (0usize, 0usize);
        screen_weight_tensor(&self.lm_head, gpu, &mut safe, &mut unsafe_count);
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Conv(weights) => {
                    screen_weight_tensor(&weights.in_proj, gpu, &mut safe, &mut unsafe_count);
                    screen_weight_tensor(&weights.out_proj, gpu, &mut safe, &mut unsafe_count);
                }
                Mixer::Attention(weights) => {
                    for weight in [&weights.wq, &weights.wk, &weights.wv, &weights.wo] {
                        screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
                    }
                    screen_weight_tensor(&weights.gate_proj, gpu, &mut safe, &mut unsafe_count);
                }
            }
            for weight in [&layer.ffn.w1, &layer.ffn.w3, &layer.ffn.w2] {
                screen_weight_tensor(weight, gpu, &mut safe, &mut unsafe_count);
            }
        }
        (safe, unsafe_count)
    }
}

impl ConvWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.in_proj.free_all(gpu);
        let _ = gpu.free_tensor(self.conv_weight);
        self.out_proj.free_all(gpu);
    }
}

impl AttnWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
        self.gate_proj.free_all(gpu);
    }
}

impl Mixer {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            Mixer::Conv(c) => c.free_gpu(gpu),
            Mixer::Attention(a) => a.free_gpu(gpu),
        }
    }
}

impl DenseFfn {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.w1.free_all(gpu);
        self.w3.free_all(gpu);
        self.w2.free_all(gpu);
    }
}

impl Lfm3LayerWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.operator_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        if let Some(n) = self.post_operator_norm {
            let _ = gpu.free_tensor(n);
        }
        if let Some(n) = self.post_ffn_norm {
            let _ = gpu.free_tensor(n);
        }
        self.mixer.free_gpu(gpu);
        self.ffn.free_gpu(gpu);
    }
}

impl Lfm3Weights {
    pub fn load(hfq: &mut HfqFile, cfg: &Lfm3Config, gpu: &mut Gpu) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let head_dim = cfg.head_dim;
        let inter = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let k_conv = cfg.conv_kernel_size;
        let sandwich = cfg.use_sandwich_norm;

        // Globals. embed_tokens is the shared (tied) lm_head.
        let (_eqt, embed_bytes) = read_tensor(hfq, "model.embed_tokens.weight")?;
        let embed = gpu
            .upload_raw(&embed_bytes, &[embed_bytes.len()])
            .map_err(|e| format!("lfm3: upload embed: {e:?}"))?;
        let embedding_norm = load_f32(hfq, gpu, "model.embedding_norm.weight", &[hidden])?;
        let lm_head = load_wt(hfq, gpu, "model.embed_tokens.weight", cfg.vocab_size, hidden)?;

        let mut conv_state_count = 0usize;
        let mut kv_count = 0usize;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let operator_norm = load_f32(hfq, gpu, &format!("{p}.operator_norm.weight"), &[hidden])?;
            let ffn_norm = load_f32(hfq, gpu, &format!("{p}.ffn_norm.weight"), &[hidden])?;
            let post_operator_norm = if sandwich {
                Some(load_f32(
                    hfq,
                    gpu,
                    &format!("{p}.post_operator_norm.weight"),
                    &[hidden],
                )?)
            } else {
                None
            };
            let post_ffn_norm = if sandwich {
                Some(load_f32(
                    hfq,
                    gpu,
                    &format!("{p}.post_ffn_norm.weight"),
                    &[hidden],
                )?)
            } else {
                None
            };

            // ── Mixer: conv vs attention (full/sliding share weights) ──────
            let mixer = match cfg.mixer(l) {
                MixerKind::Conv => {
                    let in_proj =
                        load_wt(hfq, gpu, &format!("{p}.conv.in_proj.weight"), 3 * hidden, hidden)?;
                    // conv.conv.weight ships [hidden,1,K] → loaded flat [hidden,K] f32.
                    let conv_weight =
                        load_f32(hfq, gpu, &format!("{p}.conv.conv.weight"), &[hidden * k_conv])?;
                    let out_proj =
                        load_wt(hfq, gpu, &format!("{p}.conv.out_proj.weight"), hidden, hidden)?;
                    let conv_state_idx = conv_state_count;
                    conv_state_count += 1;
                    Mixer::Conv(ConvWeights {
                        in_proj,
                        conv_weight,
                        out_proj,
                        conv_state_idx,
                    })
                }
                MixerKind::FullAttention | MixerKind::SlidingAttention => {
                    let wq = load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, hidden)?;
                    let wk = load_wt(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, hidden)?;
                    let wv = load_wt(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, hidden)?;
                    let wo = load_wt(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), hidden, q_dim)?;
                    // LFM3 names these q_norm / k_norm (NOT LFM2's q_layernorm).
                    let q_norm = load_f32(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), &[head_dim])?;
                    let k_norm = load_f32(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), &[head_dim])?;
                    // Sigmoid output gate: [n_heads, hidden].
                    let gate_proj = load_wt(
                        hfq,
                        gpu,
                        &format!("{p}.self_attn.gate_proj.weight"),
                        n_heads,
                        hidden,
                    )?;
                    let kv_idx = kv_count;
                    kv_count += 1;
                    Mixer::Attention(AttnWeights {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm,
                        k_norm,
                        gate_proj,
                        kv_idx,
                    })
                }
            };

            // ── FFN: dense SwiGLU (no MoE in the LFM3 text backbone) ───────
            let w1 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w1.weight"), inter, hidden)?;
            let w3 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w3.weight"), inter, hidden)?;
            let w2 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w2.weight"), hidden, inter)?;
            let ffn = DenseFfn { w1, w3, w2 };

            layers.push(Lfm3LayerWeights {
                operator_norm,
                ffn_norm,
                post_operator_norm,
                post_ffn_norm,
                mixer,
                ffn,
            });
        }

        let _ = (conv_state_count, kv_count);
        Ok(Lfm3Weights {
            embed,
            embedding_norm,
            lm_head,
            layers,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embed);
        let _ = gpu.free_tensor(self.embedding_norm);
        self.lm_head.free_all(gpu);
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
    }
}

// ─────────────── Source-based load helpers (ModelSource) ──────────────

fn classify_tensor_bytes(bytes: &[u8], numel: usize) -> (bool, bool) {
    let is_f16 = bytes.len() == numel * 2;
    let q8_0_expected = ((numel + 31) / 32) * 34;
    let is_q8_0 = !is_f16 && bytes.len() == q8_0_expected;
    (is_f16, is_q8_0)
}

fn effective_quant_type(
    info: &hipfire_runtime::model_source::TensorInfo,
    data: &[u8],
) -> u8 {
    if info.quant_type != 0xFF {
        return info.quant_type;
    }
    match info.dtype.as_str() {
        "F16" | "BF16" => 1,
        "F32" => 2,
        "I8" | "INT8" => 3,
        _ => {
            let numel: usize = info.shape.iter().product();
            let (is_f16, is_q8_0) = classify_tensor_bytes(data, numel);
            if is_f16 {
                1
            } else if is_q8_0 {
                3
            } else {
                1
            }
        }
    }
}

fn read_tensor_from_source<'a>(
    source: &'a dyn ModelSource,
    name: &str,
) -> Result<(u8, &'a str, &'a [u8]), String> {
    if let Some((info, data)) = source.tensor_data(name) {
        return Ok((effective_quant_type(info, data), info.dtype.as_str(), data));
    }
    // LFM3-VL nests the text backbone under `model.language_model.*`;
    // LFM3-Audio nests it under `lfm.model.*`.
    if let Some(stripped) = name.strip_prefix("model.") {
        let prefixed = format!("model.language_model.{stripped}");
        if let Some((info, data)) = source.tensor_data(&prefixed) {
            return Ok((effective_quant_type(info, data), info.dtype.as_str(), data));
        }
        let prefixed = format!("lfm.model.{stripped}");
        if let Some((info, data)) = source.tensor_data(&prefixed) {
            return Ok((effective_quant_type(info, data), info.dtype.as_str(), data));
        }
    }
    Err(format!("lfm3: tensor not found in source: {name}"))
}

fn load_f32_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (qt, dtype, data) = read_tensor_from_source(source, name)?;
    let f32_data: Vec<f32> = match qt {
        1 | 2 => source_bytes_to_f32_vec(dtype, data),
        3 => dequant_q8_0(data),
        _ => {
            return Err(format!("lfm3: expected F16/F32/Q8 for {name}, got qt={qt}"))
        }
    };
    gpu.upload_f32(&f32_data, shape)
        .map_err(|e| format!("lfm3: upload {name}: {e:?}"))
}

fn load_wt_from_source(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (qt, src_dtype, data) = read_tensor_from_source(source, name)?;
    wt_from_source_raw(gpu, qt, src_dtype, data, m, k)
        .map_err(|e| format!("lfm3: load_wt_from_source {name}: {e}"))
}

fn wt_from_source_raw(
    gpu: &mut Gpu,
    qt: u8,
    src_dtype: &str,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        18 => DType::MQ2G256,
        19 => DType::MQ2G256Lloyd,
        20 => DType::MQ3G256Lloyd,
        30 => DType::MQ4G256Lloyd,
        1 => DType::F16,
        other => return Err(format!("unsupported quant_type {other}")),
    };
    let converted: Vec<u8>;
    let data: &[u8] = if qt == 1 && src_dtype == "BF16" {
        converted = bf16_bytes_to_f16(data);
        &converted
    } else {
        data
    };
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: dtype,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

/// Load full LFM3 weights from a `&dyn ModelSource` (safetensors or HFQ).
/// Mirrors `Lfm3Weights::load` but reads through the ModelSource trait.
pub fn load_weights_from_source(
    source: &dyn ModelSource,
    cfg: &Lfm3Config,
    gpu: &mut Gpu,
) -> Result<Lfm3Weights, String> {
    let hidden = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let inter = cfg.intermediate_size;
    let n_heads = cfg.num_attention_heads;
    let k_conv = cfg.conv_kernel_size;
    let sandwich = cfg.use_sandwich_norm;

    let (eqt, edt, embed_bytes) = read_tensor_from_source(source, "model.embed_tokens.weight")?;
    let embed_converted: Vec<u8>;
    let embed_bytes: &[u8] = if eqt == 1 && edt == "BF16" {
        embed_converted = bf16_bytes_to_f16(embed_bytes);
        &embed_converted
    } else {
        embed_bytes
    };
    let embed = gpu
        .upload_raw(embed_bytes, &[embed_bytes.len()])
        .map_err(|e| format!("lfm3: upload embed: {e:?}"))?;
    let embedding_norm =
        load_f32_from_source(source, gpu, "model.embedding_norm.weight", &[hidden])?;
    let lm_head = load_wt_from_source(source, gpu, "model.embed_tokens.weight", cfg.vocab_size, hidden)?;

    let mut conv_state_count = 0usize;
    let mut kv_count = 0usize;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let operator_norm =
            load_f32_from_source(source, gpu, &format!("{p}.operator_norm.weight"), &[hidden])?;
        let ffn_norm =
            load_f32_from_source(source, gpu, &format!("{p}.ffn_norm.weight"), &[hidden])?;
        let post_operator_norm = if sandwich {
            Some(load_f32_from_source(
                source,
                gpu,
                &format!("{p}.post_operator_norm.weight"),
                &[hidden],
            )?)
        } else {
            None
        };
        let post_ffn_norm = if sandwich {
            Some(load_f32_from_source(
                source,
                gpu,
                &format!("{p}.post_ffn_norm.weight"),
                &[hidden],
            )?)
        } else {
            None
        };

        let mixer = match cfg.mixer(l) {
            MixerKind::Conv => {
                let in_proj = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.in_proj.weight"),
                    3 * hidden,
                    hidden,
                )?;
                let conv_weight = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.conv.weight"),
                    &[hidden * k_conv],
                )?;
                let out_proj = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.conv.out_proj.weight"),
                    hidden,
                    hidden,
                )?;
                let conv_state_idx = conv_state_count;
                conv_state_count += 1;
                Mixer::Conv(ConvWeights {
                    in_proj,
                    conv_weight,
                    out_proj,
                    conv_state_idx,
                })
            }
            MixerKind::FullAttention | MixerKind::SlidingAttention => {
                let wq = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    hidden,
                )?;
                let wk = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wv = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    hidden,
                )?;
                let wo = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                    q_dim,
                )?;
                let q_norm = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[head_dim],
                )?;
                let k_norm = load_f32_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[head_dim],
                )?;
                let gate_proj = load_wt_from_source(
                    source,
                    gpu,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    n_heads,
                    hidden,
                )?;
                let kv_idx = kv_count;
                kv_count += 1;
                Mixer::Attention(AttnWeights {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    gate_proj,
                    kv_idx,
                })
            }
        };

        let w1 = load_wt_from_source(
            source,
            gpu,
            &format!("{p}.feed_forward.w1.weight"),
            inter,
            hidden,
        )?;
        let w3 = load_wt_from_source(
            source,
            gpu,
            &format!("{p}.feed_forward.w3.weight"),
            inter,
            hidden,
        )?;
        let w2 = load_wt_from_source(
            source,
            gpu,
            &format!("{p}.feed_forward.w2.weight"),
            hidden,
            inter,
        )?;
        let ffn = DenseFfn { w1, w3, w2 };

        layers.push(Lfm3LayerWeights {
            operator_norm,
            ffn_norm,
            post_operator_norm,
            post_ffn_norm,
            mixer,
            ffn,
        });
    }

    let _ = (conv_state_count, kv_count);
    Ok(Lfm3Weights {
        embed,
        embedding_norm,
        lm_head,
        layers,
    })
}

// ──────────────── Decode state ────────────────

/// Per-decode GPU scratch + KV cache (attention layers) + conv-state cache
/// (conv layers). Buffers are eager-allocated.
pub struct Lfm3State {
    pub kv: KvCache,
    /// One rolling [hidden, K-1] f32 ring buffer per conv layer (zero-init).
    pub conv_states: Vec<GpuTensor>,
    pub pos_buf: hip_bridge::DeviceBuffer, // device i32 position scalar
    pub max_seq: usize,
    pub n_tokens: usize,

    // residual + shared scratch
    pub h: GpuTensor,   // [hidden] residual stream
    pub tmp: GpuTensor, // [hidden] operator_norm output (mixer input)

    // attention scratch
    pub fa_q: GpuTensor,        // [q_dim]
    pub fa_k: GpuTensor,        // [kv_dim]
    pub fa_v: GpuTensor,        // [kv_dim]
    pub fa_attn_out: GpuTensor, // [q_dim]
    pub fa_gate: GpuTensor,     // [n_heads] sigmoid gate logits
    /// Flash-attention online-softmax partials for the windowed Q8 flash kernel
    /// (`AttnFlashQ8_0Windowed`, used for BOTH full and sliding layers — window=0
    /// on full layers == plain flash). `[n_heads · ceil(max_seq/128) · (2+head_dim)
    /// · FLASH_PREFILL_SUBBATCH]`. Mirrors cohere2moe's allocation.
    pub flash_partials: GpuTensor,

    // conv scratch
    pub conv_bcx: GpuTensor, // [3*hidden] in_proj output (B|C|x)
    pub conv_y: GpuTensor,   // [hidden] gated conv output (out_proj input)

    // ffn scratch
    pub ffn_tmp: GpuTensor,    // [hidden] ffn_norm(h)
    pub dense_gate: GpuTensor, // [inter]
    pub dense_up: GpuTensor,   // [inter]
    pub dense_act: GpuTensor,  // [inter] silu(gate)*up

    // mixer/ffn output + sandwich-norm scratch (the post-operator/post-ffn
    // norm is applied to the projection output BEFORE the residual add, so the
    // fused `weight_gemv_residual` cannot be used when sandwich norm is on).
    pub proj_out: GpuTensor,    // [hidden] out_proj / o_proj / w2 output
    pub sandwich_buf: GpuTensor, // [hidden] post_*_norm output (sandwich only)

    // head
    pub final_norm_buf: GpuTensor, // [hidden]
    pub logits: GpuTensor,         // [vocab]
}

impl Lfm3State {
    pub fn new(gpu: &mut Gpu, cfg: &Lfm3Config) -> Result<Self, String> {
        let max_seq = cfg.max_position_embeddings.min(8192);
        Self::new_with_max_seq(gpu, cfg, max_seq)
    }

    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Lfm3Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let k_conv = cfg.conv_kernel_size;

        // FWHT sign LUT must exist before any rotate_x_mq / fused rotate kernel.
        gpu.ensure_mq_signs()
            .map_err(|e| format!("lfm3: ensure_mq_signs: {e:?}"))?;

        // KV cache: one slot per ATTENTION layer (conv layers carry no KV).
        let n_attn = cfg.num_attention_layers().max(1);
        let dims = hipfire_runtime::llama::KvDims {
            layers: hipfire_runtime::llama::KvLayers::Flat(n_attn),
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            max_seq,
            physical_cap: None,
        };
        let kv = KvCache::from_mode(
            hipfire_runtime::kv_mode::resolve("", &hipfire_runtime::kv_mode::HFQ_Q8_ONLY_POLICY).mode,
            gpu,
            &dims,
        )
        .map_err(|e| format!("lfm3: kv cache: {e:?}"))?;

        // Conv-state cache: one [hidden,(K-1)] f32 ring buffer per CONV layer.
        let conv_hist = hidden * (k_conv - 1);
        let zeros = vec![0u8; conv_hist * 4];
        let mut conv_states = Vec::with_capacity(cfg.num_conv_layers());
        for _ in 0..cfg.num_conv_layers() {
            let cs = gpu
                .alloc_tensor(&[conv_hist], DType::F32)
                .map_err(|e| format!("lfm3: alloc conv_state: {e:?}"))?;
            gpu.hip
                .memcpy_htod(&cs.buf, &zeros)
                .map_err(|e| format!("lfm3: zero conv_state: {e:?}"))?;
            conv_states.push(cs);
        }

        let pos_buf = gpu
            .hip
            .malloc(4)
            .map_err(|e| format!("lfm3: pos_buf malloc: {e:?}"))?;

        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32)
                .map_err(|e| format!("lfm3: alloc {label}: {e:?}"))
        };

        Ok(Lfm3State {
            kv,
            conv_states,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            tmp: alloc(gpu, hidden, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            fa_gate: alloc(gpu, n_heads, "fa_gate")?,
            flash_partials: alloc(
                gpu,
                n_heads * ((max_seq + 127) / 128) * (2 + cfg.head_dim) * FLASH_PREFILL_SUBBATCH,
                "flash_partials",
            )?,
            conv_bcx: alloc(gpu, 3 * hidden, "conv_bcx")?,
            conv_y: alloc(gpu, hidden, "conv_y")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            dense_gate: alloc(gpu, inter, "dense_gate")?,
            dense_up: alloc(gpu, inter, "dense_up")?,
            dense_act: alloc(gpu, inter, "dense_act")?,
            proj_out: alloc(gpu, hidden, "proj_out")?,
            sandwich_buf: alloc(gpu, hidden, "sandwich_buf")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
            logits: alloc(gpu, cfg.vocab_size, "logits")?,
        })
    }

    /// Reset for a new sequence: clear conv state and token count.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.n_tokens = 0;
        for cs in &self.conv_states {
            let zeros = vec![0u8; cs.numel() * 4];
            gpu.hip
                .memcpy_htod(&cs.buf, &zeros)
                .map_err(|e| format!("lfm3: reset conv_state: {e:?}"))?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.h);
        let _ = gpu.free_tensor(self.tmp);
        let _ = gpu.free_tensor(self.fa_q);
        let _ = gpu.free_tensor(self.fa_k);
        let _ = gpu.free_tensor(self.fa_v);
        let _ = gpu.free_tensor(self.fa_attn_out);
        let _ = gpu.free_tensor(self.fa_gate);
        let _ = gpu.free_tensor(self.flash_partials);
        let _ = gpu.free_tensor(self.conv_bcx);
        let _ = gpu.free_tensor(self.conv_y);
        let _ = gpu.free_tensor(self.ffn_tmp);
        let _ = gpu.free_tensor(self.dense_gate);
        let _ = gpu.free_tensor(self.dense_up);
        let _ = gpu.free_tensor(self.dense_act);
        let _ = gpu.free_tensor(self.proj_out);
        let _ = gpu.free_tensor(self.sandwich_buf);
        let _ = gpu.free_tensor(self.final_norm_buf);
        let _ = gpu.free_tensor(self.logits);
        // LEAK-1: the 4-byte position scalar buffer was allocated in the
        // constructor but never freed (inherited from the lfm2moe template).
        let _ = gpu.hip.free(self.pos_buf);
    }
}
