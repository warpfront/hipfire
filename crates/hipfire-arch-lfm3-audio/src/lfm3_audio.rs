// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Audio speech encoder + projector (Phase 3).
//!
//! Ground truth: transformers `modeling_lfm3_audio.py` /
//! `configuration_lfm3_audio.py` / `feature_extraction_lfm3_audio.py`
//! (read 2026-10-31, `LiquidAI/LFM3-Audio-2B`).
//!
//! Three components:
//!   * **Downsampler** (`Lfm3AudioDownsampler`) — 3 stride-2 conv2d layers
//!     (conv_in 1→256, depthwise conv_mid, depthwise conv_end) + 1×1 lins
//!     (lin_mid, lin_end) + SiLU + a final Conv1d `lin_out` (4096→768).
//!     128 mel bins × 8 frames → one 768-dim encoder token. Runs on CPU
//!     (small; the mel is already host-side and must byte-match the HF
//!     extractor — no new GPU conv2d kernels needed for the stride-2 path).
//!   * **Encoder** (`Lfm3AudioEncoder.lfm`) — a 17-layer LFM2 backbone
//!     (hidden 768, conv K=3, 8Q/4KV heads, head_dim 96, sliding-window 32
//!     causal attention, full-rotary RoPE θ=1e6, dense SwiGLU, NO sandwich
//!     norm, NO sigmoid output gate). Runs on GPU as a sequential prefill
//!     (token-by-token, reusing the LFM3 conv/attn/RoPE/GQA primitives).
//!   * **Projector** (`Lfm3AudioProjector`) — LayerNorm(768) + GELU(erf) MLP
//!     768→2048→2048. Runs on GPU (reuses `layernorm_batched` / `gelu_erf_f32`).
//!
//! arch_id stays 24 (audio detected by `encoder.downsample.*` /
//! `model.audio_encoder.downsample.*` tensors — see AUDIO-NAME-1 below).
//! presence). The text decoder is owned by `hipfire-arch-lfm3::Lfm3`.

use crate::mel::MelConfig;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{f16_to_f32, f32_to_f16, weight_gemv, KvCache, KvCacheExt, WeightTensor};
use hip_bridge::{HipError, HipResult};
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ──────────────────────────────────────────────────────────────────

/// LFM2 encoder layer kind (the audio encoder has only conv + full_attention;
/// full_attention layers use a sliding-window-32 causal mask per the HF
/// `create_sliding_window_causal_mask`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncLayerKind {
    Conv,
    Attention,
}

impl EncLayerKind {
    pub fn is_attention(self) -> bool {
        matches!(self, EncLayerKind::Attention)
    }
}

/// Conv2d downsampler shape constants (HF `Lfm3AudioEncoderConfig`).
#[derive(Clone, Debug)]
pub struct DownsamplerConfig {
    pub dim_in: usize,   // 128 (num mel bins)
    pub dim_conv: usize, // 256 (conv channels)
    pub dim_out: usize,  // 768 (encoder hidden)
}

/// 17-layer LFM2 backbone config (HF `encoder.lfm`).
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub conv_kernel_size: usize,
    /// Dense SwiGLU intermediate size — **derived from the actual w1 tensor
    /// shape at load time** (the config's `intermediate_size` is the
    /// pre-`block_auto_adjust_ff_dim` value 2560; the real weights are 1792).
    pub intermediate_size: usize,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub layer_types: Vec<EncLayerKind>,
}

impl EncoderConfig {
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn num_attention_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.is_attention()).count()
    }
    pub fn num_conv_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| !t.is_attention()).count()
    }
}

/// Audio projector config (HF `Lfm3AudioProjector`).
#[derive(Clone, Debug)]
pub struct ProjectorConfig {
    pub in_features: usize,  // 768 (encoder hidden)
    pub out_features: usize, // 2048 (text hidden)
    pub norm_eps: f32,
}

/// Top-level audio config (mel + downsampler + encoder + projector).
#[derive(Clone, Debug)]
pub struct AudioConfig {
    pub mel: MelConfig,
    pub down: DownsamplerConfig,
    pub enc: EncoderConfig,
    pub proj: ProjectorConfig,
}

/// Parse the LFM3-Audio config from the HFQ metadata `config` envelope.
///
/// AUDIO-NAME-1: two artifact shapes are accepted.
/// - LFM3-Audio-*: `preprocessor` (mel) + `encoder` {dim_in/dim_conv/dim_out +
///   `lfm` backbone} + top `lfm` (text backbone — projector width).
/// - LFM3-Omni-*: `audio_encoder_config` {num_mel_bins, conv_channels,
///   hidden_size, sliding_window, `text_config` backbone} + top `text_config`
///   (text backbone). There is no `preprocessor` block; the mel fields fall
///   back to the shared defaults, which byte-match the Omni feature extractor
///   (n_fft 400 / hop 160 / win 400 / sr 16k / floor -12 / norm 4±).
///
/// The encoder `intermediate_size` is taken from the actual
/// `layers.0.feed_forward.w1.weight` tensor shape (the config value is
/// pre-auto-adjust and unreliable).
pub fn audio_config_from_hfq(hfq: &HfqFile) -> Option<AudioConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config").or(Some(&meta))?;
    let (pre, enc_cfg, lfm_cfg, text_cfg) = if let Some(ae) = config.get("audio_encoder_config") {
        // Omni shape.
        (
            ae,
            ae,
            ae.get("text_config")?,
            config.get("text_config")?,
        )
    } else {
        // Audio-standalone shape.
        (
            config.get("preprocessor")?,
            config.get("encoder")?,
            config.get("encoder")?.get("lfm")?,
            config.get("lfm").or_else(|| config.get("text_config"))?,
        )
    };

    // Mel front-end (config.preprocessor).
    let mel = MelConfig {
        n_fft: pre.get("n_fft").and_then(|v| v.as_u64()).unwrap_or(400) as usize,
        hop_length: pre.get("hop_length").and_then(|v| v.as_u64()).unwrap_or(160) as usize,
        win_length: pre.get("win_length").and_then(|v| v.as_u64()).unwrap_or(400) as usize,
        n_mels: pre.get("n_mels").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
        sample_rate: pre.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as u32,
        f_max: pre.get("f_max").and_then(|v| v.as_f64()).unwrap_or(8000.0) as f32,
        f_min: pre.get("f_min").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        log_floor: pre.get("logmel_floor").and_then(|v| v.as_f64()).unwrap_or(-12.0) as f32,
        norm_offset: pre
            .get("normalization")
            .and_then(|n| n.get("offset"))
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0) as f32,
        norm_divisor: pre
            .get("normalization")
            .and_then(|n| n.get("divisor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0) as f32,
    };

    let down = DownsamplerConfig {
        // Audio standalone: explicit dim_in/dim_conv/dim_out. Omni: derive from
        // num_mel_bins / conv_channels / the encoder hidden size.
        dim_in: enc_cfg
            .get("dim_in")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(mel.n_mels),
        dim_conv: enc_cfg
            .get("dim_conv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                enc_cfg.get("conv_channels").and_then(|v| v.as_u64()).unwrap_or(256) as usize
            }),
        dim_out: enc_cfg
            .get("dim_out")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                enc_cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as usize
            }),
    };

    let hidden = lfm_cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as usize;
    let n_heads = lfm_cfg.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
    let n_kv = lfm_cfg
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;
    let head_dim = lfm_cfg
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden / n_heads) as u64) as usize;
    let num_layers = lfm_cfg.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(17) as usize;
    let conv_k = lfm_cfg.get("conv_L_cache").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let sliding_window = enc_cfg
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let rope_theta = lfm_cfg
        .get("rope_parameters")
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1_000_000.0) as f32;
    let norm_eps = lfm_cfg.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;

    let layer_types: Vec<EncLayerKind> = lfm_cfg
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|s| match s.as_str().unwrap_or("conv") {
                    "full_attention" | "attention" | "attn" => EncLayerKind::Attention,
                    _ => EncLayerKind::Conv,
                })
                .collect()
        })
        .unwrap_or_else(|| (0..num_layers).map(|i| if i % 3 == 2 { EncLayerKind::Attention } else { EncLayerKind::Conv }).collect());

    // Intermediate size: derive from the actual w1 tensor shape (config value
    // is pre-`block_auto_adjust_ff_dim` and unreliable — 2560 vs real 1792).
    // Probe both artifact namings (AUDIO-NAME-1).
    let intermediate = hfq
        .find_tensor_info("encoder.lfm.lfm.layers.0.feed_forward.w1.weight")
        .or_else(|| hfq.find_tensor_info("model.audio_encoder.lfm.layers.0.feed_forward.w1.weight"))
        .and_then(|i| i.shape.first().copied())
        .map(|s| s as usize)
        .unwrap_or_else(|| {
            lfm_cfg.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(1792) as usize
        });

    let proj = ProjectorConfig {
        in_features: down.dim_out,
        out_features: text_cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(2048) as usize,
        // HF `nn.LayerNorm(in_features)` default (modeling_lfm3_audio.py projector).
        norm_eps: 1e-5,
    };

    Some(AudioConfig {
        mel,
        down,
        enc: EncoderConfig {
            hidden_size: hidden,
            num_hidden_layers: num_layers,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            head_dim,
            conv_kernel_size: conv_k,
            intermediate_size: intermediate,
            sliding_window,
            rope_theta,
            norm_eps,
            layer_types,
        },
        proj,
    })
}

// ─── Downsampler weights (CPU F32) ───────────────────────────────────────────

/// Conv2d / linear weight as a flat F32 host vector + logical shape.
#[derive(Clone)]
struct HostW {
    data: Vec<f32>,
    /// PyTorch row-major shape (e.g. [256,1,3,3] for conv2d, [768,4096] for lin).
    shape: Vec<usize>,
}

/// Downsampler weights, host-side F32 (the downsampler runs on CPU).
pub struct DownsamplerWeights {
    conv_in_w: HostW,   // [256,1,3,3]
    conv_in_b: Vec<f32>, // [256]
    conv_mid_w: HostW,  // [256,1,3,3] (depthwise)
    conv_mid_b: Vec<f32>,
    lin_mid_w: HostW,   // [256,256,1,1]
    lin_mid_b: Vec<f32>,
    conv_end_w: HostW,  // [256,1,3,3] (depthwise)
    conv_end_b: Vec<f32>,
    lin_end_w: HostW,   // [256,256,1,1]
    lin_end_b: Vec<f32>,
    lin_out_w: HostW,   // [768,4096,1]
    lin_out_b: Vec<f32>, // [768]
}

// ─── Downsampler forward (CPU) ───────────────────────────────────────────────
// Mirrors HF `Lfm3AudioDownsampler.forward` (single clip, B=1, no length mask):
//   h = silu(conv_in(mel))                     [256, 64,  T/2]
//   h = silu(lin_mid(conv_mid(h)))             [256, 32,  T/4]
//   h = silu(lin_end(conv_end(h)))             [256, 16,  T/8]
//   out = lin_out(flatten(h, channels·height)) [768, T/8] → transpose [T/8, 768]

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Conv2d output dim for k=3, stride=2, pad=1.
#[inline]
fn conv_out_dim(d: usize) -> usize {
    (d + 2 - 3) / 2 + 1 // = (d - 1)/2 + 1  (floor)
}

/// Standard Conv2d (k=3, s=2, p=1). `x`: [Cin, H, W] row-major. `w`: [Cout, Cin, 3, 3].
fn conv2d_k3s2p1(x: &[f32], cin: usize, h: usize, w: usize, w_w: &HostW, bias: &[f32]) -> Vec<f32> {
    let cout = w_w.shape[0];
    let ho = conv_out_dim(h);
    let wo = conv_out_dim(w);
    let mut out = vec![0.0f32; cout * ho * wo];
    for co in 0..cout {
        let b = bias[co];
        let wbase = co * cin * 9;
        for hoi in 0..ho {
            for woi in 0..wo {
                let mut acc = b;
                let h0 = hoi * 2; // window start in padded coords (pad=1 → unpadded h0-1)
                let w0 = woi * 2;
                for ci in 0..cin {
                    let xbase = ci * h * w;
                    let wci = wbase + ci * 9;
                    for kh in 0..3 {
                        let ih = h0 as isize + kh as isize - 1; // -1 for pad
                        if ih < 0 || ih >= h as isize {
                            continue;
                        }
                        for kw in 0..3 {
                            let iw = w0 as isize + kw as isize - 1;
                            if iw < 0 || iw >= w as isize {
                                continue;
                            }
                            acc += x[xbase + ih as usize * w + iw as usize] * w_w.data[wci + kh * 3 + kw];
                        }
                    }
                }
                out[co * ho * wo + hoi * wo + woi] = acc;
            }
        }
    }
    out
}

/// Depthwise Conv2d (k=3, s=2, p=1, groups=C). `x`: [C, H, W]. `w`: [C, 1, 3, 3].
fn conv2d_dw_k3s2p1(x: &[f32], c: usize, h: usize, w: usize, w_w: &HostW, bias: &[f32]) -> Vec<f32> {
    let ho = conv_out_dim(h);
    let wo = conv_out_dim(w);
    let mut out = vec![0.0f32; c * ho * wo];
    for ci in 0..c {
        let b = bias[ci];
        let wci = ci * 9; // [C,1,3,3] → flat [C*9]
        let xbase = ci * h * w;
        for hoi in 0..ho {
            for woi in 0..wo {
                let mut acc = b;
                let h0 = hoi * 2;
                let w0 = woi * 2;
                for kh in 0..3 {
                    let ih = h0 as isize + kh as isize - 1;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    for kw in 0..3 {
                        let iw = w0 as isize + kw as isize - 1;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        acc += x[xbase + ih as usize * w + iw as usize] * w_w.data[wci + kh * 3 + kw];
                    }
                }
                out[ci * ho * wo + hoi * wo + woi] = acc;
            }
        }
    }
    out
}

/// 1×1 Conv2d (linear over channels, per spatial position). `x`: [Cin, H, W]. `w`: [Cout, Cin].
fn conv2d_1x1(x: &[f32], cin: usize, h: usize, w: usize, w_w: &HostW, bias: &[f32]) -> Vec<f32> {
    let cout = w_w.shape[0];
    let hw = h * w;
    let mut out = vec![0.0f32; cout * hw];
    for co in 0..cout {
        let b = bias[co];
        let wbase = co * cin;
        for hw_i in 0..hw {
            let mut acc = b;
            for ci in 0..cin {
                acc += x[ci * hw + hw_i] * w_w.data[wbase + ci];
            }
            out[co * hw + hw_i] = acc;
        }
    }
    out
}

/// Run the conv2d downsampler over a mel spectrogram.
///
/// `mel`: `[n_mels, num_frames]` row-major F32. Returns `[n_tokens, dim_out]`
/// row-major F32 — one 768-dim vector per 8 mel frames, ready for the encoder.
pub fn downsampler_forward(mel: &[f32], n_mels: usize, num_frames: usize, dw: &DownsamplerWeights) -> Vec<f32> {
    let c = dw.conv_in_w.shape[0]; // 256

    // h = silu(conv_in(mel))  — mel is [1, n_mels, num_frames] (Cin=1).
    let mut h = conv2d_k3s2p1(mel, 1, n_mels, num_frames, &dw.conv_in_w, &dw.conv_in_b);
    let (h1, w1) = (conv_out_dim(n_mels), conv_out_dim(num_frames)); // (64, T/2)
    for v in h.iter_mut() {
        *v = silu(*v);
    }

    // h = silu(lin_mid(conv_mid(h)))  — depthwise (halves H,W) then 1×1.
    let h2 = conv2d_dw_k3s2p1(&h, c, h1, w1, &dw.conv_mid_w, &dw.conv_mid_b);
    let (h2d, w2) = (conv_out_dim(h1), conv_out_dim(w1)); // (32, T/4)
    let mut h3 = conv2d_1x1(&h2, c, h2d, w2, &dw.lin_mid_w, &dw.lin_mid_b);
    for v in h3.iter_mut() {
        *v = silu(*v);
    }

    // h = silu(lin_end(conv_end(h)))  — depthwise (halves H,W) then 1×1.
    let h4 = conv2d_dw_k3s2p1(&h3, c, h2d, w2, &dw.conv_end_w, &dw.conv_end_b);
    let (h3d, w3) = (conv_out_dim(h2d), conv_out_dim(w2)); // (16, T/8)
    let mut h5 = conv2d_1x1(&h4, c, h3d, w3, &dw.lin_end_w, &dw.lin_end_b);
    for v in h5.iter_mut() {
        *v = silu(*v);
    }

    // flatten(channels · height) → [C*H, W] then lin_out (Conv1d k1, [dim_out, C*H]).
    // PyTorch flatten(1,2) on [C, H, W] → combined index = c*H + h.
    let flat_cin = c * h3d; // 256*16 = 4096
    let mut flat = vec![0.0f32; flat_cin * w3];
    for ci in 0..c {
        for hi in 0..h3d {
            for wi in 0..w3 {
                flat[(ci * h3d + hi) * w3 + wi] = h5[ci * h3d * w3 + hi * w3 + wi];
            }
        }
    }
    // lin_out: [dim_out, flat_cin] · [flat_cin, W] → [dim_out, W]
    let dim_out = dw.lin_out_w.shape[0]; // 768
    let mut out = vec![0.0f32; dim_out * w3];
    for co in 0..dim_out {
        let b = dw.lin_out_b[co];
        let wbase = co * flat_cin;
        for wi in 0..w3 {
            let mut acc = b;
            for ci in 0..flat_cin {
                acc += flat[ci * w3 + wi] * dw.lin_out_w.data[wbase + ci];
            }
            out[co * w3 + wi] = acc;
        }
    }

    // transpose [dim_out, n_tokens] → [n_tokens, dim_out].
    let n_tokens = w3;
    let mut tokens = vec![0.0f32; n_tokens * dim_out];
    for t in 0..n_tokens {
        for d in 0..dim_out {
            tokens[t * dim_out + d] = out[d * n_tokens + t];
        }
    }
    tokens
}

// ─── HFQ load helpers (mirror lfm3.rs) ───────────────────────────────────────

fn read_tensor(hfq: &HfqFile, name: &str) -> Result<(u8, Vec<u8>), String> {
    if let Some((info, data)) = hfq.tensor_data_vec(name) {
        return Ok((info.quant_type, data));
    }
    Err(format!("lfm3-audio: tensor not found in HFQ: {name}"))
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

/// Load a 1D/raw F16/F32/Q8 vector → F32 GpuTensor (norms, conv filter).
fn load_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: &[usize]) -> Result<GpuTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let f32_data: Vec<f32> = match qt {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        3 => dequant_q8_0(&data),
        other => return Err(format!("lfm3-audio: expected F16/F32/Q8 for {name}, got qt={other}")),
    };
    gpu.upload_f32(&f32_data, shape).map_err(|e| format!("lfm3-audio: upload {name}: {e:?}"))
}

/// Load a 2D weight as a WeightTensor (Q8/F16/MQ-family) for `weight_gemv`.
fn load_wt(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize) -> Result<WeightTensor, String> {
    let (qt, data) = read_tensor(hfq, name)?;
    let dtype = match qt {
        3 => DType::Q8_0,
        6 => DType::HFQ4G256,
        8 => DType::HFQ6G256,
        13 => DType::MQ4G256,
        15 => DType::MQ6G256,
        17 => DType::MQ3G256,
        1 => DType::F16,
        other => return Err(format!("lfm3-audio: unsupported quant_type {other} for {name}")),
    };
    let buf = gpu.upload_raw(&data, &[data.len()]).map_err(|e| format!("upload_raw: {e:?}"))?;
    Ok(WeightTensor { buf, gpu_dtype: dtype, m, k, row_stride: 0, paro: None, awq_scale: None, lloyd_lut_e4m3: None, lloyd_lut_f16: None, lloyd_lut_c16: None })
}

/// Load a downsampler weight (F16/F32/Q8) → host F32 Vec + shape.
fn load_host_w(hfq: &HfqFile, name: &str) -> Result<HostW, String> {
    let (info, data) = hfq.tensor_data_vec(name).ok_or_else(|| format!("lfm3-audio: {name} not found"))?;
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        3 => dequant_q8_0(&data),
        other => return Err(format!("lfm3-audio: downsampler {name}: unsupported qt={other}")),
    };
    Ok(HostW { data: f32_data, shape: info.shape.iter().map(|&s| s as usize).collect() })
}

fn load_host_vec(hfq: &HfqFile, name: &str) -> Result<Vec<f32>, String> {
    Ok(load_host_w(hfq, name)?.data)
}

// ─── Encoder weights (GPU) ───────────────────────────────────────────────────

/// LFM2 conv mixer weights (K=3 for the audio encoder).
pub struct EncConvWeights {
    pub in_proj: WeightTensor,   // [3*hidden, hidden]
    pub conv_weight: GpuTensor,  // [hidden, K] F32
    pub out_proj: WeightTensor,  // [hidden, hidden]
    pub conv_state_idx: usize,
}

/// LFM2 attention weights (full-rotary, QK-norm; NO sigmoid gate in LFM2).
pub struct EncAttnWeights {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wo: WeightTensor,
    pub q_norm: GpuTensor, // [head_dim] — LFM2 names this q_layernorm
    pub k_norm: GpuTensor,
    pub kv_idx: usize,
}

pub enum EncMixer {
    Conv(EncConvWeights),
    Attention(EncAttnWeights),
}

pub struct EncLayerWeights {
    pub operator_norm: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub mixer: EncMixer,
    pub ffn: DenseFfn,
}

pub struct DenseFfn {
    pub w1: WeightTensor,
    pub w3: WeightTensor,
    pub w2: WeightTensor,
}

pub struct EncoderWeights {
    pub embedding_norm: GpuTensor,
    pub layers: Vec<EncLayerWeights>,
}

/// Audio projector weights (LayerNorm + GELU MLP, GPU F16).
pub struct ProjectorWeights {
    pub layer_norm_w: GpuTensor, // [768] F32
    pub layer_norm_b: GpuTensor,
    pub linear_1_w: GpuTensor, // [2048, 768] F16
    pub linear_1_b: GpuTensor, // [2048] F32
    pub linear_2_w: GpuTensor, // [2048, 2048] F16
    pub linear_2_b: GpuTensor, // [2048] F32
}

pub struct AudioWeights {
    pub downsampler: DownsamplerWeights,
    pub encoder: EncoderWeights,
    pub projector: ProjectorWeights,
}

// ─── GPU teardown ────────────────────────────────────────────────────────────

impl EncConvWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.in_proj.free_all(gpu);
        let _ = gpu.free_tensor(self.conv_weight);
        self.out_proj.free_all(gpu);
    }
}

impl EncAttnWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.q_norm);
        let _ = gpu.free_tensor(self.k_norm);
    }
}

impl EncMixer {
    fn free_gpu(self, gpu: &mut Gpu) {
        match self {
            EncMixer::Conv(c) => c.free_gpu(gpu),
            EncMixer::Attention(a) => a.free_gpu(gpu),
        }
    }
}

impl DenseFfn {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.w1.free_all(gpu);
        self.w3.free_all(gpu);
        self.w2.free_all(gpu);
    }
}

impl EncLayerWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.operator_norm);
        let _ = gpu.free_tensor(self.ffn_norm);
        self.mixer.free_gpu(gpu);
        self.ffn.free_gpu(gpu);
    }
}

impl EncoderWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.embedding_norm);
        for l in self.layers {
            l.free_gpu(gpu);
        }
    }
}

impl ProjectorWeights {
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.layer_norm_w);
        let _ = gpu.free_tensor(self.layer_norm_b);
        let _ = gpu.free_tensor(self.linear_1_w);
        let _ = gpu.free_tensor(self.linear_1_b);
        let _ = gpu.free_tensor(self.linear_2_w);
        let _ = gpu.free_tensor(self.linear_2_b);
    }
}

impl AudioWeights {
    /// Free GPU tensors (encoder + projector). The downsampler weights are
    /// host-side CPU `Vec<f32>` and are dropped automatically.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.encoder.free_gpu(gpu);
        self.projector.free_gpu(gpu);
    }
}

// ─── Weight loading ──────────────────────────────────────────────────────────

fn load_downsampler(hfq: &HfqFile) -> Result<DownsamplerWeights, String> {
    // AUDIO-NAME-1: Audio-2B raw names the subtree `encoder.downsample.*`;
    // Omni-2B names it `model.audio_encoder.downsample.*`.
    let p = if hfq
        .find_tensor_info("model.audio_encoder.downsample.conv_in.weight")
        .is_some()
    {
        "model.audio_encoder.downsample."
    } else {
        "encoder.downsample."
    };
    Ok(DownsamplerWeights {
        conv_in_w: load_host_w(hfq, &format!("{p}conv_in.weight"))?,
        conv_in_b: load_host_vec(hfq, &format!("{p}conv_in.bias"))?,
        conv_mid_w: load_host_w(hfq, &format!("{p}conv_mid.weight"))?,
        conv_mid_b: load_host_vec(hfq, &format!("{p}conv_mid.bias"))?,
        lin_mid_w: load_host_w(hfq, &format!("{p}lin_mid.weight"))?,
        lin_mid_b: load_host_vec(hfq, &format!("{p}lin_mid.bias"))?,
        conv_end_w: load_host_w(hfq, &format!("{p}conv_end.weight"))?,
        conv_end_b: load_host_vec(hfq, &format!("{p}conv_end.bias"))?,
        lin_end_w: load_host_w(hfq, &format!("{p}lin_end.weight"))?,
        lin_end_b: load_host_vec(hfq, &format!("{p}lin_end.bias"))?,
        lin_out_w: load_host_w(hfq, &format!("{p}lin_out.weight"))?,
        lin_out_b: load_host_vec(hfq, &format!("{p}lin_out.bias"))?,
    })
}

fn load_encoder(hfq: &HfqFile, cfg: &EncoderConfig, gpu: &mut Gpu) -> Result<EncoderWeights, String> {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let head_dim = cfg.head_dim;
    let inter = cfg.intermediate_size;
    let k_conv = cfg.conv_kernel_size;
    // AUDIO-NAME-1: Audio-2B nests the backbone `encoder.lfm.lfm`; Omni-2B
    // flattens it to `model.audio_encoder.lfm`.
    let pfx = if hfq
        .find_tensor_info("model.audio_encoder.lfm.layers.0.operator_norm.weight")
        .is_some()
    {
        "model.audio_encoder.lfm"
    } else {
        "encoder.lfm.lfm"
    };

    let embedding_norm = load_f32(hfq, gpu, &format!("{pfx}.embedding_norm.weight"), &[h])?;

    let mut conv_count = 0usize;
    let mut kv_count = 0usize;
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let p = format!("{pfx}.layers.{l}");
        let operator_norm = load_f32(hfq, gpu, &format!("{p}.operator_norm.weight"), &[h])?;
        let ffn_norm = load_f32(hfq, gpu, &format!("{p}.ffn_norm.weight"), &[h])?;

        let mixer = match cfg.layer_types[l] {
            EncLayerKind::Conv => {
                let in_proj = load_wt(hfq, gpu, &format!("{p}.conv.in_proj.weight"), 3 * h, h)?;
                let conv_weight = load_f32(hfq, gpu, &format!("{p}.conv.conv.weight"), &[h * k_conv])?;
                let out_proj = load_wt(hfq, gpu, &format!("{p}.conv.out_proj.weight"), h, h)?;
                let conv_state_idx = conv_count;
                conv_count += 1;
                EncMixer::Conv(EncConvWeights { in_proj, conv_weight, out_proj, conv_state_idx })
            }
            EncLayerKind::Attention => {
                let wq = load_wt(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
                let wk = load_wt(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
                let wv = load_wt(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
                let wo = load_wt(hfq, gpu, &format!("{p}.self_attn.out_proj.weight"), h, q_dim)?;
                // LFM2 names these q_layernorm / k_layernorm.
                let q_norm = load_f32(hfq, gpu, &format!("{p}.self_attn.q_layernorm.weight"), &[head_dim])?;
                let k_norm = load_f32(hfq, gpu, &format!("{p}.self_attn.k_layernorm.weight"), &[head_dim])?;
                let kv_idx = kv_count;
                kv_count += 1;
                EncMixer::Attention(EncAttnWeights { wq, wk, wv, wo, q_norm, k_norm, kv_idx })
            }
        };

        let w1 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w1.weight"), inter, h)?;
        let w3 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w3.weight"), inter, h)?;
        let w2 = load_wt(hfq, gpu, &format!("{p}.feed_forward.w2.weight"), h, inter)?;
        let ffn = DenseFfn { w1, w3, w2 };

        layers.push(EncLayerWeights { operator_norm, ffn_norm, mixer, ffn });
    }
    let _ = (conv_count, kv_count);
    Ok(EncoderWeights { embedding_norm, layers })
}

fn load_projector(hfq: &HfqFile, cfg: &ProjectorConfig, gpu: &mut Gpu) -> Result<ProjectorWeights, String> {
    // AUDIO-NAME-1: two projector namings. Audio-2B raw names it `mlp.model.*`
    // with numeric indices 0=LayerNorm, 1=linear_1, 3=linear_2 (GELU at 2).
    // Omni-2B names it `model.audio_projector.{layer_norm,linear_1,linear_2}`.
    if hfq.find_tensor_info("model.audio_projector.linear_1.weight").is_some() {
        let p = "model.audio_projector.";
        let layer_norm_w = load_f32(hfq, gpu, &format!("{p}layer_norm.weight"), &[cfg.in_features])?;
        let layer_norm_b = load_f32(hfq, gpu, &format!("{p}layer_norm.bias"), &[cfg.in_features])?;
        let linear_1_w = load_f16_gpu_host(hfq, gpu, &format!("{p}linear_1.weight"))?;
        let linear_1_b = load_f32(hfq, gpu, &format!("{p}linear_1.bias"), &[cfg.out_features])?;
        let linear_2_w = load_f16_gpu_host(hfq, gpu, &format!("{p}linear_2.weight"))?;
        let linear_2_b = load_f32(hfq, gpu, &format!("{p}linear_2.bias"), &[cfg.out_features])?;
        return Ok(ProjectorWeights { layer_norm_w, layer_norm_b, linear_1_w, linear_1_b, linear_2_w, linear_2_b });
    }
    let p = "mlp.model.";
    let layer_norm_w = load_f32(hfq, gpu, &format!("{p}0.weight"), &[cfg.in_features])?;
    let layer_norm_b = load_f32(hfq, gpu, &format!("{p}0.bias"), &[cfg.in_features])?;
    let linear_1_w = load_f16_gpu_host(hfq, gpu, &format!("{p}1.weight"))?;
    let linear_1_b = load_f32(hfq, gpu, &format!("{p}1.bias"), &[cfg.out_features])?;
    let linear_2_w = load_f16_gpu_host(hfq, gpu, &format!("{p}3.weight"))?;
    let linear_2_b = load_f32(hfq, gpu, &format!("{p}3.bias"), &[cfg.out_features])?;
    Ok(ProjectorWeights { layer_norm_w, layer_norm_b, linear_1_w, linear_1_b, linear_2_w, linear_2_b })
}

/// Load an F16 (or dequant Q8→F16) weight as a raw GPU tensor for `gemm_f16`.
fn load_f16_gpu_host(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let (info, data) = hfq.tensor_data_vec(name).ok_or_else(|| format!("lfm3-audio: {name} not found"))?;
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match info.quant_type {
        1 => gpu.upload_raw(&data, &[n]).map_err(|e| format!("lfm3-audio: upload {name}: {e:?}")),
        3 => {
            let f32_data = dequant_q8_0(&data);
            let f16_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect();
            gpu.upload_raw(&f16_bytes, &[n]).map_err(|e| format!("lfm3-audio: upload {name}: {e:?}"))
        }
        other => Err(format!("lfm3-audio: {name}: unsupported qt={other} (expected F16=1 or Q8=3)")),
    }
}

pub fn load_audio_weights(hfq: &HfqFile, cfg: &AudioConfig, gpu: &mut Gpu) -> HipResult<AudioWeights> {
    eprintln!("  loading LFM3-Audio downsampler weights (CPU)...");
    let downsampler = load_downsampler(hfq).map_err(|e| HipError::new(999, &e))?;
    eprintln!("  loading LFM3-Audio encoder weights (GPU)...");
    let encoder = load_encoder(hfq, &cfg.enc, gpu).map_err(|e| HipError::new(999, &e))?;
    eprintln!("  loading LFM3-Audio projector weights (GPU)...");
    let projector = load_projector(hfq, &cfg.proj, gpu).map_err(|e| HipError::new(999, &e))?;
    Ok(AudioWeights { downsampler, encoder, projector })
}

// ─── Encoder state (GPU scratch + KV cache + conv state) ─────────────────────

/// Per-encode GPU scratch. Allocated once per `audio_forward` call and freed
/// at the end (the encoder is one-shot per query, unlike the text decoder
/// whose state persists across turns).
pub struct AudioEncoderState {
    pub kv: KvCache,
    pub conv_states: Vec<GpuTensor>,
    pub pos_buf: hip_bridge::DeviceBuffer,
    /// Absolute position counter (carries across chunks for streaming).
    pub n_tokens: usize,
    pub h: GpuTensor,
    pub tmp: GpuTensor,
    pub fa_q: GpuTensor,
    pub fa_k: GpuTensor,
    pub fa_v: GpuTensor,
    pub fa_attn_out: GpuTensor,
    pub flash_partials: GpuTensor,
    pub conv_bcx: GpuTensor,
    pub conv_y: GpuTensor,
    pub ffn_tmp: GpuTensor,
    pub dense_gate: GpuTensor,
    pub dense_up: GpuTensor,
    pub dense_act: GpuTensor,
    pub proj_out: GpuTensor,
    pub final_norm_buf: GpuTensor,
}

const ENC_FLASH_SUBBATCH: usize = 64;

impl AudioEncoderState {
    pub fn new(gpu: &mut Gpu, cfg: &EncoderConfig, max_seq: usize) -> Result<Self, String> {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let k_conv = cfg.conv_kernel_size;
        let head_dim = cfg.head_dim;

        gpu.ensure_mq_signs().map_err(|e| format!("lfm3-audio: ensure_mq_signs: {e:?}"))?;

        let n_attn = cfg.num_attention_layers().max(1);
        let dims = hipfire_runtime::llama::KvDims {
            layers: hipfire_runtime::llama::KvLayers::Flat(n_attn),
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            max_seq,
            physical_cap: None,
        };
        let kv = KvCache::from_mode(
            hipfire_runtime::kv_mode::resolve("", &hipfire_runtime::kv_mode::HFQ_Q8_ONLY_POLICY).mode,
            gpu,
            &dims,
        )
        .map_err(|e| format!("lfm3-audio: kv cache: {e:?}"))?;

        let conv_hist = hidden * (k_conv - 1);
        let zeros = vec![0u8; conv_hist * 4];
        let mut conv_states = Vec::with_capacity(cfg.num_conv_layers());
        for _ in 0..cfg.num_conv_layers() {
            let cs = gpu.alloc_tensor(&[conv_hist], DType::F32).map_err(|e| format!("lfm3-audio: alloc conv_state: {e:?}"))?;
            gpu.hip.memcpy_htod(&cs.buf, &zeros).map_err(|e| format!("lfm3-audio: zero conv_state: {e:?}"))?;
            conv_states.push(cs);
        }

        let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("lfm3-audio: pos_buf: {e:?}"))?;
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32).map_err(|e| format!("lfm3-audio: alloc {label}: {e:?}"))
        };

        Ok(AudioEncoderState {
            kv,
            conv_states,
            pos_buf,
            n_tokens: 0,
            h: alloc(gpu, hidden, "h")?,
            tmp: alloc(gpu, hidden, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            flash_partials: alloc(
                gpu,
                n_heads * ((max_seq + 127) / 128) * (2 + head_dim) * ENC_FLASH_SUBBATCH,
                "flash_partials",
            )?,
            conv_bcx: alloc(gpu, 3 * hidden, "conv_bcx")?,
            conv_y: alloc(gpu, hidden, "conv_y")?,
            ffn_tmp: alloc(gpu, hidden, "ffn_tmp")?,
            dense_gate: alloc(gpu, inter, "dense_gate")?,
            dense_up: alloc(gpu, inter, "dense_up")?,
            dense_act: alloc(gpu, inter, "dense_act")?,
            proj_out: alloc(gpu, hidden, "proj_out")?,
            final_norm_buf: alloc(gpu, hidden, "final_norm_buf")?,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        for cs in &self.conv_states {
            let zeros = vec![0u8; cs.numel() * 4];
            gpu.hip.memcpy_htod(&cs.buf, &zeros).map_err(|e| format!("lfm3-audio: reset conv_state: {e:?}"))?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.h, self.tmp, self.fa_q, self.fa_k, self.fa_v, self.fa_attn_out, self.flash_partials,
            self.conv_bcx, self.conv_y, self.ffn_tmp, self.dense_gate, self.dense_up, self.dense_act,
            self.proj_out, self.final_norm_buf,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // LEAK-1: the 4-byte position scalar buffer was never freed.
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── Encoder forward (GPU, sequential prefill) ───────────────────────────────
// Mirrors lfm3::forward::decode_step_inner minus sandwich norm / sigmoid gate,
// with conv K=3, full-rotary RoPE (n_rot=head_dim, θ=1e6), and window=32 causal
// attention on every attention layer (the HF encoder bounds full_attention to
// sliding_window=32 via create_sliding_window_causal_mask). Input is the
// downsampler token embedding (not a token-id lookup); output is the
// embedding_norm'd hidden state per token (last_hidden_state).

fn enc_conv_mixer(
    gpu: &mut Gpu, cfg: &EncoderConfig, c: &EncConvWeights, st: &AudioEncoderState, l: usize,
) -> Result<(), String> {
    let hidden = cfg.hidden_size;
    weight_gemv(gpu, &c.in_proj, &st.tmp, &st.conv_bcx)
        .map_err(|e| format!("lfm3-audio L{l}: conv in_proj: {e}"))?;
    gpu.conv1d_gated_decode_f32(&st.conv_bcx, &st.conv_states[c.conv_state_idx], &c.conv_weight, &st.conv_y, 1, hidden, cfg.conv_kernel_size)
        .map_err(|e| format!("lfm3-audio L{l}: conv gated decode: {e:?}"))?;
    weight_gemv(gpu, &c.out_proj, &st.conv_y, &st.proj_out)
        .map_err(|e| format!("lfm3-audio L{l}: conv out_proj: {e}"))
}

fn enc_attn_mixer(
    gpu: &mut Gpu, cfg: &EncoderConfig, a: &EncAttnWeights, st: &mut AudioEncoderState,
    l: usize, seq_len: usize,
) -> Result<(), String> {
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.norm_eps;

    weight_gemv(gpu, &a.wq, &st.tmp, &st.fa_q).map_err(|e| format!("lfm3-audio L{l}: q_proj: {e}"))?;
    weight_gemv(gpu, &a.wk, &st.tmp, &st.fa_k).map_err(|e| format!("lfm3-audio L{l}: k_proj: {e}"))?;
    weight_gemv(gpu, &a.wv, &st.tmp, &st.fa_v).map_err(|e| format!("lfm3-audio L{l}: v_proj: {e}"))?;

    // Per-head QK-norm (LFM2 q_layernorm / k_layernorm).
    gpu.rmsnorm_batched(&st.fa_q, &a.q_norm, &st.fa_q, n_heads, head_dim, eps)
        .map_err(|e| format!("lfm3-audio L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&st.fa_k, &a.k_norm, &st.fa_k, n_kv, head_dim, eps)
        .map_err(|e| format!("lfm3-audio L{l}: k_norm: {e:?}"))?;

    // Full-rotary RoPE (n_rot = head_dim, θ = rope_theta).
    gpu.rope_partial_interleaved_f32(&st.fa_q, &st.fa_k, &st.pos_buf, n_heads, n_kv, head_dim, head_dim, cfg.rope_theta)
        .map_err(|e| format!("lfm3-audio L{l}: rope: {e:?}"))?;

    // KV cache write (Q8).
    let kv_idx = a.kv_idx;
    gpu.kv_cache_write_q8_0(&st.kv.k_gpu[kv_idx], &st.fa_k, &st.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm3-audio L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&st.kv.v_gpu[kv_idx], &st.fa_v, &st.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("lfm3-audio L{l}: kv write v: {e:?}"))?;

    // Windowed causal attention (window=32 — the encoder bounds every attn layer).
    let ctx = DispatchCtx::new(gpu);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: seq_len - 1,
        q8_windowed: true,
        window: cfg.sliding_window as i32,
        ..st.kv.tier_inputs()
    })
    .map_err(|e| format!("lfm3-audio L{l}: kv tier: {e}"))?;
    let io = AttnParams {
        q: &st.fa_q,
        k: &st.fa_k,
        v: &st.fa_v,
        k_cache: &st.kv.k_gpu[kv_idx],
        v_cache: &st.kv.v_gpu[kv_idx],
        k_scales: None,
        v_scales: None,
        pos_buf: &st.pos_buf,
        pos: seq_len - 1,
        positions: None,
        n_heads,
        n_kv_heads: n_kv,
        head_dim,
        physical_cap: st.kv.physical_cap,
        batch_size: 1,
        max_ctx_len: 0,
        flash_partials: Some(&st.flash_partials),
        givens_cos: None,
        givens_sin: None,
        tree_bias: None,
        block_start: 0,
        block_cols: 0,
        output_gate: None,
        output_awq_scale: None,
        output: &st.fa_attn_out,
    };
    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }])
        .map_err(|e| format!("lfm3-audio L{l}: attention: {e:?}"))?;

    // o_proj → proj_out (NO sigmoid gate in LFM2).
    weight_gemv(gpu, &a.wo, &st.fa_attn_out, &st.proj_out)
        .map_err(|e| format!("lfm3-audio L{l}: out_proj: {e}"))
}

/// Run the 17-layer encoder over `tokens` (`[n_tokens, hidden]` host F32),
/// returning `[n_tokens, hidden]` last_hidden_state (after embedding_norm).
/// Uses the caller-provided `st` (which carries the KV cache + conv state +
/// position across calls for streaming). The position starts at `st.n_tokens`
/// and advances by `n_tokens`.
pub fn encoder_forward(
    gpu: &mut Gpu,
    weights: &EncoderWeights,
    cfg: &EncoderConfig,
    st: &mut AudioEncoderState,
    tokens: &[f32],
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let hidden = cfg.hidden_size;
    let eps = cfg.norm_eps;
    let base = st.n_tokens; // absolute position offset (carries across chunks)

    let mut out = vec![0.0f32; n_tokens * hidden];

    for i in 0..n_tokens as u32 {
        let pos = base as u32 + i;
        let seq_len = pos as usize + 1;
        gpu.hip.memcpy_htod(&st.pos_buf, &(pos as i32).to_ne_bytes())
            .map_err(|e| format!("lfm3-audio: htod pos: {e:?}"))?;

        // Input = downsampler token embedding (not a token-id lookup).
        let embed = &tokens[i as usize * hidden..(i as usize + 1) * hidden];
        let bytes: Vec<u8> = embed.iter().flat_map(|&f| f.to_le_bytes()).collect();
        gpu.hip.memcpy_htod(&st.h.buf, &bytes).map_err(|e| format!("lfm3-audio: htod embed: {e:?}"))?;

        for (l, layer) in weights.layers.iter().enumerate() {
            // operator_norm → tmp
            gpu.rmsnorm_f32(&st.h, &layer.operator_norm, &st.tmp, eps)
                .map_err(|e| format!("lfm3-audio L{l}: operator rmsnorm: {e:?}"))?;
            match &layer.mixer {
                EncMixer::Conv(c) => enc_conv_mixer(gpu, cfg, c, st, l)?,
                EncMixer::Attention(a) => enc_attn_mixer(gpu, cfg, a, st, l, seq_len)?,
            }

            // Residual (no sandwich norm in the audio encoder).
            gpu.add_f32(&st.proj_out, &st.h, &st.h).map_err(|e| format!("lfm3-audio L{l}: operator residual: {e:?}"))?;

            // AUDIO-DBG: per-layer hidden hash — the first layer whose hash
            // diverges across identical runs pins the nondeterministic op.

            // FFN: ffn_norm → SwiGLU.
            gpu.rmsnorm_f32(&st.h, &layer.ffn_norm, &st.ffn_tmp, eps)
                .map_err(|e| format!("lfm3-audio L{l}: ffn rmsnorm: {e:?}"))?;
            let d = &layer.ffn;
            weight_gemv(gpu, &d.w1, &st.ffn_tmp, &st.dense_gate).map_err(|e| format!("lfm3-audio L{l}: w1: {e}"))?;
            weight_gemv(gpu, &d.w3, &st.ffn_tmp, &st.dense_up).map_err(|e| format!("lfm3-audio L{l}: w3: {e}"))?;
            gpu.silu_mul_f32(&st.dense_gate, &st.dense_up, &st.dense_act).map_err(|e| format!("lfm3-audio L{l}: silu_mul: {e:?}"))?;
            weight_gemv(gpu, &d.w2, &st.dense_act, &st.proj_out).map_err(|e| format!("lfm3-audio L{l}: w2: {e}"))?;
            gpu.add_f32(&st.proj_out, &st.h, &st.h).map_err(|e| format!("lfm3-audio L{l}: ffn residual: {e:?}"))?;
        }

        // Final embedding_norm → last_hidden_state for this token.
        gpu.rmsnorm_f32(&st.h, &weights.embedding_norm, &st.final_norm_buf, eps)
            .map_err(|e| format!("lfm3-audio: final rmsnorm: {e:?}"))?;
        let h = gpu.download_f32(&st.final_norm_buf).map_err(|e| format!("lfm3-audio: download hidden: {e:?}"))?;
        out[i as usize * hidden..(i as usize + 1) * hidden].copy_from_slice(&h);
    }

    st.n_tokens = base + n_tokens; // advance the absolute position
    Ok(out)
}

// ─── Projector forward (GPU) ─────────────────────────────────────────────────
// LayerNorm(768) → Linear(768→2048) + bias → GELU(erf) → Linear(2048→2048) + bias.
// Reuses the lfm3-vl linear helpers (gemm_f16 + transpose + bias_add).

fn linear_f16_bias(
    gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, bias: &GpuTensor, out_dim: usize, in_dim: usize, n: usize,
) -> Result<GpuTensor, String> {
    let yt = gpu.alloc_tensor(&[out_dim * n], DType::F32).map_err(|e| format!("alloc yt: {e:?}"))?;
    gpu.gemm_f16(w, x, &yt, out_dim, in_dim, n).map_err(|e| format!("gemm_f16: {e:?}"))?;
    let y = gpu.alloc_tensor(&[n * out_dim], DType::F32).map_err(|e| format!("alloc y: {e:?}"))?;
    gpu.transpose_f32(&yt, &y, out_dim, n).map_err(|e| format!("transpose: {e:?}"))?;
    gpu.free_tensor(yt).map_err(|e| format!("free yt: {e:?}"))?;
    gpu.bias_add_f32(&y, bias, n, out_dim).map_err(|e| format!("bias_add: {e:?}"))?;
    Ok(y)
}

pub fn projector_forward(
    gpu: &mut Gpu, w: &ProjectorWeights, cfg: &ProjectorConfig, enc_out: &[f32], n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let in_f = cfg.in_features;
    let out_f = cfg.out_features;
    let x = gpu.upload_f32(enc_out, &[n_tokens * in_f]).map_err(|e| format!("upload enc: {e:?}"))?;

    // LayerNorm(768) with weight + bias.
    let normed = gpu.alloc_tensor(&[n_tokens * in_f], DType::F32).map_err(|e| format!("alloc normed: {e:?}"))?;
    gpu.layernorm_batched(&x, &w.layer_norm_w, &w.layer_norm_b, &normed, n_tokens, in_f, cfg.norm_eps)
        .map_err(|e| format!("projector layernorm: {e:?}"))?;
    gpu.free_tensor(x).map_err(|e| format!("free x: {e:?}"))?;

    // linear_1 (768→2048) + bias → GELU(erf).
    let m1 = linear_f16_bias(gpu, &w.linear_1_w, &normed, &w.linear_1_b, out_f, in_f, n_tokens)?;
    gpu.free_tensor(normed).map_err(|e| format!("free normed: {e:?}"))?;
    gpu.gelu_erf_f32(&m1, &m1, n_tokens * out_f).map_err(|e| format!("gelu_erf: {e:?}"))?;

    // linear_2 (2048→2048) + bias.
    let m2 = linear_f16_bias(gpu, &w.linear_2_w, &m1, &w.linear_2_b, out_f, out_f, n_tokens)?;
    gpu.free_tensor(m1).map_err(|e| format!("free m1: {e:?}"))?;

    let result = gpu.download_f32(&m2).map_err(|e| format!("download proj: {e:?}"))?;
    gpu.free_tensor(m2).map_err(|e| format!("free m2: {e:?}"))?;
    Ok(result)
}

// ─── Full audio forward: mel → downsampler → encoder → projector ─────────────

// AUDIO-DBG helpers (HIPFIRE_AUDIO_DEBUG=1): per-stage FNV-1a hashes so a
// hash that differs across identical runs pins the nondeterministic GPU stage.
fn dbg_audio() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("HIPFIRE_AUDIO_DEBUG").ok().as_deref() == Some("1"))
}
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn audio_forward(
    gpu: &mut Gpu,
    weights: &AudioWeights,
    cfg: &AudioConfig,
    mel: &[f32],
    num_frames: usize,
) -> HipResult<Vec<f32>> {
    let n_mels = cfg.mel.n_mels;
    // AUDIO-DBG: per-stage FNV-1a hashes (HIPFIRE_AUDIO_DEBUG=1). The mel is
    // pure CPU (deterministic); a hash that differs across identical runs pins
    // the nondeterministic GPU stage (encoder vs projector).
    let dbg = dbg_audio();
    // 1. CPU downsampler: mel [n_mels, num_frames] → [n_tokens, 768].
    let tokens = downsampler_forward(mel, n_mels, num_frames, &weights.downsampler);
    let n_tokens = tokens.len() / cfg.enc.hidden_size;
    if dbg {
        let b: Vec<u8> = tokens.iter().flat_map(|f| f.to_le_bytes()).collect();
        eprintln!(
            "[lfm3-audio][dbg] downsampler hash {:016x} ({} tok)",
            fnv1a(&b),
            n_tokens
        );
    }

    // 2. GPU encoder: [n_tokens, 768] → last_hidden_state [n_tokens, 768].
    let enc_out = {
        let max_seq = n_tokens.max(64);
        let mut st = AudioEncoderState::new(gpu, &cfg.enc, max_seq)
            .map_err(|e| HipError::new(999, &format!("lfm3-audio: encoder state: {e}")))?;
        st.reset(gpu).map_err(|e| HipError::new(999, &e))?;
        let out = encoder_forward(gpu, &weights.encoder, &cfg.enc, &mut st, &tokens, n_tokens)
            .map_err(|e| HipError::new(999, &e))?;
        st.free_gpu(gpu);
        if dbg {
            let b: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
            eprintln!("[lfm3-audio][dbg] encoder hash {:016x}", fnv1a(&b));
        }
        out
    };

    // 3. GPU projector: [n_tokens, 768] → [n_tokens, 2048] (text-hidden audio tokens).
    let proj = projector_forward(gpu, &weights.projector, &cfg.proj, &enc_out, n_tokens)
        .map_err(|e| HipError::new(999, &e))?;
    if dbg {
        let b: Vec<u8> = proj.iter().flat_map(|f| f.to_le_bytes()).collect();
        eprintln!("[lfm3-audio][dbg] projector hash {:016x}", fnv1a(&b));
    }

    eprintln!("  LFM3-Audio forward done: {n_tokens} audio tokens x {} dims", cfg.proj.out_features);
    Ok(proj)
}

// ─── Streaming ASR (Phase 4d) ────────────────────────────────────────────────

/// Streaming ASR state: carries the encoder LFM2 KV cache + conv state +
/// position across audio chunks. Mirrors HF `Lfm3OmniEncoderStreamingCache`
/// (the Phase 3 deferred item) — enables `transcribe_live` (live
/// transcription) and chunked `prefill_audio` (audio tokens appended to the
/// LM cache as chunks complete).
///
/// ASR-STR-1: unlike the first cut (per-chunk mel + per-chunk downsample +
/// permanent drop of every <8-frame tail), the cache carries enough history
/// that streaming matches the batch path exactly: the mel and downsampler
/// see the same inputs they would over the full stream, tokens are emitted
/// only when their full 8-frame group exists, and `is_final` flushes the
/// partial trailing group as one last token (HF: `(num_frames + 7) // 8`).
///
/// REVIEW-3 (bounded streaming): the first cut accumulated EVERY raw sample
/// and recomputed the mel + downsampler over the whole stream on each call —
/// bit-identical to batch, but O(n²) CPU in the stream length with unbounded
/// host memory over a live session. Both stages are local in time, so the
/// recompute is bounded now with ZERO output change:
///   * mel frame f covers samples [f·hop, f·hop + n_fft) — independent per
///     frame (no global normalization; the affine is per-element), so only
///     the sample tail from `frames_computed · hop` onward is retained and
///     only NEW frames are computed;
///   * one downsampler output token depends only on mel frames
///     [8t−8, 8t+8) (three k3s2p1 convs = ±8-frame receptive field), so the
///     downsampler runs on a bounded window starting at a multiple of 8
///     (the window's conv padding then matches the full stream exactly for
///     every emitted token — see `downsampler_window_bit_identity`).
pub struct AudioEncoderStreamingCache {
    pub state: AudioEncoderState,
    /// Raw samples from absolute sample `frames_computed * hop` onward —
    /// exactly the history future mel frames still need (< n_fft samples
    /// between chunks, plus the just-arrived chunk). Bounded now.
    pub accum_samples: Vec<f32>,
    /// Trailing mel frames stored FRAME-MAJOR (`[frame, n_mels]`, absolute
    /// frames `[mel_frame_start, frames_computed)`) so a frame range is a
    /// contiguous slice — NOTE this is the transpose of
    /// `log_mel_spectrogram`'s `[n_mels, num_frames]` output layout. Capped
    /// at [`DOWNSAMPLER_RETAIN_FRAMES`].
    pub mel_cache: Vec<f32>,
    /// Absolute frame index of `mel_cache[0]`.
    pub mel_frame_start: usize,
    /// Total mel frames computed so far (= absolute stream frame count).
    pub frames_computed: usize,
    /// Tokens already emitted (HF `cache.num_tokens`).
    pub num_tokens_emitted: usize,
    /// Set by the `is_final` flush; further chunks are rejected (HF raises
    /// "This stream is finalized").
    pub finished: bool,
}

/// Mel frames retained for the downsampler windows. One window needs at most
/// frames [8·t−8, 8·t+8) (24 frames for the longest final-flush window);
/// 32 gives margin. Old frames can never enter a future window.
const DOWNSAMPLER_RETAIN_FRAMES: usize = 32;

impl AudioEncoderStreamingCache {
    pub fn new(gpu: &mut Gpu, cfg: &AudioConfig, max_tokens: usize) -> Result<Self, String> {
        let mut state = AudioEncoderState::new(gpu, &cfg.enc, max_tokens.max(64))?;
        state.reset(gpu)?;
        Ok(Self {
            state,
            accum_samples: Vec::new(),
            mel_cache: Vec::new(),
            mel_frame_start: 0,
            frames_computed: 0,
            num_tokens_emitted: 0,
            finished: false,
        })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.state.free_gpu(gpu);
    }
}

/// Streaming audio forward: process one audio chunk (mono F32 @ 16 kHz),
/// carrying the encoder state across chunks. Returns the projected audio
/// tokens `[n_tokens, 2048]` for the COMPLETE tokens in this chunk (a
/// partial trailing 8-frame group is held back — pass `is_final = true` on
/// the last chunk to flush it as one final token, mirroring HF
/// `num_tokens = (num_frames + (7 if is_final else 0)) // 8`).
///
/// Ground truth: HF `Lfm3OmniForConditionalGeneration.prefill_audio` /
/// `transcribe_live` / `Lfm3AudioEncoder.stream` (streaming ASR).
pub fn audio_forward_streaming(
    gpu: &mut Gpu,
    weights: &AudioWeights,
    cfg: &AudioConfig,
    cache: &mut AudioEncoderStreamingCache,
    audio_chunk: &[f32],
    is_final: bool,
) -> HipResult<Vec<f32>> {
    if cache.finished && !audio_chunk.is_empty() {
        return Err(HipError::new(
            999,
            "lfm3-audio: stream is finalized; start a new cache for the next utterance",
        ));
    }
    let n_mels = cfg.mel.n_mels;
    let hop = cfg.mel.hop_length;
    let n_fft = cfg.mel.n_fft;

    // 1. Accumulate the sample tail + compute only the NEW mel frames (REVIEW-3).
    //    `accum_samples` starts at absolute sample `frames_computed·hop`, so its
    //    frame j IS absolute frame `frames_computed + j` — each frame depends
    //    only on its own [f·hop, f·hop + n_fft) window, making the incremental
    //    frames bit-identical to the full-stream `log_mel_spectrogram`.
    cache.accum_samples.extend_from_slice(audio_chunk);
    let new_mel = crate::mel::log_mel_spectrogram(&cache.accum_samples, &cfg.mel);
    let new_frames = new_mel.len() / n_mels;
    cache.frames_computed += new_frames;
    // Transpose-append the new frames (log_mel is [n_mels, frames], the
    // cache is frame-major).
    for f in 0..new_frames {
        for m in 0..n_mels {
            cache.mel_cache.push(new_mel[m * new_frames + f]);
        }
    }
    let num_frames = cache.frames_computed;

    // Bound the sample tail: future frames start at frame `num_frames`, whose
    // window begins at absolute sample `num_frames·hop`. Everything before it
    // can no longer affect any output. (Bound: n_fft − hop + a sub-hop
    // remainder between chunks.)
    let drop_samples = new_frames * hop;
    cache.accum_samples.drain(..drop_samples.min(cache.accum_samples.len()));

    // 2. Token accounting (HF `stream()`): a token needs its full 8-frame
    //    group, except on the final flush where the partial group emits one
    //    last token.
    let num_tokens = if is_final { (num_frames + 7) / 8 } else { num_frames / 8 };
    let start_token = cache.num_tokens_emitted;
    let new_tokens = num_tokens.saturating_sub(start_token);
    if new_tokens == 0 {
        cache.num_tokens_emitted = num_tokens;
        cache.finished = is_final;
        return Ok(Vec::new());
    }

    // 3. CPU downsampler over a BOUNDED window of the mel (REVIEW-3), then
    //    slice this call's tokens — bit-identical to running it over the
    //    full stream (see the struct doc + `downsampler_window_bit_identity`).
    //    Token t depends only on mel frames [8t−8, 8t+8); a window starting
    //    at a multiple of 8 has conv-padding behavior identical to the full
    //    stream for every token it produces, and the window's token j maps to
    //    the full stream's token `ws/8 + j`.
    let win_start = 8 * start_token.saturating_sub(1); // multiple of 8
    if win_start < cache.mel_frame_start {
        return Err(HipError::new(
            999,
            &format!(
                "lfm3-audio: downsampler window underflow (need frames from {win_start}, cache retains from {})",
                cache.mel_frame_start
            ),
        ));
    }
    let win_frames = num_frames - win_start;
    // Materialize the window in downsampler layout ([n_mels, win_frames],
    // mel-major) from the frame-major cache.
    let mut win_mel = vec![0.0f32; n_mels * win_frames];
    let cache_base = (win_start - cache.mel_frame_start) * n_mels;
    for f in 0..win_frames {
        let src = &cache.mel_cache[cache_base + f * n_mels..cache_base + (f + 1) * n_mels];
        for m in 0..n_mels {
            win_mel[m * win_frames + f] = src[m];
        }
    }
    let tokens_win = downsampler_forward(&win_mel, n_mels, win_frames, &weights.downsampler);
    // Bound the mel cache AFTER the window is materialized: frames can be
    // dropped once they can never enter a future window — the next call's
    // window starts at 8·(num_tokens−1), and this call's window started at
    // `win_start`; nothing before min(those, num_frames−RETAIN) is needed.
    let keep_from = num_frames
        .saturating_sub(DOWNSAMPLER_RETAIN_FRAMES)
        .min(8 * num_tokens.saturating_sub(1))
        .min(win_start);
    if keep_from > cache.mel_frame_start {
        cache
            .mel_cache
            .drain(..(keep_from - cache.mel_frame_start) * n_mels);
        cache.mel_frame_start = keep_from;
    }
    let avail = tokens_win.len() / cfg.enc.hidden_size;
    let first_idx = start_token - win_start / 8;
    let end_idx = first_idx + new_tokens;
    if end_idx > avail {
        return Err(HipError::new(
            999,
            &format!(
                "lfm3-audio: stream token accounting desync (want window tokens {first_idx}..{end_idx}, downsample produced {avail})"
            ),
        ));
    }
    let slice = &tokens_win[first_idx * cfg.enc.hidden_size..end_idx * cfg.enc.hidden_size];

    // 4. GPU encoder forward (persistent state — position carries).
    let enc_out =
        encoder_forward(gpu, &weights.encoder, &cfg.enc, &mut cache.state, slice, new_tokens)
            .map_err(|e| HipError::new(999, &e))?;

    // 5. GPU projector.
    let proj = projector_forward(gpu, &weights.projector, &cfg.proj, &enc_out, new_tokens)
        .map_err(|e| HipError::new(999, &e))?;

    cache.num_tokens_emitted = num_tokens;
    cache.finished = is_final;
    Ok(proj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn host_w(name: &str, dir: &str) -> HostW {
        let idx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&format!("{dir}/downsample_weights_idx.json")).unwrap()).unwrap();
        let entry = idx.get(name).unwrap();
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let data = read_f32(&format!("{dir}/dw_{}.bin", name.replace('.', "_")));
        HostW { data, shape }
    }

    /// Validate the conv2d downsampler against a numpy reference that loads
    /// the actual LFM3-Audio-2B downsampler weights and runs the HF
    /// `Lfm3AudioDownsampler.forward` (tests/fixtures/gen_downsample_reference.py).
    #[test]
    fn downsampler_matches_numpy_reference() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let mel = read_f32(&format!("{dir}/log_mel.bin"));
        let ref_out = read_f32(&format!("{dir}/downsample_out.bin"));

        fn parse_meta(path: &str) -> std::collections::HashMap<String, usize> {
            let txt = std::fs::read_to_string(path).unwrap();
            txt.lines()
                .filter_map(|l| {
                    let (k, v) = l.split_once('=')?;
                    Some((k.trim().to_string(), v.trim().parse().ok()?))
                })
                .collect()
        }
        let m = parse_meta(&format!("{dir}/meta.txt"));
        let n_mels = m["n_mels"];
        let num_frames = m["num_frames"];
        let dm = parse_meta(&format!("{dir}/downsample_meta.txt"));
        let n_tokens = dm["n_tokens"];
        let hidden = dm["hidden"];

        let dw = DownsamplerWeights {
            conv_in_w: host_w("conv_in.weight", dir),
            conv_in_b: read_f32(&format!("{dir}/dw_conv_in_bias.bin")),
            conv_mid_w: host_w("conv_mid.weight", dir),
            conv_mid_b: read_f32(&format!("{dir}/dw_conv_mid_bias.bin")),
            lin_mid_w: host_w("lin_mid.weight", dir),
            lin_mid_b: read_f32(&format!("{dir}/dw_lin_mid_bias.bin")),
            conv_end_w: host_w("conv_end.weight", dir),
            conv_end_b: read_f32(&format!("{dir}/dw_conv_end_bias.bin")),
            lin_end_w: host_w("lin_end.weight", dir),
            lin_end_b: read_f32(&format!("{dir}/dw_lin_end_bias.bin")),
            lin_out_w: host_w("lin_out.weight", dir),
            lin_out_b: read_f32(&format!("{dir}/dw_lin_out_bias.bin")),
        };

        let out = downsampler_forward(&mel, n_mels, num_frames, &dw);
        assert_eq!(out.len(), n_tokens * hidden);
        assert_eq!(out.len(), ref_out.len());
        let mut max_err = 0.0f32;
        let mut rel_err = 0.0f32;
        for (a, b) in out.iter().zip(ref_out.iter()) {
            let d = (a - b).abs();
            max_err = max_err.max(d);
            if b.abs() > 0.05 {
                rel_err = rel_err.max(d / b.abs());
            }
        }
        eprintln!("downsampler max abs err {max_err:e}, max rel err {rel_err:e}");
        // CPU f32 vs numpy f32 (BF16 weights → f32) → near-exact; allow tiny slack.
        assert!(max_err < 1e-3, "downsampler max abs err {max_err:e}");
        assert!(rel_err < 1e-3, "downsampler max rel err {rel_err:e}");
    }

    /// REVIEW-3: the bounded streaming downsampler window must be
    /// BIT-IDENTICAL to the full-stream pass. Token t depends only on mel
    /// frames [8t−8, 8t+8) (three k3s2p1 convs, ±8-frame receptive field);
    /// a window starting at a multiple of 8 sees the same conv-padding
    /// behavior as the full stream for every token it produces, with window
    /// output j mapping to full-stream token `ws/8 + j`. Uses the real
    /// LFM3-Audio-2B downsampler fixture weights.
    #[test]
    fn downsampler_window_bit_identity() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let mel = read_f32(&format!("{dir}/log_mel.bin"));

        fn parse_meta(path: &str) -> std::collections::HashMap<String, usize> {
            let txt = std::fs::read_to_string(path).unwrap();
            txt.lines()
                .filter_map(|l| {
                    let (k, v) = l.split_once('=')?;
                    Some((k.trim().to_string(), v.trim().parse().ok()?))
                })
                .collect()
        }
        let m = parse_meta(&format!("{dir}/meta.txt"));
        let n_mels = m["n_mels"];
        let num_frames = m["num_frames"];

        let dw = DownsamplerWeights {
            conv_in_w: host_w("conv_in.weight", dir),
            conv_in_b: read_f32(&format!("{dir}/dw_conv_in_bias.bin")),
            conv_mid_w: host_w("conv_mid.weight", dir),
            conv_mid_b: read_f32(&format!("{dir}/dw_conv_mid_bias.bin")),
            lin_mid_w: host_w("lin_mid.weight", dir),
            lin_mid_b: read_f32(&format!("{dir}/dw_lin_mid_bias.bin")),
            conv_end_w: host_w("conv_end.weight", dir),
            conv_end_b: read_f32(&format!("{dir}/dw_conv_end_bias.bin")),
            lin_end_w: host_w("lin_end.weight", dir),
            lin_end_b: read_f32(&format!("{dir}/dw_lin_end_bias.bin")),
            lin_out_w: host_w("lin_out.weight", dir),
            lin_out_b: read_f32(&format!("{dir}/dw_lin_out_bias.bin")),
        };

        let full = downsampler_forward(&mel, n_mels, num_frames, &dw);
        // hidden from the fixture meta — NOT `num_frames/8`: three stride-2
        // convs make the token count (num_frames−7)/8-ish, e.g. 149 frames →
        // 19 tokens, not 149/8 = 18.
        let dm = parse_meta(&format!("{dir}/downsample_meta.txt"));
        let hidden = dm["hidden"];
        let n_tokens = full.len() / hidden;
        assert_eq!(n_tokens * hidden, full.len(), "fixture token grid must be exact");
        assert!(n_tokens >= 2, "fixture too short to exercise windows");

        // Window helper: materialize frames [ws, hi) of the mel fixture in
        // downsampler layout ([n_mels, hi-ws], mel-major — the fixture is
        // mel-major, so a frame range is NOT a contiguous slice).
        let window = |mel: &[f32], num_frames: usize, ws: usize, hi: usize| {
            let wf = hi - ws;
            let mut win = vec![0.0f32; n_mels * wf];
            for f in 0..wf {
                for m in 0..n_mels {
                    win[m * wf + f] = mel[m * num_frames + ws + f];
                }
            }
            win
        };

        // Non-final tokens: window [8(t−1), num_frames), output index 1.
        for t in 0..n_tokens {
            let ws = 8 * t.saturating_sub(1);
            let win = window(&mel, num_frames, ws, num_frames);
            let win_tokens = downsampler_forward(&win, n_mels, num_frames - ws, &dw);
            let j = t - ws / 8;
            assert!(j < win_tokens.len() / hidden, "t={t}: window too short");
            let got = &win_tokens[j * hidden..(j + 1) * hidden];
            let want = &full[t * hidden..(t + 1) * hidden];
            assert_eq!(got, want, "t={t}: window token diverged from full stream");
        }

        // Final-flush partial token: truncate 3 frames off the end (leaves a
        // 5-frame partial group) and verify the flush token against a
        // full-length pass over the truncated mel.
        let trunc = num_frames - 3;
        let trunc_mel = window(&mel, num_frames, 0, trunc);
        let full_trunc = downsampler_forward(&trunc_mel, n_mels, trunc, &dw);
        assert_eq!(full_trunc.len() / hidden, (trunc + 7) / 8, "flush token must exist");
        let t = (trunc + 7) / 8 - 1;
        let ws = 8 * t.saturating_sub(1);
        let win = window(&mel, num_frames, ws, trunc);
        let win_tokens = downsampler_forward(&win, n_mels, trunc - ws, &dw);
        let j = t - ws / 8;
        let got = &win_tokens[j * hidden..(j + 1) * hidden];
        let want = &full_trunc[t * hidden..(t + 1) * hidden];
        assert_eq!(got, want, "flush token t={t}: window diverged from full stream");
    }
}
