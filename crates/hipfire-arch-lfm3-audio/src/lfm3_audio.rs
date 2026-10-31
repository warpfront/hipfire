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
//! arch_id stays 13 (audio detected by `encoder.downsample.*` tensor
//! presence). The text decoder is owned by `hipfire-arch-lfm3::Lfm3`.

use crate::mel::MelConfig;
use hipfire_runtime::hfq::HfqFile;
use hip_bridge::{HipError, HipResult};
use rdna_compute::Gpu;

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
/// The audio config.json nests: `preprocessor` (mel), `encoder`
/// (downsampler + `encoder.lfm` backbone), `lfm` (text backbone — projector
/// output = `lfm.hidden_size`). The encoder `intermediate_size` is taken from
/// the actual `encoder.lfm.lfm.layers.0.feed_forward.w1.weight` tensor shape
/// (the config value is pre-auto-adjust and unreliable).
pub fn audio_config_from_hfq(hfq: &HfqFile) -> Option<AudioConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config").or(Some(&meta))?;
    let pre = config.get("preprocessor")?;
    let enc_cfg = config.get("encoder")?;
    let lfm_cfg = enc_cfg.get("lfm")?;
    let text_cfg = config.get("lfm").or_else(|| config.get("text_config"))?;

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
        dim_in: enc_cfg.get("dim_in").and_then(|v| v.as_u64()).unwrap_or(128) as usize,
        dim_conv: enc_cfg.get("dim_conv").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
        dim_out: enc_cfg.get("dim_out").and_then(|v| v.as_u64()).unwrap_or(768) as usize,
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
    let intermediate = hfq
        .find_tensor_info("encoder.lfm.lfm.layers.0.feed_forward.w1.weight")
        .and_then(|i| i.shape.first().copied())
        .map(|s| s as usize)
        .unwrap_or_else(|| {
            lfm_cfg.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(1792) as usize
        });

    let proj = ProjectorConfig {
        in_features: down.dim_out,
        out_features: text_cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(2048) as usize,
        norm_eps: 1e-6,
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

impl HostW {
    fn numel(&self) -> usize {
        self.data.len()
    }
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

// ─── Encoder + projector weights (GPU) — stubs, filled next ──────────────────

pub struct AudioWeights {
    pub downsampler: DownsamplerWeights,
    // encoder + projector land in the next commit.
}

pub fn load_audio_weights(_hfq: &HfqFile, _cfg: &AudioConfig, _gpu: &mut Gpu) -> HipResult<AudioWeights> {
    Err(HipError::new(999, "lfm3-audio: encoder/projector load not yet implemented"))
}

pub fn audio_forward(
    _gpu: &mut Gpu,
    _weights: &AudioWeights,
    _cfg: &AudioConfig,
    _mel: &[f32],
    _num_frames: usize,
) -> HipResult<Vec<f32>> {
    Err(HipError::new(999, "lfm3-audio: encoder/projector forward not yet implemented"))
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
}
