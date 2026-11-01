// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni Liquid Codec **decode** path (Phase 4b): 12 codebooks → 24 kHz
//! waveform.
//!
//! Ground truth: transformers `modeling_liquid_codec.py` `LiquidCodecModel`
//! / `LiquidCodecDecoder` / `LiquidCodecQuantizer` / `LiquidCodecISTFT`
//! (read 2026-10-31, `LiquidAI/LFM3-Omni-2B/audio_tokenizer/`).
//!
//! Decode path (codes → waveform):
//!   1. **FSQ decode** (CPU): each code (0..2015) → 4-dim latent via the
//!      `[8,7,6,6]` levels; 12 codebooks × 4 = 48-dim; `project_out`
//!      Linear(48→256) → `[num_frames, 256]`.
//!   2. `from_quantizer` Linear(256→512) → embeddings `[num_frames, 512]`.
//!   3. **Upsample ×6** (nearest): each codec frame repeated 6× →
//!      `[num_frames*6, 512]` (12.5 fps → 75 Hz STFT frame rate).
//!   4. **8-layer LFM3 decoder** (hidden 512, conv K=7, sliding-window 30,
//!      sandwich norm ON, sigmoid gate, dense SwiGLU) — causal sequential
//!      prefill reusing the LFM3 GPU primitives (mirrors the Phase-3 audio
//!      encoder's `encoder_forward` + the text backbone's sandwich/gate).
//!      Returns `last_hidden_state` per token `[num_frames*6, 512]`.
//!   5. `lin` Linear(512→1282) → (log_magnitude, angle) `[num_frames*6, 641]`.
//!   6. **iSTFT** (CPU): `polar(exp(log_mag), angle)` → irfft → windowed
//!      overlap-add (n_fft 1280, hop 320, win 1280) → trim → 24 kHz waveform.
//!
//! Weights are F16 (per the Phase-4 decision: depth decoder + codec F16, match
//! the Phase 2/3 towers). Loaded directly from a `&dyn ModelSource` (the
//! `audio_tokenizer/` safetensors dir) for standalone validation; the daemon
//! wiring (4c) bundles or sidecars the codec.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::attention::AttnParams;
use hipfire_dispatch::families::kv_tier::{KvTierInputs, KvTierPlan};
use hipfire_dispatch::pipeline::{execute_steps, Step};
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::model_source::ModelSource;
// (DTYPE-1: the bf16 conversion helpers moved to the shared
// `safetensors_source::source_bytes_to_*` converters, used via full paths
// below.)
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::f32::consts::PI;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Liquid Codec config (HF `LiquidCodecConfig`). Defaults match
/// `LiquidAI/LFM3-Omni-2B/audio_tokenizer/`.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    // FSQ quantizer
    pub fsq_levels: Vec<u32>, // [8,7,6,6]
    pub n_q: usize,           // 12
    pub quantizer_dim: usize, // 256
    pub encoder_dim: usize,   // 512
    // STFT
    pub stft_n_fft: usize,    // 1280
    pub stft_hop: usize,      // 320
    pub stft_win: usize,      // 1280
    pub stft_frame_stack: usize, // 6
    // LFM3 decoder
    pub hidden_size: usize,       // 512
    pub num_hidden_layers: usize, // 8
    pub num_attention_heads: usize, // 8
    pub num_key_value_heads: usize, // 2
    pub head_dim: usize,          // 64
    pub conv_kernel_size: usize,  // 7
    pub intermediate_size: usize, // 2304
    pub sliding_window: usize,    // 30
    pub rope_theta: f32,          // 1e6
    pub norm_eps: f32,            // 1e-5
    /// Per-layer kind: layers 2,4,6 = sliding_attention; 0,1,3,5,7 = conv.
    pub layer_is_attn: Vec<bool>,
}

impl CodecConfig {
    pub fn codebook_size(&self) -> usize {
        self.fsq_levels.iter().map(|&l| l as usize).product()
    }
    pub fn frame_samples(&self) -> usize {
        self.stft_hop * self.stft_frame_stack
    }
    pub fn num_attn_layers(&self) -> usize {
        self.layer_is_attn.iter().filter(|&&b| b).count()
    }
    pub fn num_conv_layers(&self) -> usize {
        self.layer_is_attn.iter().filter(|&&b| !b).count()
    }
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Parse from the codec `config.json` (a `serde_json::Value`).
    pub fn from_config_value(c: &serde_json::Value) -> Self {
        let fsq_levels = c
            .get("fsq_levels")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect())
            .unwrap_or_else(|| vec![8, 7, 6, 6]);
        let n_layers = c.get("model").and_then(|m| m.get("decoder_lfm_config")).and_then(|v| v.get("num_hidden_layers")).and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        // The released codec config nests the LFM3 dims under `model` (dropped by
        // LiquidCodecConfig.__post_init__). Fall back to the omni defaults.
        let dec = c.get("model").unwrap_or(c);
        let hidden = dec.get("encoder_dim").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
        let n_heads = dec.get("decoder_lfm_config").and_then(|v| v.get("num_attention_heads")).and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let n_kv = dec.get("decoder_lfm_config").and_then(|v| v.get("num_key_value_heads")).and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let head_dim = dec.get("decoder_lfm_config").and_then(|v| v.get("head_dim")).and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        let inter = dec.get("decoder_lfm_config").and_then(|v| v.get("intermediate_size")).and_then(|v| v.as_u64()).unwrap_or(2304) as usize;
        let conv_k = dec.get("decoder_lfm_config").and_then(|v| v.get("conv_L_cache")).and_then(|v| v.as_u64()).unwrap_or(7) as usize;
        let sliding = c.get("decoder_window").and_then(|v| v.as_u64()).unwrap_or(30) as usize;
        // layer_types: attention at indices 2,4,6; conv elsewhere (lfm_config).
        let layer_is_attn = (0..n_layers).map(|i| i == 2 || i == 4 || i == 6).collect();
        Self {
            fsq_levels,
            n_q: c.get("n_q").and_then(|v| v.as_u64()).unwrap_or(12) as usize,
            quantizer_dim: c.get("quantizer_dim").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
            encoder_dim: hidden,
            stft_n_fft: c.get("stft_n_fft").and_then(|v| v.as_u64()).unwrap_or(1280) as usize,
            stft_hop: c.get("stft_hop_length").and_then(|v| v.as_u64()).unwrap_or(320) as usize,
            stft_win: c.get("stft_win_length").and_then(|v| v.as_u64()).unwrap_or(1280) as usize,
            stft_frame_stack: c.get("stft_frame_stack").and_then(|v| v.as_u64()).unwrap_or(6) as usize,
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            head_dim,
            conv_kernel_size: conv_k,
            intermediate_size: inter,
            sliding_window: sliding,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            layer_is_attn,
        }
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

pub struct CodecConvWeights {
    pub in_proj: GpuTensor,    // [3*hidden, hidden] F16
    pub conv_weight: GpuTensor, // [hidden, K] F32 (depthwise causal conv filter)
    pub out_proj: GpuTensor,   // [hidden, hidden] F16
    pub conv_state_idx: usize,
}

pub struct CodecAttnWeights {
    pub wq: GpuTensor, // [q_dim, hidden] F16
    pub wk: GpuTensor, // [kv_dim, hidden] F16
    pub wv: GpuTensor, // [kv_dim, hidden] F16
    pub wo: GpuTensor, // [q_dim, q_dim] F16
    pub q_norm: GpuTensor, // [head_dim] F32
    pub k_norm: GpuTensor, // [head_dim] F32
    pub gate_proj: GpuTensor, // [n_heads, hidden] F16
    pub kv_idx: usize,
}

pub enum CodecMixer {
    Conv(CodecConvWeights),
    Attn(CodecAttnWeights),
}

pub struct CodecLayer {
    pub operator_norm: GpuTensor, // [hidden] F32
    pub ffn_norm: GpuTensor,      // [hidden] F32
    pub post_operator_norm: GpuTensor, // [hidden] F32 (sandwich)
    pub post_ffn_norm: GpuTensor,      // [hidden] F32 (sandwich)
    pub mixer: CodecMixer,
    pub w1: GpuTensor, // [inter, hidden] F16
    pub w2: GpuTensor, // [hidden, inter] F16
    pub w3: GpuTensor, // [inter, hidden] F16
}

pub struct CodecWeights {
    // FSQ + projections
    pub project_out_w: GpuTensor, // [quantizer_dim, n_q*codebook_dim] F16
    pub project_out_b: GpuTensor, // [quantizer_dim] F32
    pub from_quantizer_w: GpuTensor, // [encoder_dim, quantizer_dim] F16
    pub from_quantizer_b: GpuTensor, // [encoder_dim] F32
    // decoder LFM3
    pub layers: Vec<CodecLayer>,
    pub embedding_norm: GpuTensor, // [hidden] F32
    // lin + istft
    pub lin_w: GpuTensor, // [2*(n_fft//2+1), hidden] F16
    pub lin_b: GpuTensor, // [2*(n_fft//2+1)] F32
    pub istft_window: Vec<f32>, // [win] F32 (host — the synthesis Hann window)
}

// ─── Source load helpers (BF16/F16/F32 → GPU F16 / host F32) ─────────────────

fn src_bytes<'a>(
    source: &'a dyn ModelSource,
    name: &str,
) -> Result<(&'a str, &'a [u8], Vec<usize>), String> {
    let (info, data) = source
        .tensor_data(name)
        .ok_or_else(|| format!("codec: tensor not found: {name}"))?;
    Ok((&info.dtype, data, info.shape.clone()))
}

fn load_f16_source(source: &dyn ModelSource, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let (dtype, data, shape) = src_bytes(source, name)?;
    let n: usize = shape.iter().product();
    // DTYPE-1: delegate to the shared converter — it passes genuine F16
    // through and converts BF16. The old `"F16" | "BF16" => bf16_bytes_to_f16`
    // arm decoded every F16 tensor as if it were BF16 (garbled weights on any
    // F16-stored codec artifact).
    let f16_bytes = hipfire_runtime::safetensors_source::source_bytes_to_f16_stream(dtype, data);
    gpu.upload_raw(&f16_bytes, &[n]).map_err(|e| format!("codec: upload {name}: {e:?}"))
}

fn load_f32_source_host(source: &dyn ModelSource, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let (dtype, data, _) = src_bytes(source, name)?;
    // DTYPE-1: shared converter (F16 decoded as F16, BF16 as BF16 — the old
    // arm treated both as BF16).
    let mut vals = hipfire_runtime::safetensors_source::source_bytes_to_f32_vec(dtype, data);
    vals.truncate(n);
    Ok(vals)
}

fn load_f32_source_gpu(
    source: &dyn ModelSource,
    gpu: &mut Gpu,
    name: &str,
    n: usize,
) -> Result<GpuTensor, String> {
    let vals = load_f32_source_host(source, name, n)?;
    gpu.upload_f32(&vals, &[n])
        .map_err(|e| format!("codec: upload {name}: {e:?}"))
}

/// Load the codec weights from a `&dyn ModelSource` (the `audio_tokenizer/`
/// safetensors dir).
pub fn load_codec_weights(
    source: &dyn ModelSource,
    cfg: &CodecConfig,
    gpu: &mut Gpu,
) -> Result<CodecWeights, String> {
    let h = cfg.hidden_size;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate_size;
    let k_conv = cfg.conv_kernel_size;
    let cb_dim = cfg.fsq_levels.len(); // 4

    eprintln!("  loading Liquid Codec weights (F16)...");
    let project_out_w = load_f16_source(source, gpu, "quantizer.project_out.weight")?;
    let project_out_b = load_f32_source_gpu(source, gpu, "quantizer.project_out.bias", cfg.quantizer_dim)?;
    let from_quantizer_w = load_f16_source(source, gpu, "from_quantizer.weight")?;
    let from_quantizer_b = load_f32_source_gpu(source, gpu, "from_quantizer.bias", cfg.encoder_dim)?;

    let embedding_norm = load_f32_source_gpu(source, gpu, "decoder.lfm.embedding_norm.weight", h)?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    let mut conv_count = 0usize;
    let mut kv_count = 0usize;
    for l in 0..cfg.num_hidden_layers {
        let p = format!("decoder.lfm.layers.{l}");
        let operator_norm = load_f32_source_gpu(source, gpu, &format!("{p}.operator_norm.weight"), h)?;
        let ffn_norm = load_f32_source_gpu(source, gpu, &format!("{p}.ffn_norm.weight"), h)?;
        let post_operator_norm =
            load_f32_source_gpu(source, gpu, &format!("{p}.post_operator_norm.weight"), h)?;
        let post_ffn_norm =
            load_f32_source_gpu(source, gpu, &format!("{p}.post_ffn_norm.weight"), h)?;
        let w1 = load_f16_source(source, gpu, &format!("{p}.feed_forward.w1.weight"))?;
        let w2 = load_f16_source(source, gpu, &format!("{p}.feed_forward.w2.weight"))?;
        let w3 = load_f16_source(source, gpu, &format!("{p}.feed_forward.w3.weight"))?;

        let mixer = if cfg.layer_is_attn[l] {
            let kv_idx = kv_count;
            kv_count += 1;
            CodecMixer::Attn(CodecAttnWeights {
                wq: load_f16_source(source, gpu, &format!("{p}.self_attn.q_proj.weight"))?,
                wk: load_f16_source(source, gpu, &format!("{p}.self_attn.k_proj.weight"))?,
                wv: load_f16_source(source, gpu, &format!("{p}.self_attn.v_proj.weight"))?,
                wo: load_f16_source(source, gpu, &format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: load_f32_source_gpu(source, gpu, &format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: load_f32_source_gpu(source, gpu, &format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                gate_proj: load_f16_source(source, gpu, &format!("{p}.self_attn.gate_proj.weight"))?,
                kv_idx,
            })
        } else {
            let conv_state_idx = conv_count;
            conv_count += 1;
            let conv_weight =
                load_f32_source_gpu(source, gpu, &format!("{p}.conv.conv.weight"), h * k_conv)?;
            CodecMixer::Conv(CodecConvWeights {
                in_proj: load_f16_source(source, gpu, &format!("{p}.conv.in_proj.weight"))?,
                conv_weight,
                out_proj: load_f16_source(source, gpu, &format!("{p}.conv.out_proj.weight"))?,
                conv_state_idx,
            })
        };
        layers.push(CodecLayer {
            operator_norm,
            ffn_norm,
            post_operator_norm,
            post_ffn_norm,
            mixer,
            w1,
            w2,
            w3,
        });
    }
    let _ = (q_dim, kv_dim, cb_dim);

    let lin_w = load_f16_source(source, gpu, "decoder.lin.weight")?;
    let lin_b = load_f32_source_gpu(
        source,
        gpu,
        "decoder.lin.bias",
        2 * (cfg.stft_n_fft / 2 + 1),
    )?;
    let istft_window = load_f32_source_host(source, "decoder.istft.window", cfg.stft_win)?;

    eprintln!(
        "  codec loaded: {} layers ({} attn / {} conv), fsq {:?}, codebook_size {}",
        cfg.num_hidden_layers,
        cfg.num_attn_layers(),
        cfg.num_conv_layers(),
        cfg.fsq_levels,
        cfg.codebook_size()
    );

    Ok(CodecWeights {
        project_out_w,
        project_out_b,
        from_quantizer_w,
        from_quantizer_b,
        layers,
        embedding_norm,
        lin_w,
        lin_b,
        istft_window,
    })
}

impl CodecWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for t in [self.project_out_w, self.from_quantizer_w, self.lin_w] {
            let _ = gpu.free_tensor(t);
        }
        for t in [self.project_out_b, self.from_quantizer_b, self.embedding_norm, self.lin_b] {
            let _ = gpu.free_tensor(t);
        }
        for l in self.layers {
            let _ = gpu.free_tensor(l.operator_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.post_operator_norm);
            let _ = gpu.free_tensor(l.post_ffn_norm);
            let _ = gpu.free_tensor(l.w1);
            let _ = gpu.free_tensor(l.w2);
            let _ = gpu.free_tensor(l.w3);
            match l.mixer {
                CodecMixer::Conv(c) => {
                    let _ = gpu.free_tensor(c.in_proj);
                    let _ = gpu.free_tensor(c.conv_weight);
                    let _ = gpu.free_tensor(c.out_proj);
                }
                CodecMixer::Attn(a) => {
                    for t in [a.wq, a.wk, a.wv, a.wo, a.gate_proj] {
                        let _ = gpu.free_tensor(t);
                    }
                    let _ = gpu.free_tensor(a.q_norm);
                    let _ = gpu.free_tensor(a.k_norm);
                }
            }
        }
    }
}

// ─── Decode state ────────────────────────────────────────────────────────────

pub struct CodecState {
    pub kv: KvCache,
    pub conv_states: Vec<GpuTensor>,
    pub pos_buf: hip_bridge::DeviceBuffer,
    pub max_seq: usize,
    /// Absolute position counter (carries across chunks for streaming decode).
    pub n_tokens: usize,
    pub h: GpuTensor,
    pub tmp: GpuTensor,
    pub fa_q: GpuTensor,
    pub fa_k: GpuTensor,
    pub fa_v: GpuTensor,
    pub fa_attn_out: GpuTensor,
    pub fa_gate: GpuTensor,
    pub flash_partials: GpuTensor,
    pub conv_bcx: GpuTensor,
    pub conv_y: GpuTensor,
    pub ffn_tmp: GpuTensor,
    pub dense_gate: GpuTensor,
    pub dense_up: GpuTensor,
    pub dense_act: GpuTensor,
    pub proj_out: GpuTensor,
    pub sandwich_buf: GpuTensor,
    pub final_norm_buf: GpuTensor,
}

impl CodecState {
    pub fn new(gpu: &mut Gpu, cfg: &CodecConfig, max_seq: usize) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let k_conv = cfg.conv_kernel_size;

        gpu.ensure_mq_signs().map_err(|e| format!("codec: ensure_mq_signs: {e:?}"))?;

        let n_attn = cfg.num_attn_layers().max(1);
        let dims = hipfire_runtime::llama::KvDims {
            layers: hipfire_runtime::llama::KvLayers::Flat(n_attn),
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            max_seq,
            physical_cap: None,
        };
        let kv = KvCache::from_mode(
            hipfire_runtime::kv_mode::resolve("", &hipfire_runtime::kv_mode::HFQ_Q8_ONLY_POLICY, head_dim).mode,
            hipfire_runtime::llama::KvTarget::Single(gpu),
            &dims,
        )
        .map_err(|e| format!("codec: kv cache: {e:?}"))?;

        let conv_hist = h * (k_conv - 1);
        let zeros = vec![0u8; conv_hist * 4];
        let mut conv_states = Vec::with_capacity(cfg.num_conv_layers());
        for _ in 0..cfg.num_conv_layers() {
            let cs = gpu.alloc_tensor(&[conv_hist], DType::F32).map_err(|e| format!("codec: alloc conv_state: {e:?}"))?;
            gpu.hip.memcpy_htod(&cs.buf, &zeros).map_err(|e| format!("codec: zero conv_state: {e:?}"))?;
            conv_states.push(cs);
        }

        let pos_buf = gpu.hip.malloc(4).map_err(|e| format!("codec: pos_buf: {e:?}"))?;
        let alloc = |g: &mut Gpu, n: usize, label: &str| -> Result<GpuTensor, String> {
            g.alloc_tensor(&[n], DType::F32).map_err(|e| format!("codec: alloc {label}: {e:?}"))
        };
        const SUB: usize = 2;
        Ok(CodecState {
            kv,
            conv_states,
            pos_buf,
            max_seq,
            n_tokens: 0,
            h: alloc(gpu, h, "h")?,
            tmp: alloc(gpu, h, "tmp")?,
            fa_q: alloc(gpu, q_dim, "fa_q")?,
            fa_k: alloc(gpu, kv_dim, "fa_k")?,
            fa_v: alloc(gpu, kv_dim, "fa_v")?,
            fa_attn_out: alloc(gpu, q_dim, "fa_attn_out")?,
            fa_gate: alloc(gpu, n_heads, "fa_gate")?,
            flash_partials: alloc(
                gpu,
                n_heads * ((max_seq + 127) / 128) * (2 + head_dim) * SUB,
                "flash_partials",
            )?,
            conv_bcx: alloc(gpu, 3 * h, "conv_bcx")?,
            conv_y: alloc(gpu, h, "conv_y")?,
            ffn_tmp: alloc(gpu, h, "ffn_tmp")?,
            dense_gate: alloc(gpu, inter, "dense_gate")?,
            dense_up: alloc(gpu, inter, "dense_up")?,
            dense_act: alloc(gpu, inter, "dense_act")?,
            proj_out: alloc(gpu, h, "proj_out")?,
            sandwich_buf: alloc(gpu, h, "sandwich_buf")?,
            final_norm_buf: alloc(gpu, h, "final_norm_buf")?,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        for cs in &self.conv_states {
            let zeros = vec![0u8; cs.numel() * 4];
            gpu.hip.memcpy_htod(&cs.buf, &zeros).map_err(|e| format!("codec: reset conv_state: {e:?}"))?;
        }
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.kv.free_gpu(gpu);
        for t in self.conv_states {
            let _ = gpu.free_tensor(t);
        }
        for t in [
            self.h, self.tmp, self.fa_q, self.fa_k, self.fa_v, self.fa_attn_out, self.fa_gate,
            self.flash_partials, self.conv_bcx, self.conv_y, self.ffn_tmp, self.dense_gate,
            self.dense_up, self.dense_act, self.proj_out, self.sandwich_buf, self.final_norm_buf,
        ] {
            let _ = gpu.free_tensor(t);
        }
        // LEAK-1: the 4-byte position scalar buffer was never freed.
        let _ = gpu.hip.free(self.pos_buf);
    }
}

// ─── Linear helpers (F16 weight × F32 activation) ────────────────────────────

/// gemv: y[out_dim] = W_f16[out_dim, in_dim] @ x_f32[in_dim] (n=1, no transpose
/// needed — gemm_f16 writes [out_dim, 1] == [out_dim] for N=1). Writes into the
/// pre-allocated `out` buffer (no per-call alloc).
fn gemv_f16(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, out: &GpuTensor, out_dim: usize, in_dim: usize) -> Result<(), String> {
    gpu.gemm_f16(w, x, out, out_dim, in_dim, 1).map_err(|e| format!("gemv_f16: {e:?}"))
}

/// Batched linear with bias: y[n, out_dim] = W[out_dim,in_dim] @ x[n,in_dim]^T + b.
fn linear_f16_bias_batched(
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

// ─── Decoder forward (sequential prefill) ────────────────────────────────────

fn codec_conv_mixer(
    gpu: &mut Gpu, cfg: &CodecConfig, c: &CodecConvWeights, st: &CodecState, l: usize,
) -> Result<(), String> {
    let h = cfg.hidden_size;
    gemv_f16(gpu, &c.in_proj, &st.tmp, &st.conv_bcx, 3 * h, h).map_err(|e| format!("codec L{l}: conv in_proj: {e}"))?;
    gpu.conv1d_gated_decode_f32(&st.conv_bcx, &st.conv_states[c.conv_state_idx], &c.conv_weight, &st.conv_y, 1, h, cfg.conv_kernel_size)
        .map_err(|e| format!("codec L{l}: conv gated decode: {e:?}"))?;
    gemv_f16(gpu, &c.out_proj, &st.conv_y, &st.proj_out, h, h).map_err(|e| format!("codec L{l}: conv out_proj: {e}"))
}

fn codec_attn_mixer(
    gpu: &mut Gpu, cfg: &CodecConfig, a: &CodecAttnWeights, st: &CodecState, l: usize, seq_len: usize,
) -> Result<(), String> {
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let eps = cfg.norm_eps;
    let q_dim = cfg.q_dim();

    gemv_f16(gpu, &a.wq, &st.tmp, &st.fa_q, q_dim, cfg.hidden_size).map_err(|e| format!("codec L{l}: q_proj: {e}"))?;
    gemv_f16(gpu, &a.wk, &st.tmp, &st.fa_k, cfg.kv_dim(), cfg.hidden_size).map_err(|e| format!("codec L{l}: k_proj: {e}"))?;
    gemv_f16(gpu, &a.wv, &st.tmp, &st.fa_v, cfg.kv_dim(), cfg.hidden_size).map_err(|e| format!("codec L{l}: v_proj: {e}"))?;

    gpu.rmsnorm_batched(&st.fa_q, &a.q_norm, &st.fa_q, n_heads, head_dim, eps).map_err(|e| format!("codec L{l}: q_norm: {e:?}"))?;
    gpu.rmsnorm_batched(&st.fa_k, &a.k_norm, &st.fa_k, n_kv, head_dim, eps).map_err(|e| format!("codec L{l}: k_norm: {e:?}"))?;

    // Full-rotary RoPE (n_rot = head_dim, θ = 1e6).
    gpu.rope_partial_interleaved_f32(&st.fa_q, &st.fa_k, &st.pos_buf, n_heads, n_kv, head_dim, head_dim, cfg.rope_theta)
        .map_err(|e| format!("codec L{l}: rope: {e:?}"))?;

    let kv_idx = a.kv_idx;
    gpu.kv_cache_write_q8_0(&st.kv.k_gpu[kv_idx], &st.fa_k, &st.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("codec L{l}: kv write k: {e:?}"))?;
    gpu.kv_cache_write_q8_0(&st.kv.v_gpu[kv_idx], &st.fa_v, &st.pos_buf, n_kv, head_dim)
        .map_err(|e| format!("codec L{l}: kv write v: {e:?}"))?;

    // Sliding-window causal attention (window=30).
    let ctx = DispatchCtx::new(gpu);
    let plan = KvTierPlan::derive(KvTierInputs {
        pos: seq_len - 1,
        q8_windowed: true,
        window: cfg.sliding_window as i32,
        ..st.kv.tier_inputs()
    })
    .map_err(|e| format!("codec L{l}: kv tier: {e}"))?;
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
        output: &st.fa_attn_out,
    };
    execute_steps(gpu, &ctx, &[Step::Attend { plan, io }]).map_err(|e| format!("codec L{l}: attention: {e:?}"))?;

    // Sigmoid output gate: gate_proj(tmp) → [n_heads]; attn_out *= sigmoid(gate).
    gemv_f16(gpu, &a.gate_proj, &st.tmp, &st.fa_gate, n_heads, cfg.hidden_size)
        .map_err(|e| format!("codec L{l}: gate_proj: {e}"))?;
    gpu.sigmoid_head_gate_mul_f32(&st.fa_attn_out, &st.fa_gate, head_dim)
        .map_err(|e| format!("codec L{l}: sigmoid gate: {e:?}"))?;

    // o_proj → proj_out.
    gemv_f16(gpu, &a.wo, &st.fa_attn_out, &st.proj_out, q_dim, q_dim).map_err(|e| format!("codec L{l}: o_proj: {e}"))
}

/// Run the 8-layer decoder over `inputs` (`[n_tokens, hidden]` host F32),
/// returning `[n_tokens, hidden]` last_hidden_state (after embedding_norm).
/// Uses the caller-provided `st` (which carries the KV cache + conv state +
/// position across calls for streaming). The position starts at `st.n_tokens`
/// and advances by `n_tokens`.
fn codec_decoder_forward(
    gpu: &mut Gpu,
    weights: &CodecWeights,
    cfg: &CodecConfig,
    st: &mut CodecState,
    inputs: &[f32],
    n_tokens: usize,
) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let eps = cfg.norm_eps;
    let base = st.n_tokens; // absolute position offset (carries across chunks)
    let mut out = vec![0.0f32; n_tokens * h];

    for i in 0..n_tokens as u32 {
        let pos = base as u32 + i; // absolute position (RoPE + KV write)
        let seq_len = pos as usize + 1;
        gpu.hip.memcpy_htod(&st.pos_buf, &(pos as i32).to_ne_bytes()).map_err(|e| format!("codec: htod pos: {e:?}"))?;

        let embed = &inputs[i as usize * h..(i as usize + 1) * h];
        let bytes: Vec<u8> = embed.iter().flat_map(|&f| f.to_le_bytes()).collect();
        gpu.hip.memcpy_htod(&st.h.buf, &bytes).map_err(|e| format!("codec: htod embed: {e:?}"))?;

        for layer in &weights.layers {
            gpu.rmsnorm_f32(&st.h, &layer.operator_norm, &st.tmp, eps).map_err(|e| format!("codec: operator rmsnorm: {e:?}"))?;

            match &layer.mixer {
                CodecMixer::Conv(c) => codec_conv_mixer(gpu, cfg, c, st, 0)?,
                CodecMixer::Attn(a) => codec_attn_mixer(gpu, cfg, a, st, 0, seq_len)?,
            }

            // Sandwich norm (post_operator) + residual.
            gpu.rmsnorm_f32(&st.proj_out, &layer.post_operator_norm, &st.sandwich_buf, eps).map_err(|e| format!("codec: post_op norm: {e:?}"))?;
            gpu.add_f32(&st.sandwich_buf, &st.h, &st.h).map_err(|e| format!("codec: op residual: {e:?}"))?;

            // FFN: ffn_norm → SwiGLU.
            gpu.rmsnorm_f32(&st.h, &layer.ffn_norm, &st.ffn_tmp, eps).map_err(|e| format!("codec: ffn rmsnorm: {e:?}"))?;
            gemv_f16(gpu, &layer.w1, &st.ffn_tmp, &st.dense_gate, cfg.intermediate_size, h).map_err(|e| format!("codec: w1: {e}"))?;
            gemv_f16(gpu, &layer.w3, &st.ffn_tmp, &st.dense_up, cfg.intermediate_size, h).map_err(|e| format!("codec: w3: {e}"))?;
            gpu.silu_mul_f32(&st.dense_gate, &st.dense_up, &st.dense_act).map_err(|e| format!("codec: silu_mul: {e:?}"))?;
            gemv_f16(gpu, &layer.w2, &st.dense_act, &st.proj_out, h, cfg.intermediate_size).map_err(|e| format!("codec: w2: {e}"))?;

            // Sandwich norm (post_ffn) + residual.
            gpu.rmsnorm_f32(&st.proj_out, &layer.post_ffn_norm, &st.sandwich_buf, eps).map_err(|e| format!("codec: post_ffn norm: {e:?}"))?;
            gpu.add_f32(&st.sandwich_buf, &st.h, &st.h).map_err(|e| format!("codec: ffn residual: {e:?}"))?;
        }

        // Final embedding_norm → last_hidden_state for this token.
        gpu.rmsnorm_f32(&st.h, &weights.embedding_norm, &st.final_norm_buf, eps).map_err(|e| format!("codec: final norm: {e:?}"))?;
        let hidden = gpu.download_f32(&st.final_norm_buf).map_err(|e| format!("codec: download hidden: {e:?}"))?;
        out[i as usize * h..(i as usize + 1) * h].copy_from_slice(&hidden);
    }
    st.n_tokens = base + n_tokens; // advance the absolute position
    Ok(out)
}

// ─── FSQ decode (CPU) ────────────────────────────────────────────────────────

/// FSQ decode: codes `[num_frames, n_q]` → FSQ latents `[num_frames, n_q*cb_dim]`.
/// Each code is expanded into `cb_dim` (4) components via the `[8,7,6,6]` levels.
/// The `project_out` Linear is applied on GPU by the caller (see `codec_decode`).
pub fn fsq_decode(cfg: &CodecConfig, codes: &[u32]) -> Result<Vec<f32>, String> {
    let n_q = cfg.n_q;
    let cb_dim = cfg.fsq_levels.len(); // 4
    let levels = &cfg.fsq_levels;
    let half_width: Vec<u32> = levels.iter().map(|&l| l / 2).collect();
    let basis: Vec<u32> = {
        let mut b = vec![0u32; cb_dim];
        let mut acc = 1u32;
        for i in 0..cb_dim {
            b[i] = acc;
            acc *= levels[i];
        }
        b
    };
    let num_frames = codes.len() / n_q;
    let in_dim = n_q * cb_dim; // 48
    let mut latents = vec![0.0f32; num_frames * in_dim];

    for f in 0..num_frames {
        for q in 0..n_q {
            let c = codes[f * n_q + q] as u32;
            for i in 0..cb_dim {
                let comp = ((c / basis[i]) % levels[i]) as i32 - half_width[i] as i32;
                latents[f * in_dim + q * cb_dim + i] = comp as f32 / half_width[i] as f32;
            }
        }
    }
    Ok(latents)
}

// ─── iSTFT (CPU) ─────────────────────────────────────────────────────────────

/// Inverse real FFT (naive O(N²)): half-spectrum `[N/2+1]` (re, im) → `[N]` real.
/// norm="backward" (1/N scaling), matching `torch.fft.irfft(..., norm="backward")`.
fn irfft(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    let half = n / 2;
    let inv_n = 1.0 / n as f32;
    let mut out = vec![0.0f32; n];
    // Precompute twiddles.
    let mut cos_t = vec![0.0f32; (half + 1) * n];
    let mut sin_t = vec![0.0f32; (half + 1) * n];
    for k in 0..=half {
        for j in 0..n {
            let ang = 2.0 * PI * k as f32 * j as f32 / n as f32;
            cos_t[k * n + j] = ang.cos();
            sin_t[k * n + j] = ang.sin();
        }
    }
    for k in 0..=half {
        let scale = if k == 0 || k == half { inv_n } else { 2.0 * inv_n };
        let c = cos_t[k * n..k * n + n].as_ptr();
        let s = sin_t[k * n..k * n + n].as_ptr();
        let rk = re[k];
        let ik = im[k];
        for j in 0..n {
            unsafe {
                out[j] += scale * (rk * *c.add(j) - ik * *s.add(j));
            }
        }
    }
    out
}

/// iSTFT overlap-add: `(log_mag, angle)` `[num_frames, n_bins]` → waveform.
/// Returns the trimmed 24 kHz mono samples (one `frame_samples` per codec frame
/// after the ×6 upsample).
/// Raw iSTFT overlap-add: returns `(audio, envelope)` of length
/// `(num_frames-1)*hop + win` (NOT trimmed). The caller divides + trims.
fn istft_overlap_add_raw(
    cfg: &CodecConfig,
    window: &[f32],
    log_mag: &[f32], // [num_frames, n_bins]
    angle: &[f32],   // [num_frames, n_bins]
    num_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_fft = cfg.stft_n_fft;
    let hop = cfg.stft_hop;
    let win = cfg.stft_win;
    let n_bins = n_fft / 2 + 1;
    let output_size = (num_frames.saturating_sub(1)) * hop + win;

    let mut audio = vec![0.0f32; output_size];
    let mut envelope = vec![0.0f32; output_size];
    let win_sq: Vec<f32> = window.iter().map(|w| w * w).collect();

    for t in 0..num_frames {
        // polar(exp(log_mag), angle) → cartesian re/im for the irfft:
        //   Re(X[k]) = exp(log_mag[k])·cos(angle[k])
        //   Im(X[k]) = exp(log_mag[k])·sin(angle[k])
        // (mirrors HF `torch.polar(log_magnitude.exp(), angle)` before
        // `torch.fft.irfft`; feeding magnitude/angle directly as re/im
        // injects the raw phase values as spectral samples — static.)
        let re: Vec<f32> = (0..n_bins)
            .map(|k| {
                let mag = log_mag[t * n_bins + k].exp();
                mag * angle[t * n_bins + k].cos()
            })
            .collect();
        let im: Vec<f32> = (0..n_bins)
            .map(|k| {
                let mag = log_mag[t * n_bins + k].exp();
                mag * angle[t * n_bins + k].sin()
            })
            .collect();
        let frame = irfft(&re, &im, n_fft);
        let off = t * hop;
        for j in 0..win {
            let idx = off + j;
            if idx < output_size {
                audio[idx] += frame[j] * window[j];
                envelope[idx] += win_sq[j];
            }
        }
    }
    (audio, envelope)
}

/// iSTFT overlap-add + trim + divide (non-streaming path).
fn istft_overlap_add(
    cfg: &CodecConfig,
    window: &[f32],
    log_mag: &[f32],
    angle: &[f32],
    num_frames: usize,
) -> Vec<f32> {
    let win = cfg.stft_win;
    let hop = cfg.stft_hop;
    let trim = (win - hop) / 2;
    let (audio, envelope) = istft_overlap_add_raw(cfg, window, log_mag, angle, num_frames);
    let output_size = audio.len();
    let lo = trim;
    let hi = output_size.saturating_sub(trim);
    audio[lo..hi]
        .iter()
        .zip(&envelope[lo..hi])
        .map(|(&a, &e)| if e > 1e-8 { a / e } else { 0.0 })
        .collect()
}

// ─── Full decode ─────────────────────────────────────────────────────────────

/// Decode audio codes → 24 kHz mono waveform (F32 samples in [-1, 1]).
///
/// `codes`: `[num_frames * n_q]` row-major (one row of `n_q` codebooks per
/// codec frame). Returns the waveform + the sample rate.
pub fn codec_decode(
    cfg: &CodecConfig,
    weights: &CodecWeights,
    gpu: &mut Gpu,
    codes: &[u32],
) -> Result<(Vec<f32>, u32), String> {
    let n_q = cfg.n_q;
    let num_frames = codes.len() / n_q;
    if num_frames == 0 {
        return Ok((Vec::new(), 24_000));
    }
    let h = cfg.hidden_size;
    let fs = cfg.stft_frame_stack; // 6
    let n_bins = cfg.stft_n_fft / 2 + 1; // 641

    eprintln!("codec_decode: {num_frames} frames ({:.1}s @ 12.5 fps)", num_frames as f32 / 12.5);

    // 1. FSQ decode (CPU): codes → [num_frames, n_q*cb_dim=48] FSQ latents.
    let cb_dim = cfg.fsq_levels.len();
    let fsq_in = n_q * cb_dim; // 48
    let latents = fsq_decode(cfg, codes)?;

    // 2. project_out Linear(48→256) + from_quantizer Linear(256→512) (batched, GPU).
    let lat_gpu = gpu.upload_f32(&latents, &[num_frames * fsq_in]).map_err(|e| format!("codec: upload latents: {e:?}"))?;
    let proj = linear_f16_bias_batched(gpu, &weights.project_out_w, &lat_gpu, &weights.project_out_b, cfg.quantizer_dim, fsq_in, num_frames)?;
    gpu.free_tensor(lat_gpu).map_err(|e| format!("codec: free lat: {e:?}"))?;
    let emb = linear_f16_bias_batched(gpu, &weights.from_quantizer_w, &proj, &weights.from_quantizer_b, h, cfg.quantizer_dim, num_frames)?;
    gpu.free_tensor(proj).map_err(|e| format!("codec: free proj: {e:?}"))?;
    let emb_host = gpu.download_f32(&emb).map_err(|e| format!("codec: dl emb: {e:?}"))?;
    gpu.free_tensor(emb).map_err(|e| format!("codec: free emb: {e:?}"))?;
    // emb_host: [num_frames, 512]

    // 3. Upsample ×6 (nearest): each frame repeated 6× → [num_frames*6, 512].
    let n_up = num_frames * fs;
    let mut upsampled = vec![0.0f32; n_up * h];
    for f in 0..num_frames {
        for r in 0..fs {
            let src = f * h;
            let dst = (f * fs + r) * h;
            upsampled[dst..dst + h].copy_from_slice(&emb_host[src..src + h]);
        }
    }

    // 4. 8-layer decoder forward → [n_up, 512] last_hidden_state.
    let hidden = {
        let mut st = CodecState::new(gpu, cfg, n_up.max(64))?;
        st.reset(gpu)?;
        let h_out = codec_decoder_forward(gpu, weights, cfg, &mut st, &upsampled, n_up)?;
        st.free_gpu(gpu);
        h_out
    };

    // 5. lin Linear(512→2*n_bins) (batched) → (log_mag, angle).
    let hidden_gpu = gpu.upload_f32(&hidden, &[n_up * h]).map_err(|e| format!("codec: upload hidden: {e:?}"))?;
    let lin_out = linear_f16_bias_batched(gpu, &weights.lin_w, &hidden_gpu, &weights.lin_b, 2 * n_bins, h, n_up)?;
    gpu.free_tensor(hidden_gpu).map_err(|e| format!("codec: free hidden_gpu: {e:?}"))?;
    let lin_host = gpu.download_f32(&lin_out).map_err(|e| format!("codec: dl lin: {e:?}"))?;
    gpu.free_tensor(lin_out).map_err(|e| format!("codec: free lin: {e:?}"))?;
    // lin_host: [n_up, 2*n_bins] — first n_bins = log_mag, next n_bins = angle.

    // 6. iSTFT overlap-add → waveform.
    let mut log_mag = vec![0.0f32; n_up * n_bins];
    let mut angle = vec![0.0f32; n_up * n_bins];
    for t in 0..n_up {
        for k in 0..n_bins {
            log_mag[t * n_bins + k] = lin_host[t * (2 * n_bins) + k];
            angle[t * n_bins + k] = lin_host[t * (2 * n_bins) + n_bins + k];
        }
    }
    let waveform = istft_overlap_add(cfg, &weights.istft_window, &log_mag, &angle, n_up);

    eprintln!(
        "codec_decode: waveform {} samples ({:.2}s @ 24kHz), finite={}",
        waveform.len(),
        waveform.len() as f32 / 24_000.0,
        waveform.iter().all(|v| v.is_finite())
    );
    Ok((waveform, 24_000))
}

// ─── Streaming decode (Phase 4d) ─────────────────────────────────────────────

/// Streaming codec decode state: carries the decoder LFM3 KV cache + conv
/// state + the iSTFT overlap-add tail across chunks. Mirrors HF
/// `LiquidCodecStreamingCache`. `codec_decode_streaming` decodes one chunk of
/// frames at a time, enabling low-latency TTS (audio emitted as frames are
/// generated, not after the full sequence).
pub struct CodecStreamingCache {
    pub state: CodecState,
    tail_audio: Vec<f32>,
    tail_envelope: Vec<f32>,
    started: bool,
    pub finished: bool,
}

impl CodecStreamingCache {
    /// `max_frames` bounds the total frames expected (for the KV-cache +
    /// flash-partials allocation). Pass an upper bound (e.g. the request's
    /// max_tokens).
    pub fn new(gpu: &mut Gpu, cfg: &CodecConfig, max_frames: usize) -> Result<Self, String> {
        let max_seq = (max_frames * cfg.stft_frame_stack).max(64);
        let mut state = CodecState::new(gpu, cfg, max_seq)?;
        state.reset(gpu)?;
        Ok(Self {
            state,
            tail_audio: Vec::new(),
            tail_envelope: Vec::new(),
            started: false,
            finished: false,
        })
    }
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.state.free_gpu(gpu);
    }
}

/// Decode one chunk of audio frames with a streaming cache. Returns the
/// chunk's waveform samples (divided by the envelope, tail-overlap-added with
/// the previous chunk). `finalize=true` on the last chunk flushes the held-back
/// tail; pass an empty `codes` with `finalize=true` to flush after the last
/// frame chunk. Concatenating all chunks' output == `codec_decode` on the full
/// codes.
pub fn codec_decode_streaming(
    cfg: &CodecConfig,
    weights: &CodecWeights,
    gpu: &mut Gpu,
    codes_chunk: &[u32],
    cache: &mut CodecStreamingCache,
    finalize: bool,
) -> Result<Vec<f32>, String> {
    let n_q = cfg.n_q;
    let h = cfg.hidden_size;
    let fs = cfg.stft_frame_stack;
    let n_bins = cfg.stft_n_fft / 2 + 1;
    let win = cfg.stft_win;
    let hop = cfg.stft_hop;
    let trim = (win - hop) / 2;
    let tail = 2 * trim;
    let n_chunk = codes_chunk.len() / n_q;

    // Produce (audio, envelope) for this chunk (or zeros for a finalize flush).
    let (mut audio, mut envelope) = if n_chunk > 0 {
        // FSQ + project_out + from_quantizer + upsample.
        let cb_dim = cfg.fsq_levels.len();
        let fsq_in = n_q * cb_dim;
        let latents = fsq_decode(cfg, codes_chunk)?;
        let lat_gpu = gpu.upload_f32(&latents, &[n_chunk * fsq_in]).map_err(|e| format!("codec-str: upload lat: {e:?}"))?;
        let proj = linear_f16_bias_batched(gpu, &weights.project_out_w, &lat_gpu, &weights.project_out_b, cfg.quantizer_dim, fsq_in, n_chunk)?;
        gpu.free_tensor(lat_gpu).map_err(|e| format!("codec-str: free lat: {e:?}"))?;
        let emb = linear_f16_bias_batched(gpu, &weights.from_quantizer_w, &proj, &weights.from_quantizer_b, h, cfg.quantizer_dim, n_chunk)?;
        gpu.free_tensor(proj).map_err(|e| format!("codec-str: free proj: {e:?}"))?;
        let emb_host = gpu.download_f32(&emb).map_err(|e| format!("codec-str: dl emb: {e:?}"))?;
        gpu.free_tensor(emb).map_err(|e| format!("codec-str: free emb: {e:?}"))?;
        let n_up = n_chunk * fs;
        let mut upsampled = vec![0.0f32; n_up * h];
        for f in 0..n_chunk {
            for r in 0..fs {
                let src = f * h;
                let dst = (f * fs + r) * h;
                upsampled[dst..dst + h].copy_from_slice(&emb_host[src..src + h]);
            }
        }
        // Decoder forward (persistent state) + lin.
        let hidden = codec_decoder_forward(gpu, weights, cfg, &mut cache.state, &upsampled, n_up)?;
        let hidden_gpu = gpu.upload_f32(&hidden, &[n_up * h]).map_err(|e| format!("codec-str: upload hidden: {e:?}"))?;
        let lin_out = linear_f16_bias_batched(gpu, &weights.lin_w, &hidden_gpu, &weights.lin_b, 2 * n_bins, h, n_up)?;
        gpu.free_tensor(hidden_gpu).map_err(|e| format!("codec-str: free hidden: {e:?}"))?;
        let lin_host = gpu.download_f32(&lin_out).map_err(|e| format!("codec-str: dl lin: {e:?}"))?;
        gpu.free_tensor(lin_out).map_err(|e| format!("codec-str: free lin: {e:?}"))?;
        let mut log_mag = vec![0.0f32; n_up * n_bins];
        let mut angle = vec![0.0f32; n_up * n_bins];
        for t in 0..n_up {
            for k in 0..n_bins {
                log_mag[t * n_bins + k] = lin_host[t * (2 * n_bins) + k];
                angle[t * n_bins + k] = lin_host[t * (2 * n_bins) + n_bins + k];
            }
        }
        istft_overlap_add_raw(cfg, &weights.istft_window, &log_mag, &angle, n_up)
    } else {
        // Finalize flush with no new frames: zeros of the tail length.
        (vec![0.0f32; tail], vec![0.0f32; tail])
    };

    // Overlap-add the previous chunk's tail into this chunk's leading edge.
    if !cache.tail_audio.is_empty() {
        let n = cache.tail_audio.len().min(audio.len());
        for i in 0..n {
            audio[i] += cache.tail_audio[i];
            envelope[i] += cache.tail_envelope[i];
        }
    }

    // First chunk: trim the leading edge.
    let lead_trim = if !cache.started {
        cache.started = true;
        trim
    } else {
        0
    };

    // Finalize: trim the trailing edge + clear the tail. Else: hold back the tail.
    let trail_trim = if finalize {
        cache.tail_audio.clear();
        cache.tail_envelope.clear();
        cache.finished = true;
        trim
    } else {
        let tlen = tail.min(audio.len());
        cache.tail_audio = audio[audio.len() - tlen..].to_vec();
        cache.tail_envelope = envelope[envelope.len() - tlen..].to_vec();
        tlen
    };

    // Emit [lead_trim .. len - trail_trim], divided by the envelope.
    let lo = lead_trim;
    let hi = audio.len().saturating_sub(trail_trim);
    let mut out = Vec::with_capacity(hi.saturating_sub(lo));
    for i in lo..hi {
        out.push(if envelope[i] > 1e-8 { audio[i] / envelope[i] } else { 0.0 });
    }
    Ok(out)
}

// ─── WAV writer (16-bit PCM mono) ────────────────────────────────────────────

/// Encode a 16-bit PCM mono WAV into an in-memory byte vector (for base64
/// transport over the daemon's JSON-lines protocol).
pub fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n = samples.len();
    let data_len = n * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    use std::io::Write;
    let _ = (&mut out).write_all(b"RIFF");
    let _ = (&mut out).write_all(&(36 + data_len as u32).to_le_bytes());
    let _ = (&mut out).write_all(b"WAVE");
    let _ = (&mut out).write_all(b"fmt ");
    let _ = (&mut out).write_all(&16u32.to_le_bytes());
    let _ = (&mut out).write_all(&1u16.to_le_bytes()); // PCM
    let _ = (&mut out).write_all(&1u16.to_le_bytes()); // mono
    let _ = (&mut out).write_all(&sample_rate.to_le_bytes());
    let _ = (&mut out).write_all(&(sample_rate * 2).to_le_bytes()); // byte rate
    let _ = (&mut out).write_all(&2u16.to_le_bytes()); // block align
    let _ = (&mut out).write_all(&16u16.to_le_bytes()); // bits per sample
    let _ = (&mut out).write_all(b"data");
    let _ = (&mut out).write_all(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Write a 16-bit PCM mono WAV file.
pub fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    std::fs::write(path, wav_bytes(samples, sample_rate))
}
