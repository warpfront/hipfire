// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3-Omni depth decoder + MaskGIT audio-frame sampling (Phase 4a).
//!
//! Ground truth: transformers `modeling_lfm3_omni.py` `Lfm3OmniDepthDecoder`
//! + `Lfm3OmniForConditionalGeneration.sample_audio_frame` (read 2026-10-31,
//! `LiquidAI/LFM3-Omni-2B`).
//!
//! The depth decoder predicts the 12 codebooks of one speech frame from the
//! language model's hidden state at the position preceding the frame. It is a
//! 3-layer **bidirectional** LFM3 (hidden 1024, 32Q/8KV heads, head_dim 32,
//! `partial_rotary_factor` 1.0 → full rotary, θ 1e6, NO sandwich norm, NO conv,
//! sigmoid output gate ON, dense SwiGLU) run over the 12 codebook slots as a
//! seq=12 batched prefill. Each slot's hidden is `depth_linear(LM hidden) +
//! (mask_embedding | code_embedding)`; the trunk attends bidirectionally over
//! the 12 slots; per-codebook tied logits = `Linear(embedding_norm(slot),
//! embedding.weight)`.
//!
//! MaskGIT sampling reveals the 12 codebooks in a random order over 3 passes of
//! `maskgit_schedule [4,4,4]`, each pass re-running the depth decoder to score
//! the still-masked codebooks. Sampling: repetition penalty 1.2 (over the last
//! 64 frames' codes per codebook) → temperature 0.4 → multinomial.
//!
//! GPU primitives (all pre-existing — NO new kernel needed):
//!   * linears        → `gemm_f16` + `transpose_f32` (+ `bias_add_f32` for
//!                      depth_linear, the only biased linear)
//!   * norms          → `rmsnorm_batched` (operator/ffn/QK/final/per-codebook)
//!   * RoPE           → `rope_partial_interleaved_f32_batched` (halfsplit default,
//!                      full rotary n_rot=head_dim, positions 0..11)
//!   * attention      → `vit_attention_gqa_f32` (bidirectional GQA, reused from
//!                      the Phase-2 vision tower — scale 1/√head_dim internal)
//!   * sigmoid gate   → `sigmoid_head_gate_mul_f32` (per-head broadcast; handles
//!                      the 12-position batch via `gate[i/head_dim]` indexing)
//!   * SwiGLU / resid → `silu_mul_f32` / `add_inplace_f32`
//!
//! The 12 per-codebook embeddings + norms + mask_embedding + the shared
//! `codebook_embedding` (used to feed sampled codes back to the LM) are kept
//! host-side F32/F16: the input gather (`embed_codes`) and the output logits
//! (`embedding_norm` + tied matmul) are small (12 × 2017×1024 ≈ 25M MACs) and
//! run on CPU, avoiding GPU sub-tensor slicing for the per-codebook norm. The
//! trunk (the only GPU-heavy part) runs entirely on GPU.

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::f16_to_f32;
use hipfire_runtime::safetensors_source::bf16_to_f32;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Depth decoder config (HF `Lfm3OmniConfig.depth_config` + the omni-level
/// speech fields). Defaults match `LiquidAI/LFM3-Omni-2B`.
#[derive(Clone, Debug)]
pub struct DepthConfig {
    pub text_hidden_size: usize, // 2048 (LM hidden — depth_linear input)
    pub hidden_size: usize,      // 1024 (depth decoder hidden)
    pub num_codebooks: usize,    // 12
    pub codebook_size: usize,    // 2017 (codec codes + eos)
    pub num_hidden_layers: usize,    // 3
    pub num_attention_heads: usize,  // 32
    pub num_key_value_heads: usize,  // 8
    pub head_dim: usize,         // 32
    pub intermediate_size: usize, // 2816
    pub rope_theta: f32,         // 1e6
    pub norm_eps: f32,           // 1e-5
    pub maskgit_schedule: Vec<usize>, // [4,4,4]
    pub audio_eos_code: u32,     // 2016
    // Sampling (HF generation_config).
    pub audio_temperature: f32,          // 0.4
    pub audio_repetition_penalty: f32,   // 1.2
    pub audio_repetition_window: usize,  // 64
}

impl DepthConfig {
    /// Default config for `LiquidAI/LFM3-Omni-2B` (used when the HFQ metadata
    /// omits a field).
    pub fn omni_default(text_hidden_size: usize) -> Self {
        Self {
            text_hidden_size,
            hidden_size: 1024,
            num_codebooks: 12,
            codebook_size: 2017,
            num_hidden_layers: 3,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 32,
            intermediate_size: 2816,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-5,
            maskgit_schedule: vec![4, 4, 4],
            audio_eos_code: 2016,
            audio_temperature: 0.4,
            audio_repetition_penalty: 1.2,
            audio_repetition_window: 64,
        }
    }
}

/// Parse the depth decoder config from the HFQ metadata envelope.
/// Returns None if no `depth_config` is present (text-only / VL / audio-only
/// LFM3 checkpoints carry no depth decoder).
pub fn depth_config_from_hfq(hfq: &HfqFile) -> Option<DepthConfig> {
    let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    let config = meta.get("config").or(Some(&meta))?;
    let dc = config.get("depth_config")?;
    let tc = config.get("text_config")?;
    let text_hidden_size = tc.get("hidden_size")?.as_u64()? as usize;

    let mut cfg = DepthConfig::omni_default(text_hidden_size);
    if let Some(v) = dc.get("hidden_size").and_then(|v| v.as_u64()) {
        cfg.hidden_size = v as usize;
    }
    if let Some(v) = dc.get("num_hidden_layers").and_then(|v| v.as_u64()) {
        cfg.num_hidden_layers = v as usize;
    }
    if let Some(v) = dc.get("num_attention_heads").and_then(|v| v.as_u64()) {
        cfg.num_attention_heads = v as usize;
    }
    if let Some(v) = dc.get("num_key_value_heads").and_then(|v| v.as_u64()) {
        cfg.num_key_value_heads = v as usize;
    }
    if let Some(v) = dc.get("head_dim").and_then(|v| v.as_u64()) {
        cfg.head_dim = v as usize;
    }
    if let Some(v) = dc.get("intermediate_size").and_then(|v| v.as_u64()) {
        cfg.intermediate_size = v as usize;
    }
    if let Some(v) = dc.get("norm_eps").and_then(|v| v.as_f64()) {
        cfg.norm_eps = v as f32;
    }
    // rope_parameters.full_attention.rope_theta
    if let Some(theta) = dc
        .get("rope_parameters")
        .and_then(|r| r.get("full_attention"))
        .and_then(|r| r.get("rope_theta"))
        .and_then(|v| v.as_f64())
    {
        cfg.rope_theta = theta as f32;
    }
    // Omni-level speech fields.
    if let Some(v) = config.get("num_codebooks").and_then(|v| v.as_u64()) {
        cfg.num_codebooks = v as usize;
    }
    if let Some(v) = config.get("codebook_size").and_then(|v| v.as_u64()) {
        cfg.codebook_size = v as usize;
    }
    if let Some(v) = config.get("audio_eos_code").and_then(|v| v.as_u64()) {
        cfg.audio_eos_code = v as u32;
    }
    if let Some(arr) = config.get("maskgit_schedule").and_then(|v| v.as_array()) {
        let sched: Vec<usize> = arr
            .iter()
            .map(|s| s.as_u64().unwrap_or(4) as usize)
            .collect();
        if !sched.is_empty() {
            cfg.maskgit_schedule = sched;
        }
    }
    Some(cfg)
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// One depth-decoder transformer layer (all `full_attention`, no conv).
pub struct DepthLayer {
    pub operator_norm: GpuTensor, // [hidden] F32
    pub ffn_norm: GpuTensor,      // [hidden] F32
    pub q_proj: GpuTensor,        // [n_heads*head_dim, hidden] F16
    pub k_proj: GpuTensor,        // [n_kv*head_dim, hidden] F16
    pub v_proj: GpuTensor,        // [n_kv*head_dim, hidden] F16
    pub o_proj: GpuTensor,        // [n_heads*head_dim, hidden] F16
    pub q_norm: GpuTensor,        // [head_dim] F32
    pub k_norm: GpuTensor,        // [head_dim] F32
    pub gate_proj: GpuTensor,     // [n_heads, hidden] F16
    pub w1: GpuTensor,            // [intermediate, hidden] F16
    pub w2: GpuTensor,            // [hidden, intermediate] F16
    pub w3: GpuTensor,            // [intermediate, hidden] F16
}

/// Depth decoder weights. The trunk runs on GPU; the 12 per-codebook
/// embeddings + norms + mask + shared codebook embedding are host-side for the
/// gather/logits paths.
pub struct DepthWeights {
    // ── GPU trunk ──
    pub depth_linear_w: GpuTensor, // [num_codebooks*hidden, text_hidden] F16
    pub depth_linear_b: GpuTensor, // [num_codebooks*hidden] F32
    pub embedding_norm: GpuTensor, // [hidden] F32 (trunk final norm)
    pub layers: Vec<DepthLayer>,
    // ── Host codebook tables (input gather + output logits) ──
    /// 12 × [codebook_size * hidden] F32 — per-codebook tied embedding.
    pub host_codebook_embeddings: Vec<Vec<f32>>,
    /// 12 × [hidden] F32 — per-codebook embedding_norm (applied before the
    /// tied output projection).
    pub host_codebook_norms: Vec<Vec<f32>>,
    /// [hidden] F32 — the learned mask token (substituted for masked codebooks).
    pub host_mask_embedding: Vec<f32>,
    // ── Host shared codebook embedding (feed sampled codes back to the LM) ──
    /// [num_codebooks*codebook_size * text_hidden] F16 — `model.codebook_embedding`.
    /// `get_audio_out_embeds` gathers 12 rows (one per codebook) and sums them.
    pub host_shared_codebook_embedding: Vec<u16>,
}

impl DepthWeights {
    /// Free the GPU trunk tensors (the codebook tables are host-side Vecs and
    /// drop naturally). Mirrors `VisionWeights::free_gpu`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.depth_linear_w);
        let _ = gpu.free_tensor(self.depth_linear_b);
        let _ = gpu.free_tensor(self.embedding_norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.operator_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.q_proj);
            let _ = gpu.free_tensor(l.k_proj);
            let _ = gpu.free_tensor(l.v_proj);
            let _ = gpu.free_tensor(l.o_proj);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.gate_proj);
            let _ = gpu.free_tensor(l.w1);
            let _ = gpu.free_tensor(l.w2);
            let _ = gpu.free_tensor(l.w3);
        }
    }
}

// ─── HFQ load helpers (F16/BF16/F32 → GPU F16 / host F32) ────────────────────

/// Dequantize HFQ4 blocks to F32 (G256 or G128). Mirrors lfm3-vl's helper.
fn dequant_hfq4(data: &[u8], n: usize, group_size: usize) -> Vec<f32> {
    let nibble_bytes = group_size / 2;
    let block_size = 8 + nibble_bytes;
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

/// Decode HFQ tensor bytes → F32 vec, handling F16(1)/F32(2)/BF16(16)/HFQ4(6,7).
fn bytes_to_f32(qt: u8, data: &[u8], n: usize, group_size: usize) -> Vec<f32> {
    match qt {
        1 => data.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        2 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        16 => data.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        6 | 7 => dequant_hfq4(data, n, group_size),
        other => panic!("depth: unsupported quant_type={other} (expected F16=1/F32=2/BF16=16/HFQ4=6,7)"),
    }
}

/// Decode HFQ tensor bytes → F16 (u16) vec, handling F16(1)/BF16(16)/F32(2).
fn bytes_to_f16(qt: u8, data: &[u8]) -> Vec<u16> {
    use hipfire_runtime::llama::f32_to_f16;
    match qt {
        1 => data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        // DTYPE-1: BF16 bits are not F16 bits — the old pass-through copied
        // BF16 patterns into an F16 buffer (noise). Route through F32.
        16 => data
            .chunks_exact(2)
            .map(|c| f32_to_f16(bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        other => panic!("depth: unsupported quant_type={other} for F16 upload"),
    }
}

/// Upload an F16 weight tensor to the GPU (from F16/BF16/F32/HFQ4 source).
fn load_f16_gpu(hfq: &HfqFile, gpu: &mut Gpu, name: &str) -> HipResult<GpuTensor> {
    let (info, data) = hfq.tensor_data(name).ok_or_else(|| {
        hip_bridge::HipError::new(0, &format!("depth: tensor not found: {name}"))
    })?;
    let qt = info.quant_type;
    // DTYPE-1: element count comes from the tensor shape. The old
    // data.len()/2 derivation is wrong for HFQ4 (8-byte scale/zero header +
    // group_size/2 nibbles per group) and mis-shaped the upload + dequant.
    let n: usize = info.shape.iter().map(|&s| s as usize).product();
    match qt {
        1 => gpu.upload_raw(data, &[n]),
        // DTYPE-1: BF16 must be converted to F16 before upload, never copied.
        16 => {
            let f16_bytes: Vec<u8> = bytes_to_f16(qt, data)
                .iter()
                .flat_map(|&h| h.to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        2 => {
            let f16_bytes: Vec<u8> = bytes_to_f16(qt, data)
                .iter()
                .flat_map(|&h| h.to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        6 | 7 => {
            let f32_data = dequant_hfq4(data, n, info.group_size as usize);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| hipfire_runtime::llama::f32_to_f16(v).to_le_bytes())
                .collect();
            gpu.upload_raw(&f16_bytes, &[n])
        }
        other => panic!("depth: {name}: unsupported quant_type={other}"),
    }
}

/// Read a 1-D norm/bias tensor → F32 vec (host).
fn load_f32_host(hfq: &HfqFile, name: &str, n: usize) -> Vec<f32> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("depth: tensor not found: {name}"));
    let mut vals = bytes_to_f32(info.quant_type, data, n, info.group_size as usize);
    vals.truncate(n);
    vals
}

/// Load the depth decoder weights from the HFQ. The trunk weights go to GPU
/// (F16 linears / F32 norms); the codebook tables stay host-side.
pub fn load_depth_weights(
    hfq: &HfqFile,
    cfg: &DepthConfig,
    gpu: &mut Gpu,
) -> HipResult<DepthWeights> {
    let h = cfg.hidden_size;
    let pfx = "model.depth_decoder";
    eprintln!("  loading LFM3-Omni depth decoder weights (GPU trunk + host codebooks)...");

    let depth_linear_w = load_f16_gpu(hfq, gpu, &format!("{pfx}.depth_linear.weight"))?;
    let depth_linear_b = {
        let vals = load_f32_host(hfq, &format!("{pfx}.depth_linear.bias"), cfg.num_codebooks * h);
        gpu.upload_f32(&vals, &[cfg.num_codebooks * h])?
    };
    let embedding_norm = {
        let vals = load_f32_host(hfq, &format!("{pfx}.model.embedding_norm.weight"), h);
        gpu.upload_f32(&vals, &[h])?
    };

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let lp = format!("{pfx}.model.layers.{i}");
        let operator_norm = {
            let v = load_f32_host(hfq, &format!("{lp}.operator_norm.weight"), h);
            gpu.upload_f32(&v, &[h])?
        };
        let ffn_norm = {
            let v = load_f32_host(hfq, &format!("{lp}.ffn_norm.weight"), h);
            gpu.upload_f32(&v, &[h])?
        };
        let q_proj = load_f16_gpu(hfq, gpu, &format!("{lp}.self_attn.q_proj.weight"))?;
        let k_proj = load_f16_gpu(hfq, gpu, &format!("{lp}.self_attn.k_proj.weight"))?;
        let v_proj = load_f16_gpu(hfq, gpu, &format!("{lp}.self_attn.v_proj.weight"))?;
        let o_proj = load_f16_gpu(hfq, gpu, &format!("{lp}.self_attn.o_proj.weight"))?;
        let q_norm = {
            let v = load_f32_host(hfq, &format!("{lp}.self_attn.q_norm.weight"), cfg.head_dim);
            gpu.upload_f32(&v, &[cfg.head_dim])?
        };
        let k_norm = {
            let v = load_f32_host(hfq, &format!("{lp}.self_attn.k_norm.weight"), cfg.head_dim);
            gpu.upload_f32(&v, &[cfg.head_dim])?
        };
        let gate_proj = load_f16_gpu(hfq, gpu, &format!("{lp}.self_attn.gate_proj.weight"))?;
        let w1 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w1.weight"))?;
        let w2 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w2.weight"))?;
        let w3 = load_f16_gpu(hfq, gpu, &format!("{lp}.feed_forward.w3.weight"))?;
        layers.push(DepthLayer {
            operator_norm,
            ffn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            gate_proj,
            w1,
            w2,
            w3,
        });
    }

    // Host codebook tables (12 per-codebook embeddings + norms).
    let cb = cfg.codebook_size;
    let mut host_codebook_embeddings = Vec::with_capacity(cfg.num_codebooks);
    let mut host_codebook_norms = Vec::with_capacity(cfg.num_codebooks);
    for i in 0..cfg.num_codebooks {
        let emb = load_f32_host(
            hfq,
            &format!("{pfx}.depth_embeddings.{i}.embedding.weight"),
            cb * h,
        );
        host_codebook_embeddings.push(emb);
        let nm = load_f32_host(hfq, &format!("{pfx}.depth_embeddings.{i}.embedding_norm.weight"), h);
        host_codebook_norms.push(nm);
    }
    let host_mask_embedding = load_f32_host(hfq, &format!("{pfx}.mask_embedding"), h);

    // Shared codebook embedding (model.codebook_embedding) — host F16.
    let host_shared_codebook_embedding = {
        let (info, data) = hfq.tensor_data("model.codebook_embedding.weight").ok_or_else(|| {
            hip_bridge::HipError::new(0, "depth: model.codebook_embedding.weight not found")
        })?;
        bytes_to_f16(info.quant_type, data)
    };

    eprintln!(
        "  depth decoder loaded: {} layers, {} codebooks ({} entries), shared emb {} rows",
        cfg.num_hidden_layers,
        cfg.num_codebooks,
        cb,
        host_shared_codebook_embedding.len() / cfg.text_hidden_size
    );

    Ok(DepthWeights {
        depth_linear_w,
        depth_linear_b,
        embedding_norm,
        layers,
        host_codebook_embeddings,
        host_codebook_norms,
        host_mask_embedding,
        host_shared_codebook_embedding,
    })
}

// ─── Linear helpers (mirror lfm3-vl) ─────────────────────────────────────────

/// Y[n, out_dim] = W_f16[out_dim, in_dim] @ X_f32[n, in_dim]^T  (no bias).
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

// ─── Depth decoder forward (trunk + logits) ──────────────────────────────────

/// Run the depth decoder for ONE frame.
///
/// `hidden`: LM hidden state at the position preceding the frame
///           (length `text_hidden_size`, host F32).
/// `codes`:  current 12 codebook values (host); masked slots are ignored.
/// `masked`: which codebooks are hidden from the decoder (it predicts these).
///
/// Returns `[num_codebooks * codebook_size]` logits (host F32), row-major
/// `[codebook, code]` — i.e. logits for codebook `i` are at
/// `[i*codebook_size .. (i+1)*codebook_size]`.
pub fn depth_forward(
    cfg: &DepthConfig,
    weights: &DepthWeights,
    gpu: &mut Gpu,
    hidden: &[f32],
    codes: &[u32],
    masked: &[bool],
) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let th = cfg.text_hidden_size;
    let ncb = cfg.num_codebooks;
    let cb = cfg.codebook_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let seq = ncb; // 12 codebook slots == sequence length
    assert_eq!(hidden.len(), th);
    assert_eq!(codes.len(), ncb);
    assert_eq!(masked.len(), ncb);

    // ── 1. depth_linear(hidden) → [num_codebooks, hidden] (GPU), + bias ──
    let hidden_gpu = gpu
        .upload_f32(hidden, &[th])
        .map_err(|e| format!("depth: upload hidden: {e:?}"))?;
    let dl = linear_f16_no_bias(gpu, &weights.depth_linear_w, &hidden_gpu, ncb * h, th, 1)
        .map_err(|e| format!("depth: depth_linear: {e:?}"))?;
    gpu.bias_add_f32(&dl, &weights.depth_linear_b, 1, ncb * h)
        .map_err(|e| format!("depth: depth_linear bias: {e:?}"))?;
    gpu.free_tensor(hidden_gpu).map_err(|e| format!("depth: free: {e:?}"))?;
    let dl_host = gpu
        .download_f32(&dl)
        .map_err(|e| format!("depth: download depth_linear: {e:?}"))?;
    gpu.free_tensor(dl).map_err(|e| format!("depth: free: {e:?}"))?;

    // ── 2. CPU: depth_in[slot] = depth_linear[slot] + (masked ? mask : code_emb) ──
    let mut depth_in = vec![0.0f32; seq * h];
    for slot in 0..seq {
        let base = slot * h;
        let dl_row = &dl_host[base..base + h];
        let emb: &[f32] = if masked[slot] {
            &weights.host_mask_embedding
        } else {
            let code = codes[slot] as usize;
            let off = code * h;
            &weights.host_codebook_embeddings[slot][off..off + h]
        };
        for k in 0..h {
            depth_in[base + k] = dl_row[k] + emb[k];
        }
    }
    let x = gpu
        .upload_f32(&depth_in, &[seq * h])
        .map_err(|e| format!("depth: upload depth_in: {e:?}"))?;

    // ── 3. positions 0..seq-1 (i32 bytes into an F32-alloc'd tensor) ──
    let positions = gpu
        .alloc_tensor(&[seq], DType::F32)
        .map_err(|e| format!("depth: alloc positions: {e:?}"))?;
    let pos_bytes: Vec<u8> = (0..seq as i32).flat_map(|p| p.to_ne_bytes()).collect();
    gpu.hip
        .memcpy_htod(&positions.buf, &pos_bytes)
        .map_err(|e| format!("depth: upload positions: {e:?}"))?;

    // ── 4. Bidirectional trunk: 3 full_attention layers ──
    for layer in &weights.layers {
        // operator_norm → tmp [seq, h]
        let tmp = gpu
            .alloc_tensor(&[seq * h], DType::F32)
            .map_err(|e| format!("depth: alloc tmp: {e:?}"))?;
        gpu.rmsnorm_batched(&x, &layer.operator_norm, &tmp, seq, h, eps)
            .map_err(|e| format!("depth: operator rmsnorm: {e:?}"))?;

        // q/k/v projections
        let q = linear_f16_no_bias(gpu, &layer.q_proj, &tmp, n_heads * hd, h, seq)
            .map_err(|e| format!("depth: q_proj: {e:?}"))?;
        let k = linear_f16_no_bias(gpu, &layer.k_proj, &tmp, n_kv * hd, h, seq)
            .map_err(|e| format!("depth: k_proj: {e:?}"))?;
        let v = linear_f16_no_bias(gpu, &layer.v_proj, &tmp, n_kv * hd, h, seq)
            .map_err(|e| format!("depth: v_proj: {e:?}"))?;

        // per-head QK RMSNorm (head_dim slices)
        gpu.rmsnorm_batched(&q, &layer.q_norm, &q, seq * n_heads, hd, eps)
            .map_err(|e| format!("depth: q_norm: {e:?}"))?;
        gpu.rmsnorm_batched(&k, &layer.k_norm, &k, seq * n_kv, hd, eps)
            .map_err(|e| format!("depth: k_norm: {e:?}"))?;

        // full-rotary RoPE (n_rot = head_dim), positions 0..seq-1
        gpu.rope_partial_interleaved_f32_batched(
            &q, &k, &positions, n_heads, n_kv, hd, hd, cfg.rope_theta, seq, 0,
        )
        .map_err(|e| format!("depth: rope: {e:?}"))?;

        // bidirectional GQA attention
        let attn_out = gpu
            .alloc_tensor(&[seq * n_heads * hd], DType::F32)
            .map_err(|e| format!("depth: alloc attn_out: {e:?}"))?;
        gpu.vit_attention_gqa_f32(&q, &k, &v, &attn_out, seq, n_heads, n_kv, hd)
            .map_err(|e| format!("depth: attention: {e:?}"))?;
        gpu.free_tensor(q).map_err(|e| format!("depth: free: {e:?}"))?;
        gpu.free_tensor(k).map_err(|e| format!("depth: free: {e:?}"))?;
        gpu.free_tensor(v).map_err(|e| format!("depth: free: {e:?}"))?;

        // sigmoid output gate: gate_proj(tmp) → [seq, n_heads]; attn_out *= sigmoid(gate)
        let gate = linear_f16_no_bias(gpu, &layer.gate_proj, &tmp, n_heads, h, seq)
            .map_err(|e| format!("depth: gate_proj: {e:?}"))?;
        gpu.sigmoid_head_gate_mul_f32(&attn_out, &gate, hd)
            .map_err(|e| format!("depth: sigmoid gate: {e:?}"))?;
        gpu.free_tensor(gate).map_err(|e| format!("depth: free: {e:?}"))?;
        gpu.free_tensor(tmp).map_err(|e| format!("depth: free: {e:?}"))?;

        // o_proj → proj_out [seq, h]
        let proj_out = linear_f16_no_bias(gpu, &layer.o_proj, &attn_out, n_heads * hd, n_heads * hd, seq)
            .map_err(|e| format!("depth: o_proj: {e:?}"))?;
        gpu.free_tensor(attn_out).map_err(|e| format!("depth: free: {e:?}"))?;

        // residual (no sandwich norm)
        gpu.add_inplace_f32(&x, &proj_out)
            .map_err(|e| format!("depth: attn residual: {e:?}"))?;
        gpu.free_tensor(proj_out).map_err(|e| format!("depth: free: {e:?}"))?;

        // FFN: ffn_norm → SwiGLU
        let ffn_tmp = gpu
            .alloc_tensor(&[seq * h], DType::F32)
            .map_err(|e| format!("depth: alloc ffn_tmp: {e:?}"))?;
        gpu.rmsnorm_batched(&x, &layer.ffn_norm, &ffn_tmp, seq, h, eps)
            .map_err(|e| format!("depth: ffn rmsnorm: {e:?}"))?;
        let g = linear_f16_no_bias(gpu, &layer.w1, &ffn_tmp, cfg.intermediate_size, h, seq)
            .map_err(|e| format!("depth: w1: {e:?}"))?;
        let u = linear_f16_no_bias(gpu, &layer.w3, &ffn_tmp, cfg.intermediate_size, h, seq)
            .map_err(|e| format!("depth: w3: {e:?}"))?;
        gpu.free_tensor(ffn_tmp).map_err(|e| format!("depth: free: {e:?}"))?;
        let act = gpu
            .alloc_tensor(&[seq * cfg.intermediate_size], DType::F32)
            .map_err(|e| format!("depth: alloc act: {e:?}"))?;
        gpu.silu_mul_f32(&g, &u, &act)
            .map_err(|e| format!("depth: silu_mul: {e:?}"))?;
        gpu.free_tensor(g).map_err(|e| format!("depth: free: {e:?}"))?;
        gpu.free_tensor(u).map_err(|e| format!("depth: free: {e:?}"))?;
        let ffn_out = linear_f16_no_bias(gpu, &layer.w2, &act, h, cfg.intermediate_size, seq)
            .map_err(|e| format!("depth: w2: {e:?}"))?;
        gpu.free_tensor(act).map_err(|e| format!("depth: free: {e:?}"))?;
        gpu.add_inplace_f32(&x, &ffn_out)
            .map_err(|e| format!("depth: ffn residual: {e:?}"))?;
        gpu.free_tensor(ffn_out).map_err(|e| format!("depth: free: {e:?}"))?;
    }

    // ── 5. final embedding_norm → depth_out [seq, h] ──
    let depth_out = gpu
        .alloc_tensor(&[seq * h], DType::F32)
        .map_err(|e| format!("depth: alloc depth_out: {e:?}"))?;
    gpu.rmsnorm_batched(&x, &weights.embedding_norm, &depth_out, seq, h, eps)
        .map_err(|e| format!("depth: final norm: {e:?}"))?;
    gpu.free_tensor(x).map_err(|e| format!("depth: free: {e:?}"))?;
    gpu.free_tensor(positions).map_err(|e| format!("depth: free: {e:?}"))?;

    let depth_out_host = gpu
        .download_f32(&depth_out)
        .map_err(|e| format!("depth: download depth_out: {e:?}"))?;
    gpu.free_tensor(depth_out).map_err(|e| format!("depth: free: {e:?}"))?;

    // ── 6. CPU: per-codebook logits = embedding_norm[slot](depth_out[slot]) @ emb[slot] ──
    let mut logits = vec![0.0f32; ncb * cb];
    for slot in 0..ncb {
        let h_row = &depth_out_host[slot * h..slot * h + h];
        let normed = rmsnorm_cpu(h_row, &weights.host_codebook_norms[slot], eps);
        let emb = &weights.host_codebook_embeddings[slot]; // [cb, h] row-major
        let out = &mut logits[slot * cb..slot * cb + cb];
        for code in 0..cb {
            let row = &emb[code * h..code * h + h];
            let mut s = 0.0f32;
            for k in 0..h {
                s += row[k] * normed[k];
            }
            out[code] = s;
        }
    }
    Ok(logits)
}

/// CPU RMSNorm: `(x / sqrt(mean(x^2) + eps)) * weight`.
fn rmsnorm_cpu(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut var = 0.0f32;
    for &v in x {
        var += v * v;
    }
    var /= n as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().zip(w).map(|(&v, &wi)| v * inv * wi).collect()
}

// ─── MaskGIT sampling ────────────────────────────────────────────────────────

/// A minimal seeded RNG (xorshift64) for MaskGIT order + multinomial sampling.
/// Deterministic given the seed — set `HIPFIRE_DETERMINISTIC=1` to pin it.
pub struct DepthRng {
    state: u64,
}

impl DepthRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Unbiased bounded draw from [0, bound). The naive `next_u64() % bound`
    /// carries the (tiny but free to remove) modulo bias of
    /// 2^64 % bound — NITS-1. Rejection is unreachable in practice for
    /// the depth decoder's bound (≤ codebook_size ≈ 2017: rejection
    /// probability < 2^-51) but keeps the distribution exactly uniform.
    pub fn randbounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        // threshold = 2^64 % bound, computed without overflow.
        let threshold = u64::MIN.wrapping_sub(bound) % bound;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % bound;
            }
        }
    }
    /// Random permutation of 0..n (Fisher-Yates).
    pub fn randperm(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.randbounded((i + 1) as u64) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// Sample one code from a codebook's logits: repetition penalty → temperature →
/// multinomial. Mirrors HF `Lfm3OmniForConditionalGeneration._sample` with
/// `temperature` + `repetition_penalty` (no top_k for audio).
fn sample_code(
    logits: &[f32],
    temperature: f32,
    rep_penalty: f32,
    history: &[u32],
    rng: &mut DepthRng,
) -> u32 {
    let n = logits.len();
    let mut scores: Vec<f32> = logits.to_vec();
    // Repetition penalty (HF RepetitionPenaltyLogitsProcessor): for each prior
    // code, score > 0 → /= penalty, else *= penalty.
    if rep_penalty != 1.0 {
        // REPPEN-1: HF's gather→where→scatter is idempotent per UNIQUE token.
        // Iterating the raw history window compounded the penalty once per
        // DUPLICATE occurrence, over-penalizing codes that legitimately recur
        // across speech frames (and diverging from HF the more a code repeats
        // inside the 64-frame window). Penalize each code at most once.
        let mut penalized = vec![false; n];
        for &c in history {
            let ci = c as usize;
            if ci < n && !penalized[ci] {
                penalized[ci] = true;
                if scores[ci] > 0.0 {
                    scores[ci] /= rep_penalty;
                } else {
                    scores[ci] *= rep_penalty;
                }
            }
        }
    }
    // Temperature.
    for s in scores.iter_mut() {
        *s /= temperature;
    }
    // Softmax (numerically stable) → multinomial sample.
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = (scores[i] - max).exp();
        exps[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= 0.0 {
        // Degenerate (all -inf) — fall back to argmax of the raw logits.
        let (mi, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return mi as u32;
    }
    let u = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        acc += e;
        if acc >= u {
            return i as u32;
        }
    }
    (n - 1) as u32
}

/// Sample the 12 codebooks of one speech frame via MaskGIT.
///
/// `hidden`: LM hidden state at the position preceding the frame.
/// `history`: previous frames' codes (each `[u32; num_codebooks]`), newest last,
///            for the repetition penalty (truncated to `audio_repetition_window`).
/// Returns the 12 sampled codes. A frame holding `audio_eos_code` in ANY
/// codebook ends the speech segment (the caller checks — HF
/// `(codes == audio_eos_code).any()`, and the whole frame is then replaced
/// with the eos code; see the omni.rs EOS-1 notes).
pub fn sample_audio_frame(
    cfg: &DepthConfig,
    weights: &DepthWeights,
    gpu: &mut Gpu,
    hidden: &[f32],
    history: &[Vec<u32>],
    rng: &mut DepthRng,
) -> Result<Vec<u32>, String> {
    let ncb = cfg.num_codebooks;
    let cb = cfg.codebook_size;
    let mut codes = vec![0u32; ncb];
    let mut revealed = vec![false; ncb];
    let order = rng.randperm(ncb);

    // History window (newest last) per codebook.
    let win = cfg.audio_repetition_window;
    let hist_start = history.len().saturating_sub(win);

    // Walk the schedule, taking schedule[i] indices each pass (MaskGIT).
    let mut idx = 0usize;
    let mut pass = 0usize;
    while idx < ncb {
        let take = if pass < cfg.maskgit_schedule.len() {
            cfg.maskgit_schedule[pass]
        } else {
            ncb - idx // remainder (shouldn't happen for [4,4,4])
        };
        let pick = &order[idx..(idx + take).min(ncb)];
        idx += take;
        pass += 1;

        let masked: Vec<bool> = (0..ncb).map(|i| !revealed[i]).collect();
        let logits = depth_forward(cfg, weights, gpu, hidden, &codes, &masked)?;
        for &slot in pick {
            let row = &logits[slot * cb..slot * cb + cb];
            let hist: Vec<u32> = history[hist_start..]
                .iter()
                .map(|f| f[slot])
                .collect();
            codes[slot] = sample_code(
                row,
                cfg.audio_temperature,
                cfg.audio_repetition_penalty,
                &hist,
                rng,
            );
            revealed[slot] = true;
        }
    }
    Ok(codes)
}

// ─── Shared codebook embedding (feed sampled codes back to the LM) ───────────

/// Gather the LM embedding for one sampled frame: the sum of the 12 codebook
/// embeddings (`model.codebook_embedding[codes[i] + i*codebook_size]` summed
/// over the codebook axis). Mirrors HF `Lfm3OmniModel.get_audio_out_embeds`.
///
/// Returns `[text_hidden_size]` F32 — the embedding spliced into the LM context
/// at the `audio_out_token_id` position via `decode_step_with_embed`.
pub fn get_audio_out_embeds(
    cfg: &DepthConfig,
    weights: &DepthWeights,
    codes: &[u32],
) -> Vec<f32> {
    let th = cfg.text_hidden_size;
    let cb = cfg.codebook_size;
    let mut out = vec![0.0f32; th];
    for (i, &code) in codes.iter().enumerate() {
        let offset = i * cb + code as usize;
        let row = &weights.host_shared_codebook_embedding[offset * th..(offset + 1) * th];
        for (o, &h) in out.iter_mut().zip(row) {
            *o += f16_to_f32(h);
        }
    }
    out
}
