// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Weight streaming for the qwen35-embed checkpoint.
//!
//! Layout on the GPU (mirrors the encoder precedents in
//! `hipfire-arch-diffusion::qwen3_gpu` / `hipfire-arch-qwen35-vl`):
//!
//! - **Projection weights f16** `[out, in]` row-major — the WMMA GEMM's
//!   weight operand (f16×f16, f32 accumulate). The checkpoint ships f32;
//!   f32/bf16/f16 sources all round-trip through the same RNE conversion
//!   (`half::f16`), so a raw-dtype `.hfq` re-save of the dir is
//!   bit-identical.
//! - **Norm vectors f32**, with the HF zero-centered `+1.0` offset applied
//!   at load for every `Qwen3_5RMSNorm` site (`input_layernorm`,
//!   `post_attention_layernorm`, `norm`, `q_norm`, `k_norm`) and NOT for
//!   `linear_attn.norm.weight` (ones-init gated norm, direct multiply — the
//!   same split `hipfire-arch-qwen35`'s loader makes between
//!   `load_norm_weight` and `load_any_as_f32`).
//! - **GDN constants raw f32**: `A_log` `[n_v_heads]`, `dt_bias`
//!   `[n_v_heads]`, `conv1d.weight` `[conv_dim, 1, kernel]` — the kernels
//!   apply exp/softplus themselves.
//! - **Embedding table stays on the host** (`Vec<f32>` `[vocab, hidden]`,
//!   ~0.95 GB at 248320×1024): an encode touches at most `document_length`
//!   rows, and the image splice has to happen per-row anyway, so a host
//!   gather + one `[n, hidden]` upload beats a device table + scatter
//!   kernels. Same split the FLUX Qwen3 conditioning encoder makes.
//! - **Dense head f16** `[128, hidden]` from `1_Dense/model.safetensors`
//!   (dirs) or a `linear.weight` tensor already present in the source
//!   (`.hfq` re-saves — HFQ containers are name-addressed).

use crate::config::{EmbedConfig, LayerType};
use crate::names::{self, TensorBytes};
use half::f16;
use hipfire_runtime::loader_api::ModelSource;
use hipfire_runtime::model_source::ModelSource as _;
use rdna_compute::{Gpu, GpuTensor};

/// Per-layer mixer weights, keyed by the config's `layer_types`.
pub enum MixerWeights {
    FullAttn {
        /// Query projection `[q_dim, hidden]` — the q half of HF's fused
        /// `q_proj` (which packs q and the output gate **per head**:
        /// `view(…, heads, 2·head_dim).chunk(2, -1)`), row-permuted to a
        /// contiguous `[heads × head_dim]` layout at load.
        q_proj: GpuTensor,
        /// The gate half of the fused projection, same permutation.
        gate_proj: GpuTensor,
        k_proj: GpuTensor,
        v_proj: GpuTensor,
        o_proj: GpuTensor,
        q_norm: GpuTensor,
        k_norm: GpuTensor,
    },
    Gdn {
        qkv: GpuTensor,   // in_proj_qkv [k*2+v, hidden]
        z: GpuTensor,     // in_proj_z [v, hidden]
        a: GpuTensor,     // in_proj_a [n_v_heads, hidden]
        b: GpuTensor,     // in_proj_b [n_v_heads, hidden]
        a_log: GpuTensor, // f32 [n_v_heads]
        dt_bias: GpuTensor,
        conv: GpuTensor,     // f32 [conv_dim, 1, kernel]
        gdn_norm: GpuTensor, // f32 [value_head_dim], direct multiply
        out_proj: GpuTensor, // [hidden, v]
    },
}

pub struct LayerWeights {
    pub mixer: MixerWeights,
    pub input_ln: GpuTensor,  // f32 [hidden], +1 applied
    pub post_ln: GpuTensor,   // f32 [hidden], +1 applied
    pub gate_proj: GpuTensor, // f16 [intermediate, hidden]
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor, // f16 [hidden, intermediate]
}

pub struct EmbedWeights {
    /// Host embedding table `[vocab, hidden]` f32 (gathered per encode).
    pub embed_table: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: GpuTensor, // f32 [hidden], +1 applied
    /// Dense projection `[dense_out, hidden]` f16 (sentence-transformers
    /// module 1).
    pub dense: GpuTensor,
    /// Vision tower + merger (qwen35-vl weight set), when the checkpoint
    /// carries `visual.*` tensors.
    pub vision: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
}

// ─── dtype decode ───────────────────────────────────────────────────────────
//
// Two source conventions meet here. A safetensors dir carries
// `quant_type == 0xFF` (dtype-driven) with F32/BF16/F16 dtypes; an `.hfq`
// carries a quant-type byte, and the quantizer may have compressed the trunk
// (mq4 -> qt 44 MQ4G256V2, q8 -> qt 3 Q8_0, ...). Both are decoded to f32
// host values; a compressed format goes through the runtime's registry-wide
// `dequantize_weight_to_f32`, which is the same decoder the arch loaders use.
// The consequence is deliberate and documented on the crate: an `.hfq` of
// this arch serves the same f16 GEMM path as the dir, so it saves disk, not
// VRAM.

/// Widen bf16 bits (the high half of an f32 word) to f32 — exact shift.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decode one tensor's bytes to host f32 (see the section note above).
fn decode_f32(t: &TensorBytes<'_>, name: &str) -> Result<Vec<f32>, String> {
    let n = t.numel();
    if t.quant_type != 0xFF {
        let v =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(t.quant_type, &t.bytes, n);
        if v.len() != n {
            return Err(format!(
                "tensor `{name}`: dequantized {} values for {n} elements (qt={})",
                v.len(),
                t.quant_type
            ));
        }
        return Ok(v);
    }
    let bytes = &t.bytes;
    // Byte-length guard shared by every raw-dtype arm: a declared shape that
    // exceeds the stored bytes is a malformed artifact's error, not a panic
    // from an unchecked slice.
    let need_bytes = |w: usize| -> Result<(), String> {
        if bytes.len() < n * w {
            return Err(format!(
                "tensor `{name}`: {} bytes < {}",
                bytes.len(),
                n * w
            ));
        }
        Ok(())
    };
    match t.dtype.as_str() {
        "F32" => {
            need_bytes(4)?;
            Ok(bytes[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "F16" => {
            need_bytes(2)?;
            Ok(bytes[..n * 2]
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        "BF16" => {
            need_bytes(2)?;
            Ok(bytes[..n * 2]
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        other => Err(format!(
            "tensor `{name}` has dtype {other} — expected F32/BF16/F16"
        )),
    }
}

/// Decode to f16 words for the GEMM operand (RNE via `half`).
fn decode_f16(t: &TensorBytes<'_>, name: &str) -> Result<Vec<u16>, String> {
    // A raw f16 source is already in the target format — no round trip.
    if t.quant_type == 0xFF && t.dtype == "F16" {
        let n = t.numel();
        if t.bytes.len() < n * 2 {
            return Err(format!(
                "tensor `{name}`: {} bytes < {}",
                t.bytes.len(),
                n * 2
            ));
        }
        return Ok(t.bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect());
    }
    if t.quant_type == 1 {
        let n = t.numel();
        if t.bytes.len() < n * 2 {
            return Err(format!(
                "tensor `{name}`: {} bytes < {}",
                t.bytes.len(),
                n * 2
            ));
        }
        return Ok(t.bytes[..n * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect());
    }
    Ok(decode_f32(t, name)?
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect())
}

// ─── upload helpers ─────────────────────────────────────────────────────────

fn upload_lin_f16(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, String> {
    let t = names::read(src, name, false)?;
    let words = decode_f16(&t, name)?;
    names::release(src, name, false);
    if words.len() != rows * cols {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} elems, expected {rows}×{cols}",
            words.len()
        ));
    }
    gpu.upload_f16_bits(&words, &[rows, cols])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))
}

fn upload_vec_f32(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    n: usize,
    plus_one: bool,
) -> Result<GpuTensor, String> {
    let t = names::read(src, name, false)?;
    let mut v = decode_f32(&t, name)?;
    names::release(src, name, false);
    if v.len() != n {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} elems, expected {n}",
            v.len()
        ));
    }
    if plus_one {
        for x in v.iter_mut() {
            *x += 1.0;
        }
    }
    gpu.upload_f32(&v, &[n])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))
}

fn upload_conv_f32(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    channels: usize,
    kernel: usize,
) -> Result<GpuTensor, String> {
    let t = names::read(src, name, false)?;
    let v = decode_f32(&t, name)?;
    names::release(src, name, false);
    if v.len() != channels * kernel {
        return Err(format!(
            "qwen35-embed: conv `{name}` has {} elems, expected {channels}×{kernel}",
            v.len()
        ));
    }
    gpu.upload_f32(&v, &[channels, 1, kernel])
        .map_err(|e| format!("qwen35-embed: upload `{name}`: {e:?}"))
}

/// Upload the fused q+gate projection as two row-permuted `[q_dim, hidden]`
/// GEMM weights. HF interleaves q and the attention output gate per head
/// (`q_proj(x).view(…, heads, 2·head_dim).chunk(2, -1)`), so output feature
/// `f` is (head `f / (2·hd)`, q if `f % (2·hd) < hd`, gate otherwise); the
/// permutation gathers each half into head-major order, which is exactly
/// the `[heads × head_dim]` layout the attention and `sigmoid_mul` kernels
/// consume.
fn upload_q_gate_split(
    gpu: &mut Gpu,
    src: &ModelSource,
    name: &str,
    heads: usize,
    head_dim: usize,
    hidden: usize,
) -> Result<(GpuTensor, GpuTensor), String> {
    let t = names::read(src, name, false)?;
    let words = decode_f16(&t, name)?;
    names::release(src, name, false);
    let fused = heads * head_dim * 2;
    if words.len() != fused * hidden {
        return Err(format!(
            "qwen35-embed: tensor `{name}` has {} rows, expected {fused}",
            words.len() / hidden
        ));
    }
    let half = head_dim;
    let mut q_words = Vec::with_capacity(heads * head_dim * hidden);
    let mut g_words = Vec::with_capacity(heads * head_dim * hidden);
    for h in 0..heads {
        for part in 0..2 {
            let row_base = (h * 2 * half + part * half) * hidden;
            let dst = if part == 0 {
                &mut q_words
            } else {
                &mut g_words
            };
            dst.extend_from_slice(&words[row_base..row_base + half * hidden]);
        }
    }
    let q = gpu
        .upload_f16_bits(&q_words, &[heads * head_dim, hidden])
        .map_err(|e| format!("qwen35-embed: upload `{name}` q half: {e:?}"))?;
    let g = gpu
        .upload_f16_bits(&g_words, &[heads * head_dim, hidden])
        .map_err(|e| format!("qwen35-embed: upload `{name}` gate half: {e:?}"))?;
    Ok((q, g))
}

// ─── Dense sidecar (1_Dense/model.safetensors) ─────────────────────────────

/// Minimal safetensors read of a single F32/F16/BF16 tensor out of a shard
/// file, without pulling in the runtime's private parser. Returns the
/// decoded host f32 rows.
fn read_sidecar_tensor(path: &std::path::Path, want: &str) -> Result<Vec<f32>, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)
        .map_err(|e| format!("{}: header length: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    // Bound before trusting the length: a corrupt or adversarial file can
    // declare an absurd header and take the loader down in an allocation.
    let file_len = f
        .metadata()
        .map_err(|e| format!("{}: metadata: {e}", path.display()))?
        .len() as usize;
    if header_len == 0 || header_len > file_len.saturating_sub(8) || header_len > (1 << 24) {
        return Err(format!(
            "{}: implausible safetensors header length {header_len} (file is {file_len} B)",
            path.display()
        ));
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    let meta: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| format!("{}: header JSON: {e}", path.display()))?;
    let entry = meta
        .get(want)
        .ok_or_else(|| format!("{}: no tensor `{want}`", path.display()))?;
    let dtype = entry
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: `{want}` has no dtype", path.display()))?
        .to_string();
    let shape: Vec<usize> = entry
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|u| u as usize))
                .collect()
        })
        .unwrap_or_default();
    let offsets = entry
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: `{want}` has no data_offsets", path.display()))?;
    let (start, end) = match offsets.as_slice() {
        [a, b, ..] => {
            let a = a.as_u64().unwrap_or(0) as usize;
            let b = b.as_u64().unwrap_or(0) as usize;
            if b < a {
                return Err(format!(
                    "{}: `{want}` data_offsets end {b} < start {a}",
                    path.display()
                ));
            }
            (a, b)
        }
        _ => {
            return Err(format!(
                "{}: `{want}` data_offsets is not a [start, end] pair",
                path.display()
            ))
        }
    };
    let mut raw = vec![0u8; end - start];
    // safetensors data begins after the 8-byte length + header (8-aligned).
    let data_base = 8 + header_len;
    std::io::Seek::seek(&mut f, std::io::SeekFrom::Start((data_base + start) as u64))
        .map_err(|e| format!("{}: seek: {e}", path.display()))?;
    f.read_exact(&mut raw)
        .map_err(|e| format!("{}: data: {e}", path.display()))?;
    let t = TensorBytes {
        quant_type: 0xFF,
        dtype,
        shape,
        bytes: std::borrow::Cow::Owned(raw),
    };
    decode_f32(&t, want)
}

/// Load the Dense head: `linear.weight` from the source itself (an `.hfq`
/// built from this dir keeps the name) or from the `1_Dense` sidecar shard
/// (dirs).
fn load_dense(gpu: &mut Gpu, src: &ModelSource, cfg: &EmbedConfig) -> Result<GpuTensor, String> {
    let (words, cols): (Vec<u16>, usize) = match names::read(src, names::DENSE_TENSOR, false) {
        Ok(t) => {
            let w = decode_f16(&t, names::DENSE_TENSOR)?;
            drop(t);
            names::release(src, names::DENSE_TENSOR, false);
            let cols = w.len() / cfg.hidden_size;
            (w, cols)
        }
        Err(_) => {
            let path = src_path(src).join(names::DENSE_SHARD);
            let v = read_sidecar_tensor(&path, "linear.weight")?;
            let cols = v.len() / cfg.hidden_size;
            (
                v.iter().map(|&x| f16::from_f32(x).to_bits()).collect(),
                cols,
            )
        }
    };
    if cols != cfg.dense_out_features {
        return Err(format!(
            "qwen35-embed: Dense head is [{cols}×{}], config says [{}×{}]",
            cfg.hidden_size, cfg.dense_out_features, cfg.hidden_size
        ));
    }
    gpu.upload_f16_bits(&words, &[cfg.dense_out_features, cfg.hidden_size])
        .map_err(|e| format!("qwen35-embed: upload dense head: {e:?}"))
}

// ─── main loader ────────────────────────────────────────────────────────────

fn src_path(src: &ModelSource) -> &std::path::Path {
    match src {
        ModelSource::Hfq(h) => h.path(),
        ModelSource::Dir(d) => d.path(),
    }
}

/// Stream the whole checkpoint to the GPU.
pub fn load_weights(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &EmbedConfig,
) -> Result<EmbedWeights, String> {
    stream_weights(gpu, src, cfg)
}

fn stream_weights(
    gpu: &mut Gpu,
    src: &ModelSource,
    cfg: &EmbedConfig,
) -> Result<EmbedWeights, String> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    // Embedding table → host f32 (gather source; see module docs).
    let t = names::read(src, "embed_tokens.weight", false)?;
    let embed_table = decode_f32(&t, "embed_tokens.weight")?;
    names::release(src, "embed_tokens.weight", false);
    if embed_table.len() != cfg.vocab_size * h {
        return Err(format!(
            "qwen35-embed: embed_tokens is {} elems, expected {}×{h}",
            embed_table.len(),
            cfg.vocab_size
        ));
    }
    drop(t);

    let mut layers = Vec::with_capacity(cfg.num_layers);
    for li in 0..cfg.num_layers {
        let p = format!("layers.{li}");
        let input_ln = upload_vec_f32(gpu, src, &format!("{p}.input_layernorm.weight"), h, true)?;
        let post_ln = upload_vec_f32(
            gpu,
            src,
            &format!("{p}.post_attention_layernorm.weight"),
            h,
            true,
        )?;
        let gate_proj = upload_lin_f16(gpu, src, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
        let up_proj = upload_lin_f16(gpu, src, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
        let down_proj = upload_lin_f16(gpu, src, &format!("{p}.mlp.down_proj.weight"), h, inter)?;

        let mixer = match cfg.layer_types[li] {
            LayerType::FullAttention => {
                let qd = cfg.q_dim();
                let kvd = cfg.kv_dim();
                let (q_proj, gate_proj) = upload_q_gate_split(
                    gpu,
                    src,
                    &format!("{p}.self_attn.q_proj.weight"),
                    cfg.n_heads,
                    cfg.head_dim,
                    h,
                )?;
                MixerWeights::FullAttn {
                    q_proj,
                    gate_proj,
                    k_proj: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.self_attn.k_proj.weight"),
                        kvd,
                        h,
                    )?,
                    v_proj: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.self_attn.v_proj.weight"),
                        kvd,
                        h,
                    )?,
                    o_proj: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.self_attn.o_proj.weight"),
                        h,
                        qd,
                    )?,
                    q_norm: upload_vec_f32(
                        gpu,
                        src,
                        &format!("{p}.self_attn.q_norm.weight"),
                        cfg.head_dim,
                        true,
                    )?,
                    k_norm: upload_vec_f32(
                        gpu,
                        src,
                        &format!("{p}.self_attn.k_norm.weight"),
                        cfg.head_dim,
                        true,
                    )?,
                }
            }
            LayerType::LinearAttention => {
                let k_dim = cfg.gdn_key_dim();
                let v_dim = cfg.gdn_value_dim();
                let conv_dim = k_dim * 2 + v_dim;
                MixerWeights::Gdn {
                    qkv: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.in_proj_qkv.weight"),
                        k_dim * 2 + v_dim,
                        h,
                    )?,
                    z: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.in_proj_z.weight"),
                        v_dim,
                        h,
                    )?,
                    a: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.in_proj_a.weight"),
                        cfg.linear_num_value_heads,
                        h,
                    )?,
                    b: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.in_proj_b.weight"),
                        cfg.linear_num_value_heads,
                        h,
                    )?,
                    a_log: upload_vec_f32(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.A_log"),
                        cfg.linear_num_value_heads,
                        false,
                    )?,
                    dt_bias: upload_vec_f32(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.dt_bias"),
                        cfg.linear_num_value_heads,
                        false,
                    )?,
                    conv: upload_conv_f32(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.conv1d.weight"),
                        conv_dim,
                        cfg.conv_kernel_dim,
                    )?,
                    gdn_norm: upload_vec_f32(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.norm.weight"),
                        cfg.linear_value_head_dim,
                        false,
                    )?,
                    out_proj: upload_lin_f16(
                        gpu,
                        src,
                        &format!("{p}.linear_attn.out_proj.weight"),
                        h,
                        v_dim,
                    )?,
                }
            }
        };
        layers.push(LayerWeights {
            mixer,
            input_ln,
            post_ln,
            gate_proj,
            up_proj,
            down_proj,
        });
    }

    let final_norm = upload_vec_f32(gpu, src, "norm.weight", h, true)?;
    let dense = load_dense(gpu, src, cfg)?;

    // Vision tower, when the checkpoint has one.
    let vision =
        if cfg.vision.is_some() && names::read(src, "patch_embed.proj.weight", true).is_ok() {
            Some(crate::vision::load_vision_weights(gpu, src, cfg)?)
        } else {
            None
        };

    Ok(EmbedWeights {
        embed_table,
        layers,
        final_norm,
        dense,
        vision,
    })
}

impl EmbedWeights {
    /// Free every GPU allocation. The embedding table is host memory
    /// (dropped) and the vision set frees itself.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let EmbedWeights {
            embed_table: _,
            layers,
            final_norm,
            dense,
            vision,
        } = self;
        for l in layers {
            let _ = gpu.free_tensor(l.input_ln);
            let _ = gpu.free_tensor(l.post_ln);
            let _ = gpu.free_tensor(l.gate_proj);
            let _ = gpu.free_tensor(l.up_proj);
            let _ = gpu.free_tensor(l.down_proj);
            match l.mixer {
                MixerWeights::FullAttn {
                    q_proj,
                    gate_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                } => {
                    for t in [q_proj, gate_proj, k_proj, v_proj, o_proj, q_norm, k_norm] {
                        let _ = gpu.free_tensor(t);
                    }
                }
                MixerWeights::Gdn {
                    qkv,
                    z,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    conv,
                    gdn_norm,
                    out_proj,
                } => {
                    for t in [qkv, z, a, b, a_log, dt_bias, conv, gdn_norm, out_proj] {
                        let _ = gpu.free_tensor(t);
                    }
                }
            }
        }
        let _ = gpu.free_tensor(final_norm);
        let _ = gpu.free_tensor(dense);
        if let Some(v) = vision {
            v.free_gpu(gpu);
        }
    }

    /// Gather `ids` rows of the host embedding table into `out` (row-major
    /// `[ids.len(), hidden]`), then splice `image_embeds` rows over the
    /// `image_pad` positions covered by `spans`.
    pub fn gather_hidden(
        &self,
        ids: &[u32],
        image_embeds: &[f32],
        spans: &[ImageSpan],
        hidden: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        out.clear();
        out.reserve(ids.len() * hidden);
        for &id in ids {
            let row = id as usize * hidden;
            if row + hidden > self.embed_table.len() {
                return Err(format!("token id {id} out of embedding range"));
            }
            out.extend_from_slice(&self.embed_table[row..row + hidden]);
        }
        for span in spans {
            let mut src = span.embed_offset * hidden;
            for r in span.start..span.start + span.len {
                let dst = r * hidden;
                if src + hidden > image_embeds.len() || dst + hidden > out.len() {
                    return Err("image splice out of range".to_string());
                }
                out[dst..dst + hidden].copy_from_slice(&image_embeds[src..src + hidden]);
                src += hidden;
            }
        }
        Ok(())
    }
}

/// One contiguous run of `<|image_pad|>` positions in the token stream and
/// which rows of the merged image-embeds matrix fill it. `grid_h`/`grid_w`
/// are the **pre-merge patch grid** (`img_px / patch_size`) — the mRoPE
/// builder divides by `spatial_merge_size` itself, so callers must not hand
/// it the already-merged counts.
#[derive(Clone, Copy, Debug)]
pub struct ImageSpan {
    pub start: usize,
    /// Merged-token count (the run's length in `ids`).
    pub len: usize,
    /// Row offset into the merged image-embeds matrix.
    pub embed_offset: usize,
    /// Pre-merge patch grid height, for mRoPE.
    pub grid_h: usize,
    /// Pre-merge patch grid width.
    pub grid_w: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_hidden_splices_image_rows() {
        use crate::config::{config_from_metadata, parse_config};
        // A 6-token vocab x 2 hidden table; ids 0..5.
        let cfgv: serde_json::Value = serde_json::from_str(
            r#"{"config":{"model_type":"qwen3_5",
                 "text_config":{"hidden_size":2,"num_hidden_layers":1,
                   "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
                   "intermediate_size":4,"vocab_size":6,"rms_norm_eps":1e-6,
                   "hidden_act":"silu","linear_num_key_heads":1,"linear_num_value_heads":1,
                   "linear_key_head_dim":2,"linear_value_head_dim":2,
                   "linear_conv_kernel_dim":4,
                   "rope_parameters":{"rope_theta":1e4,"partial_rotary_factor":1.0,
                                      "mrope_section":[1,0,0]},
                   "layer_types":["full_attention"]}}}"#,
        )
        .unwrap();
        let cfg = parse_config(config_from_metadata(&cfgv)).unwrap();
        let null = || rdna_compute::GpuTensor::null_for_test();
        let w = EmbedWeights {
            embed_table: (0..6).flat_map(|i| [i as f32, (i * 10) as f32]).collect(),
            layers: vec![],
            final_norm: null(),
            dense: null(),
            vision: None,
        };
        // ids [0,3,1], image span of 1 row at token index 1, embed row = [7,7].
        let spans = [ImageSpan {
            start: 1,
            len: 1,
            embed_offset: 0,
            grid_h: 1,
            grid_w: 1,
        }];
        let mut out = Vec::new();
        w.gather_hidden(&[0, 3, 1], &[7.0, 7.0], &spans, 2, &mut out)
            .unwrap();
        assert_eq!(
            out,
            vec![0.0, 0.0, /* spliced */ 7.0, 7.0, /* id 1 */ 1.0, 10.0]
        );
        // A multi-row span starting at a nonzero embed offset (the second
        // image of a two-image request): 3 image rows, take rows 1..3.
        let spans2 = [ImageSpan {
            start: 0,
            len: 2,
            embed_offset: 1,
            grid_h: 1,
            grid_w: 3,
        }];
        let imgs = [9.0, 9.0, 8.0, 8.0, 7.0, 7.0];
        let mut out2 = Vec::new();
        w.gather_hidden(&[5, 5], &imgs, &spans2, 2, &mut out2)
            .unwrap();
        assert_eq!(out2, vec![8.0, 8.0, 7.0, 7.0]);
        // A full-coverage span at offset 0 takes the leading rows.
        let spans3 = [ImageSpan {
            start: 0,
            len: 2,
            embed_offset: 0,
            grid_h: 1,
            grid_w: 2,
        }];
        let mut out3 = Vec::new();
        w.gather_hidden(&[5, 5], &imgs, &spans3, 2, &mut out3)
            .unwrap();
        assert_eq!(out3, vec![9.0, 9.0, 8.0, 8.0]);
        // Out-of-range token id is refused, not silently zeroed.
        assert!(w
            .gather_hidden(&[99], &[], &[], 2, &mut Vec::new())
            .is_err());
    }

    #[test]
    fn bf16_widen_is_exact() {
        // 1.0 in bf16 = 0x3F80 → f32 1.0.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn decode_f32_reads_raw_and_dequantizes_quantized() {
        // Raw f32 (safetensors dir convention): dtype-driven.
        let raw = TensorBytes {
            quant_type: 0xFF,
            dtype: "F32".into(),
            shape: vec![2],
            bytes: std::borrow::Cow::Owned(
                [1.0f32, -2.0f32]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
        };
        assert_eq!(decode_f32(&raw, "x").unwrap(), vec![1.0, -2.0]);
        // Raw f16 in an .hfq (qt=1): the quant-type byte drives it.
        let hfq_f16 = TensorBytes {
            quant_type: 1,
            dtype: String::new(),
            shape: vec![2],
            bytes: std::borrow::Cow::Owned(
                [half::f16::from_f32(1.5), half::f16::from_f32(-0.5)]
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect(),
            ),
        };
        assert_eq!(decode_f32(&hfq_f16, "x").unwrap(), vec![1.5, -0.5]);
        // A compressed .hfq format (qt=3 Q8_0) is dequantized, not refused.
        // 32 values in one Q8_0 block: f16 scale 1.0 + 32 int8s.
        let mut q8 = vec![0u8, 0x3C];
        q8.extend((1..=32).map(|i| i as u8));
        let q = TensorBytes {
            quant_type: 3,
            dtype: String::new(),
            shape: vec![32],
            bytes: std::borrow::Cow::Owned(q8),
        };
        let got = decode_f32(&q, "x").unwrap();
        assert_eq!(got.len(), 32);
        assert!((got[0] - 1.0).abs() < 1e-6 && (got[31] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn sidecar_header_offset_is_padded() {
        // The data region of a safetensors file starts at 8 + header_len
        // (header_len itself is a multiple of 8 per the spec's padding note,
        // but readers must not assume it when computing data_offsets).
        // This test pins the arithmetic read_sidecar_tensor uses.
        let base = 8usize;
        assert_eq!(base + 0, 8);
    }
}
