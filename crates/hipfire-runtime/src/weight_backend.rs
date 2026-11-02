// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-agnostic weight loading: dequant primitives, HF tensor-name resolution,
//! and a `WeightBackend` trait abstracting HFQ vs ParoQuant on-disk formats.
//! Per-arch crates build their `load_layer` schema on top of this; the only
//! arch-varying knobs are the RMSNorm `+bias` and the name-candidate resolver.

use crate::hfq::HfqFile;
use crate::llama::{f16_to_f32, EmbeddingFormat, KvCache, WeightTensor};
use hip_bridge::HipResult;
use rdna_compute::{
    DType, Gpu, GpuTensor, HipError, MQ2G256V2_GROUP_BYTES, MQ3G256V2_GROUP_BYTES,
    MQ4C_GROUP_BYTES, MQ4G128V2_GROUP_BYTES, MQ5G256V2_GROUP_BYTES, MQ6G256V2_GROUP_BYTES,
};

/// Appended to every fixed-layout blob size refusal: the size only disagrees
/// when the file is damaged or was written by an older quantizer.
const BLOB_FIX: &str = "; the model file is corrupt or was written by an older quantizer — re-download it with `hipfire pull <model>` or re-quantize it";

/// Widen a little-endian BF16 byte stream to F32 (lossless: bf16 is the high
/// 16 bits of an f32). Used by the qt=16 paths in dequant_weight_raw/dequant_f32.
fn widen_bf16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// ── HF tensor-name resolution ───────────────────────────────────────────────

/// Candidate on-disk names for a logical tensor, covering the HF nested
/// vision-wrapper layout (`model.language_model.*`), the flat layout (`model.*`),
/// and the bare name, plus the `lm_head` special-case. Layout convention only —
/// not model-specific math — so any HF text tower can share it.
pub fn hf_name_candidates(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(4);
    let mut push = |s: String| {
        if !out.iter().any(|x| x == &s) {
            out.push(s);
        }
    };
    if name == "lm_head.weight" {
        push(name.to_string());
        push("model.language_model.lm_head.weight".to_string());
        push("model.lm_head.weight".to_string());
        return out;
    }
    if name.starts_with("model.") {
        push(name.to_string());
    } else {
        push(format!("model.language_model.{name}"));
        push(format!("model.{name}"));
        push(name.to_string());
    }
    out
}

/// Flat-only resolver for arches stored without the vision-wrapper nesting
/// (qwen2, llama). Tries `model.{name}` then bare.
pub fn flat_name_candidates(name: &str) -> Vec<String> {
    if name.starts_with("model.") {
        vec![name.to_string()]
    } else {
        vec![format!("model.{name}"), name.to_string()]
    }
}

// ── Layer-relative name builders ────────────────────────────────────────────

/// HFQ projection name: `layers.{layer}.{rel}.weight` (the backend's candidate
/// resolver then adds any layout prefix).
pub fn hfq_proj_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}.weight")
}
/// HFQ norm / raw-f32 name: `layers.{layer}.{rel}` (rel already carries `.weight`
/// where the on-disk tensor has it, e.g. `input_layernorm.weight`).
pub fn hfq_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}
/// PaRo projection base (augmentor appends `.qweight`/`.weight`): `{mp}.layers.{layer}.{rel}`.
pub fn paro_proj_name(mp: &str, layer: usize, rel: &str) -> String {
    format!("{mp}.layers.{layer}.{rel}")
}
/// PaRo norm/raw-f32 name for `paro_load_norm`/`paro_load_f32`, which prepend `mp`
/// THEMSELVES — so this is prefix-LESS: `layers.{layer}.{rel}`.
pub fn paro_plain_name(layer: usize, rel: &str) -> String {
    format!("layers.{layer}.{rel}")
}

// ── Embedding / tied-lm_head primitives ──────────────────────────────────

/// How an embedding table's on-disk bytes map to the device.
#[derive(Debug)]
pub enum EmbedPlan {
    /// Upload bytes verbatim; the lookup kernel dequantizes on the fly.
    Raw(EmbeddingFormat),
    /// Host-decode to f32 (via `dequant_f32`) then upload as F32.
    HostF32,
}

/// Pure quant_type → plan. GPU-free, unit-testable.
///
/// qt 6 → Raw(HFQ4G256), 7 → Raw(HFQ4G128), 3 → Raw(Q8_0),
/// qt 1|2|16|40|41|44|45|47-50 → HostF32, else → error with the supported list.
pub fn embed_classify(quant_type: u8) -> HipResult<EmbedPlan> {
    match quant_type {
        6 => Ok(EmbedPlan::Raw(EmbeddingFormat::HFQ4G256)),
        7 => Ok(EmbedPlan::Raw(EmbeddingFormat::HFQ4G128)),
        3 => Ok(EmbedPlan::Raw(EmbeddingFormat::Q8_0)),
        1 | 2 | 16 | 40 | 41 | 44 | 45 | 47 | 48 | 49 | 50 => Ok(EmbedPlan::HostF32),
        other => Err(hip_bridge::HipError::new(
            0,
            &format!(
                "unsupported embedding quant_type {other}: this model's embedding table is stored in a \
                 weight format hipfire cannot load. Supported embedding formats: 1 (F16), 2 (F32), 3 (Q8_0), \
                 6 (HFQ4G256), 7 (HFQ4G128), 16 (BF16), 40 (TQ2G128), 41 (BQ1G128), 44/45/47-50 (MQ*G256V2). \
                 Update hipfire, or re-quantize / re-download the model with a supported embedding format."
            ),
        )),
    }
}

/// Load an embedding table to the device. Unifies the qwen35 and qwen2
/// hand-written matches. Returns the device tensor + its on-GPU format.
pub fn load_embedding(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    vocab: usize,
    dim: usize,
) -> HipResult<(GpuTensor, EmbeddingFormat)> {
    match embed_classify(quant_type)? {
        EmbedPlan::Raw(fmt) => {
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok((buf, fmt))
        }
        EmbedPlan::HostF32 => {
            // dequant_f32 uploads with shape [n] (1D). The embedding-lookup
            // kernels compute byte offsets from token_id + dim against buf
            // directly and never read the shape, so the 1D vs 2D difference
            // is behaviorally identical.
            let buf = dequant_f32(gpu, quant_type, data, vocab * dim)?;
            Ok((buf, EmbeddingFormat::F32))
        }
    }
}

/// EmbeddingFormat → the DType tag for a tied lm_head WeightTensor.
/// Replaces both arches' inline matches. Q4K is not a valid tied format → panic.
pub fn embedding_format_dtype(fmt: EmbeddingFormat) -> DType {
    match fmt {
        EmbeddingFormat::HFQ4G256 => DType::HFQ4G256,
        EmbeddingFormat::HFQ4G128 => DType::HFQ4G128,
        EmbeddingFormat::Q8_0 => DType::Q8_0,
        EmbeddingFormat::F32 => DType::F32,
        EmbeddingFormat::Ggml(dtype) => dtype,
        EmbeddingFormat::Q4K => panic!("embedding_format_dtype: Q4K not valid for tied lm_head"),
    }
}

/// Load an AWQ sidecar tensor from an HFQ file.
///
/// Looks up `{stem}.awq_scale.weight` where `name` is `{stem}.weight`.
/// Returns `None` when no sidecar exists or when the sidecar has an
/// unexpected quant_type/shape.
///
/// Moved from `hipfire-arch-qwen35::qwen35::load_awq_scale_for`.
pub fn load_awq_scale_for(hfq: &HfqFile, gpu: &Gpu, name: &str, k: usize) -> Option<GpuTensor> {
    let sidecar_name = match name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.awq_scale.weight"),
        None => format!("{name}.awq_scale.weight"),
    };
    let (sc_info, sc_data) = hfq.tensor_data_pread(&sidecar_name)?;
    // Must be 1D F16, length K. quant_type 1 = F16.
    if sc_info.quant_type != 1 {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} has quant_type={} (expected 1=F16); skipping",
            sc_info.quant_type
        );
        return None;
    }
    if sc_info.shape.len() != 1 || sc_info.shape[0] as usize != k {
        eprintln!(
            "warning: AWQ sidecar {sidecar_name} shape mismatch ({:?} vs expected [{}]); skipping",
            sc_info.shape, k
        );
        return None;
    }
    // F16 → F32 on host so the kernel takes a plain `const float*`.
    let f32_data: Vec<f32> = sc_data
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    gpu.upload_raw(&f32_bytes, &[f32_bytes.len()]).ok()
}

/// Decode a little-endian f16 byte buffer to `f32`. Pure (GPU-free). The shared
/// core of every tied-lm_head reupload across HFQ + ParoQuant arches.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Reupload an f16 weight buffer as a device F32 `WeightTensor [m, k]`. The
/// canonical tied-lm_head / ParoQuant-output reupload — replaces three hand-rolled
/// `unsafe { from_raw_parts }` copies.
pub fn reupload_f16_as_f32(
    gpu: &mut Gpu,
    f16_bytes: &[u8],
    m: usize,
    k: usize,
) -> HipResult<WeightTensor> {
    let f32_data = f16_bytes_to_f32(f16_bytes);
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4) };
    let buf = gpu.upload_raw(bytes, &[m, k])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::F32,
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

/// Build a tied lm_head `WeightTensor` that ALIASES the embedding device
/// buffer (`shallow_clone` — a non-owning view). The owning weights struct
/// must record the alias (`lm_head_aliases_embd` / `tied_lm_head`) so the
/// embedding buffer is freed exactly once and the view is never freed.
/// Panics via `embedding_format_dtype` on Q4K (no tied-lm_head / GEMV weight
/// path for Q4K).
pub fn tied_lm_head_alias(
    embd: &GpuTensor,
    embd_fmt: EmbeddingFormat,
    m: usize,
    k: usize,
) -> WeightTensor {
    WeightTensor {
        buf: embd.shallow_clone(),
        gpu_dtype: embedding_format_dtype(embd_fmt),
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

/// Requantize a Q8_0 `[m, k]` weight tensor to HFQ4G256 on the host, streaming
/// one row at a time to bound host memory (a 248320x2560 lm_head never holds
/// more than one row of f32).
///
/// Dev lever (`HIPFIRE_LM_HEAD_HFQ4=1`) for tied-embedding models whose q8
/// output GEMV dominates decode bandwidth: the HFQ4G256 buffer is ~53% of the
/// q8 bytes, so `gemv_hfq4g256` replaces `gemv_q8_0` for the logits matmul.
/// The embedding table keeps the original buffer — only the output projection
/// switches. Group math matches hipfire-quantize's `quantize_hfq4g256`
/// (f32 scale = range/15, f32 min, 128 packed nibble bytes per 256-group) and
/// gguf_input's `dequant_q8_0` (34 B blocks: f16 scale + 32 i8).
pub fn requantize_weight_q8_0_to_hfq4g256(
    gpu: &Gpu,
    src: &WeightTensor,
) -> HipResult<WeightTensor> {
    let m = src.m;
    let k = src.k;
    assert!(src.gpu_dtype == DType::Q8_0, "source must be Q8_0");
    assert!(k % 256 == 0, "k must be a multiple of 256");
    let q8_row_bytes = (k / 32) * 34;
    let mut raw = vec![0u8; m * q8_row_bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &src.buf.buf)?;

    let groups_per_row = k / 256;
    let out_row_bytes = groups_per_row * 136;
    let mut out = vec![0u8; m * out_row_bytes];

    let inv15 = 1.0f32 / 15.0;
    let mut row_f32 = vec![0.0f32; k];
    for row in 0..m {
        let row_in = &raw[row * q8_row_bytes..(row + 1) * q8_row_bytes];
        for b in 0..(k / 32) {
            let off = b * 34;
            let scale =
                crate::llama::f16_to_f32(u16::from_le_bytes([row_in[off], row_in[off + 1]]));
            for j in 0..32 {
                row_f32[b * 32 + j] = row_in[off + 2 + j] as i8 as f32 * scale;
            }
        }
        for g in 0..groups_per_row {
            let group = &row_f32[g * 256..(g + 1) * 256];
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in group {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            let range = max_val - min_val;
            let scale = if range > 0.0 { range * inv15 } else { 1.0 };
            let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
            let o = row * out_row_bytes + g * 136;
            out[o..o + 4].copy_from_slice(&scale.to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&min_val.to_le_bytes());
            for i in 0..128 {
                let lo = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
                let hi = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
                out[o + 8 + i] = lo.min(15) | (hi.min(15) << 4);
            }
        }
    }

    let buf = gpu.upload_raw(&out, &[out.len()])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::HFQ4G256,
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

/// Requantize a Q8_0 `[m, k]` weight tensor to HFQ3G256 on the host, streaming
/// one row at a time (`HIPFIRE_LM_HEAD_HFQ3=1`). Same contract as
/// [`requantize_weight_q8_0_to_hfq4g256`] with the 3-bit codec from
/// hipfire-quantize's `quantize_hfq3g256` (f32 scale = range/7, f32 min,
/// 96 packed bytes per 256-group, little-endian bitstream).
pub fn requantize_weight_q8_0_to_hfq3g256(
    gpu: &Gpu,
    src: &WeightTensor,
) -> HipResult<WeightTensor> {
    let m = src.m;
    let k = src.k;
    assert!(src.gpu_dtype == DType::Q8_0, "source must be Q8_0");
    assert!(k % 256 == 0, "k must be a multiple of 256");
    let q8_row_bytes = (k / 32) * 34;
    let mut raw = vec![0u8; m * q8_row_bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &src.buf.buf)?;

    let groups_per_row = k / 256;
    let out_row_bytes = groups_per_row * 104;
    let mut out = vec![0u8; m * out_row_bytes];

    let inv7 = 1.0f32 / 7.0;
    let mut row_f32 = vec![0.0f32; k];
    for row in 0..m {
        let row_in = &raw[row * q8_row_bytes..(row + 1) * q8_row_bytes];
        for b in 0..(k / 32) {
            let off = b * 34;
            let scale =
                crate::llama::f16_to_f32(u16::from_le_bytes([row_in[off], row_in[off + 1]]));
            for j in 0..32 {
                row_f32[b * 32 + j] = row_in[off + 2 + j] as i8 as f32 * scale;
            }
        }
        for g in 0..groups_per_row {
            let group = &row_f32[g * 256..(g + 1) * 256];
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in group {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            let range = max_val - min_val;
            let scale = if range > 0.0 { range * inv7 } else { 1.0 };
            let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
            let o = row * out_row_bytes + g * 104;
            out[o..o + 4].copy_from_slice(&scale.to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&min_val.to_le_bytes());
            for chunk in 0..32 {
                let ci = chunk * 8;
                let mut q = [0u8; 8];
                for j in 0..8 {
                    let val = group[ci + j];
                    q[j] = (((val - min_val) * inv_scale + 0.5) as u8).clamp(0, 7);
                }
                let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
                let b1 =
                    ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
                let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);
                let bo = o + 8 + chunk * 3;
                out[bo] = b0;
                out[bo + 1] = b1;
                out[bo + 2] = b2;
            }
        }
    }

    let buf = gpu.upload_raw(&out, &[out.len()])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::HFQ3G256,
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

/// Requantize a Q8_0 `[m, k]` weight tensor to HFQ2G256 on the host
/// (`HIPFIRE_LM_HEAD_HFQ2=1`). Same contract as the HFQ4/HFQ3 variants;
/// 2-bit codec matches hipfire-quantize's `quantize_hfq2g256` (f32 scale =
/// range/3, f32 min, 64 packed bytes per 256-group). QUALITY WARNING: 4
/// levels per 256-group on a 248k vocab head - ship only if quality checks
/// pass.
pub fn requantize_weight_q8_0_to_hfq2g256(
    gpu: &Gpu,
    src: &WeightTensor,
) -> HipResult<WeightTensor> {
    let m = src.m;
    let k = src.k;
    assert!(src.gpu_dtype == DType::Q8_0, "source must be Q8_0");
    assert!(k % 256 == 0, "k must be a multiple of 256");
    let q8_row_bytes = (k / 32) * 34;
    let mut raw = vec![0u8; m * q8_row_bytes];
    gpu.hip.memcpy_dtoh(&mut raw, &src.buf.buf)?;

    let groups_per_row = k / 256;
    let out_row_bytes = groups_per_row * 72;
    let mut out = vec![0u8; m * out_row_bytes];

    let inv3 = 1.0f32 / 3.0;
    let mut row_f32 = vec![0.0f32; k];
    for row in 0..m {
        let row_in = &raw[row * q8_row_bytes..(row + 1) * q8_row_bytes];
        for b in 0..(k / 32) {
            let off = b * 34;
            let scale =
                crate::llama::f16_to_f32(u16::from_le_bytes([row_in[off], row_in[off + 1]]));
            for j in 0..32 {
                row_f32[b * 32 + j] = row_in[off + 2 + j] as i8 as f32 * scale;
            }
        }
        for g in 0..groups_per_row {
            let group = &row_f32[g * 256..(g + 1) * 256];
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for &v in group {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            let range = max_val - min_val;
            let scale = if range > 0.0 { range * inv3 } else { 1.0 };
            let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };
            let o = row * out_row_bytes + g * 72;
            out[o..o + 4].copy_from_slice(&scale.to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&min_val.to_le_bytes());
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 {
                    let val = group[4 * i + j];
                    let q = (((val - min_val) * inv_scale + 0.5) as u8).clamp(0, 3);
                    byte_val |= q << (j * 2);
                }
                out[o + 8 + i] = byte_val;
            }
        }
    }

    let buf = gpu.upload_raw(&out, &[out.len()])?;
    Ok(WeightTensor {
        buf,
        gpu_dtype: DType::HFQ2G256,
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

/// Resolve the output / lm_head weight, returning `(output, aliases_embd)`.
/// `aliases_embd == true` iff `output.buf` is a view of the embedding buffer
/// and must NOT be freed by the owning struct.
///
/// - `has_separate`        → `(load_separate(gpu)?, false)` — a distinct
///   `lm_head.weight` (or `output.weight`) tensor exists on disk.
/// - else `can_alias`      → `(tied_lm_head_alias(...), true)` — single device:
///   share the embedding buffer.
/// - else (multi-GPU)      → `(reupload_tied(gpu)?, false)` — output lives on a
///   different device than embed, so re-materialize it.
///
/// `gpu` is threaded into whichever closure runs (only one is called), so
/// neither closure needs to capture `&mut Gpu`.
pub fn resolve_lm_head<S, R>(
    gpu: &mut Gpu,
    has_separate: bool,
    can_alias: bool,
    embd: &GpuTensor,
    embd_fmt: EmbeddingFormat,
    m: usize,
    k: usize,
    load_separate: S,
    reupload_tied: R,
) -> HipResult<(WeightTensor, bool)>
where
    S: FnOnce(&mut Gpu) -> HipResult<WeightTensor>,
    R: FnOnce(&mut Gpu) -> HipResult<WeightTensor>,
{
    if has_separate {
        eprintln!("  loading output (separate lm_head)...");
        Ok((load_separate(gpu)?, false))
    } else if can_alias {
        eprintln!("  loading output (tied embeddings, aliased)...");
        Ok((tied_lm_head_alias(embd, embd_fmt, m, k), true))
    } else {
        eprintln!("  loading output (tied embeddings, reupload)...");
        Ok((reupload_tied(gpu)?, false))
    }
}

// ── Dequant primitives ───────────────────────────────────────────────

// ── Raw-passthrough quant codec registry ─────────────────────────────────────
//
// Single source of truth mapping the `.hfq` wire byte `quant_type` → compute
// `DType`, for formats whose load is a verbatim byte upload
// (`upload_raw(data, &[data.len()])`) + a dtype tag. Consumed by both
// `dequant_weight_raw` and `hfq::load_weight_tensor`. Layout facts (row_stride,
// K%256 guard) live on `DType`, not here. Formats that host-decode (qt 1, 2, 16)
// are deliberately absent — they stay explicit in their consumer.
//
// WEIGHT-DECODE ONLY. Embedding tables are NOT routed here (different output
// type + Q4K divergence); see embed_classify / load_embedding_llama.
//
// Adding a passthrough format = one row here, plus (if it has a non-trivial
// stride or K constraint) the matching arm in DType::row_stride /
// DType::requires_k_mod_256, plus (if the quantizer emits sidecars) one line in
// DType::supports_awq_sidecar.

/// One passthrough quant format: upload bytes verbatim, tag `dtype`.
pub(crate) struct RawCodec {
    pub quant_type: u8,
    pub dtype: DType,
}

/// The registry. Order is irrelevant (lookup is by quant_type); ascending for
/// readability. qt 1/2/16 are intentionally absent (host-decode, see consumers).
pub(crate) const RAW_CODECS: &[RawCodec] = &[
    RawCodec {
        quant_type: 0,
        dtype: DType::Q4F16G64,
    },
    RawCodec {
        quant_type: 3,
        dtype: DType::Q8_0,
    },
    RawCodec {
        quant_type: 4,
        dtype: DType::Q4K,
    },
    RawCodec {
        quant_type: 5,
        dtype: DType::Q8HFQ,
    },
    RawCodec {
        quant_type: 6,
        dtype: DType::HFQ4G256,
    },
    RawCodec {
        quant_type: 7,
        dtype: DType::HFQ4G128,
    },
    RawCodec {
        quant_type: 8,
        dtype: DType::HFQ6G256,
    },
    RawCodec {
        quant_type: 9,
        dtype: DType::HFQ2G256,
    },
    RawCodec {
        quant_type: 10,
        dtype: DType::HFQ2G128,
    },
    RawCodec {
        quant_type: 11,
        dtype: DType::HFQ3G256,
    },
    RawCodec {
        quant_type: 12,
        dtype: DType::HFQ3G128,
    },
    RawCodec {
        quant_type: 13,
        dtype: DType::MQ4G256,
    },
    RawCodec {
        quant_type: 14,
        dtype: DType::MQ8G256,
    },
    RawCodec {
        quant_type: 15,
        dtype: DType::MQ6G256,
    },
    RawCodec {
        quant_type: 17,
        dtype: DType::MQ3G256,
    },
    RawCodec {
        quant_type: 18,
        dtype: DType::MQ2G256,
    },
    RawCodec {
        quant_type: 19,
        dtype: DType::MQ2G256Lloyd,
    },
    // qt=51: unrotated MQ2-Lloyd (Maple native ternary). Same 72 B/group
    // layout as qt=19, so the same raw codec carries it.
    RawCodec {
        quant_type: 51,
        dtype: DType::MQ2G256LloydU,
    },
    RawCodec {
        quant_type: 20,
        dtype: DType::MQ3G256Lloyd,
    },
    RawCodec {
        quant_type: 21,
        dtype: DType::HFP4G32,
    },
    RawCodec {
        quant_type: 24,
        dtype: DType::MFP4G32,
    },
    RawCodec {
        quant_type: 30,
        dtype: DType::MQ4G256Lloyd,
    },
    // MQ2/MQ3-G256-GL ("global Lloyd"): 2- resp. 3-bit codes against ONE
    // tensor-global codebook plus a per-block fp16 scale, stored SoA as
    // `[M*gpr*IDX B indices][M*gpr*2 B scales]` (IDX = 64 / 96). Passthrough
    // like every other MQ*: the bytes go to the GPU verbatim and the indexed
    // MoE GEMVs decode them. The codebook is NOT in the file — the runtime
    // supplies `rdna_compute::GL_CB2` / `GL_CB3` as scalar kernel args, which
    // MUST match the quantizer's constants (see their doc comments).
    // `decode_raw_codec` enforces K%256==0 for both via
    // `DType::requires_k_mod_256` — `gpr = K/256` sets the scale-region base,
    // so a bad K silently corrupts rather than erroring.
    RawCodec {
        quant_type: 38,
        dtype: DType::MQ2G256GL,
    },
    RawCodec {
        quant_type: 39,
        dtype: DType::MQ3G256GL,
    },
    // PrismML Bonsai ternary / binary. Renumbered 38/39 -> 40/41 when master
    // claimed 38/39 for the GL codebook formats; the IDs are on-disk contract,
    // so a clash silently mis-decodes (64/96 B GL groups read as 34/18 B
    // ternary blocks) rather than erroring.
    RawCodec {
        quant_type: 40,
        dtype: DType::TQ2G128,
    },
    RawCodec {
        quant_type: 41,
        dtype: DType::BQ1G128,
    },
    RawCodec {
        quant_type: 44,
        dtype: DType::MQ4G256V2,
    },
    // qt=52: MQ4V2-Lloyd. Wire layout byte-identical to qt=44 (136 B/group),
    // so the same verbatim upload carries it; the per-tensor codebook LUTs
    // ride the WeightTensor (built by the caller from the lloyd_levels
    // sidecar) and the headers are centered at load. Length check rejects the
    // stale +32B-prefix prototype layout — re-quantize to pure 136 B/group.
    RawCodec {
        quant_type: 52,
        dtype: DType::MQ4G256V2Lloyd,
    },
    RawCodec {
        quant_type: 45,
        dtype: DType::MQ4CG256,
    },
    // MQ4G128V2 (qt=53) is a row-local format admitted by typed Qwen4
    // sealed/dense consumers. It is registered here for exact payload
    // validation and verbatim storage; generic consumers reject it explicitly.
    RawCodec {
        quant_type: 53,
        dtype: DType::MQ4G128V2,
    },
    // FP8E4M3B128 (qt=55): OCP e4m3fn codes + 128x128 bf16 s_inv, expert-blocked
    // (`rdna_compute::fp8_e4m3::b128_block_stride`). Verbatim storage; only the
    // typed gfx942 Qwen4 FP8 route reads it, generic consumers reject it.
    RawCodec {
        quant_type: 55,
        dtype: DType::FP8E4M3B128,
    },
    // Neutral-size Magnum V2 family (qt47-50): same neutral header as qt44
    // (LE `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1, `[6..8)` fp16 z1,
    // `[8..B)` legacy payload). Half 0 covers q[0..128), half 1 q[128..256);
    // `w = q*f32(s[h])+f32(z[h])`; `K%256==0`; B=200/168/104/72.
    RawCodec {
        quant_type: 47,
        dtype: DType::MQ6G256V2,
    },
    RawCodec {
        quant_type: 48,
        dtype: DType::MQ5G256V2,
    },
    RawCodec {
        quant_type: 49,
        dtype: DType::MQ3G256V2,
    },
    RawCodec {
        quant_type: 50,
        dtype: DType::MQ2G256V2,
    },
];
/// Look up the passthrough codec for `quant_type`, or `None` if it is host-decode
/// (1/2/16) or genuinely unsupported.
pub(crate) fn raw_codec(quant_type: u8) -> Option<&'static RawCodec> {
    RAW_CODECS.iter().find(|c| c.quant_type == quant_type)
}

/// Return the compute representation used for a raw HFQ weight payload.
///
/// Host-decoded source types (F16/F32/BF16) intentionally return `None`;
/// callers must widen those payloads before upload so the result matches the
/// established LLaMA loader semantics.
pub fn hfq_weight_dtype(quant_type: u8) -> Option<DType> {
    raw_codec(quant_type).map(|codec| codec.dtype)
}

/// Decode a passthrough quant format: enforce the K%256 guard (via DType),
/// upload bytes verbatim to `target`, build the `WeightTensor` with the dtype +
/// its DType-derived row_stride. `name` is the caller context for the guard
/// error. AWQ sidecars are attached by the caller (hfq), never here.
pub(crate) fn decode_raw_codec(
    gpu: &mut Gpu,
    codec: &RawCodec,
    data: &[u8],
    m: usize,
    k: usize,
    name: &str,
    target: MemoryTarget,
) -> HipResult<WeightTensor> {
    // Low-bit layout validation — centralized before any upload/host-dequant.
    // TQ2G128: 34 B per 128-elem group, BQ1G128: 18 B per 128-elem group.
    // Both require K%128==0 and exact packed length m*(k/128)*block_bytes.
    // Checked arithmetic so overflow is an actionable error, not a silent wrap.
    if let Some(block_bytes) = lowbit_block_bytes(codec.dtype) {
        validate_lowbit_layout(codec.dtype, data.len(), m, k, name, block_bytes)?;
    } else if codec.dtype.requires_k_mod_256() && k % 256 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor {name} has K={k} columns, but this format needs K to be a multiple of 256; the model file does not match its declared format — re-quantize or re-download it",
                codec.dtype
            ),
        ));
    }
    if codec.dtype == DType::HFQ4G128 {
        // Same per-row extent the weight_store manifest paths enforce: a
        // pre-`b4846285e` file packs groups across rows and decodes to NaN.
        crate::weight_store::validate_payload_extent(&[m, k], codec.dtype, data.len()).map_err(
            |reason| hip_bridge::HipError::new(0, &format!("tensor {name}: {reason}{BLOB_FIX}")),
        )?;
    }
    if codec.dtype == DType::MQ4G256V2 {
        let gpr = k / 256;
        let expected = m * gpr * 136;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ4G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ4G128V2 {
        let groups_per_row = k
            .checked_add(127)
            .and_then(|rounded| rounded.checked_div(128))
            .ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("MQ4G128V2 K rounding overflow: K={k} (caller: {name})"),
                )
            })?;
        let expected = m
            .checked_mul(groups_per_row)
            .and_then(|groups| groups.checked_mul(MQ4G128V2_GROUP_BYTES))
            .ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("MQ4G128V2 blob length overflow: M={m} K={k} (caller: {name})"),
                )
            })?;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ4G128V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::FP8E4M3B128 {
        // Generic callers describe one 2-D `[m, k]` matrix, i.e. a single expert
        // block (E=1). Expert-stacked `[E, R, K]` payloads are sized and
        // validated by the typed Qwen4 loader via `b128_tensor_bytes(E, R, K)`.
        let expected = rdna_compute::fp8_e4m3::b128_tensor_bytes(1, m, k).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!("FP8E4M3B128 blob length overflow: M={m} K={k} (caller: {name})"),
            )
        })?;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "FP8E4M3B128 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ4G256V2Lloyd {
        let gpr = k / 256;
        let expected = m * gpr * 136;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ4G256V2Lloyd blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}; artifacts from the old +32B-prefix layout are rejected){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ4CG256 {
        let gpr = k / 256;
        // Pad layout: 136 B/group (fp16 scale+zero @+0, 4 B zero pad @+4,
        // 128 B nibbles @+8). Compact 132 B groups are not a production path.
        let expected = m * gpr * MQ4C_GROUP_BYTES;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ4CG256 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ6G256V2 {
        let gpr = k / 256;
        let expected = m * gpr * MQ6G256V2_GROUP_BYTES;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ6G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ5G256V2 {
        let gpr = k / 256;
        let expected = m * gpr * MQ5G256V2_GROUP_BYTES;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ5G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ3G256V2 {
        let gpr = k / 256;
        let expected = m * gpr * MQ3G256V2_GROUP_BYTES;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ3G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    if codec.dtype == DType::MQ2G256V2 {
        let gpr = k / 256;
        let expected = m * gpr * MQ2G256V2_GROUP_BYTES;
        if data.len() != expected {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "MQ2G256V2 blob length mismatch: expected {expected}, got {} (M={m} K={k} caller: {name}){BLOB_FIX}",
                    data.len()
                ),
            ));
        }
    }
    let buf = match target {
        MemoryTarget::Device => gpu.upload_raw(data, &[data.len()])?,
        // Zero-copy CPU copy into the registered host pointer; the pad past the
        // logical bytes is zeroed so a tail overread reads deterministic zeros.
        MemoryTarget::HostMapped => gpu.upload_raw_host_mapped(data, &[data.len()])?,
    };
    Ok(WeightTensor {
        buf,
        gpu_dtype: codec.dtype,
        m,
        k,
        row_stride: codec.dtype.row_stride(k),
        paro: None,
        awq_scale: None,
        // Lloyd LUTs are attached by the caller (hfq / arch loader) from the
        // lloyd_levels sidecar — same pattern as AWQ sidecars, never here.
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    })
}

/// Block bytes for low-bit codecs, or None for other codecs.
fn lowbit_block_bytes(dtype: DType) -> Option<usize> {
    match dtype {
        DType::TQ2G128 => Some(34),
        DType::BQ1G128 => Some(18),
        _ => None,
    }
}

/// Compute expected packed byte length for TQ2G128/BQ1G128 with checked arithmetic.
/// Returns error with dtype/shape/caller context if K%128!=0 or arithmetic overflows.
fn lowbit_expected_bytes(
    dtype: DType,
    m: usize,
    k: usize,
    name: &str,
    block_bytes: usize,
) -> HipResult<usize> {
    if k % 128 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor {name} [m={m}, k={k}] has K={k} columns, but this format needs K to be a multiple of 128; the model file does not match its declared format — re-quantize or re-download it",
                dtype
            ),
        ));
    }
    let groups = k / 128;
    let bytes_per_row = groups.checked_mul(block_bytes).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor [m={m}, k={k}] (caller: {name}) byte length overflow: (k/128)*{block_bytes} overflows usize (k/128={groups})",
                dtype
            ),
        )
    })?;
    bytes_per_row.checked_mul(m).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor [m={m}, k={k}] (caller: {name}) byte length overflow: m*(k/128)*{block_bytes} overflows usize (bytes_per_row={bytes_per_row}, m={m})",
                dtype
            ),
        )
    })
}

/// Validate that `data_len` exactly matches the published low-bit layout.
/// Checked arithmetic so overflow is an error; includes dtype, shape/caller,
/// expected and actual length in the message. GPU-free, unit-testable.
fn validate_lowbit_layout(
    dtype: DType,
    data_len: usize,
    m: usize,
    k: usize,
    name: &str,
    block_bytes: usize,
) -> HipResult<()> {
    let expected = lowbit_expected_bytes(dtype, m, k, name, block_bytes)?;
    if data_len != expected {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "{:?} tensor [m={m}, k={k}] (caller: {name}) expects {expected} bytes (m*(k/128)*{block_bytes}) but got {data_len}",
                dtype
            ),
        ));
    }
    Ok(())
}

/// Upload bytes a host-decode arm produced (f16→f32, bf16→f32, or a raw F32
/// payload) under `target`. No host-mapped upload exists for the widened form,
/// so `HostMapped` fails closed ([`HOST_DECODE_REFUSAL`]) rather than silently
/// consuming the VRAM an offloaded layer frees.
pub(crate) fn upload_decoded_bytes(
    gpu: &mut Gpu,
    target: MemoryTarget,
    bytes: &[u8],
    shape: &[usize],
) -> HipResult<GpuTensor> {
    match target {
        MemoryTarget::Device => gpu.upload_raw(bytes, shape),
        MemoryTarget::HostMapped => Err(HipError::new(0, HOST_DECODE_REFUSAL)),
    }
}

/// Quant `data` → `WeightTensor [m, k]` at `target`. Moved from
/// `hipfire-arch-qwen35::qwen35::load_weight_tensor_raw` (Task 2).
///
/// The host-decode arms (qt 1/2/16) widen bytes in host memory and upload the
/// widened form; there is no host-mapped upload for that, so they fail closed
/// under [`MemoryTarget::HostMapped`] rather than silently consuming VRAM inside
/// an offloaded layer.
pub fn dequant_weight_raw(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    m: usize,
    k: usize,
    target: MemoryTarget,
) -> HipResult<WeightTensor> {
    // Host-decode formats stay explicit (NOT passthrough table rows):
    match quant_type {
        1 => {
            // F16 — keep as F16 bytes (the HFQ path host-decodes qt 1 to F32 instead;
            // this divergence is why qt 1 is not a RAW_CODECS row).
            let buf = upload_decoded_bytes(gpu, target, data, &[data.len()])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F16,
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
        2 => {
            // F32 — upload as [m, k].
            let buf = upload_decoded_bytes(gpu, target, data, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
                m,
                k,
                row_stride: 0,
                paro: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
                awq_scale: None,
            })
        }
        16 => {
            // bf16 is the high 16 bits of an f32, so widening is lossless/exact.
            let f32_data = widen_bf16(data);
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
            };
            let buf = upload_decoded_bytes(gpu, target, bytes, &[m, k])?;
            Ok(WeightTensor {
                buf,
                gpu_dtype: DType::F32,
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
        other => match raw_codec(other) {
            Some(c) => decode_raw_codec(gpu, c, data, m, k, "dequant_weight_raw", target),
            None => Err(hip_bridge::HipError::new(
                0,
                &format!("unsupported quant_type {other} in dequant_weight_raw; this hipfire build cannot read that weight format — update hipfire or re-download the model in a supported format"),
            )),
        },
    }
}

/// CPU-side dequant of an HTQ norm weight to F32 (qt 1/2/16), adding `bias`. Factored
/// out of [`dequant_norm`] so this device path and its host-located counterpart decode
/// byte-for-byte identically — a sign/normalization drift here is the "token soup"
/// attractor failure mode, so there is exactly one copy.
fn dequantize_norm(quant_type: u8, data: &[u8], shape: &[usize], bias: f32) -> Vec<f32> {
    let mut f32_data: Vec<f32> = match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => widen_bf16(data),
        _ => panic!("expected F16/F32/BF16 for norm, got qt={quant_type}"),
    };
    let expected: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        expected,
        "dequant_norm: tensor has {} elements, expected {expected} (shape {shape:?})",
        f32_data.len()
    );
    for v in &mut f32_data {
        *v += bias;
    }
    f32_data
}

/// RMSNorm scale `data` → device `GpuTensor [shape]`, adding `bias` to every element
/// (`1.0` for qwen3.5/gemma, `0.0` for qwen2/llama/minimax). Moved from `load_norm_weight`
/// (Task 2), with the `+= 1.0` generalised to `+= bias`. The CPU dequant is factored into
/// [`dequantize_norm`] so the host-located offload path stays byte-identical.
pub fn dequant_norm(
    gpu: &mut Gpu,
    quant_type: u8,
    data: &[u8],
    shape: &[usize],
    bias: f32,
) -> HipResult<GpuTensor> {
    let f32_data = dequantize_norm(quant_type, data, shape, bias);
    gpu.upload_f32(&f32_data, shape)
}

/// Raw f16/f32 `data` → device `GpuTensor [n]` (no bias). Moved from
/// `load_any_as_f32` (Task 2).
/// Inverse FWHT-256 un-rotation applied per 256-element group during dequant of
/// every FWHT-rotated format (MQ4/6, MQ3, MFP4, HFQ*-rotated, codebook 19/20/30):
/// pre-multiply by `signs2`, in-place radix-2 Hadamard butterfly, then post-scale
/// by `0.0625 * signs1` (0.0625 = 1/16 = the orthonormal 1/√256 normalization).
///
/// This is attractor-critical math — a sign/normalization error here is the
/// "token soup" failure mode. It was previously inlined byte-for-byte in 6
/// dequant arms; keep it single-source so any fix lands once. `signs1`/`signs2`
/// come from `KvCache::gen_fwht_signs(42|1042, 256)` and are generated once per
/// call site outside the group loop.
fn fwht256_inplace(group: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    debug_assert_eq!(group.len(), 256);
    for i in 0..256 {
        group[i] *= signs2[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut j = 0;
        while j < 256 {
            for k in 0..stride {
                let a = group[j + k];
                let b = group[j + k + stride];
                group[j + k] = a + b;
                group[j + k + stride] = a - b;
            }
            j += stride * 2;
        }
        stride <<= 1;
    }
    let scale_inv = 0.0625;
    for i in 0..256 {
        group[i] *= scale_inv * signs1[i];
    }
}

/// CPU-side dequant of HTQ weight bytes to F32 for the quant_types this path
/// still decodes locally (F16/F32/BF16, Q8_0, the legacy qt14 group). Every
/// block-quantized format delegates to the canonical decoder in `hipfire-cpu`
/// (`quant::dequant_group`), whose expectation tables pin those bit-for-bit — so
/// this device path and its host-located counterpart stay identical while there
/// is one decoder, not two. A sign/normalization drift here is the "token soup"
/// attractor failure mode.
fn dequantize_to_f32(quant_type: u8, data: &[u8], n: usize) -> Vec<f32> {
    match quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        16 => widen_bf16(data),
        3 => crate::llama::dequantize_q8_0(data, n),
        14 => {
            let group_size: usize = 256;
            let bytes_per_group: usize = 258;
            let n_groups = data.len() / bytes_per_group;
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * group_size);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let scale_bits = data[off] as u16 | ((data[off + 1] as u16) << 8);
                let scale = f16_to_f32(scale_bits);
                let start = out.len();
                for i in 0..256 {
                    let q = data[off + 2 + i] as i8;
                    out.push(scale * q as f32);
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        45 | 47 | 48 | 49 | 50 => {
            // V2 MQ family over the G256 container: [0..8) per-128 fp16
            // scale/zero pairs (h0 covers weights 0..127, h1 128..255), then a
            // packed code payload at offset 8. qt45 (MQ4CG256) differs only in
            // the header: one packed fp16 scale+zero dword + 4 B zero pad, so
            // both halves decode with the same (s, z) — the loop below unifies
            // that by keying `s`/`z` on the half index. Rotated formats: the
            // FWHT inverse runs per group exactly like qt44/qt13 above.
            // Code packing (same as hipfire-cpu quant.rs code2/3/5/6):
            //   qt45  4-bit low-nibble-first, 128 B
            //   qt47  6-bit, 4 codes / 3 B,    192 B
            //   qt48  5-bit, 8 codes / 5 B,    160 B
            //   qt49  3-bit, 8 codes / 3 B,     96 B
            //   qt50  2-bit LSB-first, 4/B,     64 B
            let bytes_per_group: usize = match quant_type {
                45 => 136,
                47 => 200,
                48 => 168,
                49 => 104,
                _ => 72, // qt50
            };
            let n_groups = data.len() / bytes_per_group;
            let signs1 = KvCache::gen_fwht_signs(42, 256);
            let signs2 = KvCache::gen_fwht_signs(1042, 256);
            let mut out = Vec::with_capacity(n_groups * 256);
            for g in 0..n_groups {
                let off = g * bytes_per_group;
                let h16 =
                    |o: usize| f16_to_f32(u16::from_le_bytes([data[off + o], data[off + o + 1]]));
                // qt45: [0..2) s, [2..4) z for BOTH halves. Others: per-half s/z.
                let (s0, z0) = (h16(0), h16(2));
                let (s1, z1) = if quant_type == 45 {
                    (s0, z0)
                } else {
                    (h16(4), h16(6))
                };
                let start = out.len();
                for i in 0..256 {
                    let p = off + 8;
                    let code = match quant_type {
                        45 => {
                            let b = data[p + i / 2] as u32;
                            if i % 2 == 0 {
                                b & 0xF
                            } else {
                                b >> 4
                            }
                        }
                        47 => {
                            let bo = p + (i / 4) * 3;
                            let (b0, b1, b2) =
                                (data[bo] as u32, data[bo + 1] as u32, data[bo + 2] as u32);
                            [
                                b0 & 0x3F,
                                ((b0 >> 6) | (b1 << 2)) & 0x3F,
                                ((b1 >> 4) | (b2 << 4)) & 0x3F,
                                (b2 >> 2) & 0x3F,
                            ][i % 4]
                        }
                        48 => {
                            let bo = p + (i / 8) * 5;
                            let b = [
                                data[bo] as u32,
                                data[bo + 1] as u32,
                                data[bo + 2] as u32,
                                data[bo + 3] as u32,
                                data[bo + 4] as u32,
                            ];
                            [
                                b[0] & 0x1f,
                                ((b[0] >> 5) & 0x07) | ((b[1] & 0x03) << 3),
                                (b[1] >> 2) & 0x1f,
                                ((b[1] >> 7) & 0x01) | ((b[2] & 0x0f) << 1),
                                ((b[2] >> 4) & 0x0f) | ((b[3] & 0x01) << 4),
                                (b[3] >> 1) & 0x1f,
                                ((b[3] >> 6) & 0x03) | ((b[4] & 0x07) << 2),
                                (b[4] >> 3) & 0x1f,
                            ][i % 8]
                        }
                        49 => {
                            let bo = p + (i / 8) * 3;
                            let (b0, b1, b2) =
                                (data[bo] as u32, data[bo + 1] as u32, data[bo + 2] as u32);
                            [
                                b0 & 7,
                                (b0 >> 3) & 7,
                                ((b0 >> 6) | (b1 << 2)) & 7,
                                (b1 >> 1) & 7,
                                (b1 >> 4) & 7,
                                ((b1 >> 7) | (b2 << 1)) & 7,
                                (b2 >> 2) & 7,
                                (b2 >> 5) & 7,
                            ][i % 8]
                        }
                        _ => (data[p + i / 4] as u32 >> (2 * (i % 4))) & 0x3, // qt50
                    };
                    let (s, z) = if i < 128 { (s0, z0) } else { (s1, z1) };
                    out.push(s * code as f32 + z);
                }
                let group = &mut out[start..start + 256];
                fwht256_inplace(group, &signs1, &signs2);
            }
            out
        }
        6 | 7 | 8 | 11 | 12 | 13 | 15 | 17 | 18 | 19 | 20 | 30 | 40 | 41 | 44 => {
            // Block-quantized formats decode through the canonical CPU decoder in
            // `hipfire-cpu` (`dequant_group`). Its expectation tables pin each of
            // these bit-for-bit to the arithmetic this arm used to hold, so the
            // delegation is behaviour-preserving — and there is now one decoder
            // instead of two. Element formats stay local above: their "group" is a
            // single element, so the block decoder's fixed chunking does not apply
            // to sub-block norms/biases.
            let q = hipfire_cpu::quant::CpuQuant::from_quant_type(quant_type)
                .expect("routed quant_type has a CpuQuant");
            let (group_bytes, group_elems) = (q.group_bytes(), q.group_elems());
            let mut out = vec![0.0f32; (data.len() / group_bytes) * group_elems];
            for (g, chunk) in data.chunks_exact(group_bytes).enumerate() {
                hipfire_cpu::quant::dequant_group(
                    q,
                    chunk,
                    &mut out[g * group_elems..(g + 1) * group_elems],
                );
            }
            out
        }
        53 => panic!("MQ4G128V2 (qt=53) is typed-Qwen4-only; generic dequant_f32 refuses it"),
        55 => panic!("FP8E4M3B128 (qt=55) is typed-Qwen4-gfx942-only; generic dequant_f32 refuses it"),
        56 => panic!("FP8E4M3PT (qt=56) is read by the Qwen4 PLE row store only; generic dequant_f32 refuses it"),
        131..=134 => panic!(
            "MQ-I4 (qt={quant_type}) is a typed-Qwen4 expert-slab format; generic dequant_f32 refuses it"
        ),
        _ => panic!("unsupported quant_type {quant_type} for dequant_f32"),
    }
}

/// Dequantize an HTQ weight tensor to a device `F32 [n]` tensor. The CPU dequant is
/// factored into [`dequantize_to_f32`] so the host-located offload path stays byte for
/// byte identical — the only difference from the device path is the upload target
/// (device memory vs host-mapped system RAM readable over PCIe).
pub fn dequant_f32(gpu: &mut Gpu, quant_type: u8, data: &[u8], n: usize) -> HipResult<GpuTensor> {
    let f32_data = dequantize_to_f32(quant_type, data, n);
    gpu.upload_f32(&f32_data[..n], &[n])
}

/// Public delegation to the canonical per-tensor CPU decoder [`dequantize_to_f32`].
///
/// The decoder itself is deliberately private (it is an implementation detail
/// of the weight-loading path); this wrapper exists for the two places that
/// need to hold a *second* decoder to the same bytes:
///
/// * `crates/hipfire-runtime/tests/cpu_quant_cross_check.rs`, which asserts
///   `hipfire_cpu::quant::dequant_group` reproduces this arithmetic bit-for-bit
///   over the real tensors of the on-disk fixtures, and
/// * whatever generated the literal expectation tables in that crate.
///
/// Integrating a second decoder is the real risk in the CPU-offload path, so
/// the check is a test rather than a comment. See [`dequantize_to_f32`] for the
/// per-quant byte layouts.
pub fn dequantize_weight_to_f32(quant_type: u8, data: &[u8], n: usize) -> Vec<f32> {
    dequantize_to_f32(quant_type, data, n)
}

/// Whether [`dequantize_to_f32`] implements this HFQ quant type.
///
/// Host-dequant loaders (embed encoders, draft/MTP heads, the host-offload
/// path) should screen an artifact with this instead of discovering a gap as
/// the decoder's `panic!` mid-load — that panic takes the daemon and any
/// resident model down with it. Formats outside the dense HFQ set (Q4K qt=4 is
/// GGUF/Maple-side, qt=53 is typed-Qwen4-only) are deliberately absent.
///
/// `dequant_f32_supported_matches_the_decoder` sweeps every `u8` and ties this
/// set to the decoder's actual behaviour, so a new arm without a matching entry
/// (or the reverse) fails the test rather than a user's load.
pub fn dequant_f32_supported(quant_type: u8) -> bool {
    matches!(
        quant_type,
        1 | 2
            | 3
            | 6
            | 7
            | 8
            | 11
            | 12
            | 13
            | 14
            | 15
            | 16
            | 17
            | 18
            | 19
            | 20
            | 30
            | 40
            | 41
            | 44
            | 45
            | 47
            | 48
            | 49
            | 50
    )
}

// ── WeightBackend trait ─────────────────────────────────────────────────────

/// Where a weight's bytes physically land at load. One parameter, one upload
/// path: `Device` puts them in VRAM, `HostMapped` in pinned system RAM that the
/// *same* kernels dereference over PCIe through the device-visible alias
/// (`hipHostGetDevicePointer`). Contents are byte-identical either way, so
/// placement changes only where the bytes live, never the arithmetic.
///
/// This is the load-side placement of a weight's storage. It is deliberately not
/// the execution question ("who multiplies these bytes") — that is the caller's
/// `memory.offload_exec` policy and, from the loader's point of view, is recorded
/// on the loaded weight (stage 2), not decided here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTarget {
    /// VRAM (`hip.malloc` / pool).
    Device,
    /// Pinned system RAM (`hipHostMalloc(hipHostMallocMapped)`), padded by
    /// `rdna_compute::HOST_TAIL_PAD_BYTES`. Costs no device heap; a `Gpu` kernel
    /// read traverses PCIe.
    HostMapped,
}

/// Refusal for the host-decode arms (qt 1/2/16, and every f32 fallback) under
/// [`MemoryTarget::HostMapped`]. Those arms widen to f32 in host memory and then
/// upload the widened bytes; no host-mapped upload exists, so silently placing
/// them on the device would consume exactly the VRAM an offloaded layer frees.
pub const HOST_DECODE_REFUSAL: &str =
    "this quantized weight format has no host-memory (offload) upload path, and placing it \
     in device memory would defeat the offload; use a model quantized in a supported format, \
     or unset memory.gpu_layer_budget / memory.moe_expert_budget";

use crate::augmentor::{try_augmentors, DEFAULT_AUGMENTORS};
use crate::model_source::ModelSource;
use crate::paro::{load_fp16_weight_from_source, paro_load_f32, paro_load_norm};
pub use crate::weight_manifest::WeightResidency;

/// Pluggable weight-loading backend. `rel` is a layer-relative path: for `proj`
/// it carries NO file extension (the backend appends `.weight` / tries `.qweight`);
/// for `norm`/`raw_f32` it carries the on-disk suffix (e.g. `input_layernorm.weight`).
/// Set the active layer with `set_layer` before each layer's calls.
pub trait WeightBackend {
    fn set_layer(&mut self, layer: usize);
    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor>;
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor>;
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
    /// Load a bias vector (f32). Only qwen2 attention biases use this today.
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor>;
    /// Return an already allocated tensor to this backend's GPU pool.
    ///
    /// Layer loading uses this narrow seam to roll back staged owners without
    /// exposing the backend's device handle to arch crates.
    fn free_tensor(&mut self, tensor: GpuTensor);
}

/// HFQ backend. `norm_bias`: `1.0` (qwen3.5/gemma) or `0.0` (qwen2/llama).
/// `candidates`: layout resolver (`hf_name_candidates` or `flat_name_candidates`).
/// `read`: the arch's pread+awq weight reader (see `HfqRead`).
pub struct HfqBackend<'a> {
    pub hfq: &'a HfqFile,
    pub gpu: &'a mut Gpu,
    pub norm_bias: f32,
    pub candidates: fn(&str) -> Vec<String>,
    pub read_proj: fn(
        &HfqFile,
        &mut Gpu,
        &str,
        usize,
        usize,
        fn(&str) -> Vec<String>,
        MemoryTarget,
    ) -> HipResult<WeightTensor>,
    pub layer: usize,
    /// Where this layer's own weight tensors live. `HostMapped` allocates them in
    /// pinned system RAM the kernels read over PCIe instead of device memory,
    /// leaving VRAM free for a larger KV cache while keeping numerics
    /// byte-identical to resident mode. Set per layer by the loader from the
    /// resolved placement; `Resident` is the fully-resident, zero-diff default.
    ///
    /// It selects the [`MemoryTarget`] passed to [`Self::read_proj`] for projections
    /// and the host uploader for `norm`/`raw_f32`/`bias`. One upload path: an arch
    /// whose reader cannot honour `HostMapped` refuses inside its own decode (e.g.
    /// the f32-dequant fallback, [`HOST_DECODE_REFUSAL`]) rather than landing the
    /// weights in the VRAM the offload exists to free.
    pub residency: WeightResidency,
}

impl<'a> WeightBackend for HfqBackend<'a> {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let name = hfq_proj_name(self.layer, rel);
        let target = crate::offload::memory_target(self.residency);
        (self.read_proj)(self.hfq, self.gpu, &name, m, k, self.candidates, target)
    }
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let f32_data = dequantize_norm(info.quant_type, &data, shape, self.norm_bias);
        if self.residency == WeightResidency::HostMapped {
            self.gpu.upload_f32_host(&f32_data, shape)
        } else {
            self.gpu.upload_f32(&f32_data, shape)
        }
    }
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let f32_data = dequantize_to_f32(info.quant_type, &data, n);
        if self.residency == WeightResidency::HostMapped {
            self.gpu.upload_f32_host(&f32_data[..n], &[n])
        } else {
            self.gpu.upload_f32(&f32_data[..n], &[n])
        }
    }
    fn bias(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        let name = hfq_plain_name(self.layer, rel);
        let (info, data) = read_first(self.hfq, &name, self.candidates)
            .unwrap_or_else(|| panic!("tensor not found: {name}"));
        let f32_data = dequantize_to_f32(info.quant_type, &data, n);
        let t = if self.residency == WeightResidency::HostMapped {
            self.gpu.upload_f32_host(&f32_data[..n], &[n])?
        } else {
            self.gpu.upload_f32(&f32_data[..n], &[n])?
        };
        assert_eq!(
            t.numel(),
            n,
            "bias {name} has {} elements, expected {n}",
            t.numel()
        );
        Ok(t)
    }
    fn free_tensor(&mut self, tensor: GpuTensor) {
        let _ = self.gpu.free_tensor(tensor);
    }
}

/// Resolve `name` via `candidates` and return the first tensor's `(info, bytes)`.
pub fn read_first(
    hfq: &HfqFile,
    name: &str,
    candidates: fn(&str) -> Vec<String>,
) -> Option<(crate::hfq::HfqTensorInfo, Vec<u8>)> {
    for c in candidates(name) {
        if let Some((info, buf)) = hfq.tensor_data_vec(&c) {
            return Some((info.clone(), buf));
        }
    }
    None
}

/// PaRo backend (augmentor chain + paro primitives) — fully arch-agnostic.
/// `mp` is the text-tower prefix from `paro_text_prefix`.
pub struct ParoBackend<'a> {
    pub source: &'a dyn ModelSource,
    pub gpu: &'a mut Gpu,
    pub mp: &'static str,
    pub layer: usize,
    /// `1.0` (qwen3.5/gemma) or `0.0` (qwen2/llama).
    pub norm_bias: f32,
}

impl<'a> WeightBackend for ParoBackend<'a> {
    fn set_layer(&mut self, layer: usize) {
        self.layer = layer;
    }

    fn proj(&mut self, rel: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
        let base = paro_proj_name(self.mp, self.layer, rel);
        match try_augmentors(self.source, &base, m, k, self.gpu, DEFAULT_AUGMENTORS)? {
            Some(t) => Ok(t),
            None => {
                load_fp16_weight_from_source(self.source, self.gpu, &format!("{base}.weight"), m, k)
            }
        }
    }
    fn norm(&mut self, rel: &str, shape: &[usize]) -> HipResult<GpuTensor> {
        paro_load_norm(
            self.source,
            self.gpu,
            &paro_plain_name(self.layer, rel),
            shape,
            self.norm_bias,
        )
    }
    fn raw_f32(&mut self, rel: &str, n: usize) -> HipResult<GpuTensor> {
        paro_load_f32(self.source, self.gpu, &paro_plain_name(self.layer, rel), n)
    }
    fn free_tensor(&mut self, tensor: GpuTensor) {
        let _ = self.gpu.free_tensor(tensor);
    }
    fn bias(&mut self, _rel: &str, _n: usize) -> HipResult<GpuTensor> {
        Err(hip_bridge::HipError::new(
            0,
            "ParoBackend: attention biases unsupported",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fwht256_inplace` must be bit-identical to the per-arm inlined FWHT it
    /// replaced (the version that shipped in 6 dequant arms). Pin it: run a
    /// verbatim copy of the old inline sequence and the helper on the same
    /// pseudo-random group + signs, assert exact f32 equality. A drift here is
    /// the attractor / "token soup" failure mode, so equality must be exact.
    #[test]
    fn fwht256_inplace_matches_inlined_reference() {
        // Deterministic pseudo-random inputs (no rand dep).
        let mut x: u32 = 0x1234_5678;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let group_init: Vec<f32> = (0..256).map(|_| next()).collect();
        // signs are ±1 in practice; mirror that.
        let signs1: Vec<f32> = (0..256)
            .map(|_| if next() < 0.0 { -1.0 } else { 1.0 })
            .collect();
        let signs2: Vec<f32> = (0..256)
            .map(|_| if next() < 0.0 { -1.0 } else { 1.0 })
            .collect();

        // Reference: verbatim old inline sequence.
        let mut reference = group_init.clone();
        {
            let group = &mut reference[..];
            for i in 0..256 {
                group[i] *= signs2[i];
            }
            let mut stride = 1;
            while stride < 256 {
                let mut j = 0;
                while j < 256 {
                    for k in 0..stride {
                        let a = group[j + k];
                        let b = group[j + k + stride];
                        group[j + k] = a + b;
                        group[j + k + stride] = a - b;
                    }
                    j += stride * 2;
                }
                stride <<= 1;
            }
            let scale_inv = 0.0625;
            for i in 0..256 {
                group[i] *= scale_inv * signs1[i];
            }
        }

        let mut got = group_init.clone();
        fwht256_inplace(&mut got, &signs1, &signs2);

        assert_eq!(got.len(), 256);
        for i in 0..256 {
            assert_eq!(
                got[i].to_bits(),
                reference[i].to_bits(),
                "fwht256_inplace diverged from inlined reference at index {i}"
            );
        }
    }

    #[test]
    fn nested_candidates_cover_both_layouts() {
        let c = hf_name_candidates("layers.0.self_attn.q_proj.weight");
        assert_eq!(
            c[0],
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(c[1], "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(c[2], "layers.0.self_attn.q_proj.weight");
    }
    #[test]
    fn lm_head_special_case() {
        let c = hf_name_candidates("lm_head.weight");
        assert_eq!(c[0], "lm_head.weight");
        assert!(c.contains(&"model.language_model.lm_head.weight".to_string()));
    }
    #[test]
    fn flat_candidates_are_two() {
        assert_eq!(
            flat_name_candidates("layers.0.mlp.down_proj.weight"),
            vec![
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "layers.0.mlp.down_proj.weight".to_string()
            ]
        );
    }
    #[test]
    fn name_builders() {
        assert_eq!(
            hfq_proj_name(3, "self_attn.q_proj"),
            "layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            hfq_plain_name(3, "input_layernorm.weight"),
            "layers.3.input_layernorm.weight"
        );
        assert_eq!(
            paro_proj_name("model.language_model", 0, "linear_attn.in_proj_qkv"),
            "model.language_model.layers.0.linear_attn.in_proj_qkv"
        );
        assert_eq!(
            paro_plain_name(0, "input_layernorm.weight"),
            "layers.0.input_layernorm.weight"
        );
    }

    // ── Embedding / tied-lm_head tests ──────────────────────────────────────

    #[test]
    fn embed_classify_raw_hfq4g256() {
        match embed_classify(6).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::HFQ4G256) => {}
            other => panic!("expected Raw(HFQ4G256), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_raw_hfq4g128() {
        match embed_classify(7).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::HFQ4G128) => {}
            other => panic!("expected Raw(HFQ4G128), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_raw_q8_0() {
        match embed_classify(3).unwrap() {
            EmbedPlan::Raw(EmbeddingFormat::Q8_0) => {}
            other => panic!("expected Raw(Q8_0), got {other:?}"),
        }
    }
    #[test]
    fn embed_classify_host_f32() {
        for qt in [1, 2, 16, 40, 41] {
            match embed_classify(qt).unwrap() {
                EmbedPlan::HostF32 => {}
                other => panic!("qt={qt}: expected HostF32, got {other:?}"),
            }
        }
    }
    /// quant_type 40 (TQ2G128) → `EmbedPlan::HostF32`, so
    /// `token_embd` routes through the existing host-decode-to-F32 embedding
    /// path instead of tripping the "unsupported embedding quant_type"
    /// panic seen in Task 16's diagnosis run.
    #[test]
    fn embed_classify_tq2g128_is_host_f32() {
        match embed_classify(40).unwrap() {
            EmbedPlan::HostF32 => {}
            other => panic!("qt=40: expected HostF32, got {other:?}"),
        }
    }
    /// quant_type 41 (BQ1G128) → `EmbedPlan::HostF32`,
    /// mirroring the qt=40 TQ2G128 arm above.
    #[test]
    fn embed_classify_bq1g128_is_host_f32() {
        match embed_classify(41).unwrap() {
            EmbedPlan::HostF32 => {}
            other => panic!("qt=41: expected HostF32, got {other:?}"),
        }
    }
    /// Task 15b RED→GREEN gate: `dequant_tq2_to_f32` on a single 34-byte
    /// Q2_0 block, `d=2.0` (FP16 bytes `[0x00, 0x40]`), `qs[0]=0xE4` (codes
    /// 0,1,2,3 LSB-first) and the rest of `qs` zeroed (code 0 everywhere).
    /// `value = (code-1)*d` so: code0→-2.0, code1→0.0, code2→2.0, code3→4.0,
    /// then 124 more code-0 elements at -2.0. Mirrors the proven Task-5/
    /// Task-8v oracle for `dequant_tq2g128_to_f16`.
    #[test]
    fn dequant_tq2_to_f32_single_block() {
        let mut data = [0u8; 34];
        data[0] = 0x00;
        data[1] = 0x40; // FP16 2.0
        data[2] = 0xE4; // codes [0,1,2,3] LSB-first (0b11_10_01_00)
                        // data[3..34] already zero => codes 0 for elements 4..127
        let out = dequantize_to_f32(40, &data, 128);
        assert_eq!(out.len(), 128);
        assert_eq!(&out[0..4], &[-2.0, 0.0, 2.0, 4.0]);
        for (i, &v) in out.iter().enumerate().skip(4) {
            assert_eq!(v, -2.0, "expected tail code-0 => -d at index {i}");
        }
    }
    /// SP-B final-review cleanup: `dequant_bq1_to_f32` had no dedicated unit
    /// test (the bug it once had was missed by every per-task review). Single
    /// 18-byte Q1_0 block, `d=0.5` (FP16 bytes `[0x00, 0x38]`), all 16 `qs`
    /// bytes `0xFF` (every sign bit set) => all 128 elements decode to `+d`.
    /// Then clearing bit 0 of `qs[0]` flips element 0 to `-d` while element 1
    /// stays `+d`. Mirrors the Task-9 device-parity oracle and the already-
    /// passing `dequant_q1_0_sign_only` test in `gguf_input.rs`.
    #[test]
    fn dequant_bq1_to_f32_single_block() {
        let mut data = [0u8; 18];
        data[0] = 0x00;
        data[1] = 0x38; // FP16 0.5
        for b in data[2..18].iter_mut() {
            *b = 0xFF; // all 128 sign bits set => all +d
        }
        let out = dequantize_to_f32(41, &data, 128);
        assert_eq!(out.len(), 128);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-3, "expected +d at index {i}, got {v}");
        }

        data[2] &= !1; // clear bit 0 of qs[0] => element 0 flips to -d
        let out = dequantize_to_f32(41, &data, 128);
        assert!(
            (out[0] - (-0.5)).abs() < 1e-3,
            "expected -d at index 0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.5).abs() < 1e-3,
            "expected +d at index 1, got {}",
            out[1]
        );
    }
    #[test]
    fn embed_classify_errors_on_unknown() {
        let err = embed_classify(99).unwrap_err();
        assert!(err.message.contains("unsupported embedding quant_type"));
    }
    #[test]
    fn embedding_format_dtype_mapping() {
        assert_eq!(
            embedding_format_dtype(EmbeddingFormat::HFQ4G256),
            DType::HFQ4G256
        );
        assert_eq!(
            embedding_format_dtype(EmbeddingFormat::HFQ4G128),
            DType::HFQ4G128
        );
        assert_eq!(embedding_format_dtype(EmbeddingFormat::Q8_0), DType::Q8_0);
        assert_eq!(embedding_format_dtype(EmbeddingFormat::F32), DType::F32);
    }
    #[test]
    #[should_panic(expected = "Q4K not valid")]
    fn embedding_format_dtype_q4k_panics() {
        embedding_format_dtype(EmbeddingFormat::Q4K);
    }

    #[test]
    fn f16_bytes_to_f32_roundtrips_known_values() {
        // 1.0 = 0x3C00, 2.0 = 0x4000, -1.0 = 0xBC00 (little-endian byte pairs).
        let bytes = [0x00, 0x3C, 0x00, 0x40, 0x00, 0xBC];
        assert_eq!(f16_bytes_to_f32(&bytes), vec![1.0, 2.0, -1.0]);
    }

    #[test]
    fn tied_alias_assembles_f32_view() {
        let embd = GpuTensor::null_for_test();
        let wt = tied_lm_head_alias(&embd, EmbeddingFormat::F32, 100, 64);
        assert_eq!(wt.gpu_dtype, DType::F32);
        assert_eq!(wt.m, 100);
        assert_eq!(wt.k, 64);
        assert_eq!(wt.row_stride, 0);
        assert!(wt.paro.is_none());
        assert!(wt.awq_scale.is_none());
    }

    #[test]
    fn tied_alias_maps_hfq4g256_dtype() {
        let embd = GpuTensor::null_for_test();
        let wt = tied_lm_head_alias(&embd, EmbeddingFormat::HFQ4G256, 8, 8);
        assert_eq!(wt.gpu_dtype, DType::HFQ4G256);
    }

    #[test]
    #[should_panic(expected = "Q4K not valid for tied lm_head")]
    fn tied_alias_q4k_panics() {
        let embd = GpuTensor::null_for_test();
        let _ = tied_lm_head_alias(&embd, EmbeddingFormat::Q4K, 8, 8);
    }

    /// Pins every RAW_CODECS row against the dtype the *production* arms produced,
    /// transcribed with source citations so the oracle is independent of the table.
    /// A drift here mis-tags a quant format → "token soup"; equality must be exact.
    #[test]
    fn raw_codecs_golden_against_production_arms() {
        // (quant_type, dtype) — RHS copied from the pre-refactor arms:
        //   wb = weight_backend.rs (dequant_weight_raw), hfq = hfq.rs (load_weight_tensor)
        let expected: &[(u8, DType)] = &[
            (0, DType::Q4F16G64),       // hfq:712
            (3, DType::Q8_0),           // wb:487 / hfq:725
            (4, DType::Q4K),            // hfq:738
            (5, DType::Q8HFQ),          // hfq:754
            (6, DType::HFQ4G256),       // wb:299 / hfq:767
            (7, DType::HFQ4G128),       // wb:311 / hfq:780
            (8, DType::HFQ6G256),       // wb:323 / hfq:793
            (9, DType::HFQ2G256),       // hfq:807
            (10, DType::HFQ2G128),      // hfq:819
            (11, DType::HFQ3G256),      // wb:335 / hfq:832
            (12, DType::HFQ3G128),      // wb:347 / hfq:845
            (13, DType::MQ4G256),       // wb:359 / hfq:858
            (14, DType::MQ8G256),       // wb:371 / hfq:871
            (15, DType::MQ6G256),       // wb:383
            (17, DType::MQ3G256),       // wb:395 / hfq:884
            (18, DType::MQ2G256),       // wb:407 / hfq:897
            (19, DType::MQ2G256Lloyd),  // wb:419 / hfq:910
            (51, DType::MQ2G256LloydU), // unrotated sibling of 19
            (20, DType::MQ3G256Lloyd),  // wb:431 / hfq:923
            (21, DType::HFP4G32),       // wb:459 / hfq:944
            (24, DType::MFP4G32),       // wb:475 / hfq:963
            (30, DType::MQ4G256Lloyd),  // wb:443 / hfq:978 (renumbered from 21; do not swap)
            // GL ("global Lloyd") codebook formats — MoE-routed-expert only.
            // RHS pinned against hipfire-quantize `QuantType::MQ2G256GL = 38` /
            // `MQ3G256GL = 39`; a swap here mis-decodes 64 B/group indices as
            // 96 B/group (or vice versa) → token soup, not a crash.
            (38, DType::MQ2G256GL),
            (39, DType::MQ3G256GL),
            // Bonsai ternary/binary — renumbered off 38/39 (taken by GL above).
            (40, DType::TQ2G128), // ternary Bonsai-27B, 34 B/group-128
            (41, DType::BQ1G128), // binary Bonsai-27B, 18 B/group-128
            // qt=44/45: 136 B/group pad layouts (PR599). MQ4C is NOT 132.
            (44, DType::MQ4G256V2),
            (45, DType::MQ4CG256),
            // qt=52: MQ4V2 container + per-tensor Lloyd codebook sidecar (136 B/G256).
            (52, DType::MQ4G256V2Lloyd),
            // qt=53 is row-local MQ4G128V2 (68 B per ceil(K/128) group), admitted
            // only by typed Qwen4 sealed/dense consumers; generic paths refuse it.
            (53, DType::MQ4G128V2),
            // qt=55 FP8E4M3B128: expert-blocked FP8 payload, typed gfx942 route only.
            (55, DType::FP8E4M3B128),
            // Neutral-size Magnum V2 family (qt47-50): preserve qtype distinction;
            // do not alias to legacy MQ2/3/5/6. Each maps one-to-one to its V2 DType.
            (47, DType::MQ6G256V2), // 200 B/G256 6.25bpw
            (48, DType::MQ5G256V2), // 168 B/G256 5.25bpw
            (49, DType::MQ3G256V2), // 104 B/G256 3.25bpw
            (50, DType::MQ2G256V2), // 72 B/G256 2.25bpw
        ];
        for &(qt, dt) in expected {
            let c = raw_codec(qt).unwrap_or_else(|| panic!("no RAW_CODECS row for qt={qt}"));
            assert_eq!(c.dtype, dt, "qt={qt} dtype");
        }
        assert_eq!(
            RAW_CODECS.len(),
            expected.len(),
            "RAW_CODECS has unlisted rows"
        );
        // Host-decode formats must NOT be in the passthrough table:
        for qt in [1u8, 2, 16] {
            assert!(
                raw_codec(qt).is_none(),
                "qt={qt} is host-decode, must not be a raw codec"
            );
        }
    }

    /// No two rows may claim the same quant_type (find() is first-match-wins).
    #[test]
    fn raw_codecs_unique_quant_types() {
        for (i, a) in RAW_CODECS.iter().enumerate() {
            for b in &RAW_CODECS[i + 1..] {
                assert_ne!(
                    a.quant_type, b.quant_type,
                    "duplicate quant_type {}",
                    a.quant_type
                );
            }
        }
    }

    /// quant_type 40 (ternary Bonsai-27B TQ2G128) must resolve to
    /// DType::TQ2G128 via the RAW_CODECS loader table.
    #[test]
    fn tq2g128_quant_type_40_maps_to_tq2g128() {
        let codec = raw_codec(40).expect("quant_type 40 registered");
        assert_eq!(codec.dtype, DType::TQ2G128);
    }
    /// quant_type 41 (binary Bonsai-27B BQ1G128)
    /// must resolve to DType::BQ1G128 via the RAW_CODECS loader table.
    #[test]
    fn bq1g128_quant_type_41_maps_to_bq1g128() {
        let c = raw_codec(41).expect("no RAW_CODECS row for qt=41");
        assert_eq!(c.dtype, DType::BQ1G128);
    }

    // ── Low-bit layout-validation contract (GPU-free) ─────────────────────────
    // TQ2G128: 34 B per 128, BQ1G128: 18 B per 128, K%128==0, exact byte length
    // m*(k/128)*block_bytes with checked arithmetic. Centralized in
    // decode_raw_codec via validate_lowbit_layout / lowbit_expected_bytes.

    #[test]
    fn lowbit_block_bytes_mapping() {
        assert_eq!(lowbit_block_bytes(DType::TQ2G128), Some(34));
        assert_eq!(lowbit_block_bytes(DType::BQ1G128), Some(18));
        assert_eq!(lowbit_block_bytes(DType::HFQ4G256), None);
        assert_eq!(lowbit_block_bytes(DType::Q8_0), None);
        assert_eq!(lowbit_block_bytes(DType::HFP4G32), None);
    }

    #[test]
    fn lowbit_block_bytes_none_for_other_dtypes() {
        for dt in [
            DType::Q4K,
            DType::Q8HFQ,
            DType::MQ4G256,
            DType::MQ2G256GL,
            DType::F32,
            DType::F16,
        ] {
            assert_eq!(lowbit_block_bytes(dt), None, "{dt:?} must not be low-bit");
        }
    }

    #[test]
    fn lowbit_expected_bytes_tq2g128_valid_layouts() {
        // Single-row, single-group: m=1,k=128 => 1*1*34 =34
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 1, 128, "test", 34).unwrap(),
            34
        );
        // m=32,k=128 => 32*1*34=1088
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 32, 128, "test", 34).unwrap(),
            32 * 34
        );
        // m=1,k=256 => 1*2*34=68
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 1, 256, "test", 34).unwrap(),
            68
        );
        // Real Bonsai shape: m=8192,k=4096 => 8192*(4096/128)*34 =8192*32*34=8912896
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 8192, 4096, "test", 34).unwrap(),
            8192 * 32 * 34
        );
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 8192, 4096, "test", 34).unwrap(),
            8912896
        );
    }

    #[test]
    fn lowbit_expected_bytes_bq1g128_valid_layouts() {
        // m=1,k=128 => 18
        assert_eq!(
            lowbit_expected_bytes(DType::BQ1G128, 1, 128, "test", 18).unwrap(),
            18
        );
        // m=32,k=128 => 576
        assert_eq!(
            lowbit_expected_bytes(DType::BQ1G128, 32, 128, "test", 18).unwrap(),
            576
        );
        // m=8192,k=4096 => 8192*32*18=4718592
        assert_eq!(
            lowbit_expected_bytes(DType::BQ1G128, 8192, 4096, "test", 18).unwrap(),
            4718592
        );
        // m=4096,k=512 => 4096*4*18=294912
        assert_eq!(
            lowbit_expected_bytes(DType::BQ1G128, 4096, 512, "test", 18).unwrap(),
            4096 * 4 * 18
        );
    }

    #[test]
    fn lowbit_expected_bytes_rejects_k_not_divisible() {
        for k in [1, 127, 129, 256 - 1, 255, 257, 1000] {
            let err = lowbit_expected_bytes(DType::TQ2G128, 1, k, "caller_ctx", 34).unwrap_err();
            assert!(
                err.message.contains("multiple of 128"),
                "k={k}: expected K%128 error, got {}",
                err.message
            );
            assert!(
                err.message.contains("TQ2G128"),
                "must include dtype: {}",
                err.message
            );
            assert!(
                err.message.contains("caller_ctx"),
                "must include caller: {}",
                err.message
            );
            assert!(
                err.message.contains(&format!("k={k}")) || err.message.contains(&format!("K={k}")),
                "must include shape K: {}",
                err.message
            );
        }
        // BQ1G128 same guard
        let err = lowbit_expected_bytes(DType::BQ1G128, 4, 200, "my_layer", 18).unwrap_err();
        assert!(err.message.contains("BQ1G128"));
        assert!(err.message.contains("my_layer"));
        assert!(err.message.contains("multiple of 128"));
    }

    #[test]
    fn validate_lowbit_layout_accepts_exact() {
        // TQ2G128 m=2,k=128 => 68 bytes
        validate_lowbit_layout(DType::TQ2G128, 68, 2, 128, "accept", 34).unwrap();
        // BQ1G128 m=2,k=128 => 36 bytes
        validate_lowbit_layout(DType::BQ1G128, 36, 2, 128, "accept", 18).unwrap();
        // m=0 => 0 bytes expected (degenerate but valid)
        validate_lowbit_layout(DType::TQ2G128, 0, 0, 128, "zero_m", 34).unwrap();
        validate_lowbit_layout(DType::BQ1G128, 0, 0, 256, "zero_m", 18).unwrap();
    }

    #[test]
    fn validate_lowbit_layout_rejects_short_and_long() {
        // TQ2G128 m=1,k=128 expects 34, give 33 and 35
        let exp = 34;
        for bad in [exp - 1, exp + 1, exp + 10, 0] {
            let err =
                validate_lowbit_layout(DType::TQ2G128, bad, 1, 128, "my_caller", 34).unwrap_err();
            assert!(
                err.message.contains("TQ2G128"),
                "dtype in msg: {}",
                err.message
            );
            assert!(
                err.message.contains("m=1"),
                "shape m in msg: {}",
                err.message
            );
            assert!(
                err.message.contains("k=128"),
                "shape k in msg: {}",
                err.message
            );
            assert!(
                err.message.contains("my_caller"),
                "caller in msg: {}",
                err.message
            );
            assert!(
                err.message.contains(&exp.to_string()),
                "expected in msg: {}",
                err.message
            );
            assert!(
                err.message.contains(&bad.to_string()),
                "actual in msg: {}",
                err.message
            );
            assert!(
                err.message.contains("expects"),
                "expects phrase: {}",
                err.message
            );
            assert!(
                err.message.contains("but got"),
                "but got phrase: {}",
                err.message
            );
        }
        // BQ1G128 m=4,k=256 => 4*2*18=144, test short
        let err = validate_lowbit_layout(DType::BQ1G128, 100, 4, 256, "bq_caller", 18).unwrap_err();
        assert!(err.message.contains("BQ1G128"));
        assert!(err.message.contains("expects 144"));
        assert!(err.message.contains("but got 100"));
    }

    #[test]
    fn validate_lowbit_layout_error_contains_context() {
        let err =
            validate_lowbit_layout(DType::TQ2G128, 10, 8, 256, "attn.q_proj", 34).unwrap_err();
        // 8 rows *2 groups *34 =544 expected, got 10
        assert!(err.message.contains("TQ2G128"));
        assert!(err.message.contains("m=8"));
        assert!(err.message.contains("k=256"));
        assert!(err.message.contains("attn.q_proj"));
        assert!(err.message.contains("expects 544"));
        assert!(err.message.contains("but got 10"));
        assert!(err.message.contains("m*(k/128)*34"));
    }

    #[test]
    fn lowbit_expected_bytes_overflow() {
        // bytes_per_row*m overflows: choose m=usize::MAX, k=128 => bytes_per_row=34 => 34*MAX overflows
        let err =
            lowbit_expected_bytes(DType::TQ2G128, usize::MAX, 128, "overflow_m", 34).unwrap_err();
        assert!(err.message.contains("overflow"), "got {}", err.message);
        assert!(err.message.contains("TQ2G128"), "dtype: {}", err.message);
        assert!(
            err.message.contains("overflow_m"),
            "caller: {}",
            err.message
        );
        assert!(err.message.contains("m="), "shape: {}", err.message);
        // BQ1G128 overflow same
        let err = lowbit_expected_bytes(DType::BQ1G128, usize::MAX, 256, "ov_bq", 18).unwrap_err();
        assert!(err.message.contains("overflow"));
        assert!(err.message.contains("BQ1G128"));
        assert!(err.message.contains("ov_bq"));
    }

    #[test]
    fn lowbit_expected_bytes_zero_m() {
        // m=0 => 0 expected regardless of K (as long as K%128==0)
        assert_eq!(
            lowbit_expected_bytes(DType::TQ2G128, 0, 4096, "zero", 34).unwrap(),
            0
        );
        assert_eq!(
            lowbit_expected_bytes(DType::BQ1G128, 0, 128, "zero", 18).unwrap(),
            0
        );
    }

    /// MQ4C (qt=45) ships as 136 B/group pad layout — same total as MQ4 v1/v2.
    /// Compact 132 B groups are rejected at load; do not reintroduce them.
    #[test]
    fn mq4c_group_bytes_is_136_not_compact_132() {
        assert_eq!(MQ4C_GROUP_BYTES, 136, "MQ4C pad layout is 136 B/group");
        assert_ne!(
            MQ4C_GROUP_BYTES, 132,
            "compact 132 B MQ4C is not production"
        );
        // decode_raw_codec expected length: m * (k/256) * MQ4C_GROUP_BYTES
        let m = 4usize;
        let k = 512usize;
        let gpr = k / 256;
        let expected = m * gpr * MQ4C_GROUP_BYTES;
        assert_eq!(expected, m * gpr * 136);
        assert_ne!(expected, m * gpr * 132);
        let codec = raw_codec(45).expect("qt=45 MQ4CG256 codec");
        assert_eq!(codec.dtype, DType::MQ4CG256);
    }

    #[test]
    fn mq_v2_group_bytes_match_spec() {
        assert_eq!(MQ6G256V2_GROUP_BYTES, 200, "qt47 MQ6G256V2 is 200 B/group");
        assert_eq!(MQ5G256V2_GROUP_BYTES, 168, "qt48 MQ5G256V2 is 168 B/group");
        assert_eq!(MQ3G256V2_GROUP_BYTES, 104, "qt49 MQ3G256V2 is 104 B/group");
        assert_eq!(MQ2G256V2_GROUP_BYTES, 72, "qt50 MQ2G256V2 is 72 B/group");
        // Each qt maps one-to-one to its DType and exact block bytes.
        assert_eq!(raw_codec(47).unwrap().dtype, DType::MQ6G256V2);
        assert_eq!(raw_codec(48).unwrap().dtype, DType::MQ5G256V2);
        assert_eq!(raw_codec(49).unwrap().dtype, DType::MQ3G256V2);
        assert_eq!(raw_codec(50).unwrap().dtype, DType::MQ2G256V2);
        // Existing qts unchanged.
        assert_eq!(raw_codec(44).unwrap().dtype, DType::MQ4G256V2);
        assert_eq!(raw_codec(45).unwrap().dtype, DType::MQ4CG256);
        assert_eq!(raw_codec(15).unwrap().dtype, DType::MQ6G256);
        assert_eq!(raw_codec(17).unwrap().dtype, DType::MQ3G256);
        assert_eq!(raw_codec(18).unwrap().dtype, DType::MQ2G256);
    }

    #[test]
    fn mq_v2_require_k_mod_256_and_awq() {
        for dt in [
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
        ] {
            assert!(dt.requires_k_mod_256(), "{dt:?} must require K%256==0");
            assert!(dt.supports_awq_sidecar(), "{dt:?} must support AWQ sidecar");
        }
        // Legacy counterparts remain distinct DTypes (no alias).
        assert_ne!(DType::MQ6G256V2, DType::MQ6G256);
        assert_ne!(DType::MQ5G256V2, DType::MQ5G256);
        assert_ne!(DType::MQ3G256V2, DType::MQ3G256);
        assert_ne!(DType::MQ2G256V2, DType::MQ2G256);
    }

    /// The supported-set predicate must equal what the decoder actually does.
    ///
    /// Probes every `u8`: a type the decoder handles (no panic on a small
    /// buffer) must be `true`, and one it panics on must be `false`. This is
    /// what keeps a new decoder arm from being invisible to the host-dequant
    /// loaders that screen artifacts with the predicate — the failure mode
    /// otherwise is a false refusal, or a panic mid-load.
    #[test]
    fn dequant_f32_supported_matches_the_decoder() {
        let data = vec![0u8; 1024];
        for qt in 0..=u8::MAX {
            let handled = std::panic::catch_unwind(|| dequantize_to_f32(qt, &data, 32)).is_ok();
            assert_eq!(
                dequant_f32_supported(qt),
                handled,
                "qt={qt}: predicate says {}, decoder {}",
                dequant_f32_supported(qt),
                if handled { "handles it" } else { "panics" }
            );
        }
    }
}
