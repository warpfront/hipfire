// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! dspine_convert: Convert a HuggingFace DSpine draft safetensors + config.json
//! (the Tencent/Qwen3.8-27B-DSpine-b16 layout) into a hipfire `.hfq` sidecar
//! with `arch_id = 24`.
//!
//! Usage:
//!     dspine_convert --input <dir_or_hf_id> --output <file.hfq>
//!     dspine_convert --input <dir_or_hf_id> --output <file.hfq> --mq4
//!     dspine_convert --input <dir_or_hf_id> --output <file.hfq> --mq4v2 [--mq4v2-symmetric]
//!     dspine_convert --input <dir_or_hf_id> --output <file.hfq> --keep-f32
//!
//! Tier surface (mutually exclusive; default is the BF16 tier):
//!   default / --bf16  — every weight stays BF16. The upstream DSpine runtime
//!                       contract is BF16-only and refuses quantized weights,
//!                       so the bit-exact arm is the default; bf16→f16 would
//!                       lose exponent range and is never implicit here.
//!   --mq4 / --mq4v2   — decoder GEMM weights quantized to the MQ family.
//!                       Bridge tensors stay raw BF16 in every tier (they are
//!                       lookup-table data feeding bf16 GEMMs, not GEMM
//!                       weights) and norms stay F32.
//!   --keep-f32        — debug upcast of everything.
//!
//! The DSpine draft is a DFlash decoder trunk plus a `selected_prefix_bridge`
//! code-bridge section and four I64 version scalars. Trunk tensor names match
//! the arch-20 DFlash layout exactly (`fc.weight`, `hidden_norm.weight`,
//! `norm.weight`, `layers.N.*`), so decoder weights ride the existing DFlash
//! draft-load path; the bridge section and its metadata are what the arch-24
//! loader consumes. The draft has no embed_tokens and no lm_head: it borrows
//! the target's (vocab 248320, hidden 5120), enforced at load time.
//!
//! Fail-closed contract — the converter refuses, and the loader re-checks at
//! load time (mirroring the upstream runtime's own config validation):
//!   - discriminator: `dflash_config.selected_prefix_code_bridge` must be
//!     present; a plain DFlash draft must go through dflash_convert instead
//!   - dtype bfloat16; terminal disabled (`selected_prefix_terminal == false`
//!     AND `selected_prefix_bridge.out.weight` all-zero)
//!   - `selected_prefix_feature_write_mode == "independent"`
//!   - `draft_block_causal == true`
//!   - `code_bridge_second_pass.mode == "serial_table"` with
//!     `first_exact == num_hidden_layers - 1`
//!   - `head_topk` a power of two in [2, 64] on both refine heads
//!   - all four version scalars == 1
//!   - no DFlash2/Domino selector tensors (`prefix_gru`, `embed_proj`,
//!     `candidate_selector`)
//!
//! Metadata JSON layout:
//!
//! ```json
//! {
//!   "architecture": "dspine",
//!   "config": {<full HF config.json>},
//!   "dflash": {<trunk knobs, same flat shape as dflash_convert>},
//!   "dspine": {
//!     "code_size": 1024,
//!     "bridge_rank": 512,
//!     "feature_layers": [1, 2, 3, 4, 5],
//!     "second_pass": { "mode": "serial_table", "first_exact": 4, "head_topk": 16 },
//!     "last_write_refine": { "kind": "exact_sender", "head_topk": 16 },
//!     "versions": { "draft_block_causal": 1, "last_write_refine": 1,
//!                   "independent_write": 1, "terminal_disabled": 1 }
//!   },
//!   "tokenizer": null
//! }
//! ```
//!
//! arch_id for the dspine draft is 24. The shipped config carries
//! `model_type: "qwen3"` and `architectures: ["DFlashDraftModel"]` (upstream
//! rewrites the class name at plugin registration), so neither string can
//! discriminate DSpine from a trunk or a plain DFlash draft — the converter
//! keys off the bridge config section and stamps 24 directly, exactly as
//! dflash_convert stamps 20. `dspine_draft` is registered in arch_mapping as
//! a lookup alias for packers.

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported input dtype: {other}"),
    }
}

fn f32_slice_to_f32_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 4);
    for &v in f32_data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

/// Truncating f32 → bf16 (top 16 bits). DSpine checkpoints are natively BF16,
/// and `bf16_to_f32` is exact, so truncation of converted values is a bit-exact
/// round trip — no invented rounding. Values that never came from bf16 may
/// lose their low mantissa bits, which is what the BF16 tier means anyway.
fn f32_to_bf16_bits(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

fn f32_slice_to_bf16_bytes(f32_data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(f32_data.len() * 2);
    for &v in f32_data {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

// ─── FWHT + MQ quantization ───────────────────────────────────────────────

/// CPU-side FWHT on a 256-element group. Matches the GPU-side
/// fwht_forward_256 in rdna_compute: signs1 → butterfly → scale → signs2.
/// Kept in sync with dflash_convert's copy (seeds 42/1042 pair with the
/// engine's `rdna_compute::Gpu::ensure_mq_signs()`).
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

/// Generate FWHT sign table matching the engine's gen_fwht_signs.
/// Standard MQ4 seeds are 42 (signs1) and 1042 (signs2).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// MagnumQuant MQ4-G256: FWHT-rotated 4-bit quantization.
/// 136 bytes per 256 weights (0.531 B/w). Same binary layout as HFQ4-G256;
/// the rotation is baked into the weights so the GEMM kernel just rotates
/// the input x instead of inverse-rotating W.
fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());
        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }
    output
}

/// MagnumQuant MQ4G256 v2 (qt=44): per-128 asymmetric (or, with `symmetric`, the
/// shared symmetric grid of `hipfire_quantize::mq_symmetric`), fp16 header, G256 layout.
///
/// 136 B/group: `[0..2)` fp16 scale h0, `[2..4)` fp16 zero h0, `[4..6)` fp16
/// scale h1, `[6..8)` fp16 zero h1, `[8..136)` packed nibbles (same order as
/// MQ4-G256).
const MQ4V2_GROUP_BYTES: usize = 136;

fn quantize_mq4g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    symmetric: bool,
) -> Vec<u8> {
    // m,k are the logical 2D shape; encoding is linear over w (API parity with main).
    let _ = (m, k);
    let group_size = 256;
    let block_bytes = MQ4V2_GROUP_BYTES;
    let n = w.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];
    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&w[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut q = [0u8; 256];
        if symmetric {
            for h in 0..2 {
                let off = h * 128;
                let (s, z) = hipfire_quantize::mq_symmetric::quantize_symmetric_half(
                    &group[off..off + 128],
                    15,
                    &mut q[off..off + 128],
                );
                scales[h] = s;
                zeros[h] = z;
            }
        } else {
            let mut sts = [0.0f32; 2];
            let mut zs = [0.0f32; 2];
            let mut degenerate = [false; 2];
            for h in 0..2 {
                let off = h * 128;
                let slice = &group[off..off + 128];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let sc_bits = if hi == lo { 0u16 } else { f32_to_f16(step_f32) };
                let z_bits = f32_to_f16(lo);
                scales[h] = sc_bits;
                zeros[h] = z_bits;
                let st = f16_to_f32(sc_bits);
                let z = f16_to_f32(z_bits);
                sts[h] = st;
                zs[h] = z;
                degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
            }
            for h in 0..2 {
                let off = h * 128;
                if !degenerate[h] {
                    let st = sts[h];
                    let z = zs[h];
                    let inv = 1.0 / st;
                    for i in 0..128 {
                        let v = group[off + i];
                        q[off + i] = ((v - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                    }
                }
            }
        }
        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&scales[0].to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&zeros[0].to_le_bytes());
        output[out_off + 4..out_off + 6].copy_from_slice(&scales[1].to_le_bytes());
        output[out_off + 6..out_off + 8].copy_from_slice(&zeros[1].to_le_bytes());
        for i in 0..128 {
            let lo_q = q[2 * i];
            let hi_q = q[2 * i + 1];
            output[out_off + 8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4);
        }
    }
    output
}

// ─── Serial tables (TG / TSV) from the TARGET's lm_head ──────────────────

/// One parsed index entry of an existing HFQ container.
#[derive(Clone)]
struct HfqIndexEntry {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    offset: u64, // absolute offset of the tensor payload in the file
    len: u64,
}

/// Read an HFQ container index (the layout `write_hfq` emits and the runtime
/// `HfqFile::open` parses): 32-byte header, metadata JSON (brace-scanned to
/// its end, mirroring the runtime reader), tensor index, contiguous data.
fn read_hfq_index(path: &Path) -> (u32, Vec<HfqIndexEntry>) {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut header = [0u8; 32];
    f.read_exact(&mut header).expect("hfq header");
    assert_eq!(&header[..4], b"HFQM", "bad HFQ magic in {}", path.display());
    let arch = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let n_tensors = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let data_off = u64::from_le_bytes(header[24..32].try_into().unwrap());

    f.seek(SeekFrom::Start(32)).unwrap();
    let mut region = vec![0u8; (data_off - 32) as usize];
    f.read_exact(&mut region).expect("hfq metadata+index region");
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut json_end = 0usize;
    for (i, &b) in region.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if b == b'{' {
                depth += 1;
            }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    assert!(json_end > 0, "metadata JSON not terminated in {}", path.display());

    let mut pos = json_end;
    let idx_n = u32::from_le_bytes(region[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n_tensors, "tensor-index count mismatch in {}", path.display());
    pos += 4;
    let mut entries = Vec::with_capacity(idx_n);
    let mut running = data_off;
    for _ in 0..idx_n {
        let name_len = u16::from_le_bytes(region[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&region[pos..pos + name_len]).to_string();
        pos += name_len;
        let qt = region[pos];
        pos += 1;
        let nd = region[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(nd);
        for _ in 0..nd {
            shape.push(u32::from_le_bytes(region[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let gs = u32::from_le_bytes(region[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let len = u64::from_le_bytes(region[pos..pos + 8].try_into().unwrap());
        pos += 8;
        entries.push(HfqIndexEntry {
            name,
            quant_type: qt,
            shape,
            group_size: gs,
            offset: running,
            len,
        });
        running += len;
    }
    (arch, entries)
}

fn read_hfq_tensor(path: &Path, entry: &HfqIndexEntry) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).expect("reopen hfq");
    f.seek(SeekFrom::Start(entry.offset)).unwrap();
    let mut buf = vec![0u8; entry.len as usize];
    f.read_exact(&mut buf).expect("hfq tensor read");
    buf
}

/// MQ4G256V2 dequant: fp16 scale/zero per 128 half-group, packed nibbles,
/// then the inverse FWHT. The rotation `Q = S2·D·H·S1` (signs1 → butterfly →
/// scale → signs2) is orthogonal, so its inverse is the transposed order
/// `S1·D·H·S2`: the same butterfly with the two sign tables swapped.
fn dequant_mq4v2(raw: &[u8], n_weights: usize, signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    const BLOCK: usize = 136;
    let n_blocks = n_weights.div_ceil(256);
    assert_eq!(raw.len(), n_blocks * BLOCK, "mq4v2 payload size");
    (0..n_blocks)
        .into_par_iter()
        .flat_map_iter(|b| {
            let off = b * BLOCK;
            let s0 = f16_to_f32(u16::from_le_bytes(raw[off..off + 2].try_into().unwrap()));
            let z0 = f16_to_f32(u16::from_le_bytes(raw[off + 2..off + 4].try_into().unwrap()));
            let s1 = f16_to_f32(u16::from_le_bytes(raw[off + 4..off + 6].try_into().unwrap()));
            let z1 = f16_to_f32(u16::from_le_bytes(raw[off + 6..off + 8].try_into().unwrap()));
            let mut group = [0f32; 256];
            for i in 0..128 {
                let byte = raw[off + 8 + i];
                group[2 * i] = z0 + (byte & 0xF) as f32 * s0;
                group[2 * i + 1] = z1 + (byte >> 4) as f32 * s1;
            }
            cpu_fwht_256(&mut group, signs2, signs1); // inverse: swap the signs
            group.to_vec()
        })
        .collect::<Vec<f32>>()
}

/// F16 payload → f32 (norms and other exact small tensors in the target HFQ).
fn f16_bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Build the sender tables for the serial last-write refinement
/// (upstream `DSpineDraftSampler.build_serial_tables`):
///   TG       [V,R] = bf16((LmHead ⊙ norm_w) @ W_last)
///   TSV_gate [V,R] = sqrt(C) · codebook @ predecessor_gate^T
///   TSV_val  [V,R] = sqrt(C) · codebook @ prev^T
/// (the upstream `[G_p; A] @ (sqrt(C)·codebook)^T` transposed to one row per
/// vocab id, so `tsv[sid]` is a contiguous lookup like `tg[cand]`.)
/// All math in f32; the caller packs to BF16 for the HFQ.
fn build_serial_tables(
    lm_head: &[f32],    // [V, H] dequantized target lm_head
    norm_w: &[f32],     // [H] target final norm
    write_last: &[f32], // [H, R] row-major, draft bridge write of the last layer
    codebook: &[f32],   // [V, C] row-major
    pred_gate: &[f32],  // [R, C] row-major
    prev_w: &[f32],     // [R, C] row-major
    v: usize,
    h: usize,
    c: usize,
    r: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    assert_eq!(lm_head.len(), v * h, "lm_head shape");
    assert_eq!(norm_w.len(), h, "norm shape");
    assert_eq!(write_last.len(), h * r, "write_last shape");
    assert_eq!(codebook.len(), v * c, "codebook shape");
    assert_eq!(pred_gate.len(), r * c, "predecessor_gate shape");
    assert_eq!(prev_w.len(), r * c, "prev shape");
    let sqrt_c = (c as f32).sqrt();

    let tg: Vec<f32> = (0..v)
        .into_par_iter()
        .flat_map_iter(|vi| {
            let row = &lm_head[vi * h..(vi + 1) * h];
            let mut acc = vec![0f32; r];
            for (j, &x) in row.iter().enumerate() {
                let x = x * norm_w[j];
                let wrow = &write_last[j * r..(j + 1) * r];
                for (t, &w) in wrow.iter().enumerate() {
                    acc[t] += x * w;
                }
            }
            acc
        })
        .collect::<Vec<f32>>();

    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..v)
        .into_par_iter()
        .map(|vi| {
            let cb = &codebook[vi * c..(vi + 1) * c];
            let mut g = vec![0f32; r];
            let mut a = vec![0f32; r];
            for (t, pg) in pred_gate.chunks_exact(c).enumerate() {
                let mut s = 0f32;
                for j in 0..c {
                    s += cb[j] * pg[j];
                }
                g[t] = s * sqrt_c;
            }
            for (t, pv) in prev_w.chunks_exact(c).enumerate() {
                let mut s = 0f32;
                for j in 0..c {
                    s += cb[j] * pv[j];
                }
                a[t] = s * sqrt_c;
            }
            (g, a)
        })
        .collect::<Vec<(Vec<f32>, Vec<f32>)>>();
    let mut tsv_gate = Vec::with_capacity(v * r);
    let mut tsv_val = Vec::with_capacity(v * r);
    for (g, a) in pairs {
        tsv_gate.extend(g);
        tsv_val.extend(a);
    }
    (tg, tsv_gate, tsv_val)
}

// ─── HFQ File Format ──────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;
const ARCH_ID_DSPINE_DRAFT: u32 = 24;

#[repr(u8)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
enum QuantType {
    F16 = 1,
    F32 = 2,
    /// Raw BF16 weights (qt=16 in hipfire-quantize/src/hfq.rs). The bridge
    /// tensors and the default tier ride this; the byte values must stay in
    /// sync with the shared `QuantType::from_u8` table.
    BF16 = 16,
    MQ4G256 = 13,
    /// MQ4G256 v2 (qt=44): per-128 asymmetric fp16 header, 136 B/group G256 layout.
    MQ4G256V2 = 44,
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();

    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type as u8);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }

    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    f.write_all(metadata_bytes)?;
    f.write_all(&index_bytes)?;

    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    for t in tensors {
        f.write_all(&t.data)?;
    }

    Ok(())
}

// ─── Model discovery ───────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);
    if path.join("config.json").exists() {
        return input.to_string();
    }
    if input.contains('/') {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_root =
                format!("{home}/.cache/huggingface/hub/models--{org}--{name}/snapshots");
            if let Ok(entries) = std::fs::read_dir(&cache_root) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.join("config.json").exists() {
                        return p.to_string_lossy().into_owned();
                    }
                }
            }
        }
    }
    input.to_string()
}

// ─── Tensor classification ────────────────────────────────────────────────

/// Norms must stay F32 for numerical fidelity (same rule as dflash_convert).
fn is_norm_tensor(name: &str) -> bool {
    name.contains("input_layernorm")
        || name.contains("post_attention_layernorm")
        || name.contains("q_norm")
        || name.contains("k_norm")
        || name == "hidden_norm.weight"
        || name == "norm.weight"
}

/// The code-bridge section: lookup-table data feeding bf16 GEMMs. Stored raw
/// BF16 in every tier — never weight-quantized.
fn is_bridge_tensor(name: &str) -> bool {
    name.starts_with("selected_prefix_bridge.")
}

/// The four upstream version markers (two top-level, two under the bridge
/// prefix). They are validated and recorded in metadata, not emitted as
/// tensors.
fn is_version_scalar(name: &str) -> bool {
    name.ends_with("_version")
}

/// DFlash2 dynamic-conv selector machinery must never ride a DSpine sidecar;
/// their presence means the wrong repo was pointed at this converter.
fn is_forbidden_tensor(name: &str) -> bool {
    name.contains("prefix_gru")
        || name.contains("embed_proj")
        || name.contains("candidate_selector")
}

// ─── Config contract ──────────────────────────────────────────────────────

fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_u64())
}

fn parse_int_array(json: &serde_json::Value) -> Vec<i64> {
    json.as_array()
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default()
}

/// Contracts extracted from config.json after validation. Shapes of the
/// bridge tensors are cross-checked against these before conversion.
#[derive(Debug)]
struct DspineContracts {
    code_size: usize,
    bridge_rank: usize,
    second_pass_mode: String,
    first_exact: u64,
    second_pass_head_topk: u64,
    refine_kind: String,
    refine_head_topk: u64,
}

fn validate_head_topk(v: u64) -> Result<(), String> {
    if !(2..=64).contains(&v) {
        return Err(format!("{v} outside [2, 64]"));
    }
    if !v.is_power_of_two() {
        return Err(format!("{v} is not a power of two"));
    }
    Ok(())
}

/// Full config contract for a DSpine draft. Every branch is a pointing error
/// naming the offending field, mirroring the upstream runtime's own config
/// validation (which raises instead of silently degrading).
fn validate_dspine_config(config: &serde_json::Value) -> Result<DspineContracts, String> {
    let dflash_cfg = config
        .get("dflash_config")
        .ok_or("config.json missing dflash_config")?;
    let bridge = dflash_cfg
        .get("selected_prefix_code_bridge")
        .ok_or(
            "config.json missing dflash_config.selected_prefix_code_bridge: not a DSpine \
             draft (plain DFlash drafts go through dflash_convert, arch 20)",
        )?;
    let code_size = bridge
        .get("code_size")
        .and_then(|v| v.as_u64())
        .ok_or("selected_prefix_code_bridge missing code_size")? as usize;
    let bridge_rank = bridge
        .get("rank")
        .and_then(|v| v.as_u64())
        .ok_or("selected_prefix_code_bridge missing rank")? as usize;

    let dtype = config
        .get("dtype")
        .or_else(|| config.get("torch_dtype"))
        .and_then(|v| v.as_str())
        .ok_or("config.json missing dtype/torch_dtype")?;
    if dtype != "bfloat16" {
        return Err(format!(
            "dtype {dtype:?} != \"bfloat16\": the DSpine runtime contract is BF16-only"
        ));
    }

    let vocab = json_u64(config, "vocab_size")
        .ok_or("config.json missing vocab_size")?;
    if let Some(draft_vocab) = json_u64(config, "draft_vocab_size") {
        if draft_vocab != vocab {
            return Err(format!(
                "draft_vocab_size {draft_vocab} != vocab_size {vocab}: the draft borrows the \
                 target embedding and lm_head, vocab sizes must match"
            ));
        }
    }

    let terminal = dflash_cfg
        .get("selected_prefix_terminal")
        .and_then(|v| v.as_bool())
        .ok_or("dflash_config missing selected_prefix_terminal")?;
    if terminal {
        return Err(
            "selected_prefix_terminal is true: hipfire requires the terminal projection \
             disabled (selected_prefix_terminal=false, out.weight all-zero)"
                .into(),
        );
    }

    let write_mode = dflash_cfg
        .get("selected_prefix_feature_write_mode")
        .and_then(|v| v.as_str())
        .ok_or("dflash_config missing selected_prefix_feature_write_mode")?;
    if write_mode != "independent" {
        return Err(format!(
            "selected_prefix_feature_write_mode {write_mode:?} != \"independent\""
        ));
    }

    let block_causal = dflash_cfg
        .get("draft_block_causal")
        .and_then(|v| v.as_bool())
        .ok_or("dflash_config missing draft_block_causal")?;
    if !block_causal {
        return Err(
            "draft_block_causal is false: the block-causal draft trunk is a hard requirement"
                .into(),
        );
    }

    let second = config
        .get("code_bridge_second_pass")
        .ok_or("config.json missing code_bridge_second_pass")?;
    let mode = second
        .get("mode")
        .and_then(|v| v.as_str())
        .ok_or("code_bridge_second_pass missing mode")?;
    if mode != "serial_table" {
        return Err(format!(
            "code_bridge_second_pass.mode {mode:?} != \"serial_table\""
        ));
    }
    let first_exact = second
        .get("first_exact")
        .and_then(|v| v.as_u64())
        .ok_or("code_bridge_second_pass missing first_exact")?;
    let num_hidden_layers = json_u64(config, "num_hidden_layers")
        .ok_or("config.json missing num_hidden_layers")?;
    if first_exact != num_hidden_layers.saturating_sub(1) {
        return Err(format!(
            "code_bridge_second_pass.first_exact {first_exact} != num_hidden_layers \
             {num_hidden_layers} - 1"
        ));
    }
    let second_topk = second
        .get("head_topk")
        .and_then(|v| v.as_u64())
        .ok_or("code_bridge_second_pass missing head_topk")?;
    validate_head_topk(second_topk)
        .map_err(|e| format!("code_bridge_second_pass.head_topk: {e}"))?;

    let refine = dflash_cfg
        .get("selected_prefix_last_write_refine")
        .ok_or("dflash_config missing selected_prefix_last_write_refine")?;
    let kind = refine
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or("selected_prefix_last_write_refine missing kind")?;
    if kind != "exact_sender" {
        return Err(format!(
            "selected_prefix_last_write_refine.kind {kind:?} != \"exact_sender\""
        ));
    }
    let refine_topk = refine
        .get("head_topk")
        .and_then(|v| v.as_u64())
        .ok_or("selected_prefix_last_write_refine missing head_topk")?;
    validate_head_topk(refine_topk)
        .map_err(|e| format!("selected_prefix_last_write_refine.head_topk: {e}"))?;

    Ok(DspineContracts {
        code_size,
        bridge_rank,
        second_pass_mode: mode.to_string(),
        first_exact,
        second_pass_head_topk: second_topk,
        refine_kind: kind.to_string(),
        refine_head_topk: refine_topk,
    })
}

/// The DSpine fc reads one target hidden state per draft layer (concat along
/// K), so K must be exactly layers × hidden. (DFlash's multi-stream fc needs
/// `target_streams_per_layer`; DSpine has none.)
fn validate_fc_k(fc_k: usize, num_layers: usize, hidden: usize) -> Result<(), String> {
    let expected = num_layers * hidden;
    if fc_k != expected {
        return Err(format!(
            "fc.weight K {fc_k} != num_hidden_layers {num_layers} x hidden_size {hidden} = \
             {expected}"
        ));
    }
    Ok(())
}

fn validate_version_scalars(values: &[(String, i64)]) -> Result<(), String> {
    for (name, v) in values {
        if *v != 1 {
            return Err(format!("version scalar {name} = {v}, expected 1"));
        }
    }
    Ok(())
}

/// Cross-check the nine bridge tensor shapes against the config contracts
/// plus the trunk dims. Also proves presence (a missing bridge tensor is a
/// refusal, not a load-time surprise).
fn validate_bridge_shapes(
    c: &DspineContracts,
    hidden: usize,
    num_layers: usize,
    vocab: usize,
    shapes: &HashMap<String, TensorMeta>,
) -> Result<(), String> {
    let expect: [(&str, Vec<usize>); 9] = [
        (
            "selected_prefix_bridge.codebook",
            vec![vocab, c.code_size],
        ),
        (
            "selected_prefix_bridge.read_weight",
            vec![c.code_size, hidden],
        ),
        (
            "selected_prefix_bridge.early_read_weights",
            vec![num_layers - 1, c.code_size, hidden],
        ),
        (
            "selected_prefix_bridge.feature_write_weights",
            vec![num_layers, hidden, c.bridge_rank],
        ),
        (
            "selected_prefix_bridge.gate.weight",
            vec![c.bridge_rank, c.code_size],
        ),
        ("selected_prefix_bridge.gate.bias", vec![c.bridge_rank]),
        (
            "selected_prefix_bridge.predecessor_gate.weight",
            vec![c.bridge_rank, c.code_size],
        ),
        (
            "selected_prefix_bridge.prev.weight",
            vec![c.bridge_rank, c.code_size],
        ),
        (
            "selected_prefix_bridge.out.weight",
            vec![c.code_size, c.bridge_rank],
        ),
    ];
    for (name, want) in &expect {
        match shapes.get(*name) {
            None => return Err(format!("bridge tensor {name} missing from safetensors")),
            Some(meta) => {
                if meta.shape != *want {
                    return Err(format!(
                        "bridge tensor {name} shape {:?} != expected {:?}",
                        meta.shape, want
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The terminal projection must be disabled: upstream zeroes it at load, and
/// a hipfire sidecar refuses instead of silently zeroing.
fn validate_out_weight_zero(st: &SafetensorsFile) -> Result<(), String> {
    let name = "selected_prefix_bridge.out.weight";
    let (meta, raw) = st
        .tensor_data(name)
        .ok_or_else(|| format!("bridge tensor {name} missing from safetensors"))?;
    if meta.dtype != "BF16" {
        return Err(format!("{name} dtype {} != BF16", meta.dtype));
    }
    for c in raw.chunks_exact(2) {
        if bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) != 0.0 {
            return Err(format!(
                "{name} contains non-zero values: the terminal projection must be disabled \
                 (upstream zeroes it at load; hipfire refuses instead)"
            ));
        }
    }
    Ok(())
}

/// Every trunk tensor the arch-24 loader will demand (bridge tensors are
/// proven by [`validate_bridge_shapes`], version scalars by the version pass).
fn required_trunk_tensors(num_layers: usize) -> Vec<String> {
    let mut names = vec![
        "fc.weight".to_string(),
        "hidden_norm.weight".to_string(),
        "norm.weight".to_string(),
    ];
    for l in 0..num_layers {
        let p = format!("layers.{l}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("{p}.{suffix}"));
        }
    }
    names
}

// ─── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut keep_f32 = false;
    let mut use_bf16 = false;
    let mut use_mq4 = false;
    let mut use_mq4v2 = false;
    let mut mq4v2_symmetric = false;
    let mut target_hfq: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" | "-i" => {
                input_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--output" | "-o" => {
                output_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--mq4" => {
                use_mq4 = true;
                i += 1;
            }
            "--mq4v2" => {
                use_mq4v2 = true;
                i += 1;
            }
            "--mq4v2-symmetric" => {
                mq4v2_symmetric = true;
                i += 1;
            }
            "--target-hfq" => {
                target_hfq = Some(args[i + 1].clone());
                i += 2;
            }
            "--bf16" => {
                use_bf16 = true;
                i += 1;
            }
            "--keep-f32" => {
                keep_f32 = true;
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!(
                    "Usage: dspine_convert --input <dir_or_hf_id> --output <file.hfq> \
                     [--bf16 (default) | --mq4 | --mq4v2 [--mq4v2-symmetric] | --keep-f32]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let n_format_flags =
        (keep_f32 as u8) + (use_bf16 as u8) + (use_mq4 as u8) + (use_mq4v2 as u8);
    if n_format_flags > 1 {
        eprintln!("--bf16, --keep-f32, --mq4, --mq4v2 are mutually exclusive");
        std::process::exit(1);
    }
    if mq4v2_symmetric && !use_mq4v2 {
        eprintln!("error: --mq4v2-symmetric requires --mq4v2");
        std::process::exit(1);
    }

    let input_dir = input_dir.expect("--input required");
    let output_path = output_path.expect("--output required");
    let input_dir = resolve_model_path(&input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(&output_path);

    let dtype_desc = if keep_f32 {
        "F32"
    } else if use_mq4 {
        "MQ4-G256 (decoder GEMMs), BF16 (bridge), F32 (norms)"
    } else if use_mq4v2 {
        "MQ4V2-G256/qt44 (decoder GEMMs), BF16 (bridge), F32 (norms)"
    } else {
        "BF16 (default tier: upstream-equivalent, bit-exact)"
    };
    eprintln!("dspine_convert");
    eprintln!("  input : {}", input_dir.display());
    eprintln!("  output: {}", output_path.display());
    eprintln!("  dtype : {dtype_desc}");

    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_path.display()));
    let config: serde_json::Value =
        serde_json::from_str(&config_str).expect("config.json parse failed");

    let architectures = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let is_dflash_class = architectures
        .iter()
        .any(|v| v.as_str() == Some("DFlashDraftModel"));
    if !is_dflash_class {
        eprintln!(
            "warning: config.json architectures = {architectures:?}; DSpine drafts ship as \
             DFlashDraftModel"
        );
    }

    let contracts = match validate_dspine_config(&config) {
        Ok(c) => c,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };

    let block_size = json_u64(&config, "block_size")
        .expect("config.json missing block_size") as u32;
    let mask_token_id = config
        .get("dflash_config")
        .and_then(|d| json_u64(d, "mask_token_id"))
        .or_else(|| json_u64(&config, "mask_token_id"))
        .expect("missing mask_token_id (dflash_config or top-level)")
        as u32;
    let target_layer_ids = {
        let arr = config
            .get("dflash_config")
            .and_then(|d| d.get("target_layer_ids"))
            .or_else(|| config.get("target_layer_ids"))
            .expect("missing target_layer_ids (dflash_config or top-level)");
        parse_int_array(arr)
    };
    let num_target_layers = config
        .get("num_target_layers")
        .and_then(|v| v.as_u64())
        .expect("config.json missing num_target_layers");
    let num_hidden_layers =
        json_u64(&config, "num_hidden_layers").expect("config.json missing num_hidden_layers")
            as usize;
    let hidden_size = json_u64(&config, "hidden_size")
        .expect("config.json missing hidden_size") as usize;
    let num_attention_heads = json_u64(&config, "num_attention_heads")
        .expect("config.json missing num_attention_heads")
        as usize;
    let num_key_value_heads = json_u64(&config, "num_key_value_heads")
        .expect("config.json missing num_key_value_heads")
        as usize;
    let head_dim = json_u64(&config, "head_dim")
        .map(|v| v as usize)
        .unwrap_or(hidden_size / num_attention_heads);
    let intermediate_size = json_u64(&config, "intermediate_size")
        .expect("config.json missing intermediate_size") as usize;
    let vocab_size = json_u64(&config, "vocab_size")
        .expect("config.json missing vocab_size") as usize;

    eprintln!(
        "  dspine: block_size={}, mask_token_id={}, target_layers={:?}, hidden_layers={}, \
         hidden={}, code_size={}, bridge_rank={}",
        block_size,
        mask_token_id,
        target_layer_ids,
        num_hidden_layers,
        hidden_size,
        contracts.code_size,
        contracts.bridge_rank,
    );

    let draft_dtype = if keep_f32 {
        "f32"
    } else if use_mq4 {
        "mq4"
    } else if use_mq4v2 {
        "mq4v2"
    } else {
        "bf16"
    };
    // FWHT sign tables for MQ rotation. Seeds 42/1042 match the engine's
    // `rdna_compute::Gpu::ensure_mq_signs()` so quantized weights here can
    // be dequantized/used correctly on GPU at inference.
    let needs_fwht = use_mq4 || use_mq4v2;
    let signs1: Vec<f32> = if needs_fwht {
        gen_fwht_signs(42, 256)
    } else {
        Vec::new()
    };
    let signs2: Vec<f32> = if needs_fwht {
        gen_fwht_signs(1042, 256)
    } else {
        Vec::new()
    };
    let rope_theta = config
        .get("rope_theta")
        .cloned()
        .or_else(|| {
            config
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta").cloned())
        })
        .unwrap_or_else(|| serde_json::Value::from(10_000_000.0));

    let mut dflash_meta = json!({
        "block_size": block_size,
        "mask_token_id": mask_token_id,
        "target_layer_ids": target_layer_ids,
        "num_target_layers": num_target_layers,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "rms_norm_eps": config.get("rms_norm_eps").cloned().unwrap_or_else(|| serde_json::Value::from(1e-6)),
        "rope_theta": rope_theta,
        "vocab_size": config.get("vocab_size").cloned(),
        "draft_dtype": draft_dtype,
    });
    if let Some(obj) = dflash_meta.as_object_mut() {
        // Mirror the trunk knobs the arch-20 loader reads off `dflash`, so the
        // DFlash decoder-load path can be reused for the DSpine trunk as-is.
        for key in [
            "layer_types",
            "sliding_window",
            "use_sliding_window",
            "is_causal",
            "max_window_layers",
        ] {
            if let Some(v) = config.get(key) {
                obj.insert(key.into(), v.clone());
            }
        }
    }
    let feature_layers = config
        .get("dflash_config")
        .and_then(|d| d.get("selected_prefix_feature_layers"))
        .map(parse_int_array)
        .unwrap_or_default();
    let dspine_meta = json!({
        "code_size": contracts.code_size,
        "bridge_rank": contracts.bridge_rank,
        "feature_layers": feature_layers,
        "second_pass": {
            "mode": contracts.second_pass_mode,
            "first_exact": contracts.first_exact,
            "head_topk": contracts.second_pass_head_topk,
        },
        "last_write_refine": {
            "kind": contracts.refine_kind,
            "head_topk": contracts.refine_head_topk,
        },
    });
    let metadata = json!({
        "architecture": "dspine",
        "config": config,
        "dflash": dflash_meta,
        "dspine": dspine_meta,
        "tokenizer": serde_json::Value::Null,
    });

    // Load + convert all safetensors files (draft is one file in practice).
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .inspect(|p| eprintln!("  loading: {}", p.display()))
        .map(|p| SafetensorsFile::open(p).expect("safetensors open failed"))
        .collect();
    assert!(
        !st_files.is_empty(),
        "no .safetensors files found in input dir"
    );

    let mut name_to_file: Vec<(String, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            name_to_file.push((name.to_string(), fi));
        }
    }
    name_to_file.sort_by_key(|(name, _)| name.clone());
    eprintln!("  tensors: {}", name_to_file.len());

    // Fail-closed structural passes before any byte is rewritten.
    if let Some((name, _)) = name_to_file.iter().find(|(n, _)| is_forbidden_tensor(n)) {
        eprintln!(
            "error: tensor {name} belongs to the DFlash2/Domino selector machinery, not a \
             DSpine draft"
        );
        std::process::exit(1);
    }
    let shapes: HashMap<String, TensorMeta> = st_files
        .iter()
        .flat_map(|st| st.tensors.iter().map(|(n, m)| (n.clone(), m.clone())))
        .collect();
    let missing: Vec<String> = required_trunk_tensors(num_hidden_layers)
        .into_iter()
        .filter(|n| !shapes.contains_key(n))
        .collect();
    if !missing.is_empty() {
        eprintln!("error: trunk tensors missing from safetensors: {missing:?}");
        std::process::exit(1);
    }
    if let Err(error) = validate_bridge_shapes(
        &contracts,
        hidden_size,
        num_hidden_layers,
        vocab_size,
        &shapes,
    ) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    if let Err(error) = validate_out_weight_zero(&st_files[0]) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    let fc_meta = shapes.get("fc.weight").expect("fc.weight presence proven above");
    if fc_meta.shape.len() != 2 {
        eprintln!("error: fc.weight must be 2-D, got shape {:?}", fc_meta.shape);
        std::process::exit(1);
    }
    if let Err(error) = validate_fc_k(fc_meta.shape[1], num_hidden_layers, hidden_size) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
    let mut version_values: Vec<(String, i64)> = Vec::new();
    for (name, fi) in name_to_file.iter().filter(|(n, _)| is_version_scalar(n)) {
        let (meta, raw) = st_files[*fi]
            .tensor_data(name)
            .unwrap_or_else(|| panic!("tensor lookup failed: {name}"));
        if meta.dtype != "I64" || !meta.shape.is_empty() {
            eprintln!(
                "error: version scalar {name} must be a scalar I64, got {} {:?}",
                meta.dtype, meta.shape
            );
            std::process::exit(1);
        }
        let bytes: [u8; 8] = raw
            .try_into()
            .unwrap_or_else(|_| panic!("version scalar {name} is not 8 bytes"));
        version_values.push((name.clone(), i64::from_le_bytes(bytes)));
    }
    if version_values.is_empty() {
        eprintln!(
            "error: no version scalars found (expected draft_block_causal_version, \
             last_write_refine_version, selected_prefix_bridge.independent_write_version, \
             selected_prefix_bridge.terminal_disabled_version)"
        );
        std::process::exit(1);
    }
    if let Err(error) = validate_version_scalars(&version_values) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }

    // Version scalars are recorded in metadata, not emitted as tensors.
    let versions_json: serde_json::Map<String, serde_json::Value> = version_values
        .iter()
        .map(|(n, v)| (n.clone(), serde_json::Value::from(*v)))
        .collect();

    // ── Serial tables (TG / TSV): prebuilt from the TARGET's lm_head ─────
    // `--target-hfq` names the target's hipfire container (e.g. the 27B
    // mq4-xts artifact): its lm_head (MQ4G256V2) and final norm (F16) feed
    // the upstream `build_serial_tables` math, so the runtime lifts the
    // three tables straight out of this sidecar with no dequant work of
    // its own. Without the flag the sidecar carries no tables and the
    // loader stays on the plain greedy-argmax refinement (fail-closed).
    let serial_tables: Option<[HfqTensor; 3]> = target_hfq.as_ref().map(|th| {
        let tpath = Path::new(th);
        eprintln!("  serial tables: target lm_head from {}", tpath.display());
        let (t_arch, entries) = read_hfq_index(tpath);
        assert_eq!(t_arch, 5, "target container must be arch 5 (qwen3_5)");
        let find = |name: &str| {
            entries
                .iter()
                .find(|e| e.name == name)
                .unwrap_or_else(|| panic!("target HFQ missing tensor {name}"))
                .clone()
        };
        let lm_entry = find("lm_head.weight");
        assert_eq!(
            lm_entry.quant_type, 44,
            "lm_head must be MQ4G256V2 (qt 44) for the dequant path"
        );
        assert_eq!(
            lm_entry.shape,
            vec![vocab_size as u32, hidden_size as u32],
            "lm_head shape must match the draft contract (V,H)"
        );
        let lm_raw = read_hfq_tensor(tpath, &lm_entry);

        let tensor_f32 = |name: &str, expect_shape: &[u32]| -> Vec<f32> {
            let idx = name_to_file
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("safetensors missing {name}"))
                .1;
            let (meta, raw) = st_files[idx]
                .tensor_data(name)
                .unwrap_or_else(|| panic!("tensor lookup failed: {name}"));
            assert_eq!(
                meta.shape.iter().map(|d| *d as u32).collect::<Vec<u32>>(),
                expect_shape,
                "{name} shape"
            );
            to_f32(raw, &meta.dtype)
        };
        let c = contracts.code_size;
        let r = contracts.bridge_rank;
        let write_weights = tensor_f32(
            "selected_prefix_bridge.feature_write_weights",
            &[5, hidden_size as u32, r as u32],
        );
        let write_last = write_weights[4 * hidden_size * r..].to_vec();
        let codebook = tensor_f32(
            "selected_prefix_bridge.codebook",
            &[vocab_size as u32, c as u32],
        );
        let pred_gate = tensor_f32(
            "selected_prefix_bridge.predecessor_gate.weight",
            &[r as u32, c as u32],
        );
        let prev_w = tensor_f32("selected_prefix_bridge.prev.weight", &[r as u32, c as u32]);

        // The upstream build_serial_tables pairs the target lm_head with
        // the DRAFT model's own norm.weight (dm.norm.weight), not the
        // target's final norm.
        let norm_w = tensor_f32("norm.weight", &[hidden_size as u32]);
        assert_eq!(norm_w.len(), hidden_size, "norm width");
        // The target lm_head rotation uses the same standard seed pair as
        // the engine (`ensure_mq_signs`), independent of this run's tier.
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        eprintln!(
            "  serial tables: dequant lm_head ({} weights)...",
            vocab_size * hidden_size
        );
        let lm_head = dequant_mq4v2(&lm_raw, vocab_size * hidden_size, &signs1, &signs2);
        eprintln!(
            "  serial tables: building TG / TSV (V={vocab_size}, H={hidden_size}, C={c}, R={r})..."
        );
        let (tg, tsv_gate, tsv_val) = build_serial_tables(
            &lm_head,
            &norm_w,
            &write_last,
            &codebook,
            &pred_gate,
            &prev_w,
            vocab_size,
            hidden_size,
            c,
            r,
        );
        drop(lm_head);
        let shape = vec![vocab_size as u32, r as u32];
        // Upstream keeps the sender tables in fp32 on the GPU; the BF16
        // rounding here showed up as refinement-score noise, so store exact.
        let mk = |data: Vec<f32>, name: &str| HfqTensor {
            name: name.to_string(),
            quant_type: QuantType::F32,
            shape: shape.clone(),
            group_size: 0,
            data: f32_slice_to_f32_bytes(&data),
        };
        eprintln!(
            "  serial tables: TG/TSV built ({:.1} MiB bf16 each)",
            (vocab_size * r * 2) as f64 / (1024.0 * 1024.0)
        );
        [
            mk(tg, "dspine.tg"),
            mk(tsv_gate, "dspine.tsv_gate"),
            mk(tsv_val, "dspine.tsv_val"),
        ]
    });

    let metadata_json = {
        let mut root = metadata;
        let obj = root.as_object_mut().unwrap();
        let dspine = obj.get_mut("dspine").unwrap().as_object_mut().unwrap();
        dspine.insert("versions".into(), serde_json::Value::Object(versions_json));
        if serial_tables.is_some() {
            dspine.insert(
                "serial_tables".into(),
                json!({
                    "kind": contracts.refine_kind,
                    "head_topk": contracts.refine_head_topk,
                    "version": 1,
                    "weight": 1.0,
                    "tensors": ["dspine.tg", "dspine.tsv_gate", "dspine.tsv_val"],
                }),
            );
        }
        serde_json::to_string(&root).unwrap()
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(name_to_file.len());
    let mut total_params = 0u64;
    let mut total_bytes_out = 0usize;

    for (name, fi) in &name_to_file {
        if is_version_scalar(name) {
            continue;
        }
        let (meta, raw) = st_files[*fi]
            .tensor_data(name)
            .expect("tensor lookup failed");
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // Classification rules:
        //   version scalars → validated above, recorded in metadata, skipped.
        //   norms → always F32 (small, precision-critical).
        //   bridge tensors → always raw BF16 (lookup-table data, bf16 GEMMs).
        //   everything else → F32 if --keep-f32, MQ4 if requested (2-D, >=256
        //   elements), else BF16 (the default tier; DSpine ships BF16, so the
        //   raw bytes pass through unchanged).
        let bf16_bytes = if meta.dtype == "BF16" {
            raw.to_vec()
        } else {
            f32_slice_to_bf16_bytes(&to_f32(raw, &meta.dtype))
        };
        let (quant_type, group_size, bytes) = if is_norm_tensor(name) {
            let f32_data = to_f32(raw, &meta.dtype);
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if is_bridge_tensor(name) {
            (QuantType::BF16, 0u32, bf16_bytes)
        } else if keep_f32 {
            let f32_data = to_f32(raw, &meta.dtype);
            (QuantType::F32, 0u32, f32_slice_to_f32_bytes(&f32_data))
        } else if use_mq4 && n_elements >= 256 {
            let q = quantize_mq4g256(&to_f32(raw, &meta.dtype), &signs1, &signs2);
            (QuantType::MQ4G256, 256u32, q)
        } else if use_mq4v2 && meta.shape.len() == 2 && n_elements >= 256 {
            let m = meta.shape[0];
            let k = meta.shape[1];
            let q = quantize_mq4g256v2(
                &to_f32(raw, &meta.dtype),
                m,
                k,
                &signs1,
                &signs2,
                mq4v2_symmetric,
            );
            (QuantType::MQ4G256V2, 256u32, q)
        } else {
            (QuantType::BF16, 0u32, bf16_bytes)
        };

        total_bytes_out += bytes.len();
        hfq_tensors.push(HfqTensor {
            name: name.clone(),
            quant_type,
            shape: meta.shape.iter().map(|d| *d as u32).collect(),
            group_size,
            data: bytes,
        });
    }

    eprintln!(
        "  total params: {:.3}B ({} tensors)",
        total_params as f64 / 1e9,
        hfq_tensors.len()
    );
    if let Some(tables) = serial_tables {
        hfq_tensors.extend(tables);
    }
    eprintln!(
        "  total out  : {:.2} MiB",
        total_bytes_out as f64 / (1024.0 * 1024.0)
    );

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("mkdir -p output parent");
        }
    }

    write_hfq(
        output_path,
        ARCH_ID_DSPINE_DRAFT,
        &metadata_json,
        &hfq_tensors,
    )
    .expect("write_hfq failed");

    eprintln!("  wrote: {}", output_path.display());
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn fwht_inverse_is_the_swapped_sign_order() {
        // Q = S2·D·H·S1; Q⁻¹ = S1·D·H·S2 — the same butterfly with the two
        // sign tables swapped (H symmetric, D diagonal, S diagonal ±1).
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let mut x: Vec<f32> = (0..256).map(|i| ((i * 37) % 61) as f32 - 30.0).collect();
        let orig = x.clone();
        cpu_fwht_256(&mut x, &s1, &s2); // forward
        cpu_fwht_256(&mut x, &s2, &s1); // inverse (swapped)
        for i in 0..256 {
            assert!(
                (x[i] - orig[i]).abs() < 1e-2 + orig[i].abs() * 1e-3,
                "fwht inverse mismatch at {i}: {} vs {}",
                x[i],
                orig[i]
            );
        }
    }

    #[test]
    fn mq4v2_roundtrip_matches_within_quantization_error() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let w: Vec<f32> = (0..512)
            .map(|i| (((i * 91) % 197) as f32 / 197.0 - 0.5) * 0.8)
            .collect();
        let amax = w.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let q = quantize_mq4g256v2(&w, 2, 256, &s1, &s2, true);
        let back = dequant_mq4v2(&q, 512, &s1, &s2);
        // A symmetric 4-bit grid has a half-step of ~amax/16, but the
        // Hadamard redistributes each group's code error across its 256
        // elements: single elements land within ~1-2 steps. The honest
        // semantic check is the group energy (FWHT preserves it) plus a
        // loose per-element bound; an ordering/half-group/sign-swap bug
        // blows both up by orders of magnitude.
        let mut sum_sq = 0.0f64;
        let mut worst = 0.0f32;
        for i in 0..512 {
            let err = (back[i] - w[i]).abs();
            sum_sq += (err as f64) * (err as f64);
            worst = worst.max(err);
        }
        let rms = (sum_sq / 512.0).sqrt();
        assert!(
            rms <= (amax * 0.12) as f64,
            "mq4v2 roundtrip RMS {rms} exceeds amax*0.12 (amax {amax})"
        );
        assert!(
            worst <= amax * 0.35,
            "mq4v2 roundtrip worst element {worst} exceeds amax*0.35 (amax {amax})"
        );
    }

    #[test]
    fn serial_tables_match_closed_form() {
        // V=2, H=2, C=2, R=2 hand-checked: tg = (lm⊙nw)@W_last (identity
        // write), tsv = sqrt(C)·cb@g^T with identity codebook rows.
        let lm = vec![1.0, 2.0, 3.0, 4.0];
        let nw = vec![0.5, 2.0];
        let write_last = vec![1.0, 0.0, 0.0, 1.0];
        let cb = vec![1.0, 0.0, 0.0, 1.0];
        let pg = vec![1.0, 2.0, 3.0, 4.0];
        let pv = vec![5.0, 6.0, 7.0, 8.0];
        let (tg, tsv_gate, tsv_val) =
            build_serial_tables(&lm, &nw, &write_last, &cb, &pg, &pv, 2, 2, 2, 2);
        assert_eq!(tg, vec![0.5, 4.0, 1.5, 8.0]);
        let s = 2.0f32.sqrt();
        assert!((tsv_gate[0] - s).abs() < 1e-6);
        assert!((tsv_gate[1] - 3.0 * s).abs() < 1e-6);
        assert!((tsv_gate[2] - 2.0 * s).abs() < 1e-6);
        assert!((tsv_gate[3] - 4.0 * s).abs() < 1e-6);
        assert!((tsv_val[0] - 5.0 * s).abs() < 1e-6);
        assert!((tsv_val[1] - 7.0 * s).abs() < 1e-6);
        assert!((tsv_val[2] - 6.0 * s).abs() < 1e-6);
        assert!((tsv_val[3] - 8.0 * s).abs() < 1e-6);
    }

    fn valid_config() -> serde_json::Value {
        json!({
            "dtype": "bfloat16",
            "num_hidden_layers": 5,
            "vocab_size": 248320,
            "draft_vocab_size": 248320,
            "code_bridge_second_pass": {"mode": "serial_table", "first_exact": 4, "head_topk": 16},
            "dflash_config": {
                "selected_prefix_code_bridge": {"code_size": 1024, "rank": 512},
                "selected_prefix_terminal": false,
                "selected_prefix_feature_write_mode": "independent",
                "draft_block_causal": true,
                "selected_prefix_last_write_refine": {"kind": "exact_sender", "head_topk": 16}
            }
        })
    }

    fn with_dflash_key(config: &serde_json::Value, key: &str, value: serde_json::Value) -> serde_json::Value {
        let mut c = config.clone();
        c.get_mut("dflash_config")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value);
        c
    }

    #[test]
    fn accepts_the_shipped_contract_shape() {
        let c = validate_dspine_config(&valid_config()).unwrap();
        assert_eq!(c.code_size, 1024);
        assert_eq!(c.bridge_rank, 512);
        assert_eq!(c.second_pass_mode, "serial_table");
        assert_eq!(c.first_exact, 4);
        assert_eq!(c.second_pass_head_topk, 16);
        assert_eq!(c.refine_kind, "exact_sender");
    }

    #[test]
    fn a_plain_dflash_draft_is_refused_at_the_bridge_discriminator() {
        let mut c = valid_config();
        c.get_mut("dflash_config")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("selected_prefix_code_bridge");
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("not a DSpine"), "{err}");
        assert!(err.contains("dflash_convert"), "{err}");
    }

    #[test]
    fn non_bf16_dtype_is_refused() {
        let mut c = valid_config();
        c["dtype"] = json!("float16");
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("BF16-only"), "{err}");
    }

    #[test]
    fn mismatched_draft_vocab_is_refused() {
        let mut c = valid_config();
        c["draft_vocab_size"] = json!(151936);
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("draft_vocab_size"), "{err}");
    }

    #[test]
    fn an_enabled_terminal_is_refused() {
        let c = with_dflash_key(&valid_config(), "selected_prefix_terminal", json!(true));
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("terminal"), "{err}");
    }

    #[test]
    fn non_independent_write_mode_is_refused() {
        let c = with_dflash_key(
            &valid_config(),
            "selected_prefix_feature_write_mode",
            json!("additive"),
        );
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("independent"), "{err}");
    }

    #[test]
    fn non_block_causal_is_refused() {
        let c = with_dflash_key(&valid_config(), "draft_block_causal", json!(false));
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("block-causal"), "{err}");
    }

    #[test]
    fn non_serial_table_second_pass_is_refused() {
        let mut c = valid_config();
        c["code_bridge_second_pass"]["mode"] = json!("chain");
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("serial_table"), "{err}");
    }

    #[test]
    fn first_exact_must_be_layers_minus_one() {
        let mut c = valid_config();
        c["code_bridge_second_pass"]["first_exact"] = json!(3);
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("first_exact"), "{err}");
    }

    #[test]
    fn non_exact_sender_refine_is_refused() {
        let c = with_dflash_key(
            &valid_config(),
            "selected_prefix_last_write_refine",
            json!({"kind": "approx_sender", "head_topk": 16}),
        );
        let err = validate_dspine_config(&c).unwrap_err();
        assert!(err.contains("exact_sender"), "{err}");
    }

    #[test]
    fn head_topk_bounds_and_pow2() {
        assert_eq!(validate_head_topk(2), Ok(()));
        assert_eq!(validate_head_topk(16), Ok(()));
        assert_eq!(validate_head_topk(64), Ok(()));
        assert!(validate_head_topk(1).unwrap_err().contains("[2, 64]"));
        assert!(validate_head_topk(128).unwrap_err().contains("[2, 64]"));
        assert!(validate_head_topk(3).unwrap_err().contains("power of two"));
        assert!(validate_head_topk(48).unwrap_err().contains("power of two"));
    }

    #[test]
    fn version_scalars_must_be_one() {
        let ok = vec![
            ("draft_block_causal_version".to_string(), 1),
            ("last_write_refine_version".to_string(), 1),
            (
                "selected_prefix_bridge.independent_write_version".to_string(),
                1,
            ),
            (
                "selected_prefix_bridge.terminal_disabled_version".to_string(),
                1,
            ),
        ];
        assert_eq!(validate_version_scalars(&ok), Ok(()));
        let bad = vec![("last_write_refine_version".to_string(), 2)];
        let err = validate_version_scalars(&bad).unwrap_err();
        assert!(err.contains("expected 1"), "{err}");
    }

    #[test]
    fn fc_k_must_be_layers_times_hidden() {
        assert_eq!(validate_fc_k(25600, 5, 5120), Ok(()));
        let err = validate_fc_k(20480, 5, 5120).unwrap_err();
        assert!(err.contains("fc.weight"), "{err}");
    }

    #[test]
    fn bridge_shapes_are_cross_checked_against_the_contracts() {
        let c = validate_dspine_config(&valid_config()).unwrap();
        let hidden = 5120usize;
        let layers = 5usize;
        let vocab = 248320usize;
        let meta = |shape: Vec<usize>| TensorMeta {
            dtype: "BF16".into(),
            shape,
            data_offsets: [0, 0],
        };
        let good: HashMap<String, TensorMeta> = [
            ("selected_prefix_bridge.codebook", vec![vocab, 1024]),
            ("selected_prefix_bridge.read_weight", vec![1024, hidden]),
            ("selected_prefix_bridge.early_read_weights", vec![4, 1024, hidden]),
            ("selected_prefix_bridge.feature_write_weights", vec![layers, hidden, 512]),
            ("selected_prefix_bridge.gate.weight", vec![512, 1024]),
            ("selected_prefix_bridge.gate.bias", vec![512]),
            ("selected_prefix_bridge.predecessor_gate.weight", vec![512, 1024]),
            ("selected_prefix_bridge.prev.weight", vec![512, 1024]),
            ("selected_prefix_bridge.out.weight", vec![1024, 512]),
        ]
        .into_iter()
        .map(|(n, s)| (n.to_string(), meta(s)))
        .collect();
        assert_eq!(
            validate_bridge_shapes(&c, hidden, layers, vocab, &good),
            Ok(())
        );
        let mut wrong_codebook = good.clone();
        wrong_codebook.insert(
            "selected_prefix_bridge.codebook".into(),
            meta(vec![151936, 1024]),
        );
        let err = validate_bridge_shapes(&c, hidden, layers, vocab, &wrong_codebook).unwrap_err();
        assert!(err.contains("codebook"), "{err}");
        let mut missing = good.clone();
        missing.remove("selected_prefix_bridge.prev.weight");
        let err = validate_bridge_shapes(&c, hidden, layers, vocab, &missing).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn classification_rules_match_the_tier_surface() {
        assert!(is_norm_tensor("layers.0.self_attn.q_norm.weight"));
        assert!(is_norm_tensor("norm.weight"));
        assert!(!is_norm_tensor("fc.weight"));
        assert!(is_bridge_tensor("selected_prefix_bridge.codebook"));
        assert!(!is_bridge_tensor("layers.0.self_attn.q_proj.weight"));
        assert!(is_version_scalar("draft_block_causal_version"));
        assert!(is_version_scalar(
            "selected_prefix_bridge.terminal_disabled_version"
        ));
        assert!(!is_version_scalar("fc.weight"));
        assert!(is_forbidden_tensor("selected_prefix_gru.weight"));
        assert!(is_forbidden_tensor("embed_proj.weight"));
        assert!(is_forbidden_tensor("candidate_selector.codebook"));
        assert!(!is_forbidden_tensor("selected_prefix_bridge.codebook"));
    }

    #[test]
    fn bf16_round_trip_is_bit_exact_for_bf16_sourced_values() {
        // Finite values (all a weight tensor can hold) round-trip exactly.
        // NaN payloads are exempt: the bf16 crate canonicalizes them on
        // to_f32 (0xffab → 0xffeb), which is irrelevant for weights.
        for bits in [0x0000u16, 0x8000, 0x3f80, 0xbc00, 0x7f7f, 0x0081] {
            let f = bf16_to_f32(bits);
            assert_eq!(f32_to_bf16_bits(f), bits, "bits={bits:#06x}");
        }
        assert!(bf16_to_f32(0xffab).is_nan());
        let nan_back = f32_to_bf16_bits(bf16_to_f32(0xffab));
        assert!(bf16_to_f32(nan_back).is_nan());
        // Zero (including negative zero) must survive: out.weight all-zero
        // detection depends on it.
        assert_eq!(bf16_to_f32(0x8000), -0.0);
        assert!(bf16_to_f32(0x8000) == 0.0);
    }
}

#[cfg(test)]
mod mq4v2_symmetric_tests {
    use super::*;

    #[test]
    fn symmetric_converter_headers_and_codes_match_shared_grid() {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        for zero in [true, false] {
            let weights: Vec<f32> = (0..1024)
                .map(|i| {
                    if zero {
                        0.0
                    } else {
                        (i as f32 * 0.017).sin() * 0.42
                    }
                })
                .collect();
            let blob = quantize_mq4g256v2(&weights, 2, 512, &signs1, &signs2, true);
            let stride = 8 + 256 * 4 / 8;
            assert_eq!(blob.len(), 4 * stride);
            for (input, packed) in weights.chunks_exact(256).zip(blob.chunks_exact(stride)) {
                let mut rotated = [0.0f32; 256];
                rotated.copy_from_slice(input);
                cpu_fwht_256(&mut rotated, &signs1, &signs2);
                let mut codes = [0u8; 256];
                for h in 0..2 {
                    let off = h * 128;
                    let (scale, zero) = hipfire_quantize::mq_symmetric::quantize_symmetric_half(
                        &rotated[off..off + 128],
                        15,
                        &mut codes[off..off + 128],
                    );
                    assert_eq!(&packed[h * 4..h * 4 + 2], &scale.to_le_bytes());
                    assert_eq!(&packed[h * 4 + 2..h * 4 + 4], &zero.to_le_bytes());
                }
                for (i, &expected) in codes.iter().enumerate() {
                    let mut code = 0;
                    for b in 0..4 {
                        let bit = i * 4 + b;
                        code |= ((packed[8 + bit / 8] >> (bit % 8)) & 1) << b;
                    }
                    assert_eq!(code, expected, "index={i}");
                }
            }
        }
    }
}
