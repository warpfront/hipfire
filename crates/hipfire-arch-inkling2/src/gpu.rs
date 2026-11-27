// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU weight bundle + per-visit decode state for Inkling-2-Mini (arch 24).
//!
//! Brief 01 owns this file. It uploads the full weight set to VRAM and
//! allocates the per-visit decode state, with a correct free path. No
//! forward, no kernels, no generation — the forward lives in brief 03's
//! `gpu_forward.rs`, which consumes these structs read-only.
//!
//! Load path mirrors `hipfire-arch-cohere2moe` (`load_wt` / `wt_from_raw` /
//! `load_f32` / `LoadTransaction`): quantized tensors stay packed on the GPU,
//! one-dimensional keeps land as F32, and a mid-load failure frees everything
//! already allocated via [`LoadTx`]'s `Drop`.
//!
//! Binding decisions from the brief (do not "simplify" these away):
//! * `down_proj` is `[2304, 1152]` (`K % 256 != 0`) on disk as a flat 1D
//!   stream. It is decoded to F32, each row zero-padded `1152 -> 1280`,
//!   forward-FWHT-256-rotated per 256-group and re-encoded to MQ4G256V2,
//!   uploaded as legal `[2304, 1280]` (`2304*5*136 = 1,566,720` B, +0.45 GiB
//!   model-wide). Brief 03 sees `K = 1280` only.
//! * `embed.weight` stays packed on the host (246 MB qt44, never widened to
//!   F32); one row (9 groups x 136 B = 1,224 B) decodes per token on demand.
//! * Short conv weights are pre-transformed at upload to `[kernel, channels]`
//!   F32 with row 0 the oldest tap and row `kernel-1` the current token; full
//!   layers' `static_w` is flipped along the kernel axis once here, so brief
//!   03 never flips/transposes per visit.
//! * KV dtype is [`INKLING2_KV_DTYPE`] (F16); brief 03 reads that constant.
//! * Indexed by visit (0..48), never by physical layer.

use crate::config::Inkling2Config;
use crate::weights::Inkling2Weights;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::inkling2_attention::{
    INKLING2_ATTN_HEADS, INKLING2_ATTN_HEAD_DIM, INKLING2_ATTN_MAX_SPLITS,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Constants (brief 03 reads the KV dtype + conv width) ─────────────────

/// KV element type for every per-visit K/V buffer. F16: SWA-aware total
/// 1,241,513,984 B = 1.156 GiB at max_seq 32768 (the plan's f16 row); fits
/// the 24 GiB budget with the +0.45 GiB down-proj repack inside ~14.5 GiB.
pub const INKLING2_KV_DTYPE: DType = DType::F16;

/// Unified conv state width: `2*n_embd_kv + 2*n_embd` = 2*1024 + 2*2304.
pub const INKLING2_CONV_WIDTH: usize = 6656;

/// Ragged-K repack: logical (on-disk) vs physical (on-device) down_proj K.
pub const DOWN_K_LOGICAL: usize = 1152;
pub const DOWN_K_PADDED: usize = 1280;

/// SWA sliding-window rows kept per SWA visit; full visits keep `max_seq`.
/// SWA rows are a physical ring: past this many tokens the oldest row is
/// evicted on write (slot `pos % rows`, true positions in `visit_pos_tags`).
pub const SWA_WINDOW_ROWS: usize = 1024;

/// Physical slot for `pos` in a ring of `rows` rows. Below the ring size
/// this is the identity (`slot == pos`).
pub fn swa_slot(pos: usize, rows: usize) -> usize {
    pos % rows
}

/// Trailing window length held at `pos` in a ring of `rows` rows.
pub fn swa_window_len(pos: usize, rows: usize) -> usize {
    (pos + 1).min(rows)
}

/// Logical window at `pos`: per-row physical slots in oldest-to-newest order
/// plus the relative-bias distances, both read from the position tags (the
/// slot index no longer implies position once the ring wraps). Fails closed
/// on a tag mismatch (out-of-order write) instead of mis-addressing keys.
pub fn swa_gather_map(
    tags: &[u32],
    pos: usize,
    rows: usize,
) -> Result<(Vec<i32>, Vec<i32>), String> {
    if rows == 0 {
        return Err("inkling2: SWA ring has 0 rows".to_string());
    }
    if tags.len() != rows {
        return Err(format!(
            "inkling2: SWA tags len {} != rows {rows}",
            tags.len()
        ));
    }
    let n_kv = swa_window_len(pos, rows);
    let start = pos + 1 - n_kv;
    let mut slots = Vec::with_capacity(n_kv);
    let mut dists = Vec::with_capacity(n_kv);
    for k in 0..n_kv {
        let key_pos = start + k;
        let slot = swa_slot(key_pos, rows);
        let tag = tags[slot] as usize;
        if tag != key_pos {
            return Err(format!(
                "inkling2: SWA tag mismatch at slot {slot}: tag {tag} != key_pos {key_pos}"
            ));
        }
        slots.push(slot as i32);
        dists.push((pos - key_pos) as i32);
    }
    Ok((slots, dists))
}

/// Loader slack carried in the VRAM accounting (qwen35 precedent).
pub const VRAM_SLACK_BYTES: u64 = 128 << 20;

// ── Load helpers (cohere2moe shape) ───────────────────────────────────────

/// Quant type -> GPU dtype. cohere2moe's table plus `44 => MQ4G256V2` (every
/// dense tensor in `inkling2-mini-mq4.hfq`) and `47 => MQ6G256V2` (admitted,
/// unused by this artifact).
fn dtype_for_quant_type(qt: u8) -> Result<DType, String> {
    match qt {
        1 => Ok(DType::F16),
        2 => Ok(DType::F32),
        16 => Ok(DType::BF16),
        3 => Ok(DType::Q8_0),
        6 => Ok(DType::HFQ4G256),
        8 => Ok(DType::HFQ6G256),
        13 => Ok(DType::MQ4G256),
        15 => Ok(DType::MQ6G256),
        17 => Ok(DType::MQ3G256),
        18 => Ok(DType::MQ2G256),
        19 => Ok(DType::MQ2G256Lloyd),
        20 => Ok(DType::MQ3G256Lloyd),
        30 => Ok(DType::MQ4G256Lloyd),
        44 => Ok(DType::MQ4G256V2),
        47 => Ok(DType::MQ6G256V2),
        other => Err(format!("inkling2: unsupported quant_type {other}")),
    }
}

/// Exact encoded byte count of one logical row; rejects ragged K the way
/// `cohere2moe::validate_raw_layout` does (ragged tensors go through the
/// dedicated repack path instead).
fn encoded_row_bytes(dtype: DType, k: usize) -> Result<usize, String> {
    let group_k = match dtype {
        DType::F32 | DType::F16 | DType::BF16 => 1,
        DType::Q8_0 => 32,
        DType::HFQ4G256
        | DType::MQ4G256
        | DType::MQ4G256V2
        | DType::HFQ6G256
        | DType::MQ6G256
        | DType::MQ6G256V2
        | DType::MQ3G256
        | DType::MQ2G256
        | DType::MQ2G256Lloyd
        | DType::MQ3G256Lloyd
        | DType::MQ4G256Lloyd => 256,
        other => return Err(format!("inkling2: no encoded row formula for {other:?}")),
    };
    if k % group_k != 0 {
        return Err(format!(
            "{dtype:?} weights require K divisible by {group_k}, got K={k}"
        ));
    }
    dtype
        .row_bytes(k)
        .ok_or_else(|| format!("{dtype:?} encoded row byte length for K={k} overflows"))
}

/// Validate + upload one packed tensor. Returns the weight and the exact GPU
/// bytes (the accounting source of truth, not allocator rounding).
fn wt_from_raw(
    gpu: &mut Gpu,
    name: &str,
    qt: u8,
    data: &[u8],
    m: usize,
    k: usize,
) -> Result<(WeightTensor, u64), String> {
    let dtype = dtype_for_quant_type(qt)?;
    let row_stride = encoded_row_bytes(dtype, k)?;
    let expected = row_stride
        .checked_mul(m)
        .ok_or_else(|| format!("inkling2: {name} byte count overflows"))?;
    if data.len() != expected {
        return Err(format!(
            "inkling2: {name} bytes {} != {m} rows x {row_stride} ({expected} total)",
            data.len()
        ));
    }
    let buf = gpu
        .upload_raw(data, &[data.len()])
        .map_err(|e| format!("inkling2: upload {name}: {e:?}"))?;
    Ok((
        WeightTensor {
            buf,
            gpu_dtype: dtype,
            m,
            k,
            row_stride,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        },
        data.len() as u64,
    ))
}

/// Load a 1D/raw F16/F32/Q8 keep -> F32 tensor with the exact stored shape.
fn load_f32(
    gpu: &mut Gpu,
    weights: &Inkling2Weights,
    name: &str,
    shape: &[usize],
) -> Result<(GpuTensor, u64), String> {
    let t = weights
        .get(name)
        .ok_or_else(|| format!("inkling2: missing tensor {name}"))?;
    let vals: Vec<f32> = match t.info.quant_type {
        1 => t
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        2 => t
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        3 => hipfire_runtime::llama::dequantize_q8_0(&t.data, shape.iter().product::<usize>()),
        qt => {
            return Err(format!(
                "inkling2: expected F16/F32/Q8 keep for {name}, got qt={qt}"
            ))
        }
    };
    let want: usize = shape.iter().product();
    if vals.len() != want {
        return Err(format!(
            "inkling2: {name} decoded to {} values, expected {want}",
            vals.len()
        ));
    }
    let bytes = (vals.len() * 4) as u64;
    let tensor = gpu
        .upload_f32(&vals, shape)
        .map_err(|e| format!("inkling2: upload {name}: {e:?}"))?;
    Ok((tensor, bytes))
}

// ── FWHT-256 + MQ4G256V2 repack (down_proj ragged-K) ──────────────────────
// Mirrors `hipfire-quantize/src/quant_fwht.rs::quantize_mq4g256v2_impl`
// (asymmetric): seeds 42/1042, signs1 -> butterfly -> 1/16 -> signs2, per-half
// f16 (scale, zero), `floor((w-z)/st + 0.5)` codes, degenerate half -> all 0.

fn fwht_signs(seed: u32) -> [f32; 256] {
    let mut state = seed;
    let mut out = [0.0f32; 256];
    for v in out.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        *v = if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 };
    }
    out
}

fn fwht256_forward(group: &mut [f32; 256], s1: &[f32; 256], s2: &[f32; 256]) {
    for i in 0..256 {
        group[i] *= s1[i];
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
    for i in 0..256 {
        group[i] *= 0.0625 * s2[i];
    }
}

/// Repack one `1152`-wide F32 row to legal MQ4G256V2 bytes (`5 x 136` B):
/// zero-pad to 1280, forward-rotate each 256-group, per-half min/max grid.
fn repack_row_mq4g256v2(row1152: &[f32]) -> Vec<u8> {
    debug_assert_eq!(row1152.len(), DOWN_K_LOGICAL);
    let s1 = fwht_signs(42);
    let s2 = fwht_signs(1042);
    let mut padded = [0.0f32; DOWN_K_PADDED];
    padded[..DOWN_K_LOGICAL].copy_from_slice(row1152);
    let mut out = vec![0u8; 5 * 136];
    for (g, chunk) in out.chunks_exact_mut(136).enumerate() {
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&padded[g * 256..(g + 1) * 256]);
        fwht256_forward(&mut group, &s1, &s2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut codes = [0u8; 256];
        for half in 0..2 {
            let v = &group[half * 128..(half + 1) * 128];
            let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
            scales[half] = if hi == lo {
                0
            } else {
                half::f16::from_f32(step).to_bits()
            };
            zeros[half] = half::f16::from_f32(lo).to_bits();
            let st = half::f16::from_bits(scales[half]).to_f32();
            let z = half::f16::from_bits(zeros[half]).to_f32();
            if hi != lo && step != 0.0 && st != 0.0 {
                let inv = 1.0 / st;
                for i in 0..128 {
                    codes[half * 128 + i] = ((group[half * 128 + i] - z) * inv + 0.5)
                        .floor()
                        .clamp(0.0, 15.0) as u8;
                }
            }
        }
        chunk[..2].copy_from_slice(&scales[0].to_le_bytes());
        chunk[2..4].copy_from_slice(&zeros[0].to_le_bytes());
        chunk[4..6].copy_from_slice(&scales[1].to_le_bytes());
        chunk[6..8].copy_from_slice(&zeros[1].to_le_bytes());
        for i in 0..128 {
            chunk[8 + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        }
    }
    out
}

/// Repack a full `down_proj` tensor (`[m, 1152]` flat 1D qt13 stream) to
/// legal `[m, 1280]` MQ4G256V2 bytes. Rows are independent: rayon over rows.
fn repack_down_proj(qt: u8, data: &[u8], m: usize) -> Result<Vec<u8>, String> {
    use rayon::prelude::*;
    if qt != 13 {
        return Err(format!(
            "inkling2: down_proj repack expects flat qt13, got qt={qt}"
        ));
    }
    let n = m
        .checked_mul(DOWN_K_LOGICAL)
        .ok_or_else(|| "inkling2: down_proj element count overflows".to_string())?;
    if data.len() != n / 256 * 136 {
        return Err(format!(
            "inkling2: down_proj flat bytes {} != {m}x{DOWN_K_LOGICAL} stream",
            data.len()
        ));
    }
    let vals = hipfire_runtime::weight_backend::dequantize_weight_to_f32(qt, data, n);
    if vals.len() != n {
        return Err(format!(
            "inkling2: down_proj decoded to {} values, expected {n}",
            vals.len()
        ));
    }
    let rows: Vec<Vec<u8>> = vals
        .par_chunks_exact(DOWN_K_LOGICAL)
        .map(repack_row_mq4g256v2)
        .collect();
    if rows.len() != m {
        return Err(format!(
            "inkling2: down_proj repacked {} rows, expected {m}",
            rows.len()
        ));
    }
    Ok(rows.concat())
}

// ── Conv pre-transform ────────────────────────────────────────────────────
// Upload layout (brief 03's contract): `[kernel, channels]` row-major F32,
// row 0 scales the OLDEST tap, row `kernel-1` the current token — the same
// tap convention as `oracle::convolve_static` (`weight[l*n_ch+c]` multiplies
// `x[t+l]`, `x = [past_0..past_{k-2}, current]`). SWA `[C,1,K]` transposes;
// full-layer `static_w` `[K,C]` flips along the kernel axis once here.

fn transform_conv(
    shape: &[u32],
    vals: Vec<f32>,
    kernel: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    let n_ch = vals.len() / kernel;
    if vals.len() != n_ch * kernel {
        return Err(format!("inkling2: {name} length mismatch"));
    }
    let mut out = vec![0.0f32; vals.len()];
    if shape.len() == 3 {
        // [channels, 1, kernel] -> [kernel, channels], no flip.
        if shape[0] as usize != n_ch || shape[2] as usize != kernel {
            return Err(format!(
                "inkling2: {name} shape {shape:?} vs kernel {kernel}"
            ));
        }
        for c in 0..n_ch {
            for k in 0..kernel {
                out[k * n_ch + c] = vals[c * kernel + k];
            }
        }
    } else if shape.len() == 2 && shape[0] as usize == kernel {
        // static_w [kernel, channels]: flip the kernel axis once.
        if shape[1] as usize != n_ch {
            return Err(format!(
                "inkling2: {name} shape {shape:?} vs kernel {kernel}"
            ));
        }
        for k in 0..kernel {
            for c in 0..n_ch {
                out[k * n_ch + c] = vals[(kernel - 1 - k) * n_ch + c];
            }
        }
    } else {
        return Err(format!("inkling2: {name} unexpected conv shape {shape:?}"));
    }
    Ok(out)
}

// ── Transactional owner registry ──────────────────────────────────────────

enum PendingOwner {
    Tensor(GpuTensor),
    Weight(WeightTensor),
}

/// Mid-load failure frees everything already allocated (cohere2moe pattern).
struct LoadTx {
    gpu: *mut Gpu,
    owners: Vec<Option<PendingOwner>>,
    committed: bool,
}

impl LoadTx {
    fn new(gpu: &mut Gpu) -> Self {
        Self {
            gpu,
            owners: Vec::new(),
            committed: false,
        }
    }

    fn hold_tensor(&mut self, tensor: GpuTensor) -> usize {
        self.owners.push(Some(PendingOwner::Tensor(tensor)));
        self.owners.len() - 1
    }

    fn hold_weight(&mut self, weight: WeightTensor) -> usize {
        self.owners.push(Some(PendingOwner::Weight(weight)));
        self.owners.len() - 1
    }

    fn take_tensor(&mut self, slot: usize) -> Result<GpuTensor, String> {
        match self.owners.get_mut(slot).and_then(Option::take) {
            Some(PendingOwner::Tensor(t)) => Ok(t),
            _ => Err(format!("inkling2: load slot {slot} is not a tensor")),
        }
    }

    fn take_weight(&mut self, slot: usize) -> Result<WeightTensor, String> {
        match self.owners.get_mut(slot).and_then(Option::take) {
            Some(PendingOwner::Weight(w)) => Ok(w),
            _ => Err(format!("inkling2: load slot {slot} is not a weight")),
        }
    }

    /// Mutable access to a still-pending weight (typed-XTS sidecar attach).
    fn weight_mut(&mut self, slot: usize) -> Result<&mut WeightTensor, String> {
        match self.owners.get_mut(slot) {
            Some(Some(PendingOwner::Weight(w))) => Ok(w),
            _ => Err(format!(
                "inkling2: load slot {slot} is not a pending weight"
            )),
        }
    }

    fn commit(&mut self) {
        self.committed = true;
        self.owners.clear();
    }
}

impl Drop for LoadTx {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let gpu = unsafe { &mut *self.gpu };
        for owner in self.owners.drain(..).rev().flatten() {
            match owner {
                PendingOwner::Tensor(t) => {
                    let _ = gpu.free_tensor(t);
                }
                PendingOwner::Weight(w) => w.free_all(gpu),
            }
        }
    }
}

// ── Bundle types ──────────────────────────────────────────────────────────

/// One loop controller triple: `down [rank, 2*n_embd]`, `gain`/`forget`
/// `[n_embd, rank]`, verbatim HFQ layout (matches `oracle::control`).
pub struct Inkling2Controller {
    pub down: GpuTensor,
    pub gain: GpuTensor,
    pub forget: GpuTensor,
}

impl Inkling2Controller {
    fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.down);
        let _ = gpu.free_tensor(self.gain);
        let _ = gpu.free_tensor(self.forget);
    }
}

pub struct Inkling2Expert {
    pub gate_up: WeightTensor, // [2*inter, n_embd], original qt
    pub down: WeightTensor,    // [n_embd, 1280] MQ4G256V2 (repacked)
}

impl Inkling2Expert {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.gate_up.free_all(gpu);
        self.down.free_all(gpu);
    }
}

/// Typed-layout (`qt44-padded-awq-v1`) decode tables of one physical layer,
/// built ONCE at load and owned by the layer (freed with it, on unload and on
/// a failed load). Legacy layers carry `None`.
///
/// The per-expert F32 AWQ vectors live packed in two row-major tensors
/// (`gu_awq` `[n_experts, n_embd]`, `down_awq` `[n_experts, down_k_physical]`,
/// the down tail exactly 1.0); the pointer tables are `n_experts` `u64`s of raw
/// device addresses, in the form the indexed launchers expect.
pub struct Inkling2XtsLayer {
    pub gu_awq: GpuTensor,
    pub down_awq: GpuTensor,
    /// entry e -> expert e's `[2*inter, n_embd]` qt44 gate_up buffer.
    pub gu_ptrs: GpuTensor,
    /// entry e -> expert e's `[n_embd, down_k_physical]` qt44 down buffer.
    pub down_ptrs: GpuTensor,
    /// entry e -> row e of `gu_awq` (`F32[n_embd]`).
    pub gu_awq_ptrs: GpuTensor,
    /// entry e -> row e of `down_awq` (`F32[down_k_physical]`).
    pub down_awq_ptrs: GpuTensor,
}

impl Inkling2XtsLayer {
    fn free_gpu(self, gpu: &mut Gpu) {
        for t in [
            self.gu_awq,
            self.down_awq,
            self.gu_ptrs,
            self.down_ptrs,
            self.gu_awq_ptrs,
            self.down_awq_ptrs,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

pub struct Inkling2GpuLayer {
    pub wq: WeightTensor,
    pub wk: WeightTensor,
    pub wv: WeightTensor,
    pub wr: WeightTensor,
    pub wgate: WeightTensor,
    pub wo: WeightTensor,
    pub sink: GpuTensor,                 // [n_head, n_embd] F32
    pub rel_table: GpuTensor,            // [n_rel, n_extent] F32
    pub q_clip: GpuTensor,               // [n_head, head_dim] F32
    pub rel_clip: GpuTensor,             // [n_head, n_rel] F32
    pub attn_norm: Option<GpuTensor>,    // SWA layers only
    pub q_input_norm: Option<GpuTensor>, // full layers only
    pub k_input_norm: Option<GpuTensor>,
    pub v_input_norm: Option<GpuTensor>,
    pub mlp_norm: GpuTensor,
    /// `[kernel, channels]` F32, row 0 = oldest tap (see module docs).
    pub k_conv: GpuTensor,
    pub v_conv: GpuTensor,
    pub k_proj: Option<GpuTensor>, // full layers only, verbatim shape
    pub v_proj: Option<GpuTensor>,
    pub attn_post_conv: GpuTensor, // [kernel, n_embd] F32
    pub mlp_post_conv: GpuTensor,
    /// Loop controllers; `Some` on core layers only (6..18 carry them).
    pub controllers: Option<[Inkling2Controller; 4]>,
    /// `attn_queries.{phys-1}`; `None` on layer 0 (no query feeds it).
    pub attn_query: Option<GpuTensor>,
    pub mlp_query: GpuTensor,
    pub router: WeightTensor,           // [n_experts, n_embd]
    pub router_bias: Option<GpuTensor>, // [n_experts] F32
    pub experts: Vec<Inkling2Expert>,   // n_experts
    /// `Some` iff the file is the typed XTS layout; see [`Inkling2XtsLayer`].
    pub xts: Option<Inkling2XtsLayer>,
}

impl Inkling2GpuLayer {
    fn free_gpu(self, gpu: &mut Gpu) {
        self.wq.free_all(gpu);
        self.wk.free_all(gpu);
        self.wv.free_all(gpu);
        self.wr.free_all(gpu);
        self.wgate.free_all(gpu);
        self.wo.free_all(gpu);
        let _ = gpu.free_tensor(self.sink);
        let _ = gpu.free_tensor(self.rel_table);
        let _ = gpu.free_tensor(self.q_clip);
        let _ = gpu.free_tensor(self.rel_clip);
        if let Some(t) = self.attn_norm {
            let _ = gpu.free_tensor(t);
        }
        for t in [self.q_input_norm, self.k_input_norm, self.v_input_norm] {
            if let Some(t) = t {
                let _ = gpu.free_tensor(t);
            }
        }
        let _ = gpu.free_tensor(self.mlp_norm);
        let _ = gpu.free_tensor(self.k_conv);
        let _ = gpu.free_tensor(self.v_conv);
        if let Some(t) = self.k_proj {
            let _ = gpu.free_tensor(t);
        }
        if let Some(t) = self.v_proj {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.attn_post_conv);
        let _ = gpu.free_tensor(self.mlp_post_conv);
        if let Some(cs) = self.controllers {
            for c in cs {
                c.free_gpu(gpu);
            }
        }
        if let Some(t) = self.attn_query {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.mlp_query);
        self.router.free_all(gpu);
        if let Some(t) = self.router_bias {
            let _ = gpu.free_tensor(t);
        }
        for e in self.experts {
            e.free_gpu(gpu);
        }
        if let Some(x) = self.xts {
            x.free_gpu(gpu);
        }
    }
}

/// Packed `embed.weight` retained on the host (row-lookup side is HOST):
/// 246 MB qt44, never widened to F32 (1.85 GiB would blow the weight budget).
pub struct EmbedPacked {
    pub quant_type: u8,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

impl EmbedPacked {
    /// Decode one row to 2304 F32 on demand (~9 KB of device traffic later).
    pub fn row_f32(&self, token: u32) -> Result<Vec<f32>, String> {
        if self.quant_type != 44 {
            return Err(format!(
                "inkling2: embed row decode expects qt44, got qt={}",
                self.quant_type
            ));
        }
        let row = token as usize;
        if row >= self.rows {
            return Err(format!(
                "inkling2: token {row} out of embed range {}",
                self.rows
            ));
        }
        // K = 2304 = 9 x 256 groups, rows are group-aligned.
        let groups_per_row = self.cols / 256;
        let stride = groups_per_row * 136;
        let off = row * stride;
        let slice = self
            .bytes
            .get(off..off + stride)
            .ok_or_else(|| "inkling2: embed packed bytes truncated".to_string())?;
        Ok(hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            self.quant_type,
            slice,
            self.cols,
        ))
    }
}

/// Exact VRAM census of one load: every byte below was uploaded by name.
pub struct VramSummary {
    pub weight_bytes: u64,
    pub weight_tensors: usize,
    pub kv_bytes: u64,
    pub conv_bytes: u64,
    pub down_pad_extra_bytes: u64,
    pub max_seq: usize,
    pub full_visits: usize,
    pub swa_visits: usize,
}

impl VramSummary {
    fn gib(b: u64) -> f64 {
        b as f64 / (1u64 << 30) as f64
    }

    pub fn total_bytes(&self) -> u64 {
        self.weight_bytes + self.kv_bytes + self.conv_bytes + VRAM_SLACK_BYTES
    }

    pub fn text(&self) -> String {
        format!(
            "inkling2 gpu load summary: weights {:.3} GiB ({} tensors, incl. down_proj pad +{:.3} GiB), \
             KV {:.3} GiB ({} full + {} SWA visits, max_seq {}), conv {:.4} GiB, slack {:.3} GiB -> \
             total {:.3} GiB (fits 24 GiB: {})",
            Self::gib(self.weight_bytes),
            self.weight_tensors,
            Self::gib(self.down_pad_extra_bytes),
            Self::gib(self.kv_bytes),
            self.full_visits,
            self.swa_visits,
            self.max_seq,
            Self::gib(self.conv_bytes),
            Self::gib(VRAM_SLACK_BYTES),
            Self::gib(self.total_bytes()),
            if self.total_bytes() < (24u64 << 30) {
                "yes"
            } else {
                "NO"
            },
        )
    }
}

/// Whether one Qwen-class activation-quantized route is usable by a typed-XTS
/// load, as resolved from the existing config/flag readers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XtsRouteState {
    /// Capability holds (flags, device, symmetric verdict).
    Admitted,
    /// Capability does not hold and nothing forced it: informational.
    NotAdmitted(String),
    /// The user explicitly forced the route and it cannot run.
    Refused(String),
}

impl XtsRouteState {
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted)
    }
}

/// Raw knob/device facts [`resolve_xts_routes`] decides from (pure data so the
/// resolution is CPU-testable). Every value comes from the existing readers:
/// `FeatureFlags` (`kernel.iu4_prefill` / `HIPFIRE_IU4_PREFILL`,
/// `HIPFIRE_A8_PREFILL` / `kernel.gfx11_a8_prefill`, `HIPFIRE_QWEN4_MOE_EP_A8`),
/// `Gpu::moe_sym_iu4_requested` and `hipfire_config::developer_typed_bool`
/// (`developer.moe_sym_iu4` / `HIPFIRE_MOE_SYM_IU4`), `Gpu::qwen4_moe_down_a8_requested`.
#[derive(Clone, Debug)]
pub struct XtsRouteInputs {
    pub arch: String,
    /// `kernel.iu4_prefill` as set (`None` = unset).
    pub iu4_prefill_flag: Option<bool>,
    /// `FeatureFlags::iu4_prefill_enabled`: on unless killed, on an arch with an IU4 consumer.
    pub iu4_prefill_enabled: bool,
    /// `developer.moe_sym_iu4` as set (`None` = unset).
    pub moe_sym_iu4_flag: Option<bool>,
    /// `Gpu::moe_sym_iu4_requested`: not killed and the device has the symmetric IU4 MoE objects.
    pub moe_sym_iu4_requested: bool,
    /// `FeatureFlags::a8_prefill_enabled` (opt-in, arch scoped).
    pub a8_prefill_enabled: bool,
    /// Either raw A8 opt-in set (`a8_prefill` / `gfx11_a8_prefill`).
    pub a8_prefill_forced: bool,
    /// `Gpu::qwen4_moe_down_a8_requested`.
    pub qwen4_down_a8_requested: bool,
    /// `HIPFIRE_QWEN4_MOE_EP_A8`.
    pub qwen4_ep_a8: Option<bool>,
}

/// Route-capability record of a typed-XTS load: exposed on
/// [`Inkling2GpuWeights::xts_route_capability`] for a future blocked
/// (multi-row) forward. The scalar decode forward executes none of these
/// routes; a state here is capability, not execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XtsRouteCapability {
    pub n_layers: usize,
    /// Layers whose attention qt44 halves are all symmetric.
    pub dense_symmetric_layers: usize,
    /// Layers whose routed-expert qt44 halves are all symmetric.
    pub expert_symmetric_layers: usize,
    /// Layers symmetric in both (24/24 expected).
    pub symmetric_layers: usize,
    /// The standalone unembed is symmetric.
    pub unembed_symmetric: bool,
    pub awq_sidecars: usize,
    pub down_k_logical: usize,
    pub down_k_physical: usize,
    /// `kernel.iu4_prefill` W4A4 dense/expert prefill.
    pub iu4_prefill: XtsRouteState,
    /// `developer.moe_sym_iu4` symmetric IU4 grouped routed-expert prefill.
    pub moe_sym_iu4: XtsRouteState,
    /// A8 expert route family (`HIPFIRE_A8_PREFILL`, `kernel.gfx11_a8_prefill`,
    /// `HIPFIRE_QWEN4_MOE_DOWN_A8`, `HIPFIRE_QWEN4_MOE_EP_A8`).
    pub moe_a8: XtsRouteState,
}

impl XtsRouteCapability {
    /// The one load line (stderr). Routes are listed as admitted by capability;
    /// an explicitly forced route that cannot run is reported as refused.
    pub fn load_line(&self) -> String {
        let routes = [
            ("iu4_prefill", &self.iu4_prefill),
            ("moe_sym_iu4", &self.moe_sym_iu4),
            ("moe_a8", &self.moe_a8),
        ];
        let admitted: Vec<&str> = routes
            .iter()
            .filter(|(_, s)| s.is_admitted())
            .map(|(n, _)| *n)
            .collect();
        let refused: Vec<String> = routes
            .iter()
            .filter_map(|(n, s)| match s {
                XtsRouteState::Refused(why) => Some(format!("{n} ({why})")),
                _ => None,
            })
            .collect();
        format!(
            "inkling2 xts: symmetric {}/{} layers (dense+experts), awq {} sidecars; \
             routes admitted by capability: {}; {}executing: scalar decode (no multi-row consumer yet)",
            self.symmetric_layers,
            self.n_layers,
            self.awq_sidecars,
            if admitted.is_empty() {
                "none".to_string()
            } else {
                admitted.join(", ")
            },
            if refused.is_empty() {
                String::new()
            } else {
                format!("refused (forced, cannot run): {}; ", refused.join(", "))
            },
        )
    }
}

/// Pure route resolution behind [`XtsRouteCapability`]. `dense_all` / `experts_all`:
/// every layer's dense / routed-expert qt44 halves verified symmetric.
/// `moe_geometry_ok`: top-k <= 8, hidden % 256 == 0 and intermediate % 128 == 0.
pub fn resolve_xts_routes(
    dense_all: bool,
    experts_all: bool,
    moe_geometry_ok: bool,
    f: &XtsRouteInputs,
) -> (XtsRouteState, XtsRouteState, XtsRouteState) {
    let pick = |capable: bool, forced: bool, why: String| {
        if capable {
            XtsRouteState::Admitted
        } else if forced {
            XtsRouteState::Refused(why)
        } else {
            XtsRouteState::NotAdmitted(why)
        }
    };
    let sym_all = dense_all && experts_all;
    let iu4 = pick(
        sym_all && f.iu4_prefill_enabled,
        f.iu4_prefill_flag == Some(true),
        if !sym_all {
            "not every layer's dense+expert qt44 is symmetric".to_string()
        } else if f.iu4_prefill_flag == Some(false) {
            "kernel.iu4_prefill=false".to_string()
        } else {
            format!("no W4A4 consumer on {}", f.arch)
        },
    );
    let moe_sym = pick(
        experts_all && f.moe_sym_iu4_requested && moe_geometry_ok,
        f.moe_sym_iu4_flag == Some(true),
        if !experts_all {
            "routed experts not verified symmetric".to_string()
        } else if f.moe_sym_iu4_flag == Some(false) || f.iu4_prefill_flag == Some(false) {
            "killed by HIPFIRE_MOE_SYM_IU4=0 / HIPFIRE_IU4_PREFILL=0".to_string()
        } else if !f.moe_sym_iu4_requested {
            format!("{} lacks the symmetric IU4 MoE objects", f.arch)
        } else {
            "expert geometry outside top-k<=8, hidden%256==0, intermediate%128==0".to_string()
        },
    );
    let qwen4_forced = f.qwen4_down_a8_requested || f.qwen4_ep_a8 == Some(true);
    let a8 = pick(
        experts_all && f.a8_prefill_enabled,
        f.a8_prefill_forced || qwen4_forced,
        if !experts_all {
            "routed experts not verified symmetric".to_string()
        } else if f.a8_prefill_forced {
            format!("A8 prefill opt-in set but {} has no A8 consumer", f.arch)
        } else if qwen4_forced {
            "Qwen4 expert-parallel A8 knobs have no Inkling consumer".to_string()
        } else {
            "A8 prefill is opt-in (HIPFIRE_A8_PREFILL / kernel.gfx11_a8_prefill)".to_string()
        },
    );
    (iu4, moe_sym, a8)
}

/// Build the route-capability record of a certified XTS load from the device's
/// existing flag readers.
fn xts_route_capability(
    cert: &crate::weights::XtsCertificate,
    cfg: &Inkling2Config,
    gpu: &Gpu,
) -> XtsRouteCapability {
    let inputs = XtsRouteInputs {
        arch: gpu.arch.clone(),
        iu4_prefill_flag: gpu.flags.iu4_prefill,
        iu4_prefill_enabled: gpu.flags.iu4_prefill_enabled(),
        moe_sym_iu4_flag: hipfire_config::developer_typed_bool("HIPFIRE_MOE_SYM_IU4"),
        moe_sym_iu4_requested: gpu.moe_sym_iu4_requested(),
        a8_prefill_enabled: gpu.flags.a8_prefill_enabled(),
        a8_prefill_forced: gpu.flags.a8_prefill || gpu.flags.gfx11_a8_prefill,
        qwen4_down_a8_requested: gpu.qwen4_moe_down_a8_requested(),
        qwen4_ep_a8: gpu.flags.qwen4_moe_ep_a8,
    };
    let dense = cert.layers.iter().filter(|l| l.dense_symmetric).count();
    let experts = cert.layers.iter().filter(|l| l.experts_symmetric).count();
    let n = cert.layers.len();
    let geometry_ok = cfg.n_expert_used <= 8 && cfg.n_embd % 256 == 0 && cfg.inter % 128 == 0;
    let (iu4_prefill, moe_sym_iu4, moe_a8) =
        resolve_xts_routes(dense == n, experts == n, geometry_ok, &inputs);
    XtsRouteCapability {
        n_layers: n,
        dense_symmetric_layers: dense,
        expert_symmetric_layers: experts,
        symmetric_layers: cert.symmetric_layers(),
        unembed_symmetric: cert.unembed_symmetric,
        awq_sidecars: cert.awq_sidecars,
        down_k_logical: cert.down_k_logical,
        down_k_physical: cert.down_k_physical,
        iu4_prefill,
        moe_sym_iu4,
        moe_a8,
    }
}

impl Inkling2GpuWeights {
    /// The typed-XTS route-capability record, `None` for the legacy layout.
    pub fn xts_route_capability(&self) -> Option<&XtsRouteCapability> {
        self.xts_routes.as_ref()
    }
}

pub struct Inkling2GpuWeights {
    pub layers: Vec<Inkling2GpuLayer>,
    pub unembed: WeightTensor,
    pub norm: GpuTensor,
    pub embed_norm: GpuTensor,
    pub output_query: GpuTensor,
    /// 12 per-core-layer `[4, n_embd]` F32 tensors, rows `[Q, K, V, FFN]`
    /// (packed at load from the two cross-pass globals via
    /// `forward::pack_message_queries`).
    pub cross_pass: Vec<GpuTensor>,
    pub embed_packed: EmbedPacked,
    pub summary: VramSummary,
    /// Route-capability record of a typed-XTS load (`None` for the legacy layout).
    pub xts_routes: Option<XtsRouteCapability>,
    /// `Some(previous)` when this typed-XTS load set the device flag
    /// `Gpu::mq4v2_symmetric` (after certifying every qt44 attention / expert /
    /// unembed header); `free_gpu` restores it. `None` for the legacy layout,
    /// which never touches the flag.
    pub mq4v2_symmetric_prev: Option<bool>,
}

impl Inkling2GpuWeights {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for layer in self.layers {
            layer.free_gpu(gpu);
        }
        self.unembed.free_all(gpu);
        let _ = gpu.free_tensor(self.norm);
        let _ = gpu.free_tensor(self.embed_norm);
        let _ = gpu.free_tensor(self.output_query);
        for t in self.cross_pass {
            let _ = gpu.free_tensor(t);
        }
        if let Some(prev) = self.mq4v2_symmetric_prev {
            gpu.mq4v2_symmetric = prev;
        }
        drop(self.embed_packed);
    }
}

pub struct Inkling2GpuState {
    /// `[visit][kv_head]` K buffers, `[cap_rows, head_dim]` F16.
    pub k_cache: Vec<Vec<GpuTensor>>,
    /// `[visit][kv_head]` V buffers, `[cap_rows, head_dim]` F16.
    pub v_cache: Vec<Vec<GpuTensor>>,
    /// One unified `[kernel-1, 6656]` F32 conv state per visit (zeroed).
    pub conv: Vec<GpuTensor>,
    /// Rows already cached per visit (KV cursors).
    pub visit_cap: Vec<usize>,
    /// Row budget per visit (full visits: max_seq, SWA: min(max_seq, 1024)).
    pub visit_rows: Vec<usize>,
    /// True `key_pos` per physical KV row (host-side, zero VRAM). The slot
    /// index no longer implies position once an SWA ring wraps, so the
    /// relative-bias distance (`pos - key_pos`) is read from these tags.
    /// Full visits maintain them too (tag == slot there); SWA visits wrap.
    pub visit_pos_tags: Vec<Vec<u32>>,
    pub seq_pos: usize,
    pub max_seq: usize,
    pub kv_bytes: u64,
    pub conv_bytes: u64,
    /// Decode attention route: `false` = the scalar `inkling2_gf_attn_decode_f32`,
    /// `true` = the split-KV FA2 kernel (`Gpu::inkling2_attn_decode_fa2`,
    /// gfx1201 only; it fails closed elsewhere). Same operands either way.
    /// Defaults to FA2 on gfx1201 and scalar on every other arch.
    pub attn_fa2: bool,
    /// FA2 partition scratch, `lse [MAX_SPLITS,24]` and `oacc [MAX_SPLITS,24,128]`
    /// F32 (~400 KB), allocated once per state so a visit does no HIP
    /// alloc/free for it. Visits run in order on one stream, so one pair is
    /// shared by all 48 visits.
    pub fa2_lse: GpuTensor,
    pub fa2_oacc: GpuTensor,
}

impl Inkling2GpuState {
    /// Allocate per-visit state. `max_seq` comes from `LoadCtx` (never a
    /// silent default): full visits keep `max_seq` rows, SWA visits keep
    /// `min(max_seq, 1024)` — never 48 x max_seq.
    pub fn new_with_max_seq(
        gpu: &mut Gpu,
        cfg: &Inkling2Config,
        max_seq: usize,
    ) -> Result<Self, String> {
        if max_seq == 0 {
            return Err("inkling2: max_seq must be >= 1".into());
        }
        if cfg.swa_layers.len() != cfg.n_phys {
            return Err("inkling2: swa_layers length != n_phys".into());
        }
        let n_slot = cfg.loop_cfg.n_slot();
        let n_kv = cfg.n_head_kv;
        let hd = cfg.head_dim;
        let mut k_cache = Vec::with_capacity(n_slot);
        let mut v_cache = Vec::with_capacity(n_slot);
        let mut conv = Vec::with_capacity(n_slot);
        let visit_cap = vec![0usize; n_slot];
        let mut visit_rows = Vec::with_capacity(n_slot);
        let mut visit_pos_tags = Vec::with_capacity(n_slot);
        let mut kv_bytes = 0u64;
        let mut conv_bytes = 0u64;
        // Local cleanup on mid-alloc failure (weights have their own tx).
        let free_partial = |k_cache: &mut Vec<Vec<GpuTensor>>,
                            v_cache: &mut Vec<Vec<GpuTensor>>,
                            conv: &mut Vec<GpuTensor>,
                            gpu: &mut Gpu| {
            for row in k_cache.drain(..) {
                for t in row {
                    let _ = gpu.free_tensor(t);
                }
            }
            for row in v_cache.drain(..) {
                for t in row {
                    let _ = gpu.free_tensor(t);
                }
            }
            for t in conv.drain(..) {
                let _ = gpu.free_tensor(t);
            }
        };
        for visit in 0..n_slot {
            let phys = cfg.loop_cfg.layer_phys(visit);
            let full = !cfg.swa_layers[phys];
            let rows = if full {
                max_seq
            } else {
                max_seq.min(SWA_WINDOW_ROWS)
            };
            visit_rows.push(rows);
            visit_pos_tags.push(vec![0u32; rows]);
            let mut k_row = Vec::with_capacity(n_kv);
            let mut v_row = Vec::with_capacity(n_kv);
            for _ in 0..n_kv {
                match gpu.alloc_tensor(&[rows, hd], INKLING2_KV_DTYPE) {
                    Ok(t) => {
                        kv_bytes += t.byte_size() as u64;
                        k_row.push(t);
                    }
                    Err(e) => {
                        for t in k_row.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        for t in v_row.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        free_partial(&mut k_cache, &mut v_cache, &mut conv, gpu);
                        return Err(format!("inkling2: KV alloc visit {visit}: {e:?}"));
                    }
                }
                match gpu.alloc_tensor(&[rows, hd], INKLING2_KV_DTYPE) {
                    Ok(t) => {
                        kv_bytes += t.byte_size() as u64;
                        v_row.push(t);
                    }
                    Err(e) => {
                        for t in k_row.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        for t in v_row.drain(..) {
                            let _ = gpu.free_tensor(t);
                        }
                        free_partial(&mut k_cache, &mut v_cache, &mut conv, gpu);
                        return Err(format!("inkling2: KV alloc visit {visit}: {e:?}"));
                    }
                }
            }
            k_cache.push(k_row);
            v_cache.push(v_row);
            match gpu.zeros(&[cfg.conv_kernel - 1, INKLING2_CONV_WIDTH], DType::F32) {
                Ok(t) => {
                    conv_bytes += t.byte_size() as u64;
                    conv.push(t);
                }
                Err(e) => {
                    free_partial(&mut k_cache, &mut v_cache, &mut conv, gpu);
                    return Err(format!("inkling2: conv alloc visit {visit}: {e:?}"));
                }
            }
        }
        let (s, h, d) = (
            INKLING2_ATTN_MAX_SPLITS,
            INKLING2_ATTN_HEADS,
            INKLING2_ATTN_HEAD_DIM,
        );
        let fa2_lse = match gpu.zeros(&[s, h], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                free_partial(&mut k_cache, &mut v_cache, &mut conv, gpu);
                return Err(format!("inkling2: FA2 lse scratch alloc: {e:?}"));
            }
        };
        let fa2_oacc = match gpu.zeros(&[s, h, d], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(fa2_lse);
                free_partial(&mut k_cache, &mut v_cache, &mut conv, gpu);
                return Err(format!("inkling2: FA2 oacc scratch alloc: {e:?}"));
            }
        };
        Ok(Self {
            k_cache,
            v_cache,
            conv,
            visit_cap,
            visit_rows,
            visit_pos_tags,
            seq_pos: 0,
            max_seq,
            kv_bytes,
            conv_bytes,
            attn_fa2: gpu.arch == "gfx1201",
            fa2_lse,
            fa2_oacc,
        })
    }

    /// Zero the 48 conv states, reset KV cursors, tags and `seq_pos`. KV payloads
    /// are not zeroed: cursors gate every read, and brief 03 owns the
    /// write-before-read invariant.
    pub fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        for (v, t) in self.conv.iter().enumerate() {
            gpu.fill_f32(t, 0.0)
                .map_err(|e| format!("inkling2: reset conv visit {v}: {e:?}"))?;
        }
        for c in self.visit_cap.iter_mut() {
            *c = 0;
        }
        for tags in self.visit_pos_tags.iter_mut() {
            for t in tags.iter_mut() {
                *t = 0;
            }
        }
        self.seq_pos = 0;
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        for row in self.k_cache {
            for t in row {
                let _ = gpu.free_tensor(t);
            }
        }
        for row in self.v_cache {
            for t in row {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in self.conv {
            let _ = gpu.free_tensor(t);
        }
        let _ = gpu.free_tensor(self.fa2_lse);
        let _ = gpu.free_tensor(self.fa2_oacc);
    }
}

// ── Upload ────────────────────────────────────────────────────────────────

fn raw<'a>(
    weights: &'a Inkling2Weights,
    name: &str,
) -> Result<&'a crate::weights::RawTensor, String> {
    weights
        .get(name)
        .ok_or_else(|| format!("inkling2: missing tensor {name}"))
}

fn shape_of(weights: &Inkling2Weights, name: &str) -> Result<Vec<usize>, String> {
    Ok(raw(weights, name)?
        .info
        .shape
        .iter()
        .map(|&d| d as usize)
        .collect())
}

/// Weight-slab scope guard: packs this load's weight uploads at 64 MiB
/// granularity (qwen35 precedent). Nest-safe: under the loader's outer scope
/// the outer mode wins; standalone (tests, harnesses) this scope is outermost
/// and 64m applies. Always closed, even on the error path.
struct SlabScope {
    gpu: *mut Gpu,
}

impl SlabScope {
    fn begin(gpu: &mut Gpu) -> Self {
        gpu.weight_slab_scope_begin_with(Some(hip_bridge::SlabMode::Up64M));
        Self { gpu }
    }
}

impl Drop for SlabScope {
    fn drop(&mut self) {
        let gpu = unsafe { &*self.gpu };
        gpu.weight_slab_scope_end();
    }
}

/// Sets the device flag `Gpu::mq4v2_symmetric` for a certified typed-XTS load
/// and restores the previous value unless [`Self::keep`] disarms it (failure path).
struct SymFlagGuard {
    gpu: *mut Gpu,
    prev: bool,
    armed: bool,
}

impl SymFlagGuard {
    fn set(gpu: &mut Gpu) -> Self {
        let prev = gpu.mq4v2_symmetric;
        gpu.mq4v2_symmetric = true;
        Self {
            gpu,
            prev,
            armed: true,
        }
    }

    /// The load committed: leave the flag set, hand back the value to restore on unload.
    fn keep(&mut self) -> bool {
        self.armed = false;
        self.prev
    }
}

impl Drop for SymFlagGuard {
    fn drop(&mut self) {
        if self.armed {
            unsafe { (*self.gpu).mq4v2_symmetric = self.prev };
        }
    }
}

// ── Typed layout `qt44-padded-awq-v1` upload ──────────────────────────────
// Everything below is reached only when `Inkling2Config::mq4_layout` is the
// typed layout (`weights.xts` certified at `Inkling2Weights::load`).

/// Transaction slots of one XTS layer's decode tables (see [`Inkling2XtsLayer`]).
struct XtsTableSlots {
    gu_awq: usize,
    down_awq: usize,
    gu_ptrs: usize,
    down_ptrs: usize,
    gu_awq_ptrs: usize,
    down_awq_ptrs: usize,
}

/// F16 `[k]` sidecar of `weight_name` widened to F32 (exact).
fn xts_sidecar_f32(
    weights: &Inkling2Weights,
    weight_name: &str,
    k: usize,
) -> Result<Vec<f32>, String> {
    let sn = crate::weights::awq_sidecar_name(weight_name);
    let t = raw(weights, &sn)?;
    if t.info.quant_type != 1 || t.data.len() != 2 * k {
        return Err(format!(
            "inkling2 xts: {sn} must be F16 [{k}] (qt {} / {} bytes)",
            t.info.quant_type,
            t.data.len()
        ));
    }
    Ok(t.data
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// Attach the weight's F32 `[k]` AWQ sidecar as `WeightTensor::awq_scale`, so
/// `weight_gemv` divides x by it before the FWHT.
fn xts_attach_awq(
    tx: &mut LoadTx,
    weights: &Inkling2Weights,
    slot: usize,
    weight_name: &str,
    k: usize,
    bytes: &mut u64,
    tensors: &mut usize,
) -> Result<(), String> {
    let s = xts_sidecar_f32(weights, weight_name, k)?;
    let g = unsafe { &mut *tx.gpu };
    let t = g
        .upload_f32(&s, &[k])
        .map_err(|e| format!("inkling2: upload sidecar of {weight_name}: {e:?}"))?;
    match tx.weight_mut(slot) {
        Ok(w) => w.awq_scale = Some(t),
        Err(e) => {
            let g = unsafe { &mut *tx.gpu };
            let _ = g.free_tensor(t);
            return Err(e);
        }
    }
    *bytes += (4 * k) as u64;
    *tensors += 1;
    Ok(())
}

/// Upload one XTS layer's routed experts directly (qt44 gate_up `[2*inter, e]`,
/// qt44 down `[e, phys]` row stride 680 B, NO repack) plus the per-expert F32
/// AWQ vectors and the four device pointer tables, all held by `tx`.
/// Returns the `(gate_up, down)` weight slots and the table slots.
fn xts_upload_layer(
    tx: &mut LoadTx,
    weights: &Inkling2Weights,
    cfg: &Inkling2Config,
    il: usize,
    bytes: &mut u64,
    tensors: &mut usize,
) -> Result<(Vec<(usize, usize)>, XtsTableSlots), String> {
    let e = cfg.n_embd;
    let (_, phys) = crate::weights::xts_down_dims(cfg);
    let n = cfg.n_experts;
    let p = format!("model.llm.layers.{il}.");
    let mut slots = Vec::with_capacity(n);
    let mut gu_ptrs: Vec<u8> = Vec::with_capacity(8 * n);
    let mut dn_ptrs: Vec<u8> = Vec::with_capacity(8 * n);
    let mut gu_awq: Vec<f32> = Vec::with_capacity(n * e);
    let mut dn_awq: Vec<f32> = Vec::with_capacity(n * phys);
    for ex in 0..n {
        let gname = format!("{p}mlp.experts.{ex}.gate_up_proj.weight");
        let dname = format!("{p}mlp.experts.{ex}.down_proj.weight");
        for (name, m, k) in [(&gname, 2 * cfg.inter, e), (&dname, e, phys)] {
            let t = raw(weights, name)?;
            let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
            if t.info.quant_type != 44 || shape != [m, k] {
                return Err(format!(
                    "inkling2 xts: {name} must be qt44 [{m}, {k}], got qt={} {shape:?}",
                    t.info.quant_type
                ));
            }
        }
        let gt = raw(weights, &gname)?;
        let g = unsafe { &mut *tx.gpu };
        let (gate_up, gb) = wt_from_raw(g, &gname, 44, &gt.data, 2 * cfg.inter, e)?;
        *bytes += gb;
        *tensors += 1;
        gu_ptrs.extend_from_slice(&(gate_up.buf.buf.as_ptr() as u64).to_ne_bytes());
        let gs = tx.hold_weight(gate_up);
        let dt = raw(weights, &dname)?;
        let g = unsafe { &mut *tx.gpu };
        let (down, db) = wt_from_raw(g, &dname, 44, &dt.data, e, phys)?;
        *bytes += db;
        *tensors += 1;
        dn_ptrs.extend_from_slice(&(down.buf.buf.as_ptr() as u64).to_ne_bytes());
        let ds = tx.hold_weight(down);
        slots.push((gs, ds));
        gu_awq.extend_from_slice(&xts_sidecar_f32(weights, &gname, e)?);
        dn_awq.extend_from_slice(&xts_sidecar_f32(weights, &dname, phys)?);
    }
    let g = unsafe { &mut *tx.gpu };
    let gu_awq_t = g
        .upload_f32(&gu_awq, &[n, e])
        .map_err(|e2| format!("inkling2: upload layer {il} gate_up AWQ: {e2:?}"))?;
    *bytes += (4 * gu_awq.len()) as u64;
    *tensors += 1;
    let gu_awq_base = gu_awq_t.buf.as_ptr() as u64;
    let gu_awq_slot = tx.hold_tensor(gu_awq_t);
    let g = unsafe { &mut *tx.gpu };
    let dn_awq_t = g
        .upload_f32(&dn_awq, &[n, phys])
        .map_err(|e2| format!("inkling2: upload layer {il} down AWQ: {e2:?}"))?;
    *bytes += (4 * dn_awq.len()) as u64;
    *tensors += 1;
    let dn_awq_base = dn_awq_t.buf.as_ptr() as u64;
    let dn_awq_slot = tx.hold_tensor(dn_awq_t);
    let row_ptrs = |base: u64, row_bytes: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|ex| (base + (ex * row_bytes) as u64).to_ne_bytes())
            .collect()
    };
    let gu_awq_ptrs = row_ptrs(gu_awq_base, 4 * e);
    let dn_awq_ptrs = row_ptrs(dn_awq_base, 4 * phys);
    let mut table = |tx: &mut LoadTx, data: &[u8]| -> Result<usize, String> {
        let g = unsafe { &mut *tx.gpu };
        let t = g
            .upload_raw(data, &[data.len()])
            .map_err(|e2| format!("inkling2: upload layer {il} pointer table: {e2:?}"))?;
        *bytes += data.len() as u64;
        *tensors += 1;
        Ok(tx.hold_tensor(t))
    };
    let gu_ptrs_slot = table(tx, &gu_ptrs)?;
    let down_ptrs_slot = table(tx, &dn_ptrs)?;
    let gu_awq_ptrs_slot = table(tx, &gu_awq_ptrs)?;
    let down_awq_ptrs_slot = table(tx, &dn_awq_ptrs)?;
    Ok((
        slots,
        XtsTableSlots {
            gu_awq: gu_awq_slot,
            down_awq: dn_awq_slot,
            gu_ptrs: gu_ptrs_slot,
            down_ptrs: down_ptrs_slot,
            gu_awq_ptrs: gu_awq_ptrs_slot,
            down_awq_ptrs: down_awq_ptrs_slot,
        },
    ))
}

/// Transaction slots for one layer: every allocated tensor is held by the tx
/// until the whole load commits, so a mid-load failure frees all of them.
struct LayerSlots {
    wq: usize,
    wk: usize,
    wv: usize,
    wr: usize,
    wgate: usize,
    wo: usize,
    sink: usize,
    rel_table: usize,
    q_clip: usize,
    rel_clip: usize,
    attn_norm: Option<usize>,
    q_in: Option<usize>,
    k_in: Option<usize>,
    v_in: Option<usize>,
    mlp_norm: usize,
    k_conv: usize,
    v_conv: usize,
    k_proj: Option<usize>,
    v_proj: Option<usize>,
    attn_post: usize,
    mlp_post: usize,
    controllers: Option<[[usize; 3]; 4]>,
    attn_query: Option<usize>,
    mlp_query: usize,
    router: usize,
    router_bias: Option<usize>,
    experts: Vec<(usize, usize)>,
    /// Typed-XTS decode-table slots (`None` for the legacy layout).
    xts: Option<XtsTableSlots>,
}

/// Upload the full weight set + allocate per-visit state. `max_seq` comes
/// from `LoadCtx` (carrier). On any error the transaction frees what landed.
pub fn upload(
    weights: &Inkling2Weights,
    cfg: &Inkling2Config,
    gpu: &mut Gpu,
    max_seq: usize,
) -> Result<(Inkling2GpuWeights, Inkling2GpuState), String> {
    use rayon::prelude::*;
    let e = cfg.n_embd;
    let nh = cfg.n_head;
    let nkv = cfg.n_head_kv;
    let hd = cfg.head_dim;
    let ekv = nkv * hd;
    let eattn = nh * hd;
    let erel = nh * cfg.n_rel;
    let kk = cfg.conv_kernel;
    // Pack this load's weight uploads into slabs (64m default when outermost;
    // nested under the loader's scope the outer mode wins). Closed by Drop on
    // every path so a failed load never leaks the scope.
    let _slab = SlabScope::begin(gpu);
    // Reborrow for the tx (cohere2moe pattern): uploads use `gpu` directly,
    // the tx only derefs its pointer in `Drop` on the failure path.
    let mut tx = LoadTx::new(&mut *gpu);

    // Byte accounting: one accumulator struct (sequential method calls
    // borrow its counters one at a time).
    struct UpAcc {
        bytes: u64,
        tensors: usize,
    }
    impl UpAcc {
        // Upload one packed tensor with exact-layout validation.
        fn wt(
            &mut self,
            gpu: &mut Gpu,
            weights: &Inkling2Weights,
            name: &str,
            m: usize,
            k: usize,
        ) -> Result<WeightTensor, String> {
            let t = raw(weights, name)?;
            let (w, bytes) = wt_from_raw(gpu, name, t.info.quant_type, &t.data, m, k)?;
            self.bytes += bytes;
            self.tensors += 1;
            Ok(w)
        }
        // Upload one F32 keep with exact element-count validation.
        fn f32t(
            &mut self,
            gpu: &mut Gpu,
            weights: &Inkling2Weights,
            name: &str,
            shape: &[usize],
        ) -> Result<GpuTensor, String> {
            let (t, bytes) = load_f32(gpu, weights, name, shape)?;
            self.bytes += bytes;
            self.tensors += 1;
            Ok(t)
        }
        // Upload one pre-transformed conv ([C,1,K] or [K,C] -> [K,C] F32).
        fn conv(
            &mut self,
            gpu: &mut Gpu,
            weights: &Inkling2Weights,
            name: &str,
            ch: usize,
            kk: usize,
        ) -> Result<GpuTensor, String> {
            let t = raw(weights, name)?;
            if t.info.quant_type != 1 {
                return Err(format!(
                    "inkling2: {name} conv keep must be F16 qt1, got qt={}",
                    t.info.quant_type
                ));
            }
            let vals: Vec<f32> = t
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            if vals.len() != ch * kk {
                return Err(format!(
                    "inkling2: {name} has {} values, expected {ch}x{kk}",
                    vals.len()
                ));
            }
            let out = transform_conv(&t.info.shape, vals, kk, name)?;
            let bytes = (out.len() * 4) as u64;
            let ten = gpu
                .upload_f32(&out, &[kk, ch])
                .map_err(|e| format!("inkling2: upload {name}: {e:?}"))?;
            self.bytes += bytes;
            self.tensors += 1;
            Ok(ten)
        }
    }
    let mut acc = UpAcc {
        bytes: 0,
        tensors: 0,
    };

    // Fresh `&mut Gpu` for each upload (reborrowed from the tx pointer; the
    // tx itself never touches the device while uncommitted).
    // NOTE: each `let g = ...` shadows the previous borrow, which ends at
    // the `up_*` call that consumes it.
    let mut layer_slots = Vec::with_capacity(cfg.n_phys);
    let xts = cfg.mq4_layout.is_xts();
    if xts && weights.xts.is_none() {
        return Err(
            "inkling2: typed XTS layout without a load-time certificate (Inkling2Weights::load certifies it)"
                .into(),
        );
    }
    // Certified (`weights.xts`): every qt44 attention / expert / unembed header
    // was verified symmetric on the host, so the device-level symmetric
    // contract the Qwen-class integer routes key on holds for this file. The
    // embedding is exempt (affine, a host row lookup, never an activation GEMM).
    let mut sym_flag = if xts {
        Some(SymFlagGuard::set(gpu))
    } else {
        None
    };
    for il in 0..cfg.n_phys {
        let p = format!("model.llm.layers.{il}.");
        let swa = *cfg
            .swa_layers
            .get(il)
            .ok_or_else(|| format!("inkling2: no swa flag for layer {il}"))?;
        let nm = |s: &str| format!("{p}{s}");

        // Decode + repack the 128 down_proj tensors in parallel (CPU-bound:
        // ~2.65M F32 decode + FWHT + re-encode each), then upload serially.
        // `Inkling2Weights` is not Sync (f32 cache), so stage owned blobs
        // first; the rayon jobs then only touch Sync data.
        let down_names: Vec<String> = (0..cfg.n_experts)
            .filter(|_| !xts)
            .map(|ex| nm(&format!("mlp.experts.{ex}.down_proj.weight")))
            .collect();
        let mut down_src: Vec<(String, u8, Vec<u8>)> = Vec::with_capacity(cfg.n_experts);
        for name in &down_names {
            let t = raw(weights, name)?;
            let shape: Vec<usize> = t.info.shape.iter().map(|&d| d as usize).collect();
            if shape != [e, cfg.inter] {
                return Err(format!(
                    "inkling2: {name} shape {shape:?}, expected [{e}, {}]",
                    cfg.inter
                ));
            }
            down_src.push((name.clone(), t.info.quant_type, t.data.clone()));
        }
        let down_blobs: Vec<Vec<u8>> = down_src
            .par_iter()
            .map(|(name, qt, data)| {
                repack_down_proj(*qt, data, e).map_err(|e2| format!("{name}: {e2}"))
            })
            .collect::<Result<_, _>>()?;

        // Every upload lands in the tx immediately: a later failure frees it.
        let g = unsafe { &mut *tx.gpu };
        let wq = tx.hold_weight(acc.wt(g, weights, &nm("attn.wq_du.weight"), eattn, e)?);
        let g = unsafe { &mut *tx.gpu };
        let wk = tx.hold_weight(acc.wt(g, weights, &nm("attn.wk_dv.weight"), ekv, e)?);
        let g = unsafe { &mut *tx.gpu };
        let wv = tx.hold_weight(acc.wt(g, weights, &nm("attn.wv_dv.weight"), ekv, e)?);
        let g = unsafe { &mut *tx.gpu };
        let wr = tx.hold_weight(acc.wt(g, weights, &nm("attn.wr_du.weight"), erel, e)?);
        let g = unsafe { &mut *tx.gpu };
        let wgate = tx.hold_weight(acc.wt(g, weights, &nm("attn.gate_attn.weight"), eattn, e)?);
        let g = unsafe { &mut *tx.gpu };
        let wo = tx.hold_weight(acc.wt(g, weights, &nm("attn.wo_ud.weight"), e, eattn)?);
        if xts {
            // Typed layout: each attention weight carries its F32 AWQ sidecar,
            // so `weight_gemv` divides x by it before the FWHT.
            for (slot, stem, k) in [
                (wq, "attn.wq_du.weight", e),
                (wk, "attn.wk_dv.weight", e),
                (wv, "attn.wv_dv.weight", e),
                (wr, "attn.wr_du.weight", e),
                (wgate, "attn.gate_attn.weight", e),
                (wo, "attn.wo_ud.weight", eattn),
            ] {
                xts_attach_awq(
                    &mut tx,
                    weights,
                    slot,
                    &nm(stem),
                    k,
                    &mut acc.bytes,
                    &mut acc.tensors,
                )?;
            }
        }
        let g = unsafe { &mut *tx.gpu };
        let sink =
            tx.hold_tensor(acc.f32t(g, weights, &nm("attn.attn_sink_proj.weight"), &[nh, e])?);
        let g = unsafe { &mut *tx.gpu };
        let rel_table = tx.hold_tensor(acc.f32t(
            g,
            weights,
            &nm("attn.rel_logits_proj.proj"),
            &[cfg.n_rel, cfg.n_extent],
        )?);
        let g = unsafe { &mut *tx.gpu };
        let q_clip =
            tx.hold_tensor(acc.f32t(g, weights, &nm("attn.q_l2_clip.scale"), &[nh, hd])?);
        let g = unsafe { &mut *tx.gpu };
        let rel_clip = tx.hold_tensor(acc.f32t(
            g,
            weights,
            &nm("attn.rel_l2_clip.scale"),
            &[nh, cfg.n_rel],
        )?);

        let (attn_norm, q_in, k_in, v_in) = if swa {
            let g = unsafe { &mut *tx.gpu };
            (
                Some(tx.hold_tensor(acc.f32t(g, weights, &nm("attn_norm.weight"), &[e])?)),
                None,
                None,
                None,
            )
        } else {
            let g = unsafe { &mut *tx.gpu };
            let q = tx.hold_tensor(acc.f32t(g, weights, &nm("q_input_norm.weight"), &[e])?);
            let g = unsafe { &mut *tx.gpu };
            let k = tx.hold_tensor(acc.f32t(g, weights, &nm("k_input_norm.weight"), &[e])?);
            let g = unsafe { &mut *tx.gpu };
            let v = tx.hold_tensor(acc.f32t(g, weights, &nm("v_input_norm.weight"), &[e])?);
            (None, Some(q), Some(k), Some(v))
        };
        let g = unsafe { &mut *tx.gpu };
        let mlp_norm = tx.hold_tensor(acc.f32t(g, weights, &nm("mlp_norm.weight"), &[e])?);

        // Convs: SWA layers carry `.weight [C,1,K]`; full layers carry
        // `.static_w [K,C]` + `.weight_proj`. Both land as [K,C] F32.
        let k_conv_name = if weights.get(&nm("attn.k_sconv.weight")).is_some() {
            nm("attn.k_sconv.weight")
        } else {
            nm("attn.k_sconv.static_w")
        };
        let v_conv_name = if weights.get(&nm("attn.v_sconv.weight")).is_some() {
            nm("attn.v_sconv.weight")
        } else {
            nm("attn.v_sconv.static_w")
        };
        let g = unsafe { &mut *tx.gpu };
        let k_conv = tx.hold_tensor(acc.conv(g, weights, &k_conv_name, ekv, kk)?);
        let g = unsafe { &mut *tx.gpu };
        let v_conv = tx.hold_tensor(acc.conv(g, weights, &v_conv_name, ekv, kk)?);
        let (k_proj, v_proj) = if swa {
            (None, None)
        } else {
            let kpn = nm("attn.k_sconv.weight_proj");
            let vpn = nm("attn.v_sconv.weight_proj");
            let kshape = shape_of(weights, &kpn)?;
            let vshape = shape_of(weights, &vpn)?;
            let g = unsafe { &mut *tx.gpu };
            let kp = tx.hold_tensor(acc.f32t(g, weights, &kpn, &kshape)?);
            let g = unsafe { &mut *tx.gpu };
            let vp = tx.hold_tensor(acc.f32t(g, weights, &vpn, &vshape)?);
            (Some(kp), Some(vp))
        };
        let g = unsafe { &mut *tx.gpu };
        let attn_post = tx.hold_tensor(acc.conv(g, weights, &nm("attn_sconv.weight"), e, kk)?);
        let g = unsafe { &mut *tx.gpu };
        let mlp_post = tx.hold_tensor(acc.conv(g, weights, &nm("mlp_sconv.weight"), e, kk)?);

        let controllers = if il >= cfg.loop_cfg.core_start && il < cfg.loop_cfg.core_end {
            let mut arr = [[0usize; 3]; 4];
            for (ci, cname) in ["attn", "k", "v", "mlp"].iter().enumerate() {
                let g = unsafe { &mut *tx.gpu };
                arr[ci][0] = tx.hold_tensor(acc.f32t(
                    g,
                    weights,
                    &nm(&format!("controllers.{cname}.down_weight")),
                    &[cfg.loop_cfg.gate_rank, 2 * e],
                )?);
                let g = unsafe { &mut *tx.gpu };
                arr[ci][1] = tx.hold_tensor(acc.f32t(
                    g,
                    weights,
                    &nm(&format!("controllers.{cname}.gain_weight")),
                    &[e, cfg.loop_cfg.gate_rank],
                )?);
                let g = unsafe { &mut *tx.gpu };
                arr[ci][2] = tx.hold_tensor(acc.f32t(
                    g,
                    weights,
                    &nm(&format!("controllers.{cname}.forget_weight")),
                    &[e, cfg.loop_cfg.gate_rank],
                )?);
            }
            Some(arr)
        } else {
            None
        };

        let attn_query = if il == 0 {
            None
        } else {
            let qname = format!("model.llm.attn_res.attn_queries.{}", il - 1);
            let shape = shape_of(weights, &qname)?;
            let g = unsafe { &mut *tx.gpu };
            Some(tx.hold_tensor(acc.f32t(g, weights, &qname, &shape)?))
        };
        let g = unsafe { &mut *tx.gpu };
        let mlp_query = tx.hold_tensor(acc.f32t(
            g,
            weights,
            &format!("model.llm.attn_res.mlp_queries.{il}"),
            &[e],
        )?);

        let g = unsafe { &mut *tx.gpu };
        let router =
            tx.hold_weight(acc.wt(g, weights, &nm("mlp.gate.weight"), cfg.n_experts, e)?);
        let router_bias = match weights.get(&nm("mlp.gate.bias")) {
            Some(_) => {
                let g = unsafe { &mut *tx.gpu };
                Some(tx.hold_tensor(acc.f32t(
                    g,
                    weights,
                    &nm("mlp.gate.bias"),
                    &[cfg.n_experts],
                )?))
            }
            None => None,
        };

        let mut experts = Vec::with_capacity(cfg.n_experts);
        let xts_slots = if xts {
            let (slots, tables) =
                xts_upload_layer(&mut tx, weights, cfg, il, &mut acc.bytes, &mut acc.tensors)?;
            experts.extend(slots);
            Some(tables)
        } else {
            None
        };
        for (ex, down_blob) in down_blobs.into_iter().enumerate() {
            let gname = nm(&format!("mlp.experts.{ex}.gate_up_proj.weight"));
            let gt = raw(weights, &gname)?;
            let gshape: Vec<usize> = gt.info.shape.iter().map(|&d| d as usize).collect();
            if gshape != [2 * cfg.inter, e] {
                return Err(format!("inkling2: {gname} shape {gshape:?}"));
            }
            let g = unsafe { &mut *tx.gpu };
            let (gate_up, gb) =
                wt_from_raw(g, &gname, gt.info.quant_type, &gt.data, 2 * cfg.inter, e)?;
            acc.bytes += gb;
            acc.tensors += 1;
            let gs = tx.hold_weight(gate_up);
            let g = unsafe { &mut *tx.gpu };
            let buf = g
                .upload_raw(&down_blob, &[down_blob.len()])
                .map_err(|e| format!("inkling2: upload down expert {ex}: {e:?}"))?;
            acc.bytes += down_blob.len() as u64;
            acc.tensors += 1;
            let ds = tx.hold_weight(WeightTensor {
                buf,
                gpu_dtype: DType::MQ4G256V2,
                m: e,
                k: DOWN_K_PADDED,
                row_stride: 5 * 136,
                awq_scale: None,
                lloyd_lut_e4m3: None,
                lloyd_lut_f16: None,
                lloyd_lut_c16: None,
            });
            experts.push((gs, ds));
        }

        layer_slots.push(LayerSlots {
            wq,
            wk,
            wv,
            wr,
            wgate,
            wo,
            sink,
            rel_table,
            q_clip,
            rel_clip,
            attn_norm,
            q_in,
            k_in,
            v_in,
            mlp_norm,
            k_conv,
            v_conv,
            k_proj,
            v_proj,
            attn_post,
            mlp_post,
            controllers,
            attn_query,
            mlp_query,
            router,
            router_bias,
            experts,
            xts: xts_slots,
        });
    }

    let g = unsafe { &mut *tx.gpu };
    let unembed =
        tx.hold_weight(acc.wt(g, weights, "model.llm.unembed.weight", cfg.vocab_size, e)?);
    let g = unsafe { &mut *tx.gpu };
    let norm = tx.hold_tensor(acc.f32t(g, weights, "model.llm.norm.weight", &[e])?);
    let g = unsafe { &mut *tx.gpu };
    let embed_norm = tx.hold_tensor(acc.f32t(g, weights, "model.llm.embed_norm.weight", &[e])?);
    let g = unsafe { &mut *tx.gpu };
    let output_query =
        tx.hold_tensor(acc.f32t(g, weights, "model.llm.attn_res.output_query", &[e])?);

    // Cross-pass globals -> per-core-layer [4, n_embd] [Q, K, V, FFN].
    let packed_msgs = crate::forward::pack_message_queries(weights, cfg)?;
    let mut cross_slots = Vec::with_capacity(packed_msgs.len());
    for (i, msgs) in packed_msgs.iter().enumerate() {
        let mut flat = Vec::with_capacity(4 * e);
        for m in msgs {
            if m.len() != e {
                return Err(format!("inkling2: cross-pass message {i} len {}", m.len()));
            }
            flat.extend_from_slice(m);
        }
        let g = unsafe { &mut *tx.gpu };
        let ten = g
            .upload_f32(&flat, &[4, e])
            .map_err(|e| format!("inkling2: upload cross_pass {i}: {e:?}"))?;
        acc.bytes += (flat.len() * 4) as u64;
        acc.tensors += 1;
        cross_slots.push(tx.hold_tensor(ten));
    }

    // embed stays packed on the host (row-lookup side is HOST).
    let et = raw(weights, "model.llm.embed.weight")?;
    let eshape: Vec<usize> = et.info.shape.iter().map(|&d| d as usize).collect();
    if eshape != [cfg.vocab_size, e] {
        return Err(format!("inkling2: embed shape {eshape:?}"));
    }
    if et.info.quant_type != 44 {
        return Err(format!(
            "inkling2: embed must be packed qt44, got qt={}",
            et.info.quant_type
        ));
    }
    let embed_packed = EmbedPacked {
        quant_type: et.info.quant_type,
        rows: cfg.vocab_size,
        cols: e,
        bytes: et.data.clone(),
    };

    // Per-visit state (own cleanup on failure; weights stay in the tx).
    let g = unsafe { &mut *tx.gpu };
    let state = Inkling2GpuState::new_with_max_seq(g, cfg, max_seq)?;

    let full_visits = state.visit_rows.iter().filter(|&&r| r == max_seq).count();
    let swa_visits = state.visit_rows.len() - full_visits;
    let down_pad_extra_bytes =
        (cfg.n_phys * cfg.n_experts * e * (DOWN_K_PADDED - DOWN_K_LOGICAL) / 256 * 136) as u64;

    // Materialize: move everything out of the transaction, then commit.
    let take_w = |tx: &mut LoadTx, slot: usize| tx.take_weight(slot);
    let take_t = |tx: &mut LoadTx, slot: usize| tx.take_tensor(slot);
    let take_opt_t = |tx: &mut LoadTx, slot: Option<usize>| -> Result<Option<GpuTensor>, String> {
        match slot {
            Some(s) => Ok(Some(tx.take_tensor(s)?)),
            None => Ok(None),
        }
    };
    let take_ctrl = |tx: &mut LoadTx, slots: [usize; 3]| -> Result<Inkling2Controller, String> {
        Ok(Inkling2Controller {
            down: tx.take_tensor(slots[0])?,
            gain: tx.take_tensor(slots[1])?,
            forget: tx.take_tensor(slots[2])?,
        })
    };
    let mut layers = Vec::with_capacity(layer_slots.len());
    for s in layer_slots {
        let mut experts = Vec::with_capacity(s.experts.len());
        for (gs, ds) in s.experts {
            experts.push(Inkling2Expert {
                gate_up: take_w(&mut tx, gs)?,
                down: take_w(&mut tx, ds)?,
            });
        }
        layers.push(Inkling2GpuLayer {
            wq: take_w(&mut tx, s.wq)?,
            wk: take_w(&mut tx, s.wk)?,
            wv: take_w(&mut tx, s.wv)?,
            wr: take_w(&mut tx, s.wr)?,
            wgate: take_w(&mut tx, s.wgate)?,
            wo: take_w(&mut tx, s.wo)?,
            sink: take_t(&mut tx, s.sink)?,
            rel_table: take_t(&mut tx, s.rel_table)?,
            q_clip: take_t(&mut tx, s.q_clip)?,
            rel_clip: take_t(&mut tx, s.rel_clip)?,
            attn_norm: take_opt_t(&mut tx, s.attn_norm)?,
            q_input_norm: take_opt_t(&mut tx, s.q_in)?,
            k_input_norm: take_opt_t(&mut tx, s.k_in)?,
            v_input_norm: take_opt_t(&mut tx, s.v_in)?,
            mlp_norm: take_t(&mut tx, s.mlp_norm)?,
            k_conv: take_t(&mut tx, s.k_conv)?,
            v_conv: take_t(&mut tx, s.v_conv)?,
            k_proj: take_opt_t(&mut tx, s.k_proj)?,
            v_proj: take_opt_t(&mut tx, s.v_proj)?,
            attn_post_conv: take_t(&mut tx, s.attn_post)?,
            mlp_post_conv: take_t(&mut tx, s.mlp_post)?,
            controllers: match s.controllers {
                Some([a, b, c, d]) => Some([
                    take_ctrl(&mut tx, a)?,
                    take_ctrl(&mut tx, b)?,
                    take_ctrl(&mut tx, c)?,
                    take_ctrl(&mut tx, d)?,
                ]),
                None => None,
            },
            attn_query: take_opt_t(&mut tx, s.attn_query)?,
            mlp_query: take_t(&mut tx, s.mlp_query)?,
            router: take_w(&mut tx, s.router)?,
            router_bias: take_opt_t(&mut tx, s.router_bias)?,
            experts,
            xts: match s.xts {
                Some(x) => Some(Inkling2XtsLayer {
                    gu_awq: take_t(&mut tx, x.gu_awq)?,
                    down_awq: take_t(&mut tx, x.down_awq)?,
                    gu_ptrs: take_t(&mut tx, x.gu_ptrs)?,
                    down_ptrs: take_t(&mut tx, x.down_ptrs)?,
                    gu_awq_ptrs: take_t(&mut tx, x.gu_awq_ptrs)?,
                    down_awq_ptrs: take_t(&mut tx, x.down_awq_ptrs)?,
                }),
                None => None,
            },
        });
    }
    let unembed = take_w(&mut tx, unembed)?;
    let norm = take_t(&mut tx, norm)?;
    let embed_norm = take_t(&mut tx, embed_norm)?;
    let output_query = take_t(&mut tx, output_query)?;
    let mut cross_pass = Vec::with_capacity(cross_slots.len());
    for slot in cross_slots {
        cross_pass.push(take_t(&mut tx, slot)?);
    }
    tx.commit();
    let mq4v2_symmetric_prev = sym_flag.as_mut().map(SymFlagGuard::keep);

    let kv_bytes = state.kv_bytes;
    let conv_bytes = state.conv_bytes;
    let summary = VramSummary {
        weight_bytes: acc.bytes,
        weight_tensors: acc.tensors,
        kv_bytes,
        conv_bytes,
        down_pad_extra_bytes,
        max_seq,
        full_visits,
        swa_visits,
    };
    let xts_routes = weights
        .xts
        .as_ref()
        .map(|cert| xts_route_capability(cert, cfg, &*gpu));
    Ok((
        Inkling2GpuWeights {
            layers,
            unembed,
            norm,
            embed_norm,
            output_query,
            cross_pass,
            embed_packed,
            summary,
            xts_routes,
            mq4v2_symmetric_prev,
        },
        state,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_map_adds_44_and_47() {
        assert_eq!(dtype_for_quant_type(44).unwrap(), DType::MQ4G256V2);
        assert_eq!(dtype_for_quant_type(47).unwrap(), DType::MQ6G256V2);
        assert_eq!(dtype_for_quant_type(13).unwrap(), DType::MQ4G256);
        assert_eq!(dtype_for_quant_type(1).unwrap(), DType::F16);
        assert!(dtype_for_quant_type(99).is_err());
    }

    #[test]
    fn encoded_rows_reject_ragged_k() {
        assert!(encoded_row_bytes(DType::MQ4G256V2, 1152).is_err());
        assert_eq!(encoded_row_bytes(DType::MQ4G256V2, 1280).unwrap(), 5 * 136);
    }

    #[test]
    fn down_pad_byte_math_matches_brief() {
        // 3072 tensors x 156,672 B = +0.45 GiB model-wide.
        assert_eq!(DOWN_K_PADDED - DOWN_K_LOGICAL, 128);
        assert_eq!(2304 * 5 * 136, 1_566_720);
        assert_eq!(1_566_720 - 1_410_048, 156_672);
        assert_eq!(3072 * 156_672, 481_296_384);
        assert!((481_296_384f64 / (1u64 << 30) as f64 - 0.448).abs() < 0.01);
    }

    #[test]
    fn fwht_forward_is_inverse_of_decode() {
        // rotate then inverse == identity (orthonormal up to fp error).
        let s1 = fwht_signs(42);
        let s2 = fwht_signs(1042);
        let mut g = [0.0f32; 256];
        for i in 0..256 {
            g[i] = ((i * 7919) % 257) as f32 / 257.0 - 0.5;
        }
        let orig = g;
        fwht256_forward(&mut g, &s1, &s2);
        // Inverse: pre-scale s2, butterfly, post-scale 1/16 * s1.
        for i in 0..256 {
            g[i] *= s2[i];
        }
        let mut stride = 1;
        while stride < 256 {
            let mut j = 0;
            while j < 256 {
                for k in 0..stride {
                    let a = g[j + k];
                    let b = g[j + k + stride];
                    g[j + k] = a + b;
                    g[j + k + stride] = a - b;
                }
                j += stride * 2;
            }
            stride <<= 1;
        }
        for i in 0..256 {
            g[i] *= 0.0625 * s1[i];
        }
        let err: f32 = g
            .iter()
            .zip(orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-4, "fwht roundtrip err {err}");
    }

    #[test]
    fn repack_row_has_legal_layout_and_zero_tail() {
        let row: Vec<f32> = (0..DOWN_K_LOGICAL)
            .map(|i| ((i as u32).wrapping_mul(2654435761) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let bytes = repack_row_mq4g256v2(&row);
        assert_eq!(bytes.len(), 5 * 136);
        let back = hipfire_runtime::weight_backend::dequantize_weight_to_f32(44, &bytes, 1280);
        assert_eq!(back.len(), 1280);
        // Encoder contract, checked in the rotated domain: each weight must
        // land within half its group's grid step (rotation preserves energy
        // but not range, so no natural-domain bound this tight exists).
        let s1 = fwht_signs(42);
        let s2 = fwht_signs(1042);
        let mut padded = [0.0f32; DOWN_K_PADDED];
        padded[..DOWN_K_LOGICAL].copy_from_slice(&row);
        let mut rot_orig = padded;
        for c in rot_orig.chunks_exact_mut(256) {
            let g: &mut [f32; 256] = c.try_into().unwrap();
            fwht256_forward(g, &s1, &s2);
        }
        let mut rot_back = [0.0f32; DOWN_K_PADDED];
        rot_back.copy_from_slice(&back);
        for c in rot_back.chunks_exact_mut(256) {
            let g: &mut [f32; 256] = c.try_into().unwrap();
            fwht256_forward(g, &s1, &s2);
        }
        for g in 0..5 {
            let o = &rot_orig[g * 256..(g + 1) * 256];
            let b = &rot_back[g * 256..(g + 1) * 256];
            for half in 0..2 {
                let v = &o[half * 128..(half + 1) * 128];
                let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let st = half::f16::from_f32(if hi == lo { 0.0 } else { step }).to_f32();
                for i in 0..128 {
                    let err = (b[half * 128 + i] - v[i]).abs();
                    assert!(
                        err <= st / 2.0 + 2e-3,
                        "group {g} half {half} rotated err {err} step {st}"
                    );
                }
            }
        }
        // Pad tail: natural-domain reconstruction of the zero region stays at
        // the grid-noise floor (loose inverse-spread bound, catches a missing
        // pad which would read O(range) garbage).
        let mut tail = 0.0f32;
        for i in DOWN_K_LOGICAL..DOWN_K_PADDED {
            tail = tail.max(back[i].abs());
        }
        let range: f32 = row.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            tail <= range / 15.0 * 8.0 + 1e-2,
            "pad tail {tail} range {range}"
        );
    }

    #[test]
    fn repack_row_degenerate_half_is_all_zero_codes() {
        let row = vec![0.0f32; DOWN_K_LOGICAL];
        let bytes = repack_row_mq4g256v2(&row);
        // Every group header is (scale 0, zero +/-0.0) with zero nibbles.
        // -0.0 arises from the FWHT sign flips on exact zeros and decodes
        // identically; the reference encoder behaves the same.
        for g in 0..5 {
            let h = &bytes[g * 136..(g + 1) * 136];
            assert_eq!(&h[..2], &[0, 0], "group {g}");
            assert!(h[2..4] == [0, 0] || h[2..4] == [0, 128], "group {g}");
            assert_eq!(&h[4..6], &[0, 0], "group {g}");
            assert!(h[6..8] == [0, 0] || h[6..8] == [0, 128], "group {g}");
            assert!(h[8..].iter().all(|&b| b == 0), "group {g}");
        }
    }

    #[test]
    fn conv_transform_matches_oracle_tap_convention() {
        // 3D SWA [C,1,K]: out[k][c] = v[c*K+k], no flip.
        let vals: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = transform_conv(&[2, 1, 3], vals, 3, "t").unwrap();
        assert_eq!(out, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        // 2D static_w [K,C]: flip the kernel axis once.
        let vals: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let out = transform_conv(&[3, 2], vals, 3, "t").unwrap();
        assert_eq!(out, vec![4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
        assert!(transform_conv(&[5], vec![0.0; 5], 3, "t").is_err());
    }

    #[test]
    fn kv_cap_math_matches_vram_plan() {
        // max_seq 32768: 8 full slots + 40 SWA(1024) slots, F16, 8 heads x 128.
        let full_rows = 32768usize;
        let swa_rows = 1024usize;
        let per_vec = 128 * 2; // F16
        let per_slot = 8 * 2 * per_vec;
        let kv = (8 * full_rows + 40 * swa_rows) as u64 * per_slot as u64;
        assert_eq!(kv, 1_241_513_984);
        assert!((kv as f64 / (1u64 << 30) as f64 - 1.156).abs() < 0.001);
        // conv: 48 visits x 6656 x 3 taps x 4 B.
        assert_eq!(48 * INKLING2_CONV_WIDTH * 3 * 4, 3_833_856);
    }

    #[test]
    fn embed_row_slices_group_aligned_rows() {
        // Slicing proof with distinctive impulses: row 0 is zeros, row 1
        // carries two spikes. A row-stride bug shows up as O(signal) error,
        // far above the grid-noise floor.
        let row0 = vec![0.0f32; 2304];
        let mut row1 = vec![0.0f32; 2304];
        row1[100] = 5.0;
        row1[2000] = -3.0;
        let mut bytes = repack_row_to_2304qt44(&row0);
        bytes.extend_from_slice(&repack_row_to_2304qt44(&row1));
        let packed = EmbedPacked {
            quant_type: 44,
            rows: 2,
            cols: 2304,
            bytes,
        };
        let got0 = packed.row_f32(0).unwrap();
        assert_eq!(got0.len(), 2304);
        assert!(got0.iter().all(|&v| v.abs() < 0.35), "row 0 must stay ~0");
        let got1 = packed.row_f32(1).unwrap();
        assert!((got1[100] - 5.0).abs() < 0.35, "spike {}", got1[100]);
        assert!((got1[2000] + 3.0).abs() < 0.35, "spike {}", got1[2000]);
        assert!(packed.row_f32(2).is_err());
    }

    /// Test-only: encode a 2304-wide row to qt44 (9 groups, same header/code
    /// layout as [`repack_row_mq4g256v2`], no padding).
    fn repack_row_to_2304qt44(row: &[f32]) -> Vec<u8> {
        assert_eq!(row.len(), 2304);
        let s1 = fwht_signs(42);
        let s2 = fwht_signs(1042);
        let mut out = vec![0u8; 9 * 136];
        for (g, chunk) in out.chunks_exact_mut(136).enumerate() {
            let mut group = [0.0f32; 256];
            group.copy_from_slice(&row[g * 256..(g + 1) * 256]);
            fwht256_forward(&mut group, &s1, &s2);
            for half in 0..2 {
                let v = &group[half * 128..(half + 1) * 128];
                let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let sbits = if hi == lo {
                    0
                } else {
                    half::f16::from_f32(step).to_bits()
                };
                let zbits = half::f16::from_f32(lo).to_bits();
                chunk[half * 4..half * 4 + 2].copy_from_slice(&sbits.to_le_bytes());
                chunk[half * 4 + 2..half * 4 + 4].copy_from_slice(&zbits.to_le_bytes());
                let st = half::f16::from_bits(sbits).to_f32();
                let z = half::f16::from_bits(zbits).to_f32();
                if hi != lo && step != 0.0 && st != 0.0 {
                    let inv = 1.0 / st;
                    for i in 0..128 {
                        let q = ((group[half * 128 + i] - z) * inv + 0.5)
                            .floor()
                            .clamp(0.0, 15.0) as u8;
                        let at = 8 + (half * 128 + i) / 2;
                        if (half * 128 + i) % 2 == 0 {
                            chunk[at] |= q;
                        } else {
                            chunk[at] |= q << 4;
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires gfx1100 + the 14 GB inkling2-mini-mq4.hfq artifact: full upload + free x3, VRAM must return to baseline"]
    fn upload_unload_x3_returns_to_baseline() {
        let path = std::path::Path::new(r"D:\hipfire-models\inkling2-mini-mq4.hfq");
        let hfq = hipfire_runtime::hfq::HfqFile::open(path).expect("open artifact");
        let cfg = Inkling2Config::from_metadata_json(crate::TEST_METADATA).unwrap();
        let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
        let mut gpu = Gpu::init().expect("gpu init");
        let (free0, total) = gpu.device_mem_info().expect("mem info");
        eprintln!("baseline: free {free0} / total {total}");
        for i in 0..3 {
            let (gw, gs) = upload(&weights, &cfg, &mut gpu, 32768).expect("upload");
            let (free1, _) = gpu.device_mem_info().expect("mem info");
            let (pnew, reused, bnew) = gpu.pool_stats();
            eprintln!(
                "iter {i}: pool new={pnew} reused={reused} bytes_new={bnew} ({} MiB) parked={} slabs={:?}",
                bnew / (1 << 20),
                gpu.pool_parked_bytes(),
                gpu.pinned_weight_slabs().map(|s| (s.slabs, s.slab_bytes, s.live_views)),
            );
            eprintln!(
                "iter {i}: after upload free {free1} (delta {} MiB); {}",
                (free0 as i64 - free1 as i64) / (1 << 20),
                gw.summary.text()
            );
            let expect = gw.summary.total_bytes();
            let got = free0.saturating_sub(free1);
            assert!(
                (got as i64 - expect as i64).abs() < 2i64 << 30,
                "measured delta {got} vs accounting {expect}"
            );
            gs.free_gpu(&mut gpu);
            gw.free_gpu(&mut gpu);
            gpu.drain_pool();
            let (free2, _) = gpu.device_mem_info().expect("mem info");
            eprintln!(
                "iter {i}: after free+drain free {free2} (leak {} MiB)",
                (free0 as i64 - free2 as i64) / (1 << 20)
            );
            assert!(
                (free0 as i64 - free2 as i64).abs() < 16i64 << 20,
                "VRAM leak after iter {i}: baseline {free0} vs {free2}"
            );
        }
    }

    #[test]
    #[ignore = "requires the 14 GB inkling2-mini-mq4.hfq artifact: real down_proj repack fidelity"]
    fn real_down_proj_repack_fidelity() {
        let path = std::path::Path::new(r"D:\hipfire-models\inkling2-mini-mq4.hfq");
        let hfq = hipfire_runtime::hfq::HfqFile::open(path).expect("open artifact");
        let name = "model.llm.layers.0.mlp.experts.0.down_proj.weight";
        let (info, data) = hfq.tensor_data_vec(name).expect("tensor");
        assert_eq!(info.quant_type, 13);
        let m = 2304usize;
        let orig = hipfire_runtime::weight_backend::dequantize_weight_to_f32(
            13,
            &data,
            m * DOWN_K_LOGICAL,
        );
        let repacked = repack_down_proj(13, &data, m).expect("repack");
        assert_eq!(repacked.len(), m * 5 * 136);
        let back =
            hipfire_runtime::weight_backend::dequantize_weight_to_f32(44, &repacked, m * 1280);
        let mut err = 0.0f32;
        let mut tail = 0.0f32;
        for r in 0..m {
            for c in 0..DOWN_K_LOGICAL {
                err = err.max((back[r * 1280 + c] - orig[r * 1152 + c]).abs());
            }
            for c in DOWN_K_LOGICAL..DOWN_K_PADDED {
                tail = tail.max(back[r * 1280 + c].abs());
            }
        }
        let range: f32 = orig.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        eprintln!("repack fidelity: max err {err}, pad tail {tail}, range {range}");
        assert!(err <= range / 15.0 + 1e-3, "repack err {err} range {range}");
        assert!(tail <= range / 15.0 + 1e-3, "pad tail {tail}");
    }
}

#[cfg(test)]
mod xts_gpu_tests {
    use super::*;
    use crate::weights::xts_tests::{build, f16_vec, put, sym_qt44, tiny_cfg, weights_with};
    use crate::weights::{certify_xts, AWQ_SIDECAR_SUFFIX};

    fn inputs(arch: &str) -> XtsRouteInputs {
        XtsRouteInputs {
            arch: arch.to_string(),
            iu4_prefill_flag: None,
            iu4_prefill_enabled: true,
            moe_sym_iu4_flag: None,
            moe_sym_iu4_requested: true,
            a8_prefill_enabled: false,
            a8_prefill_forced: false,
            qwen4_down_a8_requested: false,
            qwen4_ep_a8: None,
        }
    }

    #[test]
    fn route_resolution_truth_table() {
        use XtsRouteState::*;
        // Capable device, defaults: W4A4 and symmetric-IU4 by capability; A8 is opt-in.
        let f = inputs("gfx1201");
        let (iu4, sym, a8) = resolve_xts_routes(true, true, true, &f);
        assert_eq!((iu4, sym), (Admitted, Admitted));
        assert!(matches!(a8, NotAdmitted(_)));

        // Explicit kill of the shared switch: nothing admitted, nothing refused (not forced).
        let mut k = inputs("gfx1201");
        k.iu4_prefill_flag = Some(false);
        k.iu4_prefill_enabled = false;
        k.moe_sym_iu4_requested = false;
        let (iu4, sym, _) = resolve_xts_routes(true, true, true, &k);
        assert!(matches!(&iu4, NotAdmitted(w) if w.contains("kernel.iu4_prefill=false")));
        assert!(matches!(&sym, NotAdmitted(w) if w.contains("killed")));

        // Forced on an arch with no consumer: refused, never admitted.
        let mut forced = inputs("gfx1030");
        forced.iu4_prefill_flag = Some(true);
        forced.iu4_prefill_enabled = false;
        forced.moe_sym_iu4_flag = Some(true);
        forced.moe_sym_iu4_requested = false;
        let (iu4, sym, _) = resolve_xts_routes(true, true, true, &forced);
        assert!(matches!(&iu4, Refused(w) if w.contains("gfx1030")));
        assert!(matches!(&sym, Refused(w) if w.contains("gfx1030")));

        // Forced A8 knobs: an enabled A8 prefill is admitted; a Qwen4-only knob is refused.
        let mut a8f = inputs("gfx1201");
        a8f.a8_prefill_enabled = true;
        a8f.a8_prefill_forced = true;
        assert_eq!(resolve_xts_routes(true, true, true, &a8f).2, Admitted);
        let mut q4 = inputs("gfx1201");
        q4.qwen4_ep_a8 = Some(true);
        assert!(
            matches!(&resolve_xts_routes(true, true, true, &q4).2, Refused(w) if w.contains("Qwen4"))
        );
        let mut a8_arch = inputs("gfx1100");
        a8_arch.a8_prefill_forced = true; // set, but not enabled on this arch
        assert!(matches!(
            &resolve_xts_routes(true, true, true, &a8_arch).2,
            Refused(w) if w.contains("no A8 consumer")
        ));

        // A layer that is not symmetric admits nothing; geometry gates only the MoE route.
        let (iu4, sym, a8) = resolve_xts_routes(true, false, true, &f);
        assert!(matches!(iu4, NotAdmitted(_)) && matches!(sym, NotAdmitted(_)));
        assert!(matches!(a8, NotAdmitted(_)));
        let (iu4, sym, _) = resolve_xts_routes(true, true, false, &f);
        assert_eq!(iu4, Admitted);
        assert!(matches!(&sym, NotAdmitted(w) if w.contains("geometry")));
    }

    fn cap(iu4: XtsRouteState, sym: XtsRouteState, a8: XtsRouteState) -> XtsRouteCapability {
        XtsRouteCapability {
            n_layers: 24,
            dense_symmetric_layers: 24,
            expert_symmetric_layers: 24,
            symmetric_layers: 24,
            unembed_symmetric: true,
            awq_sidecars: 6288,
            down_k_logical: 1152,
            down_k_physical: 1280,
            iu4_prefill: iu4,
            moe_sym_iu4: sym,
            moe_a8: a8,
        }
    }

    #[test]
    fn load_line_names_admitted_and_refused_routes() {
        use XtsRouteState::*;
        let all = cap(Admitted, Admitted, Admitted).load_line();
        assert_eq!(
            all,
            "inkling2 xts: symmetric 24/24 layers (dense+experts), awq 6288 sidecars; \
             routes admitted by capability: iu4_prefill, moe_sym_iu4, moe_a8; \
             executing: scalar decode (no multi-row consumer yet)"
        );
        let some = cap(
            Admitted,
            NotAdmitted("n/a".into()),
            Refused("Qwen4 expert-parallel A8 knobs have no Inkling consumer".into()),
        )
        .load_line();
        assert!(
            some.contains("routes admitted by capability: iu4_prefill; "),
            "{some}"
        );
        assert!(
            some.contains("refused (forced, cannot run): moe_a8 (Qwen4 expert-parallel A8 knobs"),
            "{some}"
        );
        assert!(
            some.ends_with("executing: scalar decode (no multi-row consumer yet)"),
            "{some}"
        );
        let none = cap(
            NotAdmitted("a".into()),
            NotAdmitted("b".into()),
            NotAdmitted("c".into()),
        )
        .load_line();
        assert!(
            none.contains("routes admitted by capability: none; executing"),
            "{none}"
        );
    }

    /// Non-unit, per-(expert, column) sidecars so a wrong table / wrong divide shows.
    fn vary_expert_sidecars(
        map: &mut std::collections::HashMap<String, crate::weights::RawTensor>,
        inter: usize,
    ) {
        for (name, t) in map.iter_mut() {
            if !(name.contains(".mlp.experts.") && name.ends_with(AWQ_SIDECAR_SUFFIX)) {
                continue;
            }
            let k = t.data.len() / 2;
            let live = if name.contains("down_proj") { inter } else { k };
            let salt = name.len();
            for i in 0..k {
                let v = if i < live {
                    0.5 + ((i * 13 + salt * 7) % 11) as f32 * 0.25
                } else {
                    1.0
                };
                t.data[2 * i..2 * i + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn xts_expert_stage_matches_cpu_f32_reference_hw() {
        stage_matches_cpu(tiny_cfg());
    }

    /// Production widths: hidden 2304 (gate_up K), SiTU 1152 -> physical 1280 (down K).
    #[test]
    #[ignore = "needs a GPU"]
    fn xts_expert_stage_matches_cpu_f32_reference_mini_widths_hw() {
        let mut cfg = tiny_cfg();
        cfg.n_embd = 2304;
        cfg.inter = 1152;
        cfg.n_phys = 1;
        stage_matches_cpu(cfg);
    }

    fn stage_matches_cpu(cfg: Inkling2Config) {
        let mut map = build(&cfg);
        vary_expert_sidecars(&mut map, cfg.inter);
        let cert = certify_xts(&map, &cfg).unwrap();
        let w = weights_with(map, Some(cert));
        let e = cfg.n_embd;
        let inter = cfg.inter;
        let mut gpu = Gpu::init().expect("GPU required");

        let mut tx = LoadTx::new(&mut gpu);
        let (mut bytes, mut nt) = (0u64, 0usize);
        let (slots, tabs) =
            xts_upload_layer(&mut tx, &w, &cfg, 0, &mut bytes, &mut nt).expect("xts upload");
        let mut weights_held = Vec::new();
        for (g, d) in slots {
            weights_held.push(tx.take_weight(g).unwrap());
            weights_held.push(tx.take_weight(d).unwrap());
        }
        let xl = Inkling2XtsLayer {
            gu_awq: tx.take_tensor(tabs.gu_awq).unwrap(),
            down_awq: tx.take_tensor(tabs.down_awq).unwrap(),
            gu_ptrs: tx.take_tensor(tabs.gu_ptrs).unwrap(),
            down_ptrs: tx.take_tensor(tabs.down_ptrs).unwrap(),
            gu_awq_ptrs: tx.take_tensor(tabs.gu_awq_ptrs).unwrap(),
            down_awq_ptrs: tx.take_tensor(tabs.down_awq_ptrs).unwrap(),
        };
        tx.commit();
        assert!(bytes > 0 && nt > 0);

        let mut s = 0x1234_5678u32;
        let x: Vec<f32> = (0..e)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();
        // Repeated experts and one unrouted (-1) slot.
        let ids: [i32; 8] = [0, 1, 1, 0, 1, -1, 0, 1];
        let xt = gpu.upload_f32(&x, &[e]).unwrap();
        let idt = gpu.alloc_tensor(&[8], DType::F32).unwrap();
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
        gpu.hip.memcpy_htod(&idt.buf, &idb).unwrap();
        let mut tmp = vec![xt, idt];
        let mut capture = None;
        let di =
            crate::gpu_forward::xts_expert_stage(&mut gpu, &cfg, &xl, &mut tmp, 0, 1, &mut capture)
                .expect("xts stage");
        let got = gpu.download_f32(&tmp[di]).unwrap();
        assert_eq!(got.len(), 8 * e);

        let matvec = |m: &[f32], rows: usize, cols: usize, v: &[f32]| -> Vec<f32> {
            (0..rows)
                .map(|r| {
                    m[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(v)
                        .map(|(a, b)| a * b)
                        .sum()
                })
                .collect()
        };
        let mut worst = 0.0f32;
        let mut peak = 0.0f32;
        for (r, &id) in ids.iter().enumerate() {
            let row = &got[r * e..(r + 1) * e];
            if id < 0 {
                assert!(
                    row.iter().all(|&v| v == 0.0),
                    "unrouted rank {r} must be zero"
                );
                continue;
            }
            let p = "model.llm.layers.0.mlp.experts";
            let (_, wgu) = w.f32(&format!("{p}.{id}.gate_up_proj.weight")).unwrap();
            let (dshape, wd) = w.f32(&format!("{p}.{id}.down_proj.weight")).unwrap();
            assert_eq!(dshape, vec![e, inter]);
            let gu = matvec(&wgu, 2 * inter, e, &x);
            let act: Vec<f32> = (0..inter)
                .map(|j| {
                    crate::oracle::situ(gu[j], gu[inter + j], cfg.situ_beta, cfg.situ_linear_beta)
                })
                .collect();
            let want = matvec(&wd, e, inter, &act);
            for (a, b) in row.iter().zip(&want) {
                worst = worst.max((a - b).abs());
                peak = peak.max(b.abs());
            }
        }
        assert!(
            peak > 0.0,
            "reference is all zero: the test would be vacuous"
        );
        assert!(
            worst <= 2e-3 * peak,
            "xts stage vs CPU f32 reference: max abs err {worst}, peak {peak}"
        );

        for t in tmp {
            let _ = gpu.free_tensor(t);
        }
        xl.free_gpu(&mut gpu);
        for wt in weights_held {
            wt.free_all(&mut gpu);
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn xts_upload_failure_unwinds_cleanly_hw() {
        let cfg = tiny_cfg();
        let mut map = build(&cfg);
        // Expert 1's down sidecar goes missing after expert 0 already uploaded.
        map.remove(&crate::weights::awq_sidecar_name(
            "model.llm.layers.0.mlp.experts.1.down_proj.weight",
        ));
        let w = weights_with(map, None);
        let mut gpu = Gpu::init().expect("GPU required");
        {
            let mut tx = LoadTx::new(&mut gpu);
            let (mut bytes, mut nt) = (0u64, 0usize);
            let err = xts_upload_layer(&mut tx, &w, &cfg, 0, &mut bytes, &mut nt)
                .err()
                .expect("must fail");
            assert!(
                err.contains("missing") && err.contains("awq_scale"),
                "{err}"
            );
            // tx drops here uncommitted and frees everything it held.
        }
        let t = gpu
            .alloc_tensor(&[16], DType::F32)
            .expect("device still usable");
        let _ = gpu.free_tensor(t);
    }

    /// An attention-shaped qt44 weight with a non-unit F32 sidecar attached by the
    /// loader's own `xts_attach_awq` runs through `weight_gemv` (x / S before the
    /// FWHT) and equals the CPU `deq(W) / S` matvec; the F16 router goes through the
    /// existing F16 GEMV dispatch of `weight_gemv`.
    #[test]
    #[ignore = "needs a GPU"]
    fn xts_dense_awq_gemv_and_f16_router_match_cpu_hw() {
        use hipfire_runtime::llama::weight_gemv;
        let (m, k) = (128usize, 2304usize);
        let name = "model.llm.layers.0.attn.wk_dv.weight";
        let w_bytes = sym_qt44(m, k, 5);
        // Multiples of 1/8: exact in F16, so the F32 vector is the sidecar bit for bit.
        let s: Vec<f32> = (0..k).map(|i| 0.5 + (i % 9) as f32 * 0.375).collect();
        let mut map = std::collections::HashMap::new();
        put(&mut map, name, 44, &[m, k], w_bytes.clone());
        put(
            &mut map,
            &crate::weights::awq_sidecar_name(name),
            1,
            &[k],
            f16_vec(&s),
        );
        let w = weights_with(map, None);
        let mut gpu = Gpu::init().expect("GPU required");

        let mut x_state = 0x9e37_79b9u32;
        let mut rnd = move || {
            x_state = x_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (x_state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let x: Vec<f32> = (0..k).map(|_| rnd()).collect();
        let xt = gpu.upload_f32(&x, &[k]).unwrap();

        // Dense qt44 + sidecar.
        let mut tx = LoadTx::new(&mut gpu);
        let t = raw(&w, name).unwrap();
        let (wt, _) = wt_from_raw(&mut gpu, name, 44, &t.data, m, k).unwrap();
        let slot = tx.hold_weight(wt);
        let (mut b, mut n) = (0u64, 0usize);
        xts_attach_awq(&mut tx, &w, slot, name, k, &mut b, &mut n).unwrap();
        let wt = tx.take_weight(slot).unwrap();
        tx.commit();
        assert!(wt.awq_scale.is_some() && n == 1 && b == 4 * k as u64);
        let yt = gpu.alloc_tensor(&[m], DType::F32).unwrap();
        weight_gemv(&mut gpu, &wt, &xt, &yt).unwrap();
        let got = gpu.download_f32(&yt).unwrap();
        let base = hipfire_runtime::weight_backend::dequantize_weight_to_f32(44, &w_bytes, m * k);
        let want: Vec<f32> = (0..m)
            .map(|r| (0..k).map(|c| base[r * k + c] / s[c] * x[c]).sum())
            .collect();
        let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |a, (g, r)| a.max((g - r).abs()));
        eprintln!("dense awq gemv: worst {worst} peak {peak}");
        assert!(
            peak > 0.0 && worst <= 2e-3 * peak,
            "awq gemv: worst {worst} peak {peak}"
        );
        // Without the sidecar the result must differ (the divide is observable).
        let plain_want: Vec<f32> = (0..m)
            .map(|r| (0..k).map(|c| base[r * k + c] * x[c]).sum())
            .collect();
        let gap = plain_want
            .iter()
            .zip(&want)
            .fold(0.0f32, |a, (p, q)| a.max((p - q).abs()));
        assert!(
            gap > 10.0 * worst,
            "sidecar not observable: gap {gap} vs worst {worst}"
        );
        wt.free_all(&mut gpu);
        let _ = gpu.free_tensor(yt);

        // F16 router [128, 2304] through the F16 GEMV dispatch.
        let rvals: Vec<f32> = (0..128 * k).map(|_| rnd() * 0.5).collect();
        let rbytes = f16_vec(&rvals);
        let (router, _) = wt_from_raw(&mut gpu, "router", 1, &rbytes, 128, k).unwrap();
        assert_eq!(router.gpu_dtype, DType::F16);
        let ry = gpu.alloc_tensor(&[128], DType::F32).unwrap();
        weight_gemv(&mut gpu, &router, &xt, &ry).unwrap();
        let rgot = gpu.download_f32(&ry).unwrap();
        let rdec: Vec<f32> = rbytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let rwant: Vec<f32> = (0..128)
            .map(|r| (0..k).map(|c| rdec[r * k + c] * x[c]).sum())
            .collect();
        let rpeak = rwant.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let rworst = rgot
            .iter()
            .zip(&rwant)
            .fold(0.0f32, |a, (g, r)| a.max((g - r).abs()));
        eprintln!("f16 router gemv: worst {rworst} peak {rpeak}");
        assert!(
            rpeak > 0.0 && rworst <= 5e-3 * rpeak,
            "router: worst {rworst} peak {rpeak}"
        );
        router.free_all(&mut gpu);
        let _ = gpu.free_tensor(ry);
        let _ = gpu.free_tensor(xt);
    }
}
