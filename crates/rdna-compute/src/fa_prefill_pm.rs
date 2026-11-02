// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Certified gfx1201 H16/KV2/D256 Q8_0 and native fp8-E4M3 prefill: f32 Q -> f16 Q -> direct
//! attention. The Q preconvert (and the F5 Q-prep fold that replaces it) is shared; only the
//! direct body and the cache-row layout differ by [`FaPrefillKvCodec`].
//! Production admits 64..=8192 query rows. The exact oracle primitives also admit
//! 1..=63-row tails. Positions are packed i32 values, not row indices: duplicates,
//! nonmonotone order and nonaligned causal horizons are all supported.
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

pub const MODULE: &str = "fa_prefill_pm_gfx1201";
pub const PRECONVERT: &str = "attention_fa2_q_preconvert_pm_gfx1201";
pub const DIRECT: &str = "attention_q8_0_fa2_gqa_pm_gfx1201";
pub const IMAGE: &[u8] = include_bytes!("../../../kernels/fa_prefill_pm_gfx1201.hxaco");
pub const MIN_ROWS: usize = 64;
pub const MAX_ROWS: usize = 8192;
pub const MAX_CONTEXT: usize = 262_144;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 256;
const ROW: usize = HEADS * HEAD_DIM;
const CACHE_ROW_BYTES: usize = KV_HEADS * (HEAD_DIM / 32) * 34;
pub const PRE_KERNARG_BYTES: usize = 40;
pub const DIRECT_KERNARG_BYTES: usize = 60;
pub const PRE_BLOCK: [u32; 3] = [128, 1, 1];
pub const DIRECT_BLOCK: [u32; 3] = [160, 1, 1];
pub const DIRECT_LDS: u32 = 65_536;
/// FP8 direct launch dynamic LDS bytes: one 32-KiB stage reused sequentially
/// for K then V (four barrier pairs). Q8 keeps the two resident 32-KiB planes
/// behind [`DIRECT_LDS`]; this selection is private so the generated Q8 byte
/// path and its public launch constant stay untouched.
const FP8_DIRECT_LDS: u32 = 32_768;
/// Native fp8-E4M3 direct body: its own single-export module (the Q16 preconvert stays the Q8
/// module's [`PRECONVERT`]). Same 60-byte kernarg ABI, grid and block as [`DIRECT`]; LDS differs: the FP8 body runs the
/// single 32-KiB sequential K-then-V stage (`FP8_DIRECT_LDS`), not the Q8 two-plane 64-KiB layout.
pub const FP8_MODULE: &str = "fa_prefill_fp8_pm_gfx1201";
pub const FP8_DIRECT: &str = "attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201";
pub const FP8_IMAGE: &[u8] = include_bytes!("../../../kernels/fa_prefill_fp8_pm_gfx1201.hxaco");
/// Native fp8 token row: `[KV_HEADS*HEAD_DIM e4m3 codes (head-major)][KV_HEADS f16 scales]`.
const FP8_CACHE_ROW_BYTES: usize = KV_HEADS * (HEAD_DIM + 2);
const _: () = assert!(FP8_CACHE_ROW_BYTES == 516);

/// KV cache wire format a PM prefill body reads. `None` at an admission site means a format
/// no PM body serves (bf16, asym, windowed Q8, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaPrefillKvCodec {
    Q8_0,
    Fp8E4M3,
}

impl FaPrefillKvCodec {
    /// Bytes of one token row of one K or V side.
    pub const fn cache_row_bytes(self) -> usize {
        match self {
            Self::Q8_0 => CACHE_ROW_BYTES,
            Self::Fp8E4M3 => FP8_CACHE_ROW_BYTES,
        }
    }

    /// Embedded module holding this codec's direct export.
    pub const fn direct_module(self) -> &'static str {
        match self {
            Self::Q8_0 => MODULE,
            Self::Fp8E4M3 => FP8_MODULE,
        }
    }

    /// This codec's direct export.
    pub const fn direct_symbol(self) -> &'static str {
        match self {
            Self::Q8_0 => DIRECT,
            Self::Fp8E4M3 => FP8_DIRECT,
        }
    }

    fn direct_image(self) -> &'static [u8] {
        match self {
            Self::Q8_0 => IMAGE,
            Self::Fp8E4M3 => FP8_IMAGE,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Fp8E4M3 => "fp8-E4M3",
        }
    }

    /// Dynamic LDS bytes for this codec's direct launch: Q8 keeps the public
    /// two-plane 64-KiB layout; FP8 uses the single 32-KiB sequential stage.
    const fn direct_lds_bytes(self) -> u32 {
        match self {
            Self::Q8_0 => DIRECT_LDS,
            Self::Fp8E4M3 => FP8_DIRECT_LDS,
        }
    }
}

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("fa_prefill_pm: {}", message.as_ref()))
}

/// Pure production admission; no device access or allocation on success.
pub fn validate_domain(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rows: usize,
    context: usize,
) -> HipResult<()> {
    if (heads, kv_heads, head_dim) != (HEADS, KV_HEADS, HEAD_DIM) {
        return Err(err("requires H16/KV2/D256"));
    }
    if !(MIN_ROWS..=MAX_ROWS).contains(&rows) {
        return Err(err("production rows must be in 64..=8192"));
    }
    // The context range is codec independent; the Q8 row stride is the larger one.
    exact_extents(FaPrefillKvCodec::Q8_0, rows, context).map(|_| ())
}

/// Validate known host positions before uploading them. Device callers must
/// uphold the same precondition; the hot-path launcher does not read them back.
/// No alignment, uniqueness or monotonicity requirement is imposed.
pub fn validate_positions(positions: &[i32], capacity: usize) -> HipResult<()> {
    if capacity == 0 || capacity > MAX_CONTEXT {
        return Err(err("context must be in 1..=262144"));
    }
    for &position in positions {
        if position < 0 || position as usize >= capacity {
            return Err(err("position must be nonnegative and below cache context"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extents {
    q32: usize,
    q16: usize,
    cache: usize,
    positions: usize,
}

fn checked_bytes(count: usize, stride: usize) -> HipResult<usize> {
    count
        .checked_mul(stride)
        .ok_or_else(|| err("buffer extent overflow"))
}

fn exact_extents(codec: FaPrefillKvCodec, rows: usize, context: usize) -> HipResult<Extents> {
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(err("exact primitive rows must be in 1..=8192"));
    }
    if context == 0 || context > MAX_CONTEXT {
        return Err(err("context must be in 1..=262144"));
    }
    Ok(Extents {
        q32: checked_bytes(rows, ROW * 4)?,
        q16: checked_bytes(rows, ROW * 2)?,
        cache: checked_bytes(context, codec.cache_row_bytes())?,
        positions: checked_bytes(rows, 4)?,
    })
}

fn require_span(tensor: &GpuTensor, bytes: usize, name: &str) -> HipResult<()> {
    // Inspect the allocation span rather than numel()/byte_size(): shape products
    // can overflow and quantized dtypes do not describe their wire byte count.
    require_byte_span(tensor.buf.size(), bytes, name)
}

fn require_byte_span(have: usize, need: usize, name: &str) -> HipResult<()> {
    if have < need {
        return Err(err(format!("{name} needs {need} bytes, has {have}")));
    }
    Ok(())
}

fn require_cache(
    codec: FaPrefillKvCodec,
    tensor: &GpuTensor,
    bytes: usize,
    name: &str,
) -> HipResult<()> {
    // A VMM-backed native cache reports its mapped prefix, not the logical capacity (mapping
    // grows with use); admission already validated the reservation before publishing it.
    if codec == FaPrefillKvCodec::Fp8E4M3 && tensor.buf.is_vmm_owner() {
        return require_cache_dtype(codec, tensor.dtype);
    }
    require_cache_bytes(codec, tensor.dtype, tensor.buf.size(), bytes, name)
}

fn require_cache_dtype(codec: FaPrefillKvCodec, dtype: DType) -> HipResult<()> {
    // Production KvCache::new_gpu_q8* / new_gpu_fp8* planes are F32-tagged byte allocations.
    // The caller's route admission guarantees the actual wire format; this storage tag alone
    // does not describe the payload's quantization.
    let ok = match codec {
        FaPrefillKvCodec::Q8_0 => matches!(dtype, DType::Raw | DType::Q8_0 | DType::F32),
        FaPrefillKvCodec::Fp8E4M3 => matches!(dtype, DType::Raw | DType::F32),
    };
    if !ok {
        return Err(err(format!(
            "{} K/V require Raw{} or F32-tagged byte storage",
            codec.label(),
            if codec == FaPrefillKvCodec::Q8_0 {
                ", Q8_0"
            } else {
                ""
            },
        )));
    }
    Ok(())
}

pub(crate) fn require_cache_bytes(
    codec: FaPrefillKvCodec,
    dtype: DType,
    have: usize,
    need: usize,
    name: &str,
) -> HipResult<()> {
    require_cache_dtype(codec, dtype)?;
    require_byte_span(have, need, name)
}

fn pre_grid(rows: usize) -> [u32; 3] {
    // The launcher validates rows before reaching this arithmetic.
    [((rows * HEADS).div_ceil(4)) as u32, 1, 1]
}
fn direct_grid(rows: usize) -> [u32; 3] {
    [rows.div_ceil(8) as u32, KV_HEADS as u32, 1]
}

// Stack-resident blobs: exact explicit-argument sizes, no trailing segment pad.
// All ABI scalars are little endian, as are the supported host and GPU targets.
fn pre_kernargs(q: *mut c_void, q16: *mut c_void, rows: usize) -> [u8; PRE_KERNARG_BYTES] {
    let mut args = [0; PRE_KERNARG_BYTES];
    args[0..8].copy_from_slice(&(q as u64).to_le_bytes());
    args[8..16].copy_from_slice(&(q16 as u64).to_le_bytes());
    // signs1 @16, signs2 @24, do_rotate @36: ignored, frozen zero values.
    args[32..36].copy_from_slice(&(rows as i32).to_le_bytes());
    args
}

fn direct_kernargs(
    q16: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    out: *mut c_void,
    positions: *mut c_void,
    rows: usize,
) -> [u8; DIRECT_KERNARG_BYTES] {
    let mut args = [0; DIRECT_KERNARG_BYTES];
    for (offset, ptr) in [(0, q16), (8, k), (16, v), (24, out), (32, positions)] {
        args[offset..offset + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
    }
    for (offset, value) in [(40, HEADS), (44, KV_HEADS), (48, HEAD_DIM), (52, rows)] {
        args[offset..offset + 4].copy_from_slice(&(value as i32).to_le_bytes());
    }
    args[56..60].copy_from_slice(&(1.0f32 / 16.0).to_le_bytes());
    args
}

impl Gpu {
    fn fa_prefill_pm_require_eager(&self) -> HipResult<()> {
        if self.graphs.capture_mode || self.replay.is_recording() {
            return Err(err(
                "eager only: capture and replay recording are unsupported",
            ));
        }
        if self.arch != "gfx1201" {
            return Err(err("requires gfx1201; no HIP fallback"));
        }
        Ok(())
    }

    /// Load the images a launch needs. The Q16 preconvert always lives in the Q8 module, so the
    /// Q8 path loads that one module (both exports) exactly as before; the fp8 path loads the
    /// preconvert from it only when the launch preconverts, plus its own direct module.
    fn fa_prefill_pm_load(&mut self, codec: FaPrefillKvCodec, preconvert: bool) -> HipResult<()> {
        self.bind_thread()?;
        if preconvert || codec == FaPrefillKvCodec::Q8_0 {
            self.ensure_embedded_kernel(MODULE, IMAGE, PRECONVERT)?;
        }
        self.ensure_embedded_kernel(
            codec.direct_module(),
            codec.direct_image(),
            codec.direct_symbol(),
        )
    }

    fn fa_prefill_pm_enqueue_pre(
        &mut self,
        q: *mut c_void,
        q16: *mut c_void,
        rows: usize,
    ) -> HipResult<()> {
        let mut args = pre_kernargs(q, q16, rows);
        self.launch_blob_recorded(
            PRECONVERT,
            pre_grid(rows),
            PRE_BLOCK,
            0,
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fa_prefill_pm_enqueue_direct(
        &mut self,
        codec: FaPrefillKvCodec,
        q16: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        out: *mut c_void,
        positions: *mut c_void,
        rows: usize,
    ) -> HipResult<()> {
        let mut args = direct_kernargs(q16, k, v, out, positions, rows);
        self.launch_blob_recorded(
            codec.direct_symbol(),
            direct_grid(rows),
            DIRECT_BLOCK,
            codec.direct_lds_bytes(),
            &mut args,
            crate::dispatch::ReplayLaunchBindings::NONE,
        )
    }

    /// Production full causal attention for either codec. Q and output are packed f32
    /// `[rows,16,256]`; K/V are packed `[context,2,..]` rows of `codec`. `positions` holds at
    /// least `rows` packed i32 values satisfying [`validate_positions`] for `max_ctx_len`.
    /// Positions remain device-resident; their values are a caller precondition, not
    /// synchronously downloaded. No windows, slots, rotated cache, capture or replay recording.
    /// The unchanged Q is converted on the active stream into Gpu-owned F4b `fa2_q16_scratch`
    /// by the shared preconvert, then consumed by the codec's direct kernel on that stream.
    #[allow(clippy::too_many_arguments)]
    fn fa_prefill_pm_attention(
        &mut self,
        codec: FaPrefillKvCodec,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_require_eager()?;
        validate_domain(n_heads, n_kv_heads, head_dim, batch_size, max_ctx_len)?;
        let extents = exact_extents(codec, batch_size, max_ctx_len)?;
        if q.dtype != DType::F32 || out.dtype != DType::F32 {
            return Err(err("Q and output must be F32"));
        }
        require_span(q, extents.q32, "Q")?;
        require_span(out, extents.q32, "output")?;
        require_span(positions, extents.positions, "positions")?;
        require_cache(codec, k_cache, extents.cache, "K cache")?;
        require_cache(codec, v_cache, extents.cache, "V cache")?;
        self.fa_prefill_pm_load(codec, true)?;
        if crate::scratch::scratch_will_grow(
            self.scratch.fa2_q16_scratch_bytes,
            self.scratch.fa2_q16_scratch.is_some(),
            extents.q16,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let q16 = self
            .scratch
            .ensure_fa2_q16_scratch(&self.hip, extents.q16)?;
        self.fa_prefill_pm_enqueue_pre(q.buf.as_ptr(), q16, batch_size)?;
        self.fa_prefill_pm_enqueue_direct(
            codec,
            q16,
            k_cache.buf.as_ptr(),
            v_cache.buf.as_ptr(),
            out.buf.as_ptr(),
            positions.buf.as_ptr(),
            batch_size,
        )
    }

    /// Production full causal Q8_0 attention. Q and output are packed f32
    /// `[rows,16,256]`; K/V are packed `[context,2,8,34]` Q8_0 bytes. See
    /// [`Self::fa_prefill_pm_attention`] for the shared contract.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_q8_0_fa_prefill_pm_gfx1201(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_attention(
            FaPrefillKvCodec::Q8_0,
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx_len,
            batch_size,
        )
    }

    /// Production full causal native fp8-E4M3 attention. K/V are packed 516-byte token rows
    /// `[512 head-major e4m3 codes][2 f16 scales]`. Same arithmetic DAG as the Q8_0 route
    /// downstream of the KV dequant. See [`Self::fa_prefill_pm_attention`].
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fp8_e4m3_fa_prefill_pm_gfx1201(
        &mut self,
        q: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx_len: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_attention(
            FaPrefillKvCodec::Fp8E4M3,
            q,
            k_cache,
            v_cache,
            out,
            positions,
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx_len,
            batch_size,
        )
    }

    /// Exact preconvert primitive for guarded oracle scratch, including tiny tails.
    /// Production uses this same enqueue/packing implementation with owned scratch.
    #[doc(hidden)]
    pub fn fa_prefill_pm_preconvert(
        &mut self,
        q: &GpuTensor,
        q16: &GpuTensor,
        rows: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_require_eager()?;
        let extents = exact_extents(FaPrefillKvCodec::Q8_0, rows, 1)?;
        if q.dtype != DType::F32 || !matches!(q16.dtype, DType::F16 | DType::Raw) {
            return Err(err("preconvert requires F32 Q and F16/Raw Q16"));
        }
        require_span(q, extents.q32, "Q")?;
        require_span(q16, extents.q16, "Q16")?;
        self.fa_prefill_pm_load(FaPrefillKvCodec::Q8_0, true)?;
        self.fa_prefill_pm_enqueue_pre(q.buf.as_ptr(), q16.buf.as_ptr(), rows)
    }

    /// Exact Q8_0 direct primitive for guarded oracle scratch, including tiny tails.
    /// Positions obey the same device-resident caller precondition as production.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn fa_prefill_pm_direct(
        &mut self,
        q16: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        context: usize,
        rows: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_direct_codec(
            FaPrefillKvCodec::Q8_0,
            q16,
            k_cache,
            v_cache,
            out,
            positions,
            context,
            rows,
        )
    }

    /// Exact direct primitive of either codec (the F5 fold and the oracles launch it on an
    /// already prepared Q16; the fp8 path loads only its own direct module).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn fa_prefill_pm_direct_codec(
        &mut self,
        codec: FaPrefillKvCodec,
        q16: &GpuTensor,
        k_cache: &GpuTensor,
        v_cache: &GpuTensor,
        out: &GpuTensor,
        positions: &GpuTensor,
        context: usize,
        rows: usize,
    ) -> HipResult<()> {
        self.fa_prefill_pm_require_eager()?;
        let extents = exact_extents(codec, rows, context)?;
        if !matches!(q16.dtype, DType::F16 | DType::Raw) || out.dtype != DType::F32 {
            return Err(err("direct requires F16/Raw Q16 and F32 output"));
        }
        require_span(q16, extents.q16, "Q16")?;
        require_span(out, extents.q32, "output")?;
        require_span(positions, extents.positions, "positions")?;
        require_cache(codec, k_cache, extents.cache, "K cache")?;
        require_cache(codec, v_cache, extents.cache, "V cache")?;
        self.fa_prefill_pm_load(codec, false)?;
        self.fa_prefill_pm_enqueue_direct(
            codec,
            q16.buf.as_ptr(),
            k_cache.buf.as_ptr(),
            v_cache.buf.as_ptr(),
            out.buf.as_ptr(),
            positions.buf.as_ptr(),
            rows,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_buffer_extent_rejects_one_byte_short() {
        for codec in [FaPrefillKvCodec::Q8_0, FaPrefillKvCodec::Fp8E4M3] {
            let sizes = exact_extents(codec, 65, 123).unwrap();
            assert_eq!(sizes.cache, 123 * codec.cache_row_bytes());
            for (name, need) in [
                ("Q", sizes.q32),
                ("Q16", sizes.q16),
                ("output", sizes.q32),
                ("K cache", sizes.cache),
                ("V cache", sizes.cache),
                ("positions", sizes.positions),
            ] {
                assert!(require_byte_span(need, need, name).is_ok());
                assert!(require_byte_span(need + 1, need, name).is_ok());
                assert!(require_byte_span(need - 1, need, name).is_err());
            }
        }
    }

    #[test]
    fn q8_cache_storage_tags_preserve_byte_span_validation() {
        let codec = FaPrefillKvCodec::Q8_0;
        let need = exact_extents(codec, 64, 123).unwrap().cache;
        for dtype in [DType::Raw, DType::Q8_0, DType::F32] {
            assert!(require_cache_bytes(codec, dtype, need, need, "K cache").is_ok());
            assert!(require_cache_bytes(codec, dtype, need + 4, need, "V cache").is_ok());
            assert!(require_cache_bytes(codec, dtype, need - 1, need, "K cache").is_err());
        }
        for dtype in [DType::F16, DType::BF16, DType::Q4K] {
            assert!(require_cache_bytes(codec, dtype, need, need, "K cache").is_err());
        }
    }

    #[test]
    fn fp8_cache_is_516_byte_rows_with_raw_or_f32_storage_only() {
        let codec = FaPrefillKvCodec::Fp8E4M3;
        assert_eq!(codec.cache_row_bytes(), 516);
        assert_eq!(FaPrefillKvCodec::Q8_0.cache_row_bytes(), 544);
        let need = exact_extents(codec, 64, 123).unwrap().cache;
        assert_eq!(need, 123 * 516);
        for dtype in [DType::Raw, DType::F32] {
            assert!(require_cache_bytes(codec, dtype, need, need, "K cache").is_ok());
            assert!(require_cache_bytes(codec, dtype, need - 1, need, "K cache").is_err());
        }
        // Q8_0-tagged and every non-byte tag is a Q8/other-format signal, never fp8 storage.
        for dtype in [DType::Q8_0, DType::F16, DType::BF16, DType::Q4K] {
            assert!(require_cache_bytes(codec, dtype, need, need, "K cache").is_err());
        }
    }

    #[test]
    fn codecs_select_distinct_direct_exports_and_share_the_preconvert_module() {
        assert_eq!(FaPrefillKvCodec::Q8_0.direct_module(), MODULE);
        assert_eq!(FaPrefillKvCodec::Q8_0.direct_symbol(), DIRECT);
        assert_eq!(FaPrefillKvCodec::Fp8E4M3.direct_module(), FP8_MODULE);
        assert_eq!(FaPrefillKvCodec::Fp8E4M3.direct_symbol(), FP8_DIRECT);
        assert_ne!(MODULE, FP8_MODULE);
        assert_ne!(DIRECT, FP8_DIRECT);
        assert!(!FP8_IMAGE.is_empty());
    }

    #[test]
    fn codec_direct_lds_splits_q8_two_plane_from_fp8_single_stage() {
        // Q8 keeps the public two-plane 64-KiB layout and its generated byte
        // path; FP8 launches the single 32-KiB sequential K-then-V stage.
        assert_eq!(DIRECT_LDS, 65_536);
        assert_eq!(FP8_DIRECT_LDS, 32_768);
        assert_eq!(FaPrefillKvCodec::Q8_0.direct_lds_bytes(), DIRECT_LDS);
        assert_eq!(FaPrefillKvCodec::Fp8E4M3.direct_lds_bytes(), FP8_DIRECT_LDS);
    }

    #[test]
    fn launcher_constants_match_certified_authoring_contract() {
        use peacemaker_isa::kernels::fa_prefill as author;
        assert_eq!(MODULE, author::MODULE);
        assert_eq!(PRECONVERT, author::SYMBOL_PRECONVERT);
        assert_eq!(DIRECT, author::SYMBOL_DIRECT);
        assert_eq!(PRE_KERNARG_BYTES as u32, author::PRECONVERT_KERNARG_BYTES);
        assert_eq!(DIRECT_KERNARG_BYTES as u32, author::DIRECT_KERNARG_BYTES);
        assert_eq!(PRE_BLOCK[0], author::PRECONVERT_BLOCK);
        assert_eq!(DIRECT_BLOCK[0], author::DIRECT_BLOCK);
        assert_eq!(DIRECT_LDS, author::DYNAMIC_LDS_BYTES);
        assert_eq!(MAX_ROWS, author::MAX_BATCH);
        assert_eq!(FP8_MODULE, author::MODULE_FP8);
        assert_eq!(FP8_DIRECT, author::SYMBOL_FP8_DIRECT);
        assert_eq!(FP8_CACHE_ROW_BYTES as u32, author::FP8_ROW_BYTES);
        assert_eq!((KV_HEADS * HEAD_DIM) as u32, author::FP8_SCALE_OFFSET);
    }

    #[test]
    fn production_domain_and_exact_tail_are_distinct() {
        for rows in [64, 65, 127, 8191, 8192] {
            assert!(validate_domain(16, 2, 256, rows, MAX_CONTEXT).is_ok());
        }
        for rows in [0, 1, 63, 8193, usize::MAX] {
            assert!(validate_domain(16, 2, 256, rows, 8192).is_err());
        }
        for shape in [
            (24, 4, 256),
            (16, 4, 256),
            (16, 2, 128),
            (usize::MAX, 2, 256),
        ] {
            assert!(validate_domain(shape.0, shape.1, shape.2, 64, 8192).is_err());
        }
        for rows in [1, 7, 63] {
            for codec in [FaPrefillKvCodec::Q8_0, FaPrefillKvCodec::Fp8E4M3] {
                assert!(exact_extents(codec, rows, 8192).is_ok());
            }
        }
        for context in [0, MAX_CONTEXT + 1, usize::MAX] {
            assert!(validate_domain(16, 2, 256, 64, context).is_err());
        }
        assert!(checked_bytes(usize::MAX, 544).is_err());
        assert_eq!(
            exact_extents(FaPrefillKvCodec::Q8_0, 8192, MAX_CONTEXT).unwrap(),
            Extents {
                q32: 134_217_728,
                q16: 67_108_864,
                cache: 142_606_336,
                positions: 32_768,
            }
        );
        assert_eq!(
            exact_extents(FaPrefillKvCodec::Fp8E4M3, 8192, MAX_CONTEXT).unwrap(),
            Extents {
                q32: 134_217_728,
                q16: 67_108_864,
                cache: 135_266_304,
                positions: 32_768,
            }
        );
    }

    #[test]
    fn positions_accept_arbitrary_causal_order() {
        assert!(validate_positions(&[65, 0, 65, 7, 3, 2], 66).is_ok());
        assert!(validate_positions(&[MAX_CONTEXT as i32 - 1], MAX_CONTEXT).is_ok());
        for positions in [&[-1][..], &[66][..], &[i32::MAX][..]] {
            assert!(validate_positions(positions, 66).is_err());
        }
        assert!(validate_positions(&[0], 0).is_err());
        assert!(validate_positions(&[0], usize::MAX).is_err());
    }

    #[test]
    fn frozen_preconvert_abi_and_grid() {
        let args = pre_kernargs(0x1000usize as *mut c_void, 0x2000usize as *mut c_void, 65);
        assert_eq!(args.len(), 40);
        assert_eq!(u64::from_le_bytes(args[0..8].try_into().unwrap()), 0x1000);
        assert_eq!(u64::from_le_bytes(args[8..16].try_into().unwrap()), 0x2000);
        assert_eq!(&args[16..32], &[0; 16]);
        assert_eq!(i32::from_le_bytes(args[32..36].try_into().unwrap()), 65);
        assert_eq!(&args[36..40], &[0; 4]);
        assert_eq!(PRE_BLOCK, [128, 1, 1]);
        for rows in [1, 7, 8, 9, 63, 64, 65, 8191, 8192] {
            assert_eq!(pre_grid(rows), [(rows * 4) as u32, 1, 1]);
        }
    }

    #[test]
    fn frozen_direct_abi_and_grid() {
        let args = direct_kernargs(
            0x1000usize as *mut c_void,
            0x2000usize as *mut c_void,
            0x3000usize as *mut c_void,
            0x4000usize as *mut c_void,
            0x5000usize as *mut c_void,
            8191,
        );
        assert_eq!(args.len(), 60);
        for (i, value) in [0x1000u64, 0x2000, 0x3000, 0x4000, 0x5000]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                u64::from_le_bytes(args[i * 8..i * 8 + 8].try_into().unwrap()),
                value
            );
        }
        for (i, value) in [16, 2, 256, 8191].into_iter().enumerate() {
            assert_eq!(
                i32::from_le_bytes(args[40 + i * 4..44 + i * 4].try_into().unwrap()),
                value
            );
        }
        assert_eq!(f32::from_le_bytes(args[56..60].try_into().unwrap()), 0.0625);
        assert_eq!(DIRECT_BLOCK, [160, 1, 1]);
        assert_eq!(DIRECT_LDS, 65536);
        for rows in [1, 7, 8, 9, 63, 64, 65, 8191, 8192] {
            assert_eq!(direct_grid(rows), [rows.div_ceil(8) as u32, 2, 1]);
        }
    }
}
