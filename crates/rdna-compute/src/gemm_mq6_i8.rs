// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-INT8 prefill GEMM for MQ6G256V2 weights on exact gfx1201 and gfx1151.
//!
//! The 6-bit codes enter `v_wmma_i32_16x16x16_iu8` raw and unsigned
//! (`kernels/src/gemm_mq6g256v2_mmq_i8.gfx12.hip` / `.gfx11.hip`); activations
//! are the existing A8 `block_i8_128` blocks (136 B per token per 128 K, layout
//! `[K/128][N]`) the MQ4 A8 route already produces. Per (row, token, 128-K
//! block) the accumulator is seeded `I8_MAGIC - 32*s` so it ends at
//! `I8_MAGIC + sum (q - 32) * qx` (`|C| <= 32*127*128 < 2^22`, exact in f32),
//! then folded in K order with `fma(RN(s_h*d), C, acc)` and
//! `fma(RN(z'_h*d), s, acc)`, `z'_h = z_h + 32*s_h`. Output F32 or BF16 bits
//! (RNE, the incumbent's formula), overwrite (SET).
//!
//! Producer ([`Gpu::mq6_i8_prepare_x`]): one pass, [`Gpu::rotate_x_mq_i8_batched`]
//! (`mq_rotate_x_i8_gfx12` / `_gfx11`): the same FWHT as the F16 route emitting the
//! A8 blocks in registers, the 128-K blocks coinciding with the MQ6 128-halves.
//! The blocks are byte-identical to the former two-pass chain
//! (`rotate_x_mq_batched` into an F32 scratch, then `ensure_int8_mmq_x`).
//!
//! Admission: default on for exact gfx1201 inside the Qwen4 forward at
//! `route_rows >= QWEN4_F16_WMMA_MIN_TOKENS` (the F16 PM route it replaces);
//! measured on Flash-Next pp16384 +1.1% with KLD inside the bar
//! (`release-0.4.3/fp8-lean/RESULTS.md`). gfx1151 is opt-in only: there the
//! iu8 rate equals F16 and the route measured 0.79-0.95x of the incumbent GEMM.
//! Env (developer): `HIPFIRE_MQ6_I8` (`1` on for any scope, `0` off, unset =
//! the default above); `HIPFIRE_MQ6_I8_PM` (the GEMM itself runs the certified
//! builder (PeaceMaker) twin `kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201` on exact
//! gfx1201 whenever [`mq6_i8_pm_admits`] holds, else the hipcc kernel; `0` = hipcc
//! always, `1` and unset = the twin where admitted). The twin's output is
//! bit-identical to the hipcc kernel's.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernel_abi;
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use hipfire_kernel_abi::families::mq6_i8::{
    FORMAT_BF16_NORM_256, FORMAT_BF16_N_M, FORMAT_BLOCK_I8_128_KB_N, FORMAT_F32_N_M,
    FORMAT_F32_ROPE_N_64, FORMAT_MQ6G256V2_WEIGHTS,
};
use hipfire_kernel_abi::{InvocationV1, MAX_FIELDS};
use std::ffi::c_void;

/// MQ6G256V2 group bytes (8 B header + 192 B 6-bit payload).
const MQ6_GROUP_BYTES: usize = 200;
/// A8 `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_I8_128_BYTES: usize = 136;

/// Archs where the unset-flag default runs the INT8 route in qwen4 scope.
pub const MQ6_I8_DEFAULT_ARCHS: &[&str] = &["gfx1201"];

/// Dynamic LDS of the gfx12 entries (header constant `MQ6_I8_GFX12_LDS_BYTES`
/// of `gemm_mq6g256v2_mmq_i8.gfx12.hip`: two K64 slots of 8192 B A + 8192 B
/// unpacked W, plus 2 x 1024 B DS and 2 x 1024 B SZ planes).
const MQ6_I8_GFX12_LDS_BYTES: u32 = 36864;
/// Dynamic LDS of the gfx11 128-token build (512 threads).
const MQ6_I8_GFX11_LDS_BYTES: u32 = 32768;
/// Dynamic LDS of the gfx11 256-token build (1024 threads, `_t256` entries).
const MQ6_I8_GFX11_T256_LDS_BYTES: u32 = 65536;

/// Embedded module of the PeaceMaker twin (gfx1201) and its entries.
const MQ6_I8_PM_MODULE: &str = "gemm_mq6g256v2_mmq_i8_pm_gfx1201";
const MQ6_I8_PM_ENTRY_F32: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201";
const MQ6_I8_PM_ENTRY_BF16: &str = "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201";
/// The 128 x 256 entries of the same module (16 waves, 512 threads).
const MQ6_I8_PM_ENTRY_F32_T256: &str = "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201_t128x256";
const MQ6_I8_PM_ENTRY_BF16_T256: &str = "gemm_mq6g256v2_mmq_i8_pm_set_bf16_gfx1201_t128x256";
/// The 256 x 128 entry of the same module (16 waves, 512 threads): F32 output
/// whose head-aligned 256-row tiles come out RMS-normed, weighted and RoPE'd
/// (QSA q/k norm+RoPE epilogue).
const MQ6_I8_PM_ENTRY_F32_NORM_ROPE: &str = "gemm_mq6g256v2_mmq_i8_pm_set_nr_gfx1201_t256x128";
/// Rows per CTA and tokens per CTA of that entry, its threads and dynamic LDS.
const MQ6_I8_PM_NR_ROWS: usize = 256;
const MQ6_I8_PM_NR_TOKENS: usize = 128;
/// `HIPFIRE_QWEN4_QSA_NORM_EPI` (developer, kill switch): on unless `0`; acted on for exact gfx1201 only.
static QWEN4_QSA_NORM_EPI: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_NORM_EPI", true));
/// Head width and RoPE width the entry is built for.
pub const QSA_NORM_EPI_HEAD_DIM: usize = 256;
pub const QSA_NORM_EPI_ROTARY_DIM: usize = 64;
/// Rows per CTA of both PM tiles.
const MQ6_I8_PM_ROWS: usize = 128;
/// Token-tile widths of the PM tiles: the 128 x 128 tile (256 threads,
/// [`MQ6_I8_GFX12_LDS_BYTES`]) and the 128 x 256 tile (512 threads).
const MQ6_I8_PM_TOKENS_128: usize = 128;
const MQ6_I8_PM_TOKENS_256: usize = 256;
const MQ6_I8_PM_T256_THREADS: u32 = 512;
const MQ6_I8_PM_T256_LDS_BYTES: u32 = 55296;

/// Token-tile width of the PM launch of `m` rows x `n` tokens on a device of `cu_count`
/// CUs (two per WGP). The 128 x 256 tile wins by ~6-8 % (fewer weight decodes and folds per
/// MAC) when its CTAs spread as evenly over the WGPs as the 128 x 128 ones do, and loses up
/// to 37 % when its grid leaves WGPs idle or one WGP an extra round (card B, `fn-10k` runs
/// `i8pm/ship/json{,2}`). It is used iff
/// 1. it pads no more tokens than the 128-token tile (`n % 256` is 0 or above 128), and
/// 2. the most loaded WGP does not carry more work: a WGP's load is its CTA count times the
///    tile's token width, CTAs round-robin over the WGPs, and the wide tile's per-token cost is
///    taken as 0.94 of the narrow one's: `2 * 94 * ceil(c256 / wgps) <= 100 * ceil(c128 / wgps)`.
fn mq6_i8_pm_token_tile(m: usize, n: usize, cu_count: usize) -> usize {
    let r = n % MQ6_I8_PM_TOKENS_256;
    if r != 0 && r <= MQ6_I8_PM_TOKENS_128 {
        return MQ6_I8_PM_TOKENS_128;
    }
    let wgps = (cu_count / 2).max(1);
    let rows = m.div_ceil(MQ6_I8_PM_ROWS);
    let rounds256 = (rows * n.div_ceil(MQ6_I8_PM_TOKENS_256)).div_ceil(wgps);
    let rounds128 = (rows * n.div_ceil(MQ6_I8_PM_TOKENS_128)).div_ceil(wgps);
    if 188 * rounds256 <= 100 * rounds128 {
        MQ6_I8_PM_TOKENS_256
    } else {
        MQ6_I8_PM_TOKENS_128
    }
}

/// `(entry, threads, dynamic LDS, token tile)` of one PM launch of `m` x `n` tokens.
fn mq6_i8_pm_launch(
    m: usize,
    n: usize,
    cu_count: usize,
    bf16: bool,
) -> (&'static str, u32, u32, usize) {
    match (mq6_i8_pm_token_tile(m, n, cu_count), bf16) {
        (MQ6_I8_PM_TOKENS_256, false) => (
            MQ6_I8_PM_ENTRY_F32_T256,
            MQ6_I8_PM_T256_THREADS,
            MQ6_I8_PM_T256_LDS_BYTES,
            MQ6_I8_PM_TOKENS_256,
        ),
        (MQ6_I8_PM_TOKENS_256, true) => (
            MQ6_I8_PM_ENTRY_BF16_T256,
            MQ6_I8_PM_T256_THREADS,
            MQ6_I8_PM_T256_LDS_BYTES,
            MQ6_I8_PM_TOKENS_256,
        ),
        (_, false) => (
            MQ6_I8_PM_ENTRY_F32,
            256,
            MQ6_I8_GFX12_LDS_BYTES,
            MQ6_I8_PM_TOKENS_128,
        ),
        (_, true) => (
            MQ6_I8_PM_ENTRY_BF16,
            256,
            MQ6_I8_GFX12_LDS_BYTES,
            MQ6_I8_PM_TOKENS_128,
        ),
    }
}

/// Whether the PeaceMaker twin runs one launch (`flag` = `HIPFIRE_MQ6_I8_PM`): not
/// switched off (`0`; unset and `1` both admit), exact gfx1201, `k % 256 == 0`, `m % 4 == 0` (its row quads are stored
/// whole), a `y` aligned to 4 bytes (F32) or 8 bytes (BF16: its quad stores are `buffer_store_b64`), and the 32-bit address contract (`m`, `k`, `n`
/// and the output and weight extents under 2^31 bytes). Anything else keeps the
/// hipcc kernel.
fn mq6_i8_pm_admits(
    flag: Option<bool>,
    arch: &str,
    m: usize,
    k: usize,
    n: usize,
    out_elem_bytes: usize,
    y_ptr: usize,
) -> bool {
    flag != Some(false)
        && arch == "gfx1201"
        && k > 0
        && k % 256 == 0
        && m > 0
        && m % 4 == 0
        && n > 0
        && y_ptr % if out_elem_bytes == 2 { 8 } else { 4 } == 0
        && m.max(k).max(n) <= i32::MAX as usize
        && crate::gemm::pm_output_extent_bytes(n, m, out_elem_bytes).is_some()
        && m.checked_mul(k / 256)
            .and_then(|g| g.checked_mul(MQ6_GROUP_BYTES))
            .is_some_and(|w| w <= crate::gemm::PM_OUTPUT_EXTENT_LIMIT_BYTES)
}

/// Scope decision of the MQ6 INT8 route past the device/shape preconditions:
/// `Some(false)` off, `Some(true)` on for any scope, unset on in qwen4 scope
/// iff `arch` is in [`MQ6_I8_DEFAULT_ARCHS`] and `route_rows` reaches
/// [`crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS`].
fn mq6_i8_scope_admits(
    flag: Option<bool>,
    arch: &str,
    qwen4_scope: bool,
    route_rows: usize,
) -> bool {
    match flag {
        Some(on) => on,
        None => {
            qwen4_scope
                && MQ6_I8_DEFAULT_ARCHS.contains(&arch)
                && route_rows >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        }
    }
}

/// `(module, source, block threads, dynamic LDS bytes, CTA tile)` of one
/// launch of `rows` weight rows x `n` tokens. gfx1201 runs the 128 x 128 tile;
/// gfx1151 takes the 256 x 256 build (one 1024-thread CTA per WGP) whenever it
/// still yields at least `cu_count` CTAs (the A8 MQ4 rule, `a8_launch`), else
/// the 128 x 128 build. `None` off those archs.
fn mq6_i8_launch(
    arch: &str,
    rows: usize,
    n: usize,
    cu_count: usize,
) -> Option<(&'static str, &'static str, u32, u32, usize)> {
    match arch {
        "gfx1201" => Some((
            "gemm_mq6g256v2_mmq_i8_gfx12",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC,
            256,
            MQ6_I8_GFX12_LDS_BYTES,
            128,
        )),
        "gfx1151" if rows.div_ceil(256) * n.div_ceil(256) >= cu_count => Some((
            "gemm_mq6g256v2_mmq_i8_gfx11_t256",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_T256_SRC,
            1024,
            MQ6_I8_GFX11_T256_LDS_BYTES,
            256,
        )),
        "gfx1151" => Some((
            "gemm_mq6g256v2_mmq_i8_gfx11",
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_SRC,
            512,
            MQ6_I8_GFX11_LDS_BYTES,
            128,
        )),
        _ => None,
    }
}

/// Entry symbol of the SET kernel in the module [`mq6_i8_launch`] picked.
fn mq6_i8_entry(arch: &str, tile: usize, bf16: bool) -> &'static str {
    match (arch == "gfx1201", tile, bf16) {
        (true, _, false) => "gemm_mq6g256v2_mmq_i8_set_gfx12",
        (true, _, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12",
        (false, 256, false) => "gemm_mq6g256v2_mmq_i8_set_gfx11_t256",
        (false, 256, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256",
        (false, _, false) => "gemm_mq6g256v2_mmq_i8_set_gfx11",
        (false, _, true) => "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11",
    }
}

impl Gpu {
    /// Whether the MQ6 INT8 route admits one launch of `m` weight rows, `k`
    /// and `batch_size` tokens: exact gfx1201 or gfx1151, eager (no replay
    /// recording / graph capture), `k % 256 == 0`, non-empty dims, and the
    /// scope decision (`HIPFIRE_MQ6_I8`: `1` any scope, `0` off, unset = qwen4
    /// scope on [`MQ6_I8_DEFAULT_ARCHS`] at `route_rows >= 512`).
    pub fn gemm_mq6g256v2_i8_applies(&self, m: usize, k: usize, batch_size: usize) -> bool {
        matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && k > 0
            && k % 256 == 0
            && m > 0
            && batch_size > 0
            && mq6_i8_scope_admits(
                self.flags.mq6_i8,
                self.arch.as_str(),
                self.qwen4_scope,
                self.route_rows(batch_size),
            )
    }

    /// Name of the kernel [`Gpu::gemm_mq6g256v2_i8_prequant`] launches for these
    /// arguments (the PeaceMaker twin entry or the hipcc one); `None` off gfx1201 /
    /// gfx1151.
    pub fn gemm_mq6g256v2_i8_kernel_name(
        &self,
        m: usize,
        k: usize,
        batch_size: usize,
        y: &GpuTensor,
    ) -> Option<&'static str> {
        let bf16 = y.dtype == DType::BF16;
        if mq6_i8_pm_admits(
            self.flags.mq6_i8_pm,
            &self.arch,
            m,
            k,
            batch_size,
            if bf16 { 2 } else { 4 },
            y.buf.as_ptr() as usize,
        ) {
            return Some(if bf16 {
                MQ6_I8_PM_ENTRY_BF16
            } else {
                MQ6_I8_PM_ENTRY_F32
            });
        }
        let (_, _, _, _, tile) =
            mq6_i8_launch(&self.arch, m, batch_size, self.cu_count_or_default())?;
        Some(mq6_i8_entry(&self.arch, tile, bf16))
    }

    /// Rotate F32 `x` (`[batch_size x k]`) with the MQ FWHT (same signs as
    /// [`Gpu::rotate_x_mq_batched_f16`]) and quantize it into A8 `block_i8_128`
    /// blocks (`[k/128][batch_size]`) in one fused launch
    /// ([`Gpu::rotate_x_mq_i8_batched`]).
    ///
    /// The returned pointer belongs to this `Gpu` and stays valid until the
    /// next INT8 activation producer call.
    pub fn mq6_i8_prepare_x(
        &mut self,
        x: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<*mut c_void> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "mq6_i8_prepare_x: eager-only"));
        }
        if x.dtype != DType::F32 {
            return Err(HipError::new(0, "mq6_i8_prepare_x: x must be F32"));
        }
        if !matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
            || k == 0
            || k % 256 != 0
            || batch_size == 0
        {
            return Err(HipError::new(
                0,
                "mq6_i8_prepare_x: exact gfx1201/gfx1151, K%256==0, N>0 required",
            ));
        }
        let elems = batch_size
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "mq6_i8_prepare_x: batch*k overflow"))?;
        if x.numel() < elems {
            return Err(HipError::new(0, "mq6_i8_prepare_x: undersized x"));
        }
        // `x` may be a residual owed an add (the deferred fold): read it flushed.
        self.flush_residual_fold()?;
        // One pass: FWHT rotate straight into the A8 blocks (no F32 round trip).
        let reservation = self.reserve_int8_mmq(k, batch_size)?;
        let prepared = self.rotate_x_mq_i8_batched(x, None, reservation, k, batch_size)?;
        self.int8_mmq_prepared_ptr(&prepared, k, batch_size)
    }

    /// Overwrite `y` (`[batch_size x m]`, F32, or BF16 bits when
    /// `y.dtype == DType::BF16`) with `W . X` for MQ6G256V2 weights `a`
    /// (`m` rows x `k/256` groups x 200 B) and the A8 blocks `xq` of
    /// [`Gpu::mq6_i8_prepare_x`]. `m` and `batch_size` need not be tile
    /// multiples. gfx1151 takes the 256-token build when
    /// `ceil(m/256) * ceil(n/256) >= cu_count`, else the 128-token one.
    pub fn gemm_mq6g256v2_i8_prequant(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "gemm_mq6g256v2_i8: eager-only"));
        }
        if m == 0 || batch_size == 0 || k == 0 || k % 256 != 0 || xq.is_null() {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: need M>0, N>0, K>0, K%256==0 and prepared blocks",
            ));
        }
        let bf16 = match y.dtype {
            DType::F32 => false,
            DType::BF16 => true,
            _ => return Err(HipError::new(0, "gemm_mq6g256v2_i8: y must be F32 or BF16")),
        };
        let weight_bytes = m
            .checked_mul(k / 256)
            .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_i8: weight bytes overflow"))?;
        let y_elems = batch_size
            .checked_mul(m)
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_i8: batch*m overflow"))?;
        let out_bytes = y_elems * if bf16 { 2 } else { 4 };
        if a.buf.size() < weight_bytes || y.buf.size() < out_bytes {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: undersized weight or output tensor",
            ));
        }
        let Some((module, source, threads, lds, tile)) =
            mq6_i8_launch(&self.arch, m, batch_size, self.cu_count_or_default())
        else {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8: exact gfx1201 or gfx1151 required",
            ));
        };
        let pm = mq6_i8_pm_admits(
            self.flags.mq6_i8_pm,
            &self.arch,
            m,
            k,
            batch_size,
            if bf16 { 2 } else { 4 },
            y.buf.as_ptr() as usize,
        );
        let (kernel, threads, lds, tile_tokens, tile_rows) = if pm {
            let (e, th, l, t) = mq6_i8_pm_launch(m, batch_size, self.cu_count_or_default(), bf16);
            (e, th, l, t, MQ6_I8_PM_ROWS)
        } else {
            (
                mq6_i8_entry(&self.arch, tile, bf16),
                threads,
                lds,
                tile,
                tile,
            )
        };
        // One typed contract per kernel (hipfire-kernel-abi): every operand
        // format/alignment/extent, the launch geometry and the kernarg image
        // are validated and encoded once, before the module is ensured.
        let arch = kernel_abi::parse_arch(&self.arch)?;
        let abi = kernel_abi::contract(&self.arch, kernel)?;
        let y_format = if bf16 {
            FORMAT_BF16_N_M
        } else {
            FORMAT_F32_N_M
        };
        let operands = [
            kernel_abi::binding(a, FORMAT_MQ6G256V2_WEIGHTS),
            kernel_abi::binding_raw(xq, self.int8_mmq_xq_bytes(xq), FORMAT_BLOCK_I8_128_KB_N),
            kernel_abi::binding(y, y_format),
        ];
        let grid = [
            m.div_ceil(tile_rows) as u32,
            batch_size.div_ceil(tile_tokens) as u32,
            1,
        ];
        let block = [threads, 1, 1];
        let mut v = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch,
                symbol: kernel,
                params: &[m as u64, k as u64, batch_size as u64],
                operands: &operands,
                grid,
                block,
                dynamic_lds: lds,
            },
        )?;
        if pm {
            self.ensure_embedded_kernel(
                MQ6_I8_PM_MODULE,
                kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201,
                kernel,
            )?;
        } else {
            self.ensure_kernel(module, source, kernel)?;
        }
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
        let nptrs = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
        let bytes = weight_bytes + batch_size * (k / 128) * BLOCK_I8_128_BYTES + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(kernel, grid, block, lds, &mut ptrs[..nptrs], || {
            kernel_abi::blob(v.explicit())
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // `y` was overwritten: caches keyed on its pointer are stale.
        self.invalidate_x_caches_for(y.buf.as_ptr());
        result
    }

    /// Whether [`Gpu::gemm_mq6g256v2_i8_prequant_norm_rope`] would launch the
    /// fused entry for these arguments: `HIPFIRE_QWEN4_QSA_NORM_EPI` on (exact
    /// gfx1201), the MQ6 INT8 route admitted ([`Gpu::gemm_mq6g256v2_i8_applies`]),
    /// the PM twin admitted for F32 `y`, `m % 256 == 0`, and `head_stride_rows`
    /// (`head_mask + 1`: 512 for the interleaved q-gate rows, 256 for plain k
    /// rows) dividing `m`.
    pub fn gemm_mq6g256v2_i8_norm_rope_applies(
        &self,
        m: usize,
        k: usize,
        batch_size: usize,
        y: &GpuTensor,
        head_stride_rows: usize,
    ) -> bool {
        (self.arch == "gfx1201" && *QWEN4_QSA_NORM_EPI)
            && y.dtype == DType::F32
            && matches!(head_stride_rows, 256 | 512)
            && m % head_stride_rows == 0
            && m % MQ6_I8_PM_NR_ROWS == 0
            && self.gemm_mq6g256v2_i8_applies(m, k, batch_size)
            && mq6_i8_pm_admits(
                self.flags.mq6_i8_pm,
                &self.arch,
                m,
                k,
                batch_size,
                4,
                y.buf.as_ptr() as usize,
            )
    }

    /// [`Gpu::gemm_mq6g256v2_i8_prequant`] into F32 `y` (`[batch_size x m]`)
    /// through the `..._pm_set_nr_gfx1201_t256x128` entry, which also applies
    /// the QSA RMSNorm (`norm`, BF16 `[256]`, `1 + w` weight), then the
    /// partial RoPE (`rope_table`: F32 `[batch_size][64]` of
    /// [`crate::tensor_ops::indexed_attention_rope_table`]) to every 256-wide
    /// head whose row start `rs` has `(rs & (head_stride_rows - 1)) < 256`.
    /// The result is byte-identical to the plain GEMM followed by
    /// `indexed_attention_norm_rope_batch` over those heads; other rows
    /// (the gate half of q-gate rows) are the plain GEMM output.
    ///
    /// Returns `Ok(true)` when it launched, `Ok(false)` (nothing launched)
    /// when [`Gpu::gemm_mq6g256v2_i8_norm_rope_applies`] is false, so the
    /// caller runs the unfused chain.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq6g256v2_i8_prequant_norm_rope(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        head_stride_rows: usize,
        norm: &GpuTensor,
        rope_table: &GpuTensor,
    ) -> HipResult<bool> {
        self.bind_thread()?;
        if !self.gemm_mq6g256v2_i8_norm_rope_applies(m, k, batch_size, y, head_stride_rows) {
            return Ok(false);
        }
        if xq.is_null()
            || norm.dtype != DType::BF16
            || norm.numel() < QSA_NORM_EPI_HEAD_DIM
            || rope_table.dtype != DType::F32
            || batch_size
                .checked_mul(QSA_NORM_EPI_ROTARY_DIM)
                .is_none_or(|e| rope_table.numel() < e)
        {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8_norm_rope: need prepared blocks, BF16 norm[256] and an F32 rope table of N rows x 64",
            ));
        }
        let weight_bytes = m
            .checked_mul(k / 256)
            .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
            .ok_or_else(|| {
                HipError::new(0, "gemm_mq6g256v2_i8_norm_rope: weight bytes overflow")
            })?;
        let out_bytes = batch_size
            .checked_mul(m)
            .and_then(|e| e.checked_mul(4))
            .ok_or_else(|| HipError::new(0, "gemm_mq6g256v2_i8_norm_rope: batch*m overflow"))?;
        if a.buf.size() < weight_bytes || y.buf.size() < out_bytes {
            return Err(HipError::new(
                0,
                "gemm_mq6g256v2_i8_norm_rope: undersized weight or output tensor",
            ));
        }
        let kernel = MQ6_I8_PM_ENTRY_F32_NORM_ROPE;
        let arch = kernel_abi::parse_arch(&self.arch)?;
        let abi = kernel_abi::contract(&self.arch, kernel)?;
        let operands = [
            kernel_abi::binding(a, FORMAT_MQ6G256V2_WEIGHTS),
            kernel_abi::binding_raw(xq, self.int8_mmq_xq_bytes(xq), FORMAT_BLOCK_I8_128_KB_N),
            kernel_abi::binding(y, FORMAT_F32_N_M),
            kernel_abi::binding(norm, FORMAT_BF16_NORM_256),
            kernel_abi::binding(rope_table, FORMAT_F32_ROPE_N_64),
        ];
        let grid = [
            (m / MQ6_I8_PM_NR_ROWS) as u32,
            batch_size.div_ceil(MQ6_I8_PM_NR_TOKENS) as u32,
            1,
        ];
        let block = [MQ6_I8_PM_T256_THREADS, 1, 1];
        let mut v = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch,
                symbol: kernel,
                params: &[
                    m as u64,
                    k as u64,
                    batch_size as u64,
                    head_stride_rows as u64,
                ],
                operands: &operands,
                grid,
                block,
                dynamic_lds: MQ6_I8_PM_T256_LDS_BYTES,
            },
        )?;
        self.ensure_embedded_kernel(
            MQ6_I8_PM_MODULE,
            kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201,
            kernel,
        )?;
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
        let nptrs = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
        let bytes = weight_bytes
            + batch_size * (k / 128) * BLOCK_I8_128_BYTES
            + out_bytes
            + batch_size * QSA_NORM_EPI_ROTARY_DIM * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(
            kernel,
            grid,
            block,
            MQ6_I8_PM_T256_LDS_BYTES,
            &mut ptrs[..nptrs],
            || kernel_abi::blob(v.explicit()),
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // `y` was overwritten: caches keyed on its pointer are stale.
        self.invalidate_x_caches_for(y.buf.as_ptr());
        result.map(|()| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_is_gfx1201_qwen4_prefill_only() {
        assert_eq!(MQ6_I8_DEFAULT_ARCHS, &["gfx1201"]);
        assert!(mq6_i8_scope_admits(None, "gfx1201", true, 512));
        assert!(mq6_i8_scope_admits(None, "gfx1201", true, 8192));
        assert!(!mq6_i8_scope_admits(None, "gfx1201", true, 511));
        assert!(!mq6_i8_scope_admits(None, "gfx1201", false, 8192));
        for arch in ["gfx1151", "gfx1100"] {
            assert!(!mq6_i8_scope_admits(None, arch, true, 8192));
        }
    }

    #[test]
    fn flag_forces_on_any_scope_and_off_wins() {
        for scope in [false, true] {
            for arch in ["gfx1201", "gfx1151"] {
                assert!(mq6_i8_scope_admits(Some(true), arch, scope, 1));
                assert!(!mq6_i8_scope_admits(Some(false), arch, scope, 8192));
            }
        }
    }

    #[test]
    fn launch_selection_per_arch() {
        // gfx1201: one 128x128 build, 256 threads.
        for (rows, n) in [(48, 1), (10240, 4096), (248_320, 8192)] {
            let (module, _, threads, lds, tile) = mq6_i8_launch("gfx1201", rows, n, 48).unwrap();
            assert_eq!(
                (module, threads, lds, tile),
                (
                    "gemm_mq6g256v2_mmq_i8_gfx12",
                    256,
                    MQ6_I8_GFX12_LDS_BYTES,
                    128
                )
            );
        }
        // gfx1151 (cu_count 20): t256 once ceil(m/256)*ceil(n/256) >= 20.
        let t128 = mq6_i8_launch("gfx1151", 2560, 512, 20).unwrap(); // 10 * 2 = 20 -> t256
        assert_eq!(t128.0, "gemm_mq6g256v2_mmq_i8_gfx11_t256");
        assert_eq!(
            (t128.2, t128.3, t128.4),
            (1024, MQ6_I8_GFX11_T256_LDS_BYTES, 256)
        );
        let small = mq6_i8_launch("gfx1151", 2560, 256, 20).unwrap(); // 10 * 1 = 10 -> t128
        assert_eq!(small.0, "gemm_mq6g256v2_mmq_i8_gfx11");
        assert_eq!(
            (small.2, small.3, small.4),
            (512, MQ6_I8_GFX11_LDS_BYTES, 128)
        );
        assert!(mq6_i8_launch("gfx1100", 2560, 512, 20).is_none());
        assert!(mq6_i8_launch("gfx942", 2560, 512, 20).is_none());
        // Budgets stay within the 64 KiB LDS of one CU.
        for lds in [
            MQ6_I8_GFX12_LDS_BYTES,
            MQ6_I8_GFX11_LDS_BYTES,
            MQ6_I8_GFX11_T256_LDS_BYTES,
        ] {
            assert!(lds <= 65536);
        }
    }

    #[test]
    fn pm_twin_is_the_gfx1201_default_with_the_32_bit_address_contract() {
        let ok = |flag, arch, m, k, n, eb, y| mq6_i8_pm_admits(flag, arch, m, k, n, eb, y);
        assert!(ok(Some(true), "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(ok(Some(true), "gfx1201", 48, 256, 1, 2, 8));
        assert!(ok(Some(true), "gfx1201", 48, 256, 1, 4, 4));
        assert!(!ok(Some(true), "gfx1201", 48, 256, 1, 2, 4));
        // FN's largest M at a vocabulary-sized token count stays inside 2^31 bytes.
        assert!(ok(Some(true), "gfx1201", 248_320, 5120, 2162, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 248_320, 5120, 4000, 4, 256));
        // Unset is the default: PM wherever admitted, hipcc where not; `0` keeps hipcc.
        assert!(ok(None, "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(!ok(None, "gfx1201", 248_320, 5120, 4000, 4, 256));
        assert!(!ok(Some(false), "gfx1201", 2560, 5120, 4000, 4, 256));
        assert!(!ok(Some(false), "gfx1201", 48, 256, 1, 2, 8));
        // Other archs, K % 256, M % 4, an unaligned Y and empty dims fall back.
        assert!(!ok(Some(true), "gfx1151", 2560, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5000, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2562, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5120, 4000, 4, 258));
        assert!(!ok(Some(true), "gfx1201", 0, 5120, 4000, 4, 256));
        assert!(!ok(Some(true), "gfx1201", 2560, 5120, 0, 4, 256));
        // The embedded image carries all four entries.
        for entry in [
            MQ6_I8_PM_ENTRY_F32,
            MQ6_I8_PM_ENTRY_BF16,
            MQ6_I8_PM_ENTRY_F32_T256,
            MQ6_I8_PM_ENTRY_BF16_T256,
        ] {
            assert!(
                kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201
                    .windows(entry.len())
                    .any(|w| w == entry.as_bytes()),
                "image lacks {entry}"
            );
        }
        // The 128 x 128 entries use the hipcc gfx12 build's threads and LDS.
        let (_, _, threads, lds, tile) = mq6_i8_launch("gfx1201", 2560, 512, 48).unwrap();
        assert_eq!(
            (threads, lds, tile),
            (256, MQ6_I8_GFX12_LDS_BYTES, MQ6_I8_PM_TOKENS_128)
        );
    }

    /// The 128 x 256 tile is the default; the 128 x 128 one serves launches where the
    /// wide tile pads more tokens or loads the WGPs worse. The table is card B's measured
    /// `(M, N, K = 2560) -> 128x256 / 128x128` time ratio (64 CUs).
    #[test]
    fn pm_tile_follows_the_padding_and_balance_rule() {
        // (m, n, measured 128x256 / 128x128 time ratio): wide wins iff ratio < 1.
        let measured: [(usize, usize, f64); 28] = [
            (10240, 16384, 0.930),
            (10240, 4096, 0.933),
            (12288, 16384, 0.940),
            (12288, 4096, 0.932),
            (6144, 16384, 0.934),
            (6144, 4096, 0.937),
            (2560, 16384, 0.919),
            (2560, 4096, 0.919),
            (512, 16384, 0.940),
            (512, 4096, 0.952),
            (128, 16384, 0.951),
            (128, 4096, 1.336),
            (48, 4096, 1.372),
            (48, 16384, 0.945),
            (128, 6144, 0.931),
            (128, 8192, 0.969),
            (128, 12288, 1.125),
            (256, 4096, 0.996),
            (256, 8192, 0.960),
            (384, 4096, 1.119),
            (384, 8192, 0.940),
            (512, 1024, 1.357),
            (512, 2048, 0.960),
            (640, 2048, 1.126),
            (640, 4096, 1.053),
            (1024, 1024, 0.957),
            (1024, 2048, 0.953),
            (2560, 1024, 1.064),
        ];
        for (m, n, ratio) in measured {
            let (e, th, l, t) = mq6_i8_pm_launch(m, n, 64, false);
            let wide = t == 256;
            // The 0.996 point is a tie either way.
            if ratio != 0.996 {
                assert_eq!(wide, ratio < 1.0, "m={m} n={n} ratio={ratio}");
            }
            if wide {
                assert_eq!((e, th, l), (MQ6_I8_PM_ENTRY_F32_T256, 512, 55296));
            } else {
                assert_eq!(
                    (e, th, l),
                    (MQ6_I8_PM_ENTRY_F32, 256, MQ6_I8_GFX12_LDS_BYTES)
                );
            }
        }
        assert_eq!(
            mq6_i8_pm_launch(10240, 16384, 64, true).0,
            MQ6_I8_PM_ENTRY_BF16_T256
        );
        assert_eq!(mq6_i8_pm_launch(48, 4096, 64, true).0, MQ6_I8_PM_ENTRY_BF16);
        // Token tails: the wide tile never pads more tokens than the narrow one.
        for n in [
            1usize,
            48,
            128,
            129,
            255,
            383,
            384,
            385,
            4097,
            4224,
            16385,
            16384 + 129,
        ] {
            for m in [48usize, 128, 512, 640, 2560, 10240, 248_320] {
                let t = mq6_i8_pm_token_tile(m, n, 64);
                let pad = |t: usize| n.div_ceil(t) * t - n;
                assert!(pad(t) <= pad(128), "m={m} n={n}");
            }
        }
        assert_eq!(mq6_i8_pm_token_tile(10240, 4097, 64), 128);
        assert_eq!(mq6_i8_pm_token_tile(10240, 4096 + 129, 64), 256);
        // Grid rows of every Flash-Next M (ragged M tiles are masked by the kernel).
        for (m, gx) in [
            (48usize, 1usize),
            (128, 1),
            (512, 4),
            (640, 5),
            (2560, 20),
            (6144, 48),
            (10240, 80),
            (12288, 96),
            (248_320, 1940),
        ] {
            assert_eq!(m.div_ceil(MQ6_I8_PM_ROWS), gx, "m={m}");
        }
    }

    #[test]
    fn entries_follow_arch_tile_and_output() {
        assert_eq!(
            mq6_i8_entry("gfx1201", 128, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx12"
        );
        assert_eq!(
            mq6_i8_entry("gfx1201", 128, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx12"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 128, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx11"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 128, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 256, false),
            "gemm_mq6g256v2_mmq_i8_set_gfx11_t256"
        );
        assert_eq!(
            mq6_i8_entry("gfx1151", 256, true),
            "gemm_mq6g256v2_mmq_i8_set_bf16_gfx11_t256"
        );
    }

    /// Every entry the launcher names exists in the module source it compiles
    /// from, and the gfx12 LDS contract matches the kernel header.
    #[test]
    fn kernel_sources_carry_the_launcher_symbols_and_lds_contract() {
        let sets = [
            (
                kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC,
                "gfx1201",
                128usize,
            ),
            (kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_SRC, "gfx1151", 128),
            (
                kernels::GEMM_MQ6G256V2_MMQ_I8_GFX11_T256_SRC,
                "gfx1151",
                256,
            ),
        ];
        for (src, arch, tile) in sets {
            for bf16 in [false, true] {
                let entry = mq6_i8_entry(arch, tile, bf16);
                // The gfx11 source names its entries through `I8_ENTRY(base)`,
                // which appends `_t256` under `I8_TILE 256`.
                let (base, t256) = match entry.strip_suffix("_t256") {
                    Some(base) => (base, true),
                    None => (entry, false),
                };
                let named = if arch == "gfx1201" {
                    src.contains(base)
                } else {
                    src.contains(&format!("I8_ENTRY({base})"))
                };
                assert!(named, "source lacks entry `{base}`");
                assert_eq!(
                    src.contains("#define I8_TILE 256"),
                    t256,
                    "tile define of `{entry}`"
                );
            }
            assert!(
                src.contains("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1"),
                "module must not re-export the standalone A8 quantizer"
            );
        }
        let gfx12 = format!("#define MQ6_I8_GFX12_LDS_BYTES {MQ6_I8_GFX12_LDS_BYTES}");
        assert!(
            kernels::GEMM_MQ6G256V2_MMQ_I8_GFX12_SRC.contains(&gfx12),
            "gfx12 LDS contract drifted from the kernel header"
        );
    }

    /// The one-pass producer is the plain A8 rotate recipe: quantizer prelude without its
    /// standalone kernel, the emit aliases, the entry define, then the unmodified rotate body
    /// with neither the sigmoid gate nor AWQ.
    #[test]
    fn one_pass_producer_sources_are_the_plain_a8_rotate_recipe() {
        for (src, entry) in [
            (kernels::MQ_ROTATE_X_I8_GFX12_SRC, "mq_rotate_x_i8_gfx12"),
            (kernels::MQ_ROTATE_X_I8_GFX11_SRC, "mq_rotate_x_i8_gfx11"),
        ] {
            assert!(src.starts_with("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n"));
            assert!(src.contains(&format!("#define HIPFIRE_ROTATE_KERNEL {entry}\n")));
            assert!(src.contains(
                "#define emit_iu4_sidecar_from_producer8 emit_i8_sidecar_from_producer8"
            ));
            for gate in [
                "#define HIPFIRE_ROTATE_SIGMOID_GATE",
                "#define HIPFIRE_ROTATE_AWQ",
                "#define HIPFIRE_IU4_RUNTIME_CAND",
            ] {
                assert!(!src.contains(gate), "{entry}: unexpected {gate}");
            }
            let emit = src.find("emit_i8_sidecar_from_producer8(").unwrap();
            let body = src.find("HIPFIRE_ROTATE_KERNEL(").unwrap();
            assert!(emit < body, "{entry}: emit must precede the rotate body");
        }
    }

    #[test]
    fn ensure_kernel_module_names_are_stable() {
        // The lazy-kernel allowlist rows name these three modules.
        let names: Vec<&str> = [
            ("gfx1201", 48, 1),
            ("gfx1151", 2560, 512),
            ("gfx1151", 48, 1),
        ]
        .iter()
        .map(|&(a, r, n)| mq6_i8_launch(a, r, n, 20).unwrap().0)
        .collect();
        assert_eq!(
            names,
            [
                "gemm_mq6g256v2_mmq_i8_gfx12",
                "gemm_mq6g256v2_mmq_i8_gfx11_t256",
                "gemm_mq6g256v2_mmq_i8_gfx11"
            ]
        );
    }

    // ---- KernelAbiV1 contract coverage (pure CPU) ----

    use hipfire_kernel_abi::{eval, families, AbiError, Arch, KernelAbiV1, OperandBinding};

    const FAKE_BYTES: u64 = 1 << 40;

    /// `expected` of the export `symbol` on `arch` against what the launcher
    /// derives: block, dynamic LDS and every grid axis.
    fn assert_export_matches(
        arch: Arch,
        symbol: &str,
        threads: u32,
        lds: u32,
        grid: [u32; 3],
        params: &[u64],
    ) {
        let (abi, export) = families::find_export(arch, symbol)
            .unwrap_or_else(|| panic!("no contract export for {symbol} on {}", arch.name()));
        assert_eq!(export.block, [threads, 1, 1], "{symbol} block");
        assert_eq!(export.dynamic_lds, lds, "{symbol} dynamic LDS");
        for axis in 0..3 {
            assert_eq!(
                eval(&export.grid[axis], abi, export, params).unwrap(),
                u64::from(grid[axis]),
                "{symbol} grid[{axis}] for {params:?}"
            );
        }
    }

    /// Every launch any selector can pick (PM entries on gfx1201, hipcc entries
    /// on gfx1201 and gfx1151, F32 and BF16) names an export whose block, LDS
    /// and grid are what the launcher passes.
    #[test]
    fn contract_covers_every_selectable_launch() {
        let ms = [4usize, 48, 128, 129, 256, 2560, 10240, 248_320];
        let ns = [1usize, 127, 128, 129, 255, 256, 257, 384, 4096, 16385];
        let mut seen = std::collections::BTreeSet::new();
        for arch in ["gfx1201", "gfx1151"] {
            let parsed = kernel_abi::parse_arch(arch).unwrap();
            for bf16 in [false, true] {
                for cu in [20usize, 48, 64] {
                    for &m in &ms {
                        for &n in &ns {
                            let params = [m as u64, 256, n as u64];
                            let (_, _, threads, lds, tile) = mq6_i8_launch(arch, m, n, cu).unwrap();
                            let entry = mq6_i8_entry(arch, tile, bf16);
                            assert_export_matches(
                                parsed,
                                entry,
                                threads,
                                lds,
                                [m.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1],
                                &params,
                            );
                            seen.insert((arch, entry));
                            if arch == "gfx1201" {
                                let (entry, threads, lds, tile) = mq6_i8_pm_launch(m, n, cu, bf16);
                                assert_export_matches(
                                    parsed,
                                    entry,
                                    threads,
                                    lds,
                                    [
                                        m.div_ceil(MQ6_I8_PM_ROWS) as u32,
                                        n.div_ceil(tile) as u32,
                                        1,
                                    ],
                                    &params,
                                );
                                seen.insert((arch, entry));
                            }
                        }
                    }
                }
            }
        }
        // PM: 2 tiles x {F32, BF16}; hipcc gfx12 x {F32, BF16}; hipcc gfx11 x 2 builds x {F32, BF16}.
        assert_eq!(seen.len(), 4 + 2 + 4, "{seen:?}");
        // The fused norm/RoPE entry.
        for (m, n) in [(256usize, 1usize), (2560, 129), (5120, 4096)] {
            assert_export_matches(
                Arch::Gfx1201,
                MQ6_I8_PM_ENTRY_F32_NORM_ROPE,
                MQ6_I8_PM_T256_THREADS,
                MQ6_I8_PM_T256_LDS_BYTES,
                [
                    (m / MQ6_I8_PM_NR_ROWS) as u32,
                    n.div_ceil(MQ6_I8_PM_NR_TOKENS) as u32,
                    1,
                ],
                &[m as u64, 256, n as u64, 256],
            );
        }
    }

    fn fake_bindings(abi: &KernelAbiV1) -> Vec<OperandBinding> {
        abi.operands
            .iter()
            .enumerate()
            .map(|(i, o)| OperandBinding {
                address: 0x1000_0000 * (i as u64 + 1),
                bytes: Some(FAKE_BYTES),
                format: o.format,
            })
            .collect()
    }

    /// The kernarg tail the removed per-launcher packing appended after
    /// `A, Xq, Y, M, K, N`.
    #[derive(Clone, Copy)]
    enum LegacyTail {
        /// hipcc: nothing.
        None,
        /// PM SET: the seventh by-value word `unused`.
        Unused,
        /// PM NR: `head_mask`, the runtime's hidden block up to 296, `norm`, `rope`.
        NormRope { head_mask: i32 },
    }

    /// Byte image exactly as the removed `KernargBlob` closures built it (fake
    /// pointers `0x1000_0000 * (i + 1)`), tail-padded to 16.
    fn legacy_image(tail: LegacyTail, [m, k, n]: [i32; 3]) -> Vec<u8> {
        let ptr = |i: usize| (0x1000_0000usize * (i + 1)) as *const c_void;
        let mut b = hip_bridge::KernargBlob::new();
        b.push_ptr(ptr(0));
        b.push_ptr(ptr(1));
        b.push_ptr(ptr(2));
        b.push_i32(m);
        b.push_i32(k);
        b.push_i32(n);
        match tail {
            LegacyTail::None => {}
            LegacyTail::Unused => b.push_i32(0),
            LegacyTail::NormRope { head_mask } => {
                b.push_i32(head_mask);
                while b.len() < 296 {
                    b.push_u32(0);
                }
                b.push_ptr(ptr(3));
                b.push_ptr(ptr(4));
            }
        }
        b.pad_to(16);
        b.as_bytes().to_vec()
    }

    /// Validate one launch through the contract and compare its encoded image
    /// (and the `kernelParams` pointers) with the legacy packing.
    #[allow(clippy::too_many_arguments)]
    fn assert_bytes_equal_legacy(
        arch: Arch,
        symbol: &str,
        params: &[u64],
        grid: [u32; 3],
        threads: u32,
        lds: u32,
        tail: LegacyTail,
        nparams: usize,
    ) {
        let abi = kernel_abi::contract(arch.name(), symbol).unwrap();
        let operands = fake_bindings(abi);
        let mut v = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch,
                symbol,
                params,
                operands: &operands,
                grid,
                block: [threads, 1, 1],
                dynamic_lds: lds,
            },
        )
        .unwrap_or_else(|e| panic!("{symbol}: {e}"));
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
        let n = kernel_abi::kernel_params(&mut v, &mut ptrs).unwrap();
        assert_eq!(n, nparams, "{symbol} kernelParams count");
        let mut blob = kernel_abi::blob(v.explicit());
        blob.pad_to(16);
        let legacy = legacy_image(tail, [params[0] as i32, params[1] as i32, params[2] as i32]);
        assert_eq!(blob.as_bytes(), legacy.as_slice(), "{symbol} blob bytes");
        // SAFETY: the pointers address `v`'s explicit image, live and unmoved here.
        unsafe {
            for (i, p) in ptrs[..3].iter().enumerate() {
                assert_eq!(
                    std::ptr::read_unaligned(*p as *const u64),
                    0x1000_0000 * (i as u64 + 1),
                    "{symbol} ptr {i}"
                );
            }
            for (i, p) in ptrs[3..6].iter().enumerate() {
                assert_eq!(
                    std::ptr::read_unaligned(*p as *const i32) as u64,
                    params[i],
                    "{symbol} dim {i}"
                );
            }
        }
    }

    #[test]
    fn contract_bytes_equal_legacy_packing() {
        let g = Arch::Gfx1201;
        // PM F32: 128 x 128 tile and the t128x256 tile.
        for (m, k, n, cu, wide) in [
            (2560usize, 5120usize, 100usize, 64usize, false),
            (10240, 2560, 4096, 64, true),
        ] {
            let (entry, threads, lds, tile) = mq6_i8_pm_launch(m, n, cu, false);
            assert_eq!(tile == MQ6_I8_PM_TOKENS_256, wide);
            assert_bytes_equal_legacy(
                g,
                entry,
                &[m as u64, k as u64, n as u64],
                [
                    m.div_ceil(MQ6_I8_PM_ROWS) as u32,
                    n.div_ceil(tile) as u32,
                    1,
                ],
                threads,
                lds,
                LegacyTail::Unused,
                7,
            );
        }
        // PM BF16 (both tiles).
        for (m, k, n, cu) in [
            (48usize, 256usize, 1usize, 64usize),
            (10240, 2560, 4096, 64),
        ] {
            let (entry, threads, lds, tile) = mq6_i8_pm_launch(m, n, cu, true);
            assert_bytes_equal_legacy(
                g,
                entry,
                &[m as u64, k as u64, n as u64],
                [
                    m.div_ceil(MQ6_I8_PM_ROWS) as u32,
                    n.div_ceil(tile) as u32,
                    1,
                ],
                threads,
                lds,
                LegacyTail::Unused,
                7,
            );
        }
        // PM norm/RoPE: head stride 256 (plain k rows) and 512 (q-gate rows).
        for (m, k, n, hsr) in [
            (2560usize, 2560usize, 300usize, 256usize),
            (5120, 2560, 4096, 512),
        ] {
            assert_bytes_equal_legacy(
                g,
                MQ6_I8_PM_ENTRY_F32_NORM_ROPE,
                &[m as u64, k as u64, n as u64, hsr as u64],
                [
                    (m / MQ6_I8_PM_NR_ROWS) as u32,
                    n.div_ceil(MQ6_I8_PM_NR_TOKENS) as u32,
                    1,
                ],
                MQ6_I8_PM_T256_THREADS,
                MQ6_I8_PM_T256_LDS_BYTES,
                LegacyTail::NormRope {
                    head_mask: (hsr - 1) as i32,
                },
                9,
            );
        }
        // hipcc: gfx1201 (F32, BF16), gfx1151 128-token build and t256 build.
        for (arch, m, k, n, cu, bf16, t256) in [
            (
                "gfx1201", 2560usize, 5120usize, 100usize, 64usize, false, false,
            ),
            ("gfx1201", 2562, 2560, 4000, 64, true, false),
            ("gfx1151", 2560, 2560, 256, 20, false, false),
            ("gfx1151", 2560, 2560, 512, 20, false, true),
            ("gfx1151", 2560, 2560, 256, 20, true, false),
            ("gfx1151", 2560, 2560, 512, 20, true, true),
        ] {
            let (_, _, threads, lds, tile) = mq6_i8_launch(arch, m, n, cu).unwrap();
            assert_eq!(tile == 256, t256, "{arch} {m}x{n}");
            let entry = mq6_i8_entry(arch, tile, bf16);
            assert_bytes_equal_legacy(
                kernel_abi::parse_arch(arch).unwrap(),
                entry,
                &[m as u64, k as u64, n as u64],
                [m.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1],
                threads,
                lds,
                LegacyTail::None,
                6,
            );
        }
    }

    /// The validator refuses, before any enqueue, the launches the legacy path
    /// would have let through or caught only by the admission predicate.
    #[test]
    fn contract_rejects_unsafe_launches_before_enqueue() {
        let abi = &families::mq6_i8::MQ6_I8_PM_SET_BF16;
        let symbol = MQ6_I8_PM_ENTRY_BF16;
        let (m, k, n) = (2560u64, 2560u64, 100u64);
        let weight_bytes = m * (k / 256) * 200;
        let run = |k: u64, a_bytes: u64, y_addr: u64, threads: u32, lds: u32| {
            let operands = [
                OperandBinding {
                    address: 0x1000_0000,
                    bytes: Some(a_bytes),
                    format: FORMAT_MQ6G256V2_WEIGHTS,
                },
                OperandBinding {
                    address: 0x2000_0000,
                    bytes: None,
                    format: FORMAT_BLOCK_I8_128_KB_N,
                },
                OperandBinding {
                    address: y_addr,
                    bytes: Some(FAKE_BYTES),
                    format: FORMAT_BF16_N_M,
                },
            ];
            hipfire_kernel_abi::validate_invocation(
                abi,
                &InvocationV1 {
                    arch: Arch::Gfx1201,
                    symbol,
                    params: &[m, k, n],
                    operands: &operands,
                    grid: [20, 1, 1],
                    block: [threads, 1, 1],
                    dynamic_lds: lds,
                },
            )
            .map(|v| v.unchecked_extents().map(|o| o.name).collect::<Vec<_>>())
        };
        // The admitted launch passes; the unbound Xq extent is recorded, not proven.
        assert_eq!(
            run(k, weight_bytes, 0x3000_0000, 256, MQ6_I8_GFX12_LDS_BYTES).unwrap(),
            ["Xq"]
        );
        // BF16 Y at 4 mod 8: the quad stores are buffer_store_b64.
        assert!(matches!(
            run(k, weight_bytes, 0x3000_0004, 256, MQ6_I8_GFX12_LDS_BYTES),
            Err(AbiError::Misaligned { operand: "Y", .. })
        ));
        // K not a multiple of 256 (2560 + 128).
        assert!(matches!(
            run(
                2560 + 128,
                weight_bytes,
                0x3000_0000,
                256,
                MQ6_I8_GFX12_LDS_BYTES
            ),
            Err(AbiError::Domain { .. })
        ));
        // One byte short of the weight extent.
        assert!(matches!(
            run(
                k,
                weight_bytes - 1,
                0x3000_0000,
                256,
                MQ6_I8_GFX12_LDS_BYTES
            ),
            Err(AbiError::Extent { operand: "A", .. })
        ));
        // Block and dynamic LDS of another entry.
        assert!(matches!(
            run(k, weight_bytes, 0x3000_0000, 512, MQ6_I8_GFX12_LDS_BYTES),
            Err(AbiError::Launch { what: "block", .. })
        ));
        assert!(matches!(
            run(
                k,
                weight_bytes,
                0x3000_0000,
                256,
                MQ6_I8_GFX12_LDS_BYTES + 4
            ),
            Err(AbiError::Launch { .. })
        ));
        // The launcher-facing helper maps the rejection to a HipError naming the symbol.
        let operands = fake_bindings(abi);
        let err = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch: Arch::Gfx1201,
                symbol,
                params: &[m, k, n],
                operands: &operands,
                grid: [20, 1, 1],
                block: [128, 1, 1],
                dynamic_lds: MQ6_I8_GFX12_LDS_BYTES,
            },
        )
        .map(|_| ())
        .unwrap_err();
        assert!(err.to_string().contains(symbol), "{err}");
    }
}
