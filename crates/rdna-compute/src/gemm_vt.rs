// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12 MQ4G256V2 "verify-tile" dense WMMA GEMMs for 16 < N < 64 rows.
//!
//! The multi-request speculative-verify trunk packs several requests'
//! 4-row (MTP) / 16-row (DFlash) blocks into one forward of N <= 63 rows. The
//! one-tile gfx12 kernels (`gemm_{qkvza,qkv,gate_up}_mq4g256v2_wmma_gfx12`,
//! `gemm_mq4g256v2_residual_wmma_gfx12`) launch one wave per 16x16 tile, so
//! every block re-streams the whole activation panel (N x K fp16) through the
//! caches for just 16 weight rows: the GEMM is activation-bandwidth bound and
//! its cost scales ~linearly with N (batch-tiling inside one wave, the BT4
//! kernels, shares the weight fragment but not that traffic). The
//! `_vt{2,3,4}w{4,8}` kernels
//! (`kernels/src/gemm_mq4g256v2_wmma_gfx12_vt_core.hip`) run 4 or 8 waves per
//! block, each owning 16 weight rows, over the whole batch panel (BT =
//! ceil(N/16) 16-row tiles, one grid column): each 128-K slab of the panel is
//! staged in LDS once per block and feeds BT independent accumulators per wave.
//! Per (weight row, batch row) output the arithmetic is the one-tile kernel's
//! exactly (same WMMA chain, K order, headers, dequant, epilogue), so rows are
//! byte-identical to the one-tile launch at any N.
//!
//! Selection ([`Gpu::gfx12_verify_bt_active`]): exact gfx1201, `16 < N < 64`,
//! K % 256 == 0, outside capture/replay recording (the historical launch
//! contract), and `HIPFIRE_WMMA_BATCH_TILES` not `0`. N <= 16 never takes it.
//!
//! # Exact verify entry points, 1 <= N <= 128
//!
//! Aggregated speculative verify (continuous-batching DFlash block16 x C8) is
//! 128 rows. Above 63 the *generic* dispatch switches arithmetic (IU4 W4A4 /
//! FP8 quantised activations), so the 64..=128 window is reached only through
//! the explicit `Gpu::gemm_*_verify_exact` methods below; ordinary prefill
//! dispatch and every singleton path keep their selection unchanged. Each
//! method validates its domain (exact gfx1201, `K % 256 == 0`,
//! `1 <= N <= 128`) before any launch and picks one launch by `N`:
//!
//! - `N <= 16`: the named singleton one-tile WMMA kernel (`gemm_*_wmma_gfx12`).
//! - `17 <= N <= 63`: the `_vt{2,3,4}w{4,8}` kernels above (the one-tile
//!   kernel, grid.y = ceil(N / 16), when `HIPFIRE_WMMA_BATCH_TILES=0`).
//! - `64 <= N <= 128`: the K32-slab wide kernels (BT4 at N = 64, BT8 above;
//!   LDS 10 KiB / 20 KiB), per-output arithmetic identical to the singleton.
//!   They need `wmma_batch_tiles`; otherwise the call fails closed.
//!
//! The wide twins ship twice with one ABI, grid and output byte contract:
//! the PeaceMaker bundle ([`kernels::MQ4_VERIFY_PM_GFX1201`], symbols
//! `mq4_verify_{qkvza,qkv,gate_up,residual}_pm_gfx1201_bt{4,8}w{4,8}`) and the
//! hipcc symbols `gemm_*_wmma_gfx12_vt{4,8}w{4,8}_k32` in the `_vt` modules.
//! `HIPFIRE_CB_VERIFY_PM` (default on) selects PM; `=0` selects the hipcc
//! twins. A PM bundle or symbol that fails to load is an error, never a hipcc
//! fallback. [`Gpu::mq4_verify_chunk_rows`] is the admission capacity
//! (`HIPFIRE_CB_VERIFY_CHUNK128`, default on for gfx1201, `=0` kills). The launches use fixed-size
//! argument arrays: no per-launch allocation beyond the existing blob path.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;

/// Wide (`w8`) blocks share each staged activation slab across more weight
/// rows and win when there are enough 16-row tiles to keep every CU busy;
/// narrow (`w4`) blocks keep small-M projections (residual) spread over the
/// GPU. Cut measured on gfx1201 (R9700) at N = 32..63.
const VT_WIDE_MIN_ROW_TILES: usize = 1024;

/// Symbol table `[BT - 2][wide]` for a projection prefix.
macro_rules! vt_symbols {
    ($p:literal) => {
        [
            [concat!($p, "_vt2w4"), concat!($p, "_vt2w8")],
            [concat!($p, "_vt3w4"), concat!($p, "_vt3w8")],
            [concat!($p, "_vt4w4"), concat!($p, "_vt4w8")],
        ]
    };
}

/// Largest row count of the exact verify entry points.
const VERIFY_EXACT_MAX_ROWS: usize = 128;
/// First row count served by the wide K32 kernels (the VT kernels end at 63).
const VERIFY_WIDE_MIN_ROWS: usize = 64;
/// Rows per supported chunk without the wide route.
const VERIFY_NARROW_CHUNK_ROWS: usize = 63;
/// Module the PeaceMaker wide bundle is loaded under.
const VERIFY_PM_MODULE: &str = "mq4_verify_pm_gfx1201";
/// Widest kernarg list of any entry point (QKVZA: 9 pointers + 6 ints).
const VERIFY_MAX_KERNARGS: usize = 16;

/// `[bt == 8][wide]` hipcc K32 symbols for a projection prefix.
macro_rules! k32_symbols {
    ($p:literal) => {
        [
            [concat!($p, "_vt4w4_k32"), concat!($p, "_vt4w8_k32")],
            [concat!($p, "_vt8w4_k32"), concat!($p, "_vt8w8_k32")],
        ]
    };
}

/// `[bt == 8][wide]` PeaceMaker wide symbols for an op name.
macro_rules! pm_symbols {
    ($p:literal) => {
        [
            [concat!($p, "_bt4w4"), concat!($p, "_bt4w8")],
            [concat!($p, "_bt8w4"), concat!($p, "_bt8w8")],
        ]
    };
}

/// Everything an exact verify op needs to name its three kernel tiers.
struct VerifyOpInfo {
    one_tile_module: &'static str,
    one_tile_source: &'static str,
    one_tile_symbol: &'static str,
    vt_module: &'static str,
    vt_source: &'static str,
    vt: [[&'static str; 2]; 3],
    k32: [[&'static str; 2]; 2],
    pm: [[&'static str; 2]; 2],
}

const VERIFY_QKVZA: VerifyOpInfo = VerifyOpInfo {
    one_tile_module: "gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2",
    one_tile_source: kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_SRC,
    one_tile_symbol: "gemm_qkvza_mq4g256v2_wmma_gfx12",
    vt_module: "gemm_qkvza_mq4g256v2_wmma_gfx12_vt",
    vt_source: kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC,
    vt: vt_symbols!("gemm_qkvza_mq4g256v2_wmma_gfx12"),
    k32: k32_symbols!("gemm_qkvza_mq4g256v2_wmma_gfx12"),
    pm: pm_symbols!("mq4_verify_qkvza_pm_gfx1201"),
};

const VERIFY_QKV: VerifyOpInfo = VerifyOpInfo {
    one_tile_module: "gemm_qkv_hfq4g256_wmma_gfx12_mq4v2",
    one_tile_source: kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_SRC,
    one_tile_symbol: "gemm_qkv_mq4g256v2_wmma_gfx12",
    vt_module: "gemm_qkv_mq4g256v2_wmma_gfx12_vt",
    vt_source: kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC,
    vt: vt_symbols!("gemm_qkv_mq4g256v2_wmma_gfx12"),
    k32: k32_symbols!("gemm_qkv_mq4g256v2_wmma_gfx12"),
    pm: pm_symbols!("mq4_verify_qkv_pm_gfx1201"),
};

const VERIFY_GATE_UP: VerifyOpInfo = VerifyOpInfo {
    one_tile_module: "gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2",
    one_tile_source: kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_SRC,
    one_tile_symbol: "gemm_gate_up_mq4g256v2_wmma_gfx12",
    vt_module: "gemm_gate_up_mq4g256v2_wmma_gfx12_vt",
    vt_source: kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC,
    vt: vt_symbols!("gemm_gate_up_mq4g256v2_wmma_gfx12"),
    k32: k32_symbols!("gemm_gate_up_mq4g256v2_wmma_gfx12"),
    pm: pm_symbols!("mq4_verify_gate_up_pm_gfx1201"),
};

const VERIFY_RESIDUAL: VerifyOpInfo = VerifyOpInfo {
    one_tile_module: "gemm_hfq4g256_residual_wmma_gfx12_mq4v2",
    one_tile_source: kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_SRC,
    one_tile_symbol: "gemm_mq4g256v2_residual_wmma_gfx12",
    vt_module: "gemm_mq4g256v2_residual_wmma_gfx12_vt",
    vt_source: kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC,
    vt: vt_symbols!("gemm_mq4g256v2_residual_wmma_gfx12"),
    k32: k32_symbols!("gemm_mq4g256v2_residual_wmma_gfx12"),
    pm: pm_symbols!("mq4_verify_residual_pm_gfx1201"),
};

/// Where one exact verify launch's code object comes from.
#[derive(Clone, Copy)]
enum VerifyImage {
    /// A hipcc module (`ensure_kernel`).
    Hip {
        module: &'static str,
        source: &'static str,
    },
    /// The embedded PeaceMaker wide bundle (`ensure_embedded_kernel`).
    Pm,
}

/// One resolved exact verify launch; `Copy`, no allocation.
#[derive(Clone, Copy)]
struct VerifyPlan {
    func_name: &'static str,
    image: VerifyImage,
    grid: [u32; 3],
    block: [u32; 3],
}

/// Batch tiles and waves of a wide launch: BT4 holds exactly 64 rows, BT8 up
/// to 128; the w4/w8 choice keeps the VT row-tile threshold.
fn verify_wide_geometry(n: usize, total_m: usize) -> (usize, usize) {
    let bt = if n <= 64 { 4 } else { 8 };
    let waves = if (total_m + 15) / 16 >= VT_WIDE_MIN_ROW_TILES {
        8
    } else {
        4
    };
    (bt, waves)
}

impl Gpu {
    /// Whether the batch-tiled verify route applies to a dense MQ4V2 gfx12
    /// GEMM of `batch_size` rows with reduction length `k`.
    pub(crate) fn gfx12_verify_bt_active(&self, batch_size: usize, k: usize) -> bool {
        self.flags.wmma_batch_tiles
            && self.arch == "gfx1201"
            && (17..64).contains(&batch_size)
            && k % 256 == 0
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
    }

    /// Shared launch: `ptrs` then `ints` are the kernel arguments in ABI order
    /// (the `_vt` kernels keep the one-tile kernels' ABI). Grid x covers
    /// `total_m` weight rows at `16 * waves` per block; grid y is the batch
    /// panel in 16 * BT-row blocks.
    #[allow(clippy::too_many_arguments)]
    fn launch_gemm_vt(
        &mut self,
        module: &'static str,
        source: &'static str,
        symbols: [[&'static str; 2]; 3],
        total_m: usize,
        batch_size: usize,
        mut ptrs: Vec<*mut c_void>,
        mut ints: Vec<i32>,
        bytes: usize,
    ) -> HipResult<()> {
        let bt = (batch_size + 15) / 16;
        debug_assert!((2..=4).contains(&bt));
        let wide = (total_m + 15) / 16 >= VT_WIDE_MIN_ROW_TILES;
        let waves = if wide { 8 } else { 4 };
        let func_name = symbols[bt - 2][wide as usize];
        self.ensure_kernel(module, source, func_name)?;
        let mut params: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| p as *mut _ as *mut c_void)
            .chain(ints.iter_mut().map(|v| v as *mut _ as *mut c_void))
            .collect();
        let grid_x = (total_m + 16 * waves - 1) / (16 * waves);
        let grid_y = (batch_size + 16 * bt - 1) / (16 * bt);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", func_name, bytes);
        let result = self.launch_maybe_blob(
            func_name,
            [grid_x as u32, grid_y as u32, 1],
            [(32 * waves) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in &ptrs {
                    b.push_ptr(*p);
                }
                for v in &ints {
                    b.push_i32(*v);
                }
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// `gemm_qkvza_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_qkvza_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_qkvza_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKVZA_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_qkvza_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_qkv.buf.as_ptr(),
                a_z.buf.as_ptr(),
                a_beta.buf.as_ptr(),
                a_alpha.buf.as_ptr(),
                x_f16,
                y_qkv.buf.as_ptr(),
                y_z.buf.as_ptr(),
                y_beta.buf.as_ptr(),
                y_alpha.buf.as_ptr(),
            ],
            vec![
                qkv_m as i32,
                z_m as i32,
                beta_m as i32,
                alpha_m as i32,
                k as i32,
                batch_size as i32,
            ],
            bytes,
        )
    }

    /// `gemm_qkv_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_qkv_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = q_m + k_m + v_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_qkv_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_QKV_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_qkv_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_q.buf.as_ptr(),
                a_k.buf.as_ptr(),
                a_v.buf.as_ptr(),
                x_f16,
                y_q.buf.as_ptr(),
                y_k.buf.as_ptr(),
                y_v.buf.as_ptr(),
            ],
            vec![
                q_m as i32,
                k_m as i32,
                v_m as i32,
                k as i32,
                batch_size as i32,
            ],
            bytes,
        )
    }

    /// `gemm_gate_up_mq4g256v2_wmma_gfx12` for 16 < N < 64 (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_gate_up_mq4g256v2_wmma_gfx12_vt(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let total_m = gate_m + up_m;
        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + batch_size * k * 2
            + batch_size * total_m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_gate_up_mq4g256v2_wmma_gfx12_vt",
            kernels::GEMM_GATE_UP_MQ4G256V2_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_gate_up_mq4g256v2_wmma_gfx12"),
            total_m,
            batch_size,
            vec![
                a_gate.buf.as_ptr(),
                a_up.buf.as_ptr(),
                x_f16,
                y_gate.buf.as_ptr(),
                y_up.buf.as_ptr(),
            ],
            vec![gate_m as i32, up_m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }

    /// `gemm_mq4g256v2_residual_wmma_gfx12` (`Y += W @ X`, also the batched
    /// lm_head) for 16 < N < 64 (see module docs).
    pub(crate) fn gemm_mq4g256v2_residual_wmma_gfx12_vt(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4 * 2;
        self.launch_gemm_vt(
            "gemm_mq4g256v2_residual_wmma_gfx12_vt",
            kernels::GEMM_MQ4G256V2_RESIDUAL_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_mq4g256v2_residual_wmma_gfx12"),
            m,
            batch_size,
            vec![a_raw.buf.as_ptr(), x_f16, y.buf.as_ptr()],
            vec![m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }

    /// Overwrite twin of [`Self::gemm_mq4g256v2_residual_wmma_gfx12_vt`]
    /// (`Y = W @ X`, no pre-zero fill; store is `acc + 0.0f`, bit-identical to
    /// memset + `Y += acc`). Same symbol/geometry selection and ABI; used by
    /// the gfx1201 DFlash draft route (`gemm_mq4g256v2_overwrite_gfx12_dflash`).
    pub(crate) fn gemm_mq4g256v2_overwrite_wmma_gfx12_vt(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let x_f16 = self.ensure_fp16_x(x, batch_size * k)?;
        let bytes =
            crate::profile::gemv_hfq4g256_bytes(m, k) + batch_size * k * 2 + batch_size * m * 4;
        self.launch_gemm_vt(
            "gemm_mq4g256v2_overwrite_wmma_gfx12_vt",
            kernels::GEMM_MQ4G256V2_OVERWRITE_WMMA_GFX12_VT_SRC,
            vt_symbols!("gemm_mq4g256v2_overwrite_wmma_gfx12"),
            m,
            batch_size,
            vec![a_raw.buf.as_ptr(), x_f16, y.buf.as_ptr()],
            vec![m as i32, k as i32, batch_size as i32],
            bytes,
        )
    }

    /// Rows of one aggregated exact verify chunk this GPU admits: 128 on exact
    /// gfx1201 when `HIPFIRE_CB_VERIFY_CHUNK128` (default on for gfx1201) and
    /// `HIPFIRE_WMMA_BATCH_TILES` are both on, otherwise the historical 63.
    /// Admission of a particular target (dense MQ4V2 projections and head
    /// only, singleton chain per [`Self::mq4_verify_singleton_chain`]) stays
    /// with the caller.
    pub fn mq4_verify_chunk_rows(&self) -> usize {
        Self::mq4_verify_chunk_rows_for(&self.arch, &self.flags)
    }

    /// [`Self::mq4_verify_chunk_rows`] of an `arch` / flag set (no device).
    pub fn mq4_verify_chunk_rows_for(arch: &str, flags: &crate::FeatureFlags) -> usize {
        if arch == "gfx1201" && flags.cb_verify_chunk128 && flags.wmma_batch_tiles {
            VERIFY_EXACT_MAX_ROWS
        } else {
            VERIFY_NARROW_CHUNK_ROWS
        }
    }

    /// Does every singleton dense MQ4G256V2 dispatch below 64 rows run the
    /// one-tile F16-input / f32-WMMA chain (or its byte-identical `_vt`
    /// twin) that the exact wide kernels reproduce? Read-only, from the flags
    /// the dispatches themselves consult:
    ///
    /// - `gemm_mq4g256v2_batched_lmhead`'s `wmma_eligible` gate:
    ///   `HIPFIRE_FP16=0` (`fp16_disabled`) or `HIPFIRE_LM_HEAD_WMMA=0`
    ///   (`lm_head_wmma_disabled`) sends the head to the scalar
    ///   `gemm_mq4g256v2_xbatch`.
    /// - `HIPFIRE_HFQ4G256_LDSSTAGE=1` (`hfq4g256_ldsstage_wmma`): gate/up,
    ///   residual (wo/down, the S4 F16 producers) and the WMMA lm_head take
    ///   `_ldsstage` at `N <= LDSSTAGE_MAX_BATCH`, `K % 512 == 0`.
    /// - `HIPFIRE_WMMA_BATCH_TILES=0`: no wide kernels at all.
    ///
    /// The other arms of those dispatches (A8, IU4 W4A4, FP8, the BT/QKV
    /// weight-reuse tiles) switch only at `>= 64` rows, which no singleton
    /// verify reaches.
    pub fn mq4_verify_singleton_chain(&self) -> bool {
        Self::mq4_verify_singleton_chain_for(&self.arch, &self.flags)
    }

    /// [`Self::mq4_verify_singleton_chain`] of an `arch` / flag set (no device).
    pub fn mq4_verify_singleton_chain_for(arch: &str, flags: &crate::FeatureFlags) -> bool {
        arch == "gfx1201"
            && flags.wmma_batch_tiles
            && !flags.fp16_disabled
            && !flags.lm_head_wmma_disabled
            && !flags.hfq4g256_ldsstage_wmma
    }

    /// Whether the wide verify kernels run from the PeaceMaker bundle
    /// (`HIPFIRE_CB_VERIFY_PM`, default on) rather than the hipcc twins.
    pub fn mq4_verify_pm_selected(&self) -> bool {
        self.flags.cb_verify_pm
    }

    /// Resolve the launch for one exact verify op: fails closed on arch,
    /// reduction length and row-count domain before anything is loaded or
    /// launched. See the module docs for the tiers.
    fn verify_exact_plan(
        &self,
        info: &'static VerifyOpInfo,
        what: &str,
        total_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<VerifyPlan> {
        if self.arch != "gfx1201" {
            return Err(hip_bridge::HipError::new(
                1,
                &format!(
                    "{what}: exact MQ4V2 verify GEMMs exist only on gfx1201, not {}",
                    self.arch
                ),
            ));
        }
        if k == 0 || k % 256 != 0 || total_m == 0 {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("{what}: K must be a positive multiple of 256 and M non-zero (M={total_m}, K={k})"),
            ));
        }
        if n == 0 || n > VERIFY_EXACT_MAX_ROWS {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("{what}: N must be 1..={VERIFY_EXACT_MAX_ROWS}, got {n}"),
            ));
        }
        let vt = self.flags.wmma_batch_tiles;
        if n <= 16 || (n < VERIFY_WIDE_MIN_ROWS && !vt) {
            return Ok(VerifyPlan {
                func_name: info.one_tile_symbol,
                image: VerifyImage::Hip {
                    module: info.one_tile_module,
                    source: info.one_tile_source,
                },
                grid: [((total_m + 15) / 16) as u32, ((n + 15) / 16) as u32, 1],
                block: [32, 1, 1],
            });
        }
        if n < VERIFY_WIDE_MIN_ROWS {
            let bt = (n + 15) / 16;
            let wide = (total_m + 15) / 16 >= VT_WIDE_MIN_ROW_TILES;
            let waves = if wide { 8 } else { 4 };
            return Ok(VerifyPlan {
                func_name: info.vt[bt - 2][wide as usize],
                image: VerifyImage::Hip {
                    module: info.vt_module,
                    source: info.vt_source,
                },
                grid: [
                    ((total_m + 16 * waves - 1) / (16 * waves)) as u32,
                    ((n + 16 * bt - 1) / (16 * bt)) as u32,
                    1,
                ],
                block: [(32 * waves) as u32, 1, 1],
            });
        }
        if !vt {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("{what}: N={n} needs the wide exact kernels, disabled by HIPFIRE_WMMA_BATCH_TILES=0"),
            ));
        }
        let (bt, waves) = verify_wide_geometry(n, total_m);
        let slot = (bt == 8) as usize;
        let wide = (waves == 8) as usize;
        let (func_name, image) = if self.flags.cb_verify_pm {
            (info.pm[slot][wide], VerifyImage::Pm)
        } else {
            (
                info.k32[slot][wide],
                VerifyImage::Hip {
                    module: info.vt_module,
                    source: info.vt_source,
                },
            )
        };
        Ok(VerifyPlan {
            func_name,
            image,
            grid: [
                ((total_m + 16 * waves - 1) / (16 * waves)) as u32,
                ((n + 16 * bt - 1) / (16 * bt)) as u32,
                1,
            ],
            block: [(32 * waves) as u32, 1, 1],
        })
    }

    /// Load the plan's code object. A missing PM bundle or symbol is an error.
    fn load_verify_plan(&mut self, plan: &VerifyPlan) -> HipResult<()> {
        match plan.image {
            VerifyImage::Hip { module, source } => {
                self.ensure_kernel(module, source, plan.func_name)
            }
            VerifyImage::Pm => self.ensure_embedded_kernel(
                VERIFY_PM_MODULE,
                kernels::MQ4_VERIFY_PM_GFX1201,
                plan.func_name,
            ),
        }
    }

    /// Launch a resolved, loaded plan: `ptrs` then `ints` are the kernel
    /// arguments in ABI order (the one-tile, VT, K32 and PM kernels share it).
    fn launch_verify_plan<const P: usize, const I: usize>(
        &mut self,
        plan: VerifyPlan,
        mut ptrs: [*mut c_void; P],
        mut ints: [i32; I],
        bytes: usize,
    ) -> HipResult<()> {
        debug_assert!(P + I <= VERIFY_MAX_KERNARGS);
        let mut params = [std::ptr::null_mut::<c_void>(); VERIFY_MAX_KERNARGS];
        for (slot, p) in params.iter_mut().zip(ptrs.iter_mut()) {
            *slot = p as *mut _ as *mut c_void;
        }
        for (slot, v) in params[P..].iter_mut().zip(ints.iter_mut()) {
            *slot = v as *mut _ as *mut c_void;
        }
        let timer = crate::profile::begin_timer(&self.hip, "gemm", plan.func_name, bytes);
        let result = self.launch_maybe_blob(
            plan.func_name,
            plan.grid,
            plan.block,
            0,
            &mut params[..P + I],
            || {
                let mut b = hip_bridge::KernargBlob::new();
                for p in &ptrs {
                    b.push_ptr(*p);
                }
                for v in &ints {
                    b.push_i32(*v);
                }
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Exact verify QKVZA projection for `1 <= n <= 128` rows (F32 overwrite
    /// into four output matrices). Byte-identical per output to the
    /// singleton `gemm_qkvza_mq4g256v2_wmma_gfx12`; see the module docs.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_mq4g256v2_verify_exact(
        &mut self,
        a_qkv: &GpuTensor,
        a_z: &GpuTensor,
        a_beta: &GpuTensor,
        a_alpha: &GpuTensor,
        x: &GpuTensor,
        y_qkv: &GpuTensor,
        y_z: &GpuTensor,
        y_beta: &GpuTensor,
        y_alpha: &GpuTensor,
        qkv_m: usize,
        z_m: usize,
        beta_m: usize,
        alpha_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if self.fp8_verify_scope {
            return self.gemm_qkvza_mq4g256v2_verify_fp8(
                a_qkv, a_z, a_beta, a_alpha, x, y_qkv, y_z, y_beta, y_alpha, qkv_m, z_m, beta_m,
                alpha_m, k, n,
            );
        }
        let total_m = qkv_m + z_m + beta_m + alpha_m;
        self.iu4_verify_note_f16();
        let plan = self.verify_exact_plan(
            &VERIFY_QKVZA,
            "gemm_qkvza_mq4g256v2_verify_exact",
            total_m,
            k,
            n,
        )?;
        self.bind_thread()?;
        self.load_verify_plan(&plan)?;
        let x_f16 = self.ensure_fp16_x(x, n * k)?;
        let bytes = crate::profile::gemv_hfq4g256_bytes(qkv_m, k)
            + crate::profile::gemv_hfq4g256_bytes(z_m, k)
            + crate::profile::gemv_hfq4g256_bytes(beta_m, k)
            + crate::profile::gemv_hfq4g256_bytes(alpha_m, k)
            + n * k * 2
            + n * total_m * 4 * 2;
        self.launch_verify_plan(
            plan,
            [
                a_qkv.buf.as_ptr(),
                a_z.buf.as_ptr(),
                a_beta.buf.as_ptr(),
                a_alpha.buf.as_ptr(),
                x_f16,
                y_qkv.buf.as_ptr(),
                y_z.buf.as_ptr(),
                y_beta.buf.as_ptr(),
                y_alpha.buf.as_ptr(),
            ],
            [
                qkv_m as i32,
                z_m as i32,
                beta_m as i32,
                alpha_m as i32,
                k as i32,
                n as i32,
            ],
            bytes,
        )
    }

    /// Exact verify QKV projection for `1 <= n <= 128` rows (see
    /// [`Self::gemm_qkvza_mq4g256v2_verify_exact`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_mq4g256v2_verify_exact(
        &mut self,
        a_q: &GpuTensor,
        a_k: &GpuTensor,
        a_v: &GpuTensor,
        x: &GpuTensor,
        y_q: &GpuTensor,
        y_k: &GpuTensor,
        y_v: &GpuTensor,
        q_m: usize,
        k_m: usize,
        v_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if self.fp8_verify_scope {
            return self.gemm_qkv_mq4g256v2_verify_fp8(
                a_q, a_k, a_v, x, y_q, y_k, y_v, q_m, k_m, v_m, k, n,
            );
        }
        let total_m = q_m + k_m + v_m;
        self.iu4_verify_note_f16();
        let plan = self.verify_exact_plan(
            &VERIFY_QKV,
            "gemm_qkv_mq4g256v2_verify_exact",
            total_m,
            k,
            n,
        )?;
        self.bind_thread()?;
        self.load_verify_plan(&plan)?;
        let x_f16 = self.ensure_fp16_x(x, n * k)?;
        let bytes = crate::profile::gemv_hfq4g256_bytes(q_m, k)
            + crate::profile::gemv_hfq4g256_bytes(k_m, k)
            + crate::profile::gemv_hfq4g256_bytes(v_m, k)
            + n * k * 2
            + n * total_m * 4 * 2;
        self.launch_verify_plan(
            plan,
            [
                a_q.buf.as_ptr(),
                a_k.buf.as_ptr(),
                a_v.buf.as_ptr(),
                x_f16,
                y_q.buf.as_ptr(),
                y_k.buf.as_ptr(),
                y_v.buf.as_ptr(),
            ],
            [q_m as i32, k_m as i32, v_m as i32, k as i32, n as i32],
            bytes,
        )
    }

    /// Exact verify gate/up projection for `1 <= n <= 128` rows (see
    /// [`Self::gemm_qkvza_mq4g256v2_verify_exact`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_mq4g256v2_verify_exact(
        &mut self,
        a_gate: &GpuTensor,
        a_up: &GpuTensor,
        x: &GpuTensor,
        y_gate: &GpuTensor,
        y_up: &GpuTensor,
        gate_m: usize,
        up_m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if self.fp8_verify_scope {
            return self.gemm_gate_up_mq4g256v2_verify_fp8(
                a_gate, a_up, x, y_gate, y_up, gate_m, up_m, k, n,
            );
        }
        let total_m = gate_m + up_m;
        self.iu4_verify_note_f16();
        let plan = self.verify_exact_plan(
            &VERIFY_GATE_UP,
            "gemm_gate_up_mq4g256v2_verify_exact",
            total_m,
            k,
            n,
        )?;
        self.bind_thread()?;
        self.load_verify_plan(&plan)?;
        let x_f16 = self.ensure_fp16_x(x, n * k)?;
        let bytes = crate::profile::gemv_hfq4g256_bytes(gate_m, k)
            + crate::profile::gemv_hfq4g256_bytes(up_m, k)
            + n * k * 2
            + n * total_m * 4 * 2;
        self.launch_verify_plan(
            plan,
            [
                a_gate.buf.as_ptr(),
                a_up.buf.as_ptr(),
                x_f16,
                y_gate.buf.as_ptr(),
                y_up.buf.as_ptr(),
            ],
            [gate_m as i32, up_m as i32, k as i32, n as i32],
            bytes,
        )
    }

    /// Exact verify residual projection (`Y += W @ X`, one f32 epilogue add)
    /// for `1 <= n <= 128` rows (see [`Self::gemm_qkvza_mq4g256v2_verify_exact`]).
    pub fn gemm_mq4g256v2_residual_verify_exact(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if self.fp8_verify_scope {
            return self.gemm_mq4g256v2_residual_verify_fp8(a, x, y, m, k, n);
        }
        self.iu4_verify_note_f16();
        let plan = self.verify_exact_plan(
            &VERIFY_RESIDUAL,
            "gemm_mq4g256v2_residual_verify_exact",
            m,
            k,
            n,
        )?;
        self.bind_thread()?;
        self.load_verify_plan(&plan)?;
        let x_f16 = self.ensure_fp16_x(x, n * k)?;
        let bytes = crate::profile::gemv_hfq4g256_bytes(m, k) + n * k * 2 + n * m * 4 * 2;
        self.launch_verify_plan(
            plan,
            [a.buf.as_ptr(), x_f16, y.buf.as_ptr()],
            [m as i32, k as i32, n as i32],
            bytes,
        )
    }

    /// Exact verify lm_head: +0 byte memset of `y`, then the exact residual
    /// (zero + acc, as the generic batched lm_head). Unlike the generic
    /// `gemm_mq4g256v2_batched_lmhead` it never routes `n == 1` to the GEMV:
    /// every row count runs the WMMA arithmetic of the 16-row tile reference.
    pub fn gemm_mq4g256v2_lmhead_verify_exact(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        if self.fp8_verify_scope {
            return self.gemm_mq4g256v2_lmhead_verify_fp8(a, x, y, m, k, n);
        }
        // Resolve and load before the memset so a refused or unloadable
        // launch leaves `y` untouched.
        let plan = self.verify_exact_plan(
            &VERIFY_RESIDUAL,
            "gemm_mq4g256v2_lmhead_verify_exact",
            m,
            k,
            n,
        )?;
        self.bind_thread()?;
        self.load_verify_plan(&plan)?;
        // The fp16 activation cache is keyed by source pointer: stomp it so
        // the residual below converts this call's X.
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
        match self.active_stream.as_ref() {
            Some(stream) => self.hip.memset_async(&y.buf, 0, n * m * 4, stream)?,
            None => self.hip.memset(&y.buf, 0, n * m * 4)?,
        }
        self.gemm_mq4g256v2_residual_verify_exact(a, x, y, m, k, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_geometry_covers_64_to_128_rows() {
        // BT4 holds exactly 64 rows, BT8 the rest up to 128.
        for n in 64..=128 {
            let (bt, _) = verify_wide_geometry(n, 5120);
            assert_eq!(bt, if n == 64 { 4 } else { 8 }, "n={n}");
            assert!(16 * bt >= n, "n={n}");
        }
        // The VT w4/w8 threshold is kept: 1024 sixteen-row tiles and up is w8.
        assert_eq!(verify_wide_geometry(128, 16 * 1023).1, 4);
        assert_eq!(verify_wide_geometry(128, 16 * 1024).1, 8);
    }

    #[test]
    fn symbol_tables_follow_the_frozen_names() {
        assert_eq!(
            VERIFY_QKVZA.k32,
            [
                [
                    "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w4_k32",
                    "gemm_qkvza_mq4g256v2_wmma_gfx12_vt4w8_k32"
                ],
                [
                    "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w4_k32",
                    "gemm_qkvza_mq4g256v2_wmma_gfx12_vt8w8_k32"
                ],
            ]
        );
        assert_eq!(
            VERIFY_RESIDUAL.pm,
            [
                [
                    "mq4_verify_residual_pm_gfx1201_bt4w4",
                    "mq4_verify_residual_pm_gfx1201_bt4w8"
                ],
                [
                    "mq4_verify_residual_pm_gfx1201_bt8w4",
                    "mq4_verify_residual_pm_gfx1201_bt8w8"
                ],
            ]
        );
    }
}
