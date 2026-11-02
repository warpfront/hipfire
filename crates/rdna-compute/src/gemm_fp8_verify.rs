// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! PeaceMaker FP8 verify GEMM family for the exact-gfx1201 MQ4G256V2 target
//! verify (`HIPFIRE_CB_VERIFY_FP8`, developer, default off).
//!
//! # What it is
//!
//! The 27B Qwen3.5-hybrid target verify (DFlash and MTP; singleton = 16 rows,
//! continuous batching = up to 128 rows) normally multiplies F16 activations
//! by MQ4G256V2 weights on the F16 WMMA. This family multiplies **E4M3
//! activations** by the **exactly dequantized E4M3 weights** on
//! `v_wmma_f32_16x16x16_fp8_fp8`. The kernels are the embedded bundle
//! [`kernels::MQ4_FP8_VERIFY_PM_GFX1201`], module
//! `mq4_fp8_verify_pm_gfx1201`, 35 entries
//! `mq4_fp8_verify_{qkvza,qkv,gate_up,residual,set}_pm_gfx1201_bt{BT}w{W}`
//! (see [`fp8_verify_symbol`]); the arithmetic contract is the kernel spec
//! (`KERNEL-SPEC.md`), this module owns only the host half: activation pack,
//! tile plan, scope, routing and fail-closed validation.
//!
//! # Why batched == singleton, byte for byte
//!
//! 1. *The GEMM is row-count independent by construction.* Every output
//!    `(token, weight row)` is the same instruction sequence on the same
//!    operands in every entry: per 128-K half, eight chained
//!    16x16x16 FP8 WMMAs `P = WMMA(A_t, B_t, P)` over fixed K fragments, then
//!    `T = fma(sc_half, P, T)` in ascending half order, then the epilogue
//!    (`Y = a*T`, residual `Y = fma(a, T, Y)`). The token's position in the
//!    tile, the batch tile `BT`, the wave count `W`, the grid and the row
//!    count `N` select only *which lane/register* computes an output, never
//!    the operands or their order, and each WMMA output element depends only
//!    on its own column/row inputs. Rows past `N` read zero and are never
//!    stored.
//! 2. *The activation pack is row independent.* [`Gpu::fp8_verify_pack_x`]
//!    runs `pack_f32_to_fp8_verify_rows_gfx1201`, grid `[N]`: block `n` reads
//!    only row `n`, derives that row's scale `a_n` from the row's absmax (an
//!    fmax reduction, order independent) and writes only row `n` of the E4M3
//!    bytes and `scales[n]`. Bytes and scales are bit-identical to
//!    `pack_f32_to_fp8_mq4v2_gfx12` scale_mode 1 (same
//!    `mq4v2_row_scale_from_absmax`, same `HIPFIRE_FP8_ROW_SCALE_SHIFT`, same
//!    `cvt_pk_fp8_f32` round-to-nearest-even of `v * (1/a_n)`); no half sums
//!    are produced because the symmetric kernels need none.
//!
//! So a row packed and multiplied alone (N = 1), in a 16-row singleton block
//! or inside a 128-row continuous batch yields the same 32-bit result. The
//! tests in this file pin the host-visible parts (tile plan, symbols); the
//! device arithmetic is certified by the kernel bundle's own tests and the
//! GPU oracles.
//!
//! # Tiers (production tile plan, [`fp8_verify_tile`])
//!
//! Narrow projections (fewer than 512 sixteen-row tiles, the 27B residual
//! shapes) at `n <= 16` -> `bt1w1` (direct template, one wave, no LDS).
//! Otherwise `n <= 32` -> `bt2`, `n <= 64` -> `bt4`, `n <= 128` -> `bt8`
//! (LDS-staged templates; at least `bt4` from 4096 row tiles, the lm_head),
//! with `w = 8` waves from 512 row tiles else `w = 4`. Each choice is the
//! fastest shipped tile of its tier on the 27B shapes (`--tile-sweep`).
//! Launch: block `[32 * W, 1, 1]`, grid
//! `[ceil(total_m / (16 W)), ceil(N / (16 BT)), 1]`, no dynamic LDS.
//! Because of (1) the tile choice is a performance decision only.
//! [`Gpu::gemm_mq4g256v2_verify_fp8_prepacked`] runs an explicit tile for the
//! oracles.
//!
//! # Scope semantics
//!
//! The arch crate wraps every target-verify forward (and its verify lm_head)
//! in [`Gpu::fp8_verify_enter`] / [`Gpu::fp8_verify_exit`], passing whether
//! the forward qualifies ([`Fp8VerifyTarget`]). `enter` returns the previous
//! scope and opens it only with `HIPFIRE_CB_VERIFY_FP8=1`; with the flag
//! unset/`0` the scope stays closed and **nothing in the crate changes**.
//! Inside the scope:
//!
//! - the five `gemm_*_mq4g256v2_*_verify_exact` entries and the generic gfx12
//!   MQ4V2 dense entries (`gemm_{qkvza,qkv,gate_up}_hfq4g256_wmma_gfx12_mq4v2`,
//!   `gemm_hfq4g256_residual_wmma_gfx12_mq4v2`, the gfx12 arm of
//!   `gemm_mq4g256v2_batched_lmhead`) route to the `_verify_fp8` twins as
//!   their first statement, before any other arm (A8/IU4/FP8/BT/PM);
//! - every other dense MQ4G256V2 entry that a verify forward could reach with
//!   F16 activations (the F16-producer entries of `mq_f16_producers` /
//!   `mq_f16_residual_producers`, the scalar and MMQ fallbacks) returns an
//!   error naming the scope instead of silently using F16 arithmetic.
//!
//! The receipt ([`Gpu::fp8_verify_receipt`]) counts closed outermost scopes
//! and, inside scopes, FP8 GEMM calls and activation rows, so a run can prove
//! that the route was actually taken.
//!
//! # Fail closed
//!
//! Every entry validates before any load or launch: exact `gfx1201`,
//! `Gpu::mq4v2_symmetric` (the kernels assume the symmetric grid and read no
//! zero points), `K > 0 && K % 256 == 0`, `1 <= N <= 128`, every projection
//! height and the total non-zero, `N * K < 2^31`. The activation must be F32,
//! 16-byte aligned, at least `N*K*4` bytes, `K <= 32768`. A missing bundle or
//! symbol is an error, never a fallback to another family. With
//! `HIPFIRE_CB_VERIFY_FP8=1`, `fp8_verify_enter` refuses (Err) when the arch,
//! the weights or the forward do not qualify.
//!
//! # Capture safety
//!
//! The pack scratch (E4M3 bytes `n*k`, scales `n*4`) lives in two dedicated
//! slots grown only by [`Gpu::fp8_verify_pack_x`] under the scratch-growth
//! protocol (`scratch_will_grow` -> `invalidate_for_scratch_growth` before
//! delegating), the pack always runs (no source-pointer cache, so replayed
//! graphs and stable-address X buffers with changing contents are correct),
//! and both the pack and the GEMM launch through `launch_maybe_blob`, so
//! hipGraph capture and Redline replay recording see every launch. The launches
//! use fixed-size argument arrays: no per-launch allocation beyond the
//! existing blob path.

use crate::{kernels, DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

/// Module the FP8 verify bundle is loaded under.
const FP8_VERIFY_MODULE: &str = "mq4_fp8_verify_pm_gfx1201";
/// Largest row count of the family.
const FP8_VERIFY_MAX_ROWS: usize = 128;
/// Largest reduction length the activation pack holds in registers.
const FP8_VERIFY_PACK_MAX_K: usize = 32768;
/// Sixteen-row weight tiles from which a launch takes 8 waves (and the
/// `n <= 16` tier leaves the direct template for `bt2w8`).
const FP8_VERIFY_WIDE_MIN_ROW_TILES: usize = 512;
/// Sixteen-row weight tiles from which the batch tile is at least `bt4`.
const FP8_VERIFY_HEAD_MIN_ROW_TILES: usize = 4096;
/// Most weight matrices of one entry (QKVZA).
const FP8_VERIFY_MAX_MATS: usize = 4;
/// Pointer kernargs of the widest entry: `A*4, X, S, Y*4`.
const FP8_VERIFY_MAX_PTRS: usize = 2 * FP8_VERIFY_MAX_MATS + 2;
/// Int kernargs of the widest entry: `m*4, K, N`.
const FP8_VERIFY_MAX_INTS: usize = FP8_VERIFY_MAX_MATS + 2;

/// Which projection family an FP8 verify launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8VerifyOp {
    /// Four matrices (qkv, z, beta, alpha), overwrite.
    Qkvza,
    /// Three matrices (q, k, v), overwrite.
    Qkv,
    /// Two matrices (gate, up), overwrite.
    GateUp,
    /// One matrix, `Y += a*T` (fma).
    Residual,
    /// One matrix, overwrite (verify lm_head; no memset needed).
    Set,
}

impl Fp8VerifyOp {
    /// Weight matrices (and outputs) of the op.
    pub const fn mats(self) -> usize {
        match self {
            Self::Qkvza => 4,
            Self::Qkv => 3,
            Self::GateUp => 2,
            Self::Residual | Self::Set => 1,
        }
    }

    const fn row(self) -> usize {
        match self {
            Self::Qkvza => 0,
            Self::Qkv => 1,
            Self::GateUp => 2,
            Self::Residual => 3,
            Self::Set => 4,
        }
    }
}

/// Batch tiles (16 tokens each) per wave group and waves per workgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8VerifyTile {
    pub bt: u8,
    pub waves: u8,
}

/// The seven shipped tiles in bundle order (index = symbol column).
const FP8_VERIFY_TILES: [(u8, u8); 7] = [(1, 1), (2, 4), (2, 8), (4, 4), (4, 8), (8, 4), (8, 8)];

fn tile_index(tile: Fp8VerifyTile) -> Option<usize> {
    FP8_VERIFY_TILES
        .iter()
        .position(|&(bt, w)| bt == tile.bt && w == tile.waves)
}

macro_rules! fp8_symbols {
    ($op:literal) => {
        [
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt1w1"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt2w4"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt2w8"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt4w4"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt4w8"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt8w4"),
            concat!("mq4_fp8_verify_", $op, "_pm_gfx1201_bt8w8"),
        ]
    };
}

/// `[op row][tile index]`, op rows in [`Fp8VerifyOp::row`] order.
const FP8_VERIFY_SYMBOLS: [[&str; 7]; 5] = [
    fp8_symbols!("qkvza"),
    fp8_symbols!("qkv"),
    fp8_symbols!("gate_up"),
    fp8_symbols!("residual"),
    fp8_symbols!("set"),
];

/// Production plan (measured per shape on gfx1201, `pm_fp8_verify_oracle
/// --tile-sweep`; every tile computes the same bytes, so this is speed only).
/// With `r = ceil(total_m / 16)` row tiles and `wide = r >= 512`:
/// `n <= 16` and not wide -> `(1, 1)` (the direct template streams the
/// narrow residual shapes best); otherwise `bt = 2` for `n <= 32`, `4` for
/// `n <= 64`, `8` above, raised to at least `4` when `r >= 4096` (the
/// lm_head), and `w = 8` when wide else `4`. `n` must be `1..=128` (entries
/// validate before planning).
pub fn fp8_verify_tile(n: usize, total_m: usize) -> Fp8VerifyTile {
    let row_tiles = total_m.div_ceil(16);
    let wide = row_tiles >= FP8_VERIFY_WIDE_MIN_ROW_TILES;
    if n <= 16 && !wide {
        return Fp8VerifyTile { bt: 1, waves: 1 };
    }
    let bt = if n <= 32 {
        2
    } else if n <= 64 {
        4
    } else {
        8
    };
    let bt = if row_tiles >= FP8_VERIFY_HEAD_MIN_ROW_TILES {
        bt.max(4)
    } else {
        bt
    };
    let waves = if wide { 8 } else { 4 };
    Fp8VerifyTile { bt, waves }
}

/// Bundle symbol of `op` at `tile`. Panics on a tile the bundle does not
/// ship (programming error; the launch entries validate tiles first).
pub fn fp8_verify_symbol(op: Fp8VerifyOp, tile: Fp8VerifyTile) -> &'static str {
    match tile_index(tile) {
        Some(i) => FP8_VERIFY_SYMBOLS[op.row()][i],
        None => panic!(
            "fp8_verify_symbol: no bt{}w{} tile in the FP8 verify bundle",
            tile.bt, tile.waves
        ),
    }
}

/// Packed activations; pointers valid until the next pack on this `Gpu`.
#[derive(Clone, Copy)]
pub struct Fp8VerifyX {
    /// E4M3 bytes `[n, k]` row-major.
    pub x8: *mut c_void,
    /// Per-row activation scales `a_n`, f32 `[n]`.
    pub scales: *mut c_void,
    pub n: usize,
    pub k: usize,
}

/// Process-wide route receipt: closed outermost scopes, FP8 GEMM launches
/// inside scopes, and the activation rows they covered.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp8VerifyReceipt {
    pub scopes: u64,
    pub calls: u64,
    pub rows: u64,
}

static FP8_VERIFY_SCOPES: AtomicU64 = AtomicU64::new(0);
static FP8_VERIFY_CALLS: AtomicU64 = AtomicU64::new(0);
static FP8_VERIFY_ROWS: AtomicU64 = AtomicU64::new(0);
/// Per-scope counters, drained into the receipt line when a scope closes.
static FP8_VERIFY_SCOPE_CALLS: AtomicU64 = AtomicU64::new(0);
static FP8_VERIFY_SCOPE_ROWS: AtomicU64 = AtomicU64::new(0);
static FP8_VERIFY_ANNOUNCE: std::sync::Once = std::sync::Once::new();

fn fp8_err(msg: &str) -> HipError {
    HipError::new(1, msg)
}

/// Error for an F16-input or non-FP8 MQ4G256V2 GEMM entry reached inside the
/// FP8 verify scope (fail closed: never a silent F16 route).
pub(crate) fn fp8_verify_scope_error(what: &str) -> HipError {
    fp8_err(&format!(
        "{what}: not available inside the FP8 verify scope"
    ))
}

/// Whether a target-verify forward can run inside the FP8 verify scope,
/// decided by the model crate per forward (see [`Gpu::fp8_verify_enter`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8VerifyTarget {
    /// No FP8 route for this forward (target dtype/layout, KV tier, tree
    /// verify, ...): `HIPFIRE_CB_VERIFY_FP8=1` refuses it.
    Ineligible,
    /// A dense MQ4G256V2 chain verify the FP8 family covers.
    Eligible,
}

impl Gpu {
    /// Open the FP8 verify scope for a forward that qualifies as `target`:
    /// returns the previous scope for [`Self::fp8_verify_exit`]. With
    /// `HIPFIRE_CB_VERIFY_FP8` unset/`0` the scope stays closed. With `=1`
    /// it opens, or errs when the device is not exact gfx1201, the weights
    /// are not the symmetric grid or `target` is
    /// [`Fp8VerifyTarget::Ineligible`] (the scope is left unchanged). The
    /// first opened scope prints one stderr line.
    pub fn fp8_verify_enter(&mut self, target: Fp8VerifyTarget) -> HipResult<bool> {
        let want = self.flags.cb_verify_fp8;
        if want {
            if self.arch != "gfx1201" {
                return Err(fp8_err(&format!(
                    "HIPFIRE_CB_VERIFY_FP8: FP8 verify GEMMs exist only on gfx1201, not {}",
                    self.arch
                )));
            }
            if !self.mq4v2_symmetric {
                return Err(fp8_err(
                    "HIPFIRE_CB_VERIFY_FP8: needs symmetric MQ4G256V2 weights (artifact is legacy/asymmetric)",
                ));
            }
            if target == Fp8VerifyTarget::Ineligible {
                return Err(fp8_err(
                    "HIPFIRE_CB_VERIFY_FP8: this verify forward has no FP8 route (needs a dense MQ4G256V2 \
                     target, an FP8/Q8/asym KV tier and chain verify)",
                ));
            }
            FP8_VERIFY_ANNOUNCE
                .call_once(|| eprintln!("[cb-verify-fp8] FP8 verify GEMM route active"));
        }
        let prev = self.fp8_verify_scope;
        self.fp8_verify_scope = want;
        Ok(prev)
    }

    /// Restore the scope returned by [`Self::fp8_verify_enter`]. An outermost
    /// scope that was on is counted as closed; the first 3 closed scopes and
    /// every 1000th print one stderr receipt line with that scope's FP8 GEMM
    /// call and row counts (the counters then reset).
    pub fn fp8_verify_exit(&mut self, prev: bool) {
        if self.fp8_verify_scope && !prev {
            let calls = FP8_VERIFY_SCOPE_CALLS.swap(0, Ordering::Relaxed);
            let rows = FP8_VERIFY_SCOPE_ROWS.swap(0, Ordering::Relaxed);
            let i = FP8_VERIFY_SCOPES.fetch_add(1, Ordering::Relaxed);
            if i < 3 || i % 1000 == 0 {
                eprintln!("[cb-verify-fp8] scope #{i}: {calls} FP8 GEMM calls, {rows} rows");
            }
        }
        self.fp8_verify_scope = prev;
    }

    /// Process totals of closed scopes and in-scope FP8 GEMM calls/rows.
    pub fn fp8_verify_receipt(&self) -> Fp8VerifyReceipt {
        Fp8VerifyReceipt {
            scopes: FP8_VERIFY_SCOPES.load(Ordering::Relaxed),
            calls: FP8_VERIFY_CALLS.load(Ordering::Relaxed),
            rows: FP8_VERIFY_ROWS.load(Ordering::Relaxed),
        }
    }

    /// Fail-closed domain check shared by every entry: runs before any
    /// load or launch.
    fn fp8_verify_validate(&self, what: &str, ms: &[usize], k: usize, n: usize) -> HipResult<()> {
        if self.arch != "gfx1201" {
            return Err(fp8_err(&format!(
                "{what}: FP8 verify GEMMs exist only on gfx1201, not {}",
                self.arch
            )));
        }
        if !self.mq4v2_symmetric {
            return Err(fp8_err(&format!(
                "{what}: FP8 verify GEMMs need symmetric MQ4G256V2 weights"
            )));
        }
        if k == 0 || k % 256 != 0 {
            return Err(fp8_err(&format!(
                "{what}: K must be a positive multiple of 256, got {k}"
            )));
        }
        if n == 0 || n > FP8_VERIFY_MAX_ROWS {
            return Err(fp8_err(&format!(
                "{what}: N must be 1..={FP8_VERIFY_MAX_ROWS}, got {n}"
            )));
        }
        if ms.iter().any(|&m| m == 0) {
            return Err(fp8_err(&format!(
                "{what}: every projection height must be non-zero ({ms:?})"
            )));
        }
        let total_m: usize = ms.iter().sum();
        if total_m == 0 || total_m >= (1usize << 31) {
            return Err(fp8_err(&format!(
                "{what}: total M out of range ({total_m})"
            )));
        }
        match n.checked_mul(k) {
            Some(nk) if nk < (1usize << 31) => Ok(()),
            _ => Err(fp8_err(&format!(
                "{what}: N*K must be below 2^31 (N={n}, K={k})"
            ))),
        }
    }

    /// Pack `x` (F32 `[n, k]`) into the dedicated E4M3 + per-row-scale
    /// scratch: one `pack_f32_to_fp8_verify_rows_gfx1201` launch, always
    /// executed (no source-pointer cache). F32 input only. The returned
    /// pointers stay valid until the next pack on this `Gpu`.
    pub fn fp8_verify_pack_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<Fp8VerifyX> {
        const WHAT: &str = "fp8_verify_pack_x";
        if x.dtype != DType::F32 {
            return Err(fp8_err(&format!(
                "{WHAT}: activations must be F32, got {:?}",
                x.dtype
            )));
        }
        if self.arch != "gfx1201" {
            return Err(fp8_err(&format!("{WHAT}: gfx1201 only, not {}", self.arch)));
        }
        if k == 0 || k % 256 != 0 || k > FP8_VERIFY_PACK_MAX_K {
            return Err(fp8_err(&format!(
                "{WHAT}: K must be a multiple of 256 in 256..={FP8_VERIFY_PACK_MAX_K}, got {k}"
            )));
        }
        if n == 0 || n > FP8_VERIFY_MAX_ROWS {
            return Err(fp8_err(&format!(
                "{WHAT}: N must be 1..={FP8_VERIFY_MAX_ROWS}, got {n}"
            )));
        }
        // n <= 128 and k <= 32768, so n * k cannot overflow or reach 2^31.
        let x_ptr = x.buf.as_ptr();
        if (x_ptr as usize) % 16 != 0 {
            return Err(fp8_err(&format!(
                "{WHAT}: activation pointer must be 16-byte aligned"
            )));
        }
        if x.buf.size() < n * k * 4 {
            return Err(fp8_err(&format!(
                "{WHAT}: activation buffer holds {} bytes, need {}",
                x.buf.size(),
                n * k * 4
            )));
        }
        self.bind_thread()?;
        let row_scale_shift = crate::gemv::fp8_row_scale_shift()?;
        // Scratch growth under a captured graph would free a pointer the
        // graph embeds: invalidate first (no-op unless captured). See
        // `invalidate_for_scratch_growth`.
        if crate::scratch::scratch_will_grow(
            self.scratch.fp8_verify_x8_scratch_bytes,
            self.scratch.fp8_verify_x8_scratch.is_some(),
            n * k,
        ) || crate::scratch::scratch_will_grow(
            self.scratch.fp8_verify_scales_scratch_bytes,
            self.scratch.fp8_verify_scales_scratch.is_some(),
            n * 4,
        ) {
            self.invalidate_for_scratch_growth();
        }
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        let (x8, scales) = self.scratch.pack_fp8_verify_rows(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            x_ptr,
            n,
            k,
            row_scale_shift,
        )?;
        Ok(Fp8VerifyX { x8, scales, n, k })
    }

    /// Load the bundle symbol for `op` at `tile`; a missing bundle or symbol
    /// is an error.
    fn fp8_verify_load(&mut self, op: Fp8VerifyOp, tile: Fp8VerifyTile) -> HipResult<&'static str> {
        let symbol = fp8_verify_symbol(op, tile);
        self.ensure_embedded_kernel(
            FP8_VERIFY_MODULE,
            kernels::MQ4_FP8_VERIFY_PM_GFX1201,
            symbol,
        )?;
        Ok(symbol)
    }

    /// One launch of a loaded, validated entry. `a`/`y`/`m` have `op.mats()`
    /// elements; kernarg order `A.., X, S, Y.., m.., K, N`.
    #[allow(clippy::too_many_arguments)]
    fn fp8_verify_launch(
        &mut self,
        op: Fp8VerifyOp,
        tile: Fp8VerifyTile,
        symbol: &'static str,
        a: &[*mut c_void],
        y: &[*mut c_void],
        m: &[usize],
        xq: &Fp8VerifyX,
    ) -> HipResult<()> {
        let nm = op.mats();
        debug_assert!(a.len() == nm && y.len() == nm && m.len() == nm);
        let (k, n) = (xq.k, xq.n);
        let total_m: usize = m.iter().sum();
        let np = 2 * nm + 2;
        let ni = nm + 2;
        let mut ptrs = [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_PTRS];
        ptrs[..nm].copy_from_slice(a);
        ptrs[nm] = xq.x8;
        ptrs[nm + 1] = xq.scales;
        ptrs[nm + 2..np].copy_from_slice(y);
        let mut ints = [0i32; FP8_VERIFY_MAX_INTS];
        for (slot, &mi) in ints.iter_mut().zip(m) {
            *slot = mi as i32;
        }
        ints[nm] = k as i32;
        ints[nm + 1] = n as i32;
        let mut params =
            [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_PTRS + FP8_VERIFY_MAX_INTS];
        for (slot, p) in params.iter_mut().zip(ptrs[..np].iter_mut()) {
            *slot = p as *mut _ as *mut c_void;
        }
        for (slot, v) in params[np..].iter_mut().zip(ints[..ni].iter_mut()) {
            *slot = v as *mut _ as *mut c_void;
        }
        let waves = tile.waves as usize;
        let bt = tile.bt as usize;
        let grid = [
            ((total_m + 16 * waves - 1) / (16 * waves)) as u32,
            ((n + 16 * bt - 1) / (16 * bt)) as u32,
            1,
        ];
        let block = [(32 * waves) as u32, 1, 1];
        let out_bytes = n * total_m * 4;
        let bytes = m
            .iter()
            .map(|&mi| crate::profile::hfq4g256_weight_bytes(mi, k))
            .sum::<usize>()
            + n * k
            + n * 4
            + if op == Fp8VerifyOp::Residual {
                2 * out_bytes
            } else {
                out_bytes
            };
        let timer = crate::profile::begin_timer(&self.hip, "gemm", symbol, bytes);
        let result = self.launch_maybe_blob(symbol, grid, block, 0, &mut params[..np + ni], || {
            let mut b = hip_bridge::KernargBlob::new();
            for p in &ptrs[..np] {
                b.push_ptr(*p);
            }
            for v in &ints[..ni] {
                b.push_i32(*v);
            }
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        if self.fp8_verify_scope {
            FP8_VERIFY_CALLS.fetch_add(1, Ordering::Relaxed);
            FP8_VERIFY_ROWS.fetch_add(n as u64, Ordering::Relaxed);
            FP8_VERIFY_SCOPE_CALLS.fetch_add(1, Ordering::Relaxed);
            FP8_VERIFY_SCOPE_ROWS.fetch_add(n as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Validate, load, pack `x`, launch: the shared body of the five
    /// `_verify_fp8` entries (production tile plan).
    #[allow(clippy::too_many_arguments)]
    fn fp8_verify_run(
        &mut self,
        what: &str,
        op: Fp8VerifyOp,
        a: &[&GpuTensor],
        x: &GpuTensor,
        y: &[&GpuTensor],
        m: &[usize],
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.fp8_verify_validate(what, m, k, n)?;
        self.bind_thread()?;
        let tile = fp8_verify_tile(n, m.iter().sum());
        let symbol = self.fp8_verify_load(op, tile)?;
        let xq = self.fp8_verify_pack_x(x, n, k)?;
        let mut a_ptrs = [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_MATS];
        let mut y_ptrs = [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_MATS];
        for (slot, t) in a_ptrs.iter_mut().zip(a) {
            *slot = t.buf.as_ptr();
        }
        for (slot, t) in y_ptrs.iter_mut().zip(y) {
            *slot = t.buf.as_ptr();
        }
        let nm = op.mats();
        self.fp8_verify_launch(op, tile, symbol, &a_ptrs[..nm], &y_ptrs[..nm], m, &xq)
    }

    /// FP8 verify QKVZA projection for `1 <= n <= 128` rows (F32 overwrite
    /// into four output matrices); same signature as
    /// `gemm_qkvza_mq4g256v2_verify_exact`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkvza_mq4g256v2_verify_fp8(
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
        self.fp8_verify_run(
            "gemm_qkvza_mq4g256v2_verify_fp8",
            Fp8VerifyOp::Qkvza,
            &[a_qkv, a_z, a_beta, a_alpha],
            x,
            &[y_qkv, y_z, y_beta, y_alpha],
            &[qkv_m, z_m, beta_m, alpha_m],
            k,
            n,
        )
    }

    /// FP8 verify QKV projection (see [`Self::gemm_qkvza_mq4g256v2_verify_fp8`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qkv_mq4g256v2_verify_fp8(
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
        self.fp8_verify_run(
            "gemm_qkv_mq4g256v2_verify_fp8",
            Fp8VerifyOp::Qkv,
            &[a_q, a_k, a_v],
            x,
            &[y_q, y_k, y_v],
            &[q_m, k_m, v_m],
            k,
            n,
        )
    }

    /// FP8 verify gate/up projection (see [`Self::gemm_qkvza_mq4g256v2_verify_fp8`]).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_gate_up_mq4g256v2_verify_fp8(
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
        self.fp8_verify_run(
            "gemm_gate_up_mq4g256v2_verify_fp8",
            Fp8VerifyOp::GateUp,
            &[a_gate, a_up],
            x,
            &[y_gate, y_up],
            &[gate_m, up_m],
            k,
            n,
        )
    }

    /// FP8 verify residual projection (`Y += W @ X`, one fma epilogue).
    pub fn gemm_mq4g256v2_residual_verify_fp8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.fp8_verify_run(
            "gemm_mq4g256v2_residual_verify_fp8",
            Fp8VerifyOp::Residual,
            &[a],
            x,
            &[y],
            &[m],
            k,
            n,
        )
    }

    /// FP8 verify lm_head: the `set` op overwrites `Y = a*T` (no memset),
    /// at every row count including `n == 1`.
    pub fn gemm_mq4g256v2_lmhead_verify_fp8(
        &mut self,
        a: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.fp8_verify_run(
            "gemm_mq4g256v2_lmhead_verify_fp8",
            Fp8VerifyOp::Set,
            &[a],
            x,
            &[y],
            &[m],
            k,
            n,
        )
    }

    /// Harness entry: one launch of `op` with an explicit `tile` (`None` =
    /// the production plan) on pre-packed activations `xq` (from
    /// [`Self::fp8_verify_pack_x`] or caller-built bytes/scales). `a`, `y`
    /// and `m` carry `op.mats()` entries. Validates before any load/launch;
    /// a tile outside the shipped seven is an error.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_mq4g256v2_verify_fp8_prepacked(
        &mut self,
        op: Fp8VerifyOp,
        tile: Option<Fp8VerifyTile>,
        a: &[&GpuTensor],
        xq: &Fp8VerifyX,
        y: &[&GpuTensor],
        m: &[usize],
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "gemm_mq4g256v2_verify_fp8_prepacked";
        let nm = op.mats();
        if a.len() != nm || y.len() != nm || m.len() != nm {
            return Err(fp8_err(&format!(
                "{WHAT}: {op:?} takes {nm} matrices (a={}, y={}, m={})",
                a.len(),
                y.len(),
                m.len()
            )));
        }
        self.fp8_verify_validate(WHAT, m, k, n)?;
        if xq.n != n || xq.k != k || xq.x8.is_null() || xq.scales.is_null() {
            return Err(fp8_err(&format!(
                "{WHAT}: packed activations are [{}, {}], launch is [{n}, {k}] (or null)",
                xq.n, xq.k
            )));
        }
        let tile = tile.unwrap_or_else(|| fp8_verify_tile(n, m.iter().sum()));
        if tile_index(tile).is_none() {
            return Err(fp8_err(&format!(
                "{WHAT}: no bt{}w{} tile in the FP8 verify bundle",
                tile.bt, tile.waves
            )));
        }
        self.bind_thread()?;
        let symbol = self.fp8_verify_load(op, tile)?;
        let mut a_ptrs = [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_MATS];
        let mut y_ptrs = [std::ptr::null_mut::<c_void>(); FP8_VERIFY_MAX_MATS];
        for (slot, t) in a_ptrs.iter_mut().zip(a) {
            *slot = t.buf.as_ptr();
        }
        for (slot, t) in y_ptrs.iter_mut().zip(y) {
            *slot = t.buf.as_ptr();
        }
        self.fp8_verify_launch(op, tile, symbol, &a_ptrs[..nm], &y_ptrs[..nm], m, xq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPS: [Fp8VerifyOp; 5] = [
        Fp8VerifyOp::Qkvza,
        Fp8VerifyOp::Qkv,
        Fp8VerifyOp::GateUp,
        Fp8VerifyOp::Residual,
        Fp8VerifyOp::Set,
    ];

    #[test]
    fn tile_plan_covers_one_to_128_rows() {
        // 27B shapes: residual 5120, qkv 14336, qkvza 16480, gate_up 34816,
        // lm_head 248320 rows (the measured best tile of each tier).
        for n in 1..=128usize {
            let tier = if n <= 16 {
                0
            } else if n <= 32 {
                1
            } else if n <= 64 {
                2
            } else {
                3
            };
            let narrow = [(1, 1), (2, 4), (4, 4), (8, 4)][tier];
            let wide = [(2, 8), (2, 8), (4, 8), (8, 8)][tier];
            let head = [(4, 8), (4, 8), (4, 8), (8, 8)][tier];
            for (total_m, want) in [
                (5120, narrow),
                (14336, wide),
                (16480, wide),
                (34816, wide),
                (248320, head),
            ] {
                let t = fp8_verify_tile(n, total_m);
                assert_eq!((t.bt, t.waves), want, "n={n} total_m={total_m}");
                // The batch tiles cover the rows and the tile ships in the bundle.
                assert!(16 * t.bt as usize >= n, "n={n}");
                assert!(tile_index(t).is_some(), "n={n}");
            }
        }
    }

    #[test]
    fn tile_plan_thresholds_are_512_and_4096_row_tiles() {
        for n in [1usize, 16, 17, 32, 33, 64, 65, 128] {
            let narrow = fp8_verify_tile(n, 16 * 511);
            let wide = fp8_verify_tile(n, 16 * 512);
            // ceil(total_m / 16) = 512 already at 16 * 511 + 1.
            assert_eq!(fp8_verify_tile(n, 16 * 511 + 1), wide, "n={n}");
            if n <= 16 {
                assert_eq!((narrow.bt, narrow.waves), (1, 1), "n={n}");
                assert_eq!((wide.bt, wide.waves), (2, 8), "n={n}");
            } else {
                assert_eq!(narrow.waves, 4, "n={n}");
                assert_eq!(wide.waves, 8, "n={n}");
                assert_eq!(narrow.bt, wide.bt, "n={n}");
            }
            let below_head = fp8_verify_tile(n, 16 * 4095);
            let head = fp8_verify_tile(n, 16 * 4096);
            assert_eq!(head.waves, 8, "n={n}");
            assert_eq!(head.bt, below_head.bt.max(4), "n={n}");
        }
    }

    #[test]
    fn symbol_table_names_all_35_entries() {
        let mut seen = std::collections::BTreeSet::new();
        for (op, name) in OPS
            .iter()
            .zip(["qkvza", "qkv", "gate_up", "residual", "set"])
        {
            for &(bt, w) in &FP8_VERIFY_TILES {
                let sym = fp8_verify_symbol(*op, Fp8VerifyTile { bt, waves: w });
                assert_eq!(sym, format!("mq4_fp8_verify_{name}_pm_gfx1201_bt{bt}w{w}"));
                assert!(seen.insert(sym), "duplicate symbol {sym}");
            }
        }
        assert_eq!(seen.len(), 35);
    }

    #[test]
    fn production_tiles_resolve_to_shipped_symbols() {
        for op in OPS {
            for n in 1..=128usize {
                for total_m in [1usize, 16, 5120, 16 * 1024, 34816] {
                    let sym = fp8_verify_symbol(op, fp8_verify_tile(n, total_m));
                    assert!(sym.starts_with("mq4_fp8_verify_"), "{sym}");
                }
            }
        }
    }

    #[test]
    fn unshipped_tiles_are_not_indexed() {
        for (bt, waves) in [(1, 4), (2, 1), (3, 4), (8, 2), (16, 4), (0, 0)] {
            assert!(
                tile_index(Fp8VerifyTile { bt, waves }).is_none(),
                "bt{bt}w{waves}"
            );
        }
    }

    #[test]
    fn op_matrix_counts_match_kernarg_sizes() {
        // Kernarg bytes 20 * mats + 24 (KERNEL-SPEC ABI): 104, 84, 64, 44, 44.
        let sizes: Vec<usize> = OPS.iter().map(|o| 20 * o.mats() + 24).collect();
        assert_eq!(sizes, [104, 84, 64, 44, 44]);
        assert!(OPS.iter().all(|o| 2 * o.mats() + 2 <= FP8_VERIFY_MAX_PTRS));
        assert!(OPS.iter().all(|o| o.mats() + 2 <= FP8_VERIFY_MAX_INTS));
    }
}
