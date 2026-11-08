// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Symmetric-MQ6 x A8 trunk route of Qwen4 Flash-Next on exact gfx1201.
//!
//! The weights are qt47 (`DType::MQ6G256V2`, 200 B per 256 weights) whose every header is
//! symmetric (`z_h == -32 * s_h`, [`Gpu::qwen4_mq6_sym_check`]), so `w = s_h * (q - 32)` and the
//! codes feed the signed int8 dot4 / iu8 WMMA kernels directly against the A8 `block_i8_128`
//! activations ([`Gpu::mq6_i8_prepare_x`]). Every entry computes, per 128-K block, the exact i32
//! `C` and folds `acc = fmaf(RN(s_h * d), (float)C, acc)`; output `[N][M]` F32 or BF16 (RNE),
//! overwrite.
//!
//! Entry selection by [`mq6sym_i8_select`]`(m, k, n)`:
//! * `n <= 8`: the T1 narrow tile where it measured at least even with the T0 dot4 GEMV
//!   ([`le8_uses_t1`]: `K <= 2560`, `6144 <= M < 100000`, `n` in 1 and 4..=8), else the T0 dot4
//!   GEMV `gemv_mq6sym_i8_rows_r{n}` (grid `[ceil(M/32)]`, block 128, dynamic LDS
//!   `2 * (2 * (n + B) * 144 + 8 * n)`, `B = 2` iff `n == 7`).
//! * `9..=48`: the T1 narrow tile of the measured launcher rule (receipts: `gemm_mq6sym_i8.gfx1201.hip`
//!   header): `M >= 100000` (lm_head) `m64n48`; `M <= 2560 && n >= 24` `m64n16y`; `n <= 16`
//!   `m64n16` (`m32n16` for `M <= 2560`); `n 17..=32` `m64n32` (`m32n32` for `6144 <= M <= 12288`);
//!   `n 33..=48` `m64n48`.
//! * `n >= 49`: the 128x128 prefill tile (dynamic LDS [`MQ6SYM_P128_LDS_BYTES`]).
//!
//! Env (developer): `HIPFIRE_QWEN4_TRUNK_A8` (`developer.qwen4_trunk_a8`, typed boolean): unset
//! requests the route on exact gfx1201, `0`/`false` is the kill switch. The route additionally
//! needs the loader's symmetric verdict ([`Gpu::set_qwen4_trunk_a8_verified`]).
//!
//! Multi-part form ([`Gpu::gemm_mq6sym_i8_prequant_x4`], `HIPFIRE_QWEN4_TRUNK_A8_X4`, developer,
//! default off): up to four F32 members of one K and one `xq` (GDN qkv|z|a|b, QSA q|k|v|idx) in ONE
//! `gemm_mq6sym_i8_set_{m64n16,m32n32,m64n48,m64n16y}_x4_gfx1201` launch, one tile picked from the
//! largest member; every output word is the single-matrix entry's, bit for bit.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernel_abi;
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use hipfire_kernel_abi::families::mq6_i8::{
    FORMAT_BF16_N_M, FORMAT_BLOCK_I8_128_KB_N, FORMAT_F32_N_M, FORMAT_MQ6G256V2_WEIGHTS,
};
use hipfire_kernel_abi::{InvocationV1, MAX_FIELDS};
use std::ffi::c_void;
use std::sync::LazyLock;

/// Smallest `route_rows` the trunk route serves: one more than the few-row decode limit. At 1..=8
/// rows the served F32 `x4` / split-K trunk GEMVs already stream the same MQ6 weight bytes at ~560
/// GB/s in-situ (rocprof, EP2 C1 MTP window: 4.85 ms/step), so the A8 route can only add its
/// quantize launches there (+1.4 ms/step measured); the symmetric artifact runs those rows on the
/// affine kernels (`z = -32 s` is exact in the affine container). From 9 rows (wide verify /
/// prefill chunks up to [`QWEN4_TRUNK_A8_MAX_ROWS`]) the integer tiles replace the WMMA / FP8 / affine-I8 routes.
pub const QWEN4_TRUNK_A8_MIN_ROWS: usize = crate::prefill_invariant::FEW_ROW_ROUTE_ROWS + 1;

/// Largest `route_rows` the trunk route serves: the exact EP joint prefill wave's routing cap (511).
/// A lane's prefill inside a wave routes as `route_rows = 511`, its singleton tile routes as its own
/// rows, and the two must pick the same kernels, so the route must be constant over 9..=511; above it
/// (>= 512-row tiles, never waved) the served FP8 / affine-I8 routes keep prefill: EP2 R9700 pair A/B
/// (A8 vs `HIPFIRE_QWEN4_TRUNK_A8=0`, ABBA): prefill 194 tokens +5.9 %, 8-lane burst TTFT 6.8 % faster,
/// 613..8172 tokens -1.9..-5.0 % (A8 uncapped; the 128x128 tile loses to FP8 from a few hundred rows;
/// 194..613 rows were not swept).
pub const QWEN4_TRUNK_A8_MAX_ROWS: usize =
    crate::prefill_invariant::QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;
/// qt47 group bytes (8 B headers + 192 B 6-bit payload).
const MQ6_GROUP_BYTES: usize = 200;
/// A8 `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_I8_128_BYTES: usize = 136;
/// Dynamic LDS of the 128x128 prefill entries (`MQ6SYM_P128_LDS_BYTES` of the kernel header).
pub const MQ6SYM_P128_LDS_BYTES: u32 = 36864;
/// Weight-row count `m` from which a matrix is lm_head-class (M = 248320) in the T1 launcher rule;
/// the trunk projections are M <= 12288. A shape class, not a token-row predicate.
const LM_HEAD_CLASS_M: usize = 100_000;

const GEMV_MODULE: &str = "gemv_mq6sym_i8_gfx1201";
const GEMM_MODULE: &str = "gemm_mq6sym_i8_gfx1201";
const CHECK_MODULE: &str = "qwen4_mq6_sym_check_gfx1201";
const CHECK_SYMBOL: &str = "qwen4_mq6_sym_check_gfx1201";

/// Whether `m x k` at `n <= 8` tokens takes the T1 narrow tile (`m64n16`) instead of the T0 dot4
/// GEMV. Card C, 3 processes, kernel-only medians (`release-0.4.5/parity/kernel-runs/a8/
/// wmma-time-2`, RESULT.md): T1 is even at n=1 (+-0.4 %) and n=4 (+-1 %) and wins n=8 by 5-9 % on
/// the K=2560 projections M 6144/10240/12288; T0 wins n=2 by 2-7 % everywhere, and every n on
/// out_proj/o_proj (K=6144) and lm_head (M=248320). n=3 follows n=2; unmeasured tiny M
/// (48/512/640) keeps T0.
fn le8_uses_t1(m: usize, k: usize, n: usize) -> bool {
    !matches!(n, 2 | 3) && k <= 2560 && (6144..LM_HEAD_CLASS_M).contains(&m)
}

/// `HIPFIRE_QWEN4_TRUNK_A8` (`developer.qwen4_trunk_a8`, typed boolean). Read once.
static QWEN4_TRUNK_A8: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_QWEN4_TRUNK_A8"));

/// Pure form of [`Gpu::qwen4_trunk_a8_requested`]: exact gfx1201 and the flag not explicitly false.
fn trunk_a8_requested(arch: &str, flag: Option<bool>) -> bool {
    arch == "gfx1201" && flag != Some(false)
}

/// Pure inputs of [`trunk_a8_admits`].
#[derive(Clone, Copy, Debug)]
struct TrunkA8Gate {
    requested: bool,
    verified: bool,
    qwen4_scope: bool,
    prefill_invariant_scope: bool,
    recording: bool,
    capture: bool,
    m: usize,
    k: usize,
    route_rows: usize,
    few_row_limit: usize,
}

/// Pure form of [`Gpu::qwen4_trunk_a8_applies`].
fn trunk_a8_admits(g: TrunkA8Gate) -> bool {
    g.requested
        && g.verified
        && g.qwen4_scope
        && !g.prefill_invariant_scope
        && !g.recording
        && !g.capture
        && g.k > 0
        && g.k % 256 == 0
        && g.m > 0
        && g.route_rows > g.few_row_limit
        && (QWEN4_TRUNK_A8_MIN_ROWS..=QWEN4_TRUNK_A8_MAX_ROWS).contains(&g.route_rows)
}

/// One launchable tile of the symmetric A8 route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    /// T0 dot4 GEMV with exactly this many tokens (1..=8).
    Gemv(usize),
    M32N16,
    M32N32,
    M64N16,
    M64N32,
    M64N48,
    M64N16Y,
    P128,
}

/// Everything one launch needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Select {
    tile: Tile,
    module: &'static str,
    source: &'static str,
    grid: [u32; 3],
    block: u32,
    lds: u32,
}

/// Entry symbols of the T0 GEMV by token count (index `n - 1`), F32 / BF16.
const GEMV_F32: [&str; 8] = [
    "gemv_mq6sym_i8_rows_r1_gfx1201",
    "gemv_mq6sym_i8_rows_r2_gfx1201",
    "gemv_mq6sym_i8_rows_r3_gfx1201",
    "gemv_mq6sym_i8_rows_r4_gfx1201",
    "gemv_mq6sym_i8_rows_r5_gfx1201",
    "gemv_mq6sym_i8_rows_r6_gfx1201",
    "gemv_mq6sym_i8_rows_r7_gfx1201",
    "gemv_mq6sym_i8_rows_r8_gfx1201",
];
const GEMV_BF16: [&str; 8] = [
    "gemv_mq6sym_i8_rows_r1_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r2_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r3_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r4_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r5_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r6_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r7_bf16_gfx1201",
    "gemv_mq6sym_i8_rows_r8_bf16_gfx1201",
];

/// Dynamic LDS of the T0 entry for `n` tokens (`mq6sym_x_lds_bytes` of the kernel header):
/// `2 * (2 * (n + B) * 144 + 8 * n)`, `B = 2` iff `n == 7`.
const fn gemv_lds_bytes(n: usize) -> u32 {
    let b = if n == 7 { 2 } else { 1 };
    (2 * (2 * (n + b) * 144 + 8 * n)) as u32
}

/// Entry symbol of `tile` for an F32 or BF16 `Y`.
fn tile_entry(tile: Tile, bf16: bool) -> &'static str {
    match (tile, bf16) {
        (Tile::Gemv(n), false) => GEMV_F32[n - 1],
        (Tile::Gemv(n), true) => GEMV_BF16[n - 1],
        (Tile::M32N16, false) => "gemm_mq6sym_i8_set_m32n16_gfx1201",
        (Tile::M32N16, true) => "gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        (Tile::M32N32, false) => "gemm_mq6sym_i8_set_m32n32_gfx1201",
        (Tile::M32N32, true) => "gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        (Tile::M64N16, false) => "gemm_mq6sym_i8_set_m64n16_gfx1201",
        (Tile::M64N16, true) => "gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        (Tile::M64N32, false) => "gemm_mq6sym_i8_set_m64n32_gfx1201",
        (Tile::M64N32, true) => "gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        (Tile::M64N48, false) => "gemm_mq6sym_i8_set_m64n48_gfx1201",
        (Tile::M64N48, true) => "gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        (Tile::M64N16Y, false) => "gemm_mq6sym_i8_set_m64n16y_gfx1201",
        (Tile::M64N16Y, true) => "gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        (Tile::P128, false) => "gemm_mq6sym_i8_set_gfx1201",
        (Tile::P128, true) => "gemm_mq6sym_i8_set_bf16_gfx1201",
    }
}

/// The T1 narrow tile for `1 <= n <= 48` (the measured launcher rule).
fn narrow_tile(m: usize, n: usize) -> Tile {
    if n > 8 && m >= LM_HEAD_CLASS_M {
        return Tile::M64N48;
    }
    if m <= 2560 && n >= 24 {
        return Tile::M64N16Y;
    }
    match n {
        0..=16 if m <= 2560 => Tile::M32N16,
        0..=16 => Tile::M64N16,
        17..=32 if (6144..=12288).contains(&m) => Tile::M32N32,
        17..=32 => Tile::M64N32,
        _ => Tile::M64N48,
    }
}

/// Tile choice for `m` weight rows x `n` tokens.
fn mq6sym_i8_tile(m: usize, k: usize, n: usize) -> Tile {
    match n {
        1..=8 if !le8_uses_t1(m, k, n) => Tile::Gemv(n),
        0..=48 => narrow_tile(m, n),
        _ => Tile::P128,
    }
}

/// Entry, module and launch geometry of one launch of `m` rows x `k` x `n` tokens.
fn mq6sym_i8_select(m: usize, k: usize, n: usize) -> Select {
    let tile = mq6sym_i8_tile(m, k, n);
    let rows = |r: usize| m.div_ceil(r) as u32;
    let (module, source) = match tile {
        Tile::Gemv(_) => (GEMV_MODULE, kernels::GEMV_MQ6SYM_I8_GFX1201_SRC),
        _ => (GEMM_MODULE, kernels::GEMM_MQ6SYM_I8_GFX1201_SRC),
    };
    let (grid, block, lds) = match tile {
        Tile::Gemv(t) => ([rows(32), 1, 1], 128, gemv_lds_bytes(t)),
        Tile::M32N16 | Tile::M32N32 => ([rows(32), 1, 1], 64, 0),
        Tile::M64N16 | Tile::M64N32 | Tile::M64N48 => ([rows(64), 1, 1], 128, 0),
        Tile::M64N16Y => ([rows(64), n.div_ceil(16) as u32, 1], 128, 0),
        Tile::P128 => (
            [rows(128), n.div_ceil(128) as u32, 1],
            256,
            MQ6SYM_P128_LDS_BYTES,
        ),
    };
    Select {
        tile,
        module,
        source,
        grid,
        block,
        lds,
    }
}

/// Largest token count of a multi-part (x4) launch: the narrow tiles serve 9..=48, the token-split
/// `m64n16y_x4` (grid.y = ceil(N/16)) 49..=64. Wider launches (>= 65 rows, prefill) keep one
/// 128x128 launch per member. The bound is the wide-verify window's top,
/// [`crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS`] (`hipfire_config::QWEN4_VERIFY_MAX_ROWS`): the
/// x4 entries exist to serve the verify window (the tile rule above 48 rows is the token-split
/// tile sized for it), so the window's own limit is the single source, not a tuned threshold.
pub const MQ6SYM_X4_MAX_ROWS: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
/// Most matrices of one multi-part launch (the `W0..W3` kernargs).
pub const MQ6SYM_X4_MAX_PARTS: usize = 4;
/// Explicit kernarg bytes of the x4 entries: `W0..W3 @0, Y0..Y3 @32, Xq @64, M0..M3 @72, K @88, N @92`.
const X4_EXPLICIT_BYTES: usize = 96;

/// Multi-part (F32) entry of `tile`, for the four tiles the wide-verify rule picks on the trunk
/// groups; `None` for every other tile (those launch per member).
fn x4_entry(tile: Tile) -> Option<&'static str> {
    match tile {
        Tile::M64N16 => Some("gemm_mq6sym_i8_set_m64n16_x4_gfx1201"),
        Tile::M32N32 => Some("gemm_mq6sym_i8_set_m32n32_x4_gfx1201"),
        Tile::M64N48 => Some("gemm_mq6sym_i8_set_m64n48_x4_gfx1201"),
        Tile::M64N16Y => Some("gemm_mq6sym_i8_set_m64n16y_x4_gfx1201"),
        _ => None,
    }
}

/// The one tile of a multi-part launch of `n` tokens whose largest part has `max_m` rows: the
/// single-matrix rule of that part ([`narrow_tile`]) for `9..=48`, the token-split tile for
/// `49..=64`; `None` when no x4 entry exists for it (the caller launches per member).
fn x4_tile(max_m: usize, n: usize) -> Option<Tile> {
    if !(QWEN4_TRUNK_A8_MIN_ROWS..=MQ6SYM_X4_MAX_ROWS).contains(&n) {
        return None;
    }
    let tile = if n > 48 {
        Tile::M64N16Y
    } else {
        narrow_tile(max_m, n)
    };
    x4_entry(tile).map(|_| tile)
}

/// Launch geometry `(grid, block)` of the x4 entry of `tile` over parts of `ms` rows and `n` tokens.
fn x4_geometry(tile: Tile, ms: &[usize], n: usize) -> ([u32; 3], u32) {
    let rows = if tile == Tile::M32N32 { 32 } else { 64 };
    let blocks = ms.iter().map(|m| m.div_ceil(rows)).sum::<usize>() as u32;
    match tile {
        Tile::M32N32 => ([blocks, 1, 1], 64),
        Tile::M64N16Y => ([blocks, n.div_ceil(16) as u32, 1], 128),
        _ => ([blocks, 1, 1], 128),
    }
}

impl Gpu {
    /// Whether the trunk A8 route is requested on this device: exact gfx1201 and
    /// `HIPFIRE_QWEN4_TRUNK_A8` not explicitly false (unset requests it; `0` is the kill switch).
    pub fn qwen4_trunk_a8_requested(&self) -> bool {
        trunk_a8_requested(&self.arch, *QWEN4_TRUNK_A8)
    }

    /// Record the loader's verdict: every qt47 trunk tensor of the loaded model passed
    /// [`Gpu::qwen4_mq6_sym_check`]. Defaults to false (fail closed).
    pub fn set_qwen4_trunk_a8_verified(&mut self, on: bool) {
        self.qwen4_trunk_a8_verified = on;
    }

    /// Whether one projection of `m` rows x `k` with `rows` tokens takes the trunk A8 route:
    /// requested, verified symmetric, inside the Qwen4 forward, eager (no replay recording / graph
    /// capture), outside the prefill-invariant scope, `k % 256 == 0`, and
    /// `QWEN4_TRUNK_A8_MIN_ROWS <= route_rows <= QWEN4_TRUNK_A8_MAX_ROWS` (9..=511, and above the scope's few-row limit: an admitted wide verify keeps its row-exact twins).
    pub fn qwen4_trunk_a8_applies(&self, m: usize, k: usize, rows: usize) -> bool {
        trunk_a8_admits(TrunkA8Gate {
            requested: self.qwen4_trunk_a8_requested(),
            verified: self.qwen4_trunk_a8_verified,
            qwen4_scope: self.qwen4_scope,
            prefill_invariant_scope: self.prefill_invariant_scope,
            recording: self.replay.is_recording(),
            capture: self.graphs.capture_mode,
            m,
            k,
            route_rows: self.route_rows(rows),
            few_row_limit: self.few_row_limit(),
        })
    }

    /// Name of the kernel [`Gpu::gemm_mq6sym_i8_prequant`] launches for `m` x `k` x `n` tokens.
    pub fn gemm_mq6sym_i8_kernel_name(
        &self,
        m: usize,
        k: usize,
        n: usize,
        bf16: bool,
    ) -> &'static str {
        tile_entry(mq6sym_i8_tile(m, k, n), bf16)
    }

    /// True when every half-group header of the `m x (k/256)` qt47 groups of `w` is finite and
    /// satisfies `float(z) == -32 * float(s)` (the symmetric grid). `Ok(false)` for a shape the
    /// route cannot take (`k % 256 != 0`, empty, or `w` smaller than `m * k/256 * 200` bytes);
    /// `Err` off exact gfx1201. Synchronous; load-time only, never captured.
    pub fn qwen4_mq6_sym_check(&mut self, w: &GpuTensor, m: usize, k: usize) -> HipResult<bool> {
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!(
                    "qwen4_mq6_sym_check: exact gfx1201 required, got {}",
                    self.arch
                ),
            ));
        }
        let Some(weight_bytes) = m
            .checked_mul(k / 256)
            .and_then(|g| g.checked_mul(MQ6_GROUP_BYTES))
        else {
            return Ok(false);
        };
        if m == 0 || k == 0 || k % 256 != 0 || w.byte_size() < weight_bytes {
            return Ok(false);
        }
        let halves = m * (k / 128);
        let words = halves.div_ceil(256);
        let (Ok(grid), Ok(mv), Ok(kv)) = (u32::try_from(words), i32::try_from(m), i32::try_from(k))
        else {
            return Ok(false);
        };
        self.bind_thread()?;
        self.ensure_kernel(
            CHECK_MODULE,
            kernels::QWEN4_MQ6_SYM_CHECK_GFX1201_SRC,
            CHECK_SYMBOL,
        )?;
        let bad = self.alloc_tensor(&[words], DType::F32)?;
        let wp = w.buf.as_ptr();
        let bp = bad.buf.as_ptr();
        let mut params = [
            &wp as *const _ as *mut c_void,
            &bp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let launched = self.launch_maybe_blob(
            CHECK_SYMBOL,
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(wp);
                b.push_ptr(bp);
                b.push_i32(mv);
                b.push_i32(kv);
                b
            },
        );
        let read = launched
            .and_then(|()| self.hip.device_synchronize())
            .and_then(|()| self.download_raw_bytes(&bad));
        let freed = self.free_tensor(bad);
        let read = read?;
        freed?;
        Ok(read.iter().all(|&byte| byte == 0))
    }

    /// Overwrite `y` (`[n x m]`, F32, or BF16 bits when `y.dtype == DType::BF16`) with `W . X` for
    /// verified-symmetric qt47 weights `a` (`m` rows x `k/256` groups x 200 B) and the A8 blocks
    /// `xq` of [`Gpu::mq6_i8_prepare_x`] (`n` tokens). Eager-only; the entry comes from
    /// [`mq6sym_i8_select`].
    pub fn gemm_mq6sym_i8_prequant(
        &mut self,
        a: &GpuTensor,
        xq: *mut c_void,
        y: &GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "gemm_mq6sym_i8: eager-only"));
        }
        if self.arch != "gfx1201" {
            return Err(HipError::new(0, "gemm_mq6sym_i8: exact gfx1201 required"));
        }
        if m == 0 || n == 0 || k == 0 || k % 256 != 0 || xq.is_null() {
            return Err(HipError::new(
                0,
                "gemm_mq6sym_i8: need M>0, N>0, K>0, K%256==0 and prepared blocks",
            ));
        }
        let bf16 = match y.dtype {
            DType::F32 => false,
            DType::BF16 => true,
            _ => return Err(HipError::new(0, "gemm_mq6sym_i8: y must be F32 or BF16")),
        };
        let weight_bytes = m
            .checked_mul(k / 256)
            .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
            .ok_or_else(|| HipError::new(0, "gemm_mq6sym_i8: weight bytes overflow"))?;
        let out_bytes = n
            .checked_mul(m)
            .and_then(|v| v.checked_mul(if bf16 { 2 } else { 4 }))
            .ok_or_else(|| HipError::new(0, "gemm_mq6sym_i8: n*m overflow"))?;
        if a.buf.size() < weight_bytes || y.buf.size() < out_bytes {
            return Err(HipError::new(
                0,
                "gemm_mq6sym_i8: undersized weight or output tensor",
            ));
        }
        let sel = mq6sym_i8_select(m, k, n);
        let kernel = tile_entry(sel.tile, bf16);
        // One typed contract per kernel (hipfire-kernel-abi): operand formats/extents, launch
        // geometry and the kernarg image are validated and encoded once, before the module loads.
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
        let block = [sel.block, 1, 1];
        let mut v = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch,
                symbol: kernel,
                params: &[m as u64, k as u64, n as u64],
                operands: &operands,
                grid: sel.grid,
                block,
                dynamic_lds: sel.lds,
            },
        )?;
        self.ensure_kernel(sel.module, sel.source, kernel)?;
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
        let nptrs = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
        let bytes = weight_bytes + n * (k / 128) * BLOCK_I8_128_BYTES + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result =
            self.launch_maybe_blob(kernel, sel.grid, block, sel.lds, &mut ptrs[..nptrs], || {
                kernel_abi::blob(v.explicit())
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // `y` was overwritten: caches keyed on its pointer are stale.
        self.invalidate_x_caches_for(y.buf.as_ptr());
        result
    }

    /// Name of the kernel [`Gpu::gemm_mq6sym_i8_prequant_x4`] launches for parts of `ms` rows and
    /// `n` tokens, `None` when the multi-part route does not take the shape (more than
    /// [`MQ6SYM_X4_MAX_PARTS`] parts, `n` outside 9..=[`MQ6SYM_X4_MAX_ROWS`], or a tile without an
    /// x4 entry): the caller then launches [`Gpu::gemm_mq6sym_i8_prequant`] per member.
    pub fn gemm_mq6sym_i8_x4_kernel_name(&self, ms: &[usize], n: usize) -> Option<&'static str> {
        if ms.is_empty() || ms.len() > MQ6SYM_X4_MAX_PARTS {
            return None;
        }
        x4_tile(ms.iter().copied().max()?, n).and_then(x4_entry)
    }

    /// `HIPFIRE_QWEN4_TRUNK_A8_X4` (developer, default off): the F32 members of one K of the wide
    /// A8 trunk route launch as one multi-part [`Gpu::gemm_mq6sym_i8_prequant_x4`].
    pub fn qwen4_trunk_a8_x4_enabled(&self) -> bool {
        self.flags.qwen4_trunk_a8_x4 && self.arch == "gfx1201"
    }

    /// Overwrite each part's `y` (`[n x m]`, F32) with `W . X` for up to four verified-symmetric
    /// qt47 matrices `(weights, y, m)` of one `k` against the same A8 blocks `xq`
    /// ([`Gpu::mq6_i8_prepare_x`], `n` tokens), in ONE launch. Every output word is bytewise the
    /// word [`Gpu::gemm_mq6sym_i8_prequant`] writes for that member (one tile per launch, picked
    /// from the largest part; all tiles share the per-row arithmetic). Eager-only; errors where
    /// [`Gpu::gemm_mq6sym_i8_x4_kernel_name`] is `None`.
    pub fn gemm_mq6sym_i8_prequant_x4(
        &mut self,
        parts: &[(&GpuTensor, &GpuTensor, usize)],
        xq: *mut c_void,
        k: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(0, "gemm_mq6sym_i8_x4: eager-only"));
        }
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                "gemm_mq6sym_i8_x4: exact gfx1201 required",
            ));
        }
        if parts.is_empty()
            || parts.len() > MQ6SYM_X4_MAX_PARTS
            || n == 0
            || k == 0
            || k % 256 != 0
            || xq.is_null()
            || parts.iter().any(|p| p.2 == 0)
        {
            return Err(HipError::new(
                0,
                "gemm_mq6sym_i8_x4: need 1..=4 parts with M>0, N>0, K>0, K%256==0 and prepared blocks",
            ));
        }
        let mut ms = [0usize; MQ6SYM_X4_MAX_PARTS];
        let mut weight_bytes = 0usize;
        let mut out_bytes = 0usize;
        for (i, (w, y, m)) in parts.iter().enumerate() {
            if y.dtype != DType::F32 {
                return Err(HipError::new(0, "gemm_mq6sym_i8_x4: every y must be F32"));
            }
            let wb = m
                .checked_mul(k / 256)
                .and_then(|v| v.checked_mul(MQ6_GROUP_BYTES))
                .ok_or_else(|| HipError::new(0, "gemm_mq6sym_i8_x4: weight bytes overflow"))?;
            let ob = n
                .checked_mul(*m)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| HipError::new(0, "gemm_mq6sym_i8_x4: n*m overflow"))?;
            if w.buf.size() < wb || y.buf.size() < ob {
                return Err(HipError::new(
                    0,
                    "gemm_mq6sym_i8_x4: undersized weight or output tensor",
                ));
            }
            ms[i] = *m;
            weight_bytes += wb;
            out_bytes += ob;
        }
        // Parts run concurrently inside one grid: a written extent (`n * m_i` words) that overlaps
        // another part's written extent or any part's weights would race.
        let span = |t: &GpuTensor, bytes: usize| {
            let lo = t.buf.as_ptr() as usize;
            (lo, lo + bytes)
        };
        for (i, (_, yi, mi)) in parts.iter().enumerate() {
            let (lo, hi) = span(yi, n * mi * 4);
            let hits = |(a, b): (usize, usize)| a < hi && lo < b;
            for (j, (wj, yj, mj)) in parts.iter().enumerate() {
                if (j != i && hits(span(yj, n * mj * 4)))
                    || hits(span(wj, mj * (k / 256) * MQ6_GROUP_BYTES))
                {
                    return Err(HipError::new(
                        0,
                        "gemm_mq6sym_i8_x4: an output overlaps another part's output or a weight",
                    ));
                }
            }
        }
        let max_m = ms.iter().copied().max().unwrap_or(0);
        let tile = x4_tile(max_m, n).ok_or_else(|| {
            HipError::new(0, "gemm_mq6sym_i8_x4: no multi-part entry for this shape")
        })?;
        let kernel = x4_entry(tile)
            .ok_or_else(|| HipError::new(0, "gemm_mq6sym_i8_x4: tile without x4 entry"))?;
        let (grid, block_x) = x4_geometry(tile, &ms[..parts.len()], n);
        let arch = kernel_abi::parse_arch(&self.arch)?;
        let abi = kernel_abi::contract(&self.arch, kernel)?;
        // ABI operand order: W0..W3, Y0..Y3, Xq; an absent part is a null binding.
        let absent = |format| kernel_abi::binding_raw(std::ptr::null(), None, format);
        let w_at = |i: usize| match parts.get(i) {
            Some((w, _, _)) => kernel_abi::binding(w, FORMAT_MQ6G256V2_WEIGHTS),
            None => absent(FORMAT_MQ6G256V2_WEIGHTS),
        };
        let y_at = |i: usize| match parts.get(i) {
            Some((_, y, _)) => kernel_abi::binding(y, FORMAT_F32_N_M),
            None => absent(FORMAT_F32_N_M),
        };
        let operands = [
            w_at(0),
            w_at(1),
            w_at(2),
            w_at(3),
            y_at(0),
            y_at(1),
            y_at(2),
            y_at(3),
            kernel_abi::binding_raw(xq, self.int8_mmq_xq_bytes(xq), FORMAT_BLOCK_I8_128_KB_N),
        ];
        let block = [block_x, 1, 1];
        let mut v = kernel_abi::validate(
            abi,
            &InvocationV1 {
                arch,
                symbol: kernel,
                params: &[
                    ms[0] as u64,
                    ms[1] as u64,
                    ms[2] as u64,
                    ms[3] as u64,
                    k as u64,
                    n as u64,
                ],
                operands: &operands,
                grid,
                block,
                dynamic_lds: 0,
            },
        )?;
        debug_assert_eq!(v.explicit().len(), X4_EXPLICIT_BYTES);
        self.ensure_kernel(GEMM_MODULE, kernels::GEMM_MQ6SYM_I8_GFX1201_SRC, kernel)?;
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_FIELDS];
        let nptrs = kernel_abi::kernel_params(&mut v, &mut ptrs)?;
        let bytes = weight_bytes + n * (k / 128) * BLOCK_I8_128_BYTES + out_bytes;
        let timer = crate::profile::begin_timer(&self.hip, "gemm", kernel, bytes);
        let result = self.launch_maybe_blob(kernel, grid, block, 0, &mut ptrs[..nptrs], || {
            kernel_abi::blob(v.explicit())
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        // Every `y` was overwritten: caches keyed on their pointers are stale.
        for (_, y, _) in parts {
            self.invalidate_x_caches_for(y.buf.as_ptr());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_kernel_abi::{eval, families, Arch, OperandBinding};

    const FAKE_BYTES: u64 = 1 << 40;

    #[test]
    fn selection_table() {
        let t = |m, n| mq6sym_i8_tile(m, 2560, n);
        // n <= 8: T1 m64n16 on the K=2560 6144..100000-row projections at n 1 and 4..=8, T0 else.
        for n in 1..=8 {
            for m in [64usize, 2560, 248_320] {
                assert_eq!(t(m, n), Tile::Gemv(n), "m={m} n={n}");
            }
            for m in [6144usize, 10240, 12288] {
                let want = if matches!(n, 2 | 3) {
                    Tile::Gemv(n)
                } else {
                    Tile::M64N16
                };
                assert_eq!(t(m, n), want, "m={m} n={n}");
                assert_eq!(
                    mq6sym_i8_tile(m, 6144, n),
                    Tile::Gemv(n),
                    "k=6144 m={m} n={n}"
                );
            }
        }
        // 9..=16.
        assert_eq!(t(2048, 9), Tile::M32N16);
        assert_eq!(t(2560, 16), Tile::M32N16);
        assert_eq!(t(2561, 16), Tile::M64N16);
        assert_eq!(t(10240, 12), Tile::M64N16);
        // 17..=32.
        assert_eq!(t(6144, 17), Tile::M32N32);
        assert_eq!(t(12288, 32), Tile::M32N32);
        assert_eq!(t(12289, 32), Tile::M64N32);
        assert_eq!(t(5120, 20), Tile::M64N32);
        assert_eq!(t(2560, 20), Tile::M64N32);
        // 33..=48.
        assert_eq!(t(6144, 33), Tile::M64N48);
        assert_eq!(t(10240, 48), Tile::M64N48);
        // Small M and N >= 24: the token-split twin.
        for n in 24..=48 {
            assert_eq!(t(2560, n), Tile::M64N16Y, "n={n}");
            assert_eq!(t(128, n), Tile::M64N16Y, "n={n}");
        }
        // lm_head-sized M: m64n48 at every N in 9..=48.
        for n in 9..=48 {
            assert_eq!(t(248_320, n), Tile::M64N48, "n={n}");
            assert_eq!(t(100_000, n), Tile::M64N48, "n={n}");
        }
        assert_eq!(t(248_320, 8), Tile::Gemv(8));
        // n >= 49: the prefill tile.
        for n in [49usize, 64, 65, 512, 4096] {
            for m in [128usize, 2560, 248_320] {
                assert_eq!(t(m, n), Tile::P128, "m={m} n={n}");
            }
        }
    }

    #[test]
    fn launch_geometry() {
        let s = mq6sym_i8_select(10240, 6144, 1);
        assert_eq!((s.grid, s.block, s.lds), ([320, 1, 1], 128, 1168));
        assert_eq!(s.module, GEMV_MODULE);
        assert_eq!(gemv_lds_bytes(8), 5312);
        let s = mq6sym_i8_select(10241, 6144, 4);
        assert_eq!(s.grid, [321, 1, 1]);
        let s = mq6sym_i8_select(10240, 2560, 4);
        assert_eq!((s.grid, s.block, s.lds), ([160, 1, 1], 128, 0));
        let s = mq6sym_i8_select(2048, 2560, 12);
        assert_eq!((s.grid, s.block, s.lds), ([64, 1, 1], 64, 0));
        let s = mq6sym_i8_select(2560, 6144, 40);
        assert_eq!((s.grid, s.block, s.lds), ([40, 3, 1], 128, 0));
        let s = mq6sym_i8_select(248_320, 2560, 40);
        assert_eq!((s.grid, s.block), ([3880, 1, 1], 128));
        let s = mq6sym_i8_select(4097, 2560, 129);
        assert_eq!(
            (s.grid, s.block, s.lds),
            ([33, 2, 1], 256, MQ6SYM_P128_LDS_BYTES)
        );
        assert_eq!(s.module, GEMM_MODULE);
    }

    #[test]
    fn entry_names_follow_tile_and_output() {
        assert_eq!(
            tile_entry(Tile::Gemv(3), false),
            "gemv_mq6sym_i8_rows_r3_gfx1201"
        );
        assert_eq!(
            tile_entry(Tile::Gemv(8), true),
            "gemv_mq6sym_i8_rows_r8_bf16_gfx1201"
        );
        assert_eq!(
            tile_entry(Tile::M64N16Y, true),
            "gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201"
        );
        assert_eq!(tile_entry(Tile::P128, false), "gemm_mq6sym_i8_set_gfx1201");
    }

    const ALL_TILES: [Tile; 15] = [
        Tile::Gemv(1),
        Tile::Gemv(2),
        Tile::Gemv(3),
        Tile::Gemv(4),
        Tile::Gemv(5),
        Tile::Gemv(6),
        Tile::Gemv(7),
        Tile::Gemv(8),
        Tile::M32N16,
        Tile::M32N32,
        Tile::M64N16,
        Tile::M64N32,
        Tile::M64N48,
        Tile::M64N16Y,
        Tile::P128,
    ];

    /// Every symbol a selector can name exists in the module source the launcher compiles from,
    /// the sources carry the block_i8_128 prelude and the P128 LDS constant of the header.
    #[test]
    fn kernel_sources_carry_the_launcher_symbols_and_lds_contract() {
        for tile in ALL_TILES {
            for bf16 in [false, true] {
                let entry = tile_entry(tile, bf16);
                let sel_src = match tile {
                    Tile::Gemv(_) => kernels::GEMV_MQ6SYM_I8_GFX1201_SRC,
                    _ => kernels::GEMM_MQ6SYM_I8_GFX1201_SRC,
                };
                // The GEMV names its entries through `S6_ENTRY(R)`; the GEMM narrow ones through
                // `SYM_ROW(rows, ..)` macro arguments, so check the pieces.
                match tile {
                    Tile::Gemv(n) => {
                        assert!(sel_src.contains(&format!("S6_ENTRY({n})")), "{entry}");
                        assert!(
                            sel_src.contains(if bf16 {
                                "S6_SUF, _bf16_gfx1201)"
                            } else {
                                "S6_SUF, _gfx1201)"
                            }),
                            "{entry}"
                        );
                    }
                    Tile::P128 => assert!(
                        sel_src.contains(if bf16 {
                            "gemm_mq6sym_i8_set_bf16_gfx1201"
                        } else {
                            "gemm_mq6sym_i8_set_gfx1201"
                        }),
                        "{entry}"
                    ),
                    narrow => {
                        // `SYM_ROW(R, NW, TGN, TG, SUF, YS)` instantiates both the F32 and the
                        // `set_bf16_` entry of `gemm_mq6sym_i8_set_m<R>n<TGN><SUF>_gfx1201`.
                        let inst = match narrow {
                            Tile::M32N16 => "SYM_ROW(32, 2, 16, 1, , false)",
                            Tile::M32N32 => "SYM_ROW(32, 2, 32, 2, , false)",
                            Tile::M64N16 => "SYM_ROW(64, 4, 16, 1, , false)",
                            Tile::M64N32 => "SYM_ROW(64, 4, 32, 2, , false)",
                            Tile::M64N48 => "SYM_ROW(64, 4, 48, 3, , false)",
                            Tile::M64N16Y => "SYM_ROW(64, 4, 16, 1, y, true)",
                            _ => unreachable!(),
                        };
                        assert!(sel_src.contains(inst), "{entry}: no `{inst}`");
                        assert!(sel_src.contains("gemm_mq6sym_i8_set_m##R##n##TGN##SUF##_gfx1201"));
                        assert!(
                            sel_src.contains("gemm_mq6sym_i8_set_bf16_m##R##n##TGN##SUF##_gfx1201")
                        );
                    }
                }
            }
        }
        for src in [
            kernels::GEMV_MQ6SYM_I8_GFX1201_SRC,
            kernels::GEMM_MQ6SYM_I8_GFX1201_SRC,
        ] {
            assert!(src.starts_with("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n"));
            assert!(src.contains("block_i8_128"));
        }
        assert!(kernels::GEMM_MQ6SYM_I8_GFX1201_SRC.contains(&format!(
            "#define MQ6SYM_P128_LDS_BYTES {MQ6SYM_P128_LDS_BYTES}"
        )));
        assert!(kernels::GEMV_MQ6SYM_I8_GFX1201_SRC.contains("2 * (2 * (N + B) * 144 + 8 * N)"));
        assert!(kernels::QWEN4_MQ6_SYM_CHECK_GFX1201_SRC.contains(CHECK_SYMBOL));
    }

    fn assert_export_matches(arch: Arch, symbol: &str, sel: &Select, params: &[u64]) {
        let (abi, export) = families::find_export(arch, symbol)
            .unwrap_or_else(|| panic!("no contract export for {symbol} on {}", arch.name()));
        assert_eq!(export.block, [sel.block, 1, 1], "{symbol} block");
        assert_eq!(export.dynamic_lds, sel.lds, "{symbol} dynamic LDS");
        for axis in 0..3 {
            assert_eq!(
                eval(&export.grid[axis], abi, export, params).unwrap(),
                u64::from(sel.grid[axis]),
                "{symbol} grid[{axis}] for {params:?}"
            );
        }
    }

    /// Every launch the selector can pick names an export whose block, LDS and grid are what the
    /// launcher passes, and every export of the family is reachable.
    #[test]
    fn contract_covers_every_selectable_launch() {
        let ms = [
            1usize, 4, 64, 100, 2048, 2560, 2561, 6144, 10240, 12288, 12289, 99_999, 100_000,
            248_320,
        ];
        let mut seen = std::collections::BTreeSet::new();
        for &m in &ms {
            for n in 1..=130usize {
                for bf16 in [false, true] {
                    let sel = mq6sym_i8_select(m, 256, n);
                    let entry = tile_entry(sel.tile, bf16);
                    assert_export_matches(Arch::Gfx1201, entry, &sel, &[m as u64, 256, n as u64]);
                    seen.insert(entry);
                }
            }
        }
        // Every tile the table can select (T0 is reached at n <= 8 for every M at K=256;
        // the T1 n <= 8 crossover only re-routes to narrow tiles already in the set).
        assert_eq!(seen.len(), ALL_TILES.len() * 2, "{seen:?}");
        // The family holds exactly those exports.
        let exported: usize = families::all()
            .filter(|a| a.id.starts_with("mq6sym_i8."))
            .map(|a| a.exports.len())
            .sum();
        assert_eq!(exported, ALL_TILES.len() * 2 + X4_TILES.len());
    }

    const X4_TILES: [Tile; 4] = [Tile::M64N16, Tile::M32N32, Tile::M64N48, Tile::M64N16Y];

    /// The multi-part rule on the two Flash-Next trunk groups (GDN 10240|6144|48|48, QSA
    /// 12288|512|512|640): one tile from the largest part, the single-matrix rule of that part for
    /// 9..=48 and the token-split tile for 49..=64, nothing outside 9..=64.
    #[test]
    fn x4_tile_rule_and_geometry() {
        for max_m in [10240usize, 12288] {
            for n in 9..=64usize {
                let want = match n {
                    9..=16 => Tile::M64N16,
                    17..=32 => Tile::M32N32,
                    33..=48 => Tile::M64N48,
                    _ => Tile::M64N16Y,
                };
                assert_eq!(x4_tile(max_m, n), Some(want), "m={max_m} n={n}");
            }
        }
        for n in [0usize, 1, 8, 65, 128, 4096] {
            assert_eq!(x4_tile(10240, n), None, "n={n}");
        }
        // A tile without an x4 entry is refused, not substituted.
        assert_eq!(x4_tile(2048, 12), None);
        assert_eq!(x4_tile(20000, 24), None);
        let gdn = [10240usize, 6144, 48, 48];
        assert_eq!(
            x4_geometry(Tile::M64N48, &gdn, 48),
            ([160 + 96 + 1 + 1, 1, 1], 128)
        );
        assert_eq!(
            x4_geometry(Tile::M32N32, &gdn, 24),
            ([320 + 192 + 2 + 2, 1, 1], 64)
        );
        assert_eq!(x4_geometry(Tile::M64N16Y, &gdn, 64), ([258, 4, 1], 128));
    }

    /// The x4 kernarg image is the 96 explicit bytes `W0..W3 Y0..Y3 Xq M0..M3 K N`, and every x4
    /// export's block, LDS and grid are what the launcher passes (absent parts null, M_i = 0).
    #[test]
    fn x4_contract_matches_launcher_geometry() {
        let k = 2560usize;
        for tile in X4_TILES {
            let entry = x4_entry(tile).unwrap();
            let abi = kernel_abi::contract("gfx1201", entry).unwrap();
            assert_eq!(abi.explicit_bytes as usize, X4_EXPLICIT_BYTES, "{entry}");
            for (ms, n) in [
                (vec![10240usize, 6144, 48, 48], 48usize),
                (vec![12288, 512, 512, 640], 24),
                (vec![10240, 6144], 64),
                (vec![10240], 9),
            ] {
                let (grid, block) = x4_geometry(tile, &ms, n);
                let mut m4 = [0u64; 4];
                for (i, m) in ms.iter().enumerate() {
                    m4[i] = *m as u64;
                }
                let operands: Vec<OperandBinding> = abi
                    .operands
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        let part = if i < 8 { i % 4 } else { 0 };
                        let live = i >= 8 || part < ms.len();
                        OperandBinding {
                            address: if live {
                                0x1000_0000 * (i as u64 + 1)
                            } else {
                                0
                            },
                            bytes: if live { Some(FAKE_BYTES) } else { None },
                            format: o.format,
                        }
                    })
                    .collect();
                let v = kernel_abi::validate(
                    abi,
                    &InvocationV1 {
                        arch: kernel_abi::parse_arch("gfx1201").unwrap(),
                        symbol: entry,
                        params: &[m4[0], m4[1], m4[2], m4[3], k as u64, n as u64],
                        operands: &operands,
                        grid,
                        block: [block, 1, 1],
                        dynamic_lds: 0,
                    },
                )
                .unwrap_or_else(|e| panic!("{entry} {ms:?} n={n}: {e}"));
                assert_eq!(v.explicit().len(), X4_EXPLICIT_BYTES, "{entry}");
            }
        }
    }

    #[test]
    fn le8_crossover_routes_narrow_tiles() {
        // The narrow rule itself at n <= 8 (where `le8_uses_t1` selects it).
        assert_eq!(narrow_tile(2048, 3), Tile::M32N16);
        assert_eq!(narrow_tile(10240, 8), Tile::M64N16);
        assert!(le8_uses_t1(10240, 2560, 1));
        assert!(!le8_uses_t1(10240, 2560, 2));
        assert!(!le8_uses_t1(2560, 6144, 8));
        assert!(!le8_uses_t1(248_320, 2560, 8));
        assert!(!le8_uses_t1(640, 2560, 8));
    }

    /// The kernarg image of every entry is the 36 explicit bytes `W Xq Y M K N`.
    #[test]
    fn kernarg_contract_is_36_explicit_bytes() {
        for tile in ALL_TILES {
            for bf16 in [false, true] {
                let entry = tile_entry(tile, bf16);
                let abi = kernel_abi::contract("gfx1201", entry).unwrap();
                assert_eq!(abi.explicit_bytes, 36, "{entry}");
                let (m, k, n) = (4096usize, 512usize, 8usize);
                let tile_n = if let Tile::Gemv(r) = tile { r } else { n };
                let sel = Select {
                    tile,
                    ..mq6sym_i8_select_for(tile, m, tile_n)
                };
                let operands: Vec<OperandBinding> = abi
                    .operands
                    .iter()
                    .enumerate()
                    .map(|(i, o)| OperandBinding {
                        address: 0x1000_0000 * (i as u64 + 1),
                        bytes: Some(FAKE_BYTES),
                        format: o.format,
                    })
                    .collect();
                let v = kernel_abi::validate(
                    abi,
                    &InvocationV1 {
                        arch: kernel_abi::parse_arch("gfx1201").unwrap(),
                        symbol: entry,
                        params: &[m as u64, k as u64, tile_n as u64],
                        operands: &operands,
                        grid: sel.grid,
                        block: [sel.block, 1, 1],
                        dynamic_lds: sel.lds,
                    },
                )
                .unwrap_or_else(|e| panic!("{entry}: {e}"));
                assert_eq!(v.explicit().len(), 36, "{entry}");
            }
        }
    }

    /// Geometry of `tile` for `m` x `n` (the table picks a different tile at some (m, n), so build
    /// the launch of the named tile directly).
    fn mq6sym_i8_select_for(tile: Tile, m: usize, n: usize) -> Select {
        let rows = |r: usize| m.div_ceil(r) as u32;
        let (grid, block, lds) = match tile {
            Tile::Gemv(t) => ([rows(32), 1, 1], 128, gemv_lds_bytes(t)),
            Tile::M32N16 | Tile::M32N32 => ([rows(32), 1, 1], 64, 0),
            Tile::M64N16 | Tile::M64N32 | Tile::M64N48 => ([rows(64), 1, 1], 128, 0),
            Tile::M64N16Y => ([rows(64), n.div_ceil(16) as u32, 1], 128, 0),
            Tile::P128 => (
                [rows(128), n.div_ceil(128) as u32, 1],
                256,
                MQ6SYM_P128_LDS_BYTES,
            ),
        };
        Select {
            tile,
            module: "",
            source: "",
            grid,
            block,
            lds,
        }
    }

    #[test]
    fn flag_parsing() {
        assert!(trunk_a8_requested("gfx1201", None));
        assert!(trunk_a8_requested("gfx1201", Some(true)));
        assert!(!trunk_a8_requested("gfx1201", Some(false)));
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx1201:xnack-", ""] {
            assert!(!trunk_a8_requested(arch, None), "{arch}");
            assert!(!trunk_a8_requested(arch, Some(true)), "{arch}");
        }
    }

    #[test]
    fn applies_truth_table() {
        let ok = TrunkA8Gate {
            requested: true,
            verified: true,
            qwen4_scope: true,
            prefill_invariant_scope: false,
            recording: false,
            capture: false,
            m: 5120,
            k: 2048,
            route_rows: 9,
            few_row_limit: 8,
        };
        assert!(trunk_a8_admits(ok));
        // An admitted wide verify (few-row limit raised to its rows) keeps the row-exact decode twins.
        assert!(!trunk_a8_admits(TrunkA8Gate {
            route_rows: 24,
            few_row_limit: 24,
            ..ok
        }));
        assert!(trunk_a8_admits(TrunkA8Gate {
            route_rows: 25,
            few_row_limit: 24,
            ..ok
        }));
        for r in 0..=8 {
            assert!(
                !trunk_a8_admits(TrunkA8Gate {
                    route_rows: r,
                    ..ok
                }),
                "rows {r}"
            );
        }
        for (r, want) in [
            (64usize, true),
            (65, true),
            (194, true),
            (256, true),
            (511, true),
            (512, false),
            (4096, false),
        ] {
            assert_eq!(
                trunk_a8_admits(TrunkA8Gate {
                    route_rows: r,
                    ..ok
                }),
                want,
                "rows {r}"
            );
        }
        assert!(!trunk_a8_admits(TrunkA8Gate {
            route_rows: 0,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            requested: false,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            verified: false,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            qwen4_scope: false,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            prefill_invariant_scope: true,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            recording: true,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate {
            capture: true,
            ..ok
        }));
        assert!(!trunk_a8_admits(TrunkA8Gate { k: 0, ..ok }));
        assert!(!trunk_a8_admits(TrunkA8Gate { k: 2000, ..ok }));
        assert!(!trunk_a8_admits(TrunkA8Gate { m: 0, ..ok }));
    }
}
