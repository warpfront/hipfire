// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane WMMA: purpose-built gfx1201 WMMA kernels for the Qwen3.8 Flash-Next HC mixers and the
//! dense BF16 `[rows, k]` projections at 16..=64 rows (the decode / lane / verify window),
//! for BF16 weights and planar W8 weights (int8 widened to BF16 exactly, the per-32 scale
//! applied in the f32 epilogue).
//!
//! Kernels: `kernels/src/qwen4_lane_wmma.gfx1201.hip`.
//! * [`lane_wmma_gemm`]: `Y[rows, M] = epi(X . W^T)`, optional `hc_act` epilogue, split-K with a
//!   deterministic last-block fold (the partials are summed in split order by the last block
//!   to finish a tile, so the result never depends on block scheduling).
//! * [`lane_wmma_read_up`]: HC up projection with the fused branch mix.
//!
//! Activations are f32 in memory and are fed to the BF16 WMMA as `terms` BF16 split terms:
//! * `terms == 1` is exact only when every activation is already BF16-representable. That is
//!   the HC down on a BF16 weight: `hyper_norm` and `hc_act` outputs are BF16-rounded, so the
//!   single BF16 operand loses nothing and the product accumulates in f32.
//! * `terms == 2` (hi + lo) covers general f32 activations to ~2^-17 relative per element.
//! * `terms == 3` is the lab-only extra-precision variant.
//!
//! The f32 summation order differs from the SIMT routes (WMMA K=16 fragments, split-K fold),
//! so outputs are NOT bitwise the SIMT route's; they are deterministic run to run.
//!
//! This module reads no environment variable. The product routes are gated by
//! `HIPFIRE_QWEN4_LANE_WMMA` (see `Gpu::lane_wmma_applies`); the lab-only prefill W8 twins and the Q8_0
//! repack live in [`crate::lane_wmma_lab`].

use crate::dispatch::ReplayLaunchBindings;
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};

/// The lane-WMMA kernel source (one module for every `lwmma_gemm_*` / `lwmma_read_up_*` entry).
pub const LANE_WMMA_SRC: &str = include_str!("../../../kernels/src/qwen4_lane_wmma.gfx1201.hip");

const LANE_WMMA_MODULE: &str = "qwen4_lane_wmma";

/// Fixed size of the [`Gpu`]-owned split-K scratch: `split_k * m <= 320` tiles of 16 columns
/// at most 64 rows (`320 * 16 * 64` floats), one counter per (column block, token block)
/// (`<= 320`: [`lane_wmma_auto_split_k`] keeps `blocks * split_k <= 320`).
pub const LANE_WMMA_SCRATCH_PARTIAL_FLOATS: usize = 320 * 16 * 64;
/// Counter capacity of the [`Gpu`]-owned split-K scratch.
pub const LANE_WMMA_SCRATCH_COUNTERS: usize = 320;

/// A GEMM weight operand.
pub enum LaneWmmaWeight<'a> {
    /// BF16 bits `[M, K]` row-major.
    Bf16(&'a GpuTensor),
    /// Planar W8 (block 32): `q` i8 `[M, K]` row-major, `d` f16 `[M, K / 32]` row-major;
    /// the weight value is `q * d`.
    W8 { q: &'a GpuTensor, d: &'a GpuTensor },
}

/// A1 launch configuration.
#[derive(Clone, Copy, Debug)]
pub struct LaneWmmaCfg {
    /// Output n-tiles per block, 1 or 2.
    pub nn: usize,
    /// K splits; 0 = [`lane_wmma_auto_split_k`].
    pub split_k: usize,
}

impl Default for LaneWmmaCfg {
    /// The product configuration: one n-tile (measured best) and the auto split-K.
    fn default() -> Self {
        Self { nn: 1, split_k: 0 }
    }
}

/// Split-K scratch: `partial` holds `split_k * rows * M` floats, `counters` one u32 per
/// (column block, token block) and is zeroed once here; the kernel leaves it zeroed.
pub struct LaneWmmaScratch {
    pub partial: GpuTensor,
    pub counters: GpuTensor,
}

impl LaneWmmaScratch {
    pub fn new(gpu: &mut Gpu, max_partial_floats: usize, max_counters: usize) -> HipResult<Self> {
        let partial = gpu.zeros(&[max_partial_floats.max(1)], DType::F32)?;
        let counters = gpu.zeros(&[max_counters.max(1)], DType::F32)?;
        Ok(Self { partial, counters })
    }
}

/// What a launcher dispatched, for the bench log.
#[derive(Clone, Debug)]
pub struct LaneWmmaLaunch {
    pub kernel: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

pub(crate) fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(1, &format!("lane_wmma: {}", msg.as_ref()))
}

/// Token tiles per `lane_wmma_read_up` block (1, 2 or 4; the other `8 / tiles` waves split K): the
/// fewest tiles covering `rows` up to four.
pub fn lane_wmma_read_up_tiles(rows: usize) -> usize {
    match rows.div_ceil(16) {
        0 | 1 => 1,
        2 => 2,
        _ => 4,
    }
}

/// Auto split-K: the largest divisor `s` of `k / 128` with
/// `ceil(m / (16 nn)) * s * ceil(rows / 64) <= 320` (at least 1).
pub fn lane_wmma_auto_split_k(m: usize, k: usize, rows: usize, nn: usize) -> usize {
    let tiles = m.div_ceil(16 * nn.max(1)) * rows.div_ceil(64).max(1);
    let chunks = k / 128;
    let mut best = 1;
    for s in 1..=chunks {
        if chunks % s == 0 && tiles * s <= 320 {
            best = s;
        }
    }
    best
}

pub(crate) fn u32_of(v: usize, what: &str) -> HipResult<u32> {
    u32::try_from(v).map_err(|_| err(format!("{what} {v} exceeds u32")))
}

pub(crate) fn i32_of(v: usize, what: &str) -> HipResult<i32> {
    i32::try_from(v).map_err(|_| err(format!("{what} {v} exceeds i32")))
}

pub(crate) fn need_bytes(t: &GpuTensor, bytes: usize, what: &str) -> HipResult<()> {
    if t.byte_size() < bytes {
        return Err(err(format!(
            "{what}: {} bytes < required {bytes}",
            t.byte_size()
        )));
    }
    Ok(())
}

pub(crate) fn need_f32(t: &GpuTensor, floats: usize, what: &str) -> HipResult<()> {
    if t.dtype != DType::F32 {
        return Err(err(format!("{what}: dtype {:?} is not F32", t.dtype)));
    }
    need_bytes(t, floats * 4, what)
}

/// Weight operand checks shared by both launchers: `rows_w` weight rows of `k` elements.
/// Returns (q/bf16 pointer, d pointer or null, `"bf16"` / `"w8"`).
fn weight_args(
    w: &LaneWmmaWeight<'_>,
    rows_w: usize,
    k: usize,
) -> HipResult<(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    &'static str,
)> {
    let elems = rows_w
        .checked_mul(k)
        .ok_or_else(|| err("weight extent overflow"))?;
    match w {
        LaneWmmaWeight::Bf16(t) => {
            need_bytes(t, elems * 2, "bf16 weight")?;
            Ok((t.buf.as_ptr() as *const _, std::ptr::null(), "bf16"))
        }
        LaneWmmaWeight::W8 { q, d } => {
            need_bytes(q, elems, "w8 q")?;
            need_bytes(d, elems / 32 * 2, "w8 d")?;
            Ok((q.buf.as_ptr() as *const _, d.buf.as_ptr() as *const _, "w8"))
        }
    }
}

/// A1: `y[rows, m] = epi(x[rows, k] . W[m, k]^T)`; `epi` is `hc_act(scale)` when `hc_act` is
/// `Some(scale)`, else identity. `x` is F32 (BF16-exact values need only `terms == 1`); `terms`
/// is the number of BF16 split terms (1..=3). `k % (128 * split_k) == 0`.
///
/// `scratch` must cover `split_k * rows * m` partial floats and one counter per (column block,
/// token block) when `split_k > 1` (checked). Product callers use [`Gpu::lane_wmma_gemm_fixed`].
#[allow(clippy::too_many_arguments)]
pub fn lane_wmma_gemm(
    gpu: &mut Gpu,
    w: &LaneWmmaWeight<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    rows: usize,
    terms: usize,
    hc_act: Option<f32>,
    cfg: LaneWmmaCfg,
    scratch: &LaneWmmaScratch,
) -> HipResult<LaneWmmaLaunch> {
    if m == 0 || rows == 0 {
        return Err(err("m and rows must be non-zero"));
    }
    if k == 0 || k % 128 != 0 {
        return Err(err(format!("k {k} must be a non-zero multiple of 128")));
    }
    if !(1..=3).contains(&terms) {
        return Err(err(format!("terms {terms} not in 1..=3")));
    }
    let nn = match cfg.nn {
        1 | 2 => cfg.nn,
        other => return Err(err(format!("nn {other} not 1|2"))),
    };
    let split_k = if cfg.split_k == 0 {
        lane_wmma_auto_split_k(m, k, rows, nn)
    } else {
        cfg.split_k
    };
    if (k / 128) % split_k != 0 {
        return Err(err(format!("k {k} % (128 * split_k {split_k}) != 0")));
    }
    let mt = rows.div_ceil(16).min(4);
    let grid_x = m.div_ceil(16 * nn);
    let grid_z = rows.div_ceil(64);
    let (w_ptr, d_ptr, wname) = weight_args(w, m, k)?;
    need_f32(x, rows * k, "x")?;
    need_f32(y, rows * m, "y")?;
    if split_k > 1 {
        need_f32(&scratch.partial, split_k * rows * m, "scratch.partial")?;
        need_bytes(&scratch.counters, grid_x * grid_z * 4, "scratch.counters")?;
    }
    let grid = [
        u32_of(grid_x, "grid x")?,
        u32_of(split_k, "grid y")?,
        u32_of(grid_z, "grid z")?,
    ];
    let block = [128u32, 1, 1];
    let kernel = format!("lwmma_gemm_{wname}_t{terms}_m{mt}_n{nn}");
    let (scale, act) = match hc_act {
        Some(s) => (s, 1),
        None => (0.0f32, 0),
    };
    gpu.ensure_kernel_public(LANE_WMMA_MODULE, LANE_WMMA_SRC, &kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(w_ptr);
    args.push_ptr(d_ptr);
    args.push_ptr(x.buf.as_ptr());
    args.push_ptr(y.buf.as_ptr());
    args.push_ptr(scratch.partial.buf.as_ptr());
    args.push_ptr(scratch.counters.buf.as_ptr());
    args.push_i32(i32_of(m, "m")?);
    args.push_i32(i32_of(k, "k")?);
    args.push_i32(i32_of(rows, "rows")?);
    args.push_i32(i32_of(split_k, "split_k")?);
    args.push_f32(scale);
    args.push_i32(act);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        &kernel,
        grid,
        block,
        0,
        args.as_mut_slice(),
        ReplayLaunchBindings::NONE,
    )?;
    Ok(LaneWmmaLaunch {
        kernel,
        grid,
        block,
    })
}

/// A2: HC up projection `W [4 * hidden, low_rank]` of `low [rows, low_rank]` with the branch mix
/// fused: `mixed [rows, hidden]` (the `hyper_read_up_fused_f32_sx_t1` epilogue). `normalized` is
/// F32 `[rows, 4 * hidden]`. `low_rank % 32 == 0`, `low_rank <= 512`, `hidden % 16 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn lane_wmma_read_up(
    gpu: &mut Gpu,
    w: &LaneWmmaWeight<'_>,
    low: &GpuTensor,
    normalized: &GpuTensor,
    mixed: &GpuTensor,
    rows: usize,
    hidden: usize,
    low_rank: usize,
    terms: usize,
) -> HipResult<LaneWmmaLaunch> {
    if rows == 0 || hidden == 0 || hidden % 16 != 0 {
        return Err(err(format!(
            "rows {rows} must be non-zero and hidden {hidden} a non-zero multiple of 16"
        )));
    }
    if low_rank == 0 || low_rank % 32 != 0 || low_rank > 512 {
        return Err(err(format!(
            "low_rank {low_rank} must be a multiple of 32 in 32..=512"
        )));
    }
    if !(1..=3).contains(&terms) {
        return Err(err(format!("terms {terms} not in 1..=3")));
    }
    let wide = 4 * hidden;
    let (w_ptr, d_ptr, wname) = weight_args(w, wide, low_rank)?;
    need_f32(low, rows * low_rank, "low")?;
    need_f32(normalized, rows * wide, "normalized")?;
    need_f32(mixed, rows * hidden, "mixed")?;
    let tt = lane_wmma_read_up_tiles(rows);
    let grid = [
        u32_of(hidden / 16, "grid x")?,
        u32_of(rows.div_ceil(16 * tt), "grid y")?,
        1,
    ];
    let block = [256u32, 1, 1];
    let kernel = format!("lwmma_read_up_{wname}_t{terms}_g{tt}");
    gpu.ensure_kernel_public(LANE_WMMA_MODULE, LANE_WMMA_SRC, &kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(w_ptr);
    args.push_ptr(d_ptr);
    args.push_ptr(low.buf.as_ptr());
    args.push_ptr(normalized.buf.as_ptr());
    args.push_ptr(mixed.buf.as_ptr());
    args.push_i32(i32_of(hidden, "hidden")?);
    args.push_i32(i32_of(low_rank, "low_rank")?);
    args.push_i32(i32_of(rows, "rows")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        &kernel,
        grid,
        block,
        0,
        args.as_mut_slice(),
        ReplayLaunchBindings::NONE,
    )?;
    Ok(LaneWmmaLaunch {
        kernel,
        grid,
        block,
    })
}

impl Gpu {
    /// [`lane_wmma_gemm`] with the product configuration ([`LaneWmmaCfg::default`]) and the `Gpu`-owned
    /// split-K scratch: allocated lazily on first use at the fixed
    /// [`LANE_WMMA_SCRATCH_PARTIAL_FLOATS`] / [`LANE_WMMA_SCRATCH_COUNTERS`] size, zeroed once, never
    /// grown (so a recorded launch's pointers never go stale). Too large a `split_k * rows * m`
    /// is an error from [`lane_wmma_gemm`], not a regrow.
    #[allow(clippy::too_many_arguments)]
    pub fn lane_wmma_gemm_fixed(
        &mut self,
        w: &LaneWmmaWeight<'_>,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        rows: usize,
        terms: usize,
        hc_act: Option<f32>,
    ) -> HipResult<LaneWmmaLaunch> {
        // Take the scratch out of `self` so `lane_wmma_gemm` can borrow `self` mutably, and put it
        // back on every path (including a launch error): the fixed buffer is never dropped.
        let scratch = match self.scratch.lane_wmma_scratch.take() {
            Some(s) => s,
            None => LaneWmmaScratch::new(
                self,
                LANE_WMMA_SCRATCH_PARTIAL_FLOATS,
                LANE_WMMA_SCRATCH_COUNTERS,
            )?,
        };
        let out = lane_wmma_gemm(
            self,
            w,
            x,
            y,
            m,
            k,
            rows,
            terms,
            hc_act,
            LaneWmmaCfg::default(),
            &scratch,
        );
        self.scratch.lane_wmma_scratch = Some(scratch);
        out
    }
}

/// How a BF16 multirow call over `total` rows of an EP prefill wave is split into per-lane calls
/// (`None`: one call over all rows, the ordinary route). `table` is [`Gpu::qwen4_wave_lane_rows`],
/// the wave's lanes' rows in order. A lane's own singleton takes the 16..=64-row lane WMMA arm
/// (`wmma(rows)`, keyed on the call's REAL rows and not bitwise the SIMT arm), so every such lane
/// is its own call; a lane of at most 15 rows is its own call too (it keeps its singleton's
/// few-row arms), while consecutive lanes wider than the few-row window (64 rows) stay one call
/// (the SIMT multirow arm is row-count invariant there, and one wide call is the wave's gain).
/// Split only when the table is the call's (`sum == total`) and some lane takes the lane WMMA arm.
pub fn wave_lane_groups(
    table: &[usize],
    total: usize,
    wmma: impl Fn(usize) -> bool,
) -> Option<Vec<usize>> {
    if table.is_empty()
        || table.iter().sum::<usize>() != total
        || !table.iter().any(|&rows| wmma(rows))
    {
        return None;
    }
    let wide = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
    let mut groups = Vec::with_capacity(table.len());
    let mut run = 0usize;
    for &rows in table {
        if rows > wide {
            run += rows;
        } else {
            if run > 0 {
                groups.push(run);
                run = 0;
            }
            groups.push(rows);
        }
    }
    if run > 0 {
        groups.push(run);
    }
    Some(groups)
}

impl Gpu {
    /// [`wave_lane_groups`] for a call of `rows` rows under this device's lane WMMA admission.
    pub fn wave_lane_groups(&self, rows: usize) -> Option<Vec<usize>> {
        wave_lane_groups(&self.qwen4_wave_lane_rows, rows, |r| {
            self.lane_wmma_applies(r)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wmma(rows: usize) -> bool {
        (crate::gemv::LANE_WMMA_MIN_ROWS..=64).contains(&rows)
    }

    #[test]
    fn wave_lane_groups_split_only_when_a_lane_takes_the_lane_wmma_arm() {
        // The story C8 burst: eight 60-row lanes, each its own call.
        assert_eq!(wave_lane_groups(&[60; 8], 480, wmma), Some(vec![60; 8]));
        // Small lanes split, consecutive wide lanes stay one call.
        assert_eq!(
            wave_lane_groups(&[60, 60, 149, 149, 256, 511], 1185, wmma),
            Some(vec![60, 60, 149 + 149 + 256 + 511])
        );
        // A small lane between wide lanes cuts the wide run.
        assert_eq!(
            wave_lane_groups(&[149, 60, 149, 149], 507, wmma),
            Some(vec![149, 60, 298])
        );
        // Lanes of 9..=15 rows keep their few-row arms: their own calls, but only with a WMMA lane.
        assert_eq!(wave_lane_groups(&[12, 12], 24, wmma), None);
        assert_eq!(wave_lane_groups(&[12, 40], 52, wmma), Some(vec![12, 40]));
        // No small lane, no table, or a table that is not this call's rows: one call.
        assert_eq!(wave_lane_groups(&[149, 256], 405, wmma), None);
        assert_eq!(wave_lane_groups(&[], 0, wmma), None);
        assert_eq!(wave_lane_groups(&[60, 60], 480, wmma), None);
        // The groups always cover the call exactly.
        for table in [vec![60; 8], vec![9, 16, 64, 65, 200], vec![65, 64, 65]] {
            let total: usize = table.iter().sum();
            if let Some(groups) = wave_lane_groups(&table, total, wmma) {
                assert_eq!(groups.iter().sum::<usize>(), total);
            }
        }
    }

    #[test]
    fn product_cfg_is_one_ntile_auto_split() {
        let cfg = LaneWmmaCfg::default();
        assert_eq!((cfg.nn, cfg.split_k), (1, 0));
    }

    #[test]
    fn read_up_tiles_cover_rows() {
        assert_eq!(lane_wmma_read_up_tiles(1), 1);
        assert_eq!(lane_wmma_read_up_tiles(16), 1);
        assert_eq!(lane_wmma_read_up_tiles(17), 2);
        assert_eq!(lane_wmma_read_up_tiles(32), 2);
        assert_eq!(lane_wmma_read_up_tiles(33), 4);
        assert_eq!(lane_wmma_read_up_tiles(64), 4);
    }

    #[test]
    fn auto_split_k_rule() {
        // HC down: M = 320, K = 10240 (80 chunks), nn = 2 -> 10 column blocks.
        assert_eq!(lane_wmma_auto_split_k(320, 10240, 1, 2), 20);
        assert_eq!(lane_wmma_auto_split_k(320, 10240, 16, 2), 20);
        assert_eq!(lane_wmma_auto_split_k(320, 10240, 64, 2), 20);
        // 4096 rows: 64 token blocks * 10 = 640 > 320 even at s = 1 -> 1.
        assert_eq!(lane_wmma_auto_split_k(320, 10240, 4096, 2), 1);
        // Write gate: M = 4, nn = 1: 1 block, 80 chunks -> largest divisor of 80 <= 320 = 80.
        assert_eq!(lane_wmma_auto_split_k(4, 10240, 8, 1), 80);
        // Shared gate/up: K = 2560 (20 chunks), M = 384, nn = 2 -> 12 blocks: 12 * 20 = 240.
        assert_eq!(lane_wmma_auto_split_k(384, 2560, 1, 2), 20);
        // Shared down: K = 384 (3 chunks), M = 2560, nn = 2 -> 80 blocks: 80 * 3 = 240.
        assert_eq!(lane_wmma_auto_split_k(2560, 384, 8, 2), 3);
        // Never below 1.
        assert_eq!(lane_wmma_auto_split_k(10240, 128, 4096, 2), 1);
    }

    /// The fixed `Gpu` scratch covers every product call: for 1..=64 rows (one token block)
    /// and the product `nn = 1`, `split_k * rows * m <= 320 * 16 * 64` and the block count
    /// `ceil(m / 16) * split_k <= 320` whenever the auto split-K picks `split_k > 1`.
    #[test]
    fn fixed_scratch_covers_product_shapes() {
        // (m, k) of the three product sites: HC down, shared/dense widths, wide-ish gates.
        for (m, k) in [
            (320usize, 10240usize),
            (4, 10240),
            (384, 2560),
            (2560, 384),
            (64, 4096),
        ] {
            for rows in 16..=64usize {
                let s = lane_wmma_auto_split_k(m, k, rows, 1);
                let blocks = m.div_ceil(16) * rows.div_ceil(64);
                if s > 1 {
                    assert!(
                        s * rows * m <= LANE_WMMA_SCRATCH_PARTIAL_FLOATS,
                        "m {m} k {k} rows {rows}"
                    );
                    assert!(
                        blocks <= LANE_WMMA_SCRATCH_COUNTERS,
                        "m {m} k {k} rows {rows}"
                    );
                }
            }
        }
    }
}
