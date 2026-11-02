// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launchers for the lane-batched ctx fill / fc rows of the DFlash
//! continuous-batching draft (exact gfx1201).
//!
//! The per-lane fc projection and per-layer K/V ctx fill project a handful of
//! rows per lane through ONE shared weight. Packing every lane's rows into one
//! workspace and running each singleton arithmetic class once per weight
//! removes the per-lane launch loops:
//!
//! * [`Gpu::draft_rows_gather_f32`] / [`Gpu::draft_rows_scatter_f32`]: copy
//!   row spans (f32 *bits*, plus optionally each row's i32 position) between
//!   scattered ring/scratch locations and one packed row buffer
//!   (`kernels/src/draft_rows_pack.hip`). The span table rides in the kernarg
//!   segment by value (no device upload, no synchronize).
//! * [`Gpu::gemv_mq4g256v2_multirow_r2_lanes`]: the singleton
//!   `gemv_mq4g256v2_multirow_r2` GEMV over `batch` independent F32 rotated
//!   rows with `grid.z = segment` (`kernels/src/gemv_mq4g256v2_multirow_r2_lanes.hip`).
//!   Segment `z` is byte-identical to a singleton launch on `rotated[z]`. It
//!   is the arithmetic class of ORIGINAL 1-row segments only; it is NOT
//!   interchangeable with the F16 overwrite WMMA class and must never be fed
//!   rows that the singleton code would send through `gemm_dispatch` (n >= 2).
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source of the gather/scatter pair.
pub const DRAFT_ROWS_PACK_SRC: &str = include_str!("../../../kernels/src/draft_rows_pack.hip");
/// Module key of [`DRAFT_ROWS_PACK_SRC`].
pub const DRAFT_ROWS_PACK_MODULE: &str = "draft_rows_pack";
/// Device symbol of [`Gpu::draft_rows_gather_f32`].
pub const DRAFT_ROWS_GATHER_SYMBOL: &str = "draft_rows_gather_f32";
/// Device symbol of [`Gpu::draft_rows_scatter_f32`].
pub const DRAFT_ROWS_SCATTER_SYMBOL: &str = "draft_rows_scatter_f32";
/// Row spans per launch (`DRAFT_ROW_SPANS_MAX` in the kernel).
pub const DRAFT_ROW_SPANS_MAX: usize = 32;

/// Kernel source of the lane-batched R2 GEMV twin: the SAME weight-cache source
/// prefix as `GEMV_MQ4G256V2_MULTIROW_SRC` (`kernels.rs`), followed by the twin.
/// A separate source (not an edit of `gemv_mq4g256v2_multirow.hip`) because that
/// source's sha256 is pinned by the accepted PM decode twin.
pub const GEMV_MQ4G256V2_MULTIROW_R2_LANES_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_multirow_r2_lanes.hip")
);
/// Module key of [`GEMV_MQ4G256V2_MULTIROW_R2_LANES_SRC`]. Deliberately NOT a
/// PM-decode module name, so the PM funnel never intercepts the twin.
pub const GEMV_MQ4G256V2_MULTIROW_R2_LANES_MODULE: &str = "gemv_mq4g256v2_multirow_r2_lanes_mq4v2";
/// Device symbol of [`Gpu::gemv_mq4g256v2_multirow_r2_lanes`].
pub const GEMV_MQ4G256V2_MULTIROW_R2_LANES_SYMBOL: &str = "gemv_mq4g256v2_multirow_r2_lanes";
/// Segments per [`Gpu::gemv_mq4g256v2_multirow_r2_lanes`] launch: the draft's
/// packed workspace holds at most 72 rows.
pub const DRAFT_R2_LANES_BATCH_MAX: usize = 72;

/// One row span of a gather/scatter launch. Pointer fields are device
/// addresses ALREADY advanced to the segment start; a segment's rows are
/// contiguous (`width` f32 each) on the span side. Fields a given operation
/// does not use are ignored by it (callers zero them by convention).
///
/// * gather: `src` = source rows, `positions` = source i32 positions (one per
///   row; only when a positions output is passed).
/// * scatter: `dst0` = destination of `packed0` rows, `dst1` = destination of
///   `packed1` rows (only when `packed1` is passed).
///
/// `packed_row0` / `rows` select the span's rows in the packed buffer(s).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftRowSpan {
    pub src: u64,
    pub dst0: u64,
    pub dst1: u64,
    pub positions: u64,
    pub packed_row0: i32,
    pub rows: i32,
} // 40 bytes, align 8

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<DraftRowSpan>() == 40);
    assert!(align_of::<DraftRowSpan>() == 8);
    assert!(offset_of!(DraftRowSpan, src) == 0);
    assert!(offset_of!(DraftRowSpan, dst0) == 8);
    assert!(offset_of!(DraftRowSpan, dst1) == 16);
    assert!(offset_of!(DraftRowSpan, positions) == 24);
    assert!(offset_of!(DraftRowSpan, packed_row0) == 32);
    assert!(offset_of!(DraftRowSpan, rows) == 36);
};

/// `#[repr(C)]` mirror of `struct DraftRowSpanTable` (the by-value kernel
/// argument): 32 spans, 1280 B, 8-aligned. Unused slots are zero.
#[repr(C)]
#[derive(Clone, Copy)]
struct DraftRowSpanTable {
    span: [DraftRowSpan; DRAFT_ROW_SPANS_MAX],
}

const _: () = assert!(std::mem::size_of::<DraftRowSpanTable>() == 1280);
const _: () = assert!(std::mem::align_of::<DraftRowSpanTable>() == 8);

fn err(what: &str, msg: impl AsRef<str>) -> HipError {
    HipError::new(0, &format!("{what}: {}", msg.as_ref()))
}

/// Field-wise table serialization (explicit layout, no struct-byte copy):
/// 4 x u64 + 2 x i32 per span = 40 B, spans contiguous, 8-aligned start.
fn push_table(b: &mut KernargBlob, table: &DraftRowSpanTable) {
    for s in &table.span {
        b.push_u64(s.src);
        b.push_u64(s.dst0);
        b.push_u64(s.dst1);
        b.push_u64(s.positions);
        b.push_i32(s.packed_row0);
        b.push_i32(s.rows);
    }
}

/// Half-open byte interval overlap.
fn overlaps(a: u64, an: u64, b: u64, bn: u64) -> bool {
    a < b.saturating_add(bn) && b < a.saturating_add(an)
}

/// Common span validation: span count, positive rows, i32 limits, packed row
/// ranges pairwise disjoint. Returns `(table, max_rows, max_end)`.
fn build_table(
    what: &str,
    spans: &[DraftRowSpan],
    width: usize,
) -> HipResult<(DraftRowSpanTable, usize, usize)> {
    if spans.is_empty() || spans.len() > DRAFT_ROW_SPANS_MAX {
        return Err(err(
            what,
            format!("{} spans outside 1..={DRAFT_ROW_SPANS_MAX}", spans.len()),
        ));
    }
    if width == 0 || width > i32::MAX as usize {
        return Err(err(what, format!("width {width} outside 1..=i32::MAX")));
    }
    let mut table = DraftRowSpanTable {
        span: [DraftRowSpan::default(); DRAFT_ROW_SPANS_MAX],
    };
    let mut max_rows = 0usize;
    let mut max_end = 0usize;
    for (i, s) in spans.iter().enumerate() {
        if s.rows <= 0 {
            return Err(err(what, format!("span {i}: rows {} must be > 0", s.rows)));
        }
        if s.packed_row0 < 0 {
            return Err(err(
                what,
                format!("span {i}: packed_row0 {} < 0", s.packed_row0),
            ));
        }
        let rows = s.rows as usize;
        let end = s.packed_row0 as usize + rows;
        max_rows = max_rows.max(rows);
        max_end = max_end.max(end);
        table.span[i] = *s;
    }
    // The kernel's per-span `idx = row * width + col` is 32-bit.
    if max_rows
        .checked_mul(width)
        .is_none_or(|n| n > i32::MAX as usize)
    {
        return Err(err(what, "max span rows * width exceeds i32::MAX"));
    }
    for (i, a) in spans.iter().enumerate() {
        for (j, b) in spans.iter().enumerate().skip(i + 1) {
            let (a0, a1) = (a.packed_row0 as i64, a.packed_row0 as i64 + a.rows as i64);
            let (b0, b1) = (b.packed_row0 as i64, b.packed_row0 as i64 + b.rows as i64);
            if a0 < b1 && b0 < a1 {
                return Err(err(
                    what,
                    format!(
                        "spans {i} [{a0}, {a1}) and {j} [{b0}, {b1}) overlap in the packed rows"
                    ),
                ));
            }
        }
    }
    max_end
        .checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| err(what, "packed rows * width overflow"))?;
    Ok((table, max_rows, max_end))
}

fn check_f32(what: &str, name: &str, t: &GpuTensor, need_bytes: usize) -> HipResult<()> {
    if t.dtype != DType::F32 {
        return Err(err(
            what,
            format!("{name} dtype must be F32 (got {:?})", t.dtype),
        ));
    }
    if t.buf.size() < need_bytes {
        return Err(err(
            what,
            format!(
                "{name} buffer too small (have {} need {need_bytes} bytes)",
                t.buf.size()
            ),
        ));
    }
    Ok(())
}

impl Gpu {
    /// May the lane-batched ctx/fc row path run now? Exact gfx1201; no graph
    /// capture, replay recording or G0 observation (the by-value span tables
    /// are only retained by the blob launch path, and product replay admission
    /// of these kernels is not claimed).
    pub fn draft_lane_rows_admitted(&self) -> bool {
        self.arch_caps.is_gfx1201()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.replay.is_g0_observing()
    }

    /// Gather `spans.len()` (1..=[`DRAFT_ROW_SPANS_MAX`]) row spans into the
    /// packed F32 buffer `packed` (`[rows, width]`, span `i` filling packed
    /// rows `[packed_row0, packed_row0 + rows)`) in ONE launch: a pure copy of
    /// the f32 bits of `src[r * width ..][..width]` per row. With
    /// `positions = Some(out)` each row's i32 position is also copied exactly
    /// once from `span.positions[r]` to `out[packed_row0 + r]`.
    ///
    /// Spans must have `rows > 0`, pairwise disjoint packed ranges, non-null
    /// 4-byte-aligned `src` (and `positions` when requested) that do not
    /// overlap the destination tensors. Device-side source bounds are the
    /// caller's contract; only the packed side is checkable here.
    pub fn draft_rows_gather_f32(
        &mut self,
        packed: &GpuTensor,
        positions: Option<&GpuTensor>,
        spans: &[DraftRowSpan],
        width: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "draft_rows_gather_f32";
        self.bind_thread()?;
        let (table, max_rows, max_end) = build_table(WHAT, spans, width)?;
        let packed_need = max_end * width * 4;
        check_f32(WHAT, "packed", packed, packed_need)?;
        let pos_need = max_end * 4;
        if let Some(p) = positions {
            if p.buf.size() < pos_need {
                return Err(err(
                    WHAT,
                    format!(
                        "positions buffer too small (have {} need {pos_need} bytes)",
                        p.buf.size()
                    ),
                ));
            }
        }
        let packed_ptr = packed.buf.as_ptr();
        let pos_ptr: *mut c_void = positions.map_or(std::ptr::null_mut(), |p| p.buf.as_ptr());
        let packed_lo = packed_ptr as u64;
        let pos_lo = pos_ptr as u64;
        for (i, s) in spans.iter().enumerate() {
            let src_bytes = s.rows as u64 * width as u64 * 4;
            if s.src == 0 || s.src % 4 != 0 {
                return Err(err(
                    WHAT,
                    format!("span {i}: src must be non-null and 4-byte aligned"),
                ));
            }
            if overlaps(s.src, src_bytes, packed_lo, packed_need as u64) {
                return Err(err(
                    WHAT,
                    format!("span {i}: src overlaps the packed buffer"),
                ));
            }
            if positions.is_some() {
                let pb = s.rows as u64 * 4;
                if s.positions == 0 || s.positions % 4 != 0 {
                    return Err(err(
                        WHAT,
                        format!("span {i}: positions must be non-null and 4-byte aligned"),
                    ));
                }
                if overlaps(s.positions, pb, pos_lo, pos_need as u64)
                    || overlaps(s.positions, pb, packed_lo, packed_need as u64)
                {
                    return Err(err(
                        WHAT,
                        format!("span {i}: positions source overlaps a destination"),
                    ));
                }
            }
        }

        self.ensure_kernel(
            DRAFT_ROWS_PACK_MODULE,
            DRAFT_ROWS_PACK_SRC,
            DRAFT_ROWS_GATHER_SYMBOL,
        )?;
        let width_i32 = width as i32;
        let block = 256u32;
        let grid = (max_rows * width).div_ceil(block as usize) as u32;
        let table_ref = &table;
        let mut params: Vec<*mut c_void> = vec![
            &packed_ptr as *const _ as *mut c_void,
            &pos_ptr as *const _ as *mut c_void,
            table_ref as *const _ as *mut c_void,
            &width_i32 as *const _ as *mut c_void,
        ];
        let live_rows: usize = spans.iter().map(|s| s.rows as usize).sum();
        let bytes = live_rows * width * 4 * 2
            + if positions.is_some() {
                live_rows * 8
            } else {
                0
            };
        let timer = crate::profile::begin_timer(
            &self.hip,
            "draft_lane_rows",
            DRAFT_ROWS_GATHER_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            DRAFT_ROWS_GATHER_SYMBOL,
            [grid, 1, spans.len() as u32],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(packed_ptr);
                b.push_ptr(pos_ptr);
                // 1280 B, 8-aligned: starts at offset 16, right after the two pointers.
                push_table(&mut b, table_ref);
                b.push_i32(width_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Scatter packed F32 rows back to `spans.len()` (1..=[`DRAFT_ROW_SPANS_MAX`])
    /// destinations in ONE launch: pure f32-bit copies of
    /// `packed0[(packed_row0 + r) * width ..][..width]` to `span.dst0[r * width ..]`
    /// and, with `packed1 = Some(..)`, of `packed1` rows to `span.dst1` (the
    /// ctx K and V together). Every span must then carry a non-null `dst1`.
    ///
    /// Spans must have `rows > 0`, pairwise disjoint packed ranges, and
    /// non-null 4-byte-aligned destinations whose byte ranges are pairwise
    /// disjoint (across spans and across `dst0`/`dst1`) and do not overlap the
    /// packed tensors. Device-side destination bounds are the caller's
    /// contract; only the packed side is checkable here.
    pub fn draft_rows_scatter_f32(
        &mut self,
        packed0: &GpuTensor,
        packed1: Option<&GpuTensor>,
        spans: &[DraftRowSpan],
        width: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "draft_rows_scatter_f32";
        self.bind_thread()?;
        let (table, max_rows, max_end) = build_table(WHAT, spans, width)?;
        let packed_need = max_end * width * 4;
        check_f32(WHAT, "packed0", packed0, packed_need)?;
        if let Some(p1) = packed1 {
            check_f32(WHAT, "packed1", p1, packed_need)?;
        }
        let p0_ptr = packed0.buf.as_ptr();
        let p1_ptr: *mut c_void = packed1.map_or(std::ptr::null_mut(), |p| p.buf.as_ptr());
        // Destination intervals (address, bytes, label).
        let mut dsts: Vec<(u64, u64, String)> = Vec::with_capacity(2 * spans.len());
        for (i, s) in spans.iter().enumerate() {
            let bytes = s.rows as u64 * width as u64 * 4;
            if s.dst0 == 0 || s.dst0 % 4 != 0 {
                return Err(err(
                    WHAT,
                    format!("span {i}: dst0 must be non-null and 4-byte aligned"),
                ));
            }
            dsts.push((s.dst0, bytes, format!("span {i} dst0")));
            if packed1.is_some() {
                if s.dst1 == 0 || s.dst1 % 4 != 0 {
                    return Err(err(
                        WHAT,
                        format!("span {i}: dst1 must be non-null and 4-byte aligned when packed1 is given"),
                    ));
                }
                dsts.push((s.dst1, bytes, format!("span {i} dst1")));
            }
        }
        for (i, (a, an, al)) in dsts.iter().enumerate() {
            for (b, bn, bl) in dsts.iter().skip(i + 1) {
                if overlaps(*a, *an, *b, *bn) {
                    return Err(err(WHAT, format!("{al} and {bl} overlap")));
                }
            }
            if overlaps(*a, *an, p0_ptr as u64, packed_need as u64)
                || (packed1.is_some() && overlaps(*a, *an, p1_ptr as u64, packed_need as u64))
            {
                return Err(err(WHAT, format!("{al} overlaps a packed buffer")));
            }
        }

        self.ensure_kernel(
            DRAFT_ROWS_PACK_MODULE,
            DRAFT_ROWS_PACK_SRC,
            DRAFT_ROWS_SCATTER_SYMBOL,
        )?;
        let width_i32 = width as i32;
        let block = 256u32;
        let grid = (max_rows * width).div_ceil(block as usize) as u32;
        let table_ref = &table;
        let mut params: Vec<*mut c_void> = vec![
            &p0_ptr as *const _ as *mut c_void,
            &p1_ptr as *const _ as *mut c_void,
            table_ref as *const _ as *mut c_void,
            &width_i32 as *const _ as *mut c_void,
        ];
        let live_rows: usize = spans.iter().map(|s| s.rows as usize).sum();
        let planes = if packed1.is_some() { 2 } else { 1 };
        let bytes = live_rows * width * 4 * 2 * planes;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "draft_lane_rows",
            DRAFT_ROWS_SCATTER_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            DRAFT_ROWS_SCATTER_SYMBOL,
            [grid, 1, spans.len() as u32],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(p0_ptr);
                b.push_ptr(p1_ptr);
                push_table(&mut b, table_ref);
                b.push_i32(width_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// `y[z] = W . rotated[z]` for `z < batch` (1..=[`DRAFT_R2_LANES_BATCH_MAX`])
    /// in ONE launch (grid `[ceil(m/2), 1, batch]`, block 32): the singleton
    /// `gemv_mq4g256v2_multirow_r2` arithmetic on every segment. `a_raw` is the
    /// MQ4-G256-v2 weight `[m, k]` (136 B per 256-group), `rotated` the
    /// already FWHT-rotated F32 activations `[batch, k]` (NO F16 cast), `y`
    /// the F32 output `[batch, m]`. Segment `z`'s `y` row is bit-identical to
    /// `gemv_mq4g256v2(a_raw, rotated[z], y[z], m, k)` on the exact-gfx1201
    /// route. `k` must be a multiple of 256.
    ///
    /// Compiled from its own module (not a PM-decode module): the twin is the
    /// HIP incumbent arithmetic, independent of `kernel.pm_decode`.
    pub fn gemv_mq4g256v2_multirow_r2_lanes(
        &mut self,
        a_raw: &GpuTensor,
        rotated: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        batch: usize,
    ) -> HipResult<()> {
        const WHAT: &str = "gemv_mq4g256v2_multirow_r2_lanes";
        self.bind_thread()?;
        if batch == 0 || batch > DRAFT_R2_LANES_BATCH_MAX {
            return Err(err(
                WHAT,
                format!("batch {batch} outside 1..={DRAFT_R2_LANES_BATCH_MAX}"),
            ));
        }
        if m == 0 || m > i32::MAX as usize || k == 0 || k > i32::MAX as usize {
            return Err(err(WHAT, format!("m {m} / k {k} outside 1..=i32::MAX")));
        }
        if k % 256 != 0 {
            return Err(err(WHAT, format!("k {k} must be divisible by 256")));
        }
        let groups = k / 256;
        let weight_bytes = m
            .checked_mul(groups)
            .and_then(|n| n.checked_mul(136))
            .ok_or_else(|| err(WHAT, "weight size overflow"))?;
        // The kernel's raw-buffer weight offsets are 32-bit (like the singleton R2).
        if weight_bytes > u32::MAX as usize {
            return Err(err(
                WHAT,
                format!("weight of {weight_bytes} bytes exceeds the 32-bit buffer offset range"),
            ));
        }
        if a_raw.buf.size() < weight_bytes {
            return Err(err(
                WHAT,
                format!(
                    "a_raw too small (have {} need {weight_bytes} bytes for [{m},{k}] MQ4G256V2)",
                    a_raw.buf.size()
                ),
            ));
        }
        let x_need = batch
            .checked_mul(k)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| err(WHAT, "batch*k overflow"))?;
        let y_need = batch
            .checked_mul(m)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| err(WHAT, "batch*m overflow"))?;
        check_f32(WHAT, "rotated", rotated, x_need)?;
        check_f32(WHAT, "y", y, y_need)?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = rotated.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        if overlaps(y_ptr as u64, y_need as u64, x_ptr as u64, x_need as u64)
            || overlaps(
                y_ptr as u64,
                y_need as u64,
                a_ptr as u64,
                weight_bytes as u64,
            )
        {
            return Err(err(
                WHAT,
                "y overlaps the weight or the rotated input (no in-place aliasing)",
            ));
        }

        self.ensure_kernel(
            GEMV_MQ4G256V2_MULTIROW_R2_LANES_MODULE,
            GEMV_MQ4G256V2_MULTIROW_R2_LANES_SRC,
            GEMV_MQ4G256V2_MULTIROW_R2_LANES_SYMBOL,
        )?;
        let m_val = m as i32;
        let k_val = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
        ];
        let grid = ((m as u32) + 1) / 2;
        let bytes = weight_bytes + x_need + y_need;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemv",
            GEMV_MQ4G256V2_MULTIROW_R2_LANES_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            GEMV_MQ4G256V2_MULTIROW_R2_LANES_SYMBOL,
            [grid, 1, batch as u32],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(k_val);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
