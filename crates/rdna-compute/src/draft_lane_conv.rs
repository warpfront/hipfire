// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `Gpu` launcher for the lane-batched DFlash2 dynamic causal convolution of
//! the continuous-batching draft (exact gfx1201).
//!
//! `draft_forward_lanes` keeps every lane's block rows in ONE shared row
//! scratch (lane `i` owns rows `[row0_i, row0_i + rows_i)`); the convolution is
//! causal across the rows of one lane only, so the per-lane
//! [`Gpu::dynamic_causal_conv_f32`] launch per stage collapses to one
//! [`Gpu::dynamic_causal_conv_f32_lanes`] launch whose lane table rides in the
//! kernarg segment by value (`kernels/src/dynamic_causal_conv_lanes.hip`;
//! byte-identical to the per-lane launches, every causal guard uses the
//! lane-local row).
//!
//! Self-contained via `include_str!` (no shared-registry edits), launched
//! through `launch_maybe_blob` so the kernarg bytes are retained under capture.

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;

/// Kernel source for [`Gpu::dynamic_causal_conv_f32_lanes`].
pub const DYNAMIC_CAUSAL_CONV_LANES_SRC: &str =
    include_str!("../../../kernels/src/dynamic_causal_conv_lanes.hip");
/// Module key and device symbol.
pub const DYNAMIC_CAUSAL_CONV_LANES_SYMBOL: &str = "dynamic_causal_conv_f32_lanes";
/// Lanes per launch (`DRAFT_CONV_LANES_MAX` in the kernel).
pub const DRAFT_CONV_LANES_MAX: usize = 16;

/// One lane of [`Gpu::dynamic_causal_conv_f32_lanes`]: rows
/// `[row0, row0 + rows)` of the shared row scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftConvLane {
    pub row0: usize,
    pub rows: usize,
}

/// `#[repr(C)]` mirror of `struct DraftConvTable` (the by-value kernel
/// argument): 16 x i32 row0, 16 x i32 rows (128 B, 4-aligned). Unused slots
/// are zero (`rows == 0` makes a lane inert).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftConvTable {
    pub row0: [i32; DRAFT_CONV_LANES_MAX],
    pub rows: [i32; DRAFT_CONV_LANES_MAX],
}

const _: () = assert!(std::mem::size_of::<DraftConvTable>() == 128);
const _: () = assert!(std::mem::align_of::<DraftConvTable>() == 4);

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(
        0,
        &format!("dynamic_causal_conv_f32_lanes: {}", msg.as_ref()),
    )
}

impl Gpu {
    /// May the lane-batched draft convolution run now? Exact gfx1201; no graph
    /// capture, replay recording or G0 observation (the by-value table is only
    /// retained by the blob launch path, and product replay admission of this
    /// kernel is not claimed).
    pub fn draft_lane_conv_admitted(&self) -> bool {
        self.arch_caps.is_gfx1201()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.replay.is_g0_observing()
    }

    /// DFlash2 dynamic causal convolution of `lanes.len()` (1..=16) lanes in
    /// ONE launch. `input`/`output` are the FULL shared `[rows, hidden]`
    /// scratch buffers, `dynamic` the full `[rows, dynamic_row_stride]`
    /// buffer, `base` the lane-independent `[kernel_size, hidden]` weight
    /// (pass the phase-1 subview for the finish convolution). Per lane this
    /// equals [`Gpu::dynamic_causal_conv_f32`] over that lane's row
    /// sub-views (`rows = lane.rows`, same `hidden`/`kernel_size`/
    /// `group_size`/stride/offset), bit for bit.
    ///
    /// Lanes must have `rows > 0`, pairwise disjoint row ranges, and lie
    /// inside the buffers; `output` must not overlap any input operand.
    #[allow(clippy::too_many_arguments)]
    pub fn dynamic_causal_conv_f32_lanes(
        &mut self,
        input: &GpuTensor,
        base: &GpuTensor,
        dynamic: &GpuTensor,
        output: &GpuTensor,
        lanes: &[DraftConvLane],
        hidden: usize,
        kernel_size: usize,
        group_size: usize,
        dynamic_row_stride: usize,
        dynamic_offset: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if lanes.is_empty() || lanes.len() > DRAFT_CONV_LANES_MAX {
            return Err(err(format!(
                "{} lanes outside 1..={DRAFT_CONV_LANES_MAX}",
                lanes.len()
            )));
        }
        if hidden == 0 {
            return Err(err("hidden must be > 0"));
        }
        if kernel_size == 0 {
            return Err(err("kernel_size must be > 0"));
        }
        if group_size == 0 {
            return Err(err("group_size must be > 0"));
        }
        if hidden % group_size != 0 {
            return Err(err(format!(
                "hidden {hidden} must be divisible by group_size {group_size}"
            )));
        }
        let groups = hidden / group_size;
        if dynamic_row_stride == 0 {
            return Err(err("dynamic_row_stride must be > 0"));
        }
        let window = kernel_size
            .checked_mul(groups)
            .ok_or_else(|| err("kernel_size*groups overflow"))?;
        if dynamic_offset
            .checked_add(window)
            .is_none_or(|e| e > dynamic_row_stride)
        {
            return Err(err(format!(
                "dynamic_row_stride {dynamic_row_stride} must be >= dynamic_offset {dynamic_offset} + kernel_size*groups {window} (window overruns stride)"
            )));
        }

        // Lane table: positive rows, checked ends, pairwise disjoint.
        let mut table = DraftConvTable {
            row0: [0; DRAFT_CONV_LANES_MAX],
            rows: [0; DRAFT_CONV_LANES_MAX],
        };
        let mut max_end = 0usize;
        let mut max_rows = 0usize;
        for (i, ln) in lanes.iter().enumerate() {
            if ln.rows == 0 {
                return Err(err(format!("lane {i}: rows must be > 0")));
            }
            let end = ln
                .row0
                .checked_add(ln.rows)
                .ok_or_else(|| err(format!("lane {i}: row0+rows overflow")))?;
            max_end = max_end.max(end);
            max_rows = max_rows.max(ln.rows);
            if ln.row0 > i32::MAX as usize || ln.rows > i32::MAX as usize {
                return Err(err(format!("lane {i}: row0/rows exceed i32::MAX")));
            }
            table.row0[i] = ln.row0 as i32;
            table.rows[i] = ln.rows as i32;
        }
        for (i, a) in lanes.iter().enumerate() {
            for (j, b) in lanes.iter().enumerate().skip(i + 1) {
                if a.row0 < b.row0 + b.rows && b.row0 < a.row0 + a.rows {
                    return Err(err(format!(
                        "lanes {i} [{}, {}) and {j} [{}, {}) overlap",
                        a.row0,
                        a.row0 + a.rows,
                        b.row0,
                        b.row0 + b.rows
                    )));
                }
            }
        }

        // i32 limits for kernargs / lane-local kernel index arithmetic. The
        // kernel's per-lane `idx`, `r*stride` and `r*hidden` are 32-bit;
        // only the lane rebase is 64-bit.
        let local_total = max_rows
            .checked_mul(hidden)
            .ok_or_else(|| err("rows*hidden overflow"))?;
        let local_dyn = max_rows
            .checked_mul(dynamic_row_stride)
            .and_then(|n| n.checked_add(dynamic_offset))
            .and_then(|n| n.checked_add(window))
            .ok_or_else(|| err("rows*dynamic_row_stride overflow"))?;
        if local_total > i32::MAX as usize
            || local_dyn > i32::MAX as usize
            || hidden > i32::MAX as usize
            || kernel_size > i32::MAX as usize
            || groups > i32::MAX as usize
            || group_size > i32::MAX as usize
            || dynamic_row_stride > i32::MAX as usize
            || dynamic_offset > i32::MAX as usize
        {
            return Err(err(
                "rows/hidden/kernel_size/groups/group_size/stride/offset exceed i32::MAX",
            ));
        }

        // dtype checks: F32 only (caller's contract).
        for (name, t) in [
            ("input", input),
            ("base", base),
            ("dynamic", dynamic),
            ("output", output),
        ] {
            if t.dtype != DType::F32 {
                return Err(err(format!("{name} dtype must be F32 (got {:?})", t.dtype)));
            }
        }

        // Buffer size checks (bytes), covering the highest lane end.
        let f32 = DType::F32.size();
        let input_need = max_end
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(f32))
            .ok_or_else(|| err("rows*hidden overflow"))?;
        let base_need = kernel_size
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(f32))
            .ok_or_else(|| err("kernel_size*hidden overflow"))?;
        let dynamic_need = max_end
            .checked_mul(dynamic_row_stride)
            .and_then(|n| n.checked_mul(f32))
            .ok_or_else(|| err("rows*dynamic_row_stride overflow"))?;
        let output_need = input_need;
        if input.buf.size() < input_need {
            return Err(err(format!(
                "input buffer too small (have {} need {input_need} for [{max_end},{hidden}] F32)",
                input.buf.size()
            )));
        }
        if base.buf.size() < base_need {
            return Err(err(format!(
                "base buffer too small (have {} need {base_need} for [{kernel_size},{hidden}] F32)",
                base.buf.size()
            )));
        }
        if dynamic.buf.size() < dynamic_need {
            return Err(err(format!(
                "dynamic buffer too small (have {} need {dynamic_need} for [{max_end},{dynamic_row_stride}] F32 stride {dynamic_row_stride} offset {dynamic_offset})",
                dynamic.buf.size()
            )));
        }
        if output.buf.size() < output_need {
            return Err(err(format!(
                "output buffer too small (have {} need {output_need} for [{max_end},{hidden}] F32)",
                output.buf.size()
            )));
        }

        let input_ptr = input.buf.as_ptr();
        let base_ptr = base.buf.as_ptr();
        let dynamic_ptr = dynamic.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();

        // Distinct output: no byte range overlap with any input operand.
        let overlaps = |a: *const c_void, an: usize, b: *const c_void, bn: usize| {
            let (a, b) = (a as usize, b as usize);
            a < b.saturating_add(bn) && b < a.saturating_add(an)
        };
        for (name, p, n) in [
            ("input", input_ptr, input_need),
            ("base", base_ptr, base_need),
            ("dynamic", dynamic_ptr, dynamic_need),
        ] {
            if overlaps(output_ptr, output_need, p, n) {
                return Err(err(format!(
                    "output overlaps {name} (no in-place aliasing)"
                )));
            }
        }

        self.ensure_kernel(
            DYNAMIC_CAUSAL_CONV_LANES_SYMBOL,
            DYNAMIC_CAUSAL_CONV_LANES_SRC,
            DYNAMIC_CAUSAL_CONV_LANES_SYMBOL,
        )?;

        let hidden_i32 = hidden as i32;
        let kernel_size_i32 = kernel_size as i32;
        let groups_i32 = groups as i32;
        let group_size_i32 = group_size as i32;
        let stride_i32 = dynamic_row_stride as i32;
        let offset_i32 = dynamic_offset as i32;
        let block = 256u32;
        let grid = local_total.div_ceil(block as usize) as u32;
        let table_ref = &table;
        let mut params: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &base_ptr as *const _ as *mut c_void,
            &dynamic_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            table_ref as *const _ as *mut c_void,
            &hidden_i32 as *const _ as *mut c_void,
            &kernel_size_i32 as *const _ as *mut c_void,
            &groups_i32 as *const _ as *mut c_void,
            &group_size_i32 as *const _ as *mut c_void,
            &stride_i32 as *const _ as *mut c_void,
            &offset_i32 as *const _ as *mut c_void,
        ];
        let live_rows: usize = lanes.iter().map(|l| l.rows).sum();
        let bytes = live_rows * hidden * f32 * 2 + base_need + live_rows * dynamic_row_stride * f32;
        let timer = crate::profile::begin_timer(
            &self.hip,
            "draft_lane_conv",
            DYNAMIC_CAUSAL_CONV_LANES_SYMBOL,
            bytes,
        );
        let result = self.launch_maybe_blob(
            DYNAMIC_CAUSAL_CONV_LANES_SYMBOL,
            [grid, 1, lanes.len() as u32],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(input_ptr);
                b.push_ptr(base_ptr);
                b.push_ptr(dynamic_ptr);
                b.push_ptr(output_ptr);
                // 128 B, 4-aligned: starts at offset 32, right after the four
                // pointers.
                for v in table_ref.row0.iter().chain(&table_ref.rows) {
                    b.push_i32(*v);
                }
                b.push_i32(hidden_i32);
                b.push_i32(kernel_size_i32);
                b.push_i32(groups_i32);
                b.push_i32(group_size_i32);
                b.push_i32(stride_i32);
                b.push_i32(offset_i32);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
