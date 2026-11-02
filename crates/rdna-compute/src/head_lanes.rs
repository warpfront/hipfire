// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-batched lm_head of the A3B (`qwen3_5_moe`, hidden 2048) exact VMM batch
//! route, exact gfx1201.
//!
//! * [`Gpu::gemv_mq4g256v2_head_lanes`]: `y[slot_c] = W · x[c]` for up to
//!   [`HEAD_LANES_MAX`] columns in ONE launch that reads each weight row once.
//!   Per column the arithmetic is the singleton `gemv_mq4g256v2_multirow_r2`'s
//!   (`kernels/src/gemv_mq4g256v2_head_lanes.gfx1201.hip` documents the
//!   argument); the lab test `tests/head_lanes.rs` proves every column's bytes
//!   equal the real singleton launch and the PM multi-column twin.
//! * [`Gpu::rmsnorm_f32_rowsplit_rows`]: the singleton decode RMSNorm
//!   (`rmsnorm_f32_rowsplit`, grid `n/256`) over `rows` contiguous rows with
//!   `grid.y = rows` (the kernel already indexes its row by `blockIdx.y`).
//!   Followed by the existing batched `mq_rotate_x` (same kernel as the
//!   singleton rotate, grid folded over rows) it replaces 2N per-row launches.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Most columns of one head launch.
pub const HEAD_LANES_MAX: usize = 8;
/// Contraction width the kernel is compiled for.
pub const HEAD_LANES_K: usize = 2048;
/// Bytes of one weight row (8 groups of 136 B).
pub const HEAD_LANES_ROW_BYTES: usize = 8 * 136;
/// Module key (not a PM-decode module name: the PM funnel never sees it).
pub const HEAD_LANES_MODULE: &str = "gemv_mq4g256v2_head_lanes_gfx1201";
/// Device symbols, indexed by `columns - 1`.
pub const HEAD_LANES_SYMBOLS: [&str; HEAD_LANES_MAX] = [
    "gemv_mq4g256v2_head_lanes_n1",
    "gemv_mq4g256v2_head_lanes_n2",
    "gemv_mq4g256v2_head_lanes_n3",
    "gemv_mq4g256v2_head_lanes_n4",
    "gemv_mq4g256v2_head_lanes_n5",
    "gemv_mq4g256v2_head_lanes_n6",
    "gemv_mq4g256v2_head_lanes_n7",
    "gemv_mq4g256v2_head_lanes_n8",
];
/// Rows covered by one workgroup (`HL_WAVES * HL_R * HL_TILES` in the kernel: 16 x 4 x 2).
pub const HEAD_LANES_ROWS_PER_WG: usize = 16 * 4 * 2;
/// Threads per workgroup (`HL_WAVES * 32`, 16 waves).
pub const HEAD_LANES_BLOCK: u32 = 512;

/// Kernel source: the singleton module's weight-cache prefix, then the kernel.
pub const HEAD_LANES_SRC: &str = concat!(
    "#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n",
    include_str!("../../../kernels/src/gfx12_weight_cache_policy.inc"),
    include_str!("../../../kernels/src/gemv_mq4g256v2_head_lanes.gfx1201.hip")
);

fn head_err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

impl Gpu {
    /// Whether [`Self::gemv_mq4g256v2_head_lanes`] is admitted: exact gfx1201,
    /// `k == 2048`, `1..=8` columns, 32-bit weight byte offsets. (The weight's
    /// dtype / AWQ state is the caller's to check.)
    pub fn head_lanes_admitted(&self, m: usize, k: usize, columns: usize) -> bool {
        self.arch_caps.is_gfx1201()
            && k == HEAD_LANES_K
            && (1..=HEAD_LANES_MAX).contains(&columns)
            && m > 0
            && m <= i32::MAX as usize
            && m.checked_mul(HEAD_LANES_ROW_BYTES)
                .is_some_and(|b| b <= u32::MAX as usize)
    }

    /// `y[slots[c] * m + i] = (W · x[c])[i]` for `c < slots.len()` (1..=8
    /// columns) in one launch. `a_raw` is the MQ4-G256-v2 weight `[m, 2048]`
    /// (136 B per 256-group), `x` the already FWHT-rotated f32 columns
    /// `[columns, 2048]` (row `c` at `x + c * 2048`), `y` a buffer of f32 rows
    /// of length `m` addressed by slot. Column `c`'s `y` row is
    /// byte-identical to the singleton `gemv_mq4g256v2_multirow_r2` launch on
    /// `x[c]`. Slots must be distinct.
    pub fn gemv_mq4g256v2_head_lanes(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        slots: &[usize],
    ) -> HipResult<()> {
        const WHAT: &str = "gemv_mq4g256v2_head_lanes";
        let columns = slots.len();
        if !self.head_lanes_admitted(m, HEAD_LANES_K, columns) {
            return Err(head_err(format!(
                "{WHAT}: not admitted (needs exact gfx1201, k == 2048, 1..=8 columns, m {m} with 32-bit weight offsets; {columns} columns)"
            )));
        }
        let weight_bytes = m * HEAD_LANES_ROW_BYTES;
        if a_raw.buf.size() < weight_bytes {
            return Err(head_err(format!(
                "{WHAT}: weight holds {} B, needs {weight_bytes} B",
                a_raw.buf.size()
            )));
        }
        let x_need = columns * HEAD_LANES_K * 4;
        if x.buf.size() < x_need {
            return Err(head_err(format!(
                "{WHAT}: x holds {} B, needs {x_need} B",
                x.buf.size()
            )));
        }
        let max_slot = *slots.iter().max().expect("columns >= 1");
        let y_need = (max_slot + 1) * m * 4;
        if y.buf.size() < y_need {
            return Err(head_err(format!(
                "{WHAT}: y holds {} B, needs {y_need} B for slot {max_slot}",
                y.buf.size()
            )));
        }
        for (i, s) in slots.iter().enumerate() {
            if slots[..i].contains(s) || *s > i32::MAX as usize {
                return Err(head_err(format!("{WHAT}: slots must be distinct i32s")));
            }
        }
        self.bind_thread()?;
        let symbol = HEAD_LANES_SYMBOLS[columns - 1];
        self.ensure_kernel(HEAD_LANES_MODULE, HEAD_LANES_SRC, symbol)?;
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let mut sv = [0i32; HEAD_LANES_MAX];
        for (d, s) in sv.iter_mut().zip(slots) {
            *d = *s as i32;
        }
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
        ];
        for s in &sv {
            params.push(s as *const _ as *mut c_void);
        }
        let grid = m.div_ceil(HEAD_LANES_ROWS_PER_WG) as u32;
        let bytes = weight_bytes + x_need + columns * m * 4;
        let timer = crate::profile::begin_timer(&self.hip, "gemv", symbol, bytes);
        let result = self.launch_maybe_blob(
            symbol,
            [grid, 1, 1],
            [HEAD_LANES_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                for s in &sv {
                    b.push_i32(*s);
                }
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Whether [`Self::rmsnorm_f32_rowsplit_rows`] equals `rows` singleton
    /// `rmsnorm_f32` launches: the singleton picks `rmsnorm_f32_rowsplit` for
    /// one row when the decode norm grids are on (`n > 256`, `n % 256 == 0`).
    pub fn rmsnorm_f32_rowsplit_rows_admitted(&self, n: usize, rows: usize) -> bool {
        self.flags.dec_norm_grids_enabled()
            && n > 256
            && n % 256 == 0
            && (1..=u16::MAX as usize).contains(&rows)
    }

    /// `out[r] = rmsnorm(x[r]) * weight` over `rows` contiguous rows of `n`:
    /// the singleton `rmsnorm_f32_rowsplit` kernel with `grid.y = rows`, so
    /// each row's bytes are the one-row launch's. `out` must not overlap `x`.
    pub fn rmsnorm_f32_rowsplit_rows(
        &mut self,
        x: &GpuTensor,
        weight: &GpuTensor,
        out: &GpuTensor,
        n: usize,
        rows: usize,
        eps: f32,
    ) -> HipResult<()> {
        const WHAT: &str = "rmsnorm_f32_rowsplit_rows";
        if !self.rmsnorm_f32_rowsplit_rows_admitted(n, rows) {
            return Err(head_err(format!(
                "{WHAT}: not admitted (decode norm grids, n {n} % 256 == 0 and > 256, {rows} rows in 1..=65535)"
            )));
        }
        let span = rows * n * 4;
        if x.buf.size() < span || out.buf.size() < span || weight.buf.size() < n * 4 {
            return Err(head_err(format!("{WHAT}: operand span too small")));
        }
        let x_ptr = x.buf.as_ptr();
        let w_ptr = weight.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        if (x_ptr as usize) < (o_ptr as usize) + span && (o_ptr as usize) < (x_ptr as usize) + span
        {
            return Err(head_err(format!("{WHAT}: out overlaps x")));
        }
        self.bind_thread()?;
        let symbol = "rmsnorm_f32_rowsplit";
        self.ensure_kernel(symbol, kernels::RMSNORM_ROWSPLIT_SRC, symbol)?;
        let n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &n_val as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::rmsnorm_bytes(rows * n);
        let timer = crate::profile::begin_timer(&self.hip, "rmsnorm", symbol, bytes);
        let result = self.launch_maybe_blob(
            symbol,
            [(n / 256) as u32, rows as u32, 1],
            [256, 1, 1],
            256 * 4,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(w_ptr);
                b.push_ptr(o_ptr);
                b.push_i32(n_val);
                b.push_f32(eps);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}
