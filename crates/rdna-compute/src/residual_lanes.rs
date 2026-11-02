// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-batched MQ4G256V2 residual GEMV for the A3B exact VMM batch route
//! (exact gfx1201): `y[b] += W . x[b]` for `b < lanes` (1..=8), each lane
//! byte-identical to the singleton `gemv_mq4g256v2_residual` run on
//! `(x[b], y[b])`.
//!
//! One workgroup of four wave32s owns four output rows and splits their K range
//! over the waves: wave `w` runs accumulator chain `w` (four of the sixteen
//! groups) of all four rows, so a wave holds one chain of 4 x lanes
//! accumulators; the finished chains are exchanged exactly (raw f32) through
//! LDS and combined as the singleton does. Each weight row is streamed once
//! into registers and decoded once per group while the lane columns repeat only
//! the singleton's per-lane FMA chains. See
//! `kernels/src/gemv_mq4g256v2_residual_lanes.gfx1201.hip`. Replaces the PM
//! x-batch residual (`pm_xbatch::residual`) for the A3B `wo` projection
//! (m = 2048, K = 4096) behind `HIPFIRE_A3B_LANE_WO`; every other shape, and
//! lane counts where the PM x-batch launch measures faster, stay on it.
//!
//! Support is a pure function of the shapes: exact gfx1201, 1..=8 lanes,
//! `K == 4096` and 32-bit weight byte offsets. The route admits it for
//! [`RESIDUAL_LANES_MIN_ROUTED`]..=8 lanes: with the weights and instruction
//! caches cold, as between the two `wo` launches of a decode step
//! (`LANE_WO_FLUSH=1`, flush copy inside the timed interval), this kernel is
//! 6.3..8.7 us per launch faster than the PM x-batch launch from 5 to 8 lanes
//! and within the +-0.4 us noise of it at 1, 2 and 4 lanes (warm, the PM
//! x-batch launch is faster at every lane count); see the lab gate
//! `tests/a3b_lane_wo.rs`.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Most lanes one launch covers.
pub const RESIDUAL_LANES_MAX: usize = 8;
/// Fewest lanes the route sends here (the PM x-batch launch is faster below).
pub const RESIDUAL_LANES_MIN_ROUTED: usize = 5;
/// Module (and entry) name of the lane-batched residual GEMV.
pub const RESIDUAL_LANES_MODULE: &str = "gemv_mq4g256v2_residual_lanes_gfx1201";
/// Waves per workgroup (each runs one of the four accumulator chains) and
/// rows per workgroup of the shipped module (the kernel's `KS_NSPL` / `KS_R`
/// defaults).
pub const RESIDUAL_LANES_WAVES: usize = 4;
pub const RESIDUAL_LANES_ROWS_PER_WG: usize = 4;
/// The input width the kernel is compiled for (the A3B `wo`: 32 heads x 128).
pub const RESIDUAL_LANES_K: usize = 4096;

/// Module source with a non-default workgroup geometry and entry name (lab
/// tuning sweeps: `waves` = `KS_NSPL`, `rows_per_wg` = `KS_R`); the shipped
/// module is [`kernels::GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC`].
pub fn residual_lanes_source_with(entry: &str, waves: usize, rows_per_wg: usize) -> String {
    format!(
        "#define HIPFIRE_RESIDUAL_LANES_KERNEL {entry}\n#define KS_NSPL {waves}\n#define KS_R {rows_per_wg}\n{}",
        kernels::GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC
    )
}

impl Gpu {
    /// Whether [`Self::gemv_mq4g256v2_residual_lanes`] can run this shape on
    /// this device (hardware / layout contract; any lane count in 1..=8).
    pub fn gemv_mq4g256v2_residual_lanes_supported(
        &self,
        m: usize,
        k: usize,
        lanes: usize,
    ) -> bool {
        let fits = |n: Option<usize>| n.is_some_and(|n| n <= u32::MAX as usize);
        self.arch_caps.is_gfx1201()
            && (1..=RESIDUAL_LANES_MAX).contains(&lanes)
            && m > 0
            && k == RESIDUAL_LANES_K
            && fits(
                (k / 256)
                    .checked_mul(136)
                    .and_then(|row| row.checked_mul(m)),
            )
            && fits(m.checked_mul(lanes * 4))
    }

    /// Whether the A3B route launches [`Self::gemv_mq4g256v2_residual_lanes`]
    /// instead of the PM x-batch residual for this shape: supported, and a
    /// lane count where it measures faster ([`RESIDUAL_LANES_MIN_ROUTED`]..=8).
    pub fn gemv_mq4g256v2_residual_lanes_admitted(&self, m: usize, k: usize, lanes: usize) -> bool {
        lanes >= RESIDUAL_LANES_MIN_ROUTED
            && self.gemv_mq4g256v2_residual_lanes_supported(m, k, lanes)
    }

    /// `y[b] += W . x[b]` for `b < lanes`; `x` is `[lanes][k]`, `y` is
    /// `[lanes][m]` (f32 row-major), `a_raw` an MQ4G256V2 matrix
    /// `[m][k / 256 * 136 B]`. Per lane byte-identical to the singleton
    /// `gemv_mq4g256v2_residual` on that lane's rows.
    pub fn gemv_mq4g256v2_residual_lanes(
        &mut self,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        lanes: usize,
    ) -> HipResult<()> {
        if !self.gemv_mq4g256v2_residual_lanes_supported(m, k, lanes) {
            return Err(HipError::new(
                0,
                &format!(
                    "gemv_mq4g256v2_residual_lanes: shape m {m} k {k} lanes {lanes} not supported \
                     (exact gfx1201, lanes 1..={RESIDUAL_LANES_MAX}, k == {RESIDUAL_LANES_K})"
                ),
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            RESIDUAL_LANES_MODULE,
            kernels::GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC,
            RESIDUAL_LANES_MODULE,
        )?;
        self.launch_residual_lanes(
            RESIDUAL_LANES_MODULE,
            RESIDUAL_LANES_WAVES,
            RESIDUAL_LANES_ROWS_PER_WG,
            a_raw,
            x,
            y,
            m,
            k,
            lanes,
        )
    }

    /// Launch an already loaded residual-lanes module built for `waves`
    /// (`KS_NSPL`) x `rows_per_wg` (`KS_R`) geometry (lab sweeps; no admission
    /// checks).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_residual_lanes(
        &mut self,
        module: &str,
        waves: usize,
        rows_per_wg: usize,
        a_raw: &GpuTensor,
        x: &GpuTensor,
        y: &GpuTensor,
        m: usize,
        k: usize,
        lanes: usize,
    ) -> HipResult<()> {
        let a_ptr = a_raw.buf.as_ptr();
        let x_ptr = x.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        debug_assert_eq!(k, RESIDUAL_LANES_K);
        let (m_val, b_val) = (m as i32, lanes as i32);
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &x_ptr as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        let grid = m.div_ceil(rows_per_wg) as u32;
        self.launch_maybe_blob(
            module,
            [grid, 1, 1],
            [(waves * 32) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a_ptr);
                b.push_ptr(x_ptr);
                b.push_ptr(y_ptr);
                b.push_i32(m_val);
                b.push_i32(b_val);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_source_carries_the_policy_preamble_and_entry() {
        let src = kernels::GEMV_MQ4G256V2_RESIDUAL_LANES_GFX1201_SRC;
        assert!(src.contains("HIPFIRE_WEIGHT_LOAD_SPLIT_U32"));
        assert!(src.contains("void HIPFIRE_RESIDUAL_LANES_KERNEL("));
        assert!(residual_lanes_source_with("e", 4, 4).contains("#define KS_NSPL 4\n"));
    }
}
