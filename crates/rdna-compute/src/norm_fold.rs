// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Launcher of the A3B decode norm fold
//! (`kernels/src/fused_rmsnorm_proj4_mq4g256v2_fold.gfx1201.hip`).
//!
//! On exact gfx1201, a `qwen3_5_moe` `.mq4rs` single-row decode runs 80
//! single-workgroup `fused_rmsnorm_mq_rotate` launches per token, each followed
//! by the MQ4G256V2 projection GEMV that consumes its output:
//! `fused_qkvza_mq4g256v2` (GDN mixer, 12352 rows), `fused_qkv_mq4g256v2` (FA
//! mixer, 9216 rows) and the 1281-row router + shared gate/up GEMV of the MoE
//! block. This kernel is one launch of `G` 256-thread workgroups: every
//! workgroup recomputes the row's RMSNorm + FWHT with the incumbent norm's
//! exact code into LDS and then walks its share of the output rows with the
//! consumer's exact per-row arithmetic. Output bytes equal the two-launch chain
//! (lab test `tests/norm_fold.rs`).

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Kernel symbol (and module name).
pub const NORM_FOLD_KERNEL: &str = "fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201";
/// Hidden width the kernel is compiled for.
pub const NORM_FOLD_HIDDEN: usize = 2048;
/// Workgroup size (8 x wave32); one output row per wave at a time.
pub const NORM_FOLD_BLOCK: u32 = 256;
/// Row waves per workgroup.
pub const NORM_FOLD_WAVES: usize = 8;
/// MQ4G256V2 row bytes at K = 2048 (8 groups x 136 B).
const ROW_BYTES: usize = 8 * 136;

/// One fold launch: the norm operands plus up to four weight sources whose rows
/// are routed in order (source 0 first). A three-source (QKV) launch repeats
/// source 2 in slot 3 with `m[3] == 0`.
pub struct NormFoldArgs<'a> {
    /// Residual input `x` (2048 f32).
    pub x: &'a GpuTensor,
    /// RMSNorm weight (2048 f32).
    pub norm_weight: &'a GpuTensor,
    /// Receives the rotated normalised vector (2048 f32).
    pub x_rot: &'a GpuTensor,
    /// MQ4G256V2 weights, K = 2048.
    pub weights: [&'a GpuTensor; 4],
    /// Per-source outputs.
    pub outs: [&'a GpuTensor; 4],
    /// Rows per source.
    pub m: [usize; 4],
    pub eps: f32,
}

fn too_small(what: &str) -> HipError {
    HipError::new(
        0,
        &format!("{NORM_FOLD_KERNEL}: operand span too small: {what}"),
    )
}

impl NormFoldArgs<'_> {
    fn validate(&self) -> HipResult<()> {
        let hidden = NORM_FOLD_HIDDEN * 4;
        let need = |what: &str, have: usize, need: usize| {
            if have < need {
                Err(too_small(what))
            } else {
                Ok(())
            }
        };
        need("x", self.x.buf.size(), hidden)?;
        need("norm_weight", self.norm_weight.buf.size(), hidden)?;
        need("x_rot", self.x_rot.buf.size(), hidden)?;
        for i in 0..4 {
            if self.m[i] > 0 {
                need("weights", self.weights[i].buf.size(), self.m[i] * ROW_BYTES)?;
                need("outs", self.outs[i].buf.size(), self.m[i] * 4)?;
            }
        }
        if self.m[0] == 0 {
            return Err(HipError::new(
                0,
                &format!("{NORM_FOLD_KERNEL}: source 0 must be non-empty"),
            ));
        }
        Ok(())
    }

    /// Total output rows.
    pub fn total_rows(&self) -> usize {
        self.m.iter().sum()
    }
}

/// Default grid for `total_rows` output rows: `rows_per_wave` rows per row
/// wave, at least one workgroup.
pub fn norm_fold_grid(total_rows: usize, rows_per_wave: usize) -> usize {
    let per_wg = NORM_FOLD_WAVES * rows_per_wave.max(1);
    total_rows.div_ceil(per_wg).max(1)
}

impl Gpu {
    /// Launch the norm fold with `grid` workgroups (>= 1). Any grid is
    /// correct; rows are strided over `grid * 8` row waves.
    ///
    /// Requires exact gfx1201 and hidden 2048.
    pub fn fused_rmsnorm_proj4_mq4g256v2_fold(
        &mut self,
        a: &NormFoldArgs<'_>,
        grid: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(HipError::new(
                0,
                &format!("{NORM_FOLD_KERNEL}: requires exact gfx1201"),
            ));
        }
        if grid == 0 || grid > (u32::MAX as usize) / NORM_FOLD_WAVES {
            return Err(HipError::new(
                0,
                &format!("{NORM_FOLD_KERNEL}: grid must be in 1..=u32::MAX/8"),
            ));
        }
        a.validate()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(
            NORM_FOLD_KERNEL,
            kernels::FUSED_RMSNORM_PROJ4_MQ4G256V2_FOLD_GFX1201_SRC,
            NORM_FOLD_KERNEL,
        )?;

        let ptr = |t: &GpuTensor| t.buf.as_ptr();
        let a0 = ptr(a.weights[0]);
        let a1 = ptr(a.weights[1]);
        let a2 = ptr(a.weights[2]);
        let a3 = ptr(a.weights[3]);
        let xp = ptr(a.x);
        let wp = ptr(a.norm_weight);
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let y0 = ptr(a.outs[0]);
        let y1 = ptr(a.outs[1]);
        let y2 = ptr(a.outs[2]);
        let y3 = ptr(a.outs[3]);
        let xr = ptr(a.x_rot);
        let m0 = a.m[0] as i32;
        let m1 = a.m[1] as i32;
        let m2 = a.m[2] as i32;
        let m3 = a.m[3] as i32;
        let nw = (grid * NORM_FOLD_WAVES) as i32;
        let eps = a.eps;

        let mut params: Vec<*mut c_void> = vec![
            &a0 as *const _ as *mut c_void,
            &a1 as *const _ as *mut c_void,
            &a2 as *const _ as *mut c_void,
            &a3 as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &y0 as *const _ as *mut c_void,
            &y1 as *const _ as *mut c_void,
            &y2 as *const _ as *mut c_void,
            &y3 as *const _ as *mut c_void,
            &xr as *const _ as *mut c_void,
            &m0 as *const _ as *mut c_void,
            &m1 as *const _ as *mut c_void,
            &m2 as *const _ as *mut c_void,
            &m3 as *const _ as *mut c_void,
            &nw as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];

        let bytes = a.total_rows() * ROW_BYTES + NORM_FOLD_HIDDEN * 4 * 3;
        let timer = crate::profile::begin_timer(&self.hip, "fused", NORM_FOLD_KERNEL, bytes);
        let result = self.launch_maybe_blob(
            NORM_FOLD_KERNEL,
            [grid as u32, 1, 1],
            [NORM_FOLD_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(a0);
                b.push_ptr(a1);
                b.push_ptr(a2);
                b.push_ptr(a3);
                b.push_ptr(xp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(y0);
                b.push_ptr(y1);
                b.push_ptr(y2);
                b.push_ptr(y3);
                b.push_ptr(xr);
                b.push_i32(m0);
                b.push_i32(m1);
                b.push_i32(m2);
                b.push_i32(m3);
                b.push_i32(nw);
                b.push_f32(eps);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        self.invalidate_x_caches_for(xr);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_covers_all_rows() {
        assert_eq!(norm_fold_grid(1, 4), 1);
        assert_eq!(norm_fold_grid(1281, 2), 81);
        assert_eq!(norm_fold_grid(12352, 6), 258);
        assert_eq!(norm_fold_grid(12352, 0), 1544);
    }

    #[test]
    fn device_source_matches_launcher_constants() {
        let src = kernels::FUSED_RMSNORM_PROJ4_MQ4G256V2_FOLD_GFX1201_SRC;
        assert!(src.contains(&format!("#define NF_K {NORM_FOLD_HIDDEN}")));
        assert!(src.contains(&format!("#define NF_WAVES {NORM_FOLD_WAVES}")));
        assert!(src.contains(NORM_FOLD_KERNEL));
    }
}
