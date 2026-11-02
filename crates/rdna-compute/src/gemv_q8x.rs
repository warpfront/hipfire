// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Ben1078
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact-gfx1100 int8-activation small-batch GEMV for MQ4G256V2 speculative
//! verify (default on for exact gfx1100; `HIPFIRE_GFX1100_Q8X_VERIFY=0`
//! restores the f16 WMMA tiers).
//!
//! An MTP K=3 verify forward runs every MQ4G256V2 projection with 4
//! activation rows. On exact gfx1100 those rows otherwise go to the N<=16
//! f16 WMMA residual tiers (`gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage` /
//! `_ksplit_lds`), where 12 of every 16 tile columns are padding and the
//! weight stream runs at about half of the card's bandwidth. This route
//! quantizes the `[batch x k]` activations to int8 once per projection group
//! (one f32 scale per 32 elements, `mq4v2_q8x_quant`) and reads every weight
//! row once, dotting it against all rows with `v_dot4_i32_iu8`
//! (`kernels/src/gemv_mq4g256v2_q8x.hip`). Up to four weights that share `x`
//! (q/k/v, GDN qkv/z/beta/alpha, gate/up) run in one `_multi` launch.
//!
//! Selection ([`Gpu::q8x_verify_active`]): the flag on (the exact-gfx1100
//! default), exact gfx1100,
//! `2 <= batch <= 5`, `k % 256 == 0`, `k <= 32768`, eager only (never under
//! graph capture or Redline recording, whose launch contracts stay the WMMA
//! tiers), outside the FP8 verify scope. Hooked inside the four
//! `gemm_*_hfq4g256_mq4v2` wrappers in `gemm.rs`, so model crates are
//! unchanged. Note that a 2..5-row *prefill* chunk through those wrappers
//! takes the route too. Every module is declared in the gfx1100 kernel
//! registry (`kernel_registry::entries`), so the gfx1100 kernel pack carries
//! them precompiled and the route never JIT-compiles on a first request.
//!
//! Numerics: the integer sums are exact; the activation quantization is
//! q8_1-class (relative L2 error ~4e-3 per activation vector), so the output
//! is NOT byte-identical to the WMMA tiers. The route is deterministic (no
//! atomics; split partials are reduced in a fixed order).
//!
//! Variants (fixed tuning, measured on a 7900 XTX over the Qwen3.8-27B verify
//! shapes): v4 (persistent grid, all of K staged in LDS) at batch <= 2, and at
//! batch <= 4 for K <= 5120; v3 (LDS-staged chunks, two rows per warp)
//! otherwise.

use crate::{kernels, DType, Gpu, GpuTensor};
use hip_bridge::HipResult;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

/// Largest activation batch the route takes (MTP K <= 4 verifies).
const Q8X_MAX_BATCH: usize = 5;
/// Largest reduction length (activation scratch is sized for it).
const Q8X_MAX_K: usize = 32768;
/// Scratch rows: room for every admitted batch.
const Q8X_SCRATCH_ROWS: usize = 8;
/// Warps per workgroup of both variants (`Q8X3_R`).
const Q8X_R: usize = 8;
/// v3 rows per warp (`Q8X3_W`) and staged-chunk budget (`Q8X3_LDSKB` KiB).
const Q8X_V3_W: usize = 2;
const Q8X_V3_P: usize = 2;
const Q8X_V3_LDSKB: usize = 32;
/// v4 stages all of K; its budget is the 64 KiB dynamic LDS limit.
const Q8X_V4_LDSKB: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Q8xVariant {
    V3,
    V4,
}

/// `(module = single-weight symbol, multi-weight symbol, source)` of every
/// compiled variant. Index: [`q8x_module_index`].
const Q8X_MODULES: [(&str, &str, &str); 14] = [
    (
        "mq4v2_q8x_v4_b2",
        "mq4v2_q8x_v4_b2_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B2_SRC,
    ),
    (
        "mq4v2_q8x_v4_b2_res",
        "mq4v2_q8x_v4_b2_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B2_RES_SRC,
    ),
    (
        "mq4v2_q8x_v4_b3",
        "mq4v2_q8x_v4_b3_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B3_SRC,
    ),
    (
        "mq4v2_q8x_v4_b3_res",
        "mq4v2_q8x_v4_b3_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B3_RES_SRC,
    ),
    (
        "mq4v2_q8x_v4_b4",
        "mq4v2_q8x_v4_b4_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B4_SRC,
    ),
    (
        "mq4v2_q8x_v4_b4_res",
        "mq4v2_q8x_v4_b4_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V4_B4_RES_SRC,
    ),
    (
        "mq4v2_q8x_v3_b2",
        "mq4v2_q8x_v3_b2_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B2_SRC,
    ),
    (
        "mq4v2_q8x_v3_b2_res",
        "mq4v2_q8x_v3_b2_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B2_RES_SRC,
    ),
    (
        "mq4v2_q8x_v3_b3",
        "mq4v2_q8x_v3_b3_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B3_SRC,
    ),
    (
        "mq4v2_q8x_v3_b3_res",
        "mq4v2_q8x_v3_b3_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B3_RES_SRC,
    ),
    (
        "mq4v2_q8x_v3_b4",
        "mq4v2_q8x_v3_b4_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B4_SRC,
    ),
    (
        "mq4v2_q8x_v3_b4_res",
        "mq4v2_q8x_v3_b4_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B4_RES_SRC,
    ),
    (
        "mq4v2_q8x_v3_b5",
        "mq4v2_q8x_v3_b5_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B5_SRC,
    ),
    (
        "mq4v2_q8x_v3_b5_res",
        "mq4v2_q8x_v3_b5_res_multi",
        kernels::GEMV_MQ4G256V2_Q8X_V3_B5_RES_SRC,
    ),
];

const Q8X_QUANT_MODULE: &str = "mq4v2_q8x_quant";

fn q8x_module_index(v: Q8xVariant, batch: usize, residual: bool) -> usize {
    let base = match v {
        Q8xVariant::V4 => (batch - 2) * 2,
        Q8xVariant::V3 => 6 + (batch - 2) * 2,
    };
    base + usize::from(residual)
}

/// Staged bytes per 4-group step for `batch` rows (`4 * (Q8X3_PSX + Q8X3_PDD)`).
fn q8x_step_bytes(batch: usize) -> usize {
    4 * ((batch * 256 + 16) + ((batch * 64 + 191) / 256 * 256 + 64))
}

/// Dynamic LDS bytes per workgroup (same formula as `q8x3_chunk_steps`).
fn q8x_lds_bytes(v: Q8xVariant, k: usize, batch: usize) -> usize {
    let per_step = q8x_step_bytes(batch);
    let nst = k.div_ceil(1024);
    match v {
        Q8xVariant::V4 => nst * per_step,
        Q8xVariant::V3 => {
            let mut cs = (Q8X_V3_LDSKB * 1024) / per_step;
            if nst <= cs {
                cs = nst;
            } else {
                cs -= cs % Q8X_V3_P;
                cs = cs.max(Q8X_V3_P);
            }
            cs * per_step
        }
    }
}

fn q8x_pick(k: usize, batch: usize) -> Q8xVariant {
    let v4_fits = batch <= 4 && q8x_lds_bytes(Q8xVariant::V4, k, batch) <= Q8X_V4_LDSKB * 1024;
    if v4_fits && (batch <= 2 || k <= 5120) {
        Q8xVariant::V4
    } else {
        Q8xVariant::V3
    }
}

/// One stderr line the first time each (module, single/multi) pair runs, so
/// a harness log proves the route was active.
static Q8X_ENGAGED: AtomicU64 = AtomicU64::new(0);

impl Gpu {
    /// Whether a `[batch x k]` MQ4G256V2 projection takes the int8 small-batch
    /// GEMV (see the module docs for the gates).
    pub fn q8x_verify_active(&self, k: usize, batch: usize) -> bool {
        self.flags.gfx1100_q8x_verify
            && self.arch == "gfx1100"
            && self.arch_caps.is_gfx1100()
            && (2..=Q8X_MAX_BATCH).contains(&batch)
            && k % 256 == 0
            && k <= Q8X_MAX_K
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && !self.fp8_verify_scope
    }

    /// The int8 activation scratch `(xq, dx)`: `Q8X_SCRATCH_ROWS x Q8X_MAX_K`
    /// int8 plus one f32 scale per 32 elements, allocated on first use.
    fn q8x_scratch(&mut self) -> HipResult<(*mut c_void, *mut c_void)> {
        if self.scratch.q8x_act.is_none() {
            let xq = self.alloc_tensor(&[Q8X_SCRATCH_ROWS * Q8X_MAX_K / 4], DType::F32)?;
            let dx = self.alloc_tensor(&[Q8X_SCRATCH_ROWS * Q8X_MAX_K / 32], DType::F32)?;
            self.scratch.q8x_act = Some((xq, dx));
        }
        let (xq, dx) = self.scratch.q8x_act.as_ref().unwrap();
        Ok((xq.buf.as_ptr(), dx.buf.as_ptr()))
    }

    /// Resident workgroups per multiprocessor (a WGP) for a loaded v4
    /// function. The occupancy API counts 64 KiB of LDS per CU on gfx1100,
    /// while a WGP-mode workgroup allocates from the WGP's 128 KiB, so two
    /// 256-thread workgroups of <= 64 KiB always fit: floor at 2.
    fn q8x_resident_per_mp(&self, func_name: &str, lds: u32) -> usize {
        self.occupancy_max_active_blocks(func_name, [(32 * Q8X_R) as u32, 1, 1], lds)
            .ok()
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .unwrap_or(1)
            .max(2)
    }

    fn q8x_mp_count(&self) -> usize {
        self.hip
            .get_device_attribute(
                crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                self.device_id,
            )
            .ok()
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .unwrap_or(48)
    }

    /// `y_i (+)= A_i · x[b]` for `b in 0..batch` and up to four MQ4G256V2
    /// weights `(A_i, y_i, m_i)` sharing the FWHT-rotated f32 input `x`
    /// (`[batch x k]`). `residual` adds into `y_i`, otherwise overwrites.
    /// Callers check [`Self::q8x_verify_active`] first.
    pub fn gemm_mq4g256v2_q8x(
        &mut self,
        outs: &[(&GpuTensor, &GpuTensor, usize)],
        x: &GpuTensor,
        k: usize,
        batch: usize,
        residual: bool,
    ) -> HipResult<()> {
        if !(2..=Q8X_MAX_BATCH).contains(&batch) || k % 256 != 0 || k > Q8X_MAX_K {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("gemm_mq4g256v2_q8x: unsupported shape k={k} batch={batch}"),
            ));
        }
        if outs.is_empty() || outs.len() > 4 {
            return Err(hip_bridge::HipError::new(
                1,
                &format!("gemm_mq4g256v2_q8x: {} weights (1..=4)", outs.len()),
            ));
        }
        self.bind_thread()?;
        let (xq, dx) = self.q8x_scratch()?;

        // 1. Activations -> int8 + per-32 scales, once for the whole group.
        self.ensure_kernel(
            Q8X_QUANT_MODULE,
            kernels::GEMV_MQ4G256V2_Q8X_QUANT_SRC,
            Q8X_QUANT_MODULE,
        )?;
        {
            let func = &self.functions[Q8X_QUANT_MODULE];
            let x_ptr = x.buf.as_ptr();
            let k_val = k as i32;
            let mut params: Vec<*mut c_void> = vec![
                &x_ptr as *const _ as *mut c_void,
                &xq as *const _ as *mut c_void,
                &dx as *const _ as *mut c_void,
                &k_val as *const _ as *mut c_void,
            ];
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [(k / 256) as u32, batch as u32, 1],
                    [32, 1, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
        }

        // 2. One GEMV launch for the group (or one per weight for a single weight).
        let v = q8x_pick(k, batch);
        let idx = q8x_module_index(v, batch, residual);
        let (module, multi_symbol, source) = Q8X_MODULES[idx];
        let use_multi = outs.len() > 1;
        let func_name = if use_multi { multi_symbol } else { module };
        self.ensure_kernel(module, source, func_name)?;
        let bit = 1u64 << (idx * 2 + usize::from(use_multi));
        if Q8X_ENGAGED.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            eprintln!(
                "[q8x-verify] route active: {func_name} k={k} batch={batch} weights={}",
                outs.len()
            );
        }
        let lds = q8x_lds_bytes(v, k, batch) as u32;
        let threads = (32 * Q8X_R) as u32;
        let rows_per_wg = match v {
            Q8xVariant::V3 => Q8X_R * Q8X_V3_W,
            Q8xVariant::V4 => Q8X_R,
        };
        let persistent = |gpu: &Gpu, rows: usize| -> usize {
            let per_mp = gpu.q8x_resident_per_mp(func_name, lds);
            (per_mp * gpu.q8x_mp_count()).min(rows / Q8X_R).max(1)
        };
        let k_val = k as i32;
        let b_val = batch as i32;
        if use_multi {
            let mut a_ptrs = [std::ptr::null_mut::<c_void>(); 4];
            let mut y_ptrs = [std::ptr::null_mut::<c_void>(); 4];
            let mut ms = [0i32; 4];
            let mut total = 0usize;
            let mut blocks = 0usize;
            for (i, &(a, y, m)) in outs.iter().enumerate() {
                a_ptrs[i] = a.buf.as_ptr();
                y_ptrs[i] = y.buf.as_ptr();
                ms[i] = m as i32;
                total += m;
                blocks += m.div_ceil(rows_per_wg);
            }
            if v == Q8xVariant::V4 {
                blocks = persistent(self, total);
            }
            let func = &self.functions[func_name];
            let mut params: Vec<*mut c_void> = Vec::with_capacity(16);
            for i in 0..4 {
                params.push(&a_ptrs[i] as *const _ as *mut c_void);
                params.push(&y_ptrs[i] as *const _ as *mut c_void);
                params.push(&ms[i] as *const _ as *mut c_void);
            }
            params.push(&xq as *const _ as *mut c_void);
            params.push(&dx as *const _ as *mut c_void);
            params.push(&k_val as *const _ as *mut c_void);
            params.push(&b_val as *const _ as *mut c_void);
            unsafe {
                self.hip.launch_kernel(
                    func,
                    [blocks as u32, 1, 1],
                    [threads, 1, 1],
                    lds,
                    self.stream_ref(),
                    &mut params,
                )?;
            }
            return Ok(());
        }
        let (a, y, m) = outs[0];
        let blocks = match v {
            Q8xVariant::V4 => persistent(self, m),
            Q8xVariant::V3 => m.div_ceil(rows_per_wg),
        };
        let func = &self.functions[func_name];
        let a_ptr = a.buf.as_ptr();
        let y_ptr = y.buf.as_ptr();
        let m_val = m as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &dx as *const _ as *mut c_void,
            &y_ptr as *const _ as *mut c_void,
            &m_val as *const _ as *mut c_void,
            &k_val as *const _ as *mut c_void,
            &b_val as *const _ as *mut c_void,
        ];
        unsafe {
            self.hip.launch_kernel(
                func,
                [blocks as u32, 1, 1],
                [threads, 1, 1],
                lds,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_table_is_indexed_consistently() {
        for (v, prefix, batches) in [
            (Q8xVariant::V4, "mq4v2_q8x_v4", 2..=4usize),
            (Q8xVariant::V3, "mq4v2_q8x_v3", 2..=5usize),
        ] {
            for b in batches {
                for res in [false, true] {
                    let (module, multi, _) = Q8X_MODULES[q8x_module_index(v, b, res)];
                    let want = format!("{prefix}_b{b}{}", if res { "_res" } else { "" });
                    assert_eq!(module, want);
                    assert_eq!(multi, format!("{want}_multi"));
                }
            }
        }
    }

    #[test]
    fn pick_matches_measured_policy_and_lds_limits() {
        // Qwen3.8-27B verify shapes at 4 rows.
        assert_eq!(q8x_pick(5120, 4), Q8xVariant::V4); // qkv / qkvza / gate_up / lm_head
        assert_eq!(q8x_pick(6144, 4), Q8xVariant::V3); // o_proj / GDN out
        assert_eq!(q8x_pick(17408, 4), Q8xVariant::V3); // MLP down
        assert_eq!(q8x_pick(17408, 2), Q8xVariant::V4); // fits 64 KiB at 2 rows
        assert_eq!(q8x_pick(5120, 5), Q8xVariant::V3);
        for b in 2..=Q8X_MAX_BATCH {
            for k in [256usize, 5120, 6144, 17408, 32768] {
                let v = q8x_pick(k, b);
                let lds = q8x_lds_bytes(v, k, b);
                let cap = match v {
                    Q8xVariant::V4 => Q8X_V4_LDSKB * 1024,
                    Q8xVariant::V3 => (Q8X_V3_LDSKB * 1024).max(Q8X_V3_P * q8x_step_bytes(b)),
                };
                assert!(lds <= cap && lds <= 65536, "k={k} b={b} {v:?} lds={lds}");
            }
        }
    }
}
