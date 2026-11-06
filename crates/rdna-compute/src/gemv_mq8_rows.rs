// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ8G256 decode GEMV (n = 1) and verify rows-exact GEMV (n = 2..) on weight layout B.
//!
//! [`Gpu::gemv_mq8g256_b`] / [`Gpu::gemv_mq8g256_rows_exact_b`] multiply one layout-B weight
//! ([`crate::gemm_mq8g256::mq8g256_repack_b`]: f16 scale plane `[m, k/256]`, pad to 256 B, int8 code plane
//! `[m, k]`) by the [`Mq8Activations`] the batched producer ([`Gpu::mq8_quantize_x`]) wrote, the same
//! activations the prefill GEMM consumes: one rotate + quantize launch feeds every projection that shares
//! the X (RDA q/k/v/g; MLA q_a/kv_a/g), at decode, verify and prefill.
//!
//! `y[r * m + row]` for X row `r` is BIT-IDENTICAL between the n = 1 kernel and every rows-exact kernel: all
//! `gemv_mq8g256_b_r<R>_u1_w4_p1` run one association (see `kernels/src/gemv_mq8g256_b.hip`): one wave32 per
//! weight row, lane `(h, s)` owns 16 code bytes of group `2p + h` of every group pair `p`, an exact int32
//! `sdot4` x4 per (lane, pair) folded as `acc = fmaf(RN(ws * xs), (float)dot, acc)` in pair order, then a
//! 16-lane butterfly per half and `T_0 + T_1`. R (X rows per launch, 1..=16) is a scheduling
//! choice that never changes a bit; more than 16 rows run in groups of 16.
//!
//! Plain launches, no host sync, no allocation (graph-capture safe after one eager warm run). Contracts:
//! `hipfire_kernel_abi::families::generic_mq8::{GEMV_MQ8G256_B, GEMV_MQ8G256_BX}`.

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemm_mq8g256::{mq8g256_b_bytes, Mq8Activations};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Most X rows one `gemv_mq8g256_b_r<R>_*` launch serves; [`Gpu::gemv_mq8g256_rows_exact_b`] splits larger
/// row counts into groups of this many (rows never interact, every group is exact).
pub const MQ8G256_ROWS_EXACT_KERNELS: usize = 16;

/// Weights per scale group.
const GROUP: usize = 256;

/// Waves (weight rows) per 128-thread block; matches `MQ8B_WPB` in the kernel source.
pub const MQ8G256_B_WAVES_PER_BLOCK: usize = 4;

/// Module of every layout-B GEMV symbol.
pub const MQ8G256_B_MODULE: &str = "gemv_mq8g256_b";

/// `gemv_mq8g256_b_r<R>_u1_w4_p1` symbols, index `R - 1`.
const MQ8G256_B_SYMBOLS: [&str; MQ8G256_ROWS_EXACT_KERNELS] = [
    "gemv_mq8g256_b_r1_u1_w4_p1",
    "gemv_mq8g256_b_r2_u1_w4_p1",
    "gemv_mq8g256_b_r3_u1_w4_p1",
    "gemv_mq8g256_b_r4_u1_w4_p1",
    "gemv_mq8g256_b_r5_u1_w4_p1",
    "gemv_mq8g256_b_r6_u1_w4_p1",
    "gemv_mq8g256_b_r7_u1_w4_p1",
    "gemv_mq8g256_b_r8_u1_w4_p1",
    "gemv_mq8g256_b_r9_u1_w4_p1",
    "gemv_mq8g256_b_r10_u1_w4_p1",
    "gemv_mq8g256_b_r11_u1_w4_p1",
    "gemv_mq8g256_b_r12_u1_w4_p1",
    "gemv_mq8g256_b_r13_u1_w4_p1",
    "gemv_mq8g256_b_r14_u1_w4_p1",
    "gemv_mq8g256_b_r15_u1_w4_p1",
    "gemv_mq8g256_b_r16_u1_w4_p1",
];

/// Launch-shape cap (a kernel capability, not a tuned selector): most X rows a fused-producer launch serves (`gemv_mq8g256_bx_r1_*`, `_r2_*`).
pub const MQ8G256_FUSED_N_CAP: usize = 2;

/// Largest K the fused kernels' LDS X staging holds (`MQ8B_FX_KMAX` in the kernel source).
pub const MQ8G256_FUSED_MAX_K: usize = 5120;

/// One fused-producer launch shape: kernel symbol and waves per block (one weight row per wave).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mq8FusedRoute {
    pub symbol: &'static str,
    pub waves_per_block: usize,
}

/// The fused-producer kernel for `n` X rows of length `k`, or `None` (the caller then runs the shared
/// producer and the unfused GEMV). One kernel per `(R, U)` with `U = ceil(K / 512)` pairs read ahead:
/// `K <= 512` (Raon kv_b): U 1, 10 waves; `1536 < K <= 2560` (q/k/v/g/q_a/kv_a): U 5, 24 waves;
/// `2560 < K <= 5120` (o): U 10, 20 waves. `n` is 1 or 2. Pure.
pub fn mq8g256_b_fused_route(k: usize, n: usize) -> Option<Mq8FusedRoute> {
    const SYMS: [[(&str, usize); 3]; MQ8G256_FUSED_N_CAP] = [
        [
            ("gemv_mq8g256_bx_r1_u1_w10_p1", 10),
            ("gemv_mq8g256_bx_r1_u5_w24_p1", 24),
            ("gemv_mq8g256_bx_r1_u10_w20_p1", 20),
        ],
        [
            ("gemv_mq8g256_bx_r2_u1_w10_p1", 10),
            ("gemv_mq8g256_bx_r2_u5_w24_p1", 24),
            ("gemv_mq8g256_bx_r2_u10_w20_p1", 20),
        ],
    ];
    if n == 0 || n > MQ8G256_FUSED_N_CAP || k == 0 || k % GROUP != 0 {
        return None;
    }
    let bucket = if k <= 512 {
        0
    } else if k > 1536 && k <= 2560 {
        1
    } else if k > 2560 && k <= MQ8G256_FUSED_MAX_K {
        2
    } else {
        return None;
    };
    let (symbol, waves_per_block) = SYMS[n - 1][bucket];
    Some(Mq8FusedRoute {
        symbol,
        waves_per_block,
    })
}

fn shape_error(what: &str) -> HipError {
    HipError::new(
        1,
        &format!(
            "{what}: needs m > 0, k a positive multiple of 256, 16-byte aligned layout-B weight of \
             mq8g256_b_bytes(m, k) bytes, F32 y of rows*m, current activations with the same k, and \
             dimensions within i32"
        ),
    )
}

impl Gpu {
    /// Decode GEMV on layout B: `y[row] = W[row] . x` for the single activation row of `acts`
    /// (`acts.rows() == 1`), `y` F32 `[m]`. `k` is `acts.k()`. See the module doc.
    pub fn gemv_mq8g256_b(
        &mut self,
        w: &GpuTensor,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        if acts.n != 1 {
            return Err(shape_error("gemv_mq8g256_b (acts.rows() must be 1)"));
        }
        self.gemv_mq8g256_b_launch_rows("gemv_mq8g256_b", w, acts, y, m)
    }

    /// Verify GEMV on layout B for `acts.rows()` activation rows (one row is the decode GEMV itself):
    /// `y[r * m + row]`, `y` F32 `[rows, m]`. Row `r` is bit-identical to [`Gpu::gemv_mq8g256_b`] on X row `r`
    /// of the same `acts`. Weights are read once per launch of up to
    /// [`MQ8G256_ROWS_EXACT_KERNELS`] rows.
    pub fn gemv_mq8g256_rows_exact_b(
        &mut self,
        w: &GpuTensor,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        self.gemv_mq8g256_b_launch_rows("gemv_mq8g256_rows_exact_b", w, acts, y, m)
    }

    /// `acts.rows() == 1` ? [`Gpu::gemv_mq8g256_b`] : [`Gpu::gemv_mq8g256_rows_exact_b`] (the same launches).
    pub fn gemv_mq8g256_b_rows(
        &mut self,
        w: &GpuTensor,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        self.gemv_mq8g256_b_launch_rows("gemv_mq8g256_b_rows", w, acts, y, m)
    }

    /// Decode / small-verify GEMV on layout B straight from F32 activations (`x` `[n, k]`, 16-byte aligned,
    /// `n` rows, `y` F32 `[n, m]`): the fused-producer kernel rotates and quantizes X inside the GEMV (one
    /// launch, no producer, the codes and scales are the batched producer's bit for bit), so the result
    /// equals [`Gpu::mq8_quantize_x`] + [`Gpu::gemv_mq8g256_rows_exact_b`] exactly. Admitted for `n` 1 or 2
    /// and the K buckets of [`mq8g256_b_fused_route`]; anything else runs that two-launch pair. Prefer it for
    /// every projection at n = 1 (decode) and n = 2; for n >= 3 quantize each shared X once and use the
    /// rows-exact kernels. First call per (kernel, K) compiles / loads kernels and the sign tables: warm
    /// eagerly before graph capture.
    pub fn gemv_mq8g256_b_x(
        &mut self,
        w: &GpuTensor,
        x: &GpuTensor,
        n: usize,
        y: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        let Some(route) = mq8g256_b_fused_route(k, n) else {
            let acts = self.mq8_quantize_x(x, n, k)?;
            return self.gemv_mq8g256_b_launch_rows("gemv_mq8g256_b_x", w, &acts, y, m);
        };
        let ok = m > 0
            && i32::try_from(m).is_ok()
            && m.checked_mul(k).is_some_and(|mk| mk < (1usize << 40))
            && w.buf.size() >= mq8g256_b_bytes(m, k)
            && (w.buf.as_ptr() as usize) % 16 == 0
            && x.dtype == DType::F32
            && x.buf.size() >= n * k * 4
            && (x.buf.as_ptr() as usize) % 16 == 0
            && y.dtype == DType::F32
            && y.buf.size() >= n * m * 4;
        if !ok {
            return Err(shape_error("gemv_mq8g256_b_x"));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_kernel(MQ8G256_B_MODULE, kernels::GEMV_MQ8G256_B_SRC, route.symbol)?;
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let (wp, xp, yp) = (w.buf.as_ptr(), x.buf.as_ptr(), y.buf.as_ptr());
        let (mv, kv) = (m as i32, k as i32);
        let mut params: Vec<*mut c_void> = vec![
            &wp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let grid = [m.div_ceil(route.waves_per_block) as u32, 1, 1];
        let block = [(32 * route.waves_per_block) as u32, 1, 1];
        self.launch_maybe_blob(route.symbol, grid, block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(wp);
            b.push_ptr(xp);
            b.push_ptr(s1);
            b.push_ptr(s2);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b
        })
    }

    fn gemv_mq8g256_b_launch_rows(
        &mut self,
        what: &str,
        w: &GpuTensor,
        acts: &Mq8Activations,
        y: &GpuTensor,
        m: usize,
    ) -> HipResult<()> {
        let (n, k) = (acts.n, acts.k);
        if n == 0 {
            return Ok(());
        }
        let ok = m > 0
            && k > 0
            && k % GROUP == 0
            && i32::try_from(m).is_ok()
            && i32::try_from(k).is_ok()
            && i32::try_from(n).is_ok()
            && m.checked_mul(k).is_some_and(|mk| mk < (1usize << 40))
            && w.buf.size() >= mq8g256_b_bytes(m, k)
            && (w.buf.as_ptr() as usize) % 16 == 0
            && y.dtype == DType::F32
            && y.buf.size() >= n * m * 4
            && self.mq8_acts_current(acts);
        if !ok {
            return Err(shape_error(what));
        }
        self.bind_thread()?;
        let mut first = 0;
        while first < n {
            let r = (n - first).min(MQ8G256_ROWS_EXACT_KERNELS);
            let sym = MQ8G256_B_SYMBOLS[r - 1];
            self.ensure_kernel(MQ8G256_B_MODULE, kernels::GEMV_MQ8G256_B_SRC, sym)?;
            // SAFETY: `first + r <= n`: the offsets stay inside the planes of `acts`.
            let xq_p = unsafe { (acts.xq as *mut u8).add(first * k) } as *mut c_void;
            let xs_p = unsafe { (acts.xs as *mut u8).add(first * 4) } as *mut c_void;
            let y_p = unsafe { (y.buf.as_ptr() as *mut u8).add(first * m * 4) } as *mut c_void;
            self.launch_mq8g256_b(sym, w.buf.as_ptr(), xq_p, xs_p, y_p, m, k, n)?;
            first += r;
        }
        Ok(())
    }

    /// One launch of a layout-B GEMV symbol: `(W, xq, xs, y, M, K, ldxs)`, `ldxs` = the producer's row
    /// count (the scale plane is `[k/256][ldxs]`, `xs` already offset to the first row of the launch).
    #[allow(clippy::too_many_arguments)]
    fn launch_mq8g256_b(
        &mut self,
        sym: &str,
        w_ptr: *mut c_void,
        xq_ptr: *mut c_void,
        xs_ptr: *mut c_void,
        y_ptr: *mut c_void,
        m: usize,
        k: usize,
        ldxs: usize,
    ) -> HipResult<()> {
        let (mut wp, mut xq, mut xs, mut yp) = (w_ptr, xq_ptr, xs_ptr, y_ptr);
        let (mut mv, mut kv, mut lv) = (m as i32, k as i32, ldxs as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut wp as *mut _ as *mut c_void,
            &mut xq as *mut _ as *mut c_void,
            &mut xs as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut lv as *mut _ as *mut c_void,
        ];
        let grid = [m.div_ceil(MQ8G256_B_WAVES_PER_BLOCK) as u32, 1, 1];
        let block = [(32 * MQ8G256_B_WAVES_PER_BLOCK) as u32, 1, 1];
        self.launch_maybe_blob(sym, grid, block, 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(wp);
            b.push_ptr(xq);
            b.push_ptr(xs);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b.push_i32(lv);
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_route_buckets() {
        for n in [1usize, 2] {
            assert_eq!(mq8g256_b_fused_route(512, n).unwrap().waves_per_block, 10);
            assert_eq!(mq8g256_b_fused_route(256, n).unwrap().waves_per_block, 10);
            assert_eq!(mq8g256_b_fused_route(2560, n).unwrap().waves_per_block, 24);
            assert_eq!(mq8g256_b_fused_route(5120, n).unwrap().waves_per_block, 20);
            assert!(mq8g256_b_fused_route(768, n).is_none());
            assert!(mq8g256_b_fused_route(1536, n).is_none());
            assert!(mq8g256_b_fused_route(5376, n).is_none());
        }
        assert_eq!(
            mq8g256_b_fused_route(2560, 2).unwrap().symbol,
            "gemv_mq8g256_bx_r2_u5_w24_p1"
        );
        assert!(mq8g256_b_fused_route(2560, 0).is_none());
        assert!(mq8g256_b_fused_route(2560, 3).is_none());
        assert!(mq8g256_b_fused_route(2561, 1).is_none());
    }

    #[test]
    fn symbol_table_names_match_the_kernel_macro() {
        for (i, sym) in MQ8G256_B_SYMBOLS.iter().enumerate() {
            assert_eq!(*sym, format!("gemv_mq8g256_b_r{}_u1_w4_p1", i + 1));
        }
    }
}
