// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6d: re-gridded row selectors (gfx1201, gfx1151, gfx1100).
//!
//! `topk_values_batched_f32` (DFlash2 selector, raw top-K) and
//! `argmax_f32_batched` (verify argmax) launch one 256-thread block per row;
//! the top-K merges 256 x K candidates on a single thread. On exact gfx1201,
//! gfx1151 and gfx1100 (opt out with `HIPFIRE_SELECT_REGRID_OFF=1`) both route through
//! `kernels/src/select_regrid.hip` instead: 1024 threads per row with wave32
//! butterflies. Outputs are byte-identical to the shipping kernels for every
//! input (see the kernel header for the argument; `test_select_regrid`
//! checks it, ties and NaNs included). Top-K rows whose leading K+1 values
//! tie are recomputed by `topk_values_tie_exact_f32`, which reproduces the
//! shipping kernel body's output byte for byte (same 256-thread candidate
//! layout and tie order) with a block-parallel merge instead of the shipping
//! single-thread one (1.6 ms -> tens of microseconds on a 260K vocabulary).
//! `topk_values_fixup_f32` (the shipping body behind a marker check) stays
//! defined: the published kernel pack pins its source.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipResult, KernargBlob};
use std::ffi::c_void;

/// Re-grid kernel source (both entries).
pub const SELECT_REGRID_SRC: &str = include_str!("../../../kernels/src/select_regrid.hip");
/// Compiled-module key for [`SELECT_REGRID_SRC`].
pub const SELECT_REGRID_MODULE: &str = "select_regrid";
/// Top-K re-grid entry.
pub const TOPK_VALUES_REGRID_SYMBOL: &str = "topk_values_regrid_f32";
/// Argmax re-grid entry.
pub const ARGMAX_REGRID_SYMBOL: &str = "argmax_f32_batched_regrid";
/// Shipping top-K body entered only for rows the re-grid marked (tie).
pub const TOPK_VALUES_FIXUP_SRC: &str = concat!(
    "#define HIPFIRE_TOPK_FIXUP_MARKED 1\n",
    "#define topk_logsumexp_batched_f32 topk_values_fixup_f32\n",
    include_str!("../../../kernels/src/topk_logsumexp_batched.hip")
);
/// Tie-row recompute with a block-parallel merge, byte-identical to
/// [`TOPK_VALUES_FIXUP_SRC`] (the kernel header gives the argument; the unit
/// tests below check it against a CPU model of both, and the `#[ignore]` GPU
/// test `topk_tie_exact_gpu` against the device).
pub const TOPK_VALUES_TIE_SRC: &str =
    include_str!("../../../kernels/src/topk_values_tie_exact.hip");
/// Module key of [`TOPK_VALUES_TIE_SRC`].
pub const TOPK_VALUES_TIE_MODULE: &str = "topk_values_tie_exact";
/// Symbol of [`TOPK_VALUES_TIE_SRC`].
pub const TOPK_VALUES_TIE_SYMBOL: &str = "topk_values_tie_exact_f32";
/// Threads per row block of the tie-row recompute (the shipping layout).
pub const TOPK_TIE_BLOCK: u32 = 256;
/// Module key and symbol of [`TOPK_VALUES_FIXUP_SRC`].
pub const TOPK_VALUES_FIXUP_SYMBOL: &str = "topk_values_fixup_f32";
/// Threads per row block for both re-grid kernels.
pub const SELECT_REGRID_BLOCK: u32 = 1024;

impl Gpu {
    /// Exact gfx1201, gfx1151 or gfx1100 (the archs `test_select_regrid` and
    /// `topk_tie_exact_gpu` byte-check; all three are wave32 and the kernels
    /// use only `__shfl_xor` and 1024-thread blocks), not opted out, and no
    /// open Redline recording (the re-grid kernels are not in Redline's kernel
    /// tables).
    pub fn select_regrid_enabled(&self) -> bool {
        (self.arch_caps.is_gfx1201() || self.arch_caps.is_gfx1151() || self.arch_caps.is_gfx1100())
            && !self.flags.select_regrid_off
            && !self.replay.is_recording()
    }

    /// Raw top-K over `[b x vocab]` rows: re-grid launch, then the fixup
    /// launch (a no-op for unmarked rows). Same stream, arguments and output
    /// contract as the shipping raw launch; `k` in `1..=16`.
    pub(crate) fn topk_values_regrid_f32(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_values: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            SELECT_REGRID_MODULE,
            SELECT_REGRID_SRC,
            TOPK_VALUES_REGRID_SYMBOL,
        )?;
        self.ensure_kernel(
            TOPK_VALUES_TIE_MODULE,
            TOPK_VALUES_TIE_SRC,
            TOPK_VALUES_TIE_SYMBOL,
        )?;
        let mut lp = logits.buf.as_ptr();
        let mut ti = top_idx.buf.as_ptr();
        let mut to = top_values.buf.as_ptr();
        let mut vs = vocab as i32;
        let mut kk = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut to as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        self.launch_params_or_plan(
            TOPK_VALUES_REGRID_SYMBOL,
            [b as u32, 1, 1],
            [SELECT_REGRID_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut bl = KernargBlob::new();
                bl.push_ptr(lp);
                bl.push_ptr(ti);
                bl.push_ptr(to);
                bl.push_i32(vs);
                bl.push_i32(kk);
                bl
            },
        )?;
        // Tie rows only (the kernel returns at once for unmarked rows): the
        // shipping candidate layout, merged block-parallel.
        let mut params: Vec<*mut c_void> = vec![
            &mut lp as *mut _ as *mut c_void,
            &mut ti as *mut _ as *mut c_void,
            &mut to as *mut _ as *mut c_void,
            &mut vs as *mut _ as *mut c_void,
            &mut kk as *mut _ as *mut c_void,
        ];
        self.launch_params_or_plan(
            TOPK_VALUES_TIE_SYMBOL,
            [b as u32, 1, 1],
            [TOPK_TIE_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut bl = KernargBlob::new();
                bl.push_ptr(lp);
                bl.push_ptr(ti);
                bl.push_ptr(to);
                bl.push_i32(vs);
                bl.push_i32(kk);
                bl
            },
        )
    }

    /// Batched argmax over `[batch x n]` rows with the shipping kernel's
    /// exact tie order, through `launch_maybe_blob` like the shipping launch.
    pub(crate) fn argmax_f32_batched_regrid(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.ensure_kernel(
            SELECT_REGRID_MODULE,
            SELECT_REGRID_SRC,
            ARGMAX_REGRID_SYMBOL,
        )?;
        let dp = data.buf.as_ptr();
        let rp = result.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            ARGMAX_REGRID_SYMBOL,
            [batch_size as u32, 1, 1],
            [SELECT_REGRID_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(rp);
                b.push_i32(nn);
                b
            },
        )
    }

    /// The re-grid + tie-recompute raw top-K on ANY arch, bypassing
    /// [`Self::select_regrid_enabled`] (parity gates only: lets
    /// `topk_tie_exact_gpu` byte-compare the re-grid route against the shipping
    /// kernel on an arch before that arch is admitted). Same contract as the
    /// routed launch; `k` in `1..=16`.
    pub fn topk_values_regrid_f32_forced(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_values: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!((1..=16).contains(&k), "topk: K={k} must be in [1,16]");
        self.topk_values_regrid_f32(logits, top_idx, top_values, vocab, k, b)
    }

    /// The re-grid batched argmax on ANY arch, bypassing
    /// [`Self::select_regrid_enabled`] (parity gates only).
    pub fn argmax_f32_batched_regrid_forced(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.argmax_f32_batched_regrid(data, result, n, batch_size)
    }

    /// The shipping raw top-K launch (kept callable for the parity gate).
    pub fn topk_values_batched_f32_shipping(
        &mut self,
        logits: &GpuTensor,
        top_idx: &GpuTensor,
        top_values: &GpuTensor,
        vocab: usize,
        k: usize,
        b: usize,
    ) -> HipResult<()> {
        self.launch_topk_batched_f32_shipping(logits, top_idx, top_values, vocab, k, b, true)
    }

    /// The shipping batched argmax launch (kept callable for the parity gate).
    pub fn argmax_f32_batched_shipping(
        &mut self,
        data: &GpuTensor,
        result: &GpuTensor,
        n: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "argmax_f32_batched",
            kernels::ARGMAX_BATCHED_SRC,
            "argmax_f32_batched",
        )?;
        let dp = data.buf.as_ptr();
        let rp = result.buf.as_ptr();
        let nn = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &nn as *const _ as *mut c_void,
        ];
        let block_size = 256u32;
        let shared = block_size * 8; // f32 + i32 per thread
        self.launch_maybe_blob(
            "argmax_f32_batched",
            [batch_size as u32, 1, 1],
            [block_size, 1, 1],
            shared,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(dp);
                b.push_ptr(rp);
                b.push_i32(nn);
                b
            },
        )
    }
}

#[cfg(test)]
mod tie_exact_tests {
    //! CPU models of the shipping top-K body (`topk_logsumexp_batched_f32`, raw
    //! mode: per-thread K best with replace-first-minimum, then one thread
    //! inserting the candidates in order `c = t * K + slot` with a strict `>`)
    //! and of `topk_values_tie_exact_f32` (same per-thread lists, K block-wide
    //! rounds over the total order (value desc, c asc)). They must agree on
    //! every input, ties, `-inf`, `+inf` and NaN included.
    const NTH: usize = 256;

    fn thread_lists(row: &[f32], k: usize, tracked_min: bool) -> Vec<(Vec<f32>, Vec<i32>)> {
        (0..NTH)
            .map(|t| {
                let mut val = vec![f32::NEG_INFINITY; k];
                let mut idx = vec![0i32; k];
                let first_min = |val: &[f32]| {
                    let (mut mj, mut mv) = (0usize, val[0]);
                    for j in 1..k {
                        if val[j] < mv {
                            mv = val[j];
                            mj = j;
                        }
                    }
                    (mj, mv)
                };
                let (mut min_j, mut min_v) = first_min(&val);
                let mut i = t;
                while i < row.len() {
                    let v = row[i];
                    if !tracked_min {
                        (min_j, min_v) = first_min(&val);
                    }
                    if v > min_v {
                        val[min_j] = v;
                        idx[min_j] = i as i32;
                        if tracked_min {
                            (min_j, min_v) = first_min(&val);
                        }
                    }
                    i += NTH;
                }
                (val, idx)
            })
            .collect()
    }

    /// The shipping kernel: serial insertion merge.
    fn shipping(row: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
        let lists = thread_lists(row, k, false);
        let mut fv = vec![f32::NEG_INFINITY; k];
        let mut fi = vec![0i32; k];
        for (val, idx) in &lists {
            for kk in 0..k {
                let (v, i) = (val[kk], idx[kk]);
                let mut ins = k;
                for j in (0..k).rev() {
                    if v > fv[j] {
                        ins = j;
                    }
                }
                if ins < k {
                    for j in (ins + 1..k).rev() {
                        fv[j] = fv[j - 1];
                        fi[j] = fi[j - 1];
                    }
                    fv[ins] = v;
                    fi[ins] = i;
                }
            }
        }
        (fi, fv)
    }

    /// The tie-exact kernel: tracked minimum, K selection rounds.
    fn rounds(row: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
        let mut lists = thread_lists(row, k, true);
        let (mut oi, mut ov) = (Vec::new(), Vec::new());
        for _ in 0..k {
            let mut best: Option<(f32, usize, usize)> = None; // (value, thread, slot)
            for (t, (val, _)) in lists.iter().enumerate() {
                let mut bv = f32::NEG_INFINITY;
                let mut bj = None;
                for j in 0..k {
                    if val[j] > bv {
                        bv = val[j];
                        bj = Some(j);
                    }
                }
                if let Some(j) = bj {
                    let better = match best {
                        None => true,
                        Some((v, _, _)) => bv > v,
                    };
                    // Threads are visited in ascending order: an equal value keeps the
                    // earlier thread.
                    if better {
                        best = Some((bv, t, j));
                    }
                }
            }
            match best {
                Some((v, t, j)) => {
                    ov.push(v);
                    oi.push(lists[t].1[j]);
                    lists[t].0[j] = f32::NEG_INFINITY;
                }
                None => {
                    ov.push(f32::NEG_INFINITY);
                    oi.push(0);
                }
            }
        }
        (oi, ov)
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        }
    }

    fn bf16(v: f32) -> f32 {
        f32::from_bits(v.to_bits() & 0xffff_0000)
    }

    fn assert_same(row: &[f32], what: &str) {
        for k in 1..=16 {
            let (si, sv) = shipping(row, k);
            let (ri, rv) = rounds(row, k);
            assert_eq!(si, ri, "{what} K={k}: indices");
            let sb: Vec<u32> = sv.iter().map(|v| v.to_bits()).collect();
            let rb: Vec<u32> = rv.iter().map(|v| v.to_bits()).collect();
            assert_eq!(sb, rb, "{what} K={k}: values");
        }
    }

    #[test]
    fn rounds_match_the_serial_merge_on_tie_free_rows() {
        let mut g = Lcg(1);
        let row: Vec<f32> = (0..5000).map(|_| g.unit() * 8.0).collect();
        assert_same(&row, "random");
    }

    #[test]
    fn rounds_match_the_serial_merge_on_tie_heavy_rows() {
        let mut g = Lcg(2);
        // bf16-valued logits tie constantly among the leading values.
        for (vocab, scale) in [
            (300usize, 2.0f32),
            (4096, 6.0),
            (20_000, 12.0),
            (259_840, 20.0),
        ] {
            let row: Vec<f32> = (0..vocab).map(|_| bf16(g.unit() * scale)).collect();
            assert_same(&row, &format!("bf16 vocab {vocab}"));
        }
        // A handful of distinct levels: ties everywhere.
        let row: Vec<f32> = (0..9000).map(|_| (g.next() % 5) as f32).collect();
        assert_same(&row, "five levels");
        // Every element equal.
        assert_same(&vec![1.5f32; 7777], "constant");
    }

    #[test]
    fn rounds_match_the_serial_merge_on_degenerate_rows() {
        let mut g = Lcg(3);
        // Fewer finite values than K, the rest -inf.
        let mut row = vec![f32::NEG_INFINITY; 1000];
        for slot in [3usize, 400, 999, 17] {
            row[slot] = g.unit();
        }
        assert_same(&row, "few finite");
        // All -inf, and a vocabulary shorter than the thread count.
        assert_same(&vec![f32::NEG_INFINITY; 600], "all -inf");
        let short: Vec<f32> = (0..10).map(|_| bf16(g.unit())).collect();
        assert_same(&short, "vocab 10");
        // NaN and +inf are handled like the shipping scan (NaN never selected).
        let mut row: Vec<f32> = (0..3000).map(|_| bf16(g.unit() * 3.0)).collect();
        row[5] = f32::NAN;
        row[777] = f32::INFINITY;
        row[778] = f32::INFINITY;
        row[1500] = f32::NAN;
        assert_same(&row, "nan and inf");
    }
}
