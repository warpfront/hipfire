// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `dflash_selector_greedy_exact` launcher (Design A §4): the DFlash2 greedy
//! candidate selector on the device, byte-identical to the CPU scorer
//! `propose_candidates_from_rows` at temperature 0. Kernel contract and the
//! status word are documented in `kernels/src/dflash_selector_greedy_exact.hip`.

use crate::dispatch::Gpu;
use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};
use std::ffi::c_void;

pub const DFLASH_SELECTOR_GREEDY_EXACT_SRC: &str =
    include_str!("../../../kernels/src/dflash_selector_greedy_exact.hip");
pub const DFLASH_SELECTOR_GREEDY_EXACT_MODULE: &str = "dflash_selector_greedy_exact";
pub const DFLASH_SELECTOR_GREEDY_EXACT_SYMBOL: &str = "dflash_selector_greedy_exact";
/// One wave32.
pub const DFLASH_SELECTOR_GREEDY_EXACT_BLOCK: u32 = 32;
/// Largest `rows`: block holds the seed plus one id per row (16 words).
pub const DFLASH_SELECTOR_MAX_ROWS: usize = 15;
pub const DFLASH_SELECTOR_MAX_K: usize = 16;
/// `status` word: 0 ok, else `code | (row + 1) << 8`; see the kernel header.
pub const DFLASH_SELECTOR_STATUS_OK: u32 = 0;

/// Device operands of one selector launch. Extents are validated by
/// [`Gpu::dflash_selector_greedy_exact`] before the launch.
pub struct DflashSelectorGreedyArgs<'a> {
    /// `[vocab, rank]`, f16 bit patterns when `pred_f16`, else f32.
    pub pred_cb: &'a DeviceBuffer,
    pub succ_cb: &'a DeviceBuffer,
    pub pred_f16: bool,
    pub succ_f16: bool,
    /// `[rows, rank]` f32.
    pub proj: &'a DeviceBuffer,
    /// `[rows, k]` u32 / f32.
    pub top_ids: &'a DeviceBuffer,
    pub top_vals: &'a DeviceBuffer,
    /// One u32 word: the block's first token.
    pub seed: &'a DeviceBuffer,
    /// Outputs: `[rows, k]` f32, `[rows + 1]` u32 (block[0] = seed), one u32.
    pub scores: &'a DeviceBuffer,
    pub block: &'a DeviceBuffer,
    pub status: &'a DeviceBuffer,
    pub rows: usize,
    pub k: usize,
    pub rank: usize,
    pub vocab: usize,
}

fn need(name: &str, buf: &DeviceBuffer, bytes: usize) -> HipResult<()> {
    if buf.size() < bytes {
        return Err(HipError::new(
            0,
            &format!(
                "dflash_selector_greedy_exact: {name} holds {} bytes, needs {bytes}",
                buf.size()
            ),
        ));
    }
    Ok(())
}

impl Gpu {
    pub fn ensure_dflash_selector_greedy_exact(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            DFLASH_SELECTOR_GREEDY_EXACT_MODULE,
            DFLASH_SELECTOR_GREEDY_EXACT_SRC,
            DFLASH_SELECTOR_GREEDY_EXACT_SYMBOL,
        )
    }

    /// Enqueue the selector on the default stream. Validates shapes and every
    /// buffer extent first; malformed token ids are caught on the device and
    /// reported through `status`.
    pub fn dflash_selector_greedy_exact(
        &mut self,
        a: &DflashSelectorGreedyArgs<'_>,
    ) -> HipResult<()> {
        let (rows, k, rank, vocab) = (a.rows, a.k, a.rank, a.vocab);
        if rows == 0 || rows > DFLASH_SELECTOR_MAX_ROWS {
            return Err(HipError::new(
                0,
                "dflash_selector_greedy_exact: rows not in 1..=15",
            ));
        }
        if k == 0 || k > DFLASH_SELECTOR_MAX_K {
            return Err(HipError::new(
                0,
                "dflash_selector_greedy_exact: k not in 1..=16",
            ));
        }
        if rank == 0 || vocab == 0 || vocab > u32::MAX as usize || rank > u32::MAX as usize {
            return Err(HipError::new(
                0,
                "dflash_selector_greedy_exact: bad rank/vocab",
            ));
        }
        let cb_elems = vocab.checked_mul(rank).ok_or_else(|| {
            HipError::new(0, "dflash_selector_greedy_exact: codebook extent overflow")
        })?;
        let esz = |f16: bool| if f16 { 2usize } else { 4 };
        need(
            "pred_cb",
            a.pred_cb,
            cb_elems.saturating_mul(esz(a.pred_f16)),
        )?;
        need(
            "succ_cb",
            a.succ_cb,
            cb_elems.saturating_mul(esz(a.succ_f16)),
        )?;
        need("proj", a.proj, rows * rank * 4)?;
        need("top_ids", a.top_ids, rows * k * 4)?;
        need("top_vals", a.top_vals, rows * k * 4)?;
        need("seed", a.seed, 4)?;
        need("scores", a.scores, rows * k * 4)?;
        need("block", a.block, (rows + 1) * 4)?;
        need("status", a.status, 4)?;
        self.bind_thread()?;
        self.ensure_dflash_selector_greedy_exact()?;

        let ptr = |b: &DeviceBuffer| b.as_ptr() as *const c_void;
        let (pc, sc, pj, ti, tv, sd, sco, bl, st) = (
            ptr(a.pred_cb),
            ptr(a.succ_cb),
            ptr(a.proj),
            ptr(a.top_ids),
            ptr(a.top_vals),
            ptr(a.seed),
            ptr(a.scores),
            ptr(a.block),
            ptr(a.status),
        );
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7, mut p8) =
            (pc, sc, pj, ti, tv, sd, sco, bl, st);
        let (mut u_rows, mut u_k, mut u_rank, mut u_vocab) =
            (rows as u32, k as u32, rank as u32, vocab as u32);
        let (mut u_pf, mut u_sf) = (a.pred_f16 as u32, a.succ_f16 as u32);
        let mut params: Vec<*mut c_void> = vec![
            &mut p0 as *mut _ as *mut c_void,
            &mut p1 as *mut _ as *mut c_void,
            &mut p2 as *mut _ as *mut c_void,
            &mut p3 as *mut _ as *mut c_void,
            &mut p4 as *mut _ as *mut c_void,
            &mut p5 as *mut _ as *mut c_void,
            &mut p6 as *mut _ as *mut c_void,
            &mut p7 as *mut _ as *mut c_void,
            &mut p8 as *mut _ as *mut c_void,
            &mut u_rows as *mut _ as *mut c_void,
            &mut u_k as *mut _ as *mut c_void,
            &mut u_rank as *mut _ as *mut c_void,
            &mut u_vocab as *mut _ as *mut c_void,
            &mut u_pf as *mut _ as *mut c_void,
            &mut u_sf as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            DFLASH_SELECTOR_GREEDY_EXACT_SYMBOL,
            [1, 1, 1],
            [DFLASH_SELECTOR_GREEDY_EXACT_BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                for p in [pc, sc, pj, ti, tv, sd, sco, bl, st] {
                    b.push_ptr(p);
                }
                for v in [
                    rows as u32,
                    k as u32,
                    rank as u32,
                    vocab as u32,
                    a.pred_f16 as u32,
                    a.succ_f16 as u32,
                ] {
                    b.push_u32(v);
                }
                b
            },
        )
    }
}
