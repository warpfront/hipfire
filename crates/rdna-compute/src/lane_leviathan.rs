// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-wide sampled verify (`kernels/src/lane_leviathan.hip`): one launch
//! draws every lane's speculative-verify window with Leviathan acceptance
//! (one workgroup per verify row builds `p` under the shared device sampler's
//! law; the last row of each lane to arrive runs the lane's verdict).
//!
//! The caller owns one u32 region laid out `[out | lanes | rows | inputs]`
//! (see [`LaneLeviathanLayout`]) and a p-table scratch of
//! [`LLV_P_WORDS`] floats per row, uploads the region, launches, and reads
//! back the `out` prefix.

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const MODULE: &str = "lane_leviathan";
const FUNC: &str = "lane_leviathan_verify";
const BLOCK: u32 = 512;

/// Most verify rows per lane (drafts + 1): the kernel's `LLV_MAX_ROWS`, a
/// buffer capacity (not a row-count route selector).
pub const LLV_ROWS_CAP: usize = 16;
/// Most `(token, prob)` pairs of one draft's `q`.
pub const LLV_Q_MAX: usize = 64;
/// Most penalty-scope + blocked tokens of one row.
pub const LLV_H_CAP: usize = 1024;

/// Lane record fields (u32 words).
pub const LL_ROW0: usize = 0;
pub const LL_NROWS: usize = 1;
pub const LL_RNG: usize = 2;
pub const LL_EOS: usize = 3;
pub const LL_TEMP: usize = 4;
pub const LL_TOP_P: usize = 5;
pub const LL_TOP_K: usize = 6;
pub const LL_MIN_P: usize = 7;
pub const LL_REP: usize = 8;
pub const LL_PRES: usize = 9;
pub const LL_FREQ: usize = 10;
pub const LL_VOCAB: usize = 11;
pub const LL_DRAFTS: usize = 12;
pub const LL_Q: usize = 13;
pub const LL_BLOCKED: usize = 14;
pub const LL_NBLOCKED: usize = 15;
pub const LL_OUT: usize = 16;
/// Global vocabulary id of the row pointer's element 0 (mode 1 only; the host
/// writes 0 for every other lane).
pub const LL_BASE: usize = 17;
pub const LLV_LANE_WORDS: usize = 20;

/// Row record fields (u32 words).
pub const LR_PTR_LO: usize = 0;
pub const LR_PTR_HI: usize = 1;
pub const LR_LANE: usize = 2;
pub const LR_J: usize = 3;
pub const LR_PEN_OFF: usize = 4;
pub const LR_PEN_LEN: usize = 5;
pub const LLV_ROW_WORDS: usize = 6;

/// One draft's q record: count, then `LLV_Q_MAX` tokens, then their probs.
pub const LLV_Q_WORDS: usize = 1 + 2 * LLV_Q_MAX;
/// p table per row (floats).
pub const LLV_P_WORDS: usize = 4 + 2 * 64;

/// Out record per lane.
pub const LO_TICKET: usize = 0;
pub const LO_DONE: usize = 1;
pub const LO_ACCEPTED: usize = 2;
pub const LO_NTOK: usize = 3;
pub const LO_RNG: usize = 4;
pub const LO_TOKENS: usize = 8;
pub const LLV_OUT_WORDS: usize = 8 + LLV_ROWS_CAP;

/// Candidate row per verify row (modes 1 and 2): `g`, 0, then 64 u64 sort keys
/// (`(~key(value) << 32) | global id`, ascending), in u32 words.
pub const LLV_CAND_WORDS: usize = 2 + 2 * 64;

/// `mode` kernarg: today's whole-row kernel.
const MODE_FULL: i32 = 0;
/// `mode` kernarg: scan the row as a vocabulary shard, publish the candidate row.
const MODE_SELECT: i32 = 1;
/// `mode` kernarg: merge two candidate rows, then today's p table and verdict.
const MODE_MERGED: i32 = 2;

/// "No EOS" marker of [`LL_EOS`].
pub const LLV_NO_EOS: u32 = u32::MAX;

/// Word offsets of a region holding `lanes` lanes and `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneLeviathanLayout {
    pub lanes: usize,
    pub rows: usize,
}

impl LaneLeviathanLayout {
    pub fn out_off(&self, lane: usize) -> usize {
        lane * LLV_OUT_WORDS
    }
    pub fn out_words(&self) -> usize {
        self.lanes * LLV_OUT_WORDS
    }
    pub fn lanes_off(&self) -> usize {
        self.out_words()
    }
    pub fn rows_off(&self) -> usize {
        self.lanes_off() + self.lanes * LLV_LANE_WORDS
    }
    /// First word of the free-form inputs (drafts, q records, scopes, blocked).
    pub fn inputs_off(&self) -> usize {
        self.rows_off() + self.rows * LLV_ROW_WORDS
    }
}

impl Gpu {
    /// Whether the lane-wide verify kernel runs on this device (RDNA4 only:
    /// it is proven and timed on gfx1201).
    pub fn lane_leviathan_supported(&self) -> bool {
        self.arch.starts_with("gfx12")
    }

    /// Enqueue the lane-wide verify over `layout.rows` rows. `region` holds
    /// the uploaded `[out | lanes | rows | inputs]` words (every out record
    /// zeroed); `ptab` holds `layout.rows * LLV_P_WORDS` floats. No readback.
    /// Mode 0: the whole row, every lane's [`LL_BASE`] word 0.
    pub fn lane_leviathan_verify_enqueue(
        &mut self,
        region: &GpuTensor,
        layout: LaneLeviathanLayout,
        ptab: &GpuTensor,
    ) -> HipResult<()> {
        self.lane_leviathan_launch(region, layout, Some(ptab), MODE_FULL, 0, 0)
    }

    /// Mode 1 (select): each row's workgroup scans its row `L[0..LL_VOCAB)` as the
    /// vocabulary shard starting at global id [`LL_BASE`] and writes its sorted
    /// top-g to row `r` of `cand` (`layout.rows * LLV_CAND_WORDS` words). No p
    /// table, no ticket: the `out` records are untouched.
    pub fn lane_leviathan_select_enqueue(
        &mut self,
        region: &GpuTensor,
        layout: LaneLeviathanLayout,
        cand: &GpuTensor,
    ) -> HipResult<()> {
        let cand_ptr = Self::lane_leviathan_cand_ptr(layout, cand, "select")?;
        self.lane_leviathan_launch(region, layout, None, MODE_SELECT, cand_ptr, 0)
    }

    /// Mode 2 (merged): each row's workgroup merges row `r` of `cand_local` with row
    /// `r` of the peer table at device address `cand_peer` (same layout), then runs
    /// today's p table (into `ptab`), ticket and lane verdict.
    pub fn lane_leviathan_merged_enqueue(
        &mut self,
        region: &GpuTensor,
        layout: LaneLeviathanLayout,
        ptab: &GpuTensor,
        cand_local: &GpuTensor,
        cand_peer: u64,
    ) -> HipResult<()> {
        if cand_peer == 0 {
            return Err(HipError::new(
                0,
                "lane_leviathan_merged: null peer candidate table",
            ));
        }
        let cand_ptr = Self::lane_leviathan_cand_ptr(layout, cand_local, "merged")?;
        self.lane_leviathan_launch(region, layout, Some(ptab), MODE_MERGED, cand_ptr, cand_peer)
    }

    fn lane_leviathan_cand_ptr(
        layout: LaneLeviathanLayout,
        cand: &GpuTensor,
        what: &str,
    ) -> HipResult<u64> {
        if layout.rows > 0 && cand.numel() < layout.rows * LLV_CAND_WORDS {
            return Err(HipError::new(
                0,
                &format!(
                    "lane_leviathan_{what}: cand {} words too small for {} rows x {LLV_CAND_WORDS}",
                    cand.numel(),
                    layout.rows
                ),
            ));
        }
        Ok(cand.buf.as_ptr() as u64)
    }

    fn lane_leviathan_launch(
        &mut self,
        region: &GpuTensor,
        layout: LaneLeviathanLayout,
        ptab: Option<&GpuTensor>,
        mode: i32,
        cand_local: u64,
        cand_peer: u64,
    ) -> HipResult<()> {
        if layout.rows == 0 {
            return Ok(());
        }
        if region.numel() < layout.inputs_off()
            || ptab.is_some_and(|p| p.numel() < layout.rows * LLV_P_WORDS)
            || (mode == MODE_SELECT && ptab.is_some())
            || (mode != MODE_SELECT && ptab.is_none())
        {
            return Err(HipError::new(
                0,
                &format!(
                    "lane_leviathan_verify: region {} / ptab {:?} words too small for {:?} (mode {mode})",
                    region.numel(),
                    ptab.map(|p| p.numel()),
                    layout
                ),
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(MODULE, kernels::LANE_LEVIATHAN_SRC, FUNC)?;
        let mut region_ptr = region.buf.as_ptr();
        let mut lanes_off = layout.lanes_off() as i32;
        let mut rows_off = layout.rows_off() as i32;
        let mut ptab_ptr = ptab.map_or(std::ptr::null_mut(), |p| p.buf.as_ptr());
        let mut mode_arg = mode;
        let mut cand_local_arg = cand_local;
        let mut cand_peer_arg = cand_peer;
        let mut params: Vec<*mut c_void> = vec![
            &mut region_ptr as *mut _ as *mut c_void,
            &mut lanes_off as *mut _ as *mut c_void,
            &mut rows_off as *mut _ as *mut c_void,
            &mut ptab_ptr as *mut _ as *mut c_void,
            &mut mode_arg as *mut _ as *mut c_void,
            &mut cand_local_arg as *mut _ as *mut c_void,
            &mut cand_peer_arg as *mut _ as *mut c_void,
        ];
        let func = &self.functions[FUNC];
        unsafe {
            self.hip.launch_kernel(
                func,
                [layout.rows as u32, 1, 1],
                [BLOCK, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every numeric `#define` of the kernel that has a Rust twin here holds
    /// the same value (the region layout is shared by both sides).
    #[test]
    fn layout_matches_the_kernel_defines() {
        let defines: std::collections::HashMap<&str, &str> = kernels::LANE_LEVIATHAN_SRC
            .lines()
            .filter_map(|l| l.strip_prefix("#define "))
            .filter_map(|l| l.split_once(' '))
            .map(|(k, v)| (k, v.trim()))
            .collect();
        let numeric = |name: &str| -> usize {
            defines
                .get(name)
                .unwrap_or_else(|| panic!("kernel lacks #define {name}"))
                .parse()
                .unwrap_or_else(|_| panic!("#define {name} is not a literal"))
        };
        let pairs: &[(&str, usize)] = &[
            ("LLV_MAX_ROWS", LLV_ROWS_CAP),
            ("LLV_Q_MAX", LLV_Q_MAX),
            ("LLV_H_CAP", LLV_H_CAP),
            ("LLV_LANE_WORDS", LLV_LANE_WORDS),
            ("LLV_ROW_WORDS", LLV_ROW_WORDS),
            ("LL_ROW0", LL_ROW0),
            ("LL_NROWS", LL_NROWS),
            ("LL_RNG", LL_RNG),
            ("LL_EOS", LL_EOS),
            ("LL_TEMP", LL_TEMP),
            ("LL_TOP_P", LL_TOP_P),
            ("LL_TOP_K", LL_TOP_K),
            ("LL_MIN_P", LL_MIN_P),
            ("LL_REP", LL_REP),
            ("LL_PRES", LL_PRES),
            ("LL_FREQ", LL_FREQ),
            ("LL_VOCAB", LL_VOCAB),
            ("LL_DRAFTS", LL_DRAFTS),
            ("LL_Q", LL_Q),
            ("LL_BLOCKED", LL_BLOCKED),
            ("LL_NBLOCKED", LL_NBLOCKED),
            ("LL_OUT", LL_OUT),
            ("LL_BASE", LL_BASE),
            ("LR_PTR_LO", LR_PTR_LO),
            ("LR_PTR_HI", LR_PTR_HI),
            ("LR_LANE", LR_LANE),
            ("LR_J", LR_J),
            ("LR_PEN_OFF", LR_PEN_OFF),
            ("LR_PEN_LEN", LR_PEN_LEN),
            ("LO_TICKET", LO_TICKET),
            ("LO_DONE", LO_DONE),
            ("LO_ACCEPTED", LO_ACCEPTED),
            ("LO_NTOK", LO_NTOK),
            ("LO_RNG", LO_RNG),
            ("LO_TOKENS", LO_TOKENS),
        ];
        for &(name, value) in pairs {
            assert_eq!(numeric(name), value, "{name}");
        }
        // The derived strides, as the kernel spells them.
        assert_eq!(defines["LLV_Q_WORDS"], "(1 + 2 * LLV_Q_MAX)");
        assert_eq!(LLV_Q_WORDS, 1 + 2 * LLV_Q_MAX);
        assert_eq!(defines["LLV_P_WORDS"], "(4 + 2 * LLV_G_MAX)");
        assert_eq!(LLV_P_WORDS, 4 + 2 * numeric("LLV_G_MAX"));
        assert_eq!(defines["LLV_OUT_WORDS"], "(8 + LLV_MAX_ROWS)");
        assert_eq!(LLV_OUT_WORDS, 8 + LLV_ROWS_CAP);
        assert_eq!(defines["LLV_CAND_WORDS"], "(2 + 2 * LLV_G_MAX)");
        assert_eq!(LLV_CAND_WORDS, 2 + 2 * numeric("LLV_G_MAX"));
    }
}
