// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 (arch_id 13) GPU bundle — the (config, weights, state, eos_tok) quadruple
//! the loader folds into `ModelState::Lfm3` and the daemon's `arch_id == 13`
//! generate branch drives.
//!
//! Phase 1 is AR-only (no spec-decode target), so — unlike `Lfm2MoeBundle` —
//! this does not yet impl `SpecTarget`. When n-gram / DFlash spec decode lands
//! for LFM3, the `SpecTarget` impl (conv-state snapshot/rollback, etc.) goes
//! here next to the forward it drives, mirroring `hipfire-arch-lfm2moe`'s
//! `spec_impl.rs`.

use crate::config::Lfm3Config;
use crate::lfm3::{Lfm3State, Lfm3Weights};

/// LFM3 (arch_id=13) GPU bundle. `eos_tok` is resolved at load time and rides
/// along so the generate path doesn't re-tokenize.
pub struct Lfm3Bundle {
    pub config: Lfm3Config,
    pub weights: Lfm3Weights,
    pub state: Lfm3State,
    pub eos_tok: u32,
}
