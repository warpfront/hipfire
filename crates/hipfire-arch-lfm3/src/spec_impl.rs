// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LFM3 (arch_id 24) GPU bundle — the (config, weights, state, eos_tok) quadruple
//! the loader folds into the loader's LFM3 bundle registration and the daemon's `arch_id == 24`
//! generate branch drives.
//!
//! Phase 1 is AR-only (no spec-decode target), so — unlike `Lfm2MoeBundle` —
//! this does not yet impl `SpecTarget`. When n-gram / DFlash spec decode lands
//! for LFM3, the `SpecTarget` impl (conv-state snapshot/rollback, etc.) goes
//! here next to the forward it drives, mirroring `hipfire-arch-lfm2moe`'s
//! `spec_impl.rs`.

// NOTE: the `Lfm3Bundle` that the loader stores in `LoadedModel.state` now
// lives in `crates/hipfire-loader` (mirroring `MuseGlimmerBundle`) because it
// carries the optional VL / audio / omni tower state, whose types live in the
// `hipfire-arch-lfm3-{vl,audio,omni}` crates that this crate must not depend
// on. A future `SpecTarget` impl for LFM3 still belongs here, next to the
// forward it drives (orphan rule).
