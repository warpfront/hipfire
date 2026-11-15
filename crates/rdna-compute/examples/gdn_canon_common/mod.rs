// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared support for the `gdn_canon_*` S0 kill oracles (design §6 S0).
//! Included by each example with
//! `#[path = "gdn_canon_common/mod.rs"] mod canon;`.
//!
//! * [`q8ef`] — the existing Q8/Q8Ef quantization + F32 recurrence reference
//!   (`src/gdn_q8ef_reference.rs`), compiled into the example unchanged;
//! * [`stream`] — the typed plug-in contract `GdnStreamUnderTest`;
//! * [`model`] — retained-input reference + bounded-tile replay (+ fault
//!   mutants);
//! * [`matrix`] — schedules, runner, bitwise comparison;
//! * [`kills`] — the CPU suite and the plug-in evaluator;
//! * [`util`] — pinned seeds, comparisons, receipts, exit codes;
//! * [`gpu`] — the gfx1201 production scan route (prep → KKT → scan) as a
//!   stream implementation (only compiled by examples that opt in).

#![allow(dead_code)]

#[path = "../../src/gdn_q8ef_reference.rs"]
pub mod q8ef;

pub mod gpu;
pub mod kills;
pub mod matrix;
pub mod model;
pub mod stream;
pub mod util;
