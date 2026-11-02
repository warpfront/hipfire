// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 (arch 16, Flash-Next) Tier C cases: element-wise GPU vs CPU reference
//! for every portable kernel the Qwen4 C1 forward launches, at Qwen4 shapes
//! (hidden 2560, GDN 16 key / 48 value heads × 128, QSA 24 q / 2 kv / head_dim
//! 256, HC 4 streams, router top-10 of 512).
//!
//! Included by `test_kernelsQA` (`#[path]` module); every case runs in its own
//! supervised subprocess there. Tolerances are the Qwen4 format's frozen ones
//! (`hipfire-arch-qwen4/reference_oracle/upstream.py:275-276`), or bit-exact
//! where the CPU twin reproduces the kernel's operation order.
//!
//! Census of record: `subq4/review/G942-ROUTE-CENSUS.md`.

#![allow(dead_code)]

pub mod common;
mod gdn;
mod hc;
mod io;
mod moe;
mod qsa;
mod trunk;

use rdna_compute::Gpu;

/// One Qwen4 Tier C case. `run` returns `Ok(summary)` on PASS (summary carries
/// the measured max-abs / ulp data) and `Err(detail)` on FAIL.
pub struct Q4Case {
    pub name: &'static str,
    pub timeout_s: u64,
    pub run: fn(&mut Gpu) -> Result<String, String>,
}

/// Every Qwen4 case: route cases in census order, then the `_probe_` cases
/// (direct launches of kernels the gfx942 route cannot reach) last.
pub fn cases() -> impl Iterator<Item = &'static Q4Case> {
    let all = || {
        io::CASES
            .iter()
            .chain(hc::CASES.iter())
            .chain(gdn::CASES.iter())
            .chain(qsa::CASES.iter())
            .chain(moe::CASES.iter())
            .chain(trunk::CASES.iter())
    };
    all()
        .filter(|c| !c.name.contains("_probe_"))
        .chain(all().filter(|c| c.name.contains("_probe_")))
}

pub fn find(name: &str) -> Option<&'static Q4Case> {
    cases().find(|case| case.name == name)
}
