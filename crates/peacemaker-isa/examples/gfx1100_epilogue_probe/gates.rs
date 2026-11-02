// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single definition of the gfx1100 epilogue-probe input sets. It is
//! `#[path]`-included by BOTH the probe generator (peacemaker-isa example
//! `gfx1100_epilogue_intrinsic_probe`) and the observer (hipfire-runtime example
//! `gfx1100_epilogue_intrinsic_observe`), and each side hashes it, so the code object,
//! its build manifest and the observation table are bound to the same input lists.
//!
//! Inputs are the raw f32 bits `g` handed to `silu_dense` (the gate AFTER the epilogue's
//! `round_f32` input rounding, so a finite gate has a cleared low half).
//!
//! * `finite_bf16`: every gate `hi << 16`, `hi` in `0..=0xffff` with a biased exponent
//!   (`(hi >> 7) & 0xff`) other than 255 — both signs, zeros and BF16 subnormals included.
//!   Count 65,280. Ascending `hi`.
//! * `exceptional`: full-F32 cases outside the finite-BF16 domain plus the special values the
//!   CPU twin documents as finite-domain-only: `+-0`, `+-inf`, every BF16-representable NaN
//!   (`hi` in `0x7f81..=0x7fff` / `0xff81..=0xffff`), full-payload NaNs (each single payload
//!   bit, each single payload bit with the quiet bit set, all-ones), full-F32 subnormals
//!   (each single bit, all-ones, 1024 deterministic xorshift32 samples per sign) and
//!   `+-f32::MAX`. Zeros and the BF16-representable specials intentionally repeat inputs of
//!   `finite_bf16` (a determinism cross-check); duplicates inside this list are dropped.

use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const SCHEMA: &str = "consumer24.gfx1100.epilogue-probe.v1";

pub fn finite_bf16() -> Vec<u32> {
    (0u32..=0xffff)
        .filter(|hi| (hi >> 7) & 0xff != 0xff)
        .map(|hi| hi << 16)
        .collect()
}

pub fn exceptional() -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut push = |bits: u32| {
        if seen.insert(bits) {
            out.push(bits)
        }
    };
    for bits in [0x0000_0000u32, 0x8000_0000, 0x7f80_0000, 0xff80_0000] {
        push(bits)
    }
    for sign in [0u32, 0x8000_0000] {
        for hi in 0x7f81u32..=0x7fff {
            push((hi << 16) | sign)
        }
        for k in 0..23 {
            push(0x7f80_0000 | (1 << k) | sign)
        }
        for k in 0..22 {
            push(0x7fc0_0000 | (1 << k) | sign)
        }
        push(0x7fff_ffff | sign);
        push(0x7f7f_ffff | sign);
        for k in 0..23 {
            push((1 << k) | sign)
        }
        push(0x007f_ffff | sign);
        let mut x = 0x9e37_79b9u32 ^ sign;
        for _ in 0..1024 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let m = x & 0x007f_ffff;
            if m != 0 {
                push(m | sign)
            }
        }
    }
    out
}

/// SHA-256 over the little-endian u32 stream of `gates`.
pub fn list_sha256(gates: &[u32]) -> String {
    let mut h = Sha256::new();
    for g in gates {
        h.update(g.to_le_bytes())
    }
    format!("{:x}", h.finalize())
}
