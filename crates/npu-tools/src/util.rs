// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Small helpers shared by the npu-tools binaries.
use std::time::Duration;

/// 64-bit LCG (Knuth MMIX constants) yielding the top byte as i8: the deterministic operand generator every GEMM driver uses.
pub fn lcg(seed: &mut u64) -> i8 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as i8
}

/// Nearest-rank percentile of an ascending-sorted, non-empty slice (`p` in 0..=1).
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

pub fn parse_num<T: std::str::FromStr>(name: &str, s: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("{name} expects a number, got {s:?}"))
}

pub fn round_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

pub fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}
