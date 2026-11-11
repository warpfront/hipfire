// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Small pinned helpers shared by the `gdn_canon_*` kill oracles: seeded RNG,
//! bit-exact comparisons, receipt writing.

use serde_json::{json, Value};

pub type R<T> = Result<T, Box<dyn std::error::Error>>;

/// Pinned fixture seed family (CHUNK-CANONICAL-DESIGN §6 S0: "pin fixtures/seeds").
pub const SEED_BASE: u64 = 0x6764_6e43_414e_0000;

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        )
    }
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

pub fn lcg_vec(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.range(lo, hi)).collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn fnv(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Bitwise f32 equality (NaN-safe: equal iff identical bit patterns).
pub fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// First index whose bit patterns differ (length mismatch counts as a diff at
/// the shorter length).
pub fn first_diff_f32(a: &[f32], b: &[f32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

pub fn first_diff_bytes(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Max |a-b| over positions where both are finite; non-finite disagreements
/// are reported separately through `nonfinite_mismatch`.
pub fn maxabs_diff(a: &[f32], b: &[f32]) -> (f64, usize) {
    let mut m = 0.0f64;
    let mut nonfinite = 0usize;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() == y.to_bits() {
            continue;
        }
        if x.is_finite() && y.is_finite() {
            m = m.max((*x as f64 - *y as f64).abs());
        } else {
            nonfinite += 1;
        }
    }
    (m, nonfinite)
}

pub fn count_diff<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Best-effort `git rev-parse HEAD` of the working tree the binary was run in.
pub fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Receipt envelope shared by all four kills. `verdict` is one of
/// `pass` | `fail` (production/candidate arithmetic fails the kill) |
/// `oracle_invalid` (the oracle's own controls misbehaved).
pub fn envelope(oracle: &str, arch: &str, verdict: &str, body: Value) -> Value {
    json!({
        "oracle": oracle,
        "receipt_schema": 1,
        "design": "reliability/CHUNK-CANONICAL-DESIGN.md §6 S0",
        "git_head": git_head(),
        "arch": arch,
        "verdict": verdict,
        "result": body,
    })
}

pub fn write_receipt(path: &Option<String>, receipt: &Value) -> R<()> {
    let text = serde_json::to_string_pretty(receipt)?;
    if let Some(p) = path {
        if let Some(parent) = std::path::Path::new(p).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(p, &text)?;
    }
    Ok(())
}

/// Exit codes shared by all kills: 0 = every check passes; 1 = the code under
/// test fails the kill (fail-before demonstrated); 2 = the oracle's own
/// controls are invalid (a correct model failed or a fault mutant escaped);
/// 3 = environment / unsupported configuration.
pub const EXIT_PASS: i32 = 0;
pub const EXIT_KILL_FAILS: i32 = 1;
pub const EXIT_ORACLE_INVALID: i32 = 2;
pub const EXIT_ENV: i32 = 3;

/// Parse `--flag value` style CLIs into a map-like lookup.
pub struct Args {
    pub rest: Vec<String>,
}

impl Args {
    pub fn from_env() -> Self {
        Self {
            rest: std::env::args().skip(1).collect(),
        }
    }
    pub fn value(&self, flag: &str) -> Option<String> {
        self.rest
            .iter()
            .position(|a| a == flag)
            .and_then(|i| self.rest.get(i + 1).cloned())
    }
    pub fn has(&self, flag: &str) -> bool {
        self.rest.iter().any(|a| a == flag)
    }
    pub fn csv_usize(&self, flag: &str) -> R<Option<Vec<usize>>> {
        match self.value(flag) {
            None => Ok(None),
            Some(s) => Ok(Some(
                s.split(',')
                    .map(|x| x.trim().parse())
                    .collect::<Result<_, _>>()?,
            )),
        }
    }
}
