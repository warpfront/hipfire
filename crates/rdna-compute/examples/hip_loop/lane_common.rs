// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Shared pieces of the two gfx1201 lane nodes (`lane_proj.rs`, `lane_wo.rs`): MQ4G256V2 weight
//! synthesis (the real 136-byte group layout, the weight kinds of the product gates
//! `tests/lane_proj.rs` / `tests/a3b_lane_wo.rs`), adversarial activation kinds, the word-wise
//! comparison policies, the cold-cache weight pool sizing, arm loading/launching for a node's own
//! device-side identity matrix, and the matrix report.
//!
//! Included with `#[path = "hip_loop/lane_common.rs"] mod lane_common;`.
#![allow(dead_code)]

use crate::node::*;
use hip_bridge::Function;
use serde_json::{json, Value};

/// Bytes of one MQ4G256V2 group: `[sc0:f16][zp0:f16][sc1:f16][zp1:f16][128 B nibbles]`.
pub const GROUP_BYTES: usize = 136;
/// Output pre-poison of the product gates (a quiet NaN with a recognisable payload).
pub const POISON: u32 = 0x7FC0_DEAD;
/// The most lanes either kernel supports.
pub const MAX_LANES: usize = 8;
/// The cold-cache pool and flush sizes live in `support/cold.rs` (shared with `pm_loop`).
pub use crate::cold::{pool_sets, POOL_BYTES};
/// WGP count assumed offline (R9700); the device attribute must agree on the card.
pub const WGP_COUNT: usize = 32;

// ---------------------------------------------------------------------------
// Weights

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WKind {
    /// Scales ~2^-9..2^-7 (dequantised values ~1e-2): the timing/emulator weights.
    Small,
    /// Scales ~2^-1..2^1.
    Large,
    /// Fully random header bytes (inf / nan / denormal scales and zeros).
    Wild,
    /// Scales 2^-7..2^-2, signed zero points (the `wo` product gate's `Normal`).
    Normal,
    /// Scales spanning 2^-14..2^12 (rounding at every magnitude).
    Wide,
    /// Nibbles all 0 (normal headers).
    AllZero,
    /// Nibbles all 15 (normal headers).
    AllFifteen,
}

impl WKind {
    pub fn name(self) -> &'static str {
        match self {
            WKind::Small => "small",
            WKind::Large => "large",
            WKind::Wild => "wild",
            WKind::Normal => "normal",
            WKind::Wide => "wide",
            WKind::AllZero => "all0",
            WKind::AllFifteen => "all15",
        }
    }
}

/// A normal f16 with a biased exponent in `lo..=hi`.
fn f16_normal(r: &mut Xs64, lo: u32, hi: u32, signed: bool) -> u16 {
    let exp = lo + (r.next_u64() >> 33) as u32 % (hi - lo + 1);
    let mant = (r.next_u64() >> 40) as u32 & 0x3FF;
    let sign = if signed {
        (r.next_u64() >> 41) as u32 & 1
    } else {
        0
    };
    ((sign << 15) | (exp << 10) | mant) as u16
}

/// `rows` MQ4G256V2 rows of `groups` groups in the real layout, deterministic in `seed`.
pub fn synth_weights(rows: usize, groups: usize, seed: u64, kind: WKind) -> Vec<u8> {
    let mut r = Xs64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut out = vec![0u8; rows * groups * GROUP_BYTES];
    for g in 0..rows * groups {
        let off = g * GROUP_BYTES;
        for half in 0..2 {
            let (sc, zp) = match kind {
                WKind::Small => (
                    f16_normal(&mut r, 6, 8, false),
                    f16_normal(&mut r, 5, 8, true),
                ),
                WKind::Large => (
                    f16_normal(&mut r, 13, 16, false),
                    f16_normal(&mut r, 12, 16, true),
                ),
                WKind::Normal | WKind::AllZero | WKind::AllFifteen => (
                    f16_normal(&mut r, 8, 13, false),
                    f16_normal(&mut r, 7, 13, true),
                ),
                WKind::Wide => (
                    f16_normal(&mut r, 1, 26, false),
                    f16_normal(&mut r, 1, 26, true),
                ),
                WKind::Wild => ((r.next_u64() >> 32) as u16, (r.next_u64() >> 32) as u16),
            };
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        let payload = &mut out[off + 8..off + GROUP_BYTES];
        match kind {
            WKind::AllZero => {}
            WKind::AllFifteen => payload.fill(0xFF),
            _ => {
                for c in payload.chunks_exact_mut(8) {
                    c.copy_from_slice(&r.next_u64().to_le_bytes());
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Words

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn word_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn words(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn poison_words(n: usize) -> Vec<u32> {
    vec![POISON; n]
}

pub fn is_nan_bits(v: u32) -> bool {
    (v & 0x7F80_0000) == 0x7F80_0000 && (v & 0x007F_FFFF) != 0
}

/// Word-wise comparison result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cmp {
    /// Words that count as different under the policy.
    pub mismatches: usize,
    /// NaN meets NaN with different bits (a failure only under the strict policy).
    pub nan_payload_diffs: usize,
    pub words: usize,
    /// Index of the first mismatching word.
    pub first: Option<usize>,
}

impl Cmp {
    pub fn add(&mut self, o: &Cmp) {
        self.mismatches += o.mismatches;
        self.nan_payload_diffs += o.nan_payload_diffs;
        self.words += o.words;
        if self.first.is_none() {
            self.first = o.first;
        }
    }
}

/// `strict`: every raw bit difference is a mismatch (the `wo` product gate: NaN payloads
/// included); otherwise non-NaN words must be bit-equal and a NaN must meet a NaN, payload and
/// sign differences between two NaNs being tallied only (the `lane_proj` product gate).
pub fn compare(got: &[u32], want: &[u32], strict: bool) -> Cmp {
    let mut c = Cmp {
        words: got.len().max(want.len()),
        ..Default::default()
    };
    if got.len() != want.len() {
        c.mismatches = got.len().abs_diff(want.len());
        c.first = Some(got.len().min(want.len()));
        return c;
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g == w {
            continue;
        }
        let both_nan = is_nan_bits(g) && is_nan_bits(w);
        if both_nan {
            c.nan_payload_diffs += 1;
        }
        if strict || !both_nan {
            c.mismatches += 1;
            c.first.get_or_insert(i);
        }
    }
    c
}

/// Run `f(0..n)` on worker threads (at most 16) and collect the results in order.
pub fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(4, |p| p.get())
        .clamp(1, 16)
        .min(n.max(1));
    let mut parts: Vec<Vec<(usize, T)>> = Vec::new();
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            break mine;
                        }
                        mine.push((i, f(i)));
                    }
                })
            })
            .collect();
        for h in hs {
            parts.push(h.join().expect("worker thread panicked"));
        }
    });
    let mut all: Vec<(usize, T)> = parts.into_iter().flatten().collect();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, v)| v).collect()
}

// ---------------------------------------------------------------------------
// Arms on the device

/// One arm's module with its functions resolved on demand.
pub struct LoadedArm {
    module: hip_bridge::Module,
}

impl LoadedArm {
    pub fn load(gpu: &Gpu, image: &[u8]) -> Result<Self> {
        // A fresh module per arm, as the driver's own timing loads them.
        Ok(Self {
            module: gpu.hip.module_load_data(image).map_err(err)?,
        })
    }

    pub fn function(&self, gpu: &Gpu, symbol: &str) -> Result<Function> {
        gpu.hip
            .module_get_function(&self.module, symbol)
            .map_err(err)
    }
}

/// One launch, then a device sync.
pub fn launch_sync(gpu: &Gpu, f: &Function, g: &Geometry, karg: &mut [u8]) -> Result<()> {
    // SAFETY: `karg` packs the kernel's explicit arguments per the production launcher and every
    // pointer in it addresses a live allocation owned by the caller for the whole call.
    unsafe {
        gpu.hip
            .launch_kernel_blob(f, g.grid, g.block, g.lds, None, karg)
    }
    .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)
}

// ---------------------------------------------------------------------------
// Matrix report

/// Accumulated result of a node's identity matrix: candidate vs incumbent, both vs the CPU
/// reference, and the untouched-poison guards.
#[derive(Default)]
pub struct Matrix {
    pub cases: usize,
    pub words: usize,
    /// Candidate vs incumbent (a failure under the node's policy).
    pub cand_vs_base: Cmp,
    pub base_vs_cpu: Cmp,
    pub cand_vs_cpu: Cmp,
    /// Pre-poisoned words that must stay untouched: rows/lanes beyond `lanes`, per arm.
    pub guard_violations: [usize; 2],
    /// Words an arm left at the pre-poison inside the written range (lane_proj: every output word
    /// is overwritten; a surviving poison word is a coverage hole). Per arm.
    pub poison_left: [usize; 2],
    pub failures: Vec<String>,
    pub lanes: Vec<usize>,
    pub wkinds: Vec<&'static str>,
}

impl Matrix {
    pub fn fail(&mut self, msg: String) {
        if self.failures.len() < 24 {
            self.failures.push(msg);
        }
    }

    pub fn pass(&self) -> bool {
        self.cases > 0
            && self.cand_vs_base.mismatches == 0
            && self.guard_violations == [0, 0]
            && self.poison_left == [0, 0]
    }

    pub fn json(&self, strict: bool) -> Value {
        let cpu = |c: &Cmp| {
            json!({"identical": c.mismatches == 0 && c.nan_payload_diffs == 0, "mismatches": c.mismatches,
                "nan_payload_diffs": c.nan_payload_diffs, "words": c.words})
        };
        json!({
            "pass": self.pass(),
            "policy": if strict {
                "raw bits: every word of every lane must equal the incumbent's (NaN payloads included)"
            } else {
                "non-NaN words bit-equal, NaN meets NaN (payload/sign differences between NaNs tallied, not failures)"
            },
            "cases": self.cases,
            "words_compared": self.cand_vs_base.words,
            "lanes": self.lanes,
            "weight_kinds": self.wkinds,
            "mismatches": self.cand_vs_base.mismatches,
            "nan_payload_diffs": self.cand_vs_base.nan_payload_diffs,
            "guard_violations": {"baseline": self.guard_violations[0], "candidate": self.guard_violations[1]},
            "poison_left_in_written_range": {"baseline": self.poison_left[0], "candidate": self.poison_left[1]},
            "cpu_ref_identical": {
                "baseline": self.base_vs_cpu.mismatches == 0 && self.base_vs_cpu.nan_payload_diffs == 0,
                "candidate": self.cand_vs_cpu.mismatches == 0 && self.cand_vs_cpu.nan_payload_diffs == 0,
            },
            "cpu_ref": {"baseline": cpu(&self.base_vs_cpu), "candidate": cpu(&self.cand_vs_cpu),
                "note": "informational: CPU reference hip_loop/lane_ref.rs; a mismatch is a reference bug or a hardware-numerics fact"},
            "failures": self.failures,
        })
    }
}
