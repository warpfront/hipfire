// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle: the persistent Qwen4 HC up+mix read tail `hyper_read_up_wmma_bf16_gfx1201_v2`
//! (exact gfx1201) / `hyper_read_up_wmma_bf16_gfx1151_v2` (exact gfx1151) against the retiled
//! `_gfx1201_t128` / `_swap` entry it replaces, every output u32 of `mixed`.
//! Both run through the public route API (`tensor_ops::hyper_read_up_wmma_route`), so the real
//! JIT-compiled kernels are exercised on the exact bytes.
//!
//! Inputs: hidden 2560, low_rank 320, `up_weight` BF16 `[4 * 2560, 320]`, `low` BF16 `[rows, 320]`,
//! `normalized` F16 `[rows, 4 * 2560]`. `mixed` (F32) is allocated `rows * 2560 + 4096` elements,
//! pre-filled with 0xAB bytes: the active region is compared u32 by u32 and the guard region must
//! keep its 0xAB fill after both launches.
//!
//! Classes (rows from `ROWS=a,b,..`, default 1,16,129,4096,4099,16384):
//!   a `random`      weights N(0, 0.15), low N(0,1), normalized N(0,1) * 2 as F16.
//!   b `adversarial` NaN / +-inf / +-0 / subnormal / extreme BF16 and F16 patterns sprinkled into low,
//!                   weights and normalized, plus a block of rows whose `low` is huge in magnitude.
//!   c `nanfuzz`     dense and 100x sparser random NaN payloads (BF16 sign|0x7f80|m, m in 1..127; F16
//!                   sign|0x7c00|m, m in 1..1023) and random infinities in low / weights / normalized
//!                   at densities 3e-4 / 2e-4 / 4e-3.
//!   d `sweep`       exhaustive gate sweep, 16 rows: for each branch b in 0..4 and chunk c in 0..26,
//!                   weight row (b * 2560 + j) column 0 = BF16 pattern (c * 2560 + j) (65536 patterns
//!                   over the 26 chunks; larger values wrap to 0), every other weight 0; `low` is the
//!                   one-hot 1.0 at column 0 of every row; `normalized` is F16 1.0 in branch b's
//!                   segment and F16 +0 elsewhere. Every gate input pattern reaches the epilogue exactly.
//!
//! Exit 0: every case byte-identical. 1: a mismatch (first row / col / bits, with NaN-vs-NaN
//! payload-only differences counted apart from the rest). 2: V2 is not available (neither gfx1201
//! on the F16 WMMA route nor gfx1151). Receipt: `HC_UP_ORACLE_RECEIPT` (default `./hc-up-v2-oracle.txt`).
//!
//! Run: cargo run --release --features lab --example hc_up_v2_oracle -p rdna-compute

use half::f16;
use rdna_compute::tensor_ops::{hyper_read_up_wmma_route, HcUpRoute, HyperReadUpFused};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt::Write as _;

const HIDDEN: usize = 2560;
const LOW_RANK: usize = 320;
const WIDE: usize = 4 * HIDDEN;
const GUARD: usize = 4096;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// N(0,1)-like: sum of four uniforms on [-1,1) scaled to unit variance.
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// True with probability ~ 1/period.
    fn hit(&mut self, period: u64) -> bool {
        self.next().is_multiple_of(period)
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn f16_bits(v: f32) -> u16 {
    f16::from_f32(v).to_bits()
}

fn to_bytes(v: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for h in v {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn upload_u16(gpu: &mut Gpu, v: &[u16], dtype: DType) -> GpuTensor {
    let mut t = gpu.upload_raw(&to_bytes(v), &[v.len()]).expect("upload");
    t.dtype = dtype;
    t.shape = vec![v.len()];
    t
}

const BF16_SPECIALS: [u16; 16] = [
    0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x7f7f, 0xff7f,
    0x7f81, 0xffff, 0x0080, 0x8080,
];
const F16_SPECIALS: [u16; 16] = [
    0x7e00, 0xfe00, 0x7c00, 0xfc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x7bff, 0xfbff,
    0x7c01, 0xffff, 0x0400, 0x8400,
];

/// One class's operands for `rows` rows.
struct Case {
    w: Vec<u16>,
    low: Vec<u16>,
    norm: Vec<u16>,
}

/// Replace elements with a NaN-payload or infinity at probability 1/period.
fn fuzz(v: &mut [u16], rng: &mut Rng, period: u64, bf16: bool) {
    for e in v.iter_mut() {
        if !rng.hit(period) {
            continue;
        }
        let sign = ((rng.next() & 1) as u16) << 15;
        let r = rng.next();
        *e = if r % 10 < 7 {
            if bf16 {
                sign | 0x7f80 | (1 + (r >> 8) as u16 % 127)
            } else {
                sign | 0x7c00 | (1 + (r >> 8) as u16 % 1023)
            }
        } else if bf16 {
            sign | 0x7f80
        } else {
            sign | 0x7c00
        };
    }
}

fn gen_random(rows: usize, rng: &mut Rng) -> Case {
    Case {
        w: (0..WIDE * LOW_RANK)
            .map(|_| bf16_bits(rng.gauss() * 0.15))
            .collect(),
        low: (0..rows * LOW_RANK)
            .map(|_| bf16_bits(rng.gauss()))
            .collect(),
        norm: (0..rows * WIDE)
            .map(|_| f16_bits(rng.gauss() * 2.0))
            .collect(),
    }
}

fn gen_adversarial(rows: usize, rng: &mut Rng) -> Case {
    let mut c = gen_random(rows, rng);
    for e in c.w.iter_mut() {
        if rng.hit(512) {
            *e = BF16_SPECIALS[rng.below(BF16_SPECIALS.len())];
        }
    }
    for e in c.low.iter_mut() {
        if rng.hit(64) {
            *e = BF16_SPECIALS[rng.below(BF16_SPECIALS.len())];
        }
    }
    for e in c.norm.iter_mut() {
        if rng.hit(512) {
            *e = F16_SPECIALS[rng.below(F16_SPECIALS.len())];
        }
    }
    // A block of rows (r % 61 < 5) with huge-magnitude `low`: logits overflow to +-inf / NaN.
    for r in 0..rows {
        if r % 61 < 5 {
            for e in &mut c.low[r * LOW_RANK..(r + 1) * LOW_RANK] {
                let sign = ((rng.next() & 1) as u16) << 15;
                *e = sign | ((200 + rng.below(55) as u16) << 7) | (rng.next() & 0x7f) as u16;
            }
        }
    }
    c
}

fn gen_nanfuzz(rows: usize, rng: &mut Rng, scale_down: u64) -> Case {
    let mut c = gen_random(rows, rng);
    fuzz(&mut c.w, rng, 5000 * scale_down, true);
    fuzz(&mut c.low, rng, 3333 * scale_down, true);
    fuzz(&mut c.norm, rng, 250 * scale_down, false);
    c
}

/// Exhaustive gate sweep operands for branch `b`, chunk `c` (16 rows).
fn gen_sweep(b: usize, chunk: usize) -> Case {
    let rows = 16;
    let mut w = vec![0u16; WIDE * LOW_RANK];
    for j in 0..HIDDEN {
        let v = chunk * HIDDEN + j;
        w[(b * HIDDEN + j) * LOW_RANK] = if v < 65536 { v as u16 } else { 0 };
    }
    let mut low = vec![0u16; rows * LOW_RANK];
    for r in 0..rows {
        low[r * LOW_RANK] = 0x3f80;
    }
    let mut norm = vec![0u16; rows * WIDE];
    for r in 0..rows {
        for j in 0..HIDDEN {
            norm[r * WIDE + b * HIDDEN + j] = 0x3c00;
        }
    }
    Case { w, low, norm }
}

fn git_sha() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    };
    let sha = run(&["rev-parse", "HEAD"]);
    let dirty = !run(&["status", "--porcelain", "--untracked-files=no"]).is_empty();
    format!(
        "{}{}",
        if sha.is_empty() { "unknown" } else { &sha },
        if dirty { "+dirty" } else { "" }
    )
}

fn le_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > 0x7f80_0000
}

struct Outcome {
    mismatches: usize,
    payload_only: usize,
    first_other: Option<(usize, usize, u32, u32)>,
    first_payload: Option<(usize, usize, u32, u32)>,
    guard_dirty: usize,
    nan: usize,
    inf: usize,
}

/// Run T128 and V2 on `c` and compare every `mixed` u32 plus the guard region.
fn run_case(gpu: &mut Gpu, w: &GpuTensor, c: &Case, rows: usize) -> Outcome {
    let low = upload_u16(gpu, &c.low, DType::BF16);
    let norm = upload_u16(gpu, &c.norm, DType::F16);
    let n = rows * HIDDEN;
    let mut outs = Vec::new();
    for route in [HcUpRoute::T128, HcUpRoute::V2] {
        let fill = vec![0xABu8; (n + GUARD) * 4];
        let mut mixed = gpu.upload_raw(&fill, &[n + GUARD]).expect("mixed");
        mixed.dtype = DType::F32;
        mixed.shape = vec![n + GUARD];
        hyper_read_up_wmma_route(
            gpu,
            &HyperReadUpFused {
                up_weight: w,
                low: &low,
                normalized: &norm,
                mixed: &mixed,
                rows,
                hidden: HIDDEN,
                low_rank: LOW_RANK,
                normalized_bf16: true,
            },
            route,
        )
        .unwrap_or_else(|e| panic!("route {route:?} rows {rows}: {e}"));
        gpu.hip.device_synchronize().unwrap();
        outs.push(le_u32(&gpu.download_raw_bytes(&mixed).unwrap()));
        gpu.free_tensor(mixed).unwrap();
    }
    gpu.free_tensor(low).unwrap();
    gpu.free_tensor(norm).unwrap();
    let (t, v) = (&outs[0], &outs[1]);
    let mut o = Outcome {
        mismatches: 0,
        payload_only: 0,
        first_other: None,
        first_payload: None,
        guard_dirty: v[n..].iter().filter(|&&x| x != 0xABAB_ABAB).count()
            + t[n..].iter().filter(|&&x| x != 0xABAB_ABAB).count(),
        nan: 0,
        inf: 0,
    };
    for i in 0..n {
        if is_nan(t[i]) {
            o.nan += 1;
        } else if t[i] & 0x7fff_ffff == 0x7f80_0000 {
            o.inf += 1;
        }
        if t[i] != v[i] {
            o.mismatches += 1;
            let at = (i / HIDDEN, i % HIDDEN, t[i], v[i]);
            if is_nan(t[i]) && is_nan(v[i]) {
                o.payload_only += 1;
                o.first_payload.get_or_insert(at);
            } else {
                o.first_other.get_or_insert(at);
            }
        }
    }
    o
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.to_string();
    let available = (arch == "gfx1201" && gpu.qwen4_f16_wmma_gfx1201()) || arch == "gfx1151";
    let (v2_name, ref_name) = if arch == "gfx1151" {
        (
            "hyper_read_up_wmma_bf16_gfx1151_v2",
            "hyper_read_up_wmma_bf16_swap",
        )
    } else {
        (
            "hyper_read_up_wmma_bf16_gfx1201_v2",
            "hyper_read_up_wmma_bf16_gfx1201_t128",
        )
    };
    println!(
        "arch {arch} f16_wmma_gfx1201={} v2_available={available}",
        gpu.qwen4_f16_wmma_gfx1201()
    );
    if !available {
        eprintln!("V2 is not available: it needs exact gfx1201 (F16 WMMA route) or exact gfx1151");
        std::process::exit(2);
    }

    let rows_list: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter(|t| !t.is_empty())
                .map(|t| t.parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 16, 129, 4096, 4099, 16384]);

    let receipt_path =
        std::env::var("HC_UP_ORACLE_RECEIPT").unwrap_or_else(|_| "./hc-up-v2-oracle.txt".into());
    let mut receipt = String::new();
    let date = std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    writeln!(receipt, "hc_up_v2_oracle receipt").unwrap();
    writeln!(receipt, "date {date}").unwrap();
    writeln!(receipt, "git {}", git_sha()).unwrap();
    writeln!(receipt, "arch {arch}").unwrap();
    writeln!(
        receipt,
        "kernel {v2_name} vs {ref_name} hidden={HIDDEN} low_rank={LOW_RANK} rows={rows_list:?}"
    )
    .unwrap();

    let mut failures = 0usize;
    let mut cases = 0usize;
    let mut report = |receipt: &mut String, label: &str, rows: usize, o: &Outcome| {
        cases += 1;
        let pass = o.mismatches == 0 && o.guard_dirty == 0;
        let line = if pass {
            format!(
                "ORACLE PASS {label} rows={rows} elems={} nan={} inf={}",
                rows * HIDDEN,
                o.nan,
                o.inf
            )
        } else {
            failures += 1;
            format!(
                "ORACLE FAIL {label} rows={rows} mismatches={} payload_only_nan_vs_nan={} other={} \
                 first_other(row,col,t128,v2)={:x?} first_payload_only(row,col,t128,v2)={:x?} guard_dirty={}",
                o.mismatches,
                o.payload_only,
                o.mismatches - o.payload_only,
                o.first_other,
                o.first_payload,
                o.guard_dirty
            )
        };
        println!("{line}");
        writeln!(receipt, "{line}").unwrap();
    };

    let max_rows = *rows_list.iter().max().unwrap();
    for (class, seed) in [
        ("random", 1u64),
        ("adversarial", 2),
        ("nanfuzz-dense", 3),
        ("nanfuzz-sparse", 4),
    ] {
        let mut rng = Rng(seed * 0x1234_5678_9abc_def1);
        let full = match class {
            "random" => gen_random(max_rows, &mut rng),
            "adversarial" => gen_adversarial(max_rows, &mut rng),
            "nanfuzz-dense" => gen_nanfuzz(max_rows, &mut rng, 1),
            _ => gen_nanfuzz(max_rows, &mut rng, 100),
        };
        let w = upload_u16(&mut gpu, &full.w, DType::BF16);
        for &rows in &rows_list {
            let case = Case {
                w: Vec::new(),
                low: full.low[..rows * LOW_RANK].to_vec(),
                norm: full.norm[..rows * WIDE].to_vec(),
            };
            let o = run_case(&mut gpu, &w, &case, rows);
            report(&mut receipt, &format!("class={class}"), rows, &o);
        }
        gpu.free_tensor(w).unwrap();
    }

    for b in 0..4 {
        for chunk in 0..26 {
            let case = gen_sweep(b, chunk);
            let w = upload_u16(&mut gpu, &case.w, DType::BF16);
            let o = run_case(&mut gpu, &w, &case, 16);
            gpu.free_tensor(w).unwrap();
            report(
                &mut receipt,
                &format!("class=sweep branch={b} chunk={chunk}"),
                16,
                &o,
            );
        }
    }

    let summary = format!(
        "SUMMARY cases={cases} failures={failures} {}",
        if failures == 0 {
            "ALL BYTE-IDENTICAL"
        } else {
            "MISMATCH"
        }
    );
    println!("{summary}");
    writeln!(receipt, "{summary}").unwrap();
    std::fs::write(&receipt_path, &receipt).expect("write receipt");
    println!("receipt {receipt_path}");
    if failures != 0 {
        std::process::exit(1);
    }
}
