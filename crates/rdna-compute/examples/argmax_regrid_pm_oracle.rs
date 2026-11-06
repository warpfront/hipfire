// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Byte oracle and timing for the PM batched row argmax twin
//! `argmax_f32_batched_regrid_pm_gfx1201` (exact gfx1201).
//!
//! Oracle: for every (row length, batch, input kind) case, the three arms
//! (shipping `argmax_f32_batched`, hipcc re-grid `argmax_f32_batched_regrid`, PM twin) write
//! their `[batch]` i32 result into poisoned buffers with guard words on both sides; every
//! result word must be byte-identical across the arms and to the host reference of the HIP
//! law, the result guards and the input rows (with guards) must stay untouched.
//!
//! Timing: PM vs hipcc re-grid (and shipping) per launch at vocab 248320, hipEvent timed over
//! back-to-back launches, interleaved blocks with alternating arm order.
//!
//! ```text
//! argmax_regrid_pm_oracle oracle --out run.json   # build with --features lab
//! argmax_regrid_pm_oracle timing --out run.json
//! ```

use rdna_compute::{Gpu, GpuTensor};
use serde_json::json;

const GUARD: usize = 1024;
const POISON: u32 = 0xA5A5_A5A5;
const RPOISON: u32 = 0x5A5A_5A5A;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 24
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f(&mut self) -> f32 {
        (self.next() % 2_000_001) as f32 / 1_000_000.0 - 1.0
    }
}

/// The HIP law: `v > -1e30` qualifies; larger value wins, equal values the lower
/// `rev8(i & 255)`, then the lower index; none -> 0.
fn reference(row: &[f32]) -> i32 {
    let key = |i: usize| ((i as u32 & 255) as u8).reverse_bits() as u64 * (1 << 32) + i as u64;
    let mut best: Option<(f32, usize)> = None;
    for (i, &v) in row.iter().enumerate() {
        if !(v > -1e30f32) {
            continue;
        }
        best = match best {
            None => Some((v, i)),
            Some((bv, bi)) if v > bv || (v == bv && key(i) < key(bi)) => Some((v, i)),
            b => b,
        };
    }
    best.map_or(0, |(_, i)| i as i32)
}

const KINDS: [&str; 14] = [
    "random",
    "all-equal",
    "all-zero",
    "zero-signs",
    "coarse",
    "nan-sprinkle",
    "inf-sprinkle",
    "special-mix",
    "denormals",
    "all-nan",
    "all-neg-inf",
    "all-below",
    "exactly-neg-1e30",
    "plateau-tail",
];

fn make(kind: &str, rng: &mut Rng, n: usize, batch: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; n * batch];
    for r in 0..batch {
        let row = &mut d[r * n..(r + 1) * n];
        match kind {
            "random" => row.iter_mut().for_each(|x| *x = rng.f()),
            "all-equal" => row.iter_mut().for_each(|x| *x = 0.375),
            "all-zero" => row.iter_mut().for_each(|x| *x = 0.0),
            "zero-signs" => row
                .iter_mut()
                .for_each(|x| *x = if rng.below(2) == 0 { -0.0 } else { 0.0 }),
            "coarse" => row.iter_mut().for_each(|x| *x = rng.below(4) as f32),
            "nan-sprinkle" => {
                row.iter_mut().for_each(|x| *x = rng.f());
                for _ in 0..(n / 3 + 1) {
                    row[rng.below(n)] = f32::NAN;
                }
            }
            "inf-sprinkle" => {
                row.iter_mut().for_each(|x| *x = rng.f());
                for _ in 0..(n / 7 + 1) {
                    row[rng.below(n)] = if rng.below(2) == 0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            }
            "special-mix" => {
                row.iter_mut().for_each(|x| *x = rng.below(3) as f32);
                for _ in 0..(n / 4 + 1) {
                    row[rng.below(n)] = match rng.below(6) {
                        0 => f32::NAN,
                        1 => f32::NEG_INFINITY,
                        2 => f32::INFINITY,
                        3 => -0.0,
                        4 => 0.0,
                        _ => f32::from_bits(1 + rng.below(8) as u32),
                    };
                }
            }
            "denormals" => row.iter_mut().for_each(|x| {
                let sign = if rng.below(2) == 0 { 0 } else { 0x8000_0000 };
                *x = f32::from_bits(sign | rng.below(8) as u32);
            }),
            "all-nan" => row.iter_mut().for_each(|x| *x = f32::NAN),
            "all-neg-inf" => row.iter_mut().for_each(|x| *x = f32::NEG_INFINITY),
            "all-below" => row.iter_mut().for_each(|x| *x = -2e30),
            "exactly-neg-1e30" => row.iter_mut().for_each(|x| *x = -1e30),
            "plateau-tail" => {
                row.iter_mut().for_each(|x| *x = rng.f() - 2.0);
                for _ in 0..4 {
                    row[n - 1 - rng.below(n.min(300))] = 5.0;
                }
            }
            _ => unreachable!(),
        }
    }
    d
}

fn poisoned(gpu: &mut Gpu, len: usize, word: u32) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(word); len], &[len])
        .expect("poisoned upload")
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

type Launch = fn(&mut Gpu, &GpuTensor, &GpuTensor, usize, usize) -> hip_bridge::HipResult<()>;

const ARMS: [(&str, Launch); 3] = [
    ("shipping", Gpu::argmax_f32_batched_shipping),
    ("hipcc", Gpu::argmax_f32_batched_regrid_hipcc_forced),
    ("pm", Gpu::argmax_f32_batched_regrid_pm_forced),
];

fn oracle(gpu: &mut Gpu) -> serde_json::Value {
    let mut rng = Rng(0x5EED_0B4);
    let mut cases: Vec<(usize, usize, &str)> = Vec::new();
    let ns = [
        1usize, 2, 3, 5, 31, 32, 33, 255, 256, 257, 1000, 1023, 1024, 1025, 4095, 4096, 4097, 9999,
        151_936, 248_320, 262_144,
    ];
    let batches = [1usize, 2, 3, 8, 16, 32, 64];
    for &n in &ns {
        for &b in &batches {
            // Keep the host data under ~70 MB.
            let b = if n > 100_000 { b.min(64) } else { b };
            for k in KINDS {
                if n > 100_000
                    && b > 8
                    && !matches!(k, "random" | "all-equal" | "special-mix" | "coarse")
                {
                    continue;
                }
                cases.push((n, b, k));
            }
        }
    }
    // Every batch 1..=64.
    for b in 1..=64usize {
        for &n in &[4097usize, 248_320] {
            for k in ["random", "coarse", "special-mix"] {
                cases.push((n, b, k));
            }
        }
    }
    let mut mismatches = 0u64;
    let mut guard_faults = 0u64;
    let mut launches = 0u64;
    let mut failures = Vec::new();
    let mut digest = 0u64;
    for (ci, &(n, batch, kind)) in cases.iter().enumerate() {
        let host = make(kind, &mut rng, n, batch);
        // data with guard words on both sides.
        let mut full = vec![f32::from_bits(POISON); GUARD];
        full.extend_from_slice(&host);
        full.extend(std::iter::repeat(f32::from_bits(POISON)).take(GUARD));
        let data_full = gpu.upload_f32(&full, &[full.len()]).expect("data upload");
        let data = data_full.sub_offset(GUARD, n * batch);
        let want: Vec<i32> = (0..batch)
            .map(|r| reference(&host[r * n..(r + 1) * n]))
            .collect();
        let mut results: Vec<(&str, Vec<u32>)> = Vec::new();
        for (name, launch) in ARMS {
            let res_full = poisoned(gpu, GUARD + batch + GUARD, RPOISON);
            let res = res_full.sub_offset(GUARD, batch);
            launch(gpu, &data, &res, n, batch)
                .unwrap_or_else(|e| panic!("{name} n {n} b {batch}: {e}"));
            gpu.hip.device_synchronize().expect("sync");
            launches += 1;
            let out = bits(gpu, &res_full);
            if out[..GUARD].iter().any(|&w| w != RPOISON)
                || out[GUARD + batch..].iter().any(|&w| w != RPOISON)
            {
                guard_faults += 1;
                failures.push(json!({"case": ci, "n": n, "batch": batch, "kind": kind, "arm": name, "fault": "result guard"}));
            }
            results.push((name, out[GUARD..GUARD + batch].to_vec()));
            gpu.free_tensor(res_full).ok();
        }
        let din = bits(gpu, &data_full);
        if din[..GUARD].iter().any(|&w| w != POISON)
            || din[GUARD + n * batch..].iter().any(|&w| w != POISON)
            || din[GUARD..GUARD + n * batch]
                .iter()
                .zip(&host)
                .any(|(&a, &b)| a != b.to_bits())
        {
            guard_faults += 1;
            failures.push(json!({"case": ci, "n": n, "batch": batch, "kind": kind, "fault": "input modified"}));
        }
        gpu.free_tensor(data_full).ok();
        for r in 0..batch {
            let w = want[r] as u32;
            for (name, out) in &results {
                digest = digest
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(u64::from(out[r]));
                if out[r] != w {
                    mismatches += 1;
                    if failures.len() < 50 {
                        failures.push(json!({"case": ci, "n": n, "batch": batch, "kind": kind, "row": r, "arm": name, "got": out[r] as i32, "want": want[r]}));
                    }
                }
            }
        }
    }
    json!({
        "mode": "oracle", "arch": gpu.arch, "cases": cases.len(), "launches": launches,
        "arms": ["shipping", "hipcc", "pm"], "reference": "host HIP law",
        "mismatches": mismatches, "guard_faults": guard_faults,
        "result_digest": format!("{digest:016x}"), "failures": failures,
    })
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_us(gpu: &mut Gpu, repeats: usize, mut body: impl FnMut(&mut Gpu)) -> f64 {
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..repeats {
        body(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync");
    gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1_000.0 / repeats as f64
}

fn timing(gpu: &mut Gpu) -> serde_json::Value {
    let n = 248_320usize;
    let mut rng = Rng(0x71_3E);
    let mut rows = Vec::new();
    for batch in [1usize, 8, 16, 32, 64] {
        let host = make("random", &mut rng, n, batch);
        let data = gpu.upload_f32(&host, &[n * batch]).expect("data");
        let res = poisoned(gpu, batch, RPOISON);
        // Warm every arm (module load) and prove agreement before timing.
        let mut outs = Vec::new();
        for (name, launch) in ARMS {
            launch(gpu, &data, &res, n, batch).unwrap_or_else(|e| panic!("{name}: {e}"));
            gpu.hip.device_synchronize().expect("sync");
            outs.push(bits(gpu, &res));
        }
        assert!(
            outs.iter().all(|o| *o == outs[0]),
            "arms disagree at batch {batch}"
        );
        let repeats = 400;
        let blocks = 11;
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
        for block in 0..blocks {
            let order: Vec<usize> = if block % 2 == 0 {
                vec![0, 1, 2]
            } else {
                vec![2, 1, 0]
            };
            for a in order {
                let launch = ARMS[a].1;
                let us = time_us(gpu, repeats, |g| {
                    launch(g, &data, &res, n, batch).expect("launch")
                });
                samples[a].push(us);
            }
        }
        let mut entry = json!({"batch": batch, "n": n});
        for (a, (name, _)) in ARMS.iter().enumerate() {
            let mut s = samples[a].clone();
            let min = s.iter().cloned().fold(f64::INFINITY, f64::min);
            entry[*name] =
                json!({"median_us": median(&mut s), "min_us": min, "blocks_us": samples[a]});
        }
        rows.push(entry);
        gpu.free_tensor(data).ok();
        gpu.free_tensor(res).ok();
    }
    json!({"mode": "timing", "arch": gpu.arch, "rows": rows})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("");
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let mut gpu = Gpu::init().expect("Gpu::init");
    assert_eq!(gpu.arch, "gfx1201", "exact gfx1201 required");
    let report = match mode {
        "oracle" => oracle(&mut gpu),
        "timing" => timing(&mut gpu),
        _ => panic!("usage: argmax_regrid_pm_oracle oracle|timing --out FILE"),
    };
    let text = serde_json::to_string_pretty(&report).unwrap();
    println!("{text}");
    if let Some(p) = out {
        std::fs::write(&p, &text).expect("write report");
    }
    if mode == "oracle" && (report["mismatches"] != 0 || report["guard_faults"] != 0) {
        std::process::exit(1);
    }
}
