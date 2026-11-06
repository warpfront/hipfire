// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU microbench of the D2 expert arm's GEMV on the Flash-Next shapes: planar W3 (qt 131/132)
//! against MQ4 (qt 44/53), the real `gemv_shared_sourced` path on the splice pool.
//!
//! Per layer call = 10 experts: one gate/up job (m = 1280, k = 2560) and one down job (m = 2560,
//! k = 640) per expert, `rows` activation rows each. Reports the median call time, the weight
//! bytes the call reads and the achieved GB/s, and a `digest` of every output (FNV-1a over the
//! f32 bits) so a kernel change can be proven byte-identical by running the same seed.
//!
//!   cargo run --release -p hipfire-cpu --features w3-bench --example w3_bench -- [--iters 200] [--rows 1,4] [--threads 16] [--experts 192] [--seed 1] [--single]
//!
//! `--single` also runs the cache-resident single-thread kernel rate (one pinned worker).

use hipfire_cpu::gemv::{gemv_shared_sourced, tensor_bytes, SharedJobSource};
use hipfire_cpu::quant::CpuQuant;
use hipfire_cpu::splice_pool::{install, SplicePoolConfig};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `experts` tensors of `[m, k]` in format `q`, random bytes, planar scale planes made finite.
fn weights(q: CpuQuant, m: usize, k: usize, experts: usize, rng: &mut Rng) -> (Vec<u8>, usize) {
    let tb = tensor_bytes(q, m, k);
    let mut v = vec![0u8; tb * experts];
    for chunk in v.chunks_mut(8) {
        let r = rng.next().to_le_bytes();
        chunk.copy_from_slice(&r[..chunk.len()]);
    }
    if q.is_planar() {
        // The scale plane (f16, `2 * k / 128` bytes per row) follows the m code rows: finite halves.
        let codes = m * hipfire_cpu::gemv::row_bytes(q, k);
        for e in 0..experts {
            for h in v[e * tb + codes..(e + 1) * tb].chunks_mut(2) {
                let mant = (rng.next() & 0x3ff) as u16;
                let bits = 0x3000u16 | mant; // exponent 12: about 2^-3
                h.copy_from_slice(&bits.to_le_bytes());
            }
        }
    } else {
        // Row formats: per-128 f16 header(s) inside the rows. Random bytes can be NaN/Inf; the
        // timing does not depend on values, the digest only has to be reproducible.
    }
    (v, tb)
}

struct Jobs<'a> {
    q: CpuQuant,
    blob: &'a [u8],
    stride: usize,
    picks: &'a [usize],
    xs: &'a [Vec<f32>],
    rows: usize,
}
impl SharedJobSource for Jobs<'_> {
    fn jobs(&self) -> usize {
        self.picks.len()
    }
    fn n(&self, _j: usize) -> usize {
        self.rows
    }
    fn quant(&self, _j: usize) -> CpuQuant {
        self.q
    }
    fn packed(&self, j: usize) -> &[u8] {
        &self.blob[self.picks[j] * self.stride..][..self.stride]
    }
    fn xs_row(&self, _j: usize, t: usize) -> &[f32] {
        &self.xs[t]
    }
}

fn fnv(h: &mut u64, v: &[f32]) {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let iters: usize = arg("--iters", 200);
    let threads: usize = arg("--threads", 16);
    let experts: usize = arg("--experts", 192);
    let seed: u64 = arg("--seed", 1);
    let single = std::env::args().any(|a| a == "--single");
    let rows_list: Vec<usize> = std::env::args()
        .skip_while(|a| a != "--rows")
        .nth(1)
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4]);
    let cfg = SplicePoolConfig { threads, pin: true };
    let formats: [(&str, CpuQuant, CpuQuant); 2] = [
        ("W3", CpuQuant::Mq3I4G128R256, CpuQuant::Mq3I4G128R128),
        ("MQ4", CpuQuant::Mq4G256V2, CpuQuant::Mq4G128V2),
    ];
    let mut rng = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
    let x_gu: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..2560)
                .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect()
        })
        .collect();
    let x_dn: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..640)
                .map(|_| (rng.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect()
        })
        .collect();
    println!(
        "threads={threads} pinned experts={experts} iters={iters} (arena per format ~{:.0} MB gate_up + {:.0} MB down)",
        experts as f64 * 1.3,
        experts as f64 * 0.65
    );
    for (name, qgu, qdn) in formats {
        let (gu, gu_stride) = weights(qgu, 1280, 2560, experts, &mut rng);
        let (dn, dn_stride) = weights(qdn, 2560, 640, experts, &mut rng);
        for &rows in &rows_list {
            let mut pick_rng = Rng(seed.wrapping_add(77) | 1);
            let mut times = Vec::with_capacity(iters);
            let mut digest = 0xcbf29ce484222325u64;
            let bytes = 10 * (gu_stride + dn_stride);
            for it in 0..iters + 5 {
                let mut picks = [0usize; 10];
                for p in picks.iter_mut() {
                    *p = (pick_rng.next() % experts as u64) as usize;
                }
                let jg = Jobs {
                    q: qgu,
                    blob: &gu,
                    stride: gu_stride,
                    picks: &picks,
                    xs: &x_gu,
                    rows,
                };
                let jd = Jobs {
                    q: qdn,
                    blob: &dn,
                    stride: dn_stride,
                    picks: &picks,
                    xs: &x_dn,
                    rows,
                };
                let mut og = vec![0f32; 10 * 1280 * rows];
                let mut od = vec![0f32; 10 * 2560 * rows];
                let t = Instant::now();
                install(cfg, || {
                    gemv_shared_sourced(1280, 2560, &mut og, &jg, None);
                    gemv_shared_sourced(2560, 640, &mut od, &jd, None);
                });
                let dt = t.elapsed().as_secs_f64();
                if it >= 5 {
                    times.push(dt);
                    fnv(&mut digest, &og);
                    fnv(&mut digest, &od);
                }
            }
            times.sort_by(f64::total_cmp);
            let med = times[times.len() / 2];
            println!(
                "{name:3} rows={rows}  layer call median {:7.1} us (min {:7.1})  weight bytes {:6.2} MB  {:6.1} GB/s  digest {digest:016x}",
                med * 1e6,
                times[0] * 1e6,
                bytes as f64 / 1e6,
                bytes as f64 / med / 1e9
            );
        }
        if single {
            // One pinned worker, one expert's tensors (cache resident after the first pass):
            // the kernel's own rate, no memory system. 16 experts' worth of rows.
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .start_handler(|_| unsafe {
                    // The splice pool is a process-wide OnceLock (first config wins), so the
                    // single-thread rate runs on its own pool, pinned to CPU 2.
                    let mut set: libc::cpu_set_t = std::mem::zeroed();
                    libc::CPU_SET(2, &mut set);
                    libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set);
                })
                .build()
                .unwrap();
            for &rows in &rows_list {
                let picks = [0usize];
                let jg = Jobs {
                    q: qgu,
                    blob: &gu,
                    stride: gu_stride,
                    picks: &picks,
                    xs: &x_gu,
                    rows,
                };
                let mut og = vec![0f32; 1280 * rows];
                let reps = 400;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    one.install(|| {
                        for _ in 0..reps {
                            gemv_shared_sourced(1280, 2560, &mut og, &jg, None);
                        }
                    });
                    best = best.min(t.elapsed().as_secs_f64() / reps as f64);
                }
                println!(
                    "{name:3} rows={rows}  1 thread, resident gate_up: {:7.1} us/call  {:5.2} GB/s of weights  {:6.1} ns per row-group-of-128 (per token)",
                    best * 1e6,
                    gu_stride as f64 / best / 1e9,
                    best * 1e9 / (1280.0 * 20.0 * rows as f64)
                );
            }
        }
    }
}
