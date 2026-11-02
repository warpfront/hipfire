// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Lab: time the D2 CPU arm of Flash-Next decode on this host, CPU only.
//!
//! Three measurements, each over synthetic payloads of the Flash-Next expert
//! shapes (`dim = 2560`, `mi = 640`, gate_up `[1280, 2560]`, down
//! `[2560, 640]`):
//!
//! - `layer`: one decode layer's CPU arm exactly as the split runs it —
//!   [`moe_cpu_host_slot_outputs`] for 1 row × `top_k` cold slots on the
//!   splice pool — with each slot's expert drawn from an E8P-I4 (qt 135/136)
//!   or an MQ4 V2 (qt 44/53) pool at `--e8p-share`. The pools are larger than
//!   the last-level cache, so every call streams its weights from DRAM as the
//!   host-mapped arena does. `--gap-us` idles the pool between calls (the GPU
//!   part of a decode layer), so worker wake-up is part of the figure.
//! - `regions`: the arm's two batched-GEMV regions (gate/up, then down) on
//!   their own, each split into preparation and rayon execution, to separate
//!   the GEMV time from the rest of a layer's arm.
//! - `kernel`: the single-thread GEMV rate of each format on the gate_up shape
//!   (one-thread pool), cache-resident and streaming.
//!
//! ```text
//! cargo run --release -p hipfire-dispatch --features lab --example d2_cpu_arm_bench -- layer --e8p-share 0.54
//! cargo run --release -p hipfire-dispatch --features lab --example d2_cpu_arm_bench -- regions --e8p-share 1.0
//! cargo run --release -p hipfire-dispatch --features lab --example d2_cpu_arm_bench -- kernel
//! ```
//!
//! The worker count and pinning follow `HIPFIRE_CPU_EXPERT_THREADS` /
//! `HIPFIRE_CPU_EXPERT_PIN`, as in the daemon. Pin the process to the cores
//! under test (`taskset`) to model a smaller host.

use std::time::Instant;

use hipfire_cpu::gemv::{gemv, row_bytes, tensor_bytes};
use hipfire_cpu::quant::CpuQuant;
use hipfire_dispatch::cpu_exec::{moe_cpu_host_slot_outputs, CpuMoeExpert};

const DIM: usize = 2560;
const MI: usize = 640;
const N_EXPERTS: usize = 512;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A normal-range f16 in roughly `[2^-9, 2^-5]`, either sign when `signed`.
fn f16_bits(rng: &mut Rng, signed: bool) -> [u8; 2] {
    let exp = 6 + (rng.next() % 5) as u16;
    let man = (rng.next() % 1024) as u16;
    let sign = if signed && rng.next() & 1 == 1 {
        0x8000
    } else {
        0
    };
    (sign | exp << 10 | man).to_le_bytes()
}

/// A `[m, k]` tensor of `q` with random codes and finite normal f16 metadata.
fn tensor(q: CpuQuant, m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let total = tensor_bytes(q, m, k);
    let mut t: Vec<u8> = (0..total).map(|_| rng.next() as u8).collect();
    let rb = row_bytes(q, k);
    match q {
        CpuQuant::E8pI4G128R256 | CpuQuant::E8pI4G128R128 => {
            for s in t[m * rb..].chunks_exact_mut(2) {
                s.copy_from_slice(&f16_bits(rng, false));
            }
        }
        CpuQuant::Mq4G256V2 => {
            for g in t.chunks_exact_mut(136) {
                for h in 0..2 {
                    g[4 * h..4 * h + 2].copy_from_slice(&f16_bits(rng, false));
                    g[4 * h + 2..4 * h + 4].copy_from_slice(&f16_bits(rng, true));
                }
            }
        }
        CpuQuant::Mq4G128V2 => {
            for g in t.chunks_exact_mut(68) {
                g[0..2].copy_from_slice(&f16_bits(rng, false));
                g[2..4].copy_from_slice(&f16_bits(rng, true));
            }
        }
        other => panic!("d2_cpu_arm_bench: no synthetic layout for {other:?}"),
    }
    t
}

#[derive(Clone, Copy, PartialEq)]
enum Codec {
    E8p,
    Mq4,
}

impl Codec {
    fn quants(self) -> (CpuQuant, CpuQuant) {
        match self {
            Codec::E8p => (CpuQuant::E8pI4G128R256, CpuQuant::E8pI4G128R128),
            Codec::Mq4 => (CpuQuant::Mq4G256V2, CpuQuant::Mq4G128V2),
        }
    }
}

struct Expert {
    codec: Codec,
    gate_up: Vec<u8>,
    down: Vec<u8>,
}

fn expert(codec: Codec, rng: &mut Rng) -> Expert {
    let (gq, dq) = codec.quants();
    Expert {
        codec,
        gate_up: tensor(gq, 2 * MI, DIM, rng),
        down: tensor(dq, DIM, MI, rng),
    }
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("d2_cpu_arm_bench: bad value for {name}: {v}"))
        })
        .unwrap_or(default)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn layer(args: &[String]) {
    let share: f64 = arg(args, "--e8p-share", 0.54);
    let top_k: usize = arg(args, "--top-k", 10);
    let pool: usize = arg(args, "--pool", 160);
    let iters: usize = arg(args, "--iters", 2000);
    let gap_us: u64 = arg(args, "--gap-us", 500);
    let rows: usize = arg(args, "--rows", 1);
    let mut rng = Rng(arg(args, "--seed", 0x5EED));
    let e8p_n = if share > 0.0 { pool } else { 0 };
    let mq4_n = if share < 1.0 { pool } else { 0 };
    let t0 = Instant::now();
    let codecs =
        std::iter::repeat_n(Codec::E8p, e8p_n).chain(std::iter::repeat_n(Codec::Mq4, mq4_n));
    let experts: Vec<Expert> = codecs.map(|c| expert(c, &mut rng)).collect();
    let arena: usize = experts.iter().map(|e| e.gate_up.len() + e.down.len()).sum();
    eprintln!(
        "layer: {} E8P + {} MQ4 experts ({:.0} MiB) built in {:.1} s",
        e8p_n,
        mq4_n,
        arena as f64 / (1 << 20) as f64,
        t0.elapsed().as_secs_f64()
    );
    let x: Vec<f32> = (0..rows * DIM).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let slots = rows * top_k;
    let host_slots: Vec<u32> = (0..slots as u32).collect();
    let mut out = vec![0.0f32; slots * DIM];
    let mut ns = Vec::with_capacity(iters);
    let mut weights = 0f64;
    let mut bytes = 0f64;
    let warm = 50;
    for it in 0..warm + iters {
        // `top_k` distinct experts per row, each E8P with probability `share`.
        let mut ids = Vec::with_capacity(slots);
        let mut chosen: Vec<CpuMoeExpert<'_>> = Vec::with_capacity(slots);
        for _ in 0..rows {
            let mut picked: Vec<usize> = Vec::with_capacity(top_k);
            while picked.len() < top_k {
                let e8p = (rng.unit() as f64) < share;
                let base = if e8p { 0 } else { e8p_n };
                let n = if e8p { e8p_n } else { mq4_n };
                let i = base + (rng.next() as usize) % n;
                if !picked.contains(&i) {
                    picked.push(i);
                }
            }
            for &i in &picked {
                let e = &experts[i];
                let (gq, dq) = e.codec.quants();
                ids.push((i % N_EXPERTS) as i32);
                chosen.push(CpuMoeExpert {
                    gate_up_quant: gq,
                    gate_up: &e.gate_up,
                    down_quant: dq,
                    down: &e.down,
                });
            }
        }
        if gap_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(gap_us));
        }
        let t = Instant::now();
        moe_cpu_host_slot_outputs(
            DIM,
            MI,
            rows,
            top_k,
            N_EXPERTS,
            &ids,
            None,
            &host_slots,
            &chosen,
            &x,
            false,
            &mut out,
        )
        .expect("moe_cpu_host_slot_outputs");
        let dt = t.elapsed().as_nanos() as f64;
        if it >= warm {
            ns.push(dt);
            for c in &chosen {
                weights += (2 * MI * DIM + DIM * MI) as f64;
                bytes += (c.gate_up.len() + c.down.len()) as f64;
            }
        }
    }
    std::hint::black_box(&out);
    let total: f64 = ns.iter().sum();
    let mut sorted = ns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "layer e8p_share={share:.2} rows={rows} top_k={top_k} gap_us={gap_us} iters={iters}: \
         p50 {:.1} us  mean {:.1} us  p90 {:.1} us  | {:.1} Gw/s {:.1} GB/s (mean)",
        percentile(&sorted, 0.5) / 1e3,
        total / iters as f64 / 1e3,
        percentile(&sorted, 0.9) / 1e3,
        weights / total,
        bytes / total,
    );
}

fn kernel(args: &[String]) {
    let pool_n: usize = arg(args, "--pool", 96);
    let reps: usize = arg(args, "--reps", 200);
    // A one-worker splice pool (this process builds no other splice pool), so
    // `gemv`'s rayon region runs on one pinned core.
    let cfg = hipfire_cpu::splice_pool::SplicePoolConfig {
        threads: 1,
        pin: true,
    };
    let mut rng = Rng(0xC0DE);
    let (m, k) = (2 * MI, DIM);
    let x: Vec<f32> = (0..k).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let mut y = vec![0.0f32; m];
    for q in [CpuQuant::E8pI4G128R256, CpuQuant::Mq4G256V2] {
        let tensors: Vec<Vec<u8>> = (0..pool_n).map(|_| tensor(q, m, k, &mut rng)).collect();
        for (label, n) in [("resident", 1usize), ("streaming", pool_n)] {
            let mut best = f64::MAX;
            let mut sum = 0.0;
            hipfire_cpu::splice_pool::install(cfg, || {
                for r in 0..reps + 5 {
                    let t = Instant::now();
                    gemv(q, &tensors[r % n], m, k, &x, &mut y);
                    let dt = t.elapsed().as_nanos() as f64;
                    if r >= 5 {
                        best = best.min(dt);
                        sum += dt;
                    }
                }
            });
            let w = (m * k) as f64;
            println!(
                "kernel {q:?} {label}: {:.2} Gw/s best, {:.2} Gw/s mean ({:.1} GB/s mean)",
                w / best,
                w * reps as f64 / sum,
                tensor_bytes(q, m, k) as f64 * reps as f64 / sum
            );
        }
    }
    std::hint::black_box(&y);
}

/// `top_k` experts as one batched-GEMV region per projection, the way the CPU
/// arm issues them (one job per expert, one activation row each).
struct Region<'a> {
    quants: Vec<CpuQuant>,
    packed: Vec<&'a [u8]>,
    xs: Vec<&'a [f32]>,
}

impl hipfire_cpu::gemv::SharedJobSource for Region<'_> {
    fn jobs(&self) -> usize {
        self.packed.len()
    }
    fn n(&self, _: usize) -> usize {
        1
    }
    fn quant(&self, j: usize) -> CpuQuant {
        self.quants[j]
    }
    fn packed(&self, j: usize) -> &[u8] {
        self.packed[j]
    }
    fn xs_row(&self, j: usize, _: usize) -> &[f32] {
        self.xs[j]
    }
}

/// The CPU arm's two GEMV regions timed on their own (prep / rayon exec), to
/// split a layer's arm time into the regions and the serial rest.
fn regions(args: &[String]) {
    let share: f64 = arg(args, "--e8p-share", 1.0);
    let top_k: usize = arg(args, "--top-k", 10);
    let pool: usize = arg(args, "--pool", 160);
    let iters: usize = arg(args, "--iters", 1000);
    let gap_us: u64 = arg(args, "--gap-us", 400);
    let mut rng = Rng(arg(args, "--seed", 0x5EED));
    let e8p_n = if share > 0.0 { pool } else { 0 };
    let mq4_n = if share < 1.0 { pool } else { 0 };
    let codecs =
        std::iter::repeat_n(Codec::E8p, e8p_n).chain(std::iter::repeat_n(Codec::Mq4, mq4_n));
    let experts: Vec<Expert> = codecs.map(|c| expert(c, &mut rng)).collect();
    let x: Vec<f32> = (0..DIM).map(|_| rng.unit() * 2.0 - 1.0).collect();
    let hid: Vec<Vec<f32>> = (0..top_k)
        .map(|_| (0..MI).map(|_| rng.unit() * 2.0 - 1.0).collect())
        .collect();
    let mut gu = vec![0.0f32; 2 * MI * top_k];
    let mut dn = vec![0.0f32; DIM * top_k];
    let cfg = hipfire_cpu::splice_pool::SplicePoolConfig {
        threads: std::env::var("HIPFIRE_CPU_EXPERT_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        pin: true,
    };
    let (mut gu_t, mut gu_x, mut dn_t, mut dn_x) = (vec![], vec![], vec![], vec![]);
    let mut call_t = vec![];
    for it in 0..50 + iters {
        let mut picked: Vec<usize> = Vec::with_capacity(top_k);
        while picked.len() < top_k {
            let e8p = (rng.unit() as f64) < share;
            let (base, n) = if e8p { (0, e8p_n) } else { (e8p_n, mq4_n) };
            let i = base + (rng.next() as usize) % n;
            if !picked.contains(&i) {
                picked.push(i);
            }
        }
        let gsrc = Region {
            quants: picked
                .iter()
                .map(|&i| experts[i].codec.quants().0)
                .collect(),
            packed: picked.iter().map(|&i| &experts[i].gate_up[..]).collect(),
            xs: vec![&x[..]; top_k],
        };
        let dsrc = Region {
            quants: picked
                .iter()
                .map(|&i| experts[i].codec.quants().1)
                .collect(),
            packed: picked.iter().map(|&i| &experts[i].down[..]).collect(),
            xs: hid.iter().map(|h| &h[..]).collect(),
        };
        if gap_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(gap_us));
        }
        let t_call = Instant::now();
        let (g, d) = hipfire_cpu::splice_pool::install(cfg, || {
            let g =
                hipfire_cpu::gemv::gemv_shared_sourced_traced(2 * MI, DIM, &mut gu, &gsrc, None);
            let d = hipfire_cpu::gemv::gemv_shared_sourced_traced(DIM, MI, &mut dn, &dsrc, None);
            (g, d)
        });
        let call = t_call.elapsed().as_nanos() as f64;
        if it >= 50 {
            call_t.push(call);
            gu_t.push(g.total_ns as f64);
            gu_x.push(g.exec_ns as f64);
            dn_t.push(d.total_ns as f64);
            dn_x.push(d.exec_ns as f64);
        }
    }
    std::hint::black_box((&gu, &dn));
    let p50 = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        percentile(v, 0.5) / 1e3
    };
    println!(
        "regions e8p_share={share:.2} gap_us={gap_us}: call p50 {:.1} us | gate_up p50 {:.1} us (exec {:.1}), down p50 {:.1} us (exec {:.1})",
        p50(&mut call_t),
        p50(&mut gu_t),
        p50(&mut gu_x),
        p50(&mut dn_t),
        p50(&mut dn_x)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("layer") => layer(&args),
        Some("kernel") => kernel(&args),
        Some("regions") => regions(&args),
        _ => {
            eprintln!("usage: d2_cpu_arm_bench layer|regions|kernel [--flags]");
            std::process::exit(2);
        }
    }
}
