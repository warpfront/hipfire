// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node-chain bench for prepared-A8 activation reuse (plan §6 G1).
//!
//! Compares, per row count, the *complete* Raon Q8 mixer-projection chains of one layer
//! through the production dispatch entry points:
//!
//!   * `incumbent`: every A8 consumer quantizes its own `x`, runs the GEMM, then the separate
//!     in-place BF16 round pass (what `project(.., OutRound::Bf16, A8)` ran before this change);
//!   * `reuse`: one `q8_0_prefill_prepare` per shared activation, then
//!     `q8_0_prefill_gemm_prepared` for every consumer (round still a separate pass);
//!   * `reuse+fuse`: reuse, and the BF16 round applied in the GEMM epilogue (no round pass).
//!
//! Chains (Raon-1-mini shapes, hidden 2560): RDA = q,k,v,g (4 x 5120x2560 on one `x`) + o
//! (2560x5120); MLA = q_a (576x2560), kv_a (576x2560), g (5120x2560) on one `x`, then q_b
//! (7680x576), kv_b (10240x512), o (2560x5120). Every output of the two arms is compared byte
//! for byte. Class ms = 18 RDA layers + 7 MLA layers of the measured chain.
//!
//! Also reports the quantizer alone, the GEMM alone, and a concatenated q/k/v/g weight single
//! GEMM (M = 20480; layout not split, an upper bound for the concat variant).
//!
//! Usage: bench_a8_reuse_chain [--rows 1,2,4,8,16,64,512,4096,16384] [--iters 21] [--out f.md]
//! Needs a gfx1201 GPU (run through fabric-run.sh).

use hip_bridge::Event;
use hipfire_dispatch::pipeline::{
    q8_0_prefill_gemm_prepared, q8_0_prefill_prepare, PrefillActTier,
};
use rdna_compute::{DType, Gpu, GpuTensor, Q8A8Sched, Q8A8Tile};
use std::fmt::Write as _;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// F32 -> F16 bits for a positive normal-range value (weight block scales).
fn f16_bits(f: f32) -> u16 {
    let u = f.to_bits();
    let exp = ((u >> 23) & 0xff) as i32 - 127 + 15;
    let man = (u >> 13) & 0x3ff;
    assert!((1..31).contains(&exp), "scale out of the F16 normal range");
    ((exp as u16) << 10) | man as u16
}

fn make_weights(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / 32;
    let mut bytes = Vec::with_capacity(m * nb * 34);
    for _ in 0..m * nb {
        bytes.extend_from_slice(&f16_bits(0.004 * (1.0 + 0.5 * rng.gauss().abs())).to_le_bytes());
        for _ in 0..32 {
            bytes.push((rng.gauss() * 38.0).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    bytes
}

/// BF16-valued activations with persistent massive channels and a few hot rows.
fn make_acts(n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let hot_ch: Vec<f32> = (0..k)
        .map(|_| {
            if rng.unit() < 0.008 {
                25.0 + 35.0 * rng.unit()
            } else {
                1.0
            }
        })
        .collect();
    let mut x = Vec::with_capacity(n * k);
    for _ in 0..n {
        let row = if rng.unit() < 0.015 { 8.0 } else { 1.0 };
        for c in 0..k {
            x.push(bf16_round(rng.gauss() * row * hot_ch[c]));
        }
    }
    x
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Timer {
    e0: Event,
    e1: Event,
}

fn time_ms(gpu: &mut Gpu, t: &Timer, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f32 {
    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters + 3 {
        gpu.hip.event_record(&t.e0, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&t.e1, None).unwrap();
        gpu.hip.event_synchronize(&t.e1).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&t.e0, &t.e1).unwrap();
        if i >= 3 {
            samples.push(ms);
        }
    }
    median(samples)
}

fn alloc_outs(gpu: &mut Gpu, rows: usize, ms: &[usize]) -> Vec<GpuTensor> {
    ms.iter()
        .map(|&m| gpu.alloc_tensor(&[rows * m], DType::F32).unwrap())
        .collect()
}

struct Site {
    w: GpuTensor,
    m: usize,
    k: usize,
}

fn site(gpu: &mut Gpu, m: usize, k: usize, rng: &mut Rng) -> Site {
    let b = make_weights(m, k, rng);
    let w = gpu.upload_raw(&b, &[b.len()]).expect("w");
    Site { w, m, k }
}

/// One chain = groups of consumers; a group shares one activation tensor.
struct Group<'a> {
    x: &'a GpuTensor,
    sites: Vec<(&'a Site, &'a GpuTensor)>,
}

#[derive(Clone, Copy)]
struct Arm {
    reuse: bool,
    fuse: bool,
}

const INCUMBENT: Arm = Arm {
    reuse: false,
    fuse: false,
};
const REUSE: Arm = Arm {
    reuse: true,
    fuse: false,
};
const FULL: Arm = Arm {
    reuse: true,
    fuse: true,
};

/// `project(.., Bf16, A8)` over every consumer of every group, in the arm's mode.
fn run(gpu: &mut Gpu, groups: &[Group], rows: usize, arm: Arm) {
    for g in groups {
        let k = g.sites[0].0.k;
        let prep = if arm.reuse && g.sites.len() > 1 {
            q8_0_prefill_prepare(gpu, PrefillActTier::A8, g.x, k, rows).unwrap()
        } else {
            None
        };
        for (s, y) in &g.sites {
            let r = q8_0_prefill_gemm_prepared(
                gpu,
                PrefillActTier::A8,
                prep.as_ref(),
                &s.w,
                g.x,
                y,
                s.m,
                s.k,
                rows,
                arm.fuse,
            )
            .unwrap();
            if !r.bf16_fused {
                gpu.bf16_round_trip_f32_strided(y, rows, 0, s.m, s.m)
                    .unwrap();
            }
        }
    }
}

type Chain<'a> = (
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
);

#[allow(clippy::type_complexity)]
fn split(c: Chain<'_>) -> ([Group<'_>; 2], [Group<'_>; 4]) {
    let (a, b, cc, d, e, f) = c;
    ([a, b], [cc, d, e, f])
}

struct Sites<'a> {
    rda: [&'a Site; 5],
    mla: [&'a Site; 6],
}

/// Consumer groups of the two layer types; `ys` holds the 11 outputs in site order
/// (q,k,v,g,o, q_a,kv_a,g, q_b,kv_b,o).
fn chain<'a>(
    s: &'a Sites<'a>,
    ys: &'a [GpuTensor],
    xs: [&'a GpuTensor; 4],
) -> (
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
    Group<'a>,
) {
    let [x, x5, x576, x512] = xs;
    let rda_qkvg = Group {
        x,
        sites: vec![
            (s.rda[0], &ys[0]),
            (s.rda[1], &ys[1]),
            (s.rda[2], &ys[2]),
            (s.rda[3], &ys[3]),
        ],
    };
    let rda_o = Group {
        x: x5,
        sites: vec![(s.rda[4], &ys[4])],
    };
    let mla_x = Group {
        x,
        sites: vec![(s.mla[0], &ys[5]), (s.mla[1], &ys[6]), (s.mla[2], &ys[7])],
    };
    let mla_qb = Group {
        x: x576,
        sites: vec![(s.mla[3], &ys[8])],
    };
    let mla_kvb = Group {
        x: x512,
        sites: vec![(s.mla[4], &ys[9])],
    };
    let mla_o = Group {
        x: x5,
        sites: vec![(s.mla[5], &ys[10])],
    };
    (rda_qkvg, rda_o, mla_x, mla_qb, mla_kvb, mla_o)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let iters: usize = arg("--iters").map_or(21, |s| s.parse().unwrap());
    let rows_list: Vec<usize> = arg("--rows")
        .map_or(vec![1, 2, 4, 8, 16, 64, 512, 4096, 16384], |s| {
            s.split(',').map(|v| v.parse().unwrap()).collect()
        });
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch {}", gpu.arch);
    let timer = Timer {
        e0: gpu.hip.event_create().unwrap(),
        e1: gpu.hip.event_create().unwrap(),
    };
    let mut rng = Rng(0x5eed_0a8a_6001);
    // RDA
    let (qp, kp, vp, gp) = (
        site(&mut gpu, 5120, 2560, &mut rng),
        site(&mut gpu, 5120, 2560, &mut rng),
        site(&mut gpu, 5120, 2560, &mut rng),
        site(&mut gpu, 5120, 2560, &mut rng),
    );
    let rda_o = site(&mut gpu, 2560, 5120, &mut rng);
    // concatenated q/k/v/g
    let cat = {
        let mut b = Vec::new();
        for _ in 0..4 {
            b.extend_from_slice(&make_weights(5120, 2560, &mut rng));
        }
        let w = gpu.upload_raw(&b, &[b.len()]).expect("cat");
        Site {
            w,
            m: 20480,
            k: 2560,
        }
    };
    // MLA
    let q_a = site(&mut gpu, 576, 2560, &mut rng);
    let kv_a = site(&mut gpu, 576, 2560, &mut rng);
    let mla_g = site(&mut gpu, 5120, 2560, &mut rng);
    let q_b = site(&mut gpu, 7680, 576, &mut rng);
    let kv_b = site(&mut gpu, 10240, 512, &mut rng);
    let mla_o = site(&mut gpu, 2560, 5120, &mut rng);

    let sites = Sites {
        rda: [&qp, &kp, &vp, &gp, &rda_o],
        mla: [&q_a, &kv_a, &mla_g, &q_b, &kv_b, &mla_o],
    };
    let mut md = String::from(
        "| rows | chain | incumbent ms | reuse ms | reuse+fuse ms | reuse gain | reuse+fuse gain | identical | notes |\n|---:|---|---:|---:|---:|---:|---:|---|---|\n",
    );
    let mut class_md = String::from(
        "| rows | RDA layer inc / reuse / reuse+fuse ms | MLA layer inc / reuse / reuse+fuse ms | class inc ms (18 RDA + 7 MLA) | class reuse | class reuse+fuse | reuse gain | reuse+fuse gain |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    let mut kern_md = String::from(
        "| rows | M x K | plain 128x128 ms (TOP/s) | pipelined-SB ms (TOP/s) | gain | bit-identical |\n|---:|---|---|---|---:|---|\n",
    );
    let mut all_ok = true;
    let ms_list = [
        5120, 5120, 5120, 5120, 2560, 576, 576, 5120, 7680, 10240, 2560,
    ];
    for &rows in &rows_list {
        println!("rows {rows}");
        let mkx = |gpu: &mut Gpu, rng: &mut Rng, k: usize| {
            let h = make_acts(rows, k, rng);
            gpu.upload_f32(&h, &[rows * k]).unwrap()
        };
        let x = mkx(&mut gpu, &mut rng, 2560);
        let x5 = mkx(&mut gpu, &mut rng, 5120);
        let x576 = mkx(&mut gpu, &mut rng, 576);
        let x512 = mkx(&mut gpu, &mut rng, 512);
        let ya = alloc_outs(&mut gpu, rows, &ms_list);
        let yb = alloc_outs(&mut gpu, rows, &ms_list);
        let yc = alloc_outs(&mut gpu, rows, &ms_list);
        let ycat = gpu.alloc_tensor(&[rows * 20480], DType::F32).unwrap();
        let xs = [&x, &x5, &x576, &x512];
        let (rda_a, mla_a) = split(chain(&sites, &ya, xs));
        let (rda_b, mla_b) = split(chain(&sites, &yb, xs));
        let (rda_c, mla_c) = split(chain(&sites, &yc, xs));
        for (r, m, arm) in [
            (&rda_a, &mla_a, INCUMBENT),
            (&rda_b, &mla_b, REUSE),
            (&rda_c, &mla_c, FULL),
        ] {
            run(&mut gpu, r, rows, arm);
            run(&mut gpu, m, rows, arm);
        }
        gpu.hip.device_synchronize().unwrap();
        let mut same = true;
        for i in 0..ya.len() {
            let ba = gpu.download_raw_bytes(&ya[i]).unwrap();
            same &= ba == gpu.download_raw_bytes(&yb[i]).unwrap();
            same &= ba == gpu.download_raw_bytes(&yc[i]).unwrap();
        }
        all_ok &= same;

        let t = |gpu: &mut Gpu, g: &[Group], arm: Arm| {
            time_ms(gpu, &timer, iters, |gg| run(gg, g, rows, arm))
        };
        let rda = [
            t(&mut gpu, &rda_a, INCUMBENT),
            t(&mut gpu, &rda_b, REUSE),
            t(&mut gpu, &rda_c, FULL),
        ];
        let mla = [
            t(&mut gpu, &mla_a, INCUMBENT),
            t(&mut gpu, &mla_b, REUSE),
            t(&mut gpu, &mla_c, FULL),
        ];
        let qkvg = [
            t(&mut gpu, &rda_a[..1], INCUMBENT),
            t(&mut gpu, &rda_b[..1], REUSE),
            t(&mut gpu, &rda_c[..1], FULL),
        ];
        let mlax = [
            t(&mut gpu, &mla_a[..1], INCUMBENT),
            t(&mut gpu, &mla_b[..1], REUSE),
            t(&mut gpu, &mla_c[..1], FULL),
        ];
        let admitted = q8_0_prefill_prepare(&mut gpu, PrefillActTier::A8, &x, 2560, rows)
            .unwrap()
            .is_some();
        let mut notes = if admitted {
            "A8 route".to_string()
        } else {
            "F16 fallback (prepare = None)".to_string()
        };
        let mut quant = (0.0f32, 0.0f32);
        if admitted {
            quant.0 = time_ms(&mut gpu, &timer, iters, |g| {
                g.q8_a8_quantize_x(&x, rows, 2560).unwrap();
            });
            quant.1 = time_ms(&mut gpu, &timer, iters, |g| {
                g.q8_a8_quantize_x(&x5, rows, 5120).unwrap();
            });
            let acts = gpu.q8_a8_quantize_x(&x, rows, 2560).unwrap();
            let four = time_ms(&mut gpu, &timer, iters, |g| {
                for (s, y) in [(&qp, &ya[0]), (&kp, &ya[1]), (&vp, &ya[2]), (&gp, &ya[3])] {
                    g.gemm_q8_0_a8_gemm(&s.w, &acts, y, s.m).unwrap();
                }
            });
            let cat_ms = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_a8_gemm(&cat.w, &acts, &ycat, cat.m).unwrap();
            });
            let _ = write!(
                notes,
                "; quant K2560 {:.4} ms, K5120 {:.4} ms; qkvg 4 GEMMs {four:.4} ms vs concat M20480 {cat_ms:.4} ms ({:+.2}%)",
                quant.0,
                quant.1,
                (four - cat_ms) / four * 100.0
            );
        }
        let g = |inc: f32, v: f32| (inc - v) / inc * 100.0;
        if admitted {
            // GEMM-only rate of the 128x128 FAST tile: plain vs pipelined single-buffer schedule,
            // bitwise compared.
            let ys = gpu.alloc_tensor(&[rows * 10240], DType::F32).unwrap();
            let yp = gpu.alloc_tensor(&[rows * 10240], DType::F32).unwrap();
            for (name, s_, xin) in [
                ("5120x2560", &qp, &x),
                ("2560x5120", &rda_o, &x5),
                ("576x2560", &q_a, &x),
                ("7680x576", &q_b, &x576),
                ("10240x512", &kv_b, &x512),
            ] {
                let acts = gpu.q8_a8_quantize_x(xin, rows, s_.k).unwrap();
                let tile = Q8A8Tile::T128x128;
                let tf = 2.0 * s_.m as f64 * s_.k as f64 * rows as f64 / 1e12;
                let mut ms = [0.0f32; 2];
                for (i, sched) in [Q8A8Sched::Plain, Q8A8Sched::PipelinedSb]
                    .into_iter()
                    .enumerate()
                {
                    let yy = if i == 0 { &ys } else { &yp };
                    ms[i] = time_ms(&mut gpu, &timer, iters, |gg| {
                        gg.gemm_q8_0_a8_gemm_sched(&s_.w, &acts, yy, s_.m, tile, false, sched)
                            .unwrap()
                    });
                }
                gpu.gemm_q8_0_a8_gemm_sched(&s_.w, &acts, &ys, s_.m, tile, false, Q8A8Sched::Plain)
                    .unwrap();
                gpu.gemm_q8_0_a8_gemm_sched(
                    &s_.w,
                    &acts,
                    &yp,
                    s_.m,
                    tile,
                    false,
                    Q8A8Sched::PipelinedSb,
                )
                .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let ident = gpu.download_raw_bytes(&ys).unwrap()[..rows * s_.m * 4]
                    == gpu.download_raw_bytes(&yp).unwrap()[..rows * s_.m * 4];
                all_ok &= ident;
                let _ = writeln!(
                    kern_md,
                    "| {rows} | {name} | {:.4} ({:.1}) | {:.4} ({:.1}) | {:+.2}% | {ident} |",
                    ms[0],
                    tf / (ms[0] as f64 / 1e3),
                    ms[1],
                    tf / (ms[1] as f64 / 1e3),
                    g(ms[0], ms[1]),
                );
            }
        }
        for (name, a, nt) in [
            ("RDA q/k/v/g", qkvg, notes.as_str()),
            ("RDA q/k/v/g + o", rda, ""),
            ("MLA q_a/kv_a/g", mlax, ""),
            ("MLA q_a/kv_a/g + q_b + kv_b + o", mla, ""),
        ] {
            let _ =
                writeln!(
                md,
                "| {rows} | {name} | {:.4} | {:.4} | {:.4} | {:+.2}% | {:+.2}% | {same} | {nt} |",
                a[0], a[1], a[2], g(a[0], a[1]), g(a[0], a[2])
            );
        }
        let cl = |i: usize| 18.0 * rda[i] + 7.0 * mla[i];
        let _ = writeln!(
            class_md,
            "| {rows} | {:.4} / {:.4} / {:.4} | {:.4} / {:.4} / {:.4} | {:.3} | {:.3} | {:.3} | {:+.2}% | {:+.2}% |",
            rda[0], rda[1], rda[2], mla[0], mla[1], mla[2], cl(0), cl(1), cl(2), g(cl(0), cl(1)), g(cl(0), cl(2))
        );
    }
    let outp = format!(
        "## A8 reuse / fused-round node chains (median of {iters}, GPU events; all outputs byte-identical across arms: {all_ok})\n\n{md}\n## Class (per forward, Q8 mixer projections incl. BF16 round)\n\n{class_md}\n## GEMM-only kernel rate: plain vs pipelined single-buffer schedule (T128x128)\n\n{kern_md}"
    );
    println!("\n{outp}");
    if let Some(path) = arg("--out") {
        std::fs::write(&path, &outp).expect("write --out");
    }
    assert!(all_ok, "an arm differs from the incumbent");
}
