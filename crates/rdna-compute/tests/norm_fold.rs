// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lab gate: the A3B decode norm fold (`fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201`)
//! against the two-launch chain it replaces (`fused_rmsnorm_rotate_mq` then the
//! MQ4G256V2 projection GEMV, i.e. the PeaceMaker-native twin on gfx1201), at the
//! three production shapes: GDN QKVZA (8192/4096/32/32), FA QKV (8192/512/512) and
//! the MoE router + shared quartet (256/1/512/512), hidden 2048, random MQ4G256V2
//! weights in the real layout.
//!
//! Compared byte-for-byte: the rotated vector and every projection output (all
//! outputs are poisoned first, so an unwritten row fails). Gates:
//!   * 10 random seeds + an outlier row + an all-zero row;
//!   * grid sweep (1, 2, 3, 8, 21, 64, 129, 258, 1544);
//!   * negative control (a one-ulp input perturbation must change the output);
//!   * 1000 back-to-back launches per shape, every launch compared;
//!   * informational per-iteration GPU time of both arms (hipEvent batches).
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --features lab --test norm_fold \
//!       -- --ignored --nocapture
#![cfg(feature = "lab")]

use rdna_compute::norm_fold::{norm_fold_grid, NormFoldArgs};
use rdna_compute::{DType, Gpu, GpuTensor};

const HIDDEN: usize = 2048;
const EPS: f32 = 1e-6;
const POISON: u32 = 0x7FC0_DEAD;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_unit(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        f32::from_bits(0x3F80_0000 | a) - 1.0
    }
    fn f32_signed(&mut self) -> f32 {
        let v = self.f32_unit();
        if self.next_u32() & 1 == 1 {
            -v
        } else {
            v
        }
    }
    fn f16_normal(&mut self, lo: u32, hi: u32, signed: bool) -> u16 {
        let exp = lo + self.next_u32() % (hi - lo + 1);
        let mant = self.next_u32() & 0x3FF;
        let sign = if signed { self.next_u32() & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

fn upload_u8(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc u8");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod u8");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len()], DType::F32)
        .expect("alloc f32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod f32");
    t
}
fn poison(gpu: &mut Gpu, t: &GpuTensor, n: usize) {
    let v = vec![POISON; n];
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("poison");
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// MQ4G256V2 (qt=44) rows: 8 groups x 136 B `[sc0 zp0 sc1 zp1 : f16][128 nibble bytes]`.
fn synth_qt44(rows: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let groups = 8;
    let mut out = vec![0u8; rows * groups * 136];
    for r in 0..rows * groups {
        let off = r * 136;
        for half in 0..2 {
            let sc = rng.f16_normal(6, 9, false);
            let zp = rng.f16_normal(5, 9, true);
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for w in 0..32 {
            let pk = rng.next_u32() ^ (rng.next_u32() << 16);
            out[off + 8 + 4 * w..off + 12 + 4 * w].copy_from_slice(&pk.to_le_bytes());
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    m: [usize; 4],
}
const SHAPES: [Shape; 3] = [
    Shape {
        name: "gdn-qkvza",
        m: [8192, 4096, 32, 32],
    },
    Shape {
        name: "fa-qkv",
        m: [8192, 512, 512, 0],
    },
    Shape {
        name: "moe-gate",
        m: [256, 1, 512, 512],
    },
];

struct Rig {
    shape: Shape,
    weights: Vec<GpuTensor>,
    outs: Vec<GpuTensor>,
    x: GpuTensor,
    norm_w: GpuTensor,
    x_rot: GpuTensor,
}

fn build_rig(gpu: &mut Gpu, shape: Shape, seed: u64) -> Rig {
    let mut weights = Vec::new();
    let mut outs = Vec::new();
    for (i, &m) in shape.m.iter().enumerate() {
        let rows = m.max(1);
        weights.push(upload_u8(gpu, &synth_qt44(rows, seed ^ (0x100 + i as u64))));
        outs.push(upload_f32(gpu, &vec![f32::from_bits(POISON); rows]));
    }
    let mut rng = Lcg(seed ^ 0x77);
    let x: Vec<f32> = (0..HIDDEN).map(|_| rng.f32_signed() * 4.0).collect();
    let nw: Vec<f32> = (0..HIDDEN).map(|_| 0.5 + rng.f32_unit()).collect();
    Rig {
        shape,
        weights,
        outs,
        x: upload_f32(gpu, &x),
        norm_w: upload_f32(gpu, &nw),
        x_rot: upload_f32(gpu, &vec![f32::from_bits(POISON); HIDDEN]),
    }
}

fn set_x(gpu: &mut Gpu, rig: &Rig, x: &[f32]) {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
    gpu.hip.memcpy_htod(&rig.x.buf, bytes).expect("set x");
}

fn reset(gpu: &mut Gpu, rig: &Rig) {
    poison(gpu, &rig.x_rot, HIDDEN);
    for (i, &m) in rig.shape.m.iter().enumerate() {
        poison(gpu, &rig.outs[i], m.max(1));
    }
}

#[derive(Clone, PartialEq)]
struct Out {
    x_rot: Vec<u32>,
    outs: Vec<Vec<u32>>,
}

fn download(gpu: &Gpu, rig: &Rig) -> Out {
    Out {
        x_rot: download_u32(gpu, &rig.x_rot, HIDDEN),
        outs: rig
            .shape
            .m
            .iter()
            .enumerate()
            .map(|(i, &m)| download_u32(gpu, &rig.outs[i], m.max(1)))
            .collect(),
    }
}

/// Number of differing words (rotated vector + every output), printing the first few.
fn diff(tag: &str, got: &Out, want: &Out) -> usize {
    let mut total = 0;
    let mut report = |what: &str, g: &[u32], w: &[u32]| {
        let bad: Vec<usize> = (0..g.len()).filter(|&i| g[i] != w[i]).collect();
        if !bad.is_empty() {
            eprintln!(
                "[{tag}] {what}: {} of {} words differ; first {:?}",
                bad.len(),
                g.len(),
                bad.iter()
                    .take(4)
                    .map(|&i| (i, g[i], w[i]))
                    .collect::<Vec<_>>()
            );
            total += bad.len();
        }
    };
    report("x_rot", &got.x_rot, &want.x_rot);
    for (i, (g, w)) in got.outs.iter().zip(&want.outs).enumerate() {
        report(&format!("out{i}"), g, w);
    }
    total
}

/// The incumbent chain: plain norm launch, then the consumer GEMV launch.
fn run_incumbent(gpu: &mut Gpu, rig: &Rig) {
    gpu.fused_rmsnorm_rotate_mq(&rig.x, &rig.norm_w, &rig.x_rot, HIDDEN, EPS)
        .expect("norm");
    let m = rig.shape.m;
    if m[3] > 0 {
        gpu.fused_qkvza_hfq4g256_mq4v2(
            &rig.weights[0],
            &rig.weights[1],
            &rig.weights[2],
            &rig.weights[3],
            &rig.x_rot,
            &rig.outs[0],
            &rig.outs[1],
            &rig.outs[2],
            &rig.outs[3],
            m[0],
            m[1],
            m[2],
            m[3],
            HIDDEN,
        )
        .expect("qkvza");
    } else {
        gpu.fused_qkv_hfq4g256_mq4v2(
            &rig.weights[0],
            &rig.weights[1],
            &rig.weights[2],
            &rig.x_rot,
            &rig.outs[0],
            &rig.outs[1],
            &rig.outs[2],
            m[0],
            m[1],
            m[2],
            HIDDEN,
        )
        .expect("qkv");
    }
}

fn run_fold(gpu: &mut Gpu, rig: &Rig, grid: usize) {
    let w = &rig.weights;
    let o = &rig.outs;
    let args = NormFoldArgs {
        x: &rig.x,
        norm_weight: &rig.norm_w,
        x_rot: &rig.x_rot,
        weights: [&w[0], &w[1], &w[2], &w[3]],
        outs: [&o[0], &o[1], &o[2], &o[3]],
        m: rig.shape.m,
        eps: EPS,
    };
    gpu.fused_rmsnorm_proj4_mq4g256v2_fold(&args, grid)
        .expect("fold");
}

fn default_grid(shape: &Shape) -> usize {
    let total: usize = shape.m.iter().sum();
    norm_fold_grid(total, if total > 4000 { 12 } else { 3 })
}

fn incumbent_out(gpu: &mut Gpu, rig: &Rig) -> Out {
    reset(gpu, rig);
    run_incumbent(gpu, rig);
    gpu.hip.device_synchronize().expect("sync");
    download(gpu, rig)
}

fn fold_out(gpu: &mut Gpu, rig: &Rig, grid: usize) -> Out {
    reset(gpu, rig);
    run_fold(gpu, rig, grid);
    gpu.hip.device_synchronize().expect("sync");
    download(gpu, rig)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn norm_fold_is_byte_identical_to_the_two_launch_chain() {
    let mut gpu = Gpu::init().expect("GPU required");
    assert!(
        gpu.arch.starts_with("gfx1201"),
        "needs exact gfx1201, got {}",
        gpu.arch
    );
    gpu.ensure_mq_signs().expect("signs");
    let mut failures = 0usize;

    for shape in SHAPES {
        // ---- 10 random seeds, default grid ---------------------------------
        for seed in 0..10u64 {
            let rig = build_rig(&mut gpu, shape, 0x5EED_0000 + seed * 0x9E37);
            let want = incumbent_out(&mut gpu, &rig);
            assert!(
                want.outs
                    .iter()
                    .zip(shape.m)
                    .all(|(o, m)| m == 0 || o.iter().all(|&w| w != POISON)),
                "incumbent left poison in an output (shape {})",
                shape.name
            );
            let got = fold_out(&mut gpu, &rig, default_grid(&shape));
            failures += diff(&format!("{} seed {seed}", shape.name), &got, &want);
        }

        // ---- special rows ---------------------------------------------------
        {
            let rig = build_rig(&mut gpu, shape, 0xABCD);
            let mut x: Vec<f32> = vec![0.0; HIDDEN];
            set_x(&mut gpu, &rig, &x);
            let want = incumbent_out(&mut gpu, &rig);
            let got = fold_out(&mut gpu, &rig, default_grid(&shape));
            failures += diff(&format!("{} zero row", shape.name), &got, &want);
            let mut rng = Lcg(99);
            for v in x.iter_mut() {
                *v = rng.f32_signed();
            }
            x[17] = 5000.0;
            x[1900] = -3.5e4;
            x[3] = 1e-30;
            set_x(&mut gpu, &rig, &x);
            let want = incumbent_out(&mut gpu, &rig);
            let got = fold_out(&mut gpu, &rig, default_grid(&shape));
            failures += diff(&format!("{} outlier row", shape.name), &got, &want);
        }

        // ---- grid sweep -----------------------------------------------------
        let rig = build_rig(&mut gpu, shape, 0x1234_5678);
        let want = incumbent_out(&mut gpu, &rig);
        for grid in [1usize, 2, 3, 8, 21, 64, 129, 258, 1544] {
            let got = fold_out(&mut gpu, &rig, grid);
            failures += diff(&format!("{} grid {grid}", shape.name), &got, &want);
        }

        // ---- negative control -----------------------------------------------
        // A perturbed input must move the fold output (and still equal the
        // incumbent on that input): a 2^10-ulp change must differ from the
        // unperturbed output; the one-ulp case is reported.
        for (what, flip) in [("1-ulp", 1u32), ("2^10-ulp", 1u32 << 10)] {
            let mut rng = Lcg(0x1234_5678 ^ 0x77);
            let mut x: Vec<f32> = (0..HIDDEN).map(|_| rng.f32_signed() * 4.0).collect();
            x[700] = f32::from_bits(x[700].to_bits() ^ flip);
            set_x(&mut gpu, &rig, &x);
            let inc = incumbent_out(&mut gpu, &rig);
            let got = fold_out(&mut gpu, &rig, default_grid(&shape));
            failures += diff(&format!("{} perturbed {what}", shape.name), &got, &inc);
            eprintln!(
                "[negative control] {} {what}: fold output differs from unperturbed: {}",
                shape.name,
                got != want
            );
            if flip > 1 {
                assert!(
                    got != want,
                    "negative control: a {what} input change did not change the fold output ({})",
                    shape.name
                );
            }
        }
        // restore the unperturbed row for the stress below
        {
            let mut rng = Lcg(0x1234_5678 ^ 0x77);
            let x: Vec<f32> = (0..HIDDEN).map(|_| rng.f32_signed() * 4.0).collect();
            set_x(&mut gpu, &rig, &x);
        }

        // ---- 1000 back-to-back launches, every launch compared ----------------
        let grid = default_grid(&shape);
        let mut stress_bad = 0usize;
        for i in 0..1000 {
            reset(&mut gpu, &rig);
            run_fold(&mut gpu, &rig, grid);
            gpu.hip.device_synchronize().expect("sync");
            let got = download(&gpu, &rig);
            let d = diff(&format!("{} stress {i}", shape.name), &got, &want);
            stress_bad += d;
            if stress_bad > 0 {
                break;
            }
        }
        failures += stress_bad;

        // ---- informational timing ---------------------------------------------
        {
            let ev0 = gpu.hip.event_create().expect("ev");
            let ev1 = gpu.hip.event_create().expect("ev");
            let batch = 200usize;
            let mut t_inc = Vec::new();
            let mut grids = vec![1usize, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512];
            grids.retain(|&g| g * 8 <= shape.m.iter().sum::<usize>().max(8));
            let mut t_fold: Vec<(usize, Vec<f64>)> =
                grids.iter().map(|&g| (g, Vec::new())).collect();
            for _ in 0..5 {
                gpu.hip.event_record(&ev0, None).unwrap();
                for _ in 0..batch {
                    run_incumbent(&mut gpu, &rig);
                }
                gpu.hip.event_record(&ev1, None).unwrap();
                gpu.hip.event_synchronize(&ev1).unwrap();
                t_inc.push(
                    gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1000.0 / batch as f64,
                );
                for (g, ts) in t_fold.iter_mut() {
                    gpu.hip.event_record(&ev0, None).unwrap();
                    for _ in 0..batch {
                        run_fold(&mut gpu, &rig, *g);
                    }
                    gpu.hip.event_record(&ev1, None).unwrap();
                    gpu.hip.event_synchronize(&ev1).unwrap();
                    ts.push(
                        gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1000.0
                            / batch as f64,
                    );
                }
            }
            eprintln!(
                "[timing] {}: incumbent norm+gemv {:.2} us/iter",
                shape.name,
                median(t_inc)
            );
            for (g, ts) in t_fold {
                eprintln!(
                    "[timing] {}: fold grid {:>4} ({:>5.1} rows/wave) {:.2} us/iter",
                    shape.name,
                    g,
                    shape.m.iter().sum::<usize>() as f64 / (g * 8) as f64,
                    median(ts)
                );
            }
            gpu.hip.event_destroy(ev0).unwrap();
            gpu.hip.event_destroy(ev1).unwrap();
        }
        eprintln!("[norm_fold] {} done", shape.name);
    }
    assert_eq!(failures, 0, "{failures} differing words in total");
}
