// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU gate for the gfx1201 lane-WMMA prefill tile GEMM (`lane_wmma_tile_gemm`, kernel
//! `qwen4_lwmma_tile.gfx1201.hip`):
//!
//! * `lwmma_tile_oracle_hw`: every output word of the tile kernel (the launcher's pick AND every
//!   instantiated tile configuration) must be byte-equal to
//!   `lane_wmma_gemm(.., LaneWmmaCfg { nn: 1, split_k }, ..)` at the pinned prefill split-K, for
//!   every served shape x row count. Reports the count of differing words.
//! * `lwmma_tile_bench_hw`: median us of `Gpu::lane_wmma_gemm_prefill` (incumbent) vs the tile
//!   launcher (every configuration, then the pick), printed as `[tile-bench]` lines.
//!
//! Both skip (pass) when no exact gfx1201 device is visible. Run only through the fabric:
//!
//!   cargo test --release -p rdna-compute --test lwmma_tile_hw -- --nocapture --test-threads 1

use rdna_compute::lane_wmma::{
    lane_wmma_gemm, lane_wmma_prefill_plan, LaneWmmaCfg, LaneWmmaScratch, LaneWmmaWeight,
    LANE_WMMA_PREFILL_SPLIT_ROWS,
};
use rdna_compute::lane_wmma_tile::{
    lane_wmma_tile_gemm, lane_wmma_tile_gemm_cfg, lane_wmma_tile_pick, lane_wmma_tile_set,
    LaneWmmaTile,
};
use rdna_compute::{DType, Gpu, GpuTensor};

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u32() as f64) + 0.5) / 2147483648.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit();
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let r = 0x7FFFu32 + ((b >> 16) & 1);
    (b.wrapping_add(r) >> 16) as u16
}
fn bf16r(v: f32) -> f32 {
    f32::from_bits((bf16_bits(v) as u32) << 16)
}

fn upload_bytes(gpu: &mut Gpu, dtype: DType, numel: usize, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[numel], dtype).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, DType::F32, data.len(), bytes)
}
fn upload_bf16(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) };
    upload_bytes(gpu, DType::BF16, bits.len(), bytes)
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

fn gpu_1201() -> Option<Gpu> {
    let gpu = Gpu::init().ok()?;
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return None;
    }
    Some(gpu)
}

/// The served prefill set (T2 bench): (m, k, terms, hc_act).
const SHAPES: &[(usize, usize, usize, Option<f32>)] = &[
    (320, 10240, 1, Some(0.25)),
    (512, 2560, 2, None),
    (640, 2560, 2, None),
    (2560, 640, 2, None),
    (2560, 2560, 2, None),
    (10240, 2560, 2, None),
    (320, 10240, 2, None),
    // The trunk-sharded shared expert (rank 0 keeps 384 of the 640 rows / columns, rank 1 256) and
    // the shared selector (one row): the per-lane GEMM sites of the served wave.
    (384, 2560, 2, None),
    (256, 2560, 2, None),
    (2560, 384, 2, None),
    (2560, 256, 2, None),
    (1, 2560, 2, None),
];
const ROWS: &[usize] = &[
    65, 77, 96, 149, 256, 384, 511, 596, 1043, 1191, 1192, 1536, 2048,
];
/// A grid over the served prefill rows (SP halves of singletons and waves included) that sets the
/// pick table's per-shape row ranges.
const BENCH_ROWS: &[usize] = &[
    65, 75, 96, 128, 149, 192, 224, 256, 300, 340, 384, 448, 511, 596, 700, 800, 900, 1000, 1192,
];
const MAX_ROWS: usize = 2048;

fn shape_w(gpu: &mut Gpu, m: usize, k: usize) -> GpuTensor {
    let mut rng = Lcg(0x71E0_0000 ^ ((m as u64) << 20) ^ k as u64);
    let bits: Vec<u16> = (0..m * k).map(|_| bf16_bits(rng.normal() * 0.05)).collect();
    upload_bf16(gpu, &bits)
}

/// X random normal (terms 2) or BF16-exact (terms 1).
fn shape_x(rng: &mut Lcg, rows: usize, k: usize, terms: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|_| {
            let v = rng.normal();
            if terms == 1 {
                bf16r(v)
            } else {
                v
            }
        })
        .collect()
}

#[test]
fn lwmma_tile_oracle_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    let mut total_diff = 0usize;
    let mut checks = 0usize;
    for &(m, k, terms, act) in SHAPES {
        let w = shape_w(&mut gpu, m, k);
        let (split_k, _) = lane_wmma_prefill_plan(m, k, MAX_ROWS, LANE_WMMA_PREFILL_SPLIT_ROWS);
        let scratch = LaneWmmaScratch::new(
            &mut gpu,
            if split_k > 1 {
                split_k * MAX_ROWS * m
            } else {
                1
            },
            if split_k > 1 {
                m.div_ceil(16) * MAX_ROWS.div_ceil(64)
            } else {
                1
            },
        )
        .expect("scratch");
        let mut rng = Lcg(0x71E0_1000 ^ ((m as u64) << 20) ^ k as u64);
        for &rows in ROWS {
            let x = shape_x(&mut rng, rows, k, terms);
            let x_t = upload_f32(&mut gpu, &x);
            let y_ref = gpu.zeros(&[rows * m], DType::F32).expect("y");
            lane_wmma_gemm(
                &mut gpu,
                &LaneWmmaWeight::Bf16(&w),
                &x_t,
                &y_ref,
                m,
                k,
                rows,
                terms,
                act,
                LaneWmmaCfg { nn: 1, split_k },
                &scratch,
            )
            .expect("lane_wmma_gemm");
            gpu.hip.device_synchronize().expect("sync");
            let want = download_f32(&gpu, &y_ref, rows * m);

            // The launcher's own pick (where the table picks a tile), then every exported tile.
            let mut cfgs: Vec<(String, Option<LaneWmmaTile>)> = Vec::new();
            if lane_wmma_tile_pick(m, k, rows, terms).is_some() {
                cfgs.push(("pick".into(), None));
            }
            for t in lane_wmma_tile_set(terms) {
                cfgs.push((t.name(), Some(*t)));
            }
            let mut line =
                format!("[tile-oracle] m={m} k={k} t={terms} rows={rows} split_k={split_k}");
            for (name, t) in &cfgs {
                // Poison y so an unwritten word cannot match by accident.
                let y_t = upload_f32(&mut gpu, &vec![f32::from_bits(0x7fc0_dead); rows * m]);
                match t {
                    None => lane_wmma_tile_gemm(
                        &mut gpu, &w, &x_t, &y_t, m, k, rows, terms, act, split_k,
                    ),
                    Some(t) => lane_wmma_tile_gemm_cfg(
                        &mut gpu, &w, &x_t, &y_t, m, k, rows, terms, act, split_k, *t,
                    ),
                }
                .expect("tile gemm");
                gpu.hip.device_synchronize().expect("sync");
                let got = download_f32(&gpu, &y_t, rows * m);
                let diff = got
                    .iter()
                    .zip(&want)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                total_diff += diff;
                checks += 1;
                line.push_str(&format!(" {name}:{diff}"));
            }
            eprintln!("{line}");
        }
    }
    eprintln!("[tile-oracle] total differing words {total_diff} over {checks} checks");
    assert_eq!(
        total_diff, 0,
        "tile kernel is not byte-identical to lane_wmma_gemm"
    );
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median us per call: BATCH back-to-back calls timed by one event pair, REPS batches.
fn time_us(gpu: &mut Gpu, mut run: impl FnMut(&mut Gpu)) -> f64 {
    const BATCH: usize = 6;
    const REPS: usize = 11;
    for _ in 0..4 {
        run(gpu);
    }
    gpu.hip.device_synchronize().expect("sync");
    let mut samples = Vec::new();
    for _ in 0..REPS {
        let s = gpu.hip.event_create().unwrap();
        let e = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&s, None).unwrap();
        for _ in 0..BATCH {
            run(gpu);
        }
        gpu.hip.event_record(&e, None).unwrap();
        gpu.hip.event_synchronize(&e).unwrap();
        samples.push(gpu.hip.event_elapsed_ms(&s, &e).unwrap() as f64 * 1000.0 / BATCH as f64);
        let _ = gpu.hip.event_destroy(s);
        let _ = gpu.hip.event_destroy(e);
    }
    median(&mut samples)
}

#[test]
fn lwmma_tile_bench_hw() {
    let Some(mut gpu) = gpu_1201() else { return };
    for &(m, k, terms, act) in SHAPES {
        let w = shape_w(&mut gpu, m, k);
        let (split_k, _) = lane_wmma_prefill_plan(m, k, MAX_ROWS, LANE_WMMA_PREFILL_SPLIT_ROWS);
        let mut rng = Lcg(0x71E0_2000 ^ ((m as u64) << 20) ^ k as u64);
        for &rows in BENCH_ROWS {
            let x_t = {
                let x = shape_x(&mut rng, rows, k, terms);
                upload_f32(&mut gpu, &x)
            };
            let y_t = gpu.zeros(&[rows * m], DType::F32).expect("y");
            let wref = LaneWmmaWeight::Bf16(&w);
            let us = time_us(&mut gpu, |g| {
                g.lane_wmma_gemm_prefill(&wref, &x_t, &y_t, m, k, rows, terms, act)
                    .expect("incumbent");
            });
            println!("[tile-bench] m={m} k={k} t={terms} rows={rows} arm=incumbent us={us:.1}");
            for t in lane_wmma_tile_set(terms) {
                let us = time_us(&mut gpu, |g| {
                    lane_wmma_tile_gemm_cfg(g, &w, &x_t, &y_t, m, k, rows, terms, act, split_k, *t)
                        .expect("tile");
                });
                println!(
                    "[tile-bench] m={m} k={k} t={terms} rows={rows} arm=tile-{} us={us:.1}",
                    t.name()
                );
            }
            if let Some(pick) = lane_wmma_tile_pick(m, k, rows, terms) {
                let us = time_us(&mut gpu, |g| {
                    g.lane_wmma_tile_gemm_prefill(&wref, &x_t, &y_t, m, k, rows, terms, act)
                        .expect("tile prefill");
                });
                println!(
                    "[tile-bench] m={m} k={k} t={terms} rows={rows} arm=tile-pick({}) us={us:.1}",
                    pick.name()
                );
            }
        }
    }
}
