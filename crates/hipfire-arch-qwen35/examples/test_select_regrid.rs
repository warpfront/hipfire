// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Railgun E0 / L6d parity gate (gfx1201): the re-gridded
//! `topk_values_batched_f32` (+ tie fixup) and `argmax_f32_batched` against
//! the shipping one-block-per-row kernels, byte-for-byte, on: distinct
//! Gaussian logits at the real DFlash shape (15 x 248320, K = 16), heavy
//! ties (small-integer logits), NaN / -inf / all--inf rows, values at or
//! below the argmax floor (-1e30), mixed tie and no-tie rows in one launch,
//! short and non-multiple-of-1024 rows, and K in {1, 4, 15, 16}. Outputs are
//! poisoned before every launch. Then device-times both routes at the real
//! shapes. Non-gfx1201 exits 0 with a skip note.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / u32::MAX as f64) as f32
    }
    fn gauss(&mut self) -> f32 {
        let (u1, u2) = (self.unit().max(1e-7), self.unit());
        ((-2.0 * (u1 as f64).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2 as f64).cos()) as f32
    }
}

fn upload_f32(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|f| f.to_ne_bytes()).collect();
    gpu.hip.memcpy_htod(&t.buf, &b).expect("upload");
}

fn download(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> Vec<u8> {
    let mut b = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("download");
    b
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    gpu.hip
        .memcpy_htod(&t.buf, &vec![0xA5u8; t.byte_size()])
        .expect("poison");
}

/// Row generators by case name.
fn make_rows(rng: &mut Lcg, case: &str, rows: usize, vocab: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(rows * vocab);
    for r in 0..rows {
        for _ in 0..vocab {
            let x = match case {
                "gauss" => rng.gauss() * 3.0,
                "ties" => (rng.next() % 40) as f32,
                "few-distinct-top" => {
                    if rng.next() % 5000 == 0 {
                        100.0 + (rng.next() % 3) as f32
                    } else {
                        rng.gauss()
                    }
                }
                "nan-inf" => match rng.next() % 6 {
                    0 => f32::NAN,
                    1 => f32::NEG_INFINITY,
                    _ => rng.gauss(),
                },
                "sparse-finite" => {
                    if rng.next() % 997 == 0 {
                        rng.gauss()
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                "floor" => match rng.next() % 3 {
                    0 => -1e30,
                    1 => -2e30,
                    _ => f32::NAN,
                },
                "mixed" => {
                    if r % 2 == 0 {
                        rng.gauss() * 3.0
                    } else {
                        (rng.next() % 12) as f32
                    }
                }
                _ => unreachable!(),
            };
            v.push(x);
        }
    }
    v
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !(gpu.arch_caps.is_gfx1201() || gpu.arch_caps.is_gfx1151() || gpu.arch_caps.is_gfx1100()) {
        eprintln!("SKIP: test_select_regrid requires exact gfx1201, gfx1151 or gfx1100");
        return;
    }
    assert!(
        gpu.select_regrid_enabled(),
        "re-grid must be enabled on gfx1201 / gfx1151 / gfx1100 by default"
    );
    eprintln!("=== selector re-grid parity ({}) ===", gpu.arch);
    let mut rng = Lcg(0xD1F7_5E1E);
    let max_rows = 16usize;
    let max_vocab = 248_320usize;
    let logits = gpu
        .alloc_tensor(&[max_rows * max_vocab], DType::F32)
        .unwrap();
    let idx_a = gpu.alloc_tensor(&[max_rows * 16], DType::F32).unwrap();
    let val_a = gpu.alloc_tensor(&[max_rows * 16], DType::F32).unwrap();
    let idx_b = gpu.alloc_tensor(&[max_rows * 16], DType::F32).unwrap();
    let val_b = gpu.alloc_tensor(&[max_rows * 16], DType::F32).unwrap();
    let am_a = gpu.alloc_tensor(&[max_rows], DType::F32).unwrap();
    let am_b = gpu.alloc_tensor(&[max_rows], DType::F32).unwrap();

    let cases = [
        "gauss",
        "ties",
        "few-distinct-top",
        "nan-inf",
        "sparse-finite",
        "floor",
        "mixed",
    ];
    let shapes = [
        (15usize, 248_320usize),
        (16, 248_320),
        (3, 1000),
        (5, 300),
        (2, 17),
        (4, 4097),
    ];
    let mut checked = 0usize;
    for case in cases {
        for (rows, vocab) in shapes {
            let data = make_rows(&mut rng, case, rows, vocab);
            upload_f32(&gpu, &logits, &data);
            for k in [1usize, 4, 15, 16] {
                for t in [&idx_a, &val_a, &idx_b, &val_b] {
                    poison(&gpu, t);
                }
                gpu.topk_values_batched_f32_shipping(&logits, &idx_a, &val_a, vocab, k, rows)
                    .unwrap();
                gpu.topk_values_batched_f32(&logits, &idx_b, &val_b, vocab, k, rows)
                    .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let n = rows * k * 4;
                assert_eq!(
                    download(&gpu, &idx_a, n),
                    download(&gpu, &idx_b, n),
                    "topk idx {case} {rows}x{vocab} k={k}"
                );
                assert_eq!(
                    download(&gpu, &val_a, n),
                    download(&gpu, &val_b, n),
                    "topk val {case} {rows}x{vocab} k={k}"
                );
                checked += 1;
            }
            for t in [&am_a, &am_b] {
                poison(&gpu, t);
            }
            gpu.argmax_f32_batched_shipping(&logits, &am_a, vocab, rows)
                .unwrap();
            gpu.argmax_f32_batched(&logits, &am_b, vocab, rows).unwrap();
            gpu.hip.device_synchronize().unwrap();
            assert_eq!(
                download(&gpu, &am_a, rows * 4),
                download(&gpu, &am_b, rows * 4),
                "argmax {case} {rows}x{vocab}"
            );
            checked += 1;
        }
        eprintln!(
            "  ok {case}: top-K (K=1,4,15,16) and argmax byte-identical on {} shapes",
            shapes.len()
        );
    }
    eprintln!("  {checked} launches compared");

    // Device timing at the real shapes: draft selector 15 x 248320 K=16,
    // verify argmax 16 x 248320. Gaussian (no-tie) rows, 200 launches each.
    let data = make_rows(&mut rng, "gauss", max_rows, max_vocab);
    upload_f32(&gpu, &logits, &data);
    let reps = 200;
    let time = |gpu: &mut Gpu, f: &mut dyn FnMut(&mut Gpu)| -> f64 {
        for _ in 0..10 {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    let ship_topk = time(&mut gpu, &mut |g| {
        g.topk_values_batched_f32_shipping(&logits, &idx_a, &val_a, max_vocab, 16, 15)
            .unwrap()
    });
    let new_topk = time(&mut gpu, &mut |g| {
        g.topk_values_batched_f32(&logits, &idx_b, &val_b, max_vocab, 16, 15)
            .unwrap()
    });
    let ship_am = time(&mut gpu, &mut |g| {
        g.argmax_f32_batched_shipping(&logits, &am_a, max_vocab, 16)
            .unwrap()
    });
    let new_am = time(&mut gpu, &mut |g| {
        g.argmax_f32_batched(&logits, &am_b, max_vocab, 16).unwrap()
    });
    eprintln!("  timing (us/launch, back-to-back, {reps} reps): topk 15x248320 K=16 shipping {ship_topk:.1} -> re-grid+fixup {new_topk:.1}; argmax 16x248320 shipping {ship_am:.1} -> re-grid {new_am:.1}");
    eprintln!("PASS: selector re-grid byte-identical to the shipping kernels");
}
