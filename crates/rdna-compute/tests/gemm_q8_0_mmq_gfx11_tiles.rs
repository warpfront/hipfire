// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11 tiled W8A8 Q8_0 GEMM (`Q8MmqTile::{T64x64,T64x128,T128x128}`): every tile, the batched
//! entry point and the prepared-activation split must equal the landed 64x64 kernel
//! (`gemm_q8_0_mmq_4w_gfx1151`) BIT FOR BIT at tileable and tail shapes.
//!
//! The GPU tests are `#[ignore]`d: they need a gfx1100 / gfx1151 GPU. Run through fabric-run.sh:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_mmq_gfx11_tiles -- --ignored

use rdna_compute::{q8_mmq_pick_tile, Gpu, Q8MmqTile};

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
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// Q8_0 bytes [m, k/32*34]: per 32-K block an f16 scale then 32 int8.
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / 32;
    let mut bytes = Vec::with_capacity(m * nb * 34);
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        for _ in 0..32 {
            let q = (rng.gauss() * 40.0).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
        }
    }
    bytes
}

fn make_x(n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n * k).map(|_| bf16_round(rng.gauss())).collect()
}

const NAN_BITS: u32 = 0x7fc0_dead;

fn nan_buf(len: usize) -> Vec<f32> {
    vec![f32::from_bits(NAN_BITS); len]
}

fn gfx11(gpu: &Gpu) -> bool {
    if gpu.arch == "gfx1100" || gpu.arch == "gfx1151" {
        true
    } else {
        eprintln!("skip: needs gfx1100 / gfx1151, got {}", gpu.arch);
        false
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} differs: got {:#010x} ({g}), want {:#010x} ({w})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn tiles_are_bit_identical_to_the_64x64_kernel() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    // (m, k, n): tileable, M tails, N tails (down to n = 1), K = 128 .. 5120.
    let shapes: [(usize, usize, usize); 9] = [
        (576, 2560, 512),
        (5120, 2560, 512),
        (2560, 5120, 300),
        (1000, 2560, 475),
        (1088, 2560, 65),
        (40, 2560, 512),
        (64, 128, 1),
        (130, 640, 129),
        (7000, 1280, 1023),
    ];
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let mut rng = Rng(0x1100_71e5 + i as u64);
        let wbytes = make_weights(m, k, &mut rng);
        let x = make_x(n, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        let xt = gpu.upload_f32(&x, &[n * k]).expect("x");

        let y_ref = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y ref");
        gpu.gemm_q8_0_mmq_4w_gfx1151(&w, &xt, &y_ref, m, k, n)
            .expect("incumbent 64x64");
        let want = gpu.download_f32(&y_ref).expect("download ref");
        assert!(
            want.iter().all(|v| v.is_finite()),
            "({m},{k},{n}): incumbent left a non-finite element"
        );

        for tile in Q8MmqTile::ALL {
            let label = tile.label();
            let acts = gpu.q8_mmq_quantize_x(&xt, n, k).expect("quantize x");
            assert_eq!((acts.rows(), acts.k()), (n, k));
            let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y");
            gpu.gemm_q8_0_mmq_gfx11_gemm_tile(&w, &acts, &y, m, tile)
                .unwrap_or_else(|e| panic!("tile {label} ({m},{k},{n}): {e:?}"));
            let got = gpu.download_f32(&y).expect("download tile");
            assert_bits_eq(&got, &want, &format!("tile {label} ({m},{k},{n})"));
        }

        let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y");
        gpu.gemm_q8_0_mmq_gfx11_batched(&w, &xt, &y, m, k, n)
            .unwrap_or_else(|e| panic!("batched ({m},{k},{n}): {e:?}"));
        let got = gpu.download_f32(&y).expect("download batched");
        assert_bits_eq(&got, &want, &format!("batched ({m},{k},{n})"));
        eprintln!("gfx11 mmq tiles ({m},{k},{n}): all tiles + batched bit-identical");
    }
}

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn prepared_activations_detect_overwrite() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let (m1, m2, k, n) = (576usize, 1000usize, 1280usize, 200usize);
    let mut rng = Rng(0x1100_0ac7);
    let w1b = make_weights(m1, k, &mut rng);
    let w2b = make_weights(m2, k, &mut rng);
    let x = make_x(n, k, &mut rng);
    let x_other = make_x(n, k, &mut rng);
    let w1 = gpu.upload_raw(&w1b, &[w1b.len()]).expect("w1");
    let w2 = gpu.upload_raw(&w2b, &[w2b.len()]).expect("w2");
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let xo = gpu.upload_f32(&x_other, &[n * k]).expect("x other");

    // Batched references (each re-quantizes x itself).
    let mut want = Vec::new();
    for (w, m) in [(&w1, m1), (&w2, m2)] {
        let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y ref");
        gpu.gemm_q8_0_mmq_gfx11_batched(w, &xt, &y, m, k, n)
            .expect("batched");
        want.push(gpu.download_f32(&y).expect("download ref"));
    }

    // One quantization serves both weight GEMMs.
    let acts = gpu.q8_mmq_quantize_x(&xt, n, k).expect("quantize x");
    assert!(
        gpu.q8_mmq_acts_current(&acts),
        "fresh activations must be current"
    );
    for (idx, (w, m)) in [(&w1, m1), (&w2, m2)].into_iter().enumerate() {
        let y = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y");
        gpu.gemm_q8_0_mmq_gfx11_gemm(w, &acts, &y, m)
            .expect("gemm on shared acts");
        let got = gpu.download_f32(&y).expect("download");
        assert_bits_eq(
            &got,
            &want[idx],
            &format!("shared acts, weight {idx} (m={m})"),
        );
        assert!(
            gpu.q8_mmq_acts_current(&acts),
            "a GEMM must not invalidate its activations"
        );
    }

    // A later quantization of a different tensor overwrites the plane.
    let old = acts;
    let _new = gpu.q8_mmq_quantize_x(&xo, n, k).expect("quantize other x");
    assert!(
        !gpu.q8_mmq_acts_current(&old),
        "old activations must be detected as overwritten"
    );
    let y = gpu.upload_f32(&nan_buf(n * m1), &[n * m1]).expect("y");
    assert!(
        gpu.gemm_q8_0_mmq_gfx11_gemm(&w1, &old, &y, m1).is_err(),
        "gemm with stale activations must return Err"
    );
}

#[test]
fn pick_tile_is_total() {
    let ms = [1usize, 40, 64, 65, 576, 1000, 1088, 2560, 5120, 7000, 17408];
    let ns = [1usize, 7, 64, 65, 300, 512, 1023, 2048, 4096, 8192];
    for &m in &ms {
        for &n in &ns {
            let tile = q8_mmq_pick_tile(m, n);
            assert!(Q8MmqTile::ALL.contains(&tile), "({m},{n}): tile not in ALL");
            let (bm, bn) = tile.dims();
            assert!(bm > 0 && bn > 0, "({m},{n}): zero tile dims");
            let (gx, gy) = (m.div_ceil(bm), n.div_ceil(bn));
            assert!(gx > 0 && gy > 0, "({m},{n}) {}: empty grid", tile.label());
            assert!(
                gx * bm >= m && gy * bn >= n,
                "({m},{n}) {}: grid {gx}x{gy} does not cover the output",
                tile.label()
            );
        }
    }
}
