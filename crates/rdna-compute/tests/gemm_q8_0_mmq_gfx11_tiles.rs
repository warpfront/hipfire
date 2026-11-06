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

/// The `_bf16` tile entries must equal the plain tile followed by the production round pass
/// (`Gpu::bf16_round_trip_f32_strided` over the `[n, m]` plane), bit for bit (class E), on every
/// tile, the selector route, normal values, values overflowing to Inf and deep denormals, M and N
/// tails. NaN-filled outputs prove every element is stored.
fn check_bf16_site(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    xs: f32,
    seed: u64,
    tiles: &[Q8MmqTile],
) {
    let mut rng = Rng(seed);
    let wbytes = make_weights(m, k, &mut rng);
    // Every third token row is scaled by `xs`: a huge `xs` overflows that row's f16 Q8_1 scale to
    // Inf, so tiles mix finite and non-finite outputs (the f16 scales bound the finite range).
    let x: Vec<f32> = make_x(n, k, &mut rng)
        .chunks(k)
        .enumerate()
        .flat_map(|(t, row)| {
            let f = if t % 3 == 0 { xs } else { 1.0 };
            row.iter().map(move |v| v * f)
        })
        .collect();
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let acts = gpu.q8_mmq_quantize_x(&xt, n, k).expect("quantize x");
    for &tile in tiles {
        let label = tile.label();
        let y0 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y0");
        gpu.gemm_q8_0_mmq_gfx11_gemm_tile(&w, &acts, &y0, m, tile)
            .unwrap_or_else(|e| panic!("plain {label} ({m},{k},{n}): {e:?}"));
        let nonfinite = gpu
            .download_f32(&y0)
            .expect("download plain")
            .iter()
            .filter(|v| !v.is_finite())
            .count();
        gpu.bf16_round_trip_f32_strided(&y0, n, 0, m, m)
            .expect("separate round pass");
        let want = gpu.download_f32(&y0).expect("download unfused");
        let y1 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y1");
        gpu.gemm_q8_0_mmq_gfx11_gemm_tile_bf16(&w, &acts, &y1, m, tile)
            .unwrap_or_else(|e| panic!("bf16 {label} ({m},{k},{n}): {e:?}"));
        let got = gpu.download_f32(&y1).expect("download bf16");
        assert_bits_eq(
            &got,
            &want,
            &format!("bf16 tile {label} ({m},{k},{n}) xs={xs:e}"),
        );
        eprintln!(
            "gfx11 mmq bf16 {label} ({m},{k},{n}) xs={xs:e}: bit-identical to plain + round ({nonfinite} non-finite)"
        );
    }
    // selector route (what the dispatcher calls)
    let y0 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y0");
    gpu.gemm_q8_0_mmq_gfx11_gemm(&w, &acts, &y0, m)
        .expect("plain selector");
    gpu.bf16_round_trip_f32_strided(&y0, n, 0, m, m)
        .expect("round");
    let want = gpu.download_f32(&y0).expect("download");
    let y1 = gpu.upload_f32(&nan_buf(n * m), &[n * m]).expect("y1");
    gpu.gemm_q8_0_mmq_gfx11_gemm_bf16(&w, &acts, &y1, m)
        .expect("bf16 selector");
    let got = gpu.download_f32(&y1).expect("download");
    assert_bits_eq(
        &got,
        &want,
        &format!("bf16 selector ({m},{k},{n}) xs={xs:e}"),
    );
}

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn bf16_epilogue_tails_overflow_and_denormals() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    let shapes: [(usize, usize, usize); 8] = [
        (576, 2560, 512),
        (5120, 2560, 512),
        (2560, 5120, 300),
        (1000, 2560, 475),
        (1088, 2560, 65),
        (40, 2560, 512),
        (64, 128, 1),
        (130, 640, 129),
    ];
    for (xs_i, xs) in [1.0f32, 3.0e3, 1.0e32, 1.0e37, 1.0e-37]
        .into_iter()
        .enumerate()
    {
        for (i, &(m, k, n)) in shapes.iter().enumerate() {
            check_bf16_site(
                &mut gpu,
                m,
                k,
                n,
                xs,
                0x1100_bf16 + (i * 8 + xs_i) as u64,
                &Q8MmqTile::ALL,
            );
        }
    }
}

/// Every Raon Q8_0 prefill site (K % 128 == 0, the tiled route's admission) x rows {64, 512, 1000,
/// 4096}, all three tiles and the selector: fused output == unfused GEMM + separate round.
#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn bf16_epilogue_raon_site_shapes() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gfx11(&gpu) {
        return;
    }
    // (M, K) of the Raon Q8 sites (`bench_q8_raon_shapes`); mla_q_b (7680 x 576) has K % 128 != 0 and
    // never takes the tiled route.
    let sites: [(usize, usize); 8] = [
        (5120, 2560),
        (2560, 5120),
        (10240, 512),
        (576, 2560),
        (1088, 2560),
        (2560, 1280),
        (12032, 2560),
        (2560, 12032),
    ];
    for (i, &(m, k)) in sites.iter().enumerate() {
        for (j, &n) in [64usize, 512, 1000, 4096].iter().enumerate() {
            check_bf16_site(
                &mut gpu,
                m,
                k,
                n,
                1.0,
                0x1100_5173 + (i * 4 + j) as u64,
                &Q8MmqTile::ALL,
            );
        }
    }
}
