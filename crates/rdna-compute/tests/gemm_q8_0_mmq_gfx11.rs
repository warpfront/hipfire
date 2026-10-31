// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx11 (gfx1100 / gfx1151) W8A8 Q8_0 GEMM (`gemm_q8_0_mmq_4w_gfx1151`, the i8-MMQ 64x64 kernel Raon's
//! tiered Q8_0 prefill route runs on RDNA3): error against an F64 reference at tileable and tail
//! shapes (M and N tails, K % 128 == 0), plus a canary check that nothing past `n * m` is written.
//!
//! `#[ignore]`d: needs a gfx1100 / gfx1151 GPU. Run through fabric-run.sh:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_mmq_gfx11 -- --ignored

use rdna_compute::Gpu;

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

/// Returns (Q8_0 bytes, dequantized f64 weights [m * k]).
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f64>) {
    let nb = k / 32;
    let mut bytes = Vec::new();
    let mut deq = Vec::with_capacity(m * k);
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        for _ in 0..32 {
            let q = (rng.gauss() * 40.0).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
            deq.push(f64::from(sc.to_f32()) * f64::from(q));
        }
    }
    (bytes, deq)
}

#[test]
#[ignore = "needs a gfx1100 / gfx1151 GPU"]
fn q8_0_mmq_4w_error_vs_f64_and_no_out_of_range_writes() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        gpu.arch == "gfx1100" || gpu.arch == "gfx1151",
        "needs gfx1100 / gfx1151, got {}",
        gpu.arch
    );
    const CANARY: f32 = 1234.5;
    const PAD: usize = 8192;
    // (m, k, n): tileable, M tail (40, 200, 1088 = 17 x 64), N tail (1 .. 255), K = 512 / 1280 / 2560.
    for (i, &(m, k, n)) in [
        (128usize, 640usize, 64usize),
        (5120, 2560, 512),
        (576, 2560, 100),
        (40, 2560, 70),
        (1088, 1280, 65),
        (2560, 512, 255),
        (200, 1280, 129),
    ]
    .iter()
    .enumerate()
    {
        let mut rng = Rng(0x1100_a8a8 + i as u64);
        let (wbytes, wdeq) = make_weights(m, k, &mut rng);
        let x: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
        let y = gpu
            .upload_f32(&vec![CANARY; n * m + PAD], &[n * m + PAD])
            .expect("y");
        gpu.gemm_q8_0_mmq_4w_gfx1151(&w, &xt, &y, m, k, n)
            .expect("mmq 4w");
        let got = gpu.download_f32(&y).expect("download");
        assert!(
            got[n * m..].iter().all(|&v| v == CANARY),
            "({m},{k},{n}): wrote past n*m"
        );
        let (mut num, mut den) = (0f64, 0f64);
        for t in 0..n {
            for r in 0..m {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += wdeq[r * k + kk] * f64::from(x[t * k + kk]);
                }
                let g = f64::from(got[t * m + r]);
                assert!(g.is_finite(), "({m},{k},{n}): non-finite at ({t},{r})");
                num += (g - acc).powi(2);
                den += acc * acc;
            }
        }
        let rel = (num / den.max(1e-30)).sqrt();
        // int8 activation quantization (d = amax/127 per 32-K block): ~5e-3 rms on Gaussian x.
        assert!(rel < 1.5e-2, "({m},{k},{n}): rms rel error {rel}");
        eprintln!("gfx11 mmq 4w ({m},{k},{n}): rms rel {rel:.3e}");
    }
}
