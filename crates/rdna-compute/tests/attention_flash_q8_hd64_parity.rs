// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! head_dim 64 Q8_0 flash decode (`attention_flash_q8_0_tile_hd64` + the
//! shared reduce) against `attention_q8_0_kv` and an f64 CPU oracle on the
//! SmolLM2-360M (15/5) and Llama-3.2-1B (32/8) head shapes. Contexts cover a
//! single key, odd tails inside the 8-position wave step, tile boundaries and
//! long multi-tile spans.
//!
//! Negative control: the flash output for `ctx` must miss the oracle of
//! `ctx + 1` (one extra key), so the tolerance resolves a single tail row
//! even at 8000 keys.
//!
//! `#[ignore]`d: needs a wave32 GPU and a working HIP toolchain.
//!
//!   cargo test -p rdna-compute --release --test attention_flash_q8_hd64_parity -- --ignored

use half::f16;
use rdna_compute::{DType, Gpu};

const HD: usize = 64;
const MAX_SEQ: usize = 8192;
/// Both kernels accumulate in f32 with different association (measured
/// <= 1e-6 on gfx1151); layout or masking mistakes are >= 1e-3 here.
const TOL: f32 = 1e-5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

/// `[pos][kv_head][2 blocks]` of `[f16 scale][32 x i8]` plus its dequantization.
fn q8_cache(rng: &mut Rng, nkv: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(MAX_SEQ * nkv * 2 * 34);
    let mut deq = Vec::with_capacity(MAX_SEQ * nkv * HD);
    for _ in 0..MAX_SEQ * nkv * 2 {
        let scale = f16::from_f32(0.01 + 0.02 * (rng.next() % 1024) as f32 / 1024.0);
        bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
        for _ in 0..32 {
            let c = (rng.next() % 255) as i32 - 127;
            bytes.push(c as i8 as u8);
            deq.push(scale.to_f32() * c as f32);
        }
    }
    (bytes, deq)
}

fn oracle(q: &[f32], k: &[f32], v: &[f32], nh: usize, nkv: usize, ctx: usize) -> Vec<f32> {
    let group = nh / nkv;
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f32; nh * HD];
    for h in 0..nh {
        let kvh = h / group;
        let row = |c: &[f32], t: usize| -> Vec<f64> {
            c[(t * nkv + kvh) * HD..][..HD]
                .iter()
                .map(|&x| x as f64)
                .collect()
        };
        let s: Vec<f64> = (0..ctx)
            .map(|t| {
                let kr = row(k, t);
                (0..HD).map(|d| q[h * HD + d] as f64 * kr[d]).sum::<f64>() * scale
            })
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let z: f64 = w.iter().sum();
        for t in 0..ctx {
            let vr = row(v, t);
            for d in 0..HD {
                out[h * HD + d] += (w[t] / z * vr[d]) as f32;
            }
        }
    }
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs a wave32 GPU"]
fn hd64_flash_decode_matches_q8_kv_and_oracle() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut rng = Rng(0x6864_3634);
    for (nh, nkv) in [(15usize, 5usize), (32, 8)] {
        let (kb, kd) = q8_cache(&mut rng, nkv);
        let (vb, vd) = q8_cache(&mut rng, nkv);
        let k = gpu.upload_raw(&kb, &[kb.len()]).expect("k");
        let v = gpu.upload_raw(&vb, &[vb.len()]).expect("v");
        let qh: Vec<f32> = (0..nh * HD).map(|_| rng.unit()).collect();
        let q = gpu.upload_f32(&qh, &[nh * HD]).expect("q");
        let tile = rdna_compute::attention::q8_flash_tile_size(&gpu.arch, nh, nkv, HD, MAX_SEQ);
        let partials = gpu
            .zeros(&[nh * MAX_SEQ.div_ceil(tile) * (2 + HD)], DType::F32)
            .expect("partials");
        for ctx in [1, 2, 7, 9, 127, 128, 129, 257, 712, 3445, 8000] {
            let pos = ((ctx - 1) as i32).to_le_bytes();
            let pos = gpu.upload_raw(&pos, &[1]).expect("pos");
            let flash_out = gpu.zeros(&[nh * HD], DType::F32).expect("out");
            let kv_out = gpu.zeros(&[nh * HD], DType::F32).expect("out");
            gpu.attention_flash_q8_0(
                &q, &k, &v, &flash_out, &pos.buf, ctx, nh, nkv, HD, MAX_SEQ, &partials,
            )
            .expect("flash");
            gpu.attention_q8_0_kv(&q, &k, &v, &kv_out, &pos.buf, ctx, nh, nkv, HD, MAX_SEQ)
                .expect("q8 kv");
            let flash = gpu.download_f32(&flash_out).expect("dl");
            let kv = gpu.download_f32(&kv_out).expect("dl");
            let want = oracle(&qh, &kd, &vd, nh, nkv, ctx);
            let (d_kv, d_or) = (max_diff(&flash, &kv), max_diff(&flash, &want));
            let d_neg = max_diff(&flash, &oracle(&qh, &kd, &vd, nh, nkv, ctx + 1));
            println!(
                "nh={nh} nkv={nkv} tile={tile} ctx={ctx}: |flash-q8kv|={d_kv:.2e} \
                 |flash-oracle|={d_or:.2e} |q8kv-oracle|={:.2e} neg(ctx+1)={d_neg:.2e}",
                max_diff(&kv, &want)
            );
            assert!(d_kv < TOL && d_or < TOL, "nh={nh} ctx={ctx}: {d_kv} {d_or}");
            assert!(
                d_neg > 10.0 * TOL,
                "nh={nh} ctx={ctx}: negative control {d_neg}"
            );
        }
    }
}
