// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU reference of the R128 A4 producer records (`moe_rotate_i4_pm` / `mq_i4_a4`): the FWHT128 with signs1 / signs2
//! and the C2 `{5,7}` quantizer (the CPU twin `fake_quantize_block_i4_128`), shared by the arithmetic proofs
//! (`moe_rotate_i4_pm.rs`) and the interpreter execution of the emitted kernels (`moe_rotate_i4_pm_emu.rs`).
#![allow(dead_code)]
use peacemaker_isa::kernels::mq_i4_a4;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `gen_fwht_signs` of the engine / quantizer (LCG, bit 16), as `mq_i4_a4_oracle.py`.
pub fn signs(seed: u32) -> [f32; 128] {
    let mut state = seed;
    let mut out = [0f32; 128];
    for s in &mut out {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        *s = if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 };
    }
    out
}

pub fn bf16_rne(x: f32) -> u16 {
    let u = x.to_bits();
    ((u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1)) >> 16) as u16
}

/// FWHT128 as the kernel: widen, `* signs1`, butterflies of strides 1..64 (`lo' = lo + hi`, `hi' = lo - hi`; each
/// lane-stage `fma(+-1, v, partner)` is exactly one rounded add / subtract), `(v * 0.0883883476f) * signs2`.
pub fn rotate(h: &[u16], s1: &[f32; 128], s2: &[f32; 128]) -> [f32; 128] {
    let mut v = [0f32; 128];
    for i in 0..128 {
        v[i] = f32::from_bits(u32::from(h[i]) << 16) * s1[i];
    }
    let mut stride = 1;
    while stride < 128 {
        for base in (0..128).step_by(2 * stride) {
            for j in base..base + stride {
                let (lo, hi) = (v[j], v[j + stride]);
                v[j] = lo + hi;
                v[j + stride] = lo - hi;
            }
        }
        stride *= 2;
    }
    let scale = f32::from_bits(mq_i4_a4::ROTATION_SCALE_BITS);
    for i in 0..128 {
        v[i] = (v[i] * scale) * s2[i];
    }
    v
}

/// The 72-byte `block_i4_128` record of one rotated group: `fake_quantize_block_i4_128(candidates = 2)`
/// (`hipfire-quantize/src/calibration.rs`), four values per simulated lane and the XOR 16,8,4,2,1 trees.
pub fn quantize(x: &[f32; 128]) -> [u8; 72] {
    let mut lane_amax = [0f32; 32];
    for lane in 0..32 {
        let b = lane * 4;
        lane_amax[lane] = x[b]
            .abs()
            .max(x[b + 1].abs())
            .max(x[b + 2].abs())
            .max(x[b + 3].abs());
    }
    for off in [16usize, 8, 4, 2, 1] {
        let prior = lane_amax;
        for lane in 0..32 {
            lane_amax[lane] = lane_amax[lane].max(prior[lane ^ off]);
        }
    }
    let amax = lane_amax[0];
    let mut best_d = 1.0f32;
    if amax != 0.0 {
        let base = (amax / 7.0f32) * 0.5f32;
        let mut best_mse = 1.0e30f32;
        for mult in [f32::from_bits(mq_i4_a4::CAND0_MULT_BITS), 2.0f32] {
            let d = base * mult;
            let mut lane_mse = [0f32; 32];
            for lane in 0..32 {
                let mut mse = 0f32;
                for e in 0..4 {
                    let v = x[lane * 4 + e];
                    let q = (v / d).round_ties_even().clamp(-8.0, 7.0);
                    let err = (-q).mul_add(d, v);
                    mse = err.mul_add(err, mse);
                }
                lane_mse[lane] = mse;
            }
            for off in [16usize, 8, 4, 2, 1] {
                let prior = lane_mse;
                for lane in 0..32 {
                    lane_mse[lane] += prior[lane ^ off];
                }
            }
            if lane_mse[0] < best_mse {
                best_mse = lane_mse[0];
                best_d = d;
            }
        }
    }
    let mut q = [0i32; 128];
    for i in 0..128 {
        q[i] = if amax == 0.0 {
            0
        } else {
            (x[i] / best_d).round_ties_even().clamp(-8.0, 7.0) as i32
        };
    }
    let mut rec = [0u8; 72];
    rec[0..4].copy_from_slice(&best_d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for i in 0..64 {
        rec[8 + i] = ((q[2 * i] & 15) | ((q[2 * i + 1] & 15) << 4)) as u8;
    }
    rec
}

pub fn record(h_group: &[u16], s1: &[f32; 128], s2: &[f32; 128]) -> [u8; 72] {
    quantize(&rotate(h_group, s1, s2))
}
