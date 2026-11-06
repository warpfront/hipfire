// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gate for the K = 640 four-rows-per-wave MQ4G128V2 routed down
//! (`gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4`, admitted
//! on gfx1201 next to gfx1151) against the one-row singleton kernel it replaces:
//! the real production launcher run twice on the same randomized inputs with
//! `FeatureFlags::qwen4_moe_down_r4` off (oracle) and on (candidate), raw f32
//! bits of the whole `[batch, k_top, M]` expanded output compared. The compiler's
//! FMA contraction of the originals is not reproducible by a CPU model, so the
//! original kernel is the oracle.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test moe_down_r4_hw -- --ignored --nocapture

use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Arc;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f32_signed(&mut self) -> f32 {
        let a = self.next_u32() & 0x7F_FFFF;
        let sign = self.next_u32() & 1;
        let v = f32::from_bits(0x3F80_0000 | a) - 1.0;
        if sign == 1 {
            -v
        } else {
            v
        }
    }
    /// Normal f16 bit pattern with biased exponent in `lo..=hi`.
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
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let t = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod i32");
    t
}
fn upload_u64(gpu: &mut Gpu, data: &[u64]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    let t = gpu
        .alloc_tensor(&[data.len() * 8], DType::Raw)
        .expect("alloc u64");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod u64");
    t
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

// Flash-Next routed down: M = 2560 output rows, K = 640 = 5 groups of 128, top-10.
const M: usize = 2560;
const K: usize = 640;
const GROUPS: usize = K / 128;
const ROW_BYTES: usize = GROUPS * 68;
const K_TOP: usize = 10;
const N_EXP: usize = 24;

/// One MQ4G128V2 (qt=53) expert: M rows x 5 groups of 68 B
/// `[scale:f16][zero:f16][64 payload bytes]`; alternating symmetric / affine headers.
fn synth_expert(seed: u64, symmetric: bool) -> Vec<u8> {
    let mut rng = Lcg(seed);
    let mut out = vec![0u8; M * ROW_BYTES];
    for row in 0..M {
        for g in 0..GROUPS {
            let off = row * ROW_BYTES + g * 68;
            let sc = rng.f16_normal(9, 13, false);
            let zp = if symmetric {
                // zero = -8 * scale: scale exponent + 3, same mantissa, negative sign
                (1 << 15) | ((((sc >> 10) & 0x1F) + 3) << 10) | (sc & 0x3FF)
            } else {
                rng.f16_normal(8, 14, true)
            };
            out[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
            for w in 0..16 {
                let pk = rng.next_u32() ^ (rng.next_u32() << 16);
                out[off + 4 + 4 * w..off + 8 + 4 * w].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn mq4g128v2_down_k640_r4_matches_one_row_kernel_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let mut keep: Vec<GpuTensor> = Vec::new();
    let mut ptrs: Vec<u64> = Vec::new();
    for e in 0..N_EXP {
        let blob = if e == N_EXP - 1 {
            // The EP dummy: a zero expert (all-zero rows, +0.0 outputs).
            vec![0u8; M * ROW_BYTES]
        } else {
            synth_expert(0x4D4F_4500 + e as u64, e % 2 == 0)
        };
        let t = upload_u8(&mut gpu, &blob);
        ptrs.push(t.buf.as_ptr() as u64);
        keep.push(t);
    }
    let expert_ptrs = upload_u64(&mut gpu, &ptrs);

    let original = {
        let mut f = (*gpu.flags).clone();
        f.qwen4_moe_down_r4 = false;
        Arc::new(f)
    };
    let candidate = {
        let mut f = (*gpu.flags).clone();
        f.qwen4_moe_down_r4 = true;
        Arc::new(f)
    };
    assert!(
        candidate.qwen4_moe_down_r4_enabled(),
        "candidate flag set must select the r4 twin on gfx1201"
    );
    assert!(!original.qwen4_moe_down_r4_enabled());

    let mut total_words = 0usize;
    let mut mismatches = 0usize;
    let mut first: Option<(usize, usize, u32, u32)> = None;
    let mut cases = 0usize;
    for trial in 0..24u64 {
        for batch in 1..=8usize {
            let mut rng = Lcg(0xD04E_0000 ^ trial.wrapping_mul(0x9E37_79B9) ^ (batch as u64) << 40);
            let mag = [0.25f32, 0.5, 1.0, 2.0, 4.0, 8.0, 3.0, 6.0][(trial % 8) as usize];
            let mut hidden: Vec<f32> = (0..batch * K_TOP * K)
                .map(|_| rng.f32_signed() * mag)
                .collect();
            if trial % 5 == 4 {
                for i in (0..hidden.len()).step_by(7) {
                    hidden[i] = 0.0;
                }
                for i in (3..hidden.len()).step_by(11) {
                    hidden[i] = rng.f32_signed() * 40.0;
                }
            }
            // Routing: distinct experts inside a token, repeats across tokens, a share of
            // slots on the zero dummy (non-owned under EP), invalid ids on some trials.
            let mut topk: Vec<i32> = Vec::new();
            for _ in 0..batch {
                let mut row: Vec<i32> = Vec::new();
                while row.len() < K_TOP {
                    let e = if rng.next_u32() % 2 == 0 {
                        (N_EXP - 1) as i32
                    } else {
                        (rng.next_u32() as usize % (N_EXP - 1)) as i32
                    };
                    if !row.contains(&e) || e == (N_EXP - 1) as i32 {
                        row.push(e);
                    }
                }
                topk.extend(row);
            }
            if trial % 6 == 5 {
                let n = topk.len();
                topk[1 % n] = -1;
                topk[(n / 2) % n] = N_EXP as i32 + 3;
            }
            let hidden_t = upload_f32(&mut gpu, &hidden);
            let topk_t = upload_i32(&mut gpu, &topk);
            let canary = f32::from_bits(0x7FC0_DEAD);
            let out_len = batch * K_TOP * M;
            let run = |gpu: &mut Gpu, flags: &Arc<rdna_compute::FeatureFlags>| -> Vec<u32> {
                gpu.flags = flags.clone();
                let out_t = upload_f32(gpu, &vec![canary; out_len]);
                gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
                    &expert_ptrs,
                    &topk_t,
                    &hidden_t,
                    &out_t,
                    M,
                    K,
                    batch,
                    N_EXP,
                    K_TOP,
                )
                .expect("expanded down");
                gpu.hip.device_synchronize().expect("sync");
                download_u32(gpu, &out_t, out_len)
            };
            let want = run(&mut gpu, &original);
            let got = run(&mut gpu, &candidate);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if g != w {
                    mismatches += 1;
                    first.get_or_insert((cases, i, *g, *w));
                }
            }
            total_words += out_len;
            cases += 1;
        }
    }
    eprintln!(
        "moe_down_r4: {cases} cases (24 trials x batch 1..8), {total_words} output words, {mismatches} mismatches, first {first:?}"
    );
    assert_eq!(
        mismatches, 0,
        "the r4 twin is not bit-identical to the one-row down kernel"
    );
}
