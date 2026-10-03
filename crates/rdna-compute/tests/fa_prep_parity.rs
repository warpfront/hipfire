// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100/gfx1151: the A3B 16Q/2K decode fusion; gfx1100/gfx1201:
//! the Qwen3.6/3.8-27B 24Q/4K decode fusion must produce the same bytes
//! as the chain it replaces: `deinterleave_f32`, Q/K `rmsnorm_batched`, then
//! `rope_partial_interleaved_f32` (the half-split partial RoPE). The Qwen3.5
//! lowered decode uses the fusion and the hand decode
//! (`HIPFIRE_FORWARD_LOWERED=0`) uses the chain, so any difference makes the
//! two decode paths diverge. On gfx1100 it did: left to the compiler, the
//! fused RoPE second output was `fma(x1, cos, x0 * sin)` where the chain
//! computes `fma(x0, sin, x1 * cos)`.
//!
//! Q, gate and K are compared bit for bit at positions from 0 to 2^20.
//! Each shape skips arches that do not admit its certified fusion.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test fa_prep_parity -- --ignored

#![cfg(feature = "deltanet")]

use rdna_compute::{Gpu, GpuTensor};

const HD: usize = 256;
const NROT: usize = 64;
const EPS: f32 = 1e-6;
const THETA: f32 = 10_000_000.0;
const POSITIONS: [i32; 10] = [0, 1, 2, 17, 511, 4_097, 8_191, 16_385, 65_535, 1 << 20];

/// Deterministic values in [-scale, scale).
fn fill(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn write_bytes(gpu: &Gpu, t: &GpuTensor, bytes: &[u8]) {
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

fn write(gpu: &Gpu, t: &GpuTensor, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    write_bytes(gpu, t, bytes);
}

#[test]
#[ignore = "needs a GPU"]
fn qwen36_27b_fa_prep_matches_unfused_chain_bit_for_bit() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !(gpu.arch_caps.is_gfx1100() || gpu.arch_caps.is_gfx1201()) {
        eprintln!("skip: {} does not admit the 24Q/4K FA prep fusion", gpu.arch);
        return;
    }
    check_prep(&mut gpu, 24, 4);
}

#[test]
#[ignore = "needs a GPU"]
fn qwen35_16q2k_fa_prep_matches_unfused_chain_bit_for_bit() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !(gpu.arch_caps.is_gfx1100() || gpu.arch_caps.is_gfx1151()) {
        eprintln!("skip: {} is not a corrected gfx11 twin", gpu.arch);
        return;
    }
    check_prep(&mut gpu, 16, 2);
}

#[allow(non_snake_case)]
fn check_prep(mut gpu: &mut Gpu, NQ: usize, NK: usize) {
    let alloc = |gpu: &mut Gpu, n: usize| gpu.upload_f32(&vec![0.0; n], &[n]).unwrap();
    let mut seed = 0x5eed_fa27;
    let q_full = alloc(&mut gpu, NQ * 2 * HD);
    let k_raw = fill(&mut seed, NK * HD, 4.0);
    let (q_weight, k_weight) = (alloc(&mut gpu, HD), alloc(&mut gpu, HD));
    write(&gpu, &q_weight, &fill(&mut seed, HD, 2.0));
    write(&gpu, &k_weight, &fill(&mut seed, HD, 2.0));
    let pos = alloc(&mut gpu, 1);
    let (q_ref, gate_ref, k_ref) = (alloc(&mut gpu, NQ * HD), alloc(&mut gpu, NQ * HD), alloc(&mut gpu, NK * HD));
    let (q_fused, gate_fused, k_fused) =
        (alloc(&mut gpu, NQ * HD), alloc(&mut gpu, NQ * HD), alloc(&mut gpu, NK * HD));

    for p in POSITIONS {
        write(&gpu, &q_full, &fill(&mut seed, NQ * 2 * HD, 4.0));
        write_bytes(&gpu, &pos, &p.to_ne_bytes());
        write(&gpu, &k_ref, &k_raw);
        write(&gpu, &k_fused, &k_raw);

        gpu.deinterleave_f32(&q_full, &q_ref, &gate_ref, NQ, HD).unwrap();
        gpu.rmsnorm_batched(&q_ref, &q_weight, &q_ref, NQ, HD, EPS).unwrap();
        gpu.rmsnorm_batched(&k_ref, &k_weight, &k_ref, NK, HD, EPS).unwrap();
        gpu.rope_partial_interleaved_f32(&q_ref, &k_ref, &pos.buf, NQ, NK, HD, NROT, THETA)
            .unwrap();

        gpu.qwen35_fa_prep_gfx1100(
            &q_full, &q_fused, &gate_fused, &k_fused, &q_weight, &k_weight, &pos.buf, EPS, THETA, NQ,
            NK,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        for (name, r, f) in [("q", &q_ref, &q_fused), ("gate", &gate_ref, &gate_fused), ("k", &k_ref, &k_fused)] {
            let (r, f) = (bits(&gpu, r), bits(&gpu, f));
            let differ: Vec<usize> = (0..r.len()).filter(|&i| r[i] != f[i]).collect();
            assert!(
                differ.is_empty(),
                "{} pos {p}: {} of {} {name} floats differ between the fused FA prep and the unfused chain \
                 (first in-head offsets {:?})",
                gpu.arch,
                differ.len(),
                r.len(),
                differ.iter().take(8).map(|i| i % HD).collect::<Vec<_>>(),
            );
        }
    }
    for t in [q_full, q_weight, k_weight, pos, q_ref, gate_ref, k_ref, q_fused, gate_fused, k_fused] {
        gpu.free_tensor(t).unwrap();
    }
}
