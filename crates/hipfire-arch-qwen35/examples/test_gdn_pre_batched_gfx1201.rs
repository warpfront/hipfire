// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Slice-P parity gate (gfx1201-only): byte-for-byte oracle for
//! `gdn_pre_batched_gfx1201` against the exact old 3-launch sequence
//! (`fused_sigmoid_alpha_gate_f32_batched` + `conv1d_silu_split_f32_n` +
//! `fused_qk_l2_norm_scale_interleave_f32_batched`), on fixture-shaped
//! synthetic inputs (16 key heads, 48 value heads, head_dim 128, ratio 3).
//!
//! Covers whole-N vs original sequence at N in {1,2,3,31,32,33,511,512} with
//! an identical nonzero initial conv ring; compares all output bits of
//! beta/alpha/q_raw/k_raw/v/q/k plus the final ring, and re-runs the fused
//! kernel for self-determinism. Any mismatch aborts nonzero. Non-gfx1201
//! exits 0 with a skip note.

use rdna_compute::{DType, Gpu, GpuTensor};

const HD: usize = 128;
const N_KEY: usize = 16;
const N_V: usize = 48;
const RATIO: usize = 3;
const K_DIM: usize = N_KEY * HD;
const V_DIM: usize = N_V * HD;
const QKV_DIM: usize = 2 * K_DIM + V_DIM;
const N_CH: usize = QKV_DIM;
const MAX_N: usize = 512;
const EPS: f32 = 1e-6;

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; v.len() * 4];
    for (i, f) in v.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&f.to_ne_bytes());
    }
    b
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Deterministic pseudo-random fill (LCG).
fn fill_lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
            (u as f32) * scale
        })
        .collect()
}

fn upload(gpu: &Gpu, t: &GpuTensor, v: &[f32]) {
    gpu.hip
        .memcpy_htod(&t.buf, &f32s_to_bytes(v))
        .expect("upload");
}

fn poison(gpu: &Gpu, t: &GpuTensor) {
    let n = t.byte_size();
    gpu.hip
        .memcpy_htod(&t.buf, &vec![0xABu8; n])
        .expect("poison");
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    let mut b = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).expect("download");
    b
}

fn check_eq(name: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    if a != b {
        let mut first = 0;
        while first < a.len() && a[first] == b[first] {
            first += 1;
        }
        let af = bytes_to_f32s(&a[first..(first + 4).min(a.len())]);
        let bf = bytes_to_f32s(&b[first..(first + 4).min(b.len())]);
        panic!(
            "{name}: byte mismatch at byte {first} ({} total): old={af:?} new={bf:?}",
            a.len()
        );
    }
    eprintln!("  ok {name} ({} bytes identical)", a.len());
}

struct Arm {
    beta: GpuTensor,
    alpha: GpuTensor,
    q_raw: GpuTensor,
    k_raw: GpuTensor,
    v: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    conv_state: GpuTensor,
}

impl Arm {
    fn alloc(gpu: &mut Gpu) -> Self {
        Self {
            beta: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).expect("beta"),
            alpha: gpu.alloc_tensor(&[MAX_N * N_V], DType::F32).expect("alpha"),
            q_raw: gpu
                .alloc_tensor(&[MAX_N * K_DIM], DType::F32)
                .expect("q_raw"),
            k_raw: gpu
                .alloc_tensor(&[MAX_N * K_DIM], DType::F32)
                .expect("k_raw"),
            v: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("v"),
            q: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("q"),
            k: gpu.alloc_tensor(&[MAX_N * V_DIM], DType::F32).expect("k"),
            conv_state: gpu
                .alloc_tensor(&[N_CH * 3], DType::F32)
                .expect("conv_state"),
        }
    }

    fn poison_all(&self, gpu: &Gpu) {
        poison(gpu, &self.beta);
        poison(gpu, &self.alpha);
        poison(gpu, &self.q_raw);
        poison(gpu, &self.k_raw);
        poison(gpu, &self.v);
        poison(gpu, &self.q);
        poison(gpu, &self.k);
        poison(gpu, &self.conv_state);
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: test_gdn_pre_batched_gfx1201 requires exact gfx1201");
        return;
    }
    eprintln!("=== gdn_pre_batched parity (gfx1201) ===");

    let qkv_in = gpu
        .alloc_tensor(&[MAX_N * QKV_DIM], DType::F32)
        .expect("qkv_in");
    let dt_bias = gpu.alloc_tensor(&[N_V], DType::F32).expect("dt_bias");
    let a_log = gpu.alloc_tensor(&[N_V], DType::F32).expect("a_log");
    let conv_w = gpu.alloc_tensor(&[N_CH * 4], DType::F32).expect("conv_w");
    upload(&gpu, &qkv_in, &fill_lcg(MAX_N * QKV_DIM, 0x1234, 0.6));
    upload(&gpu, &dt_bias, &fill_lcg(N_V, 0xB1A5, 1.0));
    upload(&gpu, &a_log, &fill_lcg(N_V, 0xA106, 0.5));
    upload(&gpu, &conv_w, &fill_lcg(N_CH * 4, 0xC0DE, 0.25));

    let q_scale = 1.0 / (HD as f32).sqrt();

    for n in [1usize, 2, 3, 31, 32, 33, 511, 512] {
        eprintln!("--- n={n} ---");
        let mut old = Arm::alloc(&mut gpu);
        let mut new = Arm::alloc(&mut gpu);
        let mut rep = Arm::alloc(&mut gpu);
        old.poison_all(&gpu);
        new.poison_all(&gpu);
        rep.poison_all(&gpu);

        // Identical live inputs in all arms, including a NONZERO ring.
        let beta_in = fill_lcg(MAX_N * N_V, 0xBE7A, 2.0);
        let alpha_in = fill_lcg(MAX_N * N_V, 0xA1FA, 2.0);
        let conv_init = fill_lcg(N_CH * 3, 0x57A7, 0.2);
        for arm in [&old, &new, &rep] {
            upload(&gpu, &arm.beta, &beta_in);
            upload(&gpu, &arm.alpha, &alpha_in);
            upload(&gpu, &arm.conv_state, &conv_init);
        }

        // Old path, verbatim hook order.
        gpu.fused_sigmoid_alpha_gate_f32_batched(
            &old.beta, &old.alpha, &dt_bias, &a_log, N_V, n,
        )
        .expect("old sigmoid");
        gpu.conv1d_silu_split_f32_n(
            &old.q_raw, &old.k_raw, &old.v, &qkv_in, &conv_w, &old.conv_state, K_DIM,
            V_DIM, n,
        )
        .expect("old conv");
        gpu.fused_qk_l2_norm_scale_interleave_f32_batched(
            &old.q_raw, &old.k_raw, &old.q, &old.k, N_KEY, RATIO, HD, q_scale, EPS, n,
        )
        .expect("old qk");

        // New path: single launch, twice (second = determinism check).
        for (arm, tag) in [(&new, "new fused"), (&rep, "rep fused")] {
            gpu.gdn_pre_batched_gfx1201(
                &arm.beta,
                &arm.alpha,
                &dt_bias,
                &a_log,
                &qkv_in,
                &conv_w,
                &arm.conv_state,
                &arm.q_raw,
                &arm.k_raw,
                &arm.v,
                &arm.q,
                &arm.k,
                N_V,
                N_KEY,
                RATIO,
                K_DIM,
                V_DIM,
                n,
                q_scale,
                EPS,
            )
            .expect(tag);
        }

        for (name, o, w) in [
            ("beta", &old.beta, &new.beta),
            ("alpha", &old.alpha, &new.alpha),
            ("q_raw", &old.q_raw, &new.q_raw),
            ("k_raw", &old.k_raw, &new.k_raw),
            ("v", &old.v, &new.v),
            ("q", &old.q, &new.q),
            ("k", &old.k, &new.k),
            ("conv_state", &old.conv_state, &new.conv_state),
        ] {
            check_eq(name, &download(&gpu, o), &download(&gpu, w));
        }
        // Self-determinism across repeated launches.
        for (name, a, b) in [
            ("rep/beta", &new.beta, &rep.beta),
            ("rep/q", &new.q, &rep.q),
            ("rep/k", &new.k, &rep.k),
            ("rep/v", &new.v, &rep.v),
            ("rep/conv_state", &new.conv_state, &rep.conv_state),
        ] {
            check_eq(name, &download(&gpu, a), &download(&gpu, b));
        }
    }
    eprintln!("ALL PARITY CHECKS PASSED");
}
