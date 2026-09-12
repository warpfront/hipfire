// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Indexed MoE gate_up tail parity: K2816 (eleven groups, three-group tail)
//! and K2048 (eight groups, no tail) through both the MQ4 launcher
//! (`gemv_mq4g256_moe_gate_up_k8_indexed`, FWHT-rotated input) and the HFQ
//! launcher (`gemv_hfq4g256_moe_gate_up_k8_indexed`, plain input), against a
//! CPU dequant reference built from the documented 136 B/group packed layout.
//!
//! Provenance: converted from the `maintainer_tail_smoke` throwaway after a
//! parent GPU run measured fixed-kernel maxerr 2e-6 (MQ4) / 3e-6 (HFQ) at
//! K2816 and 2e-6 both launchers at K2048, vs 2.02 (MQ4) / 5.20 (HFQ) on the
//! pre-fix generic kernel. Tolerance 1e-4 is ~30x measured fixed noise and
//! ~4 orders below the dropped-tail failure mode.
//!
//! `#[ignore]`d: needs an RDNA wave32 GPU with a working HIP toolchain.
//! Run explicitly (no model files):
//!
//!   cargo test -p rdna-compute --release --test moe_gate_up_tail_parity -- --ignored
//!
//! Tail-3 input lanes AND tail-group weights are forced nonzero, so a
//! tail-dropping kernel fails the K2816 cases by construction.

use rdna_compute::{DType, Gpu, GpuTensor};

const MI: usize = 32; // rows per gate/up half (kernel splits at M/2)
const M: usize = 2 * MI; // packed expert rows: gate 0..MI, up MI..2MI
const N_EXP: usize = 2;
const K_TOP: usize = 8; // MQ4 launcher bakes grid-y 8; topk buffer holds 8 ids
/// ~30x the parent-measured fixed-kernel noise (2e-6 MQ4 / 3e-6 HFQ);
/// the dropped-tail mode measured 2.02 / 5.20 on the same shapes.
const TOL: f32 = 1e-4;

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
fn alloc_f32_zeros(gpu: &mut Gpu, n: usize) -> GpuTensor {
    let t = gpu.alloc_tensor(&[n], DType::F32).expect("alloc zeros");
    gpu.hip.memset(&t.buf, 0, n * 4).expect("memset");
    t
}
fn download_f32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

/// Packed HFQ4/MQ4-G256 expert: groups = K/256 groups of 136 B
/// ([f32 scale][f32 zero-point][32 x u32 nibbles]), matching the indexed
/// gate_up kernel's DOG_X8 decode. All groups nonzero by construction.
fn synth_packed(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let row_bytes = groups * 136;
    let mut out = vec![0u8; m * row_bytes];
    let mut st = seed;
    let mut rng = || -> u32 {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (st >> 33) as u32
    };
    for row in 0..m {
        for g in 0..groups {
            let off = row * row_bytes + g * 136;
            let sc: f32 = 0.004 + (rng() & 0x3F) as f32 * 1e-4;
            out[off..off + 4].copy_from_slice(&sc.to_bits().to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&(-0.03f32).to_bits().to_le_bytes());
            for w in 0..32 {
                // Low nibble forced odd so no weight is ever exactly zp-only.
                let pk = rng() | 0x1111_1111;
                out[off + 8 + w * 4..off + 8 + w * 4 + 4].copy_from_slice(&pk.to_le_bytes());
            }
        }
    }
    out
}

/// CPU dequant of one packed row: value = scale * nibble + zero_point,
/// nibble n of k-lane (pk >> 4n) & F — the kernel's DOG_X8 order.
fn deq_row(packed: &[u8], row: usize, groups: usize) -> Vec<f32> {
    let rb = groups * 136;
    let mut w = vec![0f32; groups * 256];
    for g in 0..groups {
        let off = row * rb + g * 136;
        let sc = f32::from_le_bytes(packed[off..off + 4].try_into().unwrap());
        let zp = f32::from_le_bytes(packed[off + 4..off + 8].try_into().unwrap());
        for t in 0..32 {
            let pk = u32::from_le_bytes(
                packed[off + 8 + 4 * t..off + 12 + 4 * t]
                    .try_into()
                    .unwrap(),
            );
            for n in 0..8 {
                w[g * 256 + t * 8 + n] = sc * (((pk >> (4 * n)) & 0xF) as f32) + zp;
            }
        }
    }
    w
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore]
fn indexed_moe_gate_up_tail_parity() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP — no GPU ({e:?}).");
            return;
        }
    };
    if !gpu.arch_caps.is_wave32() {
        eprintln!("SKIP — requires an RDNA wave32 device.");
        return;
    }

    // topk cycles 2 experts; x tail lanes forced nonzero for K=2816.
    let topk: Vec<i32> = (0..K_TOP).map(|j| (j % N_EXP) as i32).collect();

    for k in [2816usize, 2048usize] {
        let groups = k / 256;
        let mut st = 0xABCDu64 + k as u64;
        let mut x: Vec<f32> = (0..k)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        if k == 2816 {
            for (i, v) in x[2048..].iter_mut().enumerate() {
                *v = if i % 2 == 0 { 0.75 } else { -0.75 };
            }
        }

        // One packed expert buffer per expert (gate rows 0..MI, up MI..2MI).
        let mut keep: Vec<GpuTensor> = Vec::new();
        let mut ptrs: Vec<u64> = Vec::with_capacity(N_EXP);
        let mut hosts: Vec<Vec<u8>> = Vec::with_capacity(N_EXP);
        for e in 0..N_EXP {
            let packed = synth_packed(M, k, 0x9E3779B9 ^ (k as u64) ^ (e as u64));
            let t = upload_u8(&mut gpu, &packed);
            ptrs.push(t.buf.as_ptr() as u64);
            keep.push(t);
            hosts.push(packed);
        }
        let expert_ptrs = upload_u64(&mut gpu, &ptrs);
        let topk_buf = upload_i32(&mut gpu, &topk);
        let x_buf = upload_f32(&mut gpu, &x);

        for launcher in ["mq4", "hfq"] {
            // MQ4 consumes the FWHT-rotated input; download the rotated
            // vector and use it as the CPU reference input so this case
            // isolates gate_up kernel math (rotation covered elsewhere).
            let x_ref: Vec<f32>;
            let x_gpu: GpuTensor;
            if launcher == "mq4" {
                let xr = alloc_f32_zeros(&mut gpu, k);
                gpu.rotate_x_mq(&x_buf, &xr, k).expect("rotate_x_mq");
                gpu.hip.device_synchronize().expect("rot sync");
                x_ref = download_f32(&gpu, &xr, k);
                x_gpu = xr;
            } else {
                x_ref = x.clone();
                x_gpu = upload_f32(&mut gpu, &x);
            }
            let y_gate = alloc_f32_zeros(&mut gpu, K_TOP * MI);
            let y_up = alloc_f32_zeros(&mut gpu, K_TOP * MI);
            if launcher == "mq4" {
                gpu.gemv_mq4g256_moe_gate_up_k8_indexed(
                    &expert_ptrs,
                    &topk_buf,
                    &x_gpu,
                    &y_gate,
                    &y_up,
                    M,
                    k,
                )
                .expect("mq4 idx");
            } else {
                gpu.gemv_hfq4g256_moe_gate_up_k8_indexed(
                    &expert_ptrs,
                    &topk_buf,
                    &x_gpu,
                    &y_gate,
                    &y_up,
                    M,
                    k,
                    K_TOP,
                )
                .expect("hfq idx");
            }
            gpu.hip.device_synchronize().expect("launch sync");
            let got_gate = download_f32(&gpu, &y_gate, K_TOP * MI);
            let got_up = download_f32(&gpu, &y_up, K_TOP * MI);
            for (slot, v) in got_gate.iter().chain(got_up.iter()).enumerate() {
                assert!(
                    v.is_finite(),
                    "[{launcher} K={k}] output slot {slot} nonfinite: {v}"
                );
            }

            let mut max_err: f32 = 0.0;
            for j in 0..K_TOP {
                let e = topk[j] as usize;
                for row in 0..MI {
                    let gw = deq_row(&hosts[e], row, groups);
                    let uw = deq_row(&hosts[e], row + MI, groups);
                    max_err = max_err
                        .max((got_gate[j * MI + row] - dot(&gw, &x_ref)).abs())
                        .max((got_up[j * MI + row] - dot(&uw, &x_ref)).abs());
                }
            }
            assert!(
                max_err <= TOL,
                "[{launcher} K={k}] max_abs_err={max_err:.6} exceeds tol={TOL}"
            );
        }
    }
}
