// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gate for `hyper_read_up_fused_f32_sx_t1_k320_rot`
//! (`HIPFIRE_QWEN4_HC_UP_ROT=1`, module `hyper_read_up_fused_rot_t1`): the HC read tail with the next
//! mixer's FWHT rotation fused in. For 2..=8 rows (the lane-decode row counts) on randomized
//! inputs: `mixed` must equal the production t1 entry's (`hyper_read_up_fused_entry` with
//! `TwinT1`) bit for bit, and `rotated` must equal `rotate_x_mq_batched(mixed, ...)` run on that
//! reference `mixed` bit for bit. The arrival counters must be back to zero after every launch,
//! and the next `rotate_x_mq_batched` of the same tensors must be consumed (`prerotated`), so a
//! second call with a poisoned `rotated` leaves it untouched.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test hyper_read_up_rot_hw -- --ignored --nocapture

use rdna_compute::tensor_ops::{
    hyper_read_up_fused_entry, hyper_read_up_fused_rotate, HyperReadUpFused, HyperReadUpFusedEntry,
};
use rdna_compute::{DType, Gpu, GpuTensor};

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
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn upload_bytes(gpu: &mut Gpu, dtype: DType, numel: usize, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[numel], dtype).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, DType::F32, data.len(), bytes)
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}

const HIDDEN: usize = 2560;
const LOW_RANK: usize = 320;
const WIDE: usize = 4 * HIDDEN;

#[test]
#[ignore = "needs an exact gfx1201 GPU"]
fn hyper_read_up_rot_matches_t1_and_mq_rotate_hw() {
    // Read once per process by the launcher.
    std::env::set_var("HIPFIRE_QWEN4_HC_UP_ROT", "1");
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let (mut mixed_bad, mut rot_bad, mut counter_bad, mut skip_bad) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut words, mut cases) = (0usize, 0usize);
    for seed in 0..6u64 {
        let mut rng = Lcg(0x7071_0000 + seed);
        let mag = [0.05f32, 0.2, 1.0, 0.5, 2.0, 0.1][seed as usize];
        let mut up = Vec::with_capacity(WIDE * LOW_RANK * 2);
        for _ in 0..WIDE * LOW_RANK {
            up.extend_from_slice(&bf16_bits(rng.f32_signed() * mag).to_le_bytes());
        }
        let up_t = upload_bytes(&mut gpu, DType::BF16, WIDE * LOW_RANK, &up);
        for rows in 2usize..=8 {
            for normalized_bf16 in [false, true] {
                let low: Vec<f32> = (0..rows * LOW_RANK)
                    .map(|_| rng.f32_signed() * 2.0)
                    .collect();
                let low_t = upload_f32(&mut gpu, &low);
                let norm_bytes: Vec<u8> = if normalized_bf16 {
                    let mut b = Vec::with_capacity(rows * WIDE * 4);
                    for _ in 0..rows * WIDE {
                        b.extend_from_slice(&bf16_bits(rng.f32_signed() * 3.0).to_le_bytes());
                    }
                    b.resize(rows * WIDE * 4, 0);
                    b
                } else {
                    (0..rows * WIDE)
                        .flat_map(|_| (rng.f32_signed() * 3.0).to_le_bytes())
                        .collect()
                };
                let norm_t = upload_bytes(&mut gpu, DType::F32, rows * WIDE, &norm_bytes);
                let canary = f32::from_bits(0x7FC0_DEAD);
                // Reference: production t1 entry, then the production rotation of its mixed rows.
                let mixed_ref = upload_f32(&mut gpu, &vec![canary; rows * HIDDEN]);
                let rot_ref = upload_f32(&mut gpu, &vec![canary; rows * HIDDEN]);
                let p_ref = HyperReadUpFused {
                    up_weight: &up_t,
                    low: &low_t,
                    normalized: &norm_t,
                    mixed: &mixed_ref,
                    rows,
                    hidden: HIDDEN,
                    low_rank: LOW_RANK,
                    normalized_bf16,
                };
                hyper_read_up_fused_entry(&mut gpu, &p_ref, HyperReadUpFusedEntry::TwinT1)
                    .expect("t1");
                gpu.rotate_x_mq_batched(&mixed_ref, &rot_ref, HIDDEN, rows)
                    .expect("rotate");
                gpu.hip.device_synchronize().expect("sync");
                let want_mixed = download_u32(&gpu, &mixed_ref, rows * HIDDEN);
                let want_rot = download_u32(&gpu, &rot_ref, rows * HIDDEN);
                // Candidate.
                let mixed = upload_f32(&mut gpu, &vec![canary; rows * HIDDEN]);
                let rotated = upload_f32(&mut gpu, &vec![canary; rows * HIDDEN]);
                let p = HyperReadUpFused {
                    up_weight: &up_t,
                    low: &low_t,
                    normalized: &norm_t,
                    mixed: &mixed,
                    rows,
                    hidden: HIDDEN,
                    low_rank: LOW_RANK,
                    normalized_bf16,
                };
                let launched = hyper_read_up_fused_rotate(&mut gpu, &p, &rotated).expect("rot");
                assert!(launched, "fused rotate declined (flag not read?)");
                gpu.hip.device_synchronize().expect("sync");
                let got_mixed = download_u32(&gpu, &mixed, rows * HIDDEN);
                let got_rot = download_u32(&gpu, &rotated, rows * HIDDEN);
                mixed_bad += got_mixed
                    .iter()
                    .zip(&want_mixed)
                    .filter(|(g, w)| g != w)
                    .count();
                rot_bad += got_rot
                    .iter()
                    .zip(&want_rot)
                    .filter(|(g, w)| g != w)
                    .count();
                let counters = gpu.scratch.gdn_pair_counters.as_ref().expect("counters");
                let mut cbytes = vec![0u8; 256 * 4];
                gpu.hip
                    .memcpy_dtoh(&mut cbytes, counters)
                    .expect("dtoh counters");
                counter_bad += cbytes.iter().filter(|b| **b != 0).count();
                // The matching rotate is consumed: after a fused launch into `poison`, re-poison it;
                // the next `rotate_x_mq_batched` of the same tensors must then skip (leave it).
                let poison = upload_f32(&mut gpu, &vec![canary; rows * HIDDEN]);
                let q = HyperReadUpFused { mixed: &mixed, ..p };
                assert!(hyper_read_up_fused_rotate(&mut gpu, &q, &poison).expect("rot2"));
                gpu.hip.device_synchronize().expect("sync");
                let canary_bytes: Vec<u8> = std::iter::repeat(canary.to_le_bytes())
                    .take(rows * HIDDEN)
                    .flatten()
                    .collect();
                gpu.hip
                    .memcpy_htod(&poison.buf, &canary_bytes)
                    .expect("htod");
                gpu.rotate_x_mq_batched(&mixed, &poison, HIDDEN, rows)
                    .expect("skipped rotate");
                gpu.hip.device_synchronize().expect("sync");
                let after = download_u32(&gpu, &poison, rows * HIDDEN);
                skip_bad += after.iter().filter(|g| **g != canary.to_bits()).count();
                words += rows * HIDDEN;
                cases += 1;
            }
        }
    }
    eprintln!(
        "hyper_read_up_rot: {cases} cases (6 seeds x rows 2..=8 x F32/BF16 normalized), {words} words per output, mismatches mixed {mixed_bad} / rotated {rot_bad}, nonzero counter bytes {counter_bad}, skipped-rotate mismatches {skip_bad}"
    );
    assert_eq!(mixed_bad, 0, "mixed differs from the t1 entry");
    assert_eq!(rot_bad, 0, "rotated differs from mq_rotate_x(mixed)");
    assert_eq!(counter_bad, 0, "arrival counters not reset");
    assert_eq!(skip_bad, 0, "prerotated rotate was not consumed");
}
