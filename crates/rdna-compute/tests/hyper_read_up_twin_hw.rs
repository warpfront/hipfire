// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-exactness gate for the Flash-Next HC read tail twins `hyper_read_up_fused_f32_sx`
//! (module `hyper_read_up_fused_twin`) and `hyper_read_up_fused_f32_sx_t1` (module
//! `hyper_read_up_fused_twin_t1`, one token per wave; both admitted on gfx1201 for decode-sized
//! batches of at most 8 rows, t1 by default) against the incumbent `hyper_read_up_fused_f32`:
//! the production launcher entry run with `Incumbent` (oracle), `Twin` and `TwinT1` (candidates)
//! on the same randomized inputs, raw f32 bits of the whole `[rows, hidden]` mixed output
//! compared. Both F32 and BF16-bits `normalized` inputs are covered; row counts above 8 are
//! covered too (the entries must be correct when called directly). The compiler's FMA
//! contraction of the incumbent is not reproducible by a CPU model, so the incumbent is the
//! oracle.
//!
//! `#[ignore]`d: needs an exact gfx1201 device, no model files.
//!
//!   cargo test -p rdna-compute --release --test hyper_read_up_twin_hw -- --ignored --nocapture

use rdna_compute::tensor_ops::{
    hyper_read_up_fused_entry, HyperReadUpFused, HyperReadUpFusedEntry,
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
fn hyper_read_up_twin_matches_incumbent_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return;
    }
    let mut mismatches = [0usize; 3];
    let mut words = 0usize;
    let mut cases = 0usize;
    let mut first: [Option<(usize, usize, u32, u32)>; 3] = [None, None, None];
    for seed in 0..6u64 {
        let mut rng = Lcg(0x48C1_0000 + seed);
        // BF16 up projection [WIDE, LOW_RANK], magnitudes spread over a few binades.
        let mag = [0.05f32, 0.2, 1.0, 0.5, 2.0, 0.1][seed as usize];
        let mut up = Vec::with_capacity(WIDE * LOW_RANK * 2);
        for i in 0..WIDE * LOW_RANK {
            let v = if seed == 5 && i % 13 == 0 {
                0.0
            } else {
                rng.f32_signed() * mag
            };
            up.extend_from_slice(&bf16_bits(v).to_le_bytes());
        }
        let up_t = upload_bytes(&mut gpu, DType::BF16, WIDE * LOW_RANK, &up);
        for rows in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 33] {
            for normalized_bf16 in [false, true] {
                let low: Vec<f32> = (0..rows * LOW_RANK)
                    .map(|_| rng.f32_signed() * 2.0)
                    .collect();
                let low_t = upload_f32(&mut gpu, &low);
                // `normalized`: [rows, WIDE] F32 values, or BF16 bits packed in the same buffer.
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
                let run = |gpu: &mut Gpu, entry: HyperReadUpFusedEntry| -> Vec<u32> {
                    let mixed = upload_f32(gpu, &vec![canary; rows * HIDDEN]);
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
                    hyper_read_up_fused_entry(gpu, &p, entry).expect("hyper_read_up_fused");
                    gpu.hip.device_synchronize().expect("sync");
                    download_u32(gpu, &mixed, rows * HIDDEN)
                };
                let want = run(&mut gpu, HyperReadUpFusedEntry::Incumbent);
                for (ei, entry) in [
                    HyperReadUpFusedEntry::Twin,
                    HyperReadUpFusedEntry::TwinT1,
                    HyperReadUpFusedEntry::TwinT1K320,
                ]
                .into_iter()
                .enumerate()
                {
                    let got = run(&mut gpu, entry);
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        if g != w {
                            mismatches[ei] += 1;
                            first[ei].get_or_insert((cases, i, *g, *w));
                        }
                    }
                }
                words += rows * HIDDEN;
                cases += 1;
            }
        }
    }
    eprintln!(
        "hyper_read_up_twin: {cases} cases (6 seeds x 13 row counts x F32/BF16 normalized), {words} output words per entry, mismatches sx twin {} / t1 {} / t1_k320 {}, first sx {:?} / t1 {:?} / t1_k320 {:?}",
        mismatches[0], mismatches[1], mismatches[2], first[0], first[1], first[2]
    );
    assert_eq!(
        mismatches[0], 0,
        "the sx twin is not bit-identical to hyper_read_up_fused_f32"
    );
    assert_eq!(
        mismatches[1], 0,
        "the sx_t1 twin is not bit-identical to hyper_read_up_fused_f32"
    );
    assert_eq!(
        mismatches[2], 0,
        "the sx_t1_k320 entry is not bit-identical to hyper_read_up_fused_f32"
    );
}
