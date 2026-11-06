// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte-parity gate for `Gpu::moe_down_combine_top10_batched_partitioned`.
//!
//! For every rank `r < ranks` the partitioned kernel must equal the ORIGINAL
//! `Gpu::moe_down_combine_top10_batched` run on a host copy of the expanded rows in which every
//! live slot whose expert `e % ranks != r` is replaced by +0.0f. Compared as raw f32 bits on the
//! whole residual. The expanded-row input must stay unchanged, a sentinel row one past the
//! residual must stay unchanged, and whatever sits in a foreign slot's row (random garbage,
//! NaN/inf payloads) must not matter. Only per-rank parity is asserted: the per-rank partials are
//! never required to sum to anything.
//!
//! `#[ignore]`d: needs a GPU.
//!
//!   cargo test -p rdna-compute --release --test moe_down_combine_top10_partitioned_hw -- --ignored --nocapture

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
    fn unit(&mut self) -> f32 {
        (self.next_u32() & 0xFF_FFFF) as f32 / (0x100_0000 as f32)
    }
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

const N_EXPERTS: usize = 17;
const SENTINEL: u32 = 0xDEAD_BEEF;

fn upload_bytes(gpu: &mut Gpu, numel: usize, data: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[numel], DType::F32).expect("alloc");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}
fn upload_f32(gpu: &mut Gpu, data: &[f32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, data.len(), bytes)
}
fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    upload_bytes(gpu, data.len(), bytes)
}
fn download_u32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
    out
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Random,
    InvalidIds,
    DuplicateIds,
    Cancellation,
    Subnormals,
    SignedZeros,
    NonzeroResidual,
}
const SCENARIOS: [Scenario; 7] = [
    Scenario::Random,
    Scenario::InvalidIds,
    Scenario::DuplicateIds,
    Scenario::Cancellation,
    Scenario::Subnormals,
    Scenario::SignedZeros,
    Scenario::NonzeroResidual,
];

struct Case {
    outputs: Vec<f32>,
    ids: Vec<i32>,
    weights: Vec<f32>,
    residual: Vec<f32>,
}

fn gen_case(s: Scenario, rng: &mut Lcg, tokens: usize, hidden: usize) -> Case {
    let slots = tokens * 10;
    let mut ids = vec![0i32; slots];
    for t in 0..tokens {
        // ten distinct ids by default (partial Fisher-Yates over N_EXPERTS)
        let mut pool: Vec<i32> = (0..N_EXPERTS as i32).collect();
        for k in 0..10 {
            let j = k + rng.below((N_EXPERTS - k) as u32) as usize;
            pool.swap(k, j);
            ids[t * 10 + k] = pool[k];
        }
    }
    let mut weights: Vec<f32> = (0..slots).map(|_| rng.unit()).collect();
    let mut outputs: Vec<f32> = (0..slots * hidden).map(|_| rng.signed() * 4.0).collect();
    let mut residual = vec![0.0f32; tokens * hidden];

    match s {
        Scenario::Random => {}
        Scenario::InvalidIds => {
            for id in ids.iter_mut() {
                match rng.below(5) {
                    0 => *id = -1,
                    1 => *id = N_EXPERTS as i32 + rng.below(40) as i32,
                    _ => {}
                }
            }
            // an all-unrouted row
            for (k, id) in ids.iter_mut().take(10).enumerate() {
                *id = if k % 2 == 0 { -1 } else { N_EXPERTS as i32 };
            }
            // unrouted rows may hold uninitialised data: poison them with NaN
            for (slot, &id) in ids.iter().enumerate() {
                if id < 0 || id >= N_EXPERTS as i32 {
                    for c in 0..hidden {
                        outputs[slot * hidden + c] = f32::NAN;
                    }
                }
            }
        }
        Scenario::DuplicateIds => {
            for t in 0..tokens {
                let a = rng.below(N_EXPERTS as u32) as i32;
                let b = rng.below(N_EXPERTS as u32) as i32;
                for k in 0..10 {
                    ids[t * 10 + k] = if k % 3 == 0 { a } else { b };
                }
            }
        }
        Scenario::Cancellation => {
            for t in 0..tokens {
                for k in (0..10).step_by(2) {
                    let (a, b) = (t * 10 + k, t * 10 + k + 1);
                    weights[b] = weights[a];
                    for c in 0..hidden {
                        let x = rng.signed() * 8.0;
                        outputs[a * hidden + c] = x;
                        outputs[b * hidden + c] = -x;
                    }
                }
            }
        }
        Scenario::Subnormals => {
            for v in outputs.iter_mut() {
                *v = f32::from_bits(rng.below(0x80_0000)) * if rng.below(2) == 0 { 1.0 } else { -1.0 };
            }
            for w in weights.iter_mut() {
                *w = f32::from_bits(rng.below(0x80_0000)) * if rng.below(2) == 0 { 1.0 } else { -1.0 };
            }
            for v in residual.iter_mut() {
                *v = f32::from_bits(rng.below(0x80_0000));
            }
        }
        Scenario::SignedZeros => {
            for v in outputs.iter_mut() {
                if rng.below(3) == 0 {
                    *v = -0.0;
                }
            }
            for w in weights.iter_mut() {
                *w = match rng.below(4) {
                    0 => -0.0,
                    1 => -rng.unit(),
                    _ => *w,
                };
            }
            for v in residual.iter_mut() {
                *v = if rng.below(2) == 0 { -0.0 } else { 0.0 };
            }
        }
        Scenario::NonzeroResidual => {
            for v in residual.iter_mut() {
                *v = rng.signed() * 16.0;
            }
        }
    }
    // most scenarios also exercise a nonzero seed
    if !matches!(s, Scenario::Subnormals | Scenario::SignedZeros | Scenario::NonzeroResidual) {
        for v in residual.iter_mut() {
            *v = rng.signed();
        }
    }
    Case { outputs, ids, weights, residual }
}

fn is_foreign(id: i32, rank: usize, ranks: usize) -> bool {
    id >= 0 && (id as usize) < N_EXPERTS && (id as usize) % ranks != rank
}

#[test]
#[ignore = "needs a GPU"]
fn partitioned_combine_matches_zeroed_original_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    let mut rng = Lcg(0x5EED_BA1A_0000_0001);
    let mut compared = 0usize;
    for &tokens in &[1usize, 3, 8] {
        for &hidden in &[2560usize, 97] {
            for &s in SCENARIOS.iter() {
                let case = gen_case(s, &mut rng, tokens, hidden);
                let ids = upload_i32(&mut gpu, &case.ids);
                let w = upload_f32(&mut gpu, &case.weights);
                for ranks in 2usize..=4 {
                    for rank in 0..ranks {
                        let tag = format!("{s:?} tokens={tokens} hidden={hidden} ranks={ranks} rank={rank}");
                        // Reference: foreign live slots replaced by +0.0.
                        let mut zeroed = case.outputs.clone();
                        // Garbage in foreign rows: must not matter.
                        let mut garbage = case.outputs.clone();
                        for slot in 0..tokens * 10 {
                            if is_foreign(case.ids[slot], rank, ranks) {
                                for c in 0..hidden {
                                    zeroed[slot * hidden + c] = 0.0;
                                    garbage[slot * hidden + c] = match rng.below(8) {
                                        0 => f32::NAN,
                                        1 => f32::INFINITY,
                                        2 => f32::NEG_INFINITY,
                                        _ => f32::from_bits(rng.next_u32()),
                                    };
                                }
                            }
                        }
                        let ref_out = upload_f32(&mut gpu, &zeroed);
                        let ref_res = upload_f32(&mut gpu, &case.residual);
                        gpu.moe_down_combine_top10_batched(
                            &ref_out, &ids, &w, &ref_res, hidden, tokens, N_EXPERTS,
                        )
                        .expect("original combine");
                        gpu.hip.device_synchronize().expect("sync");
                        let expect = download_u32(&gpu, &ref_res, tokens * hidden);

                        for (label, src) in [("clean", &case.outputs), ("garbage", &garbage)] {
                            let eo = upload_f32(&mut gpu, src);
                            // residual with one sentinel row past the end (canary)
                            let mut padded = case.residual.clone();
                            padded.extend(std::iter::repeat(f32::from_bits(SENTINEL)).take(hidden));
                            let res = upload_f32(&mut gpu, &padded);
                            gpu.moe_down_combine_top10_batched_partitioned(
                                &eo, &ids, &w, &res, hidden, tokens, N_EXPERTS, rank, ranks,
                            )
                            .expect("partitioned combine");
                            gpu.hip.device_synchronize().expect("sync");
                            let got = download_u32(&gpu, &res, tokens * hidden + hidden);
                            for (i, (g, e)) in got[..tokens * hidden].iter().zip(&expect).enumerate() {
                                assert_eq!(
                                    g, e,
                                    "{tag} [{label}] residual word {i}: got {g:#010x} want {e:#010x}"
                                );
                            }
                            assert!(
                                got[tokens * hidden..].iter().all(|&b| b == SENTINEL),
                                "{tag} [{label}]: canary row past the residual was written"
                            );
                            let after = download_u32(&gpu, &eo, src.len());
                            assert_eq!(after, bits(src), "{tag} [{label}]: expert_outputs was modified");
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("partitioned combine: {compared} byte-exact comparisons");
}

#[test]
#[ignore = "needs a GPU"]
fn partitioned_combine_rejects_invalid_partition_hw() {
    let mut gpu = Gpu::init().expect("GPU required");
    let (tokens, hidden) = (1usize, 64usize);
    let eo = upload_f32(&mut gpu, &vec![1.0f32; tokens * 10 * hidden]);
    let ids = upload_i32(&mut gpu, &(0..10).collect::<Vec<i32>>());
    let w = upload_f32(&mut gpu, &vec![1.0f32; 10]);
    let seed: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
    let res = upload_f32(&mut gpu, &seed);
    for &(rank, ranks) in &[(0usize, 0usize), (2, 2), (3, 2), (0, 5), (4, 4)] {
        let r = gpu.moe_down_combine_top10_batched_partitioned(
            &eo, &ids, &w, &res, hidden, tokens, N_EXPERTS, rank, ranks,
        );
        assert!(r.is_err(), "rank={rank} ranks={ranks} must be refused");
    }
    gpu.hip.device_synchronize().expect("sync");
    assert_eq!(
        download_u32(&gpu, &res, hidden),
        bits(&seed),
        "a refused call must not launch"
    );
}
