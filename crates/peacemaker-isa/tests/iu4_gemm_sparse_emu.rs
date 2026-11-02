// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The 2:4-sparse SWMMAC K-loops of `iu4_gemm` (`Mma::S24X64`, `Mma::S24X32`) on the PeaceMaker CPU emulator.
//!
//! The oracle: random 2:4-pruned weights (pruned codes = 8) packed with `sparse24::pack_block` into the dense MQ4V2
//! container, run through the whole sparse kernel, must give `Y` bitwise equal to the shipped dense kernel run on
//! the pruned dense weights, and to the exact CPU fold on them. Activations are random `block_i4_128` records.
use iu4_emu::{reference, run_gemm, Problem, Rng, GROUP_BYTES};
use peacemaker_isa::kernels::iu4_gemm::{sparse24, Epi, Mma, Spec};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
#[allow(dead_code)]
mod iu4_emu;

/// Prune every natural 2:4 quad of every K128 row block to a pseudo-randomly chosen pair (the others get code 8).
/// Returns the pruned dense problem and the same problem with every block packed for `mma`.
fn pruned_and_packed(p: &Problem, mma: Mma, seed: u64) -> (Problem, Problem) {
    let mut rng = Rng::new(seed);
    let (mut dense, mut packed) = (clone(p), clone(p));
    let groups = p.k / 256;
    for row in 0..p.m {
        for g in 0..groups {
            for half in 0..2 {
                let at = (row * groups + g) * GROUP_BYTES + 8 + 64 * half;
                let (mut codes, mut kept) = ([0u8; 128], [false; 128]);
                for j in 0..64 {
                    let byte = p.w[at + j];
                    codes[2 * j] = byte & 0xf;
                    codes[2 * j + 1] = byte >> 4;
                }
                for quad in 0..32 {
                    // Two distinct kept offsets of 0..4.
                    let a = (rng.next() % 4) as usize;
                    let b = (a + 1 + (rng.next() % 3) as usize) % 4;
                    for s in 0..4 {
                        let k = 4 * quad + s;
                        assert_eq!(sparse24::group_of(mma, k), quad);
                        kept[k] = s == a || s == b;
                        if !kept[k] {
                            codes[k] = 8;
                        }
                    }
                }
                for j in 0..64 {
                    dense.w[at + j] = codes[2 * j] | codes[2 * j + 1] << 4;
                }
                let bytes = sparse24::pack_block(mma, &codes, &kept).unwrap();
                packed.w[at..at + 64].copy_from_slice(&bytes);
            }
        }
    }
    (dense, packed)
}

fn clone(p: &Problem) -> Problem {
    Problem {
        m: p.m,
        n: p.n,
        k: p.k,
        w: p.w.clone(),
        recs: p.recs.clone(),
        y0: p.y0.clone(),
    }
}

fn bitwise_diff(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn spec(epi: Epi, mma: Mma) -> Spec {
    Spec {
        mma,
        ..Spec::control(epi)
    }
}

/// The sparse kernel on packed weights equals the dense kernel and the CPU fold on the pruned dense weights,
/// bitwise: SET at the spec's smallest shape (M = N = 128, K = 256: no full K-loop trip, the peeled group only),
/// SET with partial tiles and a full trip (M = 136, N = 40, K = 768: two full trips and the peeled group), and ADD.
fn sparse_equals_dense(mma: Mma) {
    require_rocm_tool!("llvm-mc");
    for (m, n, k, epi) in [
        (128, 128, 256, Epi::Set),
        (136, 40, 768, Epi::Set),
        (136, 40, 512, Epi::Add),
    ] {
        let base = Problem::synth(m, n, k, 0x9e37_79b9_7f4a_7c15 ^ (k as u64));
        let (pruned, packed) = pruned_and_packed(&base, mma, 0x5eed_0000 + k as u64);
        let want = reference(&pruned, epi == Epi::Add);
        let dense = run_gemm(spec(epi, Mma::Dense), &pruned).unwrap();
        assert_eq!(
            bitwise_diff(&dense, &want),
            0,
            "the dense kernel disagrees with the CPU fold on the pruned weights ({m}x{n}x{k} {epi:?})"
        );
        let got = run_gemm(spec(epi, mma), &packed).unwrap();
        assert_eq!(
            bitwise_diff(&got, &dense),
            0,
            "{mma:?} {m}x{n}x{k} {epi:?}: first got {} dense {}",
            got[0],
            dense[0]
        );
    }
}

#[test]
fn s24x64_equals_dense_on_the_pruned_weights() {
    sparse_equals_dense(Mma::S24X64);
}

#[test]
fn s24x32_equals_dense_on_the_pruned_weights() {
    sparse_equals_dense(Mma::S24X32);
}

/// A kept pair of equal-magnitude weights against extreme activations: the selectors, the signs and the clamp-free
/// i32 accumulation are exercised at the corners (code 0 = -8 and code 15 = +7 against q = -8 and +7).
#[test]
fn sparse_extremes_are_exact() {
    require_rocm_tool!("llvm-mc");
    for mma in [Mma::S24X64, Mma::S24X32] {
        for (code, q) in [(0u8, -8), (15, -8), (0, 7), (15, 7)] {
            let base = Problem::uniform(128, 16, 256, code, q);
            let (pruned, packed) = pruned_and_packed(&base, mma, 3);
            let dense = run_gemm(spec(Epi::Set, Mma::Dense), &pruned).unwrap();
            let got = run_gemm(spec(Epi::Set, mma), &packed).unwrap();
            assert_eq!(bitwise_diff(&got, &dense), 0, "{mma:?} code {code} q {q}");
            // Half of the K are kept: Y = K/2 * (code - 8) * q.
            let want = (128 * (i32::from(code) - 8) * q) as f32;
            assert!(
                got.iter().all(|&v| v == want),
                "{mma:?} code {code} q {q}: {}",
                got[0]
            );
        }
    }
}

/// The oracle discriminates: the sparse kernel on the pruned weights left in the dense layout, and on packed weights
/// with one selector corrupted, disagrees with the dense result (and the dense result is not degenerate).
#[test]
fn the_oracle_rejects_unpacked_and_corrupted_weights() {
    require_rocm_tool!("llvm-mc");
    let base = Problem::synth(128, 128, 256, 0x1234_5678);
    for mma in [Mma::S24X64, Mma::S24X32] {
        let (pruned, packed) = pruned_and_packed(&base, mma, 77);
        let dense = run_gemm(spec(Epi::Set, Mma::Dense), &pruned).unwrap();
        assert!(dense.iter().filter(|&&v| v != 0.0).count() > dense.len() / 2);
        let unpacked = run_gemm(spec(Epi::Set, mma), &pruned).unwrap();
        assert!(bitwise_diff(&unpacked, &dense) > dense.len() / 2, "{mma:?}");
        let mut bad = clone(&packed);
        // Row 0, K128 block 0, slab 0, lane half 0: flip the first selector of the index word.
        bad.w[8 + 16] ^= 0x88 ^ 0x8b; // index word is stored ^ 0x88888888: change the low selector bits
        let corrupted = run_gemm(spec(Epi::Set, mma), &bad).unwrap();
        assert!(bitwise_diff(&corrupted, &dense) > 0, "{mma:?}");
    }
}
