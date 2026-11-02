// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 2:4-sparse weights in the dense MQ4V2 container, for the SWMMAC K-loops
//! (`Mma::S24X64`, `Mma::S24X32`).
//!
//! Staging, the LDS layout and the `0x88888888` sign flip only move bytes, so a
//! sparse K128 row block lives in the same 64 code bytes as the dense block
//! (header unchanged): the packer puts the compressed 4-bit codes and the
//! index words where the dense staging routes them into the registers the
//! SWMMAC reads.
//!
//! # Dense slot map (derived from `prologue` / `publish`, checked by the
//! emulator oracle)
//!
//! Natural dense K (`codes[k]`: byte `k/2`, low nibble = even K) reaches the
//! W fragment `V<2>` of (slab `sl`, sub-block `sb`, lane half `h`, VGPR `j`,
//! nibble `n`) as `k = 64*sl + 32*sb + 16*h + 8*j + n`. The activation
//! fragments use the same map.
//!
//! # Sparse logical K and the 2:4 group
//!
//! Both SWMMAC forms take the dense activation fragments unpermuted as B, so
//! a 2:4 group (4 consecutive K') is 4 aligned consecutive natural K:
//! `group_of(_, k) = k / 4`, 32 groups per K128 block, for x64 and x32 alike.
//! The two forms read the same packed bytes; they differ only in how the
//! kernel feeds them to the MMA (one K64 SWMMAC per slab versus one K32
//! SWMMAC per sub-block with `index_key`).
//!
//! # Layout of one slab `sl` (32 bytes = natural K `64*sl .. 64*sl+64`)
//!
//! For each group `(sl, sb, h, j, t)` (natural base `K0 = 64*sl + 32*sb + 16*h
//! + 8*j + 4*t`), kept offsets `s0 < s1` in `0..4`:
//! - code `i` (the code at `K0 + s_i`) sits at natural position
//!   `64*sl + 16*h + 8*sb + 4*j + 2*t + i` (the registers dense fills with
//!   W(sb0), VGPR `sb`, nibble `4j+2t+i`);
//! - selector `s_i` sits in the lane's index word at bit `16*sb + 8*j + 4*t +
//!   2*i`, the word at natural positions `64*sl + 32 + 16*h + 0..8` (bytes
//!   `32*sl + 16 + 8*h + 0..4`, the first register of W(sb1)), stored XOR
//!   `0x88888888` because the kernel flips every W word. The second register
//!   of W(sb1) is padding (zero).
//!
//! Pruned codes must be 8 (zero after the flip) so the dense kernel on the
//! pruned dense weights computes the same `Y`.
use super::spec::Mma;

/// Groups per K128 block.
pub const GROUPS: usize = 32;
/// K per group.
pub const GROUP_K: usize = 4;

/// The 2:4 group (0..32) of each dense K (0..128) of one K128 row block.
/// Defined for every `mma`; both sparse forms use the same quads.
pub fn group_of(_mma: Mma, k: usize) -> usize {
    assert!(k < 128, "K {k} is outside a K128 block");
    k / GROUP_K
}

fn put_nibble(bytes: &mut [u8; 64], position: usize, q: u8) {
    let b = &mut bytes[position / 2];
    if position % 2 == 0 {
        *b = (*b & 0xf0) | q;
    } else {
        *b = (*b & 0x0f) | (q << 4);
    }
}

/// Pack one row's K128 block. `codes[k]` is the uint4 code (0..=15) in natural
/// dense K order (the order of the dense CPU reference and the shipped kernel);
/// exactly 2 K per group have `kept[k]`; every pruned K must have code 8.
/// Returns the 64 code bytes that replace the dense block's code bytes; the
/// block header is unchanged.
pub fn pack_block(mma: Mma, codes: &[u8; 128], kept: &[bool; 128]) -> Result<[u8; 64], String> {
    if !mma.is_sparse() {
        return Err("pack_block: Mma::Dense has no 2:4 packing".into());
    }
    if let Some(k) = codes.iter().position(|&c| c > 15) {
        return Err(format!(
            "pack_block: code {} at K {k} is not a uint4",
            codes[k]
        ));
    }
    let mut out = [0u8; 64];
    let mut index = [[0u32; 2]; 2]; // [sl][h]
    for g in 0..GROUPS {
        let k0 = g * GROUP_K;
        let sel: Vec<usize> = (0..GROUP_K).filter(|&s| kept[k0 + s]).collect();
        if sel.len() != 2 {
            return Err(format!(
                "pack_block: group {g} (K {k0}..{}) keeps {} of 4 K, 2:4 needs exactly 2",
                k0 + 3,
                sel.len()
            ));
        }
        for s in (0..GROUP_K).filter(|&s| !kept[k0 + s]) {
            if codes[k0 + s] != 8 {
                return Err(format!(
                    "pack_block: pruned K {} has code {}, a pruned code must be 8 (zero)",
                    k0 + s,
                    codes[k0 + s]
                ));
            }
        }
        let (sl, r) = (k0 / 64, k0 % 64);
        let (sb, h, j, t) = (r / 32, (r / 16) % 2, (r / 8) % 2, (r / 4) % 2);
        for (i, &s) in sel.iter().enumerate() {
            put_nibble(
                &mut out,
                64 * sl + 16 * h + 8 * sb + 4 * j + 2 * t + i,
                codes[k0 + s],
            );
            index[sl][h] |= (s as u32) << (16 * sb + 8 * j + 4 * t + 2 * i);
        }
    }
    for (sl, words) in index.iter().enumerate() {
        for (h, &w) in words.iter().enumerate() {
            let at = 32 * sl + 16 + 8 * h;
            out[at..at + 4].copy_from_slice(&(w ^ 0x8888_8888).to_le_bytes());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// A random 2:4-pruned block: pruned codes 8, kept codes random.
    fn block(seed: u64) -> ([u8; 128], [bool; 128]) {
        let mut r = Lcg(seed);
        let (mut codes, mut kept) = ([8u8; 128], [false; 128]);
        for g in 0..GROUPS {
            let a = (r.next() % 4) as usize;
            let b = (a + 1 + (r.next() % 3) as usize) % 4;
            for s in [a, b] {
                kept[4 * g + s] = true;
                codes[4 * g + s] = (r.next() % 16) as u8;
            }
        }
        (codes, kept)
    }

    #[test]
    fn groups_are_four_aligned_consecutive_k() {
        for mma in [Mma::S24X64, Mma::S24X32] {
            let mut count = [0usize; GROUPS];
            for k in 0..128 {
                let g = group_of(mma, k);
                assert!(g < GROUPS);
                count[g] += 1;
                assert_eq!(g, group_of(mma, 4 * (k / 4)), "K {k} leaves its quad");
            }
            assert!(count.iter().all(|&c| c == 4));
        }
    }

    /// The SWMMAC kernel's view of a packed block for one lane `(row, h)`
    /// slab `sl`: dense registers (W flip applied), then the products the
    /// emulator's `swmmac` forms form against a dense B, summed.
    fn sparse_dot(mma: Mma, packed: &[u8; 64], x: &[i32; 128]) -> i32 {
        // Register nibble at natural position p: flip bit 3 (the 0x88888888 XOR), signed 4-bit.
        let nib = |p: usize| -> i32 {
            let v = (packed[p / 2] >> (4 * (p % 2))) & 0xf;
            i32::from(((v ^ 8) << 4) as i8 >> 4)
        };
        let word = |at: usize| -> u32 {
            u32::from_le_bytes(packed[at..at + 4].try_into().unwrap()) ^ 0x8888_8888
        };
        let mut sum = 0;
        for sl in 0..2 {
            for h in 0..2 {
                // Lane (row, h): codes V<2> at natural 64sl + 16h + 0..16, index = first W(sb1) register.
                let idx = word(32 * sl + 16 + 8 * h);
                // Slot `e` (0..16) of the lane: VGPR e/8, nibble e%8 -> natural 64sl+16h+e.
                for e in 0..16 {
                    let a = nib(64 * sl + 16 * h + e);
                    let (sb, j, t, i) = (e / 8, (e / 4) % 2, (e / 2) % 2, e % 2);
                    let (sel, group_k0) = match mma {
                        // x64: slot e meets K' = 32h + 8jj + 4t + sel, jj = 2sb + j; selector at 2e.
                        Mma::S24X64 => (
                            ((idx >> (2 * e)) & 3) as usize,
                            64 * sl + 32 * sb + 16 * h + 8 * j + 4 * t,
                        ),
                        // x32: sub-block sb, slot e%8 of key sb: selector at 16*sb + 2*(e%8).
                        Mma::S24X32 => {
                            let (sbk, ee) = (e / 8, e % 8);
                            (
                                ((idx >> (16 * sbk + 2 * ee)) & 3) as usize,
                                64 * sl
                                    + 32 * sbk
                                    + 16 * h
                                    + 8 * ((ee / 4) % 2)
                                    + 4 * ((ee / 2) % 2),
                            )
                        }
                        Mma::Dense => unreachable!(),
                    };
                    let _ = i;
                    sum += a * x[group_k0 + sel];
                }
            }
        }
        sum
    }

    fn dense_dot(codes: &[u8; 128], x: &[i32; 128]) -> i32 {
        (0..128)
            .map(|k| i32::from((((codes[k] ^ 8) << 4) as i8) >> 4) * x[k])
            .sum()
    }

    #[test]
    fn packed_block_computes_the_pruned_dense_dot() {
        let mut r = Lcg(7);
        for seed in 1..200u64 {
            let (codes, kept) = block(seed);
            let x: [i32; 128] = std::array::from_fn(|_| (r.next() % 16) as i32 - 8);
            for mma in [Mma::S24X64, Mma::S24X32] {
                let packed = pack_block(mma, &codes, &kept).unwrap();
                assert_eq!(
                    sparse_dot(mma, &packed, &x),
                    dense_dot(&codes, &x),
                    "{mma:?} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn x64_and_x32_share_the_bytes() {
        let (codes, kept) = block(3);
        assert_eq!(
            pack_block(Mma::S24X64, &codes, &kept).unwrap(),
            pack_block(Mma::S24X32, &codes, &kept).unwrap()
        );
    }

    #[test]
    fn packing_is_a_bijection_onto_the_code_bytes() {
        // Every pruned K carries 8 and every kept K a distinct code slot:
        // flipping one kept code changes exactly one nibble.
        let (codes, kept) = block(11);
        let base = pack_block(Mma::S24X64, &codes, &kept).unwrap();
        for k in (0..128).filter(|&k| kept[k]) {
            let mut c = codes;
            c[k] = (c[k] + 1) % 16;
            let p = pack_block(Mma::S24X64, &c, &kept).unwrap();
            let diff: usize = base
                .iter()
                .zip(&p)
                .map(|(a, b)| ((a ^ b) & 0xf != 0) as usize + ((a ^ b) >> 4 != 0) as usize)
                .sum();
            assert_eq!(diff, 1, "K {k}");
        }
    }

    #[test]
    fn refuses_a_group_without_exactly_two_kept() {
        let (codes, mut kept) = block(5);
        kept[1] = !kept[1];
        let e = pack_block(Mma::S24X64, &codes, &kept).unwrap_err();
        assert!(e.contains("2:4 needs exactly 2"), "{e}");
        let (codes, mut kept) = block(5);
        kept[0..4].fill(true);
        let e = pack_block(Mma::S24X32, &codes, &kept).unwrap_err();
        assert!(e.contains("keeps 4 of 4"), "{e}");
    }

    #[test]
    fn refuses_a_pruned_code_that_is_not_eight() {
        let (mut codes, kept) = block(9);
        let k = (0..128).find(|&k| !kept[k]).unwrap();
        codes[k] = 9;
        let e = pack_block(Mma::S24X64, &codes, &kept).unwrap_err();
        assert!(e.contains("a pruned code must be 8"), "{e}");
    }

    #[test]
    fn refuses_dense_and_non_uint4_codes() {
        let (codes, kept) = block(2);
        assert!(pack_block(Mma::Dense, &codes, &kept)
            .unwrap_err()
            .contains("no 2:4 packing"));
        let mut bad = codes;
        let k = (0..128).find(|&k| kept[k]).unwrap();
        bad[k] = 16;
        assert!(pack_block(Mma::S24X64, &bad, &kept)
            .unwrap_err()
            .contains("not a uint4"));
    }
}
