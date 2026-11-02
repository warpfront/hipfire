// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! K-E0 CPU twin of the E8P-on-IU4 F32 accumulators plus the A4 activation generators.
//!
//! Per K128 epoch the twin is exactly the frozen definition of `e8p_i4_reference`: nibbles `o` from
//! `unpack_word_ref`, the exact integer chain `C = sum o*q` seeded `a = MAGIC + C`, `fold128(a, s)`, then
//! `epoch_sum(d_act, d, a', sum)` from `+0` over ascending epochs. Every F32 sum is compared bit for bit with the
//! device `_acc` kernel; the BF16 output is compared with the frozen epilogue DAG over those twin sums.

use hipfire_runtime::e8p_i4_reference::{self as e8p};
use hipfire_runtime::mq_i4_reference::A4Block;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::common::*;

pub struct Oracle {
    /// `unpack_word_ref` of every one of the 65 536 wire words (8 nibbles of `o`).
    otab: Vec<u32>,
}

impl Oracle {
    /// Build the table and self-check K-E0 over the whole wire-word domain: SWAR == reference unpack for every
    /// `(idx, sign7, shift)`, `v = 2o+1` odd in `[-11, 11]`, `o` in `[-6, 5]`, `decode8_lanes` == `decode8` halves.
    pub fn new() -> Result<(Oracle, Value)> {
        let (lut, mask) = (e8p::build_lut(), e8p::build_mask());
        let mut otab = vec![0u32; 65536];
        let (mut swar_bad, mut range_bad, mut lanes_bad) = (0u64, 0u64, 0u64);
        for w in 0..=65535u32 {
            let (idx, s7, sh) = e8p::unpack_word_fields(w as u16);
            let r = e8p::unpack_word_ref(idx, s7, sh);
            swar_bad += u64::from(r != e8p::unpack_word_swar(&lut, &mask, idx, s7, sh));
            let v = e8p::decode8(idx, s7, sh);
            for (k, &vk) in v.iter().enumerate() {
                let o = (((r >> (4 * k)) & 0xf) as u8) << 4;
                let o = (o as i8) >> 4;
                range_bad += u64::from(
                    !(-6..=5).contains(&o)
                        || 2 * i32::from(o) + 1 != i32::from(vk)
                        || vk % 2 == 0
                        || !(-11..=11).contains(&vk),
                );
            }
            for half in 0..2 {
                lanes_bad += u64::from(
                    e8p::decode8_lanes(idx, s7, sh, half)[..] != v[4 * half..4 * half + 4],
                );
            }
            otab[w as usize] = r;
        }
        if swar_bad + range_bad + lanes_bad != 0 {
            return Err(format!("K-E0 self-check failed over the 65536 wire words: swar {swar_bad} range {range_bad} lanes {lanes_bad}"));
        }
        let chk = json!({"wire_words": 65536, "swar_vs_ref_mismatches": swar_bad, "v_odd_in_11_and_o_in_6_5_violations": range_bad, "decode8_lanes_vs_decode8_mismatches": lanes_bad,
            "lut_plus_mask_bytes": lut.len() * 4 + mask.len() * 4, "magic": format!("{:#010x}", e8p::MAGIC), "books_sha256": e8p::E8P_BOOKS_SHA256});
        Ok((Oracle { otab }, chk))
    }

    /// `[m*k]` signed IU4 operands `o` (row-major) of one projection's blob.
    pub fn decode_o(&self, proj: Proj, blob: &[u8]) -> Vec<i8> {
        let words = &blob[..proj.e8p_words_bytes()];
        let mut o = vec![0i8; proj.m() * proj.k()];
        for (wi, c) in words.chunks_exact(2).enumerate() {
            let n = self.otab[u16::from_le_bytes([c[0], c[1]]) as usize];
            for lane in 0..8 {
                o[wi * 8 + lane] = ((((n >> (4 * lane)) & 0xf) as u8) << 4) as i8 >> 4;
            }
        }
        o
    }

    /// F32 sums of x rows `xrows` (token-major A4 blocks `x_tok`, `groups` blocks per row) against one expert's blob:
    /// `result[i]` is the `[m]` row of `xrows[i]`.
    pub fn dense_rows(
        &self,
        proj: Proj,
        blob: &[u8],
        x_tok: &[A4Block],
        xrows: &[usize],
    ) -> Vec<Vec<f32>> {
        let (m, k, groups) = (proj.m(), proj.k(), proj.groups());
        let o = self.decode_o(proj, blob);
        let sb: Vec<u16> = blob[proj.e8p_words_bytes()..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let xv: Vec<Vec<i8>> = xrows
            .iter()
            .map(|&xr| {
                (0..k)
                    .map(|j| x_tok[xr * groups + j / 128].element(j % 128))
                    .collect()
            })
            .collect();
        let cols: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|r| {
                xrows
                    .iter()
                    .enumerate()
                    .map(|(xi, &xr)| {
                        let blocks = &x_tok[xr * groups..(xr + 1) * groups];
                        let mut sum = 0f32;
                        for g in 0..groups {
                            let ow = &o[r * k + g * 128..r * k + (g + 1) * 128];
                            let xw = &xv[xi][g * 128..(g + 1) * 128];
                            let c: i32 = ow
                                .iter()
                                .zip(xw)
                                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                                .sum();
                            let a = e8p::MAGIC.wrapping_add(c as u32);
                            let folded = e8p::fold128(a, blocks[g].s);
                            sum = e8p::epoch_sum(blocks[g].d, sb[r * groups + g], folded, sum);
                        }
                        sum
                    })
                    .collect()
            })
            .collect();
        (0..xrows.len())
            .map(|xi| (0..m).map(|r| cols[r][xi]).collect())
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// activations (A4 blocks `[f32 d, i32 s, 64 nibble bytes]`)
// ---------------------------------------------------------------------------------------------------------------

/// Epoch-major device layout `[groups][rows][72]` of token-major blocks `[rows][groups]`.
pub fn epoch_major(tok: &[A4Block], rows: usize, groups: usize) -> Vec<u8> {
    let mut out = vec![0u8; rows * groups * BLK];
    for g in 0..groups {
        for r in 0..rows {
            let b = &tok[r * groups + g];
            let o = (g * rows + r) * BLK;
            out[o..o + 4].copy_from_slice(&b.d.to_le_bytes());
            out[o + 4..o + 8].copy_from_slice(&b.s.to_le_bytes());
            out[o + 8..o + BLK].copy_from_slice(&b.qs);
        }
    }
    out
}

/// Parity activations: extreme / zero / tiny / huge activation scales, extreme signed nibbles.
pub fn make_x_stress(groups: usize, rows: usize, seed: u64) -> (Vec<u8>, Vec<A4Block>) {
    let mut rng = Rng(seed);
    let ds = [1.0f32, 0.5, 1.0 / 64.0, 1.0e-3, 3.0, 0.0, 65536.0, 1.0e-20];
    let mut tok = Vec::with_capacity(rows * groups);
    for r in 0..rows {
        for g in 0..groups {
            let mut qs = [0u8; 64];
            let mut s = 0i32;
            for j in 0..128 {
                let v: i8 = match (r + g) % 9 {
                    0 => -8,
                    1 => 7,
                    2 => {
                        if j % 2 == 0 {
                            -8
                        } else {
                            7
                        }
                    }
                    _ => rng.below(16) as i8 - 8,
                };
                qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
                s += i32::from(v);
            }
            tok.push(A4Block {
                d: ds[(r * 3 + g) % ds.len()],
                s,
                qs,
            });
        }
    }
    (epoch_major(&tok, rows, groups), tok)
}

/// Bounded activations: row `r` is a pure function of `(key, r)`; scales `(1..=16)/1024`, triangular signed nibbles.
pub fn bounded_tok(groups: usize, rows: usize, key: u64) -> Vec<A4Block> {
    (0..rows)
        .into_par_iter()
        .flat_map_iter(|r| {
            let mut rng =
                Rng(key ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA4A4_0000_0000_0000);
            (0..groups).map(move |_| {
                let mut rng2 = Rng(rng.next());
                let d = (1 + rng2.below(16)) as f32 / 1024.0;
                let mut qs = [0u8; 64];
                let mut s = 0i32;
                for j in 0..128 {
                    let v = (rng2.below(9) + rng2.below(8)) as i8 - 8;
                    qs[j / 2] |= ((v as u8) & 0xf) << (4 * (j % 2));
                    s += i32::from(v);
                }
                A4Block { d, s, qs }
            })
        })
        .collect()
}

pub fn tok_bytes(tok: &[A4Block]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tok.len() * BLK);
    for b in tok {
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(&b.s.to_le_bytes());
        out.extend_from_slice(&b.qs);
    }
    out
}

pub fn tok_sha(tok: &[A4Block]) -> String {
    let mut h = Sha256::new();
    for b in tok {
        h.update(b.d.to_le_bytes());
        h.update(b.s.to_le_bytes());
        h.update(b.qs);
    }
    hex(&h.finalize())
}

/// Deterministic bounded synthetic A4 blocks in epoch-major layout for timing runs (values are not routed
/// activations; the real route fixes only which rows each expert sees).
pub fn gen_x_timing(groups: usize, rows: usize, key: u64) -> Vec<u8> {
    let mut out = vec![0u8; groups * rows * BLK];
    out.par_chunks_mut(BLK).enumerate().for_each(|(i, blk)| {
        let mut rng =
            Rng(key ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA4A4_0000_0000_0000);
        let d = (1 + rng.next() % 16) as f32 / 1024.0;
        let mut s = 0i32;
        for w in 0..8 {
            let v = rng.next();
            blk[8 + w * 8..16 + w * 8].copy_from_slice(&v.to_le_bytes());
        }
        for b in &blk[8..] {
            for nib in [*b & 0xf, *b >> 4] {
                s += i32::from(((nib as i8) << 4) >> 4);
            }
        }
        blk[0..4].copy_from_slice(&d.to_le_bytes());
        blk[4..8].copy_from_slice(&s.to_le_bytes());
    });
    out
}
