// SPDX-License-Identifier: Apache-2.0
//! E8P-I4 additions of the gfx1100 K2 G1 oracle (`--codec e8p`, `qwen4_moe_{gate_up_silu,down}_iu4_sym_e8p_pm_gfx1100[_acc]`).
//!
//! Wire layout (frozen K-E0, `hipfire_runtime::e8p_i4_reference`): per expert tensor `[M, K]` the planar bytes
//! `[M*K/8 u16 LE words][M*K/128 f16 LE scales]`; word `c` of a row covers weights `8c..8c+8`
//! (`idx | sign7 << 8 | shift << 15`), scale `i` is the FP16 `d` of flat K128 group `i`.
//!
//! Two CPU folds, both built only from the frozen K-E0 pieces plus `decode8`:
//!  * the TWIN (`twin = true`, the expected value of every device comparison): per K128 epoch
//!    `C = sum o*q` over the `unpack_word_ref` nibbles `o`, `a = MAGIC + C`, `fold128(a, s)` with the activation record's
//!    own `s`, then `epoch_sum(d_act, d, a', sum)` from `+0` over ascending epochs;
//!  * the independent MODEL (`twin = false`): `T = sum v*q` straight from `decode8` (`v = 4*codeword`) and the raw
//!    activation nibbles, `sum = fma(RN(d_act*f32(d)), T*0.25, sum)`. Mutations (`wrong-lut` = LUT row `idx ^ 1`,
//!    `wrong-sign-mask` = MASK entry `sign7 ^ 1`, `wrong-fold-epoch` = activation sum `s` dropped in every epoch so the term is
//!    `2C/4` instead of `(2C+s)/4`, wrong scale row, reversed epochs) apply to the model only. The twin and
//!    the model must agree on every scenario (self-check), which re-proves `2C + s == T` on the data actually used.
//!
//! Synthetic generators cover every wire word (`EveryCode`: a bijective per-65536-block scramble puts all 65536
//! `(idx, sign7, shift)` words, both parities of abs rows and sign counts and both shifts, into each expert), the extreme
//! `|v|` words (largest / smallest `sum v`, both shifts), f16 max / subnormal / zero / low-normal scales (kernel arithmetic
//! domain; `guard_trips` records how many fall outside the writer guard), scale-plane sentinels, and targeted exact dots
//! (`T` odd by construction, scale `d = 2^(a+2)` so the F32 sum equals the requested BF16 tie / signed-zero / row-id value).

use crate::model::*;
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4_reference::A4Block;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;

pub const E8P_WORDS: usize = 65536;

/// Bijective scramble of the 65536 wire words: block `b` of 65536 positions visits every word exactly once.
pub fn scramble_word(i: usize) -> u16 {
    let (b, j) = ((i >> 16) as u32, (i & 0xffff) as u32);
    let mul = 40503 + 2 * b * 977; // odd
    ((j.wrapping_mul(mul)).wrapping_add(b.wrapping_mul(12345)) & 0xffff) as u16
}

fn v_of(word: u16) -> [i8; 8] {
    let (idx, s7, sh) = e8p::unpack_word_fields(word);
    e8p::decode8(idx, s7, sh)
}

/// `(word with the largest sum v, word with the smallest sum v)` over all 65536 words (first on ties).
pub fn extreme_words() -> (u16, u16) {
    let (mut best_hi, mut best_lo) = ((i32::MIN, 0u16), (i32::MAX, 0u16));
    for w in 0..=u16::MAX {
        let s: i32 = v_of(w).iter().map(|&v| i32::from(v)).sum();
        if s > best_hi.0 {
            best_hi = (s, w);
        }
        if s < best_lo.0 {
            best_lo = (s, w);
        }
    }
    (best_hi.1, best_lo.1)
}

/// FP16 bits of `2^e` (normal range).
fn pow2_f16(e: i32) -> u16 {
    debug_assert!((-14..=15).contains(&e));
    ((15 + e) as u16) << 10
}

/// Requested F32 dot `t` (`sum = t` at `d_act = 1`) -> `(odd integer T = sum v*q, scale bits d)`; `None` for 0 (zero scale).
fn tie_target(t: i32) -> Option<(i32, u16)> {
    if t == 0 {
        return None;
    }
    let a = t.unsigned_abs().trailing_zeros() as i32;
    Some((t >> a, pow2_f16(a + 2)))
}

/// Overflow scenario targets (`d_act = FLT_MAX`, `d = 1.0`, `sum = FLT_MAX*T/4`): finite, finite, rounding to +-inf, zero scale.
fn overflow_target(idx: usize) -> Option<(i32, u16)> {
    const T: [i32; 6] = [1, 5, -1, -5, 0, 3];
    let t = T[idx % 6];
    if t == 0 {
        None
    } else {
        Some((t, 0x3c00))
    }
}

/// Reachability over the 16 words of one epoch against a fixed activation block: bitset DP over distinct per-word
/// contributions, so a requested odd `T = sum v*q` (or the nearest reachable one) is solved with randomly chosen words.
pub struct E8pSolver {
    by_pos: Vec<Vec<(i32, Vec<u16>)>>,
    reach: Vec<Vec<u64>>,
}

const OFF: i32 = 11264 + 64;
const WIDTH: usize = (2 * OFF + 1) as usize;
const NW: usize = (WIDTH + 63) / 64;

fn or_shift(dst: &mut [u64], src: &[u64], c: i32) {
    let (ws, bs) = ((c.unsigned_abs() / 64) as usize, c.unsigned_abs() % 64);
    if c >= 0 {
        for i in ws..NW {
            let mut v = src[i - ws] << bs;
            if bs > 0 && i - ws >= 1 {
                v |= src[i - ws - 1] >> (64 - bs);
            }
            dst[i] |= v;
        }
    } else {
        for i in 0..NW {
            if i + ws >= NW {
                break;
            }
            let mut v = src[i + ws] >> bs;
            if bs > 0 && i + ws + 1 < NW {
                v |= src[i + ws + 1] << (64 - bs);
            }
            dst[i] |= v;
        }
    }
}

fn bit(b: &[u64], i: i32) -> bool {
    i >= 0 && (i as usize) < WIDTH && (b[i as usize / 64] >> (i as usize % 64)) & 1 == 1
}

impl E8pSolver {
    pub fn new(x: &[i8; 128]) -> Self {
        let vs: Vec<[i8; 8]> = (0..E8P_WORDS).map(|w| v_of(w as u16)).collect();
        let mut by_pos = Vec::with_capacity(16);
        for k in 0..16 {
            let mut m: HashMap<i32, Vec<u16>> = HashMap::new();
            for (w, v) in vs.iter().enumerate() {
                let c: i32 = (0..8)
                    .map(|l| i32::from(v[l]) * i32::from(x[8 * k + l]))
                    .sum();
                m.entry(c).or_default().push(w as u16);
            }
            let mut l: Vec<(i32, Vec<u16>)> = m.into_iter().collect();
            l.sort_by_key(|e| e.0);
            by_pos.push(l);
        }
        let mut reach = vec![vec![0u64; NW]; 17];
        reach[0][OFF as usize / 64] |= 1u64 << (OFF as usize % 64);
        for k in 0..16 {
            let (a, b) = reach.split_at_mut(k + 1);
            for (c, _) in &by_pos[k] {
                or_shift(&mut b[0], &a[k], *c);
            }
        }
        Self { by_pos, reach }
    }

    fn nearest(&self, target: i32) -> i32 {
        for d in 0..=2 * OFF {
            for cand in [target + d, target - d] {
                if bit(&self.reach[16], cand + OFF) {
                    return cand;
                }
            }
        }
        0
    }

    /// 16 words whose `sum v*q` is `target` (or the nearest reachable value), and the achieved value.
    pub fn solve(&self, target: i32, rng: &mut Rng) -> (Vec<u16>, i32) {
        let t = self.nearest(target);
        let mut s = t + OFF;
        let mut out = vec![0u16; 16];
        for k in (0..16).rev() {
            let opts: Vec<&(i32, Vec<u16>)> = self.by_pos[k]
                .iter()
                .filter(|(c, _)| bit(&self.reach[k], s - c))
                .collect();
            let (c, ws) = opts[rng.below(opts.len() as u32) as usize];
            out[k] = ws[rng.below(ws.len() as u32) as usize];
            s -= c;
        }
        (out, t)
    }
}

/// Synthetic E8P expert. `gen` semantics follow the MQ generators; weights are wire words (see the module doc).
pub fn gen_expert_e8p(kind: Kind, gen: Gen, seed: u32) -> Res<Expert> {
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let wpr = k / 8;
    let mut rng = Rng(seed | 1);
    let mut words: Vec<u16> = (0..m * wpr).map(|_| (rng.next() >> 7) as u16).collect();
    let mut scales: Vec<u16> = (0..m * g)
        .map(|_| SCALE_POOL[rng.below(SCALE_POOL.len() as u32) as usize])
        .collect();
    match gen {
        Gen::Random => {}
        Gen::EveryCode => {
            for (i, w) in words.iter_mut().enumerate() {
                *w = scramble_word(i);
            }
        }
        Gen::Extreme => {
            let (pos, neg) = extreme_words();
            for (i, w) in words.iter_mut().enumerate() {
                *w = match (i / 3 + i / wpr) % 4 {
                    0 => pos,
                    1 => neg,
                    2 => pos ^ 0x8000,
                    _ => neg ^ 0x8000,
                };
            }
            for (i, s) in scales.iter_mut().enumerate() {
                *s = [0x7bff, 0x0001, 0x0400, 0x3c00, 0x03ff, 0x0401, 0x0403][i % 7];
            }
            // zero-scale groups
            for r in (0..m).step_by(97) {
                for gg in 0..g {
                    if gg % 5 == 0 {
                        scales[r * g + gg] = 0;
                    }
                }
            }
        }
        Gen::MaxDot => {
            let (pos, neg) = extreme_words();
            for r in 0..m {
                let w = if r % 2 == 0 { pos } else { neg };
                for j in 0..wpr {
                    words[r * wpr + j] = w;
                }
            }
        }
        Gen::Sentinel(first) => {
            scales[0] = first;
        }
        Gen::Targeted(var) => {
            let solver = E8pSolver::new(&pattern_p());
            let oc = kind.out_cols();
            for r in 0..m {
                let idx = if kind == Kind::GateUp { r % oc } else { r };
                let tgt = if var == Var::Overflow {
                    overflow_target(idx)
                } else {
                    tie_target(target_for(var, kind, r))
                };
                match tgt {
                    None => scales[r * g] = 0,
                    Some((t, d)) => {
                        let (w0, _achieved) = solver.solve(t, &mut rng);
                        words[r * wpr..r * wpr + 16].copy_from_slice(&w0);
                        scales[r * g] = d;
                    }
                }
            }
        }
    }
    Expert::from_e8p(
        kind,
        &words,
        &scales,
        format!("synthetic:{gen:?}:seed={seed}"),
    )
}

/// Mutation / model switches of the E8P folds (the twin ignores them).
fn decode_row(words: &[u16], o: FoldOpts, twin: bool) -> Vec<i8> {
    let mut out = Vec::with_capacity(words.len() * 8);
    for &w in words {
        let (mut idx, mut s7, sh) = e8p::unpack_word_fields(w);
        if o.lut_xor1 {
            idx ^= 1;
        }
        if o.mask_xor1 {
            s7 ^= 1;
        }
        if twin {
            let n = e8p::unpack_word_ref(idx, s7, sh);
            for l in 0..8 {
                out.push((((n >> (4 * l)) & 0xf) as u8 as i8) << 4 >> 4);
            }
        } else {
            out.extend_from_slice(&e8p::decode8(idx, s7, sh));
        }
    }
    out
}

/// E8P fold over `xrows` of `x`; see the module doc for `twin` (K-E0 pieces) vs model (decode8 + fma).
pub fn fold_e8p(
    kind: Kind,
    e: &Expert,
    xrows: &[usize],
    x: &XSet,
    o: FoldOpts,
    twin: bool,
) -> Vec<Vec<f32>> {
    let (m, k, g) = (kind.m(), kind.k(), kind.groups());
    let (wpr, n) = (k / 8, xrows.len());
    let mut xs = vec![0i8; n * g * 128];
    let mut xd = vec![0f32; n * g];
    let mut xsum = vec![0i32; n * g];
    for (jj, &j) in xrows.iter().enumerate() {
        for h in 0..g {
            let b: &A4Block = &x.tok[j * g + h];
            xd[jj * g + h] = b.d;
            xsum[jj * g + h] = b.s;
            for i in 0..128 {
                xs[(jj * g + h) * 128 + i] = b.element(i);
            }
        }
    }
    let per_row: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map(|r| {
            let mut sums = vec![0f32; n];
            let order: Vec<usize> = if o.reverse_epochs {
                (0..g).rev().collect()
            } else {
                (0..g).collect()
            };
            let sr = if o.scale_row_xor1 { r ^ 1 } else { r };
            let plane = decode_row(&e.words[r * wpr..(r + 1) * wpr], o, twin);
            for &h in &order {
                let d = e.scales[sr * g + h];
                let wrow = &plane[h * 128..h * 128 + 128];
                for jj in 0..n {
                    let xv = &xs[(jj * g + h) * 128..(jj * g + h) * 128 + 128];
                    let c: i32 = (0..128)
                        .map(|i| i32::from(wrow[i]) * i32::from(xv[i]))
                        .sum();
                    sums[jj] = if twin {
                        let folded =
                            e8p::fold128(e8p::MAGIC.wrapping_add(c as u32), xsum[jj * g + h]);
                        e8p::epoch_sum(xd[jj * g + h], d, folded, sums[jj])
                    } else {
                        let scale = xd[jj * g + h] * f16f(d);
                        let t = if o.drop_fold_sum {
                            c - xsum[jj * g + h]
                        } else {
                            c
                        };
                        scale.mul_add((t as f32) * 0.25, sums[jj])
                    };
                }
            }
            sums
        })
        .collect();
    (0..n)
        .map(|jj| (0..m).map(|r| per_row[r][jj]).collect())
        .collect()
}

/// Per-expert word coverage and arithmetic-domain statistics for the receipt.
pub fn word_stats(e: &Expert) -> Value {
    let mut seen = vec![false; E8P_WORDS];
    let (mut odd_rows, mut even_rows, mut sh1, mut pc_odd) = (0u64, 0u64, 0u64, 0u64);
    let mut idx_seen = [false; 256];
    for &w in &e.words {
        seen[w as usize] = true;
    }
    for w in 0..E8P_WORDS {
        if seen[w] {
            let (idx, s7, sh) = e8p::unpack_word_fields(w as u16);
            idx_seen[idx as usize] = true;
            if e8p::abs_row_parity(idx) == 1 {
                odd_rows += 1;
            } else {
                even_rows += 1;
            }
            sh1 += u64::from(sh);
            pc_odd += u64::from(s7.count_ones() & 1);
        }
    }
    let mut cls = std::collections::BTreeMap::new();
    for &d in &e.scales {
        let c = if d == 0 {
            "zero"
        } else if d & 0x8000 != 0 {
            "negative"
        } else if d & 0x7c00 == 0x7c00 {
            "non_finite"
        } else if d & 0x7c00 == 0 {
            "subnormal"
        } else if d == 0x7bff {
            "f16_max"
        } else {
            "normal"
        };
        *cls.entry(c).or_insert(0u64) += 1;
    }
    json!({"distinct_words": seen.iter().filter(|&&s| s).count(), "distinct_abs_rows": idx_seen.iter().filter(|&&s| s).count(),
        "words_on_odd_abs_rows": odd_rows, "words_on_even_abs_rows": even_rows, "words_shift_set": sh1, "words_odd_popcount_sign7": pc_odd,
        "scale_classes": cls, "guard_trips": e.guard_trips})
}

/// Reason strings for coverage the scenario promises but the experts did not reach.
pub fn word_gaps(sc_name: &str, e: &Expert) -> Vec<String> {
    let mut v = Vec::new();
    if e.codec == Codec::E8p && e.origin.starts_with("synthetic:EveryCode") {
        let d = word_stats(e)["distinct_words"].as_u64().unwrap_or(0);
        if d as usize != E8P_WORDS {
            v.push(format!(
                "{sc_name}: EveryCode expert reaches {d} of {E8P_WORDS} E8P wire words"
            ));
        }
    }
    v
}
