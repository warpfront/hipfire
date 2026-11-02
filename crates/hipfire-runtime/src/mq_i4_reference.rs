// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU byte/arithmetic oracles ("reference twins") for the SubQ4 MQ-I4 expert
//! kernels (HFQ qt=131..=134, see [`crate::mq_i4`]).
//!
//! Nothing here dispatches or is used by a default path: these are the frozen
//! arithmetic definitions future GPU kernels are compared against. Only `std`,
//! [`crate::mq_i4`] and `half` are used; the planar wire format is **not**
//! re-implemented here (`mq_i4::decode_tensor` / `mq_i4::tensor_bytes` are the
//! only codec entry points used).
//!
//! # Panic / validation policy
//!
//! Every function panics with a message naming the function and the failing
//! quantity on an invalid shape or buffer. Two input forms exist:
//!
//! * **Packed** (`bytes`): the tensor is validated *and* decoded with
//!   [`mq_i4::decode_tensor`] first (S0 import validation: scales finite and
//!   non-negative, `+0` scale only with midpoint codes, codes `<= L`, exact
//!   byte length), then the `_parts` code path runs.
//! * **Unpacked** (`*_parts`, `codes` = one `u8` per weight row-major
//!   `[M, K]`, `scale_bits` = f16 bit pattern of each `[M, K/128]` scale):
//!   REQUIRES that the codes/scales already passed S0 `encode_tensor` /
//!   `decode_tensor` validation. Only the shape (via [`mq_i4::tensor_bytes`])
//!   and the two lengths are asserted; nothing is re-encoded or allocated to
//!   validate. A code `> L` still panics inside [`weight_i8`] (it cannot be
//!   interpreted), but scale-finiteness / zero-group rules are the caller's
//!   precondition.
//!
//! # Weight value
//!
//! Rotated-basis dequant is `w = RN(float(d) * float(2c - L))`, `L = 2^b - 1`
//! ([`weight_i8`] gives the integer `2c - L`; [`weight_nibble`] its
//! two's-complement low nibble, `-8..=7`, **no XOR rebias** — the weights are
//! already symmetric). `float(d)` is `half::f16::from_bits(..).to_f32()`
//! (exact).
//!
//! # IU4 GEMM twin ([`iu4_gemm_f32`])
//!
//! Twin of the symmetric MQ4V2 x `block_i4_128` IU4 fold (WMMA
//! `v_wmma_i32_16x16x16_iu4` / `x32_iu4` chain, one K128 epoch per `g`):
//!
//! ```text
//! out[t*m + r]  (t < rows = x_a4.len() / (k/128), rows == 0 allowed)
//!   sum = +0.0f32
//!   for g in 0 .. k/128                       // ascending, K128 fold
//!       dot   = sum_{j in 0..128} (2*c[r, g*128+j] - L) * a4[t*(k/128)+g].element(j)
//!                                           // exact int32, any order; |dot| <= 7168
//!       scale = RN(a4.d * float(d[r,g]))      // one f32 multiply
//!       sum   = fma(scale, float(dot), sum)   // single rounding
//!   out = sum
//! ```
//!
//! `|dot| <= 128 * 7 * 8 = 7168` (W3 weights `|2c-7| <= 7`, activations in
//! `-8..=7`), far below the `2^22` bound that makes the device's magic-number
//! int->float conversion exact. The activation's `s` field (row sum) is not
//! used because the weights are symmetric.
//!
//! Source: `crates/hipfire-isa/src/kernels/iu4_fold.rs:7-13,25-52` (fold DAG,
//! `block_i4_128` = `[f32 d, i32 s, 64 nibble bytes]`, 72 B);
//! `iu4_gemm/spec.rs:9-30` (K128 fold axis, Token layout: one 72-byte record
//! per token and K128 epoch; line 94 is the generated-module naming only).
//! Activation producer (A4Block layout; line numbers read in this worktree,
//! the frozen note's 791-798/817-825/1090-1106 are shifted):
//! `iu4_v2b_a4.rs:51-62` (lane `l` holds features `8l..8l+7`, `{5,7}`
//! `block_i4_128` emit), `:817-824` (record `72*(...)`, qs word at byte
//! `8 + 4*(lane&15)`), `:1087-1106` (codes -> exact row sum `s`, nibble `k`
//! of the word = element `8*lane + k`, stores of the qs word and `(d, s)`).
//! Hence [`A4Block::element`]: element `j` = byte `j/2`, nibble `4*(j%2)`
//! (low nibble first); one producer lane owns 8 contiguous elements = 4 bytes
//! = `qs[4*lane .. 4*lane+4]`.
//!
//! # F4 decode GEMV tree twin ([`decode_gemv_f32`], [`decode_moe_slots`])
//!
//! The decode GEMV is a 32-lane wave per output row. This is the frozen
//! *minimum G128 adaptation* of the shipped G256 gate/up kernel (each lane
//! there owns 8 contiguous elements of every 256-group; here 4 contiguous
//! elements of every 128-group, as in the down kernel). The tree is selected
//! by `profile.rotation` (never by `K`): `R256` (gate/up) uses four
//! accumulator chains, `R128` (down) one.
//!
//! ```text
//! for output row r:
//!   for lane in 0..32:
//!     acc[4] = {+0,+0,+0,+0}
//!     for g in 0..k/128:                              // ascending
//!       for i in 0..4:
//!         j_i = g*128 + 4*lane + i
//!         w_i = RN(float(d[r,g]) * float(2*c[r,j_i] - L))
//!       p = RN(w_0 * x[j_0])                         // first a MUL
//!       for i in [1,2,3]: p = fma(w_i, x[j_i], p)   // then 3 left FMAs
//!       if R256: acc[g & 3] = RN(acc[g & 3] + p)
//!       if R128: acc[0] = RN(acc[0] + p)
//!     if R256: v[lane] = RN(RN(acc0 + acc1) + RN(acc2 + acc3))
//!     if R128: v[lane] = acc[0]                     // NO four-way fold
//!   for off in [16,8,4,2,1]:                         // snapshot shuffle-down
//!     next[l] = RN(v[l] + v[l+off]) if l+off < 32, else v[l]
//!     v = next
//!   y[r] = v[0]
//! ```
//!
//! (A real `__shfl_down` returns the lane's own value when `l + off >= 32`
//! and the kernel adds it anyway; lane 0's result never depends on those
//! lanes, so leaving them unchanged is lane-0 exact.)
//!
//! Sources: `kernels/src/qwen4_gemv_mq4g256v2_moe_gate_up_k8_indexed_batched.hip`
//! `:83-98` (four chains `acc[g & 3]`, fold `(acc0+acc1)+(acc2+acc3)`),
//! `:122-180` (lane ownership, DOG expression), `:234-248` (fold, `__shfl_down`
//! 16..1, lane-0 scatter; `:4-7` says the top-10 alias compiles this file);
//! `kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip`
//! `:43-60` (four contiguous elements per lane per group, one `acc +=`
//! per group), `:87-90` (shuffle, lane-0 scatter).
//!
//! **Contraction choice (required reading for PM authors).** The HIP source
//! has plain arithmetic expressions, no explicit `fmaf`; hipcc is built with
//! `-O3` and no explicit fast-math (`rdna-compute/src/compiler.rs:397-404`;
//! `-ffp-contract=off` is applied only to specific other modules,
//! `compiler.rs:705`). The oracle therefore *freezes a source-structure
//! interpretation*: left-associated `p0+p1+p2+p3` is contracted as
//! `fma(w3,x3,fma(w2,x2,fma(w1,x1, w0*x0)))` (first MUL, three left FMAs, no
//! reassociation), and the following `acc += sum` stays a separate rounded
//! add. This is **not** a claim that any compiled code object measured
//! bit-identical to this tree; a PM-authored kernel that implements exactly
//! this DAG matches the oracle, and any other contraction is a different
//! kernel with a different oracle.
//!
//! # `decode_moe_slots` bucket contract
//!
//! One call = one bucket of experts. `expert_ids[u]` owns
//! `slot_ids[expert_offsets[u] .. expert_offsets[u+1]]`
//! (`expert_offsets.len() == U + 1`, first `0`, last `slot_ids.len()`,
//! non-decreasing; empty buckets and empty segments are legal; every listed
//! expert must have a table). `slot_ids` are unique values in `0 .. 10*c`
//! (`c` tokens, top-10): omitted slots and any padding in the output buffers
//! are left untouched. Gate/up: `m` even, every expert `R256`, input row
//! `slot/10`, gate rows `0..m/2` -> `y0[slot*y_slot_stride + r]`, up rows
//! `m/2..m` -> `y1[slot*y_slot_stride + r - m/2]` (`y1` required). Down:
//! every expert `R128`, input row `slot`, rows `0..m` -> `y0` (`y1` must be
//! `None`). The result is unweighted (route weights belong to the combine
//! stage). Experts may use different codecs (W3/W2); nothing assumes a
//! uniform expert stride. Lengths, strides, slot ids and expert tables are
//! validated before the first write; the packed form decodes each distinct
//! expert once.
//!
//! # BF16 grouped-prefill epilogue ([`grouped_prefill_epilogue`])
//!
//! Twin of the GateUp/Down epilogue of the qwen4 grouped MoE kernel
//! (`hipfire-isa/src/kernels/qwen4_moe_sym.rs:522-532` BF16 gate/up -> SiLU
//! -> BF16 -> pad; `:538-568` bf16 packing/store). `sums` is row-major
//! `[slot_ids.len(), m]` f32 accumulators; output is row-major `u16` BF16
//! bits `[slot_ids.len(), m/2]` (gate/up) or `[.., m]` (down).
//!
//! ```text
//! f32(bits) means reinterpret IEEE bits, NOT an integer-to-float conversion.
//! round(u32 x) = nonfinite(x) ? x                  // full 32 bits kept
//!              : wrapping_add(x, 0x7fff + ((x >> 16) & 1)) & 0xffff0000
//! row with slot_id == -1 (pad):  output = +0 bits; sums/SiLU never read
//! gate/up row, i in 0..m/2:
//!   g = widen(round(sums[row*m + i]))      // finite: low16 cleared; else as is
//!   u = widen(round(sums[row*m + m/2 + i]))
//!   (SiLU DAG, gemm_uk/epilogue.rs silu_dense, DivNum::Gate / Finish::MulUp)
//!   ph = RN(f32(0xbfb8aa3b) * g)
//!   pl = fma(f32(0xbfb8aa3b), g, -ph);  pl = fma(f32(0xb2a5705f), g, pl)
//!   e  = rndne(ph);  a = RN(RN(ph - e) + pl)
//!   r  = ldexp(exp2(a), int(e))
//!   if f32(0x42ce8ed0) < g { r = 0 }       // ordered compare; NaN keeps r
//!   if g < f32(0xc2b17218) { r = +inf }
//!   den = RN(1 + r);  q = divide(g, den) (IEEE RN);  h = RN(q * u)
//!   out = round(h) >> 16
//! down row, i in 0..m: out = round(sums[row*m + i]) >> 16
//! ```
//!
//! Sources: `qwen4_moe_sym.rs:522-532,538-568`;
//! `gemm_uk/epilogue.rs:57-108` (precise SiLU DAG, steps `:65-99`);
//! `bf16.rs:18-21,47-53` (`rne_finite_passthrough`: `clear_low` only applies
//! to the finite candidate; a non-finite value bypasses the AND and keeps its
//! full original bits including NaN payload).
//!
//! **Portable epilogue vs device intrinsics (honesty).** `exp2` is the device
//! `v_exp_f32`, `ldexp` is `v_ldexp_f32`, and the divide is the full IEEE
//! sequence `v_div_scale_f32` x2, `v_rcp_f32`, fma refinement, `v_div_fmas_f32`,
//! `v_div_fixup_f32` (`epilogue.rs:83-96`). Those results are
//! hardware-defined (the reciprocal/exp seeds are not correctly rounded
//! functions that Rust reproduces). [`PortableEpilogue`] uses `f32::exp2`,
//! an f64-exact power-of-two scaling `ldexp` and Rust's `/`; that is a
//! **portable math reference, NOT a device byte oracle**. A byte oracle is
//! obtained only by calling [`grouped_prefill_epilogue_with`] with an
//! [`EpilogueIntrinsics`] whose three hooks return actual ISA intrinsic
//! results (e.g. captured from hardware). There is deliberately no fake
//! fallback.
//!
//! The `divide(g, den)` hook denotes this instruction DAG, in order (the
//! `div_*` operations include ISA-defined scaling and exceptional-value
//! behavior, not ordinary multiply/add substitutes):
//! ```text
//! ds = div_scale(den, den, g).value
//! (ns, mask) = div_scale(g, den, g)
//! rr = rcp(ds)
//! er = fma(-ds, rr, 1)
//! rr = fma(er, rr, rr)
//! q0 = RN(ns * rr)
//! er = fma(-ds, q0, ns)
//! q1 = fma(er, rr, q0)
//! er = fma(-ds, q1, ns)
//! q2 = div_fmas(er, rr, q1, mask)
//! q  = div_fixup(q2, den, g)
//! ```
//!
//! # f64 accuracy reference ([`gemv_f64`])
//!
//! `y[r] = sum over ascending columns j of (float(d) * (2c-L)) * x[j]`, with
//! every multiply and add a separate f64 operation starting at `+0.0`. Each
//! product is exact in f64 (`<= 11 + 3 + 24` significant bits) but the sum is
//! rounded per add: this is an accuracy reference for judging f32 trees, not
//! a mathematically exact arbitrary-real sum.

use crate::mq_i4::{self, MqI4Codec, MqI4Profile, MqI4Rotation, MqI4Tensor};
use half::f16;

const WAVE: usize = 32;
const GROUP: usize = mq_i4::GROUP;

// --------------------------------------------------------------------------
// Weight values
// --------------------------------------------------------------------------

/// The symmetric integer weight `2c - L` of code `c`. Panics if `c > L`.
pub fn weight_i8(codec: MqI4Codec, c: u8) -> i8 {
    let levels = codec.levels();
    assert!(
        c <= levels,
        "weight_i8: code {c} exceeds L={levels} of {codec:?}"
    );
    2 * (c as i8) - levels as i8
}

/// Two's-complement low nibble of [`weight_i8`] (`-8..=7`; no XOR rebias).
pub fn weight_nibble(codec: MqI4Codec, c: u8) -> u8 {
    (weight_i8(codec, c) as u8) & 0xF
}

#[inline]
fn scale_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

// --------------------------------------------------------------------------
// Shape / buffer checks
// --------------------------------------------------------------------------

fn check_parts(
    who: &str,
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
) {
    if let Err(e) = mq_i4::tensor_bytes(profile, m, k) {
        panic!("{who}: invalid {} shape [{m}, {k}]: {e}", profile.name());
    }
    assert_eq!(
        codes.len(),
        m * k,
        "{who}: codes length {} != m*k = {} for [{m}, {k}]",
        codes.len(),
        m * k
    );
    let groups = mq_i4::scale_count(m, k);
    assert_eq!(
        scale_bits.len(),
        groups,
        "{who}: scale_bits length {} != m*(k/128) = {groups} for [{m}, {k}]",
        scale_bits.len()
    );
}

fn decode_checked(who: &str, profile: MqI4Profile, m: usize, k: usize, bytes: &[u8]) -> MqI4Tensor {
    mq_i4::decode_tensor(profile, m, k, bytes).unwrap_or_else(|e| {
        panic!(
            "{who}: decode_tensor({}, [{m}, {k}], {} bytes) failed: {e}",
            profile.name(),
            bytes.len()
        )
    })
}

// --------------------------------------------------------------------------
// Dequant
// --------------------------------------------------------------------------

/// Rotated-basis dequant of a packed tensor: row-major `[m, k]`
/// `RN(float(d) * float(2c - L))`.
pub fn dequant_rotated_f32(profile: MqI4Profile, m: usize, k: usize, bytes: &[u8]) -> Vec<f32> {
    let t = decode_checked("dequant_rotated_f32", profile, m, k, bytes);
    dequant_rotated_f32_parts(profile, m, k, &t.codes, &t.scale_bits)
}

/// [`dequant_rotated_f32`] on already validated unpacked codes/scales.
pub fn dequant_rotated_f32_parts(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
) -> Vec<f32> {
    check_parts(
        "dequant_rotated_f32_parts",
        profile,
        m,
        k,
        codes,
        scale_bits,
    );
    let groups = k / GROUP;
    let mut out = Vec::with_capacity(m * k);
    for r in 0..m {
        for g in 0..groups {
            let d = scale_f32(scale_bits[r * groups + g]);
            let base = r * k + g * GROUP;
            for &c in &codes[base..base + GROUP] {
                out.push(d * weight_i8(profile.codec, c) as f32);
            }
        }
    }
    out
}

// --------------------------------------------------------------------------
// IU4 GEMM
// --------------------------------------------------------------------------

/// `block_i4_128`: `[f32 d, i32 s, 64 nibble bytes]` (72 bytes). Element `j`
/// is byte `j/2`, nibble `4*(j%2)` (low nibble first), a signed `-8..=7`
/// value. `s` is the exact element sum (unused by the symmetric fold).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct A4Block {
    pub d: f32,
    pub s: i32,
    pub qs: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<A4Block>() == 72);

impl A4Block {
    /// Signed value of element `j` (`j < 128`).
    pub fn element(&self, j: usize) -> i8 {
        assert!(j < GROUP, "A4Block::element: index {j} >= 128");
        let nibble = (self.qs[j / 2] >> (4 * (j % 2))) & 0xF;
        ((nibble << 4) as i8) >> 4
    }

    /// Build a block from 128 values in `-8..=7`; `s` is their sum.
    #[cfg(test)]
    pub fn from_values(d: f32, values: &[i8; 128]) -> Self {
        let mut qs = [0u8; 64];
        let mut s = 0i32;
        for (j, &v) in values.iter().enumerate() {
            assert!(
                (-8..=7).contains(&v),
                "A4Block::from_values: value {v} at {j} outside -8..=7"
            );
            qs[j / 2] |= ((v as u8) & 0xF) << (4 * (j % 2));
            s += i32::from(v);
        }
        Self { d, s, qs }
    }
}

/// IU4 fold twin on a packed tensor; see the module docs. `x_a4` holds
/// `rows * (k/128)` blocks, token-major (`x_a4[t*(k/128) + g]`); the result is
/// `out[t*m + r]`.
pub fn iu4_gemm_f32(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    bytes: &[u8],
    x_a4: &[A4Block],
) -> Vec<f32> {
    let t = decode_checked("iu4_gemm_f32", profile, m, k, bytes);
    iu4_gemm_f32_parts(profile, m, k, &t.codes, &t.scale_bits, x_a4)
}

/// [`iu4_gemm_f32`] on already validated unpacked codes/scales.
pub fn iu4_gemm_f32_parts(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
    x_a4: &[A4Block],
) -> Vec<f32> {
    check_parts("iu4_gemm_f32_parts", profile, m, k, codes, scale_bits);
    let groups = k / GROUP;
    assert_eq!(
        x_a4.len() % groups,
        0,
        "iu4_gemm_f32_parts: x_a4 length {} is not a multiple of k/128 = {groups}",
        x_a4.len()
    );
    let rows = x_a4.len() / groups;
    let mut out = vec![0.0f32; rows * m];
    for t in 0..rows {
        let blocks = &x_a4[t * groups..(t + 1) * groups];
        for r in 0..m {
            let mut sum = 0.0f32;
            for (g, blk) in blocks.iter().enumerate() {
                let base = r * k + g * GROUP;
                let mut dot = 0i32;
                for (j, &c) in codes[base..base + GROUP].iter().enumerate() {
                    dot += i32::from(weight_i8(profile.codec, c)) * i32::from(blk.element(j));
                }
                assert!(
                    dot.abs() <= 7168,
                    "iu4_gemm_f32_parts: |dot| = {} exceeds 7168 (row {r}, group {g})",
                    dot.abs()
                );
                let scale = blk.d * scale_f32(scale_bits[r * groups + g]);
                sum = scale.mul_add(dot as f32, sum);
            }
            out[t * m + r] = sum;
        }
    }
    out
}

// --------------------------------------------------------------------------
// F4 decode GEMV tree
// --------------------------------------------------------------------------

/// The frozen wave tree for one output row over an arbitrary weight source.
/// `weight(g, j)` is the f32 weight `RN(d_g * q_j)` of row column `j` (in
/// group `g`). This is the single definition of the tree; everything else
/// (`tree_row`, [`gemv_tree_f32_q`]) only supplies the weight source.
#[inline]
fn tree_row_core(
    four_chain: bool,
    groups: usize,
    x: &[f32],
    weight: impl Fn(usize, usize) -> f32,
) -> f32 {
    let mut lanes = [0.0f32; WAVE];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let j = g * GROUP + 4 * lane;
            let mut p = weight(g, j) * x[j];
            for i in 1..4 {
                p = weight(g, j + i).mul_add(x[j + i], p);
            }
            let chain = if four_chain { g & 3 } else { 0 };
            acc[chain] += p;
        }
        *slot = if four_chain {
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            acc[0]
        };
    }
    for off in [16usize, 8, 4, 2, 1] {
        let snapshot = lanes;
        for l in 0..WAVE - off {
            lanes[l] = snapshot[l] + snapshot[l + off];
        }
    }
    lanes[0]
}

/// One output row of the frozen wave tree. `codes`/`scale_bits` are the
/// row's `k` codes and `k/128` scales.
fn tree_row(
    codec: MqI4Codec,
    rotation: MqI4Rotation,
    codes: &[u8],
    scale_bits: &[u16],
    x: &[f32],
) -> f32 {
    tree_row_core(
        rotation == MqI4Rotation::R256,
        scale_bits.len(),
        x,
        |g, j| scale_f32(scale_bits[g]) * weight_i8(codec, codes[j]) as f32,
    )
}

/// The pre-extraction body of `tree_row`, kept verbatim so a test can prove
/// the shared tree helper is bit-equal to it.
#[cfg(test)]
fn tree_row_pre_refactor(
    codec: MqI4Codec,
    rotation: MqI4Rotation,
    codes: &[u8],
    scale_bits: &[u16],
    x: &[f32],
) -> f32 {
    let groups = scale_bits.len();
    let four_chain = rotation == MqI4Rotation::R256;
    let mut lanes = [0.0f32; WAVE];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut acc = [0.0f32; 4];
        for g in 0..groups {
            let d = scale_f32(scale_bits[g]);
            let j = g * GROUP + 4 * lane;
            let w = |i: usize| d * weight_i8(codec, codes[j + i]) as f32;
            let mut p = w(0) * x[j];
            for i in 1..4 {
                p = w(i).mul_add(x[j + i], p);
            }
            let chain = if four_chain { g & 3 } else { 0 };
            acc[chain] += p;
        }
        *slot = if four_chain {
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            acc[0]
        };
    }
    for off in [16usize, 8, 4, 2, 1] {
        let snapshot = lanes;
        for l in 0..WAVE - off {
            lanes[l] = snapshot[l] + snapshot[l + off];
        }
    }
    lanes[0]
}

/// F4 decode GEMV twin on a packed tensor; `x` is the rotated activation
/// (`k` f32). Returns `m` values.
pub fn decode_gemv_f32(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    bytes: &[u8],
    x: &[f32],
) -> Vec<f32> {
    let t = decode_checked("decode_gemv_f32", profile, m, k, bytes);
    decode_gemv_f32_parts(profile, m, k, &t.codes, &t.scale_bits, x)
}

/// [`decode_gemv_f32`] on already validated unpacked codes/scales.
pub fn decode_gemv_f32_parts(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
    x: &[f32],
) -> Vec<f32> {
    check_parts("decode_gemv_f32_parts", profile, m, k, codes, scale_bits);
    assert_eq!(
        x.len(),
        k,
        "decode_gemv_f32_parts: x length {} != k = {k}",
        x.len()
    );
    let groups = k / GROUP;
    (0..m)
        .map(|r| {
            tree_row(
                profile.codec,
                profile.rotation,
                &codes[r * k..(r + 1) * k],
                &scale_bits[r * groups..(r + 1) * groups],
                x,
            )
        })
        .collect()
}

/// The frozen F4 decode tree over integer weights `q` (one `i8` per weight,
/// row-major `[m, k]`) and one f32 scale `d` per 128-column group (row-major
/// `[m, k/128]`). The weight is `w = RN(d * q)` exactly as in
/// [`decode_gemv_f32_parts`] (which is this function on `weight_i8` codes and
/// f16-widened scales); the tree is selected by `rotation` (`R256` four
/// chains, `R128` one). Lets other codecs whose weights are `d * integer`
/// share the one frozen tree. `x` is `k` f32; returns `m` values.
///
/// Panics unless `k % 128 == 0`, `k % rotation.width() == 0`, `m, k > 0` and
/// the three slice lengths match.
pub fn gemv_tree_f32_q(
    rotation: MqI4Rotation,
    m: usize,
    k: usize,
    q: &[i8],
    d: &[f32],
    x: &[f32],
) -> Vec<f32> {
    const WHO: &str = "gemv_tree_f32_q";
    let shape = MqI4Profile {
        codec: MqI4Codec::W3,
        rotation,
    };
    if let Err(e) = mq_i4::tensor_bytes(shape, m, k) {
        panic!("{WHO}: invalid {rotation:?} shape [{m}, {k}]: {e}");
    }
    let groups = k / GROUP;
    assert_eq!(
        q.len(),
        m * k,
        "{WHO}: q length {} != m*k = {} for [{m}, {k}]",
        q.len(),
        m * k
    );
    assert_eq!(
        d.len(),
        m * groups,
        "{WHO}: d length {} != m*(k/128) = {} for [{m}, {k}]",
        d.len(),
        m * groups
    );
    assert_eq!(x.len(), k, "{WHO}: x length {} != k = {k}", x.len());
    let four_chain = rotation == MqI4Rotation::R256;
    (0..m)
        .map(|r| {
            let qr = &q[r * k..(r + 1) * k];
            let dr = &d[r * groups..(r + 1) * groups];
            tree_row_core(four_chain, groups, x, |g, j| dr[g] * qr[j] as f32)
        })
        .collect()
}

/// f64 accuracy reference on a packed tensor (ascending columns, separate
/// f64 multiply and add; see the module docs). `x` is `k` f32.
pub fn gemv_f64(profile: MqI4Profile, m: usize, k: usize, bytes: &[u8], x: &[f32]) -> Vec<f64> {
    let t = decode_checked("gemv_f64", profile, m, k, bytes);
    gemv_f64_parts(profile, m, k, &t.codes, &t.scale_bits, x)
}

/// [`gemv_f64`] on already validated unpacked codes/scales.
pub fn gemv_f64_parts(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
    x: &[f32],
) -> Vec<f64> {
    check_parts("gemv_f64_parts", profile, m, k, codes, scale_bits);
    assert_eq!(
        x.len(),
        k,
        "gemv_f64_parts: x length {} != k = {k}",
        x.len()
    );
    let groups = k / GROUP;
    (0..m)
        .map(|r| {
            let mut acc = 0.0f64;
            for j in 0..k {
                let d = f64::from(scale_f32(scale_bits[r * groups + j / GROUP]));
                let w = d * f64::from(weight_i8(profile.codec, codes[r * k + j]));
                let p = w * f64::from(x[j]);
                acc += p;
            }
            acc
        })
        .collect()
}

// --------------------------------------------------------------------------
// F4 slot (bucketed expert) decode
// --------------------------------------------------------------------------

/// Which expert projection a bucket computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeKind {
    /// Fused gate/up: `m` rows, first `m/2` gate, last `m/2` up; `R256`.
    GateUp,
    /// Down: `m` rows; `R128`.
    Down,
}

/// One expert's packed tensor.
#[derive(Clone, Copy, Debug)]
pub struct ExpertTensor<'a> {
    pub profile: MqI4Profile,
    pub bytes: &'a [u8],
}

/// One expert's already validated unpacked tensor.
#[derive(Clone, Copy, Debug)]
pub struct ExpertTensorParts<'a> {
    pub profile: MqI4Profile,
    pub codes: &'a [u8],
    pub scale_bits: &'a [u16],
}

fn check_bucket(expert_ids: &[u32], expert_offsets: &[u32], slot_ids: &[u32]) {
    assert_eq!(
        expert_offsets.len(),
        expert_ids.len() + 1,
        "decode_moe_slots: expert_offsets length {} != expert_ids length {} + 1",
        expert_offsets.len(),
        expert_ids.len()
    );
    assert_eq!(
        expert_offsets[0], 0,
        "decode_moe_slots: expert_offsets[0] = {} != 0",
        expert_offsets[0]
    );
    assert_eq!(
        expert_offsets[expert_ids.len()] as usize,
        slot_ids.len(),
        "decode_moe_slots: last expert offset {} != slot_ids length {}",
        expert_offsets[expert_ids.len()],
        slot_ids.len()
    );
    for (u, w) in expert_offsets.windows(2).enumerate() {
        assert!(
            w[0] <= w[1],
            "decode_moe_slots: expert_offsets not monotonic at {u}: {} > {}",
            w[0],
            w[1]
        );
    }
}

/// End index `index * stride + width`, panicking on overflow.
fn span_end(who: &str, what: &str, index: usize, stride: usize, width: usize) -> usize {
    index
        .checked_mul(stride)
        .and_then(|v| v.checked_add(width))
        .unwrap_or_else(|| panic!("{who}: {what} span overflows (index {index}, stride {stride})"))
}

/// Bucketed F4 decode twin on packed expert tensors; see the module docs for
/// the bucket contract. Each distinct expert is decoded once.
#[allow(clippy::too_many_arguments)]
pub fn decode_moe_slots(
    kind: DecodeKind,
    m: usize,
    k: usize,
    experts: &[Option<ExpertTensor<'_>>],
    expert_ids: &[u32],
    expert_offsets: &[u32],
    slot_ids: &[u32],
    c: usize,
    x: &[f32],
    x_row_stride: usize,
    y_slot_stride: usize,
    y0: &mut [f32],
    y1: Option<&mut [f32]>,
) {
    const WHO: &str = "decode_moe_slots";
    check_bucket(expert_ids, expert_offsets, slot_ids);
    let mut owned: Vec<Option<MqI4Tensor>> = (0..experts.len()).map(|_| None).collect();
    for &id in expert_ids {
        let id = id as usize;
        let e = experts
            .get(id)
            .unwrap_or_else(|| panic!("{WHO}: expert id {id} >= experts length {}", experts.len()))
            .as_ref()
            .unwrap_or_else(|| panic!("{WHO}: expert {id} has no tensor"));
        if owned[id].is_none() {
            owned[id] = Some(decode_checked(WHO, e.profile, m, k, e.bytes));
        }
    }
    let parts: Vec<Option<ExpertTensorParts<'_>>> = experts
        .iter()
        .zip(&owned)
        .map(|(e, t)| match (e, t) {
            (Some(e), Some(t)) => Some(ExpertTensorParts {
                profile: e.profile,
                codes: &t.codes,
                scale_bits: &t.scale_bits,
            }),
            _ => None,
        })
        .collect();
    decode_moe_slots_parts(
        kind,
        m,
        k,
        &parts,
        expert_ids,
        expert_offsets,
        slot_ids,
        c,
        x,
        x_row_stride,
        y_slot_stride,
        y0,
        y1,
    );
}

/// [`decode_moe_slots`] on already validated unpacked expert tensors.
#[allow(clippy::too_many_arguments)]
pub fn decode_moe_slots_parts(
    kind: DecodeKind,
    m: usize,
    k: usize,
    experts: &[Option<ExpertTensorParts<'_>>],
    expert_ids: &[u32],
    expert_offsets: &[u32],
    slot_ids: &[u32],
    c: usize,
    x: &[f32],
    x_row_stride: usize,
    y_slot_stride: usize,
    y0: &mut [f32],
    y1: Option<&mut [f32]>,
) {
    const WHO: &str = "decode_moe_slots_parts";
    let (out_rows, rotation) = match kind {
        DecodeKind::GateUp => {
            assert!(
                m > 0 && m % 2 == 0,
                "{WHO}: GateUp requires even m > 0, got {m}"
            );
            (m / 2, MqI4Rotation::R256)
        }
        DecodeKind::Down => {
            assert!(m > 0, "{WHO}: Down requires m > 0");
            (m, MqI4Rotation::R128)
        }
    };
    assert_eq!(
        y1.is_some(),
        kind == DecodeKind::GateUp,
        "{WHO}: y1 must be Some for GateUp and None for Down"
    );
    check_bucket(expert_ids, expert_offsets, slot_ids);

    for &id in expert_ids {
        let id = id as usize;
        let e = experts
            .get(id)
            .unwrap_or_else(|| panic!("{WHO}: expert id {id} >= experts length {}", experts.len()))
            .as_ref()
            .unwrap_or_else(|| panic!("{WHO}: expert {id} has no tensor"));
        assert_eq!(
            e.profile.rotation,
            rotation,
            "{WHO}: expert {id} profile {} has the wrong rotation for {kind:?}",
            e.profile.name()
        );
        check_parts(WHO, e.profile, m, k, e.codes, e.scale_bits);
    }

    let slot_limit = c
        .checked_mul(10)
        .unwrap_or_else(|| panic!("{WHO}: 10*c overflows (c = {c})"));
    let mut seen = vec![false; slot_limit];
    for &slot in slot_ids {
        let slot = slot as usize;
        assert!(
            slot < slot_limit,
            "{WHO}: slot id {slot} >= 10*c = {slot_limit}"
        );
        assert!(!seen[slot], "{WHO}: duplicate slot id {slot}");
        seen[slot] = true;
    }

    if !slot_ids.is_empty() {
        assert!(
            x_row_stride >= k,
            "{WHO}: x_row_stride {x_row_stride} < k = {k}"
        );
        assert!(
            y_slot_stride >= out_rows,
            "{WHO}: y_slot_stride {y_slot_stride} < output rows {out_rows}"
        );
    }
    let x_row = |slot: usize| match kind {
        DecodeKind::GateUp => slot / 10,
        DecodeKind::Down => slot,
    };
    for &slot in slot_ids {
        let slot = slot as usize;
        let x_end = span_end(WHO, "x row", x_row(slot), x_row_stride, k);
        assert!(
            x_end <= x.len(),
            "{WHO}: slot {slot} reads x up to {x_end} but x length is {}",
            x.len()
        );
        let y_end = span_end(WHO, "y slot", slot, y_slot_stride, out_rows);
        assert!(
            y_end <= y0.len(),
            "{WHO}: slot {slot} writes y0 up to {y_end} but y0 length is {}",
            y0.len()
        );
        if let Some(y1) = y1.as_deref() {
            assert!(
                y_end <= y1.len(),
                "{WHO}: slot {slot} writes y1 up to {y_end} but y1 length is {}",
                y1.len()
            );
        }
    }

    let mut y1 = y1;
    let groups = k / GROUP;
    for (u, &id) in expert_ids.iter().enumerate() {
        let e = experts[id as usize].as_ref().expect("validated above");
        let seg = &slot_ids[expert_offsets[u] as usize..expert_offsets[u + 1] as usize];
        for &slot in seg {
            let slot = slot as usize;
            let xs = &x[x_row(slot) * x_row_stride..][..k];
            for r in 0..m {
                let v = tree_row(
                    e.profile.codec,
                    e.profile.rotation,
                    &e.codes[r * k..(r + 1) * k],
                    &e.scale_bits[r * groups..(r + 1) * groups],
                    xs,
                );
                let base = slot * y_slot_stride;
                if r < out_rows {
                    y0[base + r] = v;
                } else {
                    y1.as_deref_mut().expect("checked Some for GateUp")[base + r - out_rows] = v;
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// BF16 grouped-prefill epilogue
// --------------------------------------------------------------------------

/// `Bf16::RneFinitePassthrough` on raw f32 bits: finite values get
/// `x + 0x7fff + ((x >> 16) & 1)` (wrapping; BF16 bits are the high half, the
/// low half is unspecified); every non-finite value is returned unchanged
/// (all 32 bits, including NaN payload).
pub fn bf16_rne_finite_passthrough_bits(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        bits
    } else {
        bits.wrapping_add(0x7fff).wrapping_add((bits >> 16) & 1)
    }
}

/// Gate/up input rounding: finite -> rounded with the low half cleared;
/// non-finite -> the original f32 untouched (the source's `clear_low` AND is
/// part of the finite candidate only).
fn round_f32(v: f32) -> f32 {
    let bits = v.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        v
    } else {
        f32::from_bits(bf16_rne_finite_passthrough_bits(bits) & 0xffff_0000)
    }
}

/// The three device intrinsics of the SiLU epilogue.
///
/// `exp2` is `v_exp_f32`, `ldexp` is `v_ldexp_f32`, `divide` is the complete
/// IEEE `v_div_scale_f32` / `v_rcp_f32` / fma / `v_div_fmas_f32` /
/// `v_div_fixup_f32` sequence (`gemm_uk/epilogue.rs:83-96`), i.e. a *result*
/// that Rust cannot reproduce without the hardware's reciprocal seed.
pub trait EpilogueIntrinsics {
    fn exp2(&self, x: f32) -> f32;
    fn ldexp(&self, x: f32, e: i32) -> f32;
    fn divide(&self, numerator: f32, denominator: f32) -> f32;
}

/// Portable math (`f32::exp2`, exact power-of-two scaling, Rust `/`). A
/// portable reference, **not** a device byte oracle.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortableEpilogue;

impl EpilogueIntrinsics for PortableEpilogue {
    fn exp2(&self, x: f32) -> f32 {
        x.exp2()
    }

    fn ldexp(&self, x: f32, e: i32) -> f32 {
        // |e| > 400 already forces 0 / inf for every finite non-zero f32, so
        // clamping keeps 2^e an exact normal f64; x * 2^e is exact in f64 and
        // rounded once to f32 (correct gradual underflow).
        let e = e.clamp(-400, 400);
        let pow2 = f64::from_bits(((e + 1023) as u64) << 52);
        (f64::from(x) * pow2) as f32
    }

    fn divide(&self, numerator: f32, denominator: f32) -> f32 {
        numerator / denominator
    }
}

/// [`grouped_prefill_epilogue_with`] using [`PortableEpilogue`].
pub fn grouped_prefill_epilogue(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
) -> Vec<u16> {
    grouped_prefill_epilogue_with(kind, m, sums, slot_ids, &PortableEpilogue)
}

/// BF16 epilogue over `sums` (`[slot_ids.len(), m]` row-major); see the module
/// docs. Returns row-major BF16 bits `[slot_ids.len(), m/2]` (GateUp) or
/// `[slot_ids.len(), m]` (Down). `slot_ids[i] == -1` marks a padding row
/// (output `+0`, its `sums` row is never read); any other negative id panics.
pub fn grouped_prefill_epilogue_with<I: EpilogueIntrinsics>(
    kind: DecodeKind,
    m: usize,
    sums: &[f32],
    slot_ids: &[i32],
    intrinsics: &I,
) -> Vec<u16> {
    const WHO: &str = "grouped_prefill_epilogue";
    assert!(m > 0, "{WHO}: m must be > 0");
    let out_cols = match kind {
        DecodeKind::GateUp => {
            assert!(m % 2 == 0, "{WHO}: GateUp requires even m, got {m}");
            m / 2
        }
        DecodeKind::Down => m,
    };
    let rows = slot_ids.len();
    let expected = rows
        .checked_mul(m)
        .unwrap_or_else(|| panic!("{WHO}: rows*m overflows ({rows} * {m})"));
    assert_eq!(
        sums.len(),
        expected,
        "{WHO}: sums length {} != slot_ids length {rows} * m = {expected}",
        sums.len()
    );
    let mut out = vec![0u16; rows * out_cols];
    for (row, &slot) in slot_ids.iter().enumerate() {
        assert!(slot >= -1, "{WHO}: invalid slot id {slot} at row {row}");
        if slot == -1 {
            continue;
        }
        let s = &sums[row * m..(row + 1) * m];
        for i in 0..out_cols {
            let h = match kind {
                DecodeKind::GateUp => {
                    silu_mul(intrinsics, round_f32(s[i]), round_f32(s[out_cols + i]))
                }
                DecodeKind::Down => s[i],
            };
            out[row * out_cols + i] = (bf16_rne_finite_passthrough_bits(h.to_bits()) >> 16) as u16;
        }
    }
    out
}

/// `silu_dense` (`DivNum::Gate`, `Finish::MulUp`): `RN(IEEE(g / (1 + e^-g)) * u)`.
fn silu_mul<I: EpilogueIntrinsics>(intr: &I, g: f32, u: f32) -> f32 {
    let neg_log2e = f32::from_bits(0xbfb8_aa3b);
    let neg_log2e_lo = f32::from_bits(0xb2a5_705f);
    let ph = neg_log2e * g;
    let mut pl = neg_log2e.mul_add(g, -ph);
    pl = neg_log2e_lo.mul_add(g, pl);
    let e = ph.round_ties_even();
    let a = (ph - e) + pl;
    // v_cvt_i32_f32 of an integral value; Rust's saturating cast (NaN -> 0)
    // is the same function on the finite domain the source produces.
    let mut r = intr.ldexp(intr.exp2(a), e as i32);
    // Ordered compares: NaN g leaves r untouched (v_cmp_nlt / v_cmp_ngt).
    if f32::from_bits(0x42ce_8ed0) < g {
        r = 0.0;
    }
    if g < f32::from_bits(0xc2b1_7218) {
        r = f32::INFINITY;
    }
    let den = 1.0 + r;
    let q = intr.divide(g, den);
    q * u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mq_i4::{self, MqI4Codec, MqI4Profile, MqI4Rotation};
    use half::f16;

    // ---- f16 scale bit patterns (all exact) --------------------------------
    const H_MIN_SUB: u16 = 0x0001; // 2^-24, smallest f16 subnormal
    const H_EIGHTH: u16 = 0x3000; // 0.125
    const H_QUARTER: u16 = 0x3400; // 0.25
    const H_HALF: u16 = 0x3800; // 0.5
    const H_ONE: u16 = 0x3C00; // 1.0
    const H_TWO: u16 = 0x4000; // 2.0
    const H_FOUR: u16 = 0x4400; // 4.0
    const H_EIGHT: u16 = 0x4800; // 8.0
    const H_MAX: u16 = 0x7BFF; // 65504

    const P3_R256: MqI4Profile = MqI4Profile::MQ3_R256;
    const P3_R128: MqI4Profile = MqI4Profile::MQ3_R128;
    const P2_R256: MqI4Profile = MqI4Profile::MQ2_R256;
    const P2_R128: MqI4Profile = MqI4Profile::MQ2_R128;

    const TWO24: f32 = 16_777_216.0; // 2^24

    fn enc(profile: MqI4Profile, m: usize, k: usize, codes: &[u8], scales: &[u16]) -> Vec<u8> {
        mq_i4::encode_tensor(profile, m, k, codes, scales).expect("valid S0 fixture")
    }

    fn bits32(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn bits64(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn a4_fill(d: f32, v: i8) -> A4Block {
        A4Block::from_values(d, &[v; 128])
    }

    fn is_bf16_nan(b: u16) -> bool {
        (b & 0x7F80) == 0x7F80 && (b & 0x007F) != 0
    }

    // =======================================================================
    // Weight tables
    // =======================================================================

    #[test]
    fn weight_tables_literal() {
        // W3: 2c - 7, c = 0..7. Nibble = two's complement low nibble, NO xor rebias:
        // -7 -> 0xF9 -> 9, -5 -> 0xFB -> B, -3 -> 0xFD -> D, -1 -> 0xFF -> F.
        let w3_i8: [i8; 8] = [-7, -5, -3, -1, 1, 3, 5, 7];
        let w3_nib: [u8; 8] = [0x9, 0xB, 0xD, 0xF, 0x1, 0x3, 0x5, 0x7];
        for c in 0..8u8 {
            assert_eq!(
                weight_i8(MqI4Codec::W3, c),
                w3_i8[c as usize],
                "W3 i8 c={c}"
            );
            assert_eq!(
                weight_nibble(MqI4Codec::W3, c),
                w3_nib[c as usize],
                "W3 nibble c={c}"
            );
        }
        // W2: 2c - 3, c = 0..3.
        let w2_i8: [i8; 4] = [-3, -1, 1, 3];
        let w2_nib: [u8; 4] = [0xD, 0xF, 0x1, 0x3];
        for c in 0..4u8 {
            assert_eq!(
                weight_i8(MqI4Codec::W2, c),
                w2_i8[c as usize],
                "W2 i8 c={c}"
            );
            assert_eq!(
                weight_nibble(MqI4Codec::W2, c),
                w2_nib[c as usize],
                "W2 nibble c={c}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceeds L")]
    fn weight_i8_w3_code_8_panics() {
        weight_i8(MqI4Codec::W3, 8);
    }

    #[test]
    #[should_panic(expected = "exceeds L")]
    fn weight_i8_w2_code_4_panics() {
        weight_i8(MqI4Codec::W2, 4);
    }

    // =======================================================================
    // Dequant
    // =======================================================================

    /// Four scales x every code, all four profiles, packed and `_parts`.
    ///
    /// Row r / group g uses scale id `SIDS[r][g]` (0:0.25, 1:1.0, 2:65504,
    /// 3:2^-24); the codes of every row are `j % (L+1)` so every code 0..=L
    /// meets every scale and both group slots of a row. `float(d)*(2c-L)` is
    /// exact in f32 for all of them (<= 11-bit scale x <= 4-bit odd integer).
    fn check_dequant_all(profile: MqI4Profile, tables: [&[f32]; 4]) {
        const SIDS: [[usize; 2]; 4] = [[0, 1], [1, 0], [2, 3], [3, 2]];
        const SCALES: [u16; 4] = [H_QUARTER, H_ONE, H_MAX, H_MIN_SUB];
        let (m, k) = (4usize, 256usize);
        let n = tables[0].len(); // L + 1
        let mut codes = Vec::with_capacity(m * k);
        let mut scales = Vec::new();
        for r in 0..m {
            for j in 0..k {
                codes.push((j % n) as u8);
            }
            for g in 0..2 {
                scales.push(SCALES[SIDS[r][g]]);
            }
        }
        let bytes = enc(profile, m, k, &codes, &scales);
        let packed = dequant_rotated_f32(profile, m, k, &bytes);
        let parts = dequant_rotated_f32_parts(profile, m, k, &codes, &scales);
        for r in 0..m {
            for j in 0..k {
                let want = tables[SIDS[r][j / 128]][j % n];
                let i = r * k + j;
                assert_eq!(
                    packed[i].to_bits(),
                    want.to_bits(),
                    "{} packed r={r} j={j}: got {} want {want}",
                    profile.name(),
                    packed[i]
                );
                assert_eq!(
                    parts[i].to_bits(),
                    want.to_bits(),
                    "{} parts r={r} j={j}",
                    profile.name()
                );
            }
        }
    }

    #[test]
    fn dequant_w3_every_code_every_scale_literal() {
        // c = 0..7 -> 2c-7 = -7,-5,-3,-1,1,3,5,7
        let q: [f32; 8] = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]; // x 0.25
        let one: [f32; 8] = [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]; // x 1
                                                                          // 65504 x {7,5,3,1} = 458528, 327520, 196512, 65504 (all < 2^24, exact)
        let big: [f32; 8] = [
            -458_528.0, -327_520.0, -196_512.0, -65_504.0, 65_504.0, 196_512.0, 327_520.0,
            458_528.0,
        ];
        // 2^-24 x odd integer: exact division by a power of two.
        let tiny: [f32; 8] = [
            -7.0 / 16_777_216.0,
            -5.0 / 16_777_216.0,
            -3.0 / 16_777_216.0,
            -1.0 / 16_777_216.0,
            1.0 / 16_777_216.0,
            3.0 / 16_777_216.0,
            5.0 / 16_777_216.0,
            7.0 / 16_777_216.0,
        ];
        check_dequant_all(P3_R256, [&q, &one, &big, &tiny]);
        check_dequant_all(P3_R128, [&q, &one, &big, &tiny]);
    }

    #[test]
    fn dequant_w2_every_code_every_scale_literal() {
        // c = 0..3 -> 2c-3 = -3,-1,1,3
        let q: [f32; 4] = [-0.75, -0.25, 0.25, 0.75];
        let one: [f32; 4] = [-3.0, -1.0, 1.0, 3.0];
        let big: [f32; 4] = [-196_512.0, -65_504.0, 65_504.0, 196_512.0]; // 65504 x {3,1}
        let tiny: [f32; 4] = [
            -3.0 / 16_777_216.0,
            -1.0 / 16_777_216.0,
            1.0 / 16_777_216.0,
            3.0 / 16_777_216.0,
        ];
        check_dequant_all(P2_R256, [&q, &one, &big, &tiny]);
        check_dequant_all(P2_R128, [&q, &one, &big, &tiny]);
    }

    #[test]
    fn dequant_zero_scale_midpoint_group_is_positive_zero() {
        // d == +0 forces every code to the midpoint (W3: 4 -> w=+1, W2: 2 -> w=+1);
        // +0 * (+1) = +0 (bits 0), never -0. The second group (scale 1.0) proves
        // the zero is per group.
        for (profile, mid, other) in [
            (P3_R256, 4u8, 7u8),
            (P3_R128, 4, 7),
            (P2_R256, 2, 3),
            (P2_R128, 2, 3),
        ] {
            let (m, k) = (1usize, 256usize);
            let mut codes = vec![mid; 128];
            codes.extend(std::iter::repeat(other).take(128));
            let scales = [0u16, H_ONE];
            let bytes = enc(profile, m, k, &codes, &scales);
            let out = dequant_rotated_f32(profile, m, k, &bytes);
            let top = f32::from(weight_i8(profile.codec, other));
            for j in 0..128 {
                assert_eq!(out[j].to_bits(), 0, "{} zero-group col {j}", profile.name());
                assert_eq!(
                    out[128 + j].to_bits(),
                    top.to_bits(),
                    "{} col {}",
                    profile.name(),
                    128 + j
                );
            }
        }
    }

    // =======================================================================
    // A4Block byte order and 72-byte ABI
    // =======================================================================

    /// Byte pattern of 16 consecutive values `(j % 16) - 8` = -8..=7:
    /// nibbles 8,9,A,B,C,D,E,F,0,1,2,3,4,5,6,7; low nibble = even element, so
    /// the bytes are 0x98,0xBA,0xDC,0xFE,0x10,0x32,0x54,0x76.
    const RAMP_BYTES: [u8; 8] = [0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76];

    #[test]
    fn a4_block_abi_72_bytes_d_s_qs_and_nibble_order() {
        assert_eq!(std::mem::size_of::<A4Block>(), 72);
        assert_eq!(std::mem::align_of::<A4Block>(), 4);

        let mut values = [0i8; 128];
        for (j, v) in values.iter_mut().enumerate() {
            *v = (j % 16) as i8 - 8;
        }
        let b = A4Block::from_values(1.5, &values);
        let base = &b as *const A4Block as usize;
        assert_eq!(std::ptr::addr_of!(b.d) as usize - base, 0);
        assert_eq!(std::ptr::addr_of!(b.s) as usize - base, 4);
        assert_eq!(std::ptr::addr_of!(b.qs) as usize - base, 8);

        // s = 8 repetitions of sum(-8..=7) = 8 * (-8) = -64
        assert_eq!(b.s, -64);
        assert_eq!(b.d.to_bits(), 0x3FC0_0000);
        for i in 0..64 {
            assert_eq!(b.qs[i], RAMP_BYTES[i % 8], "qs[{i}]");
        }
        for j in 0..128 {
            assert_eq!(b.element(j), values[j], "element({j})");
        }

        // Raw memory image on little-endian hosts: d LE, s LE, then qs.
        #[cfg(target_endian = "little")]
        {
            let raw: &[u8] =
                unsafe { std::slice::from_raw_parts(&b as *const A4Block as *const u8, 72) };
            assert_eq!(&raw[0..4], &[0x00, 0x00, 0xC0, 0x3F]); // 1.5f32
            assert_eq!(&raw[4..8], &[0xC0, 0xFF, 0xFF, 0xFF]); // -64i32
            assert_eq!(&raw[8..72], &b.qs[..]);
        }
    }

    #[test]
    fn a4_block_producer_lane_owns_contiguous_eight_elements() {
        // Lane l (0..16 of a K128 block) owns elements 8l..8l+8 = bytes 4l..4l+4.
        // Give lane l the constant value l-8 (-8..=7): its four bytes are the
        // nibble doubled: l=0 -> 0x88, 1 -> 0x99, .., 7 -> 0xFF, 8 -> 0x00, 9 -> 0x11 .. 15 -> 0x77.
        const LANE_BYTE: [u8; 16] = [
            0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55,
            0x66, 0x77,
        ];
        let mut values = [0i8; 128];
        for (j, v) in values.iter_mut().enumerate() {
            *v = (j / 8) as i8 - 8;
        }
        let b = A4Block::from_values(1.0, &values);
        assert_eq!(b.s, -64); // 8 * sum(-8..=7)
        for lane in 0..16 {
            for byte in 0..4 {
                assert_eq!(
                    b.qs[4 * lane + byte],
                    LANE_BYTE[lane],
                    "lane {lane} byte {byte}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside -8..=7")]
    fn a4_from_values_rejects_8() {
        let mut v = [0i8; 128];
        v[5] = 8;
        A4Block::from_values(1.0, &v);
    }

    #[test]
    #[should_panic(expected = "outside -8..=7")]
    fn a4_from_values_rejects_minus_9() {
        let mut v = [0i8; 128];
        v[127] = -9;
        A4Block::from_values(1.0, &v);
    }

    // =======================================================================
    // IU4 fold
    // =======================================================================

    #[test]
    fn iu4_m1_k128_all_w3_code7_literal_112() {
        // w = 2*7-7 = 7, a = 1: dot = 128*7*1 = 896; scale = d_act*d_w = 0.5*0.25 = 0.125;
        // sum = fma(0.125, 896, +0) = 112.
        let bytes = enc(P3_R128, 1, 128, &[7u8; 128], &[H_QUARTER]);
        let x = [a4_fill(0.5, 1)];
        let out = iu4_gemm_f32(P3_R128, 1, 128, &bytes, &x);
        assert_eq!(out, vec![112.0f32]);
        assert_eq!(out[0].to_bits(), 112.0f32.to_bits());
        // W2 analogue: code 3 -> w = 3; dot = 384; 0.125 * 384 = 48.
        let bytes = enc(P2_R128, 1, 128, &[3u8; 128], &[H_QUARTER]);
        assert_eq!(iu4_gemm_f32(P2_R128, 1, 128, &bytes, &x), vec![48.0f32]);
    }

    #[test]
    fn iu4_k384_exact_power_of_two_scales_literal() {
        // m=2, k=384 (3 groups), 2 tokens, R128 (384 % 256 != 0 rules out R256).
        // Row 0 weights: groups (code7 -> +7, code0 -> -7, code5 -> +3), scales (2.0, 0.5, 0.25).
        // Row 1 weights: code4 -> +1 everywhere, scale 1.0.
        // Token 0 blocks: (d=0.5, a=1) (d=0.25, a=-2) (d=2.0, a=3); token 1 = negated values.
        //   row0: g0 scale 2.0*0.5 = 1      dot 128*7*1      =  896  -> 896
        //         g1 scale 0.5*0.25 = 0.125 dot 128*(-7)(-2) = 1792  -> 224
        //         g2 scale 0.25*2 = 0.5     dot 128*3*3      = 1152  -> 576      sum 1696
        //   row1: 0.5*(128*1) = 64, 0.25*(128*-2) = -64, 2.0*(128*3) = 768       sum 768
        // out[t*m + r] = [1696, 768, -1696, -768]; every partial is exact.
        let (m, k) = (2usize, 384usize);
        let mut codes = vec![7u8; 128];
        codes.extend(std::iter::repeat(0u8).take(128));
        codes.extend(std::iter::repeat(5u8).take(128));
        codes.extend(std::iter::repeat(4u8).take(384));
        let scales = [H_TWO, H_HALF, H_QUARTER, H_ONE, H_ONE, H_ONE];
        let bytes = enc(P3_R128, m, k, &codes, &scales);
        let x = [
            a4_fill(0.5, 1),
            a4_fill(0.25, -2),
            a4_fill(2.0, 3),
            a4_fill(0.5, -1),
            a4_fill(0.25, 2),
            a4_fill(2.0, -3),
        ];
        let out = iu4_gemm_f32(P3_R128, m, k, &bytes, &x);
        assert_eq!(out, vec![1696.0f32, 768.0, -1696.0, -768.0]);
    }

    #[test]
    fn iu4_dot_extremes_plus_minus_7168() {
        // |dot| <= 128 * 7 * 8 = 7168: w=-7 (code 0) with a=-8 is +7168; w=+7 (code 7)
        // with a=-8 is -7168. Token 1 (a=7): +/-6272 = 128*7*7.
        let mut codes = vec![0u8; 128];
        codes.extend(std::iter::repeat(7u8).take(128));
        let bytes = enc(P3_R128, 2, 128, &codes, &[H_ONE, H_ONE]);
        let x = [a4_fill(1.0, -8), a4_fill(1.0, 7)];
        let out = iu4_gemm_f32(P3_R128, 2, 128, &bytes, &x);
        assert_eq!(out, vec![7168.0f32, -7168.0, -6272.0, 6272.0]);
    }

    #[test]
    fn iu4_nibble_order_matches_handwritten_qs_bytes() {
        // Weights c_j = j % 8 (w_j = 2*(j%8)-7), activations a_j = (j%16)-8 packed by hand
        // (RAMP_BYTES, struct literal - independent of from_values/element).
        // dot = sum_j (2(j%8)-7)((j%16)-8) = 1344 (evaluated over the 128 elements);
        // scale = 0.5 * 0.25 = 0.125 -> 1344 * 0.125 = 168.
        // A swapped nibble order (a_j taken from j^1) would give dot 1216 -> 152.
        let codes: Vec<u8> = (0..128).map(|j| (j % 8) as u8).collect();
        let bytes = enc(P3_R128, 1, 128, &codes, &[H_QUARTER]);
        let mut qs = [0u8; 64];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = RAMP_BYTES[i % 8];
        }
        let blk = A4Block { d: 0.5, s: -64, qs };
        assert_eq!(
            iu4_gemm_f32(P3_R128, 1, 128, &bytes, &[blk]),
            vec![168.0f32]
        );
    }

    #[test]
    fn iu4_fma_fold_is_fused_not_mul_then_add() {
        // K=256 R256, every weight code 4 (w=+1), d_w = 1.0 in both groups, 1 token.
        //   group 0: d_act = 1.0,        3 elements = -1  -> dot = -3, scale0 = 1
        //            sum = fma(1, -3, +0) = -3
        //   group 1: d_act = 1 + 2^-23,  3 elements = +1  -> dot = +3, scale1 = RN(d_act*1)
        //            = 1 + 2^-23 exactly.
        // Fused:   fma(scale1, 3, -3) = 3(1+2^-23) - 3 = 3*2^-23 = 1.5*2^-22 exactly
        //          (2^-22 has biased exponent 105 = 0x34800000; x1.5 -> mantissa 0x400000)
        //          = bits 0x34C00000.
        // Unfused: the product 3(1+2^-23) = 3 + 1.5*2^-22 lies on a tie between 3+2^-22 and
        //          3+2^-21 (ulp in [2,4) is 2^-22; 3 = 12582912 ulp, even); ties-to-even picks
        //          3 + 2^-21, then -3 + (3 + 2^-21) = 2^-21 = bits 0x35000000.
        let codes = vec![4u8; 256];
        let bytes = enc(P3_R256, 1, 256, &codes, &[H_ONE, H_ONE]);
        let mut neg = [0i8; 128];
        neg[..3].fill(-1);
        let mut pos = [0i8; 128];
        pos[..3].fill(1);
        let d1 = f32::from_bits(0x3F80_0001);
        let x = [
            A4Block::from_values(1.0, &neg),
            A4Block::from_values(d1, &pos),
        ];
        let out = iu4_gemm_f32(P3_R256, 1, 256, &bytes, &x);
        assert_eq!(out[0].to_bits(), 0x34C0_0000, "fused literal");

        // Prove the fixture discriminates: Rust never contracts `a*b + c`.
        let unfused = -3.0f32 + (d1 * 1.0) * 3.0;
        assert_eq!(unfused.to_bits(), 0x3500_0000, "unfused literal");
    }

    // =======================================================================
    // Decode GEMV tree
    // =======================================================================

    /// m=1 W3 row with weight +1 everywhere (code 4, scale 1.0) except `edits`
    /// (column, code); activations 0 except `xs` (column, value). Returns the
    /// F4 f32 result and the f64 reference; asserts packed == `_parts` bits.
    fn run_row(
        rot: MqI4Rotation,
        k: usize,
        edits: &[(usize, u8)],
        xs: &[(usize, f32)],
    ) -> (f32, f64) {
        let profile = MqI4Profile {
            codec: MqI4Codec::W3,
            rotation: rot,
        };
        let mut codes = vec![4u8; k];
        for &(j, c) in edits {
            codes[j] = c;
        }
        let scales = vec![H_ONE; k / 128];
        let mut x = vec![0.0f32; k];
        for &(j, v) in xs {
            x[j] = v;
        }
        let bytes = enc(profile, 1, k, &codes, &scales);
        let out = decode_gemv_f32(profile, 1, k, &bytes, &x);
        let parts = decode_gemv_f32_parts(profile, 1, k, &codes, &scales, &x);
        assert_eq!(out[0].to_bits(), parts[0].to_bits());
        let f = gemv_f64(profile, 1, k, &bytes, &x);
        (out[0], f[0])
    }

    #[test]
    fn decode_integer_answers_k128_r128_w3_w2() {
        // x[j] = (j%8)+1; sum over 128 columns = 16 * 36 = 576.
        //   row0: codes 4 (w=1), d=1              -> 576
        //   row1: codes 7 (w=7), d=0.5 (w=3.5)    -> 2016
        //   row2: codes 0 (w=-7), d=0.25 (-1.75)  -> -1008
        //   row3: codes j%8 (w=2(j%8)-7), d=2:  per 8-block sum (2c-7)(c+1) = 84, x16 = 1344, x2 -> 2688
        let x: Vec<f32> = (0..128).map(|j| ((j % 8) + 1) as f32).collect();
        let mut codes = vec![4u8; 128];
        codes.extend(std::iter::repeat(7u8).take(128));
        codes.extend(std::iter::repeat(0u8).take(128));
        codes.extend((0..128).map(|j| (j % 8) as u8));
        let scales = [H_ONE, H_HALF, H_QUARTER, H_TWO];
        let bytes = enc(P3_R128, 4, 128, &codes, &scales);
        let want = vec![576.0f32, 2016.0, -1008.0, 2688.0];
        assert_eq!(decode_gemv_f32(P3_R128, 4, 128, &bytes, &x), want);
        assert_eq!(
            decode_gemv_f32_parts(P3_R128, 4, 128, &codes, &scales, &x),
            want
        );
        assert_eq!(
            gemv_f64(P3_R128, 4, 128, &bytes, &x),
            vec![576.0f64, 2016.0, -1008.0, 2688.0]
        );

        // W2 (R128): row0 codes 3 (w=3), d=1 -> 3*576 = 1728; row1 codes 0 (w=-3), d=0.5
        // -> -1.5*576 = -864; row2 codes j%4 (w = -3,-1,1,3 repeating), d=2: per 8-block
        // (-3*1 -1*2 +1*3 +3*4) + (-3*5 -1*6 +1*7 +3*8) = 10 + 10 = 20, x16 = 320, x2 = 640.
        let mut codes = vec![3u8; 128];
        codes.extend(std::iter::repeat(0u8).take(128));
        codes.extend((0..128).map(|j| (j % 4) as u8));
        let scales = [H_ONE, H_HALF, H_TWO];
        let bytes = enc(P2_R128, 3, 128, &codes, &scales);
        assert_eq!(
            decode_gemv_f32(P2_R128, 3, 128, &bytes, &x),
            vec![1728.0f32, -864.0, 640.0]
        );
    }

    #[test]
    fn decode_integer_answers_k256_r256() {
        // Same x pattern over 256 columns: sum = 32 * 36 = 1152.
        //   row0: codes 4, scales (1,1)                      -> 1152
        //   row1: codes j%8, scales (2.0, 0.5): 1344*2 + 1344*0.5 = 3360
        let x: Vec<f32> = (0..256).map(|j| ((j % 8) + 1) as f32).collect();
        let mut codes = vec![4u8; 256];
        codes.extend((0..256).map(|j| (j % 8) as u8));
        let scales = [H_ONE, H_ONE, H_TWO, H_HALF];
        let bytes = enc(P3_R256, 2, 256, &codes, &scales);
        assert_eq!(
            decode_gemv_f32(P3_R256, 2, 256, &bytes, &x),
            vec![1152.0f32, 3360.0]
        );
        // W2 R256: all codes 3 (w=3), scales (1,1): 3 * 1152 = 3456.
        let bytes = enc(P2_R256, 1, 256, &[3u8; 256], &[H_ONE, H_ONE]);
        assert_eq!(
            decode_gemv_f32(P2_R256, 1, 256, &bytes, &x),
            vec![3456.0f32]
        );
    }

    /// Cases (x edits at weight +1; everything else 0) whose f32 result depends on the
    /// exact association order, all with exact f64 result 1.0.
    #[test]
    fn decode_catastrophic_cancellation_pins_lane_layout_and_shuffle_order() {
        // 2^24 + 1 is a tie in f32 (ulp 2 at 2^24) and rounds to even, 2^24, losing the 1.
        // Wave: lane l owns columns 4l..4l+3 (contiguous); shuffle-down offsets 16,8,4,2,1.
        //
        // A: x[0]=2^24, x[1]=1, x[2]=-2^24 all in lane 0.
        //    p = RN(1*2^24) = 2^24; fma(1,1,p) = RN(2^24+1) = 2^24; fma(1,-2^24,p) = 0.
        //    result +0 (bits 0) while the exact f64 value is 1. (A strided layout where lane l
        //    owns column l would put them in lanes 0,1,2 and finish with 1 instead.)
        // C: lanes 0,1,2 hold 2^24, 1, -2^24 (x[0], x[4], x[8]).
        //    off 2 first folds lane2 into lane0: 2^24 + -2^24 = 0, lane1 keeps 1;
        //    off 1: 0 + 1 = 1.  (Ascending lane sum or off=1 before off=2 would give 0.)
        // D: lanes 0,2,4 hold 2^24, 1, -2^24 (x[0], x[8], x[16]).
        //    off 4: lane0 += lane4 -> 0, lane2 stays 1; off 2: 0 + 1 = 1.
        // F: lanes 0,4,8 hold 2^24, 1, -2^24 (x[0], x[16], x[32]).
        //    off 8: lane0 += lane8 -> 0; lane4 stays 1; off 4: 0 + 1 = 1.
        // B: lanes 0,8,16 hold 2^24, -2^24, 1 (x[0], x[32], x[64]).
        //    off 16: lane0 = 2^24 + 1 = 2^24 (1 lost); lane8 = -2^24 (+ lane24 = 0);
        //    off 8: 2^24 + -2^24 = 0. (A sequential or 8-before-16 order gives 1.)
        let cases: [(&str, [(usize, f32); 3], u32); 5] = [
            (
                "A in-lane",
                [(0, TWO24), (1, 1.0), (2, -TWO24)],
                0x0000_0000,
            ),
            (
                "C off2 before off1",
                [(0, TWO24), (4, 1.0), (8, -TWO24)],
                0x3F80_0000,
            ),
            (
                "D off4 before off2",
                [(0, TWO24), (8, 1.0), (16, -TWO24)],
                0x3F80_0000,
            ),
            (
                "F off8 before off4",
                [(0, TWO24), (16, 1.0), (32, -TWO24)],
                0x3F80_0000,
            ),
            (
                "B off16 before off8",
                [(0, TWO24), (32, -TWO24), (64, 1.0)],
                0x0000_0000,
            ),
        ];
        // Down (R128, K128) and gate (R256, K256; data in group 0 -> acc0 only, so the same).
        for (rot, k) in [
            (MqI4Rotation::R128, 128usize),
            (MqI4Rotation::R256, 256usize),
        ] {
            for (name, xs, want_bits) in cases {
                let (f, exact) = run_row(rot, k, &[], &xs);
                assert_eq!(f.to_bits(), want_bits, "{name} {rot:?}: got {f}");
                assert_eq!(exact, 1.0f64, "{name} {rot:?} f64 reference");
            }
        }
    }

    #[test]
    fn decode_in_lane_is_mul_then_three_fma() {
        // Lane 0: w0 = -3 (code 2), x0 = 1; w1 = +3 (code 5), x1 = 1 + 2^-23.
        // p = RN(-3*1) = -3. Then fma(3, 1+2^-23, -3) = 3*2^-23 exactly (see the IU4 fused
        // test) = bits 0x34C00000. Mul-then-add would give 2^-21 = 0x35000000.
        let x1 = f32::from_bits(0x3F80_0001);
        for (rot, k) in [
            (MqI4Rotation::R128, 128usize),
            (MqI4Rotation::R256, 256usize),
        ] {
            let (f, exact) = run_row(rot, k, &[(0, 2), (1, 5)], &[(0, 1.0), (1, x1)]);
            assert_eq!(f.to_bits(), 0x34C0_0000, "{rot:?}: got {f}");
            // exact value: -3 + 3(1+2^-23) = 3*2^-23 (a few f64 ulps never matter: exact in f64)
            assert_eq!(exact, 3.0f64 * 2.0f64.powi(-23));
        }
        // And the separate-multiply result is what a non-fused chain would produce:
        assert_eq!((-3.0f32 + 3.0f32 * x1).to_bits(), 0x3500_0000);
    }

    #[test]
    fn decode_group_adds_are_separate_from_the_fma_chain() {
        // K256. Lane 0: group 0 column 0: w=-3 (code 2), x=1 -> p0 = -3.
        //               group 1 column 128: w=+3 (code 5), x = 1+2^-23 -> p1 = RN(3(1+2^-23)) = 3 + 2^-21
        // (the in-group chain restarts with a plain multiply, the group sum is a plain add).
        //   down (R128): acc0 = RN(RN(0 + p0) + p1) = -3 + 3 + 2^-21 = 2^-21  = bits 0x35000000
        //   gate (R256): acc0 = p0, acc1 = p1; (acc0+acc1)+(acc2+acc3) = 2^-21 + 0 = 0x35000000
        // Continuing the fma chain across groups would have produced 0x34C00000.
        let x1 = f32::from_bits(0x3F80_0001);
        for rot in [MqI4Rotation::R128, MqI4Rotation::R256] {
            let (f, _) = run_row(rot, 256, &[(0, 2), (128, 5)], &[(0, 1.0), (128, x1)]);
            assert_eq!(f.to_bits(), 0x3500_0000, "{rot:?}: got {f}");
        }
    }

    #[test]
    fn decode_gate_four_chains_g_and_3_pair_fold_over_eight_epochs() {
        // K=1024 (8 groups), lane 0 only, weight +1: p_g = v_g with
        //   v = [-3, 3, -0.5, 5, -1, 2, -3, -2^24].
        // Gate (R256) chains acc[g&3]:
        //   acc0 = g0+g4 = -3 + -1  = -4
        //   acc1 = g1+g5 =  3 +  2  =  5
        //   acc2 = g2+g6 = -0.5 + -3 = -3.5
        //   acc3 = g3+g7 =  5 + -2^24 = -16777211 (exact: 2^24 - 5 < 2^24)
        //   (acc0+acc1) = 1;  (acc2+acc3) = -16777214.5 -> tie, ties-to-even -> -16777214
        //   total = 1 + -16777214 = -16777213.
        // Every alternative association gives something else (same data):
        //   single chain -16777214 | g%2 chains -16777214 | (acc0+acc2)+(acc1+acc3) -16777214
        //   ((acc0+acc1)+acc2)+acc3 -16777214 | (acc0+acc3)+(acc1+acc2) -16777214
        //   chains by (g>>1)&3 -16777216 | eight chains tree -16777216 | descending order -16777216
        // Down (R128): one chain, ascending: 0-3=-3; +3=0; -0.5; +5=4.5; -1=3.5; +2=5.5;
        //   -3=2.5; 2.5 + -2^24 = -16777213.5 (tie) -> -16777214.
        // The exact value is -16777213.5.
        let v: [f32; 8] = [-3.0, 3.0, -0.5, 5.0, -1.0, 2.0, -3.0, -TWO24];
        let xs: Vec<(usize, f32)> = v.iter().enumerate().map(|(g, &p)| (g * 128, p)).collect();
        let (gate, gate_f64) = run_row(MqI4Rotation::R256, 1024, &[], &xs);
        let (down, down_f64) = run_row(MqI4Rotation::R128, 1024, &[], &xs);
        assert_eq!(gate, -16_777_213.0f32);
        assert_eq!(gate.to_bits(), 0xCB7F_FFFD);
        assert_eq!(down, -16_777_214.0f32);
        assert_eq!(down.to_bits(), 0xCB7F_FFFE);
        assert_eq!(gate_f64, -16_777_213.5f64);
        assert_eq!(down_f64, -16_777_213.5f64);
    }

    // =======================================================================
    // Random vectors vs f64 with derived error bounds
    // =======================================================================

    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            (self.next() >> 11) % n
        }
        /// f16 bits with biased exponent 12..=16 (2^-3 .. 2^1), random mantissa: never 0,
        /// never subnormal, never overflowing.
        fn scale_bits(&mut self) -> u16 {
            let e = 12 + self.below(5) as u16;
            (e << 10) | self.below(1024) as u16
        }
        /// f32 in +-[2^-2, 2^2) with a random 23-bit mantissa.
        fn x(&mut self) -> f32 {
            let sign = self.below(2) as u32;
            let e = 125 + self.below(4) as u32;
            f32::from_bits((sign << 31) | (e << 23) | self.below(1 << 23) as u32)
        }
        /// Positive f32 in [2^-4, 2^0) with a random mantissa (activation block scale).
        fn d(&mut self) -> f32 {
            let e = 123 + self.below(4) as u32;
            f32::from_bits((e << 23) | self.below(1 << 23) as u32)
        }
    }

    struct RandTensor {
        codes: Vec<u8>,
        scales: Vec<u16>,
        bytes: Vec<u8>,
    }

    fn rand_tensor(rng: &mut XorShift, profile: MqI4Profile, m: usize, k: usize) -> RandTensor {
        let levels = u64::from(profile.codec.levels()) + 1;
        let codes: Vec<u8> = (0..m * k).map(|_| rng.below(levels) as u8).collect();
        let scales: Vec<u16> = (0..m * (k / 128)).map(|_| rng.scale_bits()).collect();
        let bytes = enc(profile, m, k, &codes, &scales);
        RandTensor {
            codes,
            scales,
            bytes,
        }
    }

    fn widths(profile: MqI4Profile) -> [usize; 3] {
        match profile.rotation {
            MqI4Rotation::R256 => [256, 512, 768],
            MqI4Rotation::R128 => [128, 384, 640],
        }
    }

    /// Seeded corpus: the extracted `gemv_tree_f32_q` helper and the
    /// `decode_gemv_f32_parts` wrapper are bit-equal to the pre-refactor tree
    /// body ([`tree_row_pre_refactor`]) on every profile and width, including
    /// extreme scales (`H_MAX`, `H_MIN_SUB`, zero) and zero activations.
    #[test]
    fn extracted_tree_helper_bit_equal_pre_refactor_body() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut rows = 0usize;
        for profile in MqI4Profile::ALL {
            for k in widths(profile) {
                for _case in 0..8 {
                    let m = 5usize;
                    let levels = u64::from(profile.codec.levels()) + 1;
                    let codes: Vec<u8> = (0..m * k).map(|_| rng.below(levels) as u8).collect();
                    let scales: Vec<u16> = (0..m * (k / 128))
                        .map(|_| match rng.below(8) {
                            0 => H_MAX,
                            1 => H_MIN_SUB,
                            2 => 0,
                            _ => rng.scale_bits(),
                        })
                        .collect();
                    let x: Vec<f32> = (0..k)
                        .map(|_| if rng.below(16) == 0 { 0.0 } else { rng.x() })
                        .collect();
                    let groups = k / 128;
                    let want: Vec<f32> = (0..m)
                        .map(|r| {
                            tree_row_pre_refactor(
                                profile.codec,
                                profile.rotation,
                                &codes[r * k..(r + 1) * k],
                                &scales[r * groups..(r + 1) * groups],
                                &x,
                            )
                        })
                        .collect();
                    let got = decode_gemv_f32_parts(profile, m, k, &codes, &scales, &x);
                    assert_eq!(
                        bits32(&got),
                        bits32(&want),
                        "wrapper {} k={k}",
                        profile.name()
                    );
                    let q: Vec<i8> = codes.iter().map(|&c| weight_i8(profile.codec, c)).collect();
                    let d: Vec<f32> = scales.iter().map(|&b| scale_f32(b)).collect();
                    let direct = gemv_tree_f32_q(profile.rotation, m, k, &q, &d, &x);
                    assert_eq!(
                        bits32(&direct),
                        bits32(&want),
                        "helper {} k={k}",
                        profile.name()
                    );
                    rows += m;
                }
            }
        }
        assert_eq!(rows, 4 * 3 * 8 * 5);
    }

    /// gamma_n = n u / (1 - n u).
    fn gamma(n: usize, u: f64) -> f64 {
        let nu = n as f64 * u;
        nu / (1.0 - nu)
    }

    /// Ascending f64 oracle written in the test: `(sum, sum of |products|)` per row.
    /// Each product `d * (2c-L) * x` is exact in f64 (11 + 4 + 24 bits).
    fn f64_rows(
        profile: MqI4Profile,
        m: usize,
        k: usize,
        t: &RandTensor,
        x: &[f32],
    ) -> Vec<(f64, f64)> {
        let groups = k / 128;
        let levels = i32::from(profile.codec.levels());
        (0..m)
            .map(|r| {
                let (mut s, mut a) = (0.0f64, 0.0f64);
                for j in 0..k {
                    let d = f16::from_bits(t.scales[r * groups + j / 128]).to_f64();
                    let w = d * f64::from(2 * i32::from(t.codes[r * k + j]) - levels);
                    let p = w * f64::from(x[j]);
                    s += p;
                    a += p.abs();
                }
                (s, a)
            })
            .collect()
    }

    #[test]
    fn random_dequant_and_decode_vs_f64_within_derived_bounds() {
        // Decode error (per output): every product reaches the result through
        //   4 roundings in the lane (mul + 3 fma) + <= G group adds (G = k/128; the
        //   gate uses <= ceil(G/4) per chain but G is a safe upper bound) +
        //   2 gate fold adds + 5 shuffle adds,
        // i.e. at most n = 4 + G + 5 (+ 2 for R256) roundings, each of relative size
        // <= u = 2^-24, so |f32 - exact| <= gamma_n(2^-24) * sum|products|. The f64
        // oracle itself contributes <= gamma_k(2^-53) * sum|products|. Weights are exact
        // (d*(2c-L) has <= 15 bits) and scales/activations are far from over/underflow
        // (|d| in 2^-3..2^1, |x| in 2^-2..2^2), so no underflow term is needed.
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let (m, u32_, u64_) = (3usize, 2.0f64.powi(-24), 2.0f64.powi(-53));
        for profile in MqI4Profile::ALL {
            for k in widths(profile) {
                let t = rand_tensor(&mut rng, profile, m, k);
                let x: Vec<f32> = (0..k).map(|_| rng.x()).collect();
                let groups = k / 128;
                let n = 4
                    + groups
                    + 5
                    + if profile.rotation == MqI4Rotation::R256 {
                        2
                    } else {
                        0
                    };
                let oracle = f64_rows(profile, m, k, &t, &x);

                // dequant is exact: bits == f32 of the exact f64 value
                let dq = dequant_rotated_f32(profile, m, k, &t.bytes);
                let levels = i32::from(profile.codec.levels());
                for r in 0..m {
                    for j in 0..k {
                        let d = f16::from_bits(t.scales[r * groups + j / 128]).to_f64();
                        let want =
                            (d * f64::from(2 * i32::from(t.codes[r * k + j]) - levels)) as f32;
                        assert_eq!(
                            dq[r * k + j].to_bits(),
                            want.to_bits(),
                            "dequant {} r={r} j={j}",
                            profile.name()
                        );
                    }
                }

                let f32_out = decode_gemv_f32(profile, m, k, &t.bytes, &x);
                let f64_out = gemv_f64(profile, m, k, &t.bytes, &x);
                for r in 0..m {
                    let (s, a) = oracle[r];
                    assert!(a > 0.0);
                    let bound = gamma(n, u32_) * a + gamma(k, u64_) * a;
                    let err = (f64::from(f32_out[r]) - s).abs();
                    assert!(
                        err <= bound,
                        "{} k={k} r={r}: f32 {} vs f64 {s}: err {err} > bound {bound}",
                        profile.name(),
                        f32_out[r]
                    );
                    // gemv_f64 is an ascending f64 sum like the oracle: agree to the f64 bound.
                    let err64 = (f64_out[r] - s).abs();
                    assert!(
                        err64 <= 2.0 * gamma(k, u64_) * a,
                        "{} gemv_f64 r={r}",
                        profile.name()
                    );
                }
            }
        }
    }

    #[test]
    fn random_iu4_vs_f64_within_derived_bounds() {
        // Per output: sum_g fma(RN(da*dw), dot_g, sum): each term carries 1 rounding of the
        // scale product + at most G fma roundings => n = G + 1 roundings of relative size
        // <= u = 2^-24: |err| <= gamma_{G+1}(2^-24) * sum|term_g|. The f64 oracle terms
        // da*dw*dot are exact (24 + 11 + 14 bits); its ascending sum adds <= G*2^-52*sum|term|.
        // dot is an exact int32 (|dot| <= 7168 < 2^22 so float(dot) is exact).
        let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
        let (m, tokens) = (3usize, 2usize);
        for profile in MqI4Profile::ALL {
            for k in widths(profile) {
                let groups = k / 128;
                let t = rand_tensor(&mut rng, profile, m, k);
                let levels = i32::from(profile.codec.levels());
                let mut vals: Vec<[i8; 128]> = Vec::new();
                let mut blocks: Vec<A4Block> = Vec::new();
                for _ in 0..tokens * groups {
                    let mut v = [0i8; 128];
                    for e in v.iter_mut() {
                        *e = rng.below(16) as i8 - 8;
                    }
                    blocks.push(A4Block::from_values(rng.d(), &v));
                    vals.push(v);
                }
                let out = iu4_gemm_f32(profile, m, k, &t.bytes, &blocks);
                assert_eq!(out.len(), tokens * m);
                for tk in 0..tokens {
                    for r in 0..m {
                        let (mut s, mut a) = (0.0f64, 0.0f64);
                        for g in 0..groups {
                            let mut dot = 0i64;
                            for j in 0..128 {
                                let w =
                                    2 * i64::from(t.codes[r * k + g * 128 + j]) - i64::from(levels);
                                dot += w * i64::from(vals[tk * groups + g][j]);
                            }
                            assert!(dot.abs() <= 7168);
                            let dw = f16::from_bits(t.scales[r * groups + g]).to_f64();
                            let term = f64::from(blocks[tk * groups + g].d) * dw * dot as f64;
                            s += term;
                            a += term.abs();
                        }
                        let bound = gamma(groups + 1, 2.0f64.powi(-24)) * a
                            + groups as f64 * 2.0f64.powi(-52) * a;
                        let err = (f64::from(out[tk * m + r]) - s).abs();
                        assert!(
                            err <= bound,
                            "{} k={k} t={tk} r={r}: {} vs {s}: err {err} > {bound}",
                            profile.name(),
                            out[tk * m + r]
                        );
                    }
                }
            }
        }
    }

    // =======================================================================
    // Packed / _parts agreement (explicit self-agreement tests)
    // =======================================================================

    #[test]
    fn packed_and_parts_agree_for_all_apis() {
        let mut rng = XorShift(0xA076_1D64_78BD_642F);
        let (m, tokens) = (3usize, 2usize);
        for profile in MqI4Profile::ALL {
            for k in widths(profile) {
                let groups = k / 128;
                let t = rand_tensor(&mut rng, profile, m, k);
                let x: Vec<f32> = (0..k).map(|_| rng.x()).collect();
                let mut blocks = Vec::new();
                for _ in 0..tokens * groups {
                    let mut v = [0i8; 128];
                    for e in v.iter_mut() {
                        *e = rng.below(16) as i8 - 8;
                    }
                    blocks.push(A4Block::from_values(rng.d(), &v));
                }
                // the unpacked inputs come from the S0 decoder, as the module requires
                let dec = mq_i4::decode_tensor(profile, m, k, &t.bytes).unwrap();
                assert_eq!(dec.codes, t.codes);
                assert_eq!(dec.scale_bits, t.scales);
                let (c, s) = (&dec.codes, &dec.scale_bits);
                assert_eq!(
                    bits32(&dequant_rotated_f32(profile, m, k, &t.bytes)),
                    bits32(&dequant_rotated_f32_parts(profile, m, k, c, s)),
                    "dequant {}",
                    profile.name()
                );
                assert_eq!(
                    bits32(&iu4_gemm_f32(profile, m, k, &t.bytes, &blocks)),
                    bits32(&iu4_gemm_f32_parts(profile, m, k, c, s, &blocks)),
                    "iu4 {}",
                    profile.name()
                );
                assert_eq!(
                    bits32(&decode_gemv_f32(profile, m, k, &t.bytes, &x)),
                    bits32(&decode_gemv_f32_parts(profile, m, k, c, s, &x)),
                    "decode {}",
                    profile.name()
                );
                assert_eq!(
                    bits64(&gemv_f64(profile, m, k, &t.bytes, &x)),
                    bits64(&gemv_f64_parts(profile, m, k, c, s, &x)),
                    "gemv_f64 {}",
                    profile.name()
                );
            }
        }
    }

    // =======================================================================
    // F4 slot routing
    // =======================================================================

    /// Expert whose row `r` has the constant code / scale `rows[r]` across all of k.
    fn const_rows_expert(profile: MqI4Profile, m: usize, k: usize, rows: &[(u8, u16)]) -> Vec<u8> {
        assert_eq!(rows.len(), m);
        let mut codes = Vec::new();
        let mut scales = Vec::new();
        for &(c, s) in rows {
            codes.extend(std::iter::repeat(c).take(k));
            scales.extend(std::iter::repeat(s).take(k / 128));
        }
        enc(profile, m, k, &codes, &scales)
    }

    #[test]
    fn moe_gate_up_routes_to_original_slots_with_distinct_experts() {
        // m=4 (gate rows 0,1; up rows 2,3), k=256, c=2 tokens (original slots 0..20).
        // Table: [0] W3 (unused: empty bucket), [1] None, [2] W2, [3] W3.
        // expert_ids = [2, 0, 3], offsets [0, 2, 2, 3]: bucket 0 = expert 2 owns slots {14, 3}
        // (out of order on purpose), bucket 1 = expert 0 is EMPTY, bucket 2 = expert 3 owns {17}.
        // x row = slot/10: slots 3 -> token 0, 14 & 17 -> token 1. Token t row is the constant
        // (t+1) over k=256 (padding columns hold 1e30 and must never be read) so a row dot is
        // w * 256 * (t+1): token 0 -> 256 w, token 1 -> 512 w.
        // Row weights w = d*(2c-L):
        //   expert 2 (W2): [3*1, -3*0.5, -1*2, 3*0.25]  = [3, -1.5, -2, 0.75]
        //   expert 3 (W3): [7*1, -7*0.5, 1*4, 3*0.125]  = [7, -3.5, 4, 0.375]
        // Outputs land at slot*5 (y_slot_stride 5 > 2 output rows; x_row_stride 260 != 256):
        //   slot 14 (expert 2, 512): gate [1536, -768], up [-1024, 384]
        //   slot  3 (expert 2, 256): gate [ 768, -384], up [ -512, 192]
        //   slot 17 (expert 3, 512): gate [3584, -1792], up [2048, 192]
        // Every other cell keeps its sentinel (-77 in y0, -99 in y1).
        let (m, k, c) = (4usize, 256usize, 2usize);
        let e0 = const_rows_expert(
            P3_R256,
            m,
            k,
            &[(1, H_ONE), (2, H_ONE), (3, H_ONE), (5, H_ONE)],
        );
        let e2 = const_rows_expert(
            P2_R256,
            m,
            k,
            &[(3, H_ONE), (0, H_HALF), (1, H_TWO), (3, H_QUARTER)],
        );
        let e3 = const_rows_expert(
            P3_R256,
            m,
            k,
            &[(7, H_ONE), (0, H_HALF), (4, H_FOUR), (5, H_EIGHTH)],
        );

        let (x_stride, y_stride) = (260usize, 5usize);
        let mut x = vec![1.0e30f32; c * x_stride];
        for t in 0..c {
            for j in 0..k {
                x[t * x_stride + j] = (t + 1) as f32;
            }
        }
        let (ids, offsets, slots) = ([2u32, 0, 3], [0u32, 2, 2, 3], [14u32, 3, 17]);

        let mut want0 = vec![-77.0f32; 20 * y_stride];
        let mut want1 = vec![-99.0f32; 20 * y_stride];
        for (slot, g, u) in [
            (14usize, [1536.0f32, -768.0], [-1024.0f32, 384.0]),
            (3, [768.0, -384.0], [-512.0, 192.0]),
            (17, [3584.0, -1792.0], [2048.0, 192.0]),
        ] {
            want0[slot * y_stride..slot * y_stride + 2].copy_from_slice(&g);
            want1[slot * y_stride..slot * y_stride + 2].copy_from_slice(&u);
        }

        // packed
        let experts = vec![
            Some(ExpertTensor {
                profile: P3_R256,
                bytes: &e0,
            }),
            None,
            Some(ExpertTensor {
                profile: P2_R256,
                bytes: &e2,
            }),
            Some(ExpertTensor {
                profile: P3_R256,
                bytes: &e3,
            }),
        ];
        let mut y0 = vec![-77.0f32; 20 * y_stride];
        let mut y1 = vec![-99.0f32; 20 * y_stride];
        decode_moe_slots(
            DecodeKind::GateUp,
            m,
            k,
            &experts,
            &ids,
            &offsets,
            &slots,
            c,
            &x,
            x_stride,
            y_stride,
            &mut y0,
            Some(&mut y1[..]),
        );
        assert_eq!(bits32(&y0), bits32(&want0), "packed y0");
        assert_eq!(bits32(&y1), bits32(&want1), "packed y1");

        // parts
        let t0 = mq_i4::decode_tensor(P3_R256, m, k, &e0).unwrap();
        let t2 = mq_i4::decode_tensor(P2_R256, m, k, &e2).unwrap();
        let t3 = mq_i4::decode_tensor(P3_R256, m, k, &e3).unwrap();
        let parts = vec![
            Some(ExpertTensorParts {
                profile: P3_R256,
                codes: &t0.codes,
                scale_bits: &t0.scale_bits,
            }),
            None,
            Some(ExpertTensorParts {
                profile: P2_R256,
                codes: &t2.codes,
                scale_bits: &t2.scale_bits,
            }),
            Some(ExpertTensorParts {
                profile: P3_R256,
                codes: &t3.codes,
                scale_bits: &t3.scale_bits,
            }),
        ];
        let mut y0 = vec![-77.0f32; 20 * y_stride];
        let mut y1 = vec![-99.0f32; 20 * y_stride];
        decode_moe_slots_parts(
            DecodeKind::GateUp,
            m,
            k,
            &parts,
            &ids,
            &offsets,
            &slots,
            c,
            &x,
            x_stride,
            y_stride,
            &mut y0,
            Some(&mut y1[..]),
        );
        assert_eq!(bits32(&y0), bits32(&want0), "parts y0");
        assert_eq!(bits32(&y1), bits32(&want1), "parts y1");
    }

    #[test]
    fn moe_down_routes_x_row_equals_slot_with_distinct_experts() {
        // m=3, k=128 (R128), c=2 (original slots 0..20), x row = slot (20 rows, stride 130, padding
        // 1e30), y stride 7 > m=3. Table: [0] W2, [1] W3, [2] None.
        // expert_ids = [1, 0], offsets [0, 2, 3]: expert 1 owns slots {9, 2}, expert 0 owns {16}.
        // Row s of x is the constant (s+1): dot = w * 128 * (s+1).
        //   expert 1 (W3): [7*1, -7*0.5, 1*2]    = [7, -3.5, 2]
        //   expert 0 (W2): [3*1, -3*0.25, 1*8]   = [3, -0.75, 8]
        //   slot 9  (x=10, 1280): [8960, -4480, 2560]
        //   slot 2  (x= 3,  384): [2688, -1344,  768]
        //   slot 16 (x=17, 2176): [6528, -1632, 17408]
        let (m, k, c) = (3usize, 128usize, 2usize);
        let ew2 = const_rows_expert(P2_R128, m, k, &[(3, H_ONE), (0, H_QUARTER), (2, H_EIGHT)]);
        let ew3 = const_rows_expert(P3_R128, m, k, &[(7, H_ONE), (0, H_HALF), (4, H_TWO)]);
        let (x_stride, y_stride) = (130usize, 7usize);
        let mut x = vec![1.0e30f32; 20 * x_stride];
        for s in 0..20 {
            for j in 0..k {
                x[s * x_stride + j] = (s + 1) as f32;
            }
        }
        let (ids, offsets, slots) = ([1u32, 0], [0u32, 2, 3], [9u32, 2, 16]);
        let mut want = vec![-55.0f32; 20 * y_stride];
        for (slot, row) in [
            (9usize, [8960.0f32, -4480.0, 2560.0]),
            (2, [2688.0, -1344.0, 768.0]),
            (16, [6528.0, -1632.0, 17408.0]),
        ] {
            want[slot * y_stride..slot * y_stride + 3].copy_from_slice(&row);
        }

        let experts = vec![
            Some(ExpertTensor {
                profile: P2_R128,
                bytes: &ew2,
            }),
            Some(ExpertTensor {
                profile: P3_R128,
                bytes: &ew3,
            }),
            None,
        ];
        let mut y0 = vec![-55.0f32; 20 * y_stride];
        decode_moe_slots(
            DecodeKind::Down,
            m,
            k,
            &experts,
            &ids,
            &offsets,
            &slots,
            c,
            &x,
            x_stride,
            y_stride,
            &mut y0,
            None,
        );
        assert_eq!(bits32(&y0), bits32(&want), "packed");

        let t_w2 = mq_i4::decode_tensor(P2_R128, m, k, &ew2).unwrap();
        let t_w3 = mq_i4::decode_tensor(P3_R128, m, k, &ew3).unwrap();
        let parts = vec![
            Some(ExpertTensorParts {
                profile: P2_R128,
                codes: &t_w2.codes,
                scale_bits: &t_w2.scale_bits,
            }),
            Some(ExpertTensorParts {
                profile: P3_R128,
                codes: &t_w3.codes,
                scale_bits: &t_w3.scale_bits,
            }),
            None,
        ];
        let mut y0 = vec![-55.0f32; 20 * y_stride];
        decode_moe_slots_parts(
            DecodeKind::Down,
            m,
            k,
            &parts,
            &ids,
            &offsets,
            &slots,
            c,
            &x,
            x_stride,
            y_stride,
            &mut y0,
            None,
        );
        assert_eq!(bits32(&y0), bits32(&want), "parts");
    }

    #[test]
    fn moe_empty_bucket_and_empty_slot_list_write_nothing() {
        let (m, k) = (4usize, 256usize);
        let e = const_rows_expert(P3_R256, m, k, &[(7, H_ONE); 4]);
        let experts = vec![Some(ExpertTensor {
            profile: P3_R256,
            bytes: &e,
        })];
        let x = vec![1.0f32; 256];
        let mut y0 = vec![-77.0f32; 50];
        let mut y1 = vec![-99.0f32; 50];
        // one expert, empty bucket
        decode_moe_slots(
            DecodeKind::GateUp,
            m,
            k,
            &experts,
            &[0],
            &[0, 0],
            &[],
            1,
            &x,
            256,
            5,
            &mut y0,
            Some(&mut y1[..]),
        );
        // zero experts at all
        decode_moe_slots(
            DecodeKind::GateUp,
            m,
            k,
            &experts,
            &[],
            &[0],
            &[],
            1,
            &x,
            256,
            5,
            &mut y0,
            Some(&mut y1[..]),
        );
        assert!(y0.iter().all(|&v| v == -77.0));
        assert!(y1.iter().all(|&v| v == -99.0));
    }

    #[test]
    #[should_panic(expected = "duplicate slot id 3")]
    fn moe_duplicate_slot_panics() {
        let e = const_rows_expert(P3_R128, 1, 128, &[(4, H_ONE)]);
        let experts = vec![Some(ExpertTensor {
            profile: P3_R128,
            bytes: &e,
        })];
        let x = vec![1.0f32; 10 * 128];
        let mut y0 = vec![0.0f32; 10];
        decode_moe_slots(
            DecodeKind::Down,
            1,
            128,
            &experts,
            &[0],
            &[0, 2],
            &[3, 3],
            1,
            &x,
            128,
            1,
            &mut y0,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "slot id 10 >= 10*c")]
    fn moe_slot_outside_ten_c_panics() {
        let e = const_rows_expert(P3_R128, 1, 128, &[(4, H_ONE)]);
        let experts = vec![Some(ExpertTensor {
            profile: P3_R128,
            bytes: &e,
        })];
        let x = vec![1.0f32; 11 * 128];
        let mut y0 = vec![0.0f32; 11];
        decode_moe_slots(
            DecodeKind::Down,
            1,
            128,
            &experts,
            &[0],
            &[0, 1],
            &[10],
            1,
            &x,
            128,
            1,
            &mut y0,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "wrong rotation")]
    fn moe_down_rejects_r256_profile() {
        let e = const_rows_expert(P3_R256, 1, 256, &[(4, H_ONE)]);
        let experts = vec![Some(ExpertTensor {
            profile: P3_R256,
            bytes: &e,
        })];
        let x = vec![1.0f32; 10 * 256];
        let mut y0 = vec![0.0f32; 10];
        decode_moe_slots(
            DecodeKind::Down,
            1,
            256,
            &experts,
            &[0],
            &[0, 1],
            &[0],
            1,
            &x,
            256,
            1,
            &mut y0,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "y1 must be Some for GateUp")]
    fn moe_gate_up_requires_y1() {
        let e = const_rows_expert(P3_R256, 2, 256, &[(4, H_ONE), (4, H_ONE)]);
        let experts = vec![Some(ExpertTensor {
            profile: P3_R256,
            bytes: &e,
        })];
        let x = vec![1.0f32; 256];
        let mut y0 = vec![0.0f32; 10];
        decode_moe_slots(
            DecodeKind::GateUp,
            2,
            256,
            &experts,
            &[0],
            &[0, 1],
            &[0],
            1,
            &x,
            256,
            1,
            &mut y0,
            None,
        );
    }

    // =======================================================================
    // Grouped BF16 epilogue
    // =======================================================================

    /// `[gate..., up...]` rows for m=2: out = BF16(silu(g) * u), independent known
    /// values (silu(g) = g / (1 + e^-g) evaluated in f64, then BF16 RNE; every
    /// candidate is >= 0.19 bf16-ulp away from a rounding boundary, far beyond the
    /// ~1e-7 relative error of the f32 DAG).
    #[test]
    fn gate_silu_bf16_independent_knowns() {
        // (g, u, expected bf16 bits, derivation)
        let cases: [(f32, f32, u16, &str); 14] = [
            (1.0, 1.0, 0x3F3B, "silu(1)=0.7310586 -> 0.73046875"),
            (-1.0, 1.0, 0xBE8A, "silu(-1)=-0.2689414 -> -0.26953125"),
            (
                2.0,
                -0.5,
                0xBF61,
                "silu(2)=1.7615942, *-0.5 = -0.8807971 -> -0.87890625",
            ),
            (
                0.5,
                3.0,
                0x3F6F,
                "silu(.5)=0.3112297, *3 = 0.9336890 -> 0.93359375",
            ),
            (
                -3.0,
                2.0,
                0xBE92,
                "silu(-3)=-0.1422776, *2 = -0.2845552 -> -0.28515625",
            ),
            (10.0, 1.0, 0x4120, "silu(10)=9.99954602 -> 10.0"),
            (-10.0, 1.0, 0xB9EE, "silu(-10)=-4.5398e-4 -> -4.5395e-4"),
            (20.0, -1.0, 0xC1A0, "silu(20)=19.99999996, *-1 -> -20.0"),
            (-20.0, 1.0, 0xB331, "silu(-20)=-4.1223e-8 -> -4.1211e-8"),
            (
                100.0,
                0.5,
                0x4248,
                "e^-100 ~ 3.7e-44 vanishes in 1+r; 100*0.5 = 50",
            ),
            (
                -100.0,
                1.0,
                0x8000,
                "g < -88.72: r = +inf, q = -100/inf = -0, -0*1 = -0",
            ),
            (104.0, 1.0, 0x42D0, "g > 103.28: r forced to 0, q = 104"),
            (
                -0.25,
                4.0,
                0xBEE0,
                "silu(-.25)=-0.1094559, *4 = -0.4378235 -> -0.4375",
            ),
            (
                8.0,
                2.0,
                0x4180,
                "silu(8)=7.99731720, *2 = 15.9946344 -> 16.0",
            ),
        ];
        let mut sums = Vec::new();
        for &(g, u, _, _) in &cases {
            sums.push(g);
            sums.push(u);
        }
        let slots: Vec<i32> = (0..cases.len() as i32).collect();
        let out = grouped_prefill_epilogue(DecodeKind::GateUp, 2, &sums, &slots);
        assert_eq!(out.len(), cases.len());
        for (i, &(g, u, want, why)) in cases.iter().enumerate() {
            assert_eq!(
                out[i], want,
                "g={g} u={u}: got {:#06x} want {want:#06x} ({why})",
                out[i]
            );
        }
    }

    #[test]
    fn gate_zero_infinities_and_nan() {
        // Row = [gate, up] with m=2.
        let rows: [(f32, f32); 7] = [
            (0.0, 1.0),
            (0.0, -1.0),
            (-0.0, 1.0),
            (f32::INFINITY, 2.0),
            (f32::INFINITY, -2.0),
            (f32::NEG_INFINITY, 1.0),
            (f32::NAN, 1.0),
        ];
        let mut sums = Vec::new();
        for &(g, u) in &rows {
            sums.push(g);
            sums.push(u);
        }
        let slots: Vec<i32> = (0..rows.len() as i32).collect();
        let out = grouped_prefill_epilogue(DecodeKind::GateUp, 2, &sums, &slots);
        // g=+0: r = 2^0 = 1, den 2, q = 0/2 = +0, h = +0*1 = +0 -> 0x0000
        assert_eq!(out[0], 0x0000);
        // g=+0, u=-1: h = +0 * -1 = -0 -> 0x8000
        assert_eq!(out[1], 0x8000);
        // g=-0: q = -0/2 = -0, h = -0 * 1 = -0 -> 0x8000
        assert_eq!(out[2], 0x8000);
        // g=+inf (non-finite passes the BF16 rounding unchanged): g > 103.28 -> r = 0, den = 1,
        // q = inf/1 = inf, h = inf * 2 = +inf -> 0x7F80; with u = -2 -> -inf -> 0xFF80.
        assert_eq!(out[3], 0x7F80);
        assert_eq!(out[4], 0xFF80);
        // g=-inf: r = +inf, den = inf, q = -inf/inf = NaN (payload not asserted)
        assert!(is_bf16_nan(out[5]), "got {:#06x}", out[5]);
        // g=NaN: ordered compares are false, r stays NaN, result NaN
        assert!(is_bf16_nan(out[6]), "got {:#06x}", out[6]);
    }

    #[test]
    fn gate_up_inputs_are_bf16_rounded_ties_to_even_before_silu() {
        // g bits 0x3F804000 (1 + 2^-9) rounds to 1.0; u bits 0x3F808000 is a tie between
        // 0x3F80 and 0x3F81 -> even 0x3F80 = 1.0. silu(1)*1 = 0.7310586 -> 0x3F3B.
        //   (Skipping the input rounding would use silu(1.00195)*1.00390625 = 0.73573 -> 0x3F3C.)
        // g = 1.0, u bits 0x3F818000: tie between 0x3F81 and 0x3F82 -> even 0x3F82 = 1.015625;
        //   silu(1)*1.015625 = 0.7424814 -> 0x3F3E (unrounded u 1.01171875 would give 0x3F3D).
        let sums = [
            f32::from_bits(0x3F80_4000),
            f32::from_bits(0x3F80_8000),
            1.0,
            f32::from_bits(0x3F81_8000),
        ];
        let out = grouped_prefill_epilogue(DecodeKind::GateUp, 2, &sums, &[0, 1]);
        assert_eq!(out, vec![0x3F3Bu16, 0x3F3E]);
    }

    #[test]
    fn gate_up_row_layout_and_pads_untouched() {
        // m=4: row = [g0, g1, u0, u1]; out[row] = [silu(g0)*u0, silu(g1)*u1].
        // Pads (-1) are +0 bits; their sums rows are NaN and must not be read.
        //   silu(1)*2    = 1.4621172  -> 0x3FBB
        //   silu(-1)*0.5 = -0.1344707 -> 0xBE0A
        let nan = f32::NAN;
        let sums = [nan, nan, nan, nan, 1.0, -1.0, 2.0, 0.5, nan, nan, nan, nan];
        let out = grouped_prefill_epilogue(DecodeKind::GateUp, 4, &sums, &[-1, 5, -1]);
        assert_eq!(out, vec![0x0000u16, 0x0000, 0x3FBB, 0xBE0A, 0x0000, 0x0000]);
    }

    #[test]
    fn down_bf16_rne_ties_nonfinite_and_pads() {
        // BF16 = high16 of (bits + 0x7fff + ((bits >> 16) & 1)) for finite values.
        //   0x3F808000 tie, even 0x3F80 stays        -> 0x3F80
        //   0x3F818000 tie, odd 0x3F81 rounds up     -> 0x3F82
        //   0xBF808000 / 0xBF818000 (negatives)      -> 0xBF80 / 0xBF82
        //   0x3F808001 just above a tie              -> 0x3F81
        //   0x3F807FFF just below a tie              -> 0x3F80
        //   0x7F7FFFFF f32::MAX carries into the exponent -> 0x7F80
        //   0x00000001 min subnormal                 -> 0x0000
        //   0x00018000 tie, odd 0x0001 rounds up     -> 0x0002
        //   0x00008000 tie, even 0x0000 stays        -> 0x0000
        //   +inf 0x7F800000 / -inf 0xFF800000 / NaN 0x7FC00000 -> high16 unchanged
        let vals: [u32; 13] = [
            0x3F80_8000,
            0x3F81_8000,
            0xBF80_8000,
            0xBF81_8000,
            0x3F80_8001,
            0x3F80_7FFF,
            0x7F7F_FFFF,
            0x0000_0001,
            0x0001_8000,
            0x0000_8000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
        ];
        let want: [u16; 13] = [
            0x3F80, 0x3F82, 0xBF80, 0xBF82, 0x3F81, 0x3F80, 0x7F80, 0x0000, 0x0002, 0x0000, 0x7F80,
            0xFF80, 0x7FC0,
        ];
        let mut sums = vec![f32::NAN; 13]; // pad row first: garbage that must stay unread
        sums.extend(vals.iter().map(|&b| f32::from_bits(b)));
        let out = grouped_prefill_epilogue(DecodeKind::Down, 13, &sums, &[-1, 0]);
        assert_eq!(out.len(), 26);
        assert!(out[..13].iter().all(|&b| b == 0), "pad row must be +0");
        assert_eq!(&out[13..], &want);
    }

    // ---- intrinsic hooks ---------------------------------------------------

    struct Recorder {
        calls: std::cell::RefCell<Vec<(&'static str, u32, u32)>>,
    }

    impl Recorder {
        fn new() -> Self {
            Self {
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    /// Deterministic hook results: exp2 -> 1.5, ldexp -> 0.25, divide -> 0.375.
    impl EpilogueIntrinsics for Recorder {
        fn exp2(&self, x: f32) -> f32 {
            self.calls.borrow_mut().push(("exp2", x.to_bits(), 0));
            1.5
        }
        fn ldexp(&self, x: f32, e: i32) -> f32 {
            self.calls
                .borrow_mut()
                .push(("ldexp", x.to_bits(), e as u32));
            0.25
        }
        fn divide(&self, numerator: f32, denominator: f32) -> f32 {
            self.calls
                .borrow_mut()
                .push(("divide", numerator.to_bits(), denominator.to_bits()));
            0.375
        }
    }

    #[test]
    fn epilogue_hooks_receive_exact_dag_arguments() {
        // DAG (neg_log2e = f32 0xbfb8aa3b, neg_log2e_lo = f32 0xb2a5705f):
        //   ph = RN(c*g), pl = fma(c,g,-ph), pl = fma(c_lo,g,pl), e = rndne(ph),
        //   a = RN(RN(ph-e) + pl), r = ldexp(exp2(a), int(e)), den = RN(1+r), q = divide(g, den), h = RN(q*u).
        // g = 1.0:  ph = c = -1.4426950216 (0xbfb8aa3b), pl = 0 + c_lo = c_lo, e = -1,
        //           ph - e = -0.4426950216 exactly, + pl -> a = -0.4426950514 = bits 0xbee2a8ed.
        // g = -0.5: ph = 0.7213475108 (0x3f38aa3b), pl = c_lo*-0.5 (0x3225705f), e = 1,
        //           ph - e = -0.2786524892, + pl -> a = bits 0xbe8eab8a.
        // Hooks: exp2 -> 1.5 (0x3fc00000), ldexp -> 0.25 so r = 0.25, den = 1.25 (0x3fa00000),
        //        divide -> 0.375, h = 0.375*u: u=2 -> 0.75 (0x3F40), u=4 -> 1.5 (0x3FC0).
        // The pad row (-1) calls nothing (its sums are NaN and unread).
        let nan = f32::NAN;
        let sums = [1.0, 2.0, nan, nan, -0.5, 4.0];
        let hooks = Recorder::new();
        let out = grouped_prefill_epilogue_with(DecodeKind::GateUp, 2, &sums, &[0, -1, 3], &hooks);
        assert_eq!(out, vec![0x3F40u16, 0x0000, 0x3FC0]);
        let want: Vec<(&str, u32, u32)> = vec![
            ("exp2", 0xBEE2_A8ED, 0),
            ("ldexp", 0x3FC0_0000, 0xFFFF_FFFF), // e = -1
            ("divide", 0x3F80_0000, 0x3FA0_0000),
            ("exp2", 0xBE8E_AB8A, 0),
            ("ldexp", 0x3FC0_0000, 1), // e = +1
            ("divide", 0xBF00_0000, 0x3FA0_0000),
        ];
        assert_eq!(*hooks.calls.borrow(), want);
    }

    #[test]
    fn epilogue_hooks_see_clamped_r_in_divide_and_down_calls_none() {
        // g = 104 > f32 0x42ce8ed0 (103.28): r is forced to +0, so den = RN(1 + 0) = 1.0
        // (0x3F800000) and the divide hook receives (104.0, 1.0) regardless of what ldexp returned.
        // g = -100 < f32 0xc2b17218 (-88.72): r is forced to +inf, den = inf (0x7F800000).
        // divide is always the last hook of a row.
        for (g, num, den) in [
            (104.0f32, 0x42D0_0000u32, 0x3F80_0000u32),
            (-100.0, 0xC2C8_0000, 0x7F80_0000),
        ] {
            let hooks = Recorder::new();
            let out = grouped_prefill_epilogue_with(DecodeKind::GateUp, 2, &[g, 1.0], &[0], &hooks);
            // divide hook returns 0.375, h = 0.375 * 1 -> 0x3EC0
            assert_eq!(out, vec![0x3EC0u16]);
            assert_eq!(*hooks.calls.borrow().last().unwrap(), ("divide", num, den));
        }
        // Down never touches the intrinsics.
        let hooks = Recorder::new();
        let out = grouped_prefill_epilogue_with(DecodeKind::Down, 2, &[1.0, -2.0], &[0], &hooks);
        assert_eq!(out, vec![0x3F80u16, 0xC000]);
        assert!(hooks.calls.borrow().is_empty());
    }
}
