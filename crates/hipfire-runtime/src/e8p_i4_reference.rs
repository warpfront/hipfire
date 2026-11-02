// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU arithmetic oracles for the SubQ4 **E8P-on-IU4** route (SUBQ4-DESIGN v2
//! §5, v1 §4.2/§4.3; slice K-E0, lane-pair GEMV slice E1). Nothing here
//! dispatches or is used by a default path: these are the frozen definitions a
//! future GPU unpack / fold / epoch / GEMV kernel is compared against. Only
//! `std`, `half` and the frozen F4 tree of [`crate::mq_i4_reference`] are used.
//!
//! # E1 lane-pair GEMV API
//!
//! * [`pack_word`] / [`unpack_word_fields`]: wire word `idx | sign7 << 8 |
//!   shift << 15` (`u16`, 8 weights).
//! * [`decode8_lanes`]`(idx, sign7, shift, half) -> [i8; 4]`: the four `v`
//!   lanes one GEMV lane owns; `half` 0/1 equal `decode8[0..4]` / `[4..8]`.
//!   Decodes the whole word with the cached SWAR tables (lane pairs duplicate
//!   the word decode).
//! * [`decode_weights_q`] / [`decode_weights_f32`]: the decoded `v` / the F32
//!   weights `(f32(d)·0.25)·v` of a words plane.
//! * [`decode_gemv_f32_lanepair`]`(rotation, m, k, words, scale_bits, x)`: the
//!   GEMV twin. `words` row-major `m*k/8` `u16`, `scale_bits` row-major
//!   `m*k/128` FP16 bits, `x` `k` F32; `R256` = four accumulator chains, `R128`
//!   = one. It calls `mq_i4_reference::gemv_tree_f32_q` (the frozen F4 tree,
//!   unchanged; `decode_gemv_f32` is the same tree on MQ codes), with integer
//!   weights `v` and per-group scale `f32(d)·0.25` (exact in f32), so the
//!   weight is `RN((f32(d)·0.25)·v)`. No `α·s` fold is involved.
//!
//! **MQ3 comparison limits.** `mq_i4_reference::decode_gemv_f32{,_parts}`
//! takes MQ codes `c <= L` (`w = d·(2c − L)`, odd `|w| <= 7` for W3) and an
//! f16 scale per group, so it cannot express the E8P weights `|v| ∈ {9, 11}`
//! (a group with weights `d/4·{1, 11}` needs a 11:1 ratio; W3 offers at most
//! 7:1), nor an f16 `d/4` for subnormal or low-mantissa small `d`. Bit
//! equality against that API is therefore tested on the expressible subset
//! (E8P rows whose abs grid has no `5`: all `|v| <= 7`), with two exact
//! constructions: scale `d/4` as f16 when representable, or scale `d`
//! with `x/4` (power-of-two, every product/FMA real value identical). The
//! full range is tested through `gemv_tree_f32_q`, the very function the MQ
//! twin calls.
//!
//! # Wire word (16 bits per 8 weights)
//!
//! `idx: u8` (row of the 256-entry E8P abs grid, [`E8P_ABS2`]), `sign7: u8`
//! (bit `k` set = lane `k` negative, lanes `0..=6`; `< 128`) and
//! `shift: bool` (the QuIP# `±1/4` shift bit; **`true` = `-1/4`**, i.e. bit 7
//! of the `signs` byte of `vq_books.py::Book.nearest`). Lane 7's sign is never
//! stored.
//!
//! # Parity rule
//!
//! The grid holds half-integer `|g|` (E8 coset `D8 + 1/2`). A signed vector
//! is an E8 point iff its coordinate sum is even, i.e.
//! `#neg ≡ Σ|g| (mod 2)`. `Σ2|g|` is always even, so the rule is on `Σ|g|`
//! (**not** `Σ2|g|`): 149 of 256 rows have odd `Σ|g|` ("odd rows"). Lane 7 is
//! negative iff `popcount(sign7) + Σ|g|` is odd.
//!
//! # Integer decode and the `2C+s` route
//!
//! [`decode8`] returns `v = 4·codeword`: odd, in `[-11, 11]` (`4g = 2·(2g)`,
//! `2g ∈ {±1, ±3, ±5}`, plus `4·(±1/4) = ±1`). The IU4 operand is
//! `o = (v - 1) / 2 ∈ [-6, 5]` (arithmetic shift), `v = 2o + 1`, so with
//! `C = Σ o·q` (the unchanged WMMA chain) and `s = Σ q` (the activation
//! record's `i32 s` field, byte 4 of `block_i4_128`)
//! `Σ v·q = 2C + s`, `|2C + s| ≤ 11·8·128 = 11 264 < 2^22`.
//!
//! # SWAR unpack ([`unpack_word_swar`])
//!
//! ```text
//! word = LUT[idx]                 // 8 nibbles of 2|g| (1,3,5); lane 7 stored
//!                                 // two's-complement negated on odd rows
//! m    = MASK[sign7]              // 0xF in negated lanes; lane 7 = popcount(sign7) & 1
//! w    = (word ^ m) + (m & 0x11111111)   // negate: 2|g| >= 1, no carry between nibbles
//! w   -= shift ? 0x11111111 : 0          // o = 2g' - [shift]; every nibble odd >= 1, no borrow
//! ```
//!
//! `LUT` (1 024 B) + `MASK` (512 B) are the 1 536-B per-CTA stage
//! ([`build_lut`], [`build_mask`]). Nibble `k` of the result (bits `4k..4k+4`,
//! low nibble first, as `mq_i4_reference::A4Block::element`) is lane `k`'s `o`
//! in two's complement.
//!
//! # Fold and epoch ([`fold128`], [`epoch_sum`])
//!
//! The WMMA chain accumulator is seeded with [`MAGIC`] (`bits(1.5·2^23)`), so
//! after the K128 epoch it holds `a = MAGIC + C` (mod 2^32). The fold is the
//! single `v_lshl_add` `a' = 2a + (s - MAGIC) = MAGIC + 2C + s` (mod 2^32).
//! `t = as_f32(a') - as_f32(MAGIC)` is exact for `|2C+s| < 2^22`, and
//! `sum = fma(RN(d_act · f32(d)), t · 0.25, sum)`. `t · 0.25` is exact for
//! `|t| ≤ 11 264`. The wire stores only FP16 `d`; **no FP16 `d/4` is ever
//! formed** (see [`writer_guard`] for the range that keeps the decode
//! `(f32(d)·0.25)·v` exact for every `v`).
//!
//! Source of the constants and fold DAG:
//! `crates/peacemaker-isa/src/kernels/iu4_fold.rs:6-19` (`MAGIC`, `XBLK_BYTES`).
//! The grid is `vq/code/vq_books.json` (`e8p_abs`, file sha256
//! [`E8P_BOOKS_SHA256`]); when `HIPFIRE_VQ_BOOKS_JSON=<path>` is set the tests
//! re-read that file, check its sha256 and compare against [`E8P_ABS2`].

use crate::mq_i4::MqI4Rotation;
use crate::mq_i4_reference::gemv_tree_f32_q;
use half::f16;

/// int32 bit pattern of `12582912.0f` (`1.5·2^23`): the WMMA chain seed.
/// Mirror of `peacemaker-isa::kernels::iu4_fold::MAGIC` (no dependency edge).
pub const MAGIC: u32 = 0x4b40_0000;

/// sha256 of `vq/code/vq_books.json`, the source of [`E8P_ABS2`].
pub const E8P_BOOKS_SHA256: &str =
    "bfe91a2d8bae4bb91e3a07bd22ee61b96675795d2ddf820303f27cfc37ba5da8";

/// Smallest FP16 bit pattern the writer guard accepts: `2^-14` (the smallest
/// normal).
pub const GUARD_MIN_D_BITS: u16 = 0x0400;

/// `2|g|` of the 256 QuIP# E8P abs-grid rows, in `vq_books.json` order
/// (`idx` = row). Every entry is 1, 3 or 5.
#[rustfmt::skip]
pub static E8P_ABS2: [[u8; 8]; 256] = [
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 1, 1, 5], [1, 1, 1, 1, 1, 1, 3, 1],
    [1, 1, 1, 1, 1, 1, 3, 3], [1, 1, 1, 1, 1, 1, 3, 5], [1, 1, 1, 1, 1, 1, 5, 1], [1, 1, 1, 1, 1, 1, 5, 3],
    [1, 1, 1, 1, 1, 3, 1, 1], [1, 1, 1, 1, 1, 3, 1, 3], [1, 1, 1, 1, 1, 3, 1, 5], [1, 1, 1, 1, 1, 3, 3, 1],
    [1, 1, 1, 1, 1, 3, 3, 3], [1, 1, 1, 1, 1, 3, 5, 1], [1, 1, 1, 1, 1, 5, 1, 1], [1, 1, 1, 1, 1, 5, 1, 3],
    [1, 1, 1, 1, 1, 5, 3, 1], [1, 1, 1, 1, 3, 1, 1, 1], [1, 1, 1, 1, 3, 1, 1, 3], [1, 1, 1, 1, 3, 1, 1, 5],
    [1, 1, 1, 1, 3, 1, 3, 1], [1, 1, 1, 1, 3, 1, 3, 3], [1, 1, 1, 1, 3, 1, 5, 1], [1, 1, 1, 1, 3, 3, 1, 1],
    [1, 1, 1, 1, 3, 3, 1, 3], [1, 1, 1, 1, 3, 3, 3, 1], [1, 1, 1, 1, 3, 3, 3, 3], [1, 1, 1, 1, 3, 5, 1, 1],
    [1, 1, 1, 1, 5, 1, 1, 1], [1, 1, 1, 1, 5, 1, 1, 3], [1, 1, 1, 1, 5, 1, 3, 1], [1, 1, 1, 1, 5, 3, 1, 1],
    [1, 1, 1, 3, 1, 1, 1, 1], [1, 1, 1, 3, 1, 1, 1, 3], [1, 1, 1, 3, 1, 1, 1, 5], [1, 1, 1, 3, 1, 1, 3, 1],
    [1, 1, 1, 3, 1, 1, 3, 3], [1, 1, 1, 3, 1, 1, 5, 1], [1, 1, 1, 3, 1, 3, 1, 1], [1, 1, 1, 3, 1, 3, 1, 3],
    [1, 1, 1, 3, 1, 3, 3, 1], [1, 1, 1, 3, 1, 3, 3, 3], [1, 1, 1, 3, 1, 5, 1, 1], [1, 1, 1, 3, 3, 1, 1, 1],
    [1, 1, 1, 3, 3, 1, 1, 3], [1, 1, 1, 3, 3, 1, 3, 1], [1, 1, 1, 3, 3, 1, 3, 3], [1, 1, 1, 3, 3, 3, 1, 1],
    [1, 1, 1, 3, 3, 3, 1, 3], [1, 1, 1, 3, 3, 3, 3, 1], [1, 1, 1, 3, 5, 1, 1, 1], [1, 1, 1, 5, 1, 1, 1, 1],
    [1, 1, 1, 5, 1, 1, 1, 3], [1, 1, 1, 5, 1, 1, 3, 1], [1, 1, 1, 5, 1, 3, 1, 1], [1, 1, 1, 5, 3, 1, 1, 1],
    [1, 1, 3, 1, 1, 1, 1, 1], [1, 1, 3, 1, 1, 1, 1, 3], [1, 1, 3, 1, 1, 1, 1, 5], [1, 1, 3, 1, 1, 1, 3, 1],
    [1, 1, 3, 1, 1, 1, 3, 3], [1, 1, 3, 1, 1, 1, 5, 1], [1, 1, 3, 1, 1, 3, 1, 1], [1, 1, 3, 1, 1, 3, 1, 3],
    [1, 1, 3, 1, 1, 3, 3, 1], [1, 1, 3, 1, 1, 3, 3, 3], [1, 1, 3, 1, 1, 5, 1, 1], [1, 1, 3, 1, 3, 1, 1, 1],
    [1, 1, 3, 1, 3, 1, 1, 3], [1, 1, 3, 1, 3, 1, 3, 1], [1, 1, 3, 1, 3, 1, 3, 3], [1, 1, 3, 1, 3, 3, 1, 1],
    [1, 1, 3, 1, 3, 3, 1, 3], [1, 1, 3, 1, 3, 3, 3, 1], [1, 1, 3, 1, 5, 1, 1, 1], [1, 1, 3, 3, 1, 1, 1, 1],
    [1, 1, 3, 3, 1, 1, 1, 3], [1, 1, 3, 3, 1, 1, 3, 1], [1, 1, 3, 3, 1, 1, 3, 3], [1, 1, 3, 3, 1, 3, 1, 1],
    [1, 1, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 1], [1, 1, 3, 3, 3, 1, 1, 1], [1, 1, 3, 3, 3, 1, 1, 3],
    [1, 1, 3, 3, 3, 1, 3, 1], [1, 1, 3, 3, 3, 3, 1, 1], [1, 1, 3, 5, 1, 1, 1, 1], [1, 1, 5, 1, 1, 1, 1, 1],
    [1, 1, 5, 1, 1, 1, 1, 3], [1, 1, 5, 1, 1, 1, 3, 1], [1, 1, 5, 1, 1, 3, 1, 1], [1, 1, 5, 1, 3, 1, 1, 1],
    [1, 1, 5, 3, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1, 3], [1, 3, 1, 1, 1, 1, 1, 5],
    [1, 3, 1, 1, 1, 1, 3, 1], [1, 3, 1, 1, 1, 1, 3, 3], [1, 3, 1, 1, 1, 1, 5, 1], [1, 3, 1, 1, 1, 3, 1, 1],
    [1, 3, 1, 1, 1, 3, 1, 3], [1, 3, 1, 1, 1, 3, 3, 1], [1, 3, 1, 1, 1, 3, 3, 3], [1, 3, 1, 1, 1, 5, 1, 1],
    [1, 3, 1, 1, 3, 1, 1, 1], [1, 3, 1, 1, 3, 1, 1, 3], [1, 3, 1, 1, 3, 1, 3, 1], [1, 3, 1, 1, 3, 1, 3, 3],
    [1, 3, 1, 1, 3, 3, 1, 1], [1, 3, 1, 1, 3, 3, 1, 3], [1, 3, 1, 1, 3, 3, 3, 1], [1, 3, 1, 1, 5, 1, 1, 1],
    [1, 3, 1, 3, 1, 1, 1, 1], [1, 3, 1, 3, 1, 1, 1, 3], [1, 3, 1, 3, 1, 1, 3, 1], [1, 3, 1, 3, 1, 1, 3, 3],
    [1, 3, 1, 3, 1, 3, 1, 1], [1, 3, 1, 3, 1, 3, 1, 3], [1, 3, 1, 3, 1, 3, 3, 1], [1, 3, 1, 3, 3, 1, 1, 1],
    [1, 3, 1, 3, 3, 1, 1, 3], [1, 3, 1, 3, 3, 1, 3, 1], [1, 3, 1, 3, 3, 3, 1, 1], [1, 3, 1, 5, 1, 1, 1, 1],
    [1, 3, 3, 1, 1, 1, 1, 1], [1, 3, 3, 1, 1, 1, 1, 3], [1, 3, 3, 1, 1, 1, 3, 1], [1, 3, 3, 1, 1, 1, 3, 3],
    [1, 3, 3, 1, 1, 3, 1, 1], [1, 3, 3, 1, 1, 3, 1, 3], [1, 3, 3, 1, 1, 3, 3, 1], [1, 3, 3, 1, 3, 1, 1, 1],
    [1, 3, 3, 1, 3, 1, 1, 3], [1, 3, 3, 1, 3, 1, 3, 1], [1, 3, 3, 1, 3, 3, 1, 1], [1, 3, 3, 3, 1, 1, 1, 1],
    [1, 3, 3, 3, 1, 1, 1, 3], [1, 3, 3, 3, 1, 1, 3, 1], [1, 3, 3, 3, 1, 3, 1, 1], [1, 3, 3, 3, 3, 1, 1, 1],
    [1, 3, 5, 1, 1, 1, 1, 1], [1, 5, 1, 1, 1, 1, 1, 1], [1, 5, 1, 1, 1, 1, 1, 3], [1, 5, 1, 1, 1, 1, 3, 1],
    [1, 5, 1, 1, 1, 3, 1, 1], [1, 5, 1, 1, 3, 1, 1, 1], [1, 5, 1, 3, 1, 1, 1, 1], [1, 5, 3, 1, 1, 1, 1, 1],
    [3, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 3], [3, 1, 1, 1, 1, 1, 1, 5], [3, 1, 1, 1, 1, 1, 3, 1],
    [3, 1, 1, 1, 1, 1, 3, 3], [3, 1, 1, 1, 1, 1, 5, 1], [3, 1, 1, 1, 1, 3, 1, 1], [3, 1, 1, 1, 1, 3, 1, 3],
    [3, 1, 1, 1, 1, 3, 3, 1], [3, 1, 1, 1, 1, 3, 3, 3], [3, 1, 1, 1, 1, 5, 1, 1], [3, 1, 1, 1, 3, 1, 1, 1],
    [3, 1, 1, 1, 3, 1, 1, 3], [3, 1, 1, 1, 3, 1, 3, 1], [3, 1, 1, 1, 3, 1, 3, 3], [3, 1, 1, 1, 3, 3, 1, 1],
    [3, 1, 1, 1, 3, 3, 1, 3], [3, 1, 1, 1, 3, 3, 3, 1], [3, 1, 1, 1, 5, 1, 1, 1], [3, 1, 1, 3, 1, 1, 1, 1],
    [3, 1, 1, 3, 1, 1, 1, 3], [3, 1, 1, 3, 1, 1, 3, 1], [3, 1, 1, 3, 1, 1, 3, 3], [3, 1, 1, 3, 1, 3, 1, 1],
    [3, 1, 1, 3, 1, 3, 1, 3], [3, 1, 1, 3, 1, 3, 3, 1], [3, 1, 1, 3, 3, 1, 1, 1], [3, 1, 1, 3, 3, 1, 1, 3],
    [3, 1, 1, 3, 3, 1, 3, 1], [3, 1, 1, 3, 3, 3, 1, 1], [3, 1, 1, 5, 1, 1, 1, 1], [3, 1, 3, 1, 1, 1, 1, 1],
    [3, 1, 3, 1, 1, 1, 1, 3], [3, 1, 3, 1, 1, 1, 3, 1], [3, 1, 3, 1, 1, 1, 3, 3], [3, 1, 3, 1, 1, 3, 1, 1],
    [3, 1, 3, 1, 1, 3, 1, 3], [3, 1, 3, 1, 1, 3, 3, 1], [3, 1, 3, 1, 3, 1, 1, 1], [3, 1, 3, 1, 3, 1, 1, 3],
    [3, 1, 3, 1, 3, 1, 3, 1], [3, 1, 3, 1, 3, 3, 1, 1], [3, 1, 3, 3, 1, 1, 1, 1], [3, 1, 3, 3, 1, 1, 1, 3],
    [3, 1, 3, 3, 1, 1, 3, 1], [3, 1, 3, 3, 1, 3, 1, 1], [3, 1, 3, 3, 3, 1, 1, 1], [3, 1, 5, 1, 1, 1, 1, 1],
    [3, 3, 1, 1, 1, 1, 1, 1], [3, 3, 1, 1, 1, 1, 1, 3], [3, 3, 1, 1, 1, 1, 3, 1], [3, 3, 1, 1, 1, 1, 3, 3],
    [3, 3, 1, 1, 1, 3, 1, 1], [3, 3, 1, 1, 1, 3, 1, 3], [3, 3, 1, 1, 1, 3, 3, 1], [3, 3, 1, 1, 3, 1, 1, 1],
    [3, 3, 1, 1, 3, 1, 1, 3], [3, 3, 1, 1, 3, 1, 3, 1], [3, 3, 1, 1, 3, 3, 1, 1], [3, 3, 1, 3, 1, 1, 1, 1],
    [3, 3, 1, 3, 1, 1, 1, 3], [3, 3, 1, 3, 1, 1, 3, 1], [3, 3, 1, 3, 1, 3, 1, 1], [3, 3, 1, 3, 3, 1, 1, 1],
    [3, 3, 3, 1, 1, 1, 1, 1], [3, 3, 3, 1, 1, 1, 1, 3], [3, 3, 3, 1, 1, 1, 3, 1], [3, 3, 3, 1, 1, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 1, 1], [3, 3, 3, 3, 1, 1, 1, 1], [3, 5, 1, 1, 1, 1, 1, 1], [5, 1, 1, 1, 1, 1, 1, 1],
    [5, 1, 1, 1, 1, 1, 1, 3], [5, 1, 1, 1, 1, 1, 3, 1], [5, 1, 1, 1, 1, 3, 1, 1], [5, 1, 1, 1, 3, 1, 1, 1],
    [5, 1, 1, 3, 1, 1, 1, 1], [5, 1, 3, 1, 1, 1, 1, 1], [5, 3, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 3, 3, 3, 3],
    [1, 3, 1, 1, 3, 3, 3, 3], [1, 1, 3, 1, 3, 3, 3, 3], [1, 1, 1, 3, 3, 3, 3, 3], [3, 3, 3, 1, 3, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 3, 1], [3, 3, 3, 1, 1, 3, 3, 1], [3, 3, 3, 1, 3, 1, 1, 3], [3, 3, 3, 1, 1, 3, 1, 3],
    [3, 3, 3, 1, 1, 1, 3, 3], [3, 3, 1, 3, 3, 3, 1, 1], [3, 3, 1, 3, 3, 1, 3, 1], [3, 3, 1, 3, 1, 3, 3, 1],
    [3, 3, 1, 3, 3, 1, 1, 3], [3, 3, 1, 3, 1, 3, 1, 3], [3, 3, 1, 3, 1, 1, 3, 3], [3, 1, 3, 3, 3, 3, 1, 1],
    [3, 1, 3, 3, 3, 1, 3, 1], [3, 1, 3, 3, 1, 3, 3, 1], [3, 1, 3, 3, 3, 1, 1, 3], [3, 1, 3, 3, 1, 3, 1, 3],
    [1, 3, 3, 3, 1, 1, 3, 3], [1, 3, 3, 3, 3, 3, 1, 1], [1, 3, 3, 3, 3, 1, 3, 1], [1, 3, 3, 3, 1, 3, 3, 1],
    [1, 3, 3, 3, 3, 1, 1, 3], [1, 3, 3, 3, 1, 3, 1, 3], [1, 1, 3, 3, 1, 3, 3, 3], [3, 3, 1, 1, 3, 3, 3, 1],
];

/// Writer-guard refusal: the scale cannot be carried by the E8P-I4 wire and
/// the tensor/group must fall back to MQ4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    /// `d` is an FP16 infinity or NaN.
    NonFinite { d: u16 },
    /// `d < 2^-14` (zero, subnormal, or negative).
    BelowMin { d: u16 },
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reject::NonFinite { d } => {
                write!(f, "E8P-I4 writer guard: non-finite scale d=0x{d:04x}")
            }
            Reject::BelowMin { d } => write!(f, "E8P-I4 writer guard: scale d=0x{d:04x} < 2^-14"),
        }
    }
}

impl std::error::Error for Reject {}

/// Parity of `Σ|g|` for abs row `idx` (1 = "odd row": 149 of 256).
#[inline]
pub const fn abs_row_parity(idx: u8) -> u8 {
    let r = &E8P_ABS2[idx as usize];
    let mut s = 0u32;
    let mut k = 0;
    while k < 8 {
        s += r[k] as u32;
        k += 1;
    }
    // Σ2|g| is even (eight odd terms); Σ|g| = Σ2|g| / 2.
    ((s / 2) & 1) as u8
}

/// The 256-word nibble LUT: nibble `k` = `2|g_k|`; lane 7 is stored
/// two's-complement negated (`(-2|g_7|) & 0xF`) on odd rows.
pub const fn build_lut() -> [u32; 256] {
    let mut lut = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let r = &E8P_ABS2[i];
        let odd = abs_row_parity(i as u8) == 1;
        let mut w = 0u32;
        let mut k = 0;
        while k < 8 {
            let mut n = r[k] as u32;
            if k == 7 && odd {
                n = n.wrapping_neg() & 0xF;
            }
            w |= n << (4 * k);
            k += 1;
        }
        lut[i] = w;
        i += 1;
    }
    lut
}

/// The 128-word sign mask: lane `k < 7` = `0xF` iff bit `k` of `sign7`; lane 7
/// = `0xF` iff `popcount(sign7)` is odd.
pub const fn build_mask() -> [u32; 128] {
    let mut mask = [0u32; 128];
    let mut s = 0usize;
    while s < 128 {
        let mut w = 0u32;
        let mut k = 0;
        while k < 7 {
            if (s >> k) & 1 == 1 {
                w |= 0xF << (4 * k);
            }
            k += 1;
        }
        if (s as u32).count_ones() & 1 == 1 {
            w |= 0xF << 28;
        }
        mask[s] = w;
        s += 1;
    }
    mask
}

/// `4·codeword` of one wire word: eight odd integers in `[-11, 11]`.
///
/// Lane `k` is `2·(±2|g_k|) + (shift ? -1 : +1)`, with lane 7's sign from the
/// parity rule. Panics if `sign7 >= 128`.
pub fn decode8(idx: u8, sign7: u8, shift: bool) -> [i8; 8] {
    assert!(
        sign7 < 128,
        "e8p_i4_reference::decode8: sign7 {sign7} >= 128"
    );
    let a2 = &E8P_ABS2[idx as usize];
    let neg7 = (sign7.count_ones() as u8 + abs_row_parity(idx)) & 1;
    let mut v = [0i8; 8];
    for k in 0..8 {
        let neg = (if k < 7 { (sign7 >> k) & 1 } else { neg7 }) == 1;
        let n = a2[k] as i8;
        let n = if neg { -n } else { n };
        v[k] = 2 * n + if shift { -1 } else { 1 };
    }
    v
}

/// Reference unpack: eight `o = (v - 1) >> 1` nibbles (two's complement,
/// lane `k` at bits `4k..4k+4`) of [`decode8`].
pub fn unpack_word_ref(idx: u8, sign7: u8, shift: bool) -> u32 {
    let v = decode8(idx, sign7, shift);
    let mut w = 0u32;
    for k in 0..8 {
        let o = (v[k] - 1) >> 1;
        debug_assert!((-6..=5).contains(&o));
        w |= ((o as u8 & 0xF) as u32) << (4 * k);
    }
    w
}

/// SWAR unpack with the producer LUTs from [`build_lut`] / [`build_mask`]:
/// 2 table loads + XOR, AND, ADD, shift select. Bit-equal to
/// [`unpack_word_ref`] for every `(idx, sign7, shift)`. Panics (index) if
/// `sign7 >= 128`.
#[inline]
pub fn unpack_word_swar(
    lut: &[u32; 256],
    mask: &[u32; 128],
    idx: u8,
    sign7: u8,
    shift: bool,
) -> u32 {
    let m = mask[sign7 as usize];
    let mut w = (lut[idx as usize] ^ m).wrapping_add(m & 0x1111_1111);
    if shift {
        w = w.wrapping_sub(0x1111_1111);
    }
    w
}

/// The per-K128-epoch fold (`v_lshl_add`): `2a + (s - MAGIC)` mod 2^32. With
/// `a = MAGIC + C` this is `MAGIC + 2C + s`.
#[inline]
pub fn fold128(a: u32, s: i32) -> u32 {
    a.wrapping_shl(1)
        .wrapping_add((s as u32).wrapping_sub(MAGIC))
}

/// One epoch's accumulation: `fma(RN(d_act · f32(d)), t · 0.25, sum)` with
/// `t = as_f32(a_folded) - as_f32(MAGIC)`.
///
/// Preconditions (not checked): `|2C + s| < 2^22` (guaranteed by the format,
/// `≤ 11 264`) and `d` finite (see [`writer_guard`]).
#[inline]
pub fn epoch_sum(d_act: f32, d: u16, a_folded: u32, sum: f32) -> f32 {
    let scale = d_act * f16::from_bits(d).to_f32();
    let t = f32::from_bits(a_folded) - f32::from_bits(MAGIC);
    scale.mul_add(t * 0.25, sum)
}

/// Writer guard: accept only finite `d >= 2^-14`. Anything else (including
/// zero, subnormal and negative) is routed to MQ4 by the writer, which counts
/// the trips.
pub fn writer_guard(d: u16) -> Result<(), Reject> {
    if d & 0x7C00 == 0x7C00 {
        Err(Reject::NonFinite { d })
    } else if d & 0x8000 != 0 || d < GUARD_MIN_D_BITS {
        Err(Reject::BelowMin { d })
    } else {
        Ok(())
    }
}

/// Process-wide producer tables for [`decode8_lanes`]: built once at compile
/// time (const-evaluated), so the lane decode never rebuilds them.
static LUT: [u32; 256] = build_lut();
static MASK: [u32; 128] = build_mask();

/// Pack one wire word: `idx | sign7 << 8 | shift << 15`.
///
/// Panics if `sign7 >= 128`.
#[inline]
pub fn pack_word(idx: u8, sign7: u8, shift: bool) -> u16 {
    assert!(
        sign7 < 128,
        "e8p_i4_reference::pack_word: sign7 {sign7} >= 128"
    );
    idx as u16 | (sign7 as u16) << 8 | (shift as u16) << 15
}

/// Inverse of [`pack_word`]: `(idx, sign7, shift)`.
#[inline]
pub const fn unpack_word_fields(word: u16) -> (u8, u8, bool) {
    (
        (word & 0xFF) as u8,
        ((word >> 8) & 0x7F) as u8,
        word >> 15 == 1,
    )
}

/// The four `4·codeword` lanes one GEMV lane owns: half `0` is lanes
/// `0..=3` of [`decode8`], half `1` lanes `4..=7`.
///
/// This is the lane-pair decode: *both* lanes of a pair decode the **whole**
/// 8-lane word with the cached [`LUT`]/[`MASK`] SWAR ([`unpack_word_swar`]) and
/// each keeps its own half (`v = 2·o + 1` from the two's-complement nibble
/// `o`). Bit-equal to `decode8(..)[4*half..4*half+4]` for every
/// `(idx, sign7, shift, half)`. Panics if `half > 1` or `sign7 >= 128`.
#[inline]
pub fn decode8_lanes(idx: u8, sign7: u8, shift: bool, half: usize) -> [i8; 4] {
    assert!(half < 2, "e8p_i4_reference::decode8_lanes: half {half} > 1");
    assert!(
        sign7 < 128,
        "e8p_i4_reference::decode8_lanes: sign7 {sign7} >= 128"
    );
    let w = unpack_word_swar(&LUT, &MASK, idx, sign7, shift);
    let base = 16 * half as u32;
    let mut v = [0i8; 4];
    for (l, out) in v.iter_mut().enumerate() {
        let nib = ((w >> (base + 4 * l as u32)) & 0xF) as i8;
        let o = (nib << 4) >> 4; // sign-extend the nibble
        *out = 2 * o + 1;
    }
    v
}

fn check_words_shape(
    who: &str,
    rotation: MqI4Rotation,
    m: usize,
    k: usize,
    words: &[u16],
    scale_bits: &[u16],
) {
    assert!(
        m > 0 && k > 0 && k % 128 == 0 && k % rotation.width() == 0,
        "{who}: invalid {rotation:?} shape [{m}, {k}] (need m, k > 0, k % 128 == 0, k % {} == 0)",
        rotation.width()
    );
    assert_eq!(
        words.len(),
        m * k / 8,
        "{who}: words length {} != m*k/8 = {}",
        words.len(),
        m * k / 8
    );
    assert_eq!(
        scale_bits.len(),
        m * k / 128,
        "{who}: scale_bits length {} != m*k/128 = {}",
        scale_bits.len(),
        m * k / 128
    );
}

/// The `4·codeword` integer weights `v` of a words plane, one `i8` per weight
/// row-major `[m, k]`, decoded the way the GEMV lanes do it: weights
/// `4c..4c+4` of the flat plane belong to lane-chunk `c`, which reads
/// `words[c / 2]` and keeps half `c & 1` ([`decode8_lanes`]).
///
/// `words` is row-major `m*k/8` `u16` wire words ([`pack_word`]). Panics on a
/// bad shape or length.
pub fn decode_weights_q(m: usize, k: usize, words: &[u16]) -> Vec<i8> {
    assert_eq!(
        words.len(),
        m * k / 8,
        "decode_weights_q: words length {} != m*k/8",
        words.len()
    );
    let mut q = Vec::with_capacity(m * k);
    for c in 0..m * k / 4 {
        let (idx, sign7, shift) = unpack_word_fields(words[c / 2]);
        q.extend_from_slice(&decode8_lanes(idx, sign7, shift, c & 1));
    }
    q
}

/// Per-group f32 scale of the E8P decode: `f32(d) · 0.25`. The multiply is a
/// power-of-two scaling and is **exact** in f32 for every finite FP16 `d`
/// (including subnormals: the smallest, `2^-24`, becomes `2^-26`, a normal
/// f32). No FP16 `d/4` is formed, so there is no f16 underflow.
#[inline]
pub fn scale_quarter_f32(d: u16) -> f32 {
    f16::from_bits(d).to_f32() * 0.25
}

/// Decoded F32 weights `(f32(d)·0.25)·v` (row-major `[m, k]`): exactly the
/// `w` of the frozen F4 tree. `(f32(d)·0.25)·v` is exact for every finite `d`
/// and odd `|v| <= 11` (an 11-bit significand times a 4-bit integer).
pub fn decode_weights_f32(m: usize, k: usize, words: &[u16], scale_bits: &[u16]) -> Vec<f32> {
    assert_eq!(
        scale_bits.len(),
        m * k / 128,
        "decode_weights_f32: scale_bits length"
    );
    let q = decode_weights_q(m, k, words);
    q.iter()
        .enumerate()
        .map(|(i, &v)| scale_quarter_f32(scale_bits[i / 128]) * v as f32)
        .collect()
}

/// E8P lane-pair decode GEMV twin on the frozen F4 tree.
///
/// Inputs (all row-major): `words` `m*k/8` `u16` ([`pack_word`]), `scale_bits`
/// `m*k/128` FP16 bit patterns (one per 128-column group), `x` `k` f32.
/// Returns `m` f32 values.
///
/// Each of the 32 lanes owns four consecutive weights of every 128-group
/// (`4·lane .. 4·lane+4`), so lane pair `(2j, 2j+1)` shares word `j` of the
/// group and each decodes the whole word ([`decode8_lanes`]). The weight is
/// `w = RN((f32(d)·0.25) · v)` and the reduction is
/// `mq_i4_reference::gemv_tree_f32_q` (the frozen F4 tree, unchanged:
/// first MUL, three left FMAs, per-group `acc[g & 3]` for `R256` / one
/// accumulator for `R128`, then `shfl_down` 16,8,4,2,1). `d` must be finite
/// (the writer guard's non-finite rejection); any finite FP16 `d`, including
/// zero and subnormals, is evaluated exactly.
pub fn decode_gemv_f32_lanepair(
    rotation: MqI4Rotation,
    m: usize,
    k: usize,
    words: &[u16],
    scale_bits: &[u16],
    x: &[f32],
) -> Vec<f32> {
    const WHO: &str = "decode_gemv_f32_lanepair";
    check_words_shape(WHO, rotation, m, k, words, scale_bits);
    assert_eq!(x.len(), k, "{WHO}: x length {} != k = {k}", x.len());
    let q = decode_weights_q(m, k, words);
    let d: Vec<f32> = scale_bits.iter().map(|&b| scale_quarter_f32(b)).collect();
    gemv_tree_f32_q(rotation, m, k, &q, &d, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
    }

    /// Random finite f32 with exponent in `[e_lo, e_hi]` (unbiased).
    fn rand_f32(r: &mut Rng, e_lo: i32, e_hi: i32) -> f32 {
        let sign = (r.next() & 1) as u32;
        let e = r.range(e_lo, e_hi) + 127;
        let m = (r.next() & 0x7F_FFFF) as u32;
        f32::from_bits(sign << 31 | (e as u32) << 23 | m)
    }

    // ---- minimal SHA-256 (test-only; the crate has no sha2 dependency) ----
    fn sha256_hex(data: &[u8]) -> String {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        for chunk in msg.chunks(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for i in 0..8 {
                h[i] = h[i].wrapping_add(v[i]);
            }
        }
        h.iter().map(|x| format!("{x:08x}")).collect()
    }

    #[test]
    fn e8p_i4_reference_sha256_selftest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn e8p_i4_reference_abs_table_structure() {
        let rows: HashSet<[u8; 8]> = E8P_ABS2.iter().copied().collect();
        assert_eq!(rows.len(), 256, "abs rows must be distinct");
        assert!(E8P_ABS2
            .iter()
            .flatten()
            .all(|&x| x == 1 || x == 3 || x == 5));
        // |x|^2 = Σ(2|g|)^2 / 4: 227 rows with |x|^2 <= 10, 29 with 12.
        let n40 = E8P_ABS2
            .iter()
            .filter(|r| r.iter().map(|&x| (x as u32).pow(2)).sum::<u32>() <= 40)
            .count();
        let n48 = E8P_ABS2
            .iter()
            .filter(|r| r.iter().map(|&x| (x as u32).pow(2)).sum::<u32>() == 48)
            .count();
        assert_eq!((n40, n48), (227, 29));
        // Σ2|g| is even on every row; Σ|g| is odd on 149 rows.
        assert!(E8P_ABS2
            .iter()
            .all(|r| r.iter().map(|&x| x as u32).sum::<u32>() % 2 == 0));
        let odd = (0..=255u8).filter(|&i| abs_row_parity(i) == 1).count();
        assert_eq!(odd, 149);
        // LUT (1 024 B) + MASK (512 B) = the 1 536-B per-CTA stage.
        assert_eq!(build_lut().len() * 4 + build_mask().len() * 4, 1536);
    }

    /// `vq_books.json` from `HIPFIRE_VQ_BOOKS_JSON` (a path). Unset (or empty)
    /// means the pinned-JSON cross-check is skipped; a set-but-unreadable path
    /// is an error.
    fn books_json() -> Option<Vec<u8>> {
        match hipfire_config::developer_var("HIPFIRE_VQ_BOOKS_JSON") {
            Ok(p) if !p.is_empty() => Some(std::fs::read(&p).unwrap_or_else(|e| {
                panic!("e8p_i4_reference: cannot read HIPFIRE_VQ_BOOKS_JSON={p}: {e}")
            })),
            _ => {
                eprintln!("e8p_i4_reference: HIPFIRE_VQ_BOOKS_JSON unset, pinned vq_books.json cross-check skipped");
                None
            }
        }
    }

    #[test]
    fn e8p_i4_reference_all_65536_inputs_are_4x_codeword_of_pinned_books() {
        let Some(raw) = books_json() else { return };
        assert_eq!(
            sha256_hex(&raw),
            E8P_BOOKS_SHA256,
            "vq_books.json differs from the pinned sha256"
        );
        let doc: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let grid: Vec<Vec<f64>> = doc["e8p_abs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(grid.len(), 256);
        for (i, r) in grid.iter().enumerate() {
            assert_eq!(r.len(), 8);
            for k in 0..8 {
                assert_eq!(
                    E8P_ABS2[i][k] as f64,
                    2.0 * r[k].abs(),
                    "E8P_ABS2[{i}][{k}] differs from vq_books.json"
                );
                assert!(r[k] > 0.0);
            }
        }
        let mut n = 0u32;
        for idx in 0..=255u8 {
            let g = &grid[idx as usize];
            let par = (g.iter().sum::<f64>() as i64 & 1) as u32; // Σ|g| is an exact integer
            assert_eq!(g.iter().sum::<f64>().fract(), 0.0);
            for sign7 in 0..128u8 {
                let neg7 = (sign7.count_ones() + par) & 1;
                let signed: Vec<f64> = (0..8)
                    .map(|k| {
                        let neg = if k < 7 {
                            (sign7 >> k) & 1 == 1
                        } else {
                            neg7 == 1
                        };
                        if neg {
                            -g[k]
                        } else {
                            g[k]
                        }
                    })
                    .collect();
                let sum: f64 = signed.iter().sum();
                assert!(
                    sum.fract() == 0.0 && (sum as i64) % 2 == 0,
                    "signed vector not in E8 (idx {idx} sign7 {sign7})"
                );
                for shift in [false, true] {
                    let sigma = if shift { -0.25 } else { 0.25 };
                    let want: Vec<i8> = signed
                        .iter()
                        .map(|x| {
                            let v = 4.0 * (x + sigma);
                            assert_eq!(v.fract(), 0.0);
                            v as i8
                        })
                        .collect();
                    assert_eq!(
                        decode8(idx, sign7, shift).to_vec(),
                        want,
                        "idx {idx} sign7 {sign7} shift {shift}"
                    );
                    n += 1;
                }
            }
        }
        assert_eq!(n, 65_536);
    }

    #[test]
    fn e8p_i4_reference_codewords_distinct_odd_and_in_range() {
        let mut seen: HashSet<[i8; 8]> = HashSet::new();
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    let v = decode8(idx, sign7, shift);
                    assert!(
                        v.iter().all(|&x| x % 2 != 0 && (-11..=11).contains(&x)),
                        "{v:?}"
                    );
                    let o_ok = v.iter().all(|&x| {
                        (-6..=5).contains(&((x - 1) >> 1)) && 2 * ((x - 1) >> 1) + 1 == x
                    });
                    assert!(o_ok, "{v:?}");
                    assert!(seen.insert(v), "duplicate codeword {v:?}");
                }
            }
        }
        assert_eq!(seen.len(), 65_536);
    }

    #[test]
    fn e8p_i4_reference_swar_equals_reference_all_inputs() {
        let lut = build_lut();
        let mask = build_mask();
        // Lane-7 pre-negation is exactly the 149 odd rows; MASK lane 7 is popcount parity.
        for i in 0..256usize {
            let l7 = (lut[i] >> 28) & 0xF;
            let a = E8P_ABS2[i][7] as u32;
            assert_eq!(
                l7,
                if abs_row_parity(i as u8) == 1 {
                    a.wrapping_neg() & 0xF
                } else {
                    a
                }
            );
        }
        for s in 0..128usize {
            assert_eq!(
                (mask[s] >> 28) & 0xF,
                if s.count_ones() & 1 == 1 { 0xF } else { 0 }
            );
            assert_eq!(
                mask[s] & 0x0FFF_FFFF,
                (0..7)
                    .map(|k| if (s >> k) & 1 == 1 {
                        0xFu32 << (4 * k)
                    } else {
                        0
                    })
                    .sum::<u32>()
            );
        }
        let mut n = 0u32;
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    let r = unpack_word_ref(idx, sign7, shift);
                    assert_eq!(
                        unpack_word_swar(&lut, &mask, idx, sign7, shift),
                        r,
                        "idx {idx} sign7 {sign7} shift {shift}"
                    );
                    // Cross-check the nibbles against decode8 directly.
                    let v = decode8(idx, sign7, shift);
                    for k in 0..8 {
                        let nib = ((r >> (4 * k)) & 0xF) as i32;
                        let o = if nib >= 8 { nib - 16 } else { nib };
                        assert_eq!(2 * o + 1, v[k] as i32);
                    }
                    n += 1;
                }
            }
        }
        assert_eq!(n, 65_536);
        // const-evaluated tables equal the runtime ones.
        const CLUT: [u32; 256] = build_lut();
        const CMASK: [u32; 128] = build_mask();
        assert_eq!(CLUT, lut);
        assert_eq!(CMASK, mask);
    }

    #[test]
    fn e8p_i4_reference_odd_rows_are_149_and_use_prenegated_lane7() {
        let lut = build_lut();
        let odd: Vec<usize> = (0..256).filter(|&i| abs_row_parity(i as u8) == 1).collect();
        assert_eq!(odd.len(), 149);
        for &i in &odd {
            assert!(matches!((lut[i] >> 28) & 0xF, 15 | 13 | 11), "row {i}");
        }
        for i in (0..256).filter(|i| !odd.contains(i)) {
            assert!(matches!((lut[i] >> 28) & 0xF, 1 | 3 | 5), "row {i}");
        }
    }

    #[test]
    fn e8p_i4_reference_fold128_exact_327680() {
        let mf = f32::from_bits(MAGIC);
        assert_eq!(mf, 12_582_912.0);
        let mut total = 0u32;
        for seed in 0..5u64 {
            let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ (seed + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let mut cases: Vec<(i32, i32)> = Vec::with_capacity(65_536);
            if seed == 0 {
                // Format extremes: v = ±11 against q = -8 over all 128 elements.
                cases.push((6144, -1024)); // 2C+s = +11264
                cases.push((-5120, -1024)); // 2C+s = -11264
            }
            while cases.len() < 65_536 {
                // C = Σ o·q over 128 elements, o ∈ [-6,5], q ∈ [-8,7]; s = Σ q.
                let c = r.range(-5376, 6144);
                let s = r.range(-1024, 896);
                if (2 * c + s).abs() <= 11_264 {
                    cases.push((c, s));
                }
            }
            for (c, s) in cases {
                let t = 2 * c + s;
                let a = MAGIC.wrapping_add(c as u32);
                let folded = fold128(a, s);
                assert_eq!(folded as i64, (MAGIC as i64 + t as i64), "C={c} s={s}");
                let tf = f32::from_bits(folded) - mf;
                assert_eq!(tf, t as f32, "C={c} s={s}");
                assert_eq!((tf * 0.25) as f64, t as f64 / 4.0, "C={c} s={s}");
                total += 1;
            }
        }
        assert_eq!(total, 327_680);
        // Wrap-around: the fold is exact mod 2^32 even for out-of-range inputs.
        assert_eq!(fold128(0x8000_0000, 0), MAGIC.wrapping_neg());
    }

    /// Correctly rounded `RN_f32(a·b + c)` for `a·b` exactly representable in
    /// f64 (here: 24-bit × 12-bit): f64 add with round-to-odd, then one
    /// rounding to f32.
    fn fma_via_f64(a: f64, b: f64, c: f64) -> f32 {
        let p = a * b;
        assert_eq!(a.mul_add(b, -p), 0.0, "product not exact in f64");
        let s = p + c;
        let bb = s - p;
        let e = (p - (s - bb)) + (c - bb); // TwoSum error
        let r = if e != 0.0 && s.to_bits() & 1 == 0 {
            let inc = (e > 0.0) == (s > 0.0);
            f64::from_bits(if inc {
                s.to_bits() + 1
            } else {
                s.to_bits() - 1
            })
        } else {
            s
        };
        r as f32
    }

    #[test]
    fn e8p_i4_reference_epoch_sum_bit_equal_f64_reference_1e6() {
        let mut r = Rng(0xA5A5_1234_5678_9ABC);
        let mut cancel = 0u32;
        for i in 0..1_000_000u32 {
            let d_act = rand_f32(&mut r, -20, 20);
            let d = (r.range(0x0400, 0x7BFF)) as u16;
            let c = r.range(-5376, 6144);
            let mut s = r.range(-1024, 896);
            while (2 * c + s).abs() > 11_264 {
                s = r.range(-1024, 896);
            }
            let folded = fold128(MAGIC.wrapping_add(c as u32), s);
            let t_int = (2 * c + s) as i64;

            let scale_ref = ((d_act as f64) * (f16::from_bits(d).to_f32() as f64)) as f32; // exact product, one RN
            let tq = t_int as f64 / 4.0;
            let prod = scale_ref as f64 * tq;
            let sum = match i % 4 {
                0 | 1 => rand_f32(&mut r, -40, 40),
                2 => 0.0,
                _ => {
                    // near-cancellation: -RN(prod) with a few ulps of noise
                    cancel += 1;
                    let b = (-(prod as f32)).to_bits() as i64 + r.range(-3, 3) as i64;
                    f32::from_bits(b as u32)
                }
            };
            let want = fma_via_f64(scale_ref as f64, tq, sum as f64);
            let got = epoch_sum(d_act, d, folded, sum);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "i={i} d_act={d_act:e} d=0x{d:04x} C={c} s={s} sum={sum:e}"
            );
        }
        assert!(cancel > 200_000);
    }

    #[test]
    fn e8p_i4_reference_block_pipeline_matches_dequantized_dot() {
        // 128 weights = 16 wire words -> o nibbles; random IU4 activations q.
        let lut = build_lut();
        let mask = build_mask();
        let mut r = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..2000 {
            let d = r.range(0x0400, 0x7BFF) as u16;
            let d_act = rand_f32(&mut r, -8, 8);
            let mut c = 0i32;
            let mut s = 0i32;
            let mut exact = 0i64; // Σ v·q
            for _w in 0..16 {
                let (idx, sign7, shift) = (
                    r.range(0, 255) as u8,
                    r.range(0, 127) as u8,
                    r.next() & 1 == 1,
                );
                let word = unpack_word_swar(&lut, &mask, idx, sign7, shift);
                let v = decode8(idx, sign7, shift);
                for k in 0..8 {
                    let nib = ((word >> (4 * k)) & 0xF) as i32;
                    let o = if nib >= 8 { nib - 16 } else { nib };
                    let q = r.range(-8, 7);
                    c += o * q;
                    s += q;
                    exact += v[k] as i64 * q as i64;
                }
            }
            assert_eq!((2 * c + s) as i64, exact);
            // a = MAGIC + C (the WMMA chain), folded, accumulated.
            let folded = fold128(MAGIC.wrapping_add(c as u32), s);
            let sum0 = rand_f32(&mut r, -10, 10);
            let got = epoch_sum(d_act, d, folded, sum0);
            // Decode-form weights (f32(d)·0.25)·v are exact for every finite FP16 d.
            let df = f16::from_bits(d).to_f32();
            let wdot: f64 = exact as f64 * (df as f64 * 0.25);
            let want = fma_via_f64((d_act * df) as f64, exact as f64 / 4.0, sum0 as f64);
            assert_eq!(got.to_bits(), want.to_bits());
            assert_eq!(exact as f64 / 4.0 * df as f64, wdot);
        }
    }

    #[test]
    fn e8p_i4_reference_writer_guard() {
        for d in 0..=u16::MAX {
            let v = f16::from_bits(d).to_f32();
            let r = writer_guard(d);
            if !v.is_finite() {
                assert_eq!(r, Err(Reject::NonFinite { d }), "d=0x{d:04x}");
            } else if v < 2f32.powi(-14) {
                assert_eq!(r, Err(Reject::BelowMin { d }), "d=0x{d:04x}");
            } else {
                assert_eq!(r, Ok(()), "d=0x{d:04x}");
                // Decode weight (f32(d)·0.25)·v is exact for every accepted d and every v.
                for vv in [-11i32, -9, -1, 1, 3, 5, 9, 11] {
                    assert_eq!(((v * 0.25) * vv as f32) as f64, v as f64 * vv as f64 / 4.0);
                }
            }
        }
        assert_eq!(writer_guard(0x03FF), Err(Reject::BelowMin { d: 0x03FF })); // largest subnormal
        assert_eq!(writer_guard(0x0000), Err(Reject::BelowMin { d: 0 }));
        assert_eq!(writer_guard(0x8400), Err(Reject::BelowMin { d: 0x8400 })); // -2^-14
        assert_eq!(writer_guard(0x0400), Ok(())); // 2^-14
        assert_eq!(writer_guard(0x7BFF), Ok(())); // 65504
        assert_eq!(writer_guard(0x7C00), Err(Reject::NonFinite { d: 0x7C00 }));
        assert_eq!(writer_guard(0xFC00), Err(Reject::NonFinite { d: 0xFC00 }));
        assert_eq!(writer_guard(0x7E00), Err(Reject::NonFinite { d: 0x7E00 }));
        let n_ok = (0..=u16::MAX).filter(|&d| writer_guard(d).is_ok()).count();
        assert_eq!(n_ok, 0x7BFF - 0x0400 + 1);
    }

    // ---- E1 lane-pair decode and GEMV twin ---------------------------------

    use crate::mq_i4::{MqI4Profile, MqI4Rotation};
    use crate::mq_i4_reference::{decode_gemv_f32_parts, dequant_rotated_f32_parts};

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// f16 scale bit patterns with every finite class: max, min normal,
    /// largest/smallest subnormal, zeros (both signs), negatives, random.
    fn extreme_scale(r: &mut Rng, i: usize) -> u16 {
        match i % 11 {
            0 => 0x7BFF,
            1 => 0x0400,
            2 => 0x03FF,
            3 => 0x0001,
            4 => 0x0000,
            5 => 0x8000,
            6 => 0xFBFF,
            7 => 0x8001,
            _ => (r.range(0, 0x7BFF) as u16) | if r.next() & 1 == 1 { 0x8000 } else { 0 },
        }
    }

    /// Random x with exponents in `[-8, 8]` (so `x * 0.25` is exact) and some zeros.
    fn rand_x(r: &mut Rng, k: usize) -> Vec<f32> {
        (0..k)
            .map(|_| {
                if r.next() % 16 == 0 {
                    0.0
                } else {
                    rand_f32(r, -8, 8)
                }
            })
            .collect()
    }

    #[test]
    fn e8p_i4_reference_decode8_lanes_all_65536_both_halves() {
        let mut n = 0u32;
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    let want = decode8(idx, sign7, shift);
                    let lo = decode8_lanes(idx, sign7, shift, 0);
                    let hi = decode8_lanes(idx, sign7, shift, 1);
                    assert_eq!(
                        lo,
                        [want[0], want[1], want[2], want[3]],
                        "idx {idx} sign7 {sign7} shift {shift} half 0"
                    );
                    assert_eq!(
                        hi,
                        [want[4], want[5], want[6], want[7]],
                        "idx {idx} sign7 {sign7} shift {shift} half 1"
                    );
                    let w = pack_word(idx, sign7, shift);
                    assert_eq!(unpack_word_fields(w), (idx, sign7, shift));
                    n += 1;
                }
            }
        }
        assert_eq!(n, 65_536);
    }

    #[test]
    #[should_panic(expected = "half 2 > 1")]
    fn e8p_i4_reference_decode8_lanes_rejects_half_2() {
        let _ = decode8_lanes(0, 0, false, 2);
    }

    /// Words plane that starts with every codeword of `pool` once (in a fixed
    /// permuted order) and is random after that.
    fn words_plane(r: &mut Rng, n_words: usize, pool: &[u16]) -> Vec<u16> {
        let mut order: Vec<u16> = pool.to_vec();
        for i in (1..order.len()).rev() {
            order.swap(i, (r.next() % (i as u64 + 1)) as usize);
        }
        (0..n_words)
            .map(|i| {
                if i < order.len() {
                    order[i]
                } else {
                    pool[(r.next() % pool.len() as u64) as usize]
                }
            })
            .collect()
    }

    fn all_words() -> Vec<u16> {
        let mut v = Vec::with_capacity(65_536);
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    v.push(pack_word(idx, sign7, shift));
                }
            }
        }
        v
    }

    /// Codewords the MQ3 API can express: abs rows without a `5` (all `|v| <= 7`).
    fn mq_expressible_words() -> Vec<u16> {
        all_words()
            .into_iter()
            .filter(|&w| {
                let (idx, sign7, shift) = unpack_word_fields(w);
                decode8(idx, sign7, shift).iter().all(|v| v.abs() <= 7)
            })
            .collect()
    }

    const SHAPES: [(MqI4Rotation, MqI4Profile, usize, usize); 2] = [
        (MqI4Rotation::R256, MqI4Profile::MQ3_R256, 256, 2560),
        (MqI4Rotation::R128, MqI4Profile::MQ3_R128, 128, 640),
    ];

    /// Full-range plumbing: lane-pair weights and GEMV vs the helper fed with
    /// plain `decode8` (no lane path), with every one of the 65,536 codewords
    /// present in the R256/K2560 plane; weights `(f32(d)·0.25)·v` exact.
    #[test]
    fn e8p_i4_reference_lanepair_gemv_full_range_matches_decode8_tree() {
        let mut r = Rng(0xE1E1_0001_2345_6789);
        let all = all_words();
        for (rotation, _p, m, k) in SHAPES {
            let words = words_plane(&mut r, m * k / 8, &all);
            if m * k / 8 >= 65_536 {
                let mut seen: HashSet<u16> = words[..65_536].iter().copied().collect();
                seen.extend(words[65_536..].iter().copied());
                assert_eq!(seen.len(), 65_536);
            }
            let scales: Vec<u16> = (0..m * k / 128)
                .map(|i| {
                    let o = r.range(0, 3) as usize;
                    extreme_scale(&mut r, i + o)
                })
                .collect();
            let x = rand_x(&mut r, k);
            // Independent decode: whole-word decode8, no lanes.
            let mut q = Vec::with_capacity(m * k);
            for &w in &words {
                let (idx, sign7, shift) = unpack_word_fields(w);
                q.extend_from_slice(&decode8(idx, sign7, shift));
            }
            assert_eq!(q, decode_weights_q(m, k, &words));
            let wf = decode_weights_f32(m, k, &words, &scales);
            for (i, &v) in q.iter().enumerate() {
                let d = f16::from_bits(scales[i / 128]).to_f32();
                // exact in f64: (d * 0.25) * v == d * v / 4
                assert_eq!(wf[i] as f64, d as f64 * v as f64 / 4.0, "weight {i}");
                assert_eq!(
                    wf[i].to_bits(),
                    ((d * 0.25) * v as f32).to_bits(),
                    "weight {i}"
                );
            }
            let d: Vec<f32> = scales
                .iter()
                .map(|&b| f16::from_bits(b).to_f32() * 0.25)
                .collect();
            let want = gemv_tree_f32_q(rotation, m, k, &q, &d, &x);
            let got = decode_gemv_f32_lanepair(rotation, m, k, &words, &scales, &x);
            assert_eq!(bits(&got), bits(&want), "{rotation:?} [{m}, {k}]");
            assert!(got.iter().any(|v| *v != 0.0));
        }
    }

    /// `d/4` as an f16 bit pattern iff exactly representable (no underflow).
    fn quarter_f16_exact(d: u16) -> Option<u16> {
        let q = f16::from_bits(d).to_f32() * 0.25;
        let b = f16::from_f32(q);
        (b.to_f32().to_bits() == q.to_bits()).then(|| b.to_bits())
    }

    /// Bit equality against the existing `mq_i4_reference::decode_gemv_f32_parts`
    /// on the codewords MQ3 can express, R256/K2560 m256 and R128/K640 m128,
    /// random scales plus f16 max/subnormal/zero/negative extremes.
    /// (A) scale `d/4` as f16 where exact (also compares the decoded weights
    /// with `dequant_rotated_f32_parts`); (B) scale `d` with `x * 0.25`.
    #[test]
    fn e8p_i4_reference_lanepair_gemv_bit_equal_mq_decode_gemv_expressible_subset() {
        let mut r = Rng(0xE1E1_0002_DEAD_BEEF);
        let pool = mq_expressible_words();
        assert!(!pool.is_empty() && pool.len() < 65_536);
        let mut n_a = 0usize;
        let mut n_b = 0usize;
        for (rotation, profile, m, k) in SHAPES {
            let words = words_plane(&mut r, m * k / 8, &pool);
            // MQ3 codes c = (v + 7) / 2, one per weight.
            let q = decode_weights_q(m, k, &words);
            let codes: Vec<u8> = q
                .iter()
                .map(|&v| {
                    assert!(v.abs() <= 7 && v % 2 != 0);
                    ((v + 7) / 2) as u8
                })
                .collect();
            let x = rand_x(&mut r, k);

            // (B) every scale class, scale d with x/4.
            let scales: Vec<u16> = (0..m * k / 128).map(|i| extreme_scale(&mut r, i)).collect();
            let x4: Vec<f32> = x.iter().map(|v| v * 0.25).collect();
            for (a, b) in x.iter().zip(&x4) {
                assert_eq!(*a as f64 * 0.25, *b as f64);
            }
            let want = decode_gemv_f32_parts(profile, m, k, &codes, &scales, &x4);
            let got = decode_gemv_f32_lanepair(rotation, m, k, &words, &scales, &x);
            assert_eq!(bits(&got), bits(&want), "B {rotation:?} [{m}, {k}]");
            n_b += m;

            // (A) f16 d/4 exact: replace unrepresentable scales, direct weights + gemv.
            let scales_a: Vec<u16> = scales
                .iter()
                .enumerate()
                .map(|(i, &d)| match quarter_f16_exact(d) {
                    Some(_) if d & 0x7FFF != 0 => d,
                    _ => 0x0C00 + ((i * 37) % 0x3000) as u16,
                })
                .collect();
            let mq_scales: Vec<u16> = scales_a
                .iter()
                .map(|&d| quarter_f16_exact(d).expect("exact"))
                .collect();
            let w_mq = dequant_rotated_f32_parts(profile, m, k, &codes, &mq_scales);
            let w_e8 = decode_weights_f32(m, k, &words, &scales_a);
            assert_eq!(bits(&w_mq), bits(&w_e8), "A weights {rotation:?}");
            let want = decode_gemv_f32_parts(profile, m, k, &codes, &mq_scales, &x);
            let got = decode_gemv_f32_lanepair(rotation, m, k, &words, &scales_a, &x);
            assert_eq!(bits(&got), bits(&want), "A {rotation:?} [{m}, {k}]");
            n_a += m;
        }
        assert_eq!((n_a, n_b), (384, 384));
    }

    /// Optional ground truth against the standalone C++ E1 bench: with
    /// `HIPFIRE_E1_FIXTURE_DIR` set, every `N.words.bin` in the directory is a
    /// fixture `N = r{256|128}_m{M}_k{K}[_suffix]` (rotation and shape parsed
    /// from the name). Each reads raw little-endian `N.words.bin` (u16),
    /// `N.scales.bin` (u16 f16 bits), `N.x.bin` (f32) and `N.expected.bin`
    /// (f32 GEMV bits, one per row) and compares all rows bit-for-bit with
    /// [`decode_gemv_f32_lanepair`]. The production shapes
    /// `r256_m1280_k2560` and `r128_m2560_k640` must be present. Unset: skipped.
    #[test]
    fn e8p_i4_reference_lanepair_gemv_matches_cpp_bench_fixture() {
        let dir = match hipfire_config::developer_var("HIPFIRE_E1_FIXTURE_DIR") {
            Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
            _ => {
                eprintln!("e8p_i4_reference: HIPFIRE_E1_FIXTURE_DIR unset, C++ bench fixture check skipped");
                return;
            }
        };
        let rd = |name: &str, ext: &str| -> Vec<u8> {
            let p = dir.join(format!("{name}.{ext}.bin"));
            std::fs::read(&p)
                .unwrap_or_else(|e| panic!("e8p_i4_reference: cannot read {}: {e}", p.display()))
        };
        let u16s = |b: Vec<u8>| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let f32s = |b: Vec<u8>| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("e8p_i4_reference: cannot list {}: {e}", dir.display()))
            .filter_map(|e| {
                let f = e.ok()?.file_name().into_string().ok()?;
                f.strip_suffix(".words.bin").map(str::to_owned)
            })
            .collect();
        names.sort();
        for must in ["r256_m1280_k2560", "r128_m2560_k640"] {
            assert!(
                names.iter().any(|n| n == must),
                "fixture {must} missing in {}",
                dir.display()
            );
        }
        for name in &names {
            let mut parts = name.split('_');
            let rotation = match parts.next() {
                Some("r256") => MqI4Rotation::R256,
                Some("r128") => MqI4Rotation::R128,
                other => panic!("fixture {name}: unknown rotation prefix {other:?}"),
            };
            let m: usize = parts
                .next()
                .and_then(|p| p.strip_prefix('m'))
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(|| panic!("fixture {name}: cannot parse m{{M}}"));
            let k: usize = parts
                .next()
                .and_then(|p| p.strip_prefix('k'))
                .and_then(|p| p.parse().ok())
                .unwrap_or_else(|| panic!("fixture {name}: cannot parse k{{K}}"));
            let words = u16s(rd(name, "words"));
            let scales = u16s(rd(name, "scales"));
            let x = f32s(rd(name, "x"));
            let expected = f32s(rd(name, "expected"));
            assert_eq!(words.len(), m * k / 8, "{name} words");
            assert_eq!(scales.len(), m * k / 128, "{name} scales");
            assert_eq!(x.len(), k, "{name} x");
            assert_eq!(expected.len(), m, "{name} expected");
            let got = decode_gemv_f32_lanepair(rotation, m, k, &words, &scales, &x);
            for r in 0..m {
                assert_eq!(
                    got[r].to_bits(),
                    expected[r].to_bits(),
                    "{name} row {r}: rust {:e} cpp {:e}",
                    got[r],
                    expected[r]
                );
            }
            eprintln!(
                "e8p_i4_reference: fixture {name} [{m}, {k}] {} rows bit-equal",
                m
            );
        }
    }
}
