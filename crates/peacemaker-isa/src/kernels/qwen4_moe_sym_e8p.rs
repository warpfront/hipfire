// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WeightCodec::E8pI4` NT1 entries of the grouped symmetric IU4 MoE GEMM
//! (gfx1201): gate/up `qwen4_moe_gate_up_silu_iu4_sym_e8p_pm_gfx1201` (R256, K2560)
//! and down `qwen4_moe_down_iu4_sym_e8p_pm_gfx1201` (R128, K640), revision 1.
//!
//! Same 64-byte ABI (the 60-byte frozen ABI plus a trailing `n_experts` at 60), grid `(M/64, m_total/16)` for both entries (gate/up: `M` is the full
//! gate+up row count, each CTA owns 32 gate + 32 up rows, i.e. `(M/2)/32`; down: 64 rows per
//! CTA, `M/64`), A4 activations, gate/up two
//! waves, down four waves, padding/sentinel handling, BF16 epilogue (SiLU for
//! gate/up) and F32 `_acc` qualification epilogue as the W3 NT1 entries
//! (`qwen4_moe_sym_w3.rs`). Only the weight side and the fold differ.
//!
//! # Wire layout
//!
//! `expert_weight_ptrs[e]` is the planar E8P-I4 tensor `[M, K]`: `M*K/8` little-endian
//! `u16` words (row-major), then `M*K/128` FP16 scales. A word covers eight weights:
//! `idx | sign7 << 8 | shift << 15` (`idx` the row of the 256-entry abs grid
//! [`E8P_ABS2`], bit `k` of `sign7` negates lane `k < 7`, lane 7's sign follows from the
//! row parity, `shift` set = the QuIP# `-1/4`). Code row bytes are `K/4`, one K128 epoch
//! is 32 bytes (16 words) of a row. Source of the grid, the parity rule and every
//! constant below: the frozen K-E0 twin
//! `crates/hipfire-runtime/src/e8p_i4_reference.rs` (grid from `vq/code/vq_books.json`,
//! sha256 [`E8P_BOOKS_SHA256`]); they are copied, not linked (no PM -> runtime edge).
//!
//! # K assignment and loads
//!
//! The 16x16x32 IU4 WMMA sums over K, so the K positions a lane feeds may be permuted
//! as long as the A and X operands use the same permutation. A lane `(hi, lr)` owns the
//! eight words `8hi .. 8hi + 8` of its row's epoch (K elements `64hi .. 64hi + 64`):
//! ONE `buffer_load_b128` per tensor and epoch at `lr * row_bytes + 16hi + 32 * epoch`,
//! and step `i` takes dword `i` (words `2i`, `2i + 1`) as A and X bytes `8 + 32hi + 8i
//! .. +8` as X (two `buffer_load_b128` per epoch). The activation's `d` (f32) and `s`
//! (i32 `sum q`) are loaded together as one `b64`; `s` stays live in its set until the
//! fold. Every set (epoch parity) is register-direct and double buffered exactly like
//! the W3 entry: the next epoch's loads are issued before the current epoch is consumed.
//!
//! # LDS stage (1,536 B, once per CTA)
//!
//! `LUT[256]` (offset 0, one u32 of eight nibbles `2|g|` per abs row, lane 7 stored
//! two's-complement negated on odd-parity rows) and `MASK[128]` (offset 1,024, `0xF` in
//! the lanes `sign7` negates, lane 7 = parity of `popcount(sign7)`). One typed region
//! (`Tables`) is written by every wave, drained, published at ONE workgroup barrier
//! and then only read. MASK is computed arithmetically (the same code in every wave,
//! entries `tid + 32 * waves * j`); the 8 LUT chunks of 32 dwords are split between the
//! waves by wave-uniform `skip_unless` blocks (wave `k` writes chunks `c % waves == k`)
//! from SGPR literals combined into one VGPR with `v_writelane_b32`, one `ds_store_b32`
//! per chunk. The stage runs after the CTA-uniform early exits and after the first
//! epoch's loads are in flight, before any wave divergence. Store addresses are derived
//! from the work-item id in the entry block; table reads use `v_and_b32` immediate-masked
//! addresses (`0x3fc`, `0x1fc`), both forms the `pm_check::lds_bounds` check proves.
//!
//! # Unpack, `2C + s` fold
//!
//! Per word (`w` = the dword, lo word then hi word): two address derivations
//! (`(w << 2) & 0x3fc` / `(w >> 14) & 0x3fc` for the LUT, `(w >> 6) & 0x1fc` /
//! `(w >> 22) & 0x1fc` for the MASK), two `ds_load_b32`, then
//! `x = lut ^ mask`, `y = mask & 0x11111111`, `n = shift ? 0xeeeeeeef : 0`
//! (`-0x11111111 mod 2^32`), operand `= x + y + n` (`v_add3_u32`): `o = ±2|g| - shift`
//! per nibble with no carry or borrow across nibbles ([`unpack_emitted`], exhaustively
//! tested). `v = 2o + 1` so `sum(v q) = 2C + s`. The WMMA chain is seeded with MAGIC and
//! after the epoch each C register holds `a = MAGIC + C`. The fold is
//! `a' = v_lshl_add_u32(a, 1, s - MAGIC) = MAGIC + 2C + s`,
//! `t = as_f32(a') + (-12582912.0)`, `t *= 0.25` (exact) and
//! `sum = fma(RN(d_act * f32(d)), t, sum)`; no FP16 `d/4` exists ([`epoch_model`]).
//!
//! # Registers
//!
//! VGPR 168 (granule 8): the MQ4/W3 layout up to `SET0 = 92`, two 20-register sets
//! (code words 8, X 8, `d`/`s` 2, scale pair 2) at 92 and 112, A operands 132..147,
//! unpack pool 148..167 (four words x five). SGPR high water 88 (descriptor/length
//! additions as W3 plus eight literal staging SGPRs at 80..87).
use super::*;
use crate::vopd::{Operand, VopdF32, VopdOp};
use peacemaker_author::{ready, Free, LdsRegion, Published, Wave};

/// sha256 of `vq/code/vq_books.json`, the source of [`E8P_ABS2`].
pub const E8P_BOOKS_SHA256: &str =
    "bfe91a2d8bae4bb91e3a07bd22ee61b96675795d2ddf820303f27cfc37ba5da8";

/// `LUT[256]` bytes.
const LUT_BYTES: u32 = 256 * 4;
/// `MASK[128]` bytes (at `LUT_BYTES`).
const MASK_BYTES: u32 = 128 * 4;
/// Static LDS of every E8P entry: the per-CTA table stage.
pub const STAGE_BYTES: u32 = LUT_BYTES + MASK_BYTES;
/// Nibble ones: the `+1` of a two's-complement negation and the shift unit.
const ONES: u32 = 0x1111_1111;
/// `-ONES mod 2^32`: the shift subtraction as an addend.
const NEG_ONES: u32 = 0xeeee_eeef;
/// `-MAGIC mod 2^32`: `s - MAGIC` is `s + MAGIC_NEG_I32`.
const MINUS_MAGIC_I32: u32 = 0xb4c0_0000;
/// `0.25f`.
const QUARTER: u32 = 0x3e80_0000;

/// The staged table region (one typed LDS region, LUT then MASK).
enum Tables {}

// Copied verbatim from `crates/hipfire-runtime/src/e8p_i4_reference.rs` (K-E0 twin, `E8P_ABS2`).
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

/// Parity of `Σ|g|` for abs row `idx` (1 = "odd row": 149 of 256).
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
pub const fn build_lut256u32() -> [u32; 256] {
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

/// The 128-word sign mask: lane `k < 7` = `0xF` iff bit `k` of `sign7`; lane 7 =
/// `0xF` iff `popcount(sign7)` is odd.
pub const fn build_mask128u32() -> [u32; 128] {
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

/// `4·codeword` of one wire word: eight odd integers in `[-11, 11]` (twin `decode8`).
pub fn decode8(idx: u8, sign7: u8, shift: bool) -> [i8; 8] {
    assert!(sign7 < 128, "decode8: sign7 {sign7} >= 128");
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

/// Reference unpack: eight `o = (v - 1) >> 1` nibbles (two's complement, lane `k` at
/// bits `4k..4k+4`) of [`decode8`] (twin `unpack_word_ref`).
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

/// SWAR unpack with the stage tables (twin `unpack_word_swar`): 2 table loads, XOR,
/// AND, ADD, shift select.
pub fn unpack_swar(lut: &[u32; 256], mask: &[u32; 128], idx: u8, sign7: u8, shift: bool) -> u32 {
    let m = mask[sign7 as usize];
    let w = (lut[idx as usize] ^ m).wrapping_add(m & ONES);
    if shift {
        w.wrapping_sub(ONES)
    } else {
        w
    }
}

/// The emitted instruction network on one dword of two wire words (`hi` selects the
/// high word): masked table addresses, `x = lut ^ mask`, `y = mask & ones`,
/// `n = shift ? -ones : 0` from the sign-extended shift bit, `x + y + n`.
pub fn unpack_emitted(lut: &[u32; 256], mask: &[u32; 128], dword: u32, hi: bool) -> u32 {
    let (la, ma, sh) = if hi {
        (
            (dword >> 14) & 0x3fc,
            (dword >> 22) & 0x1fc,
            ((dword as i32) >> 31) as u32,
        )
    } else {
        (
            (dword << 2) & 0x3fc,
            (dword >> 6) & 0x1fc,
            ((dword >> 15) & 1).wrapping_neg(),
        )
    };
    let m = mask[(ma / 4) as usize];
    let x = lut[(la / 4) as usize] ^ m;
    let y = m & ONES;
    x.wrapping_add(y).wrapping_add(sh & NEG_ONES)
}

/// `a' = 2a + (s - MAGIC)`: the single `v_lshl_add_u32` fold of the chain accumulator.
pub fn fold_a(a: u32, s: i32) -> u32 {
    a.wrapping_shl(1)
        .wrapping_add((s as u32).wrapping_add(MINUS_MAGIC_I32))
}

/// One epoch's accumulation as emitted: `a = MAGIC + C`, `a'` from [`fold_a`],
/// `t = as_f32(a') + (-12582912.0)`, `t * 0.25`, `sum = fma(RN(d_act * d_w), t, sum)`
/// (`d_w` = `f32(d)`; no FP16 `d/4`).
pub fn epoch_model(d_act: f32, d_w: f32, c: i32, s: i32, sum: f32) -> f32 {
    let folded = fold_a(MAGIC.wrapping_add(c as u32), s);
    let t = f32::from_bits(folded) + f32::from_bits(iu4_fold::MAGIC_NEG);
    (d_act * d_w).mul_add(t * f32::from_bits(QUARTER), sum)
}

/// LUT chunks (32 dwords each, 8 in all) written by wave `wave` of `waves`.
pub fn stage_chunks(waves: u32, wave: u32) -> Vec<u32> {
    (0..8u32).filter(|c| c % waves == wave).collect()
}

// VGPRs added over the MQ4 layout.
const SET_LEN: u8 = 20;
const AOUT: u8 = SET0 + 2 * SET_LEN; // 132: unpacked operands, 8 dwords per tensor
const UNP: u8 = AOUT + 16; // 148: four words x five registers
const UNP_WORDS: u8 = 4;
const UNP_LEN: u8 = 5;
/// Entry-block LDS addresses (prologue temporaries 14, 15): `4 * tid` (MASK stores)
/// and `4 * lane` (LUT chunk stores).
const LDS_AM: u8 = 14;
const LDS_AL: u8 = 15;
/// Staging scratch (prologue temporaries): two MASK register sets and four LUT chunk registers.
const STG_MASK: [u8; 2] = [1, 6];
const STG_CHUNK: u8 = 10;

// SGPRs added over the MQ4 layout.
const SRD_WS: u8 = 68;
const SRD_US: u8 = 72;
const SROW: u8 = 76;
const UPOFF: u8 = 78; // gate/up acc entry: (M/2)*4, the up-row byte offset
const LITS: u8 = 80; // eight literal staging SGPRs 80..87

fn set_base(p: usize) -> u8 {
    SET0 + p as u8 * SET_LEN
}
/// Code dword `i` (words `2i`, `2i + 1`) of tensor `t`'s b128 in set `p`.
fn raw(p: usize, t: u8, i: u8) -> u8 {
    set_base(p) + 4 * t + i
}
fn xr(p: usize) -> u8 {
    set_base(p) + 8
}
/// `d_act` (f32) and `s` (i32) of the epoch's activation record.
fn dsr(p: usize) -> u8 {
    set_base(p) + 16
}
fn hr(p: usize, t: u8) -> u8 {
    set_base(p) + 18 + t
}
fn a_out(t: u8, i: u8) -> u8 {
    AOUT + 8 * t + 2 * i
}
/// Unpack pool register `k` of word `n`: 0 LUT address, 1 MASK address, 2 LUT value,
/// 3 MASK value, 4 shift addend.
fn unp(n: u8, k: u8) -> u8 {
    UNP + UNP_LEN * n + k
}
fn srd_c(t: u8) -> u8 {
    if t == 0 {
        SRD_W
    } else {
        SRD_U
    }
}
fn srd_s(t: u8) -> u8 {
    if t == 0 {
        SRD_WS
    } else {
        SRD_US
    }
}

fn ds_store_b32(addr: u8, data: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_store_b32 v{addr}, v{data}{o}"),
        vec![],
        vec![v(addr), v(data)],
    )
    .memory(MemoryClass::DsStore)
}

fn ds_load_b32(dst: u8, addr: u8, off: u32) -> Instruction {
    let o = if off == 0 {
        String::new()
    } else {
        format!(" offset:{off}")
    };
    Instruction::new(
        format!("ds_load_b32 v{dst}, v{addr}{o}"),
        vec![v(dst)],
        vec![v(addr)],
    )
    .memory(MemoryClass::DsLoad)
}

fn plan(g: &Gen, acc: bool) -> Result<RegPlan, String> {
    let tn = g.t();
    let gate_up = g.spec.kind == Kind::GateUp;
    let mut p = RegPlan::new(VGPR_CEILING, 104)?;
    let l = |n: &str| g.label(n);
    let kernel = || Live::Between("entry".into(), l("epilogue"));
    let pro = || Live::Between("entry".into(), l("k_begin"));
    let epi = || Live::Between(l("epilogue"), l("end"));
    p.v::<1>("tid", TID, pro())?;
    for i in 1..16u8 {
        p.v::<1>("prologue_tmp", i, pro())?;
    }
    for (name, r) in [
        ("x_off", XOFF),
        ("xq_off", XQOFF),
        ("w_off", WOFF),
        ("h_off", HOFF),
    ] {
        p.v::<1>(name, r, kernel())?;
    }
    p.v::<1>("y_off", YOFF, Live::Whole)?;
    p.v::<8>("magic8", MAGIC8, kernel())?;
    for t in 0..tn {
        p.v::<8>("cacc", CACC + 8 * t, kernel())?;
        p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
        p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
        p.v::<1>("scale_f32", HF + t, kernel())?;
    }
    p.v::<8>("fold_t", TPROD, kernel())?;
    for set in 0..2usize {
        for t in 0..tn {
            p.v::<4>("code_words", raw(set, t, 0), kernel())?;
        }
        for h in 0..2u8 {
            p.v::<4>("operand_x", xr(set) + 4 * h, kernel())?;
        }
        p.v::<2>("act_d_s", dsr(set), kernel())?;
        for t in 0..tn {
            p.v::<1>("scale_pair", hr(set, t), kernel())?;
        }
    }
    for t in 0..tn {
        for i in 0..4u8 {
            p.v::<2>("a_nibbles", a_out(t, i), kernel())?;
        }
    }
    for n in 0..UNP_WORDS {
        for k in 0..UNP_LEN {
            p.v::<1>("unpack_tmp", unp(n, k), kernel())?;
        }
    }
    // Epilogue registers over the dead K-loop ranges.
    for i in 0..8u8 {
        p.v::<1>("rt_tmp", RT_TMP + i, epi())?;
    }
    if gate_up {
        for i in 0..(6 * SILU_N) {
            p.v::<1>("silu_tmp", SET0 + i, epi())?;
        }
        for i in 0..(3 * SILU_N) {
            p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi())?;
        }
    }
    p.v::<4>("out", OUT, epi())?;
    p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
    p.s::<4>("srd_y", SRD_Y, epi())?;
    p.s::<1>("s_tmp0", T0, kernel())?;
    p.s::<1>("s_tmp1", T1, kernel())?;
    p.s::<2>("s_tmp64", T64, kernel())?;
    p.s::<8>("kernargs_0x00", ARGS0, kernel())?;
    p.s::<2>("y_base", ARGS1, Live::Whole)?;
    p.s::<2>("kernargs_m_k", ARGS1 + 2, kernel())?;
    p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
    p.s::<1>("x_src_rows", ARGS3, kernel())?;
    p.s::<1>("expert", EID, kernel())?;
    for t in 0..tn {
        p.s::<4>("srd_codes", srd_c(t), kernel())?;
        p.s::<4>("srd_scales", srd_s(t), kernel())?;
    }
    for r in SRD_X {
        p.s::<4>("srd_x", r, kernel())?;
    }
    p.s::<4>("srd_sorted", SRD_S, kernel())?;
    for (name, r) in [
        ("trips", TRIPS),
        ("x_stride2", XS2),
        ("row_bytes", ROWB),
        ("scale_row_bytes", SROW),
        ("tile16", TILE16),
        ("wave", WAVE),
        ("row_base", RBASE),
        ("wg_x", WGX),
        ("wg_y", WGY),
    ] {
        p.s::<1>(name, r, kernel())?;
    }
    p.s::<2>("w_ptr", WPTR, kernel())?;
    p.s::<2>("live", LIVE, Live::Whole)?;
    if acc && gate_up {
        p.s::<1>("up_offset", UPOFF, Live::Whole)?;
    }
    p.s::<8>("table_literals", LITS, pro())?;
    for m in MASKT {
        p.s::<2>("cmp_mask", m, Live::Whole)?;
    }
    let top = p.next_free_vgpr();
    if top % 8 != 0 {
        p.v::<1>("granule_pad", (top.div_ceil(8) * 8 - 1) as u8, Live::Whole)?;
    }
    Ok(p)
}

/// The shipped NT1 entry (BF16 epilogue).
pub(super) fn emit(spec: Spec) -> Result<Emitted, String> {
    emit_impl(spec, false)
}

/// Qualification entry `..._acc`: the same K loop, but stores the 16 rows' F32
/// accumulators (`Y[P][M]` f32; gate/up: gate rows then up rows; padding slots +0)
/// instead of the BF16 epilogue, so G1 compares the exact fold bytes against the
/// CPU twin. Never dispatched.
pub fn emit_acc(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    emit_impl(spec, true)
}

fn emit_impl(spec: Spec, acc: bool) -> Result<Emitted, String> {
    let g = Gen { spec, acc };
    let sfx = if acc { "_acc" } else { "" };
    let kspec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(),
        variant: format!("{}{sfx}", spec.variant()),
        arch: spec.arch,
        symbol: format!("{}{sfx}", spec.symbol()),
        kernargs: spec.kernargs(),
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.threads() as u16,
        group_segment_fixed_size: STAGE_BYTES,
        wave32: true,
        cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(&g, acc)?);
    b.enable_delay_alu();
    let mut wg = Workgroup::<Gfx1201, Builder>::new(&mut b)?;
    body(&mut wg, &g, acc)?;
    b.finish()
}

fn body(wg: &mut Wg<Gfx1201>, g: &Gen, acc: bool) -> Result<(), String> {
    let tables = wg.lds::<Tables>("e8p_tables", 0, STAGE_BYTES)?;
    let end = wg.exit(&g.label("end"))?;
    prologue(wg, g, &end, acc)?;
    let tables = stage(wg, g, tables)?;
    kloop(wg, g, tables)?;
    if acc {
        epilogue_acc(wg.isa(), g)?
    } else {
        epilogue(wg.isa(), g)?
    }
    wg.end(end)
}

fn prologue(wg: &mut Wg<Gfx1201>, g: &Gen, end: &End, acc: bool) -> Result<(), String> {
    let a = g.arch();
    let gate_up = g.spec.kind == Kind::GateUp;
    let b = wg.isa();
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
    sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(TID)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(TID)])?;
    op(
        b,
        format!("v_readfirstlane_b32 s{WAVE}, v1"),
        &[s(WAVE)],
        &[v(1)],
    )?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    // LDS stage store addresses: derived from the work-item id before the first branch.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LDS_AM}, 2, v0"),
        &[v(LDS_AM)],
        &[v(TID)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LDS_AL}, 2, v2"),
        &[v(LDS_AL)],
        &[v(2)],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"),
        &[TILE16],
        &[WGY],
    )?;
    let outside = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1),
        vec![],
        vec![s(TILE16), s(ARGS2 + 1)],
    ))?;
    wg.exit_if(outside, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, ARGS0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    // Fail closed: an expert id outside [0, n_experts) (the -1 sentinel, or a corrupt / non-finite-router id) exits
    // before any `expert_weight_ptrs[id]` load. One unsigned compare covers both bounds (negative ids are huge).
    smem(b, T1, 1, KARG, 0x3c)?;
    let sentinel = wg.scmp_wg_uniform(Instruction::new(
        format!("s_cmp_ge_u32 s{EID}, s{T1}"),
        vec![],
        vec![s(EID), s(T1)],
    ))?;
    wg.exit_if(sentinel, end)?;
    let b = wg.isa();
    sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    add64(b, T64, ARGS0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // Code row bytes K/4 (two bytes per eight weights); scale row bytes (K/128)*2.
    let (m, k) = (ARGS1 + 2, ARGS1 + 3);
    sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 2"), &[ROWB], &[k])?;
    sop(b, format!("s_lshr_b32 s{SROW}, s{k}, 6"), &[SROW], &[k])?;
    // Two-epoch trips before the tail: (K/128 - 1) / 2.
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(
        b,
        format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)),
        &[TRIPS],
        &[TRIPS],
    )?;
    sop(
        b,
        format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"),
        &[TRIPS],
        &[TRIPS],
    )?;
    // Row base of this wave: x*cta_rows + 16*wave.
    sop(
        b,
        format!(
            "s_lshl_b32 s{RBASE}, s{WGX}, {}",
            if gate_up { 5 } else { 6 }
        ),
        &[RBASE],
        &[WGX],
    )?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(
        b,
        format!("{} s{RBASE}, s{RBASE}, s{T0}", s_add_i32(a)),
        &[RBASE],
        &[RBASE, T0],
    )?;
    // Sorted slots of this tile.
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 6"), &[T0], &[WGY])?;
    add64(b, SRD_S, ARGS0 + 4, T0)?;
    sop(
        b,
        format!("s_mov_b32 s{}, 64", SRD_S + 2),
        &[SRD_S + 2],
        &[],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)),
        &[SRD_S + 3],
        &[],
    )?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    bload(b, 1, 6, 5, SRD_S, 0)?;
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{ARGS3}, 0x48"),
        &[T0],
        &[ARGS3],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0], ARGS0 + 6),
        &[SRD_X[0]],
        &[ARGS0 + 6],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_X[0] + 1, ARGS0 + 7),
        &[SRD_X[0] + 1],
        &[ARGS0 + 7],
    )?;
    srd_tail(b, SRD_X[0], Some(T0))?;
    add64(b, SRD_X[1], ARGS0 + 6, T0)?;
    srd_tail(b, SRD_X[1], Some(T0))?;
    sop(b, format!("s_lshl_b32 s{XS2}, s{T0}, 1"), &[XS2], &[T0])?;
    // Code descriptors: rows rbase.. of the expert (gate/up: up rows M/2 further).
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"),
        &[T0],
        &[RBASE, ROWB],
    )?;
    add64(b, SRD_W, WPTR, T0)?;
    srd_tail(b, SRD_W, None)?;
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"),
            &[T1],
            &[T1, ROWB],
        )?;
        add64(b, SRD_U, SRD_W, T1)?;
        srd_tail(b, SRD_U, None)?;
    }
    // Scale descriptors: the scale plane starts M*row_bytes after the tensor base.
    sop(
        b,
        format!("s_mul_i32 s{T0}, s{m}, s{ROWB}"),
        &[T0],
        &[m, ROWB],
    )?;
    add64(b, SRD_WS, WPTR, T0)?;
    sop(
        b,
        format!("s_mul_i32 s{T1}, s{RBASE}, s{SROW}"),
        &[T1],
        &[RBASE, SROW],
    )?;
    add64(b, SRD_WS, SRD_WS, T1)?;
    if gate_up {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
        sop(
            b,
            format!("s_mul_i32 s{T1}, s{T1}, s{SROW}"),
            &[T1],
            &[T1, SROW],
        )?;
        add64(b, SRD_US, SRD_WS, T1)?;
        srd_tail(b, SRD_US, None)?;
    }
    srd_tail(b, SRD_WS, None)?;
    // Lane offsets. Codes: lr*row_bytes + 16*hi (the lane's eight-word half of the epoch).
    op(
        b,
        format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"),
        &[v(WOFF)],
        &[s(ROWB), v(3)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v{WOFF}, v4, 4, v{WOFF}"),
        &[v(WOFF)],
        &[v(4), v(WOFF)],
    )?;
    // Scale row: the row whose scale this lane's broadcast slot serves.
    op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
    op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    op(
        b,
        format!("v_mul_u32_u24_e32 v{HOFF}, s{SROW}, v7"),
        &[v(HOFF)],
        &[s(SROW), v(7)],
    )?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2 (BF16) or *4 (F32 accumulators).
    // row_len is M/2 for the BF16 gate/up H and M otherwise.
    if gate_up && !acc {
        sop(b, format!("s_lshr_b32 s{T1}, s{m}, 1"), &[T1], &[m])?;
    } else {
        sop(b, format!("s_mov_b32 s{T1}, s{m}"), &[T1], &[m])?;
    }
    if gate_up && acc {
        // Up rows sit M/2 columns (4 bytes each) further.
        sop(b, format!("s_lshl_b32 s{UPOFF}, s{m}, 1"), &[UPOFF], &[m])?;
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"),
        &[v(8)],
        &[s(TILE16), v(3)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v8, v8, s{T1}"),
        &[v(8)],
        &[v(8), s(T1)],
    )?;
    op(
        b,
        format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"),
        &[v(9)],
        &[v(4), s(RBASE)],
    )?;
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{YOFF}, {}, v8", if acc { 2 } else { 1 }),
        &[v(YOFF)],
        &[v(8)],
    )?;
    // Activation row offset (padding slots get the gather's out-of-range offset).
    common::gather_offset(
        b,
        XOFF,
        6,
        Some(ARGS2),
        iu4_fold::XBLK_BYTES,
        LIVE,
        GatherTemps {
            v: [11, 12, 13, 10],
            mask: MASKT[0],
        },
    )?;
    // X nibbles of the lane's 64 K elements: byte 8 + 32*hi.
    op(
        b,
        format!("v_lshl_add_u32 v{XQOFF}, v4, 5, v{XOFF}"),
        &[v(XQOFF)],
        &[v(4), v(XOFF)],
    )?;
    load_set(b, g, 0)?;
    for j in 0..8u8 {
        op(
            b,
            format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)),
            &[v(MAGIC8 + j)],
            &[],
        )?;
    }
    for t in 0..g.t() {
        for j in 0..8u8 {
            op(
                b,
                format!("v_mov_b32_e32 v{}, 0", SUM + 8 * t + j),
                &[v(SUM + 8 * t + j)],
                &[],
            )?;
        }
    }
    Ok(())
}

/// Stage `LUT` and `MASK` once per CTA: every wave writes its share of the typed
/// region, drains, and the single workgroup barrier publishes it. All waves take the
/// same path to the barrier (the early exits are workgroup-uniform; the per-wave LUT
/// blocks are `skip_unless` joins before it).
fn stage(
    wg: &mut Wg<Gfx1201>,
    g: &Gen,
    tables: LdsRegion<Tables, Free>,
) -> Result<LdsRegion<Tables, Published>, String> {
    let waves = g.spec.waves();
    let threads = 32 * waves;
    let lut = build_lut256u32();
    let mut out = wg.begin_write(tables);
    // MASK[tid + threads*j] arithmetically, in every wave (the unconditional store that
    // puts the region into the writing state on every path).
    for j in 0..128 / threads {
        let [e, pc, x, y] = if j % 2 == 0 {
            [
                STG_MASK[0],
                STG_MASK[0] + 1,
                STG_MASK[0] + 2,
                STG_MASK[0] + 3,
            ]
        } else {
            [
                STG_MASK[1],
                STG_MASK[1] + 1,
                STG_MASK[1] + 2,
                STG_MASK[1] + 3,
            ]
        };
        let b = wg.isa();
        let src = if j == 0 {
            TID
        } else {
            op(
                b,
                format!("v_add_nc_u32_e32 v{e}, {}, v{TID}", lit(threads * j)),
                &[v(e)],
                &[v(TID)],
            )?;
            e
        };
        // x = sign7 | parity(sign7) << 7; parity by xor-shift reduction (sign7 < 128): bit 0 of
        // e ^ e>>4 ^ ... after folding 4, 2, 1.
        op(
            b,
            format!("v_lshrrev_b32_e32 v{x}, 4, v{src}"),
            &[v(x)],
            &[v(src)],
        )?;
        op(
            b,
            format!("v_xor_b32_e32 v{pc}, v{src}, v{x}"),
            &[v(pc)],
            &[v(src), v(x)],
        )?;
        for sh in [2u8, 1] {
            op(
                b,
                format!("v_lshrrev_b32_e32 v{x}, {sh}, v{pc}"),
                &[v(x)],
                &[v(pc)],
            )?;
            op(
                b,
                format!("v_xor_b32_e32 v{pc}, v{pc}, v{x}"),
                &[v(pc)],
                &[v(pc), v(x)],
            )?;
        }
        op(
            b,
            format!("v_lshlrev_b32_e32 v{pc}, 7, v{pc}"),
            &[v(pc)],
            &[v(pc)],
        )?;
        op(
            b,
            format!("v_bfi_b32 v{x}, 0x80, v{pc}, v{src}"),
            &[v(x)],
            &[v(pc), v(src)],
        )?;
        // Spread bit k to nibble k.
        for (shift, mask) in [(12u32, 0x000f_000fu32), (6, 0x0303_0303), (3, ONES)] {
            op(
                b,
                format!("v_lshl_or_b32 v{x}, v{x}, {shift}, v{x}"),
                &[v(x)],
                &[v(x)],
            )?;
            op(
                b,
                format!("v_and_b32_e32 v{x}, {}, v{x}", lit(mask)),
                &[v(x)],
                &[v(x)],
            )?;
        }
        // 0xF per set bit: (x << 4) - x.
        op(
            b,
            format!("v_lshlrev_b32_e32 v{y}, 4, v{x}"),
            &[v(y)],
            &[v(x)],
        )?;
        op(
            b,
            format!("v_sub_nc_u32_e32 v{y}, v{y}, v{x}"),
            &[v(y)],
            &[v(y), v(x)],
        )?;
        out = wg.ds_store(out, ds_store_b32(LDS_AM, y, LUT_BYTES + 4 * threads * j))?;
    }
    // LUT chunks: wave k writes chunks c % waves == k.
    for k in 0..waves {
        let cond = wg.scmp(Instruction::new(
            format!("s_cmp_eq_u32 s{WAVE}, {k}"),
            vec![],
            vec![s(WAVE)],
        ))?;
        out = wg.skip_unless(cond, &g.label(&format!("stage_w{k}")), out, |w, mut out| {
            for (n, c) in stage_chunks(waves, k).into_iter().enumerate() {
                let vc = STG_CHUNK + (n % 4) as u8;
                let b = w.isa();
                for batch in 0..4u32 {
                    for q in 0..8u32 {
                        let r = LITS + q as u8;
                        sop(
                            b,
                            format!(
                                "s_mov_b32 s{r}, {}",
                                lit(lut[(32 * c + 8 * batch + q) as usize])
                            ),
                            &[r],
                            &[],
                        )?;
                    }
                    for q in 0..8u32 {
                        let r = LITS + q as u8;
                        op(
                            b,
                            format!("v_writelane_b32 v{vc}, s{r}, {}", 8 * batch + q),
                            &[v(vc)],
                            &[v(vc), s(r)],
                        )?;
                    }
                }
                out = w.ds_store(out, ds_store_b32(LDS_AL, vc, 128 * c))?;
            }
            Ok(out)
        })?;
    }
    let (region, pending) = out;
    let drained = wg.wait(pending)?;
    let (published,) = wg.barrier((ready(region, drained),))?;
    Ok(published)
}

/// Issue set `p`'s loads (epoch parity `p` of the current trip): the trip's scale
/// pair of each tensor's header row, `d`/`s`, the activation nibbles of the lane's 64
/// K elements (two b128), then one b128 of eight words per tensor (byte `32p` of the
/// trip's two-epoch span), as one clause.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let tn = g.t();
    b.clause(|b| {
        for t in 0..tn {
            bload(b, 1, hr(p, t), HOFF, srd_s(t), 0)?;
        }
        bload(b, 2, dsr(p), XOFF, SRD_X[p], 0)?;
        bload(b, 4, xr(p), XQOFF, SRD_X[p], 8)?;
        bload(b, 4, xr(p) + 4, XQOFF, SRD_X[p], 24)?;
        for t in 0..tn {
            bload(b, 4, raw(p, t, 0), WOFF, srd_c(t), 32 * p as u32)?;
        }
        Ok(())
    })
}

/// Unpack step `i` of set `p` (every tensor's dword `i`, two wire words each) into
/// `a_out`: the table addresses and lookups of all words are issued first, then the
/// per-word network runs on the landed values.
fn unpack_step(
    w: &mut Wave<'_, Gfx1201, Builder>,
    g: &Gen,
    p: usize,
    i: u8,
    tables: &LdsRegion<Tables, Published>,
) -> Result<(), String> {
    let tn = g.t();
    for t in 0..tn {
        for which in 0..2u8 {
            let n = 2 * t + which;
            let word = raw(p, t, i);
            let (la, ma, sh) = (unp(n, 0), unp(n, 1), unp(n, 4));
            let b = w.isa();
            if which == 0 {
                op(
                    b,
                    format!("v_lshlrev_b32_e32 v{la}, 2, v{word}"),
                    &[v(la)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{la}, {}, v{la}", lit(0x3fc)),
                    &[v(la)],
                    &[v(la)],
                )?;
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{ma}, 6, v{word}"),
                    &[v(ma)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{ma}, {}, v{ma}", lit(0x1fc)),
                    &[v(ma)],
                    &[v(ma)],
                )?;
            } else {
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{la}, 14, v{word}"),
                    &[v(la)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{la}, {}, v{la}", lit(0x3fc)),
                    &[v(la)],
                    &[v(la)],
                )?;
                op(
                    b,
                    format!("v_lshrrev_b32_e32 v{ma}, 22, v{word}"),
                    &[v(ma)],
                    &[v(word)],
                )?;
                op(
                    b,
                    format!("v_and_b32_e32 v{ma}, {}, v{ma}", lit(0x1fc)),
                    &[v(ma)],
                    &[v(ma)],
                )?;
            }
            w.ds_load(tables, ds_load_b32(unp(n, 2), la, 0))?;
            w.ds_load(tables, ds_load_b32(unp(n, 3), ma, LUT_BYTES))?;
            // Shift addend `-ones` from the shift bit (independent of the lookups).
            let b = w.isa();
            if which == 0 {
                op(
                    b,
                    format!("v_bfe_i32 v{sh}, v{word}, 15, 1"),
                    &[v(sh)],
                    &[v(word)],
                )?;
            } else {
                op(
                    b,
                    format!("v_ashrrev_i32_e32 v{sh}, 31, v{word}"),
                    &[v(sh)],
                    &[v(word)],
                )?;
            }
            op(
                b,
                format!("v_and_b32_e32 v{sh}, {}, v{sh}", lit(NEG_ONES)),
                &[v(sh)],
                &[v(sh)],
            )?;
        }
    }
    let b = w.isa();
    for t in 0..tn {
        for which in 0..2u8 {
            let n = 2 * t + which;
            let (lv, mv, sh) = (unp(n, 2), unp(n, 3), unp(n, 4));
            op(
                b,
                format!("v_xor_b32_e32 v{lv}, v{mv}, v{lv}"),
                &[v(lv)],
                &[v(mv), v(lv)],
            )?;
            op(
                b,
                format!("v_and_b32_e32 v{mv}, {}, v{mv}", lit(ONES)),
                &[v(mv)],
                &[v(mv)],
            )?;
            let dst = a_out(t, i) + which;
            op(
                b,
                format!("v_add3_u32 v{dst}, v{lv}, v{mv}, v{sh}"),
                &[v(dst)],
                &[v(lv), v(mv), v(sh)],
            )?;
        }
    }
    Ok(())
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(
    w: &mut Wave<'_, Gfx1201, Builder>,
    g: &Gen,
    p: usize,
    tables: &LdsRegion<Tables, Published>,
) -> Result<(), String> {
    let a = g.arch();
    let tn = g.t();
    // Row scales: epoch parity selects the low (even) or high (odd) half of
    // the scale pair, then the ds_swizzle broadcast to the accumulator rows.
    let half = if p == 0 { "l" } else { "h" };
    for t in 0..tn {
        let b = w.isa();
        op(
            b,
            format!("v_cvt_f32_f16_e64 v{}, v{}.{half}", HF + t, hr(p, t)),
            &[v(HF + t)],
            &[v(hr(p, t))],
        )?;
        for j in 0..8u8 {
            let text = format!(
                "ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{j})",
                SCF + 8 * t + j,
                HF + t
            );
            b.ds_crosslane(
                Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)])
                    .memory(MemoryClass::DsLoad),
            )?;
        }
    }
    for i in 0..4u8 {
        unpack_step(w, g, p, i, tables)?;
        let b = w.isa();
        for t in 0..tn {
            iu4_fold::wmma_step(
                b,
                a,
                V::<8>(CACC + 8 * t),
                V::<2>(a_out(t, i)),
                V::<2>(xr(p) + 2 * i),
                i == 0,
                V::<8>(MAGIC8),
            )?;
        }
    }
    // s - MAGIC, in place (once per epoch, shared by the tensors).
    let b = w.isa();
    let (d, sm) = (dsr(p), dsr(p) + 1);
    op(
        b,
        format!("v_add_nc_u32_e32 v{sm}, {}, v{sm}", lit(MINUS_MAGIC_I32)),
        &[v(sm)],
        &[v(sm)],
    )?;
    for t in 0..tn {
        fold(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, d, sm, TPROD)?;
    }
    Ok(())
}

/// One tensor's fold of an accumulator fragment: `t_j = d_act * f32(d)_j` (four
/// packets, independent of the chain), `a' = 2a + (s - MAGIC)` (`v_lshl_add_u32`),
/// `a' + (-12582912.0)`, `* 0.25` (exact), `sum_j = fma(t_j, c_j, sum_j)`. `c`, `sc` and
/// `t` share a bank alignment as in [`iu4_fold::fold_pass`].
fn fold(b: &mut Builder, c: u8, sum: u8, sc: u8, d: u8, sm: u8, t: u8) -> Result<(), String> {
    let ts = |j: u8| t + (j ^ 2);
    for j in (0..8u8).step_by(2) {
        let x = VopdOp {
            op: VopdF32::Mul,
            dst: ts(j),
            src0: Operand::V(sc + j),
            src1: d,
        };
        let y = VopdOp {
            op: VopdF32::Mul,
            dst: ts(j + 1),
            src0: Operand::V(sc + j + 1),
            src1: d,
        };
        b.vopd(x, y)?;
    }
    for j in 0..8u8 {
        op(
            b,
            format!("v_lshl_add_u32 v{0}, v{0}, 1, v{sm}", c + j),
            &[v(c + j)],
            &[v(c + j), v(sm)],
        )?;
    }
    for (lit_bits, op_kind) in [(iu4_fold::MAGIC_NEG, VopdF32::Add), (QUARTER, VopdF32::Mul)] {
        for j in (0..8u8).step_by(2) {
            let x = VopdOp {
                op: op_kind,
                dst: c + j,
                src0: Operand::Lit(lit_bits),
                src1: c + j,
            };
            let y = VopdOp {
                op: op_kind,
                dst: c + j + 1,
                src0: Operand::Lit(lit_bits),
                src1: c + j + 1,
            };
            b.vopd(x, y)?;
        }
    }
    for j in (0..8u8).step_by(2) {
        let x = VopdOp {
            op: VopdF32::Fmac,
            dst: sum + j,
            src0: Operand::V(ts(j)),
            src1: c + j,
        };
        let y = VopdOp {
            op: VopdF32::Fmac,
            dst: sum + j + 1,
            src0: Operand::V(ts(j + 1)),
            src1: c + j + 1,
        };
        b.vopd(x, y)?;
    }
    Ok(())
}

fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.t() {
        add64_imm(b, srd_c(t), 64)?;
        add64_imm(b, srd_s(t), 4)?;
    }
    for r in SRD_X {
        add64(b, r, r, XS2)?;
    }
    Ok(())
}

fn kloop(
    wg: &mut Wg<Gfx1201>,
    g: &Gen,
    tables: LdsRegion<Tables, Published>,
) -> Result<(), String> {
    wg.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32", TRIPS)?;
    let tables = wg.wg_skip_if(no_trip, &done, tables, |wg, tables| {
        let tables = wg.loop_carried(&head, tables, |wg, tables| {
            load_set(wg.isa(), g, 1)?;
            compute(wg, g, 0, &tables)?;
            advance(wg.isa(), g)?;
            load_set(wg.isa(), g, 0)?;
            compute(wg, g, 1, &tables)?;
            op(
                wg.isa(),
                format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(Arch::Gfx1201)),
                &[s(TRIPS)],
                &[s(TRIPS)],
            )?;
            Ok((tables, trips_cmp(wg, "s_cmp_lg_u32", TRIPS)?))
        })?;
        if wg.isa().ledger.shape() != entry {
            return Err("k loop exit ledger differs from its entry".into());
        }
        Ok(tables)
    })?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => {
            load_set(wg.isa(), g, 1)?;
            compute(wg, g, 0, &tables)?;
            compute(wg, g, 1, &tables)
        }
        // K % 128 == 0: one epoch left when K/128 is odd, else two (the MQ4 down entry's branch).
        Kind::Down => {
            let odd = wg.scmp(Instruction::new(
                format!("s_bitcmp1_b32 s{}, 7", ARGS1 + 3),
                vec![],
                vec![s(ARGS1 + 3)],
            ))?;
            wg.if_else(
                odd,
                &g.label("tail_odd"),
                &g.label("epilogue"),
                |w| {
                    load_set(w.isa(), g, 1)?;
                    compute(w, g, 0, &tables)?;
                    compute(w, g, 1, &tables)
                },
                |w| compute(w, g, 0, &tables),
            )
        }
    }
}

/// F32 accumulators (`Y[P][M]` f32): lane (hi, lr) stores rows `rbase + 8hi..+8`
/// of token `lr` for each tensor (gate/up: the up rows M/2 columns further);
/// padding stores +0.
fn epilogue_acc(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let tn = g.t();
    b.label(&g.label("epilogue"))?;
    for t in 0..tn {
        for j in 0..8u8 {
            let r = SUM + 8 * t + j;
            op(
                b,
                format!("v_cndmask_b32_e64 v{r}, 0, v{r}, s{LIVE}"),
                &[v(r)],
                &[v(r), s(LIVE)],
            )?;
        }
    }
    sop(
        b,
        format!("s_mov_b32 s{SRD_Y}, s{ARGS1}"),
        &[SRD_Y],
        &[ARGS1],
    )?;
    sop(
        b,
        format!("s_mov_b32 s{}, s{}", SRD_Y + 1, ARGS1 + 1),
        &[SRD_Y + 1],
        &[ARGS1 + 1],
    )?;
    srd_tail(b, SRD_Y, None)?;
    if tn == 2 {
        op(
            b,
            format!("v_add_nc_u32_e32 v{RT_TMP}, s{UPOFF}, v{YOFF}"),
            &[v(RT_TMP)],
            &[s(UPOFF), v(YOFF)],
        )?;
    }
    for (t, voff) in [(0u8, YOFF), (1u8, RT_TMP)]
        .into_iter()
        .take(usize::from(tn))
    {
        bstore_b128(b, SUM + 8 * t, voff, SRD_Y, 0)?;
        bstore_b128(b, SUM + 8 * t + 4, voff, SRD_Y, 16)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        abs_row_parity, build_lut256u32, build_mask128u32, decode8, epoch_model, fold_a,
        stage_chunks, unpack_emitted, unpack_swar, unpack_word_ref, E8P_ABS2, LUT_BYTES,
        MINUS_MAGIC_I32, NEG_ONES, ONES, QUARTER, STAGE_BYTES,
    };
    use crate::kernels::iu4_fold::{self, MAGIC};

    const MAGIC_TWIN: u32 = 0x4b40_0000;

    /// The grid is the lexicographic list of every `{1,3,5}^8` row with `Σ(2g)^2 <= 40`
    /// (227 rows) followed by 29 distinct rows with five 3s and three 1s; 149 rows
    /// have odd `Σ|g|`.
    #[test]
    fn abs_grid_structure_is_the_norm_ball_plus_29() {
        let mut ball = Vec::new();
        let mut row = [1u8; 8];
        loop {
            if row
                .iter()
                .map(|&x| u32::from(x) * u32::from(x))
                .sum::<u32>()
                <= 40
            {
                ball.push(row);
            }
            // Odometer over {1,3,5}, last lane fastest.
            let mut k = 7;
            loop {
                if row[k] < 5 {
                    row[k] += 2;
                    break;
                }
                row[k] = 1;
                if k == 0 {
                    break;
                }
                k -= 1;
            }
            if row == [1u8; 8] {
                break;
            }
        }
        assert_eq!(ball.len(), 227);
        assert_eq!(&E8P_ABS2[..227], &ball[..]);
        let mut tail: Vec<[u8; 8]> = E8P_ABS2[227..].to_vec();
        assert_eq!(tail.len(), 29);
        for r in &tail {
            assert_eq!(r.iter().filter(|&&x| x == 3).count(), 5);
            assert_eq!(r.iter().filter(|&&x| x == 1).count(), 3);
        }
        tail.sort();
        tail.dedup();
        assert_eq!(tail.len(), 29);
        assert_eq!((0..=255u8).filter(|&i| abs_row_parity(i) == 1).count(), 149);
    }

    /// The tables and the SWAR network equal the per-lane reference for every
    /// `(idx, sign7, shift)`; the emitted network equals it on both halves of a
    /// dword for every word with arbitrary content in the other half.
    #[test]
    fn swar_and_emitted_networks_match_the_reference_for_every_word() {
        let (lut, mask) = (build_lut256u32(), build_mask128u32());
        let mut junk = 0x9e37_79b9u32;
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    let want = unpack_word_ref(idx, sign7, shift);
                    assert_eq!(
                        unpack_swar(&lut, &mask, idx, sign7, shift),
                        want,
                        "idx {idx} sign7 {sign7} shift {shift}"
                    );
                    let word = u32::from(idx) | u32::from(sign7) << 8 | u32::from(shift) << 15;
                    junk = junk.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    assert_eq!(
                        unpack_emitted(&lut, &mask, word | (junk & 0xffff_0000), false),
                        want,
                        "lo idx {idx} sign7 {sign7} shift {shift}"
                    );
                    assert_eq!(
                        unpack_emitted(&lut, &mask, (word << 16) | (junk >> 16), true),
                        want,
                        "hi idx {idx} sign7 {sign7} shift {shift}"
                    );
                }
            }
        }
    }

    /// No carry or borrow crosses a nibble and every operand is in `[-6, 5]`.
    #[test]
    fn unpacked_operands_are_signed_iu4() {
        let (lut, mask) = (build_lut256u32(), build_mask128u32());
        for idx in 0..=255u8 {
            for sign7 in 0..128u8 {
                for shift in [false, true] {
                    let w = unpack_swar(&lut, &mask, idx, sign7, shift);
                    let v = decode8(idx, sign7, shift);
                    for k in 0..8 {
                        let o = (((w >> (4 * k)) & 15) as i32) << 28 >> 28;
                        assert_eq!(2 * o + 1, i32::from(v[k]));
                        assert!((-6..=5).contains(&o));
                    }
                }
            }
        }
    }

    #[test]
    fn tables_have_the_documented_shape() {
        let (lut, mask) = (build_lut256u32(), build_mask128u32());
        assert_eq!(STAGE_BYTES, 1536);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], 0xf000_000f);
        assert_eq!(mask[3], 0x0000_00ff);
        assert_eq!(mask[127], 0xffff_ffff);
        assert_eq!(lut[0], 0x1111_1111);
        // Row 1 = [1,1,1,1,1,1,1,3]: sum|g| = 5 is odd, so lane 7 is stored negated.
        assert_eq!(abs_row_parity(1), 1);
        assert_eq!(lut[1], 0xd111_1111);
        assert_eq!(MAGIC, MAGIC_TWIN);
        assert_eq!(iu4_fold::MAGIC_NEG, MAGIC_TWIN | 0x8000_0000);
        assert_eq!(MINUS_MAGIC_I32, MAGIC_TWIN.wrapping_neg());
        assert_eq!(NEG_ONES, ONES.wrapping_neg());
        assert_eq!(f32::from_bits(QUARTER), 0.25);
    }

    /// The emitted fold (`2a + (s - MAGIC)`, `+ (-12582912.0)`, `* 0.25`, fma) is the
    /// exact `fma(RN(d_act d), (2C + s) / 4, sum)` over the whole `C`, `s` range.
    #[test]
    fn fold_is_exact_two_c_plus_s_and_quarter() {
        let mut sum = 0.37f32;
        let (d_act, d_w) = (0.0123f32, 0.8125f32);
        for c in -6144..=5120i32 {
            for s in [
                -1024i32, -513, -255, -64, -7, -1, 0, 1, 5, 64, 127, 511, 1023,
            ] {
                let a = MAGIC.wrapping_add(c as u32);
                let folded = fold_a(a, s);
                assert_eq!(
                    folded,
                    MAGIC.wrapping_add((2 * c + s) as u32),
                    "c {c} s {s}"
                );
                let t = f32::from_bits(folded) + f32::from_bits(iu4_fold::MAGIC_NEG);
                assert_eq!(t, (2 * c + s) as f32);
                let want = (d_act * d_w).mul_add(((2 * c + s) as f32) * 0.25, sum);
                assert_eq!(epoch_model(d_act, d_w, c, s, sum).to_bits(), want.to_bits());
                sum = want;
            }
        }
    }

    /// Publication coverage: the chunks partition the 8 LUT chunks across the waves,
    /// the MASK entries `tid + threads*j` cover all 128 entries, and every store ends
    /// inside the 1,536-byte stage.
    #[test]
    fn stage_partitions_cover_the_tables_inside_the_stage() {
        for waves in [2u32, 4] {
            let mut seen = [0u8; 8];
            for wave in 0..waves {
                for c in stage_chunks(waves, wave) {
                    seen[c as usize] += 1;
                    // lane 31's dword of chunk c
                    assert!(4 * 31 + 128 * c + 4 <= LUT_BYTES);
                }
            }
            assert_eq!(seen, [1u8; 8], "waves {waves}");
            let threads = 32 * waves;
            let mut entries = [0u8; 128];
            for j in 0..128 / threads {
                for tid in 0..threads {
                    entries[(tid + threads * j) as usize] += 1;
                    assert!(4 * tid + LUT_BYTES + 4 * threads * j + 4 <= STAGE_BYTES);
                }
            }
            assert_eq!(entries, [1u8; 128], "waves {waves}");
        }
    }
}
