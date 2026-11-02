// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 codec-local weight unpack adapters (B4-B).
//!
//! One K16 WMMA step consumes two dwords of A per lane: sixteen signed 4-bit
//! operands. Every codec turns its stored bits into those dwords **in place**
//! over the loaded registers, with explicit scratch:
//!
//! | codec | loaded per K16 step | operand dwords |
//! |---|---|---|
//! | MQ4 | the 8 stored nibbles per dword | `dw ^ 0x88888888` (unsigned `u` -> `u - 8`) |
//! | W3 | two dwords at epoch bytes `6i`, `6i+3`; low 24 bits = 8 codes | `(2c - 7) & 15` per nibble |
//! | W2 | two dwords at epoch bytes `4i`, `4i+2`; low 16 bits = 8 codes | `(2c - 3) & 15` per nibble |
//!
//! Only MQ4 applies the `0x88888888` rebias; W2/W3 are already signed after
//! their spread. The `*_model` functions are the CPU mirrors of the emitted
//! instruction DAGs, bit for bit, and are what the P1 proof enumerates.
//!
//! ## W2 spread (new adapter)
//!
//! Chunk `x`: code `k` (k = 0..7) is `x[2k+1:2k]`; bits above 15 are junk.
//! ```text
//! t = bfi(0x000000ff, x, x<<8)          codes 4..7 up by one byte
//! t = bfi(0x000f000f, t, t<<4)          code pairs (2,3),(6,7) up one nibble
//! t = bfi(0x03030303, t, t<<2)          odd codes into the next nibble
//! u = (t<<1) | 0x11111111               nibble = {junk, c1, c0, 1} = 2c+1 (mod 8 lanes)
//! a = u ^ 0x44444444                    bit2 = ~c1
//! v = bfi(0x77777777, a, a<<1)          bit3 = a.bit2: {~c1, ~c1, c0, 1}
//! ```
//! `{~c1,~c1,c0,1}` as a 4-bit two's complement is `2c - 3`
//! (`c = 0,1,2,3 -> -3,-1,1,3`). Every output nibble depends only on its own
//! two code bits, so junk in the other bits (high half, duplicated bytes)
//! cannot reach an output.
use super::*;

/// `v_bfi_b32 d, m, a, b = (a & m) | (b & !m)`.
pub fn bfi(m: u32, a: u32, b: u32) -> u32 {
    (a & m) | (b & !m)
}

/// MQ4: stored nibbles `u` become `u - 8` in two's complement.
pub fn mq4_unpack_model(dw: u32) -> u32 {
    dw ^ REBIAS
}

/// W3: the low 24 bits of `x` (junk above) -> eight signed nibbles
/// `(2c - 7) & 15`, code `k` in nibble `k`.
pub fn w3_spread_model(x: u32) -> u32 {
    let t = bfi(0xfff, x, x << 4);
    let t = bfi(0x003f_003f, t, t << 2);
    let t = bfi(0x0707_0707, t, t << 1);
    ((t << 1) | 0x1111_1111) ^ 0x8888_8888
}

/// W2: the low 16 bits of `x` (junk above) -> eight signed nibbles
/// `(2c - 3) & 15`, code `k` in nibble `k`.
pub fn w2_spread_model(x: u32) -> u32 {
    let t = bfi(0x0000_00ff, x, x << 8);
    let t = bfi(0x000f_000f, t, t << 4);
    let t = bfi(0x0303_0303, t, t << 2);
    let u = (t << 1) | 0x1111_1111;
    let a = u ^ 0x4444_4444;
    bfi(0x7777_7777, a, a << 1)
}

fn shl(b: &mut Builder, d: u8, n: u8, src: u8) -> Result<(), String> {
    op(
        b,
        format!("v_lshlrev_b32_e32 v{d}, {n}, v{src}"),
        &[v(d)],
        &[v(src)],
    )
}
fn bfi_op(b: &mut Builder, d: u8, mask: u32, p: u8, q: u8) -> Result<(), String> {
    op(
        b,
        format!("v_bfi_b32 v{d}, {}, v{p}, v{q}", lit(mask)),
        &[v(d)],
        &[v(p), v(q)],
    )
}
fn or_lit(b: &mut Builder, d: u8, k: u32, src: u8) -> Result<(), String> {
    op(
        b,
        format!("v_or_b32_e32 v{d}, {}, v{src}", lit(k)),
        &[v(d)],
        &[v(src)],
    )
}
fn xor_lit(b: &mut Builder, d: u8, k: u32, src: u8) -> Result<(), String> {
    op(
        b,
        format!("v_xor_b32_e32 v{d}, {}, v{src}", lit(k)),
        &[v(d)],
        &[v(src)],
    )
}

/// One W3 chunk in place: `raw` (low 24 bits = 8 codes) becomes eight signed
/// nibbles. `x`, `y` are scratch. Mirrors [`w3_spread_model`].
pub(super) fn w3_chunk(b: &mut Builder, raw: u8, x: u8, y: u8) -> Result<(), String> {
    shl(b, y, 4, raw)?;
    bfi_op(b, y, 0xfff, raw, y)?;
    shl(b, x, 2, y)?;
    bfi_op(b, y, 0x003f_003f, y, x)?;
    shl(b, x, 1, y)?;
    bfi_op(b, y, 0x0707_0707, y, x)?;
    shl(b, x, 1, y)?;
    or_lit(b, y, 0x1111_1111, x)?;
    xor_lit(b, raw, 0x8888_8888, y)
}

/// One W2 chunk in place: `raw` (low 16 bits = 8 codes) becomes eight signed
/// nibbles. `x`, `y` are scratch. Mirrors [`w2_spread_model`].
pub(super) fn w2_chunk(b: &mut Builder, raw: u8, x: u8, y: u8) -> Result<(), String> {
    shl(b, x, 8, raw)?;
    bfi_op(b, y, 0x0000_00ff, raw, x)?;
    shl(b, x, 4, y)?;
    bfi_op(b, y, 0x000f_000f, y, x)?;
    shl(b, x, 2, y)?;
    bfi_op(b, y, 0x0303_0303, y, x)?;
    shl(b, x, 1, y)?;
    or_lit(b, y, 0x1111_1111, x)?;
    xor_lit(b, y, 0x4444_4444, y)?;
    shl(b, x, 1, y)?;
    bfi_op(b, raw, 0x7777_7777, y, x)
}

/// Unpack every A register of one epoch (all tensors) into signed nibble
/// dwords, in place.
pub(super) fn unpack_epoch(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.tn() {
        match g.spec.codec {
            Codec::Mq4 => {
                for i in 0..16u8 {
                    let r = g.a(t) + i;
                    xor_lit(b, r, REBIAS, r)?;
                }
            }
            Codec::E8p => return Err("E8P operands are unpacked by `e8p::unpack_step` (LDS tables), not the in-place register spread".into()),
            Codec::W2 | Codec::W3 => {
                for step in 0..8u8 {
                    for which in 0..2u8 {
                        let n = ((t * 8 + step * 2 + which) % 2) as u8;
                        let (x, y) = g.scratch(n);
                        let raw = g.a(t) + 2 * step + which;
                        if g.spec.codec == Codec::W2 { w2_chunk(b, raw, x, y)? } else { w3_chunk(b, raw, x, y)? }
                    }
                }
            }
        }
    }
    Ok(())
}
