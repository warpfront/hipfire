// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Wave32 WMMA register-fragment gather/scatter. The numerical contract
//! (accumulation order, rounding, saturation) lives in `super::mma`; this
//! module only maps ISA fragments to logical matrices, so a mapping error
//! cannot hide inside a numerics model. All A/B/C inputs are gathered before
//! any destination write (D may alias A/B/C), and EXEC is ignored: WMMA
//! always reads and writes all 32 lanes.
use super::{mma, Result, State};
use crate::{
    operand::{Omod, Operand},
    reg::Kind,
    Arch, Inst,
};

#[derive(Clone, Copy)]
enum Shape {
    F16,
    Bf16,
    Iu8,
    Iu4 { k: usize },
}

pub(super) fn execute(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    if name.starts_with("v_swmmac") {
        return swmmac(arch, name, i, s);
    }
    let shape = match (arch, name) {
        (Arch::Gfx1151 | Arch::Gfx1201, "v_wmma_f32_16x16x16_f16") => Shape::F16,
        (Arch::Gfx1151 | Arch::Gfx1201, "v_wmma_f32_16x16x16_bf16") => Shape::Bf16,
        (Arch::Gfx1151 | Arch::Gfx1201, "v_wmma_i32_16x16x16_iu8") => Shape::Iu8,
        (Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4") => Shape::Iu4 { k: 16 },
        (Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4") => Shape::Iu4 { k: 32 },
        _ => return Err(format!("{name} on {arch:?}: no source-verified fragment mapping and probed numerics; refusing to approximate")),
    };
    let m = &i.mods;
    if m.neg != 0 || m.abs != 0 || m.omod != Omod::None || m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!(
            "{name}: VOP3 neg/abs/omod/DPP/SDWA modifiers are not part of WMMA"
        ));
    }
    if m.op_sel != 0 || !matches!(m.op_sel_hi, 0 | 7) {
        return Err(format!(
            "{name}: op_sel={:#x} op_sel_hi={:#x} has no modeled WMMA meaning",
            m.op_sel, m.op_sel_hi
        ));
    }
    let o = &i.operands;
    if o.len() != 4 {
        return Err(format!(
            "{name}: expected vdst,A,B,C operands, got {}",
            o.len()
        ));
    }
    let gfx12 = arch == Arch::Gfx1201;
    let (a_regs, b_regs) = match shape {
        Shape::F16 | Shape::Bf16 => (if gfx12 { 4 } else { 8 }, if gfx12 { 4 } else { 8 }),
        Shape::Iu8 => (if gfx12 { 2 } else { 4 }, if gfx12 { 2 } else { 4 }),
        Shape::Iu4 { .. } => (2, 2),
    };
    vgprs(&o[0], 8, "D")?;
    vgprs(&o[1], a_regs, "A")?;
    vgprs(&o[2], b_regs, "B")?;
    match &o[3] {
        Operand::Reg(_) => vgprs(&o[3], 8, "C")?,
        Operand::Inline(_) | Operand::Literal(_) => {}
        other => return Err(format!("{name}: unsupported C source {other:?}")),
    }
    if !gfx12 {
        // RDNA3 requires the A/B fragments replicated across lanes 0..15 and 16..31;
        // divergent halves are out of contract, so reject them before any gather.
        for (role, op, regs) in [("A", &o[1], a_regs), ("B", &o[2], b_regs)] {
            for vgpr in 0..usize::from(regs) {
                for lane in 0..16 {
                    let (lo, hi) = (s.read(op, lane, vgpr)?, s.read(op, lane + 16, vgpr)?);
                    if lo != hi {
                        return Err(format!("{name}: gfx11 {role} fragment VGPR {vgpr} lane {lane} ({lo:#010x}) differs from replica lane {} ({hi:#010x})", lane + 16));
                    }
                }
            }
        }
    }

    let mut out = [[0u32; 32]; 8];
    match shape {
        Shape::F16 | Shape::Bf16 => {
            let model = if matches!(shape, Shape::Bf16) {
                mma::wmma_f32_bf16
            } else {
                mma::wmma_f32_f16
            };
            if m.clamp {
                return Err(format!("{name}: clamp has no probed float-WMMA semantics"));
            }
            // NEG[1:0]/NEG_HI[1:0] negate A/B low/high halves; {NEG_HI[2],NEG[2]} = {ABS,NEG} on C.
            let flip = |operand: u8, half: usize| -> u16 {
                let bits = if half == 0 { m.neg_lo } else { m.neg_hi };
                if bits >> operand & 1 != 0 {
                    0x8000
                } else {
                    0
                }
            };
            let mut a = [[0u16; 16]; 16];
            let mut b = [[0u16; 16]; 16];
            for idx in 0..16 {
                for k in 0..16 {
                    let (lane, vgpr, half) = f16_loc(arch, idx, k);
                    a[idx][k] = (s.read(&o[1], lane, vgpr)? >> (16 * half)) as u16 ^ flip(0, half);
                    b[idx][k] = (s.read(&o[2], lane, vgpr)? >> (16 * half)) as u16 ^ flip(1, half);
                }
            }
            let mut c = [[0u32; 16]; 16];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    let mut v = s.read(&o[3], lane, vgpr)?;
                    if m.neg_hi & 4 != 0 {
                        v &= 0x7fff_ffff;
                    }
                    if m.neg_lo & 4 != 0 {
                        v ^= 0x8000_0000;
                    }
                    c[row][col] = v;
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    out[vgpr][lane] = model(arch, &a[row], &b[col], c[row][col]);
                }
            }
        }
        Shape::Iu8 => {
            // NEG[0]/NEG[1] are the A/B signedness bits; NEG[2] and NEG_HI must be zero.
            if m.neg_lo & 4 != 0 || m.neg_hi != 0 {
                return Err(format!(
                    "{name}: NEG[2]/NEG_HI are undefined for integer WMMA"
                ));
            }
            let signed = [m.neg_lo & 1 != 0, m.neg_lo & 2 != 0];
            let byte = |word: u32, n: usize, signed: bool| -> i16 {
                let v = (word >> (8 * n)) as u8;
                if signed {
                    i16::from(v as i8)
                } else {
                    i16::from(v)
                }
            };
            let mut a = [[0i16; 16]; 16];
            let mut b = [[0i16; 16]; 16];
            for idx in 0..16 {
                for k in 0..16 {
                    let (lane, vgpr, n) = iu8_loc(arch, idx, k);
                    a[idx][k] = byte(s.read(&o[1], lane, vgpr)?, n, signed[0]);
                    b[idx][k] = byte(s.read(&o[2], lane, vgpr)?, n, signed[1]);
                }
            }
            let mut c = [[0i32; 16]; 16];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    c[row][col] = s.read(&o[3], lane, vgpr)? as i32;
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    out[vgpr][lane] =
                        mma::wmma_i32_iu8(arch, &a[row], &b[col], c[row][col], m.clamp)? as u32;
                }
            }
        }
        Shape::Iu4 { k: depth } => {
            // NEG[0]/NEG[1] are the A/B signedness bits; NEG[2] and NEG_HI must be zero.
            if m.neg_lo & 4 != 0 || m.neg_hi != 0 {
                return Err(format!(
                    "{name}: NEG[2]/NEG_HI are undefined for integer WMMA"
                ));
            }
            let signed = [m.neg_lo & 1 != 0, m.neg_lo & 2 != 0];
            let nibble = |word: u32, n: usize, signed: bool| -> i8 {
                let v = ((word >> (4 * n)) & 15) as i8;
                if signed {
                    (v << 4) >> 4
                } else {
                    v
                }
            };
            let mut a = [[0i8; 32]; 16];
            let mut b = [[0i8; 32]; 16];
            for idx in 0..16 {
                for k in 0..depth {
                    let (lane, vgpr, n) = iu4_loc(arch, idx, k);
                    a[idx][k] = nibble(s.read(&o[1], lane, vgpr)?, n, signed[0]);
                    b[idx][k] = nibble(s.read(&o[2], lane, vgpr)?, n, signed[1]);
                }
            }
            let mut c = [[0i32; 16]; 16];
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    c[row][col] = s.read(&o[3], lane, vgpr)? as i32;
                }
            }
            for row in 0..16 {
                for col in 0..16 {
                    let (lane, vgpr) = acc_loc(arch, row, col);
                    out[vgpr][lane] = mma::wmma_i32_iu4(
                        arch,
                        &a[row][..depth],
                        &b[col][..depth],
                        c[row][col],
                        m.clamp,
                    ) as u32;
                }
            }
        }
    }
    for (vgpr, lanes) in out.iter().enumerate() {
        for (lane, &value) in lanes.iter().enumerate() {
            s.put(&o[0], lane, vgpr, value)?;
        }
    }
    Ok(())
}

/// Sparse WMMA (gfx1201 wave32): `D = A(sparse 16xK) * B(Kx16) + D`, K = 32 (64 for `iu4_x64`), 4:2 structured.
///
/// One layout rule serves every executing kind (receipts `rcpt/gfx1201-C-r1` one-hot discovery and `gfx1201-C-r2`
/// random raw-image waves, 0 mismatches for each kind and variant):
/// * A holds `nA` packed elements per lane (8 for the 16-bit float kinds, IU8 and IU4 x32, 16 for IU4 x64), element
///   `i` in VGPR `i / (32/bits)`, field `i % (32/bits)`. Lane `l` is row `l & 15`, half `h = l >> 4`; its elements are
///   the compressed slots `j = nA*h + i`.
/// * index: the SAME lane's word. Selector of slot `i` = `(idx >> (16*key + 2*i)) & 3` for `nA == 8`, where
///   `key = index_key` = OPSEL[0]; `(idx >> 2*i) & 3` for IU4 x64 (32 bits, no key).
/// * slot `j` meets dense `K = 4*(j >> 1) + selector`. Slots of one K-group may carry equal selectors; both then
///   contribute separately (no merging, zeros never enter).
/// * B: lane `l` is column `l & 15`, `hb = l >> 4`; element `e` (VGPR `e / (32/bits)`, field `e % (32/bits)`) is
///   `K = (Ktot/2)*hb + e`.
/// * C/D: WMMA accumulator layout, VDST read as C, EXEC ignored.
/// Float kinds (f32-out f16/bf16): the dense WMMA FP32 chain of [`mma`] over the 16 slots, term `c` taken from slot
/// `(c&3)|((c&4)<<1)|((c&8)>>1)`; `neg_lo[0]`/`neg_hi[0]` negate A of even/odd SLOTS, `neg_lo[1]`/`neg_hi[1]` the
/// selected B of even/odd slots (slot parity, not K parity). Integer kinds: `neg_lo[0]`/`neg_lo[1]` mark A/B signed;
/// CLAMP is the two-half saturation of [`mma::swmmac_i32`].
/// Exact-domain kinds (`f16_16x16x32_f16`, `bf16_16x16x32_bf16`, `f32_16x16x32_{fp8,bf8}_{fp8,bf8}`) have no
/// rounding/NaN/denormal model and execute only where the result is exactly determined ([`swmmac_exact`]): every
/// A, selected-B and C value finite normal or zero, and `|C| + sum|a*b| <= 2^P * q` (P = 24/11/8 for f32/f16/bf16
/// out, `q` the common power-of-two granule of C and the products), so every accumulation order yields the same
/// exactly representable total; anything else is a hard error. 16-bit-out D is 4 VGPRs, row `r` col `c` at lane
/// `c + 16*(r>>3)`, VGPR `(r&7)>>1`, half `r&1`. FP8 = E4M3 (`S.1111.111` NaN, no infinity), BF8 = E5M2.
/// Refused as unmeasured: neg_lo[2]/neg_hi[2] (C), NEG_HI on integers, CLAMP on floats, any neg/clamp on the FP8
/// kinds, and any other OPSEL bit.
fn swmmac(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    // (floating kind, bf16 inputs, bits per element, nA, A regs, B regs, D regs, exact-domain kind)
    let (float, bf16, bits, n_a, a_regs, b_regs, d_regs, exact) = match (arch, name) {
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_f16") => (true, false, 16, 8, 4, 8, 8, None),
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_bf16") => (true, true, 16, 8, 4, 8, 8, None),
        (Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu8") => (false, false, 8, 8, 2, 4, 8, None),
        (Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu4") => (false, false, 4, 8, 1, 2, 8, None),
        (Arch::Gfx1201, "v_swmmac_i32_16x16x64_iu4") => (false, false, 4, 16, 2, 4, 8, None),
        (Arch::Gfx1201, "v_swmmac_f16_16x16x32_f16") => (
            true,
            false,
            16,
            8,
            4,
            8,
            4,
            Some(Exact::new(FMT_F16, FMT_F16, FMT_F16, true)),
        ),
        (Arch::Gfx1201, "v_swmmac_bf16_16x16x32_bf16") => (
            true,
            true,
            16,
            8,
            4,
            8,
            4,
            Some(Exact::new(FMT_BF16, FMT_BF16, FMT_BF16, true)),
        ),
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_fp8_fp8") => (
            true,
            false,
            8,
            8,
            2,
            4,
            8,
            Some(Exact::new(FMT_FP8, FMT_FP8, FMT_F32, false)),
        ),
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_fp8_bf8") => (
            true,
            false,
            8,
            8,
            2,
            4,
            8,
            Some(Exact::new(FMT_FP8, FMT_BF8, FMT_F32, false)),
        ),
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_bf8_fp8") => (
            true,
            false,
            8,
            8,
            2,
            4,
            8,
            Some(Exact::new(FMT_BF8, FMT_FP8, FMT_F32, false)),
        ),
        (Arch::Gfx1201, "v_swmmac_f32_16x16x32_bf8_bf8") => (
            true,
            false,
            8,
            8,
            2,
            4,
            8,
            Some(Exact::new(FMT_BF8, FMT_BF8, FMT_F32, false)),
        ),
        _ => {
            return Err(format!(
            "{name} on {arch:?}: no probed SWMMAC numerics/index model; refusing to approximate"
        ))
        }
    };
    let m = &i.mods;
    if m.neg != 0 || m.abs != 0 || m.omod != Omod::None || m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!(
            "{name}: VOP3 neg/abs/omod/DPP/SDWA modifiers are not part of SWMMAC"
        ));
    }
    if !matches!(m.op_sel_hi, 0 | 7) {
        return Err(format!(
            "{name}: op_sel_hi={:#x} has no modeled SWMMAC meaning",
            m.op_sel_hi
        ));
    }
    let has_key = n_a == 8;
    if m.op_sel & !u8::from(has_key) != 0 {
        return Err(format!(
            "{name}: op_sel={:#x}: only index_key (OPSEL[0], absent on the x64 form) is measured",
            m.op_sel
        ));
    }
    let key = usize::from(m.op_sel & 1);
    if m.neg_lo & 4 != 0 || m.neg_hi & 4 != 0 {
        return Err(format!(
            "{name}: NEG[2]/NEG_HI[2] (accumulator) are not measured for SWMMAC"
        ));
    }
    if float && m.clamp {
        return Err(format!(
            "{name}: CLAMP of a floating-point SWMMAC is not measured"
        ));
    }
    if let Some(ex) = &exact {
        if !ex.neg_ok && (m.neg_lo != 0 || m.neg_hi != 0) {
            return Err(format!(
                "{name}: neg_lo/neg_hi do not exist on the FP8/BF8 SWMMAC forms"
            ));
        }
    }
    if !float && m.neg_hi != 0 {
        return Err(format!("{name}: NEG_HI is undefined for integer SWMMAC"));
    }
    let o = &i.operands;
    if o.len() != 4 {
        return Err(format!(
            "{name}: expected vdst,A,B,index operands, got {}",
            o.len()
        ));
    }
    vgprs(&o[0], d_regs, "D")?;
    vgprs(&o[1], a_regs, "A")?;
    vgprs(&o[2], b_regs, "B")?;
    vgprs(&o[3], 1, "index")?;
    let per = 32 / bits;
    let field = |word: u32, n: usize| (word >> (bits * (n % per))) & ((1u32 << bits) - 1);
    let k_total = 4 * n_a;
    let swap = |c: usize| (c & 3) | ((c & 4) << 1) | ((c & 8) >> 1);
    let signed = [m.neg_lo & 1 != 0, m.neg_lo & 2 != 0];
    let decode = |raw: u32, signed: bool| -> i16 {
        let v = raw as i16;
        if signed {
            (v << (16 - bits)) >> (16 - bits)
        } else {
            v
        }
    };
    let mut out = [[0u32; 32]; 8];
    for row in 0..16 {
        // Compressed A slots of this row and the dense K each one selects.
        let mut a_slot = [0u32; 32];
        let mut k_slot = [0usize; 32];
        for j in 0..2 * n_a {
            let (lane, e) = (row + 16 * (j / n_a), j % n_a);
            a_slot[j] = field(s.read(&o[1], lane, e / per)?, e);
            let shift = if has_key { 16 * key + 2 * e } else { 2 * e };
            k_slot[j] = 4 * (j >> 1) + (s.read(&o[3], lane, 0)? >> shift) as usize % 4;
        }
        for col in 0..16 {
            let mut b_slot = [0u32; 32];
            for j in 0..2 * n_a {
                let (b_lane, e) = (
                    col + 16 * (k_slot[j] / (k_total / 2)),
                    k_slot[j] % (k_total / 2),
                );
                b_slot[j] = field(s.read(&o[2], b_lane, e / per)?, e);
            }
            let (lane, vgpr) = acc_loc(arch, row, col);
            if let Some(ex) = &exact {
                // 16-bit-out D packs two rows per VGPR: row r -> VGPR (r&7)>>1, half r&1. f32-out is a whole word.
                let wide = ex.out.mbits == 23;
                let (vgpr, half) = if wide {
                    (vgpr, 0)
                } else {
                    (vgpr >> 1, row & 1)
                };
                let word = s.read(&o[0], lane, vgpr)?;
                let c_raw = if wide {
                    word
                } else {
                    (word >> (16 * half)) & 0xffff
                };
                let r = swmmac_exact(
                    name,
                    ex,
                    &a_slot[..2 * n_a],
                    &b_slot[..2 * n_a],
                    (m.neg_lo, m.neg_hi),
                    c_raw,
                )?;
                out[vgpr][lane] |= r << (16 * half);
                continue;
            }
            let c = s.read(&o[0], lane, vgpr)?;
            out[vgpr][lane] = if float {
                let flip = |lo: bool, hi: bool, j: usize| -> u16 {
                    let negate = if j % 2 == 0 { lo } else { hi };
                    if negate {
                        0x8000
                    } else {
                        0
                    }
                };
                let (mut a, mut b) = ([0u16; 16], [0u16; 16]);
                for t in 0..16 {
                    let j = swap(t);
                    a[t] = a_slot[j] as u16 ^ flip(m.neg_lo & 1 != 0, m.neg_hi & 1 != 0, j);
                    b[t] = b_slot[j] as u16 ^ flip(m.neg_lo & 2 != 0, m.neg_hi & 2 != 0, j);
                }
                (if bf16 {
                    mma::wmma_f32_bf16
                } else {
                    mma::wmma_f32_f16
                })(arch, &a, &b, c)
            } else {
                let (mut a, mut b) = ([0i16; 32], [0i16; 32]);
                for j in 0..2 * n_a {
                    a[j] = decode(a_slot[j], signed[0]);
                    b[j] = decode(b_slot[j], signed[1]);
                }
                mma::swmmac_i32(arch, &a[..2 * n_a], &b[..2 * n_a], c as i32, m.clamp) as u32
            };
        }
    }
    for (vgpr, lanes) in out.iter().enumerate().take(usize::from(d_regs)) {
        for (lane, &value) in lanes.iter().enumerate() {
            s.put(&o[0], lane, vgpr, value)?;
        }
    }
    Ok(())
}

fn vgprs(op: &Operand, want: u8, role: &str) -> Result<()> {
    match op {
        Operand::Reg(r) if r.kind == Kind::V && r.len == want => Ok(()),
        other => Err(format!(
            "WMMA {role} must be v[..] of {want} registers, got {other:?}"
        )),
    }
}

/// IEEE-style binary float layout for the exact-domain SWMMAC kinds. `e4m3` marks OCP FP8 E4M3 (`S.1111.111` is
/// the only NaN, no infinity); every other format reserves the all-ones exponent for inf/NaN.
#[derive(Clone, Copy)]
struct Fmt {
    ebits: u32,
    mbits: u32,
    bias: i32,
    e4m3: bool,
}
const FMT_F32: Fmt = Fmt {
    ebits: 8,
    mbits: 23,
    bias: 127,
    e4m3: false,
};
const FMT_F16: Fmt = Fmt {
    ebits: 5,
    mbits: 10,
    bias: 15,
    e4m3: false,
};
const FMT_BF16: Fmt = Fmt {
    ebits: 8,
    mbits: 7,
    bias: 127,
    e4m3: false,
};
const FMT_FP8: Fmt = Fmt {
    ebits: 4,
    mbits: 3,
    bias: 7,
    e4m3: true,
};
const FMT_BF8: Fmt = Fmt {
    ebits: 5,
    mbits: 2,
    bias: 15,
    e4m3: false,
};

/// Operand/accumulator formats of one exact-domain SWMMAC kind. `neg_ok`: the 16-bit kinds accept neg_lo/neg_hi.
#[derive(Clone, Copy)]
struct Exact {
    a: Fmt,
    b: Fmt,
    out: Fmt,
    neg_ok: bool,
}
impl Exact {
    const fn new(a: Fmt, b: Fmt, out: Fmt, neg_ok: bool) -> Self {
        Exact { a, b, out, neg_ok }
    }
}

/// A finite normal or zero value `(-1)^neg * m * 2^e` with `m` odd (`m == 0` is a zero of sign `neg`).
#[derive(Clone, Copy)]
struct Dyadic {
    neg: bool,
    m: u64,
    e: i32,
}

fn exact_decode(name: &str, fmt: Fmt, raw: u32, role: &str) -> Result<Dyadic> {
    let neg = (raw >> (fmt.ebits + fmt.mbits)) & 1 != 0;
    let ef = (raw >> fmt.mbits) & ((1 << fmt.ebits) - 1);
    let mf = raw & ((1 << fmt.mbits) - 1);
    if ef == (1 << fmt.ebits) - 1 && !(fmt.e4m3 && mf != (1 << fmt.mbits) - 1) {
        return Err(format!(
            "{name}: {role} value {raw:#x} is NaN or infinite; non-finite SWMMAC numerics are not measured"
        ));
    }
    if ef == 0 {
        if mf != 0 {
            return Err(format!(
                "{name}: {role} value {raw:#x} is denormal; denormal SWMMAC numerics are not measured"
            ));
        }
        return Ok(Dyadic { neg, m: 0, e: 0 });
    }
    let m = (1u64 << fmt.mbits) | u64::from(mf);
    let tz = m.trailing_zeros();
    Ok(Dyadic {
        neg,
        m: m >> tz,
        e: ef as i32 - fmt.bias - fmt.mbits as i32 + tz as i32,
    })
}

/// Encodes `total * 2^q_e` (`total != 0`, `|total| <= 2^P`, P = significand bits of `fmt`) as a normal number of
/// `fmt`; the value is exactly representable by construction, only the exponent range can fail.
fn exact_encode(name: &str, fmt: Fmt, total: i128, q_e: i32) -> Result<u32> {
    let tz = total.trailing_zeros();
    let m = total.unsigned_abs() >> tz;
    let bl = 128 - m.leading_zeros() as i32;
    let top = q_e + tz as i32 + bl - 1;
    let emax = (1 << fmt.ebits) - 2 - fmt.bias;
    if top < 1 - fmt.bias || top > emax {
        return Err(format!(
            "{name}: exact result 2^{top}-scale is outside the normal range of the output format"
        ));
    }
    let p = fmt.mbits as i32 + 1;
    let frac = ((m << (p - bl)) as u32) & ((1 << fmt.mbits) - 1);
    Ok(u32::from(total < 0) << (fmt.ebits + fmt.mbits)
        | ((top + fmt.bias) as u32) << fmt.mbits
        | frac)
}

/// One output element of an exact-domain SWMMAC: `C + sum_j a_j * b_j` over the 16 compressed slots, computed as an
/// exact dyadic rational. Executes only when the result is order-independent and exactly representable:
/// every input finite normal or zero, `|C| + sum|t_j| <= 2^P * q` (`q` the smallest power of two dividing C and every
/// nonzero product `t_j`), the total a normal number of the output format; an exact zero total only when C and every
/// product are `+0` (a cancelled or negative zero has an unmeasured sign). Everything else is a hard error.
/// `neg` = (neg_lo, neg_hi): bit 0 negates A and bit 1 negates B of even/odd slots respectively.
fn swmmac_exact(
    name: &str,
    ex: &Exact,
    a: &[u32],
    b: &[u32],
    neg: (u8, u8),
    c_raw: u32,
) -> Result<u32> {
    let p = ex.out.mbits + 1;
    let c = exact_decode(name, ex.out, c_raw, "C")?;
    // (sign, odd mantissa, exponent) of every nonzero term; `c` first.
    let mut terms = [(false, 0u64, 0i32); 17];
    let mut n = 0;
    let mut neg_zero = false;
    let mut push = |d: Dyadic| {
        if d.m == 0 {
            neg_zero |= d.neg;
        } else {
            terms[n] = (d.neg, d.m, d.e);
            n += 1;
        }
    };
    push(c);
    for j in 0..a.len() {
        let negate = |bit: u8, fmt: Fmt| -> u32 {
            let word = if j % 2 == 0 { neg.0 } else { neg.1 };
            let on = word & bit != 0;
            u32::from(on) << (fmt.ebits + fmt.mbits)
        };
        let x = exact_decode(name, ex.a, a[j] ^ negate(1, ex.a), "A")?;
        let y = exact_decode(name, ex.b, b[j] ^ negate(2, ex.b), "B")?;
        push(Dyadic {
            neg: x.neg ^ y.neg,
            m: if x.m == 0 || y.m == 0 { 0 } else { x.m * y.m },
            e: x.e + y.e,
        });
    }
    if n == 0 {
        if neg_zero {
            return Err(format!(
                "{name}: exact zero total with a negative-zero input has an unmeasured sign"
            ));
        }
        return Ok(0);
    }
    let q_e = terms[..n].iter().map(|t| t.2).min().unwrap_or(0);
    let (mut total, mut abs_sum) = (0i128, 0i128);
    for &(neg, m, e) in &terms[..n] {
        // |t|/q = m * 2^(e - q_e) has bit length bl + (e - q_e); beyond P+1 bits it already exceeds 2^P.
        let sh = (e - q_e) as u32;
        let bl = 64 - m.leading_zeros();
        if bl + sh > p + 1 {
            return Err(format!(
                "{name}: |C| + sum|a*b| exceeds 2^{p} granules; accumulation rounding is not measured"
            ));
        }
        let v = i128::from(m) << sh;
        abs_sum += v;
        total += if neg { -v } else { v };
    }
    if abs_sum > 1i128 << p {
        return Err(format!(
            "{name}: |C| + sum|a*b| exceeds 2^{p} granules; accumulation rounding is not measured"
        ));
    }
    if total == 0 {
        return Err(format!(
            "{name}: exact zero total by cancellation has an unmeasured sign"
        ));
    }
    exact_encode(name, ex.out, total, q_e)
}

/// F16 A[r,k] / B[k,n] (`idx` = r or n) -> (lane, VGPR, half). gfx11 replicates
/// the fragment in lanes 16..31 (validated in `execute`); the low copy is read.
fn f16_loc(arch: Arch, idx: usize, k: usize) -> (usize, usize, usize) {
    if arch == Arch::Gfx1201 {
        (
            ((k >> 2) & 1) * 16 + idx,
            (k >> 3) * 2 + ((k >> 1) & 1),
            k & 1,
        )
    } else {
        (idx, k >> 1, k & 1)
    }
}

/// IU4 A[r,k] / B[k,n] -> (lane, VGPR, nibble).
fn iu4_loc(arch: Arch, idx: usize, k: usize) -> (usize, usize, usize) {
    if arch == Arch::Gfx1201 {
        (((k >> 3) & 1) * 16 + idx, k >> 4, k & 7)
    } else {
        (idx, k >> 3, k & 7)
    }
}

/// IU8 A[r,k] / B[k,n] -> (lane, VGPR, byte). gfx12: ISA 8-bit layout lane = {k[3], idx[3:0]}, VGPR = k[2],
/// byte = k[1:0]. gfx11 (A/B replicated across the two lane halves): lane = idx, VGPR = k >> 2, byte = k & 3, the
/// same row-per-lane packing as its F16 and IU4 fragments.
fn iu8_loc(arch: Arch, idx: usize, k: usize) -> (usize, usize, usize) {
    if arch == Arch::Gfx1201 {
        (((k >> 3) & 1) * 16 + idx, (k >> 2) & 1, k & 3)
    } else {
        (idx, k >> 2, k & 3)
    }
}

/// C/D[row,col] -> (lane, VGPR).
fn acc_loc(arch: Arch, row: usize, col: usize) -> (usize, usize) {
    if arch == Arch::Gfx1201 {
        ((row >> 3) * 16 + col, row & 7)
    } else {
        ((row & 1) * 16 + col, row >> 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{operand::Modifiers, FormFields};

    fn state() -> State {
        State {
            s: [0; 106],
            t: [0; 16],
            v: Box::new([[0; 64]; 256]),
            lanes: 32,
            nan_loose: false,
            exec: 0xffff_ffff,
            vcc: 0,
            scc: false,
            m0: 0,
            pc: 0,
            ended: false,
            barrier: None,
            signals: 0,
            scratch: Vec::new(),
        }
    }
    fn reg(base: u16, len: u8) -> Operand {
        Operand::Reg(crate::reg::RegRef {
            kind: Kind::V,
            base,
            len,
        })
    }
    fn insn(arch: Arch, name: &str, ops: [Operand; 4], mods: Modifiers) -> Inst {
        let table = if arch == Arch::Gfx1201 {
            crate::isa::gfx12()
        } else {
            crate::isa::gfx1151()
        };
        let row = table.iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(
            arch,
            row.op,
            row.form,
            FormFields::default(),
            ops.into_iter().collect(),
            mods,
            None,
            Default::default(),
        )
        .unwrap()
    }
    fn mods(neg_lo: u8, neg_hi: u8) -> Modifiers {
        Modifiers {
            neg_lo,
            neg_hi,
            op_sel_hi: 7,
            ..Default::default()
        }
    }
    const D: u16 = 0;
    const A: u16 = 16;
    const B: u16 = 32;
    const C: u16 = 48;
    fn put_half(st: &mut State, base: u16, vgpr: usize, lane: usize, half: usize, bits: u16) {
        st.v[usize::from(base) + vgpr][lane] |= u32::from(bits) << (16 * half);
    }
    fn put_nib(st: &mut State, base: u16, vgpr: usize, lane: usize, nib: usize, v: u32) {
        st.v[usize::from(base) + vgpr][lane] |= (v & 15) << (4 * nib);
    }
    /// Every D word other than `hit` must equal C (zero here); returns the hit word.
    fn only(st: &State, hit: (usize, usize)) -> u32 {
        for vgpr in 0..8 {
            for lane in 0..32 {
                if (vgpr, lane) != hit {
                    assert_eq!(
                        st.v[usize::from(D) + vgpr][lane],
                        0,
                        "D vgpr {vgpr} lane {lane}"
                    );
                }
            }
        }
        st.v[usize::from(D) + hit.0][hit.1]
    }

    // gfx11 A[5,13]: lane 5 (+16 replica), VGPR 13/2=6, half 1. B[13,7]: lane 7, VGPR 6, half 1.
    // D[5,7]: VGPR 5/2=2, lane (5&1)*16+7=23.
    #[test]
    fn gfx11_f16_fragments() {
        let mut st = state();
        for lane in [5, 21] {
            put_half(&mut st, A, 6, lane, 1, 0x4200);
        }
        for lane in [7, 23] {
            put_half(&mut st, B, 6, lane, 1, 0x4000);
        }
        st.v[usize::from(C) + 2][23] = 1f32.to_bits();
        let i = insn(
            Arch::Gfx1151,
            "v_wmma_f32_16x16x16_f16",
            [reg(D, 8), reg(A, 8), reg(B, 8), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (2, 23)), 7f32.to_bits());
    }

    // gfx12 A[5,13]: k=0b1101 -> half 1, VGPR (1<<1)|0=2, lane 16+5. B[13,7]: lane 16+7.
    // D[5,7]: lane (5>>3)*16+7=7, VGPR 5. EXEC=0 must not suppress the write.
    #[test]
    fn gfx12_f16_fragments_ignore_exec() {
        let mut st = state();
        st.exec = 0;
        put_half(&mut st, A, 2, 21, 1, 0x4200);
        put_half(&mut st, B, 2, 23, 1, 0x4000);
        st.v[usize::from(C) + 5][7] = 1f32.to_bits();
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_f16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (5, 7)), 7f32.to_bits());
        // Row 9 column 2 lives in lane 16+2, VGPR 1; A[9,2] (k=0b0010: half 0, VGPR 1, lane group 0) -> lane 9; B[2,2] -> lane 2.
        let mut st = state();
        put_half(&mut st, A, 1, 9, 0, 0x3c00);
        put_half(&mut st, B, 1, 2, 0, 0x4400);
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_f16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (1, 18)), 4f32.to_bits());
    }

    // NEG[0]/NEG_HI[0] negate A's low/high half; NEG[1] negates B low; {NEG_HI[2],NEG[2]} = {ABS,NEG} on C.
    #[test]
    fn f16_per_half_neg_and_c_abs_neg() {
        // gfx12 A[0,1] (half 1, VGPR 0, lane 0), B[1,0] (half 1, lane 0): 2 * 3 = 6; C[0,0] = -10.
        let build = |m: Modifiers| {
            let mut st = state();
            put_half(&mut st, A, 0, 0, 1, 0x4000);
            put_half(&mut st, B, 0, 0, 1, 0x4200);
            st.v[usize::from(C)][0] = (-10f32).to_bits();
            let i = insn(
                Arch::Gfx1201,
                "v_wmma_f32_16x16x16_f16",
                [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
                m,
            );
            execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &i, &mut st).unwrap();
            f32::from_bits(st.v[usize::from(D)][0])
        };
        assert_eq!(build(mods(0, 0)), -4.0);
        assert_eq!(build(mods(1, 0)), -4.0); // low-half NEG does not touch the high-half element
        assert_eq!(build(mods(0, 1)), -16.0); // NEG_HI[0]: A[0,1] -> -2
        assert_eq!(build(mods(0, 3)), -4.0); // A and B both negated
        assert_eq!(build(mods(4, 0)), 16.0); // NEG[2]: C -> +10
        assert_eq!(build(mods(0, 4)), 16.0); // ABS(C) = +10
        assert_eq!(build(mods(4, 4)), -4.0); // -ABS(C) = -10
    }

    // gfx11 K16 A[3,10]: lane 3, VGPR 1, nibble 2. B[10,6]: lane 6. D[3,6]: lane 16+6, VGPR 1.
    #[test]
    fn gfx11_iu4_fragments_and_signedness() {
        let run = |neg_lo: u8| {
            let mut st = state();
            for lane in [3, 19] {
                put_nib(&mut st, A, 1, lane, 2, 0xe);
            }
            for lane in [6, 22] {
                put_nib(&mut st, B, 1, lane, 2, 3);
            }
            st.v[usize::from(C) + 1][22] = 100;
            let i = insn(
                Arch::Gfx1151,
                "v_wmma_i32_16x16x16_iu4",
                [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)],
                mods(neg_lo, 0),
            );
            execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", &i, &mut st).unwrap();
            only(&st, (1, 22)) as i32
        };
        assert_eq!(run(3), 94); // 0xe signed = -2; -2*3 + 100
        assert_eq!(run(2), 142); // A unsigned 14, B signed 3
        assert_eq!(run(0), 142);
    }

    #[test]
    fn gfx11_divergent_replicas_are_hard_errors() {
        let f16 = |a_hi: bool, b_hi: bool, exec: u32| {
            let mut st = state();
            st.exec = u64::from(exec);
            for lane in [5, 21] {
                put_half(&mut st, A, 6, lane, 1, 0x4200);
            }
            for lane in [7, 23] {
                put_half(&mut st, B, 6, lane, 1, 0x4000);
            }
            if a_hi {
                st.v[usize::from(A) + 7][31] ^= 1;
            }
            if b_hi {
                st.v[usize::from(B)][16] ^= 0x8000;
            }
            let i = insn(
                Arch::Gfx1151,
                "v_wmma_f32_16x16x16_f16",
                [reg(D, 8), reg(A, 8), reg(B, 8), reg(C, 8)],
                mods(0, 0),
            );
            let r = execute(Arch::Gfx1151, "v_wmma_f32_16x16x16_f16", &i, &mut st);
            (r, st)
        };
        assert!(f16(false, false, u32::MAX).0.is_ok());
        assert!(f16(false, false, 0).0.is_ok());
        for exec in [u32::MAX, 0] {
            for (a, b) in [(true, false), (false, true), (true, true)] {
                let (r, st) = f16(a, b, exec);
                assert!(r.is_err(), "a={a} b={b} exec={exec:#x}");
                assert!(
                    st.v[..8].iter().all(|row| row.iter().all(|&w| w == 0)),
                    "D written despite error"
                );
            }
        }
        let iu4 = |a_hi: bool, b_hi: bool, exec: u32| {
            let mut st = state();
            st.exec = u64::from(exec);
            for lane in [3, 19] {
                put_nib(&mut st, A, 1, lane, 2, 0xe);
            }
            for lane in [6, 22] {
                put_nib(&mut st, B, 1, lane, 2, 3);
            }
            if a_hi {
                put_nib(&mut st, A, 0, 31, 7, 1);
            }
            if b_hi {
                put_nib(&mut st, B, 1, 16, 0, 1);
            }
            let i = insn(
                Arch::Gfx1151,
                "v_wmma_i32_16x16x16_iu4",
                [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)],
                mods(3, 0),
            );
            execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu4", &i, &mut st)
        };
        assert!(iu4(false, false, 0).is_ok());
        for exec in [u32::MAX, 0] {
            for (a, b) in [(true, false), (false, true)] {
                assert!(iu4(a, b, exec).is_err(), "a={a} b={b} exec={exec:#x}");
            }
        }
    }

    // gfx12 K32 A[3,26]: k=0b11010 -> nibble 2, lane group 1 => lane 19, VGPR 1. B[26,6]: lane 22.
    // D[3,6]: lane 6, VGPR 3. A[9,5] -> lane 9, VGPR 0, nibble 5; B[5,15] -> lane 15; D[9,15]: lane 31, VGPR 1.
    #[test]
    fn gfx12_iu4_fragments() {
        let setup = || {
            let mut st = state();
            put_nib(&mut st, A, 1, 19, 2, 0xe);
            put_nib(&mut st, B, 1, 22, 2, 3);
            st
        };
        // Inline-constant C (0) accumulator: -2 * 3 = -6.
        let mut st = setup();
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_i32_16x16x32_iu4",
            [
                reg(D, 8),
                reg(A, 2),
                reg(B, 2),
                Operand::Inline(crate::operand::InlineConst::Integer(0)),
            ],
            mods(3, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (3, 6)) as i32, -6);
        let mut st = setup();
        st.v[usize::from(C) + 3][6] = 100;
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_i32_16x16x32_iu4",
            [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)],
            mods(3, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (3, 6)) as i32, 94);
        let mut st = state();
        put_nib(&mut st, A, 0, 9, 5, 7);
        put_nib(&mut st, B, 0, 15, 5, 7);
        execute(Arch::Gfx1201, "v_wmma_i32_16x16x32_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (1, 31)) as i32, 49);
    }

    fn put_byte(st: &mut State, base: u16, vgpr: usize, lane: usize, byte: usize, v: u32) {
        st.v[usize::from(base) + vgpr][lane] |= (v & 255) << (8 * byte);
    }

    // BF16 uses the F16 fragment mapping. gfx11 A[5,13]: lane 5 (+16), VGPR 6, half 1; B[13,7]: lane 7 (+16);
    // D[5,7]: lane 23, VGPR 2. bf16 2.0 = 0x4000, 3.0 = 0x4040.
    #[test]
    fn bf16_fragments_use_the_half_mapping_on_both_architectures() {
        let mut st = state();
        for lane in [5, 21] {
            put_half(&mut st, A, 6, lane, 1, 0x4000);
        }
        for lane in [7, 23] {
            put_half(&mut st, B, 6, lane, 1, 0x4040);
        }
        st.v[usize::from(C) + 2][23] = 1f32.to_bits();
        let i = insn(
            Arch::Gfx1151,
            "v_wmma_f32_16x16x16_bf16",
            [reg(D, 8), reg(A, 8), reg(B, 8), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1151, "v_wmma_f32_16x16x16_bf16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (2, 23)), 7f32.to_bits());
        // gfx12 A[5,13]: half 1, VGPR 2, lane 21; B[13,7]: lane 23; D[5,7]: lane 7, VGPR 5.
        let mut st = state();
        put_half(&mut st, A, 2, 21, 1, 0x4000);
        put_half(&mut st, B, 2, 23, 1, 0x4040);
        st.v[usize::from(C) + 5][7] = 1f32.to_bits();
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_bf16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_bf16", &i, &mut st).unwrap();
        assert_eq!(only(&st, (5, 7)), 7f32.to_bits());
        // BF16 clamp has no semantics.
        let clamped = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_bf16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            Modifiers {
                clamp: true,
                ..mods(0, 0)
            },
        );
        assert!(execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_bf16", &clamped, &mut st).is_err());
    }

    // Raw gfx1201 BF16 discriminator through the full fragment path: row 0 of A is 16 x 0x0001, column 0 of B is
    // [0x4040, 0x4100, 0, ...], C[0,0] = 0x80000001 -> D[0,0] = 0x000afffc (VGPR 0, lane 0).
    #[test]
    fn gfx12_bf16_subnormal_discriminator_through_fragments() {
        let mut st = state();
        for k in 0..16 {
            let (lane, vgpr, half) = f16_loc(Arch::Gfx1201, 0, k);
            put_half(&mut st, A, vgpr, lane, half, 0x0001);
            if k < 2 {
                let (lane, vgpr, half) = f16_loc(Arch::Gfx1201, 0, k);
                put_half(&mut st, B, vgpr, lane, half, [0x4040, 0x4100][k]);
            }
        }
        st.v[usize::from(C)][0] = 0x8000_0001;
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_bf16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_bf16", &i, &mut st).unwrap();
        assert_eq!(st.v[usize::from(D)][0], 0x000a_fffc);
    }

    // gfx11 IU8 A[3,10]: lane 3 (+16), VGPR 10>>2=2, byte 2. B[10,6]: lane 6 (+16), VGPR 2, byte 2. D[3,6]: lane 22, VGPR 1.
    #[test]
    fn gfx11_iu8_fragments_signedness_and_clamp() {
        let run = |neg_lo: u8, a: u32, b: u32, c: u32, clamp: bool| {
            let mut st = state();
            for lane in [3, 19] {
                put_byte(&mut st, A, 2, lane, 2, a);
            }
            for lane in [6, 22] {
                put_byte(&mut st, B, 2, lane, 2, b);
            }
            st.v[usize::from(C) + 1][22] = c;
            let i = insn(
                Arch::Gfx1151,
                "v_wmma_i32_16x16x16_iu8",
                [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
                Modifiers {
                    clamp,
                    ..mods(neg_lo, 0)
                },
            );
            execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu8", &i, &mut st).unwrap();
            only(&st, (1, 22)) as i32
        };
        assert_eq!(run(3, 0xfe, 3, 100, false), 94); // -2 * 3 + 100
        assert_eq!(run(2, 0xfe, 3, 100, false), 862); // A unsigned 254, B signed 3
        assert_eq!(run(1, 0xfe, 0xff, 0, false), -510); // A signed -2, B unsigned 255
    }

    // gfx12 IU8 A[3,10]: k=0b1010 -> lane group 1 (lane 19), VGPR (10>>2)&1 = 0, byte 2. B[10,6]: lane 22. D[3,6]: lane 6, VGPR 3.
    #[test]
    fn gfx12_iu8_fragments_signedness_wrap_and_clamp() {
        let run = |neg_lo: u8, a: u32, b: u32, c: u32, clamp: bool| {
            let mut st = state();
            put_byte(&mut st, A, 0, 19, 2, a);
            put_byte(&mut st, B, 0, 22, 2, b);
            st.v[usize::from(C) + 3][6] = c;
            let i = insn(
                Arch::Gfx1201,
                "v_wmma_i32_16x16x16_iu8",
                [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)],
                Modifiers {
                    clamp,
                    ..mods(neg_lo, 0)
                },
            );
            execute(Arch::Gfx1201, "v_wmma_i32_16x16x16_iu8", &i, &mut st).unwrap();
            only(&st, (3, 6)) as i32
        };
        assert_eq!(run(3, 0xfe, 3, 100, false), 94);
        assert_eq!(run(0, 0xfe, 0xff, 0, false), 254 * 255);
        assert_eq!(run(1, 0xfe, 0xff, 0, false), -2 * 255);
        assert_eq!(
            run(3, 0x7f, 0x7f, i32::MAX as u32, false),
            (i32::MAX as i64 + 127 * 127) as i32
        ); // wraps
        assert_eq!(run(3, 0x7f, 0x7f, i32::MAX as u32, true), i32::MAX); // CLAMP saturates the final sum
        assert_eq!(run(3, 0x80, 0x7f, i32::MIN as u32, true), i32::MIN);
    }
    // ---- iu8 SWMMAC: silicon-confirmed packed-A / index / dense-B maps (receipt rcpt/gfx1201-C-r1) ----
    const IDX: u16 = 48;
    fn swmmac_run(setup: impl Fn(&mut State)) -> State {
        let mut st = state();
        setup(&mut st);
        let i = insn(
            Arch::Gfx1201,
            "v_swmmac_i32_16x16x32_iu8",
            [reg(D, 8), reg(A, 2), reg(B, 4), reg(IDX, 1)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu8", &i, &mut st).unwrap();
        st
    }

    // Row 3 is lane 3 (half 0): slot 2 (VGPR 0 byte 2) = 7 with selector 1 -> slot j=2 -> K = 4*(2>>1)+1 = 5; slot 3
    // (byte 3) = 5 with selector 3 -> K = 7. The index word is lane 3's: selector of slot i at bits [2i+1:2i].
    // B[5][6]: lane 6, element e=5 -> VGPR 1 byte 1; B[7][6]: VGPR 1 byte 3. D[3][6]: lane 6, VGPR 3, C = 100.
    #[test]
    fn swmmac_iu8_expands_packed_a_through_the_lane_index() {
        let st = swmmac_run(|st| {
            put_byte(st, A, 0, 3, 2, 7);
            put_byte(st, A, 0, 3, 3, 5);
            st.v[usize::from(IDX)][3] = (1 << 4) | (3 << 6);
            put_byte(st, B, 1, 6, 1, 3);
            put_byte(st, B, 1, 6, 3, 4);
            st.v[3][6] = 100;
        });
        assert_eq!(only(&st, (3, 6)) as i32, 100 + 7 * 3 + 5 * 4);
        // The same A values with selectors 0 and 2 meet B[4] and B[6] (both zero): nothing accumulates.
        let st = swmmac_run(|st| {
            put_byte(st, A, 0, 3, 2, 7);
            put_byte(st, A, 0, 3, 3, 5);
            st.v[usize::from(IDX)][3] = (0 << 4) | (2 << 6);
            put_byte(st, B, 1, 6, 1, 3);
            put_byte(st, B, 1, 6, 3, 4);
        });
        assert!(st.v[..8].iter().all(|row| row.iter().all(|&w| w == 0)));
        // A different lane's index word is not consulted: lane 3 values with only lane 4's index set.
        let st = swmmac_run(|st| {
            put_byte(st, A, 0, 3, 2, 7);
            st.v[usize::from(IDX)][4] = 1 << 4;
            put_byte(st, B, 1, 6, 1, 3);
        });
        assert!(st.v[..8].iter().all(|row| row.iter().all(|&w| w == 0)));
    }

    // Row 4, half 1 (lane 20): slots 0 and 1 are compressed slots j=8 and 9 -> the same K-group 4 (K 16..19).
    // Equal selectors (2) land on the same K = 18 and BOTH values accumulate (209). B[18][9]: lane 16+9, e=2 ->
    // VGPR 0 byte 2. D[4][9]: lane 9, VGPR 4.
    #[test]
    fn swmmac_iu8_upper_half_and_equal_selectors_accumulate() {
        let st = swmmac_run(|st| {
            put_byte(st, A, 0, 20, 0, 9);
            put_byte(st, A, 0, 20, 1, 200);
            st.v[usize::from(IDX)][20] = (2 << 0) | (2 << 2);
            put_byte(st, B, 0, 25, 2, 250);
        });
        assert_eq!(only(&st, (4, 9)) as i32, 209 * 250);
    }

    // Raw gfx1201 hardware case (rcpt/gfx1201-C-r1, swmmac iu8 `discover_lane` case 0, variant 0, index_key 0): per-lane
    // index words, unsigned bytes, C = 0. Register-major, lane-minor.
    const HW_A: [u32; 64] = [
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ];
    const HW_B: [u32; 128] = [
        0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201,
        0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201, 0x04030201,
        0x04030201, 0x04030201, 0x14131211, 0x14131211, 0x14131211, 0x14131211, 0x14131211,
        0x14131211, 0x14131211, 0x14131211, 0x14131211, 0x14131211, 0x14131211, 0x14131211,
        0x14131211, 0x14131211, 0x14131211, 0x14131211, 0x08070605, 0x08070605, 0x08070605,
        0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x08070605,
        0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x08070605, 0x18171615,
        0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615,
        0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615, 0x18171615,
        0x18171615, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09,
        0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09,
        0x0c0b0a09, 0x0c0b0a09, 0x0c0b0a09, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19,
        0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19,
        0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x1c1b1a19, 0x100f0e0d, 0x100f0e0d,
        0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d,
        0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d, 0x100f0e0d,
        0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d,
        0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d, 0x201f1e1d,
        0x201f1e1d, 0x201f1e1d,
    ];
    const HW_IDX: [u32; 32] = [
        0xa66c53e4, 0x362a1535, 0xa03b102a, 0x382f74b8, 0x80c14827, 0xeb8ed4a1, 0x4ac47560,
        0x0cfe01a2, 0xd9ee8626, 0x6ab11a5e, 0x942c5251, 0x8c82c11b, 0x94264233, 0xda64e11f,
        0xdbe61c1e, 0xf97ad27a, 0x6103b7c7, 0x786419e7, 0xbf314d30, 0x16e58c21, 0xec130144,
        0x7015bb44, 0xe5f36b28, 0xbc0a3860, 0x82ee996a, 0xc104efda, 0xc6eb414b, 0xe3ee13b6,
        0x01471b6d, 0x793a0cd2, 0x924e2431, 0xf8a0ed58,
    ];
    const HW_D: [u32; 256] = [
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002, 0x00000002,
        0x00000002, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004, 0x00000004,
        0x00000004, 0x00000004, 0x00000004, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001,
        0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003, 0x00000003,
        0x00000003, 0x00000003, 0x00000003, 0x00000003,
    ];

    // The whole wave of a recorded hardware case (per-lane random index words, one-hot compressed A) reproduces.
    #[test]
    fn swmmac_iu8_reproduces_a_recorded_hardware_wave() {
        let st = swmmac_run(|st| {
            for lane in 0..32 {
                for vgpr in 0..2 {
                    st.v[usize::from(A) + vgpr][lane] = HW_A[32 * vgpr + lane];
                }
                for vgpr in 0..4 {
                    st.v[usize::from(B) + vgpr][lane] = HW_B[32 * vgpr + lane];
                }
                st.v[usize::from(IDX)][lane] = HW_IDX[lane];
            }
        });
        for vgpr in 0..8 {
            for lane in 0..32 {
                assert_eq!(
                    st.v[usize::from(D) + vgpr][lane],
                    HW_D[32 * vgpr + lane],
                    "D vgpr {vgpr} lane {lane}"
                );
            }
        }
    }

    /// One SWMMAC wave from raw register images (register-major, lane-minor): returns the state after execution.
    fn swmmac_wave(
        name: &str,
        m: Modifiers,
        regs: (u8, u8),
        a: &[u32],
        b: &[u32],
        idx: &[u32],
        c: &[u32],
    ) -> State {
        let mut st = state();
        for lane in 0..32 {
            for vgpr in 0..usize::from(regs.0) {
                st.v[usize::from(A) + vgpr][lane] = a[32 * vgpr + lane];
            }
            for vgpr in 0..usize::from(regs.1) {
                st.v[usize::from(B) + vgpr][lane] = b[32 * vgpr + lane];
            }
            for vgpr in 0..8 {
                st.v[usize::from(D) + vgpr][lane] = c[32 * vgpr + lane];
            }
            st.v[usize::from(IDX)][lane] = idx[lane];
        }
        let i = insn(
            Arch::Gfx1201,
            name,
            [reg(D, 8), reg(A, regs.0), reg(B, regs.1), reg(IDX, 1)],
            m,
        );
        execute(Arch::Gfx1201, name, &i, &mut st).unwrap();
        st
    }

    // Recorded gfx1201 waves (rcpt/gfx1201-C-r2): modifier variants that change the result, including the integer
    // two-half CLAMP (a single final saturation differs from hardware in 12 / 9 / 5 words of these waves).
    #[test]
    fn swmmac_modifier_variants_reproduce_recorded_hardware_waves() {
        use crate::emu::swmmac_fixtures as f;
        let key1 = Modifiers {
            op_sel: 1,
            ..mods(0, 0)
        };
        let neg_all = Modifiers {
            neg_lo: 3,
            neg_hi: 3,
            op_sel_hi: 7,
            ..Default::default()
        };
        let signed_clamp = Modifiers {
            clamp: true,
            ..mods(3, 0)
        };
        #[allow(clippy::type_complexity)]
        let cases: [(
            &str,
            Modifiers,
            (u8, u8),
            &[u32],
            &[u32],
            &[u32],
            &[u32],
            &[u32],
        ); 6] = [
            (
                "v_swmmac_f32_16x16x32_f16",
                neg_all,
                (4, 8),
                &f::F16_NEG_A,
                &f::F16_NEG_B,
                &f::F16_NEG_IDX,
                &f::F16_NEG_C,
                &f::F16_NEG_D,
            ),
            (
                "v_swmmac_f32_16x16x32_bf16",
                key1.clone(),
                (4, 8),
                &f::BF16_KEY1_A,
                &f::BF16_KEY1_B,
                &f::BF16_KEY1_IDX,
                &f::BF16_KEY1_C,
                &f::BF16_KEY1_D,
            ),
            (
                "v_swmmac_i32_16x16x32_iu8",
                signed_clamp.clone(),
                (2, 4),
                &f::IU8_CLAMP_A,
                &f::IU8_CLAMP_B,
                &f::IU8_CLAMP_IDX,
                &f::IU8_CLAMP_C,
                &f::IU8_CLAMP_D,
            ),
            (
                "v_swmmac_i32_16x16x32_iu4",
                key1,
                (1, 2),
                &f::IU4_KEY1_A,
                &f::IU4_KEY1_B,
                &f::IU4_KEY1_IDX,
                &f::IU4_KEY1_C,
                &f::IU4_KEY1_D,
            ),
            (
                "v_swmmac_i32_16x16x32_iu4",
                signed_clamp.clone(),
                (1, 2),
                &f::IU4_CLAMP_A,
                &f::IU4_CLAMP_B,
                &f::IU4_CLAMP_IDX,
                &f::IU4_CLAMP_C,
                &f::IU4_CLAMP_D,
            ),
            (
                "v_swmmac_i32_16x16x64_iu4",
                signed_clamp,
                (2, 4),
                &f::IU4X64_CLAMP_A,
                &f::IU4X64_CLAMP_B,
                &f::IU4X64_CLAMP_IDX,
                &f::IU4X64_CLAMP_C,
                &f::IU4X64_CLAMP_D,
            ),
        ];
        for (name, m, regs, a, b, idx, c, d) in cases {
            let st = swmmac_wave(name, m, regs, a, b, idx, c);
            for vgpr in 0..8 {
                for lane in 0..32 {
                    assert_eq!(
                        st.v[usize::from(D) + vgpr][lane],
                        d[32 * vgpr + lane],
                        "{name} D vgpr {vgpr} lane {lane}"
                    );
                }
            }
        }
    }

    // IU4 x32, hand-placed. Row 5 is lane 5 (half 0) / lane 21 (half 1). Half 1, element 3 (nibble 3 of VGPR 0) is
    // slot 8+3 = 11 -> K-group 5; selector 2 (idx bits [7:6] of lane 21) -> K = 22. B[22][9]: hb = 1, e = 6 -> lane
    // 16+9, VGPR 0, nibble 6. D[5][9]: lane 9, VGPR 5.
    #[test]
    fn swmmac_iu4_x32_layout_and_index_key() {
        let run = |m: Modifiers, a_nibble: u32, b_nibble: u32, idx: u32| {
            let mut st = state();
            put_nib(&mut st, A, 0, 21, 3, a_nibble);
            st.v[usize::from(IDX)][21] = idx;
            put_nib(&mut st, B, 0, 25, 6, b_nibble);
            let i = insn(
                Arch::Gfx1201,
                "v_swmmac_i32_16x16x32_iu4",
                [reg(D, 8), reg(A, 1), reg(B, 2), reg(IDX, 1)],
                m,
            );
            execute(Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu4", &i, &mut st).unwrap();
            st
        };
        assert_eq!(only(&run(mods(0, 0), 3, 5, 2 << 6), (5, 9)) as i32, 15);
        // index_key:1 reads the HIGH 16 bits: a selector there is ignored by key 0 and found by key 1.
        let high = 2u32 << 22;
        assert!(run(mods(0, 0), 3, 5, high).v[..8]
            .iter()
            .all(|r| r.iter().all(|&w| w == 0)));
        assert_eq!(
            only(
                &run(
                    Modifiers {
                        op_sel: 1,
                        ..mods(0, 0)
                    },
                    3,
                    5,
                    high
                ),
                (5, 9)
            ) as i32,
            15
        );
        // Signed nibbles: A = 0xd (-3), B = 0xb (-5); NEG_LO[0] alone makes only A signed (B stays 11).
        assert_eq!(only(&run(mods(3, 0), 0xd, 0xb, 2 << 6), (5, 9)) as i32, 15);
        assert_eq!(
            only(&run(mods(1, 0), 0xd, 0xb, 2 << 6), (5, 9)) as i32,
            -3 * 11
        );
        assert_eq!(
            only(&run(mods(2, 0), 0xd, 0xb, 2 << 6), (5, 9)) as i32,
            13 * -5
        );
    }

    // IU4 x64: 16 elements per lane, no index_key, the whole 32-bit index word. Row 2 half 1 = lane 18; element 11
    // (VGPR 1, nibble 3) is slot 16+11 = 27 -> K-group 13; selector at idx bits [23:22] = 1 -> K = 53. B[53][7]:
    // hb = 1, e = 21 -> lane 16+7, VGPR 2, nibble 5. D[2][7]: lane 7, VGPR 2.
    #[test]
    fn swmmac_iu4_x64_layout_uses_the_whole_index_word() {
        let mut st = state();
        put_nib(&mut st, A, 1, 18, 3, 6);
        st.v[usize::from(IDX)][18] = 1 << 22;
        put_nib(&mut st, B, 2, 23, 5, 7);
        let i = insn(
            Arch::Gfx1201,
            "v_swmmac_i32_16x16x64_iu4",
            [reg(D, 8), reg(A, 2), reg(B, 4), reg(IDX, 1)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, "v_swmmac_i32_16x16x64_iu4", &i, &mut st).unwrap();
        assert_eq!(only(&st, (2, 7)) as i32, 42);
        // The x64 form has no index_key.
        let keyed = insn(
            Arch::Gfx1201,
            "v_swmmac_i32_16x16x64_iu4",
            [reg(D, 8), reg(A, 2), reg(B, 4), reg(IDX, 1)],
            Modifiers {
                op_sel: 1,
                ..mods(0, 0)
            },
        );
        assert!(execute(Arch::Gfx1201, "v_swmmac_i32_16x16x64_iu4", &keyed, &mut st).is_err());
    }

    // Float NEG acts by compressed-SLOT parity, not by K parity. Row 0 lane 0: slot 0 (A = 2.0, selector 1 -> K = 1)
    // and slot 1 (A = 4.0, selector 3 -> K = 3); B[1][0] = 3.0 (VGPR 0 high half), B[3][0] = 5.0 (VGPR 1 high half).
    // Both K are odd, so a K-parity rule would not separate them; slot parity does.
    #[test]
    fn swmmac_float_neg_follows_slot_parity() {
        let run = |neg_lo: u8, neg_hi: u8| {
            let mut st = state();
            st.v[usize::from(A)][0] = 0x4000 | (0x4400 << 16); // f16 2.0 (slot 0), 4.0 (slot 1)
            st.v[usize::from(IDX)][0] = 1 | (3 << 2);
            st.v[usize::from(B)][0] = 0x4200 << 16; // e=1 = 3.0
            st.v[usize::from(B) + 1][0] = 0x4500 << 16; // e=3 = 5.0
            let m = Modifiers {
                neg_lo,
                neg_hi,
                op_sel_hi: 7,
                ..Default::default()
            };
            let i = insn(
                Arch::Gfx1201,
                "v_swmmac_f32_16x16x32_f16",
                [reg(D, 8), reg(A, 4), reg(B, 8), reg(IDX, 1)],
                m,
            );
            execute(Arch::Gfx1201, "v_swmmac_f32_16x16x32_f16", &i, &mut st).unwrap();
            f32::from_bits(st.v[0][0])
        };
        assert_eq!(run(0, 0), 2.0 * 3.0 + 4.0 * 5.0);
        assert_eq!(run(1, 0), -2.0 * 3.0 + 4.0 * 5.0); // A of even slots
        assert_eq!(run(0, 1), 2.0 * 3.0 - 4.0 * 5.0); // A of odd slots
        assert_eq!(run(2, 0), -2.0 * 3.0 + 4.0 * 5.0); // selected B of even slots
        assert_eq!(run(0, 2), 2.0 * 3.0 - 4.0 * 5.0); // selected B of odd slots
        assert_eq!(run(1, 2), -2.0 * 3.0 - 4.0 * 5.0);
        assert_eq!(run(3, 0), 2.0 * 3.0 + 4.0 * 5.0); // A and B both negated on the even slot
    }

    #[test]
    fn swmmac_refuses_unmeasured_modes_and_every_other_variant() {
        let mut st = state();
        let mk = |name: &str, regs: (u8, u8), m: Modifiers| {
            insn(
                Arch::Gfx1201,
                name,
                [reg(D, 8), reg(A, regs.0), reg(B, regs.1), reg(IDX, 1)],
                m,
            )
        };
        let iu8 = |m: Modifiers| mk("v_swmmac_i32_16x16x32_iu8", (2, 4), m);
        let f16 = |m: Modifiers| mk("v_swmmac_f32_16x16x32_f16", (4, 8), m);
        // Measured modes run.
        for m in [
            mods(0, 0),
            Modifiers {
                op_sel: 1,
                ..mods(0, 0)
            },
            Modifiers {
                clamp: true,
                ..mods(3, 0)
            },
        ] {
            assert!(execute(Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu8", &iu8(m), &mut st).is_ok());
        }
        assert!(execute(
            Arch::Gfx1201,
            "v_swmmac_f32_16x16x32_f16",
            &f16(Modifiers {
                neg_lo: 3,
                neg_hi: 3,
                op_sel: 1,
                op_sel_hi: 7,
                ..Default::default()
            }),
            &mut st
        )
        .is_ok());
        // Unmeasured: other OPSEL bits, accumulator NEG[2]/NEG_HI[2], NEG_HI on integers, CLAMP on floats.
        for m in [
            Modifiers {
                op_sel: 2,
                ..mods(0, 0)
            },
            Modifiers {
                op_sel: 4,
                ..mods(0, 0)
            },
            mods(4, 0),
            mods(0, 4),
            mods(0, 1),
        ] {
            assert!(execute(Arch::Gfx1201, "v_swmmac_i32_16x16x32_iu8", &iu8(m), &mut st).is_err());
        }
        for m in [
            Modifiers {
                neg_lo: 4,
                op_sel_hi: 7,
                ..Default::default()
            },
            Modifiers {
                neg_hi: 4,
                op_sel_hi: 7,
                ..Default::default()
            },
            Modifiers {
                clamp: true,
                ..mods(0, 0)
            },
            Modifiers {
                op_sel: 2,
                ..mods(0, 0)
            },
        ] {
            assert!(execute(Arch::Gfx1201, "v_swmmac_f32_16x16x32_f16", &f16(m), &mut st).is_err());
        }
        // gfx11 has no SWMMAC.
        assert!(execute(
            Arch::Gfx1151,
            "v_swmmac_i32_16x16x32_iu8",
            &iu8(mods(0, 0)),
            &mut st
        )
        .is_err());
        // The six exact-domain kinds (swmmac_exact_* tests) check their own register shapes: the iu8 operand shape
        // (D 8, A 2, B 4) fits the FP8/BF8 kinds (all-zero inputs are in the exact domain) and not the 16-bit-out ones.
        for name in ["v_swmmac_f16_16x16x32_f16", "v_swmmac_bf16_16x16x32_bf16"] {
            let i = iu8(mods(0, 0));
            assert!(execute(Arch::Gfx1201, name, &i, &mut st).is_err(), "{name}");
        }
        for name in [
            "v_swmmac_f32_16x16x32_fp8_fp8",
            "v_swmmac_f32_16x16x32_fp8_bf8",
            "v_swmmac_f32_16x16x32_bf8_fp8",
            "v_swmmac_f32_16x16x32_bf8_bf8",
        ] {
            let i = iu8(mods(0, 0));
            assert!(execute(Arch::Gfx1201, name, &i, &mut st).is_ok(), "{name}");
        }
        // Kinds without any model stay refused, and the exact-domain kinds exist on gfx1201 only.
        assert!(execute(
            Arch::Gfx1201,
            "v_swmmac_f32_16x16x32_fp8_fp9",
            &iu8(mods(0, 0)),
            &mut st
        )
        .is_err());
        assert!(execute(
            Arch::Gfx1151,
            "v_swmmac_f32_16x16x32_fp8_fp8",
            &iu8(mods(0, 0)),
            &mut st
        )
        .is_err());
    }
    // Raw gfx1201 hardware wave (rcpt/gfx1201-C-r1 swmmac f32_16x16x32_f16 `numerics` case 7, random 2:4 positions, index_key 0),
    // register-major, lane-minor.
    const SW_F16_A: [u32; 128] = [
        0x13ff2400, 0xa866b800, 0xbc7b3c80, 0xa420d410, 0xc8611000, 0x4d139c2d, 0x0c00bc56,
        0x1fff4000, 0xd4954000, 0xbd00a7ff, 0x22a34887, 0xb80079ff, 0xd401afff, 0x2b0f8000,
        0x3d399200, 0x55f40000, 0x00000000, 0x1ecf14e0, 0x80003a7b, 0xb5d223d6, 0x0000c206,
        0x3bff4080, 0x10040c00, 0xc6603af4, 0x4880b060, 0x1fc0b400, 0xcfff1ddd, 0x0000c010,
        0xb8c03e2d, 0x8000a5e0, 0x38104937, 0x299d2c25, 0x14043480, 0xc8803708, 0xc4042800,
        0xbaa8d7ff, 0x1ae09000, 0x3bffd020, 0x0c003620, 0x4bff5413, 0x00003934, 0xbffffbff,
        0x2eb71f15, 0xa1b8a7ff, 0x4806488a, 0xae8b2125, 0x3cd243ff, 0xa8008400, 0x0000b000,
        0x03ff8000, 0x80000f9e, 0x78004ba7, 0xbf0c9400, 0x4fb98000, 0x24c80000, 0xb766fbff,
        0xaa7a3907, 0x1fff9804, 0x91411bff, 0x000120dc, 0xc7ffc320, 0x4c000000, 0xbba9bc00,
        0xd059a7ff, 0xd2a80000, 0x00007801, 0xafff472a, 0x28409c04, 0x21bdd192, 0xadec98c0,
        0xc9d83000, 0x37ff29e2, 0x1400bc00, 0xadbad0c2, 0xced8c0fb, 0x9bff3000, 0x8f00c4d3,
        0x7a010d37, 0x00000000, 0x1801c220, 0x24201c40, 0xcc609508, 0xc3ff7800, 0x0000c000,
        0x33ff3a20, 0xb3ff0002, 0x4fe4577b, 0xafc20c01, 0x4400afd3, 0xafffa7ff, 0xc3ffb000,
        0x37812c00, 0x48003402, 0x3a009002, 0x4fa0356c, 0x2fff0000, 0x9d00acb9, 0x02000000,
        0x99ea2925, 0xd300ce62, 0xaedd8000, 0x53ffd25d, 0x303a5404, 0xa97713fd, 0xaa80a132,
        0x4000ce31, 0x8d004474, 0x56824d9d, 0xad389e62, 0x00003400, 0xb62954a0, 0x18000ea2,
        0x3c91a404, 0x2000d440, 0xa26e4357, 0x78008000, 0x8fff0000, 0xc60ab200, 0x00005002,
        0x4333c853, 0x00005240, 0x97291beb, 0xb4004000, 0x8000c404, 0x00020100, 0x80004fff,
        0x3c001402, 0xc8109f80,
    ];
    const SW_F16_B: [u32; 256] = [
        0x13ff38c0, 0x93ff97ff, 0xd069a400, 0x4bff1776, 0x20401000, 0x97c0924d, 0x1800cb7e,
        0xbc8010c0, 0x56409217, 0x00012c40, 0xbabc9bc7, 0x1d1423ff, 0x5401a3a8, 0x8dc04000,
        0x0ece9031, 0x80008d00, 0x54e99400, 0xd0144bff, 0x92602200, 0x0e09c0d5, 0x2fff1fd4,
        0xd08012c9, 0x00004801, 0x34cfa8a0, 0xcc40b683, 0x7a019e2e, 0xbfff20c0, 0xbbff2bc6,
        0x10088000, 0x8000a9a1, 0xa8002640, 0x393aaa80, 0x14003ce0, 0x0000a5bb, 0x80001bff,
        0x78004c85, 0x14208c08, 0x40007800, 0x8000d7e2, 0x97999cb3, 0x800095f2, 0x9c00c600,
        0x00012400, 0x461d9bff, 0xd096291f, 0xd3c20000, 0x48007800, 0x2aa07a00, 0xbc00aaeb,
        0xa2f8ab30, 0x51aa9a00, 0x3c00b401, 0x2cc02c00, 0x50005640, 0xbbffce09, 0xd000ca00,
        0xd2a62c68, 0xd0d9a000, 0x990ccf88, 0xbc00d312, 0x0401cc01, 0x5008b000, 0x28023000,
        0x8c0037ff, 0xb7aeb000, 0xd24138ea, 0xd4d326e0, 0x0fffc400, 0xb3ffc000, 0x8000b1e8,
        0xc6764320, 0x540003ff, 0xd331a46c, 0x7a0057ff, 0x2c001043, 0x0000b27e, 0x8da9d156,
        0x1000bc00, 0x47ffbb20, 0x8c80b440, 0x28ce8000, 0xc85e9c00, 0x0e0d2c04, 0xa000b7ff,
        0xa09aa19a, 0x0fff9689, 0xc0001bff, 0x447b1400, 0xa5401bae, 0x47ff8000, 0xa71c0000,
        0x0000cbff, 0xa0004000, 0x9fff4f52, 0xd5cb5492, 0x2c9d9bc1, 0xab681ae0, 0x140018d8,
        0x0000c262, 0x54c84800, 0xbfda0001, 0x1400bef3, 0x6400cc00, 0xa32ab53f, 0x1b72a5dc,
        0xc0005447, 0x0000c800, 0xd00018d0, 0x43ff52c0, 0xd4b433ff, 0xbfa08000, 0x8d1345e2,
        0xcd678001, 0x40008c02, 0x3c002801, 0xa73d20a2, 0xbc042c04, 0xd2d24800, 0x4000b100,
        0xb5eba500, 0x0000bc00, 0x0000bc00, 0x9400a7c1, 0x5321a3c0, 0x8000c93c, 0x4bff2c00,
        0x10008000, 0x9df6a049, 0xbb6f8000, 0xc1b92c73, 0x3800d000, 0x9cc40000, 0x3800bb68,
        0xa7e74000, 0x1c009500, 0xd00025a9, 0xd45ec16c, 0x9f463b6f, 0xb2c033c4, 0x41a138ca,
        0x31d26400, 0xb4009404, 0x9f809295, 0x54100e60, 0x9400a4f5, 0x00009fff, 0x4c1028e0,
        0xa7543a98, 0x2c204800, 0x1d401490, 0x7a00c400, 0xb7ffb580, 0x0d2099f5, 0x8c60190a,
        0xcbff4800, 0x52a15040, 0xcde3c0e0, 0x2d3ca2dc, 0x4200c800, 0xb6783135, 0x913eb7db,
        0x9a38b7c1, 0x3cb72700, 0x14007bff, 0x10df8000, 0xd4083c00, 0x8000a680, 0x2268b7ff,
        0x54008fff, 0x93bf2c00, 0xd02035a0, 0x277503ff, 0xbf4a5408, 0x0002c7ff, 0x42c92bff,
        0xafff1400, 0x4802b039, 0x3160c734, 0x4ca6a1ab, 0x9c4533ff, 0xd000b420, 0x95259fff,
        0xcec6bc00, 0x37a00000, 0x8000ac00, 0x24048000, 0xbc0043ff, 0xa3ff4960, 0x43f62ad3,
        0x8000cb6f, 0xd7ff1400, 0x5704bc00, 0xc63aa4ea, 0x52fd4000, 0x8000d1c8, 0x9f76b4bb,
        0x47288c31, 0x01009601, 0x35d7cf5e, 0x0fff4410, 0x4f88413c, 0x9c208000, 0xac7c3343,
        0x80002404, 0x3a763ec3, 0x3c00ce87, 0x25524ea9, 0x33641343, 0x79ff4a20, 0x9c870400,
        0x4ff1338f, 0xabff9940, 0xbc00bfa0, 0xc57a97f7, 0x4b2c4000, 0x9dcbd5cc, 0xd4004000,
        0x9d00c843, 0xaf00b6b0, 0x2180c500, 0x0df996df, 0x4800bcce, 0x1800af68, 0x9917b4e2,
        0x3bff8000, 0xa8d4b6c0, 0xc60f9620, 0x8000a36a, 0x44002480, 0x1bffc11b, 0x9c022182,
        0xc1d94800, 0x14009973, 0xf8004000, 0xbdcc1b46, 0x1bffa410, 0xb8008000, 0xb79c1a62,
        0x3210a702, 0xd3514b47, 0xc8671080, 0x8c29d480, 0x78002200, 0x14010c25, 0x5687354d,
        0xac003400, 0xa008d700, 0x4c40545c, 0xb1e097ff, 0x3bffd660, 0x4c42b300, 0x1bb1c4cd,
        0x4fffc020, 0xbc00ae7e, 0x147093b7, 0x30793771,
    ];
    const SW_F16_IDX: [u32; 32] = [
        0x00008cea, 0x0000c9dd, 0x0000d5ce, 0x0000c9e8, 0x0000dee6, 0x00004d95, 0x0000dc6a,
        0x00008d4c, 0x0000edce, 0x0000cced, 0x0000dce9, 0x0000dc96, 0x0000cee9, 0x0000dcdc,
        0x0000ed5d, 0x0000e8ac, 0x0000ed5e, 0x0000cece, 0x0000ccdd, 0x00004cdc, 0x0000ceee,
        0x00006d4a, 0x000098e8, 0x00008ce8, 0x0000aced, 0x00009dee, 0x000068ee, 0x0000ddae,
        0x00009e98, 0x0000dec8, 0x0000adde, 0x00005cdc,
    ];
    const SW_F16_C: [u32; 256] = [
        0xbd65e000, 0x39400000, 0x3c5bf487, 0x3d800002, 0xbd000000, 0x41000000, 0x80000001,
        0xc07fffff, 0xc0cbd2c1, 0x3b9d3800, 0x4b800001, 0x4030c816, 0xb88a84ff, 0xbb69a3f2,
        0xbcc93c6c, 0xbd269000, 0xbde0c60e, 0xc17fffff, 0xb994cb67, 0x3e50a3c0, 0x3b000040,
        0x39800010, 0x3b800000, 0x39646cc5, 0x3cdd72d9, 0x3dcd9f0b, 0x411fd000, 0xba3e6a65,
        0xbbefd62e, 0x3ac2c800, 0xbce363b9, 0x3fc0abcd, 0x3b944000, 0xbfbacb04, 0xff7fffff,
        0x3a800080, 0x40a76aee, 0xba7fffff, 0x3d000000, 0x3d000000, 0xbdca9000, 0x3dfd7dcf,
        0x3f7fffff, 0x387fffff, 0xbf000000, 0x3e7fffff, 0x3c800000, 0xbd7fffff, 0x39800002,
        0x407fffff, 0x3aa21686, 0x3f800080, 0x3e000008, 0xbd71f327, 0xb98ea331, 0x3d400800,
        0xbfffffff, 0x3b7bc000, 0xb9913800, 0x3d64c2e3, 0x3e352800, 0xbeffffff, 0xbc3246e4,
        0x3d90cb65, 0xbf000100, 0x38b19000, 0xbaffc151, 0x40000000, 0x3fe4c72d, 0x3bf97449,
        0x41632d27, 0x7f7fffff, 0xc1b20668, 0xba3b3800, 0x3a100000, 0xbb000100, 0x3f7fffff,
        0x3c000000, 0xbb9dd309, 0x3f808000, 0x4b000000, 0xc16928db, 0x3b200000, 0xbaa20222,
        0x3f800001, 0x3f800008, 0xb8ffffff, 0xbb000000, 0x3b56254a, 0xba615bbe, 0xbb7fffff,
        0x3b900000, 0x41800000, 0xba22fcf8, 0x4100929a, 0x3effffff, 0xba800001, 0xbc38d12e,
        0xbd000000, 0xbd801000, 0xb9fca000, 0xbe040000, 0x38369df9, 0xbd156d99, 0xbfbce97b,
        0x3b7744f9, 0x3a7da138, 0xba3e073d, 0xbf3fdd51, 0x3ab8c196, 0x4072cdc6, 0xff7fffff,
        0xbb000800, 0x3d549523, 0x4051ee32, 0x3d954ffa, 0x3d008000, 0xb823ab25, 0x397fffff,
        0xb94e8d5d, 0xbe544f78, 0x3dc26704, 0x39000000, 0xbc97f000, 0x3d9c9a09, 0xb97fffff,
        0x3dffffff, 0x3c880000, 0xc0040000, 0x3d1b06a6, 0xba099800, 0x38000000, 0x397fffff,
        0xbe2236f6, 0xba4d27ee, 0xbccd6478, 0x3d918c66, 0xbe27beca, 0x80800000, 0x3d7a7473,
        0x3ae9a000, 0x40400000, 0xbc2a0962, 0xbc999373, 0x34000000, 0x3b5a44fb, 0x3e6b5be5,
        0x3aff4800, 0x3d469000, 0xbbdc5d5f, 0xbca92369, 0xbd549d94, 0x39e00276, 0xbea860c3,
        0x3e800002, 0xbb069930, 0x41026000, 0x3db3859f, 0xbc7fffff, 0xbf7fffff, 0x3d096919,
        0x3f800000, 0xbe7c59f9, 0x3d1e26e4, 0xbf000000, 0x39000000, 0x3b800000, 0xbd67afcd,
        0xbc800100, 0xbe000040, 0x38ffffff, 0x33800000, 0x380a17a3, 0x3a7fffff, 0xb91344af,
        0x3dffffff, 0xbc139800, 0x3fffffff, 0xbf1cf153, 0xbf40e269, 0x3b8e8711, 0xb973c2e9,
        0x3f810000, 0x3c9eca5c, 0xc0894000, 0x396815d8, 0xbd800002, 0xb9ffffff, 0xbfbf1f37,
        0xbb476942, 0x3f800000, 0x37ea720b, 0xbf000000, 0xbd000000, 0xbcffffff, 0x3c800000,
        0x3b62e702, 0xba7fffff, 0x3dffffff, 0x39000001, 0xbbb051b3, 0x3f254057, 0x3b7fffff,
        0x37801000, 0xbc000000, 0xc2013799, 0xb8f125c8, 0xbae726d8, 0x3a800000, 0x3e41c2e6,
        0x3bd05800, 0x3c95b8d8, 0xbb800000, 0x3d052000, 0x3ac60d04, 0x3ca00000, 0x3affffff,
        0x3e81e3a8, 0x00800001, 0x3a800001, 0xbde27800, 0x40800004, 0x3d802000, 0xbc46da27,
        0x3eec6715, 0xb825b000, 0xbd7dd000, 0xbb040000, 0x007fffff, 0xbde8d440, 0x387fffff,
        0x00800001, 0x3e800000, 0x3d820000, 0x402f5395, 0xc0205bb6, 0xbced86f8, 0x3d000000,
        0x3d17e4c8, 0xbf800000, 0x3d004000, 0xbc800000, 0xba800080, 0x3a000080, 0x3f8e5e19,
        0x396f7f1e, 0xbfac9d13, 0x3d2bfe4b, 0xbf594800, 0x00000001, 0x3a75c5f6, 0xba5e3e4f,
        0x3d800000, 0xbc3f6459, 0x3f8d8000, 0x40e8bcf5,
    ];
    const SW_F16_D: [u32; 256] = [
        0xc11ed2f4, 0x400f605b, 0x4711e420, 0x402a33ab, 0x42ed4889, 0x457920ed, 0x434a2556,
        0x414a947b, 0xc5556c6a, 0x41a77810, 0x4b800378, 0x3fbd4548, 0x43149db7, 0x3f47e74e,
        0x43b97011, 0x444e87d7, 0x4183fc04, 0xc5617de5, 0x43f1492f, 0xca129f56, 0xc2e37f32,
        0x47805339, 0xc5ba781c, 0x454a19ba, 0xc3cb91be, 0xc623198f, 0xc0ad28fe, 0xc430c9b4,
        0x4520851e, 0x45970fef, 0x477bb65a, 0x47c0cbbf, 0xc6f5b58b, 0xc7b75afb, 0xff7fffff,
        0xc4f6dc7f, 0x4685b274, 0xc47d0553, 0xc608c24f, 0xc97eb48c, 0xca0be640, 0x46ab966e,
        0xc5d6ba4e, 0x47b58705, 0x45b62a07, 0xc5e873b3, 0x44fd5e76, 0x4a020222, 0xc1e47b32,
        0xc34b0b58, 0x484e71f2, 0xc909f911, 0x424d05fa, 0x47de1327, 0x497f2e54, 0x46a6f9d1,
        0x44bec3b4, 0xca8acf24, 0x48ffdf2b, 0xc2d78e7e, 0xca5a460a, 0xc672197b, 0xc4228069,
        0xc8bc38e0, 0xc40ce23d, 0xc39670f7, 0x443fa457, 0xc61bf40d, 0x488010a4, 0x471004e0,
        0xc80406f9, 0x7f7fffff, 0xc419aa31, 0x471c370d, 0x487fffc9, 0x498803d2, 0xc79bfb22,
        0xc292e374, 0xc85b809a, 0x476cdd49, 0x4affa019, 0x42ef0a61, 0xc4d82756, 0x42eff9ef,
        0x420661d8, 0x4491aef2, 0xc2a7d7b8, 0x44900c73, 0x447cf241, 0x448288a2, 0x445b5287,
        0xc1f6720c, 0xc4edfa7c, 0xc4746d70, 0x43a9e70f, 0x4421c339, 0x4ebfca63, 0x477eb890,
        0x49834bc6, 0xc4eb19c6, 0xc780b748, 0xc9f031ed, 0x49037558, 0xc63d69de, 0xca0000b8,
        0x49deed92, 0xc51e54d3, 0x49e429a7, 0xc5d7e060, 0x4940145c, 0xc51e7ed4, 0xff7fffff,
        0xc8063197, 0x4319a49a, 0xc4066183, 0x4958cb61, 0x440a798a, 0x4ebfdffe, 0xcabcf894,
        0xc3fd582c, 0x444b439f, 0xca625500, 0x4428f6cd, 0xc2925452, 0x4500b90e, 0xc26bbe1a,
        0x4ebfdffe, 0x4f0fe7e8, 0x426f5b9f, 0x414d1a9a, 0x4393a98d, 0xca322be3, 0xc0e58222,
        0xc36a7d2e, 0x42844974, 0x428aa11e, 0xc45ace6d, 0x455b7092, 0x42aa0e44, 0x429c05d3,
        0xc553a0ad, 0x439ff94f, 0xc1990e04, 0x425ffe7c, 0xc137c684, 0x426cb595, 0x42f929da,
        0xc89a8825, 0xc3884f11, 0xca002241, 0x4679e3f5, 0xc0a0333f, 0xc0a79669, 0x4563890e,
        0xc27e5e8d, 0xc35cdd88, 0x43a6f8dd, 0xc40e2567, 0xca002b16, 0xca402116, 0xc89107a7,
        0x45982fa0, 0x457919e6, 0x43ad39fc, 0x43efccf0, 0x43b5218b, 0x44be35c6, 0xc50cfab4,
        0x45beaf72, 0xc9c5f58f, 0xc309eb50, 0xc337aa84, 0x449e5bfe, 0x44a80c5c, 0xc58b0eaa,
        0xc06aaa44, 0x49bfbc5e, 0xc2cc812f, 0x47668a44, 0x44e681d0, 0xc2233d70, 0xca41b0a1,
        0x43b74f38, 0x441894b5, 0x4a40000e, 0x43dfb6bd, 0xc9c62124, 0x4512d1da, 0xc7af0d5c,
        0x44011208, 0x4822db99, 0xc5bd675e, 0x49c0332a, 0x455239d1, 0x447ab714, 0x42af1364,
        0x44ac2f5c, 0xc708773f, 0x42cf7e3c, 0xc227c269, 0xc458c9ec, 0xc585d52e, 0x44b643b4,
        0x4589ce5a, 0xc2889eaf, 0xc3361273, 0xc70e9bb2, 0xc750155a, 0x438296ef, 0xc3b006fa,
        0x43434996, 0x47271cbc, 0xc473e1d8, 0x44675aa8, 0x4671220b, 0x445832e3, 0xc3a2f114,
        0x48830b5c, 0xc3ae944a, 0xc41c6330, 0xc2cf136f, 0xc299d81e, 0xc56ad0ec, 0x458d85e2,
        0xc2dd952e, 0xc4778156, 0xc54d5dc2, 0xc41582c2, 0xc583f5aa, 0xc90018fa, 0x461d72ad,
        0x45566e01, 0x477c51c0, 0x4951dceb, 0x450764cd, 0x44a707e8, 0x4926c2f4, 0xc581c245,
        0x42aebb51, 0x4401ec7f, 0xc8c2ae63, 0xc288dae3, 0xc33eb625, 0x4a3e8203, 0x422c6d2c,
        0x450501f6, 0xc33de6fb, 0x4133aa00, 0x44036425, 0xc28e89a7, 0x3e9ca62e, 0xc4af6956,
        0xc5cf3bf2, 0xc5cc1ae5, 0x443b7f4f, 0x41d134a9,
    ];
    // Raw gfx1201 hardware wave (rcpt/gfx1201-C-r1 swmmac f32_16x16x32_bf16 `numerics` case 7, random 2:4 positions, index_key 0),
    // register-major, lane-minor.
    const SW_BF16_A: [u32; 128] = [
        0x800043ff, 0x437fc200, 0xc020bfff, 0x3f0842cf, 0x3d823e78, 0x3fc33f80, 0x4000bda0,
        0x40b1c3c0, 0x3f78c27f, 0x8000401c, 0xc0b43da0, 0x3f7f0100, 0xc37fc211, 0xc2484000,
        0x3fa0c3c6, 0x3e00c352, 0xc178c394, 0xc1c042c0, 0xc2fd4250, 0xc2013f20, 0x3dd6be6d,
        0x80004300, 0x41dfbea0, 0xc180417c, 0x41733e78, 0xbd984193, 0x4104417f, 0x41483f80,
        0xc07f3f9d, 0x4104c380, 0xc0190000, 0x0000c0ff, 0xbda8c2b9, 0x407f8000, 0xc27f43b9,
        0x3f7fbdda, 0xc3800000, 0x4b003e98, 0x41b0c1c6, 0x3dab434d, 0xc1b03dff, 0xc3cb3df0,
        0xc2304360, 0xc080c0b8, 0xbdccbf30, 0x43abc1de, 0xbe10c068, 0xbf847f40, 0xbd92c37d,
        0xbeb3c0ff, 0xc1c93e00, 0xbf78c3d4, 0x417fc389, 0xc1ff4001, 0xbdff3fff, 0xbf904280,
        0x80003f69, 0xc1080000, 0x43ab3f92, 0xbe104202, 0x3fe3c2ff, 0x41603df5, 0xc1803e82,
        0x00003fff, 0x7f7f0000, 0x8000c1f8, 0x800042df, 0x4080c182, 0x437f421b, 0x8000c318,
        0xc0804000, 0xbd803ecc, 0xc2b34030, 0x3e81be84, 0x3da67f7f, 0xc2c23fdd, 0xbf7f4159,
        0x42ff427f, 0x40008000, 0x407f405f, 0xbe00c281, 0x80004311, 0xc080c3a0, 0xbdffc080,
        0x3f5a43cb, 0x40004000, 0x3f4d8000, 0xc2170000, 0x3db9c0c0, 0xbd80bd98, 0xc27b3eff,
        0xc082c0e8, 0x415bbf80, 0x8000be7f, 0xc0887f01, 0x40108000, 0xff7f42f7, 0xc00ebe63,
        0x429dc280, 0x3f08c0c8, 0x3e50c2e8, 0xbf8a3f7f, 0xbf81c080, 0xc0804b00, 0x80004156,
        0x7f403f80, 0xc08440ff, 0x40fe3e7f, 0x417f3ee9, 0xc0ea0000, 0x438bc07f, 0x3e903fa1,
        0xbfcd3f82, 0xbfa6ff40, 0x3dd93dff, 0xc0c0bfde, 0x42b842f6, 0xbe7f8000, 0x41043d81,
        0x00008000, 0xc1d8bda3, 0xbeff3e10, 0x00003fda, 0x4208c329, 0xc27fc282, 0x3f0042b1,
        0x00003f92, 0x4313c3ee,
    ];
    const SW_BF16_B: [u32; 256] = [
        0x80008000, 0x3fa9bdad, 0xff003ff1, 0xff40c130, 0x413a0081, 0x00403f84, 0xbe94bdef,
        0x3e00ff7f, 0xc0ff4b80, 0x80018000, 0x3ec1c008, 0xbf814180, 0x8080c280, 0xc2ff0000,
        0xc0683dff, 0x43e0426c, 0xbeabbe80, 0x3f7fbf88, 0xc1020000, 0xc2f83e88, 0xff00c200,
        0x3df0427f, 0xc015c1c0, 0xc1b843e0, 0xc2808000, 0x800041ff, 0xc371beff, 0xbf8ac320,
        0xbf00c17f, 0x3e3b3f7a, 0xc2868000, 0x407f0081, 0x4045c077, 0x80000000, 0xc3f8c277,
        0x43803f02, 0x43003f20, 0xc303be7f, 0xbe8bc380, 0xc17f3e5e, 0x4000c2e1, 0x7f003f58,
        0x3dffc0dd, 0xbfff43b2, 0xbddebffd, 0x80004002, 0xc369bee3, 0xc2553d9d, 0x0000c380,
        0x3f81c24c, 0xc17f4082, 0x43f98000, 0xc300c3ff, 0x3e7f4100, 0xc1d23eb0, 0x42a63e2d,
        0x0000c1c4, 0x40003eff, 0x0000412b, 0x3fd73f04, 0xc080c0ff, 0xc0984380, 0x41113e80,
        0x3db03380, 0x3e22be00, 0x437abf9c, 0x43804301, 0x80007f00, 0x8000bfef, 0x4031bf00,
        0xbf883dc0, 0x3d81c110, 0x433e8000, 0x42000000, 0xbe7f3e9a, 0x40113dcf, 0x3fe84280,
        0xbe80bea0, 0xc1a8bfff, 0xc1d2c25a, 0x3e7c407f, 0x3fbd4060, 0xc1c53f80, 0x3e20c37f,
        0x4008c150, 0x43ddc310, 0x3f0740bc, 0xbd828000, 0x3e40bfa0, 0x4011c3c8, 0x4190be7f,
        0x0000422f, 0x42083e78, 0x00003e27, 0xc3384380, 0xc2208000, 0x3f003f88, 0x0000bee4,
        0x4b80c20b, 0xc0004000, 0x439a4294, 0x3380bf00, 0xc2c83e20, 0x80000000, 0x41843d90,
        0xbd97c1d2, 0x42084342, 0xc39340de, 0x3fe54057, 0x4295c214, 0x43ffbfbd, 0x8000c17f,
        0x43f44000, 0x00000000, 0x3ea90000, 0xc3f07f00, 0xc0cd0000, 0xc0ff4060, 0xbfe68000,
        0xbe80c2ff, 0xc2383d93, 0x41ff3fff, 0x3eff3e04, 0xbf88ff00, 0x3ed843d4, 0x41904363,
        0xc1ff40b8, 0x40dc3ff0, 0x00814300, 0x3fe1c1d8, 0x407d4380, 0x43004336, 0xc162c180,
        0x3f7fc250, 0xc0b80000, 0xc0a0007f, 0x3dcabf88, 0x42df42fe, 0x41003f00, 0xbdc7c186,
        0x427f3d83, 0x43883eff, 0xc37fc190, 0x7f403fc7, 0xbe88bf0a, 0xc1e53da8, 0x4117c065,
        0x3d92c2cb, 0xc101c380, 0xc14a4380, 0xc2a0c190, 0xc25d0040, 0xc180c000, 0x3ff13fa2,
        0x4164be80, 0x3e003d88, 0xc2e7c1d1, 0x4033bd80, 0x3f808000, 0x40404384, 0x3fff4006,
        0xbe7f3d80, 0xbf008000, 0xc0b6c002, 0x0000bf39, 0xc1043e44, 0x43804328, 0xc290404b,
        0x41ca8000, 0x80007f40, 0x0000be00, 0x3df0bf80, 0x80014200, 0xbeffc281, 0x3e04be80,
        0xbdd7c380, 0xc3c041e0, 0xc180bf7e, 0x417f4330, 0x4008c301, 0xc0ea3ff0, 0x8000c368,
        0x40003d84, 0xbf803f21, 0xc2e0c000, 0xc0f8bef0, 0xc27f3d98, 0xbef1c220, 0x3f36bd82,
        0xbe84437f, 0xc06e4b80, 0x437f4037, 0x41d7be34, 0x3dd80000, 0x41080000, 0x4100427f,
        0xc2ffbe80, 0x43808000, 0x41008000, 0xbf88bfff, 0x4221c11a, 0xc2283e02, 0xbefec37b,
        0xc080be68, 0xc1ff3eef, 0xbe13c020, 0xbe004200, 0x3d88c200, 0xc0204346, 0xff7f3e69,
        0xbf803d88, 0x8001bf7f, 0xff40be81, 0xc084c1ff, 0x40420000, 0xc2a03f7f, 0x42a0c020,
        0x437f4274, 0x4392c380, 0x00004050, 0x00003f2a, 0x41c53e7f, 0x4380c380, 0x43e8c03e,
        0x0000bff0, 0xc27fbf18, 0xc09d4304, 0x3f6d437f, 0x8000c1cb, 0xbeff0000, 0xbe3bc0da,
        0x4116c07f, 0xc090c0e5, 0x40a0407f, 0xc3803f00, 0xc23bbdc0, 0xc2008000, 0xbe7f8000,
        0x4183402a, 0xc3803ebd, 0xbe13ff7f, 0xbde7c310, 0x42bf0000, 0xc0708000, 0x3d90c0a4,
        0xbd80c247, 0xbe804080, 0xbd9e4210, 0x3f203f0b, 0xbf80c298, 0xbf004380, 0xc27fbe04,
        0x4b004100, 0x3f883e75, 0xc2803e88, 0xbf924200,
    ];
    const SW_BF16_IDX: [u32; 32] = [
        0x00009cde, 0x0000cded, 0x0000ad9a, 0x0000c8d9, 0x0000ecd8, 0x0000dcdd, 0x0000dacc,
        0x0000ea8c, 0x0000d6de, 0x0000cd89, 0x0000ddee, 0x000049e9, 0x000099cd, 0x00009d9c,
        0x00009cdc, 0x000045e9, 0x0000ee9d, 0x0000eece, 0x0000ecde, 0x000088ce, 0x0000ee84,
        0x00009cee, 0x0000cc9a, 0x0000ccce, 0x0000eced, 0x0000dcea, 0x0000eddc, 0x00009eed,
        0x0000aeae, 0x000088d8, 0x000094de, 0x0000d9cc,
    ];
    const SW_BF16_C: [u32; 256] = [
        0x42800008, 0xc0ca3571, 0x3f53bd35, 0xc3ffffff, 0xbfb0bb9d, 0xc4e38f69, 0xbd5183d4,
        0xc3000000, 0x3e70bfa5, 0xbe100000, 0x3e2dbcca, 0xc4f12800, 0xc620f50e, 0xc0800800,
        0x43ae0f71, 0x40000000, 0xc01fbaad, 0x43e86ddb, 0xbe0518a3, 0xbf840000, 0x00800001,
        0xbf0cc7fe, 0xbe800000, 0xbd000000, 0xc5153a78, 0x007fffff, 0x427fffff, 0x42b8e9c9,
        0xc0e084c7, 0xc3000008, 0x3effffff, 0x3f8daad5, 0x40af9257, 0x42876273, 0x45000000,
        0x42004000, 0xbf800000, 0xc0fab055, 0x435a13d0, 0x437fffff, 0xc0000001, 0xc66bbcba,
        0x46000000, 0x436f1000, 0x4b000000, 0xc07fffff, 0x429b5e73, 0x421c6e66, 0x447fffff,
        0xbf991000, 0x41abe000, 0x440b28c8, 0x4279f29a, 0xc0802000, 0xc27a0884, 0x3f800100,
        0x4694d800, 0x43dcc912, 0x3cf78b32, 0xc373a800, 0x40ffffff, 0x42000800, 0xbddd4000,
        0x44400000, 0x3d7fffff, 0x43ecb800, 0x41ffffff, 0xc3001000, 0x467fffff, 0xc57fffff,
        0x3e000000, 0x3ce1d7d7, 0x3fc0abcd, 0x3f800001, 0x43200000, 0xc6080000, 0xc1019800,
        0x3f800001, 0x3dd71000, 0xbf000000, 0xc0b64a24, 0xc3808000, 0x402b9157, 0xbe569800,
        0x007fffff, 0x447fffff, 0x44000000, 0x40400000, 0x4409ea50, 0x3f613297, 0x3f800000,
        0x3f800000, 0x439a66dc, 0xc1800000, 0xbee8841e, 0xc47fffff, 0x40e2b1ed, 0x419de800,
        0x46dac800, 0x440330f6, 0xc4846ed8, 0x40e9b486, 0x40400000, 0x4324f000, 0x5f000000,
        0x447fffff, 0x3dcf6b0b, 0x43f6f962, 0xc17fffff, 0xbef4cf2b, 0x3dced000, 0xc56c46b9,
        0x40800080, 0x44f69dfc, 0x44d53b43, 0xc48606e9, 0x43000000, 0xc3000000, 0xc1000000,
        0x5f000000, 0xc41c8d71, 0xc180ec93, 0x3d529040, 0xbfffffff, 0x3f7fffff, 0x43400000,
        0x00800001, 0xc57247c5, 0x43132c8d, 0x3e4345d4, 0x40000000, 0x4551f000, 0x4292bd3d,
        0x41f8f8b5, 0x424a5cb4, 0x407fffff, 0xc515a901, 0xc1dc993a, 0x40880000, 0x40347968,
        0x43000000, 0x4068cbaf, 0x4249ca3c, 0x40800000, 0x42ee34d7, 0x427fffff, 0xc3c7a02c,
        0x4b000000, 0x00000000, 0x40ffffff, 0x41d34000, 0xc2bda239, 0x3e93bb56, 0xc0ffffff,
        0xc6800000, 0x00000100, 0x41ba0800, 0xbd5ec338, 0x40137249, 0xc566f000, 0x41ffffff,
        0x452790aa, 0x3e000000, 0xbd7fffff, 0x4358bcde, 0xc3800000, 0x3fbaf043, 0xc2800000,
        0x448515d4, 0x422ad000, 0x3d000200, 0x40062986, 0xbcfd5e5c, 0xc0000000, 0x437fffff,
        0x3fd0bc3e, 0x43c50281, 0xbe801000, 0xc213b0a1, 0x4064cb3e, 0xc07fffff, 0xc4000020,
        0xbffa74cc, 0xc0e3d729, 0x42fe8b9d, 0xc3ae8507, 0x3f7fffff, 0xbe009eb0, 0x00000001,
        0x41afac4e, 0x3fe3a000, 0xc38d916d, 0xbed913d8, 0xbe7fffff, 0x3d080000, 0xff7fffff,
        0x461b6800, 0x3f22fd43, 0x40800000, 0xbf800000, 0x40000000, 0xc17cbc30, 0x3d7fffff,
        0xc0e373a3, 0x40800010, 0xbeeadde7, 0xc3ffc1e6, 0x450117fe, 0x4421796f, 0x43000000,
        0xc188c800, 0x007fffff, 0xc2800000, 0xc453faa8, 0xc3616800, 0x407fffff, 0x439fc000,
        0x46ffffff, 0x467fffff, 0x4100cf87, 0xc6000400, 0xc1ffffff, 0x45000002, 0x475ac2f4,
        0x3effffff, 0x40ffffff, 0x40ffffff, 0xbfd420de, 0x4319e473, 0xbf000000, 0x477fffff,
        0xc32875ec, 0xbf800000, 0x3f800000, 0xbf3ba1de, 0x46279c62, 0x3d02e4ec, 0x44f0035b,
        0x00000001, 0xc2b71f25, 0x41800100, 0x42000008, 0xc27fffff, 0x40e616d2, 0xc37fffff,
        0xc0858800, 0xc2d1bb1c, 0x3d3f298a, 0xc37fffff, 0xbe800000, 0x42bac000, 0x456da800,
        0xc4080000, 0xc27c8448, 0xc4000400, 0xc4ffffff,
    ];
    const SW_BF16_D: [u32; 256] = [
        0x7f800000, 0x7eafcf80, 0xff800000, 0xff800000, 0x7f800000, 0xff800000, 0x7f800000,
        0xff800000, 0x7f800000, 0xff800000, 0xfeff0000, 0x7e7ba800, 0xcb4d5d10, 0xfefe0100,
        0xff800000, 0xfef1cd40, 0x7da25d00, 0x43b82e2c, 0xcdaffff4, 0x7ee90000, 0xfdf80000,
        0x44d49909, 0x4696ecb6, 0x44c1d9b0, 0x459a8ee4, 0x7f800000, 0xc4817ffd, 0xfee90000,
        0xcd580158, 0xc6cd9355, 0x464f1b6e, 0xff800000, 0x7f800000, 0x7f800000, 0x7f800000,
        0x7f800000, 0x7f800000, 0x7f800000, 0xff800000, 0xff800000, 0xfed08001, 0x7f800000,
        0xff800000, 0x7dc60000, 0xff800000, 0xfe37c000, 0xfe4c0000, 0xff800000, 0xc6006137,
        0xff800000, 0xff800000, 0xff800000, 0xfdd80000, 0xfebf4000, 0xfe0c4000, 0x7f800000,
        0xff800000, 0x7f800000, 0xff800000, 0xff800000, 0xff800000, 0xfe3f4000, 0x7f800000,
        0xff800000, 0xfdfe0100, 0x47b09edb, 0x47c8fd68, 0xc43c8e30, 0x479f2f54, 0xc7a47b74,
        0x460edd44, 0xc62cf386, 0x478bffc4, 0x46c01ad9, 0xc6236439, 0xc61de207, 0x495ce88c,
        0x4733f3ec, 0xc70e3f2e, 0x7f800000, 0xff800000, 0x7f800000, 0x7f800000, 0x7f800000,
        0xff800000, 0x7f7e0100, 0xff800000, 0xff800000, 0x7dc93600, 0xff800000, 0x7f800000,
        0xfdc63900, 0x7f800000, 0x7f800000, 0xff800000, 0x7f800000, 0x7f800000, 0xc3161068,
        0xff800000, 0xff800000, 0x462d4fbc, 0x476f57bc, 0xc494200b, 0xc527e1b1, 0x5f000000,
        0x7f800000, 0x44008f54, 0xc68d310e, 0x4327dc98, 0xc64deb20, 0xca06197f, 0x471da083,
        0xff800000, 0x7f800000, 0xcc7fff7d, 0x7f800000, 0x7f800000, 0x45287818, 0xc6836acd,
        0x5f000000, 0xc6541a9e, 0xff800000, 0xc723c145, 0xff800000, 0x462792fd, 0x45fe9dc1,
        0xcce81631, 0x7f800000, 0xff800000, 0xc69d6e45, 0xcf7ffedf, 0x7f800000, 0xfd560000,
        0xc77d8b76, 0x47b0f27a, 0xfe770800, 0x4a78026b, 0x47cbfb71, 0x46b11f7a, 0xff800000,
        0x4e380072, 0x47abfaf2, 0x4fcafe6a, 0x45a262ea, 0x7f800000, 0x468edd86, 0x7f800000,
        0x7f800000, 0xc70215c3, 0x471b4540, 0xc41a3b84, 0x46689914, 0xc4f49bc9, 0xff800000,
        0xc743e864, 0x7f800000, 0xc750f929, 0xc69d4115, 0xcb7f283a, 0x7f800000, 0x7e7e0100,
        0x43cab840, 0xff000000, 0x7e840000, 0x4f19ff0a, 0x4614aaf0, 0xce480001, 0xc3a8d523,
        0x4d03ffdd, 0x7f42ffff, 0x4d88002a, 0xff800000, 0x4b64ff11, 0x4e1501fb, 0x4f7f000c,
        0x4497e46d, 0xfeff0000, 0xc5e1302a, 0x459d9a82, 0xc6519c84, 0x46acd616, 0xc661493c,
        0x471691ca, 0xff800000, 0x4bfffa1e, 0xff800000, 0x472e04a2, 0x472aa798, 0x460eb665,
        0x45d41c8e, 0xc6d80706, 0x7f800000, 0xc5e89eaa, 0xc4a55b4a, 0x4dafff96, 0xff800000,
        0x45595b9a, 0xc32edcb4, 0xc51fc9b2, 0x7d9f6000, 0xc9a01119, 0xff000002, 0x449d8c54,
        0x7d7f0000, 0x4c83ff32, 0x4609b1a6, 0x46126328, 0x4582dd72, 0xfe8b1400, 0xff800000,
        0xff800000, 0xff800000, 0xff800000, 0x7f800000, 0xff800000, 0x7f800000, 0xff800000,
        0x7f800000, 0xfe010000, 0x7d091000, 0xff800000, 0xfd010000, 0x4546c68e, 0x7f800000,
        0xcb6fd84e, 0xca97ff94, 0x4e8400b1, 0x7f800000, 0xcd4b017b, 0xc6210a65, 0xcc59bf2d,
        0x7f800000, 0xcfc1c9f9, 0x7f800000, 0x4a8022f7, 0xc93e263c, 0x47138124, 0x45b9a19c,
        0x4baa1c87, 0x4a3a617b, 0x7f4bffff, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000,
        0xfec00001, 0x7df00000, 0x470681c0, 0x7d580000, 0xff800000, 0x7f800000, 0x7f800000,
        0x7f800000, 0xff800000, 0xff800000, 0xff800000,
    ];

    fn swmmac_f32_wave(name: &str, a: &[u32], b: &[u32], idx: &[u32], c: &[u32]) -> State {
        let mut st = state();
        for lane in 0..32 {
            for vgpr in 0..4 {
                st.v[usize::from(A) + vgpr][lane] = a[32 * vgpr + lane];
            }
            for vgpr in 0..8 {
                st.v[usize::from(B) + vgpr][lane] = b[32 * vgpr + lane];
                st.v[usize::from(D) + vgpr][lane] = c[32 * vgpr + lane];
            }
            st.v[usize::from(IDX)][lane] = idx[lane];
        }
        let i = insn(
            Arch::Gfx1201,
            name,
            [reg(D, 8), reg(A, 4), reg(B, 8), reg(IDX, 1)],
            mods(0, 0),
        );
        execute(Arch::Gfx1201, name, &i, &mut st).unwrap();
        st
    }

    // The two recorded hardware waves reproduce word for word (random C, random 2:4 selectors, duplicates included).
    #[test]
    fn swmmac_f32_reproduces_recorded_hardware_waves() {
        let cases: [(&str, &[u32], &[u32], &[u32], &[u32], &[u32]); 2] = [
            (
                "v_swmmac_f32_16x16x32_f16",
                &SW_F16_A,
                &SW_F16_B,
                &SW_F16_IDX,
                &SW_F16_C,
                &SW_F16_D,
            ),
            (
                "v_swmmac_f32_16x16x32_bf16",
                &SW_BF16_A,
                &SW_BF16_B,
                &SW_BF16_IDX,
                &SW_BF16_C,
                &SW_BF16_D,
            ),
        ];
        for (name, a, b, idx, c, d) in cases {
            let st = swmmac_f32_wave(name, a, b, idx, c);
            for vgpr in 0..8 {
                for lane in 0..32 {
                    assert_eq!(
                        st.v[usize::from(D) + vgpr][lane],
                        d[32 * vgpr + lane],
                        "{name} D vgpr {vgpr} lane {lane}"
                    );
                }
            }
        }
    }

    // Hand-placed: row 3 (lane 3, half 0), packed element 2 = f16 2.0 (VGPR 1 half 0) with selector 1 -> slot 2 ->
    // K = 5; B[5][6]: lane 6, e=5 -> VGPR 2 half 1 = 3.0. D[3][6]: lane 6, VGPR 3, C = 1.0 -> 7.0.
    // A duplicate-K pair on row 4 half 1 (lane 20): slots 8 and 9 both select 2 -> K = 18, B[18][9] = 4.0 in lane 25
    // (e=2: VGPR 1 half 0): 2.0*4 + 3.0*4 = 20.
    #[test]
    fn swmmac_f32_selects_b_through_the_index_and_keeps_duplicate_terms() {
        let st = {
            let mut st = state();
            st.v[usize::from(A) + 1][3] = 0x4000; // f16 2.0, element 2
            st.v[usize::from(IDX)][3] = 1 << 4;
            st.v[usize::from(B) + 2][6] = 0x4200 << 16; // f16 3.0 at e=5
            st.v[3][6] = 1f32.to_bits();
            st.v[usize::from(A)][20] = 0x4000 | (0x4200 << 16); // elements 0 and 1 = 2.0, 3.0
            st.v[usize::from(IDX)][20] = (2 << 0) | (2 << 2);
            st.v[usize::from(B) + 1][25] = 0x4400; // f16 4.0 at e=2
            let i = insn(
                Arch::Gfx1201,
                "v_swmmac_f32_16x16x32_f16",
                [reg(D, 8), reg(A, 4), reg(B, 8), reg(IDX, 1)],
                mods(0, 0),
            );
            execute(Arch::Gfx1201, "v_swmmac_f32_16x16x32_f16", &i, &mut st).unwrap();
            st
        };
        assert_eq!(f32::from_bits(st.v[3][6]), 7.0);
        assert_eq!(f32::from_bits(st.v[4][9]), 20.0);
        // Everything else is C = 0 plus zero products.
        assert_eq!(st.v[0][0], 0);
    }

    // IU8 WMMA CLAMP on gfx1201: saturate after K 0..8, again after K 8..16 (receipt variants 6/7). Raw hardware words:
    // C = 0x7fffffff; the first half adds a large positive sum (saturates), the second half a negative one.
    #[test]
    fn gfx12_iu8_clamp_saturates_after_each_k_half() {
        let a_row = [
            115u32, 122, 82, 45, 63, 37, 83, 30, 102, 94, 95, 6, 60, 124, 83, 118,
        ];
        let b_cols: [([u32; 16], i32); 2] = [
            (
                [
                    1, 254, 3, 252, 5, 250, 7, 248, 9, 246, 11, 244, 13, 242, 15, 240,
                ],
                2147482999u32 as i32,
            ),
            (
                [
                    8, 247, 10, 245, 12, 243, 14, 241, 16, 239, 18, 237, 20, 235, 22, 233,
                ],
                2147482985u32 as i32,
            ),
        ];
        for neg_lo in [2u8, 3] {
            for (col, (b_col, want)) in b_cols.iter().enumerate() {
                let mut st = state();
                for k in 0..16 {
                    let (lane, vgpr, byte) = iu8_loc(Arch::Gfx1201, 0, k);
                    put_byte(&mut st, A, vgpr, lane, byte, a_row[k]);
                    let (lane, vgpr, byte) = iu8_loc(Arch::Gfx1201, col, k);
                    put_byte(&mut st, B, vgpr, lane, byte, b_col[k]);
                }
                st.v[usize::from(C)][0] = 0x7fff_ffff;
                st.v[usize::from(C)][1] = 0x7fff_ffff;
                let clamped = Modifiers {
                    clamp: true,
                    ..mods(neg_lo, 0)
                };
                let i = insn(
                    Arch::Gfx1201,
                    "v_wmma_i32_16x16x16_iu8",
                    [reg(D, 8), reg(A, 2), reg(B, 2), reg(C, 8)],
                    clamped,
                );
                execute(Arch::Gfx1201, "v_wmma_i32_16x16x16_iu8", &i, &mut st).unwrap();
                let word = if col == 0 {
                    st.v[usize::from(D)][0]
                } else {
                    st.v[usize::from(D)][1]
                };
                assert_eq!(word as i32, *want, "neg_lo={neg_lo} col={col}");
            }
        }
    }

    // gfx1151 IU8 CLAMP has no receipt: refused; the unclamped wrap is unchanged.
    #[test]
    fn gfx1151_iu8_clamp_is_refused() {
        let mut st = state();
        let clamped = Modifiers {
            clamp: true,
            ..mods(3, 0)
        };
        let i = insn(
            Arch::Gfx1151,
            "v_wmma_i32_16x16x16_iu8",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            clamped,
        );
        assert!(execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu8", &i, &mut st).is_err());
        let plain = insn(
            Arch::Gfx1151,
            "v_wmma_i32_16x16x16_iu8",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(3, 0),
        );
        assert!(execute(Arch::Gfx1151, "v_wmma_i32_16x16x16_iu8", &plain, &mut st).is_ok());
    }

    #[test]
    fn unsupported_variants_are_hard_errors() {
        let mut st = state();
        let i = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_f16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            mods(0, 0),
        );
        // BF16-out and FP8 have no probed model: still refused.
        assert!(execute(Arch::Gfx1201, "v_wmma_bf16_16x16x16_bf16", &i, &mut st).is_err());
        assert!(execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_fp8_fp8", &i, &mut st).is_err());
        // BF16 is not modeled on gfx1100 (no probe corpus).
        assert!(execute(Arch::Gfx1100, "v_wmma_f32_16x16x16_bf16", &i, &mut st).is_err());
        assert!(execute(Arch::Gfx1100, "v_wmma_f32_16x16x16_f16", &i, &mut st).is_err());
        assert!(execute(Arch::Gfx1151, "v_wmma_i32_16x16x32_iu4", &i, &mut st).is_err());
        let clamped = insn(
            Arch::Gfx1201,
            "v_wmma_f32_16x16x16_f16",
            [reg(D, 8), reg(A, 4), reg(B, 4), reg(C, 8)],
            Modifiers {
                clamp: true,
                ..mods(0, 0)
            },
        );
        assert!(execute(Arch::Gfx1201, "v_wmma_f32_16x16x16_f16", &clamped, &mut st).is_err());
    }

    // ---- exact-domain SWMMAC kinds (16-bit-out and FP8/BF8): independent dense reference, packing, refusals ----
    const EXACT_KINDS: [(&str, Fmt, Fmt, Fmt); 6] = [
        ("v_swmmac_f16_16x16x32_f16", FMT_F16, FMT_F16, FMT_F16),
        ("v_swmmac_bf16_16x16x32_bf16", FMT_BF16, FMT_BF16, FMT_BF16),
        ("v_swmmac_f32_16x16x32_fp8_fp8", FMT_FP8, FMT_FP8, FMT_F32),
        ("v_swmmac_f32_16x16x32_fp8_bf8", FMT_FP8, FMT_BF8, FMT_F32),
        ("v_swmmac_f32_16x16x32_bf8_fp8", FMT_BF8, FMT_FP8, FMT_F32),
        ("v_swmmac_f32_16x16x32_bf8_bf8", FMT_BF8, FMT_BF8, FMT_F32),
    ];
    const SET_FP8: &[f64] = &[
        0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 0.25,
    ];
    const SET_C_F32: &[f64] = &[0.0, 1.0, -1.0, 0.5, -0.25, 7.0, -3.5, 100.0, 0.0009765625];
    const SET_F16: &[f64] = &[0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0];
    const SET_C_F16: &[f64] = &[0.0, 1.0, -1.0, 0.5, -0.25, 3.0];
    const SET_BF16: &[f64] = &[0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5];
    const SET_C_BF16: &[f64] = &[0.0, 1.0, -1.0, 0.5, -0.25, 2.0];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() >> 11) as usize % n
        }
    }
    fn width(f: Fmt) -> u32 {
        f.ebits + f.mbits + 1
    }
    fn kind(name: &str) -> (Fmt, Fmt, Fmt) {
        let k = EXACT_KINDS.iter().find(|k| k.0 == name).unwrap();
        (k.1, k.2, k.3)
    }
    /// (element bits, A regs, B regs, D regs) of a kind.
    fn exact_regs(af: Fmt, of: Fmt) -> (u32, u8, u8, u8) {
        let bits = width(af);
        let per = 32 / bits;
        (
            bits,
            (8 / per) as u8,
            (16 / per) as u8,
            if width(of) == 16 { 4 } else { 8 },
        )
    }
    /// Bits of the exactly representable normal value (or +0) `v` in `f`.
    fn enc(f: Fmt, v: f64) -> u32 {
        if v == 0.0 {
            return 0;
        }
        let b = v.to_bits();
        let e = ((b >> 52) & 0x7ff) as i32 - 1023;
        let frac = b & ((1u64 << 52) - 1);
        let drop = 52 - f.mbits;
        assert_eq!(
            frac & ((1u64 << drop) - 1),
            0,
            "{v} is not exact in the format"
        );
        let ef = e + f.bias;
        assert!(
            ef >= 1 && ef < (1 << f.ebits) - 1,
            "{v} is not a normal value"
        );
        (u32::from(v < 0.0) << (f.ebits + f.mbits))
            | ((ef as u32) << f.mbits)
            | (frac >> drop) as u32
    }
    fn nan(f: Fmt) -> u32 {
        if f.e4m3 {
            0x7f
        } else {
            (((1u32 << f.ebits) - 1) << f.mbits) | 1
        }
    }
    fn inf(f: Fmt) -> u32 {
        ((1u32 << f.ebits) - 1) << f.mbits
    }
    fn put_elem(st: &mut State, base: u16, lane: usize, e: usize, bits: u32, v: u32) {
        let per = (32 / bits) as usize;
        st.v[usize::from(base) + e / per][lane] |= v << (bits * (e % per) as u32);
    }
    fn exact_try(name: &str, m: Modifiers, setup: impl Fn(&mut State)) -> Result<State, String> {
        let (af, _, of) = kind(name);
        let (_, a_regs, b_regs, d_regs) = exact_regs(af, of);
        let mut st = state();
        setup(&mut st);
        let i = insn(
            Arch::Gfx1201,
            name,
            [reg(D, d_regs), reg(A, a_regs), reg(B, b_regs), reg(IDX, 1)],
            m,
        );
        execute(Arch::Gfx1201, name, &i, &mut st)?;
        Ok(st)
    }
    /// A[row 0][slot 0] = `a`, B[K 0][col 0] = `b`, C[0][0] = `c` (selector 0 -> K 0); every other input is +0.
    fn probe(name: &str, a: u32, b: u32, c: u32) -> Result<State, String> {
        let (af, _, _) = kind(name);
        let bits = width(af);
        exact_try(name, mods(0, 0), move |st| {
            put_elem(st, A, 0, 0, bits, a);
            put_elem(st, B, 0, 0, bits, b);
            st.v[usize::from(D)][0] |= c;
        })
    }

    /// One crafted wave (true 2:4 selectors, mixed signs, junk in the unused index half) against a dense reference:
    /// A decompressed by the selectors, then an exact dense dot per element in f64.
    fn exact_wave(name: &str, key: usize, neg: (u8, u8), seed: u64) {
        let (af, bf, of) = kind(name);
        let (bits, _, _, _) = exact_regs(af, of);
        let out16 = width(of) == 16;
        let (a_set, b_set, c_set) = match of.mbits {
            23 => (SET_FP8, SET_FP8, SET_C_F32),
            10 => (SET_F16, SET_F16, SET_C_F16),
            _ => (SET_BF16, SET_BF16, SET_C_BF16),
        };
        let mut rng = Rng(seed | 1);
        let mut a = [[0f64; 16]; 16]; // [row][compressed slot j = 8h + e]
        let mut sel = [[0usize; 16]; 16];
        let mut idx = [0u32; 32];
        for r in 0..16 {
            for h in 0..2 {
                let mut fields = 0u32;
                for p in 0..4 {
                    let s0 = rng.below(4);
                    let s1 = (s0 + 1 + rng.below(3)) % 4; // the two slots of a K-group pick different positions
                    for (e, s) in [(2 * p, s0), (2 * p + 1, s1)] {
                        a[r][8 * h + e] = a_set[rng.below(a_set.len())];
                        sel[r][8 * h + e] = s;
                        fields |= (s as u32) << (2 * e);
                    }
                }
                let junk = rng.next() as u32;
                idx[r + 16 * h] = if key == 0 {
                    (junk & 0xffff_0000) | fields
                } else {
                    (junk & 0x0000_ffff) | (fields << 16)
                };
            }
        }
        let mut b = [[0f64; 16]; 32]; // [K][col]
        for row in b.iter_mut() {
            for v in row.iter_mut() {
                *v = b_set[rng.below(b_set.len())];
            }
        }
        // Dense decompression: slot j of row r sits at K = 4*(j>>1) + selector; neg modifiers follow the slot parity.
        let mut ad = [[0f64; 32]; 16];
        let mut sb = [[1f64; 32]; 16];
        for r in 0..16 {
            for j in 0..16 {
                let k = 4 * (j >> 1) + sel[r][j];
                let word = if j % 2 == 0 { neg.0 } else { neg.1 };
                ad[r][k] = if word & 1 != 0 { -a[r][j] } else { a[r][j] };
                if word & 2 != 0 {
                    sb[r][k] = -1.0;
                }
            }
        }
        let mut c = [[0f64; 16]; 16];
        let mut want = [[0f64; 16]; 16];
        for r in 0..16 {
            for col in 0..16 {
                let dot: f64 = (0..32).map(|k| ad[r][k] * sb[r][k] * b[k][col]).sum();
                let start = rng.below(c_set.len());
                // An exactly cancelled zero has an unmeasured sign, so the wave never produces one.
                let pick = (0..c_set.len())
                    .map(|t| c_set[(start + t) % c_set.len()])
                    .find(|&cand| dot + cand != 0.0)
                    .unwrap();
                c[r][col] = pick;
                want[r][col] = dot + pick;
            }
        }
        let m = Modifiers {
            op_sel: key as u8,
            ..mods(neg.0, neg.1)
        };
        let st = exact_try(name, m, |st| {
            for r in 0..16 {
                for h in 0..2 {
                    for e in 0..8 {
                        put_elem(st, A, r + 16 * h, e, bits, enc(af, a[r][8 * h + e]));
                    }
                }
            }
            for col in 0..16 {
                for hb in 0..2 {
                    for e in 0..16 {
                        put_elem(st, B, col + 16 * hb, e, bits, enc(bf, b[16 * hb + e][col]));
                    }
                }
            }
            for r in 0..16 {
                for col in 0..16 {
                    let lane = col + 16 * (r >> 3);
                    if out16 {
                        put_half(st, D, (r & 7) >> 1, lane, r & 1, enc(of, c[r][col]) as u16);
                    } else {
                        st.v[usize::from(D) + (r & 7)][lane] |= enc(of, c[r][col]);
                    }
                }
            }
            st.v[usize::from(IDX)][..32].copy_from_slice(&idx);
        })
        .unwrap_or_else(|e| panic!("{name} key={key} neg={neg:?} seed={seed}: {e}"));
        for r in 0..16 {
            for col in 0..16 {
                let lane = col + 16 * (r >> 3);
                let got = if out16 {
                    (st.v[usize::from(D) + ((r & 7) >> 1)][lane] >> (16 * (r & 1))) & 0xffff
                } else {
                    st.v[usize::from(D) + (r & 7)][lane]
                };
                assert_eq!(
                    got,
                    enc(of, want[r][col]),
                    "{name} key={key} neg={neg:?} seed={seed} row {r} col {col}: want {}",
                    want[r][col]
                );
            }
        }
    }

    #[test]
    fn swmmac_exact_kinds_match_a_dense_reference() {
        for (name, af, _, _) in EXACT_KINDS {
            // neg_lo/neg_hi exist only on the 16-bit-input kinds.
            let negs: &[(u8, u8)] = if width(af) == 16 {
                &[(0, 0), (3, 3), (1, 2), (2, 1)]
            } else {
                &[(0, 0)]
            };
            for key in 0..2 {
                for (n, &neg) in negs.iter().enumerate() {
                    for seed in 0..3u64 {
                        exact_wave(
                            name,
                            key,
                            neg,
                            0x9e37_79b9_7f4a_7c15 ^ (seed * 31 + n as u64 * 7 + key as u64),
                        );
                    }
                }
            }
        }
    }

    // 16-bit-out D: row r col c -> lane c + 16*(r>>3), VGPR (r&7)>>1, half r&1 (low half = even row). C is read from the
    // same half and every half of D is rewritten: D[r][c] = r (C) + (c+1) (A=1 times B[0][c]).
    #[test]
    fn swmmac_exact_16bit_out_packs_two_rows_per_vgpr() {
        for name in ["v_swmmac_f16_16x16x32_f16", "v_swmmac_bf16_16x16x32_bf16"] {
            let (f, _, _) = kind(name);
            let st = exact_try(name, mods(0, 0), |st| {
                for r in 0..16 {
                    put_elem(st, A, r, 0, 16, enc(f, 1.0));
                }
                for col in 0..16 {
                    put_elem(st, B, col, 0, 16, enc(f, (col + 1) as f64));
                }
                for r in 0..16 {
                    for col in 0..16 {
                        let lane = col + 16 * (r >> 3);
                        put_half(st, D, (r & 7) >> 1, lane, r & 1, enc(f, r as f64) as u16);
                    }
                }
            })
            .unwrap();
            for r in 0..16 {
                for col in 0..16 {
                    let lane = col + 16 * (r >> 3);
                    let word = st.v[usize::from(D) + ((r & 7) >> 1)][lane];
                    assert_eq!(
                        (word >> (16 * (r & 1))) & 0xffff,
                        enc(f, (r + col + 1) as f64),
                        "{name} row {r} col {col}"
                    );
                }
            }
            // Row 3, col 5: lane 5, VGPR 1, high half.
            assert_eq!(st.v[usize::from(D) + 1][5] >> 16, enc(f, 9.0), "{name}");
            // Row 12, col 2: lane 18, VGPR 2, low half.
            assert_eq!(
                st.v[usize::from(D) + 2][18] & 0xffff,
                enc(f, 15.0),
                "{name}"
            );
        }
    }

    #[test]
    fn swmmac_exact_accepts_the_domain_edge_and_e4m3_without_infinity() {
        let name = "v_swmmac_f32_16x16x32_fp8_fp8";
        let one = enc(FMT_FP8, 1.0);
        // C = 2^23, one product 1: |C| + |t| = 2^23 + 1 granules <= 2^24 and 2^23 + 1 is exact in f32.
        let st = probe(name, one, one, (8388608.0f32).to_bits()).unwrap();
        assert_eq!(st.v[usize::from(D)][0], 8388609.0f32.to_bits());
        // C = 2^24: 2^24 + 1 granules > 2^24 (and 2^24 + 1 is not an f32).
        let err = probe(name, one, one, (16777216.0f32).to_bits())
            .err()
            .unwrap();
        assert!(err.contains("exceeds 2^24"), "{err}");
        // E4M3 has no infinity: 0x78 (S.1111.000) is the normal value 256, 0x7f is the NaN.
        let st = probe(name, 0x78, one, 0).unwrap();
        assert_eq!(st.v[usize::from(D)][0], 256.0f32.to_bits());
        assert!(probe(name, 0x7f, one, 0).err().unwrap().contains("NaN"));
    }

    #[test]
    fn swmmac_exact_exact_values_have_the_expected_result() {
        for (name, af, bf, of) in EXACT_KINDS {
            // Positive operands only: a negative A or B against the other zeros would make a (refused) -0 product.
            let st = probe(name, enc(af, 1.5), enc(bf, 2.0), enc(of, 1.0)).unwrap();
            // 1.0 + 1.5 * 2.0 = 4.0
            let mask = ((1u64 << width(of)) - 1) as u32;
            assert_eq!(st.v[usize::from(D)][0] & mask, enc(of, 4.0), "{name}");
        }
    }

    #[test]
    fn swmmac_exact_refuses_everything_outside_the_domain() {
        for (name, af, bf, of) in EXACT_KINDS {
            let (one_a, one_b, one_c) = (enc(af, 1.0), enc(bf, 1.0), enc(of, 1.0));
            let denorm = 1u32;
            // NaN in A, selected B and C; denormal in A, selected B and C.
            let cases = [
                (nan(af), one_b, 0, "NaN or infinite"),
                (one_a, nan(bf), 0, "NaN or infinite"),
                (one_a, one_b, nan(of), "NaN or infinite"),
                (denorm, one_b, 0, "denormal"),
                (one_a, denorm, 0, "denormal"),
                (one_a, one_b, denorm, "denormal"),
                // Cancellation to zero and negative zeros: the sign of such a zero is unmeasured.
                (one_a, one_b, enc(of, -1.0), "cancellation"),
                (enc(af, -1.0), 0, 0, "negative-zero"),
                (0, 0, 1u32 << (width(of) - 1), "negative-zero"),
            ];
            for (a, b, c, needle) in cases {
                let err = probe(name, a, b, c).err().unwrap_or_else(|| {
                    panic!("{name}: a={a:#x} b={b:#x} c={c:#x} should be refused ({needle})")
                });
                assert!(err.contains(needle), "{name}: {err} (wanted {needle})");
            }
            // Infinity exists in every format but E4M3 (so: bf8 inputs, and 16-bit A/B/C).
            let infs = [
                (!af.e4m3, inf(af), one_b, 0),
                (!bf.e4m3, one_a, inf(bf), 0),
                (true, one_a, one_b, inf(of)),
            ];
            for (exists, a, b, c) in infs {
                if !exists {
                    continue;
                }
                let err = probe(name, a, b, c).err().unwrap_or_else(|| {
                    panic!("{name}: infinity a={a:#x} b={b:#x} c={c:#x} should be refused")
                });
                assert!(err.contains("NaN or infinite"), "{name}: {err}");
            }
            // Total beyond 2^P granules: C = 2^20 (f32 out) or 1.0 against the smallest-normal product.
            let tiny = |f: Fmt| enc(f, 2f64.powi(1 - f.bias));
            let big = if of.mbits == 23 {
                enc(of, 1048576.0)
            } else {
                one_c
            };
            let err = probe(name, tiny(af), tiny(bf), big).err().unwrap();
            assert!(err.contains("exceeds 2^"), "{name}: {err}");
        }
    }

    #[test]
    fn swmmac_exact_zero_without_cancellation_is_plus_zero() {
        for (name, _, _, _) in EXACT_KINDS {
            let st = probe(name, 0, 0, 0).unwrap();
            assert!(
                st.v[..8].iter().all(|row| row.iter().all(|&w| w == 0)),
                "{name}"
            );
        }
    }

    #[test]
    fn swmmac_exact_modifier_rules() {
        for (name, af, _, _) in EXACT_KINDS {
            let clamp = Modifiers {
                clamp: true,
                ..mods(0, 0)
            };
            assert!(exact_try(name, clamp, |_| {}).is_err(), "{name} clamp");
            assert!(
                exact_try(name, mods(4, 0), |_| {}).is_err(),
                "{name} NEG[2]"
            );
            let other_opsel = Modifiers {
                op_sel: 2,
                ..mods(0, 0)
            };
            assert!(
                exact_try(name, other_opsel, |_| {}).is_err(),
                "{name} OPSEL[1]"
            );
            // C = 1.0 everywhere keeps every total nonzero (a negated zero product is a refused -0 otherwise).
            let (_, _, of) = kind(name);
            let ones = move |st: &mut State| {
                let word = if width(of) == 16 {
                    enc(of, 1.0) * 0x0001_0001
                } else {
                    enc(of, 1.0)
                };
                for vgpr in 0..8 {
                    st.v[usize::from(D) + vgpr] = [word; 64];
                }
            };
            for (lo, hi) in [(1, 0), (0, 1), (2, 2), (3, 3)] {
                let r = exact_try(name, mods(lo, hi), ones);
                if width(af) == 8 {
                    let err = r.err().unwrap();
                    assert!(err.contains("neg_lo/neg_hi"), "{name}: {err}");
                } else {
                    assert!(r.is_ok(), "{name} neg_lo={lo} neg_hi={hi}");
                }
            }
        }
    }
}
