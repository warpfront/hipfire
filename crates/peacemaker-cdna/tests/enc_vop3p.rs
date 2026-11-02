// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP3P packed math and VOP3P-MAI (MFMA, accvgpr moves): encode / decode /
//! Display / footprint against the ROCm `llvm-mc` oracle.
//!
//! Oracle tests assemble hand-formatted asm (independent of `Display`), one
//! `llvm-mc` call per test, and assert for every case: our bytes == llvm's
//! bytes, `Display` == llvm's printed text, llvm's disassembly of the bytes ==
//! that text, and `decode(bytes)` == the original value. The only skip is the
//! prescribed `oracle::disabled()` early return. The CPU-only tests (golden
//! words, corpus round trips, illegal forms, decode rejections, op tables,
//! footprints) always run.
//!
//! Struct conventions the corpus follows (CDNA3 ISA §6.7, §7.1, §13.3.7-13.3.8):
//! - `op_sel_hi` of a packed / dot op defaults to `0b111` even with two
//!   sources (the absent source2 bit is 1; every other absent-source bit is 0);
//!   `v_fma_mix*` defaults to 0 (bit set = f16 source).
//! - Every `Dot*` op: `op_sel = 0`, `op_sel_hi = 7`, `neg_lo`/`neg_hi` free; clamp is free.
//! - `v_fma_mix*`: `neg_lo` is the per-source `-src` negate, `neg_hi` must be 0.
//! - Multi-dword register tuples (VGPR, AGPR, SGPR) start on an even register.
//! - `MaiOp::MfmaF64_16x16x4F64`: `blgp` carries the `neg:[..]` bits.
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp, MfmaType, Vop3p, Vop3pOp};
use peacemaker_cdna::inst::{
    classify, decode_all, encode_all, Footprint, Format, RegFile, RegSpan,
};
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::{oracle, Arch, Inst};

const ARCH: Arch = Arch::Gfx942;

// ----------------------------------------------------------------------------
// Constructors and tiny helpers
// ----------------------------------------------------------------------------

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn s(n: u8) -> Src {
    Src::Sgpr(n)
}
fn a(n: u8) -> Src {
    Src::Agpr(n)
}
fn reg(acc: bool, n: u8) -> Src {
    if acc {
        a(n)
    } else {
        v(n)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Packed,
    Mix,
    Dot,
}

fn class(op: Vop3pOp) -> Class {
    use Vop3pOp::*;
    match op {
        FmaMixF32 | FmaMixloF16 | FmaMixhiF16 => Class::Mix,
        Dot2F32F16 | Dot2I32I16 | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4 | Dot8U32U4 => {
            Class::Dot
        }
        _ => Class::Packed,
    }
}

/// Source operands the op reads (2 or 3).
fn nsrc(op: Vop3pOp) -> usize {
    use Vop3pOp::*;
    match op {
        PkFmaF16 | PkFmaF32 | FmaMixF32 | FmaMixloF16 | FmaMixhiF16 | Dot2F32F16 | Dot2I32I16
        | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4 | Dot8U32U4 => 3,
        _ => 2,
    }
}

/// Dword width of every dst/src operand of the op.
fn dw(op: Vop3pOp) -> u8 {
    use Vop3pOp::*;
    match op {
        PkFmaF32 | PkMulF32 | PkAddF32 | PkMovB32 => 2,
        _ => 1,
    }
}

fn hi_default(op: Vop3pOp) -> u8 {
    if class(op) == Class::Mix {
        0
    } else {
        7
    }
}

/// A canonical Vop3p with default modifiers.
fn pk(op: Vop3pOp, dst: Src, srcs: &[Src]) -> Vop3p {
    assert_eq!(srcs.len(), nsrc(op), "{op:?}");
    Vop3p {
        op,
        dst,
        src: [Some(srcs[0]), Some(srcs[1]), srcs.get(2).copied()],
        op_sel: 0,
        op_sel_hi: hi_default(op),
        neg_lo: 0,
        neg_hi: 0,
        clamp: false,
    }
}

/// Set modifiers given as per-source masks of `nsrc` bits; stores them in the
/// canonical struct form (absent source2: hi bit 1, the others 0).
fn modded(mut x: Vop3p, op_sel: u8, hi: u8, neg_lo: u8, neg_hi: u8, clamp: bool) -> Vop3p {
    let n = nsrc(x.op);
    let m = (1u8 << n) - 1;
    x.op_sel = op_sel & m;
    x.op_sel_hi = (hi & m) | if n == 2 { 4 } else { 0 };
    x.neg_lo = neg_lo & m;
    x.neg_hi = neg_hi & m;
    x.clamp = clamp;
    x
}

fn vwords(x: &Vop3p) -> Vec<u32> {
    let mut out = Vec::new();
    x.encode(ARCH, &mut out)
        .unwrap_or_else(|e| panic!("encode {x}: {e}"));
    out
}
fn mwords(x: &Mai) -> Vec<u32> {
    let mut out = Vec::new();
    x.encode(ARCH, &mut out)
        .unwrap_or_else(|e| panic!("encode {x}: {e}"));
    out
}

fn mai(
    op: MaiOp,
    dst: Src,
    a_: Src,
    b: Option<Src>,
    c: Option<Src>,
    cbsz: u8,
    abid: u8,
    blgp: u8,
) -> Mai {
    Mai {
        op,
        dst,
        src_a: a_,
        src_b: b,
        src_c: c,
        cbsz,
        abid,
        blgp,
    }
}

fn mfma(op: MaiOp, dst: Src, a_: Src, b: Src, c: Src, cbsz: u8, abid: u8, blgp: u8) -> Mai {
    mai(op, dst, a_, Some(b), Some(c), cbsz, abid, blgp)
}

/// Deterministic generator for the randomised modifier combinations.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ----------------------------------------------------------------------------
// Independent asm formatting (llvm-mc syntax)
// ----------------------------------------------------------------------------

fn bit_array(mask: u8, n: usize) -> String {
    format!(
        "[{}]",
        (0..n)
            .map(|i| if mask >> i & 1 == 1 { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn asm_p(x: &Vop3p) -> String {
    let w = dw(x.op);
    let n = nsrc(x.op);
    let cls = class(x.op);
    let mut out = format!("{} {}", x.op.name(), x.dst.fmt(w));
    for i in 0..n {
        let src = x.src[i].expect("source");
        let text = src.fmt(w);
        out += &if cls == Class::Mix && x.neg_lo >> i & 1 == 1 {
            if src.is_inline() {
                format!(", neg({text})")
            } else {
                format!(", -{text}")
            }
        } else {
            format!(", {text}")
        };
    }
    let sel = bit_array(x.op_sel, n);
    let hi = bit_array(x.op_sel_hi, n);
    let lo = bit_array(x.neg_lo, n);
    let nh = bit_array(x.neg_hi, n);
    match cls {
        Class::Packed => out += &format!(" op_sel:{sel} op_sel_hi:{hi} neg_lo:{lo} neg_hi:{nh}"),
        Class::Mix => out += &format!(" op_sel:{sel} op_sel_hi:{hi}"),
        Class::Dot => out += &format!(" neg_lo:{lo} neg_hi:{nh}"),
    }
    if x.clamp {
        out += " clamp"
    }
    out
}

fn asm_m(x: &Mai) -> String {
    let name = x.op.name();
    let Some(sh) = x.op.shape() else {
        return format!("{name} {}, {}", x.dst.fmt(1), x.src_a.fmt(1));
    };
    let mods = if x.op == MaiOp::MfmaF64_16x16x4F64 {
        format!(
            "cbsz:{} abid:{} neg:{}",
            x.cbsz,
            x.abid,
            bit_array(x.blgp, 3)
        )
    } else {
        format!("cbsz:{} abid:{} blgp:{}", x.cbsz, x.abid, x.blgp)
    };
    format!(
        "{name} {}, {}, {}, {} {mods}",
        x.dst.fmt(sh.cd_dwords),
        x.src_a.fmt(sh.a_dwords),
        x.src_b.expect("src_b").fmt(sh.b_dwords),
        x.src_c.expect("src_c").fmt(sh.cd_dwords)
    )
}

// ----------------------------------------------------------------------------
// Packed-math corpus
// ----------------------------------------------------------------------------

fn dsts(w: u8) -> [Src; 4] {
    if w == 1 {
        [v(1), v(0), v(200), v(255)]
    } else {
        [v(2), v(0), v(128), v(254)]
    }
}
fn base_srcs(w: u8) -> [Src; 3] {
    if w == 1 {
        [v(2), v(3), v(4)]
    } else {
        [v(4), v(6), v(8)]
    }
}

/// Operand patterns for width `w`: plain VGPRs, VGPR boundaries, one SGPR /
/// special scalar / inline constant per source position, all-inline.
fn patterns(w: u8) -> Vec<[Src; 3]> {
    use InlineFloat::*;
    if w == 1 {
        vec![
            [v(2), v(3), v(4)],
            [v(255), v(254), v(253)],
            [v(0), v(0), v(0)],
            [s(2), v(3), v(4)],
            [v(2), s(101), v(4)],
            [v(2), v(3), s(7)],
            [s(9), s(9), s(9)],
            [Src::Int(0), v(3), v(4)],
            [v(2), Src::Int(-16), v(4)],
            [v(2), v(3), Src::Int(64)],
            [Src::Int(1), Src::Int(-1), Src::Int(2)],
            [Src::Float(One), v(3), Src::Float(NegHalf)],
            [v(2), Src::Float(InvTwoPi), v(4)],
            [Src::Float(NegFour), Src::Float(Four), Src::Float(Two)],
            [Src::VccLo, v(3), v(4)],
            [v(2), Src::VccHi, v(4)],
            [v(2), v(3), Src::ExecLo],
            [Src::ExecHi, v(3), v(4)],
            [Src::M0, v(3), v(4)],
            [v(2), Src::Ttmp(3), v(4)],
            [v(2), v(3), Src::FlatScratchLo],
            [Src::SharedBase, v(3), v(4)],
            [v(2), Src::Vccz, v(4)],
            [v(2), v(3), Src::Execz],
            [Src::Scc, v(3), v(4)],
            [v(2), Src::PrivateLimit, v(4)],
            [s(2), Src::Int(5), s(2)],
            [Src::Float(Half), s(2), Src::Int(-3)],
            [s(2), s(2), Src::Float(Four)],
            [Src::SharedBase, Src::SharedBase, v(4)],
        ]
    } else {
        vec![
            [v(4), v(6), v(8)],
            [v(254), v(252), v(250)],
            [v(0), v(0), v(0)],
            [s(2), v(6), v(8)],
            [v(4), s(100), v(8)],
            [v(4), v(6), s(8)],
            [s(10), s(10), s(10)],
            [Src::VccLo, v(6), v(8)],
            [v(4), Src::ExecLo, v(8)],
            [v(4), v(6), Src::Ttmp(2)],
            [Src::Int(0), v(6), v(8)],
            [v(4), Src::Int(-16), v(8)],
            [v(4), v(6), Src::Int(64)],
            [Src::Int(1), Src::Int(-1), Src::Int(2)],
            [Src::Float(One), v(6), Src::Float(NegHalf)],
            [v(4), Src::Float(InvTwoPi), v(8)],
            [Src::Float(NegFour), Src::Float(Four), Src::Float(Two)],
            [s(2), Src::Int(5), s(2)],
            [Src::Float(Half), s(2), Src::Int(-3)],
            [s(2), s(2), Src::Float(Four)],
        ]
    }
}

fn inline_consts() -> Vec<Src> {
    (-16..=64)
        .map(Src::Int)
        .chain(InlineFloat::ALL.iter().map(|&f| Src::Float(f)))
        .collect()
}

/// Every op x operand pattern x destination boundaries.
fn operand_corpus() -> Vec<Vop3p> {
    let mut out = Vec::new();
    for &op in Vop3pOp::ALL {
        let w = dw(op);
        let n = nsrc(op);
        let d = dsts(w);
        for (j, p) in patterns(w).iter().enumerate() {
            out.push(pk(op, d[j % 4], &p[..n]));
            if j == 0 {
                for &other in &d[1..] {
                    out.push(pk(op, other, &p[..n]))
                }
            }
        }
    }
    out
}

/// Every op x every inline constant in each source position (rotating).
fn inline_corpus() -> Vec<Vop3p> {
    let mut out = Vec::new();
    for &op in Vop3pOp::ALL {
        let w = dw(op);
        let n = nsrc(op);
        for (k, &c) in inline_consts().iter().enumerate() {
            let mut srcs = base_srcs(w);
            srcs[k % n] = c;
            out.push(pk(op, dsts(w)[0], &srcs[..n]));
            if class(op) == Class::Mix {
                // neg on an inline constant is `neg(c)`; a bare `-1` / `-1.0` is the constant -1 with no neg bit.
                out.push(modded(
                    pk(op, dsts(w)[0], &srcs[..n]),
                    0,
                    0,
                    1 << (k % n),
                    0,
                    false,
                ));
                out.push(modded(pk(op, dsts(w)[0], &[c, c, c]), 0, 0, 7, 0, false));
            }
        }
        if class(op) == Class::Mix {
            let consts = inline_consts();
            for (k, &c) in consts.iter().enumerate().step_by(7) {
                let c2 = consts[(k + 11) % consts.len()];
                for m in 0..8u8 {
                    out.push(modded(
                        pk(op, dsts(w)[0], &[c, v(3), c2]),
                        0,
                        0,
                        m,
                        0,
                        false,
                    ));
                    out.push(modded(
                        pk(op, dsts(w)[0], &[v(2), c, c2]),
                        0,
                        0,
                        m,
                        0,
                        false,
                    ));
                    out.push(modded(pk(op, dsts(w)[0], &[s(2), c, c2]), 0, 0, m, 0, true));
                }
            }
        }
    }
    out
}

/// Bit i set where source i can carry a mix negate: registers print `-reg`,
/// inline constants `neg(const)` (a bare `-1`/`-1.0` is the constant -1 with no
/// neg bit); special scalars are left out.
fn negatable_mask(x: &Vop3p) -> u8 {
    (0..3)
        .filter(|&i| {
            matches!(
                x.src[i],
                Some(Src::Vgpr(_) | Src::Sgpr(_) | Src::Int(_) | Src::Float(_))
            )
        })
        .fold(0, |m, i| m | 1 << i)
}

/// Modifier sweeps: every field over all values, cross products, clamp, and
/// seeded random full combinations over the operand patterns.
fn modifier_corpus() -> Vec<Vop3p> {
    let mut out = Vec::new();
    let mut rng = Lcg(0x5eed_0f_3b17);
    for &op in Vop3pOp::ALL {
        let w = dw(op);
        let n = nsrc(op);
        let m = 1u8 << n;
        let full = m - 1;
        let base = pk(op, dsts(w)[0], &base_srcs(w)[..n]);
        let cls = class(op);
        match cls {
            Class::Packed => {
                for clamp in [false, true] {
                    for x in 0..m {
                        out.push(modded(base.clone(), x, full, 0, 0, clamp));
                        out.push(modded(base.clone(), 0, x, 0, 0, clamp));
                        out.push(modded(base.clone(), 0, full, x, 0, clamp));
                        out.push(modded(base.clone(), 0, full, 0, x, clamp));
                    }
                }
                for x in 0..m {
                    for y in 0..m {
                        out.push(modded(base.clone(), x, y, 0, 0, false));
                        out.push(modded(base.clone(), 0, full, x, y, false));
                        out.push(modded(base.clone(), x, full, y, 0, true));
                    }
                }
            }
            Class::Mix => {
                for clamp in [false, true] {
                    for x in 0..m {
                        out.push(modded(base.clone(), x, 0, 0, 0, clamp));
                        out.push(modded(base.clone(), 0, x, 0, 0, clamp));
                        out.push(modded(base.clone(), 0, 0, x, 0, clamp));
                    }
                }
                for x in 0..m {
                    for y in 0..m {
                        out.push(modded(base.clone(), x, y, 0, 0, false));
                        out.push(modded(base.clone(), 0, x, y, 0, true));
                        out.push(modded(base.clone(), x, 0, y, 0, false));
                    }
                }
            }
            Class::Dot => {
                for clamp in [false, true] {
                    for x in 0..m {
                        out.push(modded(base.clone(), 0, full, x, 0, clamp));
                        out.push(modded(base.clone(), 0, full, 0, x, clamp));
                    }
                }
                for x in 0..m {
                    for y in 0..m {
                        out.push(modded(base.clone(), 0, full, x, y, false))
                    }
                }
            }
        }
        // Seeded random combinations across operand patterns and destinations.
        let pats = patterns(w);
        let d = dsts(w);
        for _ in 0..96 {
            let p = &pats[rng.below(pats.len() as u64) as usize];
            let dst = d[rng.below(4) as usize];
            let x = pk(op, dst, &p[..n]);
            let r = |rng: &mut Lcg| rng.below(u64::from(m)) as u8;
            let clamp = rng.below(2) == 1;
            let (sel, hi, lo, nh) = (r(&mut rng), r(&mut rng), r(&mut rng), r(&mut rng));
            out.push(match cls {
                Class::Packed => modded(x, sel, hi, lo, nh, clamp),
                Class::Mix => {
                    let lo = lo & negatable_mask(&x);
                    modded(x, sel, hi, lo, 0, clamp)
                }
                Class::Dot => modded(x, 0, full, lo, nh, clamp),
            });
        }
    }
    out
}

fn vop3p_corpus() -> Vec<Vop3p> {
    let mut all = operand_corpus();
    all.extend(inline_corpus());
    all.extend(modifier_corpus());
    all
}

// ----------------------------------------------------------------------------
// MAI corpus
// ----------------------------------------------------------------------------

fn mfma_ops() -> Vec<MaiOp> {
    MaiOp::ALL
        .iter()
        .copied()
        .filter(|op| op.shape().is_some())
        .collect()
}

fn cd_bases(cd: u8) -> [u8; 4] {
    [0, cd, 128, 255 - cd + 1]
}
fn a_bases(dw_: u8) -> [u8; 4] {
    if dw_ == 1 {
        [0, 3, 101, 255]
    } else {
        [0, 2, 100, 254]
    }
}
fn b_bases(dw_: u8) -> [u8; 4] {
    if dw_ == 1 {
        [1, 4, 102, 254]
    } else {
        [2, 6, 104, 252]
    }
}

/// Inline C constants for `op`. llvm-mc refuses `0.15915494` (1/2pi) as the f64 C operand of
/// `v_mfma_f64_16x16x4_f64` (its f64 spelling is a 17-digit decimal), so that one is left out there.
fn c_inline(op: MaiOp) -> Vec<Src> {
    let mut c = vec![
        Src::Int(0),
        Src::Int(1),
        Src::Int(-1),
        Src::Int(-16),
        Src::Int(64),
    ];
    c.extend(
        InlineFloat::ALL
            .iter()
            .filter(|&&f| !(op == MaiOp::MfmaF64_16x16x4F64 && f == InlineFloat::InvTwoPi))
            .map(|&f| Src::Float(f)),
    );
    c
}

/// Every MFMA x {AGPR, VGPR} C/D x {V/A} A/B x register boundaries x
/// {C == D, C elsewhere, inline C}.
fn mfma_operand_corpus() -> Vec<Mai> {
    let mut out = Vec::new();
    for op in mfma_ops() {
        let sh = op.shape().unwrap();
        let (cd, ad, bd) = (sh.cd_dwords, sh.a_dwords, sh.b_dwords);
        let (cb, ab, bb) = (cd_bases(cd), a_bases(ad), b_bases(bd));
        for accd in [true, false] {
            for (aa, ba) in [(false, false), (true, false), (false, true), (true, true)] {
                for i in 0..4 {
                    let d = reg(accd, cb[i]);
                    out.push(mfma(op, d, reg(aa, ab[i]), reg(ba, bb[i]), d, 0, 0, 0));
                    out.push(mfma(
                        op,
                        d,
                        reg(aa, ab[i]),
                        reg(ba, bb[i]),
                        reg(accd, cb[(i + 1) % 4]),
                        0,
                        0,
                        0,
                    ));
                }
            }
            for (aa, ba) in [(false, false), (true, true)] {
                for c in c_inline(op) {
                    out.push(mfma(
                        op,
                        reg(accd, cb[1]),
                        reg(aa, ab[1]),
                        reg(ba, bb[1]),
                        c,
                        0,
                        0,
                        0,
                    ))
                }
            }
            for k in -16..=64i8 {
                out.push(mfma(
                    op,
                    reg(accd, cb[0]),
                    v(ab[0]),
                    v(bb[0]),
                    Src::Int(k),
                    0,
                    0,
                    0,
                ))
            }
            // Even C/D bases that are not width multiples (base 2 with width 16, base 4 with width 8, ...).
            for base in [2u8, 4, 6, 10, 18, 34, 66, 130, 242] {
                if u16::from(base) + u16::from(cd) > 256 {
                    continue;
                }
                let d = reg(accd, base);
                out.push(mfma(op, d, v(2), v(4), d, 0, 0, 0));
                out.push(mfma(op, d, a(6), v(8), Src::Int(0), 0, 0, 0));
            }
        }
    }
    out
}

/// cbsz x abid sweep (blgp rotating), blgp sweep, and the full
/// cbsz x abid x blgp cross for one f16 shape and the f64 shape.
fn mfma_modifier_corpus() -> Vec<Mai> {
    let mut out = Vec::new();
    for op in mfma_ops() {
        let sh = op.shape().unwrap();
        let (cd, ad, bd) = (sh.cd_dwords, sh.a_dwords, sh.b_dwords);
        let (d, x, y) = (a(0), v(ad.max(2)), v(bd.max(2) * 2));
        for cbsz in 0..8u8 {
            for abid in 0..16u8 {
                out.push(mfma(op, d, x, y, d, cbsz, abid, (3 * cbsz + abid) % 8))
            }
        }
        for blgp in 0..8u8 {
            out.push(mfma(op, d, x, y, d, 0, 0, blgp))
        }
        let dv = v(cd);
        for (cbsz, abid, blgp) in [(7, 15, 7), (1, 1, 3), (0, 15, 0), (7, 0, 7), (4, 15, 5)] {
            out.push(mfma(op, dv, a(2), a(4), dv, cbsz, abid, blgp));
            out.push(mfma(op, a(cd), x, a(4), Src::Int(0), cbsz, abid, blgp));
        }
        if matches!(op, MaiOp::MfmaF32_32x32x8F16 | MaiOp::MfmaF64_16x16x4F64) {
            for cbsz in 0..8u8 {
                for abid in 0..16u8 {
                    for blgp in 0..8u8 {
                        out.push(mfma(op, d, x, y, d, cbsz, abid, blgp))
                    }
                }
            }
        }
    }
    out
}

fn move_corpus() -> Vec<Mai> {
    let mut out = Vec::new();
    for dst in [0u8, 1, 127, 255] {
        for src in [0u8, 1, 128, 255] {
            out.push(mai(
                MaiOp::AccvgprReadB32,
                v(dst),
                a(src),
                None,
                None,
                0,
                0,
                0,
            ))
        }
        let mut srcs = vec![
            v(0),
            v(2),
            v(255),
            s(0),
            s(2),
            s(101),
            Src::VccLo,
            Src::VccHi,
            Src::ExecLo,
            Src::ExecHi,
            Src::M0,
            Src::Ttmp(0),
            Src::Ttmp(15),
            Src::FlatScratchLo,
            Src::FlatScratchHi,
            Src::XnackMaskLo,
            Src::XnackMaskHi,
            Src::SharedBase,
            Src::SharedLimit,
            Src::PrivateBase,
            Src::PrivateLimit,
            Src::Vccz,
            Src::Execz,
            Src::Scc,
        ];
        // Every SGPR and TTMP once, so no scalar alias is left unchecked.
        if dst == 1 {
            srcs.extend((0..=101u8).map(s));
            srcs.extend((0..16u8).map(Src::Ttmp));
        }
        srcs.extend(inline_consts());
        for src in srcs {
            out.push(mai(
                MaiOp::AccvgprWriteB32,
                a(dst),
                src,
                None,
                None,
                0,
                0,
                0,
            ))
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Oracle round trips
// ----------------------------------------------------------------------------

#[test]
fn oracle_vop3p_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_vop3p: ROCm oracle disabled");
        return;
    }
    let cases = vop3p_corpus();
    assert!(cases.len() > 6_000, "{}", cases.len());
    let lines: Vec<String> = cases.iter().map(asm_p).collect();
    let assembled = oracle::mc_encode(ARCH, &lines).unwrap();
    let mut byte_vecs = Vec::with_capacity(cases.len());
    for ((x, line), (mc_text, mc_bytes)) in cases.iter().zip(&lines).zip(&assembled) {
        let ours = vwords(x);
        assert_eq!(&oracle::bytes(&ours), mc_bytes, "bytes: {line}");
        assert_eq!(x.to_string(), *mc_text, "Display vs llvm text: {line}");
        let (decoded, len) =
            Vop3p::decode(ARCH, &ours).unwrap_or_else(|e| panic!("decode {line}: {e}"));
        assert_eq!((len, &decoded), (2, x), "decode identity: {line}");
        let (inst, len) = Inst::decode(ARCH, &ours).unwrap();
        assert_eq!(
            (len, &inst),
            (2, &Inst::Vop3p(x.clone())),
            "Inst::decode: {line}"
        );
        assert_eq!(inst.words(ARCH).unwrap(), ours);
        assert_eq!(inst.to_string(), *mc_text);
        assert_eq!(classify(ours[0]), Some(Format::Vop3p), "{line}");
        byte_vecs.push(mc_bytes.clone());
    }
    let disassembled = oracle::mc_disassemble(ARCH, &byte_vecs).unwrap();
    for ((line, (mc_text, _)), dis) in lines.iter().zip(&assembled).zip(&disassembled) {
        assert_eq!(dis, mc_text, "llvm disassembly: {line}")
    }
}

fn check_mai_round_trip(cases: &[Mai]) {
    let lines: Vec<String> = cases.iter().map(asm_m).collect();
    let assembled = oracle::mc_encode(ARCH, &lines).unwrap();
    let mut byte_vecs = Vec::with_capacity(cases.len());
    for ((x, line), (mc_text, mc_bytes)) in cases.iter().zip(&lines).zip(&assembled) {
        let ours = mwords(x);
        assert_eq!(&oracle::bytes(&ours), mc_bytes, "bytes: {line}");
        assert_eq!(x.to_string(), *mc_text, "Display vs llvm text: {line}");
        let (decoded, len) =
            Mai::decode(ARCH, &ours).unwrap_or_else(|e| panic!("decode {line}: {e}"));
        assert_eq!((len, &decoded), (2, x), "decode identity: {line}");
        let (inst, len) = Inst::decode(ARCH, &ours).unwrap();
        assert_eq!(
            (len, &inst),
            (2, &Inst::Mai(x.clone())),
            "Inst::decode: {line}"
        );
        assert_eq!(inst.words(ARCH).unwrap(), ours);
        assert_eq!(inst.to_string(), *mc_text);
        assert_eq!(classify(ours[0]), Some(Format::Mai), "{line}");
        byte_vecs.push(mc_bytes.clone());
    }
    let disassembled = oracle::mc_disassemble(ARCH, &byte_vecs).unwrap();
    for ((line, (mc_text, _)), dis) in lines.iter().zip(&assembled).zip(&disassembled) {
        assert_eq!(dis, mc_text, "llvm disassembly: {line}")
    }
}

#[test]
fn oracle_mfma_operand_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_vop3p: ROCm oracle disabled");
        return;
    }
    let cases = mfma_operand_corpus();
    assert!(cases.len() > 3_000, "{}", cases.len());
    check_mai_round_trip(&cases);
}

#[test]
fn oracle_mfma_modifier_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_vop3p: ROCm oracle disabled");
        return;
    }
    let cases = mfma_modifier_corpus();
    assert!(cases.len() > 2_000, "{}", cases.len());
    check_mai_round_trip(&cases);
}

#[test]
fn oracle_accvgpr_moves_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_vop3p: ROCm oracle disabled");
        return;
    }
    let cases = move_corpus();
    assert!(cases.len() > 400, "{}", cases.len());
    check_mai_round_trip(&cases);
}

/// Forms llvm-mc refuses; each must also fail to encode here.
fn oracle_rejected() -> Vec<(String, Result<(), String>)> {
    let mut out = Vec::new();
    let enc_p = |x: Vop3p| x.encode(ARCH, &mut Vec::new());
    let enc_m = |x: Mai| x.encode(ARCH, &mut Vec::new());
    // Packed / dot / mix.
    out.push((
        "v_pk_add_f32 v[3:4], v[4:5], v[6:7]".into(),
        enc_p(pk(Vop3pOp::PkAddF32, v(3), &[v(4), v(6)])),
    ));
    out.push((
        "v_pk_add_f32 v[2:3], v[5:6], v[6:7]".into(),
        enc_p(pk(Vop3pOp::PkAddF32, v(2), &[v(5), v(6)])),
    ));
    out.push((
        "v_pk_add_f32 v[2:3], s[3:4], v[6:7]".into(),
        enc_p(pk(Vop3pOp::PkAddF32, v(2), &[s(3), v(6)])),
    ));
    out.push((
        "v_pk_add_f16 a1, v2, v3".into(),
        enc_p(pk(Vop3pOp::PkAddF16, a(1), &[v(2), v(3)])),
    ));
    out.push((
        "v_pk_add_f16 s1, v2, v3".into(),
        enc_p(pk(Vop3pOp::PkAddF16, s(1), &[v(2), v(3)])),
    ));
    out.push((
        "v_pk_add_f16 v1, a2, v3".into(),
        enc_p(pk(Vop3pOp::PkAddF16, v(1), &[a(2), v(3)])),
    ));
    out.push((
        "v_pk_add_f16 v1, 0x12345, v3".into(),
        enc_p(pk(Vop3pOp::PkAddF16, v(1), &[Src::Lit(0x12345), v(3)])),
    ));
    out.push((
        "v_pk_add_f16 v1, 65, v3".into(),
        enc_p(pk(Vop3pOp::PkAddF16, v(1), &[Src::Int(65), v(3)])),
    ));
    out.push((
        "v_pk_add_f16 v1, v2, -17".into(),
        enc_p(pk(Vop3pOp::PkAddF16, v(1), &[v(2), Src::Int(-17)])),
    ));
    out.push((
        "v_pk_add_f16 v1, v2, v3, v4".into(),
        enc_p(Vop3p {
            src: [Some(v(2)), Some(v(3)), Some(v(4))],
            ..pk(Vop3pOp::PkAddF16, v(1), &[v(2), v(3)])
        }),
    ));
    out.push((
        "v_pk_fma_f16 v1, v2, v3".into(),
        enc_p(Vop3p {
            src: [Some(v(2)), Some(v(3)), None],
            ..pk(Vop3pOp::PkFmaF16, v(1), &[v(2), v(3), v(4)])
        }),
    ));
    out.push((
        "v_dot4_i32_i8 v1, v2, v3, v4 op_sel:[1,0,0]".into(),
        enc_p(Vop3p {
            op_sel: 1,
            ..pk(Vop3pOp::Dot4I32I8, v(1), &[v(2), v(3), v(4)])
        }),
    ));
    out.push((
        "v_dot2_f32_f16 v1, v2, v3, v4 op_sel_hi:[0,0,0]".into(),
        enc_p(Vop3p {
            op_sel_hi: 0,
            ..pk(Vop3pOp::Dot2F32F16, v(1), &[v(2), v(3), v(4)])
        }),
    ));
    // Constant bus: two distinct scalars (inline constants and a repeated scalar are free).
    for (asm, x) in [
        (
            "v_pk_add_f16 v1, s2, s3",
            pk(Vop3pOp::PkAddF16, v(1), &[s(2), s(3)]),
        ),
        (
            "v_pk_add_f16 v1, s2, vcc_lo",
            pk(Vop3pOp::PkAddF16, v(1), &[s(2), Src::VccLo]),
        ),
        (
            "v_pk_add_f16 v1, vcc_lo, vcc_hi",
            pk(Vop3pOp::PkAddF16, v(1), &[Src::VccLo, Src::VccHi]),
        ),
        (
            "v_pk_add_f16 v1, m0, s2",
            pk(Vop3pOp::PkAddF16, v(1), &[Src::M0, s(2)]),
        ),
        (
            "v_pk_add_f16 v1, s2, src_scc",
            pk(Vop3pOp::PkAddF16, v(1), &[s(2), Src::Scc]),
        ),
        (
            "v_pk_add_f16 v1, s2, src_vccz",
            pk(Vop3pOp::PkAddF16, v(1), &[s(2), Src::Vccz]),
        ),
        (
            "v_pk_add_f16 v1, src_scc, src_vccz",
            pk(Vop3pOp::PkAddF16, v(1), &[Src::Scc, Src::Vccz]),
        ),
        (
            "v_pk_add_f16 v1, s2, src_shared_base",
            pk(Vop3pOp::PkAddF16, v(1), &[s(2), Src::SharedBase]),
        ),
        (
            "v_pk_add_f16 v1, src_shared_base, src_shared_limit",
            pk(
                Vop3pOp::PkAddF16,
                v(1),
                &[Src::SharedBase, Src::SharedLimit],
            ),
        ),
        (
            "v_pk_add_f16 v1, ttmp3, ttmp4",
            pk(Vop3pOp::PkAddF16, v(1), &[Src::Ttmp(3), Src::Ttmp(4)]),
        ),
        (
            "v_pk_fma_f16 v1, s2, v3, s3",
            pk(Vop3pOp::PkFmaF16, v(1), &[s(2), v(3), s(3)]),
        ),
        (
            "v_pk_fma_f16 v1, s2, s2, s3",
            pk(Vop3pOp::PkFmaF16, v(1), &[s(2), s(2), s(3)]),
        ),
        (
            "v_pk_fma_f16 v1, exec_lo, vcc_lo, v1",
            pk(Vop3pOp::PkFmaF16, v(1), &[Src::ExecLo, Src::VccLo, v(1)]),
        ),
        (
            "v_pk_add_f32 v[2:3], s[2:3], s[4:5]",
            pk(Vop3pOp::PkAddF32, v(2), &[s(2), s(4)]),
        ),
        (
            "v_pk_add_f32 v[2:3], s[2:3], vcc",
            pk(Vop3pOp::PkAddF32, v(2), &[s(2), Src::VccLo]),
        ),
        (
            "v_pk_add_f32 v[2:3], vcc, exec",
            pk(Vop3pOp::PkAddF32, v(2), &[Src::VccLo, Src::ExecLo]),
        ),
        (
            "v_pk_fma_f32 v[2:3], s[2:3], s[2:3], s[4:5]",
            pk(Vop3pOp::PkFmaF32, v(2), &[s(2), s(2), s(4)]),
        ),
        (
            "v_pk_mov_b32 v[2:3], s[2:3], s[4:5]",
            pk(Vop3pOp::PkMovB32, v(2), &[s(2), s(4)]),
        ),
        (
            "v_fma_mix_f32 v1, s2, s3, v4",
            pk(Vop3pOp::FmaMixF32, v(1), &[s(2), s(3), v(4)]),
        ),
        (
            "v_fma_mix_f32 v1, s2, v3, flat_scratch_lo",
            pk(Vop3pOp::FmaMixF32, v(1), &[s(2), v(3), Src::FlatScratchLo]),
        ),
        (
            "v_dot2_f32_f16 v1, s2, s3, v4",
            pk(Vop3pOp::Dot2F32F16, v(1), &[s(2), s(3), v(4)]),
        ),
        (
            "v_dot4_i32_i8 v1, s2, s3, 5",
            pk(Vop3pOp::Dot4I32I8, v(1), &[s(2), s(3), Src::Int(5)]),
        ),
    ] {
        out.push((asm.into(), enc_p(x)))
    }
    // MFMA.
    let f = MaiOp::MfmaF32_16x16x32Fp8Fp8;
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], v[0:3]".into(),
        enc_m(mfma(f, a(0), v(2), v(4), v(0), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[2:3], v[4:5], a[0:3]".into(),
        enc_m(mfma(f, v(0), v(2), v(4), a(0), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], s[2:3], v[4:5], a[0:3]".into(),
        enc_m(mfma(f, a(0), s(2), v(4), a(0), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], s0".into(),
        enc_m(mfma(f, a(0), v(2), v(4), s(0), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], 0x41".into(),
        enc_m(mfma(f, a(0), v(2), v(4), Src::Lit(0x41), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], 65".into(),
        enc_m(mfma(f, a(0), v(2), v(4), Src::Int(65), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], vcc".into(),
        enc_m(mfma(f, a(0), v(2), v(4), Src::VccLo, 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], m0".into(),
        enc_m(mfma(f, a(0), v(2), v(4), Src::M0, 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[1:4], v[2:3], v[4:5], a[1:4]".into(),
        enc_m(mfma(f, a(1), v(2), v(4), a(1), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[3:4], v[4:5], a[0:3]".into(),
        enc_m(mfma(f, a(0), v(3), v(4), a(0), 0, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] cbsz:8".into(),
        enc_m(mfma(f, a(0), v(2), v(4), a(0), 8, 0, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] abid:16".into(),
        enc_m(mfma(f, a(0), v(2), v(4), a(0), 0, 16, 0)),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] blgp:8".into(),
        enc_m(mfma(f, a(0), v(2), v(4), a(0), 0, 0, 8)),
    ));
    out.push((
        "v_mfma_f64_16x16x4_f64 a[0:7], v[2:3], v[4:5], a[0:7] blgp:1".into(),
        enc_m(mfma(
            MaiOp::MfmaF64_16x16x4F64,
            a(0),
            v(2),
            v(4),
            a(0),
            0,
            0,
            8,
        )),
    ));
    out.push((
        "v_mfma_f32_16x16x32_fp8_fp8 a[254:257], v[2:3], v[4:5], a[254:257]".into(),
        enc_m(mfma(f, a(254), v(2), v(4), a(254), 0, 0, 0)),
    ));
    // Moves.
    out.push((
        "v_accvgpr_read_b32 v1, v2".into(),
        enc_m(mai(MaiOp::AccvgprReadB32, v(1), v(2), None, None, 0, 0, 0)),
    ));
    out.push((
        "v_accvgpr_read_b32 s1, a2".into(),
        enc_m(mai(MaiOp::AccvgprReadB32, s(1), a(2), None, None, 0, 0, 0)),
    ));
    out.push((
        "v_accvgpr_read_b32 v1, 5".into(),
        enc_m(mai(
            MaiOp::AccvgprReadB32,
            v(1),
            Src::Int(5),
            None,
            None,
            0,
            0,
            0,
        )),
    ));
    out.push((
        "v_accvgpr_write_b32 v1, v2".into(),
        enc_m(mai(MaiOp::AccvgprWriteB32, v(1), v(2), None, None, 0, 0, 0)),
    ));
    out.push((
        "v_accvgpr_write_b32 a1, a2".into(),
        enc_m(mai(MaiOp::AccvgprWriteB32, a(1), a(2), None, None, 0, 0, 0)),
    ));
    out.push((
        "v_accvgpr_write_b32 a1, 0x12345".into(),
        enc_m(mai(
            MaiOp::AccvgprWriteB32,
            a(1),
            Src::Lit(0x12345),
            None,
            None,
            0,
            0,
            0,
        )),
    ));
    out.push((
        "v_accvgpr_write_b32 a1, 65".into(),
        enc_m(mai(
            MaiOp::AccvgprWriteB32,
            a(1),
            Src::Int(65),
            None,
            None,
            0,
            0,
            0,
        )),
    ));
    out.push((
        "v_accvgpr_write_b32 a1, -17".into(),
        enc_m(mai(
            MaiOp::AccvgprWriteB32,
            a(1),
            Src::Int(-17),
            None,
            None,
            0,
            0,
            0,
        )),
    ));
    out
}

#[test]
fn oracle_rejects_what_we_reject() {
    if oracle::disabled() {
        eprintln!("enc_vop3p: ROCm oracle disabled");
        return;
    }
    for (asm, ours) in oracle_rejected() {
        assert!(ours.is_err(), "encoder accepted `{asm}`");
        assert!(
            oracle::mc_encode(ARCH, &[asm.clone()]).is_err(),
            "llvm-mc accepted `{asm}`"
        );
    }
}

// ----------------------------------------------------------------------------
// CPU-only: golden words from llvm-mc 2026 output, round trips, op tables
// ----------------------------------------------------------------------------

/// `llvm-mc -mcpu=gfx942 -show-encoding` words and printed text.
fn golden() -> Vec<(Inst, [u32; 2], &'static str)> {
    use MaiOp::*;
    use Vop3pOp::*;
    let p = |x: Vop3p| Inst::Vop3p(x);
    let m = |x: Mai| Inst::Mai(x);
    vec![
        (p(pk(PkFmaF16, v(1), &[v(2), v(3), v(4)])), [0xd38e_4001, 0x1c12_0702], "v_pk_fma_f16 v1, v2, v3, v4"),
        (p(Vop3p { op_sel: 0b101, op_sel_hi: 0b010, neg_lo: 0b101, neg_hi: 0b110, clamp: true, ..pk(PkFmaF16, v(1), &[v(2), v(3), v(4)]) }), [0xd38e_ae01, 0xb412_0702],
            "v_pk_fma_f16 v1, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,1] neg_hi:[0,1,1] clamp"),
        (p(Vop3p { op_sel: 0b11, op_sel_hi: 0b100, neg_lo: 0b11, neg_hi: 0b01, ..pk(PkAddF16, v(1), &[v(2), v(3)]) }), [0xd38f_5901, 0x6002_0702],
            "v_pk_add_f16 v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0] neg_lo:[1,1] neg_hi:[1,0]"),
        (p(pk(PkAddF16, v(1), &[Src::Float(InlineFloat::One), s(3)])), [0xd38f_4001, 0x1800_06f2], "v_pk_add_f16 v1, 1.0, s3"),
        (p(Vop3p { op_sel: 0b001, op_sel_hi: 0b011, ..pk(FmaMixF32, v(1), &[v(2), v(3), v(4)]) }), [0xd3a0_0801, 0x1c12_0702], "v_fma_mix_f32 v1, v2, v3, v4 op_sel:[1,0,0] op_sel_hi:[1,1,0]"),
        (p(pk(Dot2F32F16, v(1), &[v(2), v(3), v(4)])), [0xd3a3_4001, 0x1c12_0702], "v_dot2_f32_f16 v1, v2, v3, v4"),
        (p(pk(PkFmaF32, v(2), &[v(4), v(6), v(8)])), [0xd3b0_4002, 0x1c22_0d04], "v_pk_fma_f32 v[2:3], v[4:5], v[6:7], v[8:9]"),
        (p(pk(PkMovB32, v(2), &[s(4), Src::Int(0)])), [0xd3b3_4002, 0x1801_0004], "v_pk_mov_b32 v[2:3], s[4:5], 0"),
        (m(mai(AccvgprReadB32, v(1), a(2), None, None, 0, 0, 0)), [0xd3d8_4001, 0x1800_0102], "v_accvgpr_read_b32 v1, a2"),
        (m(mai(AccvgprWriteB32, a(1), Src::Int(5), None, None, 0, 0, 0)), [0xd3d9_4001, 0x1800_0085], "v_accvgpr_write_b32 a1, 5"),
        (m(mfma(MfmaF32_32x32x8F16, a(0), v(2), v(4), a(0), 0, 0, 0)), [0xd3cc_8000, 0x0402_0902], "v_mfma_f32_32x32x8_f16 a[0:15], v[2:3], v[4:5], a[0:15]"),
        (m(mfma(MfmaF32_32x32x8F16, v(0), v(2), v(4), v(0), 0, 0, 0)), [0xd3cc_0000, 0x0402_0902], "v_mfma_f32_32x32x8_f16 v[0:15], v[2:3], v[4:5], v[0:15]"),
        (m(mfma(MfmaF32_32x32x8F16, a(0), a(2), a(4), a(0), 0, 0, 0)), [0xd3cc_8000, 0x1c02_0902], "v_mfma_f32_32x32x8_f16 a[0:15], a[2:3], a[4:5], a[0:15]"),
        (m(mfma(MfmaF32_32x32x8F16, a(0), v(2), v(4), Src::Int(0), 0, 0, 0)), [0xd3cc_8000, 0x0202_0902], "v_mfma_f32_32x32x8_f16 a[0:15], v[2:3], v[4:5], 0"),
        (m(mfma(MfmaF32_32x32x8F16, a(0), v(2), v(4), a(0), 1, 1, 3)), [0xd3cc_8900, 0x6402_0902], "v_mfma_f32_32x32x8_f16 a[0:15], v[2:3], v[4:5], a[0:15] cbsz:1 abid:1 blgp:3"),
        (m(mfma(MfmaF32_16x16x32Fp8Fp8, a(0), v(2), v(4), Src::Float(InlineFloat::One), 0, 0, 0)), [0xd3f3_8000, 0x03ca_0902], "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], 1.0"),
        (m(mfma(MfmaF64_16x16x4F64, a(0), v(2), v(4), a(0), 0, 0, 1)), [0xd3ee_8000, 0x2402_0902], "v_mfma_f64_16x16x4_f64 a[0:7], v[2:3], v[4:5], a[0:7] neg:[1,0,0]"),
        (m(mfma(MfmaF32_16x16x4F32, a(0), v(2), v(3), a(0), 0, 0, 0)), [0xd3c5_8000, 0x0402_0702], "v_mfma_f32_16x16x4_f32 a[0:3], v2, v3, a[0:3]"),
        (m(mfma(MfmaF32_16x16x32Fp8Fp8, a(0), v(2), v(4), a(0), 7, 15, 7)), [0xd3f3_ff00, 0xe402_0902],
            "v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3] cbsz:7 abid:15 blgp:7"),
        // v_fma_mix: neg_lo on a register is `-reg`; on an inline constant `neg(c)`; a bare -1.0 is the constant with no neg bit.
        (p(Vop3p { neg_lo: 0b001, ..pk(FmaMixF32, v(1), &[Src::Float(InlineFloat::One), v(3), v(4)]) }), [0xd3a0_0001, 0x2412_06f2], "v_fma_mix_f32 v1, neg(1.0), v3, v4"),
        (p(Vop3p { neg_lo: 0b001, ..pk(FmaMixF32, v(1), &[Src::Int(-1), v(3), v(4)]) }), [0xd3a0_0001, 0x2412_06c1], "v_fma_mix_f32 v1, neg(-1), v3, v4"),
        (p(pk(FmaMixF32, v(1), &[Src::Int(-1), v(3), v(4)])), [0xd3a0_0001, 0x0412_06c1], "v_fma_mix_f32 v1, -1, v3, v4"),
        (p(pk(FmaMixF32, v(1), &[Src::Float(InlineFloat::NegOne), v(3), v(4)])), [0xd3a0_0001, 0x0412_06f3], "v_fma_mix_f32 v1, -1.0, v3, v4"),
        (p(Vop3p { neg_lo: 0b100, ..pk(FmaMixF32, v(1), &[v(2), v(3), Src::Float(InlineFloat::NegFour)]) }), [0xd3a0_0001, 0x83de_0702], "v_fma_mix_f32 v1, v2, v3, neg(-4.0)"),
        (p(Vop3p { neg_lo: 0b001, ..pk(FmaMixF32, v(1), &[s(2), v(3), v(4)]) }), [0xd3a0_0001, 0x2412_0602], "v_fma_mix_f32 v1, -s2, v3, v4"),
    ]
}

#[test]
fn golden_words_text_decode() {
    for (inst, words, text) in golden() {
        assert_eq!(inst.words(ARCH).unwrap(), words, "{text}");
        assert_eq!(inst.to_string(), text);
        let (decoded, len) = Inst::decode(ARCH, &words).unwrap();
        assert_eq!((len, &decoded), (2, &inst), "{text}");
        assert_eq!(classify(words[0]), Some(inst.format()));
    }
}

#[test]
fn cpu_round_trip_whole_corpus() {
    let mut insts: Vec<Inst> = vop3p_corpus().into_iter().map(Inst::Vop3p).collect();
    insts.extend(mfma_operand_corpus().into_iter().map(Inst::Mai));
    insts.extend(mfma_modifier_corpus().into_iter().map(Inst::Mai));
    insts.extend(move_corpus().into_iter().map(Inst::Mai));
    let words = encode_all(ARCH, &insts).unwrap();
    assert_eq!(words.len(), insts.len() * 2);
    let decoded = decode_all(ARCH, &words).unwrap();
    assert_eq!(decoded.len(), insts.len());
    for (i, ((at, got), want)) in decoded.iter().zip(&insts).enumerate() {
        assert_eq!((*at, got), (i * 2, want));
    }
}

#[test]
fn op_tables() {
    assert_eq!(Vop3pOp::ALL.len(), 26);
    assert_eq!(MaiOp::ALL.len(), 19);
    let mut seen = std::collections::HashSet::new();
    for &op in Vop3pOp::ALL {
        let code = op.opcode(ARCH).unwrap();
        assert!(
            code < 0x40,
            "{op:?}: VOP3P opcode {code:#x} would classify as MAI"
        );
        assert!(seen.insert(code), "{op:?}: duplicate opcode");
        assert_eq!(Vop3pOp::from_opcode(ARCH, code), Some(op));
        assert!(op.name().starts_with("v_"));
        let n = nsrc(op);
        let w = dw(op);
        assert_eq!(op.widths(), (w, w, w, if n == 3 { w } else { 0 }), "{op:?}");
    }
    let mut seen = std::collections::HashSet::new();
    for &op in MaiOp::ALL {
        let code = op.opcode(ARCH).unwrap();
        assert!(
            code >= 0x40,
            "{op:?}: MAI opcode {code:#x} would classify as VOP3P"
        );
        assert!(seen.insert(code), "{op:?}: duplicate opcode");
        assert_eq!(MaiOp::from_opcode(ARCH, code), Some(op));
    }
    let names: std::collections::HashSet<_> = Vop3pOp::ALL
        .iter()
        .map(|o| o.name())
        .chain(MaiOp::ALL.iter().map(|o| o.name()))
        .collect();
    assert_eq!(names.len(), 26 + 19);
    for code in 0..0x80u8 {
        assert_eq!(
            Vop3pOp::from_opcode(ARCH, code).is_some(),
            Vop3pOp::ALL.iter().any(|o| o.opcode(ARCH) == Some(code))
        );
        assert_eq!(
            MaiOp::from_opcode(ARCH, code).is_some(),
            MaiOp::ALL.iter().any(|o| o.opcode(ARCH) == Some(code))
        );
    }
}

fn bits_of(t: MfmaType) -> u32 {
    match t {
        MfmaType::F32 | MfmaType::I32 => 32,
        MfmaType::F16 | MfmaType::Bf16 => 16,
        MfmaType::Fp8 | MfmaType::Bf8 | MfmaType::I8 => 8,
        MfmaType::F64 => 64,
    }
}

fn type_name(t: MfmaType) -> &'static str {
    match t {
        MfmaType::F32 => "f32",
        MfmaType::I32 => "i32",
        MfmaType::F16 => "f16",
        MfmaType::Bf16 => "bf16",
        MfmaType::Fp8 => "fp8",
        MfmaType::Bf8 => "bf8",
        MfmaType::I8 => "i8",
        MfmaType::F64 => "f64",
    }
}

#[test]
fn mfma_shapes_agree_with_names_and_lane_layout() {
    assert!(MaiOp::AccvgprReadB32.shape().is_none() && MaiOp::AccvgprWriteB32.shape().is_none());
    for op in mfma_ops() {
        let sh = op.shape().unwrap();
        // 8-bit float MFMAs name both operand types (`fp8_fp8`, `bf8_fp8`, ...); every other op names one.
        let eight_bit_float = matches!(sh.a, MfmaType::Fp8 | MfmaType::Bf8);
        let suffix = if eight_bit_float {
            format!("{}_{}", type_name(sh.a), type_name(sh.b))
        } else {
            assert_eq!(sh.a, sh.b);
            type_name(sh.a).to_string()
        };
        assert_eq!(
            op.name(),
            format!(
                "v_mfma_{}_{}x{}x{}_{suffix}",
                type_name(sh.cd),
                sh.m,
                sh.n,
                sh.k
            ),
            "{op:?}"
        );
        assert_eq!(sh.blocks, 1);
        assert_eq!(sh.b_dwords, sh.a_dwords);
        // Per-lane element counts of the wave64 layouts (CDNA3 ISA §7.1.4 p.46-48).
        let a_bits = u32::from(sh.m) * u32::from(sh.k) * u32::from(sh.blocks) / 64 * bits_of(sh.a);
        assert_eq!(
            u32::from(sh.a_dwords),
            a_bits.div_ceil(32),
            "{op:?} A dwords"
        );
        let cd_bits =
            u32::from(sh.m) * u32::from(sh.n) * u32::from(sh.blocks) / 64 * bits_of(sh.cd);
        assert_eq!(
            u32::from(sh.cd_dwords),
            cd_bits.div_ceil(32),
            "{op:?} C/D dwords"
        );
        assert_eq!(bits_of(sh.a), bits_of(sh.b));
    }
}

// ----------------------------------------------------------------------------
// CPU-only: illegal forms
// ----------------------------------------------------------------------------

#[test]
fn encoder_accepts_the_valid_bases() {
    for x in [
        pk(Vop3pOp::PkAddF16, v(1), &[v(2), v(3)]),
        pk(Vop3pOp::PkFmaF16, v(1), &[v(2), v(3), v(4)]),
        pk(Vop3pOp::PkAddF32, v(2), &[v(4), v(6)]),
        pk(Vop3pOp::FmaMixF32, v(1), &[v(2), v(3), v(4)]),
        pk(Vop3pOp::Dot2F32F16, v(1), &[v(2), v(3), v(4)]),
        pk(Vop3pOp::Dot4I32I8, v(1), &[v(2), v(3), v(4)]),
    ] {
        assert_eq!(vwords(&x).len(), 2);
    }
    vwords(&pk(Vop3pOp::PkAddF32, v(254), &[v(252), v(254)]));
    vwords(&pk(Vop3pOp::PkAddF32, v(2), &[Src::VccLo, v(4)]));
    vwords(&pk(Vop3pOp::PkAddF32, v(2), &[v(4), Src::ExecLo]));
    vwords(&pk(Vop3pOp::PkAddF32, v(2), &[s(100), v(4)]));
    vwords(&pk(Vop3pOp::PkAddF32, v(2), &[v(4), Src::Ttmp(2)]));
    vwords(&pk(Vop3pOp::PkAddF16, v(255), &[v(255), v(255)]));
    mwords(&mfma(
        MaiOp::MfmaF32_32x32x8F16,
        a(240),
        v(254),
        v(254),
        a(240),
        7,
        15,
        7,
    ));
    mwords(&mfma(
        MaiOp::MfmaF32_16x16x4F32,
        a(252),
        v(255),
        v(255),
        Src::Int(64),
        0,
        0,
        0,
    ));
    mwords(&mai(
        MaiOp::AccvgprReadB32,
        v(255),
        a(255),
        None,
        None,
        0,
        0,
        0,
    ));
    mwords(&mai(
        MaiOp::AccvgprWriteB32,
        a(255),
        v(255),
        None,
        None,
        0,
        0,
        0,
    ));
}

#[test]
fn encoder_rejects_illegal_packed_forms() {
    use Vop3pOp::*;
    let b2 = pk(PkAddF16, v(1), &[v(2), v(3)]);
    let b3 = pk(PkFmaF16, v(1), &[v(2), v(3), v(4)]);
    let b64 = pk(PkAddF32, v(2), &[v(4), v(6)]);
    let mix = pk(FmaMixF32, v(1), &[v(2), v(3), v(4)]);
    let dot_f = pk(Dot2F32F16, v(1), &[v(2), v(3), v(4)]);
    let dot_i = pk(Dot4I32I8, v(1), &[v(2), v(3), v(4)]);
    let mut bad: Vec<(&str, Vop3p)> = vec![
        // Destination kinds.
        (
            "agpr dst",
            Vop3p {
                dst: a(1),
                ..b2.clone()
            },
        ),
        (
            "sgpr dst",
            Vop3p {
                dst: s(1),
                ..b2.clone()
            },
        ),
        (
            "vcc dst",
            Vop3p {
                dst: Src::VccLo,
                ..b2.clone()
            },
        ),
        (
            "inline dst",
            Vop3p {
                dst: Src::Int(0),
                ..b2.clone()
            },
        ),
        (
            "literal dst",
            Vop3p {
                dst: Src::Lit(1),
                ..b2.clone()
            },
        ),
        // Source kinds.
        (
            "agpr src0",
            Vop3p {
                src: [Some(a(2)), Some(v(3)), None],
                ..b2.clone()
            },
        ),
        (
            "agpr src1",
            Vop3p {
                src: [Some(v(2)), Some(a(3)), None],
                ..b2.clone()
            },
        ),
        (
            "agpr src2",
            Vop3p {
                src: [Some(v(2)), Some(v(3)), Some(a(4))],
                ..b3.clone()
            },
        ),
        (
            "literal src0",
            Vop3p {
                src: [Some(Src::Lit(0x3c00)), Some(v(3)), None],
                ..b2.clone()
            },
        ),
        (
            "literal src2",
            Vop3p {
                src: [Some(v(2)), Some(v(3)), Some(Src::Lit(5))],
                ..b3.clone()
            },
        ),
        (
            "int 65",
            Vop3p {
                src: [Some(Src::Int(65)), Some(v(3)), None],
                ..b2.clone()
            },
        ),
        (
            "int -17",
            Vop3p {
                src: [Some(v(2)), Some(Src::Int(-17)), None],
                ..b2.clone()
            },
        ),
        // Source count.
        (
            "missing src0",
            Vop3p {
                src: [None, Some(v(3)), None],
                ..b2.clone()
            },
        ),
        (
            "missing src1",
            Vop3p {
                src: [Some(v(2)), None, None],
                ..b2.clone()
            },
        ),
        (
            "missing src2",
            Vop3p {
                src: [Some(v(2)), Some(v(3)), None],
                ..b3.clone()
            },
        ),
        (
            "extra src2",
            Vop3p {
                src: [Some(v(2)), Some(v(3)), Some(v(4))],
                ..b2.clone()
            },
        ),
        (
            "extra src2 on pk_mov",
            Vop3p {
                src: [Some(v(2)), Some(v(4)), Some(v(6))],
                ..pk(PkMovB32, v(2), &[v(4), v(6)])
            },
        ),
        // 64-bit operand alignment and file bounds.
        (
            "misaligned dst",
            Vop3p {
                dst: v(3),
                ..b64.clone()
            },
        ),
        (
            "dst past v255",
            Vop3p {
                dst: v(255),
                ..b64.clone()
            },
        ),
        (
            "misaligned vgpr src",
            Vop3p {
                src: [Some(v(5)), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "vgpr src past v255",
            Vop3p {
                src: [Some(v(4)), Some(v(255)), None],
                ..b64.clone()
            },
        ),
        (
            "misaligned sgpr src",
            Vop3p {
                src: [Some(s(3)), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "sgpr pair across s101",
            Vop3p {
                src: [Some(s(101)), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "vcc_hi as 64-bit",
            Vop3p {
                src: [Some(Src::VccHi), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "exec_hi as 64-bit",
            Vop3p {
                src: [Some(Src::ExecHi), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "m0 as 64-bit",
            Vop3p {
                src: [Some(Src::M0), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        (
            "odd ttmp pair",
            Vop3p {
                src: [Some(Src::Ttmp(3)), Some(v(6)), None],
                ..b64.clone()
            },
        ),
        // Mask ranges.
        (
            "op_sel > 7",
            Vop3p {
                op_sel: 8,
                ..b3.clone()
            },
        ),
        (
            "op_sel_hi > 7",
            Vop3p {
                op_sel_hi: 8 | 7,
                ..b3.clone()
            },
        ),
        (
            "neg_lo > 7",
            Vop3p {
                neg_lo: 8,
                ..b3.clone()
            },
        ),
        (
            "neg_hi > 7",
            Vop3p {
                neg_hi: 0x10,
                ..b3.clone()
            },
        ),
        // Absent-source2 modifier bits: hi bit must be 1, the others 0.
        (
            "2-src op_sel bit2",
            Vop3p {
                op_sel: 0b100,
                ..b2.clone()
            },
        ),
        (
            "2-src neg_lo bit2",
            Vop3p {
                neg_lo: 0b100,
                ..b2.clone()
            },
        ),
        (
            "2-src neg_hi bit2",
            Vop3p {
                neg_hi: 0b100,
                ..b2.clone()
            },
        ),
        (
            "2-src op_sel_hi bit2 clear",
            Vop3p {
                op_sel_hi: 0b011,
                ..b2.clone()
            },
        ),
        (
            "pk_mov op_sel bit2",
            Vop3p {
                op_sel: 0b100,
                ..pk(PkMovB32, v(2), &[v(4), v(6)])
            },
        ),
        // Mix: neg_hi unsupported.
        (
            "mix neg_hi",
            Vop3p {
                neg_hi: 1,
                ..mix.clone()
            },
        ),
        (
            "mix neg_hi all",
            Vop3p {
                neg_hi: 7,
                ..mix.clone()
            },
        ),
        // Dot: no op_sel, op_sel_hi pinned to 7 (neg_lo/neg_hi are free on every dot op).
        (
            "dot_i op_sel",
            Vop3p {
                op_sel: 1,
                ..dot_i.clone()
            },
        ),
        (
            "dot_i hi 0",
            Vop3p {
                op_sel_hi: 0,
                ..dot_i.clone()
            },
        ),
        (
            "dot_i hi 3",
            Vop3p {
                op_sel_hi: 3,
                ..dot_i.clone()
            },
        ),
        (
            "dot_f op_sel",
            Vop3p {
                op_sel: 2,
                ..dot_f.clone()
            },
        ),
        (
            "dot_f hi 0",
            Vop3p {
                op_sel_hi: 0,
                ..dot_f.clone()
            },
        ),
        (
            "dot_f hi 6",
            Vop3p {
                op_sel_hi: 6,
                ..dot_f.clone()
            },
        ),
        // Constant bus: at most one distinct scalar (SGPR, VCC/EXEC/M0/TTMP halves, src_* values); inline constants are free.
        (
            "two sgprs src0 src1",
            Vop3p {
                src: [Some(s(2)), Some(s(3)), None],
                ..b2.clone()
            },
        ),
        (
            "sgpr + vcc_lo",
            Vop3p {
                src: [Some(s(2)), Some(Src::VccLo), None],
                ..b2.clone()
            },
        ),
        (
            "vcc_lo + vcc_hi",
            Vop3p {
                src: [Some(Src::VccLo), Some(Src::VccHi), None],
                ..b2.clone()
            },
        ),
        (
            "m0 + sgpr",
            Vop3p {
                src: [Some(Src::M0), Some(s(2)), None],
                ..b2.clone()
            },
        ),
        (
            "sgpr + scc",
            Vop3p {
                src: [Some(s(2)), Some(Src::Scc), None],
                ..b2.clone()
            },
        ),
        (
            "sgpr + vccz",
            Vop3p {
                src: [Some(s(2)), Some(Src::Vccz), None],
                ..b2.clone()
            },
        ),
        (
            "scc + vccz",
            Vop3p {
                src: [Some(Src::Scc), Some(Src::Vccz), None],
                ..b2.clone()
            },
        ),
        (
            "sgpr + shared_base",
            Vop3p {
                src: [Some(s(2)), Some(Src::SharedBase), None],
                ..b2.clone()
            },
        ),
        (
            "shared_base + shared_limit",
            Vop3p {
                src: [Some(Src::SharedBase), Some(Src::SharedLimit), None],
                ..b2.clone()
            },
        ),
        (
            "ttmp3 + ttmp4",
            Vop3p {
                src: [Some(Src::Ttmp(3)), Some(Src::Ttmp(4)), None],
                ..b2.clone()
            },
        ),
        (
            "fma: sgpr v sgpr",
            Vop3p {
                src: [Some(s(2)), Some(v(3)), Some(s(3))],
                ..b3.clone()
            },
        ),
        (
            "fma: s2 s2 s3",
            Vop3p {
                src: [Some(s(2)), Some(s(2)), Some(s(3))],
                ..b3.clone()
            },
        ),
        (
            "fma: exec_lo vcc_lo v",
            Vop3p {
                src: [Some(Src::ExecLo), Some(Src::VccLo), Some(v(1))],
                ..b3.clone()
            },
        ),
        (
            "64-bit: s[2:3] + s[4:5]",
            Vop3p {
                src: [Some(s(2)), Some(s(4)), None],
                ..b64.clone()
            },
        ),
        (
            "64-bit: s[2:3] + vcc",
            Vop3p {
                src: [Some(s(2)), Some(Src::VccLo), None],
                ..b64.clone()
            },
        ),
        (
            "64-bit: vcc + exec",
            Vop3p {
                src: [Some(Src::VccLo), Some(Src::ExecLo), None],
                ..b64.clone()
            },
        ),
        (
            "64-bit fma: s[2:3] s[2:3] s[4:5]",
            Vop3p {
                src: [Some(s(2)), Some(s(2)), Some(s(4))],
                ..pk(PkFmaF32, v(2), &[v(4), v(6), v(8)])
            },
        ),
        (
            "pk_mov: s[2:3] + s[4:5]",
            Vop3p {
                src: [Some(s(2)), Some(s(4)), None],
                ..pk(PkMovB32, v(2), &[v(4), v(6)])
            },
        ),
        (
            "mix: s2 + s3",
            Vop3p {
                src: [Some(s(2)), Some(s(3)), Some(v(4))],
                ..mix.clone()
            },
        ),
        (
            "mix: s2 v flat_scratch_lo",
            Vop3p {
                src: [Some(s(2)), Some(v(3)), Some(Src::FlatScratchLo)],
                ..mix.clone()
            },
        ),
        (
            "dot_f: s2 + s3",
            Vop3p {
                src: [Some(s(2)), Some(s(3)), Some(v(4))],
                ..dot_f.clone()
            },
        ),
        (
            "dot_i: s2 + s3",
            Vop3p {
                src: [Some(s(2)), Some(s(3)), Some(Src::Int(5))],
                ..dot_i.clone()
            },
        ),
    ];
    for op in [Dot2I32I16, Dot2U32U16, Dot4U32U8, Dot8I32I4, Dot8U32U4] {
        let d = pk(op, v(1), &[v(2), v(3), v(4)]);
        bad.push((
            "dot_i op_sel (all ops)",
            Vop3p {
                op_sel: 4,
                ..d.clone()
            },
        ));
        bad.push(("dot_i hi 0 (all ops)", Vop3p { op_sel_hi: 0, ..d }));
    }
    for (label, x) in &bad {
        assert!(
            x.encode(ARCH, &mut Vec::new()).is_err(),
            "encoder accepted {label}: {x:?}"
        );
        assert!(
            Inst::Vop3p(x.clone()).words(ARCH).is_err(),
            "Inst::words accepted {label}"
        );
    }
    // Per-op spot checks of the kinds every packed / mix / dot op must refuse.
    for &op in Vop3pOp::ALL {
        let w = dw(op);
        let n = nsrc(op);
        let base = pk(op, v(1), &base_srcs(w)[..n]);
        assert!(
            Vop3p {
                dst: a(1),
                ..base.clone()
            }
            .encode(ARCH, &mut Vec::new())
            .is_err(),
            "{op:?} agpr dst"
        );
        assert!(
            Vop3p {
                src: [Some(a(2)), base.src[1], base.src[2]],
                ..base.clone()
            }
            .encode(ARCH, &mut Vec::new())
            .is_err(),
            "{op:?} agpr src0"
        );
        assert!(
            Vop3p {
                src: [base.src[0], Some(Src::Lit(1)), base.src[2]],
                ..base.clone()
            }
            .encode(ARCH, &mut Vec::new())
            .is_err(),
            "{op:?} literal src1"
        );
        assert!(
            Vop3p {
                src: [base.src[0], None, base.src[2]],
                ..base.clone()
            }
            .encode(ARCH, &mut Vec::new())
            .is_err(),
            "{op:?} missing src1"
        );
        let extra = if n == 2 {
            Some(base.src[0].unwrap())
        } else {
            None
        };
        assert!(
            Vop3p {
                src: [base.src[0], base.src[1], extra],
                ..base.clone()
            }
            .encode(ARCH, &mut Vec::new())
            .is_err(),
            "{op:?} wrong src2 presence"
        );
    }
}

#[test]
fn encoder_rejects_illegal_mai_forms() {
    use MaiOp::*;
    let f = MfmaF32_16x16x32Fp8Fp8;
    let big = MfmaF32_32x32x8F16;
    let one = MfmaF32_16x16x4F32;
    let f64op = MfmaF64_16x16x4F64;
    let b = mfma(f, a(0), v(2), v(4), a(0), 0, 0, 0);
    let rd = mai(AccvgprReadB32, v(1), a(2), None, None, 0, 0, 0);
    let wr = mai(AccvgprWriteB32, a(1), v(2), None, None, 0, 0, 0);
    let bad: Vec<(&str, Mai)> = vec![
        // ACC_CD agreement and C kinds.
        (
            "agpr dst, vgpr C",
            Mai {
                src_c: Some(v(0)),
                ..b.clone()
            },
        ),
        (
            "vgpr dst, agpr C",
            Mai {
                dst: v(0),
                src_c: Some(a(0)),
                ..b.clone()
            },
        ),
        (
            "sgpr dst",
            Mai {
                dst: s(0),
                src_c: Some(s(0)),
                ..b.clone()
            },
        ),
        (
            "sgpr C",
            Mai {
                src_c: Some(s(0)),
                ..b.clone()
            },
        ),
        (
            "vcc C",
            Mai {
                src_c: Some(Src::VccLo),
                ..b.clone()
            },
        ),
        (
            "m0 C",
            Mai {
                src_c: Some(Src::M0),
                ..b.clone()
            },
        ),
        (
            "literal C",
            Mai {
                src_c: Some(Src::Lit(0x41)),
                ..b.clone()
            },
        ),
        (
            "int 65 C",
            Mai {
                src_c: Some(Src::Int(65)),
                ..b.clone()
            },
        ),
        (
            "int -17 C",
            Mai {
                src_c: Some(Src::Int(-17)),
                ..b.clone()
            },
        ),
        (
            "vgpr dst, int 65 C",
            Mai {
                dst: v(0),
                src_c: Some(Src::Int(65)),
                ..b.clone()
            },
        ),
        // A/B kinds.
        (
            "sgpr A",
            Mai {
                src_a: s(2),
                ..b.clone()
            },
        ),
        (
            "sgpr B",
            Mai {
                src_b: Some(s(4)),
                ..b.clone()
            },
        ),
        (
            "inline A",
            Mai {
                src_a: Src::Int(1),
                ..b.clone()
            },
        ),
        (
            "inline B",
            Mai {
                src_b: Some(Src::Float(InlineFloat::One)),
                ..b.clone()
            },
        ),
        (
            "literal A",
            Mai {
                src_a: Src::Lit(1),
                ..b.clone()
            },
        ),
        (
            "literal B",
            Mai {
                src_b: Some(Src::Lit(1)),
                ..b.clone()
            },
        ),
        (
            "vcc A",
            Mai {
                src_a: Src::VccLo,
                ..b.clone()
            },
        ),
        // Missing operands.
        (
            "no B",
            Mai {
                src_b: None,
                ..b.clone()
            },
        ),
        (
            "no C",
            Mai {
                src_c: None,
                ..b.clone()
            },
        ),
        (
            "no B and C",
            Mai {
                src_b: None,
                src_c: None,
                ..b.clone()
            },
        ),
        // Field ranges (cbsz 3 bits, abid 4 bits, blgp 3 bits).
        (
            "cbsz 8",
            Mai {
                cbsz: 8,
                ..b.clone()
            },
        ),
        (
            "abid 16",
            Mai {
                abid: 16,
                ..b.clone()
            },
        ),
        (
            "blgp 8",
            Mai {
                blgp: 8,
                ..b.clone()
            },
        ),
        (
            "f64 blgp (neg) 8",
            Mai {
                blgp: 8,
                ..mfma(f64op, a(0), v(2), v(4), a(0), 0, 0, 0)
            },
        ),
        ("f64 cbsz 8", mfma(f64op, a(0), v(2), v(4), a(0), 8, 0, 0)),
        (
            "f64 C 1/2pi (acc)",
            mfma(
                f64op,
                a(0),
                v(2),
                v(4),
                Src::Float(InlineFloat::InvTwoPi),
                0,
                0,
                0,
            ),
        ),
        (
            "f64 C 1/2pi (vgpr)",
            mfma(
                f64op,
                v(0),
                v(2),
                v(4),
                Src::Float(InlineFloat::InvTwoPi),
                0,
                0,
                0,
            ),
        ),
        // Alignment and register-file bounds from the shape widths.
        (
            "odd cd (4 dwords)",
            mfma(f, a(1), v(2), v(4), a(1), 0, 0, 0),
        ),
        (
            "odd cd (16 dwords)",
            mfma(big, a(3), v(2), v(4), a(3), 0, 0, 0),
        ),
        ("odd cd vgpr", mfma(f, v(1), v(2), v(4), v(1), 0, 0, 0)),
        (
            "odd cd (8 dwords, f64)",
            mfma(f64op, a(1), v(2), v(4), a(1), 0, 0, 0),
        ),
        (
            "cd past a255 (4 dwords)",
            mfma(f, a(254), v(2), v(4), a(254), 0, 0, 0),
        ),
        (
            "cd past a255 (16 dwords)",
            mfma(big, a(242), v(2), v(4), a(242), 0, 0, 0),
        ),
        (
            "cd past v255 (16 dwords)",
            mfma(big, v(244), v(2), v(4), v(244), 0, 0, 0),
        ),
        (
            "cd past a255 (8 dwords)",
            mfma(f64op, a(252), v(2), v(4), a(252), 0, 0, 0),
        ),
        ("C odd, D aligned", mfma(f, a(0), v(2), v(4), a(1), 0, 0, 0)),
        ("C past a255", mfma(f, a(0), v(2), v(4), a(254), 0, 0, 0)),
        ("odd A (2 dwords)", mfma(f, a(0), v(3), v(4), a(0), 0, 0, 0)),
        ("odd B (2 dwords)", mfma(f, a(0), v(2), v(5), a(0), 0, 0, 0)),
        ("odd A agpr", mfma(f, a(0), a(3), v(4), a(0), 0, 0, 0)),
        ("A past v255", mfma(f, a(0), v(255), v(4), a(0), 0, 0, 0)),
        ("B past a255", mfma(f, a(0), v(2), a(255), a(0), 0, 0, 0)),
        ("odd A f64", mfma(f64op, a(0), v(3), v(4), a(0), 0, 0, 0)),
        (
            "1-dword op: cd odd",
            mfma(one, a(2 + 1), v(2), v(3), a(2 + 1), 0, 0, 0),
        ),
        (
            "1-dword op: cd past a255",
            mfma(one, a(254), v(2), v(3), a(254), 0, 0, 0),
        ),
        // Moves.
        (
            "read: agpr dst",
            Mai {
                dst: a(1),
                ..rd.clone()
            },
        ),
        (
            "read: sgpr dst",
            Mai {
                dst: s(1),
                ..rd.clone()
            },
        ),
        (
            "read: vgpr src",
            Mai {
                src_a: v(2),
                ..rd.clone()
            },
        ),
        (
            "read: sgpr src",
            Mai {
                src_a: s(2),
                ..rd.clone()
            },
        ),
        (
            "read: inline src",
            Mai {
                src_a: Src::Int(5),
                ..rd.clone()
            },
        ),
        (
            "read: src_b present",
            Mai {
                src_b: Some(a(3)),
                ..rd.clone()
            },
        ),
        (
            "read: src_c present",
            Mai {
                src_c: Some(a(3)),
                ..rd.clone()
            },
        ),
        (
            "read: cbsz",
            Mai {
                cbsz: 1,
                ..rd.clone()
            },
        ),
        (
            "read: abid",
            Mai {
                abid: 1,
                ..rd.clone()
            },
        ),
        (
            "read: blgp",
            Mai {
                blgp: 1,
                ..rd.clone()
            },
        ),
        (
            "write: vgpr dst",
            Mai {
                dst: v(1),
                ..wr.clone()
            },
        ),
        (
            "write: sgpr dst",
            Mai {
                dst: s(1),
                ..wr.clone()
            },
        ),
        (
            "write: xnack_mask dst",
            Mai {
                dst: Src::XnackMaskLo,
                ..wr.clone()
            },
        ),
        (
            "write: agpr src",
            Mai {
                src_a: a(2),
                ..wr.clone()
            },
        ),
        (
            "write: literal src",
            Mai {
                src_a: Src::Lit(0x12345),
                ..wr.clone()
            },
        ),
        (
            "write: int 65 src",
            Mai {
                src_a: Src::Int(65),
                ..wr.clone()
            },
        ),
        (
            "write: int -17 src",
            Mai {
                src_a: Src::Int(-17),
                ..wr.clone()
            },
        ),
        (
            "write: src_b present",
            Mai {
                src_b: Some(v(3)),
                ..wr.clone()
            },
        ),
        (
            "write: src_c present",
            Mai {
                src_c: Some(Src::Int(0)),
                ..wr.clone()
            },
        ),
        (
            "write: cbsz",
            Mai {
                cbsz: 2,
                ..wr.clone()
            },
        ),
        (
            "write: abid",
            Mai {
                abid: 2,
                ..wr.clone()
            },
        ),
        (
            "write: blgp",
            Mai {
                blgp: 2,
                ..wr.clone()
            },
        ),
    ];
    for (label, x) in &bad {
        assert!(
            x.encode(ARCH, &mut Vec::new()).is_err(),
            "encoder accepted {label}: {x:?}"
        );
        assert!(
            Inst::Mai(x.clone()).words(ARCH).is_err(),
            "Inst::words accepted {label}"
        );
    }
    // And the boundary neighbours of the width checks are accepted.
    mwords(&mfma(big, a(240), v(2), v(4), a(240), 0, 0, 0));
    mwords(&mfma(f, a(252), v(254), v(254), a(252), 0, 0, 0));
    mwords(&mfma(f64op, a(248), v(2), v(4), a(248), 0, 0, 0));
    mwords(&mfma(one, a(252), v(255), v(255), a(252), 0, 0, 0));
}

/// Word0 of a VOP3P encoding with opcode `op` and vdst 1.
fn vop3p_word0(op: u8) -> u32 {
    0xd380_0000 | u32::from(op) << 16 | 1
}

#[test]
fn decode_rejects_malformed_words() {
    let good_p = vwords(&pk(Vop3pOp::PkAddF16, v(1), &[v(2), v(3)]));
    let good_m = mwords(&mfma(
        MaiOp::MfmaF32_16x16x32Fp8Fp8,
        a(0),
        v(2),
        v(4),
        a(0),
        0,
        0,
        0,
    ));
    // Truncated.
    assert!(Vop3p::decode(ARCH, &good_p[..1]).is_err());
    assert!(Vop3p::decode(ARCH, &[]).is_err());
    assert!(Mai::decode(ARCH, &good_m[..1]).is_err());
    assert!(Mai::decode(ARCH, &[]).is_err());
    assert!(Inst::decode(ARCH, &good_p[..1]).is_err());
    assert!(Inst::decode(ARCH, &good_m[..1]).is_err());
    // Not a VOP3P encoding / wrong sibling.
    assert!(Vop3p::decode(ARCH, &[0xd100_0001, 0]).is_err());
    assert!(Mai::decode(ARCH, &[0xd100_0001, 0]).is_err());
    assert!(Vop3p::decode(ARCH, &good_m).is_err());
    assert!(Mai::decode(ARCH, &good_p).is_err());
    // Every opcode outside the tables.
    for code in 0..0x40u8 {
        if Vop3pOp::from_opcode(ARCH, code).is_none() {
            assert!(
                Vop3p::decode(ARCH, &[vop3p_word0(code), 0x1c12_0702]).is_err(),
                "VOP3P opcode {code:#x}"
            );
            assert!(
                Inst::decode(ARCH, &[vop3p_word0(code), 0x1c12_0702]).is_err(),
                "Inst VOP3P opcode {code:#x}"
            );
        }
    }
    for code in 0x40..0x80u8 {
        if MaiOp::from_opcode(ARCH, code).is_none() {
            assert!(
                Mai::decode(ARCH, &[vop3p_word0(code), 0x0402_0902]).is_err(),
                "MAI opcode {code:#x}"
            );
            assert!(
                Inst::decode(ARCH, &[vop3p_word0(code), 0x0402_0902]).is_err(),
                "Inst MAI opcode {code:#x}"
            );
        }
    }
    // Absent-source and unsupported-modifier bits that the struct cannot carry.
    // word0: neg_hi [10:8], op_sel [13:11], op_sel_hi[2] bit 14; word1: src2 [26:18], op_sel_hi[1:0] [28:27], neg [31:29].
    let flip = |w: &[u32], i: usize, mask: u32, set: bool| {
        let mut x = w.to_vec();
        if set {
            x[i] |= mask
        } else {
            x[i] &= !mask
        }
        x
    };
    assert!(
        Vop3p::decode(ARCH, &flip(&good_p, 0, 1 << 13, true)).is_err(),
        "2-src op_sel[2]"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_p, 0, 1 << 10, true)).is_err(),
        "2-src neg_hi[2]"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_p, 1, 1 << 31, true)).is_err(),
        "2-src neg[2]"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_p, 0, 1 << 14, false)).is_err(),
        "2-src op_sel_hi[2] clear"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_p, 1, 0x1ff << 18, true)).is_err(),
        "2-src nonzero src2 field"
    );
    let good_dot = vwords(&pk(Vop3pOp::Dot4I32I8, v(1), &[v(2), v(3), v(4)]));
    assert!(Vop3p::decode(ARCH, &good_dot).is_ok());
    assert!(
        Vop3p::decode(ARCH, &flip(&good_dot, 0, 1 << 11, true)).is_err(),
        "dot op_sel"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_dot, 0, 1 << 14, false)).is_err(),
        "dot op_sel_hi[2] clear"
    );
    assert!(
        Vop3p::decode(ARCH, &flip(&good_dot, 1, 1 << 27, false)).is_err(),
        "dot op_sel_hi[0] clear"
    );
    // Dot ops carry neg_lo/neg_hi (all of them, integer dots included).
    for (word, mask) in [
        (1usize, 1u32 << 29),
        (1, 1 << 30),
        (1, 1 << 31),
        (0, 1 << 8),
        (0, 1 << 9),
        (0, 1 << 10),
    ] {
        let w = flip(&good_dot, word, mask, true);
        let (decoded, len) = Vop3p::decode(ARCH, &w)
            .unwrap_or_else(|e| panic!("dot neg bit {mask:#x} of word {word}: {e}"));
        assert_eq!(len, 2);
        assert_eq!(vwords(&decoded), w);
    }
    let good_mix = vwords(&pk(Vop3pOp::FmaMixF32, v(1), &[v(2), v(3), v(4)]));
    assert!(Vop3p::decode(ARCH, &good_mix).is_ok());
    assert!(
        Vop3p::decode(ARCH, &flip(&good_mix, 0, 1 << 8, true)).is_err(),
        "mix neg_hi (abs)"
    );
    // Moves carry no cbsz/abid/blgp.
    let good_rd = mwords(&mai(MaiOp::AccvgprReadB32, v(1), a(2), None, None, 0, 0, 0));
    assert!(Mai::decode(ARCH, &good_rd).is_ok());
    assert!(
        Mai::decode(ARCH, &flip(&good_rd, 0, 1 << 8, true)).is_err(),
        "move cbsz"
    );
    // Literal / SGPR / out-of-range operand codes in MFMA A/B and C.
    let mut sgpr_a = good_m.clone();
    sgpr_a[1] = (sgpr_a[1] & !0x1ff) | 0x02;
    assert!(Mai::decode(ARCH, &sgpr_a).is_err(), "SGPR A");
    let mut inline_b = good_m.clone();
    inline_b[1] = (inline_b[1] & !(0x1ff << 9)) | 0x80 << 9;
    assert!(Mai::decode(ARCH, &inline_b).is_err(), "inline B");
    let mut lit_a = good_m.clone();
    lit_a[1] = (lit_a[1] & !0x1ff) | 0xff;
    assert!(Mai::decode(ARCH, &lit_a).is_err(), "literal A");
    let mut sgpr_c = good_m.clone();
    sgpr_c[1] = (sgpr_c[1] & !(0x1ff << 18)) | 4 << 18;
    assert!(Mai::decode(ARCH, &sgpr_c).is_err(), "SGPR C");
}

// ----------------------------------------------------------------------------
// CPU-only: footprints
// ----------------------------------------------------------------------------

fn check_footprint(
    label: &str,
    fp: &Footprint,
    defs: &[RegSpan],
    uses: &[RegSpan],
    exec: Option<bool>,
) {
    assert_eq!(fp.defs, defs, "defs of {label}");
    for u in uses {
        assert!(
            fp.uses.contains(u),
            "{label}: missing use {u:?} in {:?}",
            fp.uses
        )
    }
    for u in &fp.uses {
        assert!(
            uses.contains(u) || u.file == RegFile::Exec,
            "{label}: unexpected use {u:?}"
        )
    }
    if let Some(want) = exec {
        assert_eq!(
            fp.uses.iter().any(|u| u.file == RegFile::Exec),
            want,
            "{label}: EXEC read"
        )
    }
}

fn file_of(x: Src) -> RegFile {
    if x.is_agpr() {
        RegFile::Agpr
    } else {
        RegFile::Vgpr
    }
}

#[test]
fn footprint_vop3p_matches_operands() {
    for x in vop3p_corpus() {
        let w = dw(x.op);
        let defs = [RegSpan::of(x.dst, w).unwrap()];
        let uses: Vec<RegSpan> = x
            .src
            .iter()
            .flatten()
            .filter_map(|&src| RegSpan::of(src, w))
            .collect();
        let label = asm_p(&x);
        check_footprint(&label, &x.footprint(), &defs, &uses, Some(true));
        check_footprint(
            &label,
            &Inst::Vop3p(x.clone()).footprint(),
            &defs,
            &uses,
            Some(true),
        );
    }
}

#[test]
fn footprint_vop3p_spans_by_hand() {
    let fp = pk(Vop3pOp::PkFmaF32, v(10), &[v(20), s(4), Src::Int(3)]).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Vgpr, 10, 2)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 20, 2)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Sgpr, 4, 2)));
    assert!(fp.uses.iter().any(|u| u.file == RegFile::Exec));
    assert_eq!(
        fp.uses.iter().filter(|u| u.file != RegFile::Exec).count(),
        2,
        "inline constants read no register"
    );
    let fp = pk(Vop3pOp::Dot4I32I8, v(7), &[v(8), Src::VccLo, v(9)]).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Vgpr, 7, 1)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 8, 1)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vcc, 0, 1)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 9, 1)));
    let fp = pk(Vop3pOp::PkMulF32, v(2), &[Src::VccLo, v(4)]).footprint();
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vcc, 0, 2)));
    assert!(
        fp.uses.iter().any(|u| u.file == RegFile::Exec),
        "lane-masked VALU op reads EXEC"
    );
    assert!(
        fp.defs.iter().all(|d| d.file == RegFile::Vgpr),
        "VOP3P writes no VCC/EXEC"
    );
}

#[test]
fn footprint_mai_matches_operands() {
    for x in mfma_operand_corpus()
        .into_iter()
        .chain(mfma_modifier_corpus())
    {
        let sh = x.op.shape().unwrap();
        let defs = [RegSpan::of(x.dst, sh.cd_dwords).unwrap()];
        let mut uses = vec![
            RegSpan::of(x.src_a, sh.a_dwords).unwrap(),
            RegSpan::of(x.src_b.unwrap(), sh.b_dwords).unwrap(),
        ];
        if let Some(c) = RegSpan::of(x.src_c.unwrap(), sh.cd_dwords) {
            uses.push(c)
        }
        let label = asm_m(&x);
        check_footprint(&label, &x.footprint(), &defs, &uses, None);
        check_footprint(
            &label,
            &Inst::Mai(x.clone()).footprint(),
            &defs,
            &uses,
            None,
        );
        assert_eq!(x.footprint().defs[0].file, file_of(x.dst), "{label}");
        assert_eq!(x.footprint().defs[0].len, sh.cd_dwords, "{label}");
    }
}

#[test]
fn footprint_mai_spans_by_hand() {
    // 32x32x8 f16: A/B 2 dwords, C/D 16 dwords.
    let fp = mfma(MaiOp::MfmaF32_32x32x8F16, a(16), a(2), v(4), a(32), 0, 0, 0).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Agpr, 16, 16)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Agpr, 2, 2)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 4, 2)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Agpr, 32, 16)));
    // Inline C reads nothing.
    let fp = mfma(
        MaiOp::MfmaF32_16x16x4F32,
        v(8),
        v(2),
        v(3),
        Src::Int(0),
        0,
        0,
        0,
    )
    .footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Vgpr, 8, 4)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 2, 1)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 3, 1)));
    assert!(
        fp.uses
            .iter()
            .all(|u| u.file == RegFile::Exec || (u.file == RegFile::Vgpr && u.len == 1)),
        "{:?}",
        fp.uses
    );
    // f64: A/B 2 dwords, C/D 8.
    let fp = mfma(MaiOp::MfmaF64_16x16x4F64, a(8), v(2), v(4), a(8), 0, 0, 1).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Agpr, 8, 8)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Agpr, 8, 8)));
}

#[test]
fn footprint_accvgpr_moves() {
    for x in move_corpus() {
        let label = asm_m(&x);
        let fp = x.footprint();
        match x.op {
            MaiOp::AccvgprReadB32 => {
                let uses = [RegSpan::of(x.src_a, 1).unwrap()];
                check_footprint(&label, &fp, &[RegSpan::of(x.dst, 1).unwrap()], &uses, None);
                assert_eq!(fp.defs[0].file, RegFile::Vgpr);
                assert_eq!(uses[0].file, RegFile::Agpr);
            }
            MaiOp::AccvgprWriteB32 => {
                let uses: Vec<RegSpan> = RegSpan::of(x.src_a, 1).into_iter().collect();
                check_footprint(&label, &fp, &[RegSpan::of(x.dst, 1).unwrap()], &uses, None);
                assert_eq!(fp.defs[0].file, RegFile::Agpr);
            }
            op => panic!("unexpected {op:?}"),
        }
        assert_eq!(Inst::Mai(x.clone()).footprint(), fp);
    }
    // Hand-checked spans.
    let fp = mai(MaiOp::AccvgprReadB32, v(3), a(200), None, None, 0, 0, 0).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Vgpr, 3, 1)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Agpr, 200, 1)));
    let fp = mai(MaiOp::AccvgprWriteB32, a(5), s(9), None, None, 0, 0, 0).footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Agpr, 5, 1)]);
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Sgpr, 9, 1)));
    let fp = mai(
        MaiOp::AccvgprWriteB32,
        a(5),
        Src::Float(InlineFloat::One),
        None,
        None,
        0,
        0,
        0,
    )
    .footprint();
    assert_eq!(fp.defs, [RegSpan::new(RegFile::Agpr, 5, 1)]);
    assert!(
        fp.uses.iter().all(|u| u.file == RegFile::Exec),
        "inline source reads no register: {:?}",
        fp.uses
    );
}

#[test]
fn encoder_accepts_constant_bus_neutral_forms() {
    use Vop3pOp::*;
    // A repeated scalar and inline constants share/avoid the bus; VGPRs never use it.
    for x in [
        pk(PkAddF16, v(1), &[s(2), s(2)]),
        pk(PkFmaF16, v(1), &[s(2), s(2), s(2)]),
        pk(PkFmaF16, v(1), &[s(2), v(3), s(2)]),
        pk(PkAddF16, v(1), &[s(2), Src::Float(InlineFloat::One)]),
        pk(PkAddF16, v(1), &[Src::Int(5), Src::Int(-3)]),
        pk(PkAddF16, v(1), &[Src::SharedBase, Src::SharedBase]),
        pk(PkAddF16, v(1), &[Src::Ttmp(3), Src::Ttmp(3)]),
        pk(PkAddF32, v(2), &[s(2), s(2)]),
        pk(PkFmaF32, v(2), &[s(2), v(4), s(2)]),
        pk(PkMovB32, v(2), &[s(2), s(2)]),
        pk(Dot2F32F16, v(1), &[s(2), s(2), s(2)]),
        pk(Dot4I32I8, v(1), &[s(2), v(3), Src::Int(5)]),
        pk(FmaMixF32, v(1), &[s(2), s(2), s(2)]),
        Vop3p {
            neg_lo: 0b010,
            ..pk(FmaMixF32, v(1), &[s(2), s(2), s(2)])
        },
    ] {
        assert_eq!(vwords(&x).len(), 2, "{x:?}");
    }
}

#[test]
fn mix_neg_on_inline_constants_is_a_neg_bit_not_a_value() {
    // neg(-1) sets the neg bit on the constant -1; a bare -1 is the same operand code with no neg bit.
    for c in inline_consts() {
        let plain = pk(Vop3pOp::FmaMixF32, v(1), &[c, v(3), v(4)]);
        let neg = Vop3p {
            neg_lo: 1,
            ..plain.clone()
        };
        let (pw, nw) = (vwords(&plain), vwords(&neg));
        assert_eq!(pw[0], nw[0]);
        assert_eq!(pw[1] ^ nw[1], 1 << 29, "{c:?}: only neg[0] differs");
        assert_eq!(Vop3p::decode(ARCH, &nw).unwrap().0, neg);
        let text = neg.to_string();
        assert!(text.contains(&format!("neg({})", c.fmt(1))), "{text}");
        assert!(!plain.to_string().contains("neg("), "{}", plain);
    }
    // Mixed masks across register and constant sources round-trip through decode.
    for m in 0..8u8 {
        let x = Vop3p {
            neg_lo: m,
            ..pk(
                Vop3pOp::FmaMixF32,
                v(1),
                &[Src::Int(-1), v(3), Src::Float(InlineFloat::NegHalf)],
            )
        };
        let w = vwords(&x);
        assert_eq!(w[1] >> 29, u32::from(m));
        assert_eq!(Vop3p::decode(ARCH, &w).unwrap().0, x);
    }
}

#[test]
fn dot_neg_masks_encode_for_every_dot_op() {
    for &op in Vop3pOp::ALL {
        if class(op) != Class::Dot {
            continue;
        }
        for lo in 0..8u8 {
            for hi in 0..8u8 {
                let x = modded(pk(op, v(1), &[v(2), v(3), v(4)]), 0, 7, lo, hi, lo == hi);
                let w = vwords(&x);
                assert_eq!(w[1] >> 29, u32::from(lo), "{op:?}");
                assert_eq!((w[0] >> 8) & 7, u32::from(hi), "{op:?}");
                assert_eq!(Vop3p::decode(ARCH, &w).unwrap().0, x, "{op:?}");
            }
        }
    }
}
