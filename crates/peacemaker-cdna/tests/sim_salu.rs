// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar ALU emulator tests (CDNA3 ISA §12.1-12.5): every opcode of the
//! SOP2/SOP1/SOPK/SOPC/SOPP tables is run through `sim::exec_one` on a fresh
//! `Wave` and compared with a Rust reference written from the ISA pseudocode
//! (wider integer types and bit-by-bit loops, so the reference does not share
//! the emulator's expressions). Randomised operands are biased towards the
//! values where scalar ALU semantics bite (0, 1, sign boundaries, 31/32/63/64).
//! No oracle dependency: everything here always runs.

use peacemaker_cdna::arch::Arch;
use peacemaker_cdna::enc::sop::{
    Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp,
};
use peacemaker_cdna::inst::{Inst, RegFile, RegSpan};
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::sim::{
    exec_one, salu, Config, Counter, Flow, Memory, Pending, SimError, SimResult, Wave,
};

// ---------------------------------------------------------------------------
// Harness

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn v32(&mut self) -> u32 {
        const E: [u32; 17] = [
            0,
            1,
            2,
            3,
            15,
            16,
            31,
            32,
            33,
            63,
            64,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            0xffff_fffe,
            0xffff_ffff,
            0xffff,
        ];
        let r = self.next();
        if r % 3 == 0 {
            E[(r >> 8) as usize % E.len()]
        } else {
            (r >> 16) as u32
        }
    }
    fn v64(&mut self) -> u64 {
        const E: [u64; 8] = [
            0,
            1,
            u64::MAX,
            1 << 63,
            (1 << 63) | 1,
            0x7fff_ffff_ffff_ffff,
            0xffff_ffff,
            0x1_0000_0000,
        ];
        let r = self.next();
        match r % 4 {
            0 => E[(r >> 8) as usize % E.len()],
            1 => u64::from(self.v32()) | u64::from(self.v32()) << 32,
            _ => self.next(),
        }
    }
}

const SA: u8 = 0; // source A: s[0:1]
const SB: u8 = 2; // source B: s[2:3]
const SD: u8 = 4; // destination: s[4:5]
const M32: u64 = 0xffff_ffff;
const ITERS: usize = 1500;

/// A wave (all registers 0) with A = s[0:1], B = s[2:3], D = s[4:5] preloaded.
fn fresh(a: u64, b: u64, old: u64, scc: bool) -> Wave {
    let mut w = Wave::new(Arch::Gfx942, 0, 0);
    w.s[0] = a as u32;
    w.s[1] = (a >> 32) as u32;
    w.s[2] = b as u32;
    w.s[3] = (b >> 32) as u32;
    w.s[4] = old as u32;
    w.s[5] = (old >> 32) as u32;
    w.scc = scc;
    w
}
fn step(w: &mut Wave, i: &Inst) -> SimResult<Flow> {
    exec_one(w, i, &mut [], &mut Memory::default(), &Config::default())
}
fn d64(w: &Wave) -> u64 {
    u64::from(w.s[4]) | u64::from(w.s[5]) << 32
}
fn sg(n: u8) -> Src {
    Src::Sgpr(n)
}

fn sop2(op: Sop2Op, d: Src, a: Src, b: Src) -> Inst {
    Inst::Sop2(Sop2 {
        op,
        sdst: d,
        ssrc0: a,
        ssrc1: b,
    })
}
fn sop1(op: Sop1Op, d: Option<Src>, a: Option<Src>) -> Inst {
    Inst::Sop1(Sop1 {
        op,
        sdst: d,
        ssrc0: a,
    })
}
fn sopk(op: SopkOp, d: Src, imm: u16) -> Inst {
    Inst::Sopk(Sopk {
        op,
        sdst: d,
        simm16: imm,
    })
}
fn sopc(op: SopcOp, a: Src, b: Src) -> Inst {
    Inst::Sopc(Sopc {
        op,
        ssrc0: a,
        ssrc1: b,
    })
}
fn sopp(op: SoppOp, imm: u16) -> Inst {
    Inst::Sopp(Sopp { op, simm16: imm })
}

/// Run `op` on `(a, b)` with default operand wiring; return the wave.
fn run2(op: Sop2Op, a: u64, b: u64, old: u64, scc: bool) -> Wave {
    let mut w = fresh(a, b, old, scc);
    let f = step(&mut w, &sop2(op, sg(SD), sg(SA), sg(SB))).unwrap();
    assert_eq!(f, Flow::Next);
    w
}

// ---------------------------------------------------------------------------
// SOP2

fn is64_sop2(op: Sop2Op) -> bool {
    use Sop2Op::*;
    matches!(
        op,
        CselectB64 | AndB64 | OrB64 | XorB64 | Andn2B64 | Orn2B64 | LshlB64 | LshrB64 | AshrI64
    )
}

/// BFE_U32 from the pseudocode, one bit at a time: field of S1[22:16] bits at
/// offset S1[4:0]; a width of 32 or more keeps every bit.
fn ref_bfe_u32(a: u32, c: u32) -> u32 {
    let (off, wd) = (c & 31, (c >> 16) & 0x7f);
    let mut r = 0u32;
    for i in 0..wd.min(32) {
        let s = off + i;
        let bit = if s < 32 { a >> s & 1 } else { 0 };
        r |= bit << i;
    }
    r
}
/// BFE_I32: bits shifted in past bit 31 are the sign; sign-extend from the top bit of the field.
fn ref_bfe_i32(a: u32, c: u32) -> u32 {
    let (off, wd) = (c & 31, ((c >> 16) & 0x7f).min(32));
    let mut r = 0u32;
    for i in 0..wd {
        let s = off + i;
        let bit = if s < 32 { a >> s & 1 } else { a >> 31 & 1 };
        r |= bit << i;
    }
    if wd > 0 && wd < 32 && r >> (wd - 1) & 1 == 1 {
        r |= u32::MAX << wd
    }
    r
}

/// `(D value at the destination's width, new SCC or None = unchanged)`.
fn ref_sop2(op: Sop2Op, a: u64, b: u64, scc: bool) -> (u64, Option<bool>) {
    use Sop2Op::*;
    let (a32, b32) = (a as u32, b as u32);
    let nz = |r: u64| Some(r != 0);
    match op {
        AddU32 => {
            let t = u64::from(a32) + u64::from(b32);
            (t & M32, Some(t >= 1 << 32))
        }
        SubU32 => {
            let t = i64::from(a32) - i64::from(b32);
            (t as u64 & M32, Some(t < 0))
        }
        AddI32 => {
            let t = i64::from(a32 as i32) + i64::from(b32 as i32);
            (t as u64 & M32, Some(t != i64::from(t as i32)))
        }
        SubI32 => {
            let t = i64::from(a32 as i32) - i64::from(b32 as i32);
            (t as u64 & M32, Some(t != i64::from(t as i32)))
        }
        AddcU32 => {
            let t = u64::from(a32) + u64::from(b32) + u64::from(scc);
            (t & M32, Some(t >= 1 << 32))
        }
        SubbU32 => {
            let t = i64::from(a32) - i64::from(b32) - i64::from(scc);
            (t as u64 & M32, Some(t < 0))
        }
        MinI32 => {
            let p = (a32 as i32) < (b32 as i32);
            (u64::from(if p { a32 } else { b32 }), Some(p))
        }
        MinU32 => {
            let p = a32 < b32;
            (u64::from(if p { a32 } else { b32 }), Some(p))
        }
        MaxI32 => {
            let p = (a32 as i32) >= (b32 as i32);
            (u64::from(if p { a32 } else { b32 }), Some(p))
        }
        MaxU32 => {
            let p = a32 >= b32;
            (u64::from(if p { a32 } else { b32 }), Some(p))
        }
        CselectB32 => (u64::from(if scc { a32 } else { b32 }), None),
        CselectB64 => (if scc { a } else { b }, None),
        AndB32 => {
            let r = u64::from(a32 & b32);
            (r, nz(r))
        }
        OrB32 => {
            let r = u64::from(a32 | b32);
            (r, nz(r))
        }
        XorB32 => {
            let r = u64::from(a32 ^ b32);
            (r, nz(r))
        }
        Andn2B32 => {
            let r = u64::from(a32 & !b32);
            (r, nz(r))
        }
        Orn2B32 => {
            let r = u64::from(a32 | !b32);
            (r, nz(r))
        }
        AndB64 => {
            let r = a & b;
            (r, nz(r))
        }
        OrB64 => {
            let r = a | b;
            (r, nz(r))
        }
        XorB64 => {
            let r = a ^ b;
            (r, nz(r))
        }
        Andn2B64 => {
            let r = a & !b;
            (r, nz(r))
        }
        Orn2B64 => {
            let r = a | !b;
            (r, nz(r))
        }
        LshlB32 => {
            let r = (u64::from(a32) << (b32 & 31)) & M32;
            (r, nz(r))
        }
        LshrB32 => {
            let r = u64::from(a32) >> (b32 & 31);
            (r, nz(r))
        }
        AshrI32 => {
            let r = ((i64::from(a32 as i32)) >> (b32 & 31)) as u64 & M32;
            (r, nz(r))
        }
        LshlB64 => {
            let r = ((u128::from(a)) << (b32 & 63)) as u64;
            (r, nz(r))
        }
        LshrB64 => {
            let r = ((u128::from(a)) >> (b32 & 63)) as u64;
            (r, nz(r))
        }
        AshrI64 => {
            let r = ((i128::from(a as i64)) >> (b32 & 63)) as u64;
            (r, nz(r))
        }
        BfmB32 => ((((1u64 << (a32 & 31)) - 1) << (b32 & 31)) & M32, None),
        MulI32 => (
            (i64::from(a32 as i32) * i64::from(b32 as i32)) as u64 & M32,
            None,
        ),
        BfeU32 => {
            let r = u64::from(ref_bfe_u32(a32, b32));
            (r, nz(r))
        }
        BfeI32 => {
            let r = u64::from(ref_bfe_i32(a32, b32));
            (r, nz(r))
        }
        MulHiU32 => ((u64::from(a32) * u64::from(b32)) >> 32, None),
        MulHiI32 => (
            ((i64::from(a32 as i32) * i64::from(b32 as i32)) >> 32) as u64 & M32,
            None,
        ),
        Lshl1AddU32 | Lshl2AddU32 | Lshl3AddU32 | Lshl4AddU32 => {
            let n = match op {
                Lshl1AddU32 => 1,
                Lshl2AddU32 => 2,
                Lshl3AddU32 => 3,
                _ => 4,
            };
            let t = (u64::from(a32) << n) + u64::from(b32);
            (t & M32, Some(t >= 1 << 32))
        }
        PackLlB32B16 => (u64::from((b32 & 0xffff) << 16 | (a32 & 0xffff)), None),
        PackLhB32B16 => (u64::from((b32 & 0xffff_0000) | (a32 & 0xffff)), None),
        PackHhB32B16 => (u64::from((b32 & 0xffff_0000) | (a32 >> 16)), None),
        AbsdiffI32 => {
            let d = (a32 as i32).wrapping_sub(b32 as i32);
            let r = if d < 0 { d.wrapping_neg() } else { d };
            (u64::from(r as u32), Some(r != 0))
        }
    }
}

fn check_sop2(op: Sop2Op) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (op as u64 + 1));
    let is64 = is64_sop2(op);
    for i in 0..ITERS {
        let scc = i & 1 == 1;
        let (mut a, mut b, old) = (rng.v64(), rng.v64(), rng.v64());
        if !is64 {
            // 32-bit operands: the dword above is junk that must be ignored.
            a = u64::from(rng.v32()) | a << 32;
            b = u64::from(rng.v32()) | b << 32;
        }
        let w = run2(op, a, b, old, scc);
        let (r, s) = ref_sop2(op, a, b, scc);
        let want = if is64 { r } else { old & !M32 | r & M32 };
        let ctx = format!("{op:?} a={a:#x} b={b:#x} scc={scc}");
        assert_eq!(d64(&w), want, "{ctx}: D");
        assert_eq!(w.scc, s.unwrap_or(scc), "{ctx}: SCC");
        assert_eq!(
            [w.s[0], w.s[1], w.s[2], w.s[3]],
            [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32],
            "{ctx}: sources clobbered"
        );
    }
}

macro_rules! sop2_tests {
    ($($name:ident => $op:ident),+ $(,)?) => { $(#[test] fn $name() { check_sop2(Sop2Op::$op) })+ };
}
sop2_tests! {
    s_add_u32 => AddU32, s_sub_u32 => SubU32, s_add_i32 => AddI32, s_sub_i32 => SubI32, s_addc_u32 => AddcU32,
    s_subb_u32 => SubbU32, s_min_i32 => MinI32, s_min_u32 => MinU32, s_max_i32 => MaxI32, s_max_u32 => MaxU32,
    s_cselect_b32 => CselectB32, s_cselect_b64 => CselectB64, s_and_b32 => AndB32, s_and_b64 => AndB64,
    s_or_b32 => OrB32, s_or_b64 => OrB64, s_xor_b32 => XorB32, s_xor_b64 => XorB64,
    s_andn2_b32 => Andn2B32, s_andn2_b64 => Andn2B64, s_orn2_b32 => Orn2B32, s_orn2_b64 => Orn2B64,
    s_lshl_b32 => LshlB32, s_lshl_b64 => LshlB64, s_lshr_b32 => LshrB32, s_lshr_b64 => LshrB64,
    s_ashr_i32 => AshrI32, s_ashr_i64 => AshrI64, s_bfm_b32 => BfmB32, s_mul_i32 => MulI32,
    s_bfe_u32 => BfeU32, s_bfe_i32 => BfeI32, s_mul_hi_u32 => MulHiU32, s_mul_hi_i32 => MulHiI32,
    s_lshl1_add_u32 => Lshl1AddU32, s_lshl2_add_u32 => Lshl2AddU32, s_lshl3_add_u32 => Lshl3AddU32,
    s_lshl4_add_u32 => Lshl4AddU32, s_pack_ll_b32_b16 => PackLlB32B16, s_pack_lh_b32_b16 => PackLhB32B16,
    s_pack_hh_b32_b16 => PackHhB32B16, s_absdiff_i32 => AbsdiffI32,
}

#[test]
fn sop2_all_ops_covered() {
    // The reference above is an exhaustive `match`; this runs every table entry once more.
    for &op in Sop2Op::ALL {
        check_sop2(op)
    }
}

fn d32(w: &Wave) -> u32 {
    w.s[4]
}

#[test]
fn sop2_carry_and_borrow_edges() {
    let w = run2(Sop2Op::AddU32, 0xffff_ffff, 1, 0, false);
    assert_eq!((d32(&w), w.scc), (0, true));
    let w = run2(Sop2Op::AddU32, 0x7fff_ffff, 1, 0, true);
    assert_eq!((d32(&w), w.scc), (0x8000_0000, false));
    let w = run2(Sop2Op::AddcU32, 0xffff_ffff, 0, 0, true);
    assert_eq!((d32(&w), w.scc), (0, true));
    let w = run2(Sop2Op::AddcU32, 0xffff_fffe, 1, 0, true);
    assert_eq!((d32(&w), w.scc), (0, true));
    let w = run2(Sop2Op::SubU32, 0, 1, 0, false);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    let w = run2(Sop2Op::SubbU32, 0, 0, 0, true);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    // S1 + SCC > S0 even though S0 - S1 does not wrap.
    let w = run2(Sop2Op::SubbU32, 5, 5, 0, true);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    let w = run2(Sop2Op::SubbU32, 5, 4, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
    // 64-bit add via ADD_U32 / ADDC_U32.
    let mut w = fresh(0x0000_0001_ffff_ffff, 0x0000_0002_0000_0001, 0, false);
    step(&mut w, &sop2(Sop2Op::AddU32, sg(4), sg(0), sg(2))).unwrap();
    step(&mut w, &sop2(Sop2Op::AddcU32, sg(5), sg(1), sg(3))).unwrap();
    assert_eq!(d64(&w), 0x0000_0001_ffff_ffff + 0x0000_0002_0000_0001);
}

#[test]
fn sop2_signed_overflow_edges() {
    let w = run2(Sop2Op::AddI32, 0x7fff_ffff, 1, 0, false);
    assert_eq!((d32(&w), w.scc), (0x8000_0000, true));
    let w = run2(Sop2Op::AddI32, 0x8000_0000, 0xffff_ffff, 0, false);
    assert_eq!((d32(&w), w.scc), (0x7fff_ffff, true));
    let w = run2(Sop2Op::AddI32, 0x8000_0000, 0x7fff_ffff, 0, true);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, false));
    let w = run2(Sop2Op::SubI32, 0x8000_0000, 1, 0, false);
    assert_eq!((d32(&w), w.scc), (0x7fff_ffff, true));
    let w = run2(Sop2Op::SubI32, 0x7fff_ffff, 0xffff_ffff, 0, false);
    assert_eq!((d32(&w), w.scc), (0x8000_0000, true));
    let w = run2(Sop2Op::SubI32, 1, 2, 0, true);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, false));
}

#[test]
fn sop2_minmax_scc_means_first_selected() {
    let w = run2(Sop2Op::MinI32, 0xffff_ffff, 1, 0, false); // -1 < 1
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    let w = run2(Sop2Op::MinU32, 0xffff_ffff, 1, 0, true);
    assert_eq!((d32(&w), w.scc), (1, false));
    let w = run2(Sop2Op::MaxI32, 7, 7, 0, false); // equal: S0 is selected, SCC = 1
    assert_eq!((d32(&w), w.scc), (7, true));
    let w = run2(Sop2Op::MaxU32, 1, 0xffff_ffff, 0, true);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, false));
    let w = run2(Sop2Op::MaxI32, 1, 0xffff_ffff, 0, false);
    assert_eq!((d32(&w), w.scc), (1, true));
}

#[test]
fn sop2_cselect_does_not_touch_scc() {
    let w = run2(Sop2Op::CselectB32, 11, 22, 0, true);
    assert_eq!((d32(&w), w.scc), (11, true));
    let w = run2(Sop2Op::CselectB32, 11, 22, 0, false);
    assert_eq!((d32(&w), w.scc), (22, false));
    let w = run2(
        Sop2Op::CselectB64,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0,
        true,
    );
    assert_eq!(d64(&w), 0x1111_2222_3333_4444);
    let w = run2(
        Sop2Op::CselectB64,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0,
        false,
    );
    assert_eq!(d64(&w), 0x5555_6666_7777_8888);
}

#[test]
fn sop2_shift_count_is_masked() {
    // 32-bit: count & 31.
    for (op, a, b, want) in [
        (Sop2Op::LshlB32, 1u32, 32u32, 1u32),
        (Sop2Op::LshlB32, 1, 33, 2),
        (Sop2Op::LshlB32, 1, 0xffff_ffe1, 2),
        (Sop2Op::LshrB32, 0x8000_0000, 32, 0x8000_0000),
        (Sop2Op::LshrB32, 0x8000_0000, 63, 1),
        (Sop2Op::AshrI32, 0x8000_0000, 31, 0xffff_ffff),
        (Sop2Op::AshrI32, 0x8000_0000, 32, 0x8000_0000),
        (Sop2Op::AshrI32, 0x7fff_ffff, 35, 0x0fff_ffff),
    ] {
        let w = run2(op, u64::from(a), u64::from(b), 0, false);
        assert_eq!(d32(&w), want, "{op:?} {a:#x} {b}");
        assert_eq!(w.scc, want != 0);
    }
    // 64-bit: count & 63.
    for (op, a, b, want) in [
        (Sop2Op::LshlB64, 1u64, 64u32, 1u64),
        (Sop2Op::LshlB64, 1, 63, 1 << 63),
        (Sop2Op::LshlB64, 1, 32, 1 << 32),
        (Sop2Op::LshrB64, 1 << 63, 65, 1 << 62),
        (Sop2Op::LshrB64, 1 << 63, 63, 1),
        (Sop2Op::AshrI64, 1 << 63, 63, u64::MAX),
        (Sop2Op::AshrI64, 1 << 63, 64, 1 << 63),
        (
            Sop2Op::AshrI64,
            0xf000_0000_0000_0000,
            36,
            0xffff_ffff_ff00_0000,
        ),
    ] {
        let w = run2(op, a, u64::from(b), 0, false);
        assert_eq!(d64(&w), want, "{op:?} {a:#x} {b}");
        assert_eq!(w.scc, want != 0);
    }
    // Result zero: SCC clears.
    let w = run2(Sop2Op::LshlB32, 0x8000_0000, 1, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
    let w = run2(Sop2Op::AshrI32, 0, 5, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
}

#[test]
fn sop2_bfm_bfe() {
    // BFM: size S0[4:0], offset S1[4:0].
    let w = run2(Sop2Op::BfmB32, 4, 8, 0, true);
    assert_eq!((d32(&w), w.scc), (0x0000_0f00, true));
    let w = run2(Sop2Op::BfmB32, 0, 8, 0, false);
    assert_eq!(d32(&w), 0);
    let w = run2(Sop2Op::BfmB32, 32, 4, 0, false); // size = 32 & 31 = 0
    assert_eq!(d32(&w), 0);
    let w = run2(Sop2Op::BfmB32, 31, 1, 0, false);
    assert_eq!(d32(&w), 0xffff_fffe);
    let w = run2(Sop2Op::BfmB32, 16, 20, 0, false); // field runs off the top: truncated
    assert_eq!(d32(&w), 0xfff0_0000);
    // BFE_U32: offset S1[4:0], width S1[22:16].
    let w = run2(Sop2Op::BfeU32, 0xabcd_ef01, 8 | 8 << 16, 0, false);
    assert_eq!((d32(&w), w.scc), (0xef, true));
    let w = run2(Sop2Op::BfeU32, 0xabcd_ef01, 8 | 0 << 16, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
    let w = run2(Sop2Op::BfeU32, 0xabcd_ef01, 0 | 32 << 16, 0, false);
    assert_eq!(d32(&w), 0xabcd_ef01);
    let w = run2(Sop2Op::BfeU32, 0xabcd_ef01, 28 | 8 << 16, 0, false); // past bit 31: zeros
    assert_eq!(d32(&w), 0xa);
    let w = run2(Sop2Op::BfeU32, 0xabcd_ef01, 0x0000_ffe8 | 4 << 16, 0, false); // bits 5..15 and 23.. ignored
    assert_eq!(d32(&w), 0xf);
    // BFE_I32: sign-extend from the top bit of the field.
    let w = run2(Sop2Op::BfeI32, 0x0000_0f00, 8 | 4 << 16, 0, false);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    let w = run2(Sop2Op::BfeI32, 0x0000_0700, 8 | 4 << 16, 0, false);
    assert_eq!((d32(&w), w.scc), (7, true));
    let w = run2(Sop2Op::BfeI32, 0x8000_0000, 31 | 1 << 16, 0, false);
    assert_eq!((d32(&w), w.scc), (0xffff_ffff, true));
    let w = run2(Sop2Op::BfeI32, 0x8000_0000, 28 | 8 << 16, 0, false); // arithmetic shift fills with the sign
    assert_eq!(d32(&w), 0xffff_fff8);
    let w = run2(Sop2Op::BfeI32, 0x8000_0000, 0 | 32 << 16, 0, false);
    assert_eq!(d32(&w), 0x8000_0000);
    let w = run2(Sop2Op::BfeI32, 0xffff_ffff, 4 | 0 << 16, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
}

#[test]
fn sop2_mul_absdiff_mulhi() {
    let w = run2(Sop2Op::MulI32, 0x1_0001, 0x1_0001, 0, true);
    assert_eq!((d32(&w), w.scc), (0x0002_0001, true)); // MUL does not write SCC
    let w = run2(Sop2Op::MulI32, 0xffff_ffff, 5, 0, false);
    assert_eq!(d32(&w), 0xffff_fffb);
    // Documented S_ABSDIFF_I32 results (CDNA3 ISA §12.1 p.106).
    for (a, b, want) in [
        (2u32, 5u32, 3u32),
        (0xffff_ffff, 0, 1),
        (0x8000_0000, 0, 0x8000_0000),
        (0x8000_0000, 1, 0x7fff_ffff),
        (0x8000_0000, 0xffff_ffff, 0x7fff_ffff),
        (0x8000_0000, 0xffff_fffe, 0x7fff_fffe),
    ] {
        let w = run2(Sop2Op::AbsdiffI32, u64::from(a), u64::from(b), 0, false);
        assert_eq!((d32(&w), w.scc), (want, true), "absdiff {a:#x} {b:#x}");
    }
    let w = run2(Sop2Op::AbsdiffI32, 9, 9, 0, true);
    assert_eq!((d32(&w), w.scc), (0, false));
    let w = run2(Sop2Op::MulHiU32, 0xffff_ffff, 0xffff_ffff, 0, false);
    assert_eq!(d32(&w), 0xffff_fffe);
    let w = run2(Sop2Op::MulHiI32, 0xffff_ffff, 0xffff_ffff, 0, false); // (-1)*(-1) = 1
    assert_eq!(d32(&w), 0);
    let w = run2(Sop2Op::MulHiI32, 0x8000_0000, 2, 0, false); // -2^32 >> 32 = -1
    assert_eq!(d32(&w), 0xffff_ffff);
    let w = run2(Sop2Op::MulHiI32, 0x8000_0000, 0x8000_0000, 0, false); // 2^62 >> 32
    assert_eq!(d32(&w), 0x4000_0000);
}

#[test]
fn sop2_lshl_add_pack() {
    let w = run2(Sop2Op::Lshl1AddU32, 0x8000_0000, 0, 0, false);
    assert_eq!((d32(&w), w.scc), (0, true));
    let w = run2(Sop2Op::Lshl2AddU32, 3, 4, 0, true);
    assert_eq!((d32(&w), w.scc), (16, false));
    let w = run2(Sop2Op::Lshl3AddU32, 0x2000_0000, 0xffff_ffff, 0, false);
    assert_eq!(
        (d32(&w), w.scc),
        (0xffff_ffff + 0x2000_0000u32.wrapping_mul(8), true)
    );
    let w = run2(Sop2Op::Lshl4AddU32, 0x0fff_ffff, 0x10, 0, false);
    assert_eq!((d32(&w), w.scc), (0xffff_fff0u32.wrapping_add(0x10), true));
    let w = run2(Sop2Op::PackLlB32B16, 0xaaaa_1111, 0xbbbb_2222, 0, true);
    assert_eq!((d32(&w), w.scc), (0x2222_1111, true));
    let w = run2(Sop2Op::PackLhB32B16, 0xaaaa_1111, 0xbbbb_2222, 0, false);
    assert_eq!(d32(&w), 0xbbbb_1111);
    let w = run2(Sop2Op::PackHhB32B16, 0xaaaa_1111, 0xbbbb_2222, 0, false);
    assert_eq!(d32(&w), 0xbbbb_aaaa);
}

// ---------------------------------------------------------------------------
// SOP1

fn src64_sop1(op: Sop1Op) -> bool {
    use Sop1Op::*;
    matches!(
        op,
        MovB64
            | CmovB64
            | NotB64
            | Bcnt1I32B64
            | Ff1I32B64
            | AndSaveexecB64
            | OrSaveexecB64
            | XorSaveexecB64
            | Andn2SaveexecB64
    )
}
fn dst64_sop1(op: Sop1Op) -> bool {
    use Sop1Op::*;
    matches!(
        op,
        MovB64
            | CmovB64
            | NotB64
            | AndSaveexecB64
            | OrSaveexecB64
            | XorSaveexecB64
            | Andn2SaveexecB64
    )
}

/// `(new D at the destination's width or None = untouched, new SCC or None, new EXEC or None)`.
fn ref_sop1(
    op: Sop1Op,
    a: u64,
    old: u64,
    scc: bool,
    exec: u64,
) -> (Option<u64>, Option<bool>, Option<u64>) {
    use Sop1Op::*;
    let a32 = a as u32;
    let nz = |r: u64| Some(r != 0);
    let m1 = u64::from(u32::MAX); // the "-1" of the find-bit ops, as a dword
    match op {
        MovB32 => (Some(u64::from(a32)), None, None),
        MovB64 => (Some(a), None, None),
        CmovB32 => (scc.then_some(u64::from(a32)), None, None),
        CmovB64 => (scc.then_some(a), None, None),
        NotB32 => {
            let r = u64::from(!a32);
            (Some(r), nz(r), None)
        }
        NotB64 => {
            let r = !a;
            (Some(r), nz(r), None)
        }
        BrevB32 => {
            let mut r = 0u32;
            for i in 0..32 {
                r |= (a32 >> i & 1) << (31 - i)
            }
            (Some(u64::from(r)), None, None)
        }
        Bcnt1I32B32 => {
            let n = (0..32).filter(|i| a32 >> i & 1 == 1).count() as u64;
            (Some(n), nz(n), None)
        }
        Bcnt1I32B64 => {
            let n = (0..64).filter(|i| a >> i & 1 == 1).count() as u64;
            (Some(n), nz(n), None)
        }
        Ff1I32B32 => (
            Some((0..32u32).find(|i| a32 >> i & 1 == 1).map_or(m1, u64::from)),
            None,
            None,
        ),
        Ff1I32B64 => (
            Some((0..64u32).find(|i| a >> i & 1 == 1).map_or(m1, u64::from)),
            None,
            None,
        ),
        FlbitI32B32 => (
            Some(
                (0..32u32)
                    .find(|i| a32 >> (31 - i) & 1 == 1)
                    .map_or(m1, u64::from),
            ),
            None,
            None,
        ),
        SextI32I8 => (
            Some(u64::from(if a32 & 0x80 != 0 {
                a32 & 0xff | 0xffff_ff00
            } else {
                a32 & 0xff
            })),
            None,
            None,
        ),
        SextI32I16 => (
            Some(u64::from(if a32 & 0x8000 != 0 {
                a32 & 0xffff | 0xffff_0000
            } else {
                a32 & 0xffff
            })),
            None,
            None,
        ),
        AndSaveexecB64 | OrSaveexecB64 | XorSaveexecB64 | Andn2SaveexecB64 => {
            let n = match op {
                AndSaveexecB64 => a & exec,
                OrSaveexecB64 => a | exec,
                XorSaveexecB64 => a ^ exec,
                _ => a & !exec,
            };
            (Some(exec), Some(n != 0), Some(n))
        }
        AbsI32 => {
            let r = (i64::from(a32 as i32).abs() as u64) & M32;
            (Some(r), nz(r), None)
        }
        Bitset0B32 => (
            Some(u64::from(old as u32 & !(1u32 << (a32 & 31)))),
            None,
            None,
        ),
        Bitset1B32 => (
            Some(u64::from(old as u32 | (1u32 << (a32 & 31)))),
            None,
            None,
        ),
        GetpcB64 | SetpcB64 => unreachable!("not modelled: checked separately"),
    }
}

fn check_sop1(op: Sop1Op) {
    use Sop1Op::*;
    if matches!(op, GetpcB64 | SetpcB64) {
        let mut w = fresh(0x1000, 0, 0, false);
        let i = if op == GetpcB64 {
            sop1(op, Some(sg(SD)), None)
        } else {
            sop1(op, None, Some(sg(SA)))
        };
        match step(&mut w, &i) {
            Err(SimError::Unsupported(m)) => {
                assert!(m.contains("GetpcB64") || m.contains("SetpcB64"), "{m}")
            }
            other => panic!("{op:?}: expected Unsupported, got {other:?}"),
        }
        return;
    }
    let mut rng = Rng(0xc0ff_ee00_dead_beef ^ (op as u64 + 1));
    for i in 0..ITERS {
        let scc = i & 1 == 1;
        let (mut a, old, exec) = (rng.v64(), rng.v64(), rng.v64());
        if !src64_sop1(op) {
            a = u64::from(rng.v32()) | a << 32
        }
        let mut w = fresh(a, 0, old, scc);
        w.exec = exec;
        let f = step(&mut w, &sop1(op, Some(sg(SD)), Some(sg(SA)))).unwrap();
        assert_eq!(f, Flow::Next);
        let (r, s, e) = ref_sop1(op, a, old, scc, exec);
        let want = match r {
            None => old,
            Some(r) if dst64_sop1(op) => r,
            Some(r) => old & !M32 | r & M32,
        };
        let ctx = format!("{op:?} a={a:#x} old={old:#x} scc={scc} exec={exec:#x}");
        assert_eq!(d64(&w), want, "{ctx}: D");
        assert_eq!(w.scc, s.unwrap_or(scc), "{ctx}: SCC");
        assert_eq!(w.exec, e.unwrap_or(exec), "{ctx}: EXEC");
        assert_eq!(
            [w.s[0], w.s[1]],
            [a as u32, (a >> 32) as u32],
            "{ctx}: source clobbered"
        );
    }
}

macro_rules! sop1_tests {
    ($($name:ident => $op:ident),+ $(,)?) => { $(#[test] fn $name() { check_sop1(Sop1Op::$op) })+ };
}
sop1_tests! {
    s_mov_b32 => MovB32, s_mov_b64 => MovB64, s_cmov_b32 => CmovB32, s_cmov_b64 => CmovB64,
    s_not_b32 => NotB32, s_not_b64 => NotB64, s_brev_b32 => BrevB32, s_bcnt1_i32_b32 => Bcnt1I32B32,
    s_bcnt1_i32_b64 => Bcnt1I32B64, s_ff1_i32_b32 => Ff1I32B32, s_ff1_i32_b64 => Ff1I32B64,
    s_flbit_i32_b32 => FlbitI32B32, s_sext_i32_i8 => SextI32I8, s_sext_i32_i16 => SextI32I16,
    s_and_saveexec_b64 => AndSaveexecB64, s_or_saveexec_b64 => OrSaveexecB64, s_xor_saveexec_b64 => XorSaveexecB64,
    s_andn2_saveexec_b64 => Andn2SaveexecB64, s_getpc_b64 => GetpcB64, s_setpc_b64 => SetpcB64,
    s_abs_i32 => AbsI32, s_bitset0_b32 => Bitset0B32, s_bitset1_b32 => Bitset1B32,
}

#[test]
fn sop1_all_ops_covered() {
    for &op in Sop1Op::ALL {
        check_sop1(op)
    }
}

fn run1(op: Sop1Op, a: u64) -> Wave {
    let mut w = fresh(a, 0, 0, false);
    step(&mut w, &sop1(op, Some(sg(SD)), Some(sg(SA)))).unwrap();
    w
}

#[test]
fn sop1_documented_examples() {
    // CDNA3 ISA §12.3 "Functional examples".
    assert_eq!(run1(Sop1Op::Bcnt1I32B32, 0).s[4], 0);
    assert_eq!(run1(Sop1Op::Bcnt1I32B32, 0xcccc_cccc).s[4], 16);
    assert_eq!(run1(Sop1Op::Bcnt1I32B32, 0xffff_ffff).s[4], 32);
    assert_eq!(run1(Sop1Op::Bcnt1I32B64, u64::MAX).s[4], 64);
    for (a, want) in [
        (0xaaaa_aaaau32, 1u32),
        (0x5555_5555, 0),
        (0, 0xffff_ffff),
        (0xffff_ffff, 0),
        (0x0001_0000, 16),
    ] {
        assert_eq!(
            run1(Sop1Op::Ff1I32B32, u64::from(a)).s[4],
            want,
            "ff1 {a:#x}"
        );
    }
    assert_eq!(run1(Sop1Op::Ff1I32B64, 1 << 40).s[4], 40);
    assert_eq!(run1(Sop1Op::Ff1I32B64, 0).s[4], 0xffff_ffff);
    for (a, want) in [
        (0u32, 0xffff_ffffu32),
        (0x0000_cccc, 16),
        (0xffff_3333, 0),
        (0x7fff_ffff, 1),
        (0x8000_0000, 0),
        (0xffff_ffff, 0),
    ] {
        assert_eq!(
            run1(Sop1Op::FlbitI32B32, u64::from(a)).s[4],
            want,
            "flbit {a:#x}"
        );
    }
    for (a, want) in [
        (1u32, 1u32),
        (0x7fff_ffff, 0x7fff_ffff),
        (0x8000_0000, 0x8000_0000),
        (0x8000_0001, 0x7fff_ffff),
        (0x8000_0002, 0x7fff_fffe),
        (0xffff_ffff, 1),
    ] {
        let w = run1(Sop1Op::AbsI32, u64::from(a));
        assert_eq!((w.s[4], w.scc), (want, true), "abs {a:#x}");
    }
    let w = run1(Sop1Op::AbsI32, 0);
    assert_eq!((w.s[4], w.scc), (0, false));
}

#[test]
fn sop1_scc_effects() {
    // Writers: NOT, BCNT1, ABS (result != 0). Non-writers keep SCC.
    let mut w = fresh(0xffff_ffff, 0, 0, true);
    step(&mut w, &sop1(Sop1Op::NotB32, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!((w.s[4], w.scc), (0, false));
    let mut w = fresh(0, 0, 0, false);
    step(&mut w, &sop1(Sop1Op::NotB64, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!((d64(&w), w.scc), (u64::MAX, true));
    let mut w = fresh(0, 0, 0, true);
    step(
        &mut w,
        &sop1(Sop1Op::Bcnt1I32B64, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert!(!w.scc);
    for op in [
        Sop1Op::MovB32,
        Sop1Op::BrevB32,
        Sop1Op::Ff1I32B32,
        Sop1Op::FlbitI32B32,
        Sop1Op::SextI32I8,
        Sop1Op::SextI32I16,
        Sop1Op::Bitset0B32,
        Sop1Op::Bitset1B32,
    ] {
        for scc in [false, true] {
            let mut w = fresh(0x1234_5678, 0, 0, scc);
            step(&mut w, &sop1(op, Some(sg(SD)), Some(sg(SA)))).unwrap();
            assert_eq!(w.scc, scc, "{op:?} must not touch SCC");
        }
    }
}

#[test]
fn sop1_misc_values() {
    assert_eq!(run1(Sop1Op::BrevB32, 1).s[4], 0x8000_0000);
    assert_eq!(run1(Sop1Op::BrevB32, 0x0000_00f1).s[4], 0x8f00_0000);
    assert_eq!(run1(Sop1Op::SextI32I8, 0x1234_5680).s[4], 0xffff_ff80);
    assert_eq!(run1(Sop1Op::SextI32I8, 0xffff_ff7f).s[4], 0x7f);
    assert_eq!(run1(Sop1Op::SextI32I16, 0x1234_8000).s[4], 0xffff_8000);
    assert_eq!(run1(Sop1Op::SextI32I16, 0xffff_7fff).s[4], 0x7fff);
    // 32-bit destination ops leave the next dword alone; 64-bit ops write both.
    let mut w = fresh(0x1_0000_0003, 0, 0xaaaa_bbbb_cccc_dddd, false);
    step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!(d64(&w), 0xaaaa_bbbb_0000_0003);
    step(&mut w, &sop1(Sop1Op::MovB64, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!(d64(&w), 0x1_0000_0003);
    // CMOV only when SCC is set.
    let mut w = fresh(5, 0, 9, false);
    step(&mut w, &sop1(Sop1Op::CmovB32, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!(w.s[4], 9);
    let mut w = fresh(5, 0, 9, true);
    step(&mut w, &sop1(Sop1Op::CmovB32, Some(sg(SD)), Some(sg(SA)))).unwrap();
    assert_eq!(w.s[4], 5);
    // BITSET0/1 read-modify-write the destination; the index is S0[4:0].
    let mut w = fresh(33, 0, 0xffff_ffff, false);
    step(
        &mut w,
        &sop1(Sop1Op::Bitset0B32, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert_eq!(w.s[4], 0xffff_fffd);
    let mut w = fresh(31, 0, 0, false);
    step(
        &mut w,
        &sop1(Sop1Op::Bitset1B32, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert_eq!(w.s[4], 0x8000_0000);
}

#[test]
fn sop1_saveexec_if_else_pattern() {
    // s_and_saveexec_b64 s[4:5], vcc: the classic `if` prologue.
    let mut w = fresh(0, 0, 0, true);
    w.exec = 0xffff_0000_ffff_0000;
    w.vcc = 0x0f0f_0f0f_0f0f_0f0f;
    let i = sop1(Sop1Op::AndSaveexecB64, Some(sg(SD)), Some(Src::VccLo));
    step(&mut w, &i).unwrap();
    assert_eq!(d64(&w), 0xffff_0000_ffff_0000);
    assert_eq!(w.exec, 0x0f0f_0000_0f0f_0000);
    assert!(w.scc);
    // Empty result clears SCC; saved mask is still the original.
    let mut w = fresh(0, 0, 0, true);
    w.exec = 0xf0;
    w.vcc = 0x0f;
    step(&mut w, &i).unwrap();
    assert_eq!((d64(&w), w.exec, w.scc), (0xf0, 0, false));
    // s_xor_saveexec_b64 (else) restores the complement within the saved mask.
    let mut w = fresh(0xff, 0, 0, false);
    w.exec = 0x0f;
    step(
        &mut w,
        &sop1(Sop1Op::XorSaveexecB64, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert_eq!((d64(&w), w.exec, w.scc), (0x0f, 0xf0, true));
    // s_or_saveexec_b64 s[4:5], s[0:1] with S0 = all ones.
    let mut w = fresh(u64::MAX, 0, 0, false);
    w.exec = 0x1234;
    step(
        &mut w,
        &sop1(Sop1Op::OrSaveexecB64, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert_eq!((d64(&w), w.exec, w.scc), (0x1234, u64::MAX, true));
    // s_andn2_saveexec_b64: EXEC = S0 & ~EXEC (ISA text).
    let mut w = fresh(0xff, 0, 0, false);
    w.exec = 0x0f;
    step(
        &mut w,
        &sop1(Sop1Op::Andn2SaveexecB64, Some(sg(SD)), Some(sg(SA))),
    )
    .unwrap();
    assert_eq!((d64(&w), w.exec, w.scc), (0x0f, 0xf0, true));
    // Saving into EXEC's own pair or a ttmp pair.
    let mut w = fresh(0, 0, 0, false);
    w.exec = 0xdead_beef_0000_0001;
    step(
        &mut w,
        &sop1(
            Sop1Op::AndSaveexecB64,
            Some(Src::Ttmp(2)),
            Some(Src::ExecLo),
        ),
    )
    .unwrap();
    assert_eq!(
        (w.ttmp[2], w.ttmp[3], w.exec),
        (1, 0xdead_beef, 0xdead_beef_0000_0001)
    );
}

// ---------------------------------------------------------------------------
// SOPK

fn ref_sopk(op: SopkOp, d: u32, imm: u16, scc: bool) -> (Option<u32>, Option<bool>) {
    use SopkOp::*;
    let sx = i32::from(imm as i16);
    let zx = u32::from(imm);
    let (di, du) = (d as i32, d);
    match op {
        MovkI32 => (Some(sx as u32), None),
        CmovkI32 => (scc.then_some(sx as u32), None),
        CmpkEqI32 => (None, Some(di == sx)),
        CmpkLgI32 => (None, Some(di != sx)),
        CmpkGtI32 => (None, Some(di > sx)),
        CmpkGeI32 => (None, Some(di >= sx)),
        CmpkLtI32 => (None, Some(di < sx)),
        CmpkLeI32 => (None, Some(di <= sx)),
        CmpkEqU32 => (None, Some(du == zx)),
        CmpkLgU32 => (None, Some(du != zx)),
        CmpkGtU32 => (None, Some(du > zx)),
        CmpkGeU32 => (None, Some(du >= zx)),
        CmpkLtU32 => (None, Some(du < zx)),
        CmpkLeU32 => (None, Some(du <= zx)),
        AddkI32 => {
            let t = i64::from(di) + i64::from(sx);
            (Some(t as u32), Some(t != i64::from(t as i32)))
        }
        MulkI32 => (Some((i64::from(di) * i64::from(sx)) as u32), None),
        GetregB32 | SetregB32 => unreachable!("checked in sopk_getreg_setreg_mode"),
    }
}

fn check_sopk(op: SopkOp) {
    use SopkOp::*;
    if matches!(op, GetregB32 | SetregB32) {
        sopk_getreg_setreg_mode();
        return;
    }
    let mut rng = Rng(0x5eed_5eed_1234_5678 ^ (op as u64 + 1));
    for i in 0..ITERS {
        let scc = i & 1 == 1;
        let imm = match rng.next() % 5 {
            0 => 0,
            1 => 0x7fff,
            2 => 0x8000,
            3 => 0xffff,
            _ => rng.next() as u16,
        };
        let d = match rng.next() % 4 {
            0 => i32::from(imm as i16) as u32, // equality with the sign-extended immediate
            1 => u32::from(imm),               // equality with the zero-extended immediate
            2 => i32::from(imm as i16).wrapping_add((rng.next() % 3) as i32 - 1) as u32,
            _ => rng.v32(),
        };
        let mut w = fresh(0, 0, u64::from(d) | 0x7777_7777 << 32, scc);
        let f = step(&mut w, &sopk(op, sg(SD), imm)).unwrap();
        assert_eq!(f, Flow::Next);
        let (r, s) = ref_sopk(op, d, imm, scc);
        let ctx = format!("{op:?} d={d:#x} imm={imm:#x} scc={scc}");
        assert_eq!(w.s[4], r.unwrap_or(d), "{ctx}: D");
        assert_eq!(w.s[5], 0x7777_7777, "{ctx}: next dword");
        assert_eq!(w.scc, s.unwrap_or(scc), "{ctx}: SCC");
    }
}

macro_rules! sopk_tests {
    ($($name:ident => $op:ident),+ $(,)?) => { $(#[test] fn $name() { check_sopk(SopkOp::$op) })+ };
}
sopk_tests! {
    s_movk_i32 => MovkI32, s_cmovk_i32 => CmovkI32, s_cmpk_eq_i32 => CmpkEqI32, s_cmpk_lg_i32 => CmpkLgI32,
    s_cmpk_gt_i32 => CmpkGtI32, s_cmpk_ge_i32 => CmpkGeI32, s_cmpk_lt_i32 => CmpkLtI32, s_cmpk_le_i32 => CmpkLeI32,
    s_cmpk_eq_u32 => CmpkEqU32, s_cmpk_lg_u32 => CmpkLgU32, s_cmpk_gt_u32 => CmpkGtU32, s_cmpk_ge_u32 => CmpkGeU32,
    s_cmpk_lt_u32 => CmpkLtU32, s_cmpk_le_u32 => CmpkLeU32, s_addk_i32 => AddkI32, s_mulk_i32 => MulkI32,
    s_getreg_b32 => GetregB32, s_setreg_b32 => SetregB32,
}

#[test]
fn sopk_all_ops_covered() {
    for &op in SopkOp::ALL {
        check_sopk(op)
    }
}

#[test]
fn sopk_edges() {
    // MOVK sign-extends; CMPK_*_U32 zero-extends.
    let mut w = fresh(0, 0, 0, false);
    step(&mut w, &sopk(SopkOp::MovkI32, sg(SD), 0x8000)).unwrap();
    assert_eq!(w.s[4], 0xffff_8000);
    w.s[4] = 0xffff_8000;
    step(&mut w, &sopk(SopkOp::CmpkEqU32, sg(SD), 0x8000)).unwrap();
    assert!(!w.scc);
    step(&mut w, &sopk(SopkOp::CmpkEqI32, sg(SD), 0x8000)).unwrap();
    assert!(w.scc);
    step(&mut w, &sopk(SopkOp::CmpkGtU32, sg(SD), 0xffff)).unwrap();
    assert!(w.scc);
    // ADDK overflow: 0x7fffffff + 1.
    let mut w = fresh(0, 0, 0x7fff_ffff, false);
    step(&mut w, &sopk(SopkOp::AddkI32, sg(SD), 1)).unwrap();
    assert_eq!((w.s[4], w.scc), (0x8000_0000, true));
    let mut w = fresh(0, 0, 0x8000_0000, true);
    step(&mut w, &sopk(SopkOp::AddkI32, sg(SD), 0xffff)).unwrap(); // + (-1)
    assert_eq!((w.s[4], w.scc), (0x7fff_ffff, true));
    let mut w = fresh(0, 0, 5, true);
    step(&mut w, &sopk(SopkOp::AddkI32, sg(SD), 0xfffb)).unwrap(); // + (-5)
    assert_eq!((w.s[4], w.scc), (0, false));
    // MULK keeps the low 32 bits.
    let mut w = fresh(0, 0, 0x0001_0001, false);
    step(&mut w, &sopk(SopkOp::MulkI32, sg(SD), 0xffff)).unwrap();
    assert_eq!(w.s[4], 0x0001_0001u32.wrapping_mul(0xffff_ffff));
    // CMOVK is conditional on SCC.
    let mut w = fresh(0, 0, 9, false);
    step(&mut w, &sopk(SopkOp::CmovkI32, sg(SD), 0xfffe)).unwrap();
    assert_eq!(w.s[4], 9);
    w.scc = true;
    step(&mut w, &sopk(SopkOp::CmovkI32, sg(SD), 0xfffe)).unwrap();
    assert_eq!(w.s[4], 0xffff_fffe);
}

fn hwreg(id: u16, off: u16, size: u16) -> u16 {
    id | off << 6 | (size - 1) << 11
}
const MODE_ID: u16 = 1;

fn getreg(w: &mut Wave, simm: u16) -> SimResult<u32> {
    step(w, &sopk(SopkOp::GetregB32, sg(SD), simm))?;
    Ok(w.s[4])
}
fn setreg(w: &mut Wave, simm: u16, v: u32) -> SimResult<()> {
    w.s[4] = v;
    step(w, &sopk(SopkOp::SetregB32, sg(SD), simm)).map(|_| ())
}

fn sopk_getreg_setreg_mode() {
    // Default MODE: round nearest (0), denorms preserved (3,3), DX10_CLAMP, IEEE.
    let mut w = fresh(0, 0, 0, false);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 0, 10)).unwrap(), 0x3f0);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 4, 4)).unwrap(), 0xf);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 8, 2)).unwrap(), 3);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 23, 1)).unwrap(), 0);
    // SETREG of sub-fields maps onto `Mode` and leaves the rest alone.
    setreg(&mut w, hwreg(MODE_ID, 0, 2), 0b1_01).unwrap(); // only 2 bits are taken: 01 -> round32 = +inf
    assert_eq!(
        (
            w.mode.round32,
            w.mode.round16_64,
            w.mode.denorm32,
            w.mode.denorm16_64
        ),
        (1, 0, 3, 3)
    );
    setreg(&mut w, hwreg(MODE_ID, 2, 2), 3).unwrap();
    assert_eq!((w.mode.round32, w.mode.round16_64), (1, 3));
    setreg(&mut w, hwreg(MODE_ID, 4, 2), 0).unwrap();
    setreg(&mut w, hwreg(MODE_ID, 6, 2), 2).unwrap();
    assert_eq!((w.mode.denorm32, w.mode.denorm16_64), (0, 2));
    setreg(&mut w, hwreg(MODE_ID, 8, 1), 0).unwrap();
    assert!(!w.mode.dx10_clamp && w.mode.ieee);
    setreg(&mut w, hwreg(MODE_ID, 9, 1), 0).unwrap();
    assert!(!w.mode.ieee);
    setreg(&mut w, hwreg(MODE_ID, 23, 1), 1).unwrap();
    assert!(w.mode.fp16_ovfl);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 0, 10)).unwrap(), 0x8d);
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 23, 1)).unwrap(), 1);
    // Offset 0, 10 bits, in one write.
    setreg(&mut w, hwreg(MODE_ID, 0, 10), 0x3a5).unwrap();
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 0, 10)).unwrap(), 0x3a5);
    assert!(w.mode.fp16_ovfl, "bit 23 untouched by a [9:0] write");
    // Source bits above the field are dropped.
    setreg(&mut w, hwreg(MODE_ID, 4, 4), 0xffff_ff00).unwrap();
    assert_eq!(getreg(&mut w, hwreg(MODE_ID, 0, 10)).unwrap(), 0x305);
    // Randomised: a reference MODE word over the modelled bits.
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let mut model = 0x305u32 | 1 << 23;
    for _ in 0..ITERS {
        let (off, size) = if rng.next() & 1 == 0 {
            let off = (rng.next() % 10) as u16;
            (off, (rng.next() % u64::from(10 - off)) as u16 + 1)
        } else {
            (23, 1)
        };
        let v = rng.v32();
        let mask = (if size == 32 {
            u32::MAX
        } else {
            (1u32 << size) - 1
        }) << off;
        model = model & !mask | (v << off) & mask;
        setreg(&mut w, hwreg(MODE_ID, off, size), v).unwrap();
        let lo = getreg(&mut w, hwreg(MODE_ID, 0, 10)).unwrap();
        let hi = getreg(&mut w, hwreg(MODE_ID, 23, 1)).unwrap();
        assert_eq!(
            (lo, hi),
            (model & 0x3ff, model >> 23 & 1),
            "after setreg(off={off}, size={size}, {v:#x})"
        );
        let part = getreg(&mut w, hwreg(MODE_ID, off, size)).unwrap();
        assert_eq!(part, (model >> off) & (mask >> off));
    }
    // Registers and MODE bits that are not modelled are Unsupported, never approximated.
    for (simm, what) in [
        (hwreg(2, 0, 32), "STATUS"),
        (hwreg(3, 0, 8), "TRAPSTS"),
        (hwreg(4, 0, 4), "HW_ID"),
        (hwreg(20, 0, 4), "XCC_ID"),
        (hwreg(MODE_ID, 10, 1), "MODE bit 10"),
        (hwreg(MODE_ID, 0, 32), "all of MODE"),
        (hwreg(MODE_ID, 12, 7), "EXCP_EN"),
        (hwreg(MODE_ID, 28, 1), "VSKIP"),
        (hwreg(MODE_ID, 29, 3), "CSP"),
        (hwreg(MODE_ID, 8, 3), "bits 8..10"),
    ] {
        let mut w = fresh(0, 0, 0, false);
        assert!(
            matches!(
                step(&mut w, &sopk(SopkOp::GetregB32, sg(SD), simm)),
                Err(SimError::Unsupported(_))
            ),
            "getreg {what}"
        );
        let before = w.mode;
        assert!(
            matches!(
                step(&mut w, &sopk(SopkOp::SetregB32, sg(SD), simm)),
                Err(SimError::Unsupported(_))
            ),
            "setreg {what}"
        );
        assert_eq!(w.mode, before, "setreg {what} must not modify MODE");
    }
}

#[test]
fn sopk_getreg_setreg_mode_test() {
    sopk_getreg_setreg_mode()
}

// ---------------------------------------------------------------------------
// SOPC

fn is64_sopc(op: SopcOp) -> bool {
    matches!(op, SopcOp::CmpEqU64 | SopcOp::CmpLgU64)
}

fn ref_sopc(op: SopcOp, a: u64, b: u64) -> bool {
    use SopcOp::*;
    let (a32, b32) = (a as u32, b as u32);
    match op {
        CmpEqI32 => a32 as i32 == b32 as i32,
        CmpLgI32 => a32 as i32 != b32 as i32,
        CmpGtI32 => a32 as i32 > b32 as i32,
        CmpGeI32 => a32 as i32 >= b32 as i32,
        CmpLtI32 => (a32 as i32) < b32 as i32,
        CmpLeI32 => a32 as i32 <= b32 as i32,
        CmpEqU32 => a32 == b32,
        CmpLgU32 => a32 != b32,
        CmpGtU32 => a32 > b32,
        CmpGeU32 => a32 >= b32,
        CmpLtU32 => a32 < b32,
        CmpLeU32 => a32 <= b32,
        Bitcmp0B32 => (a32 >> (b32 % 32)) & 1 == 0,
        Bitcmp1B32 => (a32 >> (b32 % 32)) & 1 == 1,
        CmpEqU64 => a == b,
        CmpLgU64 => a != b,
    }
}

fn check_sopc(op: SopcOp) {
    let mut rng = Rng(0xabad_cafe_f00d_1111 ^ (op as u64 + 1));
    for i in 0..ITERS {
        let scc_in = i & 1 == 1;
        let (mut a, mut b) = (rng.v64(), rng.v64());
        if rng.next() % 4 == 0 {
            b = a
        }
        // equal operands
        else if rng.next() % 4 == 0 {
            b = a.wrapping_add(rng.next() % 3).wrapping_sub(1)
        }
        if !is64_sopc(op) {
            a = u64::from(a as u32) | rng.next() << 32; // junk in the high dwords
            b = u64::from(b as u32) | rng.next() << 32;
            if rng.next() % 4 == 0 {
                b = b & !M32 | u64::from(rng.next() as u32 % 70)
            } // bit indices up to 69
        }
        let (a_, b_) = (a, b);
        let mut w = fresh(a_, b_, 0x1234, scc_in);
        let f = step(&mut w, &sopc(op, sg(SA), sg(SB))).unwrap();
        assert_eq!(f, Flow::Next);
        assert_eq!(w.scc, ref_sopc(op, a, b), "{op:?} a={a:#x} b={b:#x}");
        assert_eq!(
            (w.s[4], w.s[5]),
            (0x1234, 0),
            "{op:?}: SOPC writes no register"
        );
    }
}

macro_rules! sopc_tests {
    ($($name:ident => $op:ident),+ $(,)?) => { $(#[test] fn $name() { check_sopc(SopcOp::$op) })+ };
}
sopc_tests! {
    s_cmp_eq_i32 => CmpEqI32, s_cmp_lg_i32 => CmpLgI32, s_cmp_gt_i32 => CmpGtI32, s_cmp_ge_i32 => CmpGeI32,
    s_cmp_lt_i32 => CmpLtI32, s_cmp_le_i32 => CmpLeI32, s_cmp_eq_u32 => CmpEqU32, s_cmp_lg_u32 => CmpLgU32,
    s_cmp_gt_u32 => CmpGtU32, s_cmp_ge_u32 => CmpGeU32, s_cmp_lt_u32 => CmpLtU32, s_cmp_le_u32 => CmpLeU32,
    s_bitcmp0_b32 => Bitcmp0B32, s_bitcmp1_b32 => Bitcmp1B32, s_cmp_eq_u64 => CmpEqU64, s_cmp_lg_u64 => CmpLgU64,
}

#[test]
fn sopc_all_ops_covered() {
    for &op in SopcOp::ALL {
        check_sopc(op)
    }
}

#[test]
fn sopc_edges() {
    let cmp = |op, a: u64, b: u64| {
        let mut w = fresh(a, b, 0, false);
        step(&mut w, &sopc(op, sg(SA), sg(SB))).unwrap();
        w.scc
    };
    assert!(cmp(SopcOp::CmpLtI32, 0xffff_ffff, 0)); // -1 < 0
    assert!(!cmp(SopcOp::CmpLtU32, 0xffff_ffff, 0));
    assert!(cmp(SopcOp::CmpGtU32, 0x8000_0000, 0x7fff_ffff));
    assert!(!cmp(SopcOp::CmpGtI32, 0x8000_0000, 0x7fff_ffff));
    assert!(cmp(SopcOp::CmpLeI32, 7, 7) && cmp(SopcOp::CmpGeU32, 7, 7));
    // The index of BITCMP is S1[4:0]: 32 selects bit 0.
    assert!(cmp(SopcOp::Bitcmp1B32, 1, 32));
    assert!(!cmp(SopcOp::Bitcmp0B32, 1, 32));
    assert!(cmp(SopcOp::Bitcmp1B32, 0x8000_0000, 31));
    assert!(cmp(SopcOp::Bitcmp0B32, 0x8000_0000, 30));
    // 64-bit compares look at both dwords.
    assert!(!cmp(SopcOp::CmpEqU64, 0x1_0000_0000, 0));
    assert!(cmp(SopcOp::CmpLgU64, 0x1_0000_0000, 0));
    assert!(cmp(
        SopcOp::CmpEqU64,
        0x1234_5678_9abc_def0,
        0x1234_5678_9abc_def0
    ));
    // 32-bit compares ignore the high dwords.
    assert!(cmp(SopcOp::CmpEqU32, 0x1_0000_0005, 0x2_0000_0005));
}

// ---------------------------------------------------------------------------
// SOPP

#[test]
fn sopp_branches_taken_and_not_taken() {
    use SoppOp::*;
    // (op, SCC, VCC, EXEC) -> taken?
    let cases: [(SoppOp, bool, u64, u64, bool); 20] = [
        (Branch, false, 0, 0, true),
        (Branch, true, 1, 1, true),
        (CbranchScc0, false, 1, 1, true),
        (CbranchScc0, true, 0, 0, false),
        (CbranchScc1, true, 0, 0, true),
        (CbranchScc1, false, 1, 1, false),
        (CbranchVccz, false, 0, 1, true),
        (CbranchVccz, true, 1, 1, false),
        (CbranchVccz, true, 1 << 63, 1, false),
        (CbranchVccnz, false, 1 << 63, 0, true),
        (CbranchVccnz, true, 0, 1, false),
        (CbranchVccnz, false, 1, 0, true),
        (CbranchExecz, false, 1, 0, true),
        (CbranchExecz, true, 0, 1, false),
        (CbranchExecz, true, 0, 1 << 63, false),
        (CbranchExecnz, false, 0, 1 << 32, true),
        (CbranchExecnz, true, 1, 0, false),
        (CbranchExecnz, false, 0, u64::MAX, true),
        (CbranchScc0, false, 0, 0, true),
        (CbranchExecz, false, 0, 0, true),
    ];
    for (op, scc, vcc, exec, taken) in cases {
        for imm in [0u16, 1, 7, 0x7fff, 0xfff8, 0xffff, 0x8000] {
            let mut w = fresh(0, 0, 0, scc);
            w.vcc = vcc;
            w.exec = exec;
            let f = step(&mut w, &sopp(op, imm)).unwrap();
            let want = if taken {
                Flow::Branch(imm as i16)
            } else {
                Flow::Next
            };
            assert_eq!(
                f, want,
                "{op:?} scc={scc} vcc={vcc:#x} exec={exec:#x} imm={imm:#x}"
            );
            assert_eq!(
                (w.scc, w.vcc, w.exec),
                (scc, vcc, exec),
                "branches read state only"
            );
        }
    }
    // Negative offsets come back as negative dword counts.
    let mut w = fresh(0, 0, 0, false);
    assert_eq!(
        step(&mut w, &sopp(Branch, 0xfff8)).unwrap(),
        Flow::Branch(-8)
    );
}

#[test]
fn sopp_vccz_execz_use_all_64_bits() {
    let mut w = fresh(0, 0, 0, false);
    w.vcc = 1 << 40;
    assert_eq!(
        step(&mut w, &sopp(SoppOp::CbranchVccz, 3)).unwrap(),
        Flow::Next
    );
    assert_eq!(
        step(&mut w, &sopp(SoppOp::CbranchVccnz, 3)).unwrap(),
        Flow::Branch(3)
    );
    w.exec = 1 << 40;
    assert_eq!(
        step(&mut w, &sopp(SoppOp::CbranchExecz, 3)).unwrap(),
        Flow::Next
    );
    assert_eq!(
        step(&mut w, &sopp(SoppOp::CbranchExecnz, 3)).unwrap(),
        Flow::Branch(3)
    );
}

#[test]
fn sopp_nop_sleep_setprio_icache_are_noops() {
    for (op, imm) in [
        (SoppOp::Nop, 0u16),
        (SoppOp::Nop, 0xf),
        (SoppOp::Sleep, 0),
        (SoppOp::Sleep, 0x7f),
        (SoppOp::Setprio, 3),
        (SoppOp::IcacheInv, 0),
    ] {
        let mut w = fresh(0x11, 0x22, 0x33, true);
        w.vcc = 5;
        w.exec = 6;
        w.m0 = 7;
        let before = (w.s.clone(), w.scc, w.vcc, w.exec, w.m0, w.mode, w.pc);
        assert_eq!(step(&mut w, &sopp(op, imm)).unwrap(), Flow::Next, "{op:?}");
        assert_eq!(
            (w.s.clone(), w.scc, w.vcc, w.exec, w.m0, w.mode, w.pc),
            before,
            "{op:?} must not change state"
        );
        assert!(w.pending.is_empty());
    }
}

#[test]
fn sopp_sendmsg_trap_unsupported() {
    for op in [SoppOp::Sendmsg, SoppOp::Trap] {
        let mut w = fresh(0, 0, 0, false);
        match step(&mut w, &sopp(op, 1)) {
            Err(SimError::Unsupported(m)) => assert!(m.contains(&format!("{op:?}")), "{m}"),
            other => panic!("{op:?}: expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
fn sopp_mod_handled_ops_are_not_the_alus_job() {
    // `exec_one` intercepts these; asking the scalar ALU directly is a Fault, not a silent no-op.
    for op in [SoppOp::Waitcnt, SoppOp::Barrier, SoppOp::Endpgm] {
        let mut w = fresh(0, 0, 0, false);
        assert!(
            matches!(salu::exec(&mut w, &sopp(op, 0)), Err(SimError::Fault(_))),
            "{op:?}"
        );
    }
    assert_eq!(
        step(&mut fresh(0, 0, 0, false), &sopp(SoppOp::Barrier, 0)).unwrap(),
        Flow::Barrier
    );
}

// ---------------------------------------------------------------------------
// Operand typing, special registers, in-flight checks

#[test]
fn inline_constants_and_literals() {
    let mut w = fresh(0, 0, 0, false);
    // 32-bit: inline integers sign-extend, floats are their f32 bits, literals raw.
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Int(-1))),
    )
    .unwrap();
    assert_eq!(w.s[4], 0xffff_ffff);
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Int(64))),
    )
    .unwrap();
    assert_eq!(w.s[4], 64);
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Int(-16))),
    )
    .unwrap();
    assert_eq!(w.s[4], 0xffff_fff0);
    step(
        &mut w,
        &sop1(
            Sop1Op::MovB32,
            Some(sg(SD)),
            Some(Src::Float(InlineFloat::One)),
        ),
    )
    .unwrap();
    assert_eq!(w.s[4], 0x3f80_0000);
    step(
        &mut w,
        &sop1(
            Sop1Op::MovB32,
            Some(sg(SD)),
            Some(Src::Float(InlineFloat::NegFour)),
        ),
    )
    .unwrap();
    assert_eq!(w.s[4], 0xc080_0000);
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Lit(0xdead_beef))),
    )
    .unwrap();
    assert_eq!(w.s[4], 0xdead_beef);
    // 64-bit: inline integers sign-extend to 64 bits.
    step(
        &mut w,
        &sop1(Sop1Op::MovB64, Some(sg(SD)), Some(Src::Int(-1))),
    )
    .unwrap();
    assert_eq!(d64(&w), u64::MAX);
    step(
        &mut w,
        &sop1(Sop1Op::MovB64, Some(sg(SD)), Some(Src::Int(5))),
    )
    .unwrap();
    assert_eq!(d64(&w), 5);
    // SOP2 with an inline constant and with a literal.
    let mut w = fresh(10, 0, 0, false);
    step(&mut w, &sop2(Sop2Op::AddU32, sg(SD), sg(SA), Src::Int(-1))).unwrap();
    assert_eq!((w.s[4], w.scc), (9, true));
    step(
        &mut w,
        &sop2(Sop2Op::AddU32, sg(SD), Src::Lit(0xffff_fff6), sg(SA)),
    )
    .unwrap();
    assert_eq!((w.s[4], w.scc), (0, true));
    step(
        &mut w,
        &sop2(Sop2Op::LshlB32, sg(SD), Src::Int(1), Src::Lit(35)),
    )
    .unwrap();
    assert_eq!(w.s[4], 8);
    // SOPC with inline and literal.
    step(
        &mut w,
        &sopc(SopcOp::CmpEqI32, Src::Int(-1), Src::Lit(0xffff_ffff)),
    )
    .unwrap();
    assert!(w.scc);
    step(&mut w, &sopc(SopcOp::CmpEqU64, Src::Int(-1), Src::Int(-1))).unwrap();
    assert!(w.scc);
}

#[test]
fn special_register_operands() {
    // SCC / VCCZ / EXECZ read as 0 or 1.
    let mut w = fresh(0, 0, 0, true);
    w.vcc = 0;
    w.exec = 5;
    step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Scc))).unwrap();
    assert_eq!(w.s[4], 1);
    step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Vccz))).unwrap();
    assert_eq!(w.s[4], 1);
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Execz)),
    )
    .unwrap();
    assert_eq!(w.s[4], 0);
    // Destinations: M0, VCC halves, EXEC pair, ttmp.
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(Src::M0), Some(Src::Lit(0x1234))),
    )
    .unwrap();
    assert_eq!(w.m0, 0x1234);
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(Src::VccHi), Some(Src::Lit(7))),
    )
    .unwrap();
    assert_eq!(w.vcc, 7 << 32);
    step(
        &mut w,
        &sop1(Sop1Op::MovB64, Some(Src::ExecLo), Some(Src::Lit(0xf0f0))),
    )
    .unwrap();
    assert_eq!(w.exec, 0xf0f0);
    step(
        &mut w,
        &sop2(Sop2Op::AndB64, Src::ExecLo, Src::ExecLo, Src::Lit(0x00ff)),
    )
    .unwrap();
    assert_eq!((w.exec, w.scc), (0x00f0, true));
    step(
        &mut w,
        &sop2(Sop2Op::AndB64, Src::VccLo, Src::ExecLo, Src::Int(0)),
    )
    .unwrap();
    assert_eq!((w.vcc, w.scc), (0, false));
    step(
        &mut w,
        &sop1(Sop1Op::MovB32, Some(Src::Ttmp(3)), Some(Src::Lit(99))),
    )
    .unwrap();
    assert_eq!(w.ttmp[3], 99);
    // CMOV reads its source even when not taken, and honours SCC as an operand.
    let mut w = fresh(0, 0, 0, true);
    step(
        &mut w,
        &sop2(Sop2Op::CselectB32, sg(SD), Src::Scc, Src::Int(2)),
    )
    .unwrap();
    assert_eq!(w.s[4], 1);
}

#[test]
fn bad_operands_fault() {
    let mut w = fresh(0, 0, 0, false);
    // 64-bit operand at an odd SGPR.
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB64, Some(sg(SD)), Some(sg(1)))),
        Err(SimError::Fault(_))
    ));
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB64, Some(sg(5)), Some(sg(0)))),
        Err(SimError::Fault(_))
    ));
    // A vector register is not a scalar operand.
    assert!(matches!(
        step(
            &mut w,
            &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(Src::Vgpr(0)))
        ),
        Err(SimError::Fault(_))
    ));
    // Missing source / destination.
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(SD)), None)),
        Err(SimError::Fault(_))
    ));
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB32, None, Some(sg(0)))),
        Err(SimError::Fault(_))
    ));
    // Beyond the SGPR file.
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(SD)), Some(sg(250)))),
        Err(SimError::Fault(_))
    ));
    // Non-scalar instructions are not the ALU's.
    let v = Inst::Vop1(peacemaker_cdna::enc::vop::Vop1 {
        op: peacemaker_cdna::enc::vop::Vop1Op::ALL[0],
        vdst: Src::Vgpr(0),
        src0: Src::Vgpr(1),
        ext: peacemaker_cdna::enc::vop::VopExt::None,
    });
    assert!(matches!(
        salu::exec(&mut w, &v),
        Err(SimError::Unsupported(_))
    ));
}

#[test]
fn in_flight_registers_are_checked() {
    // A load still writing s0 makes both reading and writing s0 a Wait error.
    let mut w = fresh(1, 2, 3, false);
    w.pending.push_back(Pending {
        id: 0,
        counter: Counter::Lgkm,
        out_of_order: true,
        weight: 1,
        busy: vec![RegSpan::new(RegFile::Sgpr, 0, 1)],
        what: "s_load_dword s0".into(),
        retire: Box::new(|_, _, _| Ok(())),
    });
    assert!(matches!(
        step(&mut w, &sop2(Sop2Op::AddU32, sg(SD), sg(0), sg(SB))),
        Err(SimError::Wait(_))
    ));
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB32, Some(sg(0)), Some(sg(SB)))),
        Err(SimError::Wait(_))
    ));
    assert!(matches!(
        step(&mut w, &sopc(SopcOp::CmpEqU32, sg(SB), sg(0))),
        Err(SimError::Wait(_))
    ));
    assert!(matches!(
        step(&mut w, &sopk(SopkOp::AddkI32, sg(0), 1)),
        Err(SimError::Wait(_))
    ));
    assert!(matches!(
        step(&mut w, &sop1(Sop1Op::MovB64, Some(sg(SD)), Some(sg(0)))),
        Err(SimError::Wait(_))
    ));
    // Unrelated registers are unaffected.
    step(&mut w, &sop2(Sop2Op::AddU32, sg(SD), sg(SB), sg(SB))).unwrap();
    assert_eq!(w.s[4], 4);
}

#[test]
fn sop2_dst_ops_run_to_exact_register_file() {
    // Every SOP2 op accepts an arbitrary SGPR destination (not just s4) and writes only its width.
    for &op in Sop2Op::ALL {
        let mut w = Wave::new(Arch::Gfx942, 0, 0);
        w.s[10] = 3;
        w.s[11] = 0;
        w.s[12] = 1;
        w.s[13] = 0;
        w.s[20] = 0xaaaa_aaaa;
        w.s[21] = 0xbbbb_bbbb;
        w.s[22] = 0xcccc_cccc;
        step(&mut w, &sop2(op, sg(20), sg(10), sg(12))).unwrap();
        assert_eq!(w.s[22], 0xcccc_cccc, "{op:?} wrote beyond its destination");
        if !is64_sop2(op) {
            assert_eq!(
                w.s[21], 0xbbbb_bbbb,
                "{op:?} wrote the high dword of a 32-bit destination"
            )
        }
    }
}
