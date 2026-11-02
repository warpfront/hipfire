// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vector-ALU emulator tests (`sim::valu::exec`): every VOP1/VOP2/VOPC/VOP3
//! opcode against an independent per-lane reference over randomized inputs
//! that include NaN, ±Inf, denormals and -0, with EXEC partially off, plus
//! edge cases from the CDNA3 ISA examples, modifier handling, DPP/SDWA forms,
//! and the f32 division macro.
use peacemaker_cdna::arch::{Arch, Fp8Encoding};
use peacemaker_cdna::enc::vop::{
    Dpp, DppCtrl, Sdwa, SdwaSel, SdwaUnused, Vop1, Vop1Op, Vop2, Vop2Op, VopExt, Vopc, VopcOp,
};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::float::{self, F8};
use peacemaker_cdna::inst::Inst;
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::sim::{valu, SimError, Wave};
use peacemaker_cdna::WAVE;
use std::collections::BTreeSet;

const POISON: u32 = 0xdead_beef;
const ROUNDS: usize = 24;

// ----------------------------------------------------------------------------
// Harness

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 16) as u32
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }
    /// A partial EXEC mask (never all-on/all-off), sometimes full.
    fn exec(&mut self) -> u64 {
        match self.next() % 6 {
            0 => u64::MAX,
            1 => 0x0000_ffff_0000_0001,
            _ => self.next() | 1,
        }
    }
}

fn wave() -> Wave {
    Wave::new(Arch::Gfx942, 0, POISON)
}

const F32_SPECIAL: &[u32] = &[
    0,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7fa0_0000,
    0xff80_0001,
    0x0000_0001,
    0x007f_ffff,
    0x8000_0001,
    0x807f_ffff,
    0x0080_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x3f00_0000,
    0xbf00_0000,
    0x4b00_0000,
    0x3eff_ffff,
    0x3f7f_ffff,
    0x4f80_0000,
    0xcf00_0000,
    0x4040_0000,
    0x3fc0_0000,
    0xbfc0_0000,
];
const F16_SPECIAL: &[u32] = &[
    0, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0xfe01, 0x0001, 0x03ff, 0x8001, 0x3c00, 0xbc00,
    0x7bff, 0xfbff, 0x0400, 0x3800,
];
const U32_SPECIAL: &[u32] = &[
    0,
    1,
    2,
    31,
    32,
    0xffff_ffff,
    0x8000_0000,
    0x7fff_ffff,
    0x00ff_ffff,
    0x0080_0000,
    0xffff,
    0x8000,
    0x7fff,
    0x1_0000,
];

#[derive(Clone, Copy, Debug)]
enum G {
    /// Any 32-bit integer.
    U,
    /// f32 bit pattern (specials, random bits, moderate values).
    F,
    /// f16 in the low half, random high half.
    H,
    /// Masked random bits.
    M(u32),
    /// Small signed integer for ldexp.
    Exp,
}

fn gen(r: &mut Rng, g: G) -> u32 {
    match g {
        G::U => {
            if r.next() % 4 == 0 {
                r.pick(U32_SPECIAL)
            } else {
                r.u32()
            }
        }
        G::F => match r.next() % 4 {
            0 => r.pick(F32_SPECIAL),
            1 => r.u32(),
            _ => {
                (r.u32() & 0x8000_0000)
                    | ((100 + r.next() % 56) as u32) << 23
                    | (r.u32() & 0x7f_ffff)
            }
        },
        G::H => {
            let lo = match r.next() % 3 {
                0 => r.pick(F16_SPECIAL),
                1 => r.u32() & 0xffff,
                _ => (r.u32() & 0x8000) | ((8 + r.next() % 16) as u32) << 10 | (r.u32() & 0x3ff),
            };
            (r.u32() & 0xffff_0000) | lo
        }
        G::M(m) => {
            if r.next() % 4 == 0 {
                r.pick(U32_SPECIAL) & m
            } else {
                r.u32() & m
            }
        }
        G::Exp => {
            let k = (r.next() % 600) as i32 - 300;
            r.pick(&[
                0u32,
                1,
                u32::MAX,
                64,
                (-64i32) as u32,
                127,
                (-149i32) as u32,
                300,
                (-300i32) as u32,
                i32::MIN as u32,
                i32::MAX as u32,
                k as u32,
            ])
        }
    }
}

fn exec_ok(w: &mut Wave, inst: &Inst) {
    if let Err(e) = valu::exec(w, inst) {
        panic!("{inst:?}: {e}")
    }
}

fn vop3(op: Vop3Op, dst: Src, srcs: &[Src]) -> Vop3 {
    let mut src = [None; 3];
    for (i, s) in srcs.iter().enumerate() {
        src[i] = Some(*s)
    }
    Vop3 {
        op,
        dst,
        sdst: None,
        src,
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    }
}
fn v1(op: Vop1Op, vdst: Src, src0: Src) -> Inst {
    Inst::Vop1(Vop1 {
        op,
        vdst,
        src0,
        ext: VopExt::None,
    })
}
fn v2(op: Vop2Op, vdst: Src, src0: Src, src1: Src) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst,
        src0,
        src1,
        ext: VopExt::None,
    })
}
fn vc(op: VopcOp, src0: Src, src1: Src) -> Inst {
    Inst::Vopc(Vopc {
        op,
        src0,
        src1,
        ext: VopExt::None,
    })
}
fn v3(v: Vop3) -> Inst {
    Inst::Vop3(v)
}
fn vg(n: u8) -> Src {
    Src::Vgpr(n)
}

fn name(op: Vop3Op) -> &'static str {
    match op {
        Vop3Op::C(o) => o.name(),
        Vop3Op::V2(o) => o.name(),
        Vop3Op::V1(o) => o.name(),
        Vop3Op::Only(o) => o.name(),
    }
}

type Ref = fn(&[u32; 3], usize) -> u32;

/// A 32-bit-result op over VGPR sources v10.. that is a pure function of the
/// lane's sources (and lane id).
struct Case {
    op: Vop3Op,
    g: &'static [G],
    r: Ref,
}

/// Run `c` in its VOP3 form, and in the e32 form for VOP1/VOP2, over
/// randomized lanes with partial EXEC; inactive lanes must keep the poison.
fn run_case(c: &Case, seed: u64) {
    let mut r = Rng(seed | 1);
    let n = c.g.len();
    let srcs: Vec<Src> = (0..n).map(|k| vg(10 + k as u8)).collect();
    let mut forms = vec![v3(vop3(c.op, vg(1), &srcs))];
    match c.op {
        Vop3Op::V1(o) => forms.push(v1(o, vg(1), srcs[0])),
        Vop3Op::V2(o) => forms.push(v2(o, vg(1), srcs[0], srcs[1])),
        _ => {}
    }
    for inst in &forms {
        for _ in 0..ROUNDS {
            let mut w = wave();
            w.exec = r.exec();
            let mut ins = vec![[0u32; 3]; WAVE];
            for l in 0..WAVE {
                for k in 0..n {
                    ins[l][k] = gen(&mut r, c.g[k]);
                    w.v[10 + k][l] = ins[l][k];
                }
            }
            let exec = w.exec;
            exec_ok(&mut w, inst);
            for l in 0..WAVE {
                let want = if exec >> l & 1 != 0 {
                    (c.r)(&ins[l], l)
                } else {
                    POISON
                };
                assert_eq!(
                    w.v[1][l],
                    want,
                    "{} ({inst:?}) lane {l} inputs {:08x?}: got {:#010x} want {want:#010x}",
                    name(c.op),
                    &ins[l][..n],
                    w.v[1][l]
                );
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Reference helpers (independent of the implementation's internals)

fn fv(b: u32) -> f32 {
    f32::from_bits(b)
}
fn nan(b: u32) -> bool {
    fv(b).is_nan()
}
fn snan(b: u32) -> bool {
    nan(b) && b & 0x40_0000 == 0
}
fn quiet(b: u32) -> u32 {
    b | 0x40_0000
}
/// First NaN input quieted, else the result (invalid → 0xffc00000).
fn ar(ins: &[u32], r: f32) -> u32 {
    for &b in ins {
        if nan(b) {
            return quiet(b);
        }
    }
    if r.is_nan() {
        0xffc0_0000
    } else {
        r.to_bits()
    }
}
fn hv(b: u32) -> f32 {
    float::f16_to_f32(b as u16)
}
fn hnan(b: u32) -> bool {
    b & 0x7fff > 0x7c00
}
fn har(a: u32, b: u32, r: f32) -> u32 {
    let (a, b) = (a & 0xffff, b & 0xffff);
    if hnan(a) {
        return a | 0x200;
    }
    if hnan(b) {
        return b | 0x200;
    }
    if r.is_nan() {
        0xfe00
    } else {
        u32::from(float::f32_to_f16_rne(r))
    }
}
fn denorm_flush(b: u32) -> u32 {
    if b & 0x7f80_0000 == 0 {
        b & 0x8000_0000
    } else {
        b
    }
}
/// Transcendental reference: f64 evaluation rounded to f32, denormals flushed.
fn tr(b: u32, op: fn(f64) -> f64) -> u32 {
    if nan(b) {
        return quiet(b);
    }
    let r = op(f64::from(fv(denorm_flush(b)))) as f32;
    if r.is_nan() {
        0xffc0_0000
    } else {
        denorm_flush(r.to_bits())
    }
}

/// V_MIN_F32 in IEEE mode, written as a case table.
fn rmin(a: u32, b: u32) -> u32 {
    if snan(a) {
        quiet(a)
    } else if snan(b) {
        quiet(b)
    } else if nan(a) {
        b
    } else if nan(b) {
        a
    } else if fv(a) == 0.0 && fv(b) == 0.0 {
        if a >> 31 == 1 {
            a
        } else {
            b
        }
    } else if fv(a) < fv(b) {
        a
    } else {
        b
    }
}
fn rmax(a: u32, b: u32) -> u32 {
    if snan(a) {
        quiet(a)
    } else if snan(b) {
        quiet(b)
    } else if nan(a) {
        b
    } else if nan(b) {
        a
    } else if fv(a) == 0.0 && fv(b) == 0.0 {
        if a >> 31 == 0 {
            a
        } else {
            b
        }
    } else if fv(a) >= fv(b) {
        a
    } else {
        b
    }
}
fn rhmin(a: u32, b: u32) -> u32 {
    let (a, b) = (a & 0xffff, b & 0xffff);
    let hs = |x: u32| hnan(x) && x & 0x200 == 0;
    if hs(a) {
        a | 0x200
    } else if hs(b) {
        b | 0x200
    } else if hnan(a) {
        b
    } else if hnan(b) {
        a
    } else if hv(a) == 0.0 && hv(b) == 0.0 {
        if a >> 15 == 1 {
            a
        } else {
            b
        }
    } else if hv(a) < hv(b) {
        a
    } else {
        b
    }
}
fn rhmax(a: u32, b: u32) -> u32 {
    let (a, b) = (a & 0xffff, b & 0xffff);
    let hs = |x: u32| hnan(x) && x & 0x200 == 0;
    if hs(a) {
        a | 0x200
    } else if hs(b) {
        b | 0x200
    } else if hnan(a) {
        b
    } else if hnan(b) {
        a
    } else if hv(a) == 0.0 && hv(b) == 0.0 {
        if a >> 15 == 0 {
            a
        } else {
            b
        }
    } else if hv(a) >= hv(b) {
        a
    } else {
        b
    }
}
fn s24(x: u32) -> i64 {
    i64::from(((x << 8) as i32) >> 8)
}
fn rint(b: u32, op: fn(f32) -> f32) -> u32 {
    if nan(b) {
        quiet(b)
    } else {
        op(fv(b)).to_bits()
    }
}
fn rfrexp(b: u32) -> (i32, u32) {
    let x = fv(b);
    if !x.is_finite() || x == 0.0 {
        return (0, b);
    }
    let mut e = (f64::from(x).abs().log2().floor() as i32) + 1;
    let mut m = f64::from(x) / 2f64.powi(e);
    if m.abs() >= 1.0 {
        m /= 2.0;
        e += 1
    }
    if m.abs() < 0.5 {
        m *= 2.0;
        e -= 1
    }
    (e, (m as f32).to_bits())
}
fn rf16_of_f32(b: u32, rtz: bool) -> u32 {
    u32::from(if rtz {
        float::f32_to_f16_rtz(fv(b))
    } else {
        float::f32_to_f16_rne(fv(b))
    })
}

fn cases() -> Vec<Case> {
    use Vop1Op as A;
    use Vop2Op as B;
    use Vop3OnlyOp as O;
    use G::*;
    let c1 = |op, g, r| Case {
        op: Vop3Op::V1(op),
        g,
        r,
    };
    let c2 = |op, g, r| Case {
        op: Vop3Op::V2(op),
        g,
        r,
    };
    let c3 = |op, g, r| Case {
        op: Vop3Op::Only(op),
        g,
        r,
    };
    vec![
        c1(A::MovB32, &[U], |s, _| s[0]),
        c1(A::CvtF32I32, &[U], |s, _| (s[0] as i32 as f32).to_bits()),
        c1(A::CvtF32U32, &[U], |s, _| (s[0] as f32).to_bits()),
        c1(A::CvtU32F32, &[F], |s, _| {
            let x = fv(s[0]);
            if x.is_nan() || x <= 0.0 {
                0
            } else if x >= 4294967296.0 {
                u32::MAX
            } else {
                x.trunc() as u32
            }
        }),
        c1(A::CvtI32F32, &[F], |s, _| {
            let x = fv(s[0]);
            if x.is_nan() {
                0
            } else if x >= 2147483648.0 {
                i32::MAX as u32
            } else if x <= -2147483648.0 {
                i32::MIN as u32
            } else {
                x.trunc() as i32 as u32
            }
        }),
        c1(A::CvtF16F32, &[F], |s, _| rf16_of_f32(s[0], false)),
        c1(A::CvtF32F16, &[H], |s, _| {
            float::f16_to_f32(s[0] as u16).to_bits()
        }),
        c1(A::CvtRpiI32F32, &[F], |s, _| {
            let x = fv(s[0]);
            if x.is_nan() {
                0
            } else {
                (x + 0.5).floor() as i32 as u32
            }
        }),
        c1(A::CvtFlrI32F32, &[F], |s, _| {
            let x = fv(s[0]);
            if x.is_nan() {
                0
            } else {
                x.floor() as i32 as u32
            }
        }),
        c1(A::CvtF32Ubyte0, &[U], |s, _| {
            ((s[0] & 0xff) as f32).to_bits()
        }),
        c1(A::CvtF32Ubyte1, &[U], |s, _| {
            ((s[0] >> 8 & 0xff) as f32).to_bits()
        }),
        c1(A::CvtF32Ubyte2, &[U], |s, _| {
            ((s[0] >> 16 & 0xff) as f32).to_bits()
        }),
        c1(A::CvtF32Ubyte3, &[U], |s, _| {
            ((s[0] >> 24) as f32).to_bits()
        }),
        c1(A::FractF32, &[F], |s, _| {
            if nan(s[0]) {
                return quiet(s[0]);
            }
            let x = fv(s[0]);
            if x.is_infinite() {
                return 0xffc0_0000;
            }
            (x - x.floor()).min(fv(0x3f7f_ffff)).to_bits()
        }),
        c1(A::TruncF32, &[F], |s, _| rint(s[0], f32::trunc)),
        c1(A::CeilF32, &[F], |s, _| rint(s[0], f32::ceil)),
        c1(A::RndneF32, &[F], |s, _| rint(s[0], f32::round_ties_even)),
        c1(A::FloorF32, &[F], |s, _| rint(s[0], f32::floor)),
        c1(A::ExpF32, &[F], |s, _| tr(s[0], f64::exp2)),
        c1(A::LogF32, &[F], |s, _| {
            tr(s[0], |x| {
                if x == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    x.log2()
                }
            })
        }),
        c1(A::RcpF32, &[F], |s, _| tr(s[0], |x| 1.0 / x)),
        c1(A::RcpIflagF32, &[F], |s, _| tr(s[0], |x| 1.0 / x)),
        c1(A::RsqF32, &[F], |s, _| tr(s[0], |x| 1.0 / x.sqrt())),
        c1(A::SqrtF32, &[F], |s, _| tr(s[0], f64::sqrt)),
        c1(A::NotB32, &[U], |s, _| !s[0]),
        c1(A::BfrevB32, &[U], |s, _| {
            (0..32).fold(0, |a, i| a | (s[0] >> i & 1) << (31 - i))
        }),
        c1(A::FfbhU32, &[U], |s, _| {
            (0..32)
                .find(|i| s[0] >> (31 - i) & 1 == 1)
                .unwrap_or(u32::MAX)
        }),
        c1(A::FfblB32, &[U], |s, _| {
            (0..32).find(|i| s[0] >> i & 1 == 1).unwrap_or(u32::MAX)
        }),
        c1(A::FfbhI32, &[U], |s, _| {
            (1..32)
                .find(|i| s[0] >> (31 - i) & 1 != s[0] >> 31)
                .unwrap_or(u32::MAX)
        }),
        c1(A::FrexpExpI32F32, &[F], |s, _| rfrexp(s[0]).0 as u32),
        c1(A::FrexpMantF32, &[F], |s, _| rfrexp(s[0]).1),
        c2(B::AddF32, &[F, F], |s, _| ar(&s[..2], fv(s[0]) + fv(s[1]))),
        c2(B::SubF32, &[F, F], |s, _| ar(&s[..2], fv(s[0]) - fv(s[1]))),
        c2(B::SubrevF32, &[F, F], |s, _| {
            ar(&s[..2], fv(s[1]) - fv(s[0]))
        }),
        c2(B::MulF32, &[F, F], |s, _| ar(&s[..2], fv(s[0]) * fv(s[1]))),
        c2(B::MulI32I24, &[U, U], |s, _| (s24(s[0]) * s24(s[1])) as u32),
        c2(B::MulHiI32I24, &[U, U], |s, _| {
            ((s24(s[0]) * s24(s[1])) >> 32) as u32
        }),
        c2(B::MulU32U24, &[U, U], |s, _| {
            (u64::from(s[0] & 0xff_ffff) * u64::from(s[1] & 0xff_ffff)) as u32
        }),
        c2(B::MulHiU32U24, &[U, U], |s, _| {
            ((u64::from(s[0] & 0xff_ffff) * u64::from(s[1] & 0xff_ffff)) >> 32) as u32
        }),
        c2(B::MinF32, &[F, F], |s, _| rmin(s[0], s[1])),
        c2(B::MaxF32, &[F, F], |s, _| rmax(s[0], s[1])),
        c2(B::MinI32, &[U, U], |s, _| {
            (s[0] as i32).min(s[1] as i32) as u32
        }),
        c2(B::MaxI32, &[U, U], |s, _| {
            (s[0] as i32).max(s[1] as i32) as u32
        }),
        c2(B::MinU32, &[U, U], |s, _| s[0].min(s[1])),
        c2(B::MaxU32, &[U, U], |s, _| s[0].max(s[1])),
        c2(B::LshrrevB32, &[U, U], |s, _| s[1].wrapping_shr(s[0])),
        c2(B::AshrrevI32, &[U, U], |s, _| {
            (s[1] as i32).wrapping_shr(s[0]) as u32
        }),
        c2(B::LshlrevB32, &[U, U], |s, _| s[1].wrapping_shl(s[0])),
        c2(B::AndB32, &[U, U], |s, _| s[0] & s[1]),
        c2(B::OrB32, &[U, U], |s, _| s[0] | s[1]),
        c2(B::XorB32, &[U, U], |s, _| s[0] ^ s[1]),
        c2(B::XnorB32, &[U, U], |s, _| !(s[0] ^ s[1])),
        c2(B::AddU32, &[U, U], |s, _| s[0].wrapping_add(s[1])),
        c2(B::SubU32, &[U, U], |s, _| s[0].wrapping_sub(s[1])),
        c2(B::SubrevU32, &[U, U], |s, _| s[1].wrapping_sub(s[0])),
        c2(B::AddF16, &[H, H], |s, _| {
            har(s[0], s[1], hv(s[0]) + hv(s[1]))
        }),
        c2(B::SubF16, &[H, H], |s, _| {
            har(s[0], s[1], hv(s[0]) - hv(s[1]))
        }),
        c2(B::MulF16, &[H, H], |s, _| {
            har(
                s[0],
                s[1],
                (f64::from(hv(s[0])) * f64::from(hv(s[1]))) as f32,
            )
        }),
        c2(B::MaxF16, &[H, H], |s, _| rhmax(s[0], s[1])),
        c2(B::MinF16, &[H, H], |s, _| rhmin(s[0], s[1])),
        c2(B::AddU16, &[U, U], |s, _| {
            (s[0] as u16).wrapping_add(s[1] as u16) as u32
        }),
        c2(B::SubU16, &[U, U], |s, _| {
            (s[0] as u16).wrapping_sub(s[1] as u16) as u32
        }),
        c2(B::LshlrevB16, &[U, U], |s, _| {
            ((s[1] as u16) << (s[0] & 15)) as u32
        }),
        c2(B::MulLoU16, &[U, U], |s, _| {
            (s[0] as u16).wrapping_mul(s[1] as u16) as u32
        }),
        c3(O::MadI32I24, &[U, U, U], |s, _| {
            ((s24(s[0]) * s24(s[1])) as u32).wrapping_add(s[2])
        }),
        c3(O::MadU32U24, &[U, U, U], |s, _| {
            ((s[0] & 0xff_ffff).wrapping_mul(s[1] & 0xff_ffff)).wrapping_add(s[2])
        }),
        c3(O::FmaF32, &[F, F, F], |s, _| {
            ar(s, fv(s[0]).mul_add(fv(s[1]), fv(s[2])))
        }),
        c3(O::BfeU32, &[U, M(31), M(31)], |s, _| {
            if s[2] == 0 {
                0
            } else {
                (s[0] >> s[1]) & (u32::MAX >> (32 - s[2]))
            }
        }),
        c3(O::BfeI32, &[U, M(31), M(31)], |s, _| {
            if s[2] == 0 {
                0
            } else {
                ((((s[0] >> s[1]) << (32 - s[2])) as i32) >> (32 - s[2])) as u32
            }
        }),
        c3(O::BfiB32, &[U, U, U], |s, _| (s[0] & s[1]) | (!s[0] & s[2])),
        c3(O::AlignbitB32, &[U, U, U], |s, _| {
            (((u64::from(s[0]) << 32) | u64::from(s[1])) >> (s[2] % 32)) as u32
        }),
        c3(O::AlignbyteB32, &[U, U, U], |s, _| {
            (((u64::from(s[0]) << 32) | u64::from(s[1])) >> (8 * (s[2] % 4))) as u32
        }),
        c3(O::Min3F32, &[F, F, F], |s, _| rmin(rmin(s[0], s[1]), s[2])),
        c3(O::Max3F32, &[F, F, F], |s, _| rmax(rmax(s[0], s[1]), s[2])),
        c3(O::Med3F32, &[F, F, F], |s, _| {
            if s.iter().any(|&b| nan(b)) {
                return rmin(rmin(s[0], s[1]), s[2]);
            }
            let m = fv(rmax(rmax(s[0], s[1]), s[2]));
            if m == fv(s[0]) {
                rmax(s[1], s[2])
            } else if m == fv(s[1]) {
                rmax(s[0], s[2])
            } else {
                rmax(s[0], s[1])
            }
        }),
        c3(O::Min3I32, &[U, U, U], |s, _| {
            (s[0] as i32).min(s[1] as i32).min(s[2] as i32) as u32
        }),
        c3(O::Max3I32, &[U, U, U], |s, _| {
            (s[0] as i32).max(s[1] as i32).max(s[2] as i32) as u32
        }),
        c3(O::Min3U32, &[U, U, U], |s, _| s[0].min(s[1]).min(s[2])),
        c3(O::Max3U32, &[U, U, U], |s, _| s[0].max(s[1]).max(s[2])),
        c3(O::Med3I32, &[U, U, U], |s, _| {
            let mut v = [s[0] as i32, s[1] as i32, s[2] as i32];
            v.sort();
            v[1] as u32
        }),
        c3(O::Med3U32, &[U, U, U], |s, _| {
            let mut v = *s;
            v.sort();
            v[1]
        }),
        c3(O::LshlAddU32, &[U, U, U], |s, _| {
            s[0].wrapping_shl(s[1]).wrapping_add(s[2])
        }),
        c3(O::AddLshlU32, &[U, U, U], |s, _| {
            s[0].wrapping_add(s[1]).wrapping_shl(s[2])
        }),
        c3(O::Add3U32, &[U, U, U], |s, _| {
            s[0].wrapping_add(s[1]).wrapping_add(s[2])
        }),
        c3(O::LshlOrB32, &[U, U, U], |s, _| {
            s[0].wrapping_shl(s[1]) | s[2]
        }),
        c3(O::AndOrB32, &[U, U, U], |s, _| (s[0] & s[1]) | s[2]),
        c3(O::Or3B32, &[U, U, U], |s, _| s[0] | s[1] | s[2]),
        c3(O::XadU32, &[U, U, U], |s, _| {
            (s[0] ^ s[1]).wrapping_add(s[2])
        }),
        c3(O::PermB32, &[U, U, U], |s, _| {
            let bytes: Vec<u8> = s[1]
                .to_le_bytes()
                .into_iter()
                .chain(s[0].to_le_bytes())
                .collect();
            let sel = |x: u32| -> u8 {
                match x {
                    0..=7 => bytes[x as usize],
                    8 => (bytes[1] as i8 >> 7) as u8,
                    9 => (bytes[3] as i8 >> 7) as u8,
                    10 => (bytes[5] as i8 >> 7) as u8,
                    11 => (bytes[7] as i8 >> 7) as u8,
                    12 => 0,
                    _ => 0xff,
                }
            };
            u32::from_le_bytes([
                sel(s[2] & 0xff),
                sel(s[2] >> 8 & 0xff),
                sel(s[2] >> 16 & 0xff),
                sel(s[2] >> 24),
            ])
        }),
        c3(O::MulLoU32, &[U, U], |s, _| s[0].wrapping_mul(s[1])),
        c3(O::MulHiU32, &[U, U], |s, _| {
            ((u64::from(s[0]) * u64::from(s[1])) >> 32) as u32
        }),
        c3(O::MulHiI32, &[U, U], |s, _| {
            ((i64::from(s[0] as i32) * i64::from(s[1] as i32)) >> 32) as u32
        }),
        c3(O::LdexpF32, &[F, Exp], |s, _| {
            if nan(s[0]) {
                return quiet(s[0]);
            }
            let k = (s[1] as i32).clamp(-1000, 1000);
            // Exact: scale in two steps that stay inside f64's normal range.
            ((f64::from(fv(s[0])) * 2f64.powi(k / 2) * 2f64.powi(k - k / 2)) as f32).to_bits()
        }),
        c3(O::CvtPkrtzF16F32, &[F, F], |s, _| {
            rf16_of_f32(s[0], true) | rf16_of_f32(s[1], true) << 16
        }),
        c3(O::CvtPkU16U32, &[U, U], |s, _| {
            s[0].min(0xffff) | s[1].min(0xffff) << 16
        }),
        c3(O::CvtPkI16I32, &[U, U], |s, _| {
            ((s[0] as i32).clamp(-32768, 32767) as u16 as u32)
                | ((s[1] as i32).clamp(-32768, 32767) as u16 as u32) << 16
        }),
        c3(O::BcntU32B32, &[U, U], |s, _| {
            s[0].count_ones().wrapping_add(s[1])
        }),
        c3(O::MbcntLoU32B32, &[U, U], |s, l| {
            ((0..l.min(32)).filter(|i| s[0] >> i & 1 == 1).count() as u32).wrapping_add(s[1])
        }),
        c3(O::MbcntHiU32B32, &[U, U], |s, l| {
            ((32..l.max(32))
                .filter(|i| s[0] >> (i - 32) & 1 == 1)
                .count() as u32)
                .wrapping_add(s[1])
        }),
        c3(O::PackB32F16, &[H, H], |s, _| (s[0] & 0xffff) | s[1] << 16),
        c3(O::BfmB32, &[U, U], |s, _| {
            (((1u64 << (s[0] & 31)) - 1) as u32) << (s[1] & 31)
        }),
    ]
}

/// Ops exercised by dedicated tests below rather than [`cases`].
const DEDICATED: &[&str] = &[
    "v_nop",
    "v_readfirstlane_b32",
    "v_mov_b64",
    "v_accvgpr_mov_b32",
    "v_cvt_f32_fp8",
    "v_cvt_f32_bf8",
    "v_cvt_pk_f32_fp8",
    "v_cvt_pk_f32_bf8",
    "v_swap_b32",
    "v_cvt_f64_f32",
    "v_cvt_f32_f64",
    "v_cndmask_b32",
    "v_fmac_f32",
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_addc_co_u32",
    "v_subb_co_u32",
    "v_subbrev_co_u32",
    "v_div_scale_f32",
    "v_div_fmas_f32",
    "v_div_fixup_f32",
    "v_mad_u64_u32",
    "v_mad_i64_i32",
    "v_cvt_pk_fp8_f32",
    "v_cvt_pk_bf8_f32",
    "v_cvt_sr_fp8_f32",
    "v_cvt_sr_bf8_f32",
    "v_lshlrev_b64",
    "v_lshrrev_b64",
    "v_ashrrev_i64",
    "v_readlane_b32",
    "v_writelane_b32",
    "v_add_f64",
    "v_mul_f64",
    "v_fma_f64",
    "v_lshl_add_u64",
    "v_cmp_eq_u64",
    "v_cmp_ne_u64",
];

#[test]
fn every_opcode_is_covered() {
    let mut seen: BTreeSet<&str> = cases().iter().map(|c| name(c.op)).collect();
    seen.extend(DEDICATED);
    seen.extend(cmp_cases().iter().map(|c| c.0.name()));
    let all = Vop1Op::ALL
        .iter()
        .map(|o| o.name())
        .chain(Vop2Op::ALL.iter().map(|o| o.name()))
        .chain(VopcOp::ALL.iter().map(|o| o.name()))
        .chain(Vop3OnlyOp::ALL.iter().map(|o| o.name()));
    let missing: Vec<_> = all.filter(|n| !seen.contains(n)).collect();
    assert!(missing.is_empty(), "opcodes without a test: {missing:?}");
}

#[test]
fn table_ops_random_vs_reference() {
    for (i, c) in cases().iter().enumerate() {
        run_case(c, 0x9e37_79b9_7f4a_7c15 ^ (i as u64) << 20)
    }
}

// ----------------------------------------------------------------------------
// VOPC

type CmpRef = fn(u32, u32) -> bool;
fn cmp_cases() -> Vec<(VopcOp, G, CmpRef)> {
    use VopcOp::*;
    vec![
        (CmpClassF32, G::F, |a, m| {
            let x = fv(a);
            let bit = if snan(a) {
                0
            } else if nan(a) {
                1
            } else if x == f32::NEG_INFINITY {
                2
            } else if x == f32::INFINITY {
                9
            } else if x == 0.0 {
                if a >> 31 == 1 {
                    5
                } else {
                    6
                }
            } else if x.is_subnormal() {
                if x < 0.0 {
                    4
                } else {
                    7
                }
            } else if x < 0.0 {
                3
            } else {
                8
            };
            m >> bit & 1 == 1
        }),
        (CmpLtF32, G::F, |a, b| {
            let (x, y) = (fv(a), fv(b));
            x < y
        }),
        (CmpEqF32, G::F, |a, b| fv(a) == fv(b)),
        (CmpLeF32, G::F, |a, b| fv(a) <= fv(b)),
        (CmpGtF32, G::F, |a, b| fv(a) > fv(b)),
        (CmpLgF32, G::F, |a, b| !nan(a) && !nan(b) && fv(a) != fv(b)),
        (CmpGeF32, G::F, |a, b| fv(a) >= fv(b)),
        (CmpOF32, G::F, |a, b| !nan(a) && !nan(b)),
        (CmpUF32, G::F, |a, b| nan(a) || nan(b)),
        (CmpNeqF32, G::F, |a, b| !(fv(a) == fv(b))),
        (CmpLtI32, G::U, |a, b| (a as i32) < b as i32),
        (CmpEqI32, G::U, |a, b| a == b),
        (CmpLeI32, G::U, |a, b| a as i32 <= b as i32),
        (CmpGtI32, G::U, |a, b| a as i32 > b as i32),
        (CmpNeI32, G::U, |a, b| a != b),
        (CmpGeI32, G::U, |a, b| a as i32 >= b as i32),
        (CmpLtU32, G::U, |a, b| a < b),
        (CmpEqU32, G::U, |a, b| a == b),
        (CmpLeU32, G::U, |a, b| a <= b),
        (CmpGtU32, G::U, |a, b| a > b),
        (CmpNeU32, G::U, |a, b| a != b),
        (CmpGeU32, G::U, |a, b| a >= b),
        (CmpxEqU32, G::U, |a, b| a == b),
        (CmpxNeU32, G::U, |a, b| a != b),
        (CmpxGtU32, G::U, |a, b| a > b),
        (CmpxLtU32, G::U, |a, b| a < b),
        (CmpxGtI32, G::U, |a, b| a as i32 > b as i32),
        (CmpxLtI32, G::U, |a, b| (a as i32) < b as i32),
        (CmpxGtF32, G::F, |a, b| fv(a) > fv(b)),
        (CmpxLtF32, G::F, |a, b| fv(a) < fv(b)),
    ]
}

#[test]
fn vopc_e32_and_vop3_vs_reference() {
    let mut r = Rng(0x1234_5678_9abc_def1);
    for (op, g, rf) in cmp_cases() {
        let cmpx = op.name().starts_with("v_cmpx");
        let class = op == VopcOp::CmpClassF32;
        for vop3_form in [false, true] {
            for _ in 0..ROUNDS {
                let mut w = wave();
                let exec = r.exec();
                w.exec = exec;
                let mut ins = [[0u32; 2]; WAVE];
                for l in 0..WAVE {
                    ins[l] = [
                        gen(&mut r, g),
                        if class {
                            r.u32() & 0x3ff
                        } else {
                            gen(&mut r, g)
                        },
                    ];
                    // Make equality likely on some lanes.
                    if l % 7 == 0 {
                        ins[l][1] = if class { ins[l][1] } else { ins[l][0] }
                    }
                    w.v[10][l] = ins[l][0];
                    w.v[11][l] = ins[l][1];
                }
                let (inst, dst) = if vop3_form {
                    (v3(vop3(Vop3Op::C(op), Src::Sgpr(8), &[vg(10), vg(11)])), 8)
                } else {
                    (vc(op, vg(10), vg(11)), usize::MAX)
                };
                exec_ok(&mut w, &inst);
                let want = (0..WAVE)
                    .filter(|&l| exec >> l & 1 != 0 && rf(ins[l][0], ins[l][1]))
                    .fold(0u64, |m, l| m | 1 << l);
                let got = if dst == 8 {
                    u64::from(w.s[8]) | u64::from(w.s[9]) << 32
                } else {
                    w.vcc
                };
                assert_eq!(got, want, "{} vop3={vop3_form}", op.name());
                if cmpx {
                    assert_eq!(w.exec, want, "{} EXEC", op.name())
                } else {
                    assert_eq!(w.exec, exec)
                }
                if dst == 8 {
                    assert_eq!(
                        w.vcc,
                        u64::from(POISON) | u64::from(POISON) << 32,
                        "VOP3 compare must not touch VCC"
                    )
                }
            }
        }
    }
}

#[test]
fn vopc_u64_compares() {
    let mut r = Rng(77);
    for op in [VopcOp::CmpEqU64, VopcOp::CmpNeU64] {
        let mut w = wave();
        w.exec = 0xffff_0000_ffff_00ff;
        for l in 0..WAVE {
            let a = r.next();
            let b = if l % 3 == 0 {
                a
            } else if l % 3 == 1 {
                a ^ 1 << 40
            } else {
                r.next()
            };
            w.v[10][l] = a as u32;
            w.v[11][l] = (a >> 32) as u32;
            w.v[12][l] = b as u32;
            w.v[13][l] = (b >> 32) as u32;
        }
        exec_ok(&mut w, &vc(op, vg(10), vg(12)));
        for l in 0..WAVE {
            let eq = w.v[10][l] == w.v[12][l] && w.v[11][l] == w.v[13][l];
            let want = w.exec >> l & 1 != 0 && (eq == (op == VopcOp::CmpEqU64));
            assert_eq!(w.vcc >> l & 1 != 0, want, "{} lane {l}", op.name());
        }
    }
}

#[test]
fn vopc_float_modifiers_and_inline_constant() {
    // v_cmp_lt_f32_e64 s[8:9], -|v10|, 0.5
    let mut w = wave();
    for l in 0..WAVE {
        w.v[10][l] = if l % 2 == 0 { 0xbf80_0000 } else { 0x3e80_0000 }
    } // -1.0 / 0.25
    let mut v = vop3(
        Vop3Op::C(VopcOp::CmpLtF32),
        Src::Sgpr(8),
        &[vg(10), Src::Float(InlineFloat::Half)],
    );
    v.abs = 1;
    v.neg = 1;
    exec_ok(&mut w, &v3(v));
    assert_eq!(
        u64::from(w.s[8]) | u64::from(w.s[9]) << 32,
        u64::MAX,
        "-|x| < 0.5 for every lane"
    );
    // abs on an integer compare is not modelled.
    let mut v = vop3(Vop3Op::C(VopcOp::CmpLtI32), Src::Sgpr(8), &[vg(10), vg(10)]);
    v.abs = 1;
    assert!(matches!(
        valu::exec(&mut w, &v3(v)),
        Err(SimError::Unsupported(_))
    ));
}

// ----------------------------------------------------------------------------
// ISA edge cases

fn one(op: Vop1Op, x: u32) -> u32 {
    let mut w = wave();
    w.v[10][0] = x;
    exec_ok(&mut w, &v1(op, vg(1), vg(10)));
    w.v[1][0]
}

#[test]
fn transcendental_isa_examples() {
    use Vop1Op::*;
    // CDNA3 ISA §12.8 p.187-190 functional examples.
    for (op, x, want) in [
        (ExpF32, 0xff80_0000, 0x0000_0000),
        (ExpF32, 0x8000_0000, 0x3f80_0000),
        (ExpF32, 0x7f80_0000, 0x7f80_0000),
        (LogF32, 0xff80_0000, 0xffc0_0000),
        (LogF32, 0xbf80_0000, 0xffc0_0000),
        (LogF32, 0x8000_0000, 0xff80_0000),
        (LogF32, 0x0000_0000, 0xff80_0000),
        (LogF32, 0x3f80_0000, 0x0000_0000),
        (LogF32, 0x7f80_0000, 0x7f80_0000),
        (RcpF32, 0xff80_0000, 0x8000_0000),
        (RcpF32, 0xc000_0000, 0xbf00_0000),
        (RcpF32, 0x8000_0000, 0xff80_0000),
        (RcpF32, 0x0000_0000, 0x7f80_0000),
        (RcpF32, 0x7f80_0000, 0x0000_0000),
        (RsqF32, 0xff80_0000, 0xffc0_0000),
        (RsqF32, 0x8000_0000, 0xff80_0000),
        (RsqF32, 0x0000_0000, 0x7f80_0000),
        (RsqF32, 0x4080_0000, 0x3f00_0000),
        (RsqF32, 0x7f80_0000, 0x0000_0000),
        (SqrtF32, 0xff80_0000, 0xffc0_0000),
        (SqrtF32, 0x8000_0000, 0x8000_0000),
        (SqrtF32, 0x0000_0000, 0x0000_0000),
        (SqrtF32, 0x4080_0000, 0x4000_0000),
        (SqrtF32, 0x7f80_0000, 0x7f80_0000),
        // Denormals are flushed on input and output.
        (RcpF32, 0x0000_0001, 0x7f80_0000),
        (RcpF32, 0x7f00_0000, 0x0000_0000),
        (SqrtF32, 0x8000_0001, 0x8000_0000),
        (LogF32, 0x0000_0001, 0xff80_0000),
        (ExpF32, 0xc310_0000, 0x0000_0000),
    ] {
        assert_eq!(one(op, x), want, "{} {x:#010x}", op.name());
    }
}

#[test]
fn bit_count_isa_examples() {
    use Vop1Op::*;
    // §12.8 p.192-193.
    for (op, x, want) in [
        (FfbhU32, 0, u32::MAX),
        (FfbhU32, 0x8000_00ff, 0),
        (FfbhU32, 0x1000_00ff, 3),
        (FfbhU32, 0xffff, 16),
        (FfbhU32, 1, 31),
        (FfblB32, 0, u32::MAX),
        (FfblB32, 0xff00_0001, 0),
        (FfblB32, 0xff00_0008, 3),
        (FfblB32, 0xffff_0000, 16),
        (FfblB32, 0x8000_0000, 31),
        (FfbhI32, 0, u32::MAX),
        (FfbhI32, 0x4000_0000, 1),
        (FfbhI32, 0x8000_0000, 1),
        (FfbhI32, 0x0fff_ffff, 4),
        (FfbhI32, 0xffff_0000, 16),
        (FfbhI32, 0xffff_fffe, 31),
        (FfbhI32, 0xffff_ffff, u32::MAX),
    ] {
        assert_eq!(one(op, x), want, "{} {x:#010x}", op.name());
    }
}

#[test]
fn conversion_edges() {
    use Vop1Op::*;
    // §12.8 p.180-181: NaN → 0, out of range saturates.
    for (op, x, want) in [
        (CvtI32F32, 0x7fc0_0000, 0),
        (CvtI32F32, 0x7f80_0000, 0x7fff_ffff),
        (CvtI32F32, 0xff80_0000, 0x8000_0000),
        (CvtI32F32, 0x4f00_0000, 0x7fff_ffff),
        (CvtI32F32, 0xbfc0_0000, (-1i32) as u32),
        (CvtI32F32, 0x8000_0001, 0),
        (CvtU32F32, 0xbf80_0000, 0),
        (CvtU32F32, 0x4f80_0000, u32::MAX),
        (CvtU32F32, 0xffc0_0000, 0),
        (CvtRpiI32F32, 0x3f00_0000, 1),
        (CvtRpiI32F32, 0xbf00_0000, 0),
        (CvtRpiI32F32, 0xbfc0_0000, (-1i32) as u32),
        (CvtFlrI32F32, 0xbf00_0000, (-1i32) as u32),
        (CvtFlrI32F32, 0x7fc0_0000, 0),
        (RndneF32, 0x3f00_0000, 0),
        (RndneF32, 0x3fc0_0000, 0x4000_0000),
        (RndneF32, 0x4020_0000, 0x4000_0000),
        (RndneF32, 0xbe80_0000, 0x8000_0000),
        (CeilF32, 0xbf00_0000, 0x8000_0000),
        (FloorF32, 0x8000_0001, 0xbf80_0000),
        (TruncF32, 0x7fa0_0000, 0x7fe0_0000),
        // DX fract: fract(-1e-10) rounds to 1.0, clamped to 0x3f7fffff; fract(-1.25) = 0.75.
        (FractF32, 0xaedb_e6ff, 0x3f7f_ffff),
        (FractF32, 0xbfa0_0000, 0x3f40_0000),
        (FractF32, 0x7f80_0000, 0xffc0_0000),
        (FrexpMantF32, 0x0000_0001, 0x3f00_0000),
        (FrexpExpI32F32, 0x0000_0001, (-148i32) as u32),
        (FrexpExpI32F32, 0x7f80_0000, 0),
        (FrexpMantF32, 0xff80_0000, 0xff80_0000),
        (FrexpMantF32, 0x8000_0000, 0x8000_0000),
    ] {
        assert_eq!(one(op, x), want, "{} {x:#010x}", op.name());
    }
}

#[test]
fn min_max_nan_and_signed_zero_rules() {
    use Vop2Op::*;
    let run = |op, a: u32, b: u32, ieee: bool| {
        let mut w = wave();
        w.mode.ieee = ieee;
        w.v[10][0] = a;
        w.v[11][0] = b;
        exec_ok(&mut w, &v2(op, vg(1), vg(10), vg(11)));
        w.v[1][0]
    };
    // §12.7 p.163-164.
    assert_eq!(
        run(MinF32, 0x7fa0_0000, 0x3f80_0000, true),
        0x7fe0_0000,
        "IEEE: sNaN quieted"
    );
    assert_eq!(
        run(MinF32, 0x7fa0_0000, 0x3f80_0000, false),
        0x3f80_0000,
        "non-IEEE: NaN ignored"
    );
    assert_eq!(
        run(MaxF32, 0x3f80_0000, 0x7fc0_0000, true),
        0x3f80_0000,
        "qNaN ignored"
    );
    assert_eq!(run(MinF32, 0x0000_0000, 0x8000_0000, true), 0x8000_0000);
    assert_eq!(run(MinF32, 0x8000_0000, 0x0000_0000, true), 0x8000_0000);
    assert_eq!(run(MaxF32, 0x8000_0000, 0x0000_0000, true), 0x0000_0000);
    assert_eq!(run(MaxF32, 0x0000_0000, 0x8000_0000, false), 0x0000_0000);
    assert_eq!(run(MaxF16, 0x7d00, 0x3c00, true), 0x7f00);
    assert_eq!(run(MinF16, 0x8000, 0x0000, true), 0x8000);
    // §12.11 p.333: med3 with a NaN input is min3.
    let mut w = wave();
    for (l, (a, b, c, want)) in [
        (0x7fc0_0000, 0x4000_0000, 0x3f80_0000, 0x3f80_0000),
        (0x4040_0000, 0x3f80_0000, 0x4000_0000, 0x4000_0000),
        (0xff80_0000, 0x7f80_0000, 0x0000_0000, 0x0000_0000),
        (0x8000_0000, 0x0000_0000, 0x3f80_0000, 0x0000_0000),
    ]
    .into_iter()
    .enumerate()
    {
        w.v[10][l] = a;
        w.v[11][l] = b;
        w.v[12][l] = c;
        w.v[2][l] = want;
    }
    w.exec = 0xf;
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::Only(Vop3OnlyOp::Med3F32),
            vg(1),
            &[vg(10), vg(11), vg(12)],
        )),
    );
    for l in 0..4 {
        assert_eq!(w.v[1][l], w.v[2][l], "med3 lane {l}")
    }
}

// ----------------------------------------------------------------------------
// Modifiers and modes

#[test]
fn vop3_abs_neg_clamp_omod() {
    use Vop2Op::*;
    let add = |a: u32, b: u32, f: &dyn Fn(&mut Vop3), w: &mut Wave| -> Result<u32, SimError> {
        w.v[10][0] = a;
        w.v[11][0] = b;
        let mut v = vop3(Vop3Op::V2(AddF32), vg(1), &[vg(10), vg(11)]);
        f(&mut v);
        valu::exec(w, &v3(v)).map(|_| w.v[1][0])
    };
    let mut w = wave();
    // -|a| + |b|
    assert_eq!(
        add(
            0x4000_0000,
            0xc040_0000,
            &|v| {
                v.abs = 3;
                v.neg = 1
            },
            &mut w
        )
        .unwrap(),
        0x3f80_0000
    );
    // neg of a NaN flips the sign of the propagated (quieted) NaN.
    assert_eq!(
        add(0x7fa0_0000, 0, &|v| v.neg = 1, &mut w).unwrap(),
        0xffe0_0000
    );
    // Clamp: 2+2 → 1.0; -1+0 → +0; -0 + -0 → +0; NaN → 0 under DX10_CLAMP, NaN otherwise.
    assert_eq!(
        add(0x4000_0000, 0x4000_0000, &|v| v.clamp = true, &mut w).unwrap(),
        0x3f80_0000
    );
    assert_eq!(add(0xbf80_0000, 0, &|v| v.clamp = true, &mut w).unwrap(), 0);
    assert_eq!(
        add(0x8000_0000, 0x8000_0000, &|v| v.clamp = true, &mut w).unwrap(),
        0
    );
    assert_eq!(
        add(0x3e80_0000, 0, &|v| v.clamp = true, &mut w).unwrap(),
        0x3e80_0000
    );
    assert_eq!(add(0x7fc0_0000, 0, &|v| v.clamp = true, &mut w).unwrap(), 0);
    w.mode.dx10_clamp = false;
    assert_eq!(
        add(0x7fc0_0000, 0, &|v| v.clamp = true, &mut w).unwrap(),
        0x7fc0_0000
    );
    // OMOD is ignored with IEEE=1 or output denormals on (§6.2.2 p.33).
    assert_eq!(
        add(0x3f80_0000, 0, &|v| v.omod = 1, &mut w).unwrap(),
        0x3f80_0000
    );
    w.mode.ieee = false;
    assert_eq!(
        add(0x3f80_0000, 0, &|v| v.omod = 3, &mut w).unwrap(),
        0x3f80_0000
    );
    // Flushing denorm modes and non-RNE rounding are not modelled.
    w.mode.denorm32 = 1;
    assert!(matches!(
        add(0x3f80_0000, 0, &|v| v.omod = 1, &mut w),
        Err(SimError::Unsupported(_))
    ));
    let mut w = wave();
    w.mode.round32 = 3;
    assert!(matches!(
        add(0x3f80_0000, 0, &|_| {}, &mut w),
        Err(SimError::Unsupported(_))
    ));
    // op_sel on an f32 op and abs on an integer op are rejected.
    let mut w = wave();
    assert!(matches!(
        add(0, 0, &|v| v.op_sel = 1, &mut w),
        Err(SimError::Unsupported(_))
    ));
    let mut v = vop3(Vop3Op::V2(AndB32), vg(1), &[vg(10), vg(11)]);
    v.neg = 2;
    assert!(matches!(
        valu::exec(&mut w, &v3(v)),
        Err(SimError::Unsupported(_))
    ));
    let mut v = vop3(Vop3Op::V2(AndB32), vg(1), &[vg(10), vg(11)]);
    v.clamp = true;
    assert!(matches!(
        valu::exec(&mut w, &v3(v)),
        Err(SimError::Unsupported(_))
    ));
}

#[test]
fn integer_clamp_saturates() {
    use Vop2Op::*;
    let mut w = wave();
    w.exec = 0b111;
    for (l, (a, b)) in [(0xffff_fff0u32, 0x20u32), (5, 7), (0xffff, 0x2)]
        .into_iter()
        .enumerate()
    {
        w.v[10][l] = a;
        w.v[11][l] = b
    }
    let run = |w: &mut Wave, op| {
        let mut v = vop3(Vop3Op::V2(op), vg(1), &[vg(10), vg(11)]);
        v.clamp = true;
        exec_ok(w, &v3(v));
        [w.v[1][0], w.v[1][1], w.v[1][2]]
    };
    assert_eq!(run(&mut w, AddU32), [u32::MAX, 12, 0x1_0001]);
    assert_eq!(run(&mut w, SubU32), [0xffff_ffd0, 0, 0xfffd]);
    assert_eq!(run(&mut w, SubrevU32), [0, 2, 0]);
    assert_eq!(run(&mut w, AddU16), [0xffff, 12, 0xffff]);
    assert_eq!(run(&mut w, SubU16), [0xffd0, 0, 0xfffd]);
    assert_eq!(run(&mut w, MulLoU16), [0xffff, 35, 0xffff]);
}

#[test]
fn f16_vop1_vop2_zero_high_half_and_inline_constants() {
    let mut w = wave();
    w.v[10][0] = 0xabcd_3c00; // high half ignored: 1.0
    exec_ok(
        &mut w,
        &v2(Vop2Op::AddF16, vg(1), Src::Float(InlineFloat::Two), vg(10)),
    );
    assert_eq!(
        w.v[1][0], 0x4200,
        "2.0 + 1.0 = 3.0 (f16 inline constant), high half zero"
    );
    exec_ok(&mut w, &v2(Vop2Op::AddU16, vg(1), Src::Int(-1), vg(10)));
    assert_eq!(
        w.v[1][0], 0x3bff,
        "inline -1 is 0xffff for a 16-bit operand"
    );
    // v_pack_b32_f16 v1, -v10, v11 op_sel:[1,0]: S0 high half (negated), S1 low half.
    w.v[11][0] = 0x1234_4000;
    let mut v = vop3(
        Vop3Op::Only(Vop3OnlyOp::PackB32F16),
        vg(1),
        &[vg(10), vg(11)],
    );
    v.op_sel = 1;
    v.neg = 1;
    exec_ok(&mut w, &v3(v));
    assert_eq!(w.v[1][0], 0x4000_2bcd);
    // op_sel on a promoted VOP2 f16 op is not modelled on gfx9.
    let mut v = vop3(Vop3Op::V2(Vop2Op::AddF16), vg(1), &[vg(10), vg(11)]);
    v.op_sel = 1;
    assert!(matches!(
        valu::exec(&mut w, &v3(v)),
        Err(SimError::Unsupported(_))
    ));
}

// ----------------------------------------------------------------------------
// Carry, select and lane-crossing scalar ops

#[test]
fn carry_ops_e32_and_vop3b() {
    use Vop2Op::*;
    let mut r = Rng(99);
    for op in [
        AddCoU32,
        SubCoU32,
        SubrevCoU32,
        AddcCoU32,
        SubbCoU32,
        SubbrevCoU32,
    ] {
        for vop3_form in [false, true] {
            for _ in 0..ROUNDS {
                let mut w = wave();
                let exec = r.exec();
                w.exec = exec;
                let cin = r.next();
                for l in 0..WAVE {
                    w.v[10][l] = gen(&mut r, G::U);
                    w.v[11][l] = gen(&mut r, G::U)
                }
                let inst = if vop3_form {
                    w.s[6] = cin as u32;
                    w.s[7] = (cin >> 32) as u32;
                    let mut v = vop3(Vop3Op::V2(op), vg(1), &[vg(10), vg(11), Src::Sgpr(6)]);
                    v.sdst = Some(Src::Sgpr(4));
                    v3(v)
                } else {
                    w.vcc = cin;
                    v2(op, vg(1), vg(10), vg(11))
                };
                exec_ok(&mut w, &inst);
                let mut mask = 0u64;
                for l in 0..WAVE {
                    if exec >> l & 1 == 0 {
                        assert_eq!(w.v[1][l], POISON);
                        continue;
                    }
                    let (a, b) = (i128::from(w.v[10][l]), i128::from(w.v[11][l]));
                    let c = if matches!(op, AddcCoU32 | SubbCoU32 | SubbrevCoU32) {
                        i128::from((cin >> l & 1) as u8)
                    } else {
                        0
                    };
                    let t = match op {
                        AddCoU32 | AddcCoU32 => a + b + c,
                        SubCoU32 | SubbCoU32 => a - b - c,
                        _ => b - a - c,
                    };
                    let out = !(0..=0xffff_ffff).contains(&t);
                    assert_eq!(w.v[1][l], t as u32, "{} lane {l}", op.name());
                    mask |= u64::from(out) << l;
                }
                let got = if vop3_form {
                    u64::from(w.s[4]) | u64::from(w.s[5]) << 32
                } else {
                    w.vcc
                };
                assert_eq!(got, mask, "{} carry vop3={vop3_form}", op.name());
            }
        }
    }
    // VOP3B clamp saturates and keeps the carry.
    let mut w = wave();
    w.exec = 1;
    w.v[10][0] = u32::MAX;
    w.v[11][0] = 2;
    let mut v = vop3(Vop3Op::V2(AddCoU32), vg(1), &[vg(10), vg(11)]);
    v.sdst = Some(Src::VccLo);
    v.clamp = true;
    exec_ok(&mut w, &v3(v));
    assert_eq!((w.v[1][0], w.vcc), (u32::MAX, 1));
}

#[test]
fn cndmask_vcc_and_sgpr_mask_with_modifiers() {
    let mut w = wave();
    w.exec = 0x0f0f_0f0f_0f0f_0f0f;
    let sel = 0x3333_3333_3333_3333u64;
    w.vcc = sel;
    for l in 0..WAVE {
        w.v[10][l] = l as u32;
        w.v[11][l] = 0x8000_0000 | (100 + l as u32)
    }
    exec_ok(&mut w, &v2(Vop2Op::CndmaskB32, vg(1), vg(10), vg(11)));
    for l in 0..WAVE {
        let want = if w.exec >> l & 1 == 0 {
            POISON
        } else if sel >> l & 1 != 0 {
            0x8000_0000 | (100 + l as u32)
        } else {
            l as u32
        };
        assert_eq!(w.v[1][l], want, "lane {l}");
    }
    let mut w2 = wave();
    w2.s[20] = 0xffff_0000;
    w2.s[21] = 0x0000_ffff;
    for l in 0..WAVE {
        w2.v[10][l] = l as u32;
        w2.v[11][l] = 0x8000_0000 | l as u32
    }
    let mut v = vop3(
        Vop3Op::V2(Vop2Op::CndmaskB32),
        vg(2),
        &[vg(10), vg(11), Src::Sgpr(20)],
    );
    v.abs = 2;
    exec_ok(&mut w2, &v3(v));
    for l in 0..WAVE {
        let s = if l < 32 { l >= 16 } else { l < 48 };
        assert_eq!(
            w2.v[2][l], l as u32,
            "lane {l} (abs clears bit 31 of S1; select {s})"
        );
    }
}

#[test]
fn readlane_writelane_readfirstlane() {
    let mut w = wave();
    for l in 0..WAVE {
        w.v[5][l] = 1000 + l as u32
    }
    w.exec = 0; // all three ignore EXEC
    exec_ok(&mut w, &v1(Vop1Op::ReadfirstlaneB32, Src::Sgpr(3), vg(5)));
    assert_eq!(w.s[3], 1000, "EXEC = 0 reads lane 0");
    w.exec = 0xffff_0000_0000_0000;
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::V1(Vop1Op::ReadfirstlaneB32),
            Src::Sgpr(4),
            &[vg(5)],
        )),
    );
    assert_eq!(w.s[4], 1048);
    w.s[10] = 64 + 37; // lane select uses bits [5:0]
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::Only(Vop3OnlyOp::ReadlaneB32),
            Src::Sgpr(11),
            &[vg(5), Src::Sgpr(10)],
        )),
    );
    assert_eq!(w.s[11], 1037);
    w.s[12] = 0x1234_5678;
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::Only(Vop3OnlyOp::WritelaneB32),
            vg(5),
            &[Src::Sgpr(12), Src::Int(3)],
        )),
    );
    assert_eq!(w.v[5][3], 0x1234_5678);
    assert_eq!(w.v[5][2], 1002);
    // ISA §12.11 p.356 laneId example: v_mbcnt_lo(-1, 0); v_mbcnt_hi(-1, v0).
    let mut w = wave();
    w.exec = u64::MAX;
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::Only(Vop3OnlyOp::MbcntLoU32B32),
            vg(0),
            &[Src::Int(-1), Src::Int(0)],
        )),
    );
    exec_ok(
        &mut w,
        &v3(vop3(
            Vop3Op::Only(Vop3OnlyOp::MbcntHiU32B32),
            vg(0),
            &[Src::Int(-1), vg(0)],
        )),
    );
    for l in 0..WAVE {
        assert_eq!(w.v[0][l], l as u32)
    }
}

#[test]
fn nop_swap_mov64_accvgpr_fmac() {
    let mut w = wave();
    let before = w.v.clone();
    exec_ok(&mut w, &v1(Vop1Op::Nop, vg(0), Src::Sgpr(0)));
    assert!(w.v == before);
    w.exec = 0x00ff_00ff_00ff_00ff;
    for l in 0..WAVE {
        w.v[2][l] = l as u32;
        w.v[3][l] = 500 + l as u32
    }
    exec_ok(&mut w, &v1(Vop1Op::SwapB32, vg(2), vg(3)));
    for l in 0..WAVE {
        let on = w.exec >> l & 1 != 0;
        assert_eq!(
            (w.v[2][l], w.v[3][l]),
            if on {
                (500 + l as u32, l as u32)
            } else {
                (l as u32, 500 + l as u32)
            }
        );
    }
    // v_mov_b64 v[6:7], v[2:3] with neg (VOP3) flips bit 63.
    let mut v = vop3(Vop3Op::V1(Vop1Op::MovB64), vg(6), &[vg(2)]);
    v.neg = 1;
    exec_ok(&mut w, &v3(v));
    for l in 0..WAVE {
        if w.exec >> l & 1 == 0 {
            assert_eq!(w.v[6][l], POISON);
            continue;
        }
        assert_eq!((w.v[6][l], w.v[7][l]), (w.v[2][l], w.v[3][l] ^ 0x8000_0000));
    }
    exec_ok(
        &mut w,
        &v1(Vop1Op::MovB64, vg(8), Src::Float(InlineFloat::One)),
    );
    assert_eq!(
        (w.v[8][0], w.v[9][0]),
        (0, 0x3ff0_0000),
        "inline 1.0 as an f64 operand"
    );
    for l in 0..WAVE {
        w.a[4][l] = 7 * l as u32
    }
    exec_ok(
        &mut w,
        &v1(Vop1Op::AccvgprMovB32, Src::Agpr(9), Src::Agpr(4)),
    );
    for l in 0..WAVE {
        assert_eq!(
            w.a[9][l],
            if w.exec >> l & 1 != 0 {
                7 * l as u32
            } else {
                POISON
            }
        )
    }
    // v_fmac_f32: D = fma(S0, S1, D), single rounding.
    let mut w = wave();
    let (a, b, c) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON, -1.0f32);
    w.v[10][0] = a.to_bits();
    w.v[11][0] = b.to_bits();
    w.v[1][0] = c.to_bits();
    w.exec = 1;
    exec_ok(&mut w, &v2(Vop2Op::FmacF32, vg(1), vg(10), vg(11)));
    assert_eq!(w.v[1][0], a.mul_add(b, c).to_bits());
    assert_ne!(w.v[1][0], (a * b + c).to_bits(), "fused, not two roundings");
}

#[test]
fn f64_ops_and_conversions() {
    let mut r = Rng(4242);
    let g64 = |r: &mut Rng| -> u64 {
        match r.next() % 4 {
            0 => r.pick(&[
                0u64,
                1 << 63,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0000,
                0x7ff4_0000_0000_0000,
                1,
                0x3ff0_0000_0000_0000,
            ]),
            1 => r.next(),
            _ => (r.next() & 0x800f_ffff_ffff_ffff) | (900 + r.next() % 250) << 52,
        }
    };
    let ar64 = |ins: &[u64], x: f64| -> u64 {
        for &b in ins {
            if f64::from_bits(b).is_nan() {
                return b | 1 << 51;
            }
        }
        if x.is_nan() {
            0xfff8_0000_0000_0000
        } else {
            x.to_bits()
        }
    };
    for op in [Vop3OnlyOp::AddF64, Vop3OnlyOp::MulF64, Vop3OnlyOp::FmaF64] {
        for _ in 0..ROUNDS {
            let mut w = wave();
            w.exec = r.exec();
            let mut ins = vec![[0u64; 3]; WAVE];
            for l in 0..WAVE {
                for k in 0..3 {
                    ins[l][k] = g64(&mut r);
                    w.v[10 + 2 * k][l] = ins[l][k] as u32;
                    w.v[11 + 2 * k][l] = (ins[l][k] >> 32) as u32
                }
            }
            exec_ok(
                &mut w,
                &v3(vop3(Vop3Op::Only(op), vg(2), &[vg(10), vg(12), vg(14)])),
            );
            for l in 0..WAVE {
                let got = u64::from(w.v[2][l]) | u64::from(w.v[3][l]) << 32;
                if w.exec >> l & 1 == 0 {
                    assert_eq!(got, u64::from(POISON) | u64::from(POISON) << 32);
                    continue;
                }
                let [a, b, c] = ins[l].map(f64::from_bits);
                let want = match op {
                    Vop3OnlyOp::AddF64 => ar64(&ins[l][..2], a + b),
                    Vop3OnlyOp::MulF64 => ar64(&ins[l][..2], a * b),
                    _ => ar64(&ins[l], a.mul_add(b, c)),
                };
                assert_eq!(got, want, "{} lane {l} {:016x?}", op.name(), ins[l]);
            }
        }
    }
    // v_cvt_f64_f32 / v_cvt_f32_f64 (denormals preserved, RNE, NaN quieted).
    let mut w = wave();
    for (l, x) in F32_SPECIAL.iter().enumerate() {
        w.v[10][l] = *x
    }
    w.exec = (1u64 << F32_SPECIAL.len()) - 1;
    exec_ok(&mut w, &v1(Vop1Op::CvtF64F32, vg(2), vg(10)));
    for (l, &x) in F32_SPECIAL.iter().enumerate() {
        let got = f64::from_bits(u64::from(w.v[2][l]) | u64::from(w.v[3][l]) << 32);
        if nan(x) {
            assert!(got.is_nan() && w.v[3][l] & 0x8_0000 != 0, "lane {l}")
        } else {
            assert_eq!(got.to_bits(), f64::from(fv(x)).to_bits(), "lane {l}")
        }
    }
    exec_ok(&mut w, &v1(Vop1Op::CvtF32F64, vg(4), vg(2)));
    for (l, &x) in F32_SPECIAL.iter().enumerate() {
        assert_eq!(
            w.v[4][l],
            if nan(x) { quiet(x) } else { x },
            "round trip lane {l}"
        )
    }
    let mut w = wave();
    w.exec = 1;
    let x = 1.0f64 + 2f64.powi(-24) + 2f64.powi(-40); // above the f32 halfway point
    w.v[2][0] = x.to_bits() as u32;
    w.v[3][0] = (x.to_bits() >> 32) as u32;
    exec_ok(&mut w, &v1(Vop1Op::CvtF32F64, vg(4), vg(2)));
    assert_eq!(w.v[4][0], 0x3f80_0001);
}

#[test]
fn mad64_shift64_lshl_add_u64() {
    let mut r = Rng(5150);
    for _ in 0..ROUNDS {
        let mut w = wave();
        w.exec = r.exec();
        for l in 0..WAVE {
            w.v[10][l] = gen(&mut r, G::U);
            w.v[11][l] = gen(&mut r, G::U);
            w.v[12][l] = gen(&mut r, G::U);
            w.v[13][l] = gen(&mut r, G::U);
            w.v[14][l] = (r.next() % 70) as u32;
        }
        let rd =
            |w: &Wave, n: usize, l: usize| u64::from(w.v[n][l]) | u64::from(w.v[n + 1][l]) << 32;
        for op in [Vop3OnlyOp::MadU64U32, Vop3OnlyOp::MadI64I32] {
            let mut v = vop3(Vop3Op::Only(op), vg(2), &[vg(10), vg(11), vg(12)]);
            v.sdst = Some(Src::Sgpr(20));
            exec_ok(&mut w, &v3(v));
            let mut mask = 0u64;
            for l in (0..WAVE).filter(|&l| w.exec >> l & 1 != 0) {
                let (a, b, c) = (w.v[10][l], w.v[11][l], rd(&w, 12, l));
                let t = if op == Vop3OnlyOp::MadU64U32 {
                    i128::from(a) * i128::from(b) + i128::from(c)
                } else {
                    i128::from(a as i32) * i128::from(b as i32) + i128::from(c as i64)
                };
                assert_eq!(rd(&w, 2, l), t as u64, "{} lane {l}", op.name());
                mask |= ((t >> 64) as u64 & 1) << l;
            }
            assert_eq!(
                u64::from(w.s[20]) | u64::from(w.s[21]) << 32,
                mask,
                "{} carry",
                op.name()
            );
        }
        for op in [
            Vop3OnlyOp::LshlrevB64,
            Vop3OnlyOp::LshrrevB64,
            Vop3OnlyOp::AshrrevI64,
        ] {
            exec_ok(
                &mut w,
                &v3(vop3(Vop3Op::Only(op), vg(4), &[vg(14), vg(12)])),
            );
            for l in (0..WAVE).filter(|&l| w.exec >> l & 1 != 0) {
                let (sh, x) = (w.v[14][l] & 63, rd(&w, 12, l));
                let want = match op {
                    Vop3OnlyOp::LshlrevB64 => x << sh,
                    Vop3OnlyOp::LshrrevB64 => x >> sh,
                    _ => ((x as i64) >> sh) as u64,
                };
                assert_eq!(rd(&w, 4, l), want, "{} lane {l}", op.name());
            }
        }
        for l in 0..WAVE {
            w.v[15][l] = (r.next() % 5) as u32 | (r.u32() & !7)
        }
        exec_ok(
            &mut w,
            &v3(vop3(
                Vop3Op::Only(Vop3OnlyOp::LshlAddU64),
                vg(6),
                &[vg(12), vg(15), vg(2)],
            )),
        );
        for l in (0..WAVE).filter(|&l| w.exec >> l & 1 != 0) {
            assert_eq!(
                rd(&w, 6, l),
                (rd(&w, 12, l) << (w.v[15][l] & 7)).wrapping_add(rd(&w, 2, l)),
                "v_lshl_add_u64 lane {l}"
            );
        }
    }
    // ISA §12.11 p.351: shift counts above 4 are unsupported.
    let mut w = wave();
    w.exec = 1;
    w.v[15][0] = 5;
    assert!(matches!(
        valu::exec(
            &mut w,
            &v3(vop3(
                Vop3Op::Only(Vop3OnlyOp::LshlAddU64),
                vg(6),
                &[vg(12), vg(15), vg(2)]
            ))
        ),
        Err(SimError::Unsupported(_))
    ));
}

// ----------------------------------------------------------------------------
// FP8 / BF8 conversions

#[test]
fn fp8_converts() {
    let enc = Arch::Gfx942.caps().fp8;
    assert_eq!(enc, Fp8Encoding::Fnuz);
    let mut r = Rng(888);
    for (kind, pk, pk_to, sr) in [
        (
            F8::Fp8,
            Vop1Op::CvtF32Fp8,
            Vop1Op::CvtPkF32Fp8,
            Vop3OnlyOp::CvtSrFp8F32,
        ),
        (
            F8::Bf8,
            Vop1Op::CvtF32Bf8,
            Vop1Op::CvtPkF32Bf8,
            Vop3OnlyOp::CvtSrBf8F32,
        ),
    ] {
        let rne = if kind == F8::Fp8 {
            Vop3OnlyOp::CvtPkFp8F32
        } else {
            Vop3OnlyOp::CvtPkBf8F32
        };
        for ovfl in [false, true] {
            let mut w = wave();
            w.mode.fp16_ovfl = ovfl;
            w.exec = r.exec();
            for l in 0..WAVE {
                w.v[10][l] = gen(&mut r, G::F);
                w.v[11][l] = gen(&mut r, G::F);
                w.v[12][l] = r.u32();
                w.v[1][l] = 0x1122_3344
            }
            // v_cvt_pk_*_f32 into the low half, then (op_sel[3]) the high half; the other half is preserved.
            let mut v = vop3(Vop3Op::Only(rne), vg(1), &[vg(10), vg(11)]);
            v.neg = 2;
            exec_ok(&mut w, &v3(v.clone()));
            for l in 0..WAVE {
                let want = if w.exec >> l & 1 == 0 {
                    0x1122_3344
                } else {
                    let lo = u32::from(float::f32_to_f8_rne(enc, kind, fv(w.v[10][l]), ovfl));
                    let hi = u32::from(float::f32_to_f8_rne(enc, kind, -fv(w.v[11][l]), ovfl));
                    0x1122_0000 | lo | hi << 8
                };
                assert_eq!(w.v[1][l], want, "{} lane {l}", rne.name());
            }
            v.op_sel = 8;
            exec_ok(&mut w, &v3(v));
            for l in (0..WAVE).filter(|&l| w.exec >> l & 1 != 0) {
                let lo = u32::from(float::f32_to_f8_rne(enc, kind, fv(w.v[10][l]), ovfl))
                    | u32::from(float::f32_to_f8_rne(enc, kind, -fv(w.v[11][l]), ovfl)) << 8;
                assert_eq!(
                    w.v[1][l],
                    lo << 16 | lo,
                    "{} op_sel:[0,0,0,1] lane {l}",
                    rne.name()
                );
            }
            // v_cvt_sr_*_f32 into each byte.
            for byte in 0..4u8 {
                let mut v = vop3(Vop3Op::Only(sr), vg(2), &[vg(10), vg(12)]);
                v.op_sel = byte << 2;
                for l in 0..WAVE {
                    w.v[2][l] = 0xa5a5_a5a5
                }
                exec_ok(&mut w, &v3(v));
                for l in (0..WAVE).filter(|&l| w.exec >> l & 1 != 0) {
                    let b = u32::from(float::f32_to_f8_sr(
                        enc,
                        kind,
                        fv(w.v[10][l]),
                        w.v[12][l],
                        ovfl,
                    ));
                    let sh = 8 * u32::from(byte);
                    assert_eq!(
                        w.v[2][l],
                        (0xa5a5_a5a5 & !(0xff << sh)) | b << sh,
                        "{} byte {byte} lane {l}",
                        sr.name()
                    );
                }
            }
        }
        // F8 → F32: e32 byte 0, SDWA byte select, VOP3 op_sel byte; packed into a pair.
        let mut w = wave();
        w.exec = r.exec();
        for l in 0..WAVE {
            w.v[10][l] = r.u32()
        }
        exec_ok(&mut w, &v1(pk, vg(1), vg(10)));
        let mut sd = sdwa(SdwaSel::Dword);
        sd.src0_sel = SdwaSel::Byte2;
        exec_ok(
            &mut w,
            &Inst::Vop1(Vop1 {
                op: pk,
                vdst: vg(2),
                src0: vg(10),
                ext: VopExt::Sdwa(sd),
            }),
        );
        let mut v = vop3(Vop3Op::V1(pk), vg(3), &[vg(10)]);
        v.op_sel = 3;
        exec_ok(&mut w, &v3(v));
        exec_ok(&mut w, &v1(pk_to, vg(4), vg(10)));
        let mut sd = sdwa(SdwaSel::Dword);
        sd.src0_sel = SdwaSel::Word1;
        exec_ok(
            &mut w,
            &Inst::Vop1(Vop1 {
                op: pk_to,
                vdst: vg(6),
                src0: vg(10),
                ext: VopExt::Sdwa(sd),
            }),
        );
        let cv = |x: u32| float::f8_to_f32(enc, kind, x as u8).to_bits();
        for l in 0..WAVE {
            let x = w.v[10][l];
            if w.exec >> l & 1 == 0 {
                assert_eq!([w.v[1][l], w.v[2][l], w.v[4][l], w.v[6][l]], [POISON; 4]);
                continue;
            }
            assert_eq!(w.v[1][l], cv(x), "{} byte0", pk.name());
            assert_eq!(w.v[2][l], cv(x >> 16), "{} sdwa byte2", pk.name());
            assert_eq!(w.v[3][l], cv(x >> 24), "{} op_sel byte3", pk.name());
            assert_eq!(
                [w.v[4][l], w.v[5][l]],
                [cv(x), cv(x >> 8)],
                "{} word0",
                pk_to.name()
            );
            assert_eq!(
                [w.v[6][l], w.v[7][l]],
                [cv(x >> 16), cv(x >> 24)],
                "{} sdwa word1",
                pk_to.name()
            );
        }
        // Odd destination for the packed form, and a non-VGPR source for
        // CVT_PK (§7.2 p.51), are rejected.
        assert!(valu::exec(&mut w, &v1(pk_to, vg(5), vg(10))).is_err());
        assert!(matches!(
            valu::exec(
                &mut w,
                &v3(vop3(Vop3Op::Only(rne), vg(1), &[vg(10), Src::Int(0)]))
            ),
            Err(SimError::Unsupported(_))
        ));
    }
}

// ----------------------------------------------------------------------------
// DPP and SDWA

fn dpp(ctrl: DppCtrl) -> Dpp {
    Dpp {
        ctrl,
        row_mask: 0xf,
        bank_mask: 0xf,
        bound_ctrl: false,
        src0_neg: false,
        src0_abs: false,
        src1_neg: false,
        src1_abs: false,
    }
}
fn sdwa(dst_sel: SdwaSel) -> Sdwa {
    Sdwa {
        dst_sel,
        dst_unused: SdwaUnused::Preserve,
        clamp: false,
        omod: 0,
        src0_sel: SdwaSel::Dword,
        src0_sext: false,
        src0_neg: false,
        src0_abs: false,
        src1_sel: SdwaSel::Dword,
        src1_sext: false,
        src1_neg: false,
        src1_abs: false,
        sdst: None,
    }
}

#[test]
fn dpp_forms() {
    let mut w = wave();
    for l in 0..WAVE {
        w.v[10][l] = (l as f32 + 1.0).to_bits();
        w.v[11][l] = 0x3f80_0000
    }
    // v_mov_b32_dpp quad_perm:[1,0,3,2]: lane l reads lane l^1.
    exec_ok(
        &mut w,
        &Inst::Vop1(Vop1 {
            op: Vop1Op::MovB32,
            vdst: vg(1),
            src0: vg(10),
            ext: VopExt::Dpp(dpp(DppCtrl::QuadPerm([1, 0, 3, 2]))),
        }),
    );
    for l in 0..WAVE {
        assert_eq!(w.v[1][l], w.v[10][l ^ 1], "quad_perm lane {l}")
    }
    // v_add_f32_dpp with src0 neg and row_mask 0b0101: rows 1 and 3 keep their value.
    let mut d = dpp(DppCtrl::QuadPerm([3, 2, 1, 0]));
    d.row_mask = 0b0101;
    d.src0_neg = true;
    for l in 0..WAVE {
        w.v[2][l] = POISON
    }
    exec_ok(
        &mut w,
        &Inst::Vop2(Vop2 {
            op: Vop2Op::AddF32,
            vdst: vg(2),
            src0: vg(10),
            src1: vg(11),
            ext: VopExt::Dpp(d),
        }),
    );
    for l in 0..WAVE {
        let src = (l & !3) | (3 - (l & 3));
        let want = if (l / 16) % 2 == 0 {
            (1.0 - fv(w.v[10][src])).to_bits()
        } else {
            POISON
        };
        assert_eq!(w.v[2][l], want, "row-masked dpp add lane {l}");
    }
    // The DPP source reads the pre-instruction value even when it is also the destination.
    exec_ok(
        &mut w,
        &Inst::Vop1(Vop1 {
            op: Vop1Op::MovB32,
            vdst: vg(10),
            src0: vg(10),
            ext: VopExt::Dpp(dpp(DppCtrl::QuadPerm([1, 0, 3, 2]))),
        }),
    );
    for l in 0..WAVE {
        assert_eq!(
            w.v[10][l],
            (((l ^ 1) as f32) + 1.0).to_bits(),
            "in-place dpp lane {l}"
        )
    }
    // EXEC still masks DPP writes.
    w.exec = 0xaaaa_aaaa_aaaa_aaaa;
    for l in 0..WAVE {
        w.v[3][l] = POISON
    }
    exec_ok(
        &mut w,
        &Inst::Vop1(Vop1 {
            op: Vop1Op::MovB32,
            vdst: vg(3),
            src0: vg(11),
            ext: VopExt::Dpp(dpp(DppCtrl::QuadPerm([1, 1, 1, 1]))),
        }),
    );
    for l in 0..WAVE {
        assert_eq!(w.v[3][l], if l % 2 == 1 { 0x3f80_0000 } else { POISON })
    }
    // DPP on a VOPC has no gfx9 encoding.
    assert!(matches!(
        valu::exec(
            &mut w,
            &Inst::Vopc(Vopc {
                op: VopcOp::CmpEqU32,
                src0: vg(10),
                src1: vg(11),
                ext: VopExt::Dpp(dpp(DppCtrl::RowMirror))
            })
        ),
        Err(SimError::Unsupported(_))
    ));
}

#[test]
fn sdwa_forms() {
    let mut r = Rng(31337);
    let sel = |x: u32, s: SdwaSel, sext: bool| -> u32 {
        let (v, bits) = match s {
            SdwaSel::Byte0 => (x & 0xff, 8),
            SdwaSel::Byte1 => (x >> 8 & 0xff, 8),
            SdwaSel::Byte2 => (x >> 16 & 0xff, 8),
            SdwaSel::Byte3 => (x >> 24, 8),
            SdwaSel::Word0 => (x & 0xffff, 16),
            SdwaSel::Word1 => (x >> 16, 16),
            SdwaSel::Dword => return x,
        };
        if sext {
            ((v << (32 - bits)) as i32 >> (32 - bits)) as u32
        } else {
            v
        }
    };
    let mut w = wave();
    w.exec = r.exec();
    for l in 0..WAVE {
        w.v[10][l] = r.u32();
        w.v[11][l] = r.u32();
        w.v[1][l] = r.u32()
    }
    let old: Vec<u32> = (0..WAVE).map(|l| w.v[1][l]).collect();
    // v_add_u32_sdwa v1, sext(v10.byte1), v11.word0 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE
    let mut s = sdwa(SdwaSel::Word1);
    s.src0_sel = SdwaSel::Byte1;
    s.src0_sext = true;
    s.src1_sel = SdwaSel::Word0;
    exec_ok(
        &mut w,
        &Inst::Vop2(Vop2 {
            op: Vop2Op::AddU32,
            vdst: vg(1),
            src0: vg(10),
            src1: vg(11),
            ext: VopExt::Sdwa(s),
        }),
    );
    for l in 0..WAVE {
        let want = if w.exec >> l & 1 == 0 {
            old[l]
        } else {
            let t = sel(w.v[10][l], SdwaSel::Byte1, true).wrapping_add(sel(
                w.v[11][l],
                SdwaSel::Word0,
                false,
            ));
            (old[l] & 0xffff) | t << 16
        };
        assert_eq!(w.v[1][l], want, "sdwa add lane {l}");
    }
    // v_mul_f32_sdwa with src1 neg and an SGPR src1; DWORD selects; clamp.
    let mut w = wave();
    w.exec = u64::MAX;
    w.s[7] = 0x4000_0000;
    for l in 0..WAVE {
        w.v[10][l] = (l as f32 / 64.0 - 0.25).to_bits()
    }
    let mut s = sdwa(SdwaSel::Dword);
    s.src1_neg = true;
    s.clamp = true;
    exec_ok(
        &mut w,
        &Inst::Vop2(Vop2 {
            op: Vop2Op::MulF32,
            vdst: vg(2),
            src0: vg(10),
            src1: Src::Sgpr(7),
            ext: VopExt::Sdwa(s),
        }),
    );
    for l in 0..WAVE {
        let x = (l as f32 / 64.0 - 0.25) * -2.0;
        // Clamp maps -0 (lane 16: 0 * -2) to +0.
        assert_eq!(
            w.v[2][l],
            if x <= 0.0 { 0 } else { x.min(1.0).to_bits() },
            "sdwa mul lane {l}"
        );
    }
    // VOPC SDWA writes its SDST.
    let mut s = sdwa(SdwaSel::Dword);
    s.src0_sel = SdwaSel::Byte0;
    s.sdst = Some(Src::Sgpr(30));
    for l in 0..WAVE {
        w.v[11][l] = 0xffff_ff00 | l as u32;
        w.v[12][l] = 10
    }
    exec_ok(
        &mut w,
        &Inst::Vopc(Vopc {
            op: VopcOp::CmpLtU32,
            src0: vg(11),
            src1: vg(12),
            ext: VopExt::Sdwa(s),
        }),
    );
    assert_eq!(u64::from(w.s[30]) | u64::from(w.s[31]) << 32, (1 << 10) - 1);
}

// ----------------------------------------------------------------------------
// Division macro

/// The LLVM gfx9 IEEE f32 division sequence:
/// v_div_scale_f32 v3, s[2:3], v1, v1, v0 / v_div_scale_f32 v4, vcc, v0, v1, v0
/// / v_rcp_f32 v5, v3 / v_fma_f32 v6, -v3, v5, 1.0 / v_fma_f32 v5, v6, v5, v5
/// / v_mul_f32 v6, v4, v5 / v_fma_f32 v7, -v3, v6, v4 / v_fma_f32 v6, v7, v5, v6
/// / v_fma_f32 v3, -v3, v6, v4 / v_div_fmas_f32 v3, v3, v5, v6
/// / v_div_fixup_f32 v2, v3, v1, v0.
fn division_macro() -> Vec<Inst> {
    use Vop3OnlyOp::*;
    let o = |op| Vop3Op::Only(op);
    let mut s0 = vop3(o(DivScaleF32), vg(3), &[vg(1), vg(1), vg(0)]);
    s0.sdst = Some(Src::Sgpr(2));
    let mut s1 = vop3(o(DivScaleF32), vg(4), &[vg(0), vg(1), vg(0)]);
    s1.sdst = Some(Src::VccLo);
    let fma = |d, a, b, c, neg| {
        let mut v = vop3(o(FmaF32), vg(d), &[a, b, c]);
        v.neg = neg;
        v3(v)
    };
    vec![
        v3(s0),
        v3(s1),
        v1(Vop1Op::RcpF32, vg(5), vg(3)),
        fma(6, vg(3), vg(5), Src::Float(InlineFloat::One), 1),
        fma(5, vg(6), vg(5), vg(5), 0),
        v2(Vop2Op::MulF32, vg(6), vg(4), vg(5)),
        fma(7, vg(3), vg(6), vg(4), 1),
        fma(6, vg(7), vg(5), vg(6), 0),
        fma(3, vg(3), vg(6), vg(4), 1),
        v3(vop3(o(DivFmasF32), vg(3), &[vg(3), vg(5), vg(6)])),
        v3(vop3(o(DivFixupF32), vg(2), &[vg(3), vg(1), vg(0)])),
    ]
}

#[test]
fn division_macro_is_correctly_rounded() {
    let prog = division_macro();
    let mut r = Rng(0xd1d1_d1d1);
    let mut checked = 0usize;
    for round in 0..3000 {
        let mut w = wave();
        w.exec = u64::MAX;
        for l in 0..WAVE {
            let (n, d) = match round % 3 {
                0 => (r.u32(), r.u32()),
                1 => (gen(&mut r, G::F), gen(&mut r, G::F)),
                _ => (
                    (r.u32() & 0x8000_0000)
                        | ((1 + r.next() % 60) as u32) << 23
                        | (r.u32() & 0x7f_ffff),
                    r.u32() & 0x807f_ffff | ((150 + r.next() % 104) as u32) << 23,
                ),
            };
            w.v[0][l] = n;
            w.v[1][l] = d;
        }
        for inst in &prog {
            exec_ok(&mut w, inst)
        }
        for l in 0..WAVE {
            let (n, d) = (fv(w.v[0][l]), fv(w.v[1][l]));
            let q = n / d;
            let got = w.v[2][l];
            if q.is_nan() {
                assert!(
                    nan(got),
                    "{n:e}/{d:e} ({:#010x}/{:#010x}) = NaN, got {got:#010x}",
                    w.v[0][l],
                    w.v[1][l]
                )
            } else {
                assert_eq!(
                    got,
                    q.to_bits(),
                    "{n:e}/{d:e} ({:#010x}/{:#010x}): got {:e} want {q:e}",
                    w.v[0][l],
                    w.v[1][l],
                    fv(got)
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 3000 * WAVE);
}

#[test]
fn div_scale_fmas_fixup_units() {
    use Vop3OnlyOp::*;
    let run3 = |op, a: u32, b: u32, c: u32, vcc: u64| -> (u32, u64) {
        let mut w = wave();
        w.exec = 1;
        w.vcc = vcc;
        w.v[10][0] = a;
        w.v[11][0] = b;
        w.v[12][0] = c;
        let mut v = vop3(Vop3Op::Only(op), vg(1), &[vg(10), vg(11), vg(12)]);
        if op == DivScaleF32 {
            v.sdst = Some(Src::Sgpr(40));
            w.s[40] = POISON;
            w.s[41] = POISON;
        }
        exec_ok(&mut w, &v3(v));
        (
            w.v[1][0],
            if op == DivScaleF32 {
                u64::from(w.s[40]) | u64::from(w.s[41]) << 32
            } else {
                w.vcc
            },
        )
    };
    // §12.11 p.336-337 div_scale branches (S0 ∈ {S1, S2}).
    let one = 0x3f80_0000;
    assert_eq!(
        run3(DivScaleF32, one, 0, one, 0),
        (0xffc0_0000, 0),
        "zero denominator → NaN"
    );
    assert_eq!(
        run3(DivScaleF32, 0x7000_0000, 0x0f00_0000, 0x7000_0000, 0),
        (0x7000_0000, 1),
        "e2-e1 >= 96: VCC, numerator unchanged"
    );
    assert_eq!(
        run3(DivScaleF32, 0x0f00_0000, 0x0f00_0000, 0x7000_0000, 0),
        (0x2f00_0000, 1),
        "e2-e1 >= 96: denominator × 2^64"
    );
    assert_eq!(
        run3(DivScaleF32, 0x0000_0001, 0x0000_0001, 0x2180_0000, 0),
        (0x1500_0000, 0),
        "denormal denominator × 2^64"
    );
    assert_eq!(
        run3(DivScaleF32, 0x7f00_0000, 0x7f00_0000, 0x4000_0000, 0),
        (0x5f00_0000, 0),
        "1/S1 denormal: × 2^-64"
    );
    assert_eq!(
        run3(DivScaleF32, 0x0100_0000, 0x4080_0000, 0x0100_0000, 0),
        (0x2100_0000, 1),
        "quotient denormal: numerator × 2^64"
    );
    assert_eq!(
        run3(DivScaleF32, 0x0b00_0000, 0x3f80_0000, 0x0b00_0000, 0),
        (0x2b00_0000, 0),
        "tiny numerator × 2^64"
    );
    assert_eq!(run3(DivScaleF32, one, one, one, 0), (one, 0), "no scaling");
    // div_fmas: plain fma without VCC; post-scale by 2^±64 with VCC.
    let (a, b, c) = (1.5f32.to_bits(), 2.0f32.to_bits(), 1.0f32.to_bits());
    assert_eq!(run3(DivFmasF32, a, b, c, 0).0, 4.0f32.to_bits());
    assert_eq!(
        run3(DivFmasF32, a, b, c, 1).0,
        (4.0f32 * 2f32.powi(-64)).to_bits()
    );
    assert_eq!(
        run3(DivFmasF32, a, b, 0x4000_0000, 1).0,
        (5.0f32 * 2f32.powi(64)).to_bits()
    );
    // Single rounding into the denormal range: 2^-63·(1 + 3·2^-23) − 2^-103,
    // scaled by 2^-64, is 2^-127·(1 + 1.5·2^-22 − 2^-40). The correctly
    // rounded denormal is 0x00400001; rounding the fma first and then
    // scaling would produce a tie and give 0x00400002.
    assert_eq!(
        run3(DivFmasF32, 0x2000_0003, 0x3f80_0000, 0x8c00_0000, 1).0,
        0x0040_0001
    );
    // Exact zero / infinity handling.
    assert_eq!(
        run3(DivFmasF32, 0, b, 0x8000_0000, 1).0,
        0,
        "+0*2 + -0 = +0"
    );
    assert_eq!(
        run3(DivFmasF32, 0x7f80_0000, 0, c, 0).0,
        0xffc0_0000,
        "inf*0 = NaN"
    );
    // div_fixup special cases (§12.11 p.335).
    let fx = |q: u32, d: u32, n: u32| run3(DivFixupF32, q, d, n, 0).0;
    assert_eq!(fx(one, 0, 0), 0xffc0_0000, "0/0");
    assert_eq!(fx(one, 0x7f80_0000, 0xff80_0000), 0xffc0_0000, "inf/inf");
    assert_eq!(fx(one, 0x8000_0000, one), 0xff80_0000, "x/-0");
    assert_eq!(fx(one, one, 0xff80_0000), 0xff80_0000, "-inf/y");
    assert_eq!(fx(one, 0xff80_0000, one), 0x8000_0000, "x/-inf");
    assert_eq!(fx(one, one, 0x8000_0000), 0x8000_0000, "-0/y");
    assert_eq!(
        fx(one, 0x7fa0_0000, one),
        0x7fe0_0000,
        "NaN denominator quieted"
    );
    assert_eq!(
        fx(one, one, 0xffa0_0000),
        0xffe0_0000,
        "NaN numerator quieted"
    );
    assert_eq!(
        fx(0x4000_0000, 0xbf80_0000, one),
        0xc000_0000,
        "sign from S1^S2, magnitude from S0"
    );
    assert_eq!(
        fx(one, 0x7e80_0000, 0x0080_0000),
        0x0000_0000,
        "underflow branch"
    );
    assert_eq!(
        fx(0x7fc0_0000, 0x8d52_4bed, 0x7490_f1e1),
        0xff80_0000,
        "overflowed (NaN) quotient with finite inputs → ±Inf"
    );
    assert_eq!(
        fx(0x7f80_0000, one, 0x7f7f_ffff),
        0x7f80_0000,
        "Inf quotient → +Inf"
    );
}
