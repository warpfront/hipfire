// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Vector ALU semantics: VOP1, VOP2, VOPC and VOP3 (promoted and VOP3-only),
//! including DPP/SDWA operand forms via [`super::xlane`] and the FP8/BF8
//! conversions via [`crate::float`] (CDNA3 ISA §6, §7.2, §12.7-12.11).
//!
//! Every form is normalized to one operand model: three sources, per-source
//! ABS/NEG bits (VOP3, or the DPP/SDWA modifier bits), CLAMP, OMOD and OP_SEL.
//! Lanes are computed first and written afterwards, so a DPP source or a
//! destination that is also a source always reads pre-instruction values.
//!
//! Rules that hold for every instruction here:
//! - A lane writes only when its EXEC bit is set (CDNA3 ISA §6.2.2 p.33),
//!   and under DPP only when ROW_MASK/BANK_MASK enable it and the DPP source
//!   lane is readable ([`super::xlane::dpp_read`]). `v_readlane`,
//!   `v_writelane` and `v_readfirstlane` ignore EXEC (§12.8 p.179, §12.11 p.354-355).
//! - Scalar lane masks (VOPC result, VOP2/VOP3B carry-out, `v_div_scale`
//!   flag) are written as a whole 64-bit value: lanes that did not compute
//!   get 0, which is what LLVM relies on when it treats compare results as
//!   already masked by EXEC. `[UNVERIFIED: confirm on MI300X]`
//! - Floating point: f32 arithmetic is IEEE binary32 with round to nearest
//!   even, matching MODE round mode 0. f16 and f64 arithmetic is IEEE with
//!   RNE, matching MODE round_16_64 = 0. Denormals are preserved (MODE denorm 3).
//!   Any other round or denormal mode returns [`super::SimError::Unsupported`]
//!   for instructions that do floating-point arithmetic or comparisons.
//! - NaN results: the first NaN operand (S0, S1, S2 order) is returned
//!   quieted. An invalid operation without NaN inputs returns the default
//!   NaN 0xffc00000 / 0xfe00 / 0xfff8000000000000. The guide gives no
//!   payload rules; the default value matches its examples (V_LOG_F32,
//!   V_RSQ_F32, V_DIV_FIXUP_F32, §12.8 p.187-190, §12.11 p.335).
//!   `[UNVERIFIED: confirm on MI300X]`
//! - OMOD is ignored when MODE.IEEE = 1 or the result type's output
//!   denormals are enabled, and does not apply to integer results
//!   (§6.2.2 p.33). In any other mode it returns Unsupported. CLAMP clamps
//!   floating-point results to [0.0, 1.0]: NaN becomes +0 under DX10_CLAMP
//!   and passes through otherwise, and -0 becomes +0
//!   `[UNVERIFIED: confirm on MI300X]`. Integer CLAMP saturates only on
//!   instructions whose description says "Supports saturation" (§6.5 p.37).
//!   Elsewhere CLAMP returns Unsupported, except on VOPC (where it only
//!   signals exceptions) and on conversions whose CLAMP bit only gates the
//!   INEXACT exception.
//! - ABS/NEG on an operand the instruction does not read as floating point,
//!   and OP_SEL bits the instruction does not define on gfx9, return
//!   Unsupported.
//! - 16-bit VOP1/VOP2 results zero the high half of the destination, as
//!   LLVM's `zeroesHigh16BitsOfDest` does for GFX9. The guide's `D0.f16 = …`
//!   pseudocode leaves the high half unspecified. `[UNVERIFIED: confirm on MI300X]`
use super::xlane::{dpp_lane_enabled, dpp_read, sdwa_dst, sdwa_src};
use super::{fault, unsupported, Flow, SimError, SimResult, Ty, Wave};
use crate::enc::vop::{Dpp, Sdwa, SdwaSel, Vop1Op, Vop2Op, VopExt, VopcOp};
use crate::enc::vop3::{Vop3OnlyOp, Vop3Op};
use crate::float::{self, F8};
use crate::inst::Inst;
use crate::operand::Src;
use crate::WAVE;

/// Execute one vector-ALU instruction under EXEC (VOPC/`v_cmpx` and
/// `v_readlane`/`v_readfirstlane` per their scalar-result rules).
pub fn exec(w: &mut Wave, inst: &Inst) -> SimResult<Flow> {
    let (v, op) = match inst {
        Inst::Vop1(i) => (
            V::short(i.op.name(), &i.ext, i.vdst, [Some(i.src0), None, None]),
            Vop3Op::V1(i.op),
        ),
        Inst::Vop2(i) => (
            V::short(
                i.op.name(),
                &i.ext,
                i.vdst,
                [Some(i.src0), Some(i.src1), None],
            ),
            Vop3Op::V2(i.op),
        ),
        Inst::Vopc(i) => (
            V::short(
                i.op.name(),
                &i.ext,
                Src::VccLo,
                [Some(i.src0), Some(i.src1), None],
            ),
            Vop3Op::C(i.op),
        ),
        Inst::Vop3(i) => (
            V {
                name: op_name(i.op),
                form: Form::Vop3,
                dst: i.dst,
                sdst: i.sdst,
                src: i.src,
                abs: i.abs & 7,
                neg: i.neg & 7,
                clamp: i.clamp,
                omod: i.omod,
                op_sel: i.op_sel,
            },
            i.op,
        ),
        other => {
            return unsupported(format!(
                "`{other:?}` is not a vector-ALU (VOP1/VOP2/VOPC/VOP3) instruction"
            ))
        }
    };
    v.run(w, op)?;
    Ok(Flow::Next)
}

fn op_name(op: Vop3Op) -> &'static str {
    match op {
        Vop3Op::C(o) => o.name(),
        Vop3Op::V2(o) => o.name(),
        Vop3Op::V1(o) => o.name(),
        Vop3Op::Only(o) => o.name(),
    }
}

// ----------------------------------------------------------------------------
// Normalized operand model

#[derive(Clone, Copy, Debug)]
enum Form {
    E32,
    Vop3,
    Dpp(Dpp),
    Sdwa(Sdwa),
}

/// Result width class for output modifiers and MODE checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wd {
    F16,
    F32,
    F64,
}

/// What CLAMP/OMOD mean for an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Out {
    /// Floating-point result: CLAMP to [0,1], OMOD per §6.2.2.
    Float(Wd),
    /// Integer result with saturation semantics for CLAMP (the op applies it).
    IntSat,
    /// CLAMP/OMOD accepted and without effect on the result.
    Ignore,
    /// CLAMP unsupported; OMOD ignored (integer/bit result, §6.2.2 p.33).
    None,
}

struct V {
    name: &'static str,
    form: Form,
    /// Vector destination (scalar for `v_readlane`/`v_readfirstlane`, the
    /// mask destination for VOP3-encoded VOPC; VCC for VOPC e32/DPP).
    dst: Src,
    /// VOP3B scalar destination / SDWA VOPC destination.
    sdst: Option<Src>,
    src: [Option<Src>; 3],
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
    op_sel: u8,
}

/// Unwrap a lane value; `None` (DPP source lane unreadable) skips the lane.
macro_rules! opt {
    ($e:expr) => {
        match $e? {
            Some(v) => v,
            None => return Ok(None),
        }
    };
}

impl V {
    fn short(name: &'static str, ext: &VopExt, dst: Src, src: [Option<Src>; 3]) -> Self {
        let mut v = V {
            name,
            form: Form::E32,
            dst,
            sdst: None,
            src,
            abs: 0,
            neg: 0,
            clamp: false,
            omod: 0,
            op_sel: 0,
        };
        match *ext {
            VopExt::None => {}
            VopExt::Dpp(d) => {
                v.form = Form::Dpp(d);
                v.abs = u8::from(d.src0_abs) | u8::from(d.src1_abs) << 1;
                v.neg = u8::from(d.src0_neg) | u8::from(d.src1_neg) << 1;
            }
            VopExt::Sdwa(s) => {
                v.form = Form::Sdwa(s);
                v.abs = u8::from(s.src0_abs) | u8::from(s.src1_abs) << 1;
                v.neg = u8::from(s.src0_neg) | u8::from(s.src1_neg) << 1;
                v.clamp = s.clamp;
                v.omod = s.omod;
                v.sdst = s.sdst;
            }
        }
        v
    }

    fn src_k(&self, k: usize) -> SimResult<Src> {
        self.src[k].ok_or_else(|| SimError::Fault(format!("{}: missing src{k}", self.name)))
    }

    fn is_vop3(&self) -> bool {
        matches!(self.form, Form::Vop3)
    }

    /// Reject DPP and SDWA (64-bit, lane-crossing and VOP3-only semantics).
    fn plain(&self) -> SimResult<()> {
        match self.form {
            Form::Dpp(_) => unsupported(format!(
                "{}: DPP form not modelled for this instruction",
                self.name
            )),
            Form::Sdwa(_) => unsupported(format!(
                "{}: SDWA form not modelled for this instruction",
                self.name
            )),
            _ => Ok(()),
        }
    }

    fn check(&self, w: &Wave, fsrc: u8, out: Out, opsel: u8) -> SimResult<()> {
        if (self.abs | self.neg) & !fsrc & 7 != 0 {
            return unsupported(format!(
                "{}: abs/neg (abs {:#b} neg {:#b}) on an operand not read as floating point",
                self.name, self.abs, self.neg
            ));
        }
        if self.op_sel & !opsel != 0 {
            return unsupported(format!(
                "{}: op_sel {:#06b} (gfx9 defines only {:#06b} here)",
                self.name, self.op_sel, opsel
            ));
        }
        match out {
            Out::Float(wd) => {
                let dn = if wd == Wd::F32 {
                    w.mode.denorm32
                } else {
                    w.mode.denorm16_64
                };
                if self.omod != 0 && !(w.mode.ieee || dn & 2 != 0) {
                    return unsupported(format!("{}: omod {} with IEEE=0 and output denormals flushed (applied omod not modelled)", self.name, self.omod));
                }
            }
            Out::IntSat | Out::Ignore => {}
            Out::None => {
                if self.clamp {
                    return unsupported(format!(
                        "{}: clamp on an instruction without saturation semantics",
                        self.name
                    ));
                }
            }
        }
        Ok(())
    }

    fn fmode(&self, w: &Wave, m32: bool, m16_64: bool) -> SimResult<()> {
        let m = w.mode;
        if m32 && (m.round32 != 0 || m.denorm32 != 3) {
            return unsupported(format!("{}: FP32 round mode {} / denorm mode {} (only RNE with denormals preserved is modelled)", self.name, m.round32, m.denorm32));
        }
        if m16_64 && (m.round16_64 != 0 || m.denorm16_64 != 3) {
            return unsupported(format!("{}: FP16/64 round mode {} / denorm mode {} (only RNE with denormals preserved is modelled)", self.name, m.round16_64, m.denorm16_64));
        }
        Ok(())
    }

    fn lanes(&self, w: &Wave) -> Vec<usize> {
        let dpp = if let Form::Dpp(d) = self.form {
            Some(d)
        } else {
            None
        };
        (0..WAVE)
            .filter(|&l| {
                w.exec >> l & 1 != 0 && dpp.as_ref().is_none_or(|d| dpp_lane_enabled(d, l))
            })
            .collect()
    }

    /// Raw 32-bit value of source `k` for `lane` after DPP/SDWA selection.
    fn get(&self, w: &Wave, k: usize, lane: usize, ty: Ty) -> SimResult<Option<u32>> {
        let src = self.src_k(k)?;
        let raw = match (self.form, k, src) {
            (Form::Dpp(d), 0, Src::Vgpr(n)) => opt!(dpp_read(w, &d, n, false, lane)),
            (Form::Dpp(d), 0, Src::Agpr(n)) => opt!(dpp_read(w, &d, n, true, lane)),
            (Form::Dpp(_), 0, other) => {
                return fault(format!("{}: DPP src0 {other:?} is not a VGPR", self.name))
            }
            _ => w.read_v(src, lane, ty)?,
        };
        Ok(Some(match (self.form, k) {
            (Form::Sdwa(s), 0) => sdwa_src(raw, s.src0_sel, s.src0_sext),
            (Form::Sdwa(s), 1) => sdwa_src(raw, s.src1_sel, s.src1_sext),
            _ => raw,
        }))
    }

    fn fmod(&self, k: usize, b: u64, sign: u32) -> u64 {
        let mut b = b;
        if self.abs >> k & 1 != 0 {
            b &= !(1u64 << sign)
        }
        if self.neg >> k & 1 != 0 {
            b ^= 1u64 << sign
        }
        b
    }
    /// Untyped / integer 32-bit source.
    fn u(&self, w: &Wave, k: usize, l: usize) -> SimResult<Option<u32>> {
        self.get(w, k, l, Ty::B32)
    }
    /// f32 source bits with ABS/NEG.
    fn f(&self, w: &Wave, k: usize, l: usize) -> SimResult<Option<u32>> {
        Ok(self
            .get(w, k, l, Ty::F32)?
            .map(|b| self.fmod(k, u64::from(b), 31) as u32))
    }
    /// f16 source bits (low half) with ABS/NEG.
    fn h(&self, w: &Wave, k: usize, l: usize) -> SimResult<Option<u32>> {
        Ok(self
            .get(w, k, l, Ty::F16)?
            .map(|b| self.fmod(k, u64::from(b & 0xffff), 15) as u32))
    }
    /// Integer 16-bit source (low half).
    fn s16(&self, w: &Wave, k: usize, l: usize) -> SimResult<Option<u32>> {
        Ok(self.get(w, k, l, Ty::B16)?.map(|b| b & 0xffff))
    }
    /// f64 source bits with ABS/NEG (no DPP/SDWA: callers ran `plain`).
    fn d(&self, w: &Wave, k: usize, l: usize) -> SimResult<u64> {
        Ok(self.fmod(k, w.read_v64(self.src_k(k)?, l, Ty::F64)?, 63))
    }
    /// Integer 64-bit source.
    fn q(&self, w: &Wave, k: usize, l: usize) -> SimResult<u64> {
        w.read_v64(self.src_k(k)?, l, Ty::B64)
    }

    fn fout(&self, w: &Wave, wd: Wd, b: u64) -> u64 {
        if !self.clamp {
            return b;
        }
        let f = Fmt::of(wd);
        if f.is_nan(b) {
            return if w.mode.dx10_clamp { 0 } else { b };
        }
        let x = f.val(b);
        if x <= 0.0 {
            0
        } else if x >= 1.0 {
            f.one()
        } else {
            b
        }
    }

    /// Per-lane 32-bit result into `dst` (SDWA destination select applied).
    fn map1(
        &self,
        w: &mut Wave,
        f: impl Fn(&Self, &Wave, usize) -> SimResult<Option<u32>>,
    ) -> SimResult<()> {
        let mut out = Vec::with_capacity(WAVE);
        for lane in self.lanes(w) {
            if let Some(v) = f(self, w, lane)? {
                let v = match self.form {
                    Form::Sdwa(s) => sdwa_dst(
                        w.read_v(self.dst, lane, Ty::B32)?,
                        v,
                        s.dst_sel,
                        s.dst_unused,
                    ),
                    _ => v,
                };
                out.push((lane, v));
            }
        }
        for (lane, v) in out {
            w.write_v(self.dst, lane, v)?
        }
        Ok(())
    }

    /// Per-lane 64-bit result into the `dst` pair.
    fn map2(
        &self,
        w: &mut Wave,
        f: impl Fn(&Self, &Wave, usize) -> SimResult<u64>,
    ) -> SimResult<()> {
        self.plain()?;
        let mut out = Vec::with_capacity(WAVE);
        for lane in self.lanes(w) {
            out.push((lane, f(self, w, lane)?))
        }
        for (lane, v) in out {
            w.write_v64(self.dst, lane, v)?
        }
        Ok(())
    }

    /// Per-lane 32-bit result plus a mask bit written to `mask_dst`.
    fn map1_mask(
        &self,
        w: &mut Wave,
        mask_dst: Src,
        f: impl Fn(&Self, &Wave, usize) -> SimResult<Option<(u32, bool)>>,
    ) -> SimResult<()> {
        let mut out = Vec::with_capacity(WAVE);
        let mut mask = 0u64;
        for lane in self.lanes(w) {
            if let Some((v, bit)) = f(self, w, lane)? {
                let v = match self.form {
                    Form::Sdwa(s) => sdwa_dst(
                        w.read_v(self.dst, lane, Ty::B32)?,
                        v,
                        s.dst_sel,
                        s.dst_unused,
                    ),
                    _ => v,
                };
                out.push((lane, v));
                mask |= u64::from(bit) << lane;
            }
        }
        for (lane, v) in out {
            w.write_v(self.dst, lane, v)?
        }
        w.write_s64(mask_dst, mask)
    }

    /// Floating-point result instruction (f32 or zero-extended f16 bits).
    fn fres(
        &self,
        w: &mut Wave,
        wd: Wd,
        fsrc: u8,
        m: (bool, bool),
        f: impl Fn(&Self, &Wave, usize) -> SimResult<Option<u32>>,
    ) -> SimResult<()> {
        self.check(w, fsrc, Out::Float(wd), 0)?;
        self.fmode(w, m.0, m.1)?;
        self.map1(w, |s, w, l| {
            Ok(f(s, w, l)?.map(|b| s.fout(w, wd, u64::from(b)) as u32))
        })
    }
    fn fres64(
        &self,
        w: &mut Wave,
        fsrc: u8,
        m: (bool, bool),
        f: impl Fn(&Self, &Wave, usize) -> SimResult<u64>,
    ) -> SimResult<()> {
        self.check(w, fsrc, Out::Float(Wd::F64), 0)?;
        self.fmode(w, m.0, m.1)?;
        self.map2(w, |s, w, l| Ok(s.fout(w, Wd::F64, f(s, w, l)?)))
    }
    /// Integer/bit result instruction.
    fn ires(
        &self,
        w: &mut Wave,
        out: Out,
        fsrc: u8,
        m: (bool, bool),
        f: impl Fn(&Self, &Wave, usize) -> SimResult<Option<u32>>,
    ) -> SimResult<()> {
        self.check(w, fsrc, out, 0)?;
        self.fmode(w, m.0, m.1)?;
        self.map1(w, f)
    }

    fn run(&self, w: &mut Wave, op: Vop3Op) -> SimResult<()> {
        match op {
            Vop3Op::V1(o) => self.vop1(w, o),
            Vop3Op::V2(o) => self.vop2(w, o),
            Vop3Op::C(o) => self.vopc(w, o),
            Vop3Op::Only(o) => self.vop3(w, o),
        }
    }

    // ------------------------------------------------------------------------
    // VOP1 (CDNA3 ISA §12.8 p.178-205)

    fn vop1(&self, w: &mut Wave, op: Vop1Op) -> SimResult<()> {
        use Vop1Op::*;
        const F32: (bool, bool) = (true, false);
        const BOTH: (bool, bool) = (true, true);
        match op {
            Nop => self.check(w, 0, Out::None, 0),
            // §12.8 p.178: floating-point input modifiers allowed.
            MovB32 => self.ires(w, Out::None, 1, (false, false), |s, w, l| {
                Ok(Some(s.fmod(0, u64::from(opt!(s.u(w, 0, l))), 31) as u32))
            }),
            // §12.8 p.179: lowest active lane, lane 0 when EXEC = 0; ignores EXEC.
            ReadfirstlaneB32 => {
                self.plain()?;
                self.check(w, 0, Out::None, 0)?;
                let lane = if w.exec == 0 {
                    0
                } else {
                    w.exec.trailing_zeros() as usize
                };
                let v = w.read_v(self.src_k(0)?, lane, Ty::B32)?;
                w.write_s(self.dst, v)
            }
            CvtF32I32 => self.fres(w, Wd::F32, 0, F32, |s, w, l| {
                Ok(Some((opt!(s.u(w, 0, l)) as i32 as f32).to_bits()))
            }),
            CvtF32U32 => self.fres(w, Wd::F32, 0, F32, |s, w, l| {
                Ok(Some((opt!(s.u(w, 0, l)) as f32).to_bits()))
            }),
            // §12.8 p.180-181: truncation, out-of-range saturates, NaN → 0;
            // CLAMP only enables INEXACT.
            CvtU32F32 => self.ires(w, Out::Ignore, 1, F32, |s, w, l| {
                Ok(Some(f32::from_bits(opt!(s.f(w, 0, l))) as u32))
            }),
            CvtI32F32 => self.ires(w, Out::Ignore, 1, F32, |s, w, l| {
                Ok(Some(f32::from_bits(opt!(s.f(w, 0, l))) as i32 as u32))
            }),
            // f16 NaNs keep sign and the top payload bits, quieted (float.rs).
            CvtF16F32 => self.fres(w, Wd::F16, 1, BOTH, |s, w, l| {
                Ok(Some(u32::from(float::f32_to_f16_rne(f(
                    opt!(s.f(w, 0, l)),
                )))))
            }),
            CvtF32F16 => self.fres(w, Wd::F32, 1, BOTH, |s, w, l| {
                Ok(Some(h(opt!(s.h(w, 0, l))).to_bits()))
            }),
            // §12.8 p.181-182: f32_to_i32(floor(S0 + 0.5F)), f32_to_i32(floor(S0)).
            CvtRpiI32F32 => self.ires(w, Out::None, 1, F32, |s, w, l| {
                Ok(Some(
                    (f32::from_bits(opt!(s.f(w, 0, l))) + 0.5).floor() as i32 as u32,
                ))
            }),
            CvtFlrI32F32 => self.ires(w, Out::None, 1, F32, |s, w, l| {
                Ok(Some(
                    f32::from_bits(opt!(s.f(w, 0, l))).floor() as i32 as u32
                ))
            }),
            CvtF32Ubyte0 | CvtF32Ubyte1 | CvtF32Ubyte2 | CvtF32Ubyte3 => {
                let sh = 8 * (op as u32 - CvtF32Ubyte0 as u32);
                self.fres(w, Wd::F32, 0, F32, move |s, w, l| {
                    Ok(Some(((opt!(s.u(w, 0, l)) >> sh & 0xff) as f32).to_bits()))
                })
            }
            FractF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(fract32(opt!(s.f(w, 0, l)))))
            }),
            TruncF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(round32(opt!(s.f(w, 0, l)), f32::trunc)))
            }),
            CeilF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(round32(opt!(s.f(w, 0, l)), ceil_isa)))
            }),
            RndneF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(round32(opt!(s.f(w, 0, l)), f32::round_ties_even)))
            }),
            FloorF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(round32(opt!(s.f(w, 0, l)), floor_isa)))
            }),
            ExpF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(exp32(opt!(s.f(w, 0, l)))))
            }),
            LogF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(log32(opt!(s.f(w, 0, l)))))
            }),
            RcpF32 | RcpIflagF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(rcp32(opt!(s.f(w, 0, l)))))
            }),
            RsqF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(rsq32(opt!(s.f(w, 0, l)))))
            }),
            SqrtF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(sqrt32(opt!(s.f(w, 0, l)))))
            }),
            NotB32 => self.ires(w, Out::None, 0, (false, false), |s, w, l| {
                Ok(Some(!opt!(s.u(w, 0, l))))
            }),
            BfrevB32 => self.ires(w, Out::None, 0, (false, false), |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)).reverse_bits()))
            }),
            // §12.8 p.192-193: -1 when no bit found.
            FfbhU32 => self.ires(w, Out::None, 0, (false, false), |s, w, l| {
                let x = opt!(s.u(w, 0, l));
                Ok(Some(if x == 0 { u32::MAX } else { x.leading_zeros() }))
            }),
            FfblB32 => self.ires(w, Out::None, 0, (false, false), |s, w, l| {
                let x = opt!(s.u(w, 0, l));
                Ok(Some(if x == 0 { u32::MAX } else { x.trailing_zeros() }))
            }),
            FfbhI32 => self.ires(w, Out::None, 0, (false, false), |s, w, l| {
                let x = opt!(s.u(w, 0, l));
                let y = x ^ ((x as i32 >> 31) as u32);
                Ok(Some(if y == 0 { u32::MAX } else { y.leading_zeros() }))
            }),
            FrexpExpI32F32 => self.ires(w, Out::None, 1, F32, |s, w, l| {
                Ok(Some(frexp32(opt!(s.f(w, 0, l))).0 as u32))
            }),
            FrexpMantF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(frexp32(opt!(s.f(w, 0, l))).1))
            }),
            // §12.8 p.195-196: floating-point modifiers allowed.
            MovB64 => {
                self.check(w, 1, Out::None, 0)?;
                self.map2(w, |s, w, l| Ok(s.fmod(0, s.q(w, 0, l)?, 63)))
            }
            // §12.8 p.203: AccVGPR to AccVGPR move.
            AccvgprMovB32 => {
                self.plain()?;
                self.check(w, 0, Out::None, 0)?;
                self.map1(w, |s, w, l| s.u(w, 0, l))
            }
            CvtF32Fp8 | CvtF32Bf8 => {
                self.cvt_f32_f8(w, if op == CvtF32Fp8 { F8::Fp8 } else { F8::Bf8 })
            }
            CvtPkF32Fp8 | CvtPkF32Bf8 => {
                self.cvt_pk_f32_f8(w, if op == CvtPkF32Fp8 { F8::Fp8 } else { F8::Bf8 })
            }
            // §12.8 p.203: tmp = D0; D0 = S0; S0 = tmp (untyped, active lanes).
            SwapB32 => {
                self.plain()?;
                self.check(w, 0, Out::None, 0)?;
                let src = self.src_k(0)?;
                for lane in self.lanes(w) {
                    let (a, b) = (
                        w.read_v(self.dst, lane, Ty::B32)?,
                        w.read_v(src, lane, Ty::B32)?,
                    );
                    w.write_v(self.dst, lane, b)?;
                    w.write_v(src, lane, a)?;
                }
                Ok(())
            }
            CvtF64F32 => {
                self.plain()?;
                self.fres64(w, 1, BOTH, |s, w, l| {
                    Ok(f32_to_f64(opt_plain(s.f(w, 0, l))?))
                })
            }
            CvtF32F64 => {
                self.plain()?;
                self.fres(w, Wd::F32, 1, BOTH, |s, w, l| {
                    Ok(Some(f64_to_f32(s.d(w, 0, l)?)))
                })
            }
        }
    }

    /// `v_cvt_f32_{fp8,bf8}` (§12.8 p.203-204, §7.2 Table 31 p.50-51): byte
    /// from SDWA SRC0_SEL (BYTE1..3; BYTE0 implied otherwise; WORD selects
    /// illegal) or, in VOP3, OP_SEL[1:0] = byte index
    /// `[UNVERIFIED: VOP3 op_sel bit order, confirm on MI300X]`. ABS/NEG/SEXT
    /// are ignored, as are the other SDWA fields.
    fn cvt_f32_f8(&self, w: &mut Wave, kind: F8) -> SimResult<()> {
        let byte = match self.form {
            Form::Sdwa(s) => match s.src0_sel {
                SdwaSel::Byte1 => 1,
                SdwaSel::Byte2 => 2,
                SdwaSel::Byte3 => 3,
                SdwaSel::Byte0 | SdwaSel::Dword => 0,
                sel => {
                    return unsupported(format!(
                        "{}: SDWA src0_sel {sel:?} (only BYTE selects are legal)",
                        self.name
                    ))
                }
            },
            Form::Vop3 => u32::from(self.op_sel & 3),
            Form::E32 => 0,
            Form::Dpp(_) => {
                return unsupported(format!(
                    "{}: DPP form not modelled for this instruction",
                    self.name
                ))
            }
        };
        if self.clamp || self.omod != 0 {
            return unsupported(format!(
                "{}: clamp/omod on an FP8 → F32 conversion",
                self.name
            ));
        }
        let opsel = if self.is_vop3() { 3 } else { 0 };
        if self.op_sel & !opsel != 0 {
            return unsupported(format!("{}: op_sel {:#06b}", self.name, self.op_sel));
        }
        let enc = w.arch.caps().fp8;
        let src = self.src_k(0)?;
        let mut out = Vec::with_capacity(WAVE);
        for lane in self.lanes(w) {
            let code = (w.read_v(src, lane, Ty::B32)? >> (8 * byte)) as u8;
            out.push((lane, float::f8_to_f32(enc, kind, code).to_bits()));
        }
        for (lane, v) in out {
            w.write_v(self.dst, lane, v)?
        }
        Ok(())
    }

    /// `v_cvt_pk_f32_{fp8,bf8}` (§12.8 p.204-205): word from SDWA
    /// SRC0_SEL[1:0] == WORD1 (byte selects illegal) or VOP3 OP_SEL[0]
    /// `[UNVERIFIED: VOP3 op_sel, confirm on MI300X]`; writes dst, dst+1
    /// (dst must be even, §7.2 Table 31 p.51).
    fn cvt_pk_f32_f8(&self, w: &mut Wave, kind: F8) -> SimResult<()> {
        let hi = match self.form {
            Form::Sdwa(s) => match s.src0_sel {
                SdwaSel::Word1 => true,
                SdwaSel::Word0 | SdwaSel::Dword => false,
                sel => {
                    return unsupported(format!(
                        "{}: SDWA src0_sel {sel:?} (only WORD selects are legal)",
                        self.name
                    ))
                }
            },
            Form::Vop3 => self.op_sel & 1 != 0,
            Form::E32 => false,
            Form::Dpp(_) => {
                return unsupported(format!(
                    "{}: DPP form not modelled for this instruction",
                    self.name
                ))
            }
        };
        if self.clamp || self.omod != 0 {
            return unsupported(format!(
                "{}: clamp/omod on an FP8 → F32 conversion",
                self.name
            ));
        }
        let opsel = if self.is_vop3() { 1 } else { 0 };
        if self.op_sel & !opsel != 0 {
            return unsupported(format!("{}: op_sel {:#06b}", self.name, self.op_sel));
        }
        match self.dst {
            Src::Vgpr(n) if n % 2 == 0 => {}
            d => {
                return fault(format!(
                    "{}: destination {d:?} must be an even VGPR",
                    self.name
                ))
            }
        }
        let enc = w.arch.caps().fp8;
        let src = self.src_k(0)?;
        let mut out = Vec::with_capacity(WAVE);
        for lane in self.lanes(w) {
            let word = w.read_v(src, lane, Ty::B32)? >> if hi { 16 } else { 0 };
            let lo = float::f8_to_f32(enc, kind, word as u8).to_bits();
            let up = float::f8_to_f32(enc, kind, (word >> 8) as u8).to_bits();
            out.push((lane, u64::from(lo) | u64::from(up) << 32));
        }
        for (lane, v) in out {
            w.write_v64(self.dst, lane, v)?
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    // VOP2 (CDNA3 ISA §12.7 p.160-177)

    fn carry_dst(&self) -> SimResult<Src> {
        if self.is_vop3() {
            self.sdst.ok_or_else(|| {
                SimError::Fault(format!(
                    "{}: VOP3B form without an SDST carry destination",
                    self.name
                ))
            })
        } else {
            Ok(Src::VccLo)
        }
    }
    /// VCC (e32/DPP/SDWA) or the SGPR pair in S2 (VOP3): carry-in / select.
    fn lane_mask_in(&self, w: &Wave) -> SimResult<u64> {
        if self.is_vop3() {
            w.read_s64(self.src_k(2)?, Ty::B64)
        } else {
            w.read_s64(Src::VccLo, Ty::B64)
        }
    }

    fn vop2(&self, w: &mut Wave, op: Vop2Op) -> SimResult<()> {
        use Vop2Op::*;
        const F32: (bool, bool) = (true, false);
        const F16: (bool, bool) = (false, true);
        const NONE: (bool, bool) = (false, false);
        let ieee = w.mode.ieee;
        match op {
            // §12.7 p.160: D = VCC[lane] ? S1 : S0; float modifiers allowed.
            CndmaskB32 => {
                let mask = self.lane_mask_in(w)?;
                self.ires(w, Out::None, 3, NONE, move |s, w, l| {
                    let k = if mask >> l & 1 != 0 { 1 } else { 0 };
                    Ok(Some(s.fmod(k, u64::from(opt!(s.u(w, k, l))), 31) as u32))
                })
            }
            AddF32 => self.fres(w, Wd::F32, 3, F32, |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                Ok(Some(arith32(&[a, b], || f(a) + f(b))))
            }),
            SubF32 => self.fres(w, Wd::F32, 3, F32, |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                Ok(Some(arith32(&[a, b], || f(a) - f(b))))
            }),
            SubrevF32 => self.fres(w, Wd::F32, 3, F32, |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                Ok(Some(arith32(&[a, b], || f(b) - f(a))))
            }),
            MulF32 => self.fres(w, Wd::F32, 3, F32, |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                Ok(Some(arith32(&[a, b], || f(a) * f(b))))
            }),
            // §12.7 p.162-163: 24-bit operands from bits [23:0].
            MulI32I24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    (sext24(opt!(s.u(w, 0, l))) * sext24(opt!(s.u(w, 1, l)))) as u32,
                ))
            }),
            MulHiI32I24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((sext24(opt!(s.u(w, 0, l))) * sext24(opt!(s.u(w, 1, l)))) >> 32) as u32,
                ))
            }),
            MulU32U24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((u64::from(opt!(s.u(w, 0, l)) & 0xff_ffff))
                        * u64::from(opt!(s.u(w, 1, l)) & 0xff_ffff)) as u32,
                ))
            }),
            MulHiU32U24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((u64::from(opt!(s.u(w, 0, l)) & 0xff_ffff)
                        * u64::from(opt!(s.u(w, 1, l)) & 0xff_ffff))
                        >> 32) as u32,
                ))
            }),
            MinF32 => self.fres(w, Wd::F32, 3, F32, move |s, w, l| {
                Ok(Some(vmin(
                    Fmt::S,
                    ieee,
                    u64::from(opt!(s.f(w, 0, l))),
                    u64::from(opt!(s.f(w, 1, l))),
                ) as u32))
            }),
            MaxF32 => self.fres(w, Wd::F32, 3, F32, move |s, w, l| {
                Ok(Some(vmax(
                    Fmt::S,
                    ieee,
                    u64::from(opt!(s.f(w, 0, l))),
                    u64::from(opt!(s.f(w, 1, l))),
                ) as u32))
            }),
            MinI32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    (opt!(s.u(w, 0, l)) as i32).min(opt!(s.u(w, 1, l)) as i32) as u32,
                ))
            }),
            MaxI32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    (opt!(s.u(w, 0, l)) as i32).max(opt!(s.u(w, 1, l)) as i32) as u32,
                ))
            }),
            MinU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)).min(opt!(s.u(w, 1, l)))))
            }),
            MaxU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)).max(opt!(s.u(w, 1, l)))))
            }),
            LshrrevB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 1, l)) >> (opt!(s.u(w, 0, l)) & 31)))
            }),
            AshrrevI32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((opt!(s.u(w, 1, l)) as i32) >> (opt!(s.u(w, 0, l)) & 31)) as u32,
                ))
            }),
            LshlrevB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 1, l)) << (opt!(s.u(w, 0, l)) & 31)))
            }),
            AndB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)) & opt!(s.u(w, 1, l))))
            }),
            OrB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)) | opt!(s.u(w, 1, l))))
            }),
            XorB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)) ^ opt!(s.u(w, 1, l))))
            }),
            XnorB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(!(opt!(s.u(w, 0, l)) ^ opt!(s.u(w, 1, l)))))
            }),
            // §12.7 p.177: D = fma(S0, S1, D).
            FmacF32 => self.fres(w, Wd::F32, 3, F32, |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                let c = w.read_v(s.dst, l, Ty::F32)?;
                Ok(Some(arith32(&[a, b, c], || f(a).mul_add(f(b), f(c)))))
            }),
            // §12.7 p.167-169: carry/borrow out; "Supports saturation".
            AddCoU32 | SubCoU32 | SubrevCoU32 | AddcCoU32 | SubbCoU32 | SubbrevCoU32 => {
                self.check(w, 0, Out::IntSat, 0)?;
                let cin = if matches!(op, AddcCoU32 | SubbCoU32 | SubbrevCoU32) {
                    self.lane_mask_in(w)?
                } else {
                    0
                };
                let cdst = self.carry_dst()?;
                self.map1_mask(w, cdst, move |s, w, l| {
                    let (a, b) = (u64::from(opt!(s.u(w, 0, l))), u64::from(opt!(s.u(w, 1, l))));
                    let c = cin >> l & 1;
                    let (x, y, add) = match op {
                        AddCoU32 | AddcCoU32 => (a, b, true),
                        SubCoU32 | SubbCoU32 => (a, b, false),
                        _ => (b, a, false),
                    };
                    Ok(Some(if add {
                        let t = x + y + c;
                        let carry = t >> 32 != 0;
                        (if carry && s.clamp { u32::MAX } else { t as u32 }, carry)
                    } else {
                        let borrow = y + c > x;
                        (
                            if borrow && s.clamp {
                                0
                            } else {
                                x.wrapping_sub(y).wrapping_sub(c) as u32
                            },
                            borrow,
                        )
                    }))
                })
            }
            AddU32 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.u(w, 0, l)), opt!(s.u(w, 1, l)));
                Ok(Some(if s.clamp {
                    a.saturating_add(b)
                } else {
                    a.wrapping_add(b)
                }))
            }),
            SubU32 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.u(w, 0, l)), opt!(s.u(w, 1, l)));
                Ok(Some(if s.clamp {
                    a.saturating_sub(b)
                } else {
                    a.wrapping_sub(b)
                }))
            }),
            SubrevU32 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.u(w, 0, l)), opt!(s.u(w, 1, l)));
                Ok(Some(if s.clamp {
                    b.saturating_sub(a)
                } else {
                    b.wrapping_sub(a)
                }))
            }),
            AddF16 => self.fres(w, Wd::F16, 3, F16, |s, w, l| {
                let (a, b) = (opt!(s.h(w, 0, l)), opt!(s.h(w, 1, l)));
                Ok(Some(arith16(&[a, b], || h(a) + h(b))))
            }),
            SubF16 => self.fres(w, Wd::F16, 3, F16, |s, w, l| {
                let (a, b) = (opt!(s.h(w, 0, l)), opt!(s.h(w, 1, l)));
                Ok(Some(arith16(&[a, b], || h(a) - h(b))))
            }),
            MulF16 => self.fres(w, Wd::F16, 3, F16, |s, w, l| {
                let (a, b) = (opt!(s.h(w, 0, l)), opt!(s.h(w, 1, l)));
                Ok(Some(arith16(&[a, b], || h(a) * h(b))))
            }),
            MaxF16 => self.fres(w, Wd::F16, 3, F16, move |s, w, l| {
                Ok(Some(vmax(
                    Fmt::H,
                    ieee,
                    u64::from(opt!(s.h(w, 0, l))),
                    u64::from(opt!(s.h(w, 1, l))),
                ) as u32))
            }),
            MinF16 => self.fres(w, Wd::F16, 3, F16, move |s, w, l| {
                Ok(Some(vmin(
                    Fmt::H,
                    ieee,
                    u64::from(opt!(s.h(w, 0, l))),
                    u64::from(opt!(s.h(w, 1, l))),
                ) as u32))
            }),
            // §12.7 p.171-172: unsigned 16-bit; "Supports saturation".
            AddU16 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.s16(w, 0, l)), opt!(s.s16(w, 1, l)));
                let t = a + b;
                Ok(Some(if s.clamp { t.min(0xffff) } else { t & 0xffff }))
            }),
            SubU16 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.s16(w, 0, l)), opt!(s.s16(w, 1, l)));
                Ok(Some(if s.clamp {
                    a.saturating_sub(b)
                } else {
                    a.wrapping_sub(b) & 0xffff
                }))
            }),
            MulLoU16 => self.ires(w, Out::IntSat, 0, NONE, |s, w, l| {
                let (a, b) = (opt!(s.s16(w, 0, l)), opt!(s.s16(w, 1, l)));
                let t = a * b;
                Ok(Some(if s.clamp { t.min(0xffff) } else { t & 0xffff }))
            }),
            LshlrevB16 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    (opt!(s.s16(w, 1, l)) << (opt!(s.u(w, 0, l)) & 15)) & 0xffff,
                ))
            }),
        }
    }

    // ------------------------------------------------------------------------
    // VOPC (CDNA3 ISA §12.9 p.206-253; Table 58/60)

    fn vopc(&self, w: &mut Wave, op: VopcOp) -> SimResult<()> {
        use VopcOp::*;
        let cmpx = matches!(
            op,
            CmpxEqU32
                | CmpxNeU32
                | CmpxGtU32
                | CmpxLtU32
                | CmpxGtI32
                | CmpxLtI32
                | CmpxGtF32
                | CmpxLtF32
        );
        let dst = match self.form {
            Form::Vop3 => self.dst,
            Form::Sdwa(_) => self.sdst.unwrap_or(Src::VccLo),
            Form::Dpp(_) => {
                return unsupported(format!("{}: VOPC DPP has no gfx9 encoding", self.name))
            }
            Form::E32 => Src::VccLo,
        };
        let mut mask = 0u64;
        match op {
            // §12.9 p.208: class bits of S1 (integer operand).
            CmpClassF32 => {
                self.check(w, 1, Out::Ignore, 0)?;
                for l in self.lanes(w) {
                    let (Some(a), Some(m)) = (self.f(w, 0, l)?, self.u(w, 1, l)?) else {
                        continue;
                    };
                    mask |= u64::from(m >> class32(a) & 1) << l;
                }
            }
            CmpLtF32 | CmpEqF32 | CmpLeF32 | CmpGtF32 | CmpLgF32 | CmpGeF32 | CmpOF32 | CmpUF32
            | CmpNeqF32 | CmpxGtF32 | CmpxLtF32 => {
                self.check(w, 3, Out::Ignore, 0)?;
                self.fmode(w, true, false)?;
                for l in self.lanes(w) {
                    let (Some(a), Some(b)) = (self.f(w, 0, l)?, self.f(w, 1, l)?) else {
                        continue;
                    };
                    let (x, y) = (f(a), f(b));
                    let r = match op {
                        CmpLtF32 | CmpxLtF32 => x < y,
                        CmpEqF32 => x == y,
                        CmpLeF32 => x <= y,
                        CmpGtF32 | CmpxGtF32 => x > y,
                        // LG = ordered and not equal.
                        CmpLgF32 => x < y || x > y,
                        CmpGeF32 => x >= y,
                        CmpOF32 => !x.is_nan() && !y.is_nan(),
                        // Table 58 prints `!isNaN(S0) || !isNaN(S1)` for U, which
                        // contradicts its name and O; unordered = either is NaN
                        // (LLVM maps v_cmp_u_f32 to SETUO).
                        CmpUF32 => x.is_nan() || y.is_nan(),
                        _ => x != y, // NEQ = !(S0 == S1)
                    };
                    mask |= u64::from(r) << l;
                }
            }
            CmpEqU64 | CmpNeU64 => {
                self.plain()?;
                self.check(w, 0, Out::Ignore, 0)?;
                for l in self.lanes(w) {
                    let (a, b) = (self.q(w, 0, l)?, self.q(w, 1, l)?);
                    mask |= u64::from(if op == CmpEqU64 { a == b } else { a != b }) << l;
                }
            }
            _ => {
                self.check(w, 0, Out::Ignore, 0)?;
                for l in self.lanes(w) {
                    let (Some(a), Some(b)) = (self.u(w, 0, l)?, self.u(w, 1, l)?) else {
                        continue;
                    };
                    let (i, j) = (a as i32, b as i32);
                    let r = match op {
                        CmpLtI32 | CmpxLtI32 => i < j,
                        CmpEqI32 => i == j,
                        CmpLeI32 => i <= j,
                        CmpGtI32 | CmpxGtI32 => i > j,
                        CmpNeI32 => i != j,
                        CmpGeI32 => i >= j,
                        CmpLtU32 | CmpxLtU32 => a < b,
                        CmpEqU32 | CmpxEqU32 => a == b,
                        CmpLeU32 => a <= b,
                        CmpGtU32 | CmpxGtU32 => a > b,
                        CmpNeU32 | CmpxNeU32 => a != b,
                        CmpGeU32 => a >= b,
                        _ => unreachable!("float/class/u64 compares handled above"),
                    };
                    mask |= u64::from(r) << l;
                }
            }
        }
        w.write_s64(dst, mask)?;
        // §6.2.2 p.33: V_CMPX writes the result to both the SGPR pair and EXEC.
        if cmpx {
            w.write_s64(Src::ExecLo, mask)?
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    // VOP3-only (CDNA3 ISA §12.11 p.284-409)

    fn vop3(&self, w: &mut Wave, op: Vop3OnlyOp) -> SimResult<()> {
        use Vop3OnlyOp::*;
        const F32: (bool, bool) = (true, false);
        const F64: (bool, bool) = (false, true);
        const NONE: (bool, bool) = (false, false);
        if !self.is_vop3() {
            return fault(format!(
                "{}: VOP3-only opcode outside the VOP3 encoding",
                self.name
            ));
        }
        let ieee = w.mode.ieee;
        macro_rules! u3 {
            ($s:ident, $w:ident, $l:ident) => {
                (
                    opt!($s.u($w, 0, $l)),
                    opt!($s.u($w, 1, $l)),
                    opt!($s.u($w, 2, $l)),
                )
            };
        }
        macro_rules! f3 {
            ($s:ident, $w:ident, $l:ident) => {
                (
                    opt!($s.f($w, 0, $l)),
                    opt!($s.f($w, 1, $l)),
                    opt!($s.f($w, 2, $l)),
                )
            };
        }
        match op {
            // §12.11 p.328: D = S0.i24 * S1.i24 + S2.i32.
            MadI32I24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(((sext24(a) * sext24(b)) as u32).wrapping_add(c)))
            }),
            MadU32U24 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(
                    ((a & 0xff_ffff).wrapping_mul(b & 0xff_ffff)).wrapping_add(c),
                ))
            }),
            FmaF32 => self.fres(w, Wd::F32, 7, F32, |s, w, l| {
                let (a, b, c) = f3!(s, w, l);
                Ok(Some(arith32(&[a, b, c], || f(a).mul_add(f(b), f(c)))))
            }),
            // §12.11 p.330.
            BfeU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(bfe_u(a, b, c)))
            }),
            BfeI32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                let width = c & 31;
                let t = bfe_u(a, b, c);
                Ok(Some(if width == 0 {
                    0
                } else {
                    ((t << (32 - width)) as i32 >> (32 - width)) as u32
                }))
            }),
            BfiB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a & b) | (!a & c)))
            }),
            // §12.11 p.331: S0 carries the MSBs.
            AlignbitB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(
                    ((u64::from(a) << 32 | u64::from(b)) >> (c & 31)) as u32,
                ))
            }),
            AlignbyteB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(
                    ((u64::from(a) << 32 | u64::from(b)) >> ((c & 3) * 8)) as u32,
                ))
            }),
            // §12.11 p.332-333.
            Min3F32 => self.fres(w, Wd::F32, 7, F32, move |s, w, l| {
                let (a, b, c) = f3!(s, w, l);
                Ok(Some(vmin(
                    Fmt::S,
                    ieee,
                    vmin(Fmt::S, ieee, a.into(), b.into()),
                    c.into(),
                ) as u32))
            }),
            Max3F32 => self.fres(w, Wd::F32, 7, F32, move |s, w, l| {
                let (a, b, c) = f3!(s, w, l);
                Ok(Some(vmax(
                    Fmt::S,
                    ieee,
                    vmax(Fmt::S, ieee, a.into(), b.into()),
                    c.into(),
                ) as u32))
            }),
            Med3F32 => self.fres(w, Wd::F32, 7, F32, move |s, w, l| {
                let (a, b, c) = f3!(s, w, l);
                Ok(Some(med3f(ieee, a, b, c)))
            }),
            Min3I32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a as i32).min(b as i32).min(c as i32) as u32))
            }),
            Max3I32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a as i32).max(b as i32).max(c as i32) as u32))
            }),
            Min3U32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(a.min(b).min(c)))
            }),
            Max3U32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(a.max(b).max(c)))
            }),
            Med3I32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(med3i(a as i32, b as i32, c as i32) as u32))
            }),
            Med3U32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(med3i(a, b, c)))
            }),
            DivScaleF32 => {
                self.check(w, 7, Out::None, 0)?;
                self.fmode(w, true, false)?;
                let cdst = self.carry_dst()?;
                self.map1_mask(w, cdst, |s, w, l| {
                    let (a, b, c) = f3!(s, w, l);
                    Ok(Some(div_scale32(a, b, c)))
                })
            }
            DivFmasF32 => {
                let vcc = w.read_s64(Src::VccLo, Ty::B64)?;
                self.fres(w, Wd::F32, 7, F32, move |s, w, l| {
                    let (a, b, c) = f3!(s, w, l);
                    Ok(Some(div_fmas32(a, b, c, vcc >> l & 1 != 0)))
                })
            }
            DivFixupF32 => self.fres(w, Wd::F32, 7, F32, |s, w, l| {
                let (a, b, c) = f3!(s, w, l);
                Ok(Some(div_fixup32(a, b, c)))
            }),
            // §12.11 p.341: {D1.u1, D0.u64} = S0 * S1 + S2 (65-bit).
            MadU64U32 | MadI64I32 => {
                self.check(w, 0, Out::None, 0)?;
                let cdst = self.carry_dst()?;
                let mut out = Vec::with_capacity(WAVE);
                let mut mask = 0u64;
                for l in self.lanes(w) {
                    let (a, b, c) = (
                        self.u(w, 0, l)?.unwrap_or(0),
                        self.u(w, 1, l)?.unwrap_or(0),
                        self.q(w, 2, l)?,
                    );
                    let t: i128 = if op == MadU64U32 {
                        i128::from(a) * i128::from(b) + i128::from(c)
                    } else {
                        i128::from(a as i32) * i128::from(b as i32) + i128::from(c as i64)
                    };
                    out.push((l, t as u64));
                    mask |= ((t >> 64) as u64 & 1) << l;
                }
                for (l, v) in out {
                    w.write_v64(self.dst, l, v)?
                }
                w.write_s64(cdst, mask)
            }
            // §12.11 p.348.
            LshlAddU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a << (b & 31)).wrapping_add(c)))
            }),
            AddLshlU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(a.wrapping_add(b) << (c & 31)))
            }),
            Add3U32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(a.wrapping_add(b).wrapping_add(c)))
            }),
            LshlOrB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a << (b & 31)) | c))
            }),
            AndOrB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a & b) | c))
            }),
            Or3B32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(a | b | c))
            }),
            XadU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some((a ^ b).wrapping_add(c)))
            }),
            // §12.11 p.343-344.
            PermB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let (a, b, c) = u3!(s, w, l);
                Ok(Some(perm(a, b, c)))
            }),
            // §12.11 p.353.
            MulLoU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(opt!(s.u(w, 0, l)).wrapping_mul(opt!(s.u(w, 1, l)))))
            }),
            MulHiU32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((u64::from(opt!(s.u(w, 0, l))) * u64::from(opt!(s.u(w, 1, l)))) >> 32) as u32,
                ))
            }),
            MulHiI32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((i64::from(opt!(s.u(w, 0, l)) as i32) * i64::from(opt!(s.u(w, 1, l)) as i32))
                        >> 32) as u32,
                ))
            }),
            LdexpF32 => self.fres(w, Wd::F32, 1, F32, |s, w, l| {
                Ok(Some(ldexp32(opt!(s.f(w, 0, l)), opt!(s.u(w, 1, l)) as i32)))
            }),
            CvtPkFp8F32 | CvtPkBf8F32 => {
                let kind = if op == CvtPkFp8F32 { F8::Fp8 } else { F8::Bf8 };
                self.f8_inputs_vgpr(2)?;
                self.check(w, 3, Out::Ignore, 0b1000)?;
                self.fmode(w, false, false)?;
                let (enc, ovfl) = (w.arch.caps().fp8, w.mode.fp16_ovfl);
                self.map1(w, move |s, w, l| {
                    let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                    let pk = u32::from(float::f32_to_f8_rne(enc, kind, f(a), ovfl))
                        | u32::from(float::f32_to_f8_rne(enc, kind, f(b), ovfl)) << 8;
                    let old = w.read_v(s.dst, l, Ty::B32)?;
                    Ok(Some(if s.op_sel & 8 != 0 {
                        (old & 0xffff) | pk << 16
                    } else {
                        (old & 0xffff_0000) | pk
                    }))
                })
            }
            CvtSrFp8F32 | CvtSrBf8F32 => {
                let kind = if op == CvtSrFp8F32 { F8::Fp8 } else { F8::Bf8 };
                self.f8_inputs_vgpr(2)?;
                self.check(w, 1, Out::Ignore, 0b1100)?;
                let (enc, ovfl) = (w.arch.caps().fp8, w.mode.fp16_ovfl);
                self.map1(w, move |s, w, l| {
                    let (a, r) = (opt!(s.f(w, 0, l)), opt!(s.u(w, 1, l)));
                    let byte = u32::from(float::f32_to_f8_sr(enc, kind, f(a), r, ovfl));
                    let sh = 8 * u32::from(s.op_sel >> 2 & 3);
                    let old = w.read_v(s.dst, l, Ty::B32)?;
                    Ok(Some((old & !(0xff << sh)) | byte << sh))
                })
            }
            // §12.11 p.358: round toward zero regardless of MODE.
            CvtPkrtzF16F32 => self.ires(w, Out::None, 3, (false, false), |s, w, l| {
                let (a, b) = (opt!(s.f(w, 0, l)), opt!(s.f(w, 1, l)));
                Ok(Some(
                    u32::from(float::f32_to_f16_rtz(f(a)))
                        | u32::from(float::f32_to_f16_rtz(f(b))) << 16,
                ))
            }),
            // §12.11 p.359: u32_to_u16 / i32_to_i16 saturate
            // `[UNVERIFIED: saturation vs truncation not spelled out, confirm on MI300X]`.
            CvtPkU16U32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    opt!(s.u(w, 0, l)).min(0xffff) | opt!(s.u(w, 1, l)).min(0xffff) << 16,
                ))
            }),
            CvtPkI16I32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let sat = |x: u32| ((x as i32).clamp(-0x8000, 0x7fff) as u32) & 0xffff;
                Ok(Some(
                    sat(opt!(s.u(w, 0, l))) | sat(opt!(s.u(w, 1, l))) << 16,
                ))
            }),
            // §12.11 p.356-357: shift count S0[5:0], value S1.u64.
            LshlrevB64 | LshrrevB64 | AshrrevI64 => {
                self.check(w, 0, Out::None, 0)?;
                self.map2(w, |s, w, l| {
                    let (sh, x) = (s.u(w, 0, l)?.unwrap_or(0) & 63, s.q(w, 1, l)?);
                    Ok(match op {
                        LshlrevB64 => x << sh,
                        LshrrevB64 => x >> sh,
                        _ => ((x as i64) >> sh) as u64,
                    })
                })
            }
            // §12.11 p.354: lane = S1[5:0]; ignores EXEC.
            ReadlaneB32 => {
                self.check(w, 0, Out::None, 0)?;
                let lane = (w.read_s(self.src_k(1)?, Ty::B32)? & 63) as usize;
                let v = w.read_v(self.src_k(0)?, lane, Ty::B32)?;
                w.write_s(self.dst, v)
            }
            // §12.11 p.355: VGPR[S1[5:0]][VDST] = S0; ignores EXEC.
            WritelaneB32 => {
                self.check(w, 0, Out::None, 0)?;
                let lane = (w.read_s(self.src_k(1)?, Ty::B32)? & 63) as usize;
                let v = w.read_s(self.src_k(0)?, Ty::B32)?;
                w.write_v(self.dst, lane, v)
            }
            BcntU32B32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    opt!(s.u(w, 0, l))
                        .count_ones()
                        .wrapping_add(opt!(s.u(w, 1, l))),
                ))
            }),
            // §12.11 p.355-356: ThreadMask = (1 << lane) - 1.
            MbcntLoU32B32 | MbcntHiU32B32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                let tm = (1u64 << l) - 1;
                let half = if op == MbcntLoU32B32 {
                    tm as u32
                } else {
                    (tm >> 32) as u32
                };
                Ok(Some(
                    (opt!(s.u(w, 0, l)) & half)
                        .count_ones()
                        .wrapping_add(opt!(s.u(w, 1, l))),
                ))
            }),
            // §12.11 p.361: D[31:16] = S1.f16, D[15:0] = S0.f16; gfx9
            // OP_SEL[0]/[1] select the high half of S0/S1.
            PackB32F16 => {
                self.check(w, 3, Out::None, 0b11)?;
                self.map1(w, |s, w, l| {
                    let half = |k: usize, x: u32| {
                        s.fmod(
                            k,
                            u64::from(if s.op_sel >> k & 1 != 0 {
                                x >> 16
                            } else {
                                x & 0xffff
                            }),
                            15,
                        ) as u32
                    };
                    let (a, b) = (opt!(s.get(w, 0, l, Ty::F16)), opt!(s.get(w, 1, l, Ty::F16)));
                    Ok(Some(half(0, a) | half(1, b) << 16))
                })
            }
            // §12.11 p.358.
            BfmB32 => self.ires(w, Out::None, 0, NONE, |s, w, l| {
                Ok(Some(
                    ((1u32 << (opt!(s.u(w, 0, l)) & 31)) - 1) << (opt!(s.u(w, 1, l)) & 31),
                ))
            }),
            AddF64 => self.fres64(w, 3, F64, |s, w, l| {
                let (a, b) = (s.d(w, 0, l)?, s.d(w, 1, l)?);
                Ok(arith64(&[a, b], || d(a) + d(b)))
            }),
            MulF64 => self.fres64(w, 3, F64, |s, w, l| {
                let (a, b) = (s.d(w, 0, l)?, s.d(w, 1, l)?);
                Ok(arith64(&[a, b], || d(a) * d(b)))
            }),
            FmaF64 => self.fres64(w, 7, F64, |s, w, l| {
                let (a, b, c) = (s.d(w, 0, l)?, s.d(w, 1, l)?, s.d(w, 2, l)?);
                Ok(arith64(&[a, b, c], || d(a).mul_add(d(b), d(c))))
            }),
            // §12.11 p.351: D = (S0.u64 << S1[2:0]) + S2.u64; shift counts above 4
            // are unsupported by the hardware.
            LshlAddU64 => {
                self.check(w, 0, Out::None, 0)?;
                for l in self.lanes(w) {
                    let sh = self.u(w, 1, l)?.unwrap_or(0) & 7;
                    if sh > 4 {
                        return unsupported(format!(
                            "{}: shift count {sh} > 4 (lane {l}) is unsupported by the hardware",
                            self.name
                        ));
                    }
                }
                self.map2(w, |s, w, l| {
                    let sh = s.u(w, 1, l)?.unwrap_or(0) & 7;
                    Ok((s.q(w, 0, l)? << sh).wrapping_add(s.q(w, 2, l)?))
                })
            }
        }
    }

    /// §7.2 p.51: CVT_SR_* and CVT_PK_* take only VGPR inputs.
    fn f8_inputs_vgpr(&self, n: usize) -> SimResult<()> {
        for k in 0..n {
            let s = self.src_k(k)?;
            if !s.is_vgpr() {
                return unsupported(format!(
                    "{}: src{k} {s:?} is not a VGPR (CDNA3 ISA §7.2 p.51: VGPR inputs only)",
                    self.name
                ));
            }
        }
        Ok(())
    }
}

/// `Option<T>` → value for paths that run under `plain` (no DPP: never `None`).
fn opt_plain(v: SimResult<Option<u32>>) -> SimResult<u32> {
    v?.ok_or_else(|| SimError::Fault("lane value unavailable outside DPP".into()))
}

// ----------------------------------------------------------------------------
// Integer helpers

fn sext24(x: u32) -> i64 {
    i64::from(((x << 8) as i32) >> 8)
}

fn bfe_u(a: u32, off: u32, width: u32) -> u32 {
    let w = width & 31;
    (a >> (off & 31)) & ((1u32 << w).wrapping_sub(1))
}

fn med3i<T: Ord + Copy>(a: T, b: T, c: T) -> T {
    let mx = a.max(b).max(c);
    if mx == a {
        b.max(c)
    } else if mx == b {
        a.max(c)
    } else {
        a.max(b)
    }
}

/// §12.11 p.343-344 BYTE_PERMUTE over {S0, S1} (S0 = bytes 7..4).
fn perm(s0: u32, s1: u32, sel: u32) -> u32 {
    let data = u64::from(s0) << 32 | u64::from(s1);
    let byte = |i: u32| (data >> (8 * i)) as u8;
    let pick = |sel: u32| -> u32 {
        u32::from(match sel {
            13.. => 0xff,
            12 => 0,
            11 => {
                if byte(7) & 0x80 != 0 {
                    0xff
                } else {
                    0
                }
            }
            10 => {
                if byte(5) & 0x80 != 0 {
                    0xff
                } else {
                    0
                }
            }
            9 => {
                if byte(3) & 0x80 != 0 {
                    0xff
                } else {
                    0
                }
            }
            8 => {
                if byte(1) & 0x80 != 0 {
                    0xff
                } else {
                    0
                }
            }
            i => byte(i),
        })
    };
    (0..4).fold(0, |acc, i| acc | pick(sel >> (8 * i) & 0xff) << (8 * i))
}

// ----------------------------------------------------------------------------
// Floating-point helpers

fn f(b: u32) -> f32 {
    f32::from_bits(b)
}
fn d(b: u64) -> f64 {
    f64::from_bits(b)
}
fn h(b: u32) -> f32 {
    float::f16_to_f32(b as u16)
}

const QNAN16: u32 = 0xfe00;
const QNAN32: u32 = 0xffc0_0000;
const QNAN64: u64 = 0xfff8_0000_0000_0000;

fn nan32(b: u32) -> bool {
    b & 0x7fff_ffff > 0x7f80_0000
}

/// Binary format parameters for the generic min/max/clamp helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fmt {
    H,
    S,
    D,
}
impl Fmt {
    fn of(wd: Wd) -> Self {
        match wd {
            Wd::F16 => Fmt::H,
            Wd::F32 => Fmt::S,
            Wd::F64 => Fmt::D,
        }
    }
    fn sign(self) -> u64 {
        match self {
            Fmt::H => 0x8000,
            Fmt::S => 0x8000_0000,
            Fmt::D => 1 << 63,
        }
    }
    fn inf(self) -> u64 {
        match self {
            Fmt::H => 0x7c00,
            Fmt::S => 0x7f80_0000,
            Fmt::D => 0x7ff0_0000_0000_0000,
        }
    }
    fn quiet_bit(self) -> u64 {
        match self {
            Fmt::H => 0x200,
            Fmt::S => 0x40_0000,
            Fmt::D => 1 << 51,
        }
    }
    fn one(self) -> u64 {
        match self {
            Fmt::H => 0x3c00,
            Fmt::S => 0x3f80_0000,
            Fmt::D => 0x3ff0_0000_0000_0000,
        }
    }
    fn mag(self, b: u64) -> u64 {
        b & (self.sign() - 1)
    }
    fn is_nan(self, b: u64) -> bool {
        self.mag(b) > self.inf()
    }
    fn is_snan(self, b: u64) -> bool {
        self.is_nan(b) && b & self.quiet_bit() == 0
    }
    fn quiet(self, b: u64) -> u64 {
        b | self.quiet_bit()
    }
    fn is_neg(self, b: u64) -> bool {
        b & self.sign() != 0
    }
    fn val(self, b: u64) -> f64 {
        match self {
            Fmt::H => f64::from(h(b as u32)),
            Fmt::S => f64::from(f(b as u32)),
            Fmt::D => d(b),
        }
    }
}

/// V_MIN_F32 / V_MIN_F16 (§12.7 p.163-164, p.173-174).
fn vmin(t: Fmt, ieee: bool, a: u64, b: u64) -> u64 {
    if ieee && t.is_snan(a) {
        return t.quiet(a);
    }
    if ieee && t.is_snan(b) {
        return t.quiet(b);
    }
    if t.is_nan(a) {
        return b;
    }
    if t.is_nan(b) {
        return a;
    }
    let (x, y) = (t.val(a), t.val(b));
    if x == 0.0 && y == 0.0 && t.is_neg(a) != t.is_neg(b) {
        return if t.is_neg(a) { a } else { b };
    }
    if x < y {
        a
    } else {
        b
    }
}

/// V_MAX_F32 / V_MAX_F16 (§12.7 p.164, p.173).
fn vmax(t: Fmt, ieee: bool, a: u64, b: u64) -> u64 {
    if ieee && t.is_snan(a) {
        return t.quiet(a);
    }
    if ieee && t.is_snan(b) {
        return t.quiet(b);
    }
    if t.is_nan(a) {
        return b;
    }
    if t.is_nan(b) {
        return a;
    }
    let (x, y) = (t.val(a), t.val(b));
    if x == 0.0 && y == 0.0 && t.is_neg(a) != t.is_neg(b) {
        return if t.is_neg(a) { b } else { a };
    }
    if ieee {
        if x >= y {
            a
        } else {
            b
        }
    } else if x > y {
        a
    } else {
        b
    }
}

/// V_MED3_F32 (§12.11 p.333): NaN in any input → min3; equality tests are
/// floating-point comparisons.
fn med3f(ieee: bool, a: u32, b: u32, c: u32) -> u32 {
    let (a, b, c) = (u64::from(a), u64::from(b), u64::from(c));
    let t = Fmt::S;
    if t.is_nan(a) || t.is_nan(b) || t.is_nan(c) {
        return vmin(t, ieee, vmin(t, ieee, a, b), c) as u32;
    }
    let mx = t.val(vmax(t, ieee, vmax(t, ieee, a, b), c));
    (if mx == t.val(a) {
        vmax(t, ieee, b, c)
    } else if mx == t.val(b) {
        vmax(t, ieee, a, c)
    } else {
        vmax(t, ieee, a, b)
    }) as u32
}

fn arith32(ins: &[u32], op: impl FnOnce() -> f32) -> u32 {
    if let Some(&b) = ins.iter().find(|&&b| nan32(b)) {
        return b | 0x40_0000;
    }
    let r = op();
    if r.is_nan() {
        QNAN32
    } else {
        r.to_bits()
    }
}
fn arith64(ins: &[u64], op: impl FnOnce() -> f64) -> u64 {
    if let Some(&b) = ins.iter().find(|&&b| Fmt::D.is_nan(b)) {
        return Fmt::D.quiet(b);
    }
    let r = op();
    if r.is_nan() {
        QNAN64
    } else {
        r.to_bits()
    }
}
/// f16 arithmetic computed exactly enough in f32 and rounded once more to
/// f16: for +, -, * binary32 (p = 24 >= 2*11 + 2) makes the double rounding
/// innocuous, so the result is the correctly rounded f16 value.
fn arith16(ins: &[u32], op: impl FnOnce() -> f32) -> u32 {
    if let Some(&b) = ins.iter().find(|&&b| Fmt::H.is_nan(u64::from(b))) {
        return b | 0x200;
    }
    let r = op();
    if r.is_nan() {
        QNAN16
    } else {
        u32::from(float::f32_to_f16_rne(r))
    }
}

fn f32_to_f64(b: u32) -> u64 {
    if nan32(b) {
        return u64::from(b >> 31) << 63 | 0x7ff8_0000_0000_0000 | u64::from(b & 0x7f_ffff) << 29;
    }
    f64::from(f(b)).to_bits()
}
fn f64_to_f32(b: u64) -> u32 {
    if Fmt::D.is_nan(b) {
        return ((b >> 32) as u32 & 0x8000_0000) | 0x7fc0_0000 | ((b >> 29) as u32 & 0x3f_ffff);
    }
    (d(b) as f32).to_bits()
}

/// Integer-rounding ops: NaN → quieted input; otherwise `op` (all preserve
/// the sign of zero).
fn round32(b: u32, op: impl FnOnce(f32) -> f32) -> u32 {
    if nan32(b) {
        b | 0x40_0000
    } else {
        op(f(b)).to_bits()
    }
}
/// V_CEIL_F32 (§12.8 p.186): trunc, then +1 when positive and inexact.
fn ceil_isa(x: f32) -> f32 {
    let t = x.trunc();
    if x > 0.0 && x != t {
        t + 1.0
    } else {
        t
    }
}
/// V_FLOOR_F32 (§12.8 p.187): trunc, then -1 when negative and inexact.
fn floor_isa(x: f32) -> f32 {
    let t = x.trunc();
    if x < 0.0 && x != t {
        t - 1.0
    } else {
        t
    }
}
/// V_FRACT_F32 (§12.8 p.186): S0 + -floor(S0) in f32 RNE, clamped to
/// 0x3f7fffff.
fn fract32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    let x = f(b);
    let r = x + -floor_isa(x);
    if r.is_nan() {
        QNAN32
    } else if r > f(0x3f7f_ffff) {
        0x3f7f_ffff
    } else {
        r.to_bits()
    }
}

/// V_FREXP_EXP_I32_F32 / V_FREXP_MANT_F32 (§12.8 p.195): Inf/NaN → (0, S0).
/// Zero → (0, ±0) and denormals are normalized, matching C `frexp`. The
/// guide's `exponent()`/`mantissa()` are not defined for them.
/// `[UNVERIFIED: zero/denormal results, confirm on MI300X]`
fn frexp32(b: u32) -> (i32, u32) {
    let mag = b & 0x7fff_ffff;
    if mag >= 0x7f80_0000 || mag == 0 {
        return (0, b);
    }
    let (m, e) = if mag >> 23 == 0 {
        (mag, -149)
    } else {
        ((mag & 0x7f_ffff) | 0x80_0000, (mag >> 23) as i32 - 150)
    };
    let n = 32 - m.leading_zeros() as i32;
    let mant = (b & 0x8000_0000) | 126 << 23 | ((m << (24 - n)) & 0x7f_ffff);
    (e + n, mant)
}

/// IEEE class index (bit of the V_CMP_CLASS mask, §12.9 p.208).
fn class32(b: u32) -> u32 {
    let neg = b >> 31 != 0;
    let (e, m) = (b >> 23 & 0xff, b & 0x7f_ffff);
    match (e, m) {
        (255, 0) => {
            if neg {
                2
            } else {
                9
            }
        }
        (255, m) => {
            if m & 0x40_0000 != 0 {
                1
            } else {
                0
            }
        }
        (0, 0) => {
            if neg {
                5
            } else {
                6
            }
        }
        (0, _) => {
            if neg {
                4
            } else {
                7
            }
        }
        _ => {
            if neg {
                3
            } else {
                8
            }
        }
    }
}

/// Flush an f32 denormal to a signed zero.
fn flush32(b: u32) -> u32 {
    if b & 0x7f80_0000 == 0 {
        b & 0x8000_0000
    } else {
        b
    }
}

// Transcendentals (§12.8 p.187-190). The hardware result is a 1-ULP
// approximation the guide does not specify bit-exactly. Each is modelled as
// the exact function evaluated in f64 and rounded to f32 (RNE), with denormal
// inputs and outputs flushed to signed zero ("denormals are flushed") and the
// guide's special-value examples.
// `[UNVERIFIED: hardware approximation, confirm on MI300X]`

/// V_EXP_F32: 2^x. `[UNVERIFIED: hardware approximation, confirm on MI300X]`
fn exp32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    flush32((f64::from(f(flush32(b))).exp2() as f32).to_bits())
}
/// V_LOG_F32: log2(x); log(±0) = -Inf, log(x<0) = NaN.
/// `[UNVERIFIED: hardware approximation, confirm on MI300X]`
fn log32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    let x = f(flush32(b));
    if x == 0.0 {
        return 0xff80_0000;
    }
    if x < 0.0 {
        return QNAN32;
    }
    flush32((f64::from(x).log2() as f32).to_bits())
}
/// V_RCP_F32 / V_RCP_IFLAG_F32: 1/x.
/// `[UNVERIFIED: hardware approximation, confirm on MI300X]`
fn rcp32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    flush32(((1.0 / f64::from(f(flush32(b)))) as f32).to_bits())
}
/// V_RSQ_F32: 1/sqrt(x); rsq(-0) = -Inf, rsq(x<0) = NaN.
/// `[UNVERIFIED: hardware approximation, confirm on MI300X]`
fn rsq32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    let x = f(flush32(b));
    if x < 0.0 {
        return QNAN32;
    }
    flush32(((1.0 / f64::from(x).sqrt()) as f32).to_bits())
}
/// V_SQRT_F32: sqrt(-0) = -0, sqrt(x<0) = NaN.
/// `[UNVERIFIED: hardware approximation, confirm on MI300X]`
fn sqrt32(b: u32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    let x = f(flush32(b));
    if x < 0.0 {
        return QNAN32;
    }
    flush32((f64::from(x).sqrt() as f32).to_bits())
}

/// `x * 2^k` with a single RNE rounding (denormal results included): the
/// product is exact in f64 for the clamped exponent range.
fn ldexp32(b: u32, k: i32) -> u32 {
    if nan32(b) {
        return b | 0x40_0000;
    }
    let x = f(b);
    if x == 0.0 || x.is_infinite() {
        return b;
    }
    let k = k.clamp(-400, 400);
    let scale = f64::from_bits(((1023 + k) as u64) << 52);
    ((f64::from(x) * scale) as f32).to_bits()
}

/// Finite nonzero f32 as (negative, m, e) with value m * 2^e.
fn unpack32(b: u32) -> (bool, u128, i32) {
    let (e, m) = (b >> 23 & 0xff, b & 0x7f_ffff);
    if e == 0 {
        (b >> 31 != 0, u128::from(m), -149)
    } else {
        (b >> 31 != 0, u128::from(m | 0x80_0000), e as i32 - 150)
    }
}

/// Round (-1)^neg * (mag + sticky·ε) * 2^exp to f32, RNE, with gradual
/// underflow and overflow to infinity. `sticky` means the exact value lies
/// strictly above `mag` by less than one unit of 2^exp. Requires mag < 2^127.
fn round_f32(neg: bool, mag: u128, exp: i32, sticky: bool) -> u32 {
    let sbit = u32::from(neg) << 31;
    if mag == 0 {
        return sbit;
    }
    let n = 128 - mag.leading_zeros() as i32;
    let lead = exp + n - 1;
    let mut lsb = (lead - 23).max(-149);
    let shift = lsb - exp;
    let mut m = if shift <= 0 {
        mag << (-shift) as u32
    } else if shift >= 128 {
        0
    } else {
        let sh = shift as u32;
        let (q, rem, half) = (mag >> sh, mag & ((1u128 << sh) - 1), 1u128 << (sh - 1));
        let up = rem > half || (rem == half && (sticky || q & 1 == 1));
        q + u128::from(up)
    };
    if m >> 24 != 0 {
        m >>= 1;
        lsb += 1;
    }
    if m < 1 << 23 {
        return sbit | m as u32;
    }
    let biased = lsb + 23 + 127;
    if biased >= 255 {
        return sbit | 0x7f80_0000;
    }
    sbit | (biased as u32) << 23 | (m as u32 & 0x7f_ffff)
}

/// fma(a, b, c) * 2^k with one rounding, for finite nonzero a, b and finite
/// c (the caller handles NaN/Inf/zero operands).
fn fma_exact32(a: u32, b: u32, c: u32, k: i32) -> u32 {
    let (na, ma, ea) = unpack32(a);
    let (nb, mb, eb) = unpack32(b);
    let (np, mp, ep) = (na != nb, ma * mb, ea + eb);
    if c & 0x7fff_ffff == 0 {
        return round_f32(np, mp, ep + k, false);
    }
    let (nc, mc, ec) = unpack32(c);
    let bits = |m: u128| 128 - m.leading_zeros() as i32;
    let top = (ep + bits(mp)).max(ec + bits(mc));
    let base = top - 120;
    let align = |m: u128, e: i32| -> (u128, bool) {
        let s = e - base;
        if s >= 0 {
            (m << s as u32, false)
        } else if -s >= 128 {
            (0, m != 0)
        } else {
            (m >> (-s) as u32, m & ((1u128 << (-s) as u32) - 1) != 0)
        }
    };
    let ((tp, sp), (tc, sc)) = (align(mp, ep), align(mc, ec));
    if np == nc {
        return round_f32(np, tp + tc, base + k, sp || sc);
    }
    let p_bigger = tp > tc || (tp == tc && sp && !sc);
    let (big, small, s_small, neg) = if p_bigger {
        (tp, tc, sc, np)
    } else {
        (tc, tp, sp, nc)
    };
    let diff = big - small - u128::from(s_small);
    if diff == 0 && !s_small {
        return 0;
    }
    round_f32(neg, diff, base + k, s_small)
}

/// V_DIV_SCALE_F32 (§12.11 p.336-337), returning (D0, VCC bit). D0 defaults
/// to S0 when no branch assigns it. Exponents are the biased fields.
/// Interpretations where the literal pseudocode cannot drive the division
/// macro it exists for, each `[UNVERIFIED: confirm on MI300X]`:
/// - `1.0 / 64'F(S1.f32) == DENORM.f64` is read as "1/S1 lies in the f32
///   denormal range". Taken literally it can never hold, because no f32
///   reciprocal is an f64 denormal, and denominators above 2^126 would then
///   go unscaled into V_RCP_F32, which flushes the denormal result.
/// - In the branch where both 1/S1 and S2/S1 are denormal, the denominator
///   is scaled by 2^-64 instead of the printed 2^+64, which overflows a
///   denominator above 2^126. 2^-64 matches hardware measurements of the
///   same macro on RDNA2.
fn div_scale32(s0: u32, s1: u32, s2: u32) -> (u32, bool) {
    let (f0, f1, f2) = (f(s0), f(s1), f(s2));
    let (e1, e2) = ((s1 >> 23 & 0xff) as i32, (s2 >> 23 & 0xff) as i32);
    let denorm = |b: u32| b & 0x7f80_0000 == 0 && b & 0x7f_ffff != 0;
    let inv = (1.0 / f64::from(f1)).abs();
    let inv_denorm = inv != 0.0 && inv < f64::from(f32::MIN_POSITIVE);
    let q_denorm = denorm((f2 / f1).to_bits());
    if f2 == 0.0 || f1 == 0.0 {
        (QNAN32, false)
    } else if e2 - e1 >= 96 {
        (if f0 == f1 { ldexp32(s0, 64) } else { s0 }, true)
    } else if denorm(s1) {
        (ldexp32(s0, 64), false)
    } else if inv_denorm && q_denorm {
        (if f0 == f1 { ldexp32(s0, -64) } else { s0 }, true)
    } else if inv_denorm {
        (ldexp32(s0, -64), false)
    } else if q_denorm {
        (if f0 == f2 { ldexp32(s0, 64) } else { s0 }, true)
    } else if e2 <= 23 {
        (ldexp32(s0, 64), false)
    } else {
        (s0, false)
    }
}

/// V_DIV_FMAS_F32 (§12.11 p.338): fma(S0, S1, S2), post-scaled when the lane's
/// VCC bit is set. The guide prints a fixed 2^32 factor, which cannot undo
/// V_DIV_SCALE_F32's 2^±64 scaling. This model uses 2^+64 when
/// exponent(S2) >= 128 (the quotient was computed against a denominator
/// scaled up) and 2^-64 otherwise (numerator scaled up), with a single
/// rounding of the scaled exact fma. That matches hardware measurements of
/// the same macro on RDNA2. `[UNVERIFIED: confirm on MI300X]`
fn div_fmas32(a: u32, b: u32, c: u32, scale: bool) -> u32 {
    let k = if !scale {
        0
    } else if c >> 23 & 0xff >= 128 {
        64
    } else {
        -64
    };
    if let Some(&n) = [a, b, c].iter().find(|&&x| nan32(x)) {
        return n | 0x40_0000;
    }
    let special = |x: u32| x & 0x7fff_ffff == 0 || x & 0x7fff_ffff == 0x7f80_0000;
    if special(a) || special(b) || c & 0x7fff_ffff == 0x7f80_0000 {
        // Zero or infinite operands: the product is exact (0 or ±Inf) and the
        // sum is c, ±Inf, NaN or a signed zero.
        let r = f(a).mul_add(f(b), f(c));
        if r.is_nan() {
            return QNAN32;
        }
        if r.is_infinite() || r == 0.0 && c & 0x7fff_ffff == 0 {
            return r.to_bits();
        }
        return if f(a) * f(b) == 0.0 {
            ldexp_round(c, k)
        } else {
            r.to_bits()
        };
    }
    fma_exact32(a, b, c, k)
}

/// c * 2^k with one rounding (finite c).
fn ldexp_round(c: u32, k: i32) -> u32 {
    if c & 0x7fff_ffff == 0 {
        return c;
    }
    let (n, m, e) = unpack32(c);
    round_f32(n, m, e + k, false)
}

/// V_DIV_FIXUP_F32 (§12.11 p.335). UNDERFLOW_F32 is not defined in the
/// guide. It is modelled as +0, the RNE value of every quotient that branch
/// covers (|S2/S1| < 2^-150). The printed overflow test `exponent(S1.f32) ==
/// 255` can never be reached, because Inf and NaN S1 are handled earlier.
/// It is read as `exponent(S0.f32) == 255`: a quotient that overflowed in
/// the macro (Inf, or a NaN from Inf - Inf in the Newton-Raphson steps) gives
/// ±OVERFLOW_F32, modelled as ±Inf, the RNE value of every finite quotient
/// that overflows f32 by more than the 2^64 V_DIV_SCALE headroom. This
/// matches RDNA2 measurements of the macro, where a NaN quotient with finite
/// inputs gives ±Inf. `[UNVERIFIED: confirm on MI300X]`
fn div_fixup32(s0: u32, s1: u32, s2: u32) -> u32 {
    let sign = (s1 ^ s2) & 0x8000_0000;
    let (f1, f2) = (f(s1), f(s2));
    let (e0, e1, e2) = (
        (s0 >> 23 & 0xff) as i32,
        (s1 >> 23 & 0xff) as i32,
        (s2 >> 23 & 0xff) as i32,
    );
    if nan32(s2) {
        s2 | 0x40_0000
    } else if nan32(s1) {
        s1 | 0x40_0000
    } else if (f1 == 0.0 && f2 == 0.0) || (f1.is_infinite() && f2.is_infinite()) {
        QNAN32
    } else if f1 == 0.0 || f2.is_infinite() {
        sign | 0x7f80_0000
    } else if f1.is_infinite() || f2 == 0.0 {
        sign
    } else if e2 - e1 < -150 {
        sign
    } else if e0 == 255 {
        sign | 0x7f80_0000
    } else {
        sign | (s0 & 0x7fff_ffff)
    }
}
