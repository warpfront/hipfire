// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{
    arch::Arch,
    insn::Instruction,
    reg::{DistinctBanks, OppositeParity, RegRef, Vb, Vp, S, V},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VopdF32 {
    Add,
    Sub,
    Subrev,
    Mul,
    Fmac,
}
impl VopdF32 {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "v_dual_add_f32",
            Self::Sub => "v_dual_sub_f32",
            Self::Subrev => "v_dual_subrev_f32",
            Self::Mul => "v_dual_mul_f32",
            Self::Fmac => "v_dual_fmac_f32",
        }
    }
}

/// Arch/op-wide answer to "may the two multiplicand/addend factors of this VOPD F32 half be swapped without changing
/// the NaN payload/sign the hardware returns?". Always `false`.
///
/// On gfx1201 (AMD Radeon AI PRO R9700, fabric j8207, receipt `nan-swap-receipt-final`, build
/// `nan-swap-build-final/build.json`; evidence in `peacemaker-ir/isa/emu/float_edges.md`) native VOP2
/// and both VOPD halves, in all 16 FP32 denorm x round `MODE` settings, return the *first* NaN source (src0, then
/// src1) quieted with sign and payload preserved. Swapping `Add`/`Mul`/`Fmac` factors that are two NaNs with
/// different sign or quieted payload therefore changes the result (2,421,565,968 unequal swaps over the 13,089,298,944
/// measured op-cases), so no arch/op-wide swap is sound. `Sub`/`Subrev` are not commutative at all, and no other arch
/// has been measured. Callers that know the concrete input bit patterns use [`swap_payload_safe_for_inputs`].
pub const fn swap_payload_safe(_arch: Arch, _op: VopdF32) -> bool {
    false
}

/// Input-conditional swap legality for the two factors `a`, `b` (raw f32 bits, src0 and src1 of the half).
///
/// Returns `true` only for gfx1201 `Add`/`Mul`/`Fmac` and only when
/// `peacemaker_ir::emu::f32_nan_swap_payload_safe` holds: not both inputs are NaN, or both are NaN and
/// `(a | 0x0040_0000) == (b | 0x0040_0000)` (quieted bit pattern *including sign* identical, i.e. both quieted
/// payload bits and sign match). Everything else (`Sub`, `Subrev`, other arches) is `false`.
///
/// Scope of the measured claim: the factors are unmodified (no abs/neg/clamp/omod source or result modifier applied
/// to them), for `Fmac` only the two multiplicand factors are swapped (never the accumulator/destination `c`), and
/// the predicate makes no statement about source modifiers or FP exception traps. The receipt covers 16,777,214 NaN
/// encodings x 16 partners, a 64,000-triple grid and 4,194,304 random triples per op and mode, not the full
/// NaN-pair Cartesian product.
pub fn swap_payload_safe_for_inputs(arch: Arch, op: VopdF32, a: u32, b: u32) -> bool {
    match (arch, op) {
        (Arch::Gfx1201, VopdF32::Add | VopdF32::Mul | VopdF32::Fmac) => {
            peacemaker_ir::emu::f32_nan_swap_payload_safe(peacemaker_ir::Arch::Gfx1201, a, b)
        }
        _ => false,
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src0<const B: u8> {
    V(Vb<B>),
    S(S<1>),
    Lit(u32),
    Inline(i8),
}
#[derive(Clone, Copy, Debug)]
pub struct VopdF32Op<D, A, B> {
    pub op: VopdF32,
    pub dst: D,
    pub src0: A,
    pub src1: B,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    V(u8),
    S(u8),
    Lit(u32),
    Inline(i8),
}
impl Operand {
    fn text(self) -> String {
        match self {
            Self::V(n) => format!("v{n}"),
            Self::S(n) => format!("s{n}"),
            Self::Lit(n) => format!("0x{n:08x}"),
            Self::Inline(n) => n.to_string(),
        }
    }
    fn reg(self) -> Option<RegRef> {
        match self {
            Self::V(n) => Some(V::<1>(n).reg()),
            Self::S(n) => Some(S::<1>(n).reg()),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct VopdOp {
    pub op: VopdF32,
    pub dst: u8,
    pub src0: Operand,
    pub src1: u8,
}
impl VopdOp {
    fn text(self) -> String {
        format!(
            "{} v{}, {}, v{}",
            self.op.name(),
            self.dst,
            self.src0.text(),
            self.src1
        )
    }
}
pub fn validate_pair(arch: Arch, x: VopdOp, y: VopdOp) -> Result<(), String> {
    if x.dst % 2 == y.dst % 2 {
        return Err("VOPD destination parity collision".into());
    }
    // gfx12 lets both halves read the same VGPR as src0 (hipcc emits `v_dual_mul_f32 .., v189, .. :: v_dual_mul_f32 ..,
    // v189, ..`), as it does for src1.
    if let (Operand::V(a), Operand::V(b)) = (x.src0, y.src0) {
        if a % 4 == b % 4 && !(arch.gfx12() && a == b) {
            return Err("VOPD src0 bank collision".into());
        }
    }
    if x.src1 % 4 == y.src1 % 4 && !(arch.gfx12() && x.src1 == y.src1) {
        return Err("VOPD src1 bank collision".into());
    }
    let literals = [x.src0, y.src0]
        .iter()
        .filter_map(|s| {
            if let Operand::Lit(n) = s {
                Some(*n)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if literals.len() > 1 && literals[0] != literals[1] {
        return Err("VOPD supports only one shared literal".into());
    }
    // gfx11 (LLVM `checkVOPDRegConstraints` pairing): neither half may read
    // the other half's destination.
    if !arch.gfx12() {
        let reads = |op: VopdOp, reg: u8| op.src0 == Operand::V(reg) || op.src1 == reg;
        if reads(y, x.dst) || reads(x, y.dst) {
            return Err("gfx11 VOPD half reads the other half's destination".into());
        }
    }
    Ok(())
}
pub fn packet(arch: Arch, x: VopdOp, y: VopdOp) -> Result<Instruction, String> {
    validate_pair(arch, x, y)?;
    let mut uses = vec![V::<1>(x.src1).reg(), V::<1>(y.src1).reg()];
    for op in [x.src0, y.src0] {
        if let Some(r) = op.reg() {
            uses.push(r)
        }
    }
    if x.op == VopdF32::Fmac {
        uses.push(V::<1>(x.dst).reg())
    }
    if y.op == VopdF32::Fmac {
        uses.push(V::<1>(y.dst).reg())
    }
    Ok(Instruction::new(
        format!("{} :: {}", x.text(), y.text()),
        vec![V::<1>(x.dst).reg(), V::<1>(y.dst).reg()],
        uses,
    ))
}
pub fn typed<const DX: u8, const DY: u8, const AX: u8, const AY: u8, const BX: u8, const BY: u8>(
    arch: Arch,
    x: VopdF32Op<Vp<DX>, Src0<AX>, Vb<BX>>,
    y: VopdF32Op<Vp<DY>, Src0<AY>, Vb<BY>>,
) -> Result<Instruction, String>
where
    Vp<DX>: OppositeParity<DY>,
    Vb<AX>: DistinctBanks<AY>,
    Vb<BX>: DistinctBanks<BY>,
{
    fn convert<const B: u8>(s: Src0<B>) -> Operand {
        match s {
            Src0::V(v) => Operand::V(v.0),
            Src0::S(s) => Operand::S(s.0),
            Src0::Lit(n) => Operand::Lit(n),
            Src0::Inline(n) => Operand::Inline(n),
        }
    }
    if x.dst.0 % 2 != DX
        || y.dst.0 % 2 != DY
        || matches!(x.src0,Src0::V(v) if v.0%4!=AX)
        || matches!(y.src0,Src0::V(v) if v.0%4!=AY)
        || x.src1.0 % 4 != BX
        || y.src1.0 % 4 != BY
    {
        return Err("typed VOPD register carries wrong bank/parity".into());
    }
    packet(
        arch,
        VopdOp {
            op: x.op,
            dst: x.dst.0,
            src0: convert(x.src0),
            src1: x.src1.0,
        },
        VopdOp {
            op: y.op,
            dst: y.dst.0,
            src0: convert(y.src0),
            src1: y.src1.0,
        },
    )
}
pub fn shared_src1<const DX: u8, const DY: u8, const AX: u8, const AY: u8>(
    arch: Arch,
    x: VopdF32Op<Vp<DX>, Src0<AX>, V<1>>,
    y: VopdF32Op<Vp<DY>, Src0<AY>, V<1>>,
) -> Result<Instruction, String>
where
    Vp<DX>: OppositeParity<DY>,
    Vb<AX>: DistinctBanks<AY>,
{
    if !arch.gfx12() || x.src1 != y.src1 || x.dst.0 % 2 != DX || y.dst.0 % 2 != DY {
        return Err("shared-src1 packet requires gfx12, identical src1 and correct parity".into());
    }
    fn cv<const B: u8>(s: Src0<B>) -> Operand {
        match s {
            Src0::V(Vb(n)) => Operand::V(n),
            Src0::S(S(n)) => Operand::S(n),
            Src0::Lit(n) => Operand::Lit(n),
            Src0::Inline(n) => Operand::Inline(n),
        }
    }
    packet(
        arch,
        VopdOp {
            op: x.op,
            dst: x.dst.0,
            src0: cv(x.src0),
            src1: x.src1.0,
        },
        VopdOp {
            op: y.op,
            dst: y.dst.0,
            src0: cv(y.src0),
            src1: y.src1.0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul(dst: u8, src0: u8, src1: u8) -> VopdOp {
        VopdOp {
            op: VopdF32::Mul,
            dst,
            src0: Operand::V(src0),
            src1,
        }
    }

    /// hipcc's `attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` emits
    /// `v_dual_mul_f32 v230, v189, v222 :: v_dual_mul_f32 v231, v189, v223`: both halves read the same VGPR as src0.
    #[test]
    fn gfx12_allows_the_same_vgpr_as_src0_in_both_halves() {
        let (x, y) = (mul(230, 189, 222), mul(231, 189, 223));
        assert!(validate_pair(Arch::Gfx1201, x, y).is_ok());
        let text = packet(Arch::Gfx1201, x, y).unwrap().text;
        assert_eq!(
            text,
            "v_dual_mul_f32 v230, v189, v222 :: v_dual_mul_f32 v231, v189, v223"
        );
    }

    /// Different registers in one bank stay refused (src0 and src1), on every arch; gfx11 stays strict for equal src0.
    #[test]
    fn distinct_registers_in_one_bank_stay_refused() {
        for arch in [Arch::Gfx1201, Arch::Gfx1151] {
            assert_eq!(
                validate_pair(arch, mul(0, 2, 4), mul(1, 6, 5)).unwrap_err(),
                "VOPD src0 bank collision"
            );
            assert_eq!(
                validate_pair(arch, mul(0, 2, 4), mul(1, 3, 0)).unwrap_err(),
                "VOPD src1 bank collision"
            );
        }
        assert_eq!(
            validate_pair(Arch::Gfx1151, mul(0, 2, 4), mul(1, 2, 5)).unwrap_err(),
            "VOPD src0 bank collision"
        );
    }
}

#[cfg(test)]
mod swap_payload_tests {
    use super::*;

    const ALL_OPS: [VopdF32; 5] = [
        VopdF32::Add,
        VopdF32::Sub,
        VopdF32::Subrev,
        VopdF32::Mul,
        VopdF32::Fmac,
    ];
    const ALL_ARCHES: [Arch; 4] = [Arch::Gfx1201, Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx942];
    const SWAPPABLE: [VopdF32; 3] = [VopdF32::Add, VopdF32::Mul, VopdF32::Fmac];

    #[test]
    fn coarse_predicate_is_always_false() {
        for arch in ALL_ARCHES {
            for op in ALL_OPS {
                assert!(!swap_payload_safe(arch, op), "{arch:?} {op:?}");
            }
        }
    }

    #[test]
    fn gfx1201_safe_inputs_are_true() {
        let cases = [
            // no NaN
            (0x3f80_0000, 0x4000_0000),
            (0x7f80_0000, 0xff80_0000),
            // exactly one NaN, either side, either sign
            (0x7fc0_0000, 0x3f80_0000),
            (0x3f80_0000, 0xffa0_beef),
            (0x7f80_0001, 0x0000_0000),
            // both NaN, identical bits
            (0x7fc0_beef, 0x7fc0_beef),
            (0xffc0_0000, 0xffc0_0000),
            // both NaN, differ only in the quiet bit (same sign, same payload)
            (0x7f80_0001, 0x7fc0_0001),
            (0x7fc0_0001, 0x7f80_0001),
            (0xff80_0b61, 0xffc0_0b61),
        ];
        for op in SWAPPABLE {
            for (a, b) in cases {
                assert!(
                    swap_payload_safe_for_inputs(Arch::Gfx1201, op, a, b),
                    "{op:?} {a:#010x} {b:#010x}"
                );
                assert!(
                    swap_payload_safe_for_inputs(Arch::Gfx1201, op, b, a),
                    "{op:?} {b:#010x} {a:#010x}"
                );
            }
        }
    }

    #[test]
    fn gfx1201_distinct_payload_or_sign_is_false() {
        let cases = [
            // measured hardware witnesses (j8200 nan-swap receipt samples): sign differs, payload equal
            (0x7f80_0b61, 0xff80_0b61),
            (0x7f80_1601, 0xff80_1601),
            (0x7f80_0001, 0xff80_0001),
            // quieted payload differs
            (0x7fc0_0001, 0x7fc0_0002),
            (0x7fdc_cbec, 0x7fda_2209),
            (0x7fc0_beef, 0x7fc0_0000),
            // sign differs on quiet NaNs
            (0x7fc0_0000, 0xffc0_0000),
            // sign and payload differ
            (0xfffa_60dd, 0x7fc0_0000),
        ];
        for op in SWAPPABLE {
            for (a, b) in cases {
                assert!(
                    !swap_payload_safe_for_inputs(Arch::Gfx1201, op, a, b),
                    "{op:?} {a:#010x} {b:#010x}"
                );
                assert!(
                    !swap_payload_safe_for_inputs(Arch::Gfx1201, op, b, a),
                    "{op:?} {b:#010x} {a:#010x}"
                );
            }
        }
    }

    #[test]
    fn subtraction_and_other_arches_are_false() {
        // inputs that are safe on gfx1201 add/mul/fmac
        let (a, b) = (0x3f80_0000, 0x4000_0000);
        for arch in ALL_ARCHES {
            for op in ALL_OPS {
                let expect = arch == Arch::Gfx1201 && SWAPPABLE.contains(&op);
                assert_eq!(
                    swap_payload_safe_for_inputs(arch, op, a, b),
                    expect,
                    "{arch:?} {op:?}"
                );
            }
        }
        for op in [VopdF32::Sub, VopdF32::Subrev] {
            assert!(!swap_payload_safe_for_inputs(Arch::Gfx1201, op, a, b));
        }
    }
}
