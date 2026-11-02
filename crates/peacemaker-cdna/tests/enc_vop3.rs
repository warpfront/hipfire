// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP3A / VOP3B encoder, decoder, printer and footprint against ROCm `llvm-mc`.
//!
//! Oracle tests (`llvm_*`) batch every instruction of a class into one
//! `llvm-mc` assemble call and one disassemble call and compare bytes, printed
//! text and `Vop3::decode`; they return early only when
//! `oracle::disabled()` (`HIPFIRE_TEST_REQUIRE_ROCM=0`). Everything else is
//! CPU-only and always runs.
//!
//! Coverage: every opcode of `Vop3Op::all(Gfx942)` gets at least two operand
//! patterns (`patterns`: all-VGPR, SGPR / inline constant / special register
//! mixes, 64-bit pairs, VOP3B scalar destinations, carry / lane-mask operands,
//! lane selects), and every opcode with modifier support gets a sweep
//! (`sweeps`: each NEG / ABS bit, all bits, CLAMP, each OMOD, each OP_SEL
//! bit, combinations, negated / absolute inline constants).
use peacemaker_cdna::arch::Arch;
use peacemaker_cdna::enc::vop::{Vop1Op, Vop2Op, VopcOp};
use peacemaker_cdna::enc::vop3::{Vop3, Vop3OnlyOp, Vop3Op};
use peacemaker_cdna::inst::{decode_all, encode_all, Inst, RegFile, RegSpan};
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::oracle;
use std::collections::HashSet;

const ARCH: Arch = Arch::Gfx942;

fn v(n: u8) -> Src {
    Src::Vgpr(n)
}
fn sg(n: u8) -> Src {
    Src::Sgpr(n)
}
fn ttmp(n: u8) -> Src {
    Src::Ttmp(n)
}
fn int(n: i8) -> Src {
    Src::Int(n)
}
fn fl(n: usize) -> Src {
    Src::Float(InlineFloat::ALL[n])
}
fn some(w: u8, s: Src) -> Option<Src> {
    (w > 0).then_some(s)
}

const NOP: Vop3Op = Vop3Op::V1(Vop1Op::Nop);
const READLANE: Vop3Op = Vop3Op::Only(Vop3OnlyOp::ReadlaneB32);
const WRITELANE: Vop3Op = Vop3Op::Only(Vop3OnlyOp::WritelaneB32);
const LSHL_ADD_U64: Vop3Op = Vop3Op::Only(Vop3OnlyOp::LshlAddU64);
const DIV_FMAS: Vop3Op = Vop3Op::Only(Vop3OnlyOp::DivFmasF32);
const ADD_F32: Vop3Op = Vop3Op::V2(Vop2Op::AddF32);
const ADD_U32: Vop3Op = Vop3Op::V2(Vop2Op::AddU32);
const ADD_F64: Vop3Op = Vop3Op::Only(Vop3OnlyOp::AddF64);
const MOV_B32: Vop3Op = Vop3Op::V1(Vop1Op::MovB32);

fn new(op: Vop3Op, dst: Src, sdst: Option<Src>, src: [Option<Src>; 3]) -> Vop3 {
    Vop3 {
        op,
        dst,
        sdst,
        src,
        abs: 0,
        neg: 0,
        clamp: false,
        omod: 0,
        op_sel: 0,
    }
}

/// `dst, src0, src1` of a plain VOP3A instruction.
fn bin(op: Vop3Op, dst: Src, a: Src, b: Src) -> Vop3 {
    new(op, dst, None, [Some(a), Some(b), None])
}

fn is_mask_op(op: Vop3Op) -> bool {
    matches!(
        op,
        Vop3Op::V2(
            Vop2Op::CndmaskB32 | Vop2Op::AddcCoU32 | Vop2Op::SubbCoU32 | Vop2Op::SubbrevCoU32
        )
    )
}

fn is_fp8_cvt(op: Vop3Op) -> bool {
    matches!(
        op,
        Vop3Op::Only(
            Vop3OnlyOp::CvtPkFp8F32
                | Vop3OnlyOp::CvtPkBf8F32
                | Vop3OnlyOp::CvtSrFp8F32
                | Vop3OnlyOp::CvtSrBf8F32
        )
    )
}

/// Operand patterns for one opcode: at least two, all legal on gfx942.
fn patterns(op: Vop3Op) -> Vec<Vop3> {
    let (_, w0, w1, w2) = op.widths();
    if op == NOP {
        return vec![new(op, v(0), None, [None; 3])];
    }
    let dsts: Vec<Src> = match op {
        Vop3Op::C(_) => vec![sg(10), Src::VccLo, Src::ExecLo, ttmp(4)],
        READLANE => vec![sg(5), Src::VccLo, ttmp(3), Src::ExecHi, sg(101)],
        WRITELANE => vec![v(0), v(1), v(255), v(100), v(7), v(9)],
        _ => vec![v(8), v(40), v(250), v(254)],
    };
    let srcsets: Vec<[Option<Src>; 3]> = if op == READLANE {
        vec![
            [Some(v(7)), Some(sg(3)), None],
            [Some(v(255)), Some(int(5)), None],
            [Some(v(0)), Some(Src::M0), None],
            [Some(v(128)), Some(fl(0)), None],
            [Some(v(1)), Some(Src::Scc), None],
        ]
    } else if op == WRITELANE {
        vec![
            [Some(sg(2)), Some(sg(2)), None],
            [Some(int(5)), Some(Src::M0), None],
            [Some(ttmp(5)), Some(int(63)), None],
            [Some(Src::VccLo), Some(Src::M0), None],
            [Some(fl(6)), Some(int(0)), None],
            [Some(Src::Execz), Some(int(-1)), None],
        ]
    } else if is_mask_op(op) {
        vec![
            [Some(v(16)), Some(v(24)), Some(sg(4))],
            [Some(int(5)), Some(v(24)), Some(Src::VccLo)],
            [Some(fl(4)), Some(v(250)), Some(ttmp(2))],
            [Some(v(16)), Some(int(-16)), Some(Src::ExecLo)],
            [Some(v(16)), Some(v(24)), Some(Src::Scc)],
            [Some(v(30)), Some(v(31)), Some(Src::Vccz)],
        ]
    } else if is_fp8_cvt(op) {
        vec![
            [Some(v(16)), Some(v(24)), None],
            [Some(v(255)), Some(v(0)), None],
            [Some(v(1)), Some(v(2)), None],
            [Some(v(100)), Some(v(101)), None],
        ]
    } else if op == LSHL_ADD_U64 {
        vec![
            [Some(v(16)), Some(v(24)), Some(v(32))],
            [Some(sg(6)), Some(int(3)), Some(v(32))],
            [Some(fl(8)), Some(int(0)), Some(Src::ExecLo)],
            [Some(v(16)), Some(int(4)), Some(int(64))],
        ]
    } else if op == DIV_FMAS {
        // The implicit VCC read already uses the constant bus.
        vec![
            [Some(v(16)), Some(v(24)), Some(v(32))],
            [Some(int(5)), Some(int(-16)), Some(v(32))],
            [Some(fl(8)), Some(v(250)), Some(fl(2))],
            [Some(v(16)), Some(fl(5)), Some(int(64))],
        ]
    } else {
        vec![
            [some(w0, v(16)), some(w1, v(24)), some(w2, v(32))],
            [some(w0, sg(6)), some(w1, int(5)), some(w2, v(32))],
            [some(w0, fl(8)), some(w1, v(250)), some(w2, Src::ExecLo)],
            [some(w0, v(16)), some(w1, Src::VccLo), some(w2, int(64))],
            [some(w0, int(-16)), some(w1, fl(7)), some(w2, ttmp(2))],
        ]
    };
    srcsets
        .iter()
        .enumerate()
        .map(|(n, src)| {
            let sdst = op
                .is_vop3b()
                .then(|| [sg(12), Src::VccLo, ttmp(2), sg(100)][n % 4]);
            new(op, dsts[n % dsts.len()], sdst, *src)
        })
        .collect()
}

/// Modifier sweeps over the first pattern of every opcode that has any.
fn sweeps(op: Vop3Op) -> Vec<Vop3> {
    let m = op.mods();
    if m.neg == 0 && m.abs == 0 && !m.clamp && !m.omod && m.op_sel == 0 {
        return Vec::new();
    }
    let base = patterns(op).swap_remove(0);
    let mk = |neg: u8, abs: u8, clamp: bool, omod: u8, op_sel: u8| Vop3 {
        neg,
        abs,
        clamp,
        omod,
        op_sel,
        ..base.clone()
    };
    let mut out = Vec::new();
    for b in 0..3u8 {
        if (m.neg >> b) & 1 != 0 {
            out.push(mk(1 << b, 0, false, 0, 0))
        }
        if (m.abs >> b) & 1 != 0 {
            out.push(mk(0, 1 << b, false, 0, 0))
        }
    }
    if m.neg != 0 {
        out.push(mk(m.neg, 0, false, 0, 0))
    }
    if m.abs != 0 {
        out.push(mk(0, m.abs, false, 0, 0))
    }
    if m.neg != 0 && m.abs != 0 {
        out.push(mk(m.neg, m.abs, false, 0, 0))
    }
    if m.clamp {
        out.push(mk(0, 0, true, 0, 0))
    }
    if m.omod {
        for omod in 1..=3u8 {
            out.push(mk(0, 0, false, omod, 0))
        }
    }
    if m.clamp && m.omod {
        out.push(mk(m.neg, m.abs, true, 3, 0))
    }
    if m.op_sel != 0 {
        for b in 0..4u8 {
            if (m.op_sel >> b) & 1 != 0 {
                out.push(mk(0, 0, false, 0, 1 << b))
            }
        }
        out.push(mk(0, 0, false, 0, m.op_sel));
        if m.clamp {
            out.push(mk(m.neg, m.abs, true, 0, m.op_sel))
        }
    }
    if ((m.neg | m.abs) & 1) != 0 && !is_fp8_cvt(op) {
        for s0 in [fl(2), int(1)] {
            let mut src = base.src;
            src[0] = Some(s0);
            let with = |neg: u8, abs: u8| Vop3 {
                src,
                neg,
                abs,
                ..base.clone()
            };
            if (m.neg & 1) != 0 {
                out.push(with(1, 0))
            }
            if (m.abs & 1) != 0 {
                out.push(with(0, 1))
            }
            if (m.neg & 1) != 0 && (m.abs & 1) != 0 {
                out.push(with(1, 1))
            }
        }
    }
    out
}

fn all_cases() -> Vec<Vop3> {
    Vop3Op::all(ARCH)
        .into_iter()
        .flat_map(|op| {
            let mut cases = patterns(op);
            cases.extend(sweeps(op));
            cases
        })
        .collect()
}

fn words(i: &Vop3) -> Vec<u32> {
    let mut w = Vec::new();
    i.encode(ARCH, &mut w)
        .unwrap_or_else(|e| panic!("`{i}` must encode: {e}"));
    w
}

/// The assembler spelling of `i`: llvm prints `v_nop` for both encodings, so
/// the 64-bit form is spelled `v_nop_e64` on input.
fn asm_line(i: &Vop3) -> String {
    if i.op == NOP {
        "v_nop_e64".to_string()
    } else {
        i.to_string()
    }
}

// ---------------------------------------------------------------------------
// Opcode tables
// ---------------------------------------------------------------------------

#[test]
fn opcode_table_is_complete_unique_and_round_trips() {
    let ops = Vop3Op::all(ARCH);
    let count = |f: fn(&Vop3Op) -> bool| ops.iter().filter(|op| f(op)).count();
    assert_eq!(count(|op| matches!(op, Vop3Op::C(_))), VopcOp::ALL.len());
    assert_eq!(count(|op| matches!(op, Vop3Op::V2(_))), Vop2Op::ALL.len());
    // v_readfirstlane_b32, v_accvgpr_mov_b32 and v_swap_b32 have no 64-bit form in llvm.
    assert_eq!(
        count(|op| matches!(op, Vop3Op::V1(_))),
        Vop1Op::ALL.len() - 3
    );
    assert_eq!(
        count(|op| matches!(op, Vop3Op::Only(_))),
        Vop3OnlyOp::ALL.len()
    );
    assert_eq!(ops.len(), 32 + 41 + 39 + 55);

    let mut opcodes = HashSet::new();
    let mut names = HashSet::new();
    for &op in &ops {
        let opcode = op
            .opcode(ARCH)
            .unwrap_or_else(|| panic!("{op:?} has no opcode"));
        assert!(
            opcode < 0x380,
            "{op:?}: {opcode:#x} collides with the VOP3P prefix"
        );
        assert!(
            opcodes.insert(opcode),
            "{op:?}: duplicate opcode {opcode:#x}"
        );
        assert!(
            names.insert(op.name()),
            "{op:?}: duplicate name {}",
            op.name()
        );
        assert!(op.name().starts_with("v_"), "{op:?}");
        assert_eq!(Vop3Op::from_opcode(ARCH, opcode), Some(op), "{op:?}");
    }
    // Every other opcode number decodes to nothing.
    for opcode in 0..0x400u16 {
        if !opcodes.contains(&opcode) {
            assert_eq!(Vop3Op::from_opcode(ARCH, opcode), None, "{opcode:#x}");
        }
    }
}

#[test]
fn promoted_opcodes_sit_at_the_gfx9_offsets() {
    for &op in VopcOp::ALL {
        assert_eq!(
            Vop3Op::C(op).opcode(ARCH),
            op.opcode(ARCH).map(u16::from),
            "{op:?}"
        );
    }
    for &op in Vop2Op::ALL {
        assert_eq!(
            Vop3Op::V2(op).opcode(ARCH),
            op.opcode(ARCH).map(|o| 0x100 + u16::from(o)),
            "{op:?}"
        );
    }
    for &op in Vop1Op::ALL {
        let expected = op.opcode(ARCH).map(|o| 0x140 + u16::from(o));
        match op {
            Vop1Op::ReadfirstlaneB32 | Vop1Op::AccvgprMovB32 | Vop1Op::SwapB32 => {
                assert_eq!(Vop3Op::V1(op).opcode(ARCH), None, "{op:?}")
            }
            _ => assert_eq!(Vop3Op::V1(op).opcode(ARCH), expected, "{op:?}"),
        }
    }
    // Spot values from `llvm-mc -mcpu=gfx942 -show-encoding` and amdgpu_isa_cdna3.xml.
    assert_eq!(Vop3Op::C(VopcOp::CmpLtF32).opcode(ARCH), Some(65));
    assert_eq!(Vop3Op::C(VopcOp::CmpEqU32).opcode(ARCH), Some(202));
    assert_eq!(Vop3Op::V2(Vop2Op::CndmaskB32).opcode(ARCH), Some(0x100));
    assert_eq!(ADD_F32.opcode(ARCH), Some(0x101));
    assert_eq!(Vop3Op::V2(Vop2Op::AddCoU32).opcode(ARCH), Some(281));
    assert_eq!(MOV_B32.opcode(ARCH), Some(0x141));
    assert_eq!(Vop3Op::V1(Vop1Op::CvtF32Fp8).opcode(ARCH), Some(404));
    assert_eq!(Vop3Op::V1(Vop1Op::MovB64).opcode(ARCH), Some(376));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::FmaF32).opcode(ARCH), Some(459));
    assert_eq!(
        Vop3Op::Only(Vop3OnlyOp::DivScaleF32).opcode(ARCH),
        Some(480)
    );
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::MadU64U32).opcode(ARCH), Some(488));
    assert_eq!(LSHL_ADD_U64.opcode(ARCH), Some(520));
    assert_eq!(
        Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32).opcode(ARCH),
        Some(674)
    );
    assert_eq!(
        Vop3Op::Only(Vop3OnlyOp::CvtSrBf8F32).opcode(ARCH),
        Some(677)
    );
    assert_eq!(READLANE.opcode(ARCH), Some(649));
    assert_eq!(WRITELANE.opcode(ARCH), Some(650));
    assert_eq!(NOP.opcode(ARCH), Some(320));
}

#[test]
fn opcodes_without_a_64_bit_form_are_excluded() {
    for op in [
        Vop1Op::ReadfirstlaneB32,
        Vop1Op::AccvgprMovB32,
        Vop1Op::SwapB32,
    ] {
        let op = Vop3Op::V1(op);
        assert!(!Vop3Op::all(ARCH).contains(&op), "{op:?}");
        assert_eq!(op.opcode(ARCH), None);
        let inst = new(op, v(0), None, [Some(v(1)), None, None]);
        assert!(inst.encode(ARCH, &mut Vec::new()).is_err(), "{op:?}");
    }
    // 0x140 + 2, 0x140 + 81, 0x140 + 82.
    for opcode in [0x142u16, 0x191, 0x192] {
        assert_eq!(Vop3Op::from_opcode(ARCH, opcode), None, "{opcode:#x}");
    }
}

#[test]
fn vop3b_ops_are_exactly_the_carry_scale_and_wide_mad_ops() {
    let vop3b: Vec<Vop3Op> = Vop3Op::all(ARCH)
        .into_iter()
        .filter(|op| op.is_vop3b())
        .collect();
    let expected = [
        Vop3Op::V2(Vop2Op::AddCoU32),
        Vop3Op::V2(Vop2Op::SubCoU32),
        Vop3Op::V2(Vop2Op::SubrevCoU32),
        Vop3Op::V2(Vop2Op::AddcCoU32),
        Vop3Op::V2(Vop2Op::SubbCoU32),
        Vop3Op::V2(Vop2Op::SubbrevCoU32),
        Vop3Op::Only(Vop3OnlyOp::DivScaleF32),
        Vop3Op::Only(Vop3OnlyOp::MadU64U32),
        Vop3Op::Only(Vop3OnlyOp::MadI64I32),
    ];
    assert_eq!(vop3b.len(), expected.len());
    for op in expected {
        assert!(vop3b.contains(&op), "{op:?}");
        assert_eq!(op.mods().abs, 0, "{op:?}: VOP3B has no ABS field");
        assert_eq!(op.mods().op_sel, 0, "{op:?}: VOP3B has no OP_SEL field");
    }
}

#[test]
fn operand_widths() {
    assert_eq!(NOP.widths(), (0, 0, 0, 0));
    assert_eq!(ADD_F32.widths(), (1, 1, 1, 0));
    assert_eq!(MOV_B32.widths(), (1, 1, 0, 0));
    assert_eq!(Vop3Op::V1(Vop1Op::MovB64).widths(), (2, 2, 0, 0));
    assert_eq!(Vop3Op::V1(Vop1Op::CvtPkF32Fp8).widths(), (2, 1, 0, 0));
    assert_eq!(Vop3Op::V1(Vop1Op::CvtF64F32).widths(), (2, 1, 0, 0));
    assert_eq!(Vop3Op::V1(Vop1Op::CvtF32F64).widths(), (1, 2, 0, 0));
    assert_eq!(Vop3Op::C(VopcOp::CmpLtF32).widths(), (2, 1, 1, 0));
    assert_eq!(Vop3Op::C(VopcOp::CmpEqU64).widths(), (2, 2, 2, 0));
    assert_eq!(Vop3Op::V2(Vop2Op::CndmaskB32).widths(), (1, 1, 1, 2));
    assert_eq!(Vop3Op::V2(Vop2Op::AddcCoU32).widths(), (1, 1, 1, 2));
    assert_eq!(Vop3Op::V2(Vop2Op::AddCoU32).widths(), (1, 1, 1, 0));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::FmaF32).widths(), (1, 1, 1, 1));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::MadU64U32).widths(), (2, 1, 1, 2));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::LshlrevB64).widths(), (2, 1, 2, 0));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::FmaF64).widths(), (2, 2, 2, 2));
    assert_eq!(ADD_F64.widths(), (2, 2, 2, 0));
    assert_eq!(LSHL_ADD_U64.widths(), (2, 2, 1, 2));
    assert_eq!(READLANE.widths(), (1, 1, 1, 0));
    assert_eq!(WRITELANE.widths(), (1, 1, 1, 0));
    assert_eq!(Vop3Op::Only(Vop3OnlyOp::CvtSrFp8F32).widths(), (1, 1, 1, 0));
}

#[test]
fn every_op_has_at_least_two_patterns_and_every_pattern_round_trips_on_the_cpu() {
    let mut covered = HashSet::new();
    for op in Vop3Op::all(ARCH) {
        let cases = patterns(op);
        // `v_nop` has exactly one encodable form (every field is zero).
        let need = if op == NOP { 1 } else { 2 };
        assert!(cases.len() >= need, "{op:?}: {} patterns", cases.len());
        let distinct: HashSet<String> = cases.iter().map(|c| format!("{c}")).collect();
        assert!(distinct.len() >= need, "{op:?}: patterns print identically");
        for case in cases.iter().cloned().chain(sweeps(op)) {
            let w = words(&case);
            assert_eq!(w.len(), 2);
            let (decoded, n) =
                Vop3::decode(ARCH, &w).unwrap_or_else(|e| panic!("`{case}` {w:08x?}: {e}"));
            assert_eq!(n, 2);
            assert_eq!(decoded, case, "`{case}`");
            let mut again = Vec::new();
            decoded.encode(ARCH, &mut again).unwrap();
            assert_eq!(again, w, "`{case}`");
        }
        covered.insert(op);
    }
    assert_eq!(covered.len(), Vop3Op::all(ARCH).len());
}

// ---------------------------------------------------------------------------
// Bit layout, printing, footprint (llvm facts recorded from llvm-mc, gfx942)
// ---------------------------------------------------------------------------

fn assert_layout(inst: &Vop3, text: &str, lo: u32, hi: u32) {
    assert_eq!(inst.to_string(), text);
    assert_eq!(words(inst), vec![lo, hi], "{text}");
    let (decoded, n) = Vop3::decode(ARCH, &[lo, hi]).unwrap();
    assert_eq!((&decoded, n), (inst, 2), "{text}");
}

#[test]
fn bit_layout_matches_llvm_mc() {
    // v_add_f32_e64 v0, v1, v2
    assert_layout(
        &bin(ADD_F32, v(0), v(1), v(2)),
        "v_add_f32_e64 v0, v1, v2",
        0xd101_0000,
        0x0002_0501,
    );
    // v_nop_e64
    assert_layout(&new(NOP, v(0), None, [None; 3]), "v_nop", 0xd140_0000, 0);
    // v_readlane_b32 s0, v1, s2
    assert_layout(
        &bin(READLANE, sg(0), v(1), sg(2)),
        "v_readlane_b32 s0, v1, s2",
        0xd289_0000,
        0x0000_0501,
    );
    // v_div_scale_f32 v4, s[12:13], v20, v24, v28 (VOP3B: SDST in [14:8])
    let div_scale = new(
        Vop3Op::Only(Vop3OnlyOp::DivScaleF32),
        v(4),
        Some(sg(12)),
        [Some(v(20)), Some(v(24)), Some(v(28))],
    );
    assert_layout(
        &div_scale,
        "v_div_scale_f32 v4, s[12:13], v20, v24, v28",
        0xd1e0_0c04,
        0x0472_3114,
    );
    // v_fma_f32 v4, -v20, |v24|, -|v28| clamp div:2 (ABS [10:8], CLAMP [15], OMOD [28:27], NEG [31:29])
    let fma = Vop3 {
        abs: 0b110,
        neg: 0b101,
        clamp: true,
        omod: 3,
        ..new(
            Vop3Op::Only(Vop3OnlyOp::FmaF32),
            v(4),
            None,
            [Some(v(20)), Some(v(24)), Some(v(28))],
        )
    };
    assert_layout(
        &fma,
        "v_fma_f32 v4, -v20, |v24|, -|v28| clamp div:2",
        0xd1cb_8604,
        0xbc72_3114,
    );
    // v_cmp_class_f32_e64 s[4:5], -|v20|, v24
    let class = Vop3 {
        abs: 1,
        neg: 1,
        ..bin(Vop3Op::C(VopcOp::CmpClassF32), sg(4), v(20), v(24))
    };
    assert_layout(
        &class,
        "v_cmp_class_f32_e64 s[4:5], -|v20|, v24",
        0xd010_0104,
        0x2002_3114,
    );
    // v_cvt_pk_fp8_f32 v4, v20, v24 op_sel:[0,0,1] (OP_SEL[3] = destination half)
    let pk = Vop3 {
        op_sel: 0b1000,
        ..bin(Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32), v(4), v(20), v(24))
    };
    assert_layout(
        &pk,
        "v_cvt_pk_fp8_f32 v4, v20, v24 op_sel:[0,0,1]",
        0xd2a2_4004,
        0x0002_3114,
    );
    // v_cvt_sr_fp8_f32 v4, v20, v24 op_sel:[0,0,1,1] (OP_SEL[3:2] = destination byte)
    let sr = Vop3 {
        op_sel: 0b1100,
        ..bin(Vop3Op::Only(Vop3OnlyOp::CvtSrFp8F32), v(4), v(20), v(24))
    };
    assert_layout(
        &sr,
        "v_cvt_sr_fp8_f32 v4, v20, v24 op_sel:[0,0,1,1]",
        0xd2a4_6004,
        0x0002_3114,
    );
    // v_lshl_add_u64 v[4:5], v[20:21], v24, v[28:29]
    let lshl = new(
        LSHL_ADD_U64,
        v(4),
        None,
        [Some(v(20)), Some(v(24)), Some(v(28))],
    );
    assert_layout(
        &lshl,
        "v_lshl_add_u64 v[4:5], v[20:21], v24, v[28:29]",
        0xd208_0004,
        0x0472_3114,
    );
}

#[test]
fn display_matches_llvm_spelling() {
    let t = |inst: Vop3| inst.to_string();
    let x = bin(ADD_F32, v(4), v(20), v(24));
    assert_eq!(
        t(Vop3 {
            neg: 1,
            abs: 1,
            ..x.clone()
        }),
        "v_add_f32_e64 v4, -|v20|, v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(fl(2)), Some(v(24)), None],
            neg: 1,
            ..x.clone()
        }),
        "v_add_f32_e64 v4, neg(1.0), v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(fl(2)), Some(v(24)), None],
            abs: 1,
            ..x.clone()
        }),
        "v_add_f32_e64 v4, |1.0|, v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(fl(2)), Some(v(24)), None],
            neg: 1,
            abs: 1,
            ..x.clone()
        }),
        "v_add_f32_e64 v4, -|1.0|, v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(int(1)), Some(v(24)), None],
            neg: 1,
            ..x.clone()
        }),
        "v_add_f32_e64 v4, neg(1), v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(fl(3)), Some(v(24)), None],
            ..x.clone()
        }),
        "v_add_f32_e64 v4, -1.0, v24"
    );
    assert_eq!(
        t(Vop3 {
            src: [Some(int(-8)), Some(v(24)), None],
            ..x
        }),
        "v_add_f32_e64 v4, -8, v24"
    );
    // 64-bit slots print 1/(2*pi) as the f64 constant, 32-bit slots as the f32 spelling.
    assert_eq!(
        t(bin(ADD_F64, v(4), Src::Float(InlineFloat::InvTwoPi), v(24))),
        "v_add_f64 v[4:5], 0.15915494309189532, v[24:25]"
    );
    assert_eq!(
        t(bin(ADD_F32, v(4), Src::Float(InlineFloat::InvTwoPi), v(24))),
        "v_add_f32_e64 v4, 0.15915494, v24"
    );
    assert_eq!(
        t(Vop3 {
            neg: 1,
            ..bin(ADD_F64, v(4), Src::SharedBase, v(24))
        }),
        "v_add_f64 v[4:5], -src_shared_base, v[24:25]"
    );
    // Modifier order: op_sel, clamp, omod.
    let pack = Vop3 {
        op_sel: 0b1001,
        clamp: true,
        ..bin(Vop3Op::Only(Vop3OnlyOp::PackB32F16), v(4), v(20), v(24))
    };
    assert_eq!(
        pack.to_string(),
        "v_pack_b32_f16 v4, v20, v24 op_sel:[1,0,1] clamp"
    );
    let alignbit = Vop3 {
        op_sel: 0b1011,
        clamp: true,
        ..new(
            Vop3Op::Only(Vop3OnlyOp::AlignbitB32),
            v(4),
            None,
            [Some(v(20)), Some(v(24)), Some(v(28))],
        )
    };
    assert_eq!(
        alignbit.to_string(),
        "v_alignbit_b32 v4, v20, v24, v28 op_sel:[1,1,0,1] clamp"
    );
    let cvt = Vop3 {
        clamp: true,
        omod: 1,
        ..new(
            Vop3Op::V1(Vop1Op::CvtF32I32),
            v(4),
            None,
            [Some(v(20)), None, None],
        )
    };
    assert_eq!(cvt.to_string(), "v_cvt_f32_i32_e64 v4, v20 clamp mul:2");
    assert_eq!(
        Vop3 {
            omod: 2,
            ..cvt.clone()
        }
        .to_string(),
        "v_cvt_f32_i32_e64 v4, v20 clamp mul:4"
    );
    // VOP3B: sdst after dst; VCC prints as `vcc`.
    let scale = Vop3 {
        neg: 1,
        clamp: true,
        omod: 1,
        ..new(
            Vop3Op::Only(Vop3OnlyOp::DivScaleF32),
            v(4),
            Some(Src::VccLo),
            [Some(v(20)), Some(v(24)), Some(v(28))],
        )
    };
    assert_eq!(
        scale.to_string(),
        "v_div_scale_f32 v4, vcc, -v20, v24, v28 clamp mul:2"
    );
    // 64-bit scalar operands, mask operands, lane operands.
    let mad = new(
        Vop3Op::Only(Vop3OnlyOp::MadU64U32),
        v(4),
        Some(sg(12)),
        [Some(v(20)), Some(v(24)), Some(v(28))],
    );
    assert_eq!(
        mad.to_string(),
        "v_mad_u64_u32 v[4:5], s[12:13], v20, v24, v[28:29]"
    );
    let cnd = new(
        Vop3Op::V2(Vop2Op::CndmaskB32),
        v(4),
        None,
        [Some(v(20)), Some(v(24)), Some(Src::VccLo)],
    );
    assert_eq!(cnd.to_string(), "v_cndmask_b32_e64 v4, v20, v24, vcc");
    let cmp = bin(Vop3Op::C(VopcOp::CmpLtU32), Src::ExecLo, v(1), v(2));
    assert_eq!(cmp.to_string(), "v_cmp_lt_u32_e64 exec, v1, v2");
    assert_eq!(
        bin(READLANE, Src::VccHi, v(1), Src::M0).to_string(),
        "v_readlane_b32 vcc_hi, v1, m0"
    );
    assert_eq!(
        bin(WRITELANE, v(1), ttmp(3), int(7)).to_string(),
        "v_writelane_b32 v1, ttmp3, 7"
    );
}

fn span(file: RegFile, base: u8, len: u8) -> RegSpan {
    RegSpan::new(file, base, len)
}

#[test]
fn footprints() {
    let exec = span(RegFile::Exec, 0, 2);
    // Plain VOP3A: dst written, sources read, EXEC read.
    let fp = bin(ADD_F32, v(4), v(20), sg(3)).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 1)]);
    assert!(fp.uses.contains(&span(RegFile::Vgpr, 20, 1)));
    assert!(fp.uses.contains(&span(RegFile::Sgpr, 3, 1)));
    assert!(fp.uses.contains(&exec));
    // Constants are not state.
    let fp = bin(ADD_F32, v(4), int(3), fl(1)).footprint();
    assert_eq!(fp.uses, vec![exec]);
    // Compare: 64-bit SGPR / VCC destination.
    let fp = bin(Vop3Op::C(VopcOp::CmpLtU32), sg(10), v(1), v(2)).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Sgpr, 10, 2)]);
    let fp = bin(Vop3Op::C(VopcOp::CmpLtU32), Src::VccLo, v(1), v(2)).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vcc, 0, 2)]);
    // v_cmpx: also writes EXEC.
    let fp = bin(Vop3Op::C(VopcOp::CmpxGtU32), sg(10), v(1), v(2)).footprint();
    assert!(fp.defs.contains(&span(RegFile::Sgpr, 10, 2)) && fp.defs.contains(&exec));
    // VOP3B: sdst is a def, wide operands span two registers, carry-in is a read.
    let fp = new(
        Vop3Op::Only(Vop3OnlyOp::MadU64U32),
        v(4),
        Some(sg(12)),
        [Some(v(20)), Some(v(24)), Some(v(28))],
    )
    .footprint();
    assert_eq!(
        fp.defs,
        vec![span(RegFile::Vgpr, 4, 2), span(RegFile::Sgpr, 12, 2)]
    );
    assert!(fp.uses.contains(&span(RegFile::Vgpr, 28, 2)));
    let fp = new(
        Vop3Op::V2(Vop2Op::AddcCoU32),
        v(4),
        Some(Src::VccLo),
        [Some(v(20)), Some(v(24)), Some(sg(6))],
    )
    .footprint();
    assert_eq!(
        fp.defs,
        vec![span(RegFile::Vgpr, 4, 1), span(RegFile::Vcc, 0, 2)]
    );
    assert!(fp.uses.contains(&span(RegFile::Sgpr, 6, 2)));
    // v_div_fmas_f32 reads VCC implicitly.
    let fp = new(
        DIV_FMAS,
        v(4),
        None,
        [Some(v(20)), Some(v(24)), Some(v(28))],
    )
    .footprint();
    assert!(fp.uses.contains(&span(RegFile::Vcc, 0, 2)));
    // FP8 packed / stochastic converts keep part of the destination: it is an input too.
    for op in [
        Vop3OnlyOp::CvtPkFp8F32,
        Vop3OnlyOp::CvtPkBf8F32,
        Vop3OnlyOp::CvtSrFp8F32,
        Vop3OnlyOp::CvtSrBf8F32,
    ] {
        let fp = Vop3 {
            op_sel: 0b1000,
            ..bin(Vop3Op::Only(op), v(4), v(20), v(24))
        }
        .footprint();
        assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 1)], "{op:?}");
        assert!(
            fp.uses.contains(&span(RegFile::Vgpr, 4, 1)),
            "{op:?}: old dst is read"
        );
    }
    // v_readlane_b32 / v_writelane_b32 address one lane regardless of EXEC.
    let fp = bin(READLANE, sg(5), v(7), sg(3)).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Sgpr, 5, 1)]);
    assert_eq!(
        fp.uses,
        vec![span(RegFile::Vgpr, 7, 1), span(RegFile::Sgpr, 3, 1)]
    );
    let fp = bin(WRITELANE, v(9), sg(2), Src::M0).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 9, 1)]);
    assert!(
        fp.uses.contains(&span(RegFile::Vgpr, 9, 1)),
        "31 lanes are preserved"
    );
    assert!(
        fp.uses.contains(&span(RegFile::M0, 0, 1)) && fp.uses.contains(&span(RegFile::Sgpr, 2, 1))
    );
    assert!(!fp.uses.contains(&exec));
    // v_nop touches nothing.
    assert_eq!(new(NOP, v(0), None, [None; 3]).footprint().defs, vec![]);
    // v_lshl_add_u64: 64-bit destination and operands.
    let fp = new(
        LSHL_ADD_U64,
        v(4),
        None,
        [Some(v(20)), Some(v(24)), Some(v(28))],
    )
    .footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 2)]);
    assert!(
        fp.uses.contains(&span(RegFile::Vgpr, 20, 2))
            && fp.uses.contains(&span(RegFile::Vgpr, 24, 1))
            && fp.uses.contains(&span(RegFile::Vgpr, 28, 2))
    );
}

#[test]
fn inst_level_encode_and_decode_all() {
    let a = bin(ADD_F32, v(0), v(1), v(2));
    let b = Vop3 {
        clamp: true,
        ..bin(ADD_U32, v(3), sg(4), int(5))
    };
    let insts = vec![Inst::Vop3(a.clone()), Inst::Vop3(b.clone())];
    let w = encode_all(ARCH, &insts).unwrap();
    assert_eq!(w.len(), 4);
    assert_eq!(
        decode_all(ARCH, &w).unwrap(),
        vec![(0, Inst::Vop3(a.clone())), (2, Inst::Vop3(b))]
    );
    assert_eq!(Inst::Vop3(a.clone()).footprint(), a.footprint());
    assert_eq!(Inst::Vop3(a.clone()).to_string(), a.to_string());
}

// ---------------------------------------------------------------------------
// Rejections (CPU)
// ---------------------------------------------------------------------------

fn encode_err(inst: &Vop3) -> String {
    match inst.encode(ARCH, &mut Vec::new()) {
        Ok(()) => panic!("`{inst}` ({inst:?}) must not encode"),
        Err(e) => e,
    }
}

/// Forms llvm-mc rejects, with the assembler spelling. The encoder must
/// reject each of them too.
fn llvm_rejected() -> Vec<(&'static str, Vop3)> {
    let m0 = Src::M0;
    vec![
        // Constant bus: one scalar source per VOP3 on gfx9.
        ("v_add_f32_e64 v0, s1, s2", bin(ADD_F32, v(0), sg(1), sg(2))),
        (
            "v_add_f32_e64 v0, s1, vcc_lo",
            bin(ADD_F32, v(0), sg(1), Src::VccLo),
        ),
        (
            "v_div_fmas_f32 v0, s1, v2, v3",
            new(DIV_FMAS, v(0), None, [Some(sg(1)), Some(v(2)), Some(v(3))]),
        ),
        (
            "v_mad_u64_u32 v[0:1], s[4:5], s2, v1, s[2:3]",
            new(
                Vop3Op::Only(Vop3OnlyOp::MadU64U32),
                v(0),
                Some(sg(4)),
                [Some(sg(2)), Some(v(1)), Some(sg(2))],
            ),
        ),
        (
            "v_writelane_b32 v0, s1, s2",
            bin(WRITELANE, v(0), sg(1), sg(2)),
        ),
        // No literal, no AccVGPR.
        (
            "v_add_f32_e64 v0, 0x1234, v1",
            bin(ADD_F32, v(0), Src::Lit(0x1234), v(1)),
        ),
        (
            "v_add_f32_e64 v0, a1, v2",
            bin(ADD_F32, v(0), Src::Agpr(1), v(2)),
        ),
        // Register numbers and inline-constant range.
        (
            "v_mov_b32_e64 v0, s102",
            new(MOV_B32, v(0), None, [Some(sg(102)), None, None]),
        ),
        (
            "v_mov_b32_e64 v0, 65",
            new(MOV_B32, v(0), None, [Some(int(65)), None, None]),
        ),
        (
            "v_mov_b32_e64 v0, -17",
            new(MOV_B32, v(0), None, [Some(int(-17)), None, None]),
        ),
        // 64-bit alignment.
        (
            "v_add_f64 v[0:1], v[2:3], v[5:6]",
            bin(ADD_F64, v(0), v(2), v(5)),
        ),
        (
            "v_add_f64 v[1:2], v[2:3], v[4:5]",
            bin(ADD_F64, v(1), v(2), v(4)),
        ),
        (
            "v_add_f64 v[0:1], s[1:2], v[4:5]",
            bin(ADD_F64, v(0), sg(1), v(4)),
        ),
        (
            "v_cmp_eq_u32_e64 s[1:2], v0, v1",
            bin(Vop3Op::C(VopcOp::CmpEqU32), sg(1), v(0), v(1)),
        ),
        (
            "v_cndmask_b32_e64 v0, v1, v2, s[3:4]",
            new(
                Vop3Op::V2(Vop2Op::CndmaskB32),
                v(0),
                None,
                [Some(v(1)), Some(v(2)), Some(sg(3))],
            ),
        ),
        // Unsupported modifiers.
        (
            "v_add_u32_e64 v0, -v1, v2",
            Vop3 {
                neg: 1,
                ..bin(ADD_U32, v(0), v(1), v(2))
            },
        ),
        (
            "v_add_u32_e64 v0, |v1|, v2",
            Vop3 {
                abs: 1,
                ..bin(ADD_U32, v(0), v(1), v(2))
            },
        ),
        (
            "v_bfe_u32 v0, v1, v2, v3 clamp",
            Vop3 {
                clamp: true,
                ..new(
                    Vop3Op::Only(Vop3OnlyOp::BfeU32),
                    v(0),
                    None,
                    [Some(v(1)), Some(v(2)), Some(v(3))],
                )
            },
        ),
        (
            "v_add_f32_e64 v0, v1, v2 op_sel:[1,0]",
            Vop3 {
                op_sel: 1,
                ..bin(ADD_F32, v(0), v(1), v(2))
            },
        ),
        (
            "v_add_u32_e64 v0, v1, v2 mul:2",
            Vop3 {
                omod: 1,
                ..bin(ADD_U32, v(0), v(1), v(2))
            },
        ),
        (
            "v_mov_b32_e64 v0, v1 clamp",
            Vop3 {
                clamp: true,
                ..new(MOV_B32, v(0), None, [Some(v(1)), None, None])
            },
        ),
        (
            "v_div_scale_f32 v0, vcc, |v1|, v2, v3",
            Vop3 {
                abs: 1,
                ..new(
                    Vop3Op::Only(Vop3OnlyOp::DivScaleF32),
                    v(0),
                    Some(Src::VccLo),
                    [Some(v(1)), Some(v(2)), Some(v(3))],
                )
            },
        ),
        // Source presence.
        (
            "v_add_f32_e64 v0, v1, v2, v3",
            new(ADD_F32, v(0), None, [Some(v(1)), Some(v(2)), Some(v(3))]),
        ),
        (
            "v_add_f32_e64 v0, v1",
            new(ADD_F32, v(0), None, [Some(v(1)), None, None]),
        ),
        // Operand classes.
        (
            "v_mov_b32_e64 vcc_lo, v1",
            new(MOV_B32, Src::VccLo, None, [Some(v(1)), None, None]),
        ),
        (
            "v_cmp_eq_u32_e64 vcc_hi, v0, v1",
            bin(Vop3Op::C(VopcOp::CmpEqU32), Src::VccHi, v(0), v(1)),
        ),
        (
            "v_cndmask_b32_e64 v0, v1, v2, 1",
            new(
                Vop3Op::V2(Vop2Op::CndmaskB32),
                v(0),
                None,
                [Some(v(1)), Some(v(2)), Some(int(1))],
            ),
        ),
        (
            "v_cndmask_b32_e64 v0, v1, v2, m0",
            new(
                Vop3Op::V2(Vop2Op::CndmaskB32),
                v(0),
                None,
                [Some(v(1)), Some(v(2)), Some(m0)],
            ),
        ),
        (
            "v_readlane_b32 s0, s1, s2",
            bin(READLANE, sg(0), sg(1), sg(2)),
        ),
        (
            "v_readlane_b32 s0, v1, v2",
            bin(READLANE, sg(0), v(1), v(2)),
        ),
        ("v_readlane_b32 m0, v1, s2", bin(READLANE, m0, v(1), sg(2))),
        (
            "v_writelane_b32 v0, v1, s2",
            bin(WRITELANE, v(0), v(1), sg(2)),
        ),
        // No 64-bit form in llvm.
        (
            "v_readfirstlane_b32_e64 s0, v1",
            new(
                Vop3Op::V1(Vop1Op::ReadfirstlaneB32),
                sg(0),
                None,
                [Some(v(1)), None, None],
            ),
        ),
        (
            "v_swap_b32_e64 v0, v1",
            new(
                Vop3Op::V1(Vop1Op::SwapB32),
                v(0),
                None,
                [Some(v(1)), None, None],
            ),
        ),
    ]
}

/// Forms the encoder rejects although llvm-mc assembles them (hardware or ISA
/// restrictions, or fields llvm decodes but are not writable).
fn encoder_only_rejected() -> Vec<(&'static str, Vop3)> {
    let fp8 = |op: Vop3OnlyOp, a: Src, b: Src| bin(Vop3Op::Only(op), v(0), a, b);
    vec![
        (
            "FP8 pk convert: SGPR source (CDNA3 ISA §7.2 p.51)",
            fp8(Vop3OnlyOp::CvtPkFp8F32, sg(1), v(2)),
        ),
        (
            "FP8 pk convert: inline constant source",
            fp8(Vop3OnlyOp::CvtPkBf8F32, v(1), int(2)),
        ),
        (
            "FP8 sr convert: SGPR random source",
            fp8(Vop3OnlyOp::CvtSrFp8F32, v(1), sg(2)),
        ),
        (
            "FP8 sr convert: inline constant source",
            fp8(Vop3OnlyOp::CvtSrBf8F32, fl(2), v(2)),
        ),
        (
            "FP8 convert: CLAMP is ignored by the hardware and not encodable",
            Vop3 {
                clamp: true,
                ..fp8(Vop3OnlyOp::CvtPkFp8F32, v(1), v(2))
            },
        ),
        (
            "FP8 convert: OMOD",
            Vop3 {
                omod: 1,
                ..fp8(Vop3OnlyOp::CvtSrFp8F32, v(1), v(2))
            },
        ),
        (
            "pk convert: OP_SEL[2] does not exist",
            Vop3 {
                op_sel: 0b0100,
                ..fp8(Vop3OnlyOp::CvtPkFp8F32, v(1), v(2))
            },
        ),
        (
            "pack: OP_SEL[2] does not exist",
            Vop3 {
                op_sel: 0b0100,
                ..bin(Vop3Op::Only(Vop3OnlyOp::PackB32F16), v(0), v(1), v(2))
            },
        ),
        (
            "NEG on a source the op does not have",
            Vop3 {
                neg: 0b100,
                ..bin(ADD_F32, v(0), v(1), v(2))
            },
        ),
        (
            "ABS on the integer exponent of v_ldexp_f32",
            Vop3 {
                abs: 0b10,
                ..bin(Vop3Op::Only(Vop3OnlyOp::LdexpF32), v(0), v(1), v(2))
            },
        ),
        (
            "NEG on the random source of v_cvt_sr_fp8_f32",
            Vop3 {
                neg: 0b10,
                ..fp8(Vop3OnlyOp::CvtSrFp8F32, v(1), v(2))
            },
        ),
        (
            "neg out of range",
            Vop3 {
                neg: 8,
                ..bin(ADD_F32, v(0), v(1), v(2))
            },
        ),
        (
            "omod out of range",
            Vop3 {
                omod: 4,
                ..bin(ADD_F32, v(0), v(1), v(2))
            },
        ),
        (
            "op_sel out of range",
            Vop3 {
                op_sel: 16,
                ..new(
                    Vop3Op::Only(Vop3OnlyOp::AlignbitB32),
                    v(0),
                    None,
                    [Some(v(1)), Some(v(2)), Some(v(3))],
                )
            },
        ),
        (
            "VOP3B without sdst",
            new(
                Vop3Op::V2(Vop2Op::AddCoU32),
                v(0),
                None,
                [Some(v(1)), Some(v(2)), None],
            ),
        ),
        (
            "VOP3A with sdst",
            new(ADD_F32, v(0), Some(sg(2)), [Some(v(1)), Some(v(2)), None]),
        ),
        (
            "sdst not writable (src_scc)",
            new(
                Vop3Op::V2(Vop2Op::AddCoU32),
                v(0),
                Some(Src::Scc),
                [Some(v(1)), Some(v(2)), None],
            ),
        ),
        (
            "sdst odd",
            new(
                Vop3Op::V2(Vop2Op::AddCoU32),
                v(0),
                Some(sg(3)),
                [Some(v(1)), Some(v(2)), None],
            ),
        ),
        (
            "sdst vcc_hi",
            new(
                Vop3Op::V2(Vop2Op::AddCoU32),
                v(0),
                Some(Src::VccHi),
                [Some(v(1)), Some(v(2)), None],
            ),
        ),
        (
            "compare destination not writable (src_shared_base)",
            bin(Vop3Op::C(VopcOp::CmpEqU32), Src::SharedBase, v(0), v(1)),
        ),
        (
            "compare destination VGPR",
            bin(Vop3Op::C(VopcOp::CmpEqU32), v(0), v(0), v(1)),
        ),
        ("vector destination SGPR", bin(ADD_F32, sg(0), v(0), v(1))),
        ("v_nop with a destination", new(NOP, v(1), None, [None; 3])),
        (
            "v_nop with a source",
            new(NOP, v(0), None, [Some(v(1)), None, None]),
        ),
        (
            "v_lshl_add_u64 shift count 5 (CDNA3 ISA §12.11)",
            new(
                LSHL_ADD_U64,
                v(0),
                None,
                [Some(v(2)), Some(int(5)), Some(v(4))],
            ),
        ),
        (
            "v_lshl_add_u64 negative shift count",
            new(
                LSHL_ADD_U64,
                v(0),
                None,
                [Some(v(2)), Some(int(-1)), Some(v(4))],
            ),
        ),
        (
            "v_lshl_add_u64 odd src2",
            new(
                LSHL_ADD_U64,
                v(0),
                None,
                [Some(v(2)), Some(v(4)), Some(v(5))],
            ),
        ),
        ("ttmp pair odd", bin(ADD_F64, v(0), ttmp(1), v(2))),
        ("ttmp out of range", bin(ADD_F32, v(0), ttmp(16), v(2))),
        (
            "exec_hi starting a 64-bit operand",
            bin(ADD_F64, v(0), Src::ExecHi, v(2)),
        ),
        ("m0 as a 64-bit operand", bin(ADD_F64, v(0), Src::M0, v(2))),
        (
            "writelane lane select m0 does not exempt src0 from the bus",
            bin(WRITELANE, v(0), Src::M0, sg(2)),
        ),
        (
            "readlane destination m0",
            bin(READLANE, Src::M0, v(1), sg(2)),
        ),
        (
            "literal in the lane select",
            bin(READLANE, sg(0), v(1), Src::Lit(5)),
        ),
        (
            "readlane source in AccVGPR",
            bin(READLANE, sg(0), Src::Agpr(1), sg(2)),
        ),
    ]
}

#[test]
fn encoder_rejects_what_llvm_rejects() {
    for (asm, inst) in llvm_rejected() {
        let e = encode_err(&inst);
        assert!(
            e.contains(inst.op.name()),
            "`{asm}`: error `{e}` does not name the opcode"
        );
    }
}

#[test]
fn encoder_rejects_illegal_forms_beyond_llvm() {
    for (why, inst) in encoder_only_rejected() {
        let e = encode_err(&inst);
        assert!(!e.is_empty(), "{why}");
    }
}

#[test]
fn legal_forms_that_look_illegal_encode() {
    // Same SGPR twice is one constant-bus read; inline constants are free.
    for inst in [
        bin(ADD_F32, v(0), sg(1), sg(1)),
        bin(ADD_F32, v(0), sg(1), fl(2)),
        new(
            Vop3Op::Only(Vop3OnlyOp::FmaF32),
            v(0),
            None,
            [Some(sg(1)), Some(v(2)), Some(sg(1))],
        ),
        bin(ADD_F64, v(0), sg(2), sg(2)),
        // v_writelane_b32: M0 lane select is free next to one other scalar.
        bin(WRITELANE, v(0), sg(1), Src::M0),
        bin(WRITELANE, v(0), Src::M0, Src::M0),
        // Compare with VCC / EXEC / TTMP / FLAT_SCRATCH / XNACK_MASK destinations.
        bin(Vop3Op::C(VopcOp::CmpLtU32), Src::VccLo, v(1), v(2)),
        bin(Vop3Op::C(VopcOp::CmpLtU32), Src::ExecLo, v(1), v(2)),
        bin(Vop3Op::C(VopcOp::CmpLtU32), Src::FlatScratchLo, v(1), v(2)),
        bin(Vop3Op::C(VopcOp::CmpLtU32), Src::XnackMaskLo, v(1), v(2)),
        bin(Vop3Op::C(VopcOp::CmpLtU32), ttmp(14), v(1), v(2)),
        // Readlane destinations llvm accepts: VCC halves, EXEC halves, TTMP.
        bin(READLANE, Src::VccHi, v(1), sg(2)),
        bin(READLANE, Src::ExecHi, v(1), Src::M0),
    ] {
        let w = words(&inst);
        assert_eq!(Vop3::decode(ARCH, &w).unwrap().0, inst);
    }
}

fn mutate(inst: &Vop3, f: impl Fn(&mut [u32])) -> Vec<u32> {
    let mut w = words(inst);
    f(&mut w);
    w
}

#[test]
fn decoder_rejects_malformed_words() {
    let add = bin(ADD_F32, v(4), v(20), v(24));
    let mov = new(MOV_B32, v(4), None, [Some(v(20)), None, None]);
    let cmp = bin(Vop3Op::C(VopcOp::CmpEqU32), sg(10), v(20), v(24));
    let add_u32 = bin(ADD_U32, v(4), v(20), v(24));
    let scale = new(
        Vop3Op::Only(Vop3OnlyOp::DivScaleF32),
        v(4),
        Some(sg(12)),
        [Some(v(20)), Some(v(24)), Some(v(28))],
    );
    let bfe = new(
        Vop3Op::Only(Vop3OnlyOp::BfeU32),
        v(4),
        None,
        [Some(v(20)), Some(v(24)), Some(v(28))],
    );
    let nop = new(NOP, v(0), None, [None; 3]);
    let cases: Vec<(&str, Vec<u32>)> = vec![
        ("one dword", vec![words(&add)[0]]),
        ("empty", vec![]),
        ("not VOP3", vec![0x7e00_0000, 0]),
        (
            "unknown opcode 0x3ff",
            mutate(&add, |w| w[0] = (w[0] & !(0x3ff << 16)) | (0x3ff << 16)),
        ),
        (
            "v_swap_b32 (no 64-bit form)",
            mutate(&add, |w| w[0] = (w[0] & !(0x3ff << 16)) | (0x191 << 16)),
        ),
        (
            "v_readfirstlane_b32 (no 64-bit form)",
            mutate(&add, |w| w[0] = (w[0] & !(0x3ff << 16)) | (0x142 << 16)),
        ),
        (
            "unused src2 on a 2-source op",
            mutate(&add, |w| w[1] |= 0x100 << 18),
        ),
        (
            "unused src1 on a 1-source op",
            mutate(&mov, |w| w[1] |= 0x100 << 9),
        ),
        ("op_sel on v_add_f32", mutate(&add, |w| w[0] |= 1 << 11)),
        ("abs on an integer op", mutate(&add_u32, |w| w[0] |= 1 << 8)),
        (
            "neg on an integer op",
            mutate(&add_u32, |w| w[1] |= 1 << 29),
        ),
        ("neg on a missing source", mutate(&add, |w| w[1] |= 1 << 31)),
        (
            "omod on an integer op",
            mutate(&add_u32, |w| w[1] |= 1 << 27),
        ),
        ("clamp on v_bfe_u32", mutate(&bfe, |w| w[0] |= 1 << 15)),
        (
            "clamp on a compare of integers",
            mutate(&cmp, |w| w[0] |= 1 << 15),
        ),
        (
            "odd 64-bit compare destination",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 11),
        ),
        (
            "compare destination vcc_hi",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 107),
        ),
        (
            "compare destination m0",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 124),
        ),
        (
            "compare destination inline constant",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 128),
        ),
        (
            "compare destination src_scc",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 253),
        ),
        (
            "compare destination reserved code 125",
            mutate(&cmp, |w| w[0] = (w[0] & !0xff) | 125),
        ),
        (
            "literal source (code 255)",
            mutate(&add, |w| w[1] = (w[1] & !0x1ff) | 255),
        ),
        (
            "reserved source code 209",
            mutate(&add, |w| w[1] = (w[1] & !0x1ff) | 209),
        ),
        (
            "reserved source code 250 (DPP)",
            mutate(&add, |w| w[1] = (w[1] & !0x1ff) | 250),
        ),
        (
            "sdst reserved code 125",
            mutate(&scale, |w| w[0] = (w[0] & !(0x7f << 8)) | (125 << 8)),
        ),
        (
            "sdst odd",
            mutate(&scale, |w| w[0] = (w[0] & !(0x7f << 8)) | (13 << 8)),
        ),
        ("v_nop with a destination", mutate(&nop, |w| w[0] |= 1)),
        ("v_nop with a source", mutate(&nop, |w| w[1] |= 0x101)),
        (
            "constant-bus violation",
            mutate(&add, |w| w[1] = (w[1] & !0x3_ffff) | 1 | (2 << 9)),
        ),
    ];
    for (why, w) in cases {
        assert!(Vop3::decode(ARCH, &w).is_err(), "{why}: {w:08x?} decoded");
    }
}

// ---------------------------------------------------------------------------
// llvm-mc oracle
// ---------------------------------------------------------------------------

#[test]
fn llvm_round_trip_every_op_patterns_and_modifier_sweeps() {
    if oracle::disabled() {
        eprintln!("llvm_round_trip: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    let cases = all_cases();
    assert!(cases.len() > 2 * Vop3Op::all(ARCH).len());
    let lines: Vec<String> = cases.iter().map(asm_line).collect();
    let assembled = oracle::mc_encode(ARCH, &lines).expect("llvm-mc must assemble every pattern");
    let ours: Vec<Vec<u8>> = cases.iter().map(|c| oracle::bytes(&words(c))).collect();
    let disassembled =
        oracle::mc_disassemble(ARCH, &ours).expect("llvm-mc must disassemble every encoding");
    assert_eq!(assembled.len(), cases.len());
    assert_eq!(disassembled.len(), cases.len());
    for (n, case) in cases.iter().enumerate() {
        let text = case.to_string();
        assert_eq!(assembled[n].1, ours[n], "bytes of `{}`", lines[n]);
        assert_eq!(assembled[n].0, text, "llvm-mc print of `{}`", lines[n]);
        assert_eq!(
            disassembled[n], text,
            "llvm-mc disassembly of {:02x?}",
            ours[n]
        );
        let w: Vec<u32> = ours[n]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (decoded, len) =
            Vop3::decode(ARCH, &w).unwrap_or_else(|e| panic!("decode of `{text}`: {e}"));
        assert_eq!(len, 2);
        assert_eq!(&decoded, case, "decode of `{text}`");
    }
}

#[test]
fn llvm_assembles_each_opcode_to_the_table_opcode() {
    if oracle::disabled() {
        eprintln!("llvm_opcode_table: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    let first: Vec<Vop3> = Vop3Op::all(ARCH)
        .into_iter()
        .map(|op| patterns(op).swap_remove(0))
        .collect();
    let lines: Vec<String> = first.iter().map(asm_line).collect();
    let assembled = oracle::mc_encode(ARCH, &lines).unwrap();
    for ((inst, line), (_, bytes)) in first.iter().zip(&lines).zip(&assembled) {
        let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(lo >> 26, 0b11_0100, "`{line}`");
        assert_eq!(
            Some(u16::try_from((lo >> 16) & 0x3ff).unwrap()),
            inst.op.opcode(ARCH),
            "`{line}`"
        );
        assert_eq!(
            Vop3Op::from_opcode(ARCH, ((lo >> 16) & 0x3ff) as u16),
            Some(inst.op),
            "`{line}`"
        );
    }
}

#[test]
fn llvm_rejects_the_forms_the_encoder_rejects() {
    if oracle::disabled() {
        eprintln!("llvm_rejects: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    for (asm, inst) in llvm_rejected() {
        assert!(
            oracle::mc_encode(ARCH, &[asm.to_string()]).is_err(),
            "llvm-mc assembled `{asm}`"
        );
        assert!(
            inst.encode(ARCH, &mut Vec::new()).is_err(),
            "encoder accepted `{asm}`"
        );
    }
}

#[test]
fn llvm_disassembler_rejects_unsupported_modifier_bits() {
    if oracle::disabled() {
        eprintln!("llvm_rejects_bits: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    // (what, instruction, dword, bit to set)
    let cases: Vec<(&str, Vop3, usize, u32)> = vec![
        ("neg on v_add_u32", bin(ADD_U32, v(4), v(20), v(24)), 1, 29),
        (
            "neg on src2 of v_add_f32",
            bin(ADD_F32, v(4), v(20), v(24)),
            1,
            31,
        ),
        ("omod on v_add_u32", bin(ADD_U32, v(4), v(20), v(24)), 1, 27),
        (
            "clamp on v_cmp_lt_u32",
            bin(Vop3Op::C(VopcOp::CmpLtU32), sg(10), v(20), v(24)),
            0,
            15,
        ),
        (
            "clamp on v_mov_b32",
            new(MOV_B32, v(4), None, [Some(v(20)), None, None]),
            0,
            15,
        ),
        (
            "abs on v_mul_lo_u32",
            bin(Vop3Op::Only(Vop3OnlyOp::MulLoU32), v(4), v(20), v(24)),
            0,
            8,
        ),
        (
            "neg on v_bfe_u32",
            new(
                Vop3Op::Only(Vop3OnlyOp::BfeU32),
                v(4),
                None,
                [Some(v(20)), Some(v(24)), Some(v(28))],
            ),
            1,
            30,
        ),
        (
            "pk convert op_sel[2]",
            bin(Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32), v(4), v(20), v(24)),
            0,
            13,
        ),
        (
            "pk convert clamp",
            bin(Vop3Op::Only(Vop3OnlyOp::CvtPkFp8F32), v(4), v(20), v(24)),
            0,
            15,
        ),
    ];
    for (why, inst, dword, bit) in cases {
        let mut w = words(&inst);
        w[dword] |= 1 << bit;
        assert!(
            oracle::mc_disassemble(ARCH, &[oracle::bytes(&w)]).is_err(),
            "llvm decoded {why}"
        );
        assert!(Vop3::decode(ARCH, &w).is_err(), "decoder accepted {why}");
    }
}

#[test]
fn llvm_has_no_64_bit_form_of_the_excluded_opcodes() {
    if oracle::disabled() {
        eprintln!("llvm_excluded: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0)");
        return;
    }
    for asm in [
        "v_readfirstlane_b32_e64 s0, v1",
        "v_swap_b32_e64 v0, v1",
        "v_accvgpr_mov_b32_e64 a0, a1",
    ] {
        assert!(
            oracle::mc_encode(ARCH, &[asm.to_string()]).is_err(),
            "llvm-mc assembled `{asm}`"
        );
    }
    // llvm's disassembler does not decode the VOP3 encodings of those opcodes either.
    for opcode in [0x142u32, 0x191, 0x192] {
        let lo = (0b11_0100u32 << 26) | (opcode << 16) | 4;
        let hi = 256 + 20;
        assert!(
            oracle::mc_disassemble(ARCH, &[oracle::bytes(&[lo, hi])]).is_err(),
            "opcode {opcode:#x}"
        );
    }
}
