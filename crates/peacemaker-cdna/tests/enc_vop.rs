// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP1 / VOP2 / VOPC (e32, DPP, SDWA) round trip against llvm-mc gfx942:
//! every opcode with several operand patterns plus DPP and SDWA sweeps.
//! For each case: our bytes == llvm-mc's bytes for our text, llvm-mc's text ==
//! our text, llvm's disassembly of our bytes == our text, and decode(bytes)
//! == the instruction. Illegal forms are rejected.
use peacemaker_cdna::enc::vop::*;
use peacemaker_cdna::inst::{RegFile, RegSpan};
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::{decode_all, encode_all, oracle, Arch, Inst, Src};

const A: Arch = Arch::Gfx942;
use Src::*;

fn i1(op: Vop1Op, vdst: Src, src0: Src, ext: VopExt) -> Inst {
    Inst::Vop1(Vop1 {
        op,
        vdst,
        src0,
        ext,
    })
}
fn i2(op: Vop2Op, vdst: Src, src0: Src, src1: Src, ext: VopExt) -> Inst {
    Inst::Vop2(Vop2 {
        op,
        vdst,
        src0,
        src1,
        ext,
    })
}
fn ic(op: VopcOp, src0: Src, src1: Src, ext: VopExt) -> Inst {
    Inst::Vopc(Vopc {
        op,
        src0,
        src1,
        ext,
    })
}

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
fn sdwa() -> Sdwa {
    Sdwa {
        dst_sel: SdwaSel::Dword,
        dst_unused: SdwaUnused::Pad,
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

const SELS: [SdwaSel; 7] = [
    SdwaSel::Byte0,
    SdwaSel::Byte1,
    SdwaSel::Byte2,
    SdwaSel::Byte3,
    SdwaSel::Word0,
    SdwaSel::Word1,
    SdwaSel::Dword,
];
const UNUSED: [SdwaUnused; 3] = [SdwaUnused::Pad, SdwaUnused::Sext, SdwaUnused::Preserve];

fn all_ctrls() -> Vec<DppCtrl> {
    let mut v = vec![
        DppCtrl::QuadPerm([0, 1, 2, 3]),
        DppCtrl::QuadPerm([3, 2, 1, 0]),
        DppCtrl::QuadPerm([1, 0, 3, 2]),
        DppCtrl::QuadPerm([2, 2, 2, 2]),
        DppCtrl::WaveShl1,
        DppCtrl::WaveRol1,
        DppCtrl::WaveShr1,
        DppCtrl::WaveRor1,
        DppCtrl::RowMirror,
        DppCtrl::RowHalfMirror,
        DppCtrl::RowBcast15,
        DppCtrl::RowBcast31,
    ];
    for n in 1..=15 {
        v.extend([DppCtrl::RowShl(n), DppCtrl::RowShr(n), DppCtrl::RowRor(n)])
    }
    v
}

/// Non-foldable and foldable (printed `lit(…)`) literals per operand type.
fn literals_for_vop1(op: Vop1Op) -> [Src; 2] {
    match op {
        Vop1Op::MovB64 => [Lit(0x1234_5678), Lit(0x40)],
        Vop1Op::CvtF32F64 => [Lit(0x4008_0000), Lit(0x3ff0_0000)],
        Vop1Op::CvtF32F16 => [Lit(0x1234), Lit(0x3c00)],
        _ => [Lit(0x1234_5678), Lit(0x3f80_0000)],
    }
}

fn cases() -> Vec<Inst> {
    let n = VopExt::None;
    let mut c = Vec::new();
    let floats = InlineFloat::ALL;

    // ---- VOP1 e32: every op ----
    for (k, &op) in Vop1Op::ALL.iter().enumerate() {
        let f = Float(floats[k % floats.len()]);
        match op {
            Vop1Op::Nop => c.push(i1(op, Vgpr(0), Sgpr(0), n)),
            Vop1Op::ReadfirstlaneB32 => {
                for (d, s) in [
                    (Sgpr(0), Vgpr(0)),
                    (Sgpr(101), Vgpr(255)),
                    (VccLo, Vgpr(3)),
                    (VccHi, Vgpr(4)),
                    (ExecLo, Vgpr(5)),
                    (ExecHi, Vgpr(6)),
                    (Ttmp(3), Vgpr(7)),
                    (FlatScratchLo, Vgpr(8)),
                    (XnackMaskHi, Vgpr(9)),
                ] {
                    c.push(i1(op, d, s, n));
                }
            }
            Vop1Op::AccvgprMovB32 => {
                for (d, s) in [
                    (Agpr(1), Agpr(4)),
                    (Agpr(255), Agpr(0)),
                    (Agpr(0), Agpr(255)),
                ] {
                    c.push(i1(op, d, s, n))
                }
            }
            Vop1Op::SwapB32 => {
                for (d, s) in [(Vgpr(1), Vgpr(2)), (Vgpr(255), Vgpr(0)), (Vgpr(7), Vgpr(7))] {
                    c.push(i1(op, d, s, n))
                }
            }
            _ => {
                let (dw, sw) = op.widths();
                let vdst = if dw == 2 {
                    [Vgpr(2), Vgpr(254)]
                } else {
                    [Vgpr(1), Vgpr(255)]
                };
                let [lit, folded] = literals_for_vop1(op);
                let srcs: Vec<Src> = if sw == 2 {
                    vec![
                        Vgpr(4),
                        Vgpr(254),
                        Sgpr(100),
                        Sgpr(2),
                        VccLo,
                        ExecLo,
                        Ttmp(2),
                        FlatScratchLo,
                        XnackMaskLo,
                        Int(-1),
                        Int(64),
                        f,
                        SharedBase,
                        Scc,
                        lit,
                        folded,
                    ]
                } else {
                    vec![
                        Vgpr(4),
                        Vgpr(255),
                        Sgpr(7),
                        Sgpr(101),
                        VccHi,
                        M0,
                        ExecLo,
                        Ttmp(15),
                        Int(0),
                        Int(-16),
                        Int(64),
                        f,
                        PrivateLimit,
                        Execz,
                        lit,
                        folded,
                    ]
                };
                for (j, s) in srcs.into_iter().enumerate() {
                    c.push(i1(op, vdst[j % 2], s, n))
                }
            }
        }
    }

    // ---- VOP2 e32: every op ----
    for (k, &op) in Vop2Op::ALL.iter().enumerate() {
        let f = Float(floats[k % floats.len()]);
        let (lit, folded) = match op {
            Vop2Op::AddF16 | Vop2Op::SubF16 | Vop2Op::MulF16 | Vop2Op::MaxF16 | Vop2Op::MinF16 => {
                (Lit(0x1234), Lit(0xfff0))
            }
            Vop2Op::AddU16 | Vop2Op::SubU16 | Vop2Op::LshlrevB16 | Vop2Op::MulLoU16 => {
                (Lit(0xfff0), Lit(0x40))
            }
            _ => (Lit(0x1234_5678), Lit(0xffff_fff0)),
        };
        let srcs: Vec<Src> = if op.reads_vcc() {
            vec![Vgpr(2), Vgpr(255), Int(-7), Int(33), f]
        } else {
            vec![
                Vgpr(2),
                Vgpr(255),
                Sgpr(5),
                VccLo,
                M0,
                Ttmp(1),
                Int(-7),
                f,
                SharedBase,
                lit,
                folded,
            ]
        };
        for (j, s) in srcs.into_iter().enumerate() {
            c.push(i2(op, Vgpr(1 + j as u8), s, Vgpr(200 + j as u8), n))
        }
    }

    // ---- VOPC e32: every op ----
    for (k, &op) in VopcOp::ALL.iter().enumerate() {
        let f = Float(floats[k % floats.len()]);
        let srcs: Vec<(Src, Src)> = if op.src_dwords() == 2 {
            vec![
                (Vgpr(2), Vgpr(4)),
                (Sgpr(10), Vgpr(254)),
                (ExecLo, Vgpr(0)),
                (Int(5), Vgpr(6)),
                (f, Vgpr(8)),
                (Lit(0x3ff0_0000), Vgpr(10)),
                (Lit(64), Vgpr(12)),
            ]
        } else {
            vec![
                (Vgpr(2), Vgpr(3)),
                (Sgpr(10), Vgpr(255)),
                (VccHi, Vgpr(0)),
                (Int(-3), Vgpr(6)),
                (f, Vgpr(8)),
                (Lit(0xdead_beef), Vgpr(10)),
                (Lit(0x3e22_f983), Vgpr(12)),
            ]
        };
        for (s0, s1) in srcs {
            c.push(ic(op, s0, s1, n))
        }
    }

    // ---- DPP: one basic form for every op that has DPP ----
    let basic = VopExt::Dpp(dpp(DppCtrl::QuadPerm([1, 2, 3, 0])));
    for &op in Vop1Op::ALL {
        if matches!(
            op,
            Vop1Op::Nop
                | Vop1Op::ReadfirstlaneB32
                | Vop1Op::MovB64
                | Vop1Op::AccvgprMovB32
                | Vop1Op::SwapB32
                | Vop1Op::CvtF64F32
                | Vop1Op::CvtF32F64
                | Vop1Op::CvtPkF32Fp8
                | Vop1Op::CvtPkF32Bf8
        ) {
            continue;
        }
        c.push(i1(op, Vgpr(1), Vgpr(2), basic));
    }
    for &op in Vop2Op::ALL {
        c.push(i2(op, Vgpr(1), Vgpr(2), Vgpr(3), basic))
    }

    // DPP sweep: every control, masks, bound_ctrl.
    for (j, ctrl) in all_ctrls().into_iter().enumerate() {
        let j = j as u8;
        let d = Dpp {
            row_mask: j % 16,
            bank_mask: (j % 16 * 7) % 16,
            bound_ctrl: j % 2 == 1,
            ..dpp(ctrl)
        };
        c.push(i1(Vop1Op::MovB32, Vgpr(j), Vgpr(255 - j), VopExt::Dpp(d)));
        c.push(i2(
            Vop2Op::AddU32,
            Vgpr(j),
            Vgpr(100 + j),
            Vgpr(200),
            VopExt::Dpp(Dpp {
                bound_ctrl: !d.bound_ctrl,
                ..d
            }),
        ));
    }
    // neg/abs on float sources.
    for m in 0u8..16 {
        let d = Dpp {
            src0_neg: m & 1 != 0,
            src0_abs: m & 2 != 0,
            src1_neg: m & 4 != 0,
            src1_abs: m & 8 != 0,
            ..dpp(DppCtrl::RowRor(3))
        };
        c.push(i2(
            Vop2Op::AddF32,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Dpp(d),
        ));
        c.push(i2(
            Vop2Op::CndmaskB32,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Dpp(d),
        ));
        c.push(i2(
            Vop2Op::MulF16,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Dpp(d),
        ));
        if m < 4 {
            c.push(i1(Vop1Op::RcpF32, Vgpr(1), Vgpr(2), VopExt::Dpp(d)))
        }
    }
    c.push(i2(
        Vop2Op::FmacF32,
        Vgpr(1),
        Vgpr(2),
        Vgpr(3),
        VopExt::Dpp(Dpp {
            row_mask: 0xa,
            src0_neg: true,
            ..dpp(DppCtrl::RowBcast15)
        }),
    ));
    c.push(i2(
        Vop2Op::AddcCoU32,
        Vgpr(1),
        Vgpr(2),
        Vgpr(3),
        VopExt::Dpp(Dpp {
            bound_ctrl: true,
            ..dpp(DppCtrl::RowShr(1))
        }),
    ));
    c.push(i1(
        Vop1Op::CvtF32Fp8,
        Vgpr(1),
        Vgpr(2),
        VopExt::Dpp(dpp(DppCtrl::RowMirror)),
    ));

    // ---- SDWA: one basic form for every op that has SDWA ----
    let s = sdwa();
    for &op in Vop1Op::ALL {
        if matches!(
            op,
            Vop1Op::Nop
                | Vop1Op::ReadfirstlaneB32
                | Vop1Op::MovB64
                | Vop1Op::AccvgprMovB32
                | Vop1Op::SwapB32
                | Vop1Op::CvtF64F32
                | Vop1Op::CvtF32F64
        ) {
            continue;
        }
        let vdst = if op.widths().0 == 2 { Vgpr(2) } else { Vgpr(1) };
        c.push(i1(
            op,
            vdst,
            Vgpr(5),
            VopExt::Sdwa(Sdwa {
                src0_sel: SdwaSel::Byte1,
                ..s
            }),
        ));
        c.push(i1(op, vdst, Sgpr(5), VopExt::Sdwa(s)));
    }
    for &op in Vop2Op::ALL {
        if op == Vop2Op::FmacF32 {
            continue;
        }
        c.push(i2(
            op,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                src0_sel: SdwaSel::Word1,
                src1_sel: SdwaSel::Byte2,
                ..s
            }),
        ));
        let s0 = if op.reads_vcc() { Int(3) } else { Sgpr(9) };
        c.push(i2(op, Vgpr(1), s0, Vgpr(3), VopExt::Sdwa(s)));
    }
    for &op in VopcOp::ALL {
        if op.src_dwords() == 2 {
            continue;
        }
        c.push(ic(
            op,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                src0_sel: SdwaSel::Byte0,
                src1_sel: SdwaSel::Word0,
                ..s
            }),
        ));
        c.push(ic(
            op,
            Sgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                sdst: Some(Sgpr(4)),
                ..s
            }),
        ));
    }

    // SDWA sweep: every dst_sel × dst_unused × src0_sel on v_mov_b32.
    for (a, &ds) in SELS.iter().enumerate() {
        for &du in &UNUSED {
            for (b, &ss) in SELS.iter().enumerate() {
                c.push(i1(
                    Vop1Op::MovB32,
                    Vgpr(a as u8),
                    Vgpr(b as u8 + 10),
                    VopExt::Sdwa(Sdwa {
                        dst_sel: ds,
                        dst_unused: du,
                        src0_sel: ss,
                        src0_sext: b % 2 == 0,
                        ..s
                    }),
                ));
            }
        }
    }
    // src0_sel × src1_sel × sext on v_add_u32 / v_add_u16.
    for (a, &s0) in SELS.iter().enumerate() {
        for (b, &s1) in SELS.iter().enumerate() {
            let x = Sdwa {
                src0_sel: s0,
                src1_sel: s1,
                src0_sext: a % 2 == 1,
                src1_sext: b % 3 == 0,
                dst_sel: SELS[(a + b) % 7],
                dst_unused: UNUSED[(a * b) % 3],
                ..s
            };
            c.push(i2(
                Vop2Op::AddU32,
                Vgpr(1),
                Vgpr(2),
                Vgpr(3),
                VopExt::Sdwa(x),
            ));
            c.push(i2(
                Vop2Op::AddU16,
                Vgpr(1),
                Vgpr(2),
                Vgpr(3),
                VopExt::Sdwa(Sdwa { clamp: true, ..x }),
            ));
        }
    }
    // Scalar and constant sources (S0 / S1), incl. the same SGPR twice.
    for (s0, s1) in [
        (Sgpr(0), Vgpr(1)),
        (Vgpr(1), Sgpr(101)),
        (Sgpr(7), Sgpr(7)),
        (VccHi, Vgpr(2)),
        (M0, Vgpr(2)),
        (Vgpr(2), ExecLo),
        (Ttmp(9), Vgpr(2)),
        (FlatScratchLo, Vgpr(2)),
        (XnackMaskHi, Vgpr(2)),
        (SharedBase, Vgpr(2)),
        (Scc, Int(1)),
        (Vccz, Vgpr(2)),
        (Int(-16), Vgpr(2)),
        (Vgpr(2), Int(64)),
        (Float(InlineFloat::InvTwoPi), Sgpr(3)),
        (Float(InlineFloat::NegHalf), Float(InlineFloat::Four)),
    ] {
        c.push(i2(Vop2Op::AddF32, Vgpr(1), s0, s1, VopExt::Sdwa(s)));
        c.push(i2(
            Vop2Op::AndB32,
            Vgpr(1),
            s0,
            s1,
            VopExt::Sdwa(Sdwa {
                src0_sext: true,
                src1_sext: true,
                ..s
            }),
        ));
    }
    // Float modifiers (incl. on constants: neg(1.0), |1.0|) and output modifiers.
    for m in 0u8..16 {
        let x = Sdwa {
            src0_neg: m & 1 != 0,
            src0_abs: m & 2 != 0,
            src1_neg: m & 4 != 0,
            src1_abs: m & 8 != 0,
            ..s
        };
        c.push(i2(
            Vop2Op::MulF32,
            Vgpr(1),
            Vgpr(2),
            Sgpr(3),
            VopExt::Sdwa(x),
        ));
        c.push(i2(
            Vop2Op::SubF16,
            Vgpr(1),
            Float(InlineFloat::One),
            Vgpr(3),
            VopExt::Sdwa(x),
        ));
        c.push(i2(
            Vop2Op::MaxF32,
            Vgpr(1),
            Int(-2),
            Vgpr(3),
            VopExt::Sdwa(x),
        ));
        c.push(ic(VopcOp::CmpLtF32, Vgpr(2), Vgpr(3), VopExt::Sdwa(x)));
    }
    for omod in 0..4 {
        for clamp in [false, true] {
            let x = Sdwa { clamp, omod, ..s };
            c.push(i2(
                Vop2Op::AddF32,
                Vgpr(1),
                Vgpr(2),
                Vgpr(3),
                VopExt::Sdwa(x),
            ));
            c.push(i1(
                Vop1Op::CvtF16F32,
                Vgpr(1),
                Vgpr(2),
                VopExt::Sdwa(Sdwa {
                    src0_neg: true,
                    ..x
                }),
            ));
            c.push(i1(
                Vop1Op::CvtF32Fp8,
                Vgpr(1),
                Vgpr(2),
                VopExt::Sdwa(Sdwa {
                    src0_sel: SdwaSel::Byte2,
                    ..x
                }),
            ));
        }
    }
    c.push(i1(
        Vop1Op::CvtI32F32,
        Vgpr(1),
        Vgpr(2),
        VopExt::Sdwa(Sdwa {
            clamp: true,
            src0_abs: true,
            ..s
        }),
    ));
    // FP8 / BF8 converts: every select, scalar and constant sources.
    for (j, &sel) in SELS.iter().enumerate() {
        for op in [
            Vop1Op::CvtF32Fp8,
            Vop1Op::CvtF32Bf8,
            Vop1Op::CvtPkF32Fp8,
            Vop1Op::CvtPkF32Bf8,
        ] {
            let vdst = if op.widths().0 == 2 {
                Vgpr(2 * j as u8)
            } else {
                Vgpr(j as u8)
            };
            let src = [
                Vgpr(40),
                Sgpr(11),
                Int(1),
                Float(InlineFloat::Two),
                Vgpr(255),
                M0,
                Ttmp(4),
            ][j];
            c.push(i1(op, vdst, src, VopExt::Sdwa(Sdwa { src0_sel: sel, ..s })));
        }
    }
    // VOPC: SDWA destinations, sext, class.
    for (j, sdst) in [
        None,
        Some(Sgpr(0)),
        Some(Sgpr(100)),
        Some(ExecLo),
        Some(Ttmp(14)),
        Some(FlatScratchLo),
        Some(XnackMaskLo),
    ]
    .into_iter()
    .enumerate()
    {
        c.push(ic(
            VopcOp::CmpEqU32,
            Vgpr(j as u8),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                sdst,
                src0_sext: true,
                src0_sel: SELS[j],
                ..s
            }),
        ));
        c.push(ic(
            VopcOp::CmpxGtI32,
            Sgpr(6),
            Sgpr(6),
            VopExt::Sdwa(Sdwa {
                sdst,
                src1_sext: true,
                src1_sel: SELS[6 - j],
                ..s
            }),
        ));
    }
    c.push(ic(
        VopcOp::CmpClassF32,
        Vgpr(2),
        Vgpr(3),
        VopExt::Sdwa(Sdwa {
            src0_neg: true,
            src0_abs: true,
            src1_sext: true,
            ..s
        }),
    ));
    c.push(ic(
        VopcOp::CmpxLtF32,
        VccLo,
        Vgpr(3),
        VopExt::Sdwa(Sdwa {
            src1_abs: true,
            ..s
        }),
    ));
    c
}

#[test]
fn round_trip_against_llvm() {
    let cases = cases();
    let mut texts = Vec::with_capacity(cases.len());
    let mut bytes = Vec::with_capacity(cases.len());
    for inst in &cases {
        let words = inst
            .words(A)
            .unwrap_or_else(|e| panic!("encode {inst:?}: {e}"));
        let (back, n) = Inst::decode(A, &words)
            .unwrap_or_else(|e| panic!("decode {inst:?} {words:#010x?}: {e}"));
        assert_eq!(n, words.len(), "{inst:?}");
        assert_eq!(&back, inst, "decode identity for {words:#010x?}");
        texts.push(inst.to_string());
        bytes.push(oracle::bytes(&words));
    }
    // The concatenated stream decodes back too.
    let stream = encode_all(A, &cases).unwrap();
    let decoded: Vec<Inst> = decode_all(A, &stream)
        .unwrap()
        .into_iter()
        .map(|(_, i)| i)
        .collect();
    assert_eq!(decoded, cases);
    eprintln!("enc_vop: {} instructions", cases.len());

    if oracle::disabled() {
        eprintln!(
            "enc_vop: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0); llvm comparison skipped"
        );
        return;
    }
    let mc = oracle::mc_encode(A, &texts).unwrap();
    let mut bad = Vec::new();
    for (i, (text, enc)) in mc.iter().enumerate() {
        if text != &texts[i] || enc != &bytes[i] {
            bad.push(format!(
                "{:?}\n  ours: {} {:02x?}\n  llvm: {} {:02x?}",
                cases[i], texts[i], bytes[i], text, enc
            ));
        }
    }
    let dis = oracle::mc_disassemble(A, &bytes).unwrap();
    for (i, d) in dis.iter().enumerate() {
        if d != &texts[i] {
            bad.push(format!(
                "{:?}\n  ours: {}\n  objdump: {}",
                cases[i], texts[i], d
            ))
        }
    }
    assert!(
        bad.is_empty(),
        "{} mismatches:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[test]
fn illegal_forms_rejected() {
    let n = VopExt::None;
    let s = sdwa();
    let d = dpp(DppCtrl::QuadPerm([0, 1, 2, 3]));
    let bad = [
        // e32
        i1(Vop1Op::MovB32, Sgpr(1), Vgpr(2), n),
        i1(Vop1Op::MovB32, Vgpr(1), Agpr(2), n),
        i1(Vop1Op::MovB32, Vgpr(1), Sgpr(102), n),
        i1(Vop1Op::MovB32, Vgpr(1), Int(65), n),
        i1(Vop1Op::MovB32, Vgpr(1), Ttmp(16), n),
        i1(Vop1Op::MovB64, Vgpr(3), Vgpr(4), n),
        i1(Vop1Op::MovB64, Vgpr(2), Vgpr(5), n),
        i1(Vop1Op::MovB64, Vgpr(2), Sgpr(3), n),
        i1(Vop1Op::MovB64, Vgpr(2), Sgpr(101), n),
        i1(Vop1Op::MovB64, Vgpr(2), VccHi, n),
        i1(Vop1Op::MovB64, Vgpr(2), M0, n),
        i1(Vop1Op::CvtPkF32Fp8, Vgpr(3), Vgpr(2), n),
        i1(Vop1Op::CvtF32F64, Vgpr(1), Lit(64), n),
        i1(Vop1Op::CvtF32F16, Vgpr(1), Lit(0x1_0000), n),
        i1(Vop1Op::Nop, Vgpr(1), Sgpr(0), n),
        i1(Vop1Op::Nop, Vgpr(0), Sgpr(0), VopExt::Dpp(d)),
        i1(Vop1Op::ReadfirstlaneB32, M0, Vgpr(1), n),
        i1(Vop1Op::ReadfirstlaneB32, Vgpr(1), Vgpr(1), n),
        i1(Vop1Op::ReadfirstlaneB32, Sgpr(1), Sgpr(2), n),
        i1(Vop1Op::ReadfirstlaneB32, Sgpr(1), Vgpr(2), VopExt::Sdwa(s)),
        i1(Vop1Op::AccvgprMovB32, Agpr(1), Vgpr(2), n),
        i1(Vop1Op::AccvgprMovB32, Vgpr(1), Agpr(2), n),
        i1(Vop1Op::AccvgprMovB32, Agpr(1), Agpr(2), VopExt::Dpp(d)),
        i1(Vop1Op::SwapB32, Vgpr(1), Sgpr(2), n),
        i2(Vop2Op::AddF32, Vgpr(1), Vgpr(2), Sgpr(3), n),
        i2(Vop2Op::AddU16, Vgpr(1), Lit(0x1_0000), Vgpr(3), n),
        i2(Vop2Op::CndmaskB32, Vgpr(1), Sgpr(2), Vgpr(3), n),
        i2(Vop2Op::CndmaskB32, Vgpr(1), Lit(0x1234_5678), Vgpr(3), n),
        i2(Vop2Op::AddcCoU32, Vgpr(1), VccLo, Vgpr(3), n),
        i2(Vop2Op::SubbCoU32, Vgpr(1), Scc, Vgpr(3), n),
        ic(VopcOp::CmpEqU32, Vgpr(2), Sgpr(3), n),
        ic(VopcOp::CmpEqU64, Vgpr(3), Vgpr(4), n),
        ic(VopcOp::CmpEqU64, Vgpr(2), Vgpr(5), n),
        // DPP
        i1(Vop1Op::MovB32, Vgpr(1), Sgpr(2), VopExt::Dpp(d)),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(Dpp {
                src0_neg: true,
                ..d
            }),
        ),
        i1(
            Vop1Op::RcpF32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(Dpp {
                src1_abs: true,
                ..d
            }),
        ),
        i1(
            Vop1Op::CvtF32Fp8,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(Dpp {
                src0_abs: true,
                ..d
            }),
        ),
        i1(Vop1Op::MovB64, Vgpr(2), Vgpr(4), VopExt::Dpp(d)),
        i1(Vop1Op::CvtPkF32Fp8, Vgpr(2), Vgpr(4), VopExt::Dpp(d)),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(dpp(DppCtrl::QuadPerm([0, 1, 4, 3]))),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(dpp(DppCtrl::RowShl(0))),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(dpp(DppCtrl::RowShr(16))),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(Dpp {
                row_mask: 0x10,
                ..d
            }),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Dpp(Dpp {
                bank_mask: 0x1f,
                ..d
            }),
        ),
        i2(
            Vop2Op::AddU32,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Dpp(Dpp {
                src1_neg: true,
                ..d
            }),
        ),
        i2(Vop2Op::AddF32, Vgpr(1), Vgpr(2), Sgpr(3), VopExt::Dpp(d)),
        i2(Vop2Op::AddF32, Vgpr(1), Lit(5), Vgpr(3), VopExt::Dpp(d)),
        ic(VopcOp::CmpEqU32, Vgpr(2), Vgpr(3), VopExt::Dpp(d)),
        // SDWA
        i1(Vop1Op::MovB32, Vgpr(1), Lit(0x1234), VopExt::Sdwa(s)),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                src0_neg: true,
                ..s
            }),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa { omod: 1, ..s }),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                src1_sel: SdwaSel::Byte0,
                ..s
            }),
        ),
        i1(
            Vop1Op::MovB32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                sdst: Some(Sgpr(4)),
                ..s
            }),
        ),
        i1(
            Vop1Op::RcpF32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                src0_sext: true,
                ..s
            }),
        ),
        i1(
            Vop1Op::RcpF32,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa { omod: 4, ..s }),
        ),
        i1(
            Vop1Op::CvtF32Fp8,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                dst_sel: SdwaSel::Word0,
                ..s
            }),
        ),
        i1(
            Vop1Op::CvtF32Fp8,
            Vgpr(1),
            Vgpr(2),
            VopExt::Sdwa(Sdwa {
                src0_sext: true,
                ..s
            }),
        ),
        i1(Vop1Op::MovB64, Vgpr(2), Vgpr(4), VopExt::Sdwa(s)),
        i2(Vop2Op::FmacF32, Vgpr(1), Vgpr(2), Vgpr(3), VopExt::Sdwa(s)),
        i2(Vop2Op::AddF32, Vgpr(1), Sgpr(2), Sgpr(3), VopExt::Sdwa(s)),
        i2(Vop2Op::AddF32, Vgpr(1), Sgpr(2), Scc, VopExt::Sdwa(s)),
        i2(
            Vop2Op::CndmaskB32,
            Vgpr(1),
            Sgpr(2),
            Vgpr(3),
            VopExt::Sdwa(s),
        ),
        i2(
            Vop2Op::AddU32,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa { omod: 2, ..s }),
        ),
        i2(
            Vop2Op::AddU32,
            Vgpr(1),
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                src1_abs: true,
                ..s
            }),
        ),
        i2(Vop2Op::AddF32, Vgpr(1), Vgpr(2), Agpr(3), VopExt::Sdwa(s)),
        ic(VopcOp::CmpEqU32, Sgpr(2), Sgpr(3), VopExt::Sdwa(s)),
        ic(
            VopcOp::CmpEqU32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                sdst: Some(VccLo),
                ..s
            }),
        ),
        ic(
            VopcOp::CmpEqU32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                sdst: Some(Sgpr(3)),
                ..s
            }),
        ),
        ic(
            VopcOp::CmpEqU32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                sdst: Some(M0),
                ..s
            }),
        ),
        ic(
            VopcOp::CmpEqU32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa { clamp: true, ..s }),
        ),
        ic(
            VopcOp::CmpEqU32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                dst_sel: SdwaSel::Byte0,
                ..s
            }),
        ),
        ic(
            VopcOp::CmpClassF32,
            Vgpr(2),
            Vgpr(3),
            VopExt::Sdwa(Sdwa {
                src1_neg: true,
                ..s
            }),
        ),
        ic(VopcOp::CmpEqU64, Vgpr(2), Vgpr(4), VopExt::Sdwa(s)),
    ];
    for inst in &bad {
        assert!(
            inst.words(A).is_err(),
            "accepted illegal {inst:?} as {:#010x?}",
            inst.words(A)
        );
    }

    let reject = [
        vec![0x7e02_02fa, 0xff01_0002], // DPP_CTRL 0x100 (reserved)
        vec![0x7e02_02fa, 0xff01_5002], // row_newbcast (not modelled)
        vec![0x7e02_02fa, 0xff06_e402], // DPP reserved bits [50:49]
        vec![0x7e02_02f9, 0x0007_0602], // SDWA src0_sel 7
        vec![0x7e02_02f9, 0x0006_1e02], // DST_UNUSED 3
        vec![0x7e02_02f9, 0x0606_0602], // VOP1 SDWA with src1 fields
        vec![0x7e02_02f9, 0x0016_0602], // v_mov_b32 SDWA neg (int source)
        vec![0x7e02_a8f9, 0x0003_0002], // fp8 SDWA dst_sel BYTE_0
        vec![0x7d94_06f9, 0x0400_6a02], // VOPC SDWA SD=0 with SDST bits
        vec![0x7d94_06f9, 0x0400_ea02], // VOPC SDWA SD=1 SDST=VCC (non-canonical)
        vec![0x7d94_06fa, 0xff00_e402], // VOPC DPP
        vec![0x7e00_0001],              // v_nop with a nonzero field
        vec![0x7ef8_0502],              // v_readfirstlane_b32 m0
        vec![0x7e02_c800],              // VOP1 opcode 100: not in table
        vec![0x7e02_02ff],              // literal missing
        vec![0x0002_0702 | 4 << 25],    // VOP2 opcode 4: not in table
    ];
    for words in &reject {
        assert!(
            Inst::decode(A, words).is_err(),
            "decoded {words:#010x?} as {:?}",
            Inst::decode(A, words)
        );
    }
}

#[test]
fn op_properties() {
    assert_eq!(Vop1Op::Nop.widths(), (0, 0));
    assert_eq!(Vop1Op::MovB64.widths(), (2, 2));
    assert_eq!(Vop1Op::CvtPkF32Fp8.widths(), (2, 1));
    assert_eq!(Vop1Op::CvtF32F64.widths(), (1, 2));
    assert_eq!(Vop1Op::CvtF32Fp8.widths(), (1, 1));
    let reads: Vec<_> = Vop2Op::ALL
        .iter()
        .copied()
        .filter(|o| o.reads_vcc())
        .collect();
    assert_eq!(
        reads,
        [
            Vop2Op::CndmaskB32,
            Vop2Op::AddcCoU32,
            Vop2Op::SubbCoU32,
            Vop2Op::SubbrevCoU32
        ]
    );
    assert_eq!(Vop2Op::ALL.iter().filter(|o| o.writes_vcc()).count(), 6);
    assert!(!Vop2Op::CndmaskB32.writes_vcc());
    assert_eq!(VopcOp::ALL.iter().filter(|o| o.is_cmpx()).count(), 8);
    for &op in VopcOp::ALL {
        assert_eq!(op.is_cmpx(), op.name().starts_with("v_cmpx_"), "{op:?}")
    }
    for &op in VopcOp::ALL {
        assert_eq!(
            op.src_dwords(),
            if op.name().ends_with("64") { 2 } else { 1 },
            "{op:?}"
        )
    }
}

fn has(spans: &[RegSpan], file: RegFile, base: u8, len: u8) -> bool {
    spans.contains(&RegSpan::new(file, base, len))
}

#[test]
fn footprints() {
    let n = VopExt::None;
    let exec = |f: &peacemaker_cdna::inst::Footprint| has(&f.uses, RegFile::Exec, 0, 2);

    let f = i1(Vop1Op::Nop, Vgpr(0), Sgpr(0), n).footprint();
    assert!(f.uses.is_empty() && f.defs.is_empty());

    let f = i1(Vop1Op::MovB64, Vgpr(2), Sgpr(4), n).footprint();
    assert!(exec(&f) && has(&f.uses, RegFile::Sgpr, 4, 2) && has(&f.defs, RegFile::Vgpr, 2, 2));

    let f = i1(Vop1Op::ReadfirstlaneB32, Sgpr(7), Vgpr(3), n).footprint();
    assert!(exec(&f) && has(&f.defs, RegFile::Sgpr, 7, 1) && has(&f.uses, RegFile::Vgpr, 3, 1));

    let f = i1(Vop1Op::AccvgprMovB32, Agpr(1), Agpr(2), n).footprint();
    assert!(has(&f.defs, RegFile::Agpr, 1, 1) && has(&f.uses, RegFile::Agpr, 2, 1));

    let f = i1(Vop1Op::SwapB32, Vgpr(1), Vgpr(2), n).footprint();
    for r in [1, 2] {
        assert!(has(&f.defs, RegFile::Vgpr, r, 1) && has(&f.uses, RegFile::Vgpr, r, 1))
    }

    let f = i1(
        Vop1Op::MovB32,
        Vgpr(1),
        Vgpr(2),
        VopExt::Sdwa(Sdwa {
            dst_sel: SdwaSel::Word1,
            dst_unused: SdwaUnused::Preserve,
            ..sdwa()
        }),
    )
    .footprint();
    assert!(
        has(&f.uses, RegFile::Vgpr, 1, 1),
        "UNUSED_PRESERVE reads vdst"
    );

    let f = i1(Vop1Op::MovB32, Vgpr(1), Lit(5), n).footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Exec, 0, 2)]);

    let f = i2(Vop2Op::CndmaskB32, Vgpr(1), Vgpr(2), Vgpr(3), n).footprint();
    assert!(
        exec(&f)
            && has(&f.uses, RegFile::Vcc, 0, 2)
            && !f.defs.iter().any(|r| r.file == RegFile::Vcc)
    );

    let f = i2(Vop2Op::AddCoU32, Vgpr(1), Sgpr(2), Vgpr(3), n).footprint();
    assert!(
        has(&f.defs, RegFile::Vcc, 0, 2)
            && !has(&f.uses, RegFile::Vcc, 0, 2)
            && has(&f.uses, RegFile::Sgpr, 2, 1)
    );

    let f = i2(Vop2Op::AddcCoU32, Vgpr(1), Vgpr(2), Vgpr(3), n).footprint();
    assert!(has(&f.defs, RegFile::Vcc, 0, 2) && has(&f.uses, RegFile::Vcc, 0, 2));

    let f = i2(Vop2Op::FmacF32, Vgpr(1), Vgpr(2), Vgpr(3), n).footprint();
    assert!(has(&f.uses, RegFile::Vgpr, 1, 1) && has(&f.defs, RegFile::Vgpr, 1, 1));

    let f = ic(VopcOp::CmpEqU64, Sgpr(4), Vgpr(6), n).footprint();
    assert!(
        exec(&f)
            && has(&f.uses, RegFile::Sgpr, 4, 2)
            && has(&f.uses, RegFile::Vgpr, 6, 2)
            && has(&f.defs, RegFile::Vcc, 0, 2)
    );

    let f = ic(VopcOp::CmpxEqU32, Vgpr(2), Vgpr(3), n).footprint();
    assert!(has(&f.defs, RegFile::Vcc, 0, 2) && has(&f.defs, RegFile::Exec, 0, 2));

    let f = ic(
        VopcOp::CmpxEqU32,
        Vgpr(2),
        Vgpr(3),
        VopExt::Sdwa(Sdwa {
            sdst: Some(Sgpr(4)),
            ..sdwa()
        }),
    )
    .footprint();
    assert!(
        has(&f.defs, RegFile::Sgpr, 4, 2)
            && has(&f.defs, RegFile::Exec, 0, 2)
            && !has(&f.defs, RegFile::Vcc, 0, 2)
    );
}
