// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SOP1/SOP2/SOPK/SOPC/SOPP encoder, decoder and printer against llvm-mc
//! (gfx942): for every opcode of the five tables, several operand patterns
//! (SGPRs, pairs, special registers, inline integers and floats, literals
//! where legal). One batched `mc_encode` + one `mc_disassemble` per test.
use peacemaker_cdna::enc::sop::{
    waitcnt_fields, waitcnt_imm, Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp,
    SoppOp,
};
use peacemaker_cdna::inst::{RegFile, RegSpan};
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::{oracle, Arch, Inst, Src};

const ARCH: Arch = Arch::Gfx942;

/// Sources for a `dwords`-wide scalar operand: register kinds, inline
/// constants and literals (including one llvm prints as `lit(...)`).
fn sources(dwords: u8) -> Vec<Src> {
    if dwords == 2 {
        vec![
            Src::Sgpr(4),
            Src::Sgpr(100),
            Src::VccLo,
            Src::ExecLo,
            Src::FlatScratchLo,
            Src::XnackMaskLo,
            Src::Ttmp(2),
            Src::Int(-16),
            Src::Int(64),
            Src::Int(0),
            Src::Float(InlineFloat::NegFour),
            Src::Float(InlineFloat::InvTwoPi),
            Src::SharedBase,
            Src::PrivateLimit,
            Src::Vccz,
            Src::Execz,
            Src::Scc,
            Src::Lit(0x1234_5678),
            Src::Lit(0xffff_fff0),
            Src::Lit(5),
            Src::Lit(0x3ff0_0000),
        ]
    } else {
        vec![
            Src::Sgpr(3),
            Src::Sgpr(101),
            Src::VccHi,
            Src::ExecLo,
            Src::M0,
            Src::Ttmp(15),
            Src::FlatScratchHi,
            Src::XnackMaskLo,
            Src::Int(-7),
            Src::Int(33),
            Src::Float(InlineFloat::Half),
            Src::Float(InlineFloat::InvTwoPi),
            Src::SharedLimit,
            Src::PrivateBase,
            Src::Vccz,
            Src::Scc,
            Src::Lit(0x1234_5678),
            Src::Lit(0x3f80_0000),
            Src::Lit(0xffff_fff0),
            Src::Lit(0x41),
        ]
    }
}

/// Destinations for a `dwords`-wide SDST field.
fn dests(dwords: u8) -> Vec<Src> {
    if dwords == 2 {
        vec![
            Src::Sgpr(0),
            Src::Sgpr(100),
            Src::VccLo,
            Src::ExecLo,
            Src::Ttmp(14),
            Src::FlatScratchLo,
        ]
    } else {
        vec![
            Src::Sgpr(0),
            Src::Sgpr(101),
            Src::VccLo,
            Src::ExecHi,
            Src::M0,
            Src::Ttmp(7),
        ]
    }
}

/// Every pattern of the five formats. Each opcode gets at least as many
/// patterns as there are sources for its width.
fn all_patterns() -> Vec<Inst> {
    let mut out = Vec::new();
    for &op in Sop2Op::ALL {
        let (wd, w0, w1) = op.widths();
        let (s0, s1, d) = (sources(w0), sources(w1), dests(wd));
        for (i, &a) in s0.iter().enumerate() {
            // Pair each first source with a register second source, then a
            // constant/literal second source with a register first source.
            let b = s1[(i * 5 + 1) % s1.len()];
            let b = if a.is_literal() && b.is_literal() {
                s1[0]
            } else {
                b
            };
            out.push(Inst::Sop2(Sop2 {
                op,
                sdst: d[i % d.len()],
                ssrc0: a,
                ssrc1: b,
            }));
            out.push(Inst::Sop2(Sop2 {
                op,
                sdst: d[(i + 1) % d.len()],
                ssrc0: s0[0],
                ssrc1: s1[i % s1.len()],
            }));
        }
        // Both sources naming the one literal dword.
        out.push(Inst::Sop2(Sop2 {
            op,
            sdst: d[0],
            ssrc0: Src::Lit(0xdead_beef),
            ssrc1: Src::Lit(0xdead_beef),
        }));
    }
    for &op in Sop1Op::ALL {
        let (wd, w0) = op.widths();
        if w0 == 0 {
            for d in dests(wd) {
                out.push(Inst::Sop1(Sop1 {
                    op,
                    sdst: Some(d),
                    ssrc0: None,
                }))
            }
        } else if wd == 0 {
            // s_setpc_b64: register pair sources only (llvm's SReg_64).
            for s in sources(w0).into_iter().filter(|s| s.is_sreg()) {
                out.push(Inst::Sop1(Sop1 {
                    op,
                    sdst: None,
                    ssrc0: Some(s),
                }))
            }
        } else {
            let d = dests(wd);
            for (i, s) in sources(w0).into_iter().enumerate() {
                out.push(Inst::Sop1(Sop1 {
                    op,
                    sdst: Some(d[i % d.len()]),
                    ssrc0: Some(s),
                }))
            }
        }
    }
    let simm = [0u16, 1, 0x40, 0x41, 0x7fff, 0x8000, 0xfffb, 0xffff, 0x1234];
    for &op in SopkOp::ALL {
        let d = dests(1);
        let imms: Vec<u16> = match op {
            // hwreg ids with and without names, default and explicit offset/size.
            SopkOp::GetregB32 | SopkOp::SetregB32 => vec![
                0xf801, 0x1881, 0xf819, 0xffff, 0x0000, 0xf802, 0x0814, 0xf818, 0x7c7, 0x003f,
            ],
            _ => simm.to_vec(),
        };
        for (i, imm) in imms.into_iter().enumerate() {
            out.push(Inst::Sopk(Sopk {
                op,
                sdst: d[i % d.len()],
                simm16: imm,
            }))
        }
    }
    for &op in SopcOp::ALL {
        let s = sources(op.width());
        for (i, &a) in s.iter().enumerate() {
            let b = s[(i * 7 + 3) % s.len()];
            let b = if a.is_literal() && b.is_literal() {
                s[0]
            } else {
                b
            };
            out.push(Inst::Sopc(Sopc {
                op,
                ssrc0: a,
                ssrc1: b,
            }));
            out.push(Inst::Sopc(Sopc {
                op,
                ssrc0: s[1],
                ssrc1: a,
            }));
        }
    }
    for &op in SoppOp::ALL {
        let imms: Vec<u16> = match op {
            SoppOp::Barrier | SoppOp::IcacheInv => vec![0],
            SoppOp::Waitcnt => vec![
                waitcnt_imm(ARCH, 0, 0, 0).unwrap(),
                waitcnt_imm(ARCH, 63, 7, 15).unwrap(),
                waitcnt_imm(ARCH, 5, 7, 15).unwrap(),
                waitcnt_imm(ARCH, 63, 3, 15).unwrap(),
                waitcnt_imm(ARCH, 63, 7, 2).unwrap(),
                waitcnt_imm(ARCH, 17, 7, 3).unwrap(),
                waitcnt_imm(ARCH, 48, 0, 15).unwrap(),
            ],
            SoppOp::Sendmsg => vec![
                0x1, 0x2, 0x3, 0x22, 0x112, 0x1f, 0x2f, 0x4f, 0x9, 0xa, 0x0, 0x3f, 0x80, 0x1234,
            ],
            _ => vec![0, 1, 5, 64, 65, 0x7fff, 0x8000, 0xfffd, 0xffff],
        };
        for simm16 in imms {
            out.push(Inst::Sopp(Sopp { op, simm16 }))
        }
    }
    out
}

#[test]
fn llvm_mc_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_sop: HIPFIRE_TEST_REQUIRE_ROCM=0, skipping the llvm-mc oracle");
        return;
    }
    let insts = all_patterns();
    // Coverage: every opcode of every table has at least three patterns.
    let count = |f: &dyn Fn(&Inst) -> bool| insts.iter().filter(|i| f(i)).count();
    for &op in Sop2Op::ALL {
        assert!(
            count(&|i| matches!(i, Inst::Sop2(x) if x.op == op)) >= 3,
            "{op:?}"
        )
    }
    for &op in Sop1Op::ALL {
        assert!(
            count(&|i| matches!(i, Inst::Sop1(x) if x.op == op)) >= 3,
            "{op:?}"
        )
    }
    for &op in SopkOp::ALL {
        assert!(
            count(&|i| matches!(i, Inst::Sopk(x) if x.op == op)) >= 3,
            "{op:?}"
        )
    }
    for &op in SopcOp::ALL {
        assert!(
            count(&|i| matches!(i, Inst::Sopc(x) if x.op == op)) >= 3,
            "{op:?}"
        )
    }
    for &op in SoppOp::ALL {
        let need = if matches!(op, SoppOp::Barrier | SoppOp::IcacheInv) {
            1
        } else {
            3
        };
        assert!(
            count(&|i| matches!(i, Inst::Sopp(x) if x.op == op)) >= need,
            "{op:?}"
        )
    }

    let words: Vec<Vec<u32>> = insts
        .iter()
        .map(|i| i.words(ARCH).unwrap_or_else(|e| panic!("{i:?}: {e}")))
        .collect();
    let texts: Vec<String> = insts.iter().map(|i| i.to_string()).collect();
    let encoded =
        oracle::mc_encode(ARCH, &texts).expect("llvm-mc assembles every printed instruction");
    let bytes: Vec<Vec<u8>> = words.iter().map(|w| oracle::bytes(w)).collect();
    let disassembled =
        oracle::mc_disassemble(ARCH, &bytes).expect("llvm-mc disassembles every encoding");
    let mut failures = Vec::new();
    for (k, inst) in insts.iter().enumerate() {
        let (llvm_text, llvm_bytes) = &encoded[k];
        if *llvm_bytes != bytes[k] {
            failures.push(format!(
                "{}: bytes {:02x?} vs llvm {:02x?}",
                texts[k], bytes[k], llvm_bytes
            ))
        }
        if disassembled[k] != texts[k] {
            failures.push(format!(
                "{inst:?}: text {:?} vs llvm {:?}",
                texts[k], disassembled[k]
            ))
        }
        if oracle::normalize(llvm_text) != texts[k] {
            failures.push(format!(
                "{inst:?}: text {:?} vs llvm-mc echo {:?}",
                texts[k], llvm_text
            ))
        }
        let (back, n) =
            Inst::decode(ARCH, &words[k]).unwrap_or_else(|e| panic!("{}: decode: {e}", texts[k]));
        if back != *inst || n != words[k].len() {
            failures.push(format!(
                "{}: decoded {back:?} ({n} dwords of {})",
                texts[k],
                words[k].len()
            ))
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches of {}:\n{}",
        failures.len(),
        insts.len(),
        failures.join("\n")
    );
    eprintln!("enc_sop: {} instructions matched llvm-mc", insts.len());
}

/// Every S_SENDMSG and S_GETREG SIMM16: the printer matches llvm's
/// disassembler wherever the encoder accepts the value, and the accepted
/// values reassemble to the same bytes; rejected sendmsg values are exactly
/// those whose llvm text drops bits.
#[test]
fn llvm_mc_sendmsg_hwreg_exhaustive() {
    if oracle::disabled() {
        eprintln!("enc_sop: HIPFIRE_TEST_REQUIRE_ROCM=0, skipping the llvm-mc oracle");
        return;
    }
    let mut insts = Vec::new();
    for v in 0..=u16::MAX {
        insts.push(Inst::Sopp(Sopp {
            op: SoppOp::Sendmsg,
            simm16: v,
        }));
        insts.push(Inst::Sopk(Sopk {
            op: SopkOp::GetregB32,
            sdst: Src::Sgpr(0),
            simm16: v,
        }));
    }
    // Disassemble the raw words, whether or not the encoder accepts them.
    let raw: Vec<Vec<u8>> = (0..=u16::MAX)
        .flat_map(|v| [0xbf90_0000u32 | u32::from(v), 0xb880_0000 | u32::from(v)])
        .map(|w| oracle::bytes(&[w]))
        .collect();
    let llvm = oracle::mc_disassemble(ARCH, &raw).expect("llvm-mc disassembles every SIMM16");
    let mut accepted = Vec::new();
    let mut failures = Vec::new();
    for (k, inst) in insts.iter().enumerate() {
        match inst.words(ARCH) {
            Ok(w) => {
                let text = inst.to_string();
                if text != llvm[k] {
                    failures.push(format!("{inst:?}: {text:?} vs llvm {:?}", llvm[k]))
                }
                accepted.push((text, oracle::bytes(&w)));
            }
            Err(_) => {
                // Only sendmsg values whose fields re-encode differently are rejected.
                let Inst::Sopp(p) = inst else {
                    panic!("{inst:?} rejected")
                };
                let (id, op, stream) = (p.simm16 & 0xf, (p.simm16 >> 4) & 7, (p.simm16 >> 8) & 3);
                assert_ne!(
                    id | op << 4 | stream << 8,
                    p.simm16,
                    "{inst:?} rejected although its fields are exact"
                );
                assert!(
                    llvm[k].starts_with("s_sendmsg sendmsg(MSG_"),
                    "{inst:?} rejected but llvm prints {:?}",
                    llvm[k]
                );
                assert!(Sopp::decode(ARCH, &[0xbf90_0000 | u32::from(p.simm16)]).is_err());
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    let texts: Vec<String> = accepted.iter().map(|(t, _)| t.clone()).collect();
    let encoded = oracle::mc_encode(ARCH, &texts).expect("llvm-mc assembles every printed SIMM16");
    for ((text, bytes), (_, llvm_bytes)) in accepted.iter().zip(&encoded) {
        assert_eq!(bytes, llvm_bytes, "{text}")
    }
    eprintln!(
        "enc_sop: {} sendmsg/hwreg forms matched llvm-mc",
        accepted.len()
    );
}

/// Every S_WAITCNT SIMM16: llvm's text for each word, and the encoder accepts
/// exactly the words without unused bits.
#[test]
fn llvm_mc_waitcnt_exhaustive() {
    if oracle::disabled() {
        eprintln!("enc_sop: HIPFIRE_TEST_REQUIRE_ROCM=0, skipping the llvm-mc oracle");
        return;
    }
    let raw: Vec<Vec<u8>> = (0..=u16::MAX)
        .map(|v| oracle::bytes(&[0xbf8c_0000 | u32::from(v)]))
        .collect();
    let llvm = oracle::mc_disassemble(ARCH, &raw).expect("llvm-mc disassembles every s_waitcnt");
    for v in 0..=u16::MAX {
        let inst = Sopp {
            op: SoppOp::Waitcnt,
            simm16: v,
        };
        let mut out = Vec::new();
        if v & 0x3080 == 0 {
            inst.encode(ARCH, &mut out).unwrap();
            assert_eq!(out, [0xbf8c_0000 | u32::from(v)]);
            assert_eq!(inst.to_string(), llvm[usize::from(v)], "{v:#06x}");
            let (vm, exp, lgkm) = waitcnt_fields(ARCH, v);
            assert_eq!(waitcnt_imm(ARCH, vm, exp, lgkm).unwrap(), v);
        } else {
            assert!(inst.encode(ARCH, &mut out).is_err(), "{v:#06x}");
            assert!(
                Sopp::decode(ARCH, &[0xbf8c_0000 | u32::from(v)]).is_err(),
                "{v:#06x}"
            );
        }
    }
}

#[test]
fn waitcnt_round_trip_and_maxima() {
    let caps = ARCH.caps();
    for vm in 0..=caps.vmcnt_max {
        for exp in 0..=caps.expcnt_max {
            for lgkm in 0..=caps.lgkmcnt_max {
                let imm = waitcnt_imm(ARCH, vm, exp, lgkm).unwrap();
                assert_eq!(imm & 0x3080, 0);
                assert_eq!(waitcnt_fields(ARCH, imm), (vm, exp, lgkm));
            }
        }
    }
    // llvm-mc: `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` = 0xbf8c0000, the
    // all-maximum wait = 0xbf8ccf7f, `vmcnt(17) lgkmcnt(3)` = 0xbf8c4371.
    assert_eq!(waitcnt_imm(ARCH, 0, 0, 0).unwrap(), 0x0000);
    assert_eq!(waitcnt_imm(ARCH, 63, 7, 15).unwrap(), 0xcf7f);
    assert_eq!(waitcnt_imm(ARCH, 17, 7, 3).unwrap(), 0x4371);
    assert!(waitcnt_imm(ARCH, caps.vmcnt_max + 1, 0, 0).is_err());
    assert!(waitcnt_imm(ARCH, 0, caps.expcnt_max + 1, 0).is_err());
    assert!(waitcnt_imm(ARCH, 0, 0, caps.lgkmcnt_max + 1).is_err());
    assert_eq!(waitcnt_fields(ARCH, 0xffff), (63, 7, 15));
    assert_eq!(
        Sopp {
            op: SoppOp::Waitcnt,
            simm16: 0xcf7f
        }
        .to_string(),
        "s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(15)"
    );
    assert_eq!(
        Sopp {
            op: SoppOp::Waitcnt,
            simm16: 0x0f70
        }
        .to_string(),
        "s_waitcnt vmcnt(0)"
    );
}

fn enc(inst: Inst) -> peacemaker_cdna::Result<Vec<u32>> {
    inst.words(ARCH)
}

#[test]
fn illegal_forms_rejected() {
    use Src::*;
    let sop2 = |op, sdst, ssrc0, ssrc1| {
        enc(Inst::Sop2(Sop2 {
            op,
            sdst,
            ssrc0,
            ssrc1,
        }))
    };
    // Two different literals; vector registers; misaligned and out-of-range pairs.
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Lit(1000), Lit(1001)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Vgpr(1), Sgpr(2)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Agpr(1), Sgpr(2)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(1), Sgpr(2), Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(0), Sgpr(3), Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(0), Ttmp(3), Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(0), Sgpr(2), Sgpr(101)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(0), VccHi, Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AndB64, Sgpr(0), M0, Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AndB64, ExecHi, Sgpr(2), Sgpr(4)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(102), Sgpr(0), Sgpr(1)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Ttmp(16), Sgpr(1)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Int(65), Sgpr(1)).is_err());
    assert!(sop2(Sop2Op::AddU32, Sgpr(0), Int(-17), Sgpr(1)).is_err());
    // Destinations must be scalar registers.
    for d in [
        Int(1),
        Lit(5),
        Float(InlineFloat::One),
        Scc,
        Vccz,
        SharedBase,
        Vgpr(0),
    ] {
        assert!(sop2(Sop2Op::AddU32, d, Sgpr(0), Sgpr(1)).is_err(), "{d:?}");
    }
    // The lshl_b64 shift count is a 32-bit operand: a high-half name is legal there.
    assert!(sop2(Sop2Op::LshlB64, Sgpr(0), Sgpr(2), VccHi).is_ok());
    // SOP1 operand presence.
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::GetpcB64,
        sdst: Some(Sgpr(0)),
        ssrc0: Some(Sgpr(2))
    }))
    .is_err());
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::GetpcB64,
        sdst: None,
        ssrc0: None
    }))
    .is_err());
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::SetpcB64,
        sdst: Some(Sgpr(0)),
        ssrc0: Some(Sgpr(2))
    }))
    .is_err());
    for s in [Int(0), Lit(0x1234_5678), SharedBase, Scc] {
        assert!(
            enc(Inst::Sop1(Sop1 {
                op: Sop1Op::SetpcB64,
                sdst: None,
                ssrc0: Some(s)
            }))
            .is_err(),
            "{s:?}"
        );
    }
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::MovB32,
        sdst: None,
        ssrc0: Some(Sgpr(2))
    }))
    .is_err());
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::MovB32,
        sdst: Some(Sgpr(0)),
        ssrc0: None
    }))
    .is_err());
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::MovB64,
        sdst: Some(Sgpr(1)),
        ssrc0: Some(Sgpr(2))
    }))
    .is_err());
    assert!(enc(Inst::Sop1(Sop1 {
        op: Sop1Op::AndSaveexecB64,
        sdst: Some(M0),
        ssrc0: Some(VccLo)
    }))
    .is_err());
    // SOPK: no literal, scalar register only.
    assert!(enc(Inst::Sopk(Sopk {
        op: SopkOp::MovkI32,
        sdst: Lit(5),
        simm16: 1
    }))
    .is_err());
    assert!(enc(Inst::Sopk(Sopk {
        op: SopkOp::CmpkEqI32,
        sdst: Int(3),
        simm16: 1
    }))
    .is_err());
    assert!(enc(Inst::Sopk(Sopk {
        op: SopkOp::SetregB32,
        sdst: Vgpr(3),
        simm16: 1
    }))
    .is_err());
    // SOPC.
    assert!(enc(Inst::Sopc(Sopc {
        op: SopcOp::CmpEqU32,
        ssrc0: Lit(1000),
        ssrc1: Lit(2000)
    }))
    .is_err());
    assert!(enc(Inst::Sopc(Sopc {
        op: SopcOp::CmpEqU64,
        ssrc0: Sgpr(1),
        ssrc1: Sgpr(2)
    }))
    .is_err());
    assert!(enc(Inst::Sopc(Sopc {
        op: SopcOp::CmpEqU64,
        ssrc0: Sgpr(0),
        ssrc1: M0
    }))
    .is_err());
    // SOPP: no-operand ops, unused waitcnt bits, lossy sendmsg.
    assert!(enc(Inst::Sopp(Sopp {
        op: SoppOp::Barrier,
        simm16: 1
    }))
    .is_err());
    assert!(enc(Inst::Sopp(Sopp {
        op: SoppOp::IcacheInv,
        simm16: 1
    }))
    .is_err());
    assert!(enc(Inst::Sopp(Sopp {
        op: SoppOp::Waitcnt,
        simm16: 0x0080
    }))
    .is_err());
    assert!(enc(Inst::Sopp(Sopp {
        op: SoppOp::Waitcnt,
        simm16: 0x1000
    }))
    .is_err());
    assert!(enc(Inst::Sopp(Sopp {
        op: SoppOp::Sendmsg,
        simm16: 0x81
    }))
    .is_err());

    // Decoder: truncated literal, unknown opcodes, reserved codes, unused fields.
    assert!(Inst::decode(ARCH, &[0xbe80_00ff]).is_err()); // s_mov_b32 s0, <literal> without the dword
    assert!(Inst::decode(ARCH, &[0xbe80_00ff, 7]).is_ok());
    assert!(Inst::decode(ARCH, &[0xbe80_007d]).is_err()); // code 125 (llvm's `null`) is not a gfx942 operand here
    assert!(Inst::decode(ARCH, &[0xbe80_00ef]).is_err()); // code 239 reserved in Src
    assert!(Inst::decode(ARCH, &[0xbe80_1c05]).is_err()); // s_getpc_b64 with a nonzero SSRC0 field
    assert!(Inst::decode(ARCH, &[0xbe85_1d00]).is_err()); // s_setpc_b64 with a nonzero SDST field
    assert!(Inst::decode(ARCH, &[0xbe81_0102]).is_err()); // s_mov_b64 s[1:2], s[2:3]
    assert!(Inst::decode(ARCH, &[0xbe80_3f00]).is_err()); // SOP1 opcode 63: not in the table
    assert!(Inst::decode(ARCH, &[0xb980_0000]).is_err()); // SOPK opcode 19: not in the table
    assert!(Inst::decode(ARCH, &[0xbf8a_0005]).is_err()); // s_barrier 5
    assert!(Inst::decode(ARCH, &[0xbf8c_0080]).is_err()); // s_waitcnt with bit 7
    assert!(Sop2::decode(ARCH, &[0xbf8c_0000]).is_err()); // a SOPP word is not SOP2
    assert!(Sopp::decode(ARCH, &[]).is_err());
}

#[test]
fn footprints() {
    use Src::*;
    let scc = RegSpan::new(RegFile::Scc, 0, 1);
    let exec = RegSpan::new(RegFile::Exec, 0, 2);
    let f = Inst::Sop2(Sop2 {
        op: Sop2Op::AddcU32,
        sdst: Sgpr(0),
        ssrc0: Sgpr(1),
        ssrc1: Lit(9),
    })
    .footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Sgpr, 1, 1), scc]);
    assert_eq!(f.defs, [RegSpan::new(RegFile::Sgpr, 0, 1), scc]);
    let f = Inst::Sop2(Sop2 {
        op: Sop2Op::CselectB64,
        sdst: VccLo,
        ssrc0: Sgpr(2),
        ssrc1: Int(0),
    })
    .footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Sgpr, 2, 2), scc]);
    assert_eq!(f.defs, [RegSpan::new(RegFile::Vcc, 0, 2)]);
    let f = Inst::Sop2(Sop2 {
        op: Sop2Op::LshlB64,
        sdst: Sgpr(4),
        ssrc0: Sgpr(6),
        ssrc1: Sgpr(8),
    })
    .footprint();
    assert_eq!(
        f.uses,
        [
            RegSpan::new(RegFile::Sgpr, 6, 2),
            RegSpan::new(RegFile::Sgpr, 8, 1)
        ]
    );
    assert_eq!(f.defs, [RegSpan::new(RegFile::Sgpr, 4, 2), scc]);
    assert!(!Inst::Sop2(Sop2 {
        op: Sop2Op::MulI32,
        sdst: Sgpr(0),
        ssrc0: Sgpr(1),
        ssrc1: Sgpr(2)
    })
    .footprint()
    .defs
    .contains(&scc));

    let f = Inst::Sop1(Sop1 {
        op: Sop1Op::AndSaveexecB64,
        sdst: Some(Sgpr(10)),
        ssrc0: Some(VccLo),
    })
    .footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Vcc, 0, 2), exec]);
    assert_eq!(f.defs, [RegSpan::new(RegFile::Sgpr, 10, 2), exec, scc]);
    let f = Inst::Sop1(Sop1 {
        op: Sop1Op::CmovB32,
        sdst: Some(M0),
        ssrc0: Some(Sgpr(3)),
    })
    .footprint();
    assert_eq!(
        f.uses,
        [
            RegSpan::new(RegFile::Sgpr, 3, 1),
            RegSpan::new(RegFile::M0, 0, 1),
            scc
        ]
    );
    assert_eq!(f.defs, [RegSpan::new(RegFile::M0, 0, 1)]);
    let f = Inst::Sop1(Sop1 {
        op: Sop1Op::GetpcB64,
        sdst: Some(Sgpr(0)),
        ssrc0: None,
    })
    .footprint();
    assert!(f.uses.is_empty());
    assert_eq!(f.defs, [RegSpan::new(RegFile::Sgpr, 0, 2)]);
    let f = Inst::Sop1(Sop1 {
        op: Sop1Op::SetpcB64,
        sdst: None,
        ssrc0: Some(Sgpr(30)),
    })
    .footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Sgpr, 30, 2)]);
    assert!(f.defs.is_empty());

    let f = Inst::Sopk(Sopk {
        op: SopkOp::AddkI32,
        sdst: Sgpr(5),
        simm16: 3,
    })
    .footprint();
    assert_eq!(f.uses, [RegSpan::new(RegFile::Sgpr, 5, 1)]);
    assert_eq!(f.defs, [RegSpan::new(RegFile::Sgpr, 5, 1), scc]);
    let f = Inst::Sopk(Sopk {
        op: SopkOp::CmpkLgU32,
        sdst: Sgpr(5),
        simm16: 3,
    })
    .footprint();
    assert_eq!(
        (f.uses.as_slice(), f.defs.as_slice()),
        (&[RegSpan::new(RegFile::Sgpr, 5, 1)][..], &[scc][..])
    );
    let f = Inst::Sopk(Sopk {
        op: SopkOp::SetregB32,
        sdst: Sgpr(5),
        simm16: 0xf801,
    })
    .footprint();
    assert_eq!(
        (f.uses.as_slice(), f.defs.len()),
        (&[RegSpan::new(RegFile::Sgpr, 5, 1)][..], 0)
    );

    let f = Inst::Sopc(Sopc {
        op: SopcOp::CmpEqU64,
        ssrc0: ExecLo,
        ssrc1: Int(0),
    })
    .footprint();
    assert_eq!(
        (f.uses.as_slice(), f.defs.as_slice()),
        (&[exec][..], &[scc][..])
    );

    assert_eq!(
        Inst::Sopp(Sopp {
            op: SoppOp::CbranchScc1,
            simm16: 2
        })
        .footprint()
        .uses,
        [scc]
    );
    assert_eq!(
        Inst::Sopp(Sopp {
            op: SoppOp::CbranchExecz,
            simm16: 2
        })
        .footprint()
        .uses,
        [exec]
    );
    assert_eq!(
        Inst::Sopp(Sopp {
            op: SoppOp::CbranchVccnz,
            simm16: 2
        })
        .footprint()
        .uses,
        [RegSpan::new(RegFile::Vcc, 0, 2)]
    );
    assert_eq!(
        Inst::Sopp(Sopp {
            op: SoppOp::Sendmsg,
            simm16: 1
        })
        .footprint()
        .uses,
        [RegSpan::new(RegFile::M0, 0, 1)]
    );
    assert_eq!(
        Inst::Sopp(Sopp {
            op: SoppOp::Branch,
            simm16: 0xfffd
        })
        .branch_offset(),
        Some(-3)
    );
    assert!(
        Inst::Sopp(Sopp {
            op: SoppOp::Waitcnt,
            simm16: 0
        })
        .footprint()
            == Default::default()
    );
}
