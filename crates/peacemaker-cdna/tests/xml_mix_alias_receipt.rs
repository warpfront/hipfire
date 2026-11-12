// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLVM-alias promotion of the gfx942 `V_MAD_MIX_F32` `ENC_VOP3P` row.
//!
//! The XML row names opcode 32 `V_MAD_MIX_F32`; `llvm-mc -mcpu=gfx942` rejects
//! that mnemonic and assembles the same opcode as `v_fma_mix_f32`, so the
//! row-level LLVM cross-check records it `xml-only` (unverified). The
//! hand-written `Vop3pOp::FmaMixF32` model is swept against llvm-mc over its
//! whole selection space in `enc_vop3p.rs` (`oracle_fma_mix_f32_full_selection_space`);
//! the alias receipt promotes exactly the instances of that model (VGPR / one
//! SGPR / inline-constant sources, `NEG_HI` = 0). The row itself, the
//! `V_MAD_MIXLO_F16` / `V_MAD_MIXHI_F16` rows, gfx950 and every `NEG_HI`,
//! literal, reserved-operand, special-scalar or two-scalar instance stay
//! unverified.
//!
//! This file builds the swept words from the documented layout (independent of
//! the typed encoder) and needs no llvm-mc: the llvm-mc sweep is the other
//! test.
use peacemaker_cdna::enc::vop3p::{Vop3p, Vop3pOp};
use peacemaker_cdna::operand::Src;
use peacemaker_cdna::xml::{self, Decoded, Isa};
use peacemaker_cdna::{
    Arch, Inst, RowStatus, XmlInst, ALIAS_RECEIPTS, V_MAD_MIX_F32_AS_FMA_MIX_F32,
};

/// Operand codes of the seven swept source shapes (VGPR v2/v3/v4; SGPR s2,
/// s101, s7 in position 0/1/2; inline 0, 1.0, 64 in position 0/1/2).
const SHAPES: [[u32; 3]; 7] = [
    [258, 259, 260],
    [2, 259, 260],
    [258, 101, 260],
    [258, 259, 7],
    [128, 259, 260],
    [258, 242, 260],
    [258, 259, 192],
];

const ROW: &str = "V_MAD_MIX_F32";

/// `v_fma_mix_f32`-layout words of opcode `opcode` with vdst v1:
/// word0 = 0xd380_0000 | opcode<<16 | vdst | neg_hi<<8 | op_sel<<11 |
/// op_sel_hi[2]<<14 | clamp<<15; word1 = src0 | src1<<9 | src2<<18 |
/// op_sel_hi[1:0]<<27 | neg_lo<<29.
fn word(
    opcode: u32,
    codes: [u32; 3],
    sel: u32,
    hi: u32,
    lo: u32,
    nh: u32,
    clamp: u32,
) -> [u32; 2] {
    [
        0xd380_0000 | opcode << 16 | 1 | nh << 8 | sel << 11 | (hi >> 2) << 14 | clamp << 15,
        codes[0] | codes[1] << 9 | codes[2] << 18 | (hi & 3) << 27 | lo << 29,
    ]
}

fn mix(codes: [u32; 3], sel: u32, hi: u32, lo: u32, clamp: u32) -> [u32; 2] {
    word(32, codes, sel, hi, lo, 0, clamp)
}

/// The 7 x 8 x 8 x 8 x 2 = 7168 swept words.
fn sweep() -> Vec<[u32; 2]> {
    let mut out = Vec::new();
    for codes in SHAPES {
        for sel in 0..8 {
            for hi in 0..8 {
                for lo in 0..8 {
                    for clamp in 0..2 {
                        out.push(mix(codes, sel, hi, lo, clamp));
                    }
                }
            }
        }
    }
    out
}

fn decode(isa: Isa, words: &[u32]) -> Decoded {
    xml::decode(isa, words)
        .unwrap_or_else(|e| panic!("{words:08x?} does not decode on {isa}: {e}"))
        .0
}

/// `words` decode through the XML tables as the receipt row, but nothing
/// promotes them: the instance keeps the row's `xml-only` refusal, decodes
/// through `XmlInst` (when it does) and never encodes.
fn assert_not_promoted(words: &[u32], label: &str) {
    let d = decode(Isa::Cdna3, words);
    assert_eq!(d.form.name, ROW, "{label}");
    assert!(XmlInst::alias_receipt_of(&d).is_none(), "{label}: promoted");
    assert!(XmlInst::receipt_of(&d).is_none(), "{label}");
    let why = XmlInst::verification_of(&d).expect_err(label);
    assert!(why.contains("xml-only"), "{label}: {why}");
    if let Ok((x, _)) = XmlInst::decode(Arch::Gfx942, words) {
        assert!(!x.verified(), "{label}");
        assert!(x.alias_receipt().is_none(), "{label}");
        assert!(x.unsupported().contains("unverified encoding"), "{label}");
        let mut out = Vec::new();
        let e = x.encode(Arch::Gfx942, &mut out).unwrap_err();
        assert!(e.contains("unverified encoding"), "{label}: {e}");
        assert!(out.is_empty(), "{label}");
    }
}

#[test]
fn the_row_stays_unverified_and_the_receipt_names_it() {
    let r = &V_MAD_MIX_F32_AS_FMA_MIX_F32;
    assert_eq!(ALIAS_RECEIPTS.len(), 1);
    assert!(std::ptr::eq(ALIAS_RECEIPTS[0], r));
    assert!(ALIAS_RECEIPTS.iter().all(|r| r.isa == Isa::Cdna3));
    assert_eq!(
        (r.isa, r.name, r.encoding, r.llvm_mnemonic),
        (Isa::Cdna3, ROW, "ENC_VOP3P", "v_fma_mix_f32")
    );
    let rows: Vec<_> = xml::forms(r.isa)
        .iter()
        .filter(|f| f.name == r.name && f.encoding == r.encoding)
        .collect();
    assert_eq!(rows.len(), 1, "exactly one receipt row");
    let form = rows[0];
    assert_eq!(form.opcode, 32);
    assert!(
        !form.verification.is_verified(),
        "the row itself stays unverified"
    );
    // The typed model's mnemonic is llvm's name of the same opcode.
    assert_eq!(Vop3pOp::FmaMixF32.name(), r.llvm_mnemonic);
    assert_eq!(Vop3pOp::FmaMixF32.opcode(Arch::Gfx942), Some(32));
    assert_eq!(Vop3pOp::FmaMixF32.opcode(Arch::Gfx950), None);
}

#[test]
fn receipt_cites_hardware_provenance_and_the_llvm_sweep() {
    let r = &V_MAD_MIX_F32_AS_FMA_MIX_F32;
    assert!(r.evidence.contains("oracle_fma_mix_f32_full_selection_space"));
    assert!(r.evidence.contains("7168"));
    assert!(!r.oracle_receipt.is_empty(), "hardware receipt path");
    assert_eq!(r.oracle_sha256.len(), 64, "hardware receipt sha256");
    assert!(r.oracle_sha256.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[test]
fn every_swept_word_is_promoted_verified_and_encodes_back() {
    let words = sweep();
    assert_eq!(words.len(), 7 * 8 * 8 * 8 * 2);
    let distinct: std::collections::HashSet<_> = words.iter().collect();
    assert_eq!(distinct.len(), words.len(), "swept words are distinct");
    for w in &words {
        let label = format!("{w:08x?}");
        let d = decode(Isa::Cdna3, w);
        assert_eq!(d.form.name, ROW, "{label}");
        assert!(!d.form.verification.is_verified(), "{label}");
        let receipt = XmlInst::alias_receipt_of(&d).unwrap_or_else(|| panic!("{label}: refused"));
        assert!(std::ptr::eq(receipt, &V_MAD_MIX_F32_AS_FMA_MIX_F32));
        assert!(XmlInst::receipt_of(&d).is_none(), "{label}");
        assert_eq!(XmlInst::verification_of(&d), Ok(()), "{label}");

        // The generated-table instruction encodes back to the same words.
        let (x, n) = XmlInst::decode(Arch::Gfx942, w).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(n, 2, "{label}");
        assert!(x.verified() && x.verification().is_ok(), "{label}");
        assert!(x.alias_receipt().is_some() && x.receipt().is_none(), "{label}");
        assert!(!x.unsupported().contains("unverified"), "{label}");
        let mut out = Vec::new();
        x.encode(Arch::Gfx942, &mut out)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out, w, "{label}");

        // The typed instruction is the same words and reports Verified.
        let (inst, n) = Inst::decode(Arch::Gfx942, w).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(n, 2, "{label}");
        assert!(
            matches!(&inst, Inst::Vop3p(x) if x.op == Vop3pOp::FmaMixF32),
            "{label}: {inst:?}"
        );
        assert_eq!(inst.words(Arch::Gfx942).unwrap(), w, "{label}");
        assert_eq!(inst.verification(Arch::Gfx942), RowStatus::Verified, "{label}");
    }
}

#[test]
fn neg_hi_is_never_promoted() {
    // `|src|` spells NEG_HI (llvm-mc assembles it); the codec refuses it.
    for codes in SHAPES {
        for nh in 1..8 {
            for hi in 0..8 {
                let w = word(32, codes, 0, hi, 0, nh, 0);
                assert!(Vop3p::decode(Arch::Gfx942, &w).is_err());
                assert_not_promoted(&w, &format!("{w:08x?}"));
            }
        }
    }
}

#[test]
fn literal_reserved_special_and_multiple_scalar_sources_are_never_promoted() {
    let base = SHAPES[0];
    let with = |i: usize, code: u32| {
        let mut c = base;
        c[i] = code;
        c
    };
    for i in 0..3 {
        // Literal operand (code 255) followed by its literal dword.
        let w = mix(with(i, 255), 0, 0, 0, 0);
        assert_not_promoted(&[w[0], w[1], 0x3e80_0000], &format!("literal src{i}"));
        // Every reserved operand code.
        for code in 0..512u16 {
            if code == 255 || Src::from_code(code, None).is_ok() {
                continue;
            }
            let w = mix(with(i, u32::from(code)), 0, 0, 0, 0);
            if xml::decode(Isa::Cdna3, &w).is_ok() {
                assert_not_promoted(&w, &format!("reserved code {code} src{i}"));
            }
        }
        // Special scalars (flat_scratch, xnack_mask, vcc, ttmp, m0, exec,
        // shared/private base and limit, vccz, execz, scc) are outside the
        // swept source classes.
        for code in (102..=108).chain([123, 124, 126, 127, 235, 236, 237, 238, 251, 252, 253]) {
            let w = mix(with(i, code), 0, 0, 0, 0);
            assert_not_promoted(&w, &format!("special scalar code {code} src{i}"));
        }
    }
    // Two different SGPRs (the constant bus holds one), also with a special
    // scalar and across op_sel_hi.
    for codes in [
        [2, 3, 260],
        [2, 259, 3],
        [258, 3, 4],
        [2, 106, 260],
        [2, 3, 4],
    ] {
        for hi in 0..8 {
            let label = format!("scalars {codes:?} hi {hi}");
            assert_not_promoted(&mix(codes, 0, hi, 0, 0), &label);
        }
    }
    // The same SGPR in every position and an SGPR with inline constants are
    // one scalar: promoted.
    for codes in [[2, 2, 2], [2, 133, 2], [241, 2, 131]] {
        let d = decode(Isa::Cdna3, &mix(codes, 5, 3, 6, 1));
        assert!(XmlInst::alias_receipt_of(&d).is_some(), "{codes:?}");
    }
}

#[test]
fn mixlo_mixhi_and_neighbouring_opcodes_are_not_promoted() {
    // V_MAD_MIXLO_F16 (33) and V_MAD_MIXHI_F16 (34): the typed model decodes
    // them, but their rows are unverified and nothing promotes them.
    for (opcode, name, typed) in [
        (33, "V_MAD_MIXLO_F16", Vop3pOp::FmaMixloF16),
        (34, "V_MAD_MIXHI_F16", Vop3pOp::FmaMixhiF16),
    ] {
        for sel in 0..8 {
            for hi in 0..8 {
                let w = word(opcode, SHAPES[0], sel, hi, 5, 0, 1);
                let d = decode(Isa::Cdna3, &w);
                assert_eq!(d.form.name, name);
                assert!(!d.form.verification.is_verified());
                assert!(XmlInst::alias_receipt_of(&d).is_none(), "{name}");
                assert!(XmlInst::verification_of(&d).is_err(), "{name}");
                let (inst, _) = Inst::decode(Arch::Gfx942, &w).unwrap();
                assert!(matches!(&inst, Inst::Vop3p(x) if x.op == typed), "{name}");
                assert!(
                    matches!(inst.verification(Arch::Gfx942), RowStatus::Unverified(_)),
                    "{name}: {:?}",
                    inst.verification(Arch::Gfx942)
                );
            }
        }
    }
    // Every other VOP3P opcode keeps its own row's standing.
    for opcode in (0..128).filter(|&op| op != 32) {
        let w = word(opcode, SHAPES[0], 0, 0, 0, 0, 0);
        if let Ok((d, _)) = xml::decode(Isa::Cdna3, &w) {
            assert!(XmlInst::alias_receipt_of(&d).is_none(), "opcode {opcode}");
            assert_eq!(
                XmlInst::verification_of(&d).is_ok(),
                d.form.verification.is_verified(),
                "opcode {opcode}"
            );
        }
    }
    // No other unverified CDNA3 row is covered by the receipt: the default
    // instance of every unverified row is covered only for the receipt's row.
    for form in xml::forms(Isa::Cdna3) {
        if form.verification.is_verified() {
            continue;
        }
        let d = Decoded::new(Isa::Cdna3, form);
        if XmlInst::alias_receipt_of(&d).is_some() {
            assert_eq!((form.name, form.encoding), (ROW, "ENC_VOP3P"));
        }
    }
}

#[test]
fn gfx950_is_not_promoted() {
    for w in sweep().iter().step_by(37) {
        // CDNA4 decodes the words under its own rows: the alias receipt is
        // gfx942 only, so the row's own standing is all there is.
        if let Ok((d, _)) = xml::decode(Isa::Cdna4, w) {
            assert!(XmlInst::alias_receipt_of(&d).is_none(), "{w:08x?}");
            assert_eq!(
                XmlInst::verification_of(&d).is_ok(),
                d.form.verification.is_verified(),
                "{w:08x?}"
            );
        }
        if let Ok((x, _)) = XmlInst::decode(Arch::Gfx950, w) {
            assert!(x.alias_receipt().is_none(), "{w:08x?}");
            let e = x.encode(Arch::Gfx950, &mut Vec::new()).unwrap_err();
            assert!(e.contains("decode-only"), "{w:08x?}: {e}");
        }
        // And gfx950 never reports a gfx942 instruction as promoted.
        let (inst, _) = Inst::decode(Arch::Gfx942, w).unwrap();
        assert!(
            !matches!(inst.verification(Arch::Gfx950), RowStatus::Verified),
            "{w:08x?}"
        );
    }
}
