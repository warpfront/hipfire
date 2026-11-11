// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Field-specific hardware verification of the `V_CVT_PK_F32_FP8`
//! `VOP1_VOP_SDWA` row.
//!
//! The LLVM cross-check left the row unverified (its zero seed is rejected).
//! The gfx942 FP8 grouped MoE anchor executes exactly one SDWA field
//! combination (`src0_sel:WORD_1`, dst_sel DWORD, everything else zero), and
//! the MI300X oracle receipt proves it bitwise: only that combination, with
//! any register numbers, is promoted per instance. The row itself and every
//! other selector / flag combination stay unverified.
use peacemaker_cdna::xml::{self, Decoded, Isa};
use peacemaker_cdna::{Arch, Inst, RowStatus, XmlInst, FIELD_RECEIPTS, V_CVT_PK_F32_FP8_SDWA_WORD1};

/// First dword of `v_cvt_pk_f32_fp8_sdwa v[vdst:vdst+1], v? ...`.
fn w0(vdst: u32) -> u32 {
    0x7e00_acf9 | (vdst << 17)
}

/// The anchor's second dword for `vsrc0`: dst_sel DWORD (6), dst_unused PAD,
/// src0_sel WORD_1 (5), every flag and the src1 fields zero.
fn w1(vsrc0: u32) -> u32 {
    0x0005_0600 | vsrc0
}

/// Second-dword fields: (name, shift, width, anchor value).
const SEL_FIELDS: [(&str, u32, u32, u32); 14] = [
    ("DST_SEL", 8, 3, 6),
    ("DST_UNUSED", 11, 2, 0),
    ("CLAMP", 13, 1, 0),
    ("OMOD", 14, 2, 0),
    ("SRC0_SEL", 16, 3, 5),
    ("SRC0_SEXT", 19, 1, 0),
    ("SRC0_NEG", 20, 1, 0),
    ("SRC0_ABS", 21, 1, 0),
    ("S0", 23, 1, 0),
    ("SRC1_SEL", 24, 3, 0),
    ("SRC1_SEXT", 27, 1, 0),
    ("SRC1_NEG", 28, 1, 0),
    ("SRC1_ABS", 29, 1, 0),
    ("S1", 31, 1, 0),
];

/// Anchor instances (disassembly of the twin's device ELF, registers vary).
const ANCHOR_WORDS: [[u32; 2]; 6] = [
    [0x7e84_acf9, 0x0005_0636],
    [0x7ed0_acf9, 0x0005_063f],
    [0x7e80_acf9, 0x0005_0640],
    [0x7e7c_acf9, 0x0005_063d],
    [0x7ed0_acf9, 0x0005_0665],
    [0x7eb0_acf9, 0x0005_0633],
];

fn decode(isa: Isa, words: &[u32]) -> Option<Decoded> {
    xml::decode(isa, words).ok().map(|(d, _)| d)
}

fn mutate(word: u32, shift: u32, width: u32, value: u32) -> u32 {
    let mask = ((1u32 << width) - 1) << shift;
    (word & !mask) | (value << shift)
}

#[test]
fn anchor_words_are_the_receipt_row_and_the_row_stays_unverified() {
    for words in ANCHOR_WORDS {
        let d = decode(Isa::Cdna3, &words).expect("anchor words decode");
        assert_eq!(d.form.name, "V_CVT_PK_F32_FP8");
        assert_eq!(d.form.encoding, "VOP1_VOP_SDWA");
        // Row level: still unverified, the whole-row policy is untouched.
        assert!(!d.form.verification.is_verified(), "row must stay unverified");
        assert!(d.form.sample.is_empty());
        // Instance level: promoted by exactly the anchor receipt.
        assert_eq!(XmlInst::verification_of(&d), Ok(()));
        let r = XmlInst::receipt_of(&d).expect("receipt covers the anchor instance");
        assert!(std::ptr::eq(r, &V_CVT_PK_F32_FP8_SDWA_WORD1));
    }
}

#[test]
fn anchor_instances_are_verified_and_encode_back() {
    for words in ANCHOR_WORDS {
        let (x, n) = XmlInst::decode(Arch::Gfx942, &words).expect("decodes");
        assert_eq!(n, 2);
        assert_eq!(x.name(), "v_cvt_pk_f32_fp8");
        assert_eq!(x.encoding(), "VOP1_VOP_SDWA");
        assert_eq!(x.field("DST_SEL"), Some(6));
        assert_eq!(x.field("SRC0_SEL"), Some(5));
        assert!(x.verified());
        assert_eq!(x.verification(), Ok(()));
        assert!(x.receipt().is_some());
        // The row-level refusal does not leak into the promoted instance.
        assert!(!x.unsupported().contains("unverified encoding"), "{}", x.unsupported());
        let mut out = Vec::new();
        x.encode(Arch::Gfx942, &mut out).expect("promoted instance encodes");
        assert_eq!(out, words);
        // gfx950 stays decode-only.
        let mut out = Vec::new();
        assert!(x.encode(Arch::Gfx950, &mut out).is_err());
    }
}

#[test]
fn every_sdwa_instance_of_the_committed_twin_is_promoted() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gemm_fp8b128_moe_grouped_bf16mfma_pm_gfx942.hxaco"
    );
    let bytes = std::fs::read(path).expect("committed twin .hxaco");
    // The gfx942 device ELF sits at the 4 KiB-aligned bundle entry; scan its
    // dwords for the anchor combination (register numbers masked out).
    let dwords: Vec<u32> = bytes[4096..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut found = 0;
    for pair in dwords.windows(2) {
        if pair[0] & 0xfe01_ffff == 0x7e00_acf9 && pair[1] & 0xffff_ff00 == 0x0005_0600 {
            let (x, _) = XmlInst::decode(Arch::Gfx942, pair).expect("decodes");
            assert!(x.verified(), "{pair:#x?}");
            found += 1;
        }
    }
    assert_eq!(found, 32, "SDWA instances of the anchor's device ELF");
}

#[test]
fn register_numbers_may_vary() {
    for vdst in [0, 1, 64, 104, 126, 252] {
        for vsrc0 in [0, 42, 101, 255] {
            let words = [w0(vdst), w1(vsrc0)];
            let d = decode(Isa::Cdna3, &words).expect("decodes");
            assert_eq!(d.field("VDST"), Some(u64::from(vdst)));
            assert_eq!(d.field("VSRC0"), Some(u64::from(vsrc0)));
            assert_eq!(XmlInst::verification_of(&d), Ok(()), "{words:#x?}");
        }
    }
}

#[test]
fn every_neighbouring_selector_and_flag_is_refused() {
    let anchor = w1(54);
    for (name, shift, width, anchor_value) in SEL_FIELDS {
        assert_eq!(anchor >> shift & ((1 << width) - 1), anchor_value, "{name}");
        let mut checked = 0;
        for value in 0..(1u32 << width) {
            if value == anchor_value {
                continue;
            }
            let words = [w0(66), mutate(anchor, shift, width, value)];
            // A reserved or unencodable neighbour is refused by decode itself.
            let Some(d) = decode(Isa::Cdna3, &words) else {
                continue;
            };
            assert_eq!(d.form.name, "V_CVT_PK_F32_FP8", "{name}={value}");
            assert_eq!(d.field(name), Some(u64::from(value)), "{name}");
            assert!(XmlInst::verification_of(&d).is_err(), "{name}={value} was promoted");
            assert!(XmlInst::receipt_of(&d).is_none(), "{name}={value}");
            checked += 1;
        }
        assert!(checked > 0, "{name}: no neighbouring value decoded");
    }
}

#[test]
fn combined_neighbours_are_refused() {
    // Two fields off the anchor at once, and the anchor selector on the
    // wrong side (src1_sel instead of src0_sel).
    for words in [
        [w0(66), mutate(mutate(w1(54), 16, 3, 4), 8, 3, 5)],
        [w0(66), mutate(mutate(w1(54), 13, 1, 1), 16, 3, 4)],
        [w0(66), mutate(mutate(w1(54), 16, 3, 0), 24, 3, 5)],
        [w0(66), mutate(mutate(w1(54), 20, 1, 1), 21, 1, 1)],
    ] {
        let d = decode(Isa::Cdna3, &words).expect("decodes");
        assert!(XmlInst::verification_of(&d).is_err(), "{words:#x?}");
    }
}

#[test]
fn a_refused_neighbour_decodes_but_never_encodes() {
    for (shift, width, value) in [(16, 3, 4), (16, 3, 0), (8, 3, 5), (13, 1, 1), (19, 1, 1)] {
        let words = [w0(66), mutate(w1(54), shift, width, value)];
        let (x, _) = XmlInst::decode(Arch::Gfx942, &words).expect("decodes");
        assert!(!x.verified());
        assert!(x.receipt().is_none());
        assert!(x.unsupported().contains("unverified encoding"), "{}", x.unsupported());
        let mut out = Vec::new();
        let e = x.encode(Arch::Gfx942, &mut out).unwrap_err();
        assert!(e.contains("unverified encoding"), "{e}");
        assert!(out.is_empty());
    }
}

#[test]
fn neighbouring_opcodes_and_gfx950_are_not_promoted() {
    // Every other opcode of the SDWA VOP1 encoding with the anchor's fields.
    for op in (0..256u32).filter(|&op| op != 86) {
        let words = [mutate(w0(66), 9, 8, op), w1(54)];
        if let Some(d) = decode(Isa::Cdna3, &words) {
            assert!(XmlInst::receipt_of(&d).is_none(), "opcode {op}");
            assert_eq!(
                XmlInst::verification_of(&d).is_ok(),
                d.form.verification.is_verified(),
                "opcode {op}"
            );
        }
    }
    // The same words on gfx950 (CDNA4) are never promoted: the receipt is
    // gfx942 hardware.
    for words in ANCHOR_WORDS {
        if let Some(d) = decode(Isa::Cdna4, &words) {
            assert!(XmlInst::receipt_of(&d).is_none());
            assert_eq!(
                XmlInst::verification_of(&d).is_ok(),
                d.form.verification.is_verified()
            );
        }
    }
}

#[test]
fn receipt_names_every_field_of_its_row_and_cites_provenance() {
    let r = &V_CVT_PK_F32_FP8_SDWA_WORD1;
    assert_eq!(FIELD_RECEIPTS.len(), 1);
    let form = xml::forms(r.isa)
        .iter()
        .find(|f| f.name == r.name && f.encoding == r.encoding)
        .expect("receipt row");
    assert!(!form.verification.is_verified(), "the row itself stays unverified");
    // Every field of the row is either free (a register number) or pinned.
    for fl in form.fields {
        let free = r.free.contains(&fl.name);
        let fixed = r.fixed.iter().filter(|(n, _)| *n == fl.name).count();
        assert!(free ^ (fixed == 1), "{} must be free xor pinned once", fl.name);
    }
    for (n, _) in r.fixed {
        assert!(form.fields.iter().any(|fl| fl.name == *n), "{n}");
    }
    assert_eq!(r.free, ["VDST", "VSRC0"]);
    for s in [r.twin_receipt, r.oracle_receipt, r.evidence] {
        assert!(!s.is_empty());
    }
    for h in [r.oracle_sha256, r.anchor_sha256, r.twin_sha256] {
        assert_eq!(h.len(), 64);
        assert!(h.bytes().all(|b| b.is_ascii_hexdigit()));
    }
}

#[test]
fn inst_verification_reports_the_instance_status() {
    // `Inst::verification` of the typed or generated decode of the same words
    // must agree with the per-instance status.
    let (anchor, _) = Inst::decode(Arch::Gfx942, &ANCHOR_WORDS[0]).expect("anchor decodes");
    assert_eq!(anchor.verification(Arch::Gfx942), RowStatus::Verified);
    let neighbour = [w0(66), mutate(w1(54), 16, 3, 4)];
    let (neighbour, _) = Inst::decode(Arch::Gfx942, &neighbour).expect("neighbour decodes");
    assert!(
        matches!(neighbour.verification(Arch::Gfx942), RowStatus::Unverified(_)),
        "{:?}",
        neighbour.verification(Arch::Gfx942)
    );
}
