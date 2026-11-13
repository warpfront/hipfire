// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Generated (XML) instructions in [`Inst`]: decode fallback behind the typed
//! tables, structured encode, the footprint / memory class / branch offset
//! derived from the XML row, gfx950 decode-only, unverified rows, and the
//! emulator refusing a generated instruction by name.
use peacemaker_cdna::inst::{MemKind, RegFile, RegSpan, RowStatus};
use peacemaker_cdna::sim::{self, Config, Memory, SimError, Wave};
use peacemaker_cdna::xml::{self, Decoded, Isa};
use peacemaker_cdna::{Arch, Inst, XmlInst};

/// The words of the (default-condition) form `name`, fields not given zero.
fn build(isa: Isa, name: &str, set: &[(&str, u64)]) -> Vec<u32> {
    let form = xml::forms(isa)
        .iter()
        .find(|f| f.name == name && f.condition == "default")
        .unwrap_or_else(|| panic!("no default-condition form {name}"));
    let mut d = Decoded::new(isa, form);
    for &(field, value) in set {
        d.set(field, value)
            .unwrap_or_else(|e| panic!("{name}: set {field}={value}: {e}"));
    }
    let (words, n) = d
        .assemble()
        .unwrap_or_else(|e| panic!("{name}: assemble: {e}"));
    words[..n].to_vec()
}

fn xml_of(inst: &Inst) -> &peacemaker_cdna::XmlInst {
    inst.xml()
        .unwrap_or_else(|| panic!("expected a generated instruction, got {inst:?}"))
}

fn span(file: RegFile, base: u8, len: u8) -> RegSpan {
    RegSpan::new(file, base, len)
}

#[test]
fn generated_ds_sub_u32_facts() {
    // ds_sub_u32 v1, v2: not in the typed DS table.
    let words = build(Isa::Cdna3, "DS_SUB_U32", &[("ADDR", 1), ("DATA0", 2)]);
    let (inst, n) = Inst::decode(Arch::Gfx942, &words).unwrap();
    assert_eq!(n, words.len());
    let x = xml_of(&inst);
    assert_eq!(x.name(), "ds_sub_u32");
    assert_eq!(x.encoding(), "ENC_DS");
    assert_eq!(inst.memory(), Some(MemKind::DsAtomic));
    let fp = inst.footprint();
    for want in [
        span(RegFile::Vgpr, 1, 1),
        span(RegFile::Vgpr, 2, 1),
        span(RegFile::Exec, 0, 2),
        span(RegFile::M0, 0, 1),
    ] {
        assert!(fp.uses.contains(&want), "uses {:?} lack {want:?}", fp.uses);
    }
    assert!(fp.defs.is_empty(), "{:?}", fp.defs);
    // Encodes back to the same words when the row is verified, refuses by
    // name when it is not.
    let mut out = Vec::new();
    match inst.encode(Arch::Gfx942, &mut out) {
        Ok(()) => assert_eq!(out, words),
        Err(e) => assert!(e.contains("unverified encoding"), "{e}"),
    }
}

#[test]
fn generated_smem_store_and_branch_facts() {
    // s_store_dword s4, s[2:3], 0x8: a scalar store, typed table has loads only.
    let words = build(
        Isa::Cdna3,
        "S_STORE_DWORD",
        &[("SBASE", 1), ("SDATA", 4), ("IMM", 1), ("OFFSET", 8)],
    );
    let (inst, _) = Inst::decode(Arch::Gfx942, &words).unwrap();
    assert_eq!(xml_of(&inst).name(), "s_store_dword");
    assert_eq!(inst.memory(), Some(MemKind::SmemStore));
    let fp = inst.footprint();
    assert!(fp.uses.contains(&span(RegFile::Sgpr, 4, 1)), "{fp:?}");
    assert!(fp.uses.contains(&span(RegFile::Sgpr, 2, 2)), "{fp:?}");
    assert!(fp.defs.is_empty(), "{fp:?}");

    // s_call_b64 s[0:1], -2: a relative branch the typed SOPP table lacks.
    let words = build(Isa::Cdna3, "S_CALL_B64", &[("SDST", 0), ("SIMM16", 0xfffe)]);
    let (inst, _) = Inst::decode(Arch::Gfx942, &words).unwrap();
    assert_eq!(xml_of(&inst).name(), "s_call_b64");
    assert_eq!(inst.branch_offset(), Some(-2));
    assert!(inst.footprint().defs.contains(&span(RegFile::Sgpr, 0, 2)));
}

/// Every receipt-verified row is decoded from LLVM's independent sample, not
/// a zero-filled operand guess. Unverified rows are counted explicitly.
#[test]
fn every_cdna3_verified_form_decodes_typed_or_generated() {
    let mut verified = 0;
    let mut unverified = 0;
    let mut failures = Vec::new();
    for form in xml::forms(Isa::Cdna3) {
        if !form.verification.is_verified() {
            unverified += 1;
            assert!(form.sample.is_empty(), "{}", form.name);
        } else {
            verified += 1;
            assert!(!form.sample.is_empty(), "{}", form.name);
            match Inst::decode(Arch::Gfx942, form.sample) {
                Ok((inst, n)) => {
                    assert_eq!(n, form.sample.len(), "{}", form.name);
                    match inst.words(Arch::Gfx942) {
                        Ok(words) if words == form.sample => {}
                        other => failures.push(format!("{}: roundtrip {other:?}", form.name)),
                    }
                }
                Err(error) => failures.push(format!("{} {}: {error}", form.name, form.encoding)),
            }
        }
    }
    assert_eq!((verified, unverified), (2435, 30));
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn gfx950_verified_forms_are_decode_only_and_never_typed() {
    assert!(!Arch::Gfx950.encodes());
    let mut verified = 0;
    let mut unverified = 0;
    let mut failures = Vec::new();
    for form in xml::forms(Isa::Cdna4) {
        if !form.verification.is_verified() {
            unverified += 1;
            assert!(form.sample.is_empty(), "{}", form.name);
        } else {
            verified += 1;
            assert!(!form.sample.is_empty(), "{}", form.name);
            match Inst::decode(Arch::Gfx950, form.sample) {
                Ok((Inst::Xml(x), n)) => {
                    assert_eq!(n, form.sample.len(), "{}", form.name);
                    let error = Inst::Xml(x).words(Arch::Gfx950).unwrap_err();
                    assert!(error.contains("decode-only"), "{error}");
                }
                Ok((other, _)) => failures.push(format!("{}: typed {other:?}", form.name)),
                Err(error) => failures.push(format!("{} {}: {error}", form.name, form.encoding)),
            }
        }
    }
    assert_eq!((verified, unverified), (2536, 32));
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn cdna4_scale_identifier_correction_remains_unverified() {
    for (opcode, name) in [
        (0xd3ad_0800, "v_mfma_scale_f32_16x16x128_f8f6f4"),
        (0xd3ae_0800, "v_mfma_scale_f32_32x32x64_f8f6f4"),
    ] {
        let words = [0xd3ac_0000, 0x0002_2310, opcode, 0x0402_1100];
        let (inst, n) = Inst::decode(Arch::Gfx950, &words).unwrap();
        assert_eq!(n, 4);
        let x = xml_of(&inst);
        assert_eq!(x.name(), name);
        assert!(!x.verified());
        assert!(x.unsupported().contains("unverified encoding"));
    }
}

#[test]
fn simulator_refuses_a_generated_instruction_by_name() {
    let words = build(Isa::Cdna3, "DS_SUB_U32", &[("ADDR", 1), ("DATA0", 2)]);
    let (inst, _) = Inst::decode(Arch::Gfx942, &words).unwrap();
    let mut w = Wave::new(Arch::Gfx942, 0, 0);
    let mut lds = vec![0u8; 64];
    let mut mem = Memory::default();
    let err = sim::exec_one(&mut w, &inst, &mut lds, &mut mem, &Config::default()).unwrap_err();
    match err {
        SimError::Unsupported(m) => {
            assert!(m.contains("ds_sub_u32"), "{m}");
            assert!(m.contains("no hand-written model"), "{m}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn unverified_rows_decode_but_never_encode() {
    let mut seen = 0usize;
    for form in xml::forms(Isa::Cdna3) {
        if form.verification.is_verified() {
            continue;
        }
        let d = Decoded::new(Isa::Cdna3, form);
        // The one alias-receipt row (V_MAD_MIX_F32): its canonical instances
        // are promoted per instance, see tests/xml_mix_alias_receipt.rs.
        if XmlInst::alias_receipt_of(&d).is_some() {
            continue;
        }
        let Ok((words, n)) = d.assemble() else {
            continue;
        };
        let Ok((inst, _)) = Inst::decode(Arch::Gfx942, &words[..n]) else {
            continue;
        };
        seen += 1;
        match inst.verification(Arch::Gfx942) {
            RowStatus::Verified => panic!("{}: unverified row reported verified", form.name),
            RowStatus::Unverified(_) | RowStatus::Unknown(_) => {}
        }
        if let Some(x) = inst.xml() {
            let e = inst.encode(Arch::Gfx942, &mut Vec::new()).unwrap_err();
            assert!(e.contains("unverified encoding"), "{e}");
            assert!(x.unsupported().contains("unverified encoding"));
        }
    }
    eprintln!("{seen} unverified CDNA3 rows decoded");
}

/// The F16/F32 compare family is typed (VOPC and VOP3), never the generated fallback.
#[test]
fn cdna3_float_compare_forms_decode_typed() {
    let mut typed = 0;
    for form in xml::forms(Isa::Cdna3) {
        let n = &form.name;
        let float_cmp = n.starts_with("V_CMP")
            && !n.contains("_CLASS_")
            && (n.ends_with("_F16") || n.ends_with("_F32"));
        if !float_cmp || !form.verification.is_verified() || form.sample.is_empty() {
            continue;
        }
        let (inst, _) = Inst::decode(Arch::Gfx942, form.sample)
            .unwrap_or_else(|e| panic!("{n} {}: {e}", form.encoding));
        assert!(
            matches!(inst, Inst::Vopc(_) | Inst::Vop3(_)),
            "{n} {}: {inst:?}",
            form.encoding
        );
        typed += 1;
    }
    assert!(
        typed >= 128,
        "all 64 operations need both VOPC and VOP3 rows: {typed}"
    );
}
