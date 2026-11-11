// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! VOP2 mandatory-literal FMA (`v_fmamk_f32`, `v_fmaak_f32`, gfx942 VOP2
//! opcodes 23 / 24, two dwords) against llvm-mc gfx942, plus the emulator's
//! fused-multiply-add semantics.
//!
//! * `v_fmamk_f32 vdst, src0, K, src1`: `vdst = fma(src0, K, src1)`.
//! * `v_fmaak_f32 vdst, src0, src1, K`: `vdst = fma(src0, src1, K)`.
//!
//! `src0` is a VGPR or an inline constant, `src1`/`vdst` are VGPRs. llvm-mc
//! refuses every SGPR/VCC/EXEC/M0/TTMP `src0` (the literal already occupies
//! the constant bus), a different second literal, AGPRs, DPP, SDWA, VOP3
//! (`_e64`), clamp, omod and neg/abs, and so does the typed encoder.
//!
//! Encoding: our bytes == llvm-mc's bytes for our text, llvm-mc's text == our
//! text, llvm's disassembly of our bytes == our text, decode(bytes) == the
//! instruction. The oracle-free pins hard-code llvm-mc's bytes.
use peacemaker_cdna::arch::Arch;
use peacemaker_cdna::enc::vop::{Vop2K, Vop2KOp};
use peacemaker_cdna::inst::{Footprint, RegFile, RegSpan};
use peacemaker_cdna::operand::InlineFloat;
use peacemaker_cdna::sim::{valu, SimError, Wave};
use peacemaker_cdna::{decode_all, encode_all, oracle, Format, Inst, Src, WAVE};

const A: Arch = Arch::Gfx942;
use Src::*;

fn k(op: Vop2KOp, vdst: Src, src0: Src, src1: Src, literal: u32) -> Inst {
    Inst::Vop2K(Vop2K {
        op,
        vdst,
        src0,
        src1,
        literal,
    })
}
fn mk(vdst: Src, src0: Src, src1: Src, literal: u32) -> Inst {
    k(Vop2KOp::FmamkF32, vdst, src0, src1, literal)
}
fn aak(vdst: Src, src0: Src, src1: Src, literal: u32) -> Inst {
    k(Vop2KOp::FmaakF32, vdst, src0, src1, literal)
}

/// Non-inline literal constants: the exact value is what llvm assembles as a
/// literal dword, including values an inline constant could also represent
/// (1.0, -1.0, 0, 0x40): K is mandatory and never folded.
const LITERALS: [u32; 15] = [
    0x3f00_0001,
    0x1234_5678,
    0x0000_0040,
    0x0000_0000,
    0x8000_0000,
    0x7fc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0xffff_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x3e22_f983,
    0x0000_0001,
    0x007f_ffff,
    0x7f7f_ffff,
];

fn src0s() -> Vec<Src> {
    let mut v = vec![
        Vgpr(0),
        Vgpr(1),
        Vgpr(100),
        Vgpr(255),
        Int(0),
        Int(1),
        Int(-1),
        Int(-16),
        Int(33),
        Int(64),
    ];
    v.extend(InlineFloat::ALL.iter().map(|&f| Float(f)));
    v
}

fn cases() -> Vec<Inst> {
    let mut c = Vec::new();
    let mut n = 0u32;
    for &op in Vop2KOp::ALL {
        for s0 in src0s() {
            for &lit in &LITERALS {
                n += 1;
                let vdst = Vgpr((n * 37 % 256) as u8);
                let src1 = Vgpr((n * 91 % 256) as u8);
                c.push(k(op, vdst, s0, src1, lit));
            }
        }
        // Register aliasing and the file extremes.
        for lit in [0x3f00_0001, 0x7fc0_0000] {
            c.push(k(op, Vgpr(7), Vgpr(7), Vgpr(7), lit));
            c.push(k(op, Vgpr(7), Vgpr(7), Vgpr(8), lit));
            c.push(k(op, Vgpr(7), Vgpr(8), Vgpr(7), lit));
            c.push(k(op, Vgpr(255), Vgpr(255), Vgpr(255), lit));
            c.push(k(op, Vgpr(0), Vgpr(0), Vgpr(0), lit));
            c.push(k(op, Vgpr(255), Float(InlineFloat::InvTwoPi), Vgpr(0), lit));
        }
    }
    c
}

// ----------------------------------------------------------------------------
// Encoding

/// llvm-mc 0x3f000001 pins, from
/// `llvm-mc -arch=amdgcn -mcpu=gfx942 -show-encoding`.
#[test]
fn llvm_byte_pins() {
    let pins: [(Inst, [u32; 2], &str); 6] = [
        (
            mk(Vgpr(0), Vgpr(1), Vgpr(2), 0x3f00_0001),
            [0x2e00_0501, 0x3f00_0001],
            "v_fmamk_f32 v0, v1, 0x3f000001, v2",
        ),
        (
            aak(Vgpr(0), Vgpr(1), Vgpr(2), 0x3f00_0001),
            [0x3000_0501, 0x3f00_0001],
            "v_fmaak_f32 v0, v1, v2, 0x3f000001",
        ),
        (
            mk(Vgpr(0), Float(InlineFloat::Two), Vgpr(2), 0x3f00_0001),
            [0x2e00_04f4, 0x3f00_0001],
            "v_fmamk_f32 v0, 2.0, 0x3f000001, v2",
        ),
        (
            mk(Vgpr(255), Vgpr(255), Vgpr(255), 0x7fc0_0000),
            [0x2fff_ffff, 0x7fc0_0000],
            "v_fmamk_f32 v255, v255, 0x7fc00000, v255",
        ),
        (
            mk(Vgpr(1), Int(1), Vgpr(2), 0xffff_ffff),
            [0x2e02_0481, 0xffff_ffff],
            "v_fmamk_f32 v1, 1, 0xffffffff, v2",
        ),
        (
            aak(Vgpr(1), Int(64), Vgpr(2), 0x1),
            [0x3002_04c0, 0x1],
            "v_fmaak_f32 v1, 64, v2, 0x1",
        ),
    ];
    for (inst, words, text) in pins {
        assert_eq!(inst.words(A).unwrap(), words, "{text}");
        assert_eq!(inst.to_string(), text);
        let (back, n) = Inst::decode(A, &words).unwrap();
        assert_eq!((back, n), (inst.clone(), 2), "{text}");
        assert_eq!(inst.format(), Format::Vop2);
    }
    // Zero and the inline-representable literals still print as hex and
    // keep the literal dword (never folded into an inline constant).
    assert_eq!(
        aak(Vgpr(0), Float(InlineFloat::Half), Vgpr(2), 0).to_string(),
        "v_fmaak_f32 v0, 0.5, v2, 0x0"
    );
    assert_eq!(
        mk(Vgpr(0), Vgpr(1), Vgpr(2), 0x3f80_0000).words(A).unwrap(),
        [0x2e00_0501, 0x3f80_0000]
    );
    assert_eq!(
        aak(Vgpr(0), Vgpr(1), Vgpr(2), 0x8000_0000)
            .words(A)
            .unwrap(),
        [0x3000_0501, 0x8000_0000]
    );
}

#[test]
fn opcode_numbers() {
    assert_eq!(Vop2KOp::ALL.len(), 2);
    assert!(Vop2KOp::ALL.contains(&Vop2KOp::FmamkF32) && Vop2KOp::ALL.contains(&Vop2KOp::FmaakF32));
    assert_eq!(Vop2KOp::FmamkF32.opcode(A), Some(23));
    assert_eq!(Vop2KOp::FmaakF32.opcode(A), Some(24));
    assert_eq!(Vop2KOp::from_opcode(A, 23), Some(Vop2KOp::FmamkF32));
    assert_eq!(Vop2KOp::from_opcode(A, 24), Some(Vop2KOp::FmaakF32));
    for opcode in (0..=63u8).filter(|o| ![23, 24].contains(o)) {
        assert_eq!(Vop2KOp::from_opcode(A, opcode), None, "opcode {opcode}");
    }
    assert_eq!(Vop2KOp::FmamkF32.name(), "v_fmamk_f32");
    assert_eq!(Vop2KOp::FmaakF32.name(), "v_fmaak_f32");
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
        assert_eq!(words.len(), 2, "{inst:?}: exactly two dwords");
        let (back, n) = Inst::decode(A, &words)
            .unwrap_or_else(|e| panic!("decode {inst:?} {words:#010x?}: {e}"));
        assert_eq!(n, 2, "{inst:?}");
        assert_eq!(&back, inst, "decode identity for {words:#010x?}");
        texts.push(inst.to_string());
        bytes.push(oracle::bytes(&words));
    }
    // The concatenated stream decodes back too, two dwords per instruction.
    let stream = encode_all(A, &cases).unwrap();
    assert_eq!(stream.len(), 2 * cases.len());
    let decoded = decode_all(A, &stream).unwrap();
    for (i, (at, inst)) in decoded.iter().enumerate() {
        assert_eq!(*at, 2 * i);
        assert_eq!(inst, &cases[i]);
    }
    assert_eq!(decoded.len(), cases.len());
    eprintln!("enc_vopk: {} instructions", cases.len());

    if oracle::disabled() {
        eprintln!(
            "enc_vopk: ROCm oracle disabled (HIPFIRE_TEST_REQUIRE_ROCM=0); llvm comparison skipped"
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

/// Mixed with neighbouring instructions the literal dword is consumed exactly
/// once: the next instruction starts two dwords later.
#[test]
fn stream_with_neighbours() {
    use peacemaker_cdna::enc::vop::{Vop1, Vop1Op, VopExt};
    let nop = Inst::Vop1(Vop1 {
        op: Vop1Op::MovB32,
        vdst: Vgpr(3),
        src0: Vgpr(4),
        ext: VopExt::None,
    });
    let seq = vec![
        nop.clone(),
        mk(Vgpr(0), Vgpr(1), Vgpr(2), 0x3f00_0001),
        // A literal that looks like a VOP1 instruction word.
        aak(Vgpr(0), Vgpr(1), Vgpr(2), nop.words(A).unwrap()[0]),
        nop.clone(),
    ];
    let words = encode_all(A, &seq).unwrap();
    assert_eq!(words.len(), 1 + 2 + 2 + 1);
    let back = decode_all(A, &words).unwrap();
    assert_eq!(
        back.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
        [0, 1, 3, 5]
    );
    assert_eq!(back.into_iter().map(|(_, i)| i).collect::<Vec<_>>(), seq);
}

/// A VOP2 word with every opcode but 23/24 never decodes to `Vop2K`; 23/24
/// always do, and need their literal dword.
#[test]
fn vop2_dispatch_only_takes_fma_literals() {
    for opcode in 0..62u32 {
        let word = opcode << 25 | 2 << 9 | 0x101;
        match Inst::decode(A, &[word, 0x3f00_0001]) {
            Ok((Inst::Vop2K(i), n)) => {
                assert!(opcode == 23 || opcode == 24, "opcode {opcode} -> {i:?}");
                assert_eq!(n, 2);
                assert_eq!(i.literal, 0x3f00_0001);
            }
            Ok((_, _)) | Err(_) => {
                assert!(opcode != 23 && opcode != 24, "opcode {opcode} lost");
            }
        }
    }
    for word in [0x2e00_0501u32, 0x3000_0501] {
        assert!(
            Inst::decode(A, &[word]).is_err(),
            "{word:#x}: the literal dword is mandatory"
        );
        assert!(decode_all(A, &[word]).is_err());
    }
}

// ----------------------------------------------------------------------------
// Rejections

#[test]
fn illegal_forms_rejected() {
    let lit = 0x3f00_0001;
    let v = Vgpr(1);
    let mut bad: Vec<Inst> = Vec::new();
    for &op in Vop2KOp::ALL {
        let f = |vdst, src0, src1| k(op, vdst, src0, src1, lit);
        // src0: scalar registers/specials hit the constant bus together with
        // the literal; a second literal; AGPR.
        for s0 in [
            Sgpr(0),
            Sgpr(101),
            FlatScratchLo,
            FlatScratchHi,
            XnackMaskLo,
            XnackMaskHi,
            VccLo,
            VccHi,
            Ttmp(0),
            Ttmp(15),
            M0,
            ExecLo,
            ExecHi,
            SharedBase,
            SharedLimit,
            PrivateBase,
            PrivateLimit,
            Vccz,
            Execz,
            Scc,
            Lit(lit),
            Lit(0x3f80_0000),
            Lit(0),
            Lit(0x1234),
            Agpr(0),
            Agpr(255),
        ] {
            bad.push(f(Vgpr(0), s0, v));
        }
        // src1 is a VGPR.
        for s1 in [
            Sgpr(2),
            VccLo,
            M0,
            ExecLo,
            Ttmp(1),
            Int(1),
            Int(0),
            Float(InlineFloat::One),
            Lit(lit),
            Agpr(2),
        ] {
            bad.push(f(Vgpr(0), v, s1));
        }
        // vdst is a VGPR.
        for d in [
            Sgpr(0),
            VccLo,
            M0,
            ExecLo,
            Ttmp(0),
            Int(0),
            Float(InlineFloat::One),
            Lit(5),
            Agpr(1),
        ] {
            bad.push(f(d, v, Vgpr(2)));
        }
    }
    for inst in &bad {
        assert!(inst.words(A).is_err(), "{inst:?} must not encode");
        let mut out = Vec::new();
        assert!(inst.encode(A, &mut out).is_err(), "{inst:?}");
        assert!(out.is_empty(), "{inst:?}: no partial output on error");
        assert!(encode_all(A, std::slice::from_ref(inst)).is_err());
    }
}

/// Words llvm would also reject (or decode as something else): each is an
/// error here, never an approximation.
#[test]
fn illegal_words_rejected() {
    let lit = 0x3f00_0001;
    // vdst 0, vsrc1 v2 (bits 16:9), varying the 9-bit src0 field.
    let w = |op: u32, src0: u32| op << 25 | 2 << 9 | src0;
    for op in [23u32, 24] {
        let ok = w(op, 0x101);
        assert!(Inst::decode(A, &[ok, lit]).is_ok());
        // Constant-bus operands and the literal marker (a second literal).
        let mut bad_codes: Vec<u32> = Vec::new();
        bad_codes.extend(0..=101); // SGPRs
        bad_codes.extend(102..=107); // flat scratch, xnack mask, vcc
        bad_codes.extend(108..=123); // ttmp
        bad_codes.extend([124, 126, 127]); // m0, exec
        bad_codes.extend(235..=238); // apertures
        bad_codes.extend([251, 252, 253]); // vccz, execz, scc
        bad_codes.push(255); // literal marker
                             // 249/250 are the SDWA/DPP markers (no such forms), 254 is
                             // src_lds_direct; neither is a legal operand here.
        bad_codes.extend([249, 250, 254]);
        // Reserved holes in the source-code space.
        bad_codes.extend([125, 209, 234, 239]);
        for code in bad_codes {
            assert!(
                Inst::decode(A, &[w(op, code), lit]).is_err(),
                "opcode {op} src0 code {code} must be rejected"
            );
        }
        // Inline constants and VGPRs are legal at both ends of their ranges.
        for code in [128, 192, 193, 208, 240, 248, 256, 511] {
            let (i, n) = Inst::decode(A, &[w(op, code), lit]).unwrap_or_else(|e| {
                panic!("opcode {op} src0 code {code}: {e}");
            });
            assert_eq!(n, 2);
            assert_eq!(i.words(A).unwrap(), [w(op, code), lit]);
        }
    }
    // vdst/vsrc1 are 8-bit VGPR fields, so the whole field range decodes.
    for field in [0u32, 1, 127, 128, 255] {
        let (i, _) = Inst::decode(A, &[23 << 25 | field << 17 | field << 9 | 0x101, lit]).unwrap();
        assert_eq!(
            i,
            mk(Vgpr(field as u8), Vgpr(1), Vgpr(field as u8), lit),
            "field {field}"
        );
    }
}

/// llvm-mc refuses each of these spellings, which is why the typed encoder
/// has no field for them. (One `llvm-mc` run per form: the batch API fails as
/// a whole.)
#[test]
fn llvm_refuses_the_unsupported_forms() {
    if oracle::disabled() {
        eprintln!("enc_vopk: ROCm oracle disabled; llvm rejection check skipped");
        return;
    }
    let refused = [
        // constant bus: SGPR/special src0 + the mandatory literal
        "v_fmamk_f32 v0, s1, 0x3f000001, v2",
        "v_fmaak_f32 v0, s1, v2, 0x3f000001",
        "v_fmamk_f32 v0, vcc_lo, 0x3f000001, v2",
        "v_fmaak_f32 v0, m0, v2, 0x3f000001",
        "v_fmamk_f32 v0, exec_lo, 0x3f000001, v2",
        "v_fmamk_f32 v0, ttmp0, 0x3f000001, v2",
        "v_fmamk_f32 v0, flat_scratch_lo, 0x3f000001, v2",
        "v_fmamk_f32 v0, src_scc, 0x3f000001, v2",
        "v_fmamk_f32 v0, execz, 0x3f000001, v2",
        "v_fmaak_f32 v0, src_shared_base, v2, 0x3f000001",
        // a different second literal
        "v_fmamk_f32 v0, 0x1111, 0x3f000001, v2",
        // operand classes
        "v_fmamk_f32 a0, v1, 0x3f000001, v2",
        "v_fmamk_f32 s0, v1, 0x3f000001, v2",
        "v_fmamk_f32 v0, v1, 0x3f000001, s2",
        "v_fmamk_f32 v0, v1, 0x3f000001, 5",
        "v_fmaak_f32 v0, v1, 5, 0x3f000001",
        "v_fmaak_f32 v0, v1, v2, v3",
        "v_fmaak_f32 v0, v1, v2",
        // literal wider than 32 bits
        "v_fmaak_f32 v0, v1, v2, 0x100000000",
        // modifiers and other encodings
        "v_fmamk_f32 v0, v1, 0x3f000001, v2 clamp",
        "v_fmamk_f32 v0, v1, 0x3f000001, v2 mul:2",
        "v_fmaak_f32 v0, -v1, v2, 0x3f000001",
        "v_fmaak_f32 v0, v1, |v2|, 0x3f000001",
        "v_fmaak_f32 v0, abs(v1), v2, 0x3f000001",
        "v_fmaak_f32 v0, v1, v2, 0x3f000001 op_sel:[1,0]",
        "v_fmamk_f32 v0, v1, 0x3f000001, v2 row_shr:1",
        "v_fmaak_f32_dpp v0, v1, v2, 0x3f000001 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
        "v_fmamk_f32_sdwa v0, v1, 0x3f000001, v2",
        "v_fmamk_f32 v0, v1, 0x3f000001, v2 src0_sel:WORD_1",
        "v_fmamk_f32_e64 v0, v1, 0x3f000001, v2",
    ];
    let mut accepted = Vec::new();
    for text in refused {
        if oracle::mc_encode(A, &[text.to_string()]).is_ok() {
            accepted.push(text);
        }
    }
    assert!(accepted.is_empty(), "llvm-mc accepted: {accepted:#?}");
    // The one llvm form we deliberately do not model: the same literal as
    // src0 and K. Documented so a change in llvm shows up here.
    let same = oracle::mc_encode(
        A,
        &["v_fmamk_f32 v0, 0x3f000001, 0x3f000001, v2".to_string()],
    );
    assert!(same.is_ok(), "llvm no longer accepts one shared literal");
}

// ----------------------------------------------------------------------------
// Footprint

fn has(spans: &[RegSpan], file: RegFile, base: u8, len: u8) -> bool {
    spans.contains(&RegSpan::new(file, base, len))
}

fn check_footprint(f: &Footprint, uses: &[(RegFile, u8)], defs: &[(RegFile, u8)]) {
    assert_eq!(f.uses.len(), uses.len(), "{f:?}");
    assert_eq!(f.defs.len(), defs.len(), "{f:?}");
    for &(file, base) in uses {
        assert!(
            has(
                &f.uses,
                file,
                base,
                if file == RegFile::Exec { 2 } else { 1 }
            ),
            "{f:?}"
        );
    }
    for &(file, base) in defs {
        assert!(has(&f.defs, file, base, 1), "{f:?}");
    }
}

#[test]
fn footprints() {
    for &op in Vop2KOp::ALL {
        let f = k(op, Vgpr(1), Vgpr(2), Vgpr(3), 0x3f00_0001).footprint();
        check_footprint(
            &f,
            &[(RegFile::Exec, 0), (RegFile::Vgpr, 2), (RegFile::Vgpr, 3)],
            &[(RegFile::Vgpr, 1)],
        );
        assert!(
            !has(&f.uses, RegFile::Vgpr, 1, 1),
            "{op:?}: vdst is not read"
        );
        assert!(!f.uses.iter().any(|r| r.file == RegFile::Vcc));
        assert!(!f
            .defs
            .iter()
            .any(|r| r.file == RegFile::Vcc || r.file == RegFile::Exec));

        // Inline src0 reads nothing; the literal reads nothing.
        for s0 in [Int(5), Float(InlineFloat::Two)] {
            let f = k(op, Vgpr(9), s0, Vgpr(3), 0x3f00_0001).footprint();
            check_footprint(
                &f,
                &[(RegFile::Exec, 0), (RegFile::Vgpr, 3)],
                &[(RegFile::Vgpr, 9)],
            );
        }

        // vdst aliasing a source reads that source (pre-instruction value)
        // and still defines it.
        let f = k(op, Vgpr(3), Vgpr(2), Vgpr(3), 7).footprint();
        check_footprint(
            &f,
            &[(RegFile::Exec, 0), (RegFile::Vgpr, 2), (RegFile::Vgpr, 3)],
            &[(RegFile::Vgpr, 3)],
        );
        let f = k(op, Vgpr(2), Vgpr(2), Vgpr(3), 7).footprint();
        check_footprint(
            &f,
            &[(RegFile::Exec, 0), (RegFile::Vgpr, 2), (RegFile::Vgpr, 3)],
            &[(RegFile::Vgpr, 2)],
        );
    }
}

// ----------------------------------------------------------------------------
// Emulator semantics: fused f32 multiply-add (single rounding), literal
// operand roles, EXEC, register aliasing, FP mode.

const POISON: u32 = 0xdead_beef;

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
    fn exec(&mut self) -> u64 {
        match self.next() % 6 {
            0 => u64::MAX,
            1 => 0x0000_ffff_0000_0001,
            _ => self.next() | 1,
        }
    }
    /// f32 bits: specials (NaN, Inf, denormals, -0, extremes), random bit
    /// patterns, and moderate values where products and sums interact.
    fn f32(&mut self) -> u32 {
        const SPECIAL: [u32; 27] = [
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
        match self.next() % 4 {
            0 => self.pick(&SPECIAL),
            1 => self.u32(),
            _ => {
                (self.u32() & 0x8000_0000)
                    | ((100 + self.next() % 56) as u32) << 23
                    | (self.u32() & 0x7f_ffff)
            }
        }
    }
}

fn wave() -> Wave {
    Wave::new(A, 0, POISON)
}

fn fv(b: u32) -> f32 {
    f32::from_bits(b)
}

/// MI300X literal-FMA policy: invalid zero*infinity precedes a NaN addend;
/// otherwise quiet the first NaN source (S0, S1, S2), or use the fused result.
fn ar(ins: &[u32], r: f32) -> u32 {
    let (a, b) = (ins[0] & 0x7fff_ffff, ins[1] & 0x7fff_ffff);
    if (a == 0 && b == 0x7f80_0000) || (b == 0 && a == 0x7f80_0000) {
        return 0xffc0_0000;
    }
    if let Some(&b) = ins.iter().find(|&&b| fv(b).is_nan()) {
        return b | 0x40_0000;
    }
    if r.is_nan() {
        0xffc0_0000
    } else {
        r.to_bits()
    }
}

/// Bits an f32 operation reads for `src0` given the VGPR value for lanes.
fn src0_bits(src0: Src, vgpr: u32) -> u32 {
    match src0 {
        Vgpr(_) => vgpr,
        Int(i) => i32::from(i) as u32,
        Float(f) => f.f32_bits(),
        other => panic!("{other:?}"),
    }
}

/// `fma(S0, S1, S2)` over the instruction's operand roles.
fn reference(op: Vop2KOp, src0: u32, src1: u32, lit: u32) -> u32 {
    let (a, b, c) = match op {
        Vop2KOp::FmamkF32 => (src0, lit, src1),
        Vop2KOp::FmaakF32 => (src0, src1, lit),
    };
    ar(&[a, b, c], fv(a).mul_add(fv(b), fv(c)))
}

fn run(w: &mut Wave, inst: &Inst) {
    if let Err(e) = valu::exec(w, inst) {
        panic!("{inst:?}: {e}")
    }
}

#[test]
fn sim_operand_roles() {
    // 2, 3, 5: fmamk = 2*K(3)+5 = 11, fmaak = 2*3+K(5) = 11; swapping the
    // addend with a factor gives 13.
    let (two, three, five) = (2.0f32.to_bits(), 3.0f32.to_bits(), 5.0f32.to_bits());
    let mut w = wave();
    w.v[10][0] = two;
    w.v[11][0] = three;
    w.v[12][0] = five;
    w.exec = 1;
    run(&mut w, &mk(Vgpr(1), Vgpr(10), Vgpr(12), three));
    assert_eq!(fv(w.v[1][0]), 11.0, "fmamk: src0 * K + src1");
    run(&mut w, &aak(Vgpr(2), Vgpr(10), Vgpr(11), five));
    assert_eq!(fv(w.v[2][0]), 11.0, "fmaak: src0 * src1 + K");
    // vdst is write-only: the poison NaN it held never reaches the result.
    assert_eq!(w.v[1][0], 11.0f32.to_bits());
    // K is the literal dword even when its value is an inline constant.
    run(&mut w, &mk(Vgpr(3), Vgpr(10), Vgpr(12), 0x3f80_0000));
    assert_eq!(fv(w.v[3][0]), 7.0, "K = 1.0f: 2*1+5");
    run(&mut w, &aak(Vgpr(4), Vgpr(10), Vgpr(11), 0));
    assert_eq!(fv(w.v[4][0]), 6.0, "K = 0: 2*3+0");
    // Inline src0: 2.0 and -4.0 are floats; integer inline constants are
    // raw bit patterns (1 is the smallest denormal, not 1.0).
    run(
        &mut w,
        &mk(Vgpr(5), Float(InlineFloat::Two), Vgpr(12), three),
    );
    assert_eq!(fv(w.v[5][0]), 11.0);
    run(
        &mut w,
        &aak(Vgpr(6), Float(InlineFloat::NegFour), Vgpr(11), five),
    );
    assert_eq!(fv(w.v[6][0]), -7.0, "-4*3+5");
    run(&mut w, &mk(Vgpr(7), Int(1), Vgpr(12), three));
    assert_eq!(fv(w.v[7][0]), 5.0, "denormal * 3 + 5 rounds to 5");
}

#[test]
fn sim_is_fused_single_rounding() {
    let (a, b, c) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON, -1.0f32);
    let mut w = wave();
    w.exec = 1;
    w.v[10][0] = a.to_bits();
    w.v[11][0] = b.to_bits();
    w.v[12][0] = c.to_bits();
    // fmamk: a * K(b) + c
    run(&mut w, &mk(Vgpr(1), Vgpr(10), Vgpr(12), b.to_bits()));
    // fmaak: a * b + K(c)
    run(&mut w, &aak(Vgpr(2), Vgpr(10), Vgpr(11), c.to_bits()));
    for r in [w.v[1][0], w.v[2][0]] {
        assert_eq!(r, a.mul_add(b, c).to_bits());
        assert_ne!(r, (a * b + c).to_bits(), "fused, not two roundings");
        assert_eq!(fv(r), -f32::EPSILON * f32::EPSILON);
    }
    // A fused op preserves the exact residual of a product: x*y - round(x*y)
    // is nonzero here, while two roundings give 0.
    let big = 2.0f32.powi(40);
    let (x, y) = (big * (1.0 + f32::EPSILON), big * (1.0 - f32::EPSILON));
    w.v[10][0] = x.to_bits();
    w.v[12][0] = (-(x * y)).to_bits();
    run(&mut w, &mk(Vgpr(3), Vgpr(10), Vgpr(12), y.to_bits()));
    assert_eq!(w.v[3][0], x.mul_add(y, -(x * y)).to_bits());
    assert_ne!(fv(w.v[3][0]), 0.0, "residual survives");
}

#[test]
fn sim_exec_and_aliasing() {
    for &op in Vop2KOp::ALL {
        let mut w = wave();
        w.exec = 0x00ff_00ff_00ff_00ff;
        let lit = 0x4049_0fdb; // pi, not inline
        for l in 0..WAVE {
            w.v[10][l] = (1.5 + l as f32).to_bits();
            w.v[11][l] = (0.25 * l as f32 - 3.0).to_bits();
        }
        // Distinct vdst: inactive lanes keep the poison, active lanes compute.
        run(&mut w, &k(op, Vgpr(1), Vgpr(10), Vgpr(11), lit));
        for l in 0..WAVE {
            let want = if w.exec >> l & 1 != 0 {
                reference(op, w.v[10][l], w.v[11][l], lit)
            } else {
                POISON
            };
            assert_eq!(w.v[1][l], want, "{op:?} lane {l}");
        }
        // vdst == src0, vdst == src1, vdst == both: pre-instruction values.
        let (s0, s1): (Vec<u32>, Vec<u32>) = (0..WAVE).map(|l| (w.v[10][l], w.v[11][l])).unzip();
        run(&mut w, &k(op, Vgpr(10), Vgpr(10), Vgpr(11), lit));
        run(&mut w, &k(op, Vgpr(11), Vgpr(1), Vgpr(11), lit));
        // (src0 = vdst = v10 above overwrote v10; v1 still holds the first result.)
        for l in 0..WAVE {
            if w.exec >> l & 1 == 0 {
                assert_eq!((w.v[10][l], w.v[11][l]), (s0[l], s1[l]), "{op:?} lane {l}");
                continue;
            }
            assert_eq!(
                w.v[10][l],
                reference(op, s0[l], s1[l], lit),
                "{op:?} lane {l} (vdst = src0)"
            );
            let first = reference(op, s0[l], s1[l], lit);
            assert_eq!(
                w.v[11][l],
                reference(op, first, s1[l], lit),
                "{op:?} lane {l} (vdst = src1)"
            );
        }
        let mut w = wave();
        w.exec = u64::MAX;
        for l in 0..WAVE {
            w.v[5][l] = (l as f32 - 20.0).to_bits();
        }
        let before = w.v[5];
        run(&mut w, &k(op, Vgpr(5), Vgpr(5), Vgpr(5), lit));
        for l in 0..WAVE {
            assert_eq!(
                w.v[5][l],
                reference(op, before[l], before[l], lit),
                "{op:?} lane {l} (all one reg)"
            );
        }
        // EXEC = 0 writes nothing.
        let mut w = wave();
        w.exec = 0;
        let before = w.v.clone();
        run(&mut w, &k(op, Vgpr(1), Vgpr(2), Vgpr(3), lit));
        assert!(w.v == before);
    }
}

#[test]
fn sim_denormals_nan_inf() {
    let mut w = wave();
    w.exec = 1;
    let tiny = 0x0000_0001u32;
    // Default MODE preserves denormals: 1ulp_denorm * 1.0 + 1ulp_denorm = 2.
    w.v[10][0] = tiny;
    w.v[11][0] = tiny;
    run(&mut w, &mk(Vgpr(1), Vgpr(10), Vgpr(11), 0x3f80_0000));
    assert_eq!(w.v[1][0], 2, "denormals are not flushed");
    run(&mut w, &aak(Vgpr(2), Vgpr(10), Vgpr(11), 0x8000_0001));
    assert_eq!(
        w.v[2][0],
        ar(
            &[tiny, tiny, 0x8000_0001],
            fv(tiny).mul_add(fv(tiny), fv(0x8000_0001))
        ),
        "tiny*tiny + -tiny"
    );
    assert_eq!(w.v[2][0], 0x8000_0001, "tiny*tiny underflows below -tiny");
    // Cancellation to zero is +0 under RNE.
    w.v[12][0] = 1.0f32.to_bits();
    run(
        &mut w,
        &aak(Vgpr(3), Vgpr(12), Vgpr(12), (-1.0f32).to_bits()),
    );
    assert_eq!(w.v[3][0], 0);
    // Invalid operation without NaN input: default NaN.
    w.v[13][0] = 0; // 0 * Inf
    run(&mut w, &mk(Vgpr(4), Vgpr(13), Vgpr(12), 0x7f80_0000));
    assert_eq!(w.v[4][0], 0xffc0_0000);
    // Inf - Inf
    w.v[14][0] = 0xff80_0000;
    run(&mut w, &aak(Vgpr(5), Vgpr(12), Vgpr(14), 0x7f80_0000));
    assert_eq!(w.v[5][0], 0xffc0_0000);
    // Hardware receipt: invalid product dominates quiet/signalling NaN addends.
    w.v[17][0] = 0;
    w.v[18][0] = 0x7fa0_0001;
    run(&mut w, &mk(Vgpr(19), Vgpr(17), Vgpr(18), 0x7f80_0000));
    assert_eq!(w.v[19][0], 0xffc0_0000);
    w.v[18][0] = 0x7f80_0000;
    run(&mut w, &aak(Vgpr(19), Vgpr(17), Vgpr(18), 0x7fc0_0001));
    assert_eq!(w.v[19][0], 0xffc0_0000);
    // A signalling NaN in any operand slot is returned quieted.
    run(&mut w, &mk(Vgpr(6), Vgpr(12), Vgpr(12), 0x7fa0_0000));
    assert_eq!(w.v[6][0], 0x7fe0_0000, "NaN literal (fmamk K slot)");
    w.v[15][0] = 0xff80_0001;
    run(&mut w, &mk(Vgpr(7), Vgpr(15), Vgpr(12), 0x3f80_0000));
    assert_eq!(w.v[7][0], 0xffc0_0001, "NaN src0");
    run(&mut w, &aak(Vgpr(8), Vgpr(12), Vgpr(15), 0x3f80_0000));
    assert_eq!(w.v[8][0], 0xffc0_0001, "NaN src1");
    run(&mut w, &aak(Vgpr(9), Vgpr(12), Vgpr(12), 0x7fa0_0000));
    assert_eq!(w.v[9][0], 0x7fe0_0000, "NaN literal (fmaak K slot)");
    // Finite overflow and Inf propagation.
    w.v[16][0] = 0x7f7f_ffff;
    run(&mut w, &mk(Vgpr(10), Vgpr(16), Vgpr(12), 0x4000_0000));
    assert_eq!(w.v[10][0], 0x7f80_0000);
}

#[test]
fn sim_random_sweep() {
    let mut r = Rng(0x0fa1_c0de_f00d_5eed);
    for round in 0..300 {
        for &op in Vop2KOp::ALL {
            let mut w = wave();
            w.exec = r.exec();
            let src0 = match r.next() % 4 {
                0 => Int((r.next() % 81) as i8 - 16),
                1 => Float(r.pick(&InlineFloat::ALL)),
                _ => Vgpr(10),
            };
            let lit = if r.next() % 5 == 0 {
                r.pick(&[
                    0x3f80_0000,
                    0,
                    0x8000_0000,
                    0x7fc0_0000,
                    0x7f80_0000,
                    0x0000_0001,
                ])
            } else {
                r.f32()
            };
            let mut ins = [[0u32; 2]; WAVE];
            for l in 0..WAVE {
                ins[l] = [r.f32(), r.f32()];
                w.v[10][l] = ins[l][0];
                w.v[11][l] = ins[l][1];
            }
            // vdst: separate, or aliasing src1 / src0.
            let vdst = match (r.next() % 3, src0) {
                (0, _) => 11,
                (1, Vgpr(_)) => 10,
                _ => 20,
            };
            let inst = k(op, Vgpr(vdst), src0, Vgpr(11), lit);
            run(&mut w, &inst);
            for l in 0..WAVE {
                let lane_exec = w.exec >> l & 1 != 0;
                let want = if lane_exec {
                    reference(op, src0_bits(src0, ins[l][0]), ins[l][1], lit)
                } else if vdst == 11 {
                    ins[l][1]
                } else if vdst == 10 {
                    ins[l][0]
                } else {
                    POISON
                };
                assert_eq!(
                    w.v[usize::from(vdst)][l],
                    want,
                    "round {round} {inst:?} lane {l} src0 {:#010x} src1 {:#010x}",
                    src0_bits(src0, ins[l][0]),
                    ins[l][1]
                );
            }
        }
    }
}

/// Only RNE with denormals preserved is modelled; any other FP mode must
/// fail closed (the same path as `v_fma_f32`), leaving vdst untouched.
#[test]
fn sim_fp_mode_fails_closed() {
    for &op in Vop2KOp::ALL {
        for tweak in 0..4 {
            let mut w = wave();
            w.exec = 1;
            w.v[10][0] = 0x0000_0003;
            w.v[11][0] = 0x3f80_0000;
            match tweak {
                0 => w.mode.denorm32 = 0,
                1 => w.mode.denorm32 = 1,
                2 => w.mode.denorm32 = 2,
                _ => w.mode.round32 = 1,
            }
            let inst = k(op, Vgpr(1), Vgpr(10), Vgpr(11), 0x0000_0001);
            let r = valu::exec(&mut w, &inst);
            assert!(
                matches!(r, Err(SimError::Unsupported(_))),
                "{op:?} tweak {tweak}: non-default FP32 mode must be Unsupported"
            );
            assert_eq!(w.v[1][0], POISON, "{op:?} tweak {tweak}: vdst untouched");
        }
        // FP16/64 mode bits do not affect an f32 op.
        let mut w = wave();
        w.exec = 1;
        w.mode.denorm16_64 = 0;
        w.mode.round16_64 = 3;
        w.v[10][0] = 2.0f32.to_bits();
        w.v[11][0] = 1.0f32.to_bits();
        run(
            &mut w,
            &k(op, Vgpr(1), Vgpr(10), Vgpr(11), 3.0f32.to_bits()),
        );
        assert_eq!(
            fv(w.v[1][0]),
            if op == Vop2KOp::FmamkF32 { 7.0 } else { 5.0 }
        );
        // IEEE/DX10 clamp bits are irrelevant (no omod, no clamp).
        let mut w = wave();
        w.exec = 1;
        w.mode.ieee = false;
        w.mode.dx10_clamp = false;
        w.v[10][0] = 2.0f32.to_bits();
        w.v[11][0] = 1.0f32.to_bits();
        run(
            &mut w,
            &k(op, Vgpr(1), Vgpr(10), Vgpr(11), 3.0f32.to_bits()),
        );
        assert_eq!(
            fv(w.v[1][0]),
            if op == Vop2KOp::FmamkF32 { 7.0 } else { 5.0 }
        );
    }
}
