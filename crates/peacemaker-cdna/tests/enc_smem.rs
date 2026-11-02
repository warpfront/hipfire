// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SMEM encoder/decoder/printer against the llvm-mc gfx942 oracle.
use peacemaker_cdna::enc::smem::{Smem, SmemOffset, SmemOp};
use peacemaker_cdna::inst::{RegFile, RegSpan};
use peacemaker_cdna::oracle;
use peacemaker_cdna::{Arch, Src};

const ARCH: Arch = Arch::Gfx942;

fn encode(i: &Smem) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    i.encode(ARCH, &mut out)?;
    Ok(out)
}

/// Offset forms legal for the op family: `s_load_*` takes signed 21-bit
/// immediates, `s_buffer_load_*` unsigned 20-bit ones.
fn offsets(buffer: bool) -> Vec<SmemOffset> {
    use SmemOffset::*;
    let all = vec![
        Imm(0),
        Imm(0x10),
        Imm(-4),
        Imm(0xf_ffff),
        Imm(-0x10_0000),
        Sgpr(Src::Sgpr(4)),
        Sgpr(Src::Sgpr(101)),
        Sgpr(Src::M0),
        Sgpr(Src::Ttmp(3)),
        Sgpr(Src::VccLo),
        Sgpr(Src::ExecHi),
        SgprImm(Src::Sgpr(4), 0),
        SgprImm(Src::Sgpr(4), 0x10),
        SgprImm(Src::M0, -0x10),
        SgprImm(Src::Ttmp(15), 0xf_ffff),
        SgprImm(Src::Sgpr(101), -0x10_0000),
        SgprImm(Src::VccHi, 4),
    ];
    let signed = |o: &SmemOffset| matches!(o, Imm(n) | SgprImm(_, n) if *n < 0);
    all.into_iter().filter(|o| !(buffer && signed(o))).collect()
}

/// (sdata, sbase) pairs legal for `op`.
fn operands(op: SmemOp) -> Vec<(Option<Src>, Option<Src>)> {
    let w = op.data_dwords();
    let bases: Vec<Src> = if op.buffer() {
        vec![
            Src::Sgpr(4),
            Src::Sgpr(0),
            Src::Sgpr(96),
            Src::Ttmp(4),
            Src::Ttmp(12),
        ]
    } else {
        vec![
            Src::Sgpr(2),
            Src::Sgpr(0),
            Src::Sgpr(100),
            Src::Ttmp(2),
            Src::VccLo,
            Src::ExecLo,
            Src::FlatScratchLo,
            Src::XnackMaskLo,
        ]
    };
    let datas: Vec<Src> = match w {
        1 => vec![
            Src::Sgpr(0),
            Src::Sgpr(101),
            Src::Ttmp(7),
            Src::VccLo,
            Src::VccHi,
            Src::FlatScratchLo,
            Src::XnackMaskHi,
        ],
        2 => vec![
            Src::Sgpr(0),
            Src::Sgpr(100),
            Src::Ttmp(2),
            Src::VccLo,
            Src::FlatScratchLo,
            Src::XnackMaskLo,
        ],
        4 => vec![Src::Sgpr(4), Src::Sgpr(96), Src::Ttmp(12)],
        8 => vec![Src::Sgpr(8), Src::Sgpr(88), Src::Ttmp(8)],
        16 => vec![Src::Sgpr(16), Src::Sgpr(80), Src::Ttmp(0)],
        _ => unreachable!(),
    };
    let mut out = Vec::new();
    for (i, d) in datas.iter().enumerate() {
        // pair every sdata with the first base; the rest of the bases with the first sdata
        out.push((Some(*d), Some(bases[i % bases.len()])));
    }
    for b in &bases {
        out.push((Some(datas[0]), Some(*b)))
    }
    out
}

fn cases() -> Vec<Smem> {
    let mut v = Vec::new();
    for &op in SmemOp::ALL {
        if op.has_address() {
            for (sdata, sbase) in operands(op) {
                for off in offsets(op.buffer()) {
                    for glc in [false, true] {
                        v.push(Smem {
                            op,
                            sdata,
                            sbase,
                            offset: off,
                            glc,
                        });
                    }
                }
            }
        } else if op.data_dwords() == 0 {
            v.push(Smem {
                op,
                sdata: None,
                sbase: None,
                offset: SmemOffset::Imm(0),
                glc: false,
            });
        } else {
            for d in [
                Src::Sgpr(0),
                Src::Sgpr(2),
                Src::Sgpr(100),
                Src::Ttmp(2),
                Src::VccLo,
                Src::FlatScratchLo,
            ] {
                v.push(Smem {
                    op,
                    sdata: Some(d),
                    sbase: None,
                    offset: SmemOffset::Imm(0),
                    glc: false,
                });
            }
        }
    }
    v.sort_by_key(|i| format!("{i}"));
    v.dedup();
    v
}

#[test]
fn matches_llvm_mc() {
    if oracle::disabled() {
        eprintln!("enc_smem: oracle disabled");
        return;
    }
    let cases = cases();
    assert!(cases.len() > 1000);
    let texts: Vec<String> = cases.iter().map(|i| i.to_string()).collect();
    let assembled =
        oracle::mc_encode(ARCH, &texts).expect("llvm-mc accepts every printed instruction");
    let mine: Vec<Vec<u8>> = cases
        .iter()
        .map(|i| oracle::bytes(&encode(i).unwrap_or_else(|e| panic!("{i}: {e}"))))
        .collect();
    let disassembled =
        oracle::mc_disassemble(ARCH, &mine).expect("llvm-mc disassembles every encoding");
    for (n, inst) in cases.iter().enumerate() {
        assert_eq!(assembled[n].1, mine[n], "bytes of `{inst}`");
        assert_eq!(assembled[n].0, texts[n], "llvm-mc text of `{inst}`");
        assert_eq!(disassembled[n], texts[n], "disassembly of `{inst}`");
        let words: Vec<u32> = mine[n]
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (back, used) =
            Smem::decode(ARCH, &words).unwrap_or_else(|e| panic!("decode `{inst}`: {e}"));
        assert_eq!(used, 2);
        assert_eq!(&back, inst, "decode(encode(`{inst}`))");
    }
}

#[test]
fn every_op_and_offset_form_is_covered() {
    let cases = cases();
    for &op in SmemOp::ALL {
        assert!(cases.iter().any(|i| i.op == op), "{op:?} missing");
    }
    for &imm in &[0, 0x10, -4, 0xf_ffff, -0x10_0000] {
        assert!(
            cases.iter().any(|i| i.offset == SmemOffset::Imm(imm)),
            "Imm({imm})"
        );
    }
    for &imm in &[0, 0x10, 4, -0x10, 0xf_ffff, -0x10_0000] {
        assert!(
            cases
                .iter()
                .any(|i| matches!(i.offset, SmemOffset::SgprImm(_, n) if n == imm)),
            "SgprImm(_, {imm})"
        );
    }
    for &imm in &[0, 0x10, 0xf_ffff] {
        assert!(
            cases
                .iter()
                .any(|i| i.op.buffer() && i.offset == SmemOffset::Imm(imm)),
            "buffer Imm({imm})"
        );
        assert!(cases.iter().any(|i| i.op.buffer() && matches!(i.offset, SmemOffset::SgprImm(_, n) if n == imm)), "buffer SgprImm(_, {imm})");
    }
    assert!(cases
        .iter()
        .any(|i| matches!(i.offset, SmemOffset::Sgpr(_))));
    assert!(cases.iter().any(|i| i.glc) && cases.iter().any(|i| i.op.has_address() && !i.glc));
}

#[test]
fn data_dwords_table() {
    use SmemOp::*;
    let want = [
        (LoadDword, 1),
        (LoadDwordx2, 2),
        (LoadDwordx4, 4),
        (LoadDwordx8, 8),
        (LoadDwordx16, 16),
        (BufferLoadDword, 1),
        (BufferLoadDwordx2, 2),
        (BufferLoadDwordx4, 4),
        (BufferLoadDwordx8, 8),
        (BufferLoadDwordx16, 16),
        (Memtime, 2),
        (Memrealtime, 2),
        (DcacheInv, 0),
    ];
    assert_eq!(want.len(), SmemOp::ALL.len());
    for (op, n) in want {
        assert_eq!(op.data_dwords(), n, "{op:?}")
    }
}

fn load(op: SmemOp, sdata: Src, sbase: Src, offset: SmemOffset) -> Smem {
    Smem {
        op,
        sdata: Some(sdata),
        sbase: Some(sbase),
        offset,
        glc: false,
    }
}

#[test]
fn rejects_out_of_range_offsets() {
    for bad in [0x10_0000, -0x10_0001, i32::MAX, i32::MIN] {
        let a = load(
            SmemOp::LoadDword,
            Src::Sgpr(0),
            Src::Sgpr(2),
            SmemOffset::Imm(bad),
        );
        assert!(encode(&a).is_err(), "Imm({bad})");
        let b = load(
            SmemOp::LoadDword,
            Src::Sgpr(0),
            Src::Sgpr(2),
            SmemOffset::SgprImm(Src::Sgpr(4), bad),
        );
        assert!(encode(&b).is_err(), "SgprImm({bad})");
    }
    // s_buffer_load_* takes an unsigned 20-bit offset
    for bad in [-1, -0x10_0000, 0x10_0000] {
        let a = load(
            SmemOp::BufferLoadDword,
            Src::Sgpr(0),
            Src::Sgpr(4),
            SmemOffset::Imm(bad),
        );
        assert!(encode(&a).is_err(), "buffer Imm({bad})");
        let b = load(
            SmemOp::BufferLoadDword,
            Src::Sgpr(0),
            Src::Sgpr(4),
            SmemOffset::SgprImm(Src::Sgpr(4), bad),
        );
        assert!(encode(&b).is_err(), "buffer SgprImm({bad})");
    }
    assert!(encode(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::Sgpr(4),
        SmemOffset::Imm(0xf_ffff)
    ))
    .is_ok());
    // the bounds themselves are accepted
    for ok in [0xf_ffff, -0x10_0000] {
        assert!(encode(&load(
            SmemOp::LoadDword,
            Src::Sgpr(0),
            Src::Sgpr(2),
            SmemOffset::Imm(ok)
        ))
        .is_ok());
    }
}

#[test]
fn rejects_illegal_operands() {
    let imm = SmemOffset::Imm(0);
    // misaligned sbase
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::Sgpr(1), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::Ttmp(1), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::VccHi, imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::Sgpr(101), imm)).is_err());
    assert!(encode(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::Sgpr(2),
        imm
    ))
    .is_err());
    assert!(encode(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::Sgpr(98),
        imm
    ))
    .is_err());
    assert!(encode(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::VccLo,
        imm
    ))
    .is_err());
    assert!(encode(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::ExecLo,
        imm
    ))
    .is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::M0, imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Sgpr(0), Src::Vgpr(0), imm)).is_err());
    // misaligned or illegal sdata
    assert!(encode(&load(SmemOp::LoadDwordx2, Src::Sgpr(1), Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDwordx4, Src::Sgpr(2), Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDwordx8, Src::Sgpr(2), Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDwordx16, Src::Sgpr(6), Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(
        SmemOp::LoadDwordx16,
        Src::Sgpr(96),
        Src::Sgpr(2),
        imm
    ))
    .is_err());
    assert!(encode(&load(
        SmemOp::LoadDwordx2,
        Src::Sgpr(101),
        Src::Sgpr(2),
        imm
    ))
    .is_err());
    assert!(encode(&load(SmemOp::LoadDwordx2, Src::VccHi, Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::M0, Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::ExecLo, Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Vgpr(0), Src::Sgpr(2), imm)).is_err());
    assert!(encode(&load(SmemOp::LoadDword, Src::Lit(1), Src::Sgpr(2), imm)).is_err());
    // soffset: inline constants, literals, VGPRs
    for bad in [
        Src::Int(1),
        Src::Lit(4),
        Src::Vgpr(1),
        Src::Scc,
        Src::Sgpr(102),
    ] {
        assert!(
            encode(&load(
                SmemOp::LoadDword,
                Src::Sgpr(0),
                Src::Sgpr(2),
                SmemOffset::Sgpr(bad)
            ))
            .is_err(),
            "{bad:?}"
        );
        assert!(
            encode(&load(
                SmemOp::LoadDword,
                Src::Sgpr(0),
                Src::Sgpr(2),
                SmemOffset::SgprImm(bad, 4)
            ))
            .is_err(),
            "{bad:?}"
        );
    }
    // missing / extra operands
    let mut m = load(SmemOp::LoadDword, Src::Sgpr(0), Src::Sgpr(2), imm);
    m.sdata = None;
    assert!(encode(&m).is_err());
    let mut m = load(SmemOp::LoadDword, Src::Sgpr(0), Src::Sgpr(2), imm);
    m.sbase = None;
    assert!(encode(&m).is_err());
    let mt = |sdata, sbase, offset, glc| Smem {
        op: SmemOp::Memtime,
        sdata,
        sbase,
        offset,
        glc,
    };
    assert!(encode(&mt(Some(Src::Sgpr(0)), None, imm, false)).is_ok());
    assert!(encode(&mt(None, None, imm, false)).is_err());
    assert!(encode(&mt(Some(Src::Sgpr(1)), None, imm, false)).is_err());
    assert!(encode(&mt(Some(Src::Sgpr(0)), Some(Src::Sgpr(2)), imm, false)).is_err());
    assert!(encode(&mt(Some(Src::Sgpr(0)), None, SmemOffset::Imm(4), false)).is_err());
    assert!(encode(&mt(Some(Src::Sgpr(0)), None, imm, true)).is_err());
    let inv = |sdata, glc| Smem {
        op: SmemOp::DcacheInv,
        sdata,
        sbase: None,
        offset: imm,
        glc,
    };
    assert!(encode(&inv(None, false)).is_ok());
    assert!(encode(&inv(Some(Src::Sgpr(0)), false)).is_err());
    assert!(encode(&inv(None, true)).is_err());
}

#[test]
fn decode_rejects_what_encode_cannot_produce() {
    let w = |i: &Smem| encode(i).unwrap();
    let base = w(&load(
        SmemOp::LoadDword,
        Src::Sgpr(0),
        Src::Sgpr(2),
        SmemOffset::Imm(0x10),
    ));
    assert!(Smem::decode(ARCH, &base).is_ok());
    assert!(Smem::decode(ARCH, &base[..1]).is_err());
    assert!(Smem::decode(ARCH, &[]).is_err());
    // NV
    assert!(Smem::decode(ARCH, &[base[0] | 1 << 15, base[1]]).is_err());
    // wrong format
    assert!(Smem::decode(ARCH, &[base[0] & 0x03ff_ffff | 0xd000_0000, base[1]]).is_err());
    // unknown opcode (S_STORE_DWORD, not modelled)
    assert!(Smem::decode(ARCH, &[base[0] & !(0xff << 18) | 16 << 18, base[1]]).is_err());
    // IMM=0, SOE=1 (register in SOFFSET)
    assert!(Smem::decode(ARCH, &[base[0] & !(1 << 17) | 1 << 14, 4 << 25]).is_err());
    // reserved offset bits, soffset without SOE, stray bits in the register form
    assert!(Smem::decode(ARCH, &[base[0], base[1] | 1 << 21]).is_err());
    assert!(Smem::decode(ARCH, &[base[0], base[1] | 4 << 25]).is_err());
    assert!(Smem::decode(ARCH, &[base[0] & !(1 << 17), 0x84]).is_err());
    // soffset code 125 (null on later parts) and constants
    assert!(Smem::decode(ARCH, &[base[0] & !(1 << 17), 125]).is_err());
    // negative immediate on s_buffer_load_*
    let b = w(&load(
        SmemOp::BufferLoadDword,
        Src::Sgpr(0),
        Src::Sgpr(4),
        SmemOffset::Imm(0x10),
    ));
    assert!(Smem::decode(ARCH, &[b[0], b[1] | 1 << 20]).is_err());
    // misaligned sdata / exec as sdata
    let x2 = w(&Smem {
        op: SmemOp::LoadDwordx2,
        sdata: Some(Src::Sgpr(2)),
        sbase: Some(Src::Sgpr(2)),
        offset: SmemOffset::Imm(0),
        glc: false,
    });
    assert!(Smem::decode(ARCH, &[x2[0] & !(0x7f << 6) | 3 << 6, x2[1]]).is_err());
    assert!(Smem::decode(ARCH, &[base[0] & !(0x7f << 6) | 126 << 6, base[1]]).is_err());
    // m0 as sbase (field 62)
    assert!(Smem::decode(ARCH, &[base[0] & !0x3f | 62, base[1]]).is_err());
    // memtime / dcache_inv with address fields set
    let mt = w(&Smem {
        op: SmemOp::Memtime,
        sdata: Some(Src::Sgpr(0)),
        sbase: None,
        offset: SmemOffset::Imm(0),
        glc: false,
    });
    assert!(Smem::decode(ARCH, &mt).is_ok());
    assert!(Smem::decode(ARCH, &[mt[0] | 1 << 16, mt[1]]).is_err());
    assert!(Smem::decode(ARCH, &[mt[0] | 1 << 17, mt[1]]).is_err());
    assert!(Smem::decode(ARCH, &[mt[0] | 1, mt[1]]).is_err());
    assert!(Smem::decode(ARCH, &[mt[0], 8]).is_err());
    let inv = w(&Smem {
        op: SmemOp::DcacheInv,
        sdata: None,
        sbase: None,
        offset: SmemOffset::Imm(0),
        glc: false,
    });
    assert!(Smem::decode(ARCH, &inv).is_ok());
    assert!(Smem::decode(ARCH, &[inv[0] | 1 << 6, inv[1]]).is_err());
}

#[test]
fn footprint_defs_sdata_uses_sbase_and_soffset() {
    let sg = |base, len| RegSpan::new(RegFile::Sgpr, base, len);
    let i = load(
        SmemOp::LoadDwordx4,
        Src::Sgpr(8),
        Src::Sgpr(2),
        SmemOffset::Imm(4),
    );
    let fp = i.footprint();
    assert_eq!(fp.defs, vec![sg(8, 4)]);
    assert_eq!(fp.uses, vec![sg(2, 2)]);

    let i = load(
        SmemOp::BufferLoadDwordx16,
        Src::Sgpr(16),
        Src::Sgpr(4),
        SmemOffset::Sgpr(Src::Sgpr(9)),
    );
    let fp = i.footprint();
    assert_eq!(fp.defs, vec![sg(16, 16)]);
    assert_eq!(fp.uses, vec![sg(4, 4), sg(9, 1)]);

    let i = load(
        SmemOp::LoadDword,
        Src::Ttmp(3),
        Src::Sgpr(0),
        SmemOffset::SgprImm(Src::M0, 8),
    );
    let fp = i.footprint();
    assert_eq!(fp.defs, vec![RegSpan::new(RegFile::Ttmp, 3, 1)]);
    assert_eq!(fp.uses, vec![sg(0, 2), RegSpan::new(RegFile::M0, 0, 1)]);

    let i = load(
        SmemOp::LoadDwordx2,
        Src::VccLo,
        Src::ExecLo,
        SmemOffset::Imm(0),
    );
    let fp = i.footprint();
    assert_eq!(fp.defs, vec![RegSpan::new(RegFile::Vcc, 0, 2)]);
    assert_eq!(fp.uses, vec![RegSpan::new(RegFile::Exec, 0, 2)]);

    let fp = Smem {
        op: SmemOp::Memrealtime,
        sdata: Some(Src::Sgpr(6)),
        sbase: None,
        offset: SmemOffset::Imm(0),
        glc: false,
    }
    .footprint();
    assert_eq!(fp.defs, vec![sg(6, 2)]);
    assert!(fp.uses.is_empty());

    let fp = Smem {
        op: SmemOp::DcacheInv,
        sdata: None,
        sbase: None,
        offset: SmemOffset::Imm(0),
        glc: false,
    }
    .footprint();
    assert!(fp.defs.is_empty() && fp.uses.is_empty());
}

#[test]
fn display_spellings() {
    let i = load(
        SmemOp::LoadDword,
        Src::Sgpr(0),
        Src::Sgpr(2),
        SmemOffset::Imm(-4),
    );
    assert_eq!(i.to_string(), "s_load_dword s0, s[2:3], -0x4");
    let mut i = load(
        SmemOp::BufferLoadDwordx4,
        Src::Sgpr(4),
        Src::Sgpr(8),
        SmemOffset::SgprImm(Src::M0, -0x10),
    );
    i.glc = true;
    assert_eq!(
        i.to_string(),
        "s_buffer_load_dwordx4 s[4:7], s[8:11], m0 offset:-0x10 glc"
    );
    assert_eq!(
        Smem {
            op: SmemOp::Memtime,
            sdata: Some(Src::Sgpr(0)),
            sbase: None,
            offset: SmemOffset::Imm(0),
            glc: false
        }
        .to_string(),
        "s_memtime s[0:1]"
    );
    assert_eq!(
        Smem {
            op: SmemOp::DcacheInv,
            sdata: None,
            sbase: None,
            offset: SmemOffset::Imm(0),
            glc: false
        }
        .to_string(),
        "s_dcache_inv"
    );
}
