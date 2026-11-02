// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MUBUF encode/decode/Display/footprint against the ROCm llvm-mc oracle.
use peacemaker_cdna::enc::mubuf::{Mubuf, MubufOp};
use peacemaker_cdna::inst::{Format, Inst, MemKind, RegFile, RegSpan};
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::{oracle, Arch};

const ARCH: Arch = Arch::Gfx942;

fn base(op: MubufOp) -> Mubuf {
    Mubuf {
        op,
        vdata: Src::Vgpr(0),
        vaddr: Some(Src::Vgpr(2)),
        srsrc: Src::Sgpr(4),
        soffset: Src::Sgpr(0),
        offset: 0,
        offen: true,
        idxen: false,
        sc0: false,
        sc1: false,
        nt: false,
    }
}

fn words(m: &Mubuf) -> Vec<u32> {
    let mut out = Vec::new();
    m.encode(ARCH, &mut out).unwrap();
    out
}

fn soffsets() -> Vec<Src> {
    vec![
        Src::Sgpr(0),
        Src::Sgpr(101),
        Src::Int(0),
        Src::Int(64),
        Src::Int(-16),
        Src::Int(-1),
        Src::M0,
        Src::VccLo,
        Src::VccHi,
        Src::ExecLo,
        Src::ExecHi,
        Src::Ttmp(15),
        Src::FlatScratchLo,
        Src::XnackMaskHi,
        Src::SharedBase,
        Src::PrivateLimit,
        Src::Float(InlineFloat::Half),
        Src::Float(InlineFloat::NegFour),
        Src::Float(InlineFloat::InvTwoPi),
        Src::Vccz,
        Src::Execz,
        Src::Scc,
    ]
}

fn srsrcs() -> Vec<Src> {
    let mut v: Vec<Src> = (0..=24).map(|i| Src::Sgpr(i * 4)).collect();
    v.extend([0, 4, 8, 12].map(Src::Ttmp));
    v
}

/// Every op x {offen, idxen, both, neither} x offsets x soffset x cache bits,
/// with VDATA/VADDR/SRSRC cycling through legal registers (VGPR and AGPR data).
fn corpus() -> Vec<Mubuf> {
    let srsrcs = srsrcs();
    let soffs = soffsets();
    let mut out = Vec::new();
    let mut n = 0usize;
    for &op in MubufOp::ALL {
        for (offen, idxen) in [(false, false), (true, false), (false, true), (true, true)] {
            for offset in [0u16, 1, 0x7ff, 0x800, 0xfff] {
                for &soffset in &soffs {
                    for cache in 0..8u8 {
                        n += 1;
                        let slot = (n * 2) % 120;
                        let vdata = if n % 3 == 0 {
                            Src::Agpr(slot as u8)
                        } else {
                            Src::Vgpr(slot as u8)
                        };
                        let vaddr = if offen || idxen {
                            let reg = if offen && idxen {
                                (n * 2 + 130) % 250
                            } else {
                                (n * 3 + 130) % 250
                            };
                            Some(Src::Vgpr(if offen && idxen { reg & !1 } else { reg } as u8))
                        } else {
                            None
                        };
                        out.push(Mubuf {
                            op,
                            vdata,
                            vaddr,
                            srsrc: srsrcs[n % srsrcs.len()],
                            soffset,
                            offset,
                            offen,
                            idxen,
                            sc0: cache & 1 != 0,
                            sc1: cache & 2 != 0,
                            nt: cache & 4 != 0,
                        });
                    }
                }
            }
        }
    }
    out
}

#[test]
fn op_properties() {
    for &op in MubufOp::ALL {
        let name = op.name();
        assert_eq!(op.is_store(), name.starts_with("buffer_store"), "{name}");
        let expect = match name.rsplit('_').next().unwrap() {
            "dwordx2" => 2,
            "dwordx3" => 3,
            "dwordx4" => 4,
            _ => 1,
        };
        assert_eq!(op.data_dwords(), expect, "{name}");
        let m = base(op);
        assert_eq!(
            m.mem_kind(),
            if op.is_store() {
                MemKind::VmemStore
            } else {
                MemKind::VmemLoad
            }
        );
        assert_eq!(
            MubufOp::from_opcode(ARCH, op.opcode(ARCH).unwrap()),
            Some(op)
        );
    }
}

#[test]
fn oracle_round_trip() {
    if oracle::disabled() {
        eprintln!("enc_mubuf: ROCm oracle disabled");
        return;
    }
    let cases = corpus();
    assert!(cases.len() > 20_000);
    let texts: Vec<String> = cases.iter().map(|m| m.to_string()).collect();
    let assembled = oracle::mc_encode(ARCH, &texts).unwrap();
    let mut byte_vecs = Vec::with_capacity(cases.len());
    for ((m, text), (mc_text, mc_bytes)) in cases.iter().zip(&texts).zip(&assembled) {
        let ours = words(m);
        assert_eq!(&oracle::bytes(&ours), mc_bytes, "bytes: {text}");
        assert_eq!(mc_text, text, "llvm text: {text}");
        let (decoded, len) = Mubuf::decode(ARCH, &ours).unwrap();
        assert_eq!(len, 2);
        assert_eq!(&decoded, m, "decode identity: {text}");
        let (inst, len) = Inst::decode(ARCH, &ours).unwrap();
        assert_eq!(
            (len, &inst),
            (2, &Inst::Mubuf(m.clone())),
            "Inst::decode: {text}"
        );
        assert_eq!(inst.words(ARCH).unwrap(), ours);
        assert_eq!(Inst::Mubuf(m.clone()).to_string(), *text);
        byte_vecs.push(mc_bytes.clone());
    }
    let disassembled = oracle::mc_disassemble(ARCH, &byte_vecs).unwrap();
    for (text, dis) in texts.iter().zip(&disassembled) {
        assert_eq!(dis, text, "llvm disassembly")
    }
}

#[test]
fn oracle_rejects_what_we_reject() {
    if oracle::disabled() {
        eprintln!("enc_mubuf: ROCm oracle disabled");
        return;
    }
    for bad in [
        Mubuf {
            srsrc: Src::Sgpr(5),
            ..base(MubufOp::LoadDword)
        },
        Mubuf {
            vdata: Src::Vgpr(3),
            ..base(MubufOp::LoadDwordx2)
        },
        Mubuf {
            vdata: Src::Vgpr(1),
            ..base(MubufOp::StoreDwordx3)
        },
        Mubuf {
            vdata: Src::Agpr(1),
            ..base(MubufOp::StoreDwordx4)
        },
        Mubuf {
            idxen: true,
            vaddr: Some(Src::Vgpr(3)),
            ..base(MubufOp::LoadDword)
        },
        Mubuf {
            soffset: Src::Int(65),
            ..base(MubufOp::LoadDword)
        },
        Mubuf {
            soffset: Src::Lit(0x1234),
            ..base(MubufOp::LoadDword)
        },
    ] {
        assert!(bad.encode(ARCH, &mut Vec::new()).is_err(), "{bad}");
        assert!(
            oracle::mc_encode(ARCH, &[bad.to_string()]).is_err(),
            "llvm-mc accepted {bad}"
        );
    }
}

#[test]
fn rejects_illegal_forms() {
    let ok = base(MubufOp::LoadDword);
    let bad = |m: Mubuf| assert!(m.encode(ARCH, &mut Vec::new()).is_err(), "{m:?}");
    // misaligned / unreachable resource
    for srsrc in [
        Src::Sgpr(1),
        Src::Sgpr(5),
        Src::Sgpr(98),
        Src::Sgpr(100),
        Src::Sgpr(101),
        Src::Ttmp(2),
        Src::Ttmp(16),
        Src::Vgpr(0),
        Src::VccLo,
        Src::M0,
        Src::Int(0),
    ] {
        bad(Mubuf {
            srsrc,
            ..ok.clone()
        });
    }
    // offset overflow
    for offset in [0x1000, 0x1fff, u16::MAX] {
        bad(Mubuf {
            offset,
            ..ok.clone()
        })
    }
    // soffset: literal, VGPR, AGPR, out-of-range inline, s102+
    for soffset in [
        Src::Lit(0),
        Src::Lit(0x1000),
        Src::Vgpr(0),
        Src::Agpr(0),
        Src::Int(65),
        Src::Int(-17),
        Src::Sgpr(102),
        Src::Sgpr(255),
    ] {
        bad(Mubuf {
            soffset,
            ..ok.clone()
        });
    }
    // vaddr presence must match offen/idxen
    bad(Mubuf {
        vaddr: None,
        ..ok.clone()
    });
    bad(Mubuf {
        offen: false,
        idxen: false,
        ..ok.clone()
    });
    bad(Mubuf {
        vaddr: Some(Src::Sgpr(0)),
        ..ok.clone()
    });
    bad(Mubuf {
        vaddr: Some(Src::Agpr(2)),
        ..ok.clone()
    });
    bad(Mubuf {
        vaddr: Some(Src::Vgpr(255)),
        idxen: true,
        ..ok.clone()
    });
    bad(Mubuf {
        vaddr: Some(Src::Vgpr(3)),
        idxen: true,
        ..ok.clone()
    });
    // vdata
    bad(Mubuf {
        vdata: Src::Sgpr(0),
        ..ok.clone()
    });
    bad(Mubuf {
        vdata: Src::Vgpr(1),
        op: MubufOp::LoadDwordx2,
        ..ok.clone()
    });
    bad(Mubuf {
        vdata: Src::Agpr(3),
        op: MubufOp::StoreDwordx3,
        ..ok.clone()
    });
    bad(Mubuf {
        vdata: Src::Vgpr(254),
        op: MubufOp::LoadDwordx3,
        ..ok.clone()
    });
}

#[test]
fn accepts_register_boundaries() {
    let mut out = Vec::new();
    Mubuf {
        vdata: Src::Agpr(255),
        ..base(MubufOp::LoadDword)
    }
    .encode(ARCH, &mut out)
    .unwrap();
    Mubuf {
        vdata: Src::Vgpr(252),
        ..base(MubufOp::StoreDwordx4)
    }
    .encode(ARCH, &mut out)
    .unwrap();
    Mubuf {
        vaddr: Some(Src::Vgpr(254)),
        idxen: true,
        ..base(MubufOp::LoadDword)
    }
    .encode(ARCH, &mut out)
    .unwrap();
    Mubuf {
        vaddr: Some(Src::Vgpr(255)),
        offen: true,
        ..base(MubufOp::LoadDword)
    }
    .encode(ARCH, &mut out)
    .unwrap();
    assert_eq!(out.len(), 8);
}

#[test]
fn decode_rejects_unmodeled_bits() {
    let good = words(&Mubuf {
        sc0: true,
        ..base(MubufOp::LoadDword)
    });
    assert!(Mubuf::decode(ARCH, &good).is_ok());
    let patch = |i: usize, f: &dyn Fn(u32) -> u32| {
        let mut w = good.clone();
        w[i] = f(w[i]);
        w
    };
    let cases: Vec<(&str, Vec<u32>)> = vec![
        ("lds", patch(0, &|w| w | 1 << 16)),
        ("reserved w0 bit 25", patch(0, &|w| w | 1 << 25)),
        ("reserved w1 bit 21", patch(1, &|w| w | 1 << 21)),
        ("reserved w1 bit 22", patch(1, &|w| w | 1 << 22)),
        ("srsrc 25", patch(1, &|w| (w & !(0x1f << 16)) | 25 << 16)),
        ("srsrc 26", patch(1, &|w| (w & !(0x1f << 16)) | 26 << 16)),
        ("srsrc 31", patch(1, &|w| (w & !(0x1f << 16)) | 31 << 16)),
        ("soffset null", patch(1, &|w| (w & 0x00ff_ffff) | 125 << 24)),
        (
            "soffset literal",
            patch(1, &|w| (w & 0x00ff_ffff) | 255 << 24),
        ),
        (
            "soffset reserved",
            patch(1, &|w| (w & 0x00ff_ffff) | 220 << 24),
        ),
        ("opcode 0 (format load)", patch(0, &|w| w & !(0x7f << 18))),
        (
            "opcode 25 (d16_hi)",
            patch(0, &|w| (w & !(0x7f << 18)) | 25 << 18),
        ),
        (
            "vaddr without offen/idxen",
            patch(0, &|w| w & !(1 << 12))
                .into_iter()
                .enumerate()
                .map(|(i, w)| if i == 1 { w | 3 } else { w })
                .collect(),
        ),
        (
            "odd vaddr pair",
            patch(0, &|w| w | 1 << 13)
                .into_iter()
                .enumerate()
                .map(|(i, w)| if i == 1 { (w & !0xff) | 3 } else { w })
                .collect(),
        ),
        (
            "wrong format",
            patch(0, &|w| w & 0x03ff_ffff | 0b111010 << 26),
        ),
        ("one dword", vec![good[0]]),
        ("empty", vec![]),
    ];
    for (what, w) in cases {
        assert!(Mubuf::decode(ARCH, &w).is_err(), "{what}: decoded {w:08x?}")
    }
}

#[test]
fn field_layout() {
    // CDNA3 ISA §13.5.2 Table 98 p.549.
    let m = Mubuf {
        op: MubufOp::StoreDwordx4,
        vdata: Src::Agpr(8),
        vaddr: Some(Src::Vgpr(0xa6)),
        srsrc: Src::Ttmp(4),
        soffset: Src::Int(-3),
        offset: 0xabc,
        offen: true,
        idxen: true,
        sc0: true,
        sc1: true,
        nt: true,
    };
    let w = words(&m);
    assert_eq!(
        w[0],
        0b111000 << 26 | 31 << 18 | 1 << 17 | 1 << 15 | 1 << 14 | 1 << 13 | 1 << 12 | 0xabc
    );
    assert_eq!(w[1], (192 + 3) << 24 | 1 << 23 | 28 << 16 | 8 << 8 | 0xa6);
    assert_eq!(oracle_format(w[0]), Format::Mubuf);
}

fn oracle_format(w: u32) -> Format {
    peacemaker_cdna::inst::classify(w).unwrap()
}

#[test]
fn footprint_loads_and_stores() {
    let exec = RegSpan::new(RegFile::Exec, 0, 2);
    let load = Mubuf {
        vdata: Src::Vgpr(8),
        vaddr: Some(Src::Vgpr(2)),
        srsrc: Src::Sgpr(12),
        soffset: Src::Sgpr(20),
        ..base(MubufOp::LoadDwordx3)
    };
    let fp = load.footprint();
    assert_eq!(fp.defs, vec![RegSpan::new(RegFile::Vgpr, 8, 3)]);
    assert!(fp.uses.contains(&exec));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Sgpr, 12, 4)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Sgpr, 20, 1)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 2, 1)));
    assert_eq!(fp.uses.len(), 4);

    let store = Mubuf {
        vdata: Src::Agpr(4),
        vaddr: Some(Src::Vgpr(6)),
        srsrc: Src::Ttmp(8),
        soffset: Src::Int(16),
        offen: true,
        idxen: true,
        ..base(MubufOp::StoreDwordx4)
    };
    let fp = store.footprint();
    assert!(fp.defs.is_empty());
    assert!(fp.uses.contains(&exec));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Ttmp, 8, 4)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Vgpr, 6, 2)));
    assert!(fp.uses.contains(&RegSpan::new(RegFile::Agpr, 4, 4)));
    assert_eq!(fp.uses.len(), 4, "inline soffset reads no register");

    let none = Mubuf {
        vaddr: None,
        offen: false,
        idxen: false,
        soffset: Src::M0,
        ..base(MubufOp::StoreByte)
    };
    let fp = none.footprint();
    assert!(fp.uses.contains(&RegSpan::new(RegFile::M0, 0, 1)));
    assert!(!fp
        .uses
        .iter()
        .any(|r| r.file == RegFile::Vgpr && r.base == 2));
    assert_eq!(Inst::Mubuf(store).memory(), Some(MemKind::VmemStore));
}
