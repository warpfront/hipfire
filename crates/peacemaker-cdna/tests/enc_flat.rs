// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FLAT / GLOBAL / SCRATCH encode/decode/Display/footprint against the ROCm
//! llvm-mc oracle (one batched assemble and one batched disassemble per corpus).
use peacemaker_cdna::enc::flat::{Flat, FlatOp, FlatSeg};
use peacemaker_cdna::inst::{Format, Inst, MemKind, RegFile, RegSpan};
use peacemaker_cdna::operand::Src;
use peacemaker_cdna::{oracle, Arch};

const ARCH: Arch = Arch::Gfx942;

fn is_load(op: FlatOp) -> bool {
    op.widths().1 == 0
}
fn is_store(op: FlatOp) -> bool {
    op.widths().0 == 0
}
fn is_atomic(op: FlatOp) -> bool {
    !is_load(op) && !is_store(op)
}

fn words(f: &Flat) -> Vec<u32> {
    let mut out = Vec::new();
    f.encode(ARCH, &mut out).unwrap();
    out
}

/// A well-formed instruction: `slot` picks the data/vdst/addr registers,
/// `acc` moves data/vdst into the AccVGPR file.
#[allow(clippy::too_many_arguments)]
fn make(
    op: FlatOp,
    seg: FlatSeg,
    addr: Option<Src>,
    saddr: Option<Src>,
    offset: i16,
    cache: u8,
    slot: u8,
    acc: bool,
) -> Flat {
    let dw = op.widths().1;
    // llvm-mc has no AccVGPR form of the returning f32 atomic.
    let acc = acc && !(op == FlatOp::AtomicAddF32 && cache & 1 != 0);
    let reg = |n: u8| if acc { Src::Agpr(n) } else { Src::Vgpr(n) };
    let (sc0, sc1, nt) = (cache & 1 != 0, cache & 2 != 0, cache & 4 != 0);
    let vdst = if is_load(op) || (is_atomic(op) && sc0) {
        Some(reg(slot))
    } else {
        None
    };
    let data = (dw > 0).then(|| reg(slot.wrapping_add(8)));

    Flat {
        op,
        seg,
        vdst,
        addr,
        data,
        saddr,
        offset,
        sc0,
        sc1,
        nt,
    }
}

fn check(cases: &[Flat]) {
    if oracle::disabled() {
        eprintln!("enc_flat: llvm-mc oracle disabled; skipping");
        return;
    }
    let lines: Vec<String> = cases.iter().map(|c| c.to_string()).collect();
    let assembled = oracle::mc_encode(ARCH, &lines).unwrap();
    let bytes: Vec<Vec<u8>> = cases.iter().map(|c| oracle::bytes(&words(c))).collect();
    let disassembled = oracle::mc_disassemble(ARCH, &bytes).unwrap();
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(
            assembled[i].1, bytes[i],
            "bytes of `{}` ({case:?})",
            lines[i]
        );
        assert_eq!(
            assembled[i].0, lines[i],
            "llvm's text for our spelling of {case:?}"
        );
        assert_eq!(
            disassembled[i], lines[i],
            "llvm disassembly of the bytes of {case:?}"
        );
        let w = words(case);
        let (back, n) = Flat::decode(ARCH, &w).unwrap();
        assert_eq!((&back, n), (case, 2), "decode of `{}`", lines[i]);
        let (inst, n) = Inst::decode(ARCH, &w).unwrap();
        assert_eq!((inst.clone(), n), (Inst::Flat(case.clone()), 2));
        assert_eq!(inst.format(), Format::Flat);
        assert_eq!(inst.to_string(), lines[i]);
    }
}

const OFFSETS: [i16; 8] = [0, 1, -1, 16, 2047, -2048, 4095, -4096];

fn saddr_pairs() -> Vec<Src> {
    vec![
        Src::Sgpr(4),
        Src::Sgpr(0),
        Src::Sgpr(100),
        Src::VccLo,
        Src::FlatScratchLo,
        Src::XnackMaskLo,
        Src::Ttmp(4),
        Src::Ttmp(14),
    ]
}

#[test]
fn global_with_off_and_with_saddr() {
    let saddrs = saddr_pairs();
    let mut cases = Vec::new();
    let mut n = 0usize;
    for &op in FlatOp::ALL {
        for use_saddr in [false, true] {
            for &offset in &OFFSETS {
                for cache in 0..8u8 {
                    n += 1;
                    let slot = ((n * 2) % 100) as u8;
                    let acc = n % 5 == 0;
                    let (addr, saddr) = if use_saddr {
                        (
                            Some(Src::Vgpr(((n * 3) % 200) as u8)),
                            Some(saddrs[n % saddrs.len()]),
                        )
                    } else {
                        (Some(Src::Vgpr(((n * 2) % 200 + 100) as u8 & !1)), None)
                    };
                    cases.push(make(
                        op,
                        FlatSeg::Global,
                        addr,
                        saddr,
                        offset,
                        cache,
                        slot,
                        acc,
                    ));
                }
            }
        }
    }
    assert!(cases.len() > 2000);
    check(&cases);
}

#[test]
fn global_register_extremes() {
    let mut cases = Vec::new();
    for &op in FlatOp::ALL {
        let (vw, dw) = op.widths();
        let w = vw.max(dw);
        // Highest first register that fits: v255 for one dword, else the highest even start.
        let top = if w == 1 {
            255
        } else {
            ((256 - u16::from(w)) as u8) & !1
        };
        for acc in [false, true] {
            for cache in [0u8, 1, 7] {
                for (addr, saddr, offset) in [
                    (Src::Vgpr(254), None, 0i16),
                    (Src::Vgpr(255), Some(Src::Sgpr(100)), -4096),
                ] {
                    let mut f = make(
                        op,
                        FlatSeg::Global,
                        Some(addr),
                        saddr,
                        offset,
                        cache,
                        0,
                        acc,
                    );
                    let agpr = f.vdst.or(f.data).is_some_and(Src::is_agpr);
                    let reg = |n: u8| if agpr { Src::Agpr(n) } else { Src::Vgpr(n) };
                    f.vdst = f.vdst.map(|_| reg(top));
                    f.data = f.data.map(|_| reg(top));
                    cases.push(f);
                }
            }
        }
    }
    check(&cases);
}

#[test]
fn flat_spellings() {
    let mut cases = Vec::new();
    let mut n = 0usize;
    for &op in FlatOp::ALL {
        for offset in [0i16, 1, 16, 2047, 2048, 4095] {
            for cache in 0..8u8 {
                n += 1;
                let slot = ((n * 2) % 100) as u8;
                cases.push(make(
                    op,
                    FlatSeg::Flat,
                    Some(Src::Vgpr(((n * 2) % 200 + 20) as u8 & !1)),
                    None,
                    offset,
                    cache,
                    slot,
                    n % 7 == 0,
                ));
            }
        }
    }
    check(&cases);
}

#[test]
fn scratch_spellings() {
    let mut cases = Vec::new();
    let mut n = 0usize;
    let saddrs = [
        Src::Sgpr(4),
        Src::Sgpr(101),
        Src::M0,
        Src::VccHi,
        Src::Ttmp(3),
        Src::FlatScratchLo,
        Src::ExecLo,
        Src::Sgpr(0),
    ];
    for &op in FlatOp::ALL {
        if op.name(FlatSeg::Scratch).is_none() {
            continue;
        }
        for (use_addr, use_saddr) in [(true, false), (true, true), (false, true), (false, false)] {
            for offset in [0i16, -1, 12, 4095, -4096, -2048] {
                for cache in 0..8u8 {
                    n += 1;
                    let slot = ((n * 2) % 100) as u8;
                    let addr = use_addr.then(|| Src::Vgpr((n % 200) as u8));
                    let saddr = use_saddr.then(|| saddrs[n % saddrs.len()]);
                    cases.push(make(
                        op,
                        FlatSeg::Scratch,
                        addr,
                        saddr,
                        offset,
                        cache,
                        slot,
                        n % 6 == 0,
                    ));
                }
            }
        }
    }
    check(&cases);
}

#[test]
fn scratch_atomics_do_not_exist() {
    for op in [FlatOp::AtomicAdd, FlatOp::AtomicAddF32] {
        assert!(op.name(FlatSeg::Scratch).is_none());
        let f = make(
            op,
            FlatSeg::Scratch,
            Some(Src::Vgpr(2)),
            None,
            0,
            0,
            4,
            false,
        );
        assert!(f.encode(ARCH, &mut Vec::new()).is_err());
    }
}

#[test]
fn named_encodings_match_llvm_dump() {
    // Bytes printed by `llvm-mc -mcpu=gfx942 -show-encoding` while writing this encoder.
    let global_load = make(
        FlatOp::LoadDword,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        1,
        false,
    );
    assert_eq!(
        oracle::bytes(&words(&global_load)),
        [0x00, 0x80, 0x50, 0xdc, 0x02, 0x00, 0x7f, 0x01]
    );
    let mut all_bits = make(
        FlatOp::LoadDword,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        Some(Src::Sgpr(4)),
        16,
        7,
        1,
        false,
    );
    all_bits.addr = Some(Src::Vgpr(2));
    assert_eq!(
        oracle::bytes(&words(&all_bits)),
        [0x10, 0x80, 0x53, 0xde, 0x02, 0x00, 0x04, 0x01]
    );
    assert_eq!(
        all_bits.to_string(),
        "global_load_dword v1, v2, s[4:5] offset:16 sc0 nt sc1"
    );
    let flat_store = Flat {
        op: FlatOp::StoreDword,
        seg: FlatSeg::Flat,
        vdst: None,
        addr: Some(Src::Vgpr(2)),
        data: Some(Src::Vgpr(1)),
        saddr: None,
        offset: 4095,
        sc0: false,
        sc1: true,
        nt: false,
    };
    assert_eq!(
        oracle::bytes(&words(&flat_store)),
        [0xff, 0x0f, 0x70, 0xde, 0x02, 0x01, 0x00, 0x00]
    );
    assert_eq!(
        flat_store.to_string(),
        "flat_store_dword v[2:3], v1 offset:4095 sc1"
    );
    let scratch = Flat {
        op: FlatOp::StoreDword,
        seg: FlatSeg::Scratch,
        vdst: None,
        addr: Some(Src::Vgpr(2)),
        data: Some(Src::Vgpr(1)),
        saddr: Some(Src::Sgpr(4)),
        offset: 0,
        sc0: false,
        sc1: false,
        nt: false,
    };
    assert_eq!(
        oracle::bytes(&words(&scratch)),
        [0x00, 0x60, 0x70, 0xdc, 0x02, 0x01, 0x04, 0x00]
    );
    assert_eq!(scratch.to_string(), "scratch_store_dword v2, v1, s4");
}

#[test]
fn offset_overflow_is_rejected() {
    for &op in FlatOp::ALL {
        for (seg, bad) in [
            (FlatSeg::Global, [4096i16, -4097, i16::MAX, i16::MIN]),
            (FlatSeg::Flat, [4096, -1, i16::MAX, i16::MIN]),
            (FlatSeg::Scratch, [4096, -4097, i16::MAX, i16::MIN]),
        ] {
            if op.name(seg).is_none() {
                continue;
            }
            for offset in bad {
                let f = make(op, seg, Some(Src::Vgpr(2)), None, offset, 0, 4, false);
                let err = f
                    .encode(ARCH, &mut Vec::new())
                    .expect_err("overflowing offset must be rejected");
                assert!(err.contains("offset"), "{err}");
            }
        }
    }
}

#[test]
fn illegal_forms_are_rejected() {
    let ok = make(
        FlatOp::LoadDwordx2,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        4,
        false,
    );
    assert!(ok.encode(ARCH, &mut Vec::new()).is_ok());
    let bad =
        |f: Flat, why: &str| assert!(f.encode(ARCH, &mut Vec::new()).is_err(), "{why}: {f:?}");
    bad(
        Flat {
            addr: Some(Src::Vgpr(3)),
            ..ok.clone()
        },
        "misaligned 64-bit address pair",
    );
    bad(
        Flat {
            vdst: Some(Src::Vgpr(5)),
            ..ok.clone()
        },
        "misaligned 64-bit vdst",
    );
    bad(
        Flat {
            addr: Some(Src::Vgpr(255)),
            ..ok.clone()
        },
        "address pair past v255",
    );
    bad(
        Flat {
            vdst: Some(Src::Vgpr(254)),
            op: FlatOp::LoadDwordx4,
            ..ok.clone()
        },
        "vdst quad past v255",
    );
    bad(
        Flat {
            addr: None,
            ..ok.clone()
        },
        "global without an address",
    );
    bad(
        Flat {
            addr: Some(Src::Agpr(2)),
            ..ok.clone()
        },
        "AGPR address",
    );
    bad(
        Flat {
            addr: Some(Src::Sgpr(2)),
            ..ok.clone()
        },
        "SGPR address",
    );
    bad(
        Flat {
            vdst: None,
            ..ok.clone()
        },
        "load without vdst",
    );
    bad(
        Flat {
            data: Some(Src::Vgpr(8)),
            ..ok.clone()
        },
        "load with data",
    );
    bad(
        Flat {
            saddr: Some(Src::Sgpr(5)),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "odd saddr pair",
    );
    bad(
        Flat {
            saddr: Some(Src::ExecLo),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "exec as global saddr",
    );
    bad(
        Flat {
            saddr: Some(Src::M0),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "m0 as global saddr pair",
    );
    bad(
        Flat {
            saddr: Some(Src::Vgpr(4)),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "VGPR saddr",
    );
    bad(
        Flat {
            saddr: Some(Src::Int(1)),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "inline constant saddr",
    );
    bad(
        Flat {
            saddr: Some(Src::Lit(1)),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "literal saddr",
    );
    bad(
        Flat {
            seg: FlatSeg::Flat,
            saddr: Some(Src::Sgpr(4)),
            addr: Some(Src::Vgpr(2)),
            ..ok.clone()
        },
        "FLAT with saddr",
    );
    // The same operands as a SCRATCH load are a legal shape (32-bit address VGPR + one SGPR).
    let scratch = make(
        FlatOp::LoadDwordx2,
        FlatSeg::Scratch,
        Some(Src::Vgpr(2)),
        Some(Src::Sgpr(4)),
        0,
        0,
        4,
        false,
    );
    assert!(scratch.encode(ARCH, &mut Vec::new()).is_ok());
}

#[test]
fn atomic_sc0_must_match_vdst() {
    let ret = make(
        FlatOp::AtomicAdd,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        1,
        4,
        false,
    );
    assert!(ret.vdst.is_some() && ret.sc0);
    assert!(ret.encode(ARCH, &mut Vec::new()).is_ok());
    assert!(Flat {
        sc0: false,
        ..ret.clone()
    }
    .encode(ARCH, &mut Vec::new())
    .is_err());
    let no_ret = Flat {
        vdst: None,
        sc0: false,
        ..ret.clone()
    };
    assert!(no_ret.encode(ARCH, &mut Vec::new()).is_ok());
    assert!(Flat {
        sc0: true,
        ..no_ret.clone()
    }
    .encode(ARCH, &mut Vec::new())
    .is_err());
    assert!(
        Flat {
            vdst: Some(Src::Agpr(4)),
            ..ret.clone()
        }
        .encode(ARCH, &mut Vec::new())
        .is_err(),
        "vdst/data register files differ"
    );
    assert!(Flat {
        vdst: Some(Src::Agpr(4)),
        data: Some(Src::Agpr(8)),
        ..ret.clone()
    }
    .encode(ARCH, &mut Vec::new())
    .is_ok());
    let f32_ret = Flat {
        op: FlatOp::AtomicAddF32,
        vdst: Some(Src::Agpr(4)),
        data: Some(Src::Agpr(8)),
        ..ret
    };
    assert!(
        f32_ret.encode(ARCH, &mut Vec::new()).is_err(),
        "llvm-mc rejects AccVGPRs on the returning f32 atomic"
    );
    assert!(Flat {
        vdst: None,
        sc0: false,
        ..f32_ret
    }
    .encode(ARCH, &mut Vec::new())
    .is_ok());
    let store = make(
        FlatOp::StoreDword,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        0,
        false,
    );
    assert!(Flat {
        vdst: Some(Src::Vgpr(1)),
        ..store
    }
    .encode(ARCH, &mut Vec::new())
    .is_err());
}

#[test]
fn decode_rejects_reserved_and_ignored_fields() {
    let dec = |w: [u32; 2]| Flat::decode(ARCH, &w);
    let global_load = [0xdc50_8000u32, 0x017f_0002];
    assert!(dec(global_load).is_ok());
    assert!(
        dec([global_load[0] | 0x4000, global_load[1]]).is_err(),
        "SEG = 3"
    );
    assert!(
        dec([global_load[0] | 0x2000, global_load[1]]).is_err(),
        "LDS bit on a global op"
    );
    assert!(
        dec([global_load[0], global_load[1] | 0x0500]).is_err(),
        "DATA on a load"
    );
    assert!(
        dec([global_load[0], 0x017f_0003]).is_err(),
        "misaligned address pair"
    );
    assert!(dec([global_load[0], 0x017e_0002]).is_err(), "exec as saddr");
    assert!(
        dec([0xdc50_0000, global_load[1]]).is_err(),
        "FLAT with saddr = 0x7f"
    );
    assert!(
        dec([0xdc50_1000, 0x0100_0002]).is_err(),
        "FLAT offset bit 12"
    );
    assert!(
        dec([0xdc50_4000, 0x017f_0002]).is_err(),
        "scratch address without SVE"
    );
    assert!(dec([0xdc70_8000, 0x057f_0102]).is_err(), "VDST on a store");
    assert!(
        dec([(0b11_0111 << 26) | (66 << 18) | (2 << 14), 0x057f_0102]).is_err(),
        "VDST on a non-returning atomic"
    );
    assert!(
        dec([(0b11_0111 << 26) | (66 << 18) | (1 << 14), 0x007f_0102]).is_err(),
        "scratch atomic"
    );
    assert!(dec([0xdc00_8000, 0x017f_0002]).is_err(), "unknown opcode 0");
    assert!(dec([0xd800_0000, 0]).is_err(), "not a FLAT-family dword");
    assert!(Flat::decode(ARCH, &[0xdc50_8000]).is_err(), "truncated");
}

#[test]
fn mem_kind_and_widths() {
    assert_eq!(FlatOp::LoadDwordx3.widths(), (3, 0));
    assert_eq!(FlatOp::StoreDwordx4.widths(), (0, 4));
    assert_eq!(FlatOp::AtomicAdd.widths(), (1, 1));
    assert_eq!(FlatOp::AtomicAddF32.widths(), (1, 1));
    for &op in FlatOp::ALL {
        for seg in [FlatSeg::Global, FlatSeg::Flat, FlatSeg::Scratch] {
            if op.name(seg).is_none() {
                continue;
            }
            for cache in [0u8, 1] {
                let f = make(op, seg, Some(Src::Vgpr(2)), None, 0, cache, 4, false);
                let want = if is_load(op) {
                    MemKind::VmemLoad
                } else if is_store(op) {
                    MemKind::VmemStore
                } else if cache & 1 != 0 {
                    MemKind::VmemAtomicRtn
                } else {
                    MemKind::VmemAtomic
                };
                assert_eq!(f.mem_kind(), want, "{f}");
                assert_eq!(Inst::Flat(f).memory(), Some(want));
            }
        }
    }
}

#[test]
fn footprints() {
    let sp = |file, base, len| RegSpan::new(file, base, len);
    // Global load, 64-bit address, no saddr.
    let load = make(
        FlatOp::LoadDwordx4,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        8,
        false,
    );
    let fp = load.footprint();
    assert_eq!(fp.defs, vec![sp(RegFile::Vgpr, 8, 4)]);
    assert!(
        fp.uses.contains(&sp(RegFile::Exec, 0, 2)) && fp.uses.contains(&sp(RegFile::Vgpr, 2, 2))
    );
    assert_eq!(fp.uses.len(), 2);
    // Global store with saddr: 32-bit offset, SGPR-pair base, 2-dword data.
    let store = make(
        FlatOp::StoreDwordx2,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        Some(Src::Sgpr(4)),
        8,
        0,
        0,
        false,
    );
    let fp = store.footprint();
    assert!(fp.defs.is_empty());
    assert!(
        fp.uses.contains(&sp(RegFile::Vgpr, 2, 1))
            && fp.uses.contains(&sp(RegFile::Sgpr, 4, 2))
            && fp.uses.contains(&sp(RegFile::Vgpr, 8, 2))
    );
    assert_eq!(fp.uses.len(), 4);
    // Returning atomic defines vdst; non-returning does not.
    let ret = make(
        FlatOp::AtomicAdd,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        1,
        4,
        false,
    );
    assert_eq!(ret.footprint().defs, vec![sp(RegFile::Vgpr, 4, 1)]);
    assert!(make(
        FlatOp::AtomicAdd,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        4,
        false
    )
    .footprint()
    .defs
    .is_empty());
    // AccVGPR data.
    let acc = make(
        FlatOp::LoadDword,
        FlatSeg::Global,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        6,
        true,
    );
    assert_eq!(acc.footprint().defs, vec![sp(RegFile::Agpr, 6, 1)]);
    // Scratch: one 32-bit VGPR address, one SGPR, and the FLAT_SCRATCH base.
    let scratch = make(
        FlatOp::LoadDword,
        FlatSeg::Scratch,
        Some(Src::Vgpr(2)),
        Some(Src::Sgpr(4)),
        0,
        0,
        6,
        false,
    );
    let fp = scratch.footprint();
    assert!(
        fp.uses.contains(&sp(RegFile::Vgpr, 2, 1))
            && fp.uses.contains(&sp(RegFile::Sgpr, 4, 1))
            && fp.uses.contains(&sp(RegFile::FlatScratch, 0, 2))
    );
    // FLAT: 64-bit address pair.
    let flat = make(
        FlatOp::StoreDword,
        FlatSeg::Flat,
        Some(Src::Vgpr(2)),
        None,
        0,
        0,
        0,
        false,
    );
    assert!(flat.footprint().uses.contains(&sp(RegFile::Vgpr, 2, 2)));
}
