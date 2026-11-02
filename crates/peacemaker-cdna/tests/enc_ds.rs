// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! DS format: every op x offsets x operand placements against the llvm-mc
//! gfx942 assembler and disassembler (bytes, text, decode identity), the
//! exhaustive `ds_swizzle_b32` offset space, and the rejection paths.
use peacemaker_cdna::enc::ds::{Ds, DsOp};
use peacemaker_cdna::inst::{Footprint, MemKind, RegFile, RegSpan};
use peacemaker_cdna::operand::Src;
use peacemaker_cdna::{oracle, Arch, Inst};
use std::collections::HashSet;

const ARCH: Arch = Arch::Gfx942;

fn v(n: u8) -> Option<Src> {
    Some(Src::Vgpr(n))
}

fn ds(op: DsOp, regs: (u8, u8, u8, u8), offset0: u8, offset1: u8) -> Ds {
    let (wv, w0, w1) = op.widths();
    let (rv, ra, r0, r1) = regs;
    Ds {
        op,
        vdst: if wv > 0 { v(rv) } else { None },
        addr: if op.has_addr() { v(ra) } else { None },
        data0: if w0 > 0 { v(r0) } else { None },
        data1: if w1 > 0 { v(r1) } else { None },
        offset0,
        offset1,
        gds: false,
    }
}

/// (vdst, addr, data0, data1) bases. Even-aligned sets suit every op; the
/// odd set only ops whose operands are all single dwords.
const EVEN_SETS: [(u8, u8, u8, u8); 3] = [(4, 1, 8, 12), (0, 255, 2, 6), (252, 254, 248, 244)];
const ODD_SET: (u8, u8, u8, u8) = (7, 3, 5, 9);

fn single_dword(op: DsOp) -> bool {
    let (a, b, c) = op.widths();
    a <= 1 && b <= 1 && c <= 1
}

fn cases() -> Vec<Ds> {
    let mut out = Vec::new();
    for &op in DsOp::ALL {
        let mut sets = EVEN_SETS.to_vec();
        if single_dword(op) {
            sets.push(ODD_SET)
        }
        // offset0/offset1 byte pairs: 0, small, max, and each byte alone.
        // ds_swizzle_b32 offsets are modes, not byte offsets: the curated list below.
        let offsets: Vec<(u8, u8)> = if op == DsOp::Nop || op == DsOp::SwizzleB32 {
            vec![(0, 0)]
        } else if op.two_addr() {
            vec![
                (0, 0),
                (1, 0),
                (0, 1),
                (4, 8),
                (255, 255),
                (255, 0),
                (0, 255),
            ]
        } else {
            // 0, 4, 255, 256, 0x1234, 65535 as {offset1, offset0}.
            [0u32, 4, 255, 256, 0x1234, 0xffff]
                .iter()
                .map(|&o| Ds::split_offset(o).unwrap())
                .collect()
        };
        for (i, &regs) in sets.iter().enumerate() {
            for (j, &(o0, o1)) in offsets.iter().enumerate() {
                // Every offset with the first set; the other sets with a rotating subset.
                if i == 0 || (i + j) % 3 == 0 {
                    out.push(ds(op, regs, o0, o1))
                }
            }
        }
    }
    // ds_swizzle_b32: every printed mode (quad perm, bitmask forms, rotate, FFT).
    for imm in [
        0u16, 0x8000, 0x80e4, 0x80ff, 0x041f, 0x401f, 0x1c1f, 0x001f, 0x0020, 0x181f, 0xc000,
        0xc400, 0xc020, 0xe000, 0xe01f,
    ] {
        out.push(ds(
            DsOp::SwizzleB32,
            EVEN_SETS[0],
            imm as u8,
            (imm >> 8) as u8,
        ));
    }
    out
}

#[test]
fn every_op_is_covered() {
    let seen: HashSet<DsOp> = cases().iter().map(|d| d.op).collect();
    for &op in DsOp::ALL {
        assert!(seen.contains(&op), "{op:?} missing from the case list");
    }
}

#[test]
fn opcode_table_round_trips() {
    for &op in DsOp::ALL {
        let code = op
            .opcode(ARCH)
            .unwrap_or_else(|| panic!("{op:?} has no gfx942 opcode"));
        assert_eq!(DsOp::from_opcode(ARCH, code), Some(op), "{op:?}");
    }
    // 8-bit OP field [24:17]: the b96/b128 opcodes set bit 24 of the first dword.
    let mut w = Vec::new();
    ds(DsOp::ReadB128, EVEN_SETS[0], 0, 0)
        .encode(ARCH, &mut w)
        .unwrap();
    assert_eq!(w[0], 0xd9fe_0000);
}

#[test]
fn encode_text_decode_match_llvm() {
    if oracle::disabled() {
        eprintln!("enc_ds: ROCm disabled, skipping llvm-mc comparison");
        return;
    }
    let cases = cases();
    let texts: Vec<String> = cases.iter().map(|d| d.to_string()).collect();
    let assembled = oracle::mc_encode(ARCH, &texts).unwrap();
    let mut bytes = Vec::new();
    for (d, (text, (llvm_text, llvm_bytes))) in cases.iter().zip(texts.iter().zip(&assembled)) {
        let mut words = Vec::new();
        d.encode(ARCH, &mut words).unwrap();
        assert_eq!(words.len(), 2, "{text}");
        assert_eq!(
            &oracle::bytes(&words),
            llvm_bytes,
            "encode bytes: {text} ({d:?})"
        );
        assert_eq!(llvm_text, text, "llvm re-print of {d:?}");
        bytes.push(llvm_bytes.clone());
    }
    let disassembled = oracle::mc_disassemble(ARCH, &bytes).unwrap();
    for ((d, text), (dis, b)) in cases
        .iter()
        .zip(&texts)
        .zip(disassembled.iter().zip(&bytes))
    {
        assert_eq!(dis, text, "llvm disassembly of {b:02x?} ({d:?})");
        let words: Vec<u32> = b
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (back, n) = Ds::decode(ARCH, &words).unwrap();
        assert_eq!((&back, n), (d, 2), "decode {text}");
        let (inst, n) = Inst::decode(ARCH, &words).unwrap();
        assert_eq!((inst, n), (Inst::Ds(d.clone()), 2), "Inst::decode {text}");
    }
}

/// Every one of the 65536 `ds_swizzle_b32` offsets: our printer equals the
/// llvm disassembler's, decode is the identity, and llvm accepts what we
/// print (a non-canonical offset is allowed to re-assemble to a different
/// offset that prints the same text).
#[test]
fn swizzle_offsets_exhaustive() {
    if oracle::disabled() {
        eprintln!("enc_ds: ROCm disabled, skipping llvm-mc comparison");
        return;
    }
    let all: Vec<Ds> = (0..=u16::MAX)
        .map(|i| ds(DsOp::SwizzleB32, EVEN_SETS[0], i as u8, (i >> 8) as u8))
        .collect();
    let mut bytes = Vec::with_capacity(all.len());
    for d in &all {
        let mut w = Vec::new();
        d.encode(ARCH, &mut w).unwrap();
        bytes.push(oracle::bytes(&w));
    }
    let disassembled = oracle::mc_disassemble(ARCH, &bytes).unwrap();
    for (d, dis) in all.iter().zip(&disassembled) {
        assert_eq!(&d.to_string(), dis, "offset {:#06x}", d.offset());
    }
    let texts: Vec<String> = all.iter().map(|d| d.to_string()).collect();
    let assembled = oracle::mc_encode(ARCH, &texts).unwrap();
    let mut canonical = 0usize;
    let mut modes: HashSet<&str> = HashSet::new();
    for ((d, text), (llvm_text, llvm_bytes)) in all.iter().zip(&texts).zip(&assembled) {
        let words: Vec<u32> = llvm_bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (back, _) = Ds::decode(ARCH, &words).unwrap();
        assert_eq!(
            &back.to_string(),
            llvm_text,
            "decode of llvm's bytes for offset {:#06x}",
            d.offset()
        );
        if back == *d {
            canonical += 1;
            assert_eq!(llvm_text, text, "re-print of offset {:#06x}", d.offset());
            for m in [
                "QUAD_PERM",
                "BITMASK_PERM",
                "BROADCAST",
                "SWAP",
                "REVERSE",
                "ROTATE",
                "FFT",
            ] {
                if text.contains(m) {
                    modes.insert(m);
                }
            }
        }
    }
    assert!(
        canonical > 1000,
        "only {canonical} canonical swizzle offsets"
    );
    for m in [
        "QUAD_PERM",
        "BITMASK_PERM",
        "BROADCAST",
        "SWAP",
        "REVERSE",
        "ROTATE",
        "FFT",
    ] {
        assert!(modes.contains(m), "no canonical {m} offset");
    }
}

#[test]
fn llvm_rejects_what_we_reject() {
    if oracle::disabled() {
        eprintln!("enc_ds: ROCm disabled, skipping llvm-mc comparison");
        return;
    }
    for bad in [
        "ds_read_b32 v1, v2 gds",
        "ds_read2_b32 v[0:1], v2 offset0:255 offset1:255 gds",
        "ds_read_b32 v1, v2 offset:65536",
        "ds_write_b32 v1, v2 offset:65536",
        "ds_read2_b32 v[0:1], v2 offset0:256",
        "ds_read2st64_b64 v[0:3], v2 offset1:256",
        "ds_nop offset:3",
        "ds_nop gds",
        "ds_read_b64 v[1:2], v3",
        "ds_read_b96 v[1:3], v2",
        "ds_read_b128 v[1:4], v3",
        "ds_write_b64 v1, v[3:4]",
        "ds_read2_b32 v[1:2], v2",
        "ds_write_b32 v1, s2",
    ] {
        assert!(
            oracle::mc_encode(ARCH, &[bad.to_string()]).is_err(),
            "llvm-mc accepted `{bad}`"
        );
    }
}

fn words_of(d: &Ds) -> Vec<u32> {
    let mut w = Vec::new();
    d.encode(ARCH, &mut w).unwrap();
    w
}

#[test]
fn encoder_rejections() {
    let base = |op| ds(op, EVEN_SETS[0], 0, 0);
    let mut bad: Vec<(&str, Ds)> = Vec::new();
    bad.push((
        "gds",
        Ds {
            gds: true,
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "gds nop",
        Ds {
            gds: true,
            ..base(DsOp::Nop)
        },
    ));
    bad.push((
        "nop offset0",
        Ds {
            offset0: 1,
            ..base(DsOp::Nop)
        },
    ));
    bad.push((
        "nop offset1",
        Ds {
            offset1: 1,
            ..base(DsOp::Nop)
        },
    ));
    bad.push((
        "agpr vdst",
        Ds {
            vdst: Some(Src::Agpr(1)),
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "agpr data0",
        Ds {
            data0: Some(Src::Agpr(2)),
            ..base(DsOp::WriteB32)
        },
    ));
    bad.push((
        "sgpr addr",
        Ds {
            addr: Some(Src::Sgpr(2)),
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "literal data",
        Ds {
            data0: Some(Src::Lit(1)),
            ..base(DsOp::WriteB32)
        },
    ));
    bad.push((
        "inline data",
        Ds {
            data0: Some(Src::Int(1)),
            ..base(DsOp::AddU32)
        },
    ));
    bad.push((
        "odd b64 vdst",
        Ds {
            vdst: v(3),
            ..base(DsOp::ReadB64)
        },
    ));
    bad.push((
        "odd b96 vdst",
        Ds {
            vdst: v(5),
            ..base(DsOp::ReadB96)
        },
    ));
    bad.push((
        "odd b128 vdst",
        Ds {
            vdst: v(1),
            ..base(DsOp::ReadB128)
        },
    ));
    bad.push((
        "odd read2_b32 vdst",
        Ds {
            vdst: v(1),
            ..base(DsOp::Read2B32)
        },
    ));
    bad.push((
        "odd write_b64 data",
        Ds {
            data0: v(3),
            ..base(DsOp::WriteB64)
        },
    ));
    bad.push((
        "odd write2_b64 data1",
        Ds {
            data1: v(5),
            ..base(DsOp::Write2B64)
        },
    ));
    bad.push((
        "b128 past v255",
        Ds {
            vdst: v(254),
            ..base(DsOp::ReadB128)
        },
    ));
    bad.push((
        "b64 past v255",
        Ds {
            vdst: v(255),
            ..base(DsOp::ReadB64)
        },
    ));
    bad.push((
        "missing vdst",
        Ds {
            vdst: None,
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "missing addr",
        Ds {
            addr: None,
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "missing data0",
        Ds {
            data0: None,
            ..base(DsOp::WriteB32)
        },
    ));
    bad.push((
        "missing data1",
        Ds {
            data1: None,
            ..base(DsOp::Write2B32)
        },
    ));
    bad.push((
        "extra vdst on write",
        Ds {
            vdst: v(2),
            ..base(DsOp::WriteB32)
        },
    ));
    bad.push((
        "extra data1 on write",
        Ds {
            data1: v(2),
            ..base(DsOp::WriteB32)
        },
    ));
    bad.push((
        "extra data0 on read",
        Ds {
            data0: v(2),
            ..base(DsOp::ReadB32)
        },
    ));
    bad.push((
        "extra addr on nop",
        Ds {
            addr: v(1),
            ..base(DsOp::Nop)
        },
    ));
    bad.push((
        "extra data0 on swizzle",
        Ds {
            data0: v(2),
            ..base(DsOp::SwizzleB32)
        },
    ));
    bad.push((
        "extra vdst on non-rtn add",
        Ds {
            vdst: v(2),
            ..base(DsOp::AddU32)
        },
    ));
    for (what, d) in &bad {
        let mut out = vec![0xdead_beef];
        assert!(d.encode(ARCH, &mut out).is_err(), "{what}: {d:?} encoded");
        assert_eq!(
            out,
            vec![0xdead_beef],
            "{what}: a failed encode must not emit dwords"
        );
        assert!(Inst::Ds(d.clone()).words(ARCH).is_err(), "{what}");
    }
}

#[test]
fn offset_overflow_rejected() {
    assert_eq!(Ds::split_offset(0).unwrap(), (0, 0));
    assert_eq!(Ds::split_offset(4).unwrap(), (4, 0));
    assert_eq!(Ds::split_offset(0x1234).unwrap(), (0x34, 0x12));
    assert_eq!(Ds::split_offset(65535).unwrap(), (255, 255));
    assert!(Ds::split_offset(65536).is_err());
    assert!(Ds::split_offset(u32::MAX).is_err());
    assert_eq!(Ds::pair_offsets(0, 0).unwrap(), (0, 0));
    assert_eq!(Ds::pair_offsets(255, 255).unwrap(), (255, 255));
    assert!(Ds::pair_offsets(256, 0).is_err());
    assert!(Ds::pair_offsets(0, 256).is_err());
    assert!(Ds::pair_offsets(1000, 1000).is_err());
}

#[test]
fn decoder_rejections() {
    let good = words_of(&ds(DsOp::ReadB32, EVEN_SETS[0], 0, 0));
    assert!(Ds::decode(ARCH, &good).is_ok());
    assert!(Ds::decode(ARCH, &good[..1]).is_err(), "one dword");
    assert!(Ds::decode(ARCH, &[]).is_err(), "no dwords");
    let mut w = good.clone();
    w[0] |= 1 << 16;
    assert!(Ds::decode(ARCH, &w).is_err(), "GDS bit");
    let mut w = good.clone();
    w[0] |= 1 << 25;
    assert!(Ds::decode(ARCH, &w).is_err(), "ACC bit");
    let mut w = good.clone();
    w[0] = (w[0] & !0xfc00_0000) | 0xdc00_0000;
    assert!(Ds::decode(ARCH, &w).is_err(), "FLAT encoding");
    // DS opcode 1 (ds_sub_u32) is outside the table.
    let mut w = good.clone();
    w[0] = (w[0] & !(0xff << 17)) | 1 << 17;
    assert!(Ds::decode(ARCH, &w).is_err(), "opcode outside the table");
    // Unused operand fields must be zero (llvm's disassembler rejects them too).
    let mut w = good.clone();
    w[1] |= 5 << 8; // data0 on a read
    assert!(Ds::decode(ARCH, &w).is_err(), "data0 on read");
    let mut w = good.clone();
    w[1] |= 5 << 16; // data1 on a read
    assert!(Ds::decode(ARCH, &w).is_err(), "data1 on read");
    let wr = words_of(&ds(DsOp::WriteB32, EVEN_SETS[0], 0, 0));
    let mut w = wr.clone();
    w[1] |= 3 << 24; // vdst on a write
    assert!(Ds::decode(ARCH, &w).is_err(), "vdst on write");
    let nop = words_of(&ds(DsOp::Nop, EVEN_SETS[0], 0, 0));
    for patch in [
        (0usize, 3u32),
        (0, 3 << 8),
        (1, 1),
        (1, 1 << 8),
        (1, 1 << 16),
        (1, 1 << 24),
    ] {
        let mut w = nop.clone();
        w[patch.0] |= patch.1;
        assert!(Ds::decode(ARCH, &w).is_err(), "ds_nop with {patch:?}");
    }
    // Misaligned tuples.
    let mut w = words_of(&ds(DsOp::ReadB64, EVEN_SETS[0], 0, 0));
    w[1] = (w[1] & 0x00ff_ffff) | 3 << 24;
    assert!(Ds::decode(ARCH, &w).is_err(), "odd b64 vdst");
}

#[test]
fn llvm_disassembler_agrees_on_rejections() {
    if oracle::disabled() {
        eprintln!("enc_ds: ROCm disabled, skipping llvm-mc comparison");
        return;
    }
    let words = |w0: u32, w1: u32| vec![w0, w1];
    let good = words_of(&ds(DsOp::ReadB32, EVEN_SETS[0], 0, 0));
    let wr = words_of(&ds(DsOp::WriteB32, EVEN_SETS[0], 0, 0));
    let nop = words_of(&ds(DsOp::Nop, EVEN_SETS[0], 0, 0));
    let sw = words_of(&ds(DsOp::SwizzleB32, EVEN_SETS[0], 0, 0));
    let rtn = words_of(&ds(DsOp::AddRtnU32, EVEN_SETS[0], 0, 0));
    for (what, w) in [
        ("data0 on read", words(good[0], good[1] | 5 << 8)),
        ("data1 on read", words(good[0], good[1] | 5 << 16)),
        ("vdst on write", words(wr[0], wr[1] | 3 << 24)),
        ("data0 on swizzle", words(sw[0], sw[1] | 5 << 8)),
        ("data1 on rtn add", words(rtn[0], rtn[1] | 5 << 16)),
        ("nop addr", words(nop[0], nop[1] | 1)),
        ("nop offset0", words(nop[0] | 3, nop[1])),
        ("nop gds", words(nop[0] | 1 << 16, nop[1])),
    ] {
        assert!(
            oracle::mc_disassemble(ARCH, &[oracle::bytes(&w)]).is_err(),
            "llvm disassembled {what}"
        );
        assert!(Ds::decode(ARCH, &w).is_err(), "decode accepted {what}");
    }
    // GDS and ACC: llvm decodes them (GDS prints as nothing, ACC as agprs); we
    // do not model either and refuse the bytes.
    let gds =
        oracle::mc_disassemble(ARCH, &[oracle::bytes(&[good[0] | 1 << 16, good[1]])]).unwrap();
    assert_eq!(
        gds,
        ["ds_read_b32 v4, v1"],
        "llvm prints no gds modifier on gfx942"
    );
    let acc =
        oracle::mc_disassemble(ARCH, &[oracle::bytes(&[good[0] | 1 << 25, good[1]])]).unwrap();
    assert_eq!(acc, ["ds_read_b32 a4, v1"]);
}

fn span(file: RegFile, base: u8, len: u8) -> RegSpan {
    RegSpan::new(file, base, len)
}

fn has_use(fp: &Footprint, s: RegSpan) -> bool {
    fp.uses.contains(&s)
}

#[test]
fn footprints() {
    let exec = span(RegFile::Exec, 0, 2);
    let m0 = span(RegFile::M0, 0, 1);

    // ds_read2_b32 v[4:5], v1
    let fp = ds(DsOp::Read2B32, (4, 1, 8, 12), 1, 2).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 2)]);
    assert_eq!(fp.uses, vec![exec, m0, span(RegFile::Vgpr, 1, 1)]);

    // ds_read2st64_b64 v[4:7], v1
    let fp = ds(DsOp::Read2st64B64, (4, 1, 8, 12), 1, 2).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 4)]);

    // ds_write2_b64 v1, v[8:9], v[12:13]
    let fp = ds(DsOp::Write2B64, (4, 1, 8, 12), 1, 2).footprint();
    assert!(fp.defs.is_empty());
    assert_eq!(
        fp.uses,
        vec![
            exec,
            m0,
            span(RegFile::Vgpr, 1, 1),
            span(RegFile::Vgpr, 8, 2),
            span(RegFile::Vgpr, 12, 2)
        ]
    );

    // ds_write_b128 v1, v[8:11]
    let fp = ds(DsOp::WriteB128, (4, 1, 8, 12), 0, 0).footprint();
    assert_eq!(
        fp.uses,
        vec![
            exec,
            m0,
            span(RegFile::Vgpr, 1, 1),
            span(RegFile::Vgpr, 8, 4)
        ]
    );

    // ds_add_u32 (no return) and ds_add_rtn_u32
    let fp = ds(DsOp::AddU32, (4, 1, 8, 12), 0, 0).footprint();
    assert!(fp.defs.is_empty());
    assert_eq!(
        fp.uses,
        vec![
            exec,
            m0,
            span(RegFile::Vgpr, 1, 1),
            span(RegFile::Vgpr, 8, 1)
        ]
    );
    let fp = ds(DsOp::AddRtnU32, (4, 1, 8, 12), 0, 0).footprint();
    assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 1)]);
    assert!(has_use(&fp, m0) && has_use(&fp, exec));

    // Cross-lane ops access no LDS memory: no M0.
    for op in [DsOp::SwizzleB32, DsOp::PermuteB32, DsOp::BpermuteB32] {
        let fp = ds(op, (4, 1, 8, 12), 0, 0).footprint();
        assert!(has_use(&fp, exec), "{op:?}");
        assert!(!has_use(&fp, m0), "{op:?} must not use M0");
        assert_eq!(fp.defs, vec![span(RegFile::Vgpr, 4, 1)], "{op:?}");
    }
    let fp = ds(DsOp::PermuteB32, (4, 1, 8, 12), 0, 0).footprint();
    assert_eq!(
        fp.uses,
        vec![exec, span(RegFile::Vgpr, 1, 1), span(RegFile::Vgpr, 8, 1)]
    );

    // ds_nop touches nothing.
    assert_eq!(
        ds(DsOp::Nop, (4, 1, 8, 12), 0, 0).footprint(),
        Footprint::default()
    );

    // Every op: the footprint matches widths().
    for d in cases() {
        let fp = d.footprint();
        let (wv, _, _) = d.op.widths();
        let defs: u32 = fp.defs.iter().map(|s| u32::from(s.len)).sum();
        assert_eq!(defs, u32::from(wv), "{d}");
        assert_eq!(Inst::Ds(d.clone()).footprint(), fp, "{d}");
    }
}

#[test]
fn mem_kinds() {
    let expect = |op: DsOp| match op {
        DsOp::ReadB32
        | DsOp::ReadB64
        | DsOp::ReadB96
        | DsOp::ReadB128
        | DsOp::Read2B32
        | DsOp::Read2B64
        | DsOp::Read2st64B32
        | DsOp::Read2st64B64
        | DsOp::ReadU8
        | DsOp::ReadI8
        | DsOp::ReadU16
        | DsOp::ReadI16 => MemKind::DsLoad,
        DsOp::WriteB8
        | DsOp::WriteB16
        | DsOp::WriteB32
        | DsOp::WriteB64
        | DsOp::WriteB96
        | DsOp::WriteB128
        | DsOp::Write2B32
        | DsOp::Write2B64
        | DsOp::Write2st64B32
        | DsOp::Write2st64B64
        | DsOp::Nop => MemKind::DsStore,
        DsOp::AddU32 | DsOp::MaxU32 | DsOp::MaxI32 | DsOp::MaxF32 | DsOp::MinF32 | DsOp::AddF32 => {
            MemKind::DsAtomic
        }
        DsOp::AddRtnU32 => MemKind::DsAtomicRtn,
        DsOp::SwizzleB32 | DsOp::PermuteB32 | DsOp::BpermuteB32 => MemKind::DsXlane,
    };
    for &op in DsOp::ALL {
        let d = ds(op, EVEN_SETS[0], 0, 0);
        assert_eq!(d.mem_kind(), expect(op), "{op:?}");
        assert_eq!(Inst::Ds(d).memory(), Some(expect(op)), "{op:?}");
    }
}

#[test]
fn widths_and_shapes() {
    assert_eq!(DsOp::Read2B32.widths(), (2, 0, 0));
    assert_eq!(DsOp::Read2B64.widths(), (4, 0, 0));
    assert_eq!(DsOp::Write2B32.widths(), (0, 1, 1));
    assert_eq!(DsOp::Write2st64B64.widths(), (0, 2, 2));
    assert_eq!(DsOp::ReadB96.widths(), (3, 0, 0));
    assert_eq!(DsOp::AddRtnU32.widths(), (1, 1, 0));
    assert_eq!(DsOp::Nop.widths(), (0, 0, 0));
    let two: Vec<DsOp> = DsOp::ALL.iter().copied().filter(|o| o.two_addr()).collect();
    assert_eq!(
        two,
        [
            DsOp::Read2B32,
            DsOp::Read2B64,
            DsOp::Read2st64B32,
            DsOp::Read2st64B64,
            DsOp::Write2B32,
            DsOp::Write2B64,
            DsOp::Write2st64B32,
            DsOp::Write2st64B64
        ]
    );
}

#[test]
fn display_spellings() {
    let d = |op, regs, o0, o1| ds(op, regs, o0, o1).to_string();
    let r = (4, 1, 8, 12);
    assert_eq!(d(DsOp::ReadB32, r, 0, 0), "ds_read_b32 v4, v1");
    assert_eq!(
        d(DsOp::ReadB32, r, 0xff, 0xff),
        "ds_read_b32 v4, v1 offset:65535"
    );
    assert_eq!(
        d(DsOp::ReadB128, r, 16, 0),
        "ds_read_b128 v[4:7], v1 offset:16"
    );
    assert_eq!(d(DsOp::Read2B32, r, 0, 0), "ds_read2_b32 v[4:5], v1");
    assert_eq!(
        d(DsOp::Read2B32, r, 3, 0),
        "ds_read2_b32 v[4:5], v1 offset0:3"
    );
    assert_eq!(
        d(DsOp::Read2B32, r, 0, 3),
        "ds_read2_b32 v[4:5], v1 offset1:3"
    );
    assert_eq!(
        d(DsOp::Read2st64B64, r, 1, 2),
        "ds_read2st64_b64 v[4:7], v1 offset0:1 offset1:2"
    );
    assert_eq!(
        d(DsOp::Write2B64, r, 1, 2),
        "ds_write2_b64 v1, v[8:9], v[12:13] offset0:1 offset1:2"
    );
    assert_eq!(d(DsOp::WriteB96, r, 0, 0), "ds_write_b96 v1, v[8:10]");
    assert_eq!(d(DsOp::AddU32, r, 8, 0), "ds_add_u32 v1, v8 offset:8");
    assert_eq!(d(DsOp::AddRtnU32, r, 0, 0), "ds_add_rtn_u32 v4, v1, v8");
    assert_eq!(d(DsOp::SwizzleB32, r, 0, 0), "ds_swizzle_b32 v4, v1");
    assert_eq!(
        d(DsOp::SwizzleB32, r, 0xe4, 0x80),
        "ds_swizzle_b32 v4, v1 offset:swizzle(QUAD_PERM,0,1,2,3)"
    );
    assert_eq!(
        d(DsOp::SwizzleB32, r, 0x1f, 0x40),
        "ds_swizzle_b32 v4, v1 offset:swizzle(SWAP,16)"
    );
    assert_eq!(
        d(DsOp::SwizzleB32, r, 0x1f, 0x1c),
        "ds_swizzle_b32 v4, v1 offset:swizzle(REVERSE,8)"
    );
    assert_eq!(
        d(DsOp::SwizzleB32, r, 0x1e, 0),
        "ds_swizzle_b32 v4, v1 offset:swizzle(BROADCAST,2,0)"
    );
    assert_eq!(
        d(DsOp::SwizzleB32, r, 0, 0xe0),
        "ds_swizzle_b32 v4, v1 offset:swizzle(FFT,0)"
    );
    assert_eq!(d(DsOp::PermuteB32, r, 0, 0), "ds_permute_b32 v4, v1, v8");
    assert_eq!(
        d(DsOp::BpermuteB32, r, 4, 0),
        "ds_bpermute_b32 v4, v1, v8 offset:4"
    );
    assert_eq!(d(DsOp::Nop, r, 0, 0), "ds_nop");
}
