// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! T3/T9 gate tests for the C3b second opinion. Tool-gated: with a pinned
//! toolchain (`PEACEMAKER_ROCM`, else `ROCM_PATH`, else `/opt/rocm`) the live tools
//! must agree; without one the committed objdump fixture is the reference.
//! A set-but-unusable `PEACEMAKER_ROCM` fails, never skips.
//!
//! Coverage is all six KT48 kernels (4,780 instructions): boundaries,
//! mnemonics and canonical operand text equal the pinned objdump's for every
//! instruction, branch targets equal its `<sym+0x…>` annotations, and
//! assembler parity holds with the lossless/reported split.

use super::{canonical, kernel_lines};
use crate::text::support::{self, classify, lossy_mask, Parity};
use peacemaker_ir::{
    codec::gfx12,
    inst::{Arch, FormFields, Inst},
    isa,
    operand::Operand,
};
use smallvec::SmallVec;

const KT48_VA: u32 = 0x7f00;
#[test]
fn dpp_text_roundtrips_and_assembles_to_the_shipped_words() {
    for (words, expected) in [
        (
            [0x0624_24fa, 0xff09_0812],
            "v_add_f32_dpp v18, v18, v18 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1",
        ),
        (
            [0x7e04_02fa, 0xff09_0101],
            "v_mov_b32_dpp v2, v1 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
        ),
        (
            [0x7fb2_02fa, 0xff01_50ed],
            "v_mov_b32_dpp v217, v237 row_share:0 row_mask:0xf bank_mask:0xf",
        ),
    ] {
        let (inst, used) = gfx12::decode(&words).expect("shipped DPP encoding");
        assert_eq!(used, 2);
        assert_eq!(canonical(&inst, Arch::Gfx1201).unwrap(), expected);
        let parsed = crate::text::parse_line(expected, Arch::Gfx1201).unwrap();
        assert_eq!(gfx12::encode(&parsed).unwrap().as_slice(), &words);
        if let Some(tc) = support::toolchain().expect("toolchain discovery") {
            assert_eq!(
                support::mc_batch(&tc.mc, &[expected.into()]).unwrap()[0],
                words
            );
        }
    }
}

#[test]
fn every_table_example_matches_pinned_llvm_mc_except_declared_dont_cares() {
    let Some(tc) = support::toolchain().expect("toolchain discovery") else {
        return;
    };
    let rows = isa::gfx12();
    let lines: Vec<String> = rows.iter().map(|row| row.sample.into()).collect();
    let assembled = support::mc_batch(&tc.mc, &lines).expect("assemble every table example");
    let mut reported = Vec::new();
    for (row, words) in rows.iter().zip(&assembled) {
        let original: Vec<u32> = row
            .encoding
            .split_whitespace()
            .map(|w| u32::from_str_radix(w, 16).unwrap())
            .collect();
        let (inst, consumed) = gfx12::decode(&original).unwrap();
        assert_eq!(consumed, original.len(), "{}", row.name);
        assert_eq!(
            gfx12::encode(&inst).unwrap().as_slice(),
            original,
            "{}",
            row.name
        );
        match classify(&inst, words).unwrap_or(Parity::Mismatch) {
            Parity::Exact => {}
            Parity::Reported => reported.push(row.name),
            Parity::Mismatch => panic!(
                "{}: llvm-mc={words:08x?}, encoded={original:08x?}",
                row.name
            ),
        }
    }
    eprintln!(
        "{} table rows, {} syntax-lossy samples",
        rows.len(),
        reported.len()
    );
}

#[test]
fn buffer_cache_hints_roundtrip_through_objdump_and_llvm_mc_syntax() {
    for (words, line) in [
        (
            [0xc405_005c, 0x409c_50b8, 0x0000_00b6],
            "buffer_load_b32 v184, v182, s[40:43], s92 offen th:TH_LOAD_NT scope:SCOPE_SYS",
        ),
        (
            [0xc406_8051, 0x40fc_5020, 0x0000_008a],
            "buffer_store_b32 v32, v138, s[40:43], s81 offen th:TH_STORE_NT_WB scope:SCOPE_SYS",
        ),
    ] {
        let (inst, used) = gfx12::decode(&words).unwrap();
        assert_eq!(used, 3);
        assert_eq!(canonical(&inst, Arch::Gfx1201).unwrap(), line);
        let parsed = crate::text::parse_line(line, Arch::Gfx1201).unwrap();
        assert_eq!(
            (&parsed.operands, &parsed.fields, &parsed.mods),
            (&inst.operands, &inst.fields, &inst.mods),
            "{line}"
        );
        assert_eq!(gfx12::encode(&parsed).unwrap().as_slice(), &words);
        if let Some(tc) = support::toolchain().expect("toolchain discovery") {
            assert_eq!(support::mc_batch(&tc.mc, &[line.into()]).unwrap()[0], words);
        }
    }
}

const KT48_SIZE: usize = 10_604;

/// Per-kernel census pins `(symbol, instructions, branches, distinct branch
/// targets)` in VA order. The last row is the selected kernel; its
/// 1,696/67/57 pins predate the all-six extension.
const KT48_EXPECTED: [(&str, usize, usize, usize); 6] = [
    ("attention_fp8_e4m3_fa2_gqa_gfx1201", 1297, 57, 49),
    ("attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201", 70, 1, 1),
    ("attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201", 253, 2, 1),
    ("attention_fp8_e4m3_fa2_gqa_partial_gfx1201", 1334, 60, 50),
    ("attention_fp8_e4m3_fa2_gqa_merge_gfx1201", 130, 9, 6),
    (
        "attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201",
        1696,
        67,
        57,
    ),
];

struct Kt48 {
    name: &'static str,
    va: u32,
    size: usize,
    addrs: Vec<u32>,
    words: Vec<Vec<u32>>,
    insts: Vec<Inst>,
}

fn decode_all() -> Vec<Kt48> {
    let mut out = Vec::new();
    for (name, va, stream) in support::kt48_all_streams() {
        let mut addrs = Vec::new();
        let mut words = Vec::new();
        let mut insts = Vec::new();
        let mut index = 0usize;
        while index < stream.len() {
            let (inst, n) = gfx12::decode(&stream[index..])
                .unwrap_or_else(|e| panic!("{name} word {index}: {e}"));
            addrs.push(va + (index as u32) * 4);
            words.push(stream[index..index + n].to_vec());
            insts.push(inst);
            index += n;
        }
        let size = stream.len() * 4;
        assert_eq!(index * 4, size, "{name} size");
        out.push(Kt48 {
            name,
            va,
            size,
            addrs,
            words,
            insts,
        });
    }
    assert_eq!(out.len(), KT48_EXPECTED.len(), "six-kernel census");
    for (kt, (name, insts, _, _)) in out.iter().zip(KT48_EXPECTED.iter()) {
        assert_eq!(kt.name, *name, "kernel order");
        assert_eq!(kt.insts.len(), *insts, "{name} census");
    }
    // The selected kernel's range pin survives the extension.
    let sel = out.last().expect("six kernels");
    assert_eq!((sel.va, sel.size), (KT48_VA, KT48_SIZE), "selected range");
    out
}

fn objdump_all() -> Vec<support::ObjdumpLine> {
    let lines = match support::toolchain().expect("toolchain discovery must not fail") {
        Some(tc) => {
            let output = std::process::Command::new(&tc.objdump)
                .args(["--mcpu=gfx1201", "-d"])
                .arg(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/kt48/hipcc.co"
                ))
                .output()
                .expect("run pinned llvm-objdump");
            assert!(
                output.status.success(),
                "pinned llvm-objdump failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let live = support::parse_objdump_stdout(&String::from_utf8_lossy(&output.stdout));
            // The committed fixture must equal the live tool output.
            let pinned = support::pinned_all_objdump_fixture();
            assert_eq!(
                live.len(),
                pinned.len(),
                "pinned all-objdump fixture is stale (line count)"
            );
            for (l, p) in live.iter().zip(pinned.iter()) {
                assert_eq!(l.addr, p.addr, "stale fixture addr");
                assert_eq!(l.text, p.text, "stale fixture text");
                assert_eq!(l.words, p.words, "stale fixture words");
            }
            live
        }
        None => support::pinned_all_objdump_fixture(),
    };
    lines
}

/// T3: boundaries, mnemonics and canonical operand text of all 4,780
/// instructions equal the pinned objdump's; 164 distinct branch targets
/// equal its `<sym+0x…>` annotations.
#[test]
fn t3_decoder_agrees_with_pinned_objdump() {
    let kernels = decode_all();
    let lines = objdump_all();
    // Boundaries: objdump addresses must advance exactly as the codec's
    // widths say.
    const BRANCH_SET: [&str; 7] = [
        "s_branch",
        "s_cbranch_scc0",
        "s_cbranch_scc1",
        "s_cbranch_vccz",
        "s_cbranch_vccnz",
        "s_cbranch_execz",
        "s_cbranch_execnz",
    ];
    let mut total = 0usize;
    let mut total_branches = 0usize;
    let mut total_targets = std::collections::BTreeSet::new();
    for (kt, (_, _, want_branches, want_targets)) in kernels.iter().zip(KT48_EXPECTED.iter()) {
        let ol: Vec<&support::ObjdumpLine> = lines
            .iter()
            .filter(|l| l.addr >= kt.va && l.addr < kt.va + kt.size as u32)
            .collect();
        assert_eq!(
            ol.len(),
            kt.insts.len(),
            "{} objdump lines in symbol range",
            kt.name
        );
        for (i, line) in ol.iter().enumerate() {
            assert_eq!(
                line.addr, kt.addrs[i],
                "{} boundary mismatch at instruction {i}",
                kt.name
            );
            assert_eq!(
                line.words.len(),
                kt.words[i].len(),
                "{} width mismatch at {:#x}",
                kt.name,
                line.addr
            );
            for (a, b) in line.words.iter().zip(kt.words[i].iter()) {
                assert_eq!(*a, *b, "{} word mismatch at {:#x}", kt.name, line.addr);
            }
        }
        // Canonical text equality; semantic disagreement is a failure.
        let mut diffs = Vec::new();
        let mut printed = Vec::new();
        for (i, inst) in kt.insts.iter().enumerate() {
            let text = canonical(inst, Arch::Gfx1201)
                .unwrap_or_else(|e| panic!("print {:#x}: {e}", kt.addrs[i]));
            printed.push(text.clone());
            if text != ol[i].text {
                diffs.push(format!(
                    "{:#x}:\n  ours: {text}\n  ref : {}",
                    kt.addrs[i], ol[i].text
                ));
            }
        }
        assert!(
            diffs.is_empty(),
            "{}: {} canonical text mismatches:\n{}",
            kt.name,
            diffs.len(),
            diffs.into_iter().take(10).collect::<Vec<_>>().join("\n")
        );
        // Branch targets from encodings (signed simm16, dwords from next
        // PC) equal the annotated `<sym+0x…>` targets; decoding from the
        // unsigned text operand instead gives wrong targets, so this
        // also pins the print/parse direction of the pitfall.
        let mut branches = 0usize;
        let mut targets = std::collections::BTreeSet::new();
        for (i, inst) in kt.insts.iter().enumerate() {
            let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
            if !BRANCH_SET.contains(&row.name) {
                continue;
            }
            branches += 1;
            let simm = match inst.operands.first() {
                Some(Operand::Imm(peacemaker_ir::operand::ImmField::Sopp(n))) => *n as i32,
                other => panic!("branch without simm16 at {:#x}: {other:?}", kt.addrs[i]),
            };
            let target = kt.addrs[i] as i32 + (kt.words[i].len() as i32) * 4 + simm * 4;
            let ann = &ol[i].ann;
            let hex = ann
                .strip_prefix(&format!("<{}+0x", kt.name))
                .and_then(|s| s.strip_suffix('>'))
                .unwrap_or_else(|| {
                    panic!("branch at {:#x} lacks annotation: {ann:?}", kt.addrs[i])
                });
            let expected = u32::from_str_radix(hex, 16).unwrap();
            assert_eq!(
                (target - kt.va as i32) as u32,
                expected,
                "branch target mismatch at {:#x}",
                kt.addrs[i]
            );
            targets.insert(expected);
            total_targets.insert((kt.name, expected));
            // The printed unsigned operand re-parses to the same simm.
            let printed_operand = printed[i].split_whitespace().nth(1).unwrap().to_owned();
            let reparsed = printed_operand.parse::<u32>().unwrap() as u16 as i16;
            assert_eq!(reparsed, simm as i16, "unsigned branch pitfall");
        }
        assert_eq!(branches, *want_branches, "{} branch census", kt.name);
        assert_eq!(targets.len(), *want_targets, "{} distinct targets", kt.name);
        total += kt.insts.len();
        total_branches += branches;
    }
    assert_eq!(total, 4780, "all six kernels census");
    assert_eq!(total_branches, 196, "all six kernels branch census");
    assert_eq!(total_targets.len(), 164, "all six kernels distinct targets");
}

/// T9: assembler parity for every KT48 instruction (all 4,780 have
/// lossless syntax, so the reported set is empty); a synthetic VOP3b
/// with `src2_unused = 0x00` lands in the reported set, differing only
/// in that field, without failing. This also discharges T3's
/// `mc(Inst::text(i)) == encode(i)` sentence.
#[test]
fn t9_assembler_parity_where_syntax_is_lossless() {
    let kernels = decode_all();
    let insts: Vec<Inst> = kernels
        .iter()
        .flat_map(|kt| kt.insts.iter().cloned())
        .collect();
    let addrs: Vec<u32> = kernels
        .iter()
        .flat_map(|kt| kt.addrs.iter().copied())
        .collect();
    assert_eq!(insts.len(), 4780, "all six kernels census");
    let lines = kernel_lines(&insts, Arch::Gfx1201).expect("all KT48 instructions print");
    // Synthetic VOP3b with the non-canonical benign fill first, so the
    // reported-set logic is exercised even without tools.
    let vop3b_index = insts
        .iter()
        .position(|inst| matches!(inst.fields, FormFields::Vop3b { .. }))
        .expect("KT48 holds VOP3b instructions");
    let mut synth = insts[vop3b_index].clone();
    synth.fields = FormFields::Vop3b { src2_unused: 0x00 };
    let synth = Inst::from_parts(
        Arch::Gfx1201,
        synth.op,
        synth.form,
        synth.fields.clone(),
        SmallVec::from_vec(synth.operands.to_vec()),
        synth.mods.clone(),
        synth.literal,
        peacemaker_ir::provenance::Provenance::default(),
    )
    .expect("synthetic VOP3b validates");
    let synth_text = canonical(&synth, Arch::Gfx1201).expect("synthetic prints");
    assert_eq!(
        synth_text, lines[vop3b_index],
        "don't-care fill is invisible in canonical text"
    );
    let enc_synth = gfx12::encode(&synth).expect("synthetic encodes");
    let enc_canon = gfx12::encode(&insts[vop3b_index]).expect("canonical encodes");
    assert_ne!(enc_synth.as_slice(), enc_canon.as_slice());
    let mask = lossy_mask(&synth);
    assert!(
        mask.iter().any(|m| *m != 0),
        "synthetic has unspellable bits"
    );
    let only_masked = enc_synth.len() == enc_canon.len()
        && enc_synth
            .iter()
            .zip(enc_canon.iter())
            .enumerate()
            .all(|(i, (a, b))| (a ^ b) & !mask.get(i).copied().unwrap_or(0) == 0);
    assert!(only_masked, "synthetic differs only in src2");
    match support::toolchain().expect("toolchain discovery") {
        Some(tc) => {
            let mc_words =
                support::mc_batch(&tc.mc, &lines).expect("pinned llvm-mc assembles KT48 text");
            let mut reported = Vec::new();
            let mut mismatched = Vec::new();
            for (i, (inst, mc)) in insts.iter().zip(mc_words.iter()).enumerate() {
                match classify(inst, mc).unwrap_or(Parity::Mismatch) {
                    Parity::Exact => {}
                    Parity::Reported => reported.push(i),
                    Parity::Mismatch => mismatched.push(format!(
                        "{:#x}: {} mc={mc:08X?} enc={:08X?}",
                        addrs[i],
                        lines[i],
                        gfx12::encode(inst).map(|e| e.to_vec()).unwrap_or_default()
                    )),
                }
            }
            assert!(
                mismatched.is_empty(),
                "{} assembler mismatches:\n{}",
                mismatched.len(),
                mismatched
                    .into_iter()
                    .take(10)
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            assert!(
                reported.is_empty(),
                "KT48 non-canonical list is not empty: {reported:?}"
            );
            // The synthetic lands reported through the same path: mc
            // fills src2 with 0x80 while the instruction holds 0x00.
            let mc_synth = support::mc_batch(&tc.mc, &[synth_text])
                .expect("mc assembles synthetic")[0]
                .clone();
            assert_eq!(
                classify(&synth, &mc_synth).unwrap(),
                Parity::Reported,
                "synthetic VOP3b must land in the reported set"
            );
        }
        None => {
            if std::env::var("PEACEMAKER_ROCM").is_ok() {
                panic!("PEACEMAKER_ROCM is set but unusable");
            }
            // No assembler: the reported-set logic still runs against
            // the canonical encoding (which is exactly what llvm-mc
            // fills for don't-care fields).
            eprintln!("no pinned toolchain: mc parity via canonical fill");
            assert_eq!(
                classify(&synth, enc_canon.as_slice()).unwrap(),
                Parity::Reported,
                "synthetic VOP3b must land in the reported set"
            );
        }
    }
}

#[test]
fn swmmac_index_key_prints_before_neg_lo_and_roundtrips() {
    for (words, expected) in [
        (
            [0xcc50_4b00, 0x7c52_1908],
            "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]",
        ),
        (
            [0xcc57_4800, 0x1c3a_1508],
            "v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14 index_key:1",
        ),
        (
            [0xcc54_4000, 0x1c3a_1508],
            "v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14",
        ),
    ] {
        let (inst, used) = gfx12::decode(&words).unwrap();
        assert_eq!(used, 2);
        assert_eq!(canonical(&inst, Arch::Gfx1201).unwrap(), expected);
        let parsed = crate::text::parse_line(expected, Arch::Gfx1201).unwrap();
        assert_eq!(gfx12::encode(&parsed).unwrap().as_slice(), &words);
    }
}
