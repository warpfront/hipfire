//! Parser gate tests: every canonical KT48 line parses back to the
//! instruction the codec decoded (semantics plus re-encoding), and the `.s`
//! source structure carries labels/directives for `lift_text`.

use super::{label_block_ids, parse_line, parse_source, SourceKind};
use crate::text::{print::canonical, support};
use peacemaker_ir::{
    codec::{gfx11, gfx12},
    inst::{Arch, Inst},
};

const KT48_VA: u32 = 0x7f00;
const KT48_SIZE: usize = 10_604;

fn decode_kt48() -> Vec<(u32, Vec<u32>, Inst)> {
    let (stream, va) = support::kt48_stream();
    assert_eq!(va, KT48_VA);
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < stream.len() {
        let (inst, n) = gfx12::decode(&stream[index..])
            .unwrap_or_else(|e| panic!("KT48 word {index}: {e}"));
        out.push((
            va + (index as u32) * 4,
            stream[index..index + n].to_vec(),
            inst,
        ));
        index += n;
    }
    assert_eq!(index * 4, KT48_SIZE);
    out
}

fn kt48_texts() -> Vec<(u32, Vec<u32>, String)> {
    support::pinned_objdump_fixture()
        .into_iter()
        .filter(|l| l.addr >= KT48_VA && l.addr < KT48_VA + KT48_SIZE as u32)
        .map(|l| (l.addr, l.words, l.text))
        .collect()
}

/// All six KT48 kernels as `(symbol, base VA, byte size, decoded)`.
fn decode_all() -> Vec<(&'static str, u32, usize, Vec<(u32, Vec<u32>, Inst)>)> {
    support::kt48_all_streams()
        .into_iter()
        .map(|(name, va, stream)| {
            let mut out = Vec::new();
            let mut index = 0usize;
            while index < stream.len() {
                let (inst, n) = gfx12::decode(&stream[index..])
                    .unwrap_or_else(|e| panic!("{name} word {index}: {e}"));
                out.push((
                    va + (index as u32) * 4,
                    stream[index..index + n].to_vec(),
                    inst,
                ));
                index += n;
            }
            let size = stream.len() * 4;
            assert_eq!(index * 4, size, "{name} size");
            (name, va, size, out)
        })
        .collect()
}

/// The all-symbols pinned fixture; tests filter to each kernel range.
fn texts_all() -> Vec<support::ObjdumpLine> {
    support::pinned_all_objdump_fixture()
}

/// The parser agrees with the codec on all 4,780 instructions across the
/// six KT48 kernels: same op, form, operands, modifiers and literal — and
/// the parsed instruction re-encodes to the original words, proving the
/// canonical don't-care defaults. Any semantic disagreement is a test
/// failure.
#[test]
fn parser_matches_codec_on_kt48() {
    const INSTS: [(&str, usize); 6] = [
        ("attention_fp8_e4m3_fa2_gqa_gfx1201", 1297),
        ("attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201", 70),
        ("attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201", 253),
        ("attention_fp8_e4m3_fa2_gqa_partial_gfx1201", 1334),
        ("attention_fp8_e4m3_fa2_gqa_merge_gfx1201", 130),
        ("attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_gfx1201", 1696),
    ];
    let kernels = decode_all();
    let texts = texts_all();
    assert_eq!(kernels.len(), INSTS.len(), "six-kernel census");
    let mut total = 0usize;
    let mut failures = Vec::new();
    for ((name, va, size, decoded), (want_name, want_insts)) in
        kernels.iter().zip(INSTS.iter())
    {
        assert_eq!(name, want_name, "kernel order");
        assert_eq!(decoded.len(), *want_insts, "{name} census");
        let range = *va..*va + *size as u32;
        let ktexts: Vec<&support::ObjdumpLine> = texts
            .iter()
            .filter(|l| range.contains(&l.addr))
            .collect();
        assert_eq!(ktexts.len(), decoded.len(), "{name} fixture lines");
        for ((addr, words, inst), line) in decoded.iter().zip(ktexts.iter()) {
            let (taddr, text) = (line.addr, &line.text);
            assert_eq!(addr, &taddr, "{name} boundary");
            // Sanity: the test parses the canonical printing, which T3 proves
            // equals the pinned text.
            let printed =
                canonical(inst, Arch::Gfx1201).expect("canonical prints");
            if &printed != text {
                failures.push(format!("{name} {addr:#x}: T3 drift for {text}"));
                continue;
            }
            let parsed = match parse_line(text, Arch::Gfx1201) {
                Ok(inst) => inst,
                Err(e) => {
                    failures.push(format!("{name} {addr:#x}: parse {text:?}: {e}"));
                    continue;
                }
            };
            // Modifiers compare with textually invisible bits cleared (the
            // same lossiness the T9 harness defines: e.g. WMMA op_sel_hi).
            if parsed.op != inst.op
                || parsed.form != inst.form
                || parsed.operands != inst.operands
                || support::masked_mods(&parsed) != support::masked_mods(inst)
                || parsed.literal != inst.literal
            {
                failures.push(format!(
                    "{name} {addr:#x}: semantic drift on {text}\n  codec: {:?} {:?} {:?} {:?}\n  parse: {:?} {:?} {:?} {:?}",
                    inst.op,
                    inst.form,
                    inst.operands,
                    inst.mods,
                    parsed.op,
                    parsed.form,
                    parsed.operands,
                    parsed.mods
                ));
                continue;
            }
            // Re-encoding matches modulo the same unspellable bits.
            let mask = support::lossy_mask(inst);
            match gfx12::encode(&parsed) {
                Ok(enc)
                    if enc.len() == words.len()
                        && enc.iter().zip(words.iter()).enumerate().all(
                            |(i, (a, b))| {
                                (a ^ b) & !mask.get(i).copied().unwrap_or(0) == 0
                            },
                        ) => {}
                Ok(enc) => failures.push(format!(
                    "{name} {addr:#x}: re-encode drift on {text}: {enc:08X?} != {words:08X?}"
                )),
                Err(e) => failures.push(format!(
                    "{name} {addr:#x}: re-encode {text:?}: {e}"
                )),
            }
        }
        total += decoded.len();
    }
    assert_eq!(total, 4780, "all six kernels census");
    assert!(
        failures.is_empty(),
        "{} parser mismatches:\n{}",
        failures.len(),
        failures.into_iter().take(10).collect::<Vec<_>>().join("\n")
    );
}

/// `.s` structure: labels map to instruction ordinals, directives pass
/// through, every instruction line parses.
#[test]
fn source_parser_handles_labels_and_directives() {
    let src = "\
.text
.globl my_kernel
my_kernel:
    s_nop 0
.LBB0:
    s_branch 3
    // a comment
    s_endpgm
";
    let file = parse_source(src).expect("source parses");
    assert_eq!(
        file.labels,
        vec![
            ("my_kernel".to_owned(), 0),
            (".LBB0".to_owned(), 1),
        ]
    );
    assert_eq!(file.inst_len(), 3);
    let ids = label_block_ids(&file);
    assert_eq!(ids.len(), 2);
    // Every instruction line parses to a typed instruction.
    for text in file.insts() {
        parse_line(text, Arch::Gfx1201)
            .unwrap_or_else(|e| panic!("parse {text:?}: {e}"));
    }
    // Comment and directive lines are classified, not parsed as code.
    assert!(
        file.lines.iter().any(|l| matches!(
            l.kind,
            SourceKind::Directive(_)
        )),
        "directives retained"
    );
    // Duplicate labels fail closed.
    assert!(parse_source("a:\n    s_nop 0\na:\n    s_nop 0\n").is_err());
    // A KT48-sized body parses end to end through the source path.
    let body = kt48_texts()
        .into_iter()
        .map(|(_, _, t)| t)
        .collect::<Vec<_>>()
        .join("\n");
    let file =
        parse_source(&format!(".text\nmy_kernel:\n{body}\n")).expect("body");
    assert_eq!(file.inst_len(), 1696);
    assert_eq!(file.labels, vec![("my_kernel".to_owned(), 0)]);
}

/// Garbage is rejected, never normalised or panicked on.
#[test]
fn parse_rejects_garbage() {
    for bad in [
        "",
        "not_an_instruction",
        "s_nop",
        "s_nop 0, 1",
        "v_fma_f32 v177, -v175, v176",
        "v_fma_f32 v177, -v175, v176, 1.0, v0",
        "s_branch x",
        "s_delay_alu instid0(BOGUS)",
        "s_wait_alu depctr_nope(0)",
        "s_wait_alu depctr_sa_sdst(2)",
        "v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, -1, v3",
        "s_clause 0x100",
        "v_cndmask_b32_e32 v15, -1, v10",
        "s_load_b32 s3, s[0:1]",
    ] {
        assert!(
            parse_line(bad, Arch::Gfx1201).is_err(),
            "must reject {bad:?}"
        );
    }
    assert!(parse_line("s_nop 0", Arch::Gfx1201).is_ok());
}

/// C7 lift_text regressions on hipcc's own `.s`: quoted `//` inside
/// `.ident`, the `.amdgpu_metadata` YAML region, and the hidden-EXEC
/// cmpx shape (two visible operands).
#[test]
fn source_parser_handles_hipcc_s_constructs() {
    let src = "\
\t.ident\t\"AMD clang version 20.1.0 (https://github.com/rocm/llvm-project.git)\"
\t.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
# a yaml comment
  - .args:
...
\t.end_amdgpu_metadata
my_kernel:
\ts_nop 0
";
    let file = parse_source(src).expect("hipcc constructs parse");
    let directives: Vec<&str> = file
        .lines
        .iter()
        .filter_map(|l| match &l.kind {
            SourceKind::Directive(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        directives.iter().any(|d| d.starts_with(".ident")
            && d.contains("https://github.com/rocm/llvm-project.git")),
        "quoted // must not start a comment: {directives:?}"
    );
    for marker in [
        ".amdgpu_metadata",
        "amdhsa.version: [ 1, 2 ]",
        "# a yaml comment",
        "- .args:",
        "...",
        ".end_amdgpu_metadata",
    ] {
        assert!(
            directives.iter().any(|d| d.trim() == marker),
            "metadata region kept verbatim, missing {marker:?}: {directives:?}"
        );
    }
    assert_eq!(file.labels, vec![("my_kernel".to_owned(), 0)]);
    assert_eq!(file.inst_len(), 1);
}

/// `v_cmpx*` VOP3 canonical text has two visible operands; the hidden
/// exec VDST is fixed 0x7e in the encoding and absent from the type.
#[test]
fn cmpx_parses_two_visible_operands() {
    let inst = parse_line("v_cmpx_gt_i32_e64 s7, v4", Arch::Gfx1201)
        .expect("cmpx parses");
    assert_eq!(inst.operands.len(), 2);
    let enc = gfx12::encode(&inst).expect("cmpx encodes");
    assert_eq!(enc.len(), 2);
    let (back, n) = gfx12::decode(&enc).expect("cmpx re-decodes");
    assert_eq!(n, 2);
    assert_eq!(back.operands, inst.operands);
    assert_eq!(back.mods, inst.mods);
}

/// C7 lift_text census: hipcc's own `.s` omits redundant `op_sel`, uses
/// negative VMEM offsets, and relies on assembler defaults. parse_line
/// types exactly what llvm-mc encodes: op_sel implied by `.h`/`.l`,
/// assembler defaults for unspelled bits.
#[test]
fn hipcc_spellings_match_codec() {
    // op_sel implied by halves (redundant suffix omitted by hipcc).
    for (text, op_sel) in [
        ("v_mov_b16_e64 v139.l, v5.h", 1),
        ("v_cvt_f32_f16_e64 v12, v198.h", 1),
        ("v_cvt_pk_fp8_f32 v141.h, v140, v139", 8),
    ] {
        let inst =
            parse_line(text, Arch::Gfx1201).expect("hipcc spelling parses");
        assert_eq!(inst.mods.op_sel, op_sel, "{text}");
        // Re-prints with the redundant suffix objdump shows.
        let canon = canonical(&inst, Arch::Gfx1201).expect("prints");
        assert!(
            canon.contains("op_sel:["),
            "canonical restores suffix: {canon}"
        );
        gfx12::encode(&inst).expect("encodes");
    }
    // Negative VMEM offsets.
    let inst = parse_line(
        "global_load_b64 v[173:174], v[2:3], off offset:-48",
        Arch::Gfx1201,
    )
    .expect("negative vmem offset parses");
    assert!(matches!(
        inst.operands.last(),
        Some(peacemaker_ir::operand::Operand::Imm(
            peacemaker_ir::operand::ImmField::VmemOffset(-48)
        ))
    ));
    gfx12::encode(&inst).expect("encodes");
    // Omitted WMMA op_sel_hi takes the assembler default.
    let inst = parse_line(
        "v_wmma_f32_16x16x16_fp8_fp8 v[129:136], v[7:8], v[141:142], v[129:136]",
        Arch::Gfx1201,
    )
    .expect("wmma parses");
    assert_eq!(inst.mods.op_sel_hi, 7);
    gfx12::encode(&inst).expect("encodes");
}

/// `(text, gfx1100 words, gfx1151 words, gfx1201 words)` pinned from
/// `/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc -triple=amdgcn -mcpu=<arch>
/// -show-encoding`. An empty slice means the pinned assembler rejects the
/// spelling for that arch, so the parser/codec pair must refuse it too.
type Pinned = (&'static str, &'static [u32], &'static [u32], &'static [u32]);

const GFX11_ARCHES: [Arch; 2] = [Arch::Gfx1100, Arch::Gfx1151];

/// Encode through the architecture's own codec.
fn encode_for_arch(arch: Arch, inst: &Inst) -> Result<Vec<u32>, String> {
    match arch {
        Arch::Gfx1201 => gfx12::encode(inst),
        _ => gfx11::encode(arch, inst),
    }
    .map(|words| words.to_vec())
    .map_err(|e| e.to_string())
}

fn parse_encode(text: &str, arch: Arch) -> Result<Vec<u32>, String> {
    let inst = parse_line(text, arch).map_err(|e| e.to_string())?;
    encode_for_arch(arch, &inst)
}

fn pinned_words(case: &Pinned, arch: Arch) -> &'static [u32] {
    match arch {
        Arch::Gfx1100 => case.1,
        Arch::Gfx1151 => case.2,
        Arch::Gfx1201 => case.3,
        other => panic!("no pinned words for {other:?}"),
    }
}

/// Every pinned row on each arch it assembles for must parse and re-encode
/// to llvm-mc's exact words; rows the assembler rejects must be refused.
fn check_pinned(table: &[Pinned], arches: &[Arch]) {
    for case in table {
        for &arch in arches {
            let want = pinned_words(case, arch);
            let got = parse_encode(case.0, arch);
            if want.is_empty() {
                assert!(
                    got.is_err(),
                    "{arch:?}: llvm-mc rejects {:?}, parser+codec produced {got:08x?}",
                    case.0
                );
            } else {
                assert_eq!(
                    got.as_deref(),
                    Ok(want),
                    "{arch:?}: {:?} vs llvm-mc words {want:08x?}",
                    case.0
                );
            }
        }
    }
}

const PINNED_WAITCNT: &[Pinned] = &[
    ("s_waitcnt vmcnt(0) lgkmcnt(0)", &[0xbf890007], &[0xbf890007], &[]),
    ("s_waitcnt vmcnt(1) expcnt(2) lgkmcnt(3)", &[0xbf890432], &[0xbf890432], &[]),
    ("s_waitcnt expcnt(0)", &[0xbf89fff0], &[0xbf89fff0], &[]),
    ("s_waitcnt vmcnt(5)", &[0xbf8917f7], &[0xbf8917f7], &[]),
    ("s_waitcnt lgkmcnt(0)", &[0xbf89fc07], &[0xbf89fc07], &[]),
    ("s_waitcnt 0", &[0xbf890000], &[0xbf890000], &[]),
];

/// gfx11-only true16 rows: the gfx12 table has no `v_mul_f16`, so the gfx1201
/// column is unused (never checked).
const PINNED_TRUE16_GFX11: &[Pinned] = &[
    ("v_mul_f16_e32 v1.h, v2.l, v3.h", &[0x6b030702], &[0x6b030702], &[]),
    ("v_mul_f16_e32 v1.l, v2.h, v3.l", &[0x6a020782], &[0x6a020782], &[]),
    ("v_mul_f16_e64 v1.h, v2.h, v3.l", &[0xd5354801, 0x02020702], &[0xd5354801, 0x02020702], &[]),
];

const PINNED_TRUE16: &[Pinned] = &[
    ("v_fma_f16 v1.l, v2.h, v3.l, v4.h", &[0xd6482801, 0x04120702], &[0xd6482801, 0x04120702], &[0xd6482801, 0x04120702]),
    ("v_mov_b16_e32 v1.h, v2.l", &[0x7f023902], &[0x7f023902], &[0x7f023902]),
    ("v_mov_b16_e32 v5.h, 0", &[0x7f0a3880], &[0x7f0a3880], &[0x7f0a3880]),
    ("v_cvt_f16_f32_e32 v68.h, v61", &[0x7f88153d], &[0x7f88153d], &[0x7f88153d]),
    ("v_cvt_f32_f16_e32 v1, v2.h", &[0x7e021782], &[0x7e021782], &[0x7e021782]),
    ("v_cvt_f32_f16_e64 v12, v198.h", &[0xd58b080c, 0x020101c6], &[0xd58b080c, 0x020101c6], &[0xd58b080c, 0x020101c6]),
];

const PINNED_CACHE_BITS: &[Pinned] = &[
    ("buffer_load_b32 v1, v2, s[4:7], s8 offen offset:16 glc slc dlc", &[0xe0507010, 0x08410102], &[0xe0507010, 0x08410102], &[]),
    ("buffer_load_b32 v1, off, s[4:7], s8 offset:4 glc", &[0xe0504004, 0x08010100], &[0xe0504004, 0x08010100], &[]),
    ("buffer_store_b32 v1, v2, s[4:7], s8 offen slc dlc", &[0xe0683000, 0x08410102], &[0xe0683000, 0x08410102], &[]),
    ("global_load_b32 v1, v[2:3], off glc slc dlc", &[0xdc52e000, 0x017c0002], &[0xdc52e000, 0x017c0002], &[]),
    ("global_load_b32 v1, v[2:3], off offset:-48 glc", &[0xdc525fd0, 0x017c0002], &[0xdc525fd0, 0x017c0002], &[]),
    ("global_load_b32 v1, v2, s[4:5] slc", &[0xdc528000, 0x01040002], &[0xdc528000, 0x01040002], &[]),
    ("global_store_b32 v[2:3], v1, off dlc", &[0xdc6a2000, 0x007c0102], &[0xdc6a2000, 0x007c0102], &[]),
    ("s_load_b32 s1, s[2:3], 0x0 glc", &[0xf4004041, 0xf8000000], &[0xf4004041, 0xf8000000], &[]),
    ("s_load_b32 s1, s[2:3], 0x10 glc dlc", &[0xf4006041, 0xf8000010], &[0xf4006041, 0xf8000010], &[]),
];

const PINNED_VOPD: &[Pinned] = &[
    ("v_dual_mov_b32 v1, v2 :: v_dual_add_f32 v4, v5, v6", &[0xca080102, 0x01040d05], &[0xca080102, 0x01040d05], &[0xca080102, 0x01040d05]),
    ("v_dual_add_f32 v104, v104, v34 :: v_dual_add_f32 v105, v105, v35", &[0xc9084568, 0x68684769], &[0xc9084568, 0x68684769], &[0xc9084568, 0x68684769]),
    ("v_dual_fmamk_f32 v1, v2, 0x3f800000, v3 :: v_dual_mul_f32 v4, v5, v6", &[0xc8860702, 0x01040d05, 0x3f800000], &[0xc8860702, 0x01040d05, 0x3f800000], &[0xc8860702, 0x01040d05, 0x3f800000]),
    ("v_dual_fmaak_f32 v0, v1, v2, 0x40490fdb :: v_dual_mov_b32 v3, v4", &[0xc8500501, 0x00020104, 0x40490fdb], &[0xc8500501, 0x00020104, 0x40490fdb], &[0xc8500501, 0x00020104, 0x40490fdb]),
    ("v_dual_mul_f32 v0, v1, v2 :: v_dual_fmamk_f32 v3, v4, 0x40490fdb, v5", &[0xc8c40501, 0x00020b04, 0x40490fdb], &[0xc8c40501, 0x00020b04, 0x40490fdb], &[0xc8c40501, 0x00020b04, 0x40490fdb]),
];

const PINNED_BRANCH: &[Pinned] = &[
    ("s_branch 5", &[0xbfa00005], &[0xbfa00005], &[0xbfa00005]),
    ("s_branch -3", &[0xbfa0fffd], &[0xbfa0fffd], &[0xbfa0fffd]),
    ("s_cbranch_scc1 -2", &[0xbfa2fffe], &[0xbfa2fffe], &[0xbfa2fffe]),
    ("s_cbranch_execz 7", &[0xbfa50007], &[0xbfa50007], &[0xbfa50007]),
    ("s_cbranch_scc0 12", &[0xbfa1000c], &[0xbfa1000c], &[0xbfa1000c]),
    ("s_cbranch_vccz -7", &[0xbfa3fff9], &[0xbfa3fff9], &[0xbfa3fff9]),
    ("s_cbranch_execnz 0", &[0xbfa60000], &[0xbfa60000], &[0xbfa60000]),
    ("s_branch -32768", &[0xbfa08000], &[0xbfa08000], &[0xbfa08000]),
    ("s_branch 32767", &[0xbfa07fff], &[0xbfa07fff], &[0xbfa07fff]),
    ("s_branch 32768", &[0xbfa08000], &[0xbfa08000], &[0xbfa08000]),
    ("s_branch 65535", &[0xbfa0ffff], &[0xbfa0ffff], &[0xbfa0ffff]),
];

const PINNED_EDGES: &[Pinned] = &[
    ("v_fmamk_f32 v1, v2, 0x3f800000, v3", &[0x58020702, 0x3f800000], &[0x58020702, 0x3f800000], &[0x58020702, 0x3f800000]),
    ("v_fmaak_f32 v1, v2, v3, 0x3f800000", &[0x5a020702, 0x3f800000], &[0x5a020702, 0x3f800000], &[0x5a020702, 0x3f800000]),
    ("v_pk_fma_f16 v1, v2, v3, v4 op_sel:[0,1,0]", &[0xcc0e5001, 0x1c120702], &[0xcc0e5001, 0x1c120702], &[0xcc0e5001, 0x1c120702]),
    ("v_pk_fma_f16 v1, v2, v3, v4 op_sel_hi:[1,0,1]", &[0xcc0e4001, 0x0c120702], &[0xcc0e4001, 0x0c120702], &[0xcc0e4001, 0x0c120702]),
    ("v_pk_fma_f16 v1, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[0,1,0] neg_lo:[1,0,0] neg_hi:[0,0,1]", &[0xcc0e2c01, 0x34120702], &[0xcc0e2c01, 0x34120702], &[0xcc0e2c01, 0x34120702]),
    ("ds_swizzle_b32 v1, v2 offset:swizzle(BROADCAST,16,8)", &[0xd8d40110, 0x01000002], &[0xd8d40110, 0x01000002], &[0xd8d40110, 0x01000002]),
    ("ds_swizzle_b32 v1, v2 offset:swizzle(BROADCAST,2,1)", &[0xd8d4003e, 0x01000002], &[0xd8d4003e, 0x01000002], &[0xd8d4003e, 0x01000002]),
    ("ds_swizzle_b32 v1, v2 offset:swizzle(BROADCAST,32,31)", &[0xd8d403e0, 0x01000002], &[0xd8d403e0, 0x01000002], &[0xd8d403e0, 0x01000002]),
    ("buffer_load_b32 v1, v2, s[4:7], s8 offen offset:16", &[0xe0500010, 0x08410102], &[0xe0500010, 0x08410102], &[0xc4050008, 0x40800801, 0x00001002]),
    ("buffer_load_b32 v1, off, s[4:7], s8 offset:16", &[0xe0500010, 0x08010100], &[0xe0500010, 0x08010100], &[0xc4050008, 0x00800801, 0x00001000]),
];

/// gfx11 `s_waitcnt` combined counters (vmcnt/expcnt/lgkmcnt in one
/// SOPP word, unspelled counters at their maxima) and the bare numeric
/// spelling. gfx1201 has no combined counter instruction: it is refused.
#[test]
fn gfx11_waitcnt_combined_counters_match_llvm_mc() {
    check_pinned(PINNED_WAITCNT, &GFX11_ARCHES);
    for case in PINNED_WAITCNT {
        assert!(
            parse_line(case.0, Arch::Gfx1201).is_err(),
            "gfx1201 must refuse {:?}",
            case.0
        );
    }
    for bad in [
        "s_waitcnt vmcnt(64)",
        "s_waitcnt expcnt(8)",
        "s_waitcnt lgkmcnt(64)",
        "s_waitcnt bogcnt(0)",
        "s_waitcnt vmcnt",
    ] {
        for arch in GFX11_ARCHES {
            assert!(parse_line(bad, arch).is_err(), "{arch:?} must reject {bad:?}");
        }
    }
}

/// True16 `.l`/`.h` halves select the VOP1/VOP2 half bits and the VOP3
/// `op_sel` bits identically on every architecture whose table has the row
/// (`v_mul_f16` is gfx11-only).
#[test]
fn true16_halves_match_llvm_mc() {
    check_pinned(PINNED_TRUE16, &[Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]);
    check_pinned(PINNED_TRUE16_GFX11, &GFX11_ARCHES);
}

/// gfx11 `glc`/`slc`/`dlc` cache bits on buffer, global and SMEM; gfx1201
/// replaced them with `th:`/`scope:`, so the legacy spellings are refused.
#[test]
fn gfx11_cache_bits_match_llvm_mc_and_gfx1201_refuses() {
    check_pinned(PINNED_CACHE_BITS, &[Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]);
}

/// VOPD pairs, including the embedded-literal X/Y forms, encode to
/// llvm-mc's words on all three architectures.
#[test]
fn vopd_pairs_match_llvm_mc_on_all_arches() {
    check_pinned(PINNED_VOPD, &[Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]);
}

/// Numeric branch targets: negative, positive and the wrapped
/// 32768..=65535 spelling llvm-mc accepts; below -32768 and above 65535 are
/// refused on every architecture.
#[test]
fn numeric_branches_match_llvm_mc_and_reject_out_of_range() {
    let arches = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];
    check_pinned(PINNED_BRANCH, &arches);
    for arch in arches {
        for bad in ["s_branch -32769", "s_branch 65536", "s_cbranch_scc1 -32769", "s_cbranch_vccz 65536"] {
            assert!(parse_line(bad, arch).is_err(), "{arch:?} must reject {bad:?}");
        }
    }
}

/// Edge spellings pinned on all three architectures: FMAMK/FMAAK embedded
/// 32-bit literal, packed-FMA `op_sel`/`op_sel_hi`/`neg_lo`/`neg_hi` source
/// bit mapping, `ds_swizzle_b32` BROADCAST encoding and buffer `offen` /
/// `off` + `offset:` (gfx11 MUBUF dwords differ from gfx12 VBUFFER).
#[test]
fn embedded_literal_packed_swizzle_and_buffer_edges_match_llvm_mc() {
    check_pinned(PINNED_EDGES, &[Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]);
}

/// gfx1201 buffer `offen offset th scope` words from llvm-mc. llvm-mc only
/// accepts the `offen offset:N th: scope:` spelling; the parser also takes the
/// modifiers in any order and must still emit the canonical operand order.
#[test]
fn gfx1201_buffer_offen_offset_th_scope_order() {
    const WORDS: &[u32] = &[0xc4050008, 0x40940801, 0x00001002];
    for text in [
        "buffer_load_b32 v1, v2, s[4:7], s8 offen offset:16 th:TH_LOAD_NT scope:SCOPE_SE",
        "buffer_load_b32 v1, v2, s[4:7], s8 offset:16 offen th:TH_LOAD_NT scope:SCOPE_SE",
        "buffer_load_b32 v1, v2, s[4:7], s8 scope:SCOPE_SE th:TH_LOAD_NT offset:16 offen",
        "buffer_load_b32 v1, v2, s[4:7], s8 th:TH_LOAD_NT offen scope:SCOPE_SE offset:16",
    ] {
        assert_eq!(parse_encode(text, Arch::Gfx1201).as_deref(), Ok(WORDS), "{text}");
    }
}

/// The `peacemaker profile` clock/id reads and its GLOBAL `th:` record
/// store: each target takes exactly the hwreg names and cache-policy syntax
/// llvm-mc accepts for it (gfx12 also takes the gfx11 `HW_ID` names).
const PINNED_PROFILE: &[Pinned] = &[
    ("s_getreg_b32 s17, hwreg(HW_REG_HW_ID1)", &[0xb891f817], &[0xb891f817], &[0xb891f817]),
    ("s_getreg_b32 s17, hwreg(HW_REG_WAVE_HW_ID1)", &[], &[], &[0xb891f817]),
    ("s_getreg_b32 s3, hwreg(HW_REG_SHADER_CYCLES, 0, 20)", &[0xb883981d], &[0xb883981d], &[]),
    ("s_getreg_b32 s3, hwreg(HW_REG_SHADER_CYCLES_HI)", &[], &[], &[0xb883f81e]),
    ("global_store_addtid_b32 v239, s[28:29] offset:-8 th:TH_STORE_NT", &[], &[], &[0xee0a401c, 0x77900000, 0xfffff800]),
];
#[test]
fn profile_hwreg_names_and_global_th_match_llvm_mc_per_target() {
    check_pinned(PINNED_PROFILE, &[Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201]);
}
