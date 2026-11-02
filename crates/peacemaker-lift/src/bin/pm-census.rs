// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Offline packaged-object census for gfx1100, gfx1151 and gfx1201.
//! Usage: pm-census <arch> <registry.tsv> <compiled/<arch>> <output-dir>
//!                  <llvm-objdump> <arch-encodings.tsv> [builder.hxaco ...]
//! Generate the TSV from the matching pinned MR-ISA XML with tools/pm-census-xml.py.
//! Matching full XML encoding identifiers and word widths avoids guessing literal forms.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use peacemaker_ir::codec::gfx12;
use peacemaker_ir::edit::analyze;
use peacemaker_ir::envelope::Bundle;
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel};
use peacemaker_ir::lds::AddrFact;
use peacemaker_ir::passes::{barriers, windows};
use peacemaker_lift::bundle::BundleCodec;
use peacemaker_lift::elf::EnvelopeCodec;
use peacemaker_lift::{emit, lift_object, LiftError, Options};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn arch_name(arch: Arch) -> &'static str {
    match arch {
        Arch::Gfx1100 => "gfx1100",
        Arch::Gfx1151 => "gfx1151",
        Arch::Gfx1201 => "gfx1201",
        _ => unreachable!("census accepts gfx1100, gfx1151 or gfx1201 only"),
    }
}
fn xml_sha(arch: Arch) -> &'static str {
    match arch {
        Arch::Gfx1100 => "6eee5f8737172adf08c0e7d5994ea916e9555b9c23ccb9ec80d9ee38678733b4",
        Arch::Gfx1151 => "c36b6d79b1e940d74107221c985f5a7fde248025da251d2c6ef756c4cd31391a",
        Arch::Gfx1201 => "f8a290c8471e26a1071b08b61a33e4d9efa46ec6cedcdb8e5ba57d7969b60692",
        _ => unreachable!(),
    }
}

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

#[derive(Clone, Debug)]
struct Line {
    addr: u64,
    words: Vec<u32>,
    text: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn device_elf(bytes: &[u8], arch: Arch) -> Result<&[u8]> {
    if !Bundle::is_bundle(bytes) {
        return Ok(bytes);
    }
    let (bundle, payloads) = Bundle::read(bytes)?;
    Ok(payloads[bundle.device_entry(arch)?])
}

fn disassemble(elf: &[u8], objdump: &Path, scratch: &Path, arch: Arch) -> Result<Vec<Line>> {
    fs::write(scratch, elf)?;
    let result = Command::new(objdump)
        .arg(format!("--mcpu={}", arch_name(arch)))
        .arg("-d")
        .arg(scratch)
        .output()?;
    fs::remove_file(scratch)?;
    if !result.status.success() {
        return Err(format!("llvm-objdump: {}", String::from_utf8_lossy(&result.stderr)).into());
    }
    let stdout = String::from_utf8(result.stdout)?;
    let mut lines = Vec::new();
    for line in stdout.lines() {
        let Some((text, comment)) = line.strip_prefix('\t').and_then(|l| l.split_once("//")) else {
            continue;
        };
        let mut tokens = comment.split_whitespace();
        let Some(addr) = tokens
            .next()
            .and_then(|s| s.strip_suffix(':'))
            .and_then(|s| u64::from_str_radix(s, 16).ok())
        else {
            continue;
        };
        let words = tokens
            .take_while(|s| s.len() == 8 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .map(|s| u32::from_str_radix(s, 16).expect("checked hex"))
            .collect::<Vec<_>>();
        if !words.is_empty() {
            lines.push(Line {
                addr,
                words,
                text: text.trim().to_owned(),
            });
        }
    }
    Ok(lines)
}

// Pinned per-target XML digests are checked by `xml_sha`.

struct XmlRow {
    encoding: String,
    opcode: String,
    bits: usize,
    mask: u128,
    ids: Vec<u128>,
}
type XmlTable = BTreeMap<String, Vec<XmlRow>>;

fn xml_table(path: &Path, arch: Arch) -> Result<XmlTable> {
    let contents = fs::read_to_string(path)?;
    let mut lines = contents.lines();
    if lines.next() != Some(format!("# xml_sha256={}", xml_sha(arch)).as_str()) {
        return Err(format!(
            "{} is not exported from pinned {} XML",
            path.display(),
            arch_name(arch)
        )
        .into());
    }
    if lines.next() != Some("encoding\topcode\tbits\tspelling\tid_mask\tids") {
        return Err("invalid MR-ISA XML encoding-table header".into());
    }
    let mut table: XmlTable = BTreeMap::new();
    for (line_no, line) in lines.enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(
                format!("XML table line {} has {} fields", line_no + 3, fields.len()).into(),
            );
        }
        let row = XmlRow {
            encoding: fields[0].to_owned(),
            opcode: fields[1].to_owned(),
            bits: fields[2].parse()?,
            mask: u128::from_str_radix(fields[4], 16)?,
            ids: fields[5]
                .split(',')
                .map(|id| u128::from_str_radix(id, 16))
                .collect::<std::result::Result<_, _>>()?,
        };
        table.entry(fields[3].to_owned()).or_default().push(row);
    }
    Ok(table)
}

fn xml_opcodes(spelling: &str, words: &[u32], table: &XmlTable) -> Vec<String> {
    let base = spelling
        .trim_end_matches("_e32")
        .trim_end_matches("_e64")
        .trim_end_matches("_dpp");
    let encoded = words
        .iter()
        .enumerate()
        .fold(0u128, |acc, (i, word)| acc | u128::from(*word) << (32 * i));
    let mut matches = table
        .get(base)
        .into_iter()
        .flatten()
        .filter(|row| {
            row.bits == 32 * words.len()
                && row.ids.contains(&(encoded & row.mask))
                && (!spelling.ends_with("_e32") || !row.encoding.contains("VOP3"))
                && (!spelling.ends_with("_e64") || row.encoding.contains("VOP3"))
                && if spelling.ends_with("_dpp") {
                    // gfx12 SRC0=0xfa denotes the DPP16 extension in this corpus.
                    row.encoding.contains("DPP16") && words[0] & 0xff == 0xfa
                } else {
                    !row.encoding.contains("DPP")
                }
        })
        .map(|row| format!("{}:{}", row.encoding, row.opcode))
        .collect::<Vec<_>>();
    matches.sort();
    matches.dedup();
    matches
}

fn rejection(error: &LiftError) -> Value {
    match error {
        LiftError::Rejected {
            kernel,
            offset,
            rule,
            reason,
        } => {
            json!({"kernel": kernel, "offset": format!("{offset:#x}"), "rule": format!("{rule:?}"), "reason": reason})
        }
    }
}

fn kernel_facts(kernel: &Kernel, program: &peacemaker_ir::inst::Program) -> Result<Value> {
    let analysis = analyze(program.clone(), &kernel.symbol)?;
    let facts = &analysis.facts;
    let resources = &facts.resources[0];
    let (vgpr_metadata, sgpr_metadata, vgpr_descriptor, lds_descriptor) = match &kernel.abi {
        Abi::Hsa {
            descriptor,
            metadata,
        } => (
            metadata.parsed.vgpr_count,
            metadata.parsed.sgpr_count,
            descriptor
                .compute_pgm_rsrc1
                .next_free_vgpr(descriptor.kernel_code_properties.wave32()),
            descriptor.group_segment_fixed_size,
        ),
        Abi::Raw { .. } => return Err("unexpected raw ABI in packaged object".into()),
    };
    let mut obligations = BTreeMap::<String, usize>::new();
    let mut details = Vec::new();
    for item in &analysis.obligations {
        let kind = format!("{:?}", item.kind);
        *obligations
            .entry(format!("{kind}/{}", item.rule_id))
            .or_default() += 1;
        let offsets = item
            .insts
            .iter()
            .filter_map(|id| kernel.body.insts.get(*id))
            .filter_map(|inst| inst.prov.pc)
            .map(|pc| format!("{pc:#x}"))
            .collect::<Vec<_>>();
        details.push(
            json!({"kind": kind, "rule": item.rule_id, "offsets": offsets, "reason": item.text}),
        );
    }
    let lds = &facts.lds;
    let lds_unknown = lds
        .accesses
        .iter()
        .filter(|(access, _)| matches!(access.addr, AddrFact::Unknown))
        .count();
    let lds_unbounded = lds
        .accesses
        .iter()
        .filter(|(access, _)| matches!(access.addr, AddrFact::Bounded { hi: u32::MAX, .. }))
        .count();
    let windows = windows::check_windows(&kernel.body)?;
    let pairings = barriers::analyze(&kernel.body, program.target.arch)?;
    let ambiguous = windows
        .delays
        .iter()
        .filter(|delay| delay.is_ambiguous())
        .count();
    Ok(json!({
        "symbol": kernel.symbol.0, "instructions": kernel.body.layout.len(),
        "obligations": obligations, "obligation_details": details,
        "hazards": analysis.obligations.iter().filter(|o| format!("{:?}", o.kind) == "Hazard").count(),
        "resources": {"vgpr_high_water": resources.max_vgpr, "sgpr_high_water": resources.max_sgpr,
            "vcc": resources.uses_vcc, "vgpr_descriptor": vgpr_descriptor,
            "vgpr_metadata": vgpr_metadata, "sgpr_metadata": sgpr_metadata},
        "lds": {"accesses": lds.accesses.len(), "unknown": lds_unknown, "unbounded": lds_unbounded,
            "max_known_end": lds.max_end, "fixed_descriptor_bytes": lds_descriptor,
            "dynamic_extent": resources.lds_dynamic_max},
        "barriers": {"paired_signals": pairings.pairs.len(), "matched_waits": pairings.pairs.iter().map(|p| p.waits.len()).sum::<usize>(),
            "obligations": pairings.obligations.len()},
        "windows": {"clauses": windows.clauses.len(), "delays": windows.delays.len(),
            "ambiguous_delays": ambiguous, "ambiguous_clauses": 0}
    }))
}

fn scan_module(
    name: &str,
    path: &Path,
    indexed: bool,
    objdump: &Path,
    scratch: &Path,
    xml: &XmlTable,
    unknowns: &mut BTreeMap<String, Value>,
    arch: Arch,
) -> Result<Value> {
    let input = fs::read(path)?;
    let sha = hex(&Sha256::digest(&input));
    let mut packaging_index = None;
    let indexed_symbols = if indexed {
        let index: Value = serde_json::from_slice(&fs::read(path.with_extension("index.json"))?)?;
        if index["module"] != name
            || index["arch"] != arch_name(arch)
            || index["object_sha256"] != sha
        {
            return Err(format!(
                "{}: packaging index does not match module, arch, or SHA",
                path.display()
            )
            .into());
        }
        packaging_index = Some(index.clone());
        Some(
            index["symbols"]
                .as_array()
                .ok_or("index symbols are not an array")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .ok_or("index symbol is not a string")
                        .map(str::to_owned)
                })
                .collect::<std::result::Result<BTreeSet<_>, _>>()?,
        )
    } else {
        None
    };
    let elf = device_elf(&input, arch)?;
    let (_, images) = peacemaker_ir::envelope::Envelope::read(elf)?;
    let actual = images
        .iter()
        .map(|image| image.name.clone())
        .collect::<BTreeSet<_>>();
    if let Some(expected) = &indexed_symbols {
        if !expected.is_subset(&actual) {
            return Err(format!(
                "{name}: packaged index expects {expected:?}, ELF holds {actual:?}"
            )
            .into());
        }
    }
    let unindexed_elf_symbols = actual
        .difference(indexed_symbols.as_ref().unwrap_or(&actual))
        .cloned()
        .collect::<Vec<_>>();
    let lines = disassemble(elf, objdump, scratch, arch)?;
    let mut kernel_lines = BTreeMap::new();
    let mut spellings = BTreeMap::<String, usize>::new();
    let mut missing_lines = Vec::new();
    let mut checked_instructions = 0;
    let mut opcode_disagreements = Vec::new();
    for image in &images {
        let these = lines
            .iter()
            .filter(|line| {
                (image.entry_va..image.entry_va + image.code.len() as u64).contains(&line.addr)
            })
            .collect::<Vec<_>>();
        let mut next = image.entry_va;
        for line in &these {
            if line.addr != next {
                return Err(format!(
                    "{name}/{}: objdump gap/overlap at {next:#x}, next line at {:#x}",
                    image.name, line.addr
                )
                .into());
            }
            next += line.words.len() as u64 * 4;
        }
        if next != image.entry_va + image.code.len() as u64 {
            return Err(format!(
                "{name}/{}: objdump ends at {next:#x}, kernel ends at {:#x}",
                image.name,
                image.entry_va + image.code.len() as u64
            )
            .into());
        }
        for line in &these {
            let spelling = line
                .text
                .split_whitespace()
                .next()
                .unwrap_or("<no-spelling>")
                .to_owned();
            *spellings.entry(spelling.clone()).or_default() += 1;
            let decoded = gfx12::decode_for(arch, &line.words);
            if let Ok((inst, n)) = &decoded {
                if *n == line.words.len() {
                    checked_instructions += 1;
                    let decoded_name =
                        peacemaker_ir::isa::lookup(arch, inst.op, inst.form).map(|row| row.name);
                    // gfx11 objdump can elide the optional e64 spelling on a VOP3,
                    // but the form and full instruction words must still agree.
                    let rendered_matches = decoded_name == Some(spelling.as_str())
                        || (arch != Arch::Gfx1201
                            && inst.form == peacemaker_ir::inst::Form::Vop3
                            && decoded_name.and_then(|name| name.strip_suffix("_e64"))
                                == Some(spelling.as_str()));
                    if !rendered_matches {
                        opcode_disagreements.push(format!(
                            "{} at {:#x}: codec {} vs llvm {}",
                            image.name,
                            line.addr,
                            decoded_name.unwrap_or("<unknown>"),
                            spelling
                        ));
                    }
                }
            }
            if decoded
                .as_ref()
                .map_or(true, |(_, n)| *n != line.words.len())
            {
                let xml_rows = xml_opcodes(&spelling, &line.words, xml);
                let key = format!("{}@{}", spelling, xml_rows.join(","));
                let entry = unknowns.entry(key).or_insert_with(|| json!({
                    "spelling": spelling, "form": xml_rows.iter().map(|s| s.split(':').next().unwrap()).collect::<Vec<_>>(),
                    "xml_opcode": xml_rows, "table_has_spelling": peacemaker_ir::isa::table(arch).iter().any(|row| row.name == spelling),
                    "occurrences": 0, "modules": BTreeMap::<String, usize>::new(),
                    "sample": {"module": name, "kernel": image.name, "offset": format!("{:#x}", line.addr), "words": line.words.iter().map(|w| format!("0x{w:08x}")).collect::<Vec<_>>(), "text": line.text},
                    "decode_reason": decoded.err().map(|e| e.to_string())
                }));
                entry["occurrences"] = json!(entry["occurrences"].as_u64().unwrap() + 1);
                let current = entry["modules"][name].as_u64().unwrap_or(0);
                entry["modules"][name] = json!(current + 1);
            }
        }
        if these.is_empty() {
            missing_lines.push(image.name.clone());
        }
        kernel_lines.insert(image.name.clone(), these);
    }
    if !missing_lines.is_empty() {
        return Err(format!("{name}: objdump returned no lines for {missing_lines:?}").into());
    }
    let frontend = if indexed {
        Frontend::Hipcc
    } else {
        Frontend::Builder
    };
    let mut kernels = Vec::new();
    let mut rejected = Vec::new();
    let mut word_boundary_mismatches = Vec::new();
    let mut text_differences = Vec::new();
    let mut roundtrip = false;
    match lift_object(&input, Options { frontend }) {
        Err(error) => rejected.push(rejection(&error)),
        Ok(lifted) => {
            roundtrip = emit::module(&lifted.program)? == input;
            if !roundtrip {
                return Err(format!("{name}: emit::module differs after successful lift").into());
            }
            for kernel in &lifted.program.kernels {
                let these = &kernel_lines[&kernel.symbol.0];
                let emitted = emit::insts(kernel, arch)?;
                let texts = emit::text(kernel, arch)?;
                if these.len() != emitted.len() {
                    word_boundary_mismatches.push(format!(
                        "{}: {} objdump vs {} codec instructions",
                        kernel.symbol.0,
                        these.len(),
                        emitted.len()
                    ));
                }
                for (inst, (text, line)) in emitted.iter().zip(texts.iter().zip(these)) {
                    let words = gfx12::encode_for(arch, inst)?;
                    if inst.prov.pc.map(u64::from) != Some(line.addr)
                        || words.as_slice() != line.words
                    {
                        word_boundary_mismatches.push(format!(
                            "{} at {:#x}: codec pc {:?} words {:?}, llvm words {:?}",
                            kernel.symbol.0, line.addr, inst.prov.pc, words, line.words
                        ));
                    }
                    if text != &line.text {
                        text_differences.push(format!(
                            "{} at {:#x}: codec {:?}, llvm {:?}",
                            kernel.symbol.0, line.addr, text, line.text
                        ));
                    }
                }
                match kernel_facts(kernel, &lifted.program) {
                    Ok(facts) => kernels.push(facts),
                    Err(error) => rejected.push(json!({"kernel": kernel.symbol.0, "offset": "unknown", "rule": "Analysis", "reason": error.to_string()})),
                }
            }
        }
    }
    let semantic_agreement =
        if !opcode_disagreements.is_empty() || !word_boundary_mismatches.is_empty() {
            "disagrees"
        } else if !roundtrip || !text_differences.is_empty() {
            "unproven"
        } else {
            "agrees"
        };
    Ok(
        json!({"module": name, "kind": if indexed {"indexed"} else {"builder"},
        "path": path.display().to_string(), "sha256": sha,
        "packaging_index": packaging_index,
        "symbols": images.iter().map(|image| image.name.clone()).collect::<Vec<_>>(),
        "indexed_symbols": indexed_symbols, "unindexed_elf_symbols": unindexed_elf_symbols,
        "objdump_instructions": kernel_lines.values().map(Vec::len).sum::<usize>(),
        "codec_checked_instructions": checked_instructions, "codec_opcode_disagreements": opcode_disagreements,
        "spellings": spellings, "lifted_roundtrip": roundtrip, "kernels": kernels,
        "rejections": rejected, "objdump_word_boundary_mismatches": word_boundary_mismatches,
        "objdump_text_differences": text_differences, "objdump_semantic_agreement": semantic_agreement}),
    )
}

fn report(
    modules: &[Value],
    unknowns: &BTreeMap<String, Value>,
    provenance: Value,
    out: &Path,
    arch: Arch,
) -> Result<()> {
    let mut rejected = BTreeMap::<String, usize>::new();
    let mut rejected_rules = BTreeMap::<String, usize>::new();
    let mut obligations = BTreeMap::<String, usize>::new();
    let mut clean = 0;
    let mut analyzed = 0;
    let mut text_differences = 0;
    let mut word_disagreements = 0;
    let mut opcode_disagreements = 0;
    for module in modules {
        if module["lifted_roundtrip"] == true {
            clean += 1;
        }
        analyzed += module["kernels"].as_array().unwrap().len();
        text_differences += module["objdump_text_differences"].as_array().unwrap().len();
        word_disagreements += module["objdump_word_boundary_mismatches"]
            .as_array()
            .unwrap()
            .len();
        opcode_disagreements += module["codec_opcode_disagreements"]
            .as_array()
            .unwrap()
            .len();
        for r in module["rejections"].as_array().unwrap() {
            *rejected_rules
                .entry(r["rule"].as_str().unwrap().to_owned())
                .or_default() += 1;
            *rejected
                .entry(format!(
                    "{}: {}",
                    r["rule"].as_str().unwrap(),
                    r["reason"].as_str().unwrap()
                ))
                .or_default() += 1;
        }
        for kernel in module["kernels"].as_array().unwrap() {
            for (kind, value) in kernel["obligations"].as_object().unwrap() {
                *obligations.entry(kind.clone()).or_default() += value.as_u64().unwrap() as usize;
            }
        }
    }
    let indexed = modules.iter().filter(|m| m["kind"] == "indexed").count();
    let totals = json!({"objects": modules.len(), "indexed": indexed, "builder": modules.len() - indexed,
        "lifted_roundtrip": clean, "rejected_modules": modules.len() - clean,
        "analyzed_kernels": analyzed, "rejections_by_rule": rejected_rules, "rejections_by_reason": rejected, "obligations_by_kind_rule": obligations,
        "objdump_text_differences": text_differences, "objdump_word_boundary_mismatches": word_disagreements,
        "codec_opcode_disagreements": opcode_disagreements,
        "unknown_encoding_spellings": unknowns.values().map(|v| v["spelling"].as_str().unwrap()).collect::<BTreeSet<_>>().len(),
        "unknown_encoding_rows": unknowns.len(),
        "unknown_encoding_occurrences": unknowns.values().map(|v| v["occurrences"].as_u64().unwrap()).sum::<u64>()});
    let record = json!({"schema": 1, "arch": arch_name(arch), "provenance": provenance, "totals": totals, "modules": modules, "unknown_encodings": unknowns.values().collect::<Vec<_>>()});
    fs::write(out.join("census.json"), serde_json::to_vec_pretty(&record)?)?;
    let mut md = format!("# {} compiled-kernel census\n\nOffline compiler/encoder inspection only; neither strict certification nor GPU/admission evidence. `census.json` holds every symbol, spelling, offset, obligation, and diagnostic.\n\n", arch_name(arch));
    md.push_str(&format!("## Inputs and method\n\n{indexed} indexed code objects and {} additional bundles were checked. The TSV identifies the canonical module, source and compile recipe; a seventh field, when present, identifies the distinct artifact key of a JIT variant. The matching index's canonical module, arch, symbol set and object SHA-256 were verified. The emitter re-encodes typed fields. Rejected modules have no analyzed kernel or round-trip proof; independent LLVM disassembly still inventories their entire kernel ranges. Exact per-kernel results and input/tool hashes are in JSON. XML encoding names/opcodes come from the matching pinned MR-ISA XML SHA-256 `{}`; sample instruction words are matched to XML encoding bit-width and identifier fields, not inferred from mnemonic alone.\n\n", modules.len()-indexed, xml_sha(arch)));
    md.push_str(&format!("## Totals\n\nIndexed: {indexed}; builder: {}; lifted with byte-identical emit: {clean}/{}; rejected modules: {}; analyzed kernels: {analyzed}; objdump opcode disagreements: {opcode_disagreements}, word/boundary disagreements on lifted modules: {word_disagreements}, text differences on lifted modules: {text_differences}; unsupported codec encoding rows: {}, unique spellings: {}; occurrences: {}. Text differences do **not** establish semantic inequality (builder bundles contain VMEM `offen`/`offset` ordering and `m0`/`null` rendering differences); semantic agreement remains unproven where text differs or lifting rejects.\n\n", modules.len()-indexed, modules.len(), modules.len()-clean, unknowns.len(), totals["unknown_encoding_spellings"], totals["unknown_encoding_occurrences"]));
    let indexed_clean = modules
        .iter()
        .filter(|m| m["kind"] == "indexed" && m["lifted_roundtrip"] == true)
        .count();
    md.push_str(&format!("**M7 result: {}.** {indexed_clean}/{indexed} indexed modules and {}/{} builder bundles produce a byte-identical lift/emit; {} modules reject (listed above). These are diagnostic compiler/encoder receipts only: no strict receipt is issued for the indexed set, no candidate is promoted, and packaging/admission policy stays **off** by default.\n\n",
        if clean == modules.len() { "every object round-trips" } else { "fail closed" },
        clean - indexed_clean, modules.len() - indexed, modules.len() - clean));
    md.push_str("### Rejections by rule\n\n| Rule | Count |\n|---|---:|\n");
    for (rule, n) in &rejected_rules {
        md.push_str(&format!("| {rule} | {n} |\n"));
    }
    md.push_str(
        "\n### Rejections by rule and reason\n\n| Rule: reason | Modules / kernels |\n|---|---:|\n",
    );
    for (reason, n) in &rejected {
        md.push_str(&format!("| {} | {n} |\n", reason.replace('|', "\\|")));
    }
    md.push_str("\n### Obligations by kind / rule\n\n| Kind / rule | Count |\n|---|---:|\n");
    for (kind, n) in &obligations {
        md.push_str(&format!("| {kind} | {n} |\n"));
    }
    md.push_str("\n## Per-module summary\n\nA dash means no analyzed kernel (module lift rejected). Multiple-kernel resource columns are maxima; individual descriptor comparisons are recorded in JSON.\n\n| Module | Kernels | Objdump insts | Lift + emit | First rejection | Obligations | VGPR high/descriptor | SGPR high/meta | LDS known/unknown | Barrier pairs | Delay ambiguous | Hazards | LLVM semantic status / text diffs |\n|---|---:|---:|---|---|---:|---|---|---|---:|---:|---:|---|\n");
    for module in modules {
        let issue = module["rejections"]
            .as_array()
            .unwrap()
            .first()
            .map(|r| {
                format!(
                    "{} {} {}: {}",
                    r["kernel"].as_str().unwrap_or("module"),
                    r["offset"].as_str().unwrap(),
                    r["rule"].as_str().unwrap(),
                    r["reason"].as_str().unwrap()
                )
            })
            .unwrap_or_default();
        let kernels = module["kernels"].as_array().unwrap();
        let count = |field: &str, subfield: &str| {
            kernels
                .iter()
                .map(|k| k[field][subfield].as_u64().unwrap_or(0))
                .sum::<u64>()
        };
        let max = |field: &str, subfield: &str| {
            kernels
                .iter()
                .map(|k| k[field][subfield].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0)
        };
        let resources = if kernels.is_empty() {
            ("-".to_owned(), "-".to_owned(), "-".to_owned())
        } else {
            (
                format!(
                    "{}/{}",
                    max("resources", "vgpr_high_water"),
                    max("resources", "vgpr_descriptor")
                ),
                format!(
                    "{}/{}",
                    max("resources", "sgpr_high_water"),
                    max("resources", "sgpr_metadata")
                ),
                format!(
                    "{}/{}",
                    kernels
                        .iter()
                        .filter(|k| k["lds"]["max_known_end"].is_number())
                        .count(),
                    count("lds", "unknown")
                ),
            )
        };
        let obs = kernels
            .iter()
            .map(|k| k["obligation_details"].as_array().unwrap().len())
            .sum::<usize>();
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {obs} | {} | {} | {} | {} | {} | {} | {} / {} |\n",
            module["module"].as_str().unwrap(),
            module["symbols"].as_array().unwrap().len(),
            module["objdump_instructions"],
            module["lifted_roundtrip"],
            issue.replace('|', "\\|"),
            resources.0,
            resources.1,
            resources.2,
            count("barriers", "paired_signals"),
            count("windows", "ambiguous_delays"),
            kernels
                .iter()
                .map(|k| k["hazards"].as_u64().unwrap_or(0))
                .sum::<u64>(),
            module["objdump_semantic_agreement"].as_str().unwrap(),
            module["objdump_text_differences"].as_array().unwrap().len()
        ));
    }
    md.push_str("\n## Unsupported codec encodings\n\nThe `form` and `XML opcode` entries are pinned MR-ISA XML encoding names/opcodes; `table_has_spelling` in JSON separates a missing row from an unsupported field/variant of an existing row. Every listed sample has the full LLVM instruction words.\n\n| Spelling | Occurrences | Modules | Form | XML opcode | Sample words |\n|---|---:|---:|---|---|---|\n");
    for value in unknowns.values() {
        let strings = |key: &str| {
            value[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let sample = value["sample"]["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect::<Vec<_>>()
            .join(" ");
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} (`{}:{}`) |\n",
            value["spelling"].as_str().unwrap(),
            value["occurrences"],
            value["modules"].as_object().unwrap().len(),
            strings("form"),
            strings("xml_opcode"),
            sample,
            value["sample"]["module"].as_str().unwrap(),
            value["sample"]["offset"].as_str().unwrap()
        ));
    }
    fs::write(out.join("census.md"), md)?;
    Ok(())
}

/// ROCm root the census actually ran against. The objdump argument is what
/// disassembled the objects, so a `<root>/lib/llvm/bin/llvm-objdump` path names
/// its root exactly; any other layout falls back to the shared resolution of
/// the lift tests: `PEACEMAKER_ROCM`, else `ROCM_PATH`, else `/opt/rocm`.
#[path = "../../../rocm.rs"]
mod rocm;
fn rocm_root(objdump: &Path) -> PathBuf {
    let layout = objdump
        .ancestors()
        .nth(4)
        .filter(|root| root.join("lib/llvm/bin").join("llvm-objdump") == objdump);
    layout.map(Path::to_path_buf).unwrap_or_else(|| {
        std::env::var_os("PEACEMAKER_ROCM")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(rocm::root)
    })
}

fn run() -> Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let [arch_arg, registry, compiled, output, objdump, xml_tsv, builders @ ..] = args.as_slice()
    else {
        return Err("usage: pm-census <gfx1100|gfx1151|gfx1201> <registry.tsv> <compiled/arch> <output-dir> <llvm-objdump> <arch-encodings.tsv> [builder.hxaco ...]".into());
    };
    let arch = match arch_arg.to_str() {
        Some("gfx1100") => Arch::Gfx1100,
        Some("gfx1151") => Arch::Gfx1151,
        Some("gfx1201") => Arch::Gfx1201,
        _ => return Err("unsupported census architecture".into()),
    };
    fs::create_dir_all(output)?;
    let xml = xml_table(xml_tsv, arch)?;
    let mut entries = BTreeMap::<String, String>::new();
    for row in fs::read_to_string(registry)?.lines() {
        let cols = row.split('\t').collect::<Vec<_>>();
        if !(cols.len() == 6 || cols.len() == 7) || cols[0] != arch_name(arch) {
            return Err(format!("invalid registry row: {row}").into());
        }
        let key = cols.get(6).unwrap_or(&cols[1]).to_string();
        if entries.insert(key.clone(), cols[1].to_owned()).is_some() {
            return Err(format!("duplicate registry artifact key {key}").into());
        }
    }
    let objects = fs::read_dir(compiled)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.extension().is_some_and(|ext| ext == "hsaco"))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect::<BTreeSet<_>>();
    let keys = entries.keys().cloned().collect::<BTreeSet<_>>();
    if keys != objects {
        return Err(format!(
            "registry/pack object-set mismatch: missing {:?}, extra {:?}",
            keys.difference(&objects).collect::<Vec<_>>(),
            objects.difference(&keys).collect::<Vec<_>>()
        )
        .into());
    }
    let scratch = output.join(".pm-census-device.co");
    let mut unknowns = BTreeMap::new();
    let mut modules = Vec::new();
    for (key, name) in entries {
        eprintln!("census {key}");
        modules.push(scan_module(
            &name,
            &compiled.join(format!("{key}.hsaco")),
            true,
            objdump,
            &scratch,
            &xml,
            &mut unknowns,
            arch,
        )?);
    }
    for path in builders {
        let name = path
            .file_stem()
            .ok_or("builder object has no stem")?
            .to_string_lossy();
        eprintln!("census {name}");
        modules.push(scan_module(
            &name,
            path,
            false,
            objdump,
            &scratch,
            &xml,
            &mut unknowns,
            arch,
        )?);
    }
    let provenance = json!({
        "registry_sha256": hex(&Sha256::digest(&fs::read(registry)?)),
        "xml_encoding_tsv_sha256": hex(&Sha256::digest(&fs::read(xml_tsv)?)),
        "llvm_objdump_sha256": hex(&Sha256::digest(&fs::read(objdump)?)),
        "rocm": rocm_root(objdump).display().to_string(), "xml_sha256": xml_sha(arch)
    });
    report(&modules, &unknowns, provenance, output, arch)?;
    eprintln!(
        "wrote {} and {}",
        output.join("census.json").display(),
        output.join("census.md").display()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("pm-census: {error}");
        std::process::exit(1);
    }
}
