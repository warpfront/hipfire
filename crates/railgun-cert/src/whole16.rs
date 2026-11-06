// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Whole16 window manifest generation and verification (railgun DFlash
//! window design A, §5, slice A2).
//!
//! The manifest generator takes the **ordered node sequence of a window plan**
//! (symbol, execution route and the exact loaded object bytes of each node:
//! draft embeddings, the five draft layers, draft head, top-K, selector, the
//! verify body, FP8 KV writer, GDN pre-capture and recurrence, hidden commit,
//! verify head/argmax, copy/fill helpers) and certifies every object with
//! [`crate::certify`]. It refuses the plan when any node
//!
//! * is not a railgun node (`HipDirect`, `Graph`),
//! * has no exact object, no receipt kernel record, or A1 facts that are not
//!   `Proven` / are incomplete (`dynamic_kernarg_reads`, `taint_overflow`) or
//!   whose G8 metadata differential fails,
//! * loads a hidden argument railgun does not supply,
//! * is a kernel with an explicit-ABI table (the FP8 tile/reduce) whose
//!   code-object metadata and facts differ from that table,
//!
//! and when a required node class (the FP8 tile and reduce, the KV writer,
//! `qwen35_fa_prep`, the draft attention/conv/selector kernels, GDN, commit,
//! copy/fill, argmax) is absent from the sequence. No symbol or pointer
//! allowlist is consulted: a node passes only through its own object's facts.
//!
//! [`verify`] regenerates the manifest from the supplied objects and compares
//! it with a stored manifest, so a wrong digest, a changed object or a changed
//! fact refuses.
use std::collections::{BTreeMap, BTreeSet, HashMap};

use peacemaker_ir::kernarg::HIDDEN_GRID_KINDS;
use railgun_corpus::window::{
    sequence_digest, AbiArgRecord, AbiReceipt, NodeRoute, WindowManifest, WindowNodeRecord,
    WindowRefusal, WHOLE16_SHAPE, WINDOW_MANIFEST_SCHEMA, WINDOW_MANIFEST_VERSION,
};

use crate::{certify, sha256_hex, KernelRecord, Receipt};

/// A required node class: satisfied when some node's symbol equals (when
/// `exact`) or contains one of `any_of`.
#[derive(Clone, Copy, Debug)]
pub struct ClassSpec {
    pub id: &'static str,
    pub any_of: &'static [&'static str],
    pub exact: bool,
}

/// The node classes a Whole16 sequence must contain (design §4, §5). Names
/// are the registry/launcher symbols; classes whose kernels vary by route
/// (embeddings, GEMMs, norms, heads) are covered by the per-node facts rule
/// alone.
pub const REQUIRED_CLASSES: &[ClassSpec] = &[
    ClassSpec {
        id: "draft_embedding",
        // Loaded symbols: `embedding_q8` / `embedding_hfq4g*` (scalar) and their
        // `_batched` twins; the launcher name `embedding_lookup_*` never appears
        // in a plan.
        any_of: &["embedding_"],
        exact: false,
    },
    ClassSpec {
        id: "draft_attention_sliding",
        any_of: &["attention_dflash_sliding"],
        exact: false,
    },
    ClassSpec {
        id: "draft_conv",
        any_of: &["dynamic_causal_conv", "dynamic_conv"],
        exact: false,
    },
    ClassSpec {
        id: "draft_topk",
        // gfx1201 routes the raw top-K through the re-grid kernel (+ the fixup
        // body on tied rows); the shipping kernel is `topk_logsumexp_batched_f32`.
        any_of: &["topk_values_regrid", "topk_logsumexp_batched"],
        exact: false,
    },
    ClassSpec {
        id: "selector",
        any_of: &["dflash_selector_greedy_exact"],
        exact: true,
    },
    ClassSpec {
        id: "kv_writer_fp8",
        any_of: &["kv_cache_write_fp8_e4m3"],
        exact: false,
    },
    ClassSpec {
        id: "fa_prep",
        any_of: &["qwen35_fa_prep"],
        exact: false,
    },
    ClassSpec {
        id: "verify_attention_tile_fp8",
        any_of: &["attention_verify_gqa_fp8_gfx1201"],
        exact: true,
    },
    ClassSpec {
        id: "verify_attention_reduce",
        any_of: &["attention_verify_reduce_gfx1201"],
        exact: true,
    },
    ClassSpec {
        id: "gdn_pre_capture",
        any_of: &["dflash_gdn_pre_capture", "gdn_pre_batched"],
        exact: false,
    },
    ClassSpec {
        id: "gdn_recurrence",
        any_of: &["gated_delta_net"],
        exact: false,
    },
    ClassSpec {
        id: "hidden_commit",
        any_of: &["dflash_hidden_commit"],
        exact: false,
    },
    ClassSpec {
        id: "head_argmax",
        any_of: &["argmax"],
        exact: false,
    },
    ClassSpec {
        id: "copy",
        any_of: &["railgun_copy"],
        exact: false,
    },
    ClassSpec {
        id: "fill",
        any_of: &["railgun_fill_u32"],
        exact: true,
    },
];

impl ClassSpec {
    fn matches(&self, symbol: &str) -> bool {
        self.any_of.iter().any(|p| {
            if self.exact {
                symbol == *p
            } else {
                symbol.contains(p)
            }
        })
    }
}

/// What an explicit argument of an ABI-table kernel must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbiEffect {
    /// Global pointer; A1 must report exactly `read`.
    Read,
    /// Global pointer; A1 must report exactly `write`.
    Write,
    /// By-value scalar; never a pointer (A4 role `integer` or `not_loaded`).
    Scalar,
}

impl AbiEffect {
    fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Scalar => "scalar",
        }
    }
}

/// The source-ABI expectation of one kernel (design §5 table). The explicit
/// end is derived from the table; the loader segment size (which includes the
/// hidden block) is whatever the code object declares.
#[derive(Clone, Copy, Debug)]
pub struct AbiSpec {
    pub symbol: &'static str,
    /// Whether Whole16 may execute this kernel. The GQA8 sibling shares the
    /// tile ABI but is not the 27B shape: receipted, never admitted.
    pub admitted: bool,
    pub args: &'static [(u32, u32, AbiEffect)],
}

impl AbiSpec {
    pub fn explicit_end(&self) -> u32 {
        self.args.iter().map(|(o, s, _)| o + s).max().unwrap_or(0)
    }
}

use AbiEffect::{Read, Scalar, Write};

/// `attention_verify_gqa.gfx1201.hip` FP8 tile: q, k_cache, v_cache read,
/// partials write, positions read; n_rows, n_kv_heads, max_tiles,
/// batch_offset scalars at 40/44/48/52; scale_attn (f32 bits) at 56.
const TILE_ARGS: &[(u32, u32, AbiEffect)] = &[
    (0, 8, Read),
    (8, 8, Read),
    (16, 8, Read),
    (24, 8, Write),
    (32, 8, Read),
    (40, 4, Scalar),
    (44, 4, Scalar),
    (48, 4, Scalar),
    (52, 4, Scalar),
    (56, 4, Scalar),
];

pub const FP8_ABI: &[AbiSpec] = &[
    AbiSpec {
        symbol: "attention_verify_gqa_fp8_gfx1201",
        admitted: true,
        args: TILE_ARGS,
    },
    AbiSpec {
        symbol: "attention_verify_gqa_fp8_gqa8_gfx1201",
        admitted: false,
        args: TILE_ARGS,
    },
    // reduce: partials read +0, out write +8, positions read +16;
    // n_heads/max_tiles/batch_offset at +24/+28/+32.
    AbiSpec {
        symbol: "attention_verify_reduce_gfx1201",
        admitted: true,
        args: &[
            (0, 8, Read),
            (8, 8, Write),
            (16, 8, Read),
            (24, 4, Scalar),
            (28, 4, Scalar),
            (32, 4, Scalar),
        ],
    },
];

/// Hidden kinds the railgun kernarg builder supplies: the geometry words it
/// rebuilds from the bound grid, zero global offsets and the dynamic-LDS size.
/// A loaded hidden argument outside this set is not supplied by railgun.
pub fn supplied_hidden(kind: &str) -> bool {
    HIDDEN_GRID_KINDS.contains(&kind)
        || matches!(
            kind,
            "hidden_global_offset_x"
                | "hidden_global_offset_y"
                | "hidden_global_offset_z"
                | "hidden_dynamic_lds_size"
        )
}

/// End of the explicit (non-hidden) arguments of a kernel record.
pub fn explicit_end(record: &KernelRecord) -> u32 {
    record
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .map(|a| a.offset + a.size)
        .max()
        .unwrap_or(0)
}

/// Refusals for a loaded hidden argument railgun does not supply.
pub fn check_hidden(record: &KernelRecord) -> Vec<String> {
    record
        .implicit
        .hidden_loaded
        .iter()
        .filter(|k| !supplied_hidden(k))
        .map(|k| format!("unexpected hidden argument {k} is loaded"))
        .collect()
}

/// Assert `spec`'s explicit ABI against the record built from the actual code
/// object: every explicit argument offset/size/effect, the explicit end
/// derived from the table, hidden arguments only at or after that end and
/// only of supplied kinds, and a loader segment that holds the explicit
/// block (its size is read from the object, not asserted).
pub fn check_abi(record: &KernelRecord, spec: &AbiSpec) -> Result<AbiReceipt, Vec<String>> {
    let mut why = Vec::new();
    let end = spec.explicit_end();
    let explicit: Vec<_> = record
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .collect();
    if explicit.len() != spec.args.len() {
        why.push(format!(
            "{} explicit arguments in the object, ABI table has {}",
            explicit.len(),
            spec.args.len()
        ));
    }
    for (offset, size, effect) in spec.args {
        let Some(a) = explicit.iter().find(|a| a.offset == *offset) else {
            why.push(format!("no explicit argument at +{offset}"));
            continue;
        };
        if a.size != *size {
            why.push(format!("+{offset}: size {} is not {size}", a.size));
        }
        match effect {
            Read | Write => {
                let want = effect.name();
                if a.value_kind != "global_buffer" {
                    why.push(format!(
                        "+{offset}: {} is not a global_buffer",
                        a.value_kind
                    ));
                }
                if a.mode.as_deref() != Some(want) {
                    why.push(format!(
                        "+{offset}: A1 mode {:?}, table requires {want}",
                        a.mode
                    ));
                }
            }
            Scalar => {
                if a.value_kind == "global_buffer" || a.role == "pointer" || a.mode.is_some() {
                    why.push(format!(
                        "+{offset}: scalar is a pointer (kind {}, role {}, mode {:?})",
                        a.value_kind, a.role, a.mode
                    ));
                }
            }
        }
    }
    let object_end = explicit_end(record);
    if object_end != end {
        why.push(format!(
            "explicit arguments end at {object_end}, table at {end}"
        ));
    }
    if record.kernarg_size < object_end {
        why.push(format!(
            "loader segment {} does not hold the {object_end}-byte explicit block",
            record.kernarg_size
        ));
    }
    for a in record
        .args
        .iter()
        .filter(|a| a.value_kind.starts_with("hidden_"))
    {
        if a.offset < end {
            why.push(format!(
                "hidden {} at +{} overlaps the explicit block ending at {end}",
                a.value_kind, a.offset
            ));
        }
        if a.offset + a.size > record.kernarg_size {
            why.push(format!(
                "hidden {} at +{} exceeds the loader segment {}",
                a.value_kind, a.offset, record.kernarg_size
            ));
        }
    }
    why.extend(check_hidden(record));
    if !why.is_empty() {
        return Err(why);
    }
    Ok(AbiReceipt {
        symbol: spec.symbol.to_owned(),
        artifact_sha256: String::new(),
        admitted: spec.admitted,
        kernarg_size: record.kernarg_size,
        explicit_end: object_end,
        args: spec
            .args
            .iter()
            .map(|(offset, size, effect)| AbiArgRecord {
                offset: *offset,
                size: *size,
                effect: effect.name().to_owned(),
            })
            .collect(),
        hidden_loaded: record.implicit.hidden_loaded.clone(),
    })
}

/// One node of the requested window plan with its exact loaded object.
#[derive(Clone, Debug)]
pub struct SequenceEntry {
    pub phase: String,
    pub symbol: String,
    pub route: NodeRoute,
    /// The loaded object bytes (HIP bundle or ELF); `None` = no artifact.
    pub object: Option<Vec<u8>>,
}

fn facts_sha256(record: &KernelRecord) -> String {
    sha256_hex(&serde_json::to_vec(record).expect("a kernel record serialises"))
}

/// Generate the Whole16 manifest with the design's [`REQUIRED_CLASSES`].
pub fn build(
    entries: &[SequenceEntry],
    arch: &str,
    toolchain_pin: &str,
) -> Result<WindowManifest, Vec<WindowRefusal>> {
    build_with(entries, arch, toolchain_pin, REQUIRED_CLASSES)
}

/// [`build`] with an explicit class list (tests use small sequences).
pub fn build_with(
    entries: &[SequenceEntry],
    arch: &str,
    toolchain_pin: &str,
    classes: &[ClassSpec],
) -> Result<WindowManifest, Vec<WindowRefusal>> {
    let mut refusals = Vec::new();
    let mut nodes = Vec::with_capacity(entries.len());
    let mut abi: BTreeMap<(String, String), AbiReceipt> = BTreeMap::new();
    let mut receipts: HashMap<String, Result<Receipt, String>> = HashMap::new();
    let mut satisfied: BTreeSet<&'static str> = BTreeSet::new();
    if toolchain_pin.len() != 64 || !toolchain_pin.bytes().all(|b| b.is_ascii_hexdigit()) {
        refusals.push(WindowRefusal::whole(
            "toolchain pin is not a SHA-256 digest",
        ));
    }
    for (index, e) in entries.iter().enumerate() {
        let before = refusals.len();
        let refuse = |refusals: &mut Vec<WindowRefusal>, why: String| {
            refusals.push(WindowRefusal::node(index, &e.symbol, why))
        };
        if e.route != NodeRoute::Railgun {
            refuse(
                &mut refusals,
                format!("route {:?} is not a railgun node", e.route),
            );
        }
        if let Some(spec) = FP8_ABI.iter().find(|s| s.symbol == e.symbol) {
            if !spec.admitted {
                refuse(
                    &mut refusals,
                    "kernel is receipted but not admitted for the 27B Whole16 shape".into(),
                );
            }
        }
        let Some(object) = &e.object else {
            refuse(
                &mut refusals,
                "no exact artifact: the node's loaded object is absent".into(),
            );
            continue;
        };
        let artifact = sha256_hex(object);
        let receipt = receipts
            .entry(artifact.clone())
            .or_insert_with(|| certify(object, None).map_err(|err| err.to_string()));
        let receipt = match receipt {
            Ok(r) => &*r,
            Err(err) => {
                refuse(
                    &mut refusals,
                    format!("railgun-cert rejected the object: {err}"),
                );
                continue;
            }
        };
        if receipt.arch != arch {
            refuse(
                &mut refusals,
                format!("object is {}, plan is {arch}", receipt.arch),
            );
        }
        let (Some(record), Some(identity)) = (
            receipt
                .railgun
                .kernels
                .iter()
                .find(|k| k.symbol == e.symbol),
            receipt.kernels.iter().find(|k| k.symbol == e.symbol),
        ) else {
            refuse(
                &mut refusals,
                format!("the object's receipt has no kernel {}", e.symbol),
            );
            continue;
        };
        if let Err(why) = record.validate_case(
            &receipt.object_sha256,
            &receipt.device_elf_sha256,
            &receipt.arch,
        ) {
            refuse(&mut refusals, format!("guarded case invalid: {why}"));
        }
        if record.status != "proven" && record.status != "conditional" {
            let rules: BTreeSet<&str> = record.unresolved.iter().map(|u| u.rule.as_str()).collect();
            refuse(
                &mut refusals,
                format!(
                    "missing facts: A1 status {} (unresolved: {})",
                    record.status,
                    rules.into_iter().collect::<Vec<_>>().join(", ")
                ),
            );
        }
        if record.dynamic_kernarg_reads || record.taint_overflow {
            refuse(
                &mut refusals,
                "A4 facts are incomplete (dynamic kernarg read or taint overflow)".into(),
            );
        }
        if record.differential.status == "fail" {
            refuse(&mut refusals, "G8 metadata differential failed".into());
        }
        if identity.text_sha256.is_empty() || identity.kd_sha256.is_empty() {
            refuse(
                &mut refusals,
                "code identity is incomplete (no text or descriptor digest)".into(),
            );
        }
        for why in check_hidden(record) {
            refuse(&mut refusals, why);
        }
        // ABI-table kernels: this node's own spec, and every sibling the same
        // object exports (the GQA8 receipt is generated, never admitted).
        for spec in FP8_ABI {
            let own = spec.symbol == e.symbol;
            let Some(sibling) = receipt
                .railgun
                .kernels
                .iter()
                .find(|k| k.symbol == spec.symbol)
            else {
                continue;
            };
            match check_abi(sibling, spec) {
                Ok(mut r) => {
                    r.artifact_sha256 = artifact.clone();
                    abi.insert((artifact.clone(), spec.symbol.to_owned()), r);
                }
                Err(why) => refuse(
                    &mut refusals,
                    format!(
                        "{}ABI table mismatch for {}: {}",
                        if own { "" } else { "sibling " },
                        spec.symbol,
                        why.join("; ")
                    ),
                ),
            }
        }
        let class = classes
            .iter()
            .find(|c| c.matches(&e.symbol))
            .map_or("", |c| c.id);
        for c in classes.iter().filter(|c| c.matches(&e.symbol)) {
            if refusals.len() == before {
                satisfied.insert(c.id);
            }
        }
        nodes.push(WindowNodeRecord {
            index: index as u32,
            phase: e.phase.clone(),
            class: class.to_owned(),
            symbol: e.symbol.clone(),
            artifact_sha256: artifact,
            device_elf_sha256: receipt.device_elf_sha256.clone(),
            text_sha256: identity.text_sha256.clone(),
            kd_sha256: identity.kd_sha256.clone(),
            metadata_sha256: identity.metadata_sha256.clone(),
            receipt_sha256: facts_sha256(record),
            a1_status: record.status.clone(),
            case_sha256: record.case_digest.clone(),
            required_witnesses: record.required_witnesses.clone(),
            kernarg_size: record.kernarg_size,
            explicit_end: explicit_end(record),
            hidden_loaded: record.implicit.hidden_loaded.clone(),
        });
    }
    for c in classes {
        if !satisfied.contains(c.id) {
            refusals.push(WindowRefusal::whole(format!(
                "required node class {} ({}) has no certified node in the sequence",
                c.id,
                c.any_of.join("|")
            )));
        }
    }
    if !refusals.is_empty() {
        return Err(refusals);
    }
    Ok(WindowManifest {
        schema: WINDOW_MANIFEST_SCHEMA.into(),
        version: WINDOW_MANIFEST_VERSION,
        shape: WHOLE16_SHAPE.into(),
        arch: arch.into(),
        toolchain_pin: toolchain_pin.into(),
        sequence_sha256: sequence_digest(&nodes),
        nodes,
        abi: abi.into_values().collect(),
    })
}

/// Regenerate the manifest from `entries` and compare with `stored`: any
/// difference (digest, order, fact, ABI receipt, toolchain pin) refuses.
pub fn verify(
    stored: &WindowManifest,
    entries: &[SequenceEntry],
    arch: &str,
    toolchain_pin: &str,
) -> Result<(), Vec<WindowRefusal>> {
    verify_with(stored, entries, arch, toolchain_pin, REQUIRED_CLASSES)
}

pub fn verify_with(
    stored: &WindowManifest,
    entries: &[SequenceEntry],
    arch: &str,
    toolchain_pin: &str,
    classes: &[ClassSpec],
) -> Result<(), Vec<WindowRefusal>> {
    let mut out = stored.validate();
    match build_with(entries, arch, toolchain_pin, classes) {
        Err(r) => out.extend(r),
        Ok(fresh) => {
            if fresh.toolchain_pin != stored.toolchain_pin || fresh.arch != stored.arch {
                out.push(WindowRefusal::whole("arch or toolchain pin differs"));
            }
            if fresh.nodes.len() != stored.nodes.len() {
                out.push(WindowRefusal::whole(format!(
                    "{} nodes now, manifest has {}",
                    fresh.nodes.len(),
                    stored.nodes.len()
                )));
            }
            for (i, (a, b)) in fresh.nodes.iter().zip(&stored.nodes).enumerate() {
                if a != b {
                    let field = if a.artifact_sha256 != b.artifact_sha256 {
                        "object digest"
                    } else if a.receipt_sha256 != b.receipt_sha256 {
                        "receipt facts"
                    } else {
                        "code identity or ABI fields"
                    };
                    out.push(WindowRefusal::node(
                        i,
                        &a.symbol,
                        format!("{field} differs from the manifest"),
                    ));
                }
            }
            if fresh.abi != stored.abi {
                out.push(WindowRefusal::whole("ABI-table receipts differ"));
            }
            if fresh.sequence_sha256 != stored.sequence_sha256 {
                out.push(WindowRefusal::whole("sequence digest differs"));
            }
        }
    }
    if out.is_empty() {
        Ok(())
    } else {
        Err(out)
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SequenceLine {
    phase: String,
    symbol: String,
    #[serde(default = "railgun_route")]
    route: NodeRoute,
    object: Option<String>,
    module: Option<String>,
}

fn railgun_route() -> NodeRoute {
    NodeRoute::Railgun
}

/// The object file of a module in `dirs`: `NAME.hsaco` or `NAME.<key>.hsaco`.
/// Several files with different bytes are ambiguous, never guessed between.
fn find_module_object(
    module: &str,
    dirs: &[std::path::PathBuf],
) -> Result<Option<std::path::PathBuf>, String> {
    let mut found: BTreeMap<String, std::path::PathBuf> = BTreeMap::new();
    for dir in dirs {
        let Ok(read) = std::fs::read_dir(dir) else {
            return Err(format!("{}: unreadable objects directory", dir.display()));
        };
        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(stem) = name.strip_suffix(".hsaco") else {
                continue;
            };
            let hit = stem == module
                || stem
                    .strip_prefix(module)
                    .and_then(|r| r.strip_prefix('.'))
                    .is_some_and(|key| !key.contains('.'));
            if hit {
                let bytes = std::fs::read(entry.path())
                    .map_err(|e| format!("{}: {e}", entry.path().display()))?;
                found
                    .entry(sha256_hex(&bytes))
                    .or_insert_with(|| entry.path());
            }
        }
    }
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.into_values().next()),
        n => Err(format!(
            "module {module}: {n} different objects match ({}); name one with \"object\"",
            found
                .values()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// `whole16-manifest`, `whole16-verify` and `whole16-classes` (see main.rs).
pub fn cli(command: &str, args: &[String]) -> Result<(), String> {
    use std::path::PathBuf;
    if command == "whole16-classes" {
        for c in REQUIRED_CLASSES {
            println!(
                "{}\t{}\t{}",
                c.id,
                if c.exact { "exact" } else { "contains" },
                c.any_of.join("|")
            );
        }
        for s in FP8_ABI {
            println!(
                "abi\t{}\tadmitted={}\texplicit_end={}",
                s.symbol,
                s.admitted,
                s.explicit_end()
            );
        }
        return Ok(());
    }
    let mut sequence = None;
    let mut arch = None;
    let mut pin = None;
    let mut manifest_path = None;
    let mut out = PathBuf::from("railgun-cert-out");
    let mut dirs = Vec::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--sequence" => sequence = Some(PathBuf::from(value()?)),
            "--arch" => arch = Some(value()?),
            "--toolchain-id" => pin = Some(sha256_hex(value()?.as_bytes())),
            "--toolchain-pin" => pin = Some(value()?),
            "--corpus" => {
                let corpus =
                    railgun_corpus::JitCorpus::open(value()?).map_err(|e| e.to_string())?;
                pin = Some(corpus.toolchain_pin().to_owned());
            }
            "--objects" => dirs.push(PathBuf::from(value()?)),
            "--manifest" => manifest_path = Some(PathBuf::from(value()?)),
            "--out" => out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let sequence = sequence.ok_or("--sequence is required")?;
    let arch = arch.ok_or("--arch is required")?;
    let pin = pin.ok_or("one of --toolchain-id, --toolchain-pin, --corpus is required")?;
    let text =
        std::fs::read_to_string(&sequence).map_err(|e| format!("{}: {e}", sequence.display()))?;
    let mut entries = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let l: SequenceLine = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", sequence.display(), n + 1))?;
        let path = match (&l.object, &l.module) {
            (Some(o), _) => Some(PathBuf::from(o)),
            (None, Some(m)) => find_module_object(m, &dirs)?,
            (None, None) => None,
        };
        let object = match &path {
            Some(p) => {
                let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Some(bytes)
            }
            None => None,
        };
        entries.push(SequenceEntry {
            phase: l.phase,
            symbol: l.symbol,
            route: l.route,
            object,
        });
    }
    // Receipts for every distinct object, next to the manifest.
    let write_receipts = |entries: &[SequenceEntry]| -> Result<(), String> {
        let mut done = BTreeSet::new();
        for e in entries {
            let Some(bytes) = &e.object else { continue };
            if !done.insert(sha256_hex(bytes)) {
                continue;
            }
            if let Ok(receipt) = certify(bytes, None) {
                let dir = out.join("receipts").join(&receipt.arch);
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                std::fs::write(
                    dir.join(format!("{}.receipt.json", receipt.object_sha256)),
                    serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    };
    let refused = |refusals: Vec<WindowRefusal>| -> String {
        for r in &refusals {
            eprintln!("REFUSE\t{r}");
        }
        format!(
            "{} refusal(s): Whole16 plan is not certified",
            refusals.len()
        )
    };
    match command {
        "whole16-manifest" => {
            write_receipts(&entries)?;
            let manifest = build(&entries, &arch, &pin).map_err(refused)?;
            std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
            let file = out.join("whole16.manifest.json");
            std::fs::write(
                &file,
                serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            eprintln!(
                "{} nodes, {} ABI receipts, sequence {} -> {}",
                manifest.nodes.len(),
                manifest.abi.len(),
                manifest.sequence_sha256,
                file.display()
            );
            Ok(())
        }
        "whole16-verify" => {
            let path = manifest_path.ok_or("whole16-verify needs --manifest")?;
            let stored: WindowManifest = serde_json::from_slice(
                &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
            )
            .map_err(|e| format!("{}: {e}", path.display()))?;
            verify(&stored, &entries, &arch, &pin).map_err(refused)?;
            eprintln!("verified {} nodes", stored.nodes.len());
            Ok(())
        }
        other => Err(format!("unknown command {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArgRecord, Differential, ImplicitRecord, SideRecord};

    const COPY: &[u8] = include_bytes!("../tests/fixtures/railgun_copy.gfx1201.hsaco");
    const PTRTABLE: &[u8] = include_bytes!("../tests/fixtures/railgun_copy_ptrtable.gfx1201.hsaco");
    const PIN: &str = "0000000000000000000000000000000000000000000000000000000000000001";
    const COPY_CLASS: &[ClassSpec] = &[ClassSpec {
        id: "copy",
        any_of: &["railgun_copy"],
        exact: false,
    }];

    fn entry(symbol: &str, route: NodeRoute, object: Option<&[u8]>) -> SequenceEntry {
        SequenceEntry {
            phase: "transport".into(),
            symbol: symbol.into(),
            route,
            object: object.map(<[u8]>::to_vec),
        }
    }

    fn good() -> Vec<SequenceEntry> {
        vec![
            entry("railgun_copy", NodeRoute::Railgun, Some(COPY)),
            entry("railgun_copy_batch", NodeRoute::Railgun, Some(COPY)),
        ]
    }

    fn reasons(r: Result<WindowManifest, Vec<WindowRefusal>>) -> String {
        r.expect_err("refuses")
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[test]
    fn proven_exact_objects_build_and_verify() {
        let m = build_with(&good(), "gfx1201", PIN, COPY_CLASS).expect("builds");
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(m.nodes[0].artifact_sha256, sha256_hex(COPY));
        assert_eq!(m.nodes[0].class, "copy");
        assert!(m.validate().is_empty());
        verify_with(&m, &good(), "gfx1201", PIN, COPY_CLASS).expect("verifies");
    }

    #[test]
    fn wrong_digest_refuses() {
        let m = build_with(&good(), "gfx1201", PIN, COPY_CLASS).unwrap();
        // The manifest names another object digest for node 1.
        let mut forged = m.clone();
        forged.nodes[1].artifact_sha256 = sha256_hex(b"other");
        forged.sequence_sha256 = sequence_digest(&forged.nodes);
        let e = verify_with(&forged, &good(), "gfx1201", PIN, COPY_CLASS).expect_err("refuses");
        assert!(
            e.iter().any(|r| r.reason.contains("object digest")),
            "{e:?}"
        );
        // The plan loads different bytes for node 0.
        let mut flipped = COPY.to_vec();
        *flipped.last_mut().unwrap() ^= 0xff;
        let swapped = vec![
            entry("railgun_copy", NodeRoute::Railgun, Some(&flipped)),
            entry("railgun_copy_batch", NodeRoute::Railgun, Some(COPY)),
        ];
        assert!(verify_with(&m, &swapped, "gfx1201", PIN, COPY_CLASS).is_err());
        // A stale recorded sequence digest.
        let mut stale = m.clone();
        stale.sequence_sha256 = sha256_hex(b"stale");
        assert!(verify_with(&stale, &good(), "gfx1201", PIN, COPY_CLASS).is_err());
        // Another toolchain pin.
        let other = "0000000000000000000000000000000000000000000000000000000000000002";
        assert!(verify_with(&m, &good(), "gfx1201", other, COPY_CLASS).is_err());
    }

    #[test]
    fn missing_fact_or_artifact_refuses() {
        // The pointer-table kernel's accesses are memory-derived: A1 Unknown.
        let e = vec![entry(
            "railgun_copy_batch",
            NodeRoute::Railgun,
            Some(PTRTABLE),
        )];
        assert!(reasons(build_with(&e, "gfx1201", PIN, COPY_CLASS)).contains("missing facts"));
        // No object at all.
        let e = vec![entry("railgun_copy", NodeRoute::Railgun, None)];
        assert!(reasons(build_with(&e, "gfx1201", PIN, COPY_CLASS)).contains("no exact artifact"));
        // A symbol the object does not export.
        let e = vec![entry("railgun_fill_u32", NodeRoute::Railgun, Some(COPY))];
        assert!(reasons(build_with(&e, "gfx1201", PIN, &[])).contains("has no kernel"));
        // Garbage bytes.
        let e = vec![entry("railgun_copy", NodeRoute::Railgun, Some(b"nope"))];
        assert!(reasons(build_with(&e, "gfx1201", PIN, &[])).contains("rejected"));
        // Wrong arch.
        assert!(reasons(build_with(&good(), "gfx1100", PIN, COPY_CLASS)).contains("plan is"));
    }

    #[test]
    fn hipdirect_graph_and_missing_class_refuse() {
        for route in [NodeRoute::HipDirect, NodeRoute::Graph] {
            let e = vec![entry("railgun_copy", route, Some(COPY))];
            assert!(
                reasons(build_with(&e, "gfx1201", PIN, COPY_CLASS)).contains("not a railgun node")
            );
        }
        // The full design class list is absent from a copy-only sequence.
        let why = reasons(build(&good(), "gfx1201", PIN));
        assert!(why.contains("verify_attention_tile_fp8"), "{why}");
        assert!(why.contains("fill"), "{why}");
        // A bad pin is refused outright.
        assert!(reasons(build_with(&good(), "gfx1201", "pin", COPY_CLASS)).contains("SHA-256"));
    }

    fn arg(offset: u32, size: u32, kind: &str, role: &str, mode: Option<&str>) -> ArgRecord {
        ArgRecord {
            offset,
            size,
            index: None,
            name: String::new(),
            value_kind: kind.into(),
            role: role.into(),
            mode: mode.map(str::to_owned),
            bound: None,
            classes: vec![],
            paths: vec![],
            cache_bits: vec![],
            accesses: 0,
            shared: false,
            read_cache: String::new(),
            sinks: vec![],
            loaded: true,
        }
    }

    /// A kernel record shaped like `spec` with a 64-byte-aligned hidden block.
    fn record_for(spec: &AbiSpec) -> KernelRecord {
        let mut args: Vec<ArgRecord> = spec
            .args
            .iter()
            .map(|(o, s, e)| match e {
                Scalar => arg(*o, *s, "by_value", "integer", None),
                e => arg(*o, *s, "global_buffer", "pointer", Some(e.name())),
            })
            .collect();
        let hidden_at = spec.explicit_end().div_ceil(8) * 8;
        args.push(arg(hidden_at, 4, "hidden_block_count_x", "integer", None));
        KernelRecord {
            symbol: spec.symbol.into(),
            status: "proven".into(),
            unresolved: vec![],
            kernarg_size: hidden_at + 64,
            read_cache: String::new(),
            dynamic_kernarg_reads: false,
            taint_overflow: false,
            side: SideRecord::default(),
            implicit: ImplicitRecord {
                hidden_loaded: vec!["hidden_block_count_y".into()],
                ..Default::default()
            },
            args,
            dwords: vec![],
            differential: Differential {
                source: "object-metadata".into(),
                status: "pass".into(),
                args: vec![],
                radiowave_mutable_read_cache: None,
                guards: vec![],
                guarded: false,
            },
            pm_obligations: BTreeMap::new(),
            pm_analyze_error: None,
            unguarded_unresolved: vec![],
            guards: vec![],
            indexed_effects: vec![],
            sealed_tables: vec![],
            descriptor_arenas: vec![],
            required_witnesses: vec![],
            case_digest: String::new(),
            guard_flag: None,
        }
    }

    #[test]
    fn abi_table_accepts_the_table_and_refuses_each_deviation() {
        for spec in FP8_ABI {
            let r = record_for(spec);
            let receipt = check_abi(&r, spec).expect("table shape passes");
            assert_eq!(receipt.explicit_end, spec.explicit_end());
            assert_eq!(receipt.admitted, spec.admitted);
            // Unexpected hidden argument.
            let mut bad = record_for(spec);
            bad.implicit.hidden_loaded.push("hidden_queue_ptr".into());
            assert!(check_abi(&bad, spec).unwrap_err()[0].contains("hidden_queue_ptr"));
            // A scalar that is a pointer.
            let mut bad = record_for(spec);
            let last = bad
                .args
                .iter()
                .position(|a| a.value_kind == "by_value")
                .unwrap();
            bad.args[last].role = "pointer".into();
            assert!(check_abi(&bad, spec).is_err());
            // A shifted offset and a changed effect.
            let mut bad = record_for(spec);
            bad.args[0].offset += 4;
            assert!(check_abi(&bad, spec).is_err());
            let mut bad = record_for(spec);
            bad.args[0].mode = Some("read_write".into());
            assert!(check_abi(&bad, spec).is_err());
            // A hidden argument inside the explicit block, an extra explicit
            // argument, and a loader segment too small for the explicit block.
            let mut bad = record_for(spec);
            bad.args.push(arg(
                spec.explicit_end() - 4,
                4,
                "hidden_remainder_x",
                "integer",
                None,
            ));
            assert!(check_abi(&bad, spec).is_err());
            let mut bad = record_for(spec);
            bad.args
                .push(arg(spec.explicit_end(), 4, "by_value", "integer", None));
            assert!(check_abi(&bad, spec).is_err());
            let mut bad = record_for(spec);
            bad.kernarg_size = spec.explicit_end() - 1;
            assert!(check_abi(&bad, spec).is_err());
        }
        let tile = &FP8_ABI[0];
        assert_eq!((tile.explicit_end(), FP8_ABI[2].explicit_end()), (60, 36));
        assert!(!FP8_ABI[1].admitted);
    }

    #[test]
    fn hidden_check_refuses_unsupplied_kinds_only() {
        let mut r = record_for(&FP8_ABI[0]);
        assert!(check_hidden(&r).is_empty());
        r.implicit.hidden_loaded = vec![
            "hidden_block_count_x".into(),
            "hidden_global_offset_z".into(),
            "hidden_dynamic_lds_size".into(),
        ];
        assert!(check_hidden(&r).is_empty());
        for kind in [
            "hidden_queue_ptr",
            "hidden_hostcall_buffer",
            "hidden_heap_v1",
        ] {
            r.implicit.hidden_loaded = vec![kind.into()];
            assert_eq!(check_hidden(&r).len(), 1, "{kind}");
        }
    }
}
