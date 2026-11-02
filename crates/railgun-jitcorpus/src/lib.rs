// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun-jitcorpus: the offline JIT receipt corpus (railgun design §2.5,
//! milestone MJ).
//!
//! The admitted default programs (`programs.json`, one per route and arch,
//! nodes taken from each program's recorded capture) name the runtime-JIT
//! module behind every launch; `programs.json` also names railgun's own
//! kernels (the copy kernels its lowering launches), required on each arch
//! railgun certifies. The generated source and compile recipe for a
//! module are the runtime's own JIT cache-key inputs: the source expression
//! its launcher passes to `ensure_kernel`, as inventoried by
//! `rdna_compute::kernel_registry::corpus_entries`, and
//! `KernelCompiler::recipe_for_source` under the product-default process
//! configuration. `build` compiles each module with the runtime's hipcc argv
//! on this host, certifies the object with `railgun-cert`, and writes a
//! versioned, per-build-host corpus (`railgun_corpus` layout) whose pack
//! directories the runtime can load verbatim. On gfx1201 the product default
//! loads embedded PeaceMaker twin images
//! (`rdna_compute::pm_decode_twins`) instead of the hipcc objects of some
//! modules, so `build` also certifies each planned arch's twins against their
//! radiowave and records them (`origin: pm_native`, keyed by image digest,
//! under `native/<arch>/`, never in the pack). `check` is the CI gate: every
//! node of every node set must resolve, under the runtime's current key
//! (packaging key for hipcc modules, image digest for twins), to a record
//! with a verified receipt.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use hipfire_config::{ConfigLayer, ProcessConfig, CONFIG_SCHEMA_VERSION};
use railgun_corpus::{
    sha256_hex, ArtifactKey, BuildHost, CorpusIndex, CorpusRecord, JitCorpus, KernelVerdict,
    MissingNode, NodeRecord, ObjectOrigin, ProgramNode, ProgramRecord, RejectedObject,
    ToolchainPin, CORPUS_INDEX, CORPUS_SCHEMA, CORPUS_VERSION,
};
use rdna_compute::kernel_registry::{self, KernelEntry};
use rdna_compute::{pm_decode_twins, FeatureFlags, KernelCompiler};
use serde::{Deserialize, Serialize};

pub const PROGRAMS_JSON: &str = include_str!("../programs.json");
pub const PROGRAMS_SCHEMA: &str = "railgun-admitted-programs";

type Result<T, E = String> = std::result::Result<T, E>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub schema: String,
    pub version: u32,
    pub admission: String,
    pub programs: Vec<Program>,
    pub railgun_kernels: RailgunKernels,
    /// Window node sets added at run time from a plan's node sequence
    /// ([`Inventory::add_window_sequence`]); never part of `programs.json`.
    #[serde(skip)]
    pub window_sets: Vec<WindowSet>,
    /// Program node sets added at run time from a program owner's recorded
    /// launch sequence ([`Inventory::add_program_sequence`]); never part of
    /// `programs.json`.
    #[serde(skip)]
    pub sequence_sets: Vec<SequenceSet>,
}

/// A certified-window node set (`DFlash Whole16`), from the sequence the
/// window builder dumped (`HIPFIRE_DFLASH_RAILGUN_SEQUENCE_OUT`).
#[derive(Clone, Debug)]
pub struct WindowSet {
    pub id: String,
    pub arch: String,
    /// Distinct `(symbol, module)`, first-seen order.
    pub nodes: Vec<(String, String)>,
}

/// The node set of a program owner's recorded launch sequence (the Raon
/// plain-AR decode body: `sequence.jsonl` of the probe's inventory phase),
/// with the SHA-256 of the exact code object each node loaded in that run.
#[derive(Clone, Debug)]
pub struct SequenceSet {
    pub id: String,
    pub arch: String,
    /// Distinct `(symbol, module)`, first-seen order.
    pub nodes: Vec<(String, String)>,
    /// `loaded_sha256[i]` is the object node `i` ran from.
    pub loaded_sha256: Vec<String>,
    /// Kernel launches in the sequence (every node counted once per launch).
    pub launches: usize,
    /// Device fills in the sequence (no object: railgun's own fill kernel).
    pub fills: usize,
}

/// The kernels railgun's own lowering launches, on every arch in `archs`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RailgunKernels {
    pub source: String,
    pub archs: Vec<String>,
    /// `(symbol, module)`.
    pub nodes: Vec<(String, String)>,
}

/// Nodes the corpus must cover on one arch: an admitted program's, or
/// railgun's own kernels (id `railgun-kernels-<arch>`).
pub struct NodeSet<'a> {
    pub id: String,
    pub arch: &'a str,
    /// `(symbol, module)`.
    pub nodes: &'a [(String, String)],
    /// SHA-256 of the object each node loaded in the recorded run, when the
    /// node set comes from a recorded sequence ([`SequenceSet`]).
    pub loaded_sha256: Option<&'a [String]>,
}

impl Inventory {
    /// The admitted programs, then railgun's own kernels per arch.
    pub fn node_sets(&self) -> Vec<NodeSet<'_>> {
        let programs = self.programs.iter().map(|p| NodeSet {
            id: p.id.clone(),
            arch: &p.arch,
            nodes: &p.nodes,
            loaded_sha256: None,
        });
        let railgun = self.railgun_kernels.archs.iter().map(|arch| NodeSet {
            id: format!("railgun-kernels-{arch}"),
            arch,
            nodes: &self.railgun_kernels.nodes,
            loaded_sha256: None,
        });
        let windows = self.window_sets.iter().map(|w| NodeSet {
            id: w.id.clone(),
            arch: &w.arch,
            nodes: &w.nodes,
            loaded_sha256: None,
        });
        let sequences = self.sequence_sets.iter().map(|w| NodeSet {
            id: w.id.clone(),
            arch: &w.arch,
            nodes: &w.nodes,
            loaded_sha256: Some(&w.loaded_sha256),
        });
        programs
            .chain(railgun)
            .chain(windows)
            .chain(sequences)
            .collect()
    }

    /// Add the node set of a window plan on `arch`. `jsonl` is the sequence
    /// the window builder dumps (one `{"phase","symbol","route","module"}`
    /// per node, as `railgun-cert whole16-manifest --sequence` reads); only
    /// `symbol` and `module` matter here, duplicates collapse, and every node
    /// must name its module. Every module must then resolve in the registry
    /// (`plan` names each one that does not).
    pub fn add_window_sequence(&mut self, id: &str, arch: &str, jsonl: &str) -> Result<()> {
        #[derive(Deserialize)]
        struct Line {
            symbol: String,
            module: Option<String>,
        }
        let mut nodes: Vec<(String, String)> = Vec::new();
        for (n, line) in jsonl.lines().enumerate() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let l: Line = serde_json::from_str(line)
                .map_err(|e| format!("window sequence line {}: {e}", n + 1))?;
            let module = l.module.ok_or_else(|| {
                format!(
                    "window sequence line {}: node {} has no module",
                    n + 1,
                    l.symbol
                )
            })?;
            let node = (l.symbol, module);
            if !nodes.contains(&node) {
                nodes.push(node);
            }
        }
        if nodes.is_empty() {
            return Err("window sequence names no node".into());
        }
        self.window_sets.push(WindowSet {
            id: id.to_owned(),
            arch: arch.to_owned(),
            nodes,
        });
        Ok(())
    }

    /// Keep only the node sets named in `ids` (admitted programs, window and
    /// sequence sets) plus railgun's own kernels, which every railgun program
    /// needs. An id that names no node set is an error.
    pub fn retain_programs(&mut self, ids: &[String]) -> Result<()> {
        let known: Vec<String> = self
            .programs
            .iter()
            .map(|p| p.id.clone())
            .chain(self.window_sets.iter().map(|w| w.id.clone()))
            .chain(self.sequence_sets.iter().map(|w| w.id.clone()))
            .collect();
        if let Some(id) = ids.iter().find(|id| !known.contains(id)) {
            return Err(format!(
                "--only-program {id}: no such program (known: {})",
                known.join(", ")
            ));
        }
        self.programs.retain(|p| ids.contains(&p.id));
        self.window_sets.retain(|w| ids.contains(&w.id));
        self.sequence_sets.retain(|w| ids.contains(&w.id));
        Ok(())
    }

    /// Add the node set of a program owner's recorded launch sequence on
    /// `arch` under its own `id` (never a DFlash window label). `jsonl` is the
    /// sequence the Raon probe's inventory phase writes: one object per line,
    /// `{"kind":"kernel","symbol","module","sha256",..}` for a launch (grid,
    /// block, LDS and kernarg bytes are ignored here) or `{"kind":"fill",..}`
    /// for a device fill (no object). Every kernel line must name `symbol`,
    /// `module` and the SHA-256 of the object it loaded. Duplicates collapse,
    /// but one `(symbol, module)` loading two different objects, or one
    /// `symbol` under two modules, is an error: the node set would not name
    /// one object per node. Every module must then resolve in the registry
    /// (`plan` names each one that does not).
    pub fn add_program_sequence(&mut self, id: &str, arch: &str, jsonl: &str) -> Result<()> {
        #[derive(Deserialize)]
        struct Line {
            kind: String,
            symbol: Option<String>,
            module: Option<String>,
            sha256: Option<String>,
        }
        let mut nodes: Vec<(String, String)> = Vec::new();
        let mut loaded: Vec<String> = Vec::new();
        let (mut launches, mut fills) = (0usize, 0usize);
        for (n, line) in jsonl.lines().enumerate() {
            let at = n + 1;
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let l: Line = serde_json::from_str(line)
                .map_err(|e| format!("program sequence line {at}: {e}"))?;
            match l.kind.as_str() {
                "fill" => {
                    fills += 1;
                    continue;
                }
                "kernel" => {}
                other => {
                    return Err(format!(
                        "program sequence line {at}: unknown node kind `{other}` (expected `kernel` or `fill`)"
                    ))
                }
            }
            let field = |value: Option<String>, name: &str| {
                value
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| format!("program sequence line {at}: kernel node has no {name}"))
            };
            let symbol = field(l.symbol, "symbol")?;
            let module = field(l.module, "module")?;
            let sha256 = field(l.sha256, "sha256")?;
            if sha256.len() != 64
                || !sha256
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            {
                return Err(format!(
                    "program sequence line {at}: node {symbol}: sha256 `{sha256}` is not 64 lowercase hex digits"
                ));
            }
            launches += 1;
            if let Some(i) = nodes.iter().position(|(s, m)| *s == symbol && *m == module) {
                if loaded[i] != sha256 {
                    return Err(format!(
                        "program sequence line {at}: node {symbol} (module {module}) loaded object {sha256}, but an earlier launch loaded {}",
                        loaded[i]
                    ));
                }
                continue;
            }
            if let Some((_, other)) = nodes.iter().find(|(s, _)| *s == symbol) {
                return Err(format!(
                    "program sequence line {at}: symbol {symbol} appears under modules {other} and {module}"
                ));
            }
            nodes.push((symbol, module));
            loaded.push(sha256);
        }
        if nodes.is_empty() {
            return Err("program sequence names no kernel node".into());
        }
        if self.sequence_sets.iter().any(|s| s.id == id) {
            return Err(format!(
                "program sequence id {id} is already in the inventory"
            ));
        }
        self.sequence_sets.push(SequenceSet {
            id: id.to_owned(),
            arch: arch.to_owned(),
            nodes,
            loaded_sha256: loaded,
            launches,
            fills,
        });
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub id: String,
    pub arch: String,
    pub route: String,
    pub golden: Golden,
    pub nodes_from: NodesFrom,
    /// `(symbol, module)` for every distinct kernel the program launches.
    pub nodes: Vec<(String, String)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Golden {
    pub sequence_hash: String,
    pub launches: u64,
    pub unique_kernels: u64,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NodesFrom {
    pub capture: String,
    pub sequence_hash: String,
    pub launches: u64,
    /// The capture is the sealed golden itself, not an earlier recording.
    pub matches_golden: bool,
    pub note: String,
}

/// Corpus program id of the `ordinal`-th (0-based) Raon plain-AR decode
/// sequence on `arch`: `raon-ar-decode-gfx1201`, then `raon-ar-decode-gfx1201-2`.
pub fn raon_ar_program_id(arch: &str, ordinal: usize) -> String {
    if ordinal == 0 {
        format!("raon-ar-decode-{arch}")
    } else {
        format!("raon-ar-decode-{arch}-{}", ordinal + 1)
    }
}

pub fn inventory() -> Result<Inventory> {
    let inventory: Inventory =
        serde_json::from_str(PROGRAMS_JSON).map_err(|e| format!("programs.json: {e}"))?;
    if inventory.schema != PROGRAMS_SCHEMA || inventory.version != 1 {
        return Err(format!(
            "programs.json: schema {} v{}",
            inventory.schema, inventory.version
        ));
    }
    Ok(inventory)
}

fn product_config() -> ProcessConfig {
    ProcessConfig {
        schema_version: CONFIG_SCHEMA_VERSION,
        values: ConfigLayer::default(),
    }
}

/// Pin the process to the product-default configuration before anything
/// reads it, so no user TOML or `HIPFIRE_*` override reaches a recipe.
pub fn install_product_defaults() -> Result<()> {
    match hipfire_config::install_process_config(product_config()) {
        Ok(()) => Ok(()),
        Err(_) if hipfire_config::active_process_config() == Some(&product_config()) => Ok(()),
        Err(_) => Err("a non-default hipfire process configuration is already active".into()),
    }
}

fn product_flags(arch: &str) -> FeatureFlags {
    FeatureFlags::from_process_config(arch, &product_config())
}

/// `FeatureFlags::hipcc_extra_flags` under the product defaults: the
/// process-global extra flags every JIT compile on `arch` carries.
pub fn default_extra_flags(arch: &str) -> String {
    product_flags(arch).hipcc_extra_flags
}

/// Lowercase hex of a SHA-256 digest.
fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The modules one arch's programs need, as registry entries.
pub struct ArchPlan {
    pub arch: String,
    pub extra_flags: String,
    pub entries: Vec<KernelEntry>,
    /// The embedded PeaceMaker twins the product default loads instead of the
    /// hipcc objects of some `entries` (`pm_decode_twins::pm_decode_module`).
    pub twins: Vec<&'static pm_decode_twins::NativeKernelBundle>,
    /// Embedded native objects a node set loads that have no hipcc module and
    /// are not accepted `pm_decode` twins (the Raon `gated_delta_vector_pm` twin;
    /// `kernel_registry::native_object_entries`), keyed by image digest.
    pub embedded: Vec<kernel_registry::NativeObject>,
}

/// Resolve every node of every node set on `arch` to its registry entry.
pub fn plan(inventory: &Inventory, arch: &str) -> Result<ArchPlan> {
    let extra_flags = default_extra_flags(arch);
    let all = kernel_registry::corpus_entries(arch, &extra_flags)
        .map_err(|e| format!("{arch}: {e:?}"))?;
    let mut seen = BTreeSet::new();
    for entry in &all {
        if !seen.insert(entry.module) {
            return Err(format!(
                "{arch}: registry module {} appears twice",
                entry.module
            ));
        }
    }
    let natives =
        kernel_registry::native_object_entries(arch).map_err(|e| format!("{arch}: {e:?}"))?;
    let mut needed = BTreeSet::new();
    let mut needed_native = BTreeSet::new();
    let mut errors = Vec::new();
    for set in inventory.node_sets().iter().filter(|s| s.arch == arch) {
        for (symbol, module) in set.nodes {
            match all.iter().find(|e| e.module == module) {
                None => match natives.iter().find(|n| n.module == module) {
                    Some(n) if n.symbols.contains(&symbol.as_str()) => {
                        needed_native.insert(n.module);
                    }
                    Some(_) => errors.push(format!(
                        "{}: node {symbol}: embedded native module {module} does not export it",
                        set.id
                    )),
                    None => errors.push(format!(
                        "{}: node {symbol}: module {module} is not in the {arch} registry",
                        set.id
                    )),
                },
                Some(e) if !e.symbols.contains(&symbol.as_str()) => errors.push(format!(
                    "{}: node {symbol}: registry module {module} does not export it",
                    set.id
                )),
                Some(_) => {
                    needed.insert(module.as_str());
                }
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }
    let entries: Vec<KernelEntry> = all
        .into_iter()
        .filter(|e| needed.contains(e.module))
        .collect();
    let pm_decode = product_flags(arch).pm_decode;
    let mut twins = Vec::new();
    for entry in &entries {
        let Some(twin) = pm_decode_twins::pm_decode_module(arch, pm_decode, entry.module) else {
            continue;
        };
        twin.verify_image().map_err(|e| format!("{arch}: {e}"))?;
        let source_sha256 = sha256_hex(entry.source().as_bytes());
        if hex(&twin.source_sha256) != source_sha256 {
            return Err(format!(
                "{arch}: pm_decode twin {}: registry source sha256 {source_sha256} is not the twin's accepted {}; the runtime refuses to load the twin under that source",
                entry.module, hex(&twin.source_sha256)));
        }
        twins.push(twin);
    }
    let mut embedded = Vec::new();
    for native in natives
        .into_iter()
        .filter(|n| needed_native.contains(n.module))
    {
        let actual = sha256_hex(native.image);
        if actual != native.image_sha256 {
            return Err(format!(
                "{arch}: embedded native module {}: image sha256 {actual} is not the pinned {}",
                native.module, native.image_sha256
            ));
        }
        embedded.push(native);
    }
    Ok(ArchPlan {
        arch: arch.to_owned(),
        extra_flags,
        entries,
        twins,
        embedded,
    })
}

/// The archs the inventory covers, in first-seen order.
pub fn archs(inventory: &Inventory) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for s in inventory.node_sets() {
        if !out.iter().any(|a| a == s.arch) {
            out.push(s.arch.to_owned());
        }
    }
    out
}

/// A program's nodes keyed by the runtime's current JIT inputs.
#[derive(Clone, Debug)]
pub struct ProgramKeys {
    pub id: String,
    pub arch: String,
    /// `(symbol, module, key)`.
    pub nodes: Vec<(String, String, NodeKey)>,
    /// For a recorded sequence ([`SequenceSet`]): the SHA-256 of the object
    /// each node loaded in the recorded run, parallel to `nodes`. The check
    /// then requires the corpus record to be for exactly that object.
    pub loaded_sha256: Option<Vec<String>>,
}

/// How the runtime's current configuration identifies the object a node loads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKey {
    /// Compiled by hipcc: `KernelCompiler::packaging_hash_for` over the registry source and recipe.
    Hipcc(String),
    /// An embedded PeaceMaker twin image: the SHA-256 of the image the runtime loads.
    Native { artifact_sha256: String },
}

/// Key every node set's nodes with the object the runtime loads under the
/// product defaults: the embedded twin's image digest where the module has
/// one, else `KernelCompiler::packaging_hash_for` over the registry's current
/// source and recipe (no compiler needed).
pub fn current_program_keys(inventory: &Inventory) -> Result<Vec<ProgramKeys>> {
    let mut out = Vec::new();
    for arch in archs(inventory) {
        let plan = plan(inventory, &arch)?;
        let mut keys: BTreeMap<&str, NodeKey> = plan
            .entries
            .iter()
            .map(|e| {
                (
                    e.module,
                    match plan.twins.iter().find(|t| t.module == e.module) {
                        Some(twin) => NodeKey::Native {
                            artifact_sha256: hex(&twin.image_sha256),
                        },
                        None => NodeKey::Hipcc(KernelCompiler::packaging_hash_for(
                            &arch,
                            e.module,
                            e.source(),
                            &plan.extra_flags,
                        )),
                    },
                )
            })
            .collect();
        for native in &plan.embedded {
            keys.insert(
                native.module,
                NodeKey::Native {
                    artifact_sha256: native.image_sha256.to_owned(),
                },
            );
        }
        for set in inventory.node_sets().iter().filter(|s| s.arch == arch) {
            let nodes = set
                .nodes
                .iter()
                .map(|(s, m)| (s.clone(), m.clone(), keys[m.as_str()].clone()))
                .collect();
            out.push(ProgramKeys {
                id: set.id.clone(),
                arch: arch.clone(),
                nodes,
                loaded_sha256: set.loaded_sha256.map(<[String]>::to_vec),
            });
        }
    }
    Ok(out)
}

/// Coverage of one program by a corpus.
#[derive(Clone, Debug, Serialize)]
pub struct Coverage {
    pub program: String,
    pub arch: String,
    pub nodes: usize,
    pub covered: usize,
    /// Covered nodes whose A1 verdict is `proven` / `unknown`.
    pub proven: usize,
    pub unknown: usize,
    /// Covered nodes whose A1 verdict is `conditional`: valid only under a
    /// generation-bound witness the runtime must supply before submit. Reported,
    /// never an error here.
    pub conditional: usize,
    /// Every covered node whose verdict is not `proven`, with that verdict.
    pub unproven: Vec<NodeVerdict>,
    pub missing: Vec<MissingNode>,
}

/// A covered node and its A1 verdict (`proven` / `conditional` / `unknown`).
#[derive(Clone, Debug, Serialize)]
pub struct NodeVerdict {
    pub symbol: String,
    pub status: String,
}

/// The CI check: every node must resolve under its current key to a record
/// whose pack object and receipt both verify.
pub fn check(corpus: &JitCorpus, programs: &[ProgramKeys]) -> Vec<Coverage> {
    let pin = corpus.toolchain_pin().to_owned();
    programs.iter().map(|program| {
        let mut missing = Vec::new();
        let mut resolved: Vec<(&str, String, &CorpusRecord)> = Vec::new();
        if let Some(loaded) = &program.loaded_sha256 {
            if loaded.len() != program.nodes.len() {
                missing.push(MissingNode { symbol: String::new(), artifact_sha256: String::new(),
                    reason: format!("program {} has {} nodes but {} recorded object digests", program.id, program.nodes.len(), loaded.len()) });
            }
        }
        for (index, (symbol, module, key)) in program.nodes.iter().enumerate() {
            let found = match key {
                NodeKey::Hipcc(packaging_key) => corpus.record_for_module(&program.arch, module, packaging_key).ok_or_else(|| {
                    match corpus.index().rejected.iter()
                        .find(|r| r.origin == ObjectOrigin::Hipcc && r.arch == program.arch && &r.module == module && &r.packaging_key == packaging_key)
                    {
                        Some(r) => MissingNode { symbol: symbol.clone(), artifact_sha256: r.artifact_sha256.clone(),
                            reason: format!("railgun-cert rejected {module}: {}", r.reason) },
                        None => MissingNode { symbol: symbol.clone(), artifact_sha256: String::new(),
                            reason: format!("no record for module {module} under the runtime's current key {packaging_key}") },
                    }
                }),
                NodeKey::Native { artifact_sha256 } => {
                    let key = ArtifactKey { arch: &program.arch, toolchain_pin: &pin, artifact_sha256 };
                    corpus.record(&key).filter(|r| r.origin == ObjectOrigin::PmNative && &r.module == module).ok_or_else(|| {
                        let reason = match corpus.rejection(&program.arch, artifact_sha256) {
                            Some(r) => format!("railgun-cert rejected pm_native twin {module}: {}", r.reason),
                            None => format!("no pm_native record for twin {module} image {}", &artifact_sha256[..artifact_sha256.len().min(16)]),
                        };
                        MissingNode { symbol: symbol.clone(), artifact_sha256: artifact_sha256.clone(), reason }
                    })
                }
            };
            let record = match found {
                Ok(record) => record,
                Err(node) => {
                    missing.push(node);
                    continue;
                }
            };
            // A recorded sequence names the exact object each node loaded: a record for
            // other bytes of the same module (another toolchain, build path or source) is
            // not a record of what the body runs.
            if let Some(recorded) = program.loaded_sha256.as_ref().and_then(|l| l.get(index)) {
                if *recorded != record.artifact_sha256 {
                    missing.push(MissingNode { symbol: symbol.clone(), artifact_sha256: recorded.clone(),
                        reason: format!("the recorded run loaded object {recorded} for module {module}, but the corpus record under the runtime's current key is object {} (the body would not run the certified bytes)", record.artifact_sha256) });
                    continue;
                }
            }
            let object = corpus.root().join(&record.object);
            match std::fs::read(&object) {
                Ok(bytes) if sha256_hex(&bytes) == record.artifact_sha256 => resolved.push((symbol, record.artifact_sha256.clone(), record)),
                Ok(_) => missing.push(MissingNode { symbol: symbol.clone(), artifact_sha256: record.artifact_sha256.clone(),
                    reason: format!("{} does not hash to the record", object.display()) }),
                Err(e) => missing.push(MissingNode { symbol: symbol.clone(), artifact_sha256: record.artifact_sha256.clone(),
                    reason: format!("{}: {e}", object.display()) }),
            }
        }
        let nodes: Vec<ProgramNode> = resolved.iter().map(|(s, sha, _)| ProgramNode { symbol: s, artifact_sha256: sha }).collect();
        if let Err(e) = corpus.check_program(&program.id, &program.arch, &pin, &nodes) {
            missing.extend(e.missing);
        }
        let failed: BTreeSet<&str> = missing.iter().map(|m| m.symbol.as_str()).collect();
        let verdict = |s: &str, r: &CorpusRecord| r.kernels.iter().find(|k| k.symbol == s).map(|k| k.status.clone()).unwrap_or_else(|| "no-verdict".to_owned());
        let ok: Vec<(String, String)> = resolved.iter().filter(|(s, ..)| !failed.contains(s)).map(|(s, _, r)| ((*s).to_owned(), verdict(s, r))).collect();
        let count = |status: &str| ok.iter().filter(|(_, v)| v == status).count();
        Coverage {
            program: program.id.clone(), arch: program.arch.clone(), nodes: program.nodes.len(), covered: ok.len(),
            proven: count(railgun_corpus::STATUS_PROVEN), unknown: count(railgun_corpus::STATUS_UNKNOWN), conditional: count(railgun_corpus::STATUS_CONDITIONAL),
            unproven: ok.iter().filter(|(_, v)| v != railgun_corpus::STATUS_PROVEN).map(|(s, v)| NodeVerdict { symbol: s.clone(), status: v.clone() }).collect(),
            missing,
        }
    }).collect()
}

// ---------------------------------------------------------------- build

pub struct BuildOptions {
    /// Corpus base; the corpus lands in `<out>/v<version>/<host key>`.
    pub out: PathBuf,
    /// Fixed compile directory. hipcc names the `__hip_cuid_*` marker after
    /// the input path and argv, so a fixed directory repeats the bytes.
    pub build_dir: PathBuf,
    pub archs: Vec<String>,
    pub jobs: usize,
}

/// An embedded native code object the corpus records as `pm_native`: an accepted
/// `pm_decode` twin (with its incumbent source digest and Radiowave manifest) or
/// another sealed embedded image (no HIP source, no manifest).
struct NativeImage {
    module: &'static str,
    image: &'static [u8],
    image_sha256: String,
    /// SHA-256 of the incumbent HIP source; empty for a sealed image without one.
    source_sha256: String,
    radiowave_json: Option<&'static str>,
    symbols: &'static [&'static str],
}

struct Compiled {
    object: Vec<u8>,
    /// Err: the certifier rejected the object (it ships without a record).
    receipt: Result<railgun_cert::Receipt>,
    /// The Radiowave manifest the runtime reads beside a packaged object.
    radiowave_json: String,
    headers: BTreeSet<String>,
}

/// `f` over `items` on `jobs` threads, results in item order.
fn run_parallel<T: Sync, R: Send>(items: &[T], jobs: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..jobs.clamp(1, items.len().max(1)))
            .map(|_| {
                scope.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        local.push((i, f(item)));
                    }
                    local
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("worker panicked"))
            .collect()
    });
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, r)| r).collect()
}

fn run(program: &str, args: &[String], rocm_path: Option<&Path>) -> Result<std::process::Output> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(root) = rocm_path {
        cmd.env("ROCM_PATH", root);
    }
    cmd.output().map_err(|e| format!("{program}: {e}"))
}

/// Headers the device preprocess of `argv` reads (`-E -H`).
fn headers(argv: &[String], rocm_path: Option<&Path>) -> Result<BTreeSet<String>> {
    let mut args: Vec<String> = Vec::new();
    let mut it = argv[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--genco" => args.push("--cuda-device-only".into()),
            "-o" => {
                it.next();
                args.extend(["-o".into(), "/dev/null".into()]);
            }
            _ => args.push(a.clone()),
        }
    }
    args.extend(["-E".into(), "-H".into()]);
    let out = run(&argv[0], &args, rocm_path)?;
    if !out.status.success() {
        return Err(format!(
            "preprocess failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stderr)
        .lines()
        .filter_map(|l| {
            l.trim_start_matches('.')
                .strip_prefix(' ')
                .filter(|_| l.starts_with('.'))
        })
        .map(|p| {
            std::fs::canonicalize(p)
                .map(|c| c.display().to_string())
                .unwrap_or_else(|_| p.to_owned())
        })
        .collect())
}

fn compile_and_certify(
    compiler: &KernelCompiler,
    entry: &KernelEntry,
    dir: &Path,
) -> Result<Compiled> {
    let src = dir.join(format!("{}.hip", entry.module));
    let obj = dir.join(format!("{}.hsaco", entry.module));
    let _ = std::fs::remove_file(&obj);
    compiler
        .compile_to_paths(entry.module, entry.source(), &src, &obj)
        .map_err(|e| format!("{}: {e}", entry.module))?;
    let object = std::fs::read(&obj).map_err(|e| format!("{}: {e}", obj.display()))?;
    // Without the manifest Redline types the launch with no resource effects and
    // keeps a wait + acquire railgun's certified facts prove unneeded.
    let manifest = dir.join(format!("{}.radiowave.json", entry.module));
    let _ = std::fs::remove_file(&manifest);
    compiler
        .certify_radiowave(entry.module, &src, &obj, &manifest)
        .map_err(|e| format!("{}: radiowave: {e}", entry.module))?;
    let radiowave_json =
        std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let receipt = railgun_cert::certify(&object, Some(&radiowave_json)).map_err(|e| e.to_string());
    if let Ok(receipt) = &receipt {
        let missing: Vec<&str> = entry
            .symbols
            .iter()
            .copied()
            .filter(|s| !receipt.railgun.kernels.iter().any(|k| k.symbol == *s))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{}: object does not export registry symbol(s) {}",
                entry.module,
                missing.join(", ")
            ));
        }
    }
    let headers = headers(
        &compiler.jit_argv(entry.module, entry.source(), &src, &obj),
        compiler.rocm_env_root(),
    )?;
    Ok(Compiled {
        object,
        receipt,
        radiowave_json,
        headers,
    })
}

fn git_revision() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let head = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // The receipts depend on the certifier (railgun-cert over peacemaker's lift and ISA tables).
    let dirty = git(&[
        "status",
        "--porcelain",
        "--",
        "../../kernels",
        "../rdna-compute",
        "../railgun-cert",
        "../railgun-corpus",
        "../peacemaker-ir",
        "../peacemaker-lift",
        ".",
    ])
    .is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{head}-dirty")
    } else {
        head
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes)
}

/// Compile, certify and index every module the inventory's node sets use
/// on `opts.archs`, plus the embedded pm_decode twin images the product
/// default loads in their place. Returns the corpus directory.
pub fn build(inventory: &Inventory, opts: &BuildOptions) -> Result<PathBuf> {
    install_product_defaults()?;
    let build_dir = std::fs::canonicalize(&opts.build_dir)
        .or_else(|_| {
            std::fs::create_dir_all(&opts.build_dir)
                .and_then(|_| std::fs::canonicalize(&opts.build_dir))
        })
        .map_err(|e| format!("{}: {e}", opts.build_dir.display()))?;
    let versioned = opts.out.join(format!("v{CORPUS_VERSION}"));
    let staging = versioned.join(format!(".staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    let mut records = Vec::new();
    let mut rejected = Vec::new();
    let mut all_headers = BTreeSet::new();
    let mut toolchain: Option<(String, Option<PathBuf>, String)> = None;
    for arch in &opts.archs {
        let plan = plan(inventory, arch)?;
        let compiler = KernelCompiler::new(arch, plan.extra_flags.clone())
            .map_err(|e| format!("{arch}: {e}"))?;
        let id = compiler.toolchain_id().to_owned();
        if id.is_empty() {
            return Err(format!(
                "{arch}: no hipcc; the corpus is built with the pinned toolchain"
            ));
        }
        let hipcc = compiler.jit_argv(
            "probe",
            "",
            Path::new("probe.hip"),
            Path::new("probe.hsaco"),
        )[0]
        .clone();
        match &toolchain {
            Some((t, _, _)) if *t != id => {
                return Err(format!("{arch}: toolchain {id} differs from {t}"))
            }
            Some(_) => {}
            None => toolchain = Some((id, compiler.rocm_env_root().map(Path::to_path_buf), hipcc)),
        }
        let dir = build_dir.join(arch);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        eprintln!(
            "{arch}: {} modules, extra flags `{}`",
            plan.entries.len(),
            plan.extra_flags
        );
        let results = run_parallel(&plan.entries, opts.jobs, |entry| {
            compile_and_certify(&compiler, entry, &dir)
        });
        let failures: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        if !failures.is_empty() {
            return Err(failures
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n"));
        }
        for (entry, compiled) in plan
            .entries
            .iter()
            .zip(results.into_iter().map(Result::unwrap))
        {
            let artifact = sha256_hex(&compiled.object);
            let symbols: Vec<String> = entry.symbols.iter().map(|s| (*s).to_owned()).collect();
            let object_src = dir.join(format!("{}.hsaco", entry.module));
            let pack_dir = staging.join("pack").join(arch);
            compiler.publish_package(
                entry.module,
                entry.source(),
                &symbols,
                &object_src,
                &pack_dir,
            )?;
            std::fs::write(
                pack_dir.join(format!("{}.radiowave.json", entry.module)),
                &compiled.radiowave_json,
            )
            .map_err(|e| e.to_string())?;
            let source_rel = staging
                .join("sources")
                .join(arch)
                .join(format!("{}.hip", entry.module));
            std::fs::create_dir_all(source_rel.parent().unwrap()).map_err(|e| e.to_string())?;
            std::fs::write(&source_rel, entry.source()).map_err(|e| e.to_string())?;
            all_headers.extend(compiled.headers);
            let receipt = match compiled.receipt {
                Ok(receipt) => receipt,
                Err(reason) => {
                    eprintln!(
                        "{arch}: {} rejected by railgun-cert: {reason}",
                        entry.module
                    );
                    rejected.push(RejectedObject {
                        arch: arch.clone(),
                        artifact_sha256: artifact,
                        module: entry.module.to_owned(),
                        origin: ObjectOrigin::Hipcc,
                        symbols,
                        source_sha256: sha256_hex(entry.source().as_bytes()),
                        packaging_key: compiler.packaging_hash(entry.module, entry.source()),
                        object: format!("pack/{arch}/{}.hsaco", entry.module),
                        reason,
                    });
                    continue;
                }
            };
            let receipt_rel = format!("receipts/{arch}/{artifact}.receipt.json");
            let receipt_bytes = write_json(&staging.join(&receipt_rel), &receipt)?;
            records.push(CorpusRecord {
                arch: arch.clone(),
                artifact_sha256: artifact,
                module: entry.module.to_owned(),
                origin: ObjectOrigin::Hipcc,
                symbols,
                source_sha256: sha256_hex(entry.source().as_bytes()),
                extra_flags: plan.extra_flags.clone(),
                flags: entry.flags.clone(),
                scheduler_profile: entry.scheduler_profile.clone().unwrap_or_default(),
                packaging_key: compiler.packaging_hash(entry.module, entry.source()),
                jit_cache_key: compiler.jit_cache_key(entry.module, entry.source()),
                argv: compiler.jit_argv(
                    entry.module,
                    entry.source(),
                    &dir.join(format!("{}.hip", entry.module)),
                    &object_src,
                ),
                object: format!("pack/{arch}/{}.hsaco", entry.module),
                receipt: receipt_rel,
                receipt_sha256: sha256_hex(&receipt_bytes),
                kernels: receipt
                    .railgun
                    .kernels
                    .iter()
                    .map(|k| KernelVerdict {
                        symbol: k.symbol.clone(),
                        status: k.status.clone(),
                        differential: k.differential.status.clone(),
                    })
                    .collect(),
            });
        }
        let natives: Vec<NativeImage> = plan
            .twins
            .iter()
            .map(|t| NativeImage {
                module: t.module,
                image: t.image,
                image_sha256: hex(&t.image_sha256),
                source_sha256: hex(&t.source_sha256),
                radiowave_json: Some(t.radiowave_json),
                symbols: t.symbols,
            })
            .chain(plan.embedded.iter().map(|n| NativeImage {
                module: n.module,
                image: n.image,
                image_sha256: n.image_sha256.to_owned(),
                source_sha256: String::new(),
                radiowave_json: None,
                symbols: n.symbols,
            }))
            .collect();
        if !natives.is_empty() {
            eprintln!(
                "{arch}: {} embedded native object(s) ({} pm_decode twin(s))",
                natives.len(),
                plan.twins.len()
            );
        }
        let certified = run_parallel(&natives, opts.jobs, |n| {
            railgun_cert::certify(n.image, n.radiowave_json).map_err(|e| e.to_string())
        });
        for (twin, receipt) in natives.iter().zip(certified) {
            let artifact = twin.image_sha256.clone();
            let symbols: Vec<String> = twin.symbols.iter().map(|s| (*s).to_owned()).collect();
            let object_rel = format!("native/{arch}/{}.hxaco", twin.module);
            let radiowave_rel = format!("native/{arch}/{}.radiowave.json", twin.module);
            let write_native = |rel: &str, bytes: &[u8]| -> Result<()> {
                let path = staging.join(rel);
                std::fs::create_dir_all(path.parent().unwrap())
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
            };
            write_native(&object_rel, twin.image)?;
            if let Some(json) = twin.radiowave_json {
                write_native(&radiowave_rel, json.as_bytes())?;
            }
            let receipt = match receipt {
                Ok(receipt) => receipt,
                Err(reason) => {
                    eprintln!(
                        "{arch}: pm_native object {} rejected by railgun-cert: {reason}",
                        twin.module
                    );
                    rejected.push(RejectedObject {
                        arch: arch.clone(),
                        artifact_sha256: artifact,
                        module: twin.module.to_owned(),
                        origin: ObjectOrigin::PmNative,
                        symbols,
                        source_sha256: twin.source_sha256.clone(),
                        packaging_key: String::new(),
                        object: object_rel,
                        reason,
                    });
                    continue;
                }
            };
            let missing: Vec<&str> = twin
                .symbols
                .iter()
                .copied()
                .filter(|s| !receipt.railgun.kernels.iter().any(|k| k.symbol == *s))
                .collect();
            if !missing.is_empty() {
                return Err(format!(
                    "{arch}: pm_native object {}: receipt has no kernel for symbol(s) {}",
                    twin.module,
                    missing.join(", ")
                ));
            }
            let receipt_rel = format!("receipts/{arch}/{artifact}.receipt.json");
            let receipt_bytes = write_json(&staging.join(&receipt_rel), &receipt)?;
            records.push(CorpusRecord {
                arch: arch.clone(),
                artifact_sha256: artifact,
                module: twin.module.to_owned(),
                origin: ObjectOrigin::PmNative,
                symbols,
                source_sha256: twin.source_sha256.clone(),
                extra_flags: String::new(),
                flags: Vec::new(),
                scheduler_profile: String::new(),
                packaging_key: String::new(),
                jit_cache_key: String::new(),
                argv: Vec::new(),
                object: object_rel,
                receipt: receipt_rel,
                receipt_sha256: sha256_hex(&receipt_bytes),
                kernels: receipt
                    .railgun
                    .kernels
                    .iter()
                    .map(|k| KernelVerdict {
                        symbol: k.symbol.clone(),
                        status: k.status.clone(),
                        differential: k.differential.status.clone(),
                    })
                    .collect(),
            });
        }
    }
    let (id, rocm_env_root, hipcc) = toolchain.ok_or("no arch selected")?;
    let version = run(&hipcc, &["--version".into()], rocm_env_root.as_deref())?;
    let clang_root = rocm_env_root
        .clone()
        .or_else(hipfire_config::rocm::root)
        .ok_or("no ROCm root")?;
    let clang = std::fs::canonicalize(clang_root.join("lib/llvm/bin/clang"))
        .map_err(|e| format!("clang: {e}"))?;
    let clang_sha256 =
        sha256_hex(&std::fs::read(&clang).map_err(|e| format!("{}: {e}", clang.display()))?);
    let mut header_lines = String::new();
    for h in &all_headers {
        let bytes = std::fs::read(h).map_err(|e| format!("{h}: {e}"))?;
        header_lines.push_str(&format!("{h}\t{}\n", sha256_hex(&bytes)));
    }
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_default();
    let toolchain = ToolchainPin {
        pin: sha256_hex(id.as_bytes()),
        id,
        version_sha256: sha256_hex(&version.stdout),
    };
    let mut build_host = BuildHost {
        key: String::new(),
        hostname,
        hipcc,
        rocm_env_root: rocm_env_root.map(|p| p.display().to_string()),
        clang_sha256,
        header_set_sha256: sha256_hex(header_lines.as_bytes()),
        build_dir: build_dir.display().to_string(),
    };
    build_host.key = sha256_hex(
        &serde_json::to_vec(&(&build_host, &toolchain.version_sha256))
            .map_err(|e| e.to_string())?,
    );
    std::fs::write(staging.join("headers.tsv"), &header_lines).map_err(|e| e.to_string())?;

    let programs = inventory
        .node_sets()
        .iter()
        .filter(|s| opts.archs.iter().any(|a| a == s.arch))
        .map(|p| ProgramRecord {
            id: p.id.clone(),
            arch: p.arch.to_owned(),
            nodes: p
                .nodes
                .iter()
                .map(|(symbol, module)| {
                    // The product default loads the twin where the module has one, else the hipcc object.
                    let of = |origin: ObjectOrigin| {
                        records
                            .iter()
                            .find(|r| r.origin == origin && r.arch == p.arch && &r.module == module)
                            .map(|r| &r.artifact_sha256)
                            .or_else(|| {
                                rejected
                                    .iter()
                                    .find(|r| {
                                        r.origin == origin
                                            && r.arch == p.arch
                                            && &r.module == module
                                    })
                                    .map(|r| &r.artifact_sha256)
                            })
                    };
                    NodeRecord {
                        symbol: symbol.clone(),
                        module: module.clone(),
                        artifact_sha256: of(ObjectOrigin::PmNative)
                            .or_else(|| of(ObjectOrigin::Hipcc))
                            .expect("planned module built")
                            .clone(),
                    }
                })
                .collect(),
        })
        .collect();
    let index = CorpusIndex {
        schema: CORPUS_SCHEMA.into(),
        version: CORPUS_VERSION,
        source_commit: git_revision(),
        build_host,
        toolchain,
        records,
        rejected,
        programs,
    };
    write_json(&staging.join(CORPUS_INDEX), &index)?;
    let dest = versioned.join(&index.build_host.key[..16]);
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&staging, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    JitCorpus::open(&dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

// ---------------------------------------------------------------- install

/// What the runtime loads for one module of an installed pack.
#[derive(Clone, Debug, Serialize)]
pub struct InstallVerdict {
    pub arch: String,
    pub module: String,
    /// `pack` (the object file the runtime verified and loads),
    /// `embedded` (a sealed image compiled into the binary), or `elsewhere`
    /// (the runtime would JIT-compile or take a hot-cache object instead).
    pub source: String,
    /// SHA-256 of the object the runtime would load, when known.
    pub loaded_sha256: Option<String>,
    /// SHA-256 of the corpus record of that module under the runtime's current key.
    pub corpus_sha256: Option<String>,
    pub ok: bool,
    pub detail: String,
}

/// Load every hipcc module of `inventory` on `archs` through the runtime's own
/// loader (`KernelCompiler::compile_for_symbol`) with the current directory set to
/// `install_root` (the runtime probes `kernels/compiled/<arch>` under its CWD) and
/// report, per module, whether the object it resolves to is a file of that pack
/// whose SHA-256 is the corpus record's. Embedded images are checked against the
/// corpus' `pm_native` record by digest. The hot JIT cache is whatever
/// `HIPFIRE_KERNEL_CACHE` names; the loader tries the verified pack first, so a hot
/// object can never mask a pack miss here: it is reported as `elsewhere`.
/// Changes the process working directory.
pub fn verify_install(
    inventory: &Inventory,
    corpus: &JitCorpus,
    install_root: &Path,
    archs: &[String],
) -> Result<Vec<InstallVerdict>> {
    install_product_defaults()?;
    let root = std::fs::canonicalize(install_root)
        .map_err(|e| format!("{}: {e}", install_root.display()))?;
    std::env::set_current_dir(&root).map_err(|e| format!("{}: {e}", root.display()))?;
    let pin = corpus.toolchain_pin().to_owned();
    let mut out = Vec::new();
    for arch in archs {
        let plan = plan(inventory, arch)?;
        let pack = root.join("kernels").join("compiled").join(arch);
        let mut compiler = KernelCompiler::new(arch, plan.extra_flags.clone())
            .map_err(|e| format!("{arch}: {e}"))?;
        // The owner opens the corpus only under the toolchain it was built with.
        let now = sha256_hex(compiler.toolchain_id().as_bytes());
        out.push(InstallVerdict {
            arch: arch.clone(),
            module: "<toolchain pin>".to_owned(),
            source: "toolchain".into(),
            loaded_sha256: Some(now.clone()),
            corpus_sha256: Some(pin.clone()),
            ok: now == pin,
            detail: if now == pin {
                String::new()
            } else {
                format!("the runtime compiles with toolchain `{}` (pin {now}), the corpus pin is {pin}: the owner would refuse the corpus", compiler.toolchain_id())
            },
        });
        for entry in &plan.entries {
            let verdict = |source: &str,
                           loaded: Option<String>,
                           corpus_sha: Option<String>,
                           ok: bool,
                           detail: String| InstallVerdict {
                arch: arch.clone(),
                module: entry.module.to_owned(),
                source: source.to_owned(),
                loaded_sha256: loaded,
                corpus_sha256: corpus_sha,
                ok,
                detail,
            };
            if let Some(twin) = plan.twins.iter().find(|t| t.module == entry.module) {
                let sha = hex(&twin.image_sha256);
                let key = ArtifactKey {
                    arch,
                    toolchain_pin: &pin,
                    artifact_sha256: &sha,
                };
                let found = corpus
                    .record(&key)
                    .is_some_and(|r| r.origin == ObjectOrigin::PmNative);
                out.push(verdict(
                    "embedded",
                    Some(sha.clone()),
                    found.then(|| sha.clone()),
                    found,
                    if found {
                        String::new()
                    } else {
                        "no pm_native record for the embedded twin".into()
                    },
                ));
                continue;
            }
            let packaging_key = KernelCompiler::packaging_hash_for(
                arch,
                entry.module,
                entry.source(),
                &plan.extra_flags,
            );
            let record = corpus.record_for_module(arch, entry.module, &packaging_key);
            let corpus_sha = record.map(|r| r.artifact_sha256.clone());
            let symbol = entry.symbols.first().copied().unwrap_or(entry.module);
            let path = match compiler.compile_for_symbol(entry.module, entry.source(), symbol) {
                Ok(p) => p.to_path_buf(),
                Err(e) => {
                    out.push(verdict(
                        "elsewhere",
                        None,
                        corpus_sha,
                        false,
                        format!("the runtime loader failed: {e}"),
                    ));
                    continue;
                }
            };
            let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
            let in_pack = canonical.parent() == std::fs::canonicalize(&pack).ok().as_deref();
            let loaded = std::fs::read(&canonical).ok().map(|b| sha256_hex(&b));
            let (ok, detail) = match (&record, in_pack, &loaded) {
                (None, ..) => (false, format!("no corpus record under the runtime's current key {packaging_key}")),
                (Some(_), false, _) => (false, format!("the runtime loads {} (not a file of {}): it would not run the corpus bytes", canonical.display(), pack.display())),
                (Some(_), true, None) => (false, format!("{} is unreadable", canonical.display())),
                (Some(r), true, Some(l)) if *l != r.artifact_sha256 => (false, format!("the pack object {} has sha256 {l}, the corpus record is {}", canonical.display(), r.artifact_sha256)),
                (Some(_), true, Some(_)) => (true, String::new()),
            };
            out.push(verdict(
                if in_pack { "pack" } else { "elsewhere" },
                loaded,
                corpus_sha,
                ok,
                detail,
            ));
        }
        // Embedded images that are not accepted twins (never loaded from a pack).
        for native in &plan.embedded {
            let key = ArtifactKey {
                arch,
                toolchain_pin: &pin,
                artifact_sha256: native.image_sha256,
            };
            let found = corpus
                .record(&key)
                .is_some_and(|r| r.origin == ObjectOrigin::PmNative);
            out.push(InstallVerdict {
                arch: arch.clone(),
                module: native.module.to_owned(),
                source: "embedded".into(),
                loaded_sha256: Some(native.image_sha256.to_owned()),
                corpus_sha256: found.then(|| native.image_sha256.to_owned()),
                ok: found,
                detail: if found {
                    String::new()
                } else {
                    "no pm_native record for the embedded image".into()
                },
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- repro

/// One module's reproducibility result. `raw` compares object bytes;
/// `code` compares every kernel's text, descriptor and metadata digests.
#[derive(Clone, Debug, Serialize)]
pub struct Repro {
    pub arch: String,
    pub module: String,
    /// Recompiled at the corpus build paths: identical to the corpus object.
    pub rebuild_raw_equal: bool,
    /// Two compiles through runtime-style unique temporary object paths.
    pub jit_raw_equal: bool,
    pub jit_code_equal: bool,
    /// The same compile with an aliased include root before vs after `-I<root>/include`.
    pub include_order_raw_equal: bool,
    pub include_order_code_equal: bool,
    /// Runtime-style unique paths again, with `-fuse-cuid=none`.
    pub nocuid_raw_equal: bool,
    pub error: Option<String>,
}

fn code_equal(a: &[u8], b: &[u8]) -> Result<bool> {
    let id = |bytes: &[u8]| {
        railgun_cert::code_identity(bytes)
            .map_err(|e| e.to_string())
            .map(|v| {
                v.into_iter()
                    .map(|k| (k.symbol, k.text_sha256, k.kd_sha256, k.metadata_sha256))
                    .collect::<Vec<_>>()
            })
    };
    Ok(id(a)? == id(b)?)
}

fn compile_argv(argv: &[String], rocm: Option<&Path>, out: &Path) -> Result<Vec<u8>> {
    let _ = std::fs::remove_file(out);
    let o = run(&argv[0], &argv[1..], rocm)?;
    if !o.status.success() {
        return Err(format!(
            "hipcc failed: {}",
            String::from_utf8_lossy(&o.stderr)
        ));
    }
    std::fs::read(out).map_err(|e| format!("{}: {e}", out.display()))
}

fn with_output(argv: &[String], out: &Path) -> Vec<String> {
    let mut v = argv.to_vec();
    let i = v.iter().position(|a| a == "-o").expect("argv has -o");
    v[i + 1] = out.display().to_string();
    v
}

fn repro_one(
    corpus: &JitCorpus,
    record: &CorpusRecord,
    compiler: &KernelCompiler,
    work: &Path,
    alias: &str,
) -> Result<Repro> {
    let source = std::fs::read_to_string(
        corpus
            .root()
            .join(format!("sources/{}/{}.hip", record.arch, record.module)),
    )
    .map_err(|e| e.to_string())?;
    if sha256_hex(source.as_bytes()) != record.source_sha256 {
        return Err("corpus source does not match its record".into());
    }
    let rocm = compiler.rocm_env_root();
    let module = record.module.as_str();
    let build = Path::new(&corpus.index().build_host.build_dir).join(&record.arch);
    // 1. The corpus build paths again.
    let (src, obj) = (
        build.join(format!("{module}.hip")),
        build.join(format!("{module}.hsaco")),
    );
    compiler
        .compile_to_paths(module, &source, &src, &obj)
        .map_err(|e| e.to_string())?;
    let rebuild = std::fs::read(&obj).map_err(|e| e.to_string())?;
    // 2. Runtime-style: `{stem}.hip` in the cache, unique temporary objects.
    let dir = work.join(&record.arch);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stem = format!("{module}.{}", record.jit_cache_key);
    let jit_src = dir.join(format!("{stem}.hip"));
    std::fs::write(&jit_src, &source).map_err(|e| e.to_string())?;
    let temp = |tag: &str| dir.join(format!(".{stem}.{}_{tag}.hsaco.tmp", std::process::id()));
    let base = compiler.jit_argv(module, &source, &jit_src, &temp("a"));
    let jit_a = compile_argv(&base, rocm, &temp("a"))?;
    let jit_b = compile_argv(&with_output(&base, &temp("b")), rocm, &temp("b"))?;
    // 3. Include order: an alias of the include root before, then after it.
    let root_i = base
        .iter()
        .position(|a| a.starts_with("-I"))
        .ok_or("argv has no -I")?;
    let inc_out = dir.join(format!("{module}.include.hsaco"));
    let mut fwd = with_output(&base, &inc_out);
    fwd.insert(root_i, format!("-I{alias}"));
    let mut rev = with_output(&base, &inc_out);
    rev.insert(root_i + 1, format!("-I{alias}"));
    let inc_fwd = compile_argv(&fwd, rocm, &inc_out)?;
    let inc_rev = compile_argv(&rev, rocm, &inc_out)?;
    // 4. Unique paths with the cuid marker disabled.
    let nocuid = |tag: &str| {
        let mut v = with_output(&base, &temp(tag));
        v.insert(1, "-fuse-cuid=none".into());
        compile_argv(&v, rocm, &temp(tag))
    };
    let (nc_a, nc_b) = (nocuid("na")?, nocuid("nb")?);
    for tag in ["a", "b", "na", "nb"] {
        let _ = std::fs::remove_file(temp(tag));
    }
    Ok(Repro {
        arch: record.arch.clone(),
        module: module.to_owned(),
        rebuild_raw_equal: sha256_hex(&rebuild) == record.artifact_sha256,
        jit_raw_equal: jit_a == jit_b,
        jit_code_equal: code_equal(&jit_a, &jit_b)?,
        include_order_raw_equal: inc_fwd == inc_rev,
        include_order_code_equal: code_equal(&inc_fwd, &inc_rev)?,
        nocuid_raw_equal: nc_a == nc_b,
        error: None,
    })
}

/// Recompile every corpus module in the variants [`Repro`] names.
pub fn repro(corpus: &JitCorpus, work: &Path, archs: &[String], jobs: usize) -> Result<Vec<Repro>> {
    install_product_defaults()?;
    std::fs::create_dir_all(work).map_err(|e| format!("{}: {e}", work.display()))?;
    let mut out = Vec::new();
    for arch in archs {
        let records: Vec<&CorpusRecord> = corpus
            .index()
            .records
            .iter()
            .filter(|r| r.origin == ObjectOrigin::Hipcc && &r.arch == arch)
            .collect();
        let Some(first) = records.first() else {
            continue;
        };
        let compiler =
            KernelCompiler::new(arch, first.extra_flags.clone()).map_err(|e| e.to_string())?;
        if sha256_hex(compiler.toolchain_id().as_bytes()) != corpus.toolchain_pin() {
            return Err(format!("{arch}: local toolchain is not the corpus pin"));
        }
        let root_include = first
            .argv
            .iter()
            .find_map(|a| a.strip_prefix("-I"))
            .ok_or("record argv has no -I")?;
        let alias = std::fs::canonicalize(root_include)
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let alias = if alias.is_empty() || alias == root_include {
            format!("{root_include}/.")
        } else {
            alias
        };
        out.extend(run_parallel(&records, jobs, |r| {
            repro_one(corpus, r, &compiler, work, &alias).unwrap_or_else(|e| Repro {
                arch: r.arch.clone(),
                module: r.module.clone(),
                rebuild_raw_equal: false,
                jit_raw_equal: false,
                jit_code_equal: false,
                include_order_raw_equal: false,
                include_order_code_equal: false,
                nocuid_raw_equal: false,
                error: Some(e),
            })
        }));
    }
    Ok(out)
}

// ---------------------------------------------------------------- runtime cache

/// One launch of a runtime capture checked against the registry: the
/// runtime's own JIT cache entry `{module}.{key}.hip` is the generated source
/// it compiled, and `key` is its cache key over that source and the recipe.
#[derive(Clone, Debug, Serialize)]
pub struct CacheVerdict {
    pub symbol: String,
    pub module: String,
    pub runtime_key: String,
    /// The registry's source is byte-identical to the runtime's.
    pub source_equal: bool,
    /// `jit_cache_key` over the registry source and recipe equals the runtime's key.
    pub key_equal: bool,
}

/// Verify a Redline capture whose artifacts are content-keyed JIT cache
/// entries against the registry and the runtime's cache-key function.
pub fn verify_runtime_cache(capture: &Path, arch: &str) -> Result<Vec<CacheVerdict>> {
    install_product_defaults()?;
    let text =
        std::fs::read_to_string(capture).map_err(|e| format!("{}: {e}", capture.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    fn find(v: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
        match v {
            serde_json::Value::Object(m) if m.contains_key("sequence_hash") => {
                m.get("sequence").and_then(|s| s.as_array())
            }
            serde_json::Value::Object(m) => m.values().find_map(find),
            serde_json::Value::Array(a) => a.iter().find_map(find),
            _ => None,
        }
    }
    let sequence = find(&value).ok_or("capture has no redline sequence")?;
    let mut launches = BTreeMap::new();
    for l in sequence {
        let (Some(k), Some(a)) = (l["kernel"].as_str(), l["artifact"].as_str()) else {
            continue;
        };
        launches.insert(k.to_owned(), PathBuf::from(a));
    }
    let extra = default_extra_flags(arch);
    let compiler = KernelCompiler::new(arch, extra.clone()).map_err(|e| e.to_string())?;
    let registry = kernel_registry::corpus_entries(arch, &extra).map_err(|e| format!("{e:?}"))?;
    let mut out = Vec::new();
    for (symbol, artifact) in launches {
        let name = artifact
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let Some((module, key)) = name.strip_suffix(".hsaco").and_then(|s| s.rsplit_once('.'))
        else {
            return Err(format!(
                "{symbol}: {name} is not a content-keyed cache entry"
            ));
        };
        let runtime_source = std::fs::read(artifact.with_extension("hip"))
            .map_err(|e| format!("{}: {e}", artifact.display()))?;
        let entry = registry.iter().find(|e| e.module == module);
        out.push(CacheVerdict {
            symbol,
            module: module.to_owned(),
            runtime_key: key.to_owned(),
            source_equal: entry.is_some_and(|e| e.source().as_bytes() == runtime_source.as_slice()),
            key_equal: entry.is_some_and(|e| compiler.jit_cache_key(module, e.source()) == key),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_sequence_adds_a_node_set_the_plan_resolves() {
        install_product_defaults().unwrap();
        let mut inv = inventory().unwrap();
        let jsonl = concat!(
            "{\"phase\":\"transport\",\"symbol\":\"railgun_copy\",\"route\":\"railgun\",\"module\":\"railgun_copy\"}\n",
            "{\"phase\":\"transport\",\"symbol\":\"railgun_fill_u32\",\"route\":\"railgun\",\"module\":\"railgun_fill\"}\n",
            "{\"phase\":\"selector\",\"symbol\":\"dflash_selector_greedy_exact\",\"route\":\"railgun\",\"module\":\"dflash_selector_greedy_exact\"}\n",
            "{\"phase\":\"transport\",\"symbol\":\"railgun_copy\",\"route\":\"railgun\",\"module\":\"railgun_copy\"}\n",
        );
        inv.add_window_sequence("dflash-whole16-gfx1201", "gfx1201", jsonl)
            .unwrap();
        let set = inv
            .node_sets()
            .into_iter()
            .find(|s| s.id == "dflash-whole16-gfx1201")
            .expect("window set");
        assert_eq!(set.nodes.len(), 3, "duplicates collapse");
        let resolved = plan(&inv, "gfx1201").unwrap_or_else(|e| panic!("{e}"));
        for m in [
            "railgun_copy",
            "railgun_fill",
            "dflash_selector_greedy_exact",
        ] {
            assert!(resolved.entries.iter().any(|e| e.module == m), "{m}");
        }
        // A module the registry lacks is named, not skipped.
        let mut bad = inventory().unwrap();
        bad.add_window_sequence(
            "w",
            "gfx1201",
            "{\"symbol\":\"nope\",\"module\":\"no_such_module\"}\n",
        )
        .unwrap();
        let err = plan(&bad, "gfx1201").err().expect("unresolved module");
        assert!(err.contains("no_such_module"), "{err}");
        assert!(inv
            .add_window_sequence("w", "gfx1201", "{\"symbol\":\"x\"}\n")
            .is_err());
    }

    #[test]
    fn every_admitted_program_node_resolves_to_a_registry_module_that_exports_it() {
        install_product_defaults().unwrap();
        let inventory = inventory().unwrap();
        for arch in archs(&inventory) {
            let plan = plan(&inventory, &arch).unwrap_or_else(|e| panic!("{e}"));
            assert!(!plan.entries.is_empty(), "{arch}");
        }
    }

    const GFX1201_TWIN_MODULES: [&str; 5] = [
        "fused_qkvza_hfq4g256_mq4v2",
        "fused_qkv_hfq4g256_mq4v2",
        "fused_gate_up_hfq4g256_mq4v2",
        "gemv_hfq4g256_multirow_default_mq4v2",
        "gemv_hfq4g256_residual_mq4v2",
    ];

    #[test]
    fn gfx1201_product_defaults_plan_the_five_pm_decode_twins_and_key_their_nodes_by_image_digest()
    {
        install_product_defaults().unwrap();
        let inventory = inventory().unwrap();
        let plan = plan(&inventory, "gfx1201").unwrap_or_else(|e| panic!("{e}"));
        let mut modules: Vec<&str> = plan.twins.iter().map(|t| t.module).collect();
        modules.sort_unstable();
        let mut expected = GFX1201_TWIN_MODULES.to_vec();
        expected.sort_unstable();
        assert_eq!(modules, expected);

        let programs = current_program_keys(&inventory).unwrap();
        let program = programs
            .iter()
            .find(|p| p.id == "qwen35-dense-h2-gfx1201")
            .expect("program qwen35-dense-h2-gfx1201");
        let mut native: Vec<&str> = Vec::new();
        for (_, module, key) in &program.nodes {
            match plan.twins.iter().find(|t| t.module == module) {
                Some(twin) => {
                    assert_eq!(
                        key,
                        &NodeKey::Native {
                            artifact_sha256: hex(&twin.image_sha256)
                        },
                        "{module}"
                    );
                    native.push(module.as_str());
                }
                None => assert!(matches!(key, NodeKey::Hipcc(_)), "{module}"),
            }
        }
        native.sort_unstable();
        native.dedup();
        assert_eq!(native, expected);
    }
}
