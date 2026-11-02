// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WorkloadCaseV1` extraction from real routes (PM/railgun bedrock R2).
//!
//! Sources, each producing sealed [`WorkloadCaseV1`] values:
//!
//! * [`from_redline_capture`] — a daemon/bench Redline capture
//!   (`decode[].captures[].redline_capture.sequence`): one case per capture,
//!   the chain being the distinct launches in first-seen order with call
//!   counts. Kernarg bytes are decoded through the `hipfire-kernel-abi`
//!   contracts when one names the export.
//! * [`from_programs_manifest`] — `railgun-jitcorpus/programs.json`: one AR
//!   case per admitted program (node set only; no timing, no kernargs).
//! * [`from_rocprof`] — one profiled ROI of a rocprofv3 trace.
//! * [`from_pm_loop_record`] — one `pm_loop` (schema 1) microbench record.
//!
//! Nothing here fabricates identity. Whatever cannot be derived stays `None`
//! or is recorded as `unresolved: <reason>` in the node's
//! [`ObjectIdentityV1::join`]; no launch is ever dropped.
//!
//! The CLI is [`cli`] (`pm-census extract ...`).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use hipfire_atlas::workload::{
    AbiRefV1, ChainNodeV1, DenominatorV1, ExecModeV1, FixtureV1, ModelRefV1, ObjectIdentityV1,
    OriginV1, ParamsBasisV1, PhaseV1, PlacementV1, PromptRefV1, ProvenanceV1, RowsV1, StateReqV1,
    WorkloadCaseV1, WorkloadCorpusV1,
};
use hipfire_kernel_abi::{Arch, ExportV1, Expr, KernelAbiV1, ObjectRefV1};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{inventory, rocprof};

const EXTRACTED_BY: &str = concat!("pm-census extract ", env!("CARGO_PKG_VERSION"));

/// What the caller knows about a source that the source itself does not say.
#[derive(Clone, Debug)]
pub struct CaseContext {
    pub model: ModelRefV1,
    pub arch: Option<String>,
    pub phase: PhaseV1,
    pub mode: ExecModeV1,
    pub placement: PlacementV1,
    pub prompt: Option<PromptRefV1>,
    pub config: BTreeMap<String, String>,
    pub commit: Option<String>,
}

// ---------------------------------------------------------------------------
// small helpers

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("odd-length hex string ({} digits)", s.len()));
    }
    let digit = |b: u8| -> Result<u8, String> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(format!("non-hex byte {b:#04x}")),
        }
    };
    s.as_bytes()
        .chunks(2)
        .map(|p| Ok((digit(p[0])? << 4) | digit(p[1])?))
        .collect()
}

fn jstr<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

fn jstr_req<'a>(v: &'a Value, key: &str, what: &str) -> Result<&'a str, String> {
    jstr(v, key).ok_or_else(|| format!("{what}: missing string field {key:?}"))
}

fn jaxes(v: &Value, key: &str, what: &str) -> Result<[u32; 3], String> {
    let arr = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{what}: missing array field {key:?}"))?;
    if arr.len() != 3 {
        return Err(format!(
            "{what}: {key:?} has {} elements, expected 3",
            arr.len()
        ));
    }
    let mut out = [0u32; 3];
    for (slot, x) in out.iter_mut().zip(arr) {
        *slot = x
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("{what}: {key:?} element {x} is not a u32"))?;
    }
    Ok(out)
}

fn unresolved(reason: &str) -> String {
    format!("unresolved: {reason}")
}

fn base_case(
    model: ModelRefV1,
    arch: String,
    phase: PhaseV1,
    mode: ExecModeV1,
    placement: PlacementV1,
    provenance: ProvenanceV1,
) -> WorkloadCaseV1 {
    WorkloadCaseV1 {
        schema: String::new(),
        case_id: String::new(),
        model,
        prompt: None,
        arch,
        device: None,
        config: BTreeMap::new(),
        phase,
        rows: RowsV1::default(),
        mode,
        placement,
        chain: Vec::new(),
        fixtures: Vec::new(),
        state: StateReqV1::default(),
        denominator: DenominatorV1::Unmeasured {
            reason: String::new(),
        },
        provenance,
    }
}

// ---------------------------------------------------------------------------
// kernel ABI contract lookup

/// The contract and export built for exactly `(arch, symbol)`.
fn contract_for(symbol: &str, arch: &str) -> Option<(&'static KernelAbiV1, &'static ExportV1)> {
    let arch = Arch::parse(arch)?;
    hipfire_kernel_abi::families::all()
        .find_map(|abi| abi.export(arch, symbol).map(|export| (abi, export)))
}

fn abi_ref(abi: &KernelAbiV1) -> AbiRefV1 {
    AbiRefV1 {
        id: abi.id.to_string(),
        version: abi.version,
        digest: abi.digest(),
    }
}

/// Decode a launch's kernarg bytes through the contract of `(arch, symbol)`.
///
/// `None` when no contract names the export, the kernarg image is shorter
/// than the contract's explicit image, or the bytes violate the contract
/// (wrong reserved word, conflicting param, out-of-range scalar). Otherwise
/// the contract reference and every param the explicit image carries
/// directly.
pub fn decode_node(
    symbol: &str,
    arch: &str,
    kernarg: &[u8],
) -> Option<(AbiRefV1, BTreeMap<String, u64>)> {
    let (abi, _export) = contract_for(symbol, arch)?;
    let explicit = kernarg.get(..abi.explicit_bytes as usize)?;
    let decoded = hipfire_kernel_abi::decode_params(abi, explicit).ok()?;
    let params = abi
        .params
        .iter()
        .zip(&decoded.params)
        .filter_map(|(p, v)| v.map(|v| (p.name.to_string(), v)))
        .collect();
    Some((abi_ref(abi), params))
}

/// Value of `e` when it names no launch param (only literals and the
/// export's constants).
pub(crate) fn const_value(e: &Expr, abi: &KernelAbiV1, export: &ExportV1) -> Option<u64> {
    match e {
        Expr::Const(c) => Some(*c),
        Expr::Param(n) => {
            if abi.param_index(n).is_some() {
                None
            } else {
                export.consts.iter().find(|(k, _)| k == n).map(|(_, v)| *v)
            }
        }
        Expr::Add(xs) => xs
            .iter()
            .try_fold(0u64, |a, x| a.checked_add(const_value(x, abi, export)?)),
        Expr::Mul(xs) => xs
            .iter()
            .try_fold(1u64, |a, x| a.checked_mul(const_value(x, abi, export)?)),
        Expr::Sub(a, b) => const_value(a, abi, export)?.checked_sub(const_value(b, abi, export)?),
        Expr::DivExact(a, d) => {
            let v = const_value(a, abi, export)?;
            (*d != 0 && v % d == 0).then(|| v / d)
        }
        Expr::DivCeil(a, d) => {
            let v = const_value(a, abi, export)?;
            (*d != 0).then(|| v.div_ceil(*d))
        }
        Expr::Div(a, d) => {
            let v = const_value(a, abi, export)?;
            (*d != 0).then(|| v / d)
        }
    }
}

/// Solve `e == target` for the launch params `e` names, exactly. Only
/// expressions that are injective in their single unknown are inverted
/// (`Param`, `Mul` by a constant, `Add`/`Sub` of a constant, `DivExact`);
/// `Div`/`DivCeil` lose information and are refused. Binds into `out`;
/// a param bound twice must agree.
fn invert(
    e: &Expr,
    target: u64,
    abi: &KernelAbiV1,
    export: &ExportV1,
    out: &mut BTreeMap<String, u64>,
) -> bool {
    if let Some(c) = const_value(e, abi, export) {
        return c == target;
    }
    match e {
        Expr::Param(n) => match out.get(*n) {
            Some(v) => *v == target,
            None => {
                out.insert((*n).to_string(), target);
                true
            }
        },
        Expr::Mul(xs) => {
            let mut k = 1u64;
            let mut unknown = None;
            for x in xs.iter() {
                match const_value(x, abi, export) {
                    Some(c) => match k.checked_mul(c) {
                        Some(v) => k = v,
                        None => return false,
                    },
                    None => {
                        if unknown.replace(x).is_some() {
                            return false;
                        }
                    }
                }
            }
            match unknown {
                Some(x) if k != 0 && target % k == 0 => invert(x, target / k, abi, export, out),
                _ => false,
            }
        }
        Expr::Add(xs) => {
            let mut sum = 0u64;
            let mut unknown = None;
            for x in xs.iter() {
                match const_value(x, abi, export) {
                    Some(c) => match sum.checked_add(c) {
                        Some(v) => sum = v,
                        None => return false,
                    },
                    None => {
                        if unknown.replace(x).is_some() {
                            return false;
                        }
                    }
                }
            }
            match (unknown, target.checked_sub(sum)) {
                (Some(x), Some(rest)) => invert(x, rest, abi, export, out),
                _ => false,
            }
        }
        Expr::Sub(a, b) => {
            if let Some(bc) = const_value(b, abi, export) {
                target
                    .checked_add(bc)
                    .is_some_and(|t| invert(a, t, abi, export, out))
            } else if let Some(ac) = const_value(a, abi, export) {
                ac.checked_sub(target)
                    .is_some_and(|t| invert(b, t, abi, export, out))
            } else {
                false
            }
        }
        Expr::DivExact(a, d) => {
            *d != 0
                && target
                    .checked_mul(*d)
                    .is_some_and(|t| invert(a, t, abi, export, out))
        }
        Expr::Const(_) | Expr::Div(..) | Expr::DivCeil(..) => false,
    }
}

/// Params recoverable exactly from a launch's workgroup grid through the
/// grid expressions of the `(arch, symbol)` export. `None` unless the
/// export's block equals `block`, every axis is a constant match or exactly
/// invertible, and at least one param is bound.
fn grid_bound(
    symbol: &str,
    arch: &str,
    grid: [u32; 3],
    block: [u32; 3],
) -> Option<BTreeMap<String, u64>> {
    let (abi, export) = contract_for(symbol, arch)?;
    if export.block != block {
        return None;
    }
    let mut out = BTreeMap::new();
    for axis in 0..3 {
        if !invert(
            &export.grid[axis],
            u64::from(grid[axis]),
            abi,
            export,
            &mut out,
        ) {
            return None;
        }
    }
    (!out.is_empty()).then_some(out)
}

// ---------------------------------------------------------------------------
// Redline capture

/// Capture artifact names are `{module}.{jit_cache_key}.hsaco` for the
/// runtime-JIT hipcc objects (railgun-jitcorpus `programs.json`,
/// `nodes_from.note`); the key is 16 hex digits.
fn jit_stem(artifact: &str) -> Option<(String, String)> {
    let name = Path::new(artifact).file_name()?.to_str()?;
    let stem = name.strip_suffix(".hsaco")?;
    let (module, key) = stem.rsplit_once('.')?;
    (key.len() == 16 && key.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| (module.to_string(), key.to_string()))
}

fn path_arch(artifact: &str) -> Option<&str> {
    Path::new(artifact)
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .find(|s| {
            s.strip_prefix("gfx")
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_alphanumeric()))
        })
}

#[derive(Default)]
struct CodeObjectNote {
    image_sha256: Option<String>,
    origin: Option<String>,
    module: Option<String>,
}

struct Launch {
    kernel: String,
    artifact: Option<String>,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    kernarg: Vec<u8>,
    code_object: CodeObjectNote,
}

fn parse_launch(i: usize, v: &Value) -> Result<Launch, String> {
    let what = format!("sequence[{i}]");
    let kernel = jstr_req(v, "kernel", &what)?.to_string();
    let hex = jstr_req(v, "kernarg_hex", &what)?;
    let kernarg = unhex(hex).map_err(|e| format!("{what} ({kernel}): kernarg_hex: {e}"))?;
    if let Some(n) = v.get("kernarg_bytes").and_then(Value::as_u64) {
        if n != kernarg.len() as u64 {
            return Err(format!(
                "{what} ({kernel}): kernarg_bytes {n} != {} decoded bytes",
                kernarg.len()
            ));
        }
    }
    let shared_mem = v
        .get("shared_mem")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| format!("{what} ({kernel}): missing u32 shared_mem"))?;
    let code_object = match v.get("code_object") {
        Some(co) if co.is_object() => CodeObjectNote {
            image_sha256: jstr(co, "image_sha256").map(str::to_string),
            origin: jstr(co, "origin").map(str::to_string),
            module: jstr(co, "module").map(str::to_string),
        },
        _ => CodeObjectNote::default(),
    };
    Ok(Launch {
        grid: jaxes(v, "grid", &what)?,
        block: jaxes(v, "block", &what)?,
        artifact: jstr(v, "artifact").map(str::to_string),
        kernel,
        shared_mem,
        kernarg,
        code_object,
    })
}

/// Identity facts about one artifact path, shared by every node launching it.
struct ArtifactFacts {
    module: Option<String>,
    sha256: Option<String>,
    /// Join text for the artifact part of [`ObjectIdentityV1::join`].
    join: String,
    jit: bool,
    committed_pm: bool,
    native_embedded: bool,
}

fn artifact_facts(artifact: Option<&str>, co: &CodeObjectNote) -> Result<ArtifactFacts, String> {
    let Some(path) = artifact else {
        return Ok(ArtifactFacts {
            module: co.module.clone(),
            sha256: co.image_sha256.clone(),
            join: unresolved("capture launch carries no artifact"),
            jit: false,
            committed_pm: false,
            native_embedded: co.origin.as_deref() == Some("native_embedded"),
        });
    };
    let local = Path::new(path);
    let local_sha = if local.is_file() {
        Some(sha256_hex(&read_file(local)?))
    } else {
        None
    };
    let jit = jit_stem(path);
    let module = co
        .module
        .clone()
        .or_else(|| jit.as_ref().map(|(m, _)| m.clone()))
        .or_else(|| {
            local
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
        });
    let (sha256, join) = match (&co.image_sha256, &local_sha) {
        (Some(captured), Some(file)) if captured == file => (
            Some(captured.clone()),
            format!("artifact {path}; sha256 from capture code_object.image_sha256, matches the local file"),
        ),
        (Some(captured), Some(_)) => (
            Some(captured.clone()),
            format!("artifact {path}; sha256 from capture code_object.image_sha256 (the local file at that path DIFFERS)"),
        ),
        (Some(captured), None) => (
            Some(captured.clone()),
            format!("artifact {path}; sha256 from capture code_object.image_sha256; file not on this host"),
        ),
        (None, Some(file)) => (
            Some(file.clone()),
            format!("artifact {path}; sha256 of the local file"),
        ),
        (None, None) => (
            None,
            format!("artifact {path}; {}", unresolved("artifact not on this host")),
        ),
    };
    Ok(ArtifactFacts {
        module,
        sha256,
        join,
        jit: jit.is_some(),
        committed_pm: path.ends_with(".hxaco"),
        native_embedded: co.origin.as_deref() == Some("native_embedded"),
    })
}

/// Origin of a launched object. The launched artifact is the evidence; a
/// contract's claim about the same symbol is only used when the artifact says
/// nothing.
fn redline_origin(symbol: &str, arch: &str, facts: &ArtifactFacts) -> (OriginV1, String) {
    let contract = contract_for(symbol, arch);
    let contract_note = match contract {
        Some((abi, export)) => match export.object {
            ObjectRefV1::Embedded { module, .. } => format!(
                "contract {} lists an embedded PM export in module {module}",
                abi.id
            ),
            ObjectRefV1::HipccSource { module, .. } => format!(
                "contract {} lists a hipcc source export in module {module}",
                abi.id
            ),
        },
        None => "no kernel ABI contract names this export".to_string(),
    };
    if facts.jit {
        return (
            OriginV1::Hipcc,
            format!("artifact is a content-keyed JIT .hsaco stem; {contract_note}"),
        );
    }
    if facts.committed_pm {
        return (
            OriginV1::Pm,
            format!("artifact is a committed .hxaco; {contract_note}"),
        );
    }
    match contract {
        Some((_, export)) => match export.object {
            ObjectRefV1::Embedded { .. } => (
                OriginV1::Pm,
                format!("by contract only (artifact gives no origin evidence); {contract_note}"),
            ),
            ObjectRefV1::HipccSource { .. } => (
                OriginV1::Hipcc,
                format!("by contract only (artifact gives no origin evidence); {contract_note}"),
            ),
        },
        None => {
            let why = if facts.native_embedded {
                "artifact is a native-embedded image and no contract names it"
            } else {
                "artifact is neither a JIT .hsaco nor a committed .hxaco and no contract names it"
            };
            (OriginV1::Unknown, unresolved(&format!("origin: {why}")))
        }
    }
}

/// One case per `redline_capture` found under `decode`/`prefill`
/// (`captures[]`, the section being an object or an array of objects).
///
/// Grid units: the capture's `grid` is the HIP launch grid in **workgroups**,
/// not work-items. `rdna-compute` records `launch_kernel`'s `gridDim` and its
/// replay planner multiplies the three axes to count workgroups
/// (`replay.rs` `launch_workgroups`); the data agrees (`gemv_mq4g256v2_residual`,
/// block 32: grid.x 5120 = one output row per workgroup, not 5120/32).
///
/// `ctx.mode` is ignored (a Redline capture is always `RedlinePm4`);
/// `ctx.phase` is used for `decode` sections (an error if it is `Prefill`),
/// and `prefill` sections are `Prefill`.
pub fn from_redline_capture(path: &Path, ctx: &CaseContext) -> Result<Vec<WorkloadCaseV1>, String> {
    let bytes = read_file(path)?;
    let file_sha = sha256_hex(&bytes);
    let doc: Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: not valid JSON: {e}", path.display()))?;

    let mut found: Vec<(&'static str, &Value, usize, &Value)> = Vec::new();
    for section in ["prefill", "decode"] {
        let objects: Vec<&Value> = match doc.get(section) {
            Some(v) if v.is_array() => v.as_array().into_iter().flatten().collect(),
            Some(v) if v.is_object() => vec![v],
            _ => Vec::new(),
        };
        for obj in objects {
            if let Some(caps) = obj.get("captures").and_then(Value::as_array) {
                for (i, cap) in caps.iter().enumerate() {
                    if cap.get("redline_capture").is_some() {
                        found.push((section, obj, i, cap));
                    }
                }
            }
        }
    }
    if found.is_empty() {
        return Err(format!(
            "{}: no redline_capture under prefill/decode captures[]",
            path.display()
        ));
    }

    let mut cases = Vec::new();
    for (section, obj, idx, cap) in found {
        let rc = &cap["redline_capture"];
        let what = format!("{} {section}.captures[{idx}]", path.display());
        let seq = rc
            .get("sequence")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                format!("{what}: redline_capture has no sequence (capture recorded without detail)")
            })?;
        if let Some(n) = rc.get("launches").and_then(Value::as_u64) {
            if n != seq.len() as u64 {
                return Err(format!(
                    "{what}: launches {n} != sequence length {}",
                    seq.len()
                ));
            }
        }
        let launches: Vec<Launch> = seq
            .iter()
            .enumerate()
            .map(|(i, v)| parse_launch(i, v))
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{what}: {e}"))?;
        cases.push(redline_case(
            path, &file_sha, &doc, section, obj, idx, cap, rc, &launches, ctx,
        )?);
    }
    Ok(cases)
}

#[allow(clippy::too_many_arguments)]
fn redline_case(
    path: &Path,
    file_sha: &str,
    doc: &Value,
    section: &str,
    section_obj: &Value,
    idx: usize,
    cap: &Value,
    rc: &Value,
    launches: &[Launch],
    ctx: &CaseContext,
) -> Result<WorkloadCaseV1, String> {
    let what = format!("{} {section}.captures[{idx}]", path.display());

    // Arch: the caller's, else the one every artifact path names.
    let path_arches: std::collections::BTreeSet<&str> = launches
        .iter()
        .filter_map(|l| l.artifact.as_deref().and_then(path_arch))
        .collect();
    let arch = match (&ctx.arch, path_arches.len()) {
        (Some(a), _) => {
            if let Some(bad) = path_arches.iter().find(|p| **p != a.as_str()) {
                return Err(format!(
                    "{what}: --arch {a} but an artifact path names {bad}"
                ));
            }
            a.clone()
        }
        (None, 1) => (*path_arches.iter().next().unwrap()).to_string(),
        (None, 0) => {
            return Err(format!(
                "{what}: arch is not derivable (no gfx component in any artifact path); pass --arch"
            ))
        }
        (None, _) => {
            return Err(format!(
                "{what}: artifact paths name several archs {path_arches:?}; pass --arch"
            ))
        }
    };

    let phase = match section {
        "prefill" => PhaseV1::Prefill,
        _ => {
            if ctx.phase == PhaseV1::Prefill {
                return Err(format!("{what}: a decode section cannot be phase prefill"));
            }
            ctx.phase
        }
    };

    // Distinct launches, first-seen order.
    type Key = (String, Option<String>, [u32; 3], [u32; 3], u32, Vec<u8>);
    let mut index: HashMap<Key, usize> = HashMap::new();
    let mut chain: Vec<ChainNodeV1> = Vec::new();
    let mut facts_cache: HashMap<Option<String>, ArtifactFacts> = HashMap::new();
    let mut decode_failed = 0u64;
    let mut no_contract = 0u64;
    for l in launches {
        let key: Key = (
            l.kernel.clone(),
            l.artifact.clone(),
            l.grid,
            l.block,
            l.shared_mem,
            l.kernarg.clone(),
        );
        if let Some(&i) = index.get(&key) {
            chain[i].calls += 1;
            continue;
        }
        if !facts_cache.contains_key(&l.artifact) {
            let f = artifact_facts(l.artifact.as_deref(), &l.code_object)?;
            facts_cache.insert(l.artifact.clone(), f);
        }
        let facts = &facts_cache[&l.artifact];
        let (origin, basis) = redline_origin(&l.kernel, &arch, facts);
        let contract = contract_for(&l.kernel, &arch);
        let (params, params_basis) = match decode_node(&l.kernel, &arch, &l.kernarg) {
            Some((_, p)) if !p.is_empty() => (p, ParamsBasisV1::KernargDecode),
            Some(_) => (BTreeMap::new(), ParamsBasisV1::None),
            None => {
                if contract.is_some() {
                    decode_failed += 1;
                } else {
                    no_contract += 1;
                }
                (BTreeMap::new(), ParamsBasisV1::None)
            }
        };
        index.insert(key, chain.len());
        chain.push(ChainNodeV1 {
            symbol: l.kernel.clone(),
            object: ObjectIdentityV1 {
                origin,
                module: facts.module.clone(),
                artifact: l.artifact.clone(),
                artifact_sha256: facts.sha256.clone(),
                join: format!("{}; origin {origin:?}: {basis}", facts.join),
            },
            abi: contract.map(|(abi, _)| abi_ref(abi)),
            params,
            params_basis,
            grid: Some(l.grid),
            block: Some(l.block),
            dynamic_lds: Some(l.shared_mem),
            kernarg_sha256: Some(sha256_hex(&l.kernarg)),
            calls: 1,
            time_ns: None,
        });
    }

    // Model.
    let mut model = ctx.model.clone();
    let model_path = jstr(doc, "model");
    if model.name.is_empty() {
        model.name = model_path
            .and_then(|m| Path::new(m).file_stem())
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .ok_or_else(|| {
                format!("{what}: no model name (pass --model; capture has no model path)")
            })?;
    }
    if model.file_bytes.is_none() {
        model.file_bytes = doc.get("model_bytes").and_then(Value::as_u64);
    }

    // Denominator: the capture entry's own timed result.
    let iterations = cap
        .get("iterations")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .or_else(|| {
            section_obj
                .get("capture_iterations")
                .and_then(Value::as_u64)
                .filter(|n| *n > 0)
        });
    let context = cap
        .get("context_tokens")
        .or_else(|| section_obj.get("context_tokens"))
        .and_then(Value::as_u64);
    let denominator = match (cap.get("ms").and_then(Value::as_f64), iterations) {
        (Some(ms), Some(it)) if ms.is_finite() && ms > 0.0 => DenominatorV1::Measured {
            wall_ns: (ms * 1e6 / it as f64).round() as u64,
            roi: format!(
                "one {section} step of the captured sequence: {section}.captures[{idx}].ms / {it} iteration(s){}",
                context.map(|c| format!(", context {c} tokens")).unwrap_or_default()
            ),
            source: path.display().to_string(),
        },
        (None, _) => DenominatorV1::Unmeasured {
            reason: format!("capture entry {section}.captures[{idx}] has no ms (not a timed ROI)"),
        },
        _ => DenominatorV1::Unmeasured {
            reason: format!(
                "capture entry {section}.captures[{idx}] has a non-positive or non-finite ms, or no iteration count"
            ),
        },
    };

    let mut config = ctx.config.clone();
    let mut put = |k: &str, v: String| {
        config.insert(format!("capture.{k}"), v);
    };
    put("section", section.to_string());
    put("index", idx.to_string());
    put("launches", launches.len().to_string());
    put("distinct_nodes", chain.len().to_string());
    if let Some(h) = jstr(rc, "sequence_hash") {
        put("sequence_hash", h.to_string());
    }
    if let Some(n) = rc.get("unique_kernels").and_then(Value::as_u64) {
        put("unique_kernels", n.to_string());
    }
    if let Some(it) = iterations {
        put("iterations", it.to_string());
    }
    if let Some(ms) = cap.get("ms").and_then(Value::as_f64) {
        put("ms", ms.to_string());
    }
    for key in ["kv_mode", "daemon", "model"] {
        if let Some(v) = jstr(doc, key) {
            put(key, v.to_string());
        }
    }
    put("grid_unit", "workgroups".to_string());
    put("kernarg_decode_failed_nodes", decode_failed.to_string());
    put("nodes_without_contract", no_contract.to_string());

    let mut case = base_case(
        model,
        arch,
        phase,
        ExecModeV1::RedlinePm4,
        ctx.placement,
        ProvenanceV1 {
            source_kind: "redline_capture".to_string(),
            source_path: path.display().to_string(),
            source_sha256: file_sha.to_string(),
            extracted_by: EXTRACTED_BY.to_string(),
            commit: ctx.commit.clone(),
        },
    );
    case.prompt = ctx.prompt.clone();
    case.config = config;
    case.rows = RowsV1 {
        context,
        ..RowsV1::default()
    };
    case.chain = chain;
    case.state = StateReqV1 {
        notes: vec![
            "rollback/prefix_commit/graph_replay are not recorded by a Redline capture; left false"
                .to_string(),
        ],
        ..StateReqV1::default()
    };
    case.denominator = denominator;
    Ok(case.seal())
}

// ---------------------------------------------------------------------------
// railgun programs manifest

/// The `(arch, module, symbols)` twins listed in `pm_decode_twins.rs`
/// (`twin!("<module>", "<file>", &["<symbol>", ...], ...)` inside
/// `pub static GFX<nnnn>_TWINS`), read from the source text so nothing is
/// hardcoded here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PmTwins {
    pub source: String,
    entries: Vec<(String, String, Vec<String>)>,
}

impl PmTwins {
    pub fn parse(source: &str, text: &str) -> Result<Self, String> {
        let statics: Vec<(usize, String)> = text
            .match_indices("static GFX")
            .filter_map(|(at, _)| {
                let digits: String = text[at + "static GFX".len()..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                (!digits.is_empty()).then(|| (at, format!("gfx{digits}")))
            })
            .collect();
        let mut entries = Vec::new();
        for (at, _) in text.match_indices("twin!(") {
            let arch = statics
                .iter()
                .rev()
                .find(|(s, _)| *s < at)
                .map(|(_, a)| a.clone())
                .ok_or_else(|| {
                    format!("{source}: twin! at byte {at} precedes any GFX*_TWINS static")
                })?;
            let rest = &text[at + "twin!(".len()..];
            let (module, after) = string_literal(rest)
                .ok_or_else(|| format!("{source}: twin! at byte {at}: no module literal"))?;
            let list_at = after
                .find("&[")
                .ok_or_else(|| format!("{source}: twin! {module}: no symbol list"))?;
            let list = &after[list_at + 2..];
            let list_end = list
                .find(']')
                .ok_or_else(|| format!("{source}: twin! {module}: unterminated symbol list"))?;
            let mut symbols = Vec::new();
            let mut cursor = &list[..list_end];
            while let Some((sym, next)) = string_literal(cursor) {
                symbols.push(sym);
                cursor = next;
            }
            if symbols.is_empty() {
                return Err(format!("{source}: twin! {module}: empty symbol list"));
            }
            entries.push((arch, module, symbols));
        }
        if entries.is_empty() {
            return Err(format!("{source}: no twin!(...) entries found"));
        }
        Ok(Self {
            source: source.to_string(),
            entries,
        })
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&path.display().to_string(), &text)
    }

    /// `(module, symbol)` is an accepted twin on exactly `arch`.
    pub fn contains(&self, arch: &str, module: &str, symbol: &str) -> bool {
        self.entries
            .iter()
            .any(|(a, m, s)| a == arch && m == module && s.iter().any(|x| x == symbol))
    }
}

/// The first `"..."` literal in `s` (no escapes occur in the sources parsed)
/// and the text after its closing quote.
fn string_literal(s: &str) -> Option<(String, &str)> {
    let open = s.find('"')?;
    let body = &s[open + 1..];
    let close = body.find('"')?;
    Some((body[..close].to_string(), &body[close + 1..]))
}

const TWINS_RELPATH: &str = "crates/rdna-compute/src/pm_decode_twins.rs";

fn find_twins(manifest: &Path, repo: Option<&Path>) -> Option<PathBuf> {
    if let Some(r) = repo {
        let p = r.join(TWINS_RELPATH);
        return p.is_file().then_some(p);
    }
    let abs = std::fs::canonicalize(manifest).ok()?;
    abs.ancestors()
        .map(|a| a.join(TWINS_RELPATH))
        .find(|p| p.is_file())
}

/// One AR case per program of `railgun-jitcorpus/programs.json`;
/// `pm_decode_twins.rs` is looked up in the repo containing the manifest.
pub fn from_programs_manifest(path: &Path) -> Result<Vec<WorkloadCaseV1>, String> {
    from_programs_manifest_in(path, None, None)
}

/// [`from_programs_manifest`] with an explicit repo root for
/// `pm_decode_twins.rs` and a commit recorded in the provenance.
///
/// The chain is the manifest's node list, one node per entry, `calls` = 0
/// (the manifest does not count launches) and no geometry or kernargs. A
/// node is `Pm` iff its `(module, kernel)` is listed in `pm_decode_twins.rs`
/// for the program's arch; otherwise `Hipcc`, as the manifest names the
/// runtime-JIT module behind every launch (railgun-jitcorpus lib docs). The
/// mode is `RedlinePm4`: the manifest's `admission` states these are the
/// sealed default Redline programs. With no twins file locatable, origins are
/// `Unknown` (`unresolved`).
pub fn from_programs_manifest_in(
    path: &Path,
    repo: Option<&Path>,
    commit: Option<&str>,
) -> Result<Vec<WorkloadCaseV1>, String> {
    let bytes = read_file(path)?;
    let sha = sha256_hex(&bytes);
    let doc: Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: not valid JSON: {e}", path.display()))?;
    let what = path.display().to_string();
    let schema = jstr_req(&doc, "schema", &what)?;
    if schema != "railgun-admitted-programs" {
        return Err(format!(
            "{what}: schema {schema:?} is not \"railgun-admitted-programs\""
        ));
    }
    let twins_path = find_twins(path, repo);
    let twins = twins_path.as_deref().map(PmTwins::load).transpose()?;
    let programs = doc
        .get("programs")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{what}: missing programs array"))?;

    let mut cases = Vec::new();
    for p in programs {
        let id = jstr_req(p, "id", &what)?;
        let pw = format!("{what} program {id}");
        let arch = jstr_req(p, "arch", &pw)?.to_string();
        let nodes = p
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{pw}: missing nodes array"))?;

        let mut chain = Vec::with_capacity(nodes.len());
        for (i, n) in nodes.iter().enumerate() {
            let pair = n
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("{pw}: nodes[{i}] is not a [kernel, module] pair"))?;
            let (Some(kernel), Some(module)) = (pair[0].as_str(), pair[1].as_str()) else {
                return Err(format!("{pw}: nodes[{i}] entries are not strings"));
            };
            let artifact_gap = unresolved("manifest names the module stem only");
            let (origin, join) = match &twins {
                Some(t) if t.contains(&arch, module, kernel) => (
                    OriginV1::Pm,
                    format!(
                        "pm_decode twin: ({module}, {kernel}) is listed for {arch} in {}; artifact {artifact_gap}",
                        t.source
                    ),
                ),
                Some(t) => (
                    OriginV1::Hipcc,
                    format!(
                        "runtime-JIT hipcc module per the manifest; ({module}, {kernel}) is not a pm_decode twin for {arch} in {}; artifact {artifact_gap}",
                        t.source
                    ),
                ),
                None => (
                    OriginV1::Unknown,
                    unresolved(&format!(
                        "origin: {TWINS_RELPATH} not found for this manifest (pass --repo)"
                    )),
                ),
            };
            chain.push(ChainNodeV1 {
                symbol: kernel.to_string(),
                object: ObjectIdentityV1 {
                    origin,
                    module: Some(module.to_string()),
                    artifact: None,
                    artifact_sha256: None,
                    join,
                },
                abi: contract_for(kernel, &arch).map(|(abi, _)| abi_ref(abi)),
                params: BTreeMap::new(),
                params_basis: ParamsBasisV1::None,
                grid: None,
                block: None,
                dynamic_lds: None,
                kernarg_sha256: None,
                calls: 0,
                time_ns: None,
            });
        }

        let model_name = id
            .strip_suffix(&format!("-{arch}"))
            .unwrap_or(id)
            .to_string();
        let family = model_name
            .split('-')
            .next()
            .unwrap_or(&model_name)
            .to_string();
        let mut case = base_case(
            ModelRefV1 {
                name: model_name,
                family,
                weight_format: None,
                file_sha256: None,
                file_bytes: None,
                sidecar_sha256: None,
            },
            arch.clone(),
            PhaseV1::Ar,
            ExecModeV1::RedlinePm4,
            PlacementV1::Unknown,
            ProvenanceV1 {
                source_kind: "programs_manifest".to_string(),
                source_path: what.clone(),
                source_sha256: sha.clone(),
                extracted_by: EXTRACTED_BY.to_string(),
                commit: commit.map(str::to_string),
            },
        );
        let mut put = |k: &str, v: Option<String>| {
            if let Some(v) = v {
                case.config.insert(k.to_string(), v);
            }
        };
        put("manifest.program_id", Some(id.to_string()));
        put("manifest.route", jstr(p, "route").map(str::to_string));
        put(
            "manifest.admission",
            jstr(&doc, "admission").map(str::to_string),
        );
        put(
            "manifest.model_basis",
            Some("name = program id minus the arch suffix, family = its first dash token; the manifest names no model file".to_string()),
        );
        if let Some(g) = p.get("golden") {
            put(
                "golden.sequence_hash",
                jstr(g, "sequence_hash").map(str::to_string),
            );
            put("golden.launches", g.get("launches").map(Value::to_string));
            put(
                "golden.unique_kernels",
                g.get("unique_kernels").map(Value::to_string),
            );
            put("golden.source", jstr(g, "source").map(str::to_string));
        }
        let mut matches_golden = None;
        if let Some(nf) = p.get("nodes_from") {
            put(
                "nodes_from.capture",
                jstr(nf, "capture").map(str::to_string),
            );
            put(
                "nodes_from.sequence_hash",
                jstr(nf, "sequence_hash").map(str::to_string),
            );
            put(
                "nodes_from.launches",
                nf.get("launches").map(Value::to_string),
            );
            put(
                "nodes_from.matches_golden",
                nf.get("matches_golden").map(Value::to_string),
            );
            put("nodes_from.note", jstr(nf, "note").map(str::to_string));
            matches_golden = nf.get("matches_golden").and_then(Value::as_bool);
            if let Some(cap) = jstr(nf, "capture") {
                case.fixtures.push(FixtureV1 {
                    name: "nodes_from.capture".to_string(),
                    kind: "captured".to_string(),
                    path: Some(cap.to_string()),
                    sha256: None,
                    seed: None,
                });
            }
        }
        case.state.notes.push(
            "node-set manifest: per-node calls are not recorded (calls = 0), no geometry or kernargs"
                .to_string(),
        );
        case.state.notes.push(
            "railgun's own copy/fill kernels (manifest railgun_kernels) are required on every arch but are not in the program's node list"
                .to_string(),
        );
        if matches_golden == Some(false) {
            case.state.notes.push(
                "the recorded capture does not match the golden sequence: this node set is historical"
                    .to_string(),
            );
        }
        case.chain = chain;
        case.denominator = DenominatorV1::Unmeasured {
            reason: "node-set manifest: no timing".to_string(),
        };
        cases.push(case.seal());
    }
    Ok(cases)
}

// ---------------------------------------------------------------------------
// rocprofv3 trace

fn origin_ord(o: OriginV1) -> u8 {
    match o {
        OriginV1::Pm => 0,
        OriginV1::Hipcc => 1,
        OriginV1::Unknown => 2,
    }
}

/// One ROI of a rocprofv3 trace: the dispatches of `agent` (the only GPU
/// agent when `None`) wholly inside `window` (`[start_ns, end_ns]`; the whole
/// agent timeline when `None`).
///
/// Chain nodes are the distinct `(kernel name, grid in workgroups, workgroup
/// size, dispatch group segment bytes, attributed object)` in first-start
/// order with calls and summed dispatch time. `dynamic_lds` is the dispatch's
/// group segment size minus the kernel's static group segment when both are
/// known (rocprof reports the dispatch total). `params` are filled
/// (`GridBound`) only when an ABI contract's grid expression inverts exactly
/// for the node. Dispatches straddling the window edge, dispatches of other
/// agents and memory copies in the window are counted in `config`, never
/// dropped silently. The denominator is the window length.
pub fn from_rocprof(
    trace: &rocprof::Trace,
    inv: &inventory::Inventory,
    window: Option<(u64, u64)>,
    agent: Option<u64>,
    ctx: &CaseContext,
) -> Result<WorkloadCaseV1, String> {
    let src = &trace.source.path;
    let agent_id = match agent {
        Some(a) => {
            let ag = trace
                .agents
                .iter()
                .find(|x| x.id == a)
                .ok_or_else(|| format!("{src}: no agent {a} in the trace"))?;
            if ag.kind != "gpu" {
                return Err(format!(
                    "{src}: agent {a} is a {} agent, not a gpu",
                    ag.kind
                ));
            }
            a
        }
        // The single GPU agent that ran dispatches (the census rule).
        None => crate::census::resolve_agent(trace, None).map_err(|e| format!("{src}: {e}"))?,
    };
    let agent_name = trace
        .agents
        .iter()
        .find(|x| x.id == agent_id)
        .map(|a| a.name.clone())
        .unwrap_or_default();
    let arch = match &ctx.arch {
        Some(a) => a.clone(),
        None if agent_name.starts_with("gfx") => agent_name.clone(),
        None => {
            return Err(format!(
                "{src}: agent {agent_id} is named {agent_name:?}, not a gfx target; pass --arch"
            ))
        }
    };

    let on_agent: Vec<&rocprof::Dispatch> = trace
        .dispatches
        .iter()
        .filter(|d| d.agent == agent_id)
        .collect();
    let (w_start, w_end) = match window {
        Some((s, e)) if s < e => (s, e),
        Some((s, e)) => return Err(format!("{src}: empty window [{s}, {e}]")),
        None => {
            let s = on_agent.iter().map(|d| d.start_ns).min();
            let e = on_agent.iter().map(|d| d.end_ns).max();
            match (s, e) {
                (Some(s), Some(e)) if s < e => (s, e),
                _ => return Err(format!("{src}: agent {agent_id} has no dispatches")),
            }
        }
    };
    let mut selected: Vec<&rocprof::Dispatch> = on_agent
        .iter()
        .copied()
        .filter(|d| d.start_ns >= w_start && d.end_ns <= w_end)
        .collect();
    if selected.is_empty() {
        return Err(format!(
            "{src}: no dispatch of agent {agent_id} inside [{w_start}, {w_end}]"
        ));
    }
    selected.sort_by_key(|d| d.start_ns);
    let straddling = on_agent
        .iter()
        .filter(|d| d.end_ns > w_start && d.start_ns < w_end)
        .filter(|d| !(d.start_ns >= w_start && d.end_ns <= w_end))
        .count();
    let other_agents = trace
        .dispatches
        .iter()
        .filter(|d| d.agent != agent_id && d.end_ns > w_start && d.start_ns < w_end)
        .count();
    let copies: Vec<&rocprof::MemCopy> = trace
        .copies
        .iter()
        .filter(|c| c.start_ns >= w_start && c.end_ns <= w_end)
        .collect();

    struct Attr {
        origin: OriginV1,
        path: Option<String>,
        sha: Option<String>,
        join: String,
    }
    let mut attr_cache: HashMap<u64, Attr> = HashMap::new();

    type Key = (
        String,
        [u32; 3],
        [u32; 3],
        u32,
        u8,
        Option<String>,
        Option<String>,
    );
    let mut index: HashMap<Key, usize> = HashMap::new();
    let mut chain: Vec<ChainNodeV1> = Vec::new();
    let (mut hip_true, mut hip_false, mut hip_none) = (0u64, 0u64, 0u64);
    for d in &selected {
        match d.hip_api {
            Some(true) => hip_true += 1,
            Some(false) => hip_false += 1,
            None => hip_none += 1,
        }
        let kernel = trace.kernel(d.kernel_id);
        let symbol = match kernel {
            Some(k) => k.name.clone(),
            None => format!("kernel_id:{}", d.kernel_id),
        };
        if !attr_cache.contains_key(&d.kernel_id) {
            let a = match kernel {
                Some(k) => {
                    let at = inv.attribute(trace, k);
                    Attr {
                        origin: at.origin,
                        path: at.object_path,
                        sha: at.object_sha256,
                        join: at.join,
                    }
                }
                None => Attr {
                    origin: OriginV1::Unknown,
                    path: None,
                    sha: None,
                    join: unresolved(&format!(
                        "kernel_id {} has no kernel symbol record in the trace",
                        d.kernel_id
                    )),
                },
            };
            attr_cache.insert(d.kernel_id, a);
        }
        let attr = &attr_cache[&d.kernel_id];
        let grid = d.grid_workgroups();
        let key: Key = (
            symbol.clone(),
            grid,
            d.workgroup,
            d.group_segment_size,
            origin_ord(attr.origin),
            attr.path.clone(),
            attr.sha.clone(),
        );
        let dur = d.end_ns.saturating_sub(d.start_ns);
        if let Some(&i) = index.get(&key) {
            chain[i].calls += 1;
            chain[i].time_ns = Some(chain[i].time_ns.unwrap_or(0).saturating_add(dur));
            continue;
        }
        let dynamic_lds = kernel
            .and_then(|k| k.group_segment_size)
            .and_then(|stat| d.group_segment_size.checked_sub(stat));
        let bound = grid_bound(&symbol, &arch, grid, d.workgroup);
        let (params, params_basis) = match bound {
            Some(p) => (p, ParamsBasisV1::GridBound),
            None => (BTreeMap::new(), ParamsBasisV1::None),
        };
        index.insert(key, chain.len());
        chain.push(ChainNodeV1 {
            symbol: symbol.clone(),
            object: ObjectIdentityV1 {
                origin: attr.origin,
                module: None,
                artifact: attr.path.clone(),
                artifact_sha256: attr.sha.clone(),
                join: attr.join.clone(),
            },
            abi: contract_for(&symbol, &arch).map(|(abi, _)| abi_ref(abi)),
            params,
            params_basis,
            grid: Some(grid),
            block: Some(d.workgroup),
            dynamic_lds,
            kernarg_sha256: None,
            calls: 1,
            time_ns: Some(dur),
        });
    }

    let mut case = base_case(
        ctx.model.clone(),
        arch,
        ctx.phase,
        ctx.mode,
        ctx.placement,
        ProvenanceV1 {
            source_kind: "rocprof_trace".to_string(),
            source_path: trace.source.path.clone(),
            source_sha256: trace.source.sha256.clone(),
            extracted_by: EXTRACTED_BY.to_string(),
            commit: ctx.commit.clone().or_else(|| inv.commit.clone()),
        },
    );
    case.prompt = ctx.prompt.clone();
    case.device = Some(format!("rocprof agent {agent_id} ({agent_name})"));
    case.config = ctx.config.clone();
    let mut put = |k: &str, v: String| {
        case.config.insert(format!("rocprof.{k}"), v);
    };
    put("format", trace.source.format.clone());
    put("agent", agent_id.to_string());
    put("window_ns", format!("{w_start}:{w_end}"));
    put("dispatches", selected.len().to_string());
    put("distinct_nodes", chain.len().to_string());
    put("dispatches_hip_api_true", hip_true.to_string());
    put("dispatches_hip_api_false", hip_false.to_string());
    put("dispatches_hip_api_unknown", hip_none.to_string());
    put("straddling_dispatches_excluded", straddling.to_string());
    put("other_agent_dispatches_in_window", other_agents.to_string());
    put("copies_in_window", copies.len().to_string());
    put(
        "copy_bytes_in_window",
        copies.iter().map(|c| c.bytes).sum::<u64>().to_string(),
    );
    put(
        "copy_ns_in_window",
        copies
            .iter()
            .map(|c| c.end_ns.saturating_sub(c.start_ns))
            .sum::<u64>()
            .to_string(),
    );
    put(
        "dynamic_lds_basis",
        "dispatch group_segment_size minus kernel static group_segment_size".to_string(),
    );
    case.chain = chain;
    case.denominator = DenominatorV1::Measured {
        wall_ns: w_end - w_start,
        roi: match window {
            Some(_) => format!(
                "rocprofv3 window [{w_start}, {w_end}] ns, agent {agent_id}, {} dispatches",
                selected.len()
            ),
            None => format!(
                "rocprofv3 agent {agent_id} first start to last end, {} dispatches",
                selected.len()
            ),
        },
        source: trace.source.path.clone(),
    };
    Ok(case.seal())
}

// ---------------------------------------------------------------------------
// pm_loop record

/// One `pm_loop` schema-1 record as a microbench case.
///
/// The chain is the candidate's launch symbol (a PeaceMaker-emitted object);
/// the baseline (the shipped object, or a second emitted variant) is recorded
/// in `config` (`baseline.*`). Integer shape entries become `params` with
/// `params_basis` **None**: a pm_loop record carries the shape the harness
/// was asked for, not kernarg bytes, so they are not a kernarg decode;
/// non-integer shape entries go to `config` (`shape.*`). The record does not
/// state a phase; this entry point uses `Prefill` and says so in
/// `config["phase_basis"]` (see [`from_pm_loop_record_with`]).
pub fn from_pm_loop_record(record: &Value, source_path: &str) -> Result<WorkloadCaseV1, String> {
    from_pm_loop_record_with(record, source_path, None, None)
}

/// [`from_pm_loop_record`] with an explicit phase and commit.
pub fn from_pm_loop_record_with(
    record: &Value,
    source_path: &str,
    phase: Option<PhaseV1>,
    commit: Option<&str>,
) -> Result<WorkloadCaseV1, String> {
    let what = format!("pm_loop record {source_path}");
    if record.get("schema").and_then(Value::as_u64) != Some(1) {
        return Err(format!("{what}: schema is not 1"));
    }
    let kernel = jstr_req(record, "kernel", &what)?;
    let arch = jstr_req(record, "arch", &what)?.to_string();
    let cand = record.get("candidate").ok_or_else(|| {
        format!(
            "{what}: no candidate (record failed before emit: {})",
            record
                .get("error")
                .map(Value::to_string)
                .unwrap_or_default()
        )
    })?;
    let symbol = jstr_req(cand, "launch_symbol", &format!("{what} candidate"))?.to_string();

    let timing = record.get("timing");
    let calls = timing
        .and_then(|t| t.get("calls_per_arm"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let median_us = |arm: &str| {
        timing
            .and_then(|t| t.get(arm))
            .and_then(|a| a.get("median"))
            .and_then(Value::as_f64)
            .filter(|m| m.is_finite() && *m > 0.0)
    };
    let to_ns = |us: f64| (us * 1e3 * calls as f64).round() as u64;
    let cand_time = median_us("candidate_us").filter(|_| calls > 0).map(to_ns);
    let denominator = match median_us("baseline_us").filter(|_| calls > 0) {
        Some(us) => DenominatorV1::Measured {
            wall_ns: to_ns(us),
            roi: format!(
                "pm_loop baseline arm: {calls} launches at the median hipEvent time {us} us (default stream)"
            ),
            source: source_path.to_string(),
        },
        None => DenominatorV1::Unmeasured {
            reason: match timing.and_then(|t| t.get("status")).and_then(Value::as_str) {
                Some(s) => format!("pm_loop timing status: {s}"),
                None => "pm_loop record has no usable timing (baseline_us.median / calls_per_arm)"
                    .to_string(),
            },
        },
    };

    let mut params = BTreeMap::new();
    let mut config = BTreeMap::new();
    let mut seed = None;
    if let Some(shape) = record.get("shape").and_then(Value::as_object) {
        for (k, v) in shape {
            match v.as_u64() {
                Some(n) => {
                    params.insert(k.clone(), n);
                    if k == "seed" {
                        seed = Some(n);
                    }
                }
                None => {
                    config.insert(format!("shape.{k}"), v.to_string());
                }
            }
        }
    }
    config.insert("pm_loop.kernel".to_string(), kernel.to_string());
    config.insert("pm_loop.schema".to_string(), "1".to_string());
    config.insert(
        "pm_loop.params_basis".to_string(),
        "shape as requested of the harness, not decoded from kernarg bytes".to_string(),
    );
    if let Some(ok) = record
        .get("correctness")
        .and_then(|c| c.get("pass"))
        .and_then(Value::as_bool)
    {
        config.insert("pm_loop.correctness_pass".to_string(), ok.to_string());
    }
    if let Some(n) = timing.and_then(|t| t.get("n")).and_then(Value::as_u64) {
        config.insert("pm_loop.block_launches".to_string(), n.to_string());
    }
    if let Some(variant) = jstr(cand, "variant") {
        config.insert("candidate.variant".to_string(), variant.to_string());
    }
    if let Some(knobs) = cand.get("knobs").and_then(Value::as_object) {
        for (k, v) in knobs {
            let s = v
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| v.to_string());
            config.insert(format!("candidate.knob.{k}"), s);
        }
    }
    if let Some(base) = record.get("baseline") {
        let mut put = |k: &str, v: Option<String>| {
            if let Some(v) = v {
                config.insert(format!("baseline.{k}"), v);
            }
        };
        put(
            "launch_symbol",
            jstr(base, "launch_symbol").map(str::to_string),
        );
        put("co_sha256", jstr(base, "co_sha256").map(str::to_string));
        put(
            "artifact",
            jstr(base, "artifact")
                .or_else(|| base.get("source").and_then(|s| jstr(s, "artifact")))
                .map(str::to_string),
        );
        put(
            "source",
            base.get("source")
                .and_then(|s| jstr(s, "source"))
                .map(str::to_string),
        );
        if let Some(knobs) = base.get("knobs").and_then(Value::as_object) {
            for (k, v) in knobs {
                put(
                    &format!("knob.{k}"),
                    Some(
                        v.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string()),
                    ),
                );
            }
        }
    }
    if phase.is_none() {
        config.insert(
            "phase_basis".to_string(),
            "unspecified: a pm_loop record carries no phase; defaulted to prefill".to_string(),
        );
    }

    let object = ObjectIdentityV1 {
        origin: OriginV1::Pm,
        module: None,
        artifact: jstr(cand, "object_path").map(str::to_string),
        artifact_sha256: jstr(cand, "co_sha256").map(str::to_string),
        join: "pm_loop candidate: PeaceMaker native-writer object; sha256 from the record's co_sha256; the artifact path is the run's output directory and may no longer exist".to_string(),
    };
    let node = ChainNodeV1 {
        abi: contract_for(&symbol, &arch).map(|(abi, _)| abi_ref(abi)),
        symbol,
        object,
        params,
        params_basis: ParamsBasisV1::None,
        grid: None,
        block: None,
        dynamic_lds: None,
        kernarg_sha256: None,
        calls,
        time_ns: cand_time,
    };

    let mut case = base_case(
        ModelRefV1 {
            name: format!("pm_loop:{kernel}"),
            family: "microbench".to_string(),
            weight_format: None,
            file_sha256: None,
            file_bytes: None,
            sidecar_sha256: None,
        },
        arch,
        phase.unwrap_or(PhaseV1::Prefill),
        ExecModeV1::Eager,
        PlacementV1::Unknown,
        ProvenanceV1 {
            source_kind: "pm_loop_record".to_string(),
            source_path: source_path.to_string(),
            source_sha256: sha256_hex(
                &serde_json::to_vec(record).map_err(|e| format!("{what}: {e}"))?,
            ),
            extracted_by: EXTRACTED_BY.to_string(),
            commit: commit.map(str::to_string),
        },
    );
    case.device = record
        .get("device")
        .and_then(|d| jstr(d, "uuid"))
        .map(str::to_string);
    case.config = config;
    case.chain = vec![node];
    case.fixtures = vec![FixtureV1 {
        name: "pm_loop.synthetic_inputs".to_string(),
        kind: "synthetic".to_string(),
        path: None,
        sha256: None,
        seed,
    }];
    case.state.notes.push(
        "microbench: one chain node, single stream, inputs reset outside the timed interval"
            .to_string(),
    );
    case.denominator = denominator;
    Ok(case.seal())
}

// ---------------------------------------------------------------------------
// CLI

const USAGE: &str = "\
pm-census extract — WorkloadCaseV1 corpus extraction

  pm-census extract redline  <capture.json> [common] --out corpus.jsonl
  pm-census extract programs <programs.json> [common] --out corpus.jsonl
  pm-census extract rocprof  <trace dir|file> [common] [--window START_NS:END_NS] [--agent ID] --out corpus.jsonl
  pm-census extract pm-loop  <record.json|jsonl> [common] --out corpus.jsonl
  pm-census extract merge    A.jsonl B.jsonl ... --out C.jsonl
  pm-census extract verify   C.jsonl
  pm-census extract nodes    --trace T --config C [--hfq H] --tokens N --out corpus.json   (node corpus; see `pm-census extract nodes --help`)
  pm-census extract route-hist  --dump FILE [--model ID] [--experts E] --out route_hist.json   (route_hist.v1; see `pm-census extract route-hist --help`)
  pm-census extract act-capture --dump FILE --out act_capture.json                              (act_capture.v1)

common: [--model NAME] [--family F] [--arch A] [--phase prefill|ar|spec]
        [--mode eager|hip_graph|redline_pm4|railgun|unknown]
        [--placement resident|host_mapped|mixed|unknown]
        [--weight-format W] [--repo DIR] [--commit SHA]
  --repo   repo root: pm_decode_twins.rs (programs), object inventory (rocprof; default .)
  --commit read the inventory from this commit (rocprof) and record it in provenance
  redline ignores --mode (always redline_pm4); programs ignores --model/--family/--phase.
Every subcommand prints one JSON document {cases, digest, by_source_kind}.";

struct Flags {
    positional: Vec<String>,
    values: BTreeMap<String, String>,
}

impl Flags {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut values = BTreeMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let Some(flag) = a.strip_prefix("--") else {
                positional.push(a.clone());
                continue;
            };
            let (name, inline) = match flag.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (flag, None),
            };
            if !allowed.contains(&name) {
                return Err(format!("unknown flag --{name}\n\n{USAGE}"));
            }
            let value = match inline {
                Some(v) => v,
                None => it
                    .next()
                    .cloned()
                    .ok_or_else(|| format!("--{name} needs a value"))?,
            };
            if values.insert(name.to_string(), value).is_some() {
                return Err(format!("--{name} given twice"));
            }
        }
        Ok(Self { positional, values })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn input(&self) -> Result<&str, String> {
        match self.positional.as_slice() {
            [one] => Ok(one.as_str()),
            _ => Err(format!(
                "expected exactly one input, got {}\n\n{USAGE}",
                self.positional.len()
            )),
        }
    }

    fn out(&self) -> Result<PathBuf, String> {
        self.get("out")
            .map(PathBuf::from)
            .ok_or_else(|| "--out corpus.jsonl is required".to_string())
    }
}

fn parse_enum<T: serde::de::DeserializeOwned>(flag: &str, v: &str) -> Result<T, String> {
    serde_json::from_value(Value::String(v.to_string()))
        .map_err(|_| format!("--{flag}: unrecognized value {v:?}"))
}

fn context(f: &Flags) -> Result<CaseContext, String> {
    Ok(CaseContext {
        model: ModelRefV1 {
            name: f.get("model").unwrap_or("").to_string(),
            family: f
                .get("family")
                .map(str::to_string)
                .unwrap_or_else(|| unresolved("pass --family")),
            weight_format: f.get("weight-format").map(str::to_string),
            file_sha256: None,
            file_bytes: None,
            sidecar_sha256: None,
        },
        arch: f.get("arch").map(str::to_string),
        phase: match f.get("phase") {
            Some(p) => parse_enum("phase", p)?,
            None => PhaseV1::Ar,
        },
        mode: match f.get("mode") {
            Some(m) => parse_enum("mode", m)?,
            None => ExecModeV1::Unknown,
        },
        placement: match f.get("placement") {
            Some(p) => parse_enum("placement", p)?,
            None => PlacementV1::Unknown,
        },
        prompt: None,
        config: BTreeMap::new(),
        commit: f.get("commit").map(str::to_string),
    })
}

fn parse_window(s: &str) -> Result<(u64, u64), String> {
    let (a, b) = s
        .split_once(':')
        .ok_or_else(|| format!("--window {s:?}: expected START_NS:END_NS"))?;
    let num = |x: &str| {
        x.parse::<u64>()
            .map_err(|_| format!("--window {s:?}: {x:?} is not an integer"))
    };
    let (a, b) = (num(a)?, num(b)?);
    if a >= b {
        return Err(format!("--window {s:?}: start must be below end"));
    }
    Ok((a, b))
}

/// Records of a pm_loop input: one JSON object, a JSON array of them, or
/// JSONL.
fn pm_loop_records(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Array(a)) => Ok(a),
        Ok(v @ Value::Object(_)) => Ok(vec![v]),
        Ok(_) => Err(format!("{}: not a JSON object or array", path.display())),
        Err(whole) => {
            let mut out = Vec::new();
            for (i, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                out.push(serde_json::from_str::<Value>(line).map_err(|e| {
                    format!(
                        "{}: neither one JSON document ({whole}) nor JSONL (line {}: {e})",
                        path.display(),
                        i + 1
                    )
                })?);
            }
            Ok(out)
        }
    }
}

fn summary(corpus: &WorkloadCorpusV1) -> Value {
    let mut by: BTreeMap<&str, u64> = BTreeMap::new();
    for c in &corpus.cases {
        *by.entry(c.provenance.source_kind.as_str()).or_default() += 1;
    }
    serde_json::json!({
        "cases": corpus.cases.len(),
        "digest": corpus.digest,
        "by_source_kind": by,
    })
}

fn write_corpus(cases: Vec<WorkloadCaseV1>, out: &Path) -> Result<(), String> {
    let corpus = WorkloadCorpusV1::new(cases)?;
    corpus
        .write_jsonl(out)
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let mut s = summary(&corpus);
    s["out"] = Value::String(out.display().to_string());
    println!(
        "{}",
        serde_json::to_string_pretty(&s).map_err(|e| e.to_string())?
    );
    Ok(())
}

pub fn cli(args: &[String]) -> Result<(), String> {
    let Some((sub, rest)) = args.split_first() else {
        println!("{USAGE}");
        return Err("missing subcommand".to_string());
    };
    if matches!(sub.as_str(), "--help" | "-h" | "help") {
        println!("{USAGE}");
        return Ok(());
    }
    const COMMON: [&str; 9] = [
        "model",
        "family",
        "arch",
        "phase",
        "mode",
        "placement",
        "weight-format",
        "repo",
        "commit",
    ];
    let with_common = |extra: &[&'static str]| -> Vec<&'static str> {
        let mut v: Vec<&'static str> = COMMON.to_vec();
        v.push("out");
        v.extend_from_slice(extra);
        v
    };
    match sub.as_str() {
        "redline" => {
            let f = Flags::parse(rest, &with_common(&[]))?;
            let cases = from_redline_capture(Path::new(f.input()?), &context(&f)?)?;
            write_corpus(cases, &f.out()?)
        }
        "programs" => {
            let f = Flags::parse(rest, &with_common(&[]))?;
            let cases = from_programs_manifest_in(
                Path::new(f.input()?),
                f.get("repo").map(Path::new),
                f.get("commit"),
            )?;
            write_corpus(cases, &f.out()?)
        }
        "rocprof" => {
            let f = Flags::parse(rest, &with_common(&["window", "agent"]))?;
            let ctx = context(&f)?;
            let window = f.get("window").map(parse_window).transpose()?;
            let agent = f
                .get("agent")
                .map(|a| {
                    a.parse::<u64>()
                        .map_err(|_| format!("--agent {a:?} is not an integer"))
                })
                .transpose()?;
            let trace = rocprof::load(Path::new(f.input()?))?;
            let inv = inventory::build(Path::new(f.get("repo").unwrap_or(".")), f.get("commit"))?;
            let case = from_rocprof(&trace, &inv, window, agent, &ctx)?;
            write_corpus(vec![case], &f.out()?)
        }
        "pm-loop" => {
            let f = Flags::parse(rest, &with_common(&[]))?;
            let input = f.input()?;
            let phase = f.get("phase").map(|p| parse_enum("phase", p)).transpose()?;
            let records = pm_loop_records(Path::new(input))?;
            if records.is_empty() {
                return Err(format!("{input}: no pm_loop records"));
            }
            let cases = records
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let label = if records.len() == 1 {
                        input.to_string()
                    } else {
                        format!("{input}#{i}")
                    };
                    from_pm_loop_record_with(r, &label, phase, f.get("commit"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            write_corpus(cases, &f.out()?)
        }
        "nodes" => crate::ingest::cli_nodes(rest),
        "route-hist" => crate::capture::cli_route_hist(rest),
        "act-capture" => crate::capture::cli_act_capture(rest),
        "merge" => {
            let f = Flags::parse(rest, &["out"])?;
            if f.positional.is_empty() {
                return Err("merge needs at least one input corpus".to_string());
            }
            let mut cases = Vec::new();
            for p in &f.positional {
                cases.extend(WorkloadCorpusV1::read_jsonl(Path::new(p))?.cases);
            }
            write_corpus(cases, &f.out()?)
        }
        "verify" => {
            let f = Flags::parse(rest, &[])?;
            let corpus = WorkloadCorpusV1::read_jsonl(Path::new(f.input()?))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&summary(&corpus)).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        other => Err(format!("unknown extract subcommand {other:?}\n\n{USAGE}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> CaseContext {
        CaseContext {
            model: ModelRefV1 {
                name: String::new(),
                family: "qwen35".into(),
                weight_format: None,
                file_sha256: None,
                file_bytes: None,
                sidecar_sha256: None,
            },
            arch: None,
            phase: PhaseV1::Ar,
            mode: ExecModeV1::Unknown,
            placement: PlacementV1::Resident,
            prompt: None,
            config: BTreeMap::new(),
            commit: None,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pm-census-extract-{}-{name}", std::process::id()))
    }

    #[test]
    fn decode_node_is_none_for_an_unknown_symbol() {
        assert!(decode_node("no_such_kernel_symbol", "gfx1201", &[0u8; 512]).is_none());
        assert!(decode_node("no_such_kernel_symbol", "not-an-arch", &[]).is_none());
    }

    #[test]
    fn pm_twins_parse_reads_arch_module_and_symbols() {
        let text = r#"
macro_rules! twin { ($module:literal) => {}; }
pub static GFX1201_TWINS: &[NativeKernelBundle] = &[
    twin!(
        "mod_a",
        "file_a",
        &["sym_a", "sym_b"],
        source = "00"
    ),
    twin!("mod_b", "file_b", &["sym_c"], source = "11"),
];
pub static GFX9999_TWINS: &[NativeKernelBundle] = &[
    twin!("mod_c", "file_c", &["sym_d"], source = "22"),
];
"#;
        let t = PmTwins::parse("inline", text).unwrap();
        assert!(t.contains("gfx1201", "mod_a", "sym_a"));
        assert!(t.contains("gfx1201", "mod_a", "sym_b"));
        assert!(t.contains("gfx1201", "mod_b", "sym_c"));
        assert!(t.contains("gfx9999", "mod_c", "sym_d"));
        assert!(
            !t.contains("gfx1201", "mod_c", "sym_d"),
            "arch is part of the key"
        );
        assert!(
            !t.contains("gfx1201", "mod_a", "sym_c"),
            "module is part of the key"
        );
        assert!(PmTwins::parse("inline", "nothing here").is_err());
    }

    #[test]
    fn programs_manifest_yields_one_case_per_program_with_every_node() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../railgun-jitcorpus/programs.json");
        let cases = from_programs_manifest(&path).unwrap();
        let doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let programs = doc["programs"].as_array().unwrap();
        assert_eq!(cases.len(), programs.len());
        for (case, p) in cases.iter().zip(programs) {
            case.verify_id().unwrap();
            assert_eq!(case.arch, p["arch"].as_str().unwrap());
            assert_eq!(
                case.config["manifest.program_id"],
                p["id"].as_str().unwrap()
            );
            let nodes = p["nodes"].as_array().unwrap();
            assert_eq!(case.chain.len(), nodes.len(), "every node present");
            for (n, want) in case.chain.iter().zip(nodes) {
                assert_eq!(n.symbol, want[0].as_str().unwrap());
                assert_eq!(n.object.module.as_deref(), want[1].as_str());
                assert_eq!(n.calls, 0);
                assert_eq!(n.params_basis, ParamsBasisV1::None);
                assert!(n.object.artifact.is_none() && n.object.artifact_sha256.is_none());
                assert_ne!(
                    n.object.origin,
                    OriginV1::Unknown,
                    "twins file is in this repo"
                );
                assert!(n
                    .object
                    .join
                    .contains("unresolved: manifest names the module stem only"));
            }
            assert!(matches!(case.denominator, DenominatorV1::Unmeasured { .. }));
            assert_eq!(case.phase, PhaseV1::Ar);
        }
        // The dense gfx1201 program replaces five of its nineteen unique
        // kernel/module pairs with pm_decode twins.
        let dense = cases
            .iter()
            .find(|c| c.config["manifest.program_id"] == "qwen35-dense-h2-gfx1201")
            .unwrap();
        assert_eq!(dense.chain.len(), 19);
        assert_eq!(
            dense
                .chain
                .iter()
                .filter(|n| n.object.origin == OriginV1::Pm)
                .count(),
            5
        );
        // No twins on other arches.
        for c in cases.iter().filter(|c| c.arch != "gfx1201") {
            assert!(c.chain.iter().all(|n| n.object.origin == OriginV1::Hipcc));
        }
    }

    fn fixture_json() -> String {
        // Three launches, two distinct: A, B, A (identical kernarg).
        let a = serde_json::json!({
            "kernel": "kern_a", "artifact": "/nonexistent/work/gfx1201/kern_a.0123456789abcdef.hsaco",
            "grid": [20, 1, 1], "block": [256, 1, 1], "shared_mem": 1024,
            "kernarg_bytes": 8, "kernarg_hex": "0011223344556677", "kernarg_hash": "x"
        });
        let b = serde_json::json!({
            "kernel": "kern_b", "artifact": "/nonexistent/work/gfx1201/kern_b.fedcba9876543210.hsaco",
            "grid": [4, 512, 1], "block": [32, 1, 1], "shared_mem": 0,
            "kernarg_bytes": 4, "kernarg_hex": "aabbccdd", "kernarg_hash": "y"
        });
        serde_json::json!({
            "model": "/models/tiny/model.hfq", "model_bytes": 1234, "kv_mode": "fp8",
            "daemon": "/bin/daemon",
            "prefill": {},
            "decode": {
                "context_tokens": 512, "capture_iterations": 1,
                "captures": [{
                    "type": "decode_result", "context_tokens": 512, "iterations": 2, "ms": 4.0,
                    "redline_capture": {
                        "launches": 3, "unique_kernels": 2, "sequence_hash": "deadbeef",
                        "sequence": [a.clone(), b, a]
                    }
                }]
            }
        })
        .to_string()
    }

    #[test]
    fn redline_import_counts_distinct_launches_and_is_stable() {
        let path = tmp("redline-fixture.json");
        std::fs::write(&path, fixture_json()).unwrap();
        let cases = from_redline_capture(&path, &ctx()).unwrap();
        assert_eq!(cases.len(), 1);
        let c = &cases[0];
        c.verify_id().unwrap();
        assert_eq!(
            c.chain.iter().map(|n| n.calls).collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert_eq!(c.chain[0].symbol, "kern_a");
        assert_eq!(c.chain[0].grid, Some([20, 1, 1]));
        assert_eq!(c.chain[0].block, Some([256, 1, 1]));
        assert_eq!(c.chain[0].dynamic_lds, Some(1024));
        assert_eq!(
            c.chain[0].kernarg_sha256.as_deref(),
            Some(sha256_hex(&unhex("0011223344556677").unwrap()).as_str())
        );
        // Identity is never invented: the artifacts are not on this host.
        assert_eq!(c.chain[0].object.artifact_sha256, None);
        assert!(c.chain[0]
            .object
            .join
            .contains("unresolved: artifact not on this host"));
        assert_eq!(c.chain[0].object.origin, OriginV1::Hipcc);
        assert_eq!(c.chain[0].object.module.as_deref(), Some("kern_a"));
        assert_eq!(c.chain[0].params_basis, ParamsBasisV1::None);
        assert!(c.chain[0].abi.is_none());
        assert_eq!(c.arch, "gfx1201");
        assert_eq!(c.mode, ExecModeV1::RedlinePm4);
        assert_eq!(c.model.name, "model");
        assert_eq!(c.model.file_bytes, Some(1234));
        assert_eq!(c.rows.context, Some(512));
        // ms / iterations, in ns.
        assert_eq!(
            c.denominator,
            DenominatorV1::Measured {
                wall_ns: 2_000_000,
                roi: c_roi(c),
                source: path.display().to_string(),
            }
        );
        assert_eq!(c.provenance.source_kind, "redline_capture");
        assert_eq!(
            c.provenance.source_sha256,
            sha256_hex(fixture_json().as_bytes())
        );

        // Stable: a second import gives the same id.
        let again = from_redline_capture(&path, &ctx()).unwrap();
        assert_eq!(again[0].case_id, c.case_id);

        // A different kernarg for the third launch changes the chain and the id.
        let changed = fixture_json().replacen("0011223344556677", "0011223344556678", 1);
        std::fs::write(&path, changed).unwrap();
        let other = from_redline_capture(&path, &ctx()).unwrap();
        assert_ne!(other[0].case_id, c.case_id);
        std::fs::remove_file(&path).unwrap();
    }

    fn c_roi(c: &WorkloadCaseV1) -> String {
        match &c.denominator {
            DenominatorV1::Measured { roi, .. } => roi.clone(),
            other => panic!("expected a measured denominator, got {other:?}"),
        }
    }

    #[test]
    fn redline_import_refuses_a_missing_arch_and_a_short_sequence() {
        let path = tmp("redline-noarch.json");
        let doc = fixture_json().replace("/gfx1201/", "/hot/");
        std::fs::write(&path, &doc).unwrap();
        assert!(from_redline_capture(&path, &ctx())
            .unwrap_err()
            .contains("--arch"));
        let mut with_arch = ctx();
        with_arch.arch = Some("gfx1201".into());
        assert_eq!(
            from_redline_capture(&path, &with_arch).unwrap()[0].arch,
            "gfx1201"
        );

        // `launches` must equal the sequence length: nothing is dropped silently.
        std::fs::write(&path, doc.replace("\"launches\":3", "\"launches\":4")).unwrap();
        assert!(from_redline_capture(&path, &with_arch)
            .unwrap_err()
            .contains("launches 4"));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn redline_import_preserves_the_real_capture_when_present() {
        let path = Path::new("/home/kaden/qcal/railgun/m1p/out/capture/ctx512.json");
        if !path.is_file() {
            return;
        }
        let cases = from_redline_capture(path, &ctx()).unwrap();
        assert!(!cases.is_empty());
        for c in &cases {
            c.verify_id().unwrap();
            assert_eq!(c.arch, "gfx1201");
            assert_eq!(
                c.chain.iter().map(|n| n.calls).sum::<u64>(),
                755,
                "launches preserved"
            );
            let kernels: std::collections::BTreeSet<&str> =
                c.chain.iter().map(|n| n.symbol.as_str()).collect();
            assert_eq!(kernels.len(), 19, "unique kernels preserved");
            assert_eq!(c.config["capture.unique_kernels"], "19");
            assert_eq!(c.config["capture.grid_unit"], "workgroups");
            for n in &c.chain {
                if n.object
                    .artifact
                    .as_deref()
                    .is_some_and(|p| Path::new(p).is_file())
                {
                    assert!(n.object.artifact_sha256.is_some(), "{}", n.symbol);
                }
            }
        }
    }

    #[test]
    fn grid_bound_is_exact_for_every_contract_export() {
        // For every export: pick params, evaluate the grid forward, invert it
        // back; whenever inversion succeeds, re-evaluating with the recovered
        // params (others arbitrary) must reproduce the same grid.
        // (Exercised only for exports whose params can all take the sample value.)
        for abi in hipfire_kernel_abi::families::all() {
            for export in abi.exports {
                for base in [1u64, 64, 4096] {
                    let params: Vec<u64> = abi.params.iter().map(|_| base).collect();
                    let grid: Option<Vec<u64>> = export
                        .grid
                        .iter()
                        .map(|e| hipfire_kernel_abi::eval(e, abi, export, &params).ok())
                        .collect();
                    let Some(grid) = grid else { continue };
                    if grid.iter().any(|g| *g > u64::from(u32::MAX)) {
                        continue;
                    }
                    let g = [grid[0] as u32, grid[1] as u32, grid[2] as u32];
                    let Some(bound) =
                        grid_bound(export.symbol, export.arch.name(), g, export.block)
                    else {
                        continue;
                    };
                    // Inversion succeeded: check it reproduces the grid exactly.
                    let rebuilt: Vec<u64> = abi
                        .params
                        .iter()
                        .map(|p| bound.get(p.name).copied().unwrap_or(1))
                        .collect();
                    for (axis, e) in export.grid.iter().enumerate() {
                        assert_eq!(
                            hipfire_kernel_abi::eval(e, abi, export, &rebuilt),
                            Ok(grid[axis]),
                            "{} grid axis {axis}",
                            export.symbol
                        );
                    }
                    // A wrong block never matches the contract.
                    let wrong = [export.block[0] + 1, export.block[1], export.block[2]];
                    assert!(grid_bound(export.symbol, export.arch.name(), g, wrong).is_none());
                }
            }
        }
        assert!(
            hipfire_kernel_abi::families::all().next().is_some(),
            "no contracts registered"
        );
    }

    #[test]
    fn pm_loop_record_becomes_a_microbench_case() {
        let record = serde_json::json!({
            "schema": 1, "kernel": "mq6_i8_gemm", "arch": "gfx1201",
            "shape": {"M": 128, "N": 64, "K": 256, "route": "uniform"},
            "candidate": {
                "launch_symbol": "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201",
                "co_sha256": "ab".repeat(32), "object_path": "/tmp/x/candidate.co",
                "knobs": {"epi": "set"}, "variant": "128x128x8-set"
            },
            "baseline": {
                "launch_symbol": "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201",
                "co_sha256": "cd".repeat(32), "artifact": "kernels/x.hxaco",
                "source": {"source": "file:1", "artifact": "kernels/x.hxaco", "sha256": "cd"}
            },
            "correctness": {"pass": true},
            "device": {"arch": "gfx1201", "uuid": "GPU-1"},
            "timing": {
                "calls_per_arm": 10, "n": 5,
                "baseline_us": {"median": 100.0}, "candidate_us": {"median": 90.0}
            }
        });
        let c = from_pm_loop_record(&record, "rec.json").unwrap();
        c.verify_id().unwrap();
        assert_eq!(c.chain.len(), 1);
        let n = &c.chain[0];
        assert_eq!(n.object.origin, OriginV1::Pm);
        assert_eq!(n.object.artifact_sha256, Some("ab".repeat(32)));
        assert_eq!(n.params["M"], 128);
        assert!(!n.params.contains_key("route"));
        assert_eq!(c.config["shape.route"], "\"uniform\"");
        assert_eq!(n.params_basis, ParamsBasisV1::None);
        assert_eq!(n.calls, 10);
        assert_eq!(n.time_ns, Some(900_000));
        assert_eq!(c.config["baseline.artifact"], "kernels/x.hxaco");
        assert_eq!(c.device.as_deref(), Some("GPU-1"));
        assert_eq!(c.mode, ExecModeV1::Eager);
        assert_eq!(
            c.denominator,
            DenominatorV1::Measured {
                wall_ns: 1_000_000,
                roi: c_roi(&c),
                source: "rec.json".into()
            }
        );
        // No timing: unmeasured, not zero.
        let mut untimed = record.clone();
        untimed.as_object_mut().unwrap().remove("timing");
        let u = from_pm_loop_record(&untimed, "rec.json").unwrap();
        assert!(matches!(u.denominator, DenominatorV1::Unmeasured { .. }));
        assert_eq!(u.chain[0].calls, 0);
        assert_eq!(u.chain[0].time_ns, None);
        // A record that failed before emit has no node to extract.
        let failed = serde_json::json!({"schema": 1, "kernel": "k", "arch": "gfx1201", "error": {"stage": "args"}});
        assert!(from_pm_loop_record(&failed, "f").is_err());
    }

    #[test]
    fn cli_programs_merge_and_verify_round_trip() {
        let manifest =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../railgun-jitcorpus/programs.json");
        let (a, b, merged) = (tmp("a.jsonl"), tmp("b.jsonl"), tmp("merged.jsonl"));
        fn s(p: &Path) -> String {
            p.display().to_string()
        }
        cli(&["programs".into(), s(&manifest), "--out".into(), s(&a)]).unwrap();
        let rec = tmp("rec.json");
        std::fs::write(
            &rec,
            serde_json::json!({
                "schema": 1, "kernel": "iu4_gemm", "arch": "gfx1201",
                "candidate": {"launch_symbol": "sym"}
            })
            .to_string(),
        )
        .unwrap();
        cli(&["pm-loop".into(), s(&rec), "--out".into(), s(&b)]).unwrap();
        cli(&[
            "merge".into(),
            s(&a),
            s(&b),
            s(&a),
            "--out".into(),
            s(&merged),
        ])
        .unwrap();
        cli(&["verify".into(), s(&merged)]).unwrap();
        let corpus = WorkloadCorpusV1::read_jsonl(&merged).unwrap();
        let programs = WorkloadCorpusV1::read_jsonl(&a).unwrap();
        assert_eq!(
            corpus.cases.len(),
            programs.cases.len() + 1,
            "duplicates dedupe on merge"
        );
        for p in [a, b, merged, rec] {
            std::fs::remove_file(p).unwrap();
        }
    }
}
