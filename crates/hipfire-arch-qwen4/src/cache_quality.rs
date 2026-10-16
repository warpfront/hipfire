// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Hit-vs-cold logits oracle of the Qwen4 engine-owned session cache
//! (`hipfire_runtime::session_cache`), plus the opt-in production ID capture
//! sink consumed by `quality_driver.py`.
//!
//! Ported from the RC3 `state_parity::run_prefix_cache_quality` oracle: the
//! fixture schema (`hipfire.qwen4.cache_quality_corpus.v1`), the result schema
//! (`hipfire.qwen4.cache_quality_result.v1`), the row/KL statistics and the
//! raw-f32 sidecars are unchanged. Only the cache-driving backend differs: it
//! drives `SessionCache` through the bundle (`session_plan`, `prefill_final` /
//! `mtp_prefill`, `session_commit`, `session_clear`) instead of the RC3
//! prefix/radix cache, and state digests are one SHA-256 per
//! `SessionState::snapshot_parts` part.
//!
//! Per fixture case and arm: `session_clear`, then (hit arms) replay the prior
//! turns through plan -> prefill -> teacher-forced decode -> `session_commit`,
//! then plan and prefill the final prompt and score `score_rows + 1` logits
//! rows into a raw f32 sidecar. The result JSON is always written; `Err` is
//! returned afterwards if any expectation failed.

use crate::bundle::Qwen4Bundle;
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_spec::Qwen4MtpDrafter;
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::serve_contract::CacheDomain;
use hipfire_runtime::session_cache::{SessionCache, SessionRoute, SessionState, SnapshotParts};
use hipfire_runtime::spec::MtpDrafter;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_manifest::WeightEntry;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Native MTP draft depth of the oracle's drafter (the value the RC3 oracle and
/// the session-cache hardware test use).
const CQ_MTP_K: usize = 3;

/// Optional byte budget of the oracle's `SessionCache` (`u64`); default
/// `u64::MAX >> 1`, the session-cache hardware test's "unbounded".
const CQ_BUDGET_ENV: &str = "HIPFIRE_CQ_SESSION_BUDGET_BYTES";

// ---------------------------------------------------------------------------
// Capture sink (opt-in production ID capture)
// ---------------------------------------------------------------------------

const CQ_CAPTURE_ENV: &str = "HIPFIRE_QWEN4_CACHE_QUALITY_OUT";

/// The capture file named by `raw`: `None` = capture disabled (unset/empty),
/// `Err` = a relative path (capture disabled with a warning).
fn cq_capture_path_from_env(raw: Option<&std::ffi::OsStr>) -> Result<Option<PathBuf>, String> {
    let Some(raw) = raw.filter(|raw| !raw.is_empty()) else {
        return Ok(None);
    };
    let path = PathBuf::from(raw);
    if !path.is_absolute() {
        return Err(format!(
            "{CQ_CAPTURE_ENV}={} is not an absolute path; capture disabled",
            path.display()
        ));
    }
    Ok(Some(path))
}

static CQ_CAPTURE_PATH: std::sync::LazyLock<Option<PathBuf>> = std::sync::LazyLock::new(|| {
    let raw = std::env::var_os(CQ_CAPTURE_ENV);
    match cq_capture_path_from_env(raw.as_deref()) {
        Ok(path) => path,
        Err(warning) => {
            eprintln!("cache-quality: {warning}");
            None
        }
    }
});
static CQ_CAPTURE_SINK: std::sync::Mutex<Option<std::io::BufWriter<std::fs::File>>> =
    std::sync::Mutex::new(None);

fn cq_capture_path() -> Option<&'static Path> {
    CQ_CAPTURE_PATH.as_deref()
}

/// `HIPFIRE_QWEN4_CACHE_QUALITY_OUT` names an absolute JSONL file (read once).
pub fn cache_quality_capture_enabled() -> bool {
    cq_capture_path().is_some()
}

/// Append one compact JSON line (flushed per line) to the capture file. A
/// no-op when capture is disabled; I/O failures are returned, never panic.
pub fn cache_quality_event(event: &Value) -> Result<(), String> {
    let Some(path) = cq_capture_path() else {
        return Ok(());
    };
    let mut line = serde_json::to_string(event)
        .map_err(|error| format!("cache-quality event serialization: {error}"))?;
    line.push('\n');
    let mut sink = CQ_CAPTURE_SINK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if sink.is_none() {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("open {}: {error}", path.display()))?;
        *sink = Some(std::io::BufWriter::new(file));
    }
    let Some(writer) = sink.as_mut() else {
        return Err("cache-quality capture sink is not open".to_string());
    };
    let written = writer
        .write_all(line.as_bytes())
        .and_then(|()| writer.flush());
    if let Err(error) = written {
        *sink = None;
        return Err(format!("append to {}: {error}", path.display()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const CQ_FIXTURE_SCHEMA: &str = "hipfire.qwen4.cache_quality_corpus.v1";
const CQ_RESULT_SCHEMA: &str = "hipfire.qwen4.cache_quality_result.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
enum CqStream {
    File {
        path: PathBuf,
        /// Lower-case hex.
        sha256: String,
    },
    Synthetic {
        salt: u64,
        len: usize,
    },
    /// `base[..share_len] ++ tokens(salt, len - share_len)`.
    Branch {
        salt: u64,
        len: usize,
        base: String,
        share_len: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CqTurn {
    stream: String,
    prompt_end: usize,
    consumed_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CqFinal {
    stream: String,
    prompt_end: usize,
    score_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CqArmKind {
    Hit,
    Cold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CqExpect {
    None,
    BitwiseEqualCold,
}

/// Decode route of the fixture: AR (`SessionRoute::Ar`) or native MTP
/// (`SessionRoute::Mtp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CqMode {
    Ar,
    NativeMtp,
}

impl CqMode {
    fn name(self) -> &'static str {
        match self {
            CqMode::Ar => "ar",
            CqMode::NativeMtp => "native_mtp",
        }
    }

    fn route(self) -> SessionRoute {
        match self {
            CqMode::Ar => SessionRoute::Ar,
            CqMode::NativeMtp => SessionRoute::Mtp,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CqArm {
    name: String,
    kind: CqArmKind,
    expect: CqExpect,
    expect_start: Option<usize>,
    /// Free-form tag echoed into the result (cross-binary comparisons).
    label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CqCase {
    name: String,
    turns: Vec<CqTurn>,
    fin: CqFinal,
    arms: Vec<CqArm>,
}

#[derive(Clone, Debug)]
struct CqFixture {
    scenario: String,
    id_provenance: String,
    provenance: Value,
    mode: CqMode,
    /// Parsed for schema compatibility; the session cache needs no turn
    /// boundaries from the driver.
    #[allow(dead_code)]
    turn_end_id: Option<u32>,
    streams: BTreeMap<String, CqStream>,
    cases: Vec<CqCase>,
}

fn cq_obj<'a>(
    value: &'a Value,
    ctx: &str,
    allowed: &[&str],
) -> Result<&'a Map<String, Value>, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{ctx}: expected a JSON object"))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!(
                "{ctx}: unknown key {key:?} (allowed: {})",
                allowed.join(", ")
            ));
        }
    }
    Ok(object)
}

fn cq_req<'a>(object: &'a Map<String, Value>, key: &str, ctx: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("{ctx}: missing key {key:?}"))
}

fn cq_string(object: &Map<String, Value>, key: &str, ctx: &str) -> Result<String, String> {
    cq_req(object, key, ctx)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("{ctx}: {key:?} must be a string"))
}

fn cq_uint(object: &Map<String, Value>, key: &str, ctx: &str) -> Result<u64, String> {
    cq_req(object, key, ctx)?
        .as_u64()
        .ok_or_else(|| format!("{ctx}: {key:?} must be a non-negative integer"))
}

fn cq_usize(object: &Map<String, Value>, key: &str, ctx: &str) -> Result<usize, String> {
    usize::try_from(cq_uint(object, key, ctx)?)
        .map_err(|_| format!("{ctx}: {key:?} does not fit usize"))
}

/// A case/arm name: it becomes part of a sidecar file name.
fn cq_name(object: &Map<String, Value>, key: &str, ctx: &str) -> Result<String, String> {
    let name = cq_string(object, key, ctx)?;
    let valid = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if !valid {
        return Err(format!(
            "{ctx}: {key:?} {name:?} must be non-empty, not start with '.', and use only [A-Za-z0-9_.-]"
        ));
    }
    Ok(name)
}

fn cq_parse_stream(name: &str, value: &Value) -> Result<CqStream, String> {
    let ctx = format!("stream {name:?}");
    let probe = value
        .as_object()
        .ok_or_else(|| format!("{ctx}: expected a JSON object"))?;
    if probe.contains_key("ids_path") {
        let object = cq_obj(value, &ctx, &["ids_path", "sha256"])?;
        let path = PathBuf::from(cq_string(object, "ids_path", &ctx)?);
        if !path.is_absolute() {
            return Err(format!(
                "{ctx}: ids_path {} is not an absolute path",
                path.display()
            ));
        }
        let sha256 = cq_string(object, "sha256", &ctx)?.to_ascii_lowercase();
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("{ctx}: sha256 must be 64 hex digits"));
        }
        return Ok(CqStream::File { path, sha256 });
    }
    if probe.contains_key("synthetic_salt") {
        let object = cq_obj(
            value,
            &ctx,
            &["synthetic_salt", "len", "share_prefix_from", "share_len"],
        )?;
        let salt = cq_uint(object, "synthetic_salt", &ctx)?;
        let len = cq_usize(object, "len", &ctx)?;
        if len == 0 {
            return Err(format!("{ctx}: len must be positive"));
        }
        return match (
            object.contains_key("share_prefix_from"),
            object.contains_key("share_len"),
        ) {
            (false, false) => Ok(CqStream::Synthetic { salt, len }),
            (true, true) => {
                let base = cq_string(object, "share_prefix_from", &ctx)?;
                let share_len = cq_usize(object, "share_len", &ctx)?;
                if share_len == 0 || share_len > len {
                    return Err(format!(
                        "{ctx}: share_len {share_len} must be in 1..=len ({len})"
                    ));
                }
                Ok(CqStream::Branch {
                    salt,
                    len,
                    base,
                    share_len,
                })
            }
            _ => Err(format!(
                "{ctx}: share_prefix_from and share_len must be given together"
            )),
        };
    }
    Err(format!(
        "{ctx}: needs either ids_path+sha256 or synthetic_salt+len"
    ))
}

fn cq_stream_ref(
    object: &Map<String, Value>,
    ctx: &str,
    streams: &BTreeMap<String, CqStream>,
) -> Result<String, String> {
    let stream = cq_string(object, "stream", ctx)?;
    if !streams.contains_key(&stream) {
        return Err(format!("{ctx}: unknown stream {stream:?}"));
    }
    Ok(stream)
}

fn cq_parse_arm(value: &Value, ctx: &str) -> Result<CqArm, String> {
    let object = cq_obj(
        value,
        ctx,
        &["name", "kind", "cache_mode", "expect", "expect_start", "label"],
    )?;
    let name = cq_name(object, "name", ctx)?;
    let ctx = format!("{ctx} arm {name:?}");
    let label = match object.get("label") {
        None => None,
        Some(_) => Some(cq_string(object, "label", &ctx)?),
    };
    let kind = cq_string(object, "kind", &ctx)?;
    match kind.as_str() {
        "hit" => {
            // The session cache has one policy: a hit arm either omits
            // `cache_mode` or names it "default". The RC3 prefix-cache modes
            // no longer exist, so a fixture written for them is refused
            // instead of silently measuring something else.
            if let Some(value) = object.get("cache_mode") {
                match value.as_str() {
                    Some("default") => {}
                    Some(mode @ ("cold_exact" | "session")) => {
                        return Err(format!(
                            "{ctx}: cache_mode {mode:?} belongs to the RC3 prefix/radix cache and is not supported by the session-cache oracle; omit cache_mode or use \"default\""
                        ))
                    }
                    Some(other) => {
                        return Err(format!(
                            "{ctx}: unknown cache_mode {other:?} (expected \"default\" or omitted)"
                        ))
                    }
                    None => {
                        return Err(format!(
                            "{ctx}: unknown cache_mode {value} (expected \"default\" or omitted)"
                        ))
                    }
                }
            }
            let expect = match object.get("expect") {
                None => CqExpect::None,
                Some(value) => match value.as_str() {
                    Some("none") => CqExpect::None,
                    Some("bitwise_equal_cold") => CqExpect::BitwiseEqualCold,
                    _ => {
                        return Err(format!(
                            "{ctx}: unknown expect {value} (expected none or bitwise_equal_cold)"
                        ))
                    }
                },
            };
            let expect_start = if object.contains_key("expect_start") {
                Some(cq_usize(object, "expect_start", &ctx)?)
            } else {
                None
            };
            if expect_start.is_some() && expect == CqExpect::None {
                return Err(format!(
                    "{ctx}: expect_start needs expect \"bitwise_equal_cold\""
                ));
            }
            Ok(CqArm {
                name,
                kind: CqArmKind::Hit,
                expect,
                expect_start,
                label,
            })
        }
        "cold" => {
            for key in ["cache_mode", "expect", "expect_start"] {
                if object.contains_key(key) {
                    return Err(format!("{ctx}: {key:?} is not allowed on a cold arm"));
                }
            }
            Ok(CqArm {
                name,
                kind: CqArmKind::Cold,
                expect: CqExpect::None,
                expect_start: None,
                label,
            })
        }
        other => Err(format!("{ctx}: unknown kind {other:?} (expected hit or cold)")),
    }
}

fn cq_parse_case(
    value: &Value,
    index: usize,
    streams: &BTreeMap<String, CqStream>,
) -> Result<CqCase, String> {
    let ctx = format!("case #{index}");
    let object = cq_obj(value, &ctx, &["name", "turns", "final", "arms"])?;
    let name = cq_name(object, "name", &ctx)?;
    let ctx = format!("case {name:?}");
    let mut turns = Vec::new();
    for (i, turn) in cq_req(object, "turns", &ctx)?
        .as_array()
        .ok_or_else(|| format!("{ctx}: turns must be an array"))?
        .iter()
        .enumerate()
    {
        let ctx = format!("{ctx} turn {i}");
        let o = cq_obj(turn, &ctx, &["stream", "prompt_end", "consumed_end"])?;
        let stream = cq_stream_ref(o, &ctx, streams)?;
        let prompt_end = cq_usize(o, "prompt_end", &ctx)?;
        let consumed_end = cq_usize(o, "consumed_end", &ctx)?;
        if prompt_end == 0 || prompt_end > consumed_end {
            return Err(format!(
                "{ctx}: needs 0 < prompt_end ({prompt_end}) <= consumed_end ({consumed_end})"
            ));
        }
        turns.push(CqTurn {
            stream,
            prompt_end,
            consumed_end,
        });
    }
    let fctx = format!("{ctx} final");
    let fo = cq_obj(
        cq_req(object, "final", &ctx)?,
        &fctx,
        &["stream", "prompt_end", "score_rows"],
    )?;
    let fin = CqFinal {
        stream: cq_stream_ref(fo, &fctx, streams)?,
        prompt_end: cq_usize(fo, "prompt_end", &fctx)?,
        score_rows: cq_usize(fo, "score_rows", &fctx)?,
    };
    if fin.prompt_end == 0 {
        return Err(format!("{fctx}: prompt_end must be >= 1"));
    }
    let mut arms: Vec<CqArm> = Vec::new();
    for arm in cq_req(object, "arms", &ctx)?
        .as_array()
        .ok_or_else(|| format!("{ctx}: arms must be an array"))?
    {
        let arm = cq_parse_arm(arm, &ctx)?;
        if arms.iter().any(|other| other.name == arm.name) {
            return Err(format!("{ctx}: duplicate arm name {:?}", arm.name));
        }
        arms.push(arm);
    }
    if arms.is_empty() {
        return Err(format!("{ctx}: needs at least one arm"));
    }
    let colds = arms.iter().filter(|arm| arm.kind == CqArmKind::Cold).count();
    if colds > 1 {
        return Err(format!(
            "{ctx}: {colds} cold arms (the comparison reference must be unique)"
        ));
    }
    if colds == 0 && arms.iter().any(|arm| arm.expect != CqExpect::None) {
        return Err(format!(
            "{ctx}: an arm expects equality with cold but the case has no cold arm"
        ));
    }
    Ok(CqCase {
        name,
        turns,
        fin,
        arms,
    })
}

/// Strict structural parse of a `hipfire.qwen4.cache_quality_corpus.v1` fixture.
/// Lengths and file contents are checked by [`cq_resolve_streams`] and
/// [`cq_validate_cases`].
fn cq_parse_fixture(text: &str) -> Result<CqFixture, String> {
    let root: Value =
        serde_json::from_str(text).map_err(|error| format!("fixture is not valid JSON: {error}"))?;
    let top = cq_obj(
        &root,
        "fixture",
        &[
            "schema",
            "scenario",
            "id_provenance",
            "provenance",
            "mode",
            "turn_end_id",
            "streams",
            "cases",
        ],
    )?;
    let schema = cq_string(top, "schema", "fixture")?;
    if schema != CQ_FIXTURE_SCHEMA {
        return Err(format!(
            "fixture: schema {schema:?} is not {CQ_FIXTURE_SCHEMA:?}"
        ));
    }
    let scenario = cq_string(top, "scenario", "fixture")?;
    let id_provenance = cq_string(top, "id_provenance", "fixture")?;
    if !["captured", "reconstructed", "synthetic"].contains(&id_provenance.as_str()) {
        return Err(format!(
            "fixture: unknown id_provenance {id_provenance:?} (expected captured, reconstructed or synthetic)"
        ));
    }
    let provenance = cq_req(top, "provenance", "fixture")?.clone();
    if !provenance.is_object() {
        return Err("fixture: provenance must be a JSON object".to_string());
    }
    let mode = match cq_string(top, "mode", "fixture")?.as_str() {
        "native_mtp" => CqMode::NativeMtp,
        "ar" => CqMode::Ar,
        other => {
            return Err(format!(
                "fixture: unknown mode {other:?} (expected native_mtp or ar)"
            ))
        }
    };
    let turn_end_id = match top.get("turn_end_id") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or("fixture: turn_end_id must be a u32 or null")?,
        ),
    };
    let stream_values = cq_req(top, "streams", "fixture")?
        .as_object()
        .ok_or("fixture: streams must be an object")?;
    if stream_values.is_empty() {
        return Err("fixture: streams is empty".to_string());
    }
    let mut streams = BTreeMap::new();
    for (name, value) in stream_values {
        streams.insert(name.clone(), cq_parse_stream(name, value)?);
    }
    for (name, spec) in &streams {
        if let CqStream::Branch { base, .. } = spec {
            match streams.get(base) {
                None => {
                    return Err(format!(
                        "stream {name:?}: share_prefix_from {base:?} is not a stream"
                    ))
                }
                Some(CqStream::Branch { .. }) => {
                    return Err(format!(
                        "stream {name:?}: share_prefix_from {base:?} must not itself be a branch"
                    ))
                }
                Some(_) => {}
            }
        }
    }
    let mut cases: Vec<CqCase> = Vec::new();
    for (index, value) in cq_req(top, "cases", "fixture")?
        .as_array()
        .ok_or("fixture: cases must be an array")?
        .iter()
        .enumerate()
    {
        let case = cq_parse_case(value, index, &streams)?;
        if cases.iter().any(|other| other.name == case.name) {
            return Err(format!("fixture: duplicate case name {:?}", case.name));
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err("fixture: cases is empty".to_string());
    }
    Ok(CqFixture {
        scenario,
        id_provenance,
        provenance,
        mode,
        turn_end_id,
        streams,
        cases,
    })
}

fn cq_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn cq_sha256_hex(bytes: &[u8]) -> String {
    cq_hex(&Sha256::digest(bytes))
}

/// Deterministic spread over the real vocabulary, never the EOS id: the RC3
/// `SessionRig::tokens` generator (same LCG, same streams for the same salt).
fn cq_synth_tokens(vocab: usize, eos: u32, salt: u64, count: usize) -> Vec<u32> {
    let span = vocab as u64 - 10;
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ salt.wrapping_mul(0xff51_afd7_ed55_8ccd);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let token = (10 + (state >> 33) % span) as u32;
        if token != eos {
            out.push(token);
        }
    }
    out
}

/// Materialize every stream: file ids are sha256-verified little-endian u32s,
/// synthetic streams come from `synth(salt, len)` ([`cq_synth_tokens`]).
fn cq_resolve_streams(
    fixture: &CqFixture,
    read_file: &dyn Fn(&Path) -> Result<Vec<u8>, String>,
    synth: &dyn Fn(u64, usize) -> Vec<u32>,
) -> Result<BTreeMap<String, Vec<u32>>, String> {
    let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for (name, spec) in &fixture.streams {
        match spec {
            CqStream::File { path, sha256 } => {
                let bytes = read_file(path)?;
                let actual = cq_sha256_hex(&bytes);
                if actual != *sha256 {
                    return Err(format!(
                        "stream {name:?}: sha256 mismatch for {}: file is {actual}, fixture says {sha256}",
                        path.display()
                    ));
                }
                if bytes.is_empty() || bytes.len() % 4 != 0 {
                    return Err(format!(
                        "stream {name:?}: {} is {} bytes, not a positive multiple of 4",
                        path.display(),
                        bytes.len()
                    ));
                }
                let ids = bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                out.insert(name.clone(), ids);
            }
            CqStream::Synthetic { salt, len } => {
                let ids = synth(*salt, *len);
                if ids.len() != *len {
                    return Err(format!(
                        "stream {name:?}: synthetic generator returned {} ids, expected {len}",
                        ids.len()
                    ));
                }
                out.insert(name.clone(), ids);
            }
            CqStream::Branch { .. } => {}
        }
    }
    for (name, spec) in &fixture.streams {
        let CqStream::Branch {
            salt,
            len,
            base,
            share_len,
        } = spec
        else {
            continue;
        };
        let base_ids = out
            .get(base)
            .ok_or_else(|| format!("stream {name:?}: base {base:?} did not resolve"))?;
        if *share_len > base_ids.len() {
            return Err(format!(
                "stream {name:?}: share_len {share_len} exceeds base {base:?} length {}",
                base_ids.len()
            ));
        }
        let mut ids = base_ids[..*share_len].to_vec();
        let tail = synth(*salt, *len - *share_len);
        if tail.len() != *len - *share_len {
            return Err(format!(
                "stream {name:?}: synthetic generator returned {} ids, expected {}",
                tail.len(),
                *len - *share_len
            ));
        }
        ids.extend_from_slice(&tail);
        out.insert(name.clone(), ids);
    }
    Ok(out)
}

/// Bounds of every turn and final request against the resolved streams and
/// the rig's `max_seq`.
fn cq_validate_cases(
    fixture: &CqFixture,
    streams: &BTreeMap<String, Vec<u32>>,
    max_seq: usize,
) -> Result<(), String> {
    let len_of = |ctx: &str, stream: &str| -> Result<usize, String> {
        streams
            .get(stream)
            .map(Vec::len)
            .ok_or_else(|| format!("{ctx}: stream {stream:?} is not resolved"))
    };
    for case in &fixture.cases {
        for (i, turn) in case.turns.iter().enumerate() {
            let ctx = format!("case {:?} turn {i}", case.name);
            let len = len_of(&ctx, &turn.stream)?;
            if !(0 < turn.prompt_end && turn.prompt_end <= turn.consumed_end && turn.consumed_end <= len)
            {
                return Err(format!(
                    "{ctx}: needs 0 < prompt_end ({}) <= consumed_end ({}) <= stream {:?} length ({len})",
                    turn.prompt_end, turn.consumed_end, turn.stream
                ));
            }
        }
        let ctx = format!("case {:?} final", case.name);
        let fin = &case.fin;
        let len = len_of(&ctx, &fin.stream)?;
        let needed = fin
            .prompt_end
            .checked_add(fin.score_rows)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| format!("{ctx}: prompt_end + score_rows overflows"))?;
        if fin.prompt_end < 1 || needed > len || needed >= max_seq {
            return Err(format!(
                "{ctx}: prompt_end ({}) + score_rows ({}) + 1 = {needed} must be <= stream {:?} length ({len}) and < max_seq ({max_seq})",
                fin.prompt_end, fin.score_rows, fin.stream
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Scored row `row` of a final request ending at `prompt_end`: the logits row
/// after consuming `stream[prompt_end - 1 + row]` (target position
/// `prompt_end - 1 + row`), predicting `stream[prompt_end + row]`. Returns
/// `(position, input index, target index)`.
fn cq_row_map(prompt_end: usize, row: usize) -> (usize, usize, usize) {
    (prompt_end - 1 + row, prompt_end - 1 + row, prompt_end + row)
}

fn cq_require_finite(row: &[f32]) -> Result<(), String> {
    if row.len() < 2 {
        return Err(format!("logits row has {} entries", row.len()));
    }
    match row.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(format!("non-finite logit {} at index {index}", row[index])),
        None => Ok(()),
    }
}

/// Index of the first maximum.
fn cq_top1(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in row.iter().enumerate().skip(1) {
        if value > row[best] {
            best = index;
        }
    }
    best
}

fn cq_logsumexp(row: &[f32]) -> f64 {
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &value| acc.max(f64::from(value)));
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = row.iter().map(|&value| (f64::from(value) - max).exp()).sum();
    max + sum.ln()
}

/// `KL(softmax(p) || softmax(q))` in f64 over the full vocabulary. Rows must
/// have equal length (callers check).
fn cq_kl(p: &[f32], q: &[f32]) -> f64 {
    let (lse_p, lse_q) = (cq_logsumexp(p), cq_logsumexp(q));
    p.iter()
        .zip(q)
        .map(|(&a, &b)| {
            let log_p = f64::from(a) - lse_p;
            let log_q = f64::from(b) - lse_q;
            log_p.exp() * (log_p - log_q)
        })
        .sum()
}

#[derive(Clone, Debug, PartialEq)]
struct CqRowStats {
    top1_id: usize,
    top1_logit: f32,
    margin: f64,
    logsumexp: f64,
    target_logprob: f64,
}

fn cq_row_stats(row: &[f32], target_id: u32) -> Result<CqRowStats, String> {
    cq_require_finite(row)?;
    let target = target_id as usize;
    if target >= row.len() {
        return Err(format!(
            "target id {target_id} is outside the {}-entry vocabulary",
            row.len()
        ));
    }
    let top1 = cq_top1(row);
    let mut second = f32::NEG_INFINITY;
    for (index, &value) in row.iter().enumerate() {
        if index != top1 && value > second {
            second = value;
        }
    }
    let logsumexp = cq_logsumexp(row);
    Ok(CqRowStats {
        top1_id: top1,
        top1_logit: row[top1],
        margin: f64::from(row[top1]) - f64::from(second),
        logsumexp,
        target_logprob: f64::from(row[target]) - logsumexp,
    })
}

#[derive(Clone, Debug, PartialEq)]
struct CqRowCompare {
    bitwise_equal: bool,
    max_abs_diff: f64,
    kl_cold_arm: f64,
    kl_arm_cold: f64,
    top1_equal: bool,
}

fn cq_compare_rows(cold: &[f32], arm: &[f32]) -> Result<CqRowCompare, String> {
    if cold.len() != arm.len() {
        return Err(format!(
            "row lengths differ: cold {} vs arm {}",
            cold.len(),
            arm.len()
        ));
    }
    cq_require_finite(cold)?;
    cq_require_finite(arm)?;
    let bitwise_equal = cold
        .iter()
        .zip(arm)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let max_abs_diff = cold
        .iter()
        .zip(arm)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0f64, f64::max);
    Ok(CqRowCompare {
        bitwise_equal,
        max_abs_diff,
        kl_cold_arm: cq_kl(cold, arm),
        kl_arm_cold: cq_kl(arm, cold),
        top1_equal: cq_top1(cold) == cq_top1(arm),
    })
}

/// "cold" when nothing was restored, else "session" (the only reuse source of
/// the engine-owned cache).
fn cq_source_name(start: usize) -> &'static str {
    if start == 0 {
        "cold"
    } else {
        "session"
    }
}

/// Reasons a `bitwise_equal_cold` expectation fails (empty = pass).
fn cq_bitwise_reasons(
    expect_start: Option<usize>,
    selected_start: usize,
    selected_source: &str,
    all_rows_bitwise_equal: bool,
    families_equal: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !all_rows_bitwise_equal {
        reasons.push("logits rows are not all bit-identical to the cold arm".to_string());
    }
    if !families_equal {
        reasons.push("a state part or the pending hidden digest differs from cold".to_string());
    }
    if let Some(start) = expect_start {
        if selected_start != start {
            reasons.push(format!(
                "final selected_start {selected_start} != expect_start {start}"
            ));
        }
        if start > 0 && selected_source == "cold" {
            reasons.push(format!(
                "expect_start {start} > 0 but the final source is cold"
            ));
        }
    }
    reasons
}

/// The oracle's session-cache byte budget: `raw` is the value of
/// [`CQ_BUDGET_ENV`] (`None`/empty = unbounded, `u64::MAX >> 1`).
fn cq_session_budget(raw: Option<&str>) -> Result<u64, String> {
    match raw.map(str::trim).filter(|raw| !raw.is_empty()) {
        None => Ok(u64::MAX >> 1),
        Some(raw) => raw
            .parse::<u64>()
            .map_err(|error| format!("{CQ_BUDGET_ENV}={raw:?} is not a u64: {error}")),
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CqFamily {
    pub(crate) digest: String,
    pub(crate) bytes: u64,
}

fn cq_family_of(bytes: &[u8]) -> CqFamily {
    CqFamily {
        digest: cq_sha256_hex(bytes),
        bytes: bytes.len() as u64,
    }
}

fn cq_family_json(family: &CqFamily) -> Value {
    json!({"digest": family.digest, "bytes": family.bytes})
}

struct CqRow {
    row: usize,
    position: usize,
    input_id: u32,
    target_id: u32,
    stats: CqRowStats,
    sidecar_offset: u64,
}

struct CqSidecarInfo {
    path: PathBuf,
    sha256: String,
    rows: usize,
    vocab: usize,
}

struct CqSidecarWriter {
    path: PathBuf,
    writer: std::io::BufWriter<std::fs::File>,
    hasher: Sha256,
    rows: usize,
    vocab: usize,
    scratch: Vec<u8>,
}

impl CqSidecarWriter {
    fn create(path: PathBuf, vocab: usize) -> Result<Self, String> {
        let file = std::fs::File::create(&path)
            .map_err(|error| format!("create {}: {error}", path.display()))?;
        Ok(Self {
            path,
            writer: std::io::BufWriter::new(file),
            hasher: Sha256::new(),
            rows: 0,
            vocab,
            scratch: Vec::with_capacity(vocab * 4),
        })
    }

    /// Append one row; returns its byte offset.
    fn push(&mut self, row: &[f32]) -> Result<u64, String> {
        if row.len() != self.vocab {
            return Err(format!(
                "sidecar row has {} entries, vocabulary is {}",
                row.len(),
                self.vocab
            ));
        }
        let offset = (self.rows as u64) * (self.vocab as u64) * 4;
        self.scratch.clear();
        for value in row {
            self.scratch.extend_from_slice(&value.to_le_bytes());
        }
        self.hasher.update(&self.scratch);
        self.writer
            .write_all(&self.scratch)
            .map_err(|error| format!("write {}: {error}", self.path.display()))?;
        self.rows += 1;
        Ok(offset)
    }

    fn finish(mut self) -> Result<CqSidecarInfo, String> {
        self.writer
            .flush()
            .map_err(|error| format!("flush {}: {error}", self.path.display()))?;
        Ok(CqSidecarInfo {
            path: self.path,
            sha256: cq_hex(&self.hasher.finalize()),
            rows: self.rows,
            vocab: self.vocab,
        })
    }
}

fn cq_read_sidecar_row(
    reader: &mut impl std::io::Read,
    bytes: &mut [u8],
    out: &mut Vec<f32>,
) -> Result<(), String> {
    reader
        .read_exact(bytes)
        .map_err(|error| format!("read sidecar row: {error}"))?;
    out.clear();
    out.extend(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
    );
    Ok(())
}

/// Row-by-row comparison of two sidecars through two reused host rows.
fn cq_compare_sidecars(
    cold: &CqSidecarInfo,
    arm: &CqSidecarInfo,
) -> Result<Vec<CqRowCompare>, String> {
    if cold.rows != arm.rows || cold.vocab != arm.vocab {
        return Err(format!(
            "sidecar shapes differ: cold {}x{} vs arm {}x{}",
            cold.rows, cold.vocab, arm.rows, arm.vocab
        ));
    }
    let open = |path: &Path| {
        std::fs::File::open(path)
            .map(std::io::BufReader::new)
            .map_err(|error| format!("open {}: {error}", path.display()))
    };
    let (mut a, mut b) = (open(&cold.path)?, open(&arm.path)?);
    let mut bytes = vec![0u8; cold.vocab * 4];
    let mut x = Vec::with_capacity(cold.vocab);
    let mut y = Vec::with_capacity(cold.vocab);
    let mut out = Vec::with_capacity(cold.rows);
    for row in 0..cold.rows {
        cq_read_sidecar_row(&mut a, &mut bytes, &mut x)?;
        cq_read_sidecar_row(&mut b, &mut bytes, &mut y)?;
        out.push(cq_compare_rows(&x, &y).map_err(|error| format!("row {row}: {error}"))?);
    }
    Ok(out)
}

/// What one arm measured.
struct CqArmRun {
    turns: Vec<Value>,
    selected_start: usize,
    selected_source: String,
    target_position: usize,
    head_position: Option<usize>,
    /// `part.<i>` -> digest of every `snapshot_parts` part of the final prefill.
    families: BTreeMap<String, CqFamily>,
    pending: Option<CqFamily>,
    session_stored_bytes: u64,
    rows: Vec<CqRow>,
    sidecar: CqSidecarInfo,
}

fn cq_stream<'a>(
    streams: &'a BTreeMap<String, Vec<u32>>,
    name: &str,
) -> Result<&'a [u32], String> {
    streams
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("stream {name:?} is not resolved"))
}

fn cq_push_row(
    sidecar: &mut CqSidecarWriter,
    rows: &mut Vec<CqRow>,
    host: &[f32],
    stream: &[u32],
    prompt_end: usize,
    row: usize,
) -> Result<(), String> {
    let (position, input_index, target_index) = cq_row_map(prompt_end, row);
    let target_id = stream[target_index];
    let stats = cq_row_stats(host, target_id).map_err(|error| format!("row {row}: {error}"))?;
    let sidecar_offset = sidecar.push(host)?;
    rows.push(CqRow {
        row,
        position,
        input_id: stream[input_index],
        target_id,
        stats,
        sidecar_offset,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Rig: one bundle with an attached SessionCache
// ---------------------------------------------------------------------------

/// Explicit host-mapped routed-expert placement for an oracle rig on a discrete
/// GPU that cannot hold every routed expert (Flash-Next on a 32 GB R9700): with
/// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` the routed experts of trunk layers at
/// or past `N` (and the head's) live in pinned host RAM, as the loader places
/// them; HIP's runtime load then keeps that memory out of reclaim. Unset keeps
/// every weight resident. `auto` is refused: the oracle must name its
/// placement so arms and runs are comparable.
fn place_rig_experts(weights: &mut [WeightEntry], gpu: &Gpu) -> Result<(), String> {
    use crate::expert_residency::{self as residency, ExpertVramLayers};
    match residency::expert_vram_layers_from_env()? {
        None => Ok(()),
        Some(ExpertVramLayers::Auto) => Err(format!(
            "{}=auto is not supported by the oracle rigs; give an explicit layer count",
            residency::EXPERT_VRAM_LAYERS_ENV
        )),
        Some(ExpertVramLayers::Layers(_)) if gpu.is_uma() => Err(format!(
            "{} places experts in host RAM, which only a discrete GPU needs",
            residency::EXPERT_VRAM_LAYERS_ENV
        )),
        Some(ExpertVramLayers::Layers(layers)) => {
            let moved = residency::place_routed_experts(weights, layers);
            eprintln!(
                "oracle rig: routed experts of layers 0..{layers} in VRAM, {moved} expert tensors in pinned host RAM"
            );
            Ok(())
        }
    }
}

/// One fulfillment payload of the rig loader: a slab layer arena is one span of
/// the container; a host-mapped entry (which the store rejects as a range) is
/// borrowed from the resident source bytes, as the product loader does; every
/// other tensor streams as a bounded range.
fn qwen4_rig_payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    recipe: &crate::weights::Qwen4TierRecipe,
    entry: &WeightEntry,
) -> Result<hipfire_runtime::model_source::SourcePayload<'a>, String> {
    use hipfire_runtime::model_source::SourcePayload;
    if let Some(payload) = crate::expert_arena::recipe_arena_payload(source, recipe, entry) {
        return payload;
    }
    if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
        let (info, bytes) = source
            .tensor_data(&entry.name)
            .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
        return Ok(SourcePayload::Borrowed { info, bytes });
    }
    source
        .tensor_range(&entry.name)
        .map_err(|error| error.to_string())?
        .map(SourcePayload::Range)
        .ok_or_else(|| format!("missing tensor '{}'", entry.name))
}

/// One loaded bundle (forward, the MTP head when the fixture is native, and
/// the session cache) driven through sequential arms.
struct CqRig {
    bundle: Qwen4Bundle,
    logits: GpuTensor,
    max_seq: usize,
    vocab: usize,
    eos: u32,
    /// `spec_chunk_rows()`: the session cache's boundary stride.
    chunk: usize,
    /// Live VMM granule bytes before the load.
    baseline_granule_bytes: usize,
    content_digest: Vec<u8>,
    state_format: String,
}

/// Load one bundle with the shipped state formats, the product's default
/// context and the automatic KV backend: forward, MTP head (native fixtures
/// only), then the engine-owned session cache over the production
/// `CacheDomain`.
fn cq_load_rig(gpu: &mut Gpu, model_path: &Path, mode: CqMode) -> Result<CqRig, String> {
    let budget = match std::env::var(CQ_BUDGET_ENV) {
        Ok(raw) => cq_session_budget(Some(raw.as_str()))?,
        Err(std::env::VarError::NotPresent) => cq_session_budget(None)?,
        Err(error) => return Err(format!("{CQ_BUDGET_ENV}: {error}")),
    };
    let baseline_granule_bytes = hip_bridge::vmm_live_granule_bytes();
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("tokenizer from {}: {error}", model_path.display()))?;
    let receipt = crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let mut manifest = receipt.manifest.clone();
    place_rig_experts(&mut manifest.weights, gpu)?;
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let content_digest = hfq.content_digest();
    let domain = CacheDomain::for_model(&hfq, &tokenizer, None, "qwen4", gpu.device_id);
    // The loader's forward scratch plan, decided from the actual payload
    // while the mapping is readable.
    let scratch_mode =
        crate::artifact::source_forward_scratch_mode(&*gpu, &hfq, &config, &manifest)
            .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        gpu,
        expected,
        |entry| qwen4_rig_payload(&source, &manifest.identity.recipe, entry),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let state_format = crate::state::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        gpu,
        &config,
    )?;
    let backend = Qwen4KvBackend::automatic(gpu);
    let max_seq = crate::QWEN4_DEFAULT_CONTEXT.min(config.max_position_embeddings);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        &manifest.identity.recipe,
        transaction,
        &placements,
        gpu,
        max_seq,
        metadata,
        state_format,
        backend,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let attach = (|| -> Result<usize, String> {
        bundle
            .attach_forward(gpu, max_seq, scratch_mode)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        if mode == CqMode::NativeMtp {
            bundle
                .attach_mtp(gpu, max_seq)
                .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        }
        bundle.attach_session_cache(SessionCache::new(domain, budget));
        if bundle.session_cache().is_none() {
            return Err("qwen4 session cache is not attached".to_string());
        }
        bundle
            .spec_chunk_rows()
            .ok_or_else(|| "qwen4 forward resources are not attached".to_string())
    })();
    let chunk = match attach {
        Ok(chunk) => chunk,
        Err(error) => {
            let _ = bundle.free_gpu(gpu);
            return Err(error);
        }
    };
    let logits = match gpu.zeros(&[config.vocab_size], DType::F32) {
        Ok(logits) => logits,
        Err(error) => {
            let _ = bundle.free_gpu(gpu);
            return Err(format!("allocate session logits: {error}"));
        }
    };
    Ok(CqRig {
        bundle,
        logits,
        max_seq,
        vocab: config.vocab_size,
        eos: config.eos_token_id,
        chunk,
        baseline_granule_bytes,
        content_digest,
        state_format: format!("{state_format:?}"),
    })
}

/// Free a loaded bundle (its session cache with it) and require the process's
/// live VMM granule bytes to return to their pre-load value.
fn cq_unload_rig(gpu: &mut Gpu, rig: CqRig) -> Result<(), String> {
    let baseline = rig.baseline_granule_bytes;
    let CqRig { bundle, logits, .. } = rig;
    let logits_cleanup = gpu
        .free_tensor(logits)
        .err()
        .map(|error| format!("free session logits: {error}"));
    let bundle_cleanup = bundle
        .free_gpu(gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    if let Some(error) = logits_cleanup.or(bundle_cleanup) {
        return Err(error);
    }
    let _ = hip_bridge::retry_pending_granule_releases(&gpu.hip);
    let live = hip_bridge::vmm_live_granule_bytes();
    if live != baseline {
        return Err(format!(
            "unload left {live} live VMM granule bytes, {baseline} before the load"
        ));
    }
    Ok(())
}

/// Plan against the session cache. The first plan of an arm must be 0: anything
/// else is cache state that survived `session_clear`.
fn cq_plan(
    rig: &CqRig,
    route: SessionRoute,
    prompt: &[u32],
    first: &mut bool,
) -> Result<usize, String> {
    let start = rig.bundle.session_plan(prompt, route);
    if std::mem::replace(first, false) && start != 0 {
        return Err(format!(
            "cache state leaked between arms: first plan was {start}, expected 0 after session_clear"
        ));
    }
    if start >= prompt.len() {
        return Err(format!(
            "session_plan returned {start} for a {}-token prompt",
            prompt.len()
        ));
    }
    Ok(start)
}

/// Production prefill of `prompt` restoring `start` cached tokens (qwen.rs's two
/// routes). Native MTP returns the greedy seed.
fn cq_prefill(
    rig: &mut CqRig,
    gpu: &mut Gpu,
    drafter: &mut Option<Qwen4MtpDrafter>,
    mode: CqMode,
    prompt: &[u32],
    start: usize,
) -> Result<Option<u32>, String> {
    match mode {
        CqMode::Ar => {
            rig.bundle
                .prefill_final(gpu, prompt, start, &rig.logits)
                .map_err(|error| error.to_string())?;
            Ok(None)
        }
        CqMode::NativeMtp => {
            let drafter = drafter
                .as_mut()
                .ok_or("native MTP drafter is not allocated")?;
            let seed = drafter
                .mtp_prefill(
                    gpu,
                    &mut rig.bundle,
                    prompt,
                    &prompt[start..],
                    start,
                    start > 0,
                    &|| false,
                )?
                .into_first_token()?;
            Ok(Some(seed))
        }
    }
}

/// Teacher-force `tokens` from `start_pos` as the decode lineage.
fn cq_force(
    rig: &mut CqRig,
    gpu: &mut Gpu,
    drafter: &mut Option<Qwen4MtpDrafter>,
    mode: CqMode,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(), String> {
    match mode {
        CqMode::Ar => {
            for &token in tokens {
                rig.bundle
                    .forward_token(gpu, token, &rig.logits, None)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        CqMode::NativeMtp => {
            let drafter = drafter
                .as_mut()
                .ok_or("native MTP drafter is not allocated")?;
            if drafter.mtp_forced_advance(gpu, &mut rig.bundle, tokens, start_pos, &|| false)? {
                Ok(())
            } else {
                Err("mtp_forced_advance declined the teacher-forced tokens".to_string())
            }
        }
    }
}

fn cq_expect_positions(
    rig: &CqRig,
    mode: CqMode,
    expected: usize,
    what: &str,
) -> Result<(), String> {
    let target = rig.bundle.state.position;
    if target != expected {
        return Err(format!("{what}: target position {target}, expected {expected}"));
    }
    if mode == CqMode::NativeMtp {
        let head = rig.bundle.mtp_position().map_err(|error| error.to_string())?;
        if head != expected {
            return Err(format!("{what}: MTP head position {head}, expected {expected}"));
        }
    }
    Ok(())
}

/// Row 0: the final prompt row. Native MTP's final-only target forward
/// (`spec_prefill_rows` with `Qwen4OutputRows::Final`) writes its single
/// output row to row 0 of `spec_logits`; the head appends and the cache's
/// boundary captures that follow never write `spec_logits`, so it is read right
/// after `mtp_prefill`. AR's `prefill_final` leaves it in the rig logits tensor.
fn cq_read_final_row(
    rig: &CqRig,
    gpu: &Gpu,
    mode: CqMode,
    seed: Option<u32>,
    host: &mut Vec<f32>,
) -> Result<(), String> {
    match mode {
        CqMode::Ar => gpu
            .download_f32_into(&rig.logits, host)
            .map_err(|error| error.to_string()),
        CqMode::NativeMtp => {
            rig.bundle
                .spec_row_logits(gpu, 0, host)
                .map_err(|error| error.to_string())?;
            let seed = seed.ok_or("native MTP prefill returned no seed")?;
            let value = *host
                .get(seed as usize)
                .ok_or_else(|| format!("seed {seed} is outside the logits row"))?;
            let max = host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if value != max {
                return Err(format!(
                    "greedy seed {seed} has logit {value}, not the row maximum {max}: spec_logits row 0 is not the final prompt row"
                ));
            }
            Ok(())
        }
    }
}

/// SHA-256 of every part of the live state at `position` on `route`
/// (`SessionState::snapshot_parts`, as the session-cache hardware test's
/// `state_digests`): part 0 is the metadata, then every fixed part, then the
/// valid rows of every row stream. Keys are `part.<i>`.
pub(crate) fn cq_state_families(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    route: SessionRoute,
    position: usize,
) -> Result<BTreeMap<String, CqFamily>, String> {
    let SnapshotParts { meta, layout } = SessionState::snapshot_parts(bundle, route, position)?;
    gpu.hip.device_synchronize().map_err(|error| error.to_string())?;
    let mut families = BTreeMap::new();
    families.insert("part.0".to_string(), cq_family_of(&meta));
    let ranges = layout
        .fixed
        .iter()
        .map(|part| (part.buf, part.offset, part.bytes))
        .chain(
            layout
                .rows
                .iter()
                .map(|stream| (stream.buf, 0, stream.rows * stream.row_bytes)),
        );
    for (index, (buf, offset, bytes)) in ranges.enumerate() {
        let mut host = vec![0u8; bytes];
        if bytes > 0 {
            gpu.hip
                .memcpy_dtoh_at(&mut host, buf, offset)
                .map_err(|error| format!("download state part {}: {error}", index + 1))?;
        }
        families.insert(format!("part.{}", index + 1), cq_family_of(&host));
    }
    Ok(families)
}

/// Digest of the native drafter's pending target-hidden row.
fn cq_pending_family(gpu: &mut Gpu, drafter: &Qwen4MtpDrafter) -> Result<CqFamily, String> {
    let hidden = drafter.pending_hidden_for_parity()?;
    gpu.hip.device_synchronize().map_err(|error| error.to_string())?;
    let mut bytes = vec![0u8; hidden.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &hidden.buf)
        .map_err(|error| error.to_string())?;
    Ok(cq_family_of(&bytes))
}

fn cq_stored_bytes(rig: &CqRig) -> u64 {
    rig.bundle
        .session_cache()
        .map_or(0, SessionCache::stored_bytes)
}

#[allow(clippy::too_many_arguments)]
fn cq_arm_body(
    gpu: &mut Gpu,
    rig: &mut CqRig,
    drafter: &mut Option<Qwen4MtpDrafter>,
    fixture: &CqFixture,
    streams: &BTreeMap<String, Vec<u32>>,
    case: &CqCase,
    arm: &CqArm,
    sidecar_path: PathBuf,
    host: &mut Vec<f32>,
) -> Result<CqArmRun, String> {
    let mode = fixture.mode;
    let route = mode.route();
    let hit = arm.kind == CqArmKind::Hit;
    let vocab = rig.vocab;
    let mut first_plan = true;
    let mut turns = Vec::new();
    let prior: &[CqTurn] = if hit { case.turns.as_slice() } else { &[] };
    for (index, turn) in prior.iter().enumerate() {
        let stream = cq_stream(streams, &turn.stream)?;
        let prompt = &stream[..turn.prompt_end];
        let start = cq_plan(rig, route, prompt, &mut first_plan)?;
        cq_prefill(rig, gpu, drafter, mode, prompt, start)
            .map_err(|error| format!("turn {index} prefill: {error}"))?;
        cq_expect_positions(rig, mode, turn.prompt_end, &format!("turn {index} prefill"))?;
        cq_force(
            rig,
            gpu,
            drafter,
            mode,
            &stream[turn.prompt_end..turn.consumed_end],
            turn.prompt_end,
        )
        .map_err(|error| format!("turn {index} teacher-forced decode: {error}"))?;
        cq_expect_positions(
            rig,
            mode,
            turn.consumed_end,
            &format!("turn {index} teacher-forced decode"),
        )?;
        rig.bundle.session_commit();
        turns.push(json!({
            "turn": index,
            "stream": turn.stream,
            "prompt_len": turn.prompt_end,
            "local_start": start,
            "selected_start": start,
            "selected_source": cq_source_name(start),
            "consumed": turn.consumed_end - turn.prompt_end,
            "consumed_end": turn.consumed_end,
            "session_stored_bytes": cq_stored_bytes(rig),
        }));
    }
    let fin = &case.fin;
    let stream = cq_stream(streams, &fin.stream)?;
    let prompt = &stream[..fin.prompt_end];
    let start = cq_plan(rig, route, prompt, &mut first_plan)?;
    let seed = cq_prefill(rig, gpu, drafter, mode, prompt, start)
        .map_err(|error| format!("final prefill: {error}"))?;
    cq_expect_positions(rig, mode, fin.prompt_end, "final prefill")?;
    // Row 0 first: nothing may run on the target between the prefill and this
    // read.
    cq_read_final_row(rig, gpu, mode, seed, host)?;
    let mut sidecar = CqSidecarWriter::create(sidecar_path, vocab)?;
    let mut rows = Vec::with_capacity(fin.score_rows + 1);
    cq_push_row(&mut sidecar, &mut rows, host, stream, fin.prompt_end, 0)?;
    let families = cq_state_families(gpu, &mut rig.bundle, route, fin.prompt_end)?;
    let pending = match drafter.as_ref() {
        Some(drafter) => Some(cq_pending_family(gpu, drafter)?),
        None => None,
    };
    let target_position = rig.bundle.state.position;
    let head_position = match mode {
        CqMode::Ar => None,
        CqMode::NativeMtp => Some(
            rig.bundle
                .mtp_position()
                .map_err(|error| error.to_string())?,
        ),
    };
    let session_stored_bytes = cq_stored_bytes(rig);
    for row in 1..=fin.score_rows {
        // Row `row` consumes stream[prompt_end - 1 + row].
        let token = stream[fin.prompt_end + row - 1];
        rig.bundle
            .forward_token(gpu, token, &rig.logits, None)
            .map_err(|error| format!("row {row} forward: {error}"))?;
        gpu.download_f32_into(&rig.logits, host)
            .map_err(|error| error.to_string())?;
        cq_push_row(&mut sidecar, &mut rows, host, stream, fin.prompt_end, row)?;
    }
    let sidecar = sidecar.finish()?;
    Ok(CqArmRun {
        turns,
        selected_start: start,
        selected_source: cq_source_name(start).to_string(),
        target_position,
        head_position,
        families,
        pending,
        session_stored_bytes,
        rows,
        sidecar,
    })
}

/// One arm from a clean cache: `session_clear`, a fresh native drafter (freed
/// even on error), then [`cq_arm_body`].
#[allow(clippy::too_many_arguments)]
fn cq_run_arm(
    gpu: &mut Gpu,
    rig: &mut CqRig,
    fixture: &CqFixture,
    streams: &BTreeMap<String, Vec<u32>>,
    case: &CqCase,
    arm: &CqArm,
    sidecar_path: PathBuf,
    host: &mut Vec<f32>,
) -> Result<CqArmRun, String> {
    rig.bundle.session_clear(gpu);
    let mut drafter = (fixture.mode == CqMode::NativeMtp)
        .then(|| Qwen4MtpDrafter::new(CQ_MTP_K, rig.max_seq, None));
    let result = cq_arm_body(
        gpu,
        rig,
        &mut drafter,
        fixture,
        streams,
        case,
        arm,
        sidecar_path,
        host,
    );
    if let Some(drafter) = drafter {
        MtpDrafter::mtp_free(Box::new(drafter), gpu);
    }
    result
}

/// Comparison of one arm against the case's cold arm.
struct CqComparison {
    rows: Vec<CqRowCompare>,
    family_equal: BTreeMap<String, bool>,
    pending_equal: bool,
    all_rows_bitwise_equal: bool,
    families_equal: bool,
    max_kl: f64,
    mean_kl: f64,
    first_mismatch: Value,
}

fn cq_compare_arms(cold: &CqArmRun, arm: &CqArmRun) -> Result<CqComparison, String> {
    let rows = cq_compare_sidecars(&cold.sidecar, &arm.sidecar)?;
    let names = cold
        .families
        .keys()
        .chain(arm.families.keys())
        .collect::<BTreeSet<_>>();
    // A part present in only one arm (unequal part counts) is unequal.
    let family_equal = names
        .into_iter()
        .map(|name| {
            let equal = match (cold.families.get(name), arm.families.get(name)) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            (name.clone(), equal)
        })
        .collect::<BTreeMap<_, _>>();
    let pending_equal = cold.pending == arm.pending;
    let all_rows_bitwise_equal = rows.iter().all(|row| row.bitwise_equal);
    let families_equal = pending_equal && family_equal.values().all(|equal| *equal);
    let max_kl = rows
        .iter()
        .map(|row| row.kl_cold_arm.max(row.kl_arm_cold))
        .fold(0.0f64, f64::max);
    let mean_kl = if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|row| row.kl_cold_arm).sum::<f64>() / rows.len() as f64
    };
    let first_mismatch = if let Some(index) = rows.iter().position(|row| !row.bitwise_equal) {
        json!({
            "kind": "row",
            "row": index,
            "position": cold.rows[index].position,
            "max_abs_diff": rows[index].max_abs_diff,
        })
    } else if let Some((name, _)) = family_equal.iter().find(|(_, equal)| !**equal) {
        json!({"kind": "family", "name": name})
    } else if !pending_equal {
        json!({"kind": "pending"})
    } else {
        Value::Null
    };
    Ok(CqComparison {
        rows,
        family_equal,
        pending_equal,
        all_rows_bitwise_equal,
        families_equal,
        max_kl,
        mean_kl,
        first_mismatch,
    })
}

/// `cache_mode` in the result: hit arms ran the session cache's one policy
/// ("default"); the cold arm keeps the RC3 report's "cold_exact" so existing
/// consumers group it unchanged.
fn cq_arm_json(arm: &CqArm, run: &Result<CqArmRun, String>, expectation: Value) -> Value {
    let mut out = json!({
        "name": arm.name,
        "kind": match arm.kind {
            CqArmKind::Hit => "hit",
            CqArmKind::Cold => "cold",
        },
        "cache_mode": match arm.kind {
            CqArmKind::Hit => "default",
            CqArmKind::Cold => "cold_exact",
        },
        "expectation": expectation,
    });
    if let Some(label) = &arm.label {
        out["label"] = json!(label);
    }
    match run {
        Err(error) => out["error"] = json!(error),
        Ok(run) => {
            out["turns"] = Value::Array(run.turns.clone());
            let families = run
                .families
                .iter()
                .map(|(name, family)| (name.clone(), cq_family_json(family)))
                .collect::<Map<String, Value>>();
            out["final"] = json!({
                "local_start": run.selected_start,
                "selected_start": run.selected_start,
                "selected_source": run.selected_source,
                "target_position": run.target_position,
                "head_position": run.head_position,
                "families": families,
                "pending": run.pending.as_ref().map(cq_family_json),
                "session_stored_bytes": run.session_stored_bytes,
            });
            out["rows"] = Value::Array(
                run.rows
                    .iter()
                    .map(|row| {
                        json!({
                            "row": row.row,
                            "position": row.position,
                            "input_id": row.input_id,
                            "target_id": row.target_id,
                            "top1_id": row.stats.top1_id,
                            "top1_logit": row.stats.top1_logit,
                            "top1_top2_margin": row.stats.margin,
                            "logsumexp": row.stats.logsumexp,
                            "target_logprob": row.stats.target_logprob,
                            "sidecar_offset": row.sidecar_offset,
                        })
                    })
                    .collect(),
            );
            out["sidecar"] = json!({
                "path": run.sidecar.path,
                "sha256": run.sidecar.sha256,
                "rows": run.sidecar.rows,
                "vocab": run.sidecar.vocab,
            });
        }
    }
    out
}

fn cq_comparison_json(arm: &str, cold: &str, comparison: &CqComparison) -> Value {
    json!({
        "arm": arm,
        "versus": cold,
        "rows": comparison
            .rows
            .iter()
            .enumerate()
            .map(|(row, c)| json!({
                "row": row,
                "bitwise_equal": c.bitwise_equal,
                "max_abs_diff": c.max_abs_diff,
                "kl_cold_arm": c.kl_cold_arm,
                "kl_arm_cold": c.kl_arm_cold,
                "top1_equal": c.top1_equal,
            }))
            .collect::<Vec<_>>(),
        "family_equal": comparison.family_equal,
        "pending_equal": comparison.pending_equal,
        "summary": {
            "all_rows_bitwise_equal": comparison.all_rows_bitwise_equal,
            "families_equal": comparison.families_equal,
            "max_kl": comparison.max_kl,
            "mean_kl": comparison.mean_kl,
            "kl_basis": "max_kl: max over rows of both directions; mean_kl: mean of kl_cold_arm",
            "first_mismatch": comparison.first_mismatch,
        },
    })
}

/// Run every arm of `case`, compare against its cold arm, evaluate the
/// expectations. Returns the case JSON and its failure descriptions.
#[allow(clippy::too_many_arguments)]
fn cq_run_case(
    gpu: &mut Gpu,
    rig: &mut CqRig,
    fixture: &CqFixture,
    streams: &BTreeMap<String, Vec<u32>>,
    case: &CqCase,
    sidecar_base: &str,
    host: &mut Vec<f32>,
) -> (Value, Vec<String>) {
    let mut runs: Vec<Result<CqArmRun, String>> = Vec::new();
    for arm in &case.arms {
        let path = PathBuf::from(format!("{sidecar_base}.{}.{}.f32", case.name, arm.name));
        eprintln!("cache-quality: case {} arm {}", case.name, arm.name);
        let run = cq_run_arm(gpu, rig, fixture, streams, case, arm, path, host);
        if let Err(error) = &run {
            eprintln!("cache-quality: case {} arm {} failed: {error}", case.name, arm.name);
        }
        runs.push(run);
    }
    let cold_index = case.arms.iter().position(|arm| arm.kind == CqArmKind::Cold);
    let cold = cold_index.and_then(|index| runs[index].as_ref().ok());
    let mut arm_values = Vec::new();
    let mut comparisons = Vec::new();
    let mut failures = Vec::new();
    for (index, (arm, run)) in case.arms.iter().zip(&runs).enumerate() {
        let mut reasons: Vec<String> = Vec::new();
        let mut comparison: Option<CqComparison> = None;
        match run {
            Err(error) => reasons.push(format!("arm failed: {error}")),
            Ok(run) if Some(index) != cold_index => match cold {
                Some(cold) => match cq_compare_arms(cold, run) {
                    Ok(found) => comparison = Some(found),
                    Err(error) => reasons.push(format!("comparison failed: {error}")),
                },
                None => {
                    if arm.expect != CqExpect::None {
                        reasons.push("cold arm did not run".to_string());
                    }
                }
            },
            Ok(_) => {}
        }
        if let (CqExpect::BitwiseEqualCold, Ok(run), Some(found)) =
            (arm.expect, run, comparison.as_ref())
        {
            reasons.extend(cq_bitwise_reasons(
                arm.expect_start,
                run.selected_start,
                &run.selected_source,
                found.all_rows_bitwise_equal,
                found.families_equal,
            ));
        }
        let passed = reasons.is_empty();
        for reason in &reasons {
            failures.push(format!("{}/{}: {reason}", case.name, arm.name));
        }
        let expectation = json!({
            "expect": match arm.expect {
                CqExpect::None => "none",
                CqExpect::BitwiseEqualCold => "bitwise_equal_cold",
            },
            "expect_start": arm.expect_start,
            "passed": passed,
            "reasons": reasons,
        });
        arm_values.push(cq_arm_json(arm, run, expectation));
        if let (Some(found), Some(cold_index)) = (comparison.as_ref(), cold_index) {
            comparisons.push(cq_comparison_json(&arm.name, &case.arms[cold_index].name, found));
        }
    }
    let value = json!({
        "name": case.name,
        "turns": case.turns.len(),
        "final": {
            "stream": case.fin.stream,
            "prompt_end": case.fin.prompt_end,
            "score_rows": case.fin.score_rows,
        },
        "arms": arm_values,
        "comparisons": comparisons,
    });
    (value, failures)
}

fn cq_env_snapshot() -> Map<String, Value> {
    let mut vars = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy().into_owned();
        if ["HIPFIRE_", "ROCR_", "HIP_"]
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            vars.insert(key, Value::String(value.to_string_lossy().into_owned()));
        }
    }
    vars.into_iter().collect()
}

fn cq_write_report(path: &Path, report: &Value) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", path.display()))
}

/// Hit-vs-cold logits oracle of the Qwen4 session cache. Loads one bundle
/// (forward, MTP head when the fixture is native, `SessionCache` with budget
/// `HIPFIRE_CQ_SESSION_BUDGET_BYTES`), then per fixture case and arm: clears
/// the cache, replays the prior turns through plan/prefill/commit and scores
/// `score_rows + 1` logits rows of the final prompt into a raw f32 sidecar.
/// The result JSON is always written; `Err` is returned afterwards if any
/// expectation failed.
pub fn run_prefix_cache_quality(
    model_path: &Path,
    fixture_path: &Path,
    output_path: &Path,
) -> Result<(), String> {
    for (label, path) in [
        ("model", model_path),
        ("fixture", fixture_path),
        ("output", output_path),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} path {} is not absolute", path.display()));
        }
    }
    let fixture_bytes = std::fs::read(fixture_path)
        .map_err(|error| format!("read {}: {error}", fixture_path.display()))?;
    let fixture_text = std::str::from_utf8(&fixture_bytes)
        .map_err(|error| format!("fixture {} is not UTF-8: {error}", fixture_path.display()))?;
    let fixture = cq_parse_fixture(fixture_text)
        .map_err(|error| format!("fixture {}: {error}", fixture_path.display()))?;
    let fixture_sha = cq_sha256_hex(&fixture_bytes);
    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let model_size = std::fs::metadata(model_path)
        .map_err(|error| format!("stat {}: {error}", model_path.display()))?
        .len();
    let output_text = output_path.to_string_lossy().into_owned();
    let sidecar_base = output_text
        .strip_suffix(".json")
        .unwrap_or(&output_text)
        .to_string();

    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let mut loaded = cq_load_rig(&mut gpu, model_path, fixture.mode)?;
    let run = (|| -> Result<(Map<String, Value>, Vec<String>), String> {
        let rig = &mut loaded;
        if rig.vocab <= 10 {
            return Err(format!("vocabulary of {} entries is too small", rig.vocab));
        }
        let (vocab, eos) = (rig.vocab, rig.eos);
        let streams = cq_resolve_streams(
            &fixture,
            &|path| {
                std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))
            },
            &|salt, len| cq_synth_tokens(vocab, eos, salt, len),
        )?;
        cq_validate_cases(&fixture, &streams, rig.max_seq)?;
        let mut host: Vec<f32> = Vec::with_capacity(rig.vocab);
        let mut cases = Vec::new();
        let mut failures = Vec::new();
        for case in &fixture.cases {
            let (value, case_failures) =
                cq_run_case(&mut gpu, rig, &fixture, &streams, case, &sidecar_base, &mut host);
            cases.push(value);
            failures.extend(case_failures);
        }
        let mut report = Map::new();
        report.insert("schema".into(), json!(CQ_RESULT_SCHEMA));
        report.insert(
            "fixture".into(),
            json!({"path": fixture_path, "sha256": fixture_sha}),
        );
        report.insert("scenario".into(), json!(fixture.scenario));
        report.insert("id_provenance".into(), json!(fixture.id_provenance));
        report.insert("provenance".into(), fixture.provenance.clone());
        report.insert("mode".into(), json!(fixture.mode.name()));
        report.insert(
            "model".into(),
            json!({
                "path": model_path,
                "file_size": model_size,
                "content_digest": cq_hex(&rig.content_digest),
            }),
        );
        report.insert("gpu_arch".into(), json!(gpu.arch));
        report.insert("chunk_rows".into(), json!(rig.chunk));
        report.insert("max_seq".into(), json!(rig.max_seq));
        report.insert("state_format".into(), json!(rig.state_format));
        report.insert("env".into(), Value::Object(cq_env_snapshot()));
        report.insert("cases".into(), Value::Array(cases));
        Ok((report, failures))
    })();
    let unload = cq_unload_rig(&mut gpu, loaded);
    let (mut report, mut failures) = match (run, unload) {
        (Ok(pair), Ok(())) => pair,
        (Ok(pair), Err(error)) => {
            let (report, mut failures) = pair;
            failures.push(format!("unload: {error}"));
            (report, failures)
        }
        (Err(error), Ok(())) => return Err(error),
        (Err(error), Err(unload)) => {
            return Err(format!("{error}; unload also failed: {unload}"))
        }
    };
    let passed = failures.is_empty();
    report.insert("passed".into(), json!(passed));
    if !passed {
        report.insert("failures".into(), json!(failures));
    }
    cq_write_report(output_path, &Value::Object(report))?;
    if passed {
        return Ok(());
    }
    failures.sort();
    Err(format!(
        "cache-quality expectations failed: {}",
        failures.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + 3).collect()
    }

    fn le_bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter().flat_map(|id| id.to_le_bytes()).collect()
    }

    fn fake_synth(salt: u64, len: usize) -> Vec<u32> {
        (0..len).map(|i| (salt as u32) * 100_000 + i as u32).collect()
    }

    fn fixture_json(sha: &str) -> Value {
        json!({
            "schema": CQ_FIXTURE_SCHEMA,
            "scenario": "unit",
            "id_provenance": "synthetic",
            "provenance": {"note": "unit"},
            "mode": "native_mtp",
            "turn_end_id": 248046,
            "streams": {
                "main": {"ids_path": "/abs/main.u32", "sha256": sha},
                "syn": {"synthetic_salt": 17, "len": 16700},
                "branch": {
                    "synthetic_salt": 18, "len": 9000,
                    "share_prefix_from": "syn", "share_len": 4146
                }
            },
            "cases": [{
                "name": "c0",
                "turns": [{"stream": "main", "prompt_end": 100, "consumed_end": 150}],
                "final": {"stream": "main", "prompt_end": 500, "score_rows": 64},
                "arms": [
                    {"name": "hit-bitwise", "kind": "hit",
                     "expect": "bitwise_equal_cold", "expect_start": 8192,
                     "label": "rc4-canonical"},
                    {"name": "hit-default", "kind": "hit", "cache_mode": "default",
                     "expect": "none"},
                    {"name": "cold", "kind": "cold"}
                ]
            }]
        })
    }

    fn main_bytes() -> Vec<u8> {
        le_bytes(&ids(2000))
    }

    fn valid() -> Value {
        fixture_json(&cq_sha256_hex(&main_bytes()))
    }

    fn parse(value: &Value) -> Result<CqFixture, String> {
        cq_parse_fixture(&value.to_string())
    }

    fn resolve(fixture: &CqFixture) -> Result<BTreeMap<String, Vec<u32>>, String> {
        cq_resolve_streams(fixture, &|_| Ok(main_bytes()), &fake_synth)
    }

    fn parse_err(value: &Value) -> String {
        parse(value).expect_err("fixture must be rejected")
    }

    #[test]
    fn valid_fixture_parses_resolves_and_validates() {
        let fixture = parse(&valid()).expect("valid fixture");
        assert_eq!(fixture.mode, CqMode::NativeMtp);
        assert_eq!(fixture.mode.route(), SessionRoute::Mtp);
        assert_eq!(fixture.turn_end_id, Some(248046));
        let case = &fixture.cases[0];
        assert_eq!(case.arms[0].kind, CqArmKind::Hit);
        assert_eq!(case.arms[0].expect, CqExpect::BitwiseEqualCold);
        assert_eq!(case.arms[0].expect_start, Some(8192));
        assert_eq!(case.arms[0].label.as_deref(), Some("rc4-canonical"));
        assert_eq!(case.arms[1].expect, CqExpect::None);
        assert_eq!(case.arms[1].label, None);
        assert_eq!(case.arms[2].kind, CqArmKind::Cold);
        let streams = resolve(&fixture).expect("resolve");
        assert_eq!(streams["main"], ids(2000));
        assert_eq!(streams["syn"].len(), 16700);
        let branch = &streams["branch"];
        assert_eq!(branch.len(), 9000);
        assert_eq!(branch[..4146], streams["syn"][..4146]);
        assert_eq!(branch[4146..], fake_synth(18, 9000 - 4146)[..]);
        cq_validate_cases(&fixture, &streams, 32768).expect("validate");
        let ar = {
            let mut value = valid();
            value["mode"] = json!("ar");
            parse(&value).expect("ar fixture")
        };
        assert_eq!(ar.mode, CqMode::Ar);
        assert_eq!(ar.mode.route(), SessionRoute::Ar);
        assert_eq!((ar.mode.name(), fixture.mode.name()), ("ar", "native_mtp"));
    }

    #[test]
    fn rc3_cache_modes_are_refused_with_a_clear_error() {
        for mode in ["cold_exact", "session"] {
            let mut value = valid();
            value["cases"][0]["arms"][0]["cache_mode"] = json!(mode);
            let error = parse_err(&value);
            assert!(error.contains("session-cache oracle"), "{error}");
            assert!(error.contains(mode), "{error}");
        }
        let mut value = valid();
        value["cases"][0]["arms"][0]["cache_mode"] = json!("warm");
        assert!(parse_err(&value).contains("unknown cache_mode"));
        let mut value = valid();
        value["cases"][0]["arms"][0]["cache_mode"] = json!(3);
        assert!(parse_err(&value).contains("unknown cache_mode"));
        // "default" and an omitted cache_mode are the same arm.
        let mut value = valid();
        value["cases"][0]["arms"][0]["cache_mode"] = json!("default");
        parse(&value).expect("explicit default");
    }

    #[test]
    fn label_is_an_optional_string_on_either_arm_kind() {
        let mut value = valid();
        value["cases"][0]["arms"][2]["label"] = json!("cold-ref");
        let fixture = parse(&value).expect("label on cold");
        assert_eq!(fixture.cases[0].arms[2].label.as_deref(), Some("cold-ref"));
        let mut value = valid();
        value["cases"][0]["arms"][0]["label"] = json!(7);
        assert!(parse_err(&value).contains("label"));
    }

    #[test]
    fn arm_json_echoes_label_and_reports_cache_mode() {
        let hit = CqArm {
            name: "h".to_string(),
            kind: CqArmKind::Hit,
            expect: CqExpect::None,
            expect_start: None,
            label: Some("rc4".to_string()),
        };
        let cold = CqArm {
            name: "c".to_string(),
            kind: CqArmKind::Cold,
            expect: CqExpect::None,
            expect_start: None,
            label: None,
        };
        let failed: Result<CqArmRun, String> = Err("boom".to_string());
        let value = cq_arm_json(&hit, &failed, json!({"passed": false}));
        assert_eq!(value["label"], json!("rc4"));
        assert_eq!(value["cache_mode"], json!("default"));
        assert_eq!(value["error"], json!("boom"));
        let value = cq_arm_json(&cold, &failed, json!({"passed": false}));
        assert!(value.get("label").is_none());
        assert_eq!(value["cache_mode"], json!("cold_exact"));
    }

    #[test]
    fn relative_paths_are_rejected() {
        let mut value = valid();
        value["streams"]["main"]["ids_path"] = json!("main.u32");
        assert!(parse_err(&value).contains("absolute"));
    }

    #[test]
    fn sha256_mismatch_is_rejected() {
        let fixture = parse(&fixture_json(&"0".repeat(64))).expect("structurally valid");
        let error = resolve(&fixture).expect_err("sha mismatch");
        assert!(error.contains("sha256 mismatch"), "{error}");
    }

    #[test]
    fn turn_ends_beyond_the_stream_are_rejected() {
        let mut value = valid();
        value["cases"][0]["turns"][0]["prompt_end"] = json!(2500);
        value["cases"][0]["turns"][0]["consumed_end"] = json!(2600);
        let fixture = parse(&value).expect("structurally valid");
        let streams = resolve(&fixture).unwrap();
        let error = cq_validate_cases(&fixture, &streams, 32768).expect_err("beyond stream");
        assert!(error.contains("consumed_end"), "{error}");
    }

    #[test]
    fn zero_and_inverted_turns_are_rejected() {
        let mut value = valid();
        value["cases"][0]["turns"][0]["prompt_end"] = json!(0);
        assert!(parse_err(&value).contains("prompt_end"));
        let mut value = valid();
        value["cases"][0]["turns"][0]["consumed_end"] = json!(99);
        assert!(parse_err(&value).contains("consumed_end"));
    }

    #[test]
    fn final_request_overflow_is_rejected() {
        let mut value = valid();
        value["cases"][0]["final"]["prompt_end"] = json!(1936);
        let fixture = parse(&value).unwrap();
        let streams = resolve(&fixture).unwrap();
        // 1936 + 64 + 1 = 2001 > 2000.
        let error = cq_validate_cases(&fixture, &streams, 32768).expect_err("overflow");
        assert!(error.contains("score_rows"), "{error}");
        // The last fitting prompt passes.
        let mut value = valid();
        value["cases"][0]["final"]["prompt_end"] = json!(1935);
        let fixture = parse(&value).unwrap();
        cq_validate_cases(&fixture, &streams, 32768).expect("1935 + 64 + 1 == 2000");
        // max_seq bound: 500 + 64 + 1 = 565 is not < 565.
        let fixture = parse(&valid()).unwrap();
        assert!(cq_validate_cases(&fixture, &streams, 565).is_err());
        assert!(cq_validate_cases(&fixture, &streams, 566).is_ok());
    }

    #[test]
    fn final_prompt_end_must_be_positive() {
        let mut value = valid();
        value["cases"][0]["final"]["prompt_end"] = json!(0);
        assert!(parse_err(&value).contains("prompt_end"));
    }

    #[test]
    fn expectation_without_cold_arm_is_rejected() {
        let mut value = valid();
        value["cases"][0]["arms"] = json!([
            {"name": "hit", "kind": "hit", "expect": "bitwise_equal_cold"}
        ]);
        assert!(parse_err(&value).contains("no cold arm"));
        // Without expectations a cold-less case is allowed.
        value["cases"][0]["arms"] = json!([{"name": "hit", "kind": "hit"}]);
        parse(&value).expect("no expectation, no cold arm");
    }

    #[test]
    fn unknown_kinds_modes_and_keys_are_rejected() {
        let mut value = valid();
        value["cases"][0]["arms"][1]["kind"] = json!("warm");
        assert!(parse_err(&value).contains("unknown kind"));

        let mut value = valid();
        value["mode"] = json!("mtp");
        assert!(parse_err(&value).contains("unknown mode"));

        let mut value = valid();
        value["cases"][0]["arms"][0]["expect"] = json!("close_to_cold");
        assert!(parse_err(&value).contains("unknown expect"));

        let mut value = valid();
        value["extra"] = json!(1);
        assert!(parse_err(&value).contains("unknown key"));

        let mut value = valid();
        value["cases"][0]["final"]["extra"] = json!(1);
        assert!(parse_err(&value).contains("unknown key"));

        let mut value = valid();
        value["cases"][0]["arms"][0]["extra"] = json!(1);
        assert!(parse_err(&value).contains("unknown key"));

        let mut value = valid();
        value["streams"]["syn"]["extra"] = json!(1);
        assert!(parse_err(&value).contains("unknown key"));

        let mut value = valid();
        value["schema"] = json!("hipfire.qwen4.cache_quality_corpus.v0");
        assert!(parse_err(&value).contains("schema"));

        let mut value = valid();
        value["id_provenance"] = json!("guessed");
        assert!(parse_err(&value).contains("id_provenance"));
    }

    #[test]
    fn arm_shape_rules() {
        let mut value = valid();
        value["cases"][0]["arms"][2]["expect_start"] = json!(0);
        assert!(parse_err(&value).contains("not allowed on a cold arm"));

        let mut value = valid();
        value["cases"][0]["arms"][2]["cache_mode"] = json!("default");
        assert!(parse_err(&value).contains("not allowed on a cold arm"));

        let mut value = valid();
        value["cases"][0]["arms"][1]["expect_start"] = json!(0);
        assert!(parse_err(&value).contains("expect_start"));

        let mut value = valid();
        value["cases"][0]["arms"][1]["name"] = json!("cold");
        assert!(parse_err(&value).contains("duplicate arm"));

        let mut value = valid();
        value["cases"][0]["arms"][1] = json!({"name": "cold2", "kind": "cold"});
        assert!(parse_err(&value).contains("cold arms"));

        let mut value = valid();
        value["cases"][0]["arms"][1]["name"] = json!("../escape");
        assert!(parse_err(&value).contains("name"));

        // A hit arm needs no cache_mode on this backend.
        let mut value = valid();
        value["cases"][0]["arms"][1] = json!({"name": "h", "kind": "hit"});
        parse(&value).expect("hit arm without cache_mode");
    }

    #[test]
    fn stream_shape_rules() {
        let mut value = valid();
        value["streams"]["branch"]["share_prefix_from"] = json!("nope");
        assert!(parse_err(&value).contains("not a stream"));

        let mut value = valid();
        value["streams"]["branch"]["share_prefix_from"] = json!("branch");
        assert!(parse_err(&value).contains("must not itself be a branch"));

        let mut value = valid();
        value["streams"]["branch"]["share_len"] = json!(9001);
        assert!(parse_err(&value).contains("share_len"));

        let mut value = valid();
        value["streams"]["branch"].as_object_mut().unwrap().remove("share_len");
        assert!(parse_err(&value).contains("together"));

        let mut value = valid();
        value["cases"][0]["final"]["stream"] = json!("ghost");
        assert!(parse_err(&value).contains("unknown stream"));
    }

    #[test]
    fn id_files_must_be_u32_multiples() {
        let bytes = vec![1u8, 2, 3, 4, 5];
        let fixture = parse(&fixture_json(&cq_sha256_hex(&bytes))).unwrap();
        let error = cq_resolve_streams(&fixture, &|_| Ok(bytes.clone()), &fake_synth)
            .expect_err("5 bytes");
        assert!(error.contains("multiple of 4"), "{error}");
    }

    #[test]
    fn synthetic_tokens_are_deterministic_in_range_and_never_eos() {
        let (vocab, eos) = (1000usize, 7u32);
        let a = cq_synth_tokens(vocab, eos, 17, 4096);
        assert_eq!(a, cq_synth_tokens(vocab, eos, 17, 4096));
        assert_ne!(a, cq_synth_tokens(vocab, eos, 18, 4096));
        assert_eq!(a.len(), 4096);
        assert!(a.iter().all(|&t| (10..vocab as u32).contains(&t) && t != eos));
        // A longer request extends the shorter one (same LCG walk).
        assert_eq!(a[..100], cq_synth_tokens(vocab, eos, 17, 100)[..]);
        // The EOS id is skipped, not remapped: forcing it out of a tiny span
        // still yields the requested count.
        let tiny = cq_synth_tokens(12, 10, 3, 64);
        assert_eq!(tiny.len(), 64);
        assert!(tiny.iter().all(|&t| t == 11));
    }

    #[test]
    fn row_mapping_has_no_off_by_one() {
        let stream = ids(100);
        let prompt_end = 10;
        // Row 0: the final prompt token (index 9) predicts index 10.
        assert_eq!(cq_row_map(prompt_end, 0), (9, 9, 10));
        assert_eq!(stream[9], 3 + 9 * 7);
        // Row 3: after teacher-forcing stream[prompt_end + 2] (index 12).
        assert_eq!(cq_row_map(prompt_end, 3), (12, 12, 13));
        // The last scored row of score_rows = 64 targets stream[prompt_end + 64].
        assert_eq!(cq_row_map(prompt_end, 64).2, prompt_end + 64);
        // Row r >= 1 consumes stream[prompt_end + r - 1], i.e. input_index.
        for row in 1..=64usize {
            assert_eq!(cq_row_map(prompt_end, row).1, prompt_end + row - 1);
        }
    }

    #[test]
    fn logsumexp_and_row_stats() {
        assert!((cq_logsumexp(&[0.0, 0.0]) - 2f64.ln()).abs() < 1e-12);
        let row = [1.0f32, 3.0, 2.0];
        let stats = cq_row_stats(&row, 2).unwrap();
        let lse = (1f64.exp() + 3f64.exp() + 2f64.exp()).ln();
        assert_eq!(stats.top1_id, 1);
        assert_eq!(stats.top1_logit, 3.0);
        assert!((stats.margin - 1.0).abs() < 1e-12);
        assert!((stats.logsumexp - lse).abs() < 1e-12);
        assert!((stats.target_logprob - (2.0 - lse)).abs() < 1e-12);
        assert!(cq_row_stats(&row, 3).is_err());
        assert!(cq_row_stats(&[1.0, f32::NAN, 0.0], 0).is_err());
        assert!(cq_row_stats(&[1.0, f32::INFINITY], 0).is_err());
        // A tie for the maximum has margin 0 and the first index wins.
        let tie = cq_row_stats(&[2.0, 2.0, 1.0], 0).unwrap();
        assert_eq!((tie.top1_id, tie.margin), (0, 0.0));
    }

    #[test]
    fn kl_of_identical_rows_is_zero() {
        let row = [0.3f32, -1.2, 4.5, 0.0, 2.2];
        assert_eq!(cq_kl(&row, &row), 0.0);
        let compared = cq_compare_rows(&row, &row).unwrap();
        assert_eq!(compared.kl_cold_arm, 0.0);
        assert_eq!(compared.kl_arm_cold, 0.0);
    }

    #[test]
    fn kl_matches_a_hand_computed_three_logit_example() {
        // P = softmax([0, 0, 0]) = [1/3; 3]; Q = softmax([ln 2, 0, 0]) = [1/2, 1/4, 1/4].
        let p = [0.0f32; 3];
        let q = [std::f32::consts::LN_2, 0.0, 0.0];
        // KL(P||Q) = (1/3) * (ln(2/3) + 2 ln(4/3)) = ln(32/27) / 3.
        let forward = (32f64 / 27.0).ln() / 3.0;
        // KL(Q||P) = 1/2 ln(3/2) + 2 * 1/4 ln(3/4) = ln(9/8) / 2.
        let backward = (9f64 / 8.0).ln() / 2.0;
        assert!((cq_kl(&p, &q) - forward).abs() < 1e-6, "{}", cq_kl(&p, &q));
        assert!((cq_kl(&q, &p) - backward).abs() < 1e-6, "{}", cq_kl(&q, &p));
        let compared = cq_compare_rows(&p, &q).unwrap();
        assert!((compared.kl_cold_arm - forward).abs() < 1e-6);
        assert!((compared.kl_arm_cold - backward).abs() < 1e-6);
        // Both rows peak at index 0 (P is a three-way tie, first index wins).
        assert!(compared.top1_equal);
        assert!(!cq_compare_rows(&[1.0, 2.0, 0.0], &[2.0, 1.0, 0.0]).unwrap().top1_equal);
        assert!((compared.max_abs_diff - f64::from(std::f32::consts::LN_2)).abs() < 1e-12);
    }

    #[test]
    fn bitwise_comparison_sees_one_ulp_and_signed_zero() {
        let cold = [1.0f32, 2.0, 0.0];
        assert!(cq_compare_rows(&cold, &cold).unwrap().bitwise_equal);
        let mut arm = cold;
        arm[1] = f32::from_bits(arm[1].to_bits() + 1);
        let compared = cq_compare_rows(&cold, &arm).unwrap();
        assert!(!compared.bitwise_equal);
        assert!(compared.max_abs_diff > 0.0);
        assert!(compared.top1_equal);
        let mut negative_zero = cold;
        negative_zero[2] = -0.0;
        let compared = cq_compare_rows(&cold, &negative_zero).unwrap();
        assert!(!compared.bitwise_equal);
        assert_eq!(compared.max_abs_diff, 0.0);
        assert!(cq_compare_rows(&cold, &[1.0, 2.0]).is_err());
        assert!(cq_compare_rows(&cold, &[1.0, f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn source_names_follow_the_restored_prefix() {
        assert_eq!(cq_source_name(0), "cold");
        assert_eq!(cq_source_name(512), "session");
    }

    #[test]
    fn bitwise_expectation_reasons() {
        assert!(cq_bitwise_reasons(Some(8192), 8192, "session", true, true).is_empty());
        assert!(cq_bitwise_reasons(None, 0, "cold", true, true).is_empty());
        assert_eq!(cq_bitwise_reasons(None, 0, "cold", false, true).len(), 1);
        assert_eq!(cq_bitwise_reasons(None, 0, "cold", true, false).len(), 1);
        assert_eq!(cq_bitwise_reasons(Some(8192), 4096, "session", true, true).len(), 1);
        // expect_start > 0 with a cold source: right start, wrong source.
        assert_eq!(cq_bitwise_reasons(Some(8192), 8192, "cold", true, true).len(), 1);
        // expect_start == 0 may be cold.
        assert!(cq_bitwise_reasons(Some(0), 0, "cold", true, true).is_empty());
    }

    #[test]
    fn state_family_is_sha256_of_the_part_bytes() {
        let empty = cq_family_of(&[]);
        assert_eq!(empty.bytes, 0);
        assert_eq!(
            empty.digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let abc = cq_family_of(b"abc");
        assert_eq!(abc.bytes, 3);
        assert_eq!(
            abc.digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_ne!(cq_family_of(b"abd"), abc);
    }

    #[test]
    fn session_budget_parses_u64_or_defaults_to_unbounded() {
        assert_eq!(cq_session_budget(None), Ok(u64::MAX >> 1));
        assert_eq!(cq_session_budget(Some("")), Ok(u64::MAX >> 1));
        assert_eq!(cq_session_budget(Some("  ")), Ok(u64::MAX >> 1));
        assert_eq!(cq_session_budget(Some("1048576")), Ok(1 << 20));
        assert_eq!(cq_session_budget(Some(" 7 ")), Ok(7));
        assert!(cq_session_budget(Some("-1")).is_err());
        assert!(cq_session_budget(Some("1.5")).is_err());
        assert!(cq_session_budget(Some("12GiB")).is_err());
        assert!(cq_session_budget(Some("18446744073709551616")).is_err());
    }

    #[test]
    fn capture_path_is_absolute_or_disabled() {
        use std::ffi::OsStr;
        assert_eq!(cq_capture_path_from_env(None), Ok(None));
        assert_eq!(cq_capture_path_from_env(Some(OsStr::new(""))), Ok(None));
        assert!(cq_capture_path_from_env(Some(OsStr::new("rel/out.jsonl"))).is_err());
        assert_eq!(
            cq_capture_path_from_env(Some(OsStr::new("/abs/out.jsonl"))),
            Ok(Some(PathBuf::from("/abs/out.jsonl")))
        );
    }
}
