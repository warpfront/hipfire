// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_slowlink_gate — the S1 slow-link layer-major prefill gate harness.
//!
//! Frozen CLI (every flag required, every path absolute):
//!
//! ```text
//! qwen4_slowlink_gate --model ABS --fixture ABS.json --output ABS.json
//!     --mode primitives|state|product|perf --route incumbent|layer-major
//!     --manifest ABS.json
//! ```
//!
//! `--route` is a diagnostic force inside the same production selector and
//! safety predicate (`HIPFIRE_QWEN4_SLOWLINK_PREFILL=0|1`), not an alternate
//! implementation: a forced `layer-major` that the selector bypasses is a
//! failing check naming the bypass reason, never a silent incumbent run.
//!
//! The fixture (`hipfire.qwen4.slowlink_fixture.v1`) records the exact
//! prompt ids (512 / 1536 / 1537 / 8192 / 16384 rows, an irregular prompt
//! with original launch cuts, a resumed prompt with its entry prefix, a
//! boundary prompt with its requested boundaries and a natural-text prompt),
//! the divergent continuation ids and the named failure injections. The
//! manifest pins source base and tip, binary SHA-256, model identity, PLE
//! source/format/sidecars, kernel code-object hashes, host/device/link and
//! the memory ledger; the binary hash and model size are re-observed and a
//! mismatch fails the run before any GPU work.
//!
//! The output records the actual route, window and chunk manifest of every
//! request (the bundle's `slowlink_receipt`), fixed-state and final-logit
//! digests, generated ids and decoded text, useful/submitted bytes and the
//! copy/compute/barrier/non-expert spans, peak and resident bytes and the
//! status of EVERY required check. A required check that was not executed
//! makes the run `incomplete`, never `pass`; the exit status is 0 only for
//! `pass`.
//!
//! Durable progress: every run appends bounded JSONL records to the sidecar
//! `<output>.progress.jsonl` (no new flag; written with `sync_data` the moment
//! each is observed) so a SIGTERM, timeout or crash keeps the case/repeat
//! start and end, each completed repeat's raw receipts, determinism digests,
//! wall time, session-cache occupancy/evictions, the existing
//! `pre_load`/`post_load`/`end` point samples (plus a `load_failed` or
//! `run_failed` point when a failure leaves the GPU owned), every check row
//! and the error text. Records carry UTC and a monotonic offset that stderr
//! progress echoes repeat. The sidecar is evidence only, never a verdict:
//! `--output` is written once, atomically, and only a complete run reaches
//! `pass`; an aborted run writes `incomplete` with a summary of the sidecar
//! and a missing case is never filled in. A sidecar write error fails the
//! required check `progress.durability`. The legacy `perf` cell keeps its
//! informational, unchanged timing path and persists only its envelope.
//!
//! Modes:
//! * `primitives` — the production panel transport, dispatch and ordered
//!   combine self-test (payload/table identity, repeated epochs at the same
//!   addresses, delayed readers, within-expert panels with canaries, exact
//!   canonical combine) on the loaded bundle's admitted S1 profile.
//! * `state` — real prompts through the production prefill: route receipts
//!   (residency, tile, charged lookahead, one sweep per window, exactly one
//!   window at 16384 rows), same-route repeat digests of every fixed state,
//!   resume and launch-cut chronology, session restore and divergent
//!   continuation at the requested boundaries, named failure injections with
//!   restore/containment and a clean re-run, bounded head scoring of the
//!   last rows, and eager versus retained-body replay decode after the
//!   prefill.
//! * `product` — greedy generation with decoded text for review (attractor
//!   heuristics reject degenerate output; a human still reads the text).
//! * `perf` — fresh-process timing of the 16384-row prefill plus the
//!   pp512/pp8192 guards and decode, with the span attribution and the
//!   cross-layer event-overlap evidence.
//!
//! Decode "graph" proof: Qwen4 production has no hipGraph decode route
//! (`gpu_forward.rs`: no Qwen4 production path opens a hipGraph capture); its
//! retained body replay (`ReplayController`: first eligible forward records,
//! later ones replay) is the decode body replay, so that is the route proved
//! after the S1 prefill. A route that does not reach `Ready` or replay every
//! measured token fails the check.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::state_parity::{
    slowlink_load_bundle, slowlink_logit_rows_summary, slowlink_state_digest,
};
use hipfire_runtime::arch_model::ArchModel;
use hipfire_runtime::session_cache::{SessionCache, SessionCacheUsage, SessionRoute};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::byte_arena::{ModelWeightArenaPlan, NamedByteArenaPlan};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[path = "../tests/common/slowlink_receipt.rs"]
mod slowlink_receipt;
use slowlink_receipt::*;

const USAGE: &str = "usage: qwen4_slowlink_gate --model ABS --fixture ABS.json --output ABS.json --mode primitives|state|product|perf --route incumbent|layer-major --manifest ABS.json";

/// Diagnostic-only: the census is a source commitment, not a live allocation,
/// and the residual is not an estimate of the actual driver footprint.
const MEMORY_DIAGNOSTIC_NOTE: &str = "diagnostic only, never gates the verdict: owned census figures are source commitments, not live allocations, and the residual is not an actual driver estimate";

/// The checks whose execution decides the verdict for `mode`. The census
/// residual is deliberately absent: it is reported as `memory_observations`.
fn required_check_ids(mode: Mode, fixture: &Fixture, route: Expect) -> Vec<String> {
    let mut required: Vec<String> = match mode {
        Mode::Primitives => PRIMITIVE_CASES
            .iter()
            .map(|n| format!("primitive.{n}"))
            .collect(),
        Mode::State => state_required(fixture, route),
        Mode::Product => product_required(fixture),
        Mode::Perf => perf_required(fixture),
    };
    required.push("identity.manifest".to_string());
    required.push("load.placement".to_string());
    required.push("progress.durability".to_string());
    required
}

/// The primitive cell's census/residual observations as a NON-GATING
/// diagnostic. Ledger and residual problems are `info_error(s)`, never checks.
fn memory_diagnostic(samples: &[Value], ledger: Result<MemoryLedger, String>) -> Value {
    let mut diag = match ledger {
        Err(e) => json!({"samples": samples, "info_error": e}),
        Ok(ledger) => {
            let (problems, mut receipt) = memory_problems(samples, &ledger);
            receipt["info_errors"] = json!(problems);
            receipt
        }
    };
    diag["required"] = json!(false);
    diag["gating"] = json!(false);
    diag["note"] = json!(MEMORY_DIAGNOSTIC_NOTE);
    diag
}

const ROUTE_ENV: &str = "HIPFIRE_QWEN4_SLOWLINK_PREFILL";
const REPORT_SCHEMA: &str = "hipfire.qwen4.slowlink_gate.v1";
/// Greedy ids compared / reported for the restore and replay checks.
const RESTORE_DECODE: usize = 16;
/// Retained replay needs at most this many warm tokens before `Ready`.
const REPLAY_WARM_LIMIT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Primitives,
    State,
    Product,
    Perf,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Primitives => "primitives",
            Self::State => "state",
            Self::Product => "product",
            Self::Perf => "perf",
        }
    }
}

use hipfire_arch_qwen4::slowlink_admission::{
    purpose_tile_rows, AdmissionPurpose, GranuleCharge, SlowlinkCensus, SlowlinkProfile,
};

/// The admission purpose of a mode's load. Only the primitive cell is the Hold 1
/// self-test (its diagnostics are charged inside the admitted reserve); every
/// state, product and perf cell is the product route and keeps the product's
/// headroom rule. The one real load call below takes its purpose from here.
fn load_purpose(mode: Mode) -> AdmissionPurpose {
    match mode {
        Mode::Primitives => AdmissionPurpose::PrimitiveSelftest,
        Mode::State | Mode::Product | Mode::Perf => AdmissionPurpose::Product,
    }
}

/// The manifest `s1_profiles` key of a purpose.
fn purpose_name(purpose: AdmissionPurpose) -> &'static str {
    match purpose {
        AdmissionPurpose::Product => PROFILE_PRODUCT,
        AdmissionPurpose::PrimitiveSelftest => PROFILE_PRIMITIVE,
    }
}

/// The actual admission read back from the loaded bundle's profile.
fn admitted_from(
    profile: Option<&hipfire_arch_qwen4::slowlink_admission::SlowlinkProfile>,
) -> Option<(&'static str, u64)> {
    profile.map(|p| (purpose_name(p.purpose), p.tile_rows as u64))
}

/// What this cell's profile is, and is not, evidence of. `expected_*` is what
/// this binary's loader is built to admit for the cell's purpose, known at
/// startup; `admitted_*` is `null` until a load succeeded and then the values
/// read back from the loaded bundle's own profile (`None` stays `null`: a failed
/// or declined load never reports an expected value as admitted). The primitive
/// cell runs at the primitive purpose's tile and is kernel correctness only;
/// every other cell is a product-shape claim at the product tile. `admission`
/// is the actual placement's witness (`null` until a profile was admitted);
/// the manifest's resident count is reference data and is never labelled
/// observed.
fn s1_profile_evidence(
    purpose: AdmissionPurpose,
    manifest: &Value,
    admitted: Option<(&str, u64)>,
    admission: Option<&Value>,
) -> Value {
    let primitive = purpose == AdmissionPurpose::PrimitiveSelftest;
    json!({
        "expected_purpose": purpose_name(purpose),
        "expected_tile_rows": purpose_tile_rows(purpose),
        "admitted_purpose": admitted.map(|a| a.0),
        "admitted_tile_rows": admitted.map(|a| a.1),
        "manifest_product_tile_rows": manifest_profile_tile(manifest, PROFILE_PRODUCT),
        "manifest_primitive_tile_rows": manifest_profile_tile(manifest, PROFILE_PRIMITIVE),
        "manifest_resident_expert_layers_reference": {
            "role": PROJECTION_ROLE_REFERENCE,
            "product": manifest_profile_u64(manifest, PROFILE_PRODUCT, "resident_expert_layers"),
            "primitive_selftest": manifest_profile_u64(manifest, PROFILE_PRIMITIVE, "resident_expert_layers"),
        },
        "admission": admission,
        "qualification": if primitive {
            "kernel correctness only, at the primitive purpose's tile; not a product-shape qualification"
        } else {
            "product-shape claim at the product purpose's tile"
        },
    })
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    mode: Mode,
    route: Expect,
    manifest: PathBuf,
}

fn absolute(flag: &str, value: String) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path"));
    }
    Ok(path)
}

fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut args = args.into_iter();
    let (mut model, mut fixture, mut output, mut manifest) = (None, None, None, None);
    let (mut mode, mut route) = (None, None);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(absolute("--model", value()?)?),
            "--fixture" => fixture = Some(absolute("--fixture", value()?)?),
            "--output" => output = Some(absolute("--output", value()?)?),
            "--manifest" => manifest = Some(absolute("--manifest", value()?)?),
            "--mode" => {
                mode = Some(match value()?.as_str() {
                    "primitives" => Mode::Primitives,
                    "state" => Mode::State,
                    "product" => Mode::Product,
                    "perf" => Mode::Perf,
                    other => {
                        return Err(format!(
                            "--mode {other:?} is not primitives|state|product|perf"
                        ))
                    }
                })
            }
            "--route" => {
                route = Some(match value()?.as_str() {
                    "incumbent" => Expect::Incumbent,
                    "layer-major" => Expect::LayerMajor,
                    other => return Err(format!("--route {other:?} is not incumbent|layer-major")),
                })
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let args = Args {
        model: model.ok_or("--model is required")?,
        fixture: fixture.ok_or("--fixture is required")?,
        output: output.ok_or("--output is required")?,
        mode: mode.ok_or("--mode is required")?,
        route: route.ok_or("--route is required")?,
        manifest: manifest.ok_or("--manifest is required")?,
    };
    if args.mode == Mode::Primitives && args.route == Expect::Incumbent {
        return Err(
            "--mode primitives exercises the layer-major panel code; it has no incumbent arm"
                .into(),
        );
    }
    Ok(args)
}

// ── helpers ────────────────────────────────────────────────────────────

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn file_sha256(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha_u32(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// A generated sequence that fell into a short loop or collapsed to a couple
/// of tokens. A heuristic only: the decoded text is still for human review.
fn attractor(ids: &[u32]) -> Option<String> {
    if ids.len() < 8 {
        return None;
    }
    let distinct: std::collections::BTreeSet<u32> = ids.iter().copied().collect();
    if distinct.len() <= 2 {
        return Some(format!(
            "only {} distinct token(s) in {}",
            distinct.len(),
            ids.len()
        ));
    }
    for period in 1..=4usize {
        if ids.len() >= period * 8 {
            let tail = &ids[ids.len() - period * 8..];
            if tail.chunks(period).all(|c| c == &tail[..period]) {
                return Some(format!(
                    "last {} tokens repeat with period {period}",
                    period * 8
                ));
            }
        }
    }
    None
}

fn gpu_error<T>(result: Result<T, impl std::fmt::Display>, what: &str) -> Result<T, String> {
    result.map_err(|e| format!("{what}: {e}"))
}

fn sync(gpu: &Gpu) -> Result<(), String> {
    gpu_error(gpu.hip.device_synchronize(), "device synchronize")
}

/// `SessionCache::usage` as JSON: the DEVICE-only footprint (`device_total_bytes`
/// sums the device owners and never a host byte) and, separately, the pinned
/// HOST ownership categories and counters of the opt-in pressure-selected
/// host snapshot tier (`host_total_bytes` sums each host owner once).
/// `pressure_host_policy` records whether that tier is armed (a nonzero host
/// budget) and its host budget; the device-side budget is the admitted
/// profile's `device_snapshot_budget_bytes`, recorded with the placement.
fn usage_value(usage: &SessionCacheUsage) -> Value {
    let device_total = [
        usage.pool_bytes,
        usage.pending_bytes,
        usage.resident_bytes,
        usage.deferred_release_bytes,
        usage.parked_bytes,
    ]
    .into_iter()
    .fold(0u64, u64::saturating_add);
    let host_total = [
        usage.host_pool_bytes,
        usage.host_pending_bytes,
        usage.host_resident_bytes,
        usage.host_deferred_release_bytes,
        usage.host_parked_bytes,
    ]
    .into_iter()
    .fold(0u64, u64::saturating_add);
    json!({
        "pool_bytes": usage.pool_bytes, "budget_bytes": usage.budget_bytes,
        "pending_bytes": usage.pending_bytes, "resident_bytes": usage.resident_bytes,
        "deferred_release_bytes": usage.deferred_release_bytes, "parked_bytes": usage.parked_bytes,
        "evictions_total": usage.evictions_total, "device_total_bytes": device_total,
        "host_budget_bytes": usage.host_budget_bytes, "host_pool_bytes": usage.host_pool_bytes,
        "host_pending_bytes": usage.host_pending_bytes,
        "host_resident_bytes": usage.host_resident_bytes,
        "host_deferred_release_bytes": usage.host_deferred_release_bytes,
        "host_parked_bytes": usage.host_parked_bytes,
        "host_captures_total": usage.host_captures_total,
        "host_restores_total": usage.host_restores_total,
        "host_evictions_total": usage.host_evictions_total,
        "host_skips_total": usage.host_skips_total,
        "host_total_bytes": host_total,
        "byte_units": "payload bytes (requested allocation sizes), not physical granule backing; the cache applies its own physical-budget check at admission",
        "pressure_host_policy": {
            "armed": usage.host_budget_bytes > 0,
            "host_budget_bytes": usage.host_budget_bytes,
        },
    })
}

/// Current cache occupancy and eviction counter; `Null` without a cache.
fn usage_json(cache: Option<&SessionCache>) -> Value {
    cache.map_or(Value::Null, |cache| usage_value(&cache.usage()))
}

/// `after - before` of one monotonic counter; `Null` when either side lacks it
/// or it went backwards.
fn counter_delta(before: &Value, after: &Value, key: &str) -> Value {
    match (before[key].as_u64(), after[key].as_u64()) {
        (Some(b), Some(a)) => a.checked_sub(b).map_or(Value::Null, |d| json!(d)),
        _ => Value::Null,
    }
}

/// Per-request cache evidence: occupancy before and after and the ACTUAL
/// counter deltas between them (the cache's own counters, never derived from
/// the occupancy change): device evictions and the host tier's captures,
/// restores, evictions and skips. `Null` delta when either side has no counter.
fn cache_evidence(before: &Value, after: &Value) -> Value {
    json!({
        "before": before, "after": after,
        "evictions_delta": counter_delta(before, after, "evictions_total"),
        "host_captures_delta": counter_delta(before, after, "host_captures_total"),
        "host_restores_delta": counter_delta(before, after, "host_restores_total"),
        "host_evictions_delta": counter_delta(before, after, "host_evictions_total"),
        "host_skips_delta": counter_delta(before, after, "host_skips_total"),
    })
}

/// Problems with the snapshot storage a published capture left behind. A
/// stored snapshot is a device pool byte OR a pinned-host pool byte (the
/// pressure tier may place every capture on the host); an empty or missing
/// usage fails, so a capture that stored nothing never passes.
fn stored_snapshot_problems(published: &Value) -> Vec<String> {
    if !published.is_object() {
        return vec!["no session cache usage was recorded after commit".to_string()];
    }
    let device = published["pool_bytes"].as_u64().unwrap_or(0);
    let host = published["host_pool_bytes"].as_u64().unwrap_or(0);
    if device == 0 && host == 0 {
        return vec![
            "no snapshot was published to the pool (device pool_bytes 0 and host_pool_bytes 0 after commit)"
                .to_string(),
        ];
    }
    Vec::new()
}

/// Adds the case's own cache evidence (before-plan to after) once both
/// occupancy samples exist.
fn attach_case_cache_evidence(detail: &mut Value) {
    let evidence = cache_evidence(
        &detail["cache_usage_before_plan"],
        &detail["cache_usage_after"],
    );
    detail["cache_evidence"] = evidence;
}

fn term_rows(terms: &[(&'static str, u64)]) -> Value {
    Value::Array(
        terms
            .iter()
            .map(|(name, bytes)| json!({"name": name, "bytes": bytes}))
            .collect(),
    )
}

fn checked_sum(parts: &[u64]) -> Option<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
}

/// The admitted profile's allocation-granule census, additive and separate
/// from the sealed LOGICAL ledger. The census terms (seven owned, two budget)
/// stay logical payload; the granule backing padding is reported per owner
/// beside them and is never folded into a logical figure. SCOPE: the physical
/// figures are the S1 INCREMENT the profile charges beside the incumbent base
/// reserve (the base reserve's payload is not in this profile), not the total
/// model-owned bytes. `padding_s1 = fixed padding - base padding` (both from
/// the profile's own public totals, so a future owner added to either side
/// cannot vanish); the S1 terms listed are the fixed terms whose names are not
/// base-reserve owners, and `terms_close` flags any difference between that
/// list's sum and the derived total. `physical_owned_s1 = logical owned + S1
/// padding` and `physical_reserved_s1 = census total + (base + S1) padding`,
/// which must equal the profile's own `reserve_bytes()`. Resident-layer
/// padding is per admitted layer and reported on its own. Diagnostic
/// evidence only: nothing here gates a verdict, and it is a source
/// commitment, not a live allocation or an arena claim.
fn granule_evidence(
    census: &SlowlinkCensus,
    granule: &GranuleCharge,
    resident_layers: usize,
    device_snapshot_budget_bytes: u64,
    profile_reserve_bytes: Result<u64, String>,
) -> Value {
    let owned = census.owned_terms();
    let budget = census.budget_terms();
    let base = granule.base.terms();
    let fixed = granule.terms();
    let s1_padding: Vec<(&'static str, u64)> = fixed
        .iter()
        .filter(|(name, _)| !base.iter().any(|(base_name, _)| base_name == name))
        .copied()
        .collect();
    let owned_total = census.owned_total().ok();
    let budget_total = checked_sum(&budget.map(|(_, bytes)| bytes));
    let logical_reserve = census.total().ok();
    let base_padding_total = granule.base.total().ok();
    let padding_total = granule.total().ok();
    let s1_padding_total = padding_total
        .zip(base_padding_total)
        .and_then(|(fixed_total, base_total)| fixed_total.checked_sub(base_total));
    let s1_terms_sum = checked_sum(&s1_padding.iter().map(|(_, b)| *b).collect::<Vec<_>>());
    let physical_owned = owned_total
        .zip(s1_padding_total)
        .and_then(|(logical, padding)| logical.checked_add(padding));
    let physical_reserved = logical_reserve
        .zip(padding_total)
        .and_then(|(logical, padding)| logical.checked_add(padding));
    let layers_padding = granule.layer.checked_mul(resident_layers as u64);
    let (profile_reserve, profile_reserve_error) = match profile_reserve_bytes {
        Ok(bytes) => (json!(bytes), Value::Null),
        Err(error) => (Value::Null, json!(error)),
    };
    json!({
        "units": "bytes; logical_* are payload (requested allocation sizes), backing_padding_* are the \
                  allocation-granule rounding of the same owners' real allocations, physical_* are \
                  logical + padding, counted once; a source commitment, not a live allocation",
        "allocation_granule_bytes": granule.granule_bytes,
        "logical_payload": {
            "owned_bytes": owned_total,
            "owned_terms": term_rows(&owned),
            "budget_bytes": budget_total,
            "budget_terms": term_rows(&budget),
            "reserve_bytes": logical_reserve,
        },
        "backing_padding": {
            "s1_owner_bytes": s1_padding_total,
            "s1_owner_terms": term_rows(&s1_padding),
            "s1_owner_terms_sum_bytes": s1_terms_sum,
            "terms_close": s1_padding_total.is_some() && s1_padding_total == s1_terms_sum,
            "base_reserve_bytes": base_padding_total,
            "base_reserve_terms": term_rows(&base),
            "fixed_total_bytes": padding_total,
            "per_resident_layer_bytes": granule.layer,
            "resident_layers": resident_layers,
            "resident_layers_bytes": layers_padding,
        },
        "physical": {
            "scope": "S1 increment charged beside the incumbent base reserve; the base reserve's payload is not in this profile, so these are not total model-owned bytes",
            "owned_s1_bytes": physical_owned,
            "reserved_s1_bytes": physical_reserved,
            "profile_reserve_bytes": profile_reserve,
            "profile_reserve_error": profile_reserve_error,
            "reserve_consistent": physical_reserved.is_some()
                && profile_reserve.as_u64() == physical_reserved,
        },
        "device_snapshot_budget_bytes": device_snapshot_budget_bytes,
        "arena_claim": false,
    })
}

fn profile_granule_evidence(profile: &SlowlinkProfile) -> Value {
    granule_evidence(
        &profile.census,
        &profile.granule,
        profile.resident_layers,
        profile.device_snapshot_budget_bytes,
        profile.reserve_bytes(),
    )
}

/// `bytes` rounded UP to a whole number of `granule` bytes, checked.
fn round_up_to_granule(bytes: usize, granule: usize) -> Option<u64> {
    if granule == 0 {
        return None;
    }
    let granule = granule as u64;
    (bytes as u64)
        .checked_add(granule - 1)
        .map(|padded| padded / granule * granule)
}

/// One arena family's plan scope and observed capacity.
///
/// PLANNED figures (`payload_bytes`, `root_request_bytes`, `backing_bytes`)
/// come only from the plan the load built; ACTUAL figures (`actual.*`) come
/// only from the capacities the allocator reports for roots that are live
/// (`observed[i] = Some(capacity)`, `None` = a lazily planned root that has
/// not been allocated yet). A planned root is never reported as actual: an
/// unallocated root contributes nothing to `actual.capacity_bytes` and the
/// exact-capacity proof is `exact` only when EVERY planned root is live with
/// `capacity == request` and the plan's own `validate_capacities` returns the
/// planned backing. An empty `observed` means no root of the family is live.
/// `lazy` families (the F16 shadow roots) may legitimately be unallocated;
/// an eager family (the load-lifetime weight roots) with a planned root that
/// is not live after the load is a problem. Nothing here measures recovery.
fn arena_family_evidence(
    scope: &str,
    plan: &NamedByteArenaPlan,
    granule: usize,
    observed: &[Option<usize>],
    lazy: bool,
) -> (Value, Vec<String>) {
    let mut problems = Vec::new();
    let roots = plan.roots.len();
    let payload = plan.logical_bytes();
    let backing = plan.backing_bytes(granule);
    let request_total = checked_sum(&plan.roots.iter().map(|r| *r as u64).collect::<Vec<_>>());
    for result in [&payload, &backing] {
        if let Err(error) = result {
            problems.push(format!("{scope}: plan is unusable: {error}"));
        }
    }
    if request_total.is_none() {
        problems.push(format!("{scope}: planned root requests overflow"));
    }
    let census_ok = observed.is_empty() || observed.len() == roots;
    if !census_ok {
        problems.push(format!(
            "{scope}: {} observed root capacities for {roots} planned roots (census mismatch)",
            observed.len()
        ));
    }
    let live: Vec<Option<usize>> = if census_ok && !observed.is_empty() {
        observed.to_vec()
    } else {
        vec![None; roots]
    };
    let allocated = live.iter().flatten().count();
    let mut mismatches = Vec::new();
    for (index, (request, capacity)) in plan.roots.iter().zip(&live).enumerate() {
        if let Some(capacity) = capacity {
            if capacity != request {
                mismatches.push(json!({
                    "root": index, "requested_bytes": request, "actual_capacity_bytes": capacity,
                }));
                problems.push(format!(
                    "{scope} root {index}: actual capacity {capacity} != planned request {request} \
                     (an oversized or reused buffer is not an exact arena root)"
                ));
            }
        }
    }
    let capacity_total = checked_sum(&live.iter().flatten().map(|c| *c as u64).collect::<Vec<_>>());
    let actual_backing = live
        .iter()
        .flatten()
        .map(|c| round_up_to_granule(*c, granule))
        .try_fold(0u64, |sum, part| part.and_then(|p| sum.checked_add(p)));
    if !lazy && census_ok && roots > 0 && allocated < roots {
        problems.push(format!(
            "{scope}: {} of {roots} planned roots are not live after the load \
             (planned bytes are not actual capacity)",
            roots - allocated
        ));
    }
    let proof = if !census_ok {
        "census_mismatch"
    } else if !mismatches.is_empty() {
        "mismatch"
    } else if roots == 0 || allocated == 0 {
        "none"
    } else if allocated < roots {
        "partial"
    } else {
        let capacities: Vec<usize> = live.iter().flatten().copied().collect();
        match (plan.validate_capacities(&capacities, granule), &backing) {
            (Ok(proved), Ok(planned)) if proved == *planned && Some(proved) == actual_backing => {
                "exact"
            }
            (Ok(_), _) => {
                problems.push(format!(
                    "{scope}: actual backing does not equal the planned backing"
                ));
                "mismatch"
            }
            (Err(error), _) => {
                problems.push(format!("{scope}: capacity validation failed: {error}"));
                "mismatch"
            }
        }
    };
    let state = if roots == 0 {
        "empty"
    } else if allocated == 0 {
        "planned_only"
    } else if allocated < roots {
        "partial_lazy"
    } else {
        "fully_allocated"
    };
    let member_padding = request_total
        .zip(payload.as_ref().ok().copied())
        .and_then(|(request, payload)| request.checked_sub(payload));
    let granule_padding = request_total
        .zip(backing.as_ref().ok().copied())
        .and_then(|(request, backing)| backing.checked_sub(request));
    let evidence = json!({
        "members": plan.members.len(),
        "roots": roots,
        "alignment_bytes": plan.alignment,
        "max_root_bytes": plan.max_root_bytes,
        "lazy": lazy,
        "planned": {
            "payload_bytes": payload.as_ref().ok(),
            "root_request_bytes": request_total,
            "member_alignment_padding_bytes": member_padding,
            "backing_bytes": backing.as_ref().ok(),
            "granule_padding_bytes": granule_padding,
        },
        "actual": {
            "capacity_observed": allocated > 0,
            "capacity_observed_roots": allocated,
            "backing_model": "ceil(observed buf.size, queried allocation granule) per live root; a model, not a per-allocation physical measurement",
            "physical_backing_measured": false,
            "state": state,
            "live_roots": allocated,
            "unallocated_roots": roots.saturating_sub(allocated),
            "capacity_bytes": capacity_total,
            "backing_bytes": actual_backing,
            "capacity_mismatches": mismatches,
        },
        "capacity_proof": proof,
    });
    (evidence, problems)
}

/// The S1 model arena census: the weight family and the F16-shadow family of
/// the plan the load built, beside what the allocator actually holds.
/// `weights` / `shadows` are the observed per-root capacities (empty = no
/// live root). Additive receipt evidence: the second return is the list of
/// capacity/census problems (a root whose actual capacity is not its planned
/// request, a census of the wrong length, capacities with no plan, planned
/// eager roots that are not live); it never inspects the residual.
fn arena_evidence(
    plan: Option<&ModelWeightArenaPlan>,
    weights: &[Option<usize>],
    shadows: &[Option<usize>],
) -> (Value, Vec<String>) {
    let Some(plan) = plan else {
        let mut problems = Vec::new();
        if !weights.is_empty() || !shadows.is_empty() {
            problems.push(
                "arena capacities were observed but the load carries no arena plan".to_string(),
            );
        }
        return (
            json!({"present": false, "planned_is_actual": false, "capacity_observed": false,
                   "physical_backing_measured": false, "recovery_measured": false, "arena_claim": false}),
            problems,
        );
    };
    let (weight_evidence, mut problems) = arena_family_evidence(
        "weight arena",
        &plan.weights,
        plan.granule_bytes,
        weights,
        false,
    );
    let (shadow_evidence, shadow_problems) = arena_family_evidence(
        "f16 shadow arena",
        &plan.f16_shadows,
        plan.granule_bytes,
        shadows,
        true,
    );
    problems.extend(shadow_problems);
    let evidence = json!({
        "units": "bytes; planned.* are the plan's payload, root requests and granule-rounded \
                  backing; actual.* are the capacities of roots that are live now (a lazily \
                  planned root that is not allocated is NOT actual); the exact-capacity proof \
                  needs every root live with capacity == request",
        "scope": "S1 load-lifetime exact roots: the non-expert resident weights and their F16 \
                  shadows; a per-model census, not a product constant; non-S1 loads carry no plan",
        "present": true,
        "allocation_granule_bytes": plan.granule_bytes,
        "weights": weight_evidence,
        "f16_shadows": shadow_evidence,
        "planned_is_actual": false,
        "backing_model": "ceil(root request, queried allocation granule) per root; capacity_observed is true only for live owners; the physical gate is the external sysfs and HIP minimum-free (512 MiB) measurement, not this census",
        "physical_backing_measured": false,
        "recovery_measured": false,
        "arena_claim": false,
    });
    (evidence, problems)
}

/// A load that selected the S1 profile builds the model arena plan; any other
/// load builds none.
fn arena_presence_problems(present: bool, s1_profile: bool) -> Vec<String> {
    match (s1_profile, present) {
        (true, false) => {
            vec!["the S1 profile was admitted without a model weight arena plan".into()]
        }
        (false, true) => {
            vec!["a load without the S1 profile carries a model weight arena plan".into()]
        }
        _ => Vec::new(),
    }
}

/// Observe the bundle's plan and the capacities its owners report right now.
fn arena_observation(bundle: &Qwen4Bundle, gpu: &Gpu) -> (Value, Vec<String>) {
    let weights: Vec<Option<usize>> = bundle
        .weight_arena_root_capacities()
        .into_iter()
        .map(Some)
        .collect();
    let shadows = bundle.f16_shadow_arena_root_capacities(gpu);
    arena_evidence(
        bundle.slowlink_weight_arena_plan().map(|plan| &**plan),
        &weights,
        &shadows,
    )
}

/// How the gate's phase samples are taken, recorded in every sample.
const SAMPLE_METHOD: &str = "point sample at a gate phase boundary: hipMemGetInfo (total-free) + /proc/self/status + smaps clean regular-file RSS + /proc/self/fdinfo amdgpu drm-memory-vram/gtt; observed point, not a continuous driver peak";
/// The light diagnostic sample of the non-primitive cells (no smaps).
const SAMPLE_METHOD_LIGHT: &str = "light point diagnostic at a gate phase boundary: hipMemGetInfo (total-free) + /proc/self/status + /proc/self/fdinfo amdgpu drm-memory-vram/gtt; no smaps, no sampler thread; not judged, not a continuous driver peak";

/// One labeled phase sample of this process's memory. Taken only at gate
/// phase boundaries (never inside the production prefill loops).
fn mem_sample(gpu: &Gpu, label: &str, smaps: bool) -> Value {
    let device = gpu
        .hip
        .get_vram_info()
        .ok()
        .map(|(free, total)| (free as u64, total as u64));
    let method = if smaps {
        SAMPLE_METHOD
    } else {
        SAMPLE_METHOD_LIGHT
    };
    counters_json(label, method, &read_proc_counters(smaps), device)
}

/// Persist one already-taken gate point sample immediately. These are the
/// existing phase points (hipMemGetInfo total-free plus `/proc`), never an
/// allocator observation, and a failure point is not the HIP free at the
/// allocation that failed.
fn persist_memory(
    progress: &ProgressLog,
    phase: &str,
    sample: &Value,
    sampler: Option<&ProcSampler>,
) {
    progress.record(
        "memory",
        json!({
            "phase": phase, "sample": sample,
            "sampler_max": sampler.map(ProcSampler::snapshot),
            "scope": "gate phase point sample; not an allocator observation and not the HIP free at any allocation",
        }),
    );
}

/// The sampler reads the (costly) `/proc/self/smaps` on every this-many reads.
const SAMPLER_SMAPS_EVERY: u64 = 20;

/// Background `/proc` sampler: running maxima of discrete points (process
/// status and fdinfo only; it never queries HIP).
struct ProcSampler {
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<(u64, ProcCounters)>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ProcSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new((0u64, ProcCounters::default())));
        let (thread_stop, thread_state) = (Arc::clone(&stop), Arc::clone(&state));
        let handle = std::thread::spawn(move || {
            let mut tick = 0u64;
            while !thread_stop.load(Ordering::Relaxed) {
                let counters = read_proc_counters(tick % SAMPLER_SMAPS_EVERY == 0);
                tick += 1;
                {
                    let mut guard = thread_state.lock().unwrap_or_else(|e| e.into_inner());
                    guard.0 += 1;
                    guard.1.merge_max(&counters);
                }
                std::thread::sleep(Duration::from_millis(SAMPLER_PERIOD_MS));
            }
        });
        Self {
            stop,
            state,
            handle: Some(handle),
        }
    }

    /// Stop, take one last point and return the maxima as a sample.
    fn finish(mut self) -> Value {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let last = read_proc_counters(true);
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.0 += 1;
        guard.1.merge_max(&last);
        let method = format!("background thread: /proc/self/status + fdinfo every {SAMPLER_PERIOD_MS} ms (plus smaps, the clean regular-file RSS, every {SAMPLER_SMAPS_EVERY}th read and on the final read); running maxima of discrete points, no hipMemGetInfo, not a continuous peak");
        let mut sample = counters_json(SAMPLER_LABEL, &method, &guard.1, None);
        sample["samples"] = json!(guard.0);
        sample
    }

    /// The running maxima so far, without stopping the thread.
    fn snapshot(&self) -> Value {
        let guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut sample = counters_json(
            SAMPLER_LABEL,
            "background sampler running maxima at a progress record; not a final or continuous peak",
            &guard.1,
            None,
        );
        sample["samples"] = json!(guard.0);
        sample
    }
}

struct Ctx {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    logits: GpuTensor,
    vocab: usize,
    eos: u32,
    fixture: Fixture,
    route: Expect,
    /// The admitted purpose profile's tile (sealed manifest, equal to the
    /// binary's `purpose_tile_rows`): the exact tile every layer-major
    /// receipt of this load must name.
    expected_tile: u64,
    /// Labeled phase memory samples ([`mem_sample`]).
    mem: Vec<Value>,
    /// The durable progress sidecar.
    progress: ProgressLog,
}

impl Ctx {
    /// A heavy (smaps) phase sample; only the primitive cell takes these.
    fn sample_memory(&mut self, label: &str) {
        let sample = mem_sample(&self.gpu, label, true);
        self.mem.push(sample);
    }

    fn cache_usage(&self) -> Value {
        usage_json(self.bundle.session_cache())
    }

    /// The bundle's receipt alone (no cache evidence).
    fn bare_receipt(&self) -> Option<Value> {
        self.bundle.slowlink_receipt()
    }

    /// Receipt of the request that just ran, with its `cache_usage` evidence:
    /// the occupancy `before` it (taken by the caller before the request),
    /// the occupancy now and the actual eviction delta. `None` stays `None`.
    fn receipt(&self, before: &Value) -> Option<Value> {
        let mut receipt = self.bundle.slowlink_receipt()?;
        if let Some(object) = receipt.as_object_mut() {
            object.insert(
                "cache_usage".into(),
                cache_evidence(before, &self.cache_usage()),
            );
        }
        Some(receipt)
    }

    /// Receipt problems for one request call of `call_rows` rows, bound to this
    /// load's admitted profile tile ([`call_problems`]).
    fn validate(
        &self,
        receipt: Option<&Value>,
        call_rows: usize,
        single_window: bool,
        overlap: bool,
    ) -> Vec<String> {
        call_problems(
            self.route,
            self.expected_tile,
            receipt,
            call_rows,
            single_window,
            overlap,
        )
    }
}

/// Receipt problems for one request call of `call_rows` rows. A forced
/// layer-major call longer than one historical tile (`MAX_TILE_ROWS`, the
/// approved named-incumbent-bypass bound) must run layer-major; a shorter one
/// may be an explicit, named selector bypass. Every layer-major receipt must
/// name exactly `expected_tile` rows (the admitted purpose profile): being
/// within the generic 1,536 upper bound is not enough.
fn call_problems(
    route: Expect,
    expected_tile: u64,
    receipt: Option<&Value>,
    call_rows: usize,
    single_window: bool,
    overlap: bool,
) -> Vec<String> {
    let limits = Limits {
        require_single_window: single_window,
        require_overlap: overlap && route == Expect::LayerMajor,
        expected_tile_rows: Some(expected_tile),
        ..Limits::default()
    };
    match route {
        Expect::Incumbent => validate(receipt, Expect::Incumbent, &limits),
        Expect::LayerMajor if call_rows as u64 > MAX_TILE_ROWS => {
            validate(receipt, Expect::LayerMajor, &limits)
        }
        Expect::LayerMajor => validate_forced_short(receipt, &limits),
    }
}

/// One whole request: either `prefill_final` (no cuts, no entry) or the
/// consecutive `forward_chunk_final` calls the fixture's launch cuts and
/// resume entry name. Returns each call's receipt.
fn drive(
    ctx: &mut Ctx,
    ids: &[u32],
    entry: usize,
    cuts: &[usize],
) -> Result<Vec<Option<Value>>, String> {
    gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
    let mut receipts = Vec::new();
    if entry == 0 && cuts.is_empty() {
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle.prefill_final(&mut ctx.gpu, ids, 0, &ctx.logits),
            "prefill_final",
        )?;
        receipts.push(ctx.receipt(&before));
    } else {
        let mut points: Vec<usize> = cuts.to_vec();
        if entry > 0 {
            points.push(entry);
        }
        points.push(ids.len());
        points.sort_unstable();
        points.dedup();
        let mut start = 0;
        for end in points {
            let before = ctx.cache_usage();
            gpu_error(
                ctx.bundle
                    .forward_chunk_final(&mut ctx.gpu, &ids[start..end], &ctx.logits, None),
                &format!("forward_chunk_final {start}..{end}"),
            )?;
            receipts.push(ctx.receipt(&before));
            start = end;
        }
    }
    sync(&ctx.gpu)?;
    Ok(receipts)
}

struct Observation {
    receipts: Vec<Option<Value>>,
    state: Value,
    logits: Value,
    wall_s: f64,
}

fn observe(
    ctx: &mut Ctx,
    ids: &[u32],
    entry: usize,
    cuts: &[usize],
) -> Result<Observation, String> {
    let started = Instant::now();
    let receipts = drive(ctx, ids, entry, cuts)?;
    let wall_s = started.elapsed().as_secs_f64();
    let state = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
    let logits = slowlink_logit_rows_summary(&ctx.gpu, &ctx.logits, 1, ctx.vocab, 16)?;
    Ok(Observation {
        receipts,
        state,
        logits,
        wall_s,
    })
}

fn receipts_json(receipts: &[Option<Value>]) -> Value {
    Value::Array(
        receipts
            .iter()
            .map(|r| r.clone().unwrap_or(Value::Null))
            .collect(),
    )
}

/// The actual windows each call of a request ran, compactly: route, entry,
/// rows, the receipt's own window list and tile. Activations and digests of
/// state are never logged here; the raw receipts stay in `receipts`.
fn window_summary(receipts: &[Option<Value>]) -> Value {
    Value::Array(
        receipts
            .iter()
            .map(|receipt| match receipt {
                None => Value::Null,
                Some(r) => json!({
                    "route": r["route"], "entry_pos": r["entry_pos"],
                    "prompt_rows": r["prompt_rows"], "windows": r["windows"],
                    "launch_cuts": r["launch_cuts"], "tile_rows": r["tile_rows"],
                }),
            })
            .collect(),
    )
}

fn greedy(ctx: &mut Ctx, count: usize) -> Result<(Vec<u32>, &'static str), String> {
    let row = gpu_error(ctx.gpu.download_f32(&ctx.logits), "download prompt logits")?;
    if row.iter().any(|v| !v.is_finite()) {
        return Err("non-finite prefill logits".into());
    }
    let mut pending = Some(hipfire_runtime::llama::argmax(&row));
    let mut ids = Vec::with_capacity(count);
    while ids.len() < count {
        let token = gpu_error(
            ctx.bundle
                .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
            "decode",
        )?;
        ids.push(token);
        if token == ctx.eos || ctx.tokenizer.is_terminator(token) {
            return Ok((ids, "stop"));
        }
    }
    Ok((ids, "length"))
}

// ── primitives ─────────────────────────────────────────────────────────

fn run_primitives(
    ctx: &mut Ctx,
    checks: &mut Checks,
    out: &mut serde_json::Map<String, Value>,
    sampler: Option<&ProcSampler>,
) {
    // Each case is invoked alone (the facade already isolates cases) so its
    // raw report is durable the moment it finishes. A refusal before any case
    // ran is ONE `primitive.init` failure; a later refusal is that case's own
    // error and the cases after it stay unproved. It is never cloned.
    let mut reports: Vec<Value> = Vec::new();
    for name in PRIMITIVE_CASES {
        let case_id = format!("primitive.{name}");
        ctx.progress.record("case_start", json!({"case": case_id}));
        let started = Instant::now();
        let result = ctx
            .bundle
            .slowlink_primitive_selftest(&mut ctx.gpu, &[name])
            .map_err(|error| error.to_string());
        let wall_s = started.elapsed().as_secs_f64();
        let (case, stop) = match result {
            Ok(report) => (
                report["cases"]
                    .as_array()
                    .and_then(|cases| cases.iter().find(|c| c["name"] == name))
                    .cloned()
                    .unwrap_or_else(|| {
                        primitive_error(name, "the self-test report did not carry this case")
                    }),
                false,
            ),
            Err(error) if reports.is_empty() => {
                ctx.progress.record(
                    "case_end",
                    json!({
                        "case": case_id, "status": "init_error", "error": error,
                        "wall_s": wall_s, "cache_usage": ctx.cache_usage(),
                        "sampler_max": sampler.map(ProcSampler::snapshot),
                    }),
                );
                record_primitives(checks, out, Err(error));
                ctx.progress.checks(checks);
                return;
            }
            Err(error) => {
                out.insert("selftest_error".into(), json!(error));
                (
                    primitive_error(
                        name,
                        &format!("slow-link primitive {name:?} could not run: {error}"),
                    ),
                    true,
                )
            }
        };
        ctx.progress.record(
            "case_end",
            json!({
                "case": case_id, "status": case["status"], "mismatches": case["mismatches"],
                "report": case, "wall_s": wall_s, "cache_usage": ctx.cache_usage(),
                "sampler_max": sampler.map(ProcSampler::snapshot),
            }),
        );
        judge_primitive(checks, name, &case);
        ctx.progress.checks(checks);
        reports.push(case);
        if stop {
            break;
        }
    }
    out.insert("selftest".into(), json!({"cases": reports}));
}

/// One case's own error report: names the case, carries no mismatch count.
fn primitive_error(name: &str, reason: &str) -> Value {
    json!({
        "name": name, "status": "error", "mismatches": Value::Null,
        "detail": {"error": reason},
    })
}

/// Judges one case from its own report alone (never another case's).
/// `status:"pass"` with `mismatches == 0` passes; `status:"error"`
/// (`mismatches:null`, `detail.error` the reason) fails only that case with
/// its own error text and no invented mismatch count; anything else fails
/// only that case.
fn judge_primitive(checks: &mut Checks, name: &str, case: &Value) {
    let key = format!("primitive.{name}");
    match case["status"].as_str() {
        Some("pass") if case["mismatches"].as_u64() == Some(0) => checks.pass(&key, case.clone()),
        Some("error") => {
            let reason = match case["detail"]["error"].as_str() {
                Some(error) => format!("case error: {error}"),
                None => "case error with no detail.error text".to_string(),
            };
            checks.fail(&key, reason, case.clone())
        }
        status => checks.fail(
            &key,
            format!(
                "case did not pass with zero mismatches (status {}, mismatches {})",
                status.unwrap_or("absent"),
                case["mismatches"]
            ),
            case.clone(),
        ),
    }
}

/// Records a self-test result without ever cloning one failure onto several cases.
/// A global `Err` (no admitted profile, a refused request, a bundle that cannot
/// run it) happens before any case ran: ONE `primitive.init` failure is recorded
/// and every required `primitive.*` stays missing (unproved, the cell is
/// incomplete). Otherwise each case is judged alone from its own report:
/// `status:"pass"` with `mismatches == 0` passes; `status:"error"`
/// (`mismatches:null`, `detail.error` the reason) fails only that case with its
/// own error text and no invented mismatch count; `status:"fail"` or anything
/// else fails only that case; a case absent from the report stays missing.
fn record_primitives(
    checks: &mut Checks,
    out: &mut serde_json::Map<String, Value>,
    result: Result<Value, String>,
) {
    let report = match result {
        Err(error) => {
            out.insert("selftest_error".into(), json!(error));
            checks.fail(
                "primitive.init",
                format!("self-test could not start: {error}"),
                Value::Null,
            );
            return;
        }
        Ok(report) => report,
    };
    out.insert("selftest".into(), report.clone());
    let cases = report["cases"].as_array().cloned().unwrap_or_default();
    for name in PRIMITIVE_CASES {
        let Some(case) = cases.iter().find(|c| c["name"] == name) else {
            continue; // missing stays missing: the run is incomplete
        };
        judge_primitive(checks, name, case);
    }
}

// ── state ──────────────────────────────────────────────────────────────

fn same_run(a: &Observation, b: &Observation) -> Vec<String> {
    let mut bad = Vec::new();
    if a.state["combined"] != b.state["combined"] {
        bad.push("fixed-state digest differs between repeats".to_string());
    }
    if a.logits["combined"] != b.logits["combined"] {
        bad.push("final-logit digest differs between repeats".to_string());
    }
    for (x, y) in a.receipts.iter().zip(&b.receipts) {
        for key in [
            "route",
            "windows",
            "launch_cuts",
            "tile_rows",
            "resident_layers",
            "host_layers",
        ] {
            if x.as_ref().and_then(|r| r.get(key)) != y.as_ref().and_then(|r| r.get(key)) {
                bad.push(format!("receipt {key} differs between repeats"));
            }
        }
    }
    bad
}

fn prompt_check(ctx: &mut Ctx, checks: &mut Checks, report: &mut Vec<Value>, index: usize) {
    let prompt = ctx.fixture.prompts[index].clone();
    let id = prompt.id.clone();
    let case = format!("state.{id}");
    let mut runs: Vec<Observation> = Vec::new();
    ctx.progress.record(
        "case_start",
        json!({
            "case": case, "kind": prompt.kind, "rows": prompt.prompt_ids.len(),
            "entry": prompt.entry, "launch_cuts": prompt.launch_cuts,
            "repeats": ctx.fixture.repeats,
        }),
    );
    for repeat in 0..ctx.fixture.repeats {
        ctx.progress
            .record("repeat_start", json!({"case": case, "repeat": repeat}));
        let start = ProgressLog::stamp(ctx.progress.now());
        match observe(ctx, &prompt.prompt_ids, prompt.entry, &prompt.launch_cuts) {
            Ok(run) => {
                ctx.progress.record(
                    "repeat_end",
                    json!({
                        "case": case, "repeat": repeat, "status": "ok", "start": start,
                        "wall_s": run.wall_s, "calls": run.receipts.len(),
                        "windows": window_summary(&run.receipts),
                        "receipts": receipts_json(&run.receipts),
                        "state_digest": run.state["combined"],
                        "logits_digest": run.logits["combined"],
                        "top1": run.logits["per_row"][0]["top1"],
                        "determinism_vs_first_repeat": runs.first().map(|first| same_run(first, &run)),
                        "cache_usage": ctx.cache_usage(),
                    }),
                );
                runs.push(run);
            }
            Err(error) => {
                ctx.progress.record(
                    "repeat_end",
                    json!({
                        "case": case, "repeat": repeat, "status": "error", "start": start,
                        "error": error, "cache_usage": ctx.cache_usage(),
                    }),
                );
                // The repeats that did complete stay in the failure's detail.
                let detail = json!({
                    "failed_repeat": repeat,
                    "completed_repeats": runs.iter().map(|r| json!({
                        "state": r.state["combined"], "final_logits": r.logits["combined"],
                        "wall_s": r.wall_s,
                    })).collect::<Vec<_>>(),
                });
                checks.fail(
                    &format!("state.{id}.route_receipt"),
                    format!("repeat {repeat}: {error}"),
                    detail.clone(),
                );
                checks.fail(
                    &format!("state.{id}.repeat_determinism"),
                    format!("repeat {repeat}: {error}"),
                    detail,
                );
                return;
            }
        }
    }
    let mut problems = Vec::new();
    for (repeat, run) in runs.iter().enumerate() {
        for (call, receipt) in run.receipts.iter().enumerate() {
            let rows = receipt
                .as_ref()
                .and_then(|r| u64_at(r, "prompt_rows"))
                .map_or(prompt.prompt_ids.len(), |n| n as usize);
            // Only the whole-prompt 16384 request has the one-window rule.
            let single = prompt.kind == "rows"
                && prompt.launch_cuts.is_empty()
                && prompt.prompt_ids.len() as u64 == TARGET_ROWS;
            for p in ctx.validate(receipt.as_ref(), rows, single, false) {
                problems.push(format!("repeat {repeat} call {call}: {p}"));
            }
        }
    }
    let detail = json!({
        "rows": prompt.prompt_ids.len(), "kind": prompt.kind, "entry": prompt.entry,
        "launch_cuts": prompt.launch_cuts,
        "wall_s": runs.iter().map(|r| r.wall_s).collect::<Vec<_>>(),
        "receipts": runs.iter().map(|r| receipts_json(&r.receipts)).collect::<Vec<_>>(),
    });
    checks.push(&format!("state.{id}.route_receipt"), problems, detail);
    let mut bad = Vec::new();
    for run in &runs[1..] {
        bad.extend(same_run(&runs[0], run));
    }
    bad.sort();
    bad.dedup();
    let digests = json!({
        "state": runs[0].state["combined"], "final_logits": runs[0].logits["combined"],
        "top1": runs[0].logits["per_row"][0]["top1"], "repeats": runs.len(),
    });
    checks.push(
        &format!("state.{id}.repeat_determinism"),
        bad,
        digests.clone(),
    );
    report.push(json!({"id": id, "digests": digests, "state": runs[0].state, "final_logits": runs[0].logits}));
    if prompt.kind == "irregular" {
        // The named launch cuts were driven as consecutive calls; each call's
        // receipt (checked above) carries its entry position.
        let mut chrono = Vec::new();
        let mut expect_entry = 0u64;
        for (call, receipt) in runs[0].receipts.iter().enumerate() {
            let entry = receipt.as_ref().and_then(|r| u64_at(r, "entry_pos"));
            let rows = receipt.as_ref().and_then(|r| u64_at(r, "prompt_rows"));
            if entry != Some(expect_entry) {
                chrono.push(format!(
                    "call {call} entry_pos {entry:?}, expected {expect_entry}"
                ));
            }
            expect_entry += rows.unwrap_or(0);
        }
        if expect_entry != prompt.prompt_ids.len() as u64 {
            chrono.push(format!(
                "calls covered {expect_entry} rows of {}",
                prompt.prompt_ids.len()
            ));
        }
        checks.push(
            &format!("state.{id}.launch_cut_chronology"),
            chrono,
            json!({"cuts": prompt.launch_cuts}),
        );
    }
    if prompt.kind == "resume" {
        let entry = runs[0]
            .receipts
            .last()
            .and_then(|r| r.as_ref())
            .and_then(|r| u64_at(r, "entry_pos"));
        let problems = if entry == Some(prompt.entry as u64) {
            Vec::new()
        } else {
            vec![format!(
                "resumed call entry_pos {entry:?} != {}",
                prompt.entry
            )]
        };
        checks.push(
            &format!("state.{id}.resume_entry"),
            problems,
            json!({"entry": prompt.entry}),
        );
    }
}

/// Cold prefill (reused = 0) of `ids`, commit, digest, logits bytes digest,
/// greedy ids.
struct Cold {
    state: Value,
    logits: Value,
    ids: Vec<u32>,
    receipt: Option<Value>,
}

fn cold_commit(ctx: &mut Ctx, ids: &[u32], reused: usize) -> Result<Cold, String> {
    let before = ctx.cache_usage();
    gpu_error(
        ctx.bundle
            .prefill_final(&mut ctx.gpu, ids, reused, &ctx.logits),
        "prefill_final",
    )?;
    let receipt = ctx.receipt(&before);
    sync(&ctx.gpu)?;
    let state = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
    let logits = slowlink_logit_rows_summary(&ctx.gpu, &ctx.logits, 1, ctx.vocab, 16)?;
    ctx.bundle.session_commit();
    let (decoded, _) = greedy(ctx, RESTORE_DECODE)?;
    Ok(Cold {
        state,
        logits,
        ids: decoded,
        receipt,
    })
}

/// Why the boundary cases could not run: every boundary check fails (none is
/// skipped), so the run is a FAILURE, never a pass or a quiet omission.
fn fail_boundary(checks: &mut Checks, cuts: &[usize], why: &str) {
    for name in [
        "state.boundary_capture",
        "state.session_restore",
        "state.divergent_continuation",
    ] {
        checks.fail(name, why, Value::Null);
    }
    for cut in cuts {
        checks.fail(&format!("state.boundary.{cut}.restore"), why, Value::Null);
    }
}

struct BoundaryCase {
    /// Rows the cache planned to reuse (`None` if the case did not get that far).
    reused: Option<usize>,
    executed: bool,
    restore: Vec<String>,
    divergent: Vec<String>,
    detail: Value,
}

/// One requested boundary: the cache must plan exactly `cut` rows from a
/// stored snapshot; the restored request must then compute only the tail and
/// reproduce the cold recompute `cold` of the same extended prompt `ext`.
/// A plan that offers less is a failure and the restore is NOT run, so a cold
/// recompute can never pass as a restore.
fn boundary_case(
    ctx: &mut Ctx,
    cut: usize,
    ext: &[u32],
    cold: &Result<Cold, String>,
) -> BoundaryCase {
    let mut case = BoundaryCase {
        reused: None,
        executed: false,
        restore: Vec::new(),
        divergent: Vec::new(),
        detail: Value::Null,
    };
    let cold = match cold {
        Ok(cold) => cold,
        Err(e) => {
            case.restore.push(format!("cold reference: {e}"));
            return case;
        }
    };
    for p in ctx.validate(cold.receipt.as_ref(), ext.len(), false, false) {
        case.divergent.push(format!("cold reference receipt: {p}"));
    }
    // A reset drops the resident snapshot (the live rows are gone), so the
    // restore can only come from a published pool snapshot.
    if let Err(e) = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset") {
        case.restore.push(e);
        return case;
    }
    let before_plan = ctx.cache_usage();
    let planned = ctx.bundle.session_plan(ext, SessionRoute::Ar);
    let live = planned > 0
        && ctx
            .bundle
            .session_plan_is_live(ext, SessionRoute::Ar, planned);
    case.reused = Some(planned);
    case.detail = json!({
        "cut": cut, "extended_rows": ext.len(), "planned": planned, "live": live,
        "cache_usage_before_plan": before_plan,
        "cold_receipt": cold.receipt,
    });
    if planned != cut || live {
        // No restored request runs on a bad plan: only the plan is judged.
        case.restore
            .extend(boundary_plan_problems(cut, planned, live));
        case.detail["restore_run"] = json!(false);
        case.detail["cache_usage_after"] = ctx.cache_usage();
        attach_case_cache_evidence(&mut case.detail);
        return case;
    }
    let warm = match cold_commit(ctx, ext, planned) {
        Ok(warm) => warm,
        Err(e) => {
            case.restore.push(format!("restored prefill: {e}"));
            case.detail["cache_usage_after"] = ctx.cache_usage();
            attach_case_cache_evidence(&mut case.detail);
            return case;
        }
    };
    case.executed = true;
    case.restore.extend(boundary_restore_problems(
        cut,
        planned,
        live,
        warm.receipt.as_ref(),
        ext.len(),
    ));
    let mut bad = Vec::new();
    if warm.state["combined"] != cold.state["combined"] {
        bad.push("restored state digest differs from cold");
    }
    if warm.logits["combined"] != cold.logits["combined"] {
        bad.push("restored final logits differ from cold");
    }
    if warm.ids != cold.ids {
        bad.push("restored greedy ids differ from cold");
    }
    case.divergent.extend(bad.iter().map(|b| b.to_string()));
    case.divergent.extend(
        ctx.validate(warm.receipt.as_ref(), ext.len() - planned, false, false)
            .into_iter()
            .map(|p| format!("restored receipt: {p}")),
    );
    case.detail["restore_run"] = json!(true);
    case.detail["equal"] = json!(bad.is_empty());
    case.detail["problems"] = json!(bad);
    case.detail["warm_receipt"] = json!(warm.receipt);
    case.detail["cold_ids"] = json!(cold.ids);
    case.detail["warm_ids"] = json!(warm.ids);
    case.detail["cold_state"] = cold.state["combined"].clone();
    case.detail["warm_state"] = warm.state["combined"].clone();
    case.detail["cache_usage_after"] = ctx.cache_usage();
    attach_case_cache_evidence(&mut case.detail);
    case
}

/// Real SessionCache boundaries through the production prefill: the cold
/// references of every cut, then ONE whole-prompt `prefill_final` of the
/// boundary prompt (the explicit marker id makes every requested cut a
/// snapshot boundary through `set_turn_end_token` + `set_turn_snapshots`, on
/// every arm), then a restore case per cut. No case is skipped.
fn boundary_cases(
    ctx: &mut Ctx,
    checks: &mut Checks,
    boundary: &FixturePrompt,
    cuts: &[usize],
    detail: &mut Vec<Value>,
) {
    let marker = ctx.fixture.boundary_token_id;
    let extended: Vec<Vec<u32>> = cuts
        .iter()
        .map(|&cut| {
            let mut ids = boundary.prompt_ids[..cut].to_vec();
            ids.extend_from_slice(&ctx.fixture.continuation_ids);
            ids
        })
        .collect();
    // Cold references first: their own captures are cleared below, so the
    // restore cases only ever read the whole-prompt prefill's snapshots.
    ctx.progress.record(
        "phase_start",
        json!({"stage": "state.restore.marker_boundary", "phase": "cold_references",
               "cuts": cuts, "extended_rows": extended.iter().map(Vec::len).collect::<Vec<_>>()}),
    );
    let mut colds: Vec<Result<Cold, String>> = Vec::new();
    for ids in &extended {
        let cold = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")
            .and_then(|()| cold_commit(ctx, ids, 0));
        colds.push(cold);
    }
    ctx.progress.record(
        "phase_end",
        json!({"stage": "state.restore.marker_boundary", "phase": "cold_references",
               "failed": colds.iter().filter(|c| c.is_err()).count(),
               "windows": window_summary(
                   &colds.iter().map(|c| c.as_ref().ok().and_then(|c| c.receipt.clone())).collect::<Vec<_>>()
               ),
               "cache_usage": ctx.cache_usage()}),
    );
    ctx.bundle.session_clear(&mut ctx.gpu);
    let cleared = ctx.cache_usage();
    ctx.progress.record(
        "phase_start",
        json!({"stage": "state.restore.marker_boundary", "phase": "whole_prompt_capture",
               "rows": boundary.prompt_ids.len()}),
    );
    let capture = (|| -> Result<(Option<Value>, Value), String> {
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle
                .prefill_final(&mut ctx.gpu, &boundary.prompt_ids, 0, &ctx.logits),
            "whole-prompt prefill_final",
        )?;
        let receipt = ctx.receipt(&before);
        sync(&ctx.gpu)?;
        ctx.bundle.session_commit();
        Ok((receipt, ctx.cache_usage()))
    })();
    let layer_major = ctx.route == Expect::LayerMajor;
    let mut capture_problems = Vec::new();
    let capture_detail = match &capture {
        Err(e) => {
            capture_problems.push(format!("whole-prompt prefill: {e}"));
            json!({"cache_usage_after_clear": cleared})
        }
        Ok((receipt, published)) => {
            capture_problems.extend(ctx.validate(
                receipt.as_ref(),
                boundary.prompt_ids.len(),
                false,
                false,
            ));
            capture_problems.extend(boundary_capture_problems(
                cuts,
                receipt.as_ref(),
                layer_major,
            ));
            capture_problems.extend(stored_snapshot_problems(published));
            json!({
                "prompt_rows": boundary.prompt_ids.len(), "marker_id": marker, "cuts": cuts,
                "receipt": receipt, "cache_usage_after_clear": cleared,
                "cache_usage_after_commit": published,
                "cache_evidence": cache_evidence(&cleared, published),
            })
        }
    };
    checks.push(
        "state.boundary_capture",
        capture_problems.clone(),
        capture_detail.clone(),
    );
    ctx.progress.checks(checks);
    detail.push(json!({"capture": capture_detail}));
    let mut reused = Vec::new();
    let mut verified: Vec<(usize, bool)> = Vec::new();
    let mut restore_all: Vec<String> = capture_problems
        .iter()
        .map(|p| format!("capture: {p}"))
        .collect();
    let mut divergent_all = Vec::new();
    for (index, &cut) in cuts.iter().enumerate() {
        ctx.progress.record(
            "case_start",
            json!({"case": format!("state.boundary.{cut}.restore"), "extended_rows": extended[index].len()}),
        );
        let case = boundary_case(ctx, cut, &extended[index], &colds[index]);
        if let Some(rows) = case.reused {
            reused.push((cut, rows));
        }
        checks.push(
            &format!("state.boundary.{cut}.restore"),
            case.restore.clone(),
            case.detail.clone(),
        );
        ctx.progress.checks(checks);
        restore_all.extend(case.restore.iter().map(|p| format!("cut {cut}: {p}")));
        divergent_all.extend(case.divergent.iter().map(|p| format!("cut {cut}: {p}")));
        if !case.executed {
            divergent_all.push(format!("cut {cut}: the restored case did not run"));
        }
        verified.push((
            cut,
            case.executed && case.restore.is_empty() && case.divergent.is_empty(),
        ));
        detail.push(case.detail);
    }
    restore_all.extend(boundary_coverage_problems(cuts, &reused));
    restore_all.sort();
    restore_all.dedup();
    let largest = largest_verified_cut(&verified);
    detail.push(json!({
        "cache_usage_final": ctx.cache_usage(),
        "largest_verified_restore_cut": largest,
    }));
    checks.push(
        "state.session_restore",
        restore_all,
        json!({"boundaries": cuts, "marker_id": marker, "reused": reused,
               "largest_verified_restore_cut": largest}),
    );
    checks.push(
        "state.divergent_continuation",
        divergent_all,
        json!({"continuation_rows": ctx.fixture.continuation_ids.len(), "decode": RESTORE_DECODE}),
    );
    ctx.progress.checks(checks);
}

/// Strict 16K boundary snapshots: ONE whole `prefill_final` of the 16384-row
/// prompt plus three continuation rows under the UNCHANGED 8192-row cadence
/// (no marker policy), then a strict pool restore at 8192 and at 16384 against
/// cold references. A cut the capture did not publish fails (the plan offers
/// less than the cut) and is never recomputed cold. No autotuning, no cache
/// policy change: a real memory or capture failure is recorded as is.
fn boundary16k_checks(ctx: &mut Ctx, checks: &mut Checks, detail: &mut Vec<Value>) {
    let fail_all = |checks: &mut Checks, why: &str| {
        for name in boundary16k_required() {
            checks.fail(&name, why, Value::Null);
        }
    };
    let Some(rows) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() == 16384)
        .map(|p| p.prompt_ids.clone())
    else {
        return fail_all(checks, "the fixture has no 16384-row prompt");
    };
    let ids = match boundary16k_prompt(
        &rows,
        &ctx.fixture.continuation_ids,
        ctx.fixture.max_seq,
        RESTORE_DECODE,
    ) {
        Ok(ids) => ids,
        Err(e) => return fail_all(checks, &e),
    };
    if ctx.bundle.session_cache().is_none() {
        return fail_all(checks, "no session cache is attached to the bundle");
    }
    // Unchanged natural cadence: no message-end token, no turn snapshots.
    ctx.bundle.set_turn_end_token(None);
    ctx.bundle.set_turn_snapshots(false);
    let cuts = BOUNDARY16K_CUTS;
    let extended: Vec<Vec<u32>> = cuts
        .iter()
        .map(|&cut| ids[..cut].iter().chain(&ids[16384..]).copied().collect())
        .collect();
    ctx.progress.record(
        "phase_start",
        json!({"stage": "state.restore.boundary16k", "phase": "cold_references",
               "cuts": cuts, "extended_rows": extended.iter().map(Vec::len).collect::<Vec<_>>()}),
    );
    let mut colds: Vec<Result<Cold, String>> = Vec::new();
    for ext in &extended {
        let cold = gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")
            .and_then(|()| cold_commit(ctx, ext, 0));
        colds.push(cold);
    }
    ctx.progress.record(
        "phase_end",
        json!({"stage": "state.restore.boundary16k", "phase": "cold_references",
               "failed": colds.iter().filter(|c| c.is_err()).count(),
               "windows": window_summary(
                   &colds.iter().map(|c| c.as_ref().ok().and_then(|c| c.receipt.clone())).collect::<Vec<_>>()
               ),
               "cache_usage": ctx.cache_usage()}),
    );
    ctx.bundle.session_clear(&mut ctx.gpu);
    let cleared = ctx.cache_usage();
    ctx.progress.record(
        "phase_start",
        json!({"stage": "state.restore.boundary16k", "phase": "whole_16k_capture",
               "rows": ids.len()}),
    );
    let capture = (|| -> Result<(Option<Value>, Value), String> {
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        let before = ctx.cache_usage();
        gpu_error(
            ctx.bundle.prefill_final(&mut ctx.gpu, &ids, 0, &ctx.logits),
            "whole 16K prefill_final",
        )?;
        let receipt = ctx.receipt(&before);
        sync(&ctx.gpu)?;
        ctx.bundle.session_commit();
        Ok((receipt, ctx.cache_usage()))
    })();
    let layer_major = ctx.route == Expect::LayerMajor;
    let mut capture_problems = Vec::new();
    let capture_detail = match &capture {
        Err(e) => {
            capture_problems.push(format!("whole-prompt 16K prefill: {e}"));
            json!({"prompt_rows": ids.len(), "cuts": cuts, "cache_usage_after_clear": cleared,
                   "cache_usage_after_failure": ctx.cache_usage()})
        }
        Ok((receipt, published)) => {
            capture_problems.extend(ctx.validate(receipt.as_ref(), ids.len(), false, false));
            capture_problems.extend(boundary_capture_problems(
                &cuts,
                receipt.as_ref(),
                layer_major,
            ));
            capture_problems.extend(stored_snapshot_problems(published));
            json!({
                "prompt_rows": ids.len(), "cuts": cuts, "cadence": "unchanged 8192-row natural cadence",
                "receipt": receipt, "cache_usage_after_clear": cleared,
                "cache_usage_after_commit": published,
                "cache_evidence": cache_evidence(&cleared, published),
            })
        }
    };
    checks.push(
        "state.boundary16k.capture",
        capture_problems.clone(),
        capture_detail.clone(),
    );
    ctx.progress.checks(checks);
    detail.push(json!({"boundary16k_capture": capture_detail}));
    let mut verified = Vec::new();
    for (index, &cut) in cuts.iter().enumerate() {
        ctx.progress.record(
            "case_start",
            json!({"case": format!("state.boundary16k.{cut}.restore"), "extended_rows": extended[index].len()}),
        );
        let case = boundary_case(ctx, cut, &extended[index], &colds[index]);
        let mut problems: Vec<String> = case.restore.clone();
        problems.extend(case.divergent.iter().cloned());
        if !case.executed {
            problems.push("the restored case did not run".to_string());
        }
        verified.push((cut, case.executed && problems.is_empty()));
        checks.push(
            &format!("state.boundary16k.{cut}.restore"),
            problems,
            case.detail.clone(),
        );
        ctx.progress.checks(checks);
        detail.push(json!({"boundary16k_case": case.detail}));
    }
    detail.push(json!({
        "boundary16k_largest_verified_restore_cut": largest_verified_cut(&verified),
        "cache_usage_final": ctx.cache_usage(),
    }));
}

fn marker_boundary_checks(ctx: &mut Ctx, checks: &mut Checks, detail: &mut Vec<Value>) {
    let Some(boundary) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "boundary")
        .cloned()
    else {
        return;
    };
    let cuts = boundary.requested_boundaries.clone();
    let marker = ctx.fixture.boundary_token_id;
    if marker as usize >= ctx.vocab {
        let why = format!(
            "boundary_token_id {marker} is outside the vocabulary of {} ids",
            ctx.vocab
        );
        return fail_boundary(checks, &cuts, &why);
    }
    if ctx.bundle.session_cache().is_none() {
        return fail_boundary(checks, &cuts, "no session cache is attached to the bundle");
    }
    // The diagnostic snapshot policy, identical on both arms: the fixture's
    // explicit marker id is the message-end token, so each marker row makes
    // the position after it a boundary.
    ctx.bundle.set_turn_end_token(Some(marker));
    ctx.bundle.set_turn_snapshots(true);
    if ctx.bundle.turn_snapshots_active() {
        boundary_cases(ctx, checks, &boundary, &cuts, detail);
    } else {
        fail_boundary(
            checks,
            &cuts,
            "the marker snapshot policy did not activate (turn_snapshots_active is false)",
        );
    }
    // The later checks run with no message-end token (and so no marker
    // boundaries), as before this case.
    ctx.bundle.set_turn_end_token(None);
    ctx.bundle.set_turn_snapshots(false);
}

fn injection_checks(ctx: &mut Ctx, checks: &mut Checks, out: &mut Vec<Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "resume")
        .cloned()
    else {
        return;
    };
    let entry = prompt.entry;
    for point in FAILURE_POINTS {
        let name = format!("state.inject.{point}");
        ctx.progress.record("case_start", json!({"case": name}));
        let outcome = (|| -> Result<(Vec<String>, Value), String> {
            let mut bad = Vec::new();
            // A clean reference of the whole resumed request and of the entry.
            let clean = observe(ctx, &prompt.prompt_ids, entry, &[])?;
            gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
            gpu_error(
                ctx.bundle.forward_chunk_final(
                    &mut ctx.gpu,
                    &prompt.prompt_ids[..entry],
                    &ctx.logits,
                    None,
                ),
                "entry prefix",
            )?;
            sync(&ctx.gpu)?;
            let at_entry = slowlink_state_digest(&ctx.gpu, &ctx.bundle)?;
            gpu_error(
                ctx.bundle.slowlink_inject_failure(Some(point)),
                "arm injection",
            )?;
            let failed = ctx.bundle.forward_chunk_final(
                &mut ctx.gpu,
                &prompt.prompt_ids[entry..],
                &ctx.logits,
                None,
            );
            let consumed = ctx
                .bare_receipt()
                .and_then(|r| r.get("injection_consumed").and_then(Value::as_bool))
                .unwrap_or(false);
            let _ = ctx.bundle.slowlink_inject_failure(None);
            if failed.is_ok() {
                bad.push("the injected failure did not fail the request".to_string());
            }
            if !consumed {
                bad.push("the injection was not consumed (route ignored it)".to_string());
            }
            let _ = ctx.gpu.hip.device_synchronize();
            let containment = match slowlink_state_digest(&ctx.gpu, &ctx.bundle) {
                Ok(after)
                    if after["combined"] == at_entry["combined"]
                        && after["position"] == at_entry["position"] =>
                {
                    "restored"
                }
                _ => "reset",
            };
            // Whatever the containment, a fresh clean request must reproduce
            // the clean digests: no poisoned state, no donor restore residue.
            let again = observe(ctx, &prompt.prompt_ids, entry, &[])?;
            if again.state["combined"] != clean.state["combined"]
                || again.logits["combined"] != clean.logits["combined"]
            {
                bad.push(
                    "a clean re-run after the abort does not reproduce the clean digests"
                        .to_string(),
                );
            }
            for p in ctx.validate(
                again.receipts.last().and_then(|r| r.as_ref()),
                prompt.prompt_ids.len() - entry,
                false,
                false,
            ) {
                bad.push(format!("re-run receipt: {p}"));
            }
            Ok((
                bad,
                json!({"point": point, "containment": containment, "consumed": consumed, "entry": entry}),
            ))
        })();
        match outcome {
            Ok((bad, detail)) => {
                out.push(detail.clone());
                checks.push(&name, bad, detail);
            }
            Err(e) => checks.fail(&name, e, Value::Null),
        }
        ctx.progress.checks(checks);
    }
}

fn head_rows_check(ctx: &mut Ctx, checks: &mut Checks, out: &mut Vec<Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() as u64 == TARGET_ROWS)
        .cloned()
    else {
        return;
    };
    let head = ctx.fixture.head_rows;
    let outcome = (|| -> Result<(Vec<String>, Value), String> {
        let body = prompt.prompt_ids.len() - head;
        let view = gpu_error(
            ctx.gpu.zeros(&[head * ctx.vocab], DType::F32),
            "head logits",
        )?;
        let mut summaries = Vec::new();
        let mut bad = Vec::new();
        for repeat in 0..ctx.fixture.repeats {
            gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
            gpu_error(
                ctx.bundle
                    .prefill_final(&mut ctx.gpu, &prompt.prompt_ids[..body], 0, &ctx.logits),
                "prefill body",
            )?;
            // All-row API over only the last `head` rows: bounded, never N×vocab.
            gpu_error(
                ctx.bundle
                    .forward_chunk(&mut ctx.gpu, &prompt.prompt_ids[body..], &view, None),
                "head forward_chunk",
            )?;
            sync(&ctx.gpu)?;
            let summary = slowlink_logit_rows_summary(&ctx.gpu, &view, head, ctx.vocab, 16)?;
            for row in summary["per_row"].as_array().into_iter().flatten() {
                if row["finite"] != true {
                    bad.push(format!(
                        "repeat {repeat}: non-finite head row {}",
                        row["row"]
                    ));
                }
            }
            summaries.push(summary);
        }
        let _ = ctx.gpu.free_tensor(view);
        if summaries
            .iter()
            .any(|s| s["combined"] != summaries[0]["combined"])
        {
            bad.push("head-row digests differ between repeats".to_string());
        }
        Ok((
            bad,
            json!({"head_rows": head, "body_rows": body, "summary": summaries[0]}),
        ))
    })();
    match outcome {
        Ok((bad, detail)) => {
            out.push(detail.clone());
            checks.push("state.head_rows_bounded", bad, detail);
        }
        Err(e) => checks.fail("state.head_rows_bounded", e, Value::Null),
    }
}

fn decode_checks(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let Some(prompt) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "natural")
        .cloned()
    else {
        return;
    };
    let measured = ctx.fixture.decode_tokens;
    let result = (|| -> Result<(Vec<String>, Vec<String>, Value), String> {
        // Replay arm first: it fixes how many warm tokens the controller took.
        let armed = ctx.gpu.replay.configure_model_default(true);
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        ctx.progress.record(
            "phase_start",
            json!({"stage": "state.decode", "phase": "natural_prefill_replay_arm",
                   "rows": prompt.prompt_ids.len()}),
        );
        let receipts = drive(ctx, &prompt.prompt_ids, 0, &[])?;
        ctx.progress.record(
            "phase_end",
            json!({"stage": "state.decode", "phase": "natural_prefill_replay_arm",
                   "windows": window_summary(&receipts), "cache_usage": ctx.cache_usage()}),
        );
        let row = gpu_error(ctx.gpu.download_f32(&ctx.logits), "download")?;
        let mut pending = Some(hipfire_runtime::llama::argmax(&row));
        ctx.progress.record(
            "phase_start",
            json!({"stage": "state.decode", "phase": "replay_warm_and_measured_decode",
                   "warm_limit": REPLAY_WARM_LIMIT, "measured_tokens": measured}),
        );
        let mut replay_ids = Vec::new();
        let mut warm = 0usize;
        while ctx.gpu.replay.state() != rdna_compute::replay::ReplayState::Ready
            && warm < REPLAY_WARM_LIMIT
        {
            let t = gpu_error(
                ctx.bundle
                    .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
                "warm decode",
            )?;
            replay_ids.push(t);
            warm += 1;
        }
        let state_after_warm = format!("{:?}", ctx.gpu.replay.state());
        ctx.gpu.replay.begin_replay_observation_window();
        for _ in 0..measured {
            let t = gpu_error(
                ctx.bundle
                    .forward_token_or_argmax(&mut ctx.gpu, pending.take(), &ctx.logits),
                "replay decode",
            )?;
            replay_ids.push(t);
        }
        let observation = ctx.gpu.replay.replay_observation();
        ctx.progress.record(
            "phase_end",
            json!({"stage": "state.decode", "phase": "replay_warm_and_measured_decode",
                   "warm_tokens": warm, "replay_state": state_after_warm,
                   "observed_count": observation.count, "observed_failed": observation.failed}),
        );
        let mut replay_bad = Vec::new();
        if state_after_warm != "Ready" {
            replay_bad.push(format!("retained body not Ready after {warm} warm token(s): {state_after_warm} (configure_model_default returned {armed})"));
        }
        if observation.count != measured as u64 || observation.failed {
            replay_bad.push(format!(
                "replay observed {} of {measured} measured tokens (failed={})",
                observation.count, observation.failed
            ));
        }
        // Eager arm: the same prompt, the same number of tokens.
        ctx.gpu.replay.configure_model_default(false);
        gpu_error(ctx.bundle.reset(&mut ctx.gpu), "reset")?;
        ctx.progress.record(
            "phase_start",
            json!({"stage": "state.decode", "phase": "natural_prefill_and_eager_decode",
                   "rows": prompt.prompt_ids.len(), "tokens": replay_ids.len()}),
        );
        let receipts = drive(ctx, &prompt.prompt_ids, 0, &[])?;
        let (eager_ids, _) = greedy(ctx, replay_ids.len())?;
        ctx.progress.record(
            "phase_end",
            json!({"stage": "state.decode", "phase": "natural_prefill_and_eager_decode",
                   "windows": window_summary(&receipts), "tokens": eager_ids.len(),
                   "cache_usage": ctx.cache_usage()}),
        );
        let mut eager_bad = Vec::new();
        if eager_ids.is_empty() || eager_ids.iter().any(|&t| t as usize >= ctx.vocab) {
            eager_bad.push("eager decode produced no valid ids".to_string());
        }
        if eager_ids != replay_ids {
            replay_bad.push("retained-replay ids differ from the eager ids".to_string());
        }
        let detail = json!({
            "eager_ids": eager_ids, "replay_ids": replay_ids, "warm_tokens": warm,
            "replay_state": state_after_warm,
            "observation": {"count": observation.count, "failed": observation.failed},
            "eager_sha256": sha_u32(&eager_ids), "replay_sha256": sha_u32(&replay_ids),
            "decoded": ctx.tokenizer.decode(&eager_ids),
        });
        Ok((eager_bad, replay_bad, detail))
    })();
    match result {
        Ok((eager_bad, replay_bad, detail)) => {
            out.insert("decode".into(), detail.clone());
            checks.push("state.decode_eager", eager_bad, detail.clone());
            checks.push("state.decode_retained_replay", replay_bad, detail);
        }
        Err(e) => {
            checks.fail("state.decode_eager", e.clone(), Value::Null);
            checks.fail("state.decode_retained_replay", e, Value::Null);
        }
    }
}

fn state_required(fixture: &Fixture, route: Expect) -> Vec<String> {
    let mut names = Vec::new();
    for p in &fixture.prompts {
        names.push(format!("state.{}.route_receipt", p.id));
        names.push(format!("state.{}.repeat_determinism", p.id));
        if p.kind == "irregular" {
            names.push(format!("state.{}.launch_cut_chronology", p.id));
        }
        if p.kind == "resume" {
            names.push(format!("state.{}.resume_entry", p.id));
        }
        if p.kind == "boundary" {
            names.push("state.boundary_capture".to_string());
            for cut in &p.requested_boundaries {
                names.push(format!("state.boundary.{cut}.restore"));
            }
        }
    }
    if fixture
        .prompts
        .iter()
        .any(|p| p.kind == "rows" && p.prompt_ids.len() == 16384)
    {
        names.extend(boundary16k_required());
    }
    names.extend(
        [
            "state.session_restore",
            "state.divergent_continuation",
            "state.head_rows_bounded",
            "state.decode_eager",
            "state.decode_retained_replay",
        ]
        .map(String::from),
    );
    if route == Expect::LayerMajor {
        for point in FAILURE_POINTS {
            names.push(format!("state.inject.{point}"));
        }
    }
    names
}

/// One named stage of the state cell with its start/end and check rows
/// persisted around it.
fn stage(ctx: &mut Ctx, checks: &mut Checks, name: &str, body: impl FnOnce(&mut Ctx, &mut Checks)) {
    ctx.progress.record(
        "stage_start",
        json!({"stage": name, "cache_usage": ctx.cache_usage()}),
    );
    body(ctx, checks);
    ctx.progress.record(
        "stage_end",
        json!({"stage": name, "cache_usage": ctx.cache_usage()}),
    );
    ctx.progress.checks(checks);
}

fn run_state(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let mut prompts = Vec::new();
    for index in 0..ctx.fixture.prompts.len() {
        let case = format!("state.{}", ctx.fixture.prompts[index].id);
        prompt_check(ctx, checks, &mut prompts, index);
        ctx.progress.record(
            "case_end",
            json!({"case": case, "cache_usage": ctx.cache_usage()}),
        );
        ctx.progress.checks(checks);
    }
    out.insert("prompts".into(), Value::Array(prompts));
    // The marker-boundary and strict-16K restore phases are separate stages
    // so a partial log shows which one was in flight; `evidence.restore`
    // stays one array.
    let mut restore = Vec::new();
    stage(
        ctx,
        checks,
        "state.restore.marker_boundary",
        |ctx, checks| {
            marker_boundary_checks(ctx, checks, &mut restore);
        },
    );
    stage(ctx, checks, "state.restore.boundary16k", |ctx, checks| {
        boundary16k_checks(ctx, checks, &mut restore);
    });
    out.insert("restore".into(), Value::Array(restore));
    stage(ctx, checks, "state.injection", |ctx, checks| {
        if ctx.route == Expect::LayerMajor {
            let mut injections = Vec::new();
            injection_checks(ctx, checks, &mut injections);
            out.insert("injections".into(), Value::Array(injections));
        } else {
            for point in FAILURE_POINTS {
                checks.skip(
                    &format!("state.inject.{point}"),
                    "the incumbent route has no S1 transaction to inject into",
                );
            }
        }
    });
    stage(ctx, checks, "state.head_rows", |ctx, checks| {
        let mut head = Vec::new();
        head_rows_check(ctx, checks, &mut head);
        out.insert("head".into(), Value::Array(head));
    });
    stage(ctx, checks, "state.decode", |ctx, checks| {
        decode_checks(ctx, checks, out)
    });
}

// ── product ────────────────────────────────────────────────────────────

fn run_product(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    let mut records = Vec::new();
    for index in 0..ctx.fixture.prompts.len() {
        let prompt = ctx.fixture.prompts[index].clone();
        let id = prompt.id.clone();
        let name = format!("product.{id}");
        ctx.progress.record("case_start", json!({"case": name}));
        let result = (|| -> Result<(Vec<String>, Value), String> {
            let obs = observe(ctx, &prompt.prompt_ids, prompt.entry, &prompt.launch_cuts)?;
            let (ids, finish) = greedy(ctx, ctx.fixture.decode_tokens)?;
            let mut bad = Vec::new();
            let single = prompt.kind == "rows"
                && prompt.launch_cuts.is_empty()
                && prompt.prompt_ids.len() as u64 == TARGET_ROWS;
            for (call, receipt) in obs.receipts.iter().enumerate() {
                let rows = receipt
                    .as_ref()
                    .and_then(|r| u64_at(r, "prompt_rows"))
                    .map_or(prompt.prompt_ids.len(), |n| n as usize);
                for p in ctx.validate(receipt.as_ref(), rows, single, false) {
                    bad.push(format!("call {call}: {p}"));
                }
            }
            if let Some(why) = attractor(&ids) {
                bad.push(format!("attractor: {why}"));
            }
            if ids.is_empty() {
                bad.push("no tokens generated".to_string());
            }
            Ok((
                bad,
                json!({
                    "id": id, "kind": prompt.kind, "prompt_rows": prompt.prompt_ids.len(),
                    "prompt_sha256": sha_u32(&prompt.prompt_ids),
                    "token_ids": ids, "finish_reason": finish, "decoded": ctx.tokenizer.decode(&ids),
                    "first_logit_digest": obs.logits["combined"], "state_digest": obs.state["combined"],
                    "receipts": receipts_json(&obs.receipts),
                }),
            ))
        })();
        match result {
            Ok((bad, record)) => {
                ctx.progress.record(
                    "case_end",
                    json!({
                        "case": name, "status": "ok", "evidence": record,
                        "cache_usage": ctx.cache_usage(),
                    }),
                );
                records.push(record.clone());
                checks.push(&name, bad, json!({"id": id}));
            }
            Err(e) => {
                ctx.progress.record(
                    "case_end",
                    json!({
                        "case": name, "status": "error", "error": e,
                        "cache_usage": ctx.cache_usage(),
                    }),
                );
                checks.fail(&name, e, Value::Null)
            }
        }
        ctx.progress.checks(checks);
    }
    out.insert("records".into(), Value::Array(records));
    out.insert(
        "human_review".into(),
        json!("required: read every decoded text; the attractor heuristic is not a quality gate"),
    );
}

// ── perf ───────────────────────────────────────────────────────────────

fn run_perf(ctx: &mut Ctx, checks: &mut Checks, out: &mut serde_json::Map<String, Value>) {
    // One untimed short prefill materializes lazy allocations and kernels.
    if let Some(warm) = ctx
        .fixture
        .prompts
        .iter()
        .find(|p| p.kind == "rows" && p.prompt_ids.len() == 1537)
        .cloned()
    {
        if let Err(e) = drive(ctx, &warm.prompt_ids, 0, &[]) {
            checks.fail("perf.warmup", e, Value::Null);
            return;
        }
    }
    let mut rows_out = Vec::new();
    let perf_rows = ctx.fixture.perf_rows.clone();
    for rows in perf_rows {
        let name = format!("perf.pp{rows}");
        let Some(prompt) = ctx
            .fixture
            .prompts
            .iter()
            .find(|p| p.kind == "rows" && p.prompt_ids.len() == rows)
            .cloned()
        else {
            continue;
        };
        let target = rows as u64 == TARGET_ROWS;
        let result = (|| -> Result<(Vec<String>, Value), String> {
            let mut times = Vec::new();
            let mut receipts = Vec::new();
            for _ in 0..ctx.fixture.perf_repeats {
                let started = Instant::now();
                drive(ctx, &prompt.prompt_ids, 0, &[])?;
                gpu_error(ctx.gpu.download_f32(&ctx.logits), "download prompt logits")?;
                times.push(started.elapsed().as_secs_f64());
                receipts.push(ctx.bare_receipt());
            }
            let mut bad = Vec::new();
            for r in &receipts {
                bad.extend(ctx.validate(r.as_ref(), rows, target, target));
            }
            let tok_s: Vec<f64> = times.iter().map(|t| rows as f64 / t).collect();
            Ok((
                bad,
                json!({
                    "rows": rows, "prompt_sha256": sha_u32(&prompt.prompt_ids),
                    "seconds": times, "tok_s": tok_s,
                    // The fresh-process observation is the first timed prefill;
                    // later in-process repeats are informational dispersion only.
                    "process_observation_tok_s": tok_s[0],
                    "meets_1400": target.then(|| tok_s[0] >= 1400.0),
                    "meets_2000": target.then(|| tok_s[0] >= 2000.0),
                    "receipts": receipts_json(&receipts),
                }),
            ))
        })();
        match result {
            Ok((bad, detail)) => {
                rows_out.push(detail.clone());
                checks.push(&name, bad, detail);
            }
            Err(e) => checks.fail(&name, e, Value::Null),
        }
    }
    out.insert("prefill".into(), Value::Array(rows_out));
    // Decode tok/s after the last prefill, greedy.
    let decode = (|| -> Result<Value, String> {
        let count = ctx.fixture.decode_tokens;
        let started = Instant::now();
        let (ids, finish) = greedy(ctx, count)?;
        sync(&ctx.gpu)?;
        let seconds = started.elapsed().as_secs_f64();
        Ok(
            json!({"tokens": ids.len(), "seconds": seconds, "tok_s": ids.len() as f64 / seconds, "finish": finish, "ids_sha256": sha_u32(&ids)}),
        )
    })();
    match decode {
        Ok(detail) => {
            out.insert("decode".into(), detail.clone());
            checks.pass("perf.decode", detail);
        }
        Err(e) => checks.fail("perf.decode", e, Value::Null),
    }
}

fn perf_required(fixture: &Fixture) -> Vec<String> {
    let mut names = vec!["perf.warmup".to_string()];
    for &rows in &fixture.perf_rows {
        if fixture
            .prompts
            .iter()
            .any(|p| p.kind == "rows" && p.prompt_ids.len() == rows)
        {
            names.push(format!("perf.pp{rows}"));
        }
    }
    names.push("perf.decode".into());
    names
}

fn product_required(fixture: &Fixture) -> Vec<String> {
    fixture
        .prompts
        .iter()
        .map(|p| format!("product.{}", p.id))
        .collect()
}

// ── driver ─────────────────────────────────────────────────────────────

fn report_bytes(report: &Value) -> Result<Vec<u8>, String> {
    Ok((serde_json::to_string_pretty(report).map_err(|e| e.to_string())? + "\n").into_bytes())
}

/// The report is published atomically and durably, or the write fails.
fn write_report(path: &Path, report: &Value) -> Result<(), String> {
    write_atomic(path, &report_bytes(report)?)
}

/// The complete final report of a run that reached its end, its status and
/// bytes. Nothing partial passes through here: `status` is `Checks::status`,
/// which is `pass` only when every required check executed and none failed.
fn final_report(
    args: &Args,
    checks: &Checks,
    evidence: &serde_json::Map<String, Value>,
    progress: &ProgressLog,
) -> Result<(String, Vec<u8>), String> {
    let summary = checks.json();
    let status = summary["status"].as_str().unwrap_or("fail").to_string();
    let report = json!({
        "schema": REPORT_SCHEMA,
        "mode": args.mode.name(),
        "route": args.route.name(),
        "status": status,
        "model": args.model, "fixture": args.fixture, "manifest": args.manifest,
        "fixture_sha256": file_sha256(&args.fixture)?,
        "manifest_sha256": file_sha256(&args.manifest)?,
        "checks": summary,
        "evidence": evidence,
        "progress": {
            "schema": PROGRESS_SCHEMA,
            "sidecar": progress_path(&args.output),
            "records_before_final": progress.records(),
            "time_sources": PROGRESS_TIME_SOURCES,
        },
    });
    Ok((status, report_bytes(&report)?))
}

fn run(args: &Args, progress: &ProgressLog) -> Result<bool, String> {
    let fixture_json = read_json(&args.fixture)?;
    let fixture = Fixture::parse(&fixture_json)?;
    let manifest = read_json(&args.manifest)?;
    // The route is forced through the one production switch, before the
    // process config snapshot is first read. A conflicting ambient value is
    // refused rather than overridden.
    let forced = if args.route == Expect::LayerMajor {
        "1"
    } else {
        "0"
    };
    if let Ok(ambient) = std::env::var(ROUTE_ENV) {
        if ambient != forced {
            return Err(format!(
                "{ROUTE_ENV}={ambient} conflicts with --route {}",
                args.route.name()
            ));
        }
    } else {
        std::env::set_var(ROUTE_ENV, forced);
    }
    let binary = std::env::current_exe().map_err(|e| e.to_string())?;
    let observed_binary = file_sha256(&binary)?;
    let model_size = std::fs::metadata(&args.model)
        .map_err(|e| format!("stat {}: {e}", args.model.display()))?
        .len();
    // The required list is durable before any GPU work, so a partial run's
    // missing cases are named from it, never inferred.
    let required = required_check_ids(args.mode, &fixture, args.route);
    let required_refs: Vec<&str> = required.iter().map(String::as_str).collect();
    let mut checks = Checks::new(&required_refs);
    progress.record(
        "plan",
        json!({"mode": args.mode.name(), "route": args.route.name(), "required": required,
        "expected_purpose": purpose_name(load_purpose(args.mode)),
        "manifest_profiles": {
            "product_tile_rows": manifest_profile_tile(&manifest, PROFILE_PRODUCT),
            "primitive_tile_rows": manifest_profile_tile(&manifest, PROFILE_PRIMITIVE),
        }}),
    );
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    // Baseline sample: after the HIP context exists, before any model allocation.
    // Only the primitive cell (Hold 1's first cell) runs the heavy telemetry:
    // the smaps/fdinfo background sampler and the non-gating census diagnostic.
    // Timed and quality cells keep their previous pipeline (point diagnostics).
    let heavy_memory = args.mode == Mode::Primitives;
    let pre_load = mem_sample(&gpu, "pre_load", heavy_memory);
    persist_memory(progress, "pre_load", &pre_load, None);
    let mut mem_samples = vec![pre_load];
    let sampler = heavy_memory.then(ProcSampler::start);
    let mut identity_problems = manifest_problems(
        &manifest,
        &Observed {
            binary_sha256: observed_binary.clone(),
            model_size_bytes: model_size,
            gpu_arch: gpu.arch.clone(),
        },
    );
    identity_problems.extend(profile_admission_problems(
        &manifest,
        purpose_tile_rows(AdmissionPurpose::Product) as u64,
        purpose_tile_rows(AdmissionPurpose::PrimitiveSelftest) as u64,
    ));
    checks.push(
        "identity.manifest",
        identity_problems.clone(),
        json!({"binary_sha256": observed_binary, "model_size_bytes": model_size, "gpu_arch": gpu.arch,
               "model_sha256_verified": false, "note": "model sha256 is pinned by the manifest and not recomputed (125 GB)"}),
    );
    progress.checks(&checks);
    let mut out = serde_json::Map::new();
    if identity_problems.is_empty() {
        // The sealed profile of this cell's purpose is the tile every layer-major
        // receipt must name; the identity check above already proved it equals
        // what this binary's loader admits for that purpose.
        let purpose = load_purpose(args.mode);
        let profile = purpose_name(purpose);
        let expected_tile = manifest_profile_tile(&manifest, profile).unwrap_or(0);
        // Expected only: nothing is admitted until a load succeeds.
        out.insert(
            "s1_profile".into(),
            s1_profile_evidence(purpose, &manifest, None, None),
        );
        progress.record("profile", out["s1_profile"].clone());
        let mut fixture = fixture;
        let session_cache = (args.mode == Mode::State).then_some(fixture.session_cache_bytes);
        progress.record(
            "load_start",
            json!({"max_seq": fixture.max_seq, "session_cache_bytes": session_cache}),
        );
        let loaded = match slowlink_load_bundle(
            &mut gpu,
            &args.model,
            fixture.max_seq,
            session_cache,
            load_purpose(args.mode),
        ) {
            Ok(loaded) => loaded,
            Err(error) => {
                // What the process still holds after the failed load, taken
                // while the GPU is still owned. Not the HIP free at the
                // failing allocation.
                let sample = mem_sample(&gpu, "load_failed", heavy_memory);
                persist_memory(progress, "load_failed", &sample, sampler.as_ref());
                // `s1_profile` still has admitted_* null: nothing was admitted.
                progress.record("load_error", json!({"error": error}));
                return Err(error);
            }
        };
        let vocab = loaded.vocab;
        out.insert(
            "load".into(),
            json!({
                "num_layers": loaded.num_layers, "host_mapped_tensors": loaded.host_mapped_tensors,
                "resident_layers": loaded.resident_layers, "placement": loaded.placement,
                "slowlink_profile": loaded.slowlink_profile, "ple_format": loaded.ple_format,
                "vocab": vocab, "max_seq": fixture.max_seq,
            }),
        );
        progress.record("load_end", out["load"].clone());
        // The placement the load actually made: a forced layer-major run needs
        // the S1 admission's joint placement, proved by the scalar witness of
        // the shared quote (the largest count that fits, a host layer
        // remaining; ten resident layers is PREFERRED, information only). The
        // incumbent arm must not carry a profile.
        let mut placement_problems = Vec::new();
        let profile_now = loaded.bundle.slowlink_profile();
        let admitted_tile = profile_now.map(|p| p.tile_rows as u64);
        let admission = profile_now.map(admission_witness);
        let weight_delta: Option<Result<i128, String>> = profile_now.map(|p| {
            layer_payload_bytes(p)
                .ok_or_else(|| {
                    "resident layer bytes are below the layer's backing slack".to_string()
                })
                .and_then(|payload| {
                    reference_weight_delta(
                        &manifest,
                        profile,
                        loaded.resident_layers as u64,
                        payload,
                    )
                })
        });
        let accounting = profile_now.map(|p| {
            placement_accounting(
                &manifest,
                profile,
                loaded.resident_layers as u64,
                loaded.num_layers as u64,
                layer_payload_bytes(p),
            )
        });
        // The same bundle profile feeds the evidence and the load.placement row.
        out.insert(
            "s1_profile".into(),
            s1_profile_evidence(
                purpose,
                &manifest,
                admitted_from(profile_now),
                admission.as_ref(),
            ),
        );
        progress.record("profile_admitted", out["s1_profile"].clone());
        match args.route {
            Expect::LayerMajor => {
                if !loaded.slowlink_profile || loaded.placement != "slowlink-profile" {
                    placement_problems.push(format!(
                        "layer-major was forced but the load did not admit the S1 profile (placement {})",
                        loaded.placement
                    ));
                }
                // The actual admitted profile against the sealed ledger of this
                // cell's purpose: purpose, tile and window. The sealed resident
                // count is reference data and is not compared.
                placement_problems.extend(profile_load_problems(
                    &manifest,
                    profile,
                    profile_now.map_or("none", |p| purpose_name(p.purpose)),
                    admitted_tile,
                    profile_now.map(|p| p.window_rows as u64),
                ));
                // No resident floor: the actual count must be maximal-fits.
                placement_problems.extend(admission_witness_problems(
                    admission.as_ref(),
                    &AdmissionActual {
                        resident_layers: Some(loaded.resident_layers as u64),
                        trunk_layers: Some(loaded.num_layers as u64),
                        snapshot_budget_bytes: profile_now.map(|p| p.device_snapshot_budget_bytes),
                    },
                ));
            }
            Expect::Incumbent => {
                if loaded.slowlink_profile {
                    placement_problems.push("the incumbent arm loaded an S1 profile".to_string());
                }
            }
        }
        let (arena_census, arena_problems) = arena_observation(&loaded.bundle, &gpu);
        placement_problems.extend(arena_presence_problems(
            arena_census["present"].as_bool() == Some(true),
            loaded.slowlink_profile,
        ));
        placement_problems.extend(arena_problems);
        checks.push(
            "load.placement",
            placement_problems,
            json!({"placement": loaded.placement, "resident_layers": loaded.resident_layers,
                   "host_mapped_tensors": loaded.host_mapped_tensors, "slowlink_profile": loaded.slowlink_profile,
                   "expected_purpose": purpose_name(purpose), "expected_tile_rows": expected_tile,
                   "admitted_purpose": admitted_from(profile_now).map(|a| a.0),
                   "admitted_tile_rows": admitted_tile,
                   "admitted_device_snapshot_budget_bytes": profile_now.map(|p| p.device_snapshot_budget_bytes),
                   "admission": admission,
                   "preferred_resident_layers": PREFERRED_RESIDENT_LAYERS,
                   "preferred_note": "ten resident layers is PREFERRED, information only: the count is the largest that fits, proven by the admission witness",
                   "placement_accounting": accounting,
                   "granule_census": profile_now.map_or(Value::Null, profile_granule_evidence),
                   "arena_census": arena_census,
                   "pressure_host_policy": usage_json(loaded.bundle.session_cache())["pressure_host_policy"].clone()}),
        );
        progress.checks(&checks);
        // A failure from here on still owns the GPU: its point sample is
        // persisted before the error leaves the run.
        let run_failed = |gpu: &Gpu, error: String| {
            let sample = mem_sample(gpu, "run_failed", heavy_memory);
            persist_memory(progress, "run_failed", &sample, sampler.as_ref());
            error
        };
        // Natural-text prompts are tokenized with the artifact's own tokenizer.
        for prompt in fixture.prompts.iter_mut() {
            if let Some(text) = &prompt.text {
                prompt.prompt_ids = loaded.tokenizer.encode(text);
                if prompt.prompt_ids.is_empty()
                    || prompt.prompt_ids.len() + fixture.decode_tokens + 1 > fixture.max_seq
                {
                    return Err(run_failed(
                        &gpu,
                        format!(
                            "natural prompt {} tokenizes to {} ids, outside 1..max_seq",
                            prompt.id,
                            prompt.prompt_ids.len()
                        ),
                    ));
                }
            }
        }
        let logits = match gpu.zeros(&[vocab], DType::F32) {
            Ok(logits) => logits,
            Err(e) => return Err(run_failed(&gpu, e.to_string())),
        };
        let post_load = mem_sample(&gpu, "post_load", sampler.is_some());
        persist_memory(progress, "post_load", &post_load, sampler.as_ref());
        mem_samples.push(post_load);
        let mut ctx = Ctx {
            gpu,
            bundle: loaded.bundle,
            tokenizer: loaded.tokenizer,
            logits,
            vocab,
            eos: loaded.eos_token,
            fixture,
            route: args.route,
            expected_tile,
            mem: mem_samples,
            progress: progress.clone(),
        };
        progress.record(
            "stage_start",
            json!({"stage": args.mode.name(), "cache_usage": ctx.cache_usage()}),
        );
        match args.mode {
            Mode::Primitives => run_primitives(&mut ctx, &mut checks, &mut out, sampler.as_ref()),
            Mode::State => run_state(&mut ctx, &mut checks, &mut out),
            Mode::Product => run_product(&mut ctx, &mut checks, &mut out),
            Mode::Perf => run_perf(&mut ctx, &mut checks, &mut out),
        }
        progress.record(
            "stage_end",
            json!({"stage": args.mode.name(), "cache_usage": ctx.cache_usage()}),
        );
        progress.checks(&checks);
        // The lazy shadow roots are allocated by now: the same census again.
        // A failure row is recorded only when a root's capacity breaks the
        // exact proof; a clean census adds no check row and no new verdict.
        let (arena_end, arena_end_problems) = arena_observation(&ctx.bundle, &ctx.gpu);
        progress.record("arena_census_end", arena_end.clone());
        out.insert("arena_census_end".into(), arena_end);
        if !arena_end_problems.is_empty() {
            checks.fail(
                "load.arena_capacity",
                arena_end_problems.join("; "),
                Value::Null,
            );
        }
        if let Some(sampler) = sampler {
            // The primitive cell (Hold 1's first cell) only: the heavy sampler.
            // Its census residual is recorded as a non-gating diagnostic.
            ctx.sample_memory("end");
            if let Some(end) = ctx.mem.last() {
                persist_memory(progress, "end", end, Some(&sampler));
            }
            let mut samples = std::mem::take(&mut ctx.mem);
            samples.push(sampler.finish());
            out.insert(
                "memory_observations".into(),
                memory_diagnostic(
                    &samples,
                    memory_ledger(&manifest, purpose_name(load_purpose(args.mode))).and_then(
                        |ledger| match &weight_delta {
                            Some(Ok(delta)) => ledger.at_actual_placement(*delta),
                            Some(Err(error)) => Err(error.clone()),
                            None => Ok(ledger),
                        },
                    ),
                ),
            );
        } else {
            // Every other cell (state, product, timed perf): no sampler thread,
            // no smaps, no census requirement. Three light point diagnostics,
            // taken outside any timed region, are recorded and never judged.
            let mut samples = std::mem::take(&mut ctx.mem);
            let end = mem_sample(&ctx.gpu, "end", false);
            persist_memory(progress, "end", &end, None);
            samples.push(end);
            out.insert(
                "memory_diagnostics".into(),
                json!({"required": false, "samples": samples}),
            );
        }
        progress.record("cache", json!({"phase": "end", "usage": ctx.cache_usage()}));
        let peak = ctx
            .gpu
            .hip
            .get_vram_info()
            .ok()
            .map(|(free, total)| json!({"free": free, "total": total}));
        out.insert("vram_after".into(), peak.unwrap_or(Value::Null));
        let Ctx {
            mut gpu,
            bundle,
            logits,
            ..
        } = ctx;
        let _ = gpu.free_tensor(logits);
        if let Err(e) = bundle.free_gpu(&mut gpu) {
            checks.fail(
                "teardown",
                format!("bundle teardown failed: {e}"),
                Value::Null,
            );
        }
        progress.checks(&checks);
    }
    // The sidecar's own durability is a required check: a lost record is a
    // gate failure, never silent. The final record carries the report's
    // digest; if THAT record cannot be written the report is rebuilt failing.
    checks.push(
        "progress.durability",
        progress.durability_problems(),
        json!({"sidecar": progress_path(&args.output), "records": progress.records()}),
    );
    let durable_before_final = progress.error().is_none();
    let (mut status, mut bytes) = final_report(args, &checks, &out, progress)?;
    progress.record(
        "final",
        json!({
            "status": status, "report": args.output,
            "report_sha256": format!("{:x}", Sha256::digest(&bytes)),
        }),
    );
    if let (true, Some(error)) = (durable_before_final, progress.error()) {
        checks.fail(
            "progress.durability",
            format!("the final record was not durable: {error}"),
            Value::Null,
        );
        (status, bytes) = final_report(args, &checks, &out, progress)?;
    }
    write_atomic(&args.output, &bytes)?;
    eprintln!(
        "qwen4_slowlink_gate: {} {} -> {status}",
        args.mode.name(),
        args.route.name()
    );
    Ok(status == "pass")
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// An aborted run's report: always `incomplete`, with the error and a summary
/// of the durable sidecar (never a pass, never filled-in cases).
fn aborted_report(args: &Args, error: &str, progress: Option<&ProgressLog>) -> Value {
    let sidecar = progress_path(&args.output);
    let summary = match std::fs::read_to_string(&sidecar) {
        Ok(text) => progress_summary(&text),
        Err(e) => json!({"unreadable": format!("read {}: {e}", sidecar.display())}),
    };
    json!({
        "schema": REPORT_SCHEMA, "mode": args.mode.name(), "route": args.route.name(),
        "status": "incomplete", "error": error,
        "progress": {
            "schema": PROGRESS_SCHEMA, "sidecar": sidecar,
            "write_error": progress.and_then(|p| p.error()), "summary": summary,
        },
    })
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return;
    }
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("qwen4_slowlink_gate: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let mut clock = system_clock();
    let (started_ms, _) = clock();
    let run_id = format!("{}-{started_ms}", std::process::id());
    let progress = match ProgressLog::open_file(&progress_path(&args.output), run_id, true) {
        Ok(progress) => progress,
        Err(e) => {
            // No durable progress is a gate failure before any GPU work.
            eprintln!("qwen4_slowlink_gate: {e}");
            if let Err(w) = write_report(&args.output, &aborted_report(&args, &e, None)) {
                eprintln!("qwen4_slowlink_gate: {w}");
            }
            std::process::exit(1);
        }
    };
    progress.record(
        "run_start",
        json!({
            "mode": args.mode.name(), "route": args.route.name(), "pid": std::process::id(),
            "model": args.model, "fixture": args.fixture, "manifest": args.manifest,
            "output": args.output, "output_preexisting": args.output.exists(),
            "report_schema": REPORT_SCHEMA, "time_sources": PROGRESS_TIME_SOURCES,
        }),
    );
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&args, &progress)))
        .unwrap_or_else(|payload| Err(format!("gate panicked: {}", panic_message(&*payload))));
    match outcome {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            // An aborted run still leaves a report: it did not finish, so
            // INCOMPLETE, and the error is written through first.
            progress.record("abort", json!({"error": e}));
            if let Err(w) = write_report(&args.output, &aborted_report(&args, &e, Some(&progress)))
            {
                eprintln!("qwen4_slowlink_gate: {w}");
            }
            eprintln!("qwen4_slowlink_gate: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod slowlink_receipt_tests {
    use super::*;

    #[test]
    fn slowlink_receipt_load_purpose_is_selftest_only_for_primitives() {
        assert_eq!(
            load_purpose(Mode::Primitives),
            AdmissionPurpose::PrimitiveSelftest
        );
        for mode in [Mode::State, Mode::Product, Mode::Perf] {
            assert_eq!(
                load_purpose(mode),
                AdmissionPurpose::Product,
                "{} is a product cell",
                mode.name()
            );
        }
    }

    fn case_report(name: &str, status: &str) -> Value {
        match status {
            "pass" => json!({"name": name, "status": "pass", "mismatches": 0, "detail": {}}),
            "error" => json!({
                "name": name, "status": "error", "mismatches": null,
                "detail": {"error": format!("{name}: hipMalloc failed")}
            }),
            _ => json!({"name": name, "status": "fail", "mismatches": 3, "detail": {}}),
        }
    }

    fn primitive_rows(statuses: &[&str]) -> (Checks, serde_json::Map<String, Value>, Vec<String>) {
        let ids: Vec<String> = PRIMITIVE_CASES
            .iter()
            .map(|case| format!("primitive.{case}"))
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut checks = Checks::new(&refs);
        let mut out = serde_json::Map::new();
        let cases: Vec<Value> = PRIMITIVE_CASES
            .iter()
            .zip(statuses)
            .map(|(name, status)| case_report(name, status))
            .collect();
        record_primitives(&mut checks, &mut out, Ok(json!({"cases": cases})));
        (checks, out, ids)
    }

    fn rows(checks: &Checks) -> Vec<Value> {
        checks.json()["checks"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    }

    fn row(checks: &Checks, name: &str) -> Value {
        rows(checks)
            .into_iter()
            .find(|r| r["name"] == name)
            .unwrap_or_else(|| panic!("no row {name}"))
    }

    #[test]
    fn slowlink_receipt_first_or_middle_primitive_error_does_not_stop_the_rest() {
        for bad in [0usize, 3] {
            let mut statuses = vec!["pass"; PRIMITIVE_CASES.len()];
            statuses[bad] = "error";
            let (checks, _, ids) = primitive_rows(&statuses);
            assert_eq!(checks.status(), "fail");
            assert!(checks.missing().is_empty(), "every case was judged alone");
            for (index, id) in ids.iter().enumerate() {
                let r = row(&checks, id);
                if index == bad {
                    assert_eq!(r["status"], "fail", "{id}");
                    let reason = r["reason"].as_str().unwrap();
                    assert!(reason.contains("case error:"), "{reason}");
                    assert!(reason.contains("hipMalloc failed"), "{reason}");
                    assert!(
                        r["detail"]["mismatches"].is_null(),
                        "an error is not an arithmetic mismatch count"
                    );
                } else {
                    assert_eq!(r["status"], "pass", "{id} must still be proved");
                }
            }
        }
    }

    #[test]
    fn slowlink_receipt_primitive_fail_keeps_its_own_mismatch_count() {
        let mut statuses = vec!["pass"; PRIMITIVE_CASES.len()];
        statuses[1] = "fail";
        let (checks, _, ids) = primitive_rows(&statuses);
        assert_eq!(checks.status(), "fail");
        let r = row(&checks, &ids[1]);
        assert!(r["reason"].as_str().unwrap().contains("mismatches 3"));
        assert_eq!(
            ids.iter()
                .filter(|id| row(&checks, id)["status"] == "fail")
                .count(),
            1
        );
    }

    #[test]
    fn slowlink_receipt_global_selftest_error_is_one_failure_and_leaves_cases_missing() {
        let ids: Vec<String> = PRIMITIVE_CASES
            .iter()
            .map(|case| format!("primitive.{case}"))
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut checks = Checks::new(&refs);
        let mut out = serde_json::Map::new();
        record_primitives(
            &mut checks,
            &mut out,
            Err("slow-link primitive check needs an admitted slow-link profile".to_string()),
        );
        assert_eq!(checks.status(), "fail");
        let all = rows(&checks);
        let fails: Vec<&Value> = all.iter().filter(|r| r["status"] == "fail").collect();
        assert_eq!(
            fails.len(),
            1,
            "one initialization failure, not one per case"
        );
        assert_eq!(fails[0]["name"], "primitive.init");
        assert_eq!(
            checks.missing().len(),
            PRIMITIVE_CASES.len(),
            "every case stays unproved"
        );
        assert!(out.contains_key("selftest_error"));
    }

    #[test]
    fn slowlink_receipt_absent_primitive_case_stays_missing_not_pass() {
        let ids: Vec<String> = PRIMITIVE_CASES
            .iter()
            .map(|case| format!("primitive.{case}"))
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut checks = Checks::new(&refs);
        let mut out = serde_json::Map::new();
        let cases: Vec<Value> = PRIMITIVE_CASES[..2]
            .iter()
            .map(|n| case_report(n, "pass"))
            .collect();
        record_primitives(&mut checks, &mut out, Ok(json!({"cases": cases})));
        assert_eq!(checks.status(), "incomplete");
        assert_eq!(checks.missing().len(), PRIMITIVE_CASES.len() - 2);
    }

    fn receipt() -> Value {
        json!({
            "schema": RECEIPT_SCHEMA, "route": "layer-major", "selected_by": "env1", "bypass_reason": null,
            "entry_pos": 0, "prompt_rows": 16384, "windows": [{"start": 0, "rows": 16384}],
            "launch_cuts": [1536, 3072], "tile_rows": 1536, "retained_scratch_rows": 1536,
            "sweeps": 1, "resident_layers": 10, "host_layers": 38,
            "host_expert_bytes_total": 50_803_507_200u64, "submitted_bytes": 50_803_507_200u64,
            "useful_bytes": 50_000_000_000u64, "unused_prefetched_bytes": 803_507_200u64,
            "reserve": {"tile": 1u64 << 30, "arena": 1u64 << 29, "ring": 167_772_160u64,
                        "lookahead": 1_336_000_000u64, "tables": 1u64 << 20, "total": 4_000_000_000u64},
            "spans_ms": {"wall": 11000.0, "copy": 7200.0, "compute": 3000.0, "route_barrier": 100.0, "nonexpert": 1500.0},
            "overlap": {"cross_layer_intersections": 37u64, "cross_layer_overlap_ms": 900.0,
                        "copy_idle_ms": 50.0, "page_full_stalls": 3u64, "copy_starved_stalls": 1u64},
            "peak_vram_bytes": 23_000_000_000u64,
        })
    }

    fn incumbent() -> Value {
        let mut r = receipt();
        r["route"] = json!("incumbent");
        r["selected_by"] = json!("env0");
        r["reserve"] =
            json!({"tile": 0, "arena": 0, "ring": 0, "lookahead": 0, "tables": 0, "total": 0});
        r["overlap"] = json!({"cross_layer_intersections": 0, "cross_layer_overlap_ms": 0.0, "copy_idle_ms": 0.0, "page_full_stalls": 0, "copy_starved_stalls": 0});
        r
    }

    fn single() -> Limits {
        Limits {
            require_single_window: true,
            require_overlap: true,
            ..Limits::default()
        }
    }

    fn problems(r: &Value) -> Vec<String> {
        validate(Some(r), Expect::LayerMajor, &single())
    }

    #[test]
    fn slowlink_receipt_good_layer_major_and_incumbent_pass() {
        assert_eq!(problems(&receipt()), Vec::<String>::new());
        assert_eq!(
            validate(Some(&incumbent()), Expect::Incumbent, &Limits::default()),
            Vec::<String>::new()
        );
    }

    #[test]
    fn slowlink_receipt_missing_receipt_and_missing_route_fail_closed() {
        assert!(!validate(None, Expect::LayerMajor, &single()).is_empty());
        for key in [
            "route",
            "windows",
            "sweeps",
            "reserve",
            "spans_ms",
            "overlap",
            "launch_cuts",
            "schema",
        ] {
            let mut r = receipt();
            r.as_object_mut().unwrap().remove(key);
            assert!(!problems(&r).is_empty(), "removing {key} must fail");
        }
        let mut r = receipt();
        r["route"] = json!("maybe");
        assert!(!problems(&r).is_empty());
    }

    #[test]
    fn slowlink_receipt_forced_layer_major_that_ran_incumbent_fails_with_reason() {
        let mut r = incumbent();
        r["bypass_reason"] = json!("link is not narrow");
        let p = validate(Some(&r), Expect::LayerMajor, &Limits::default());
        assert!(p.iter().any(|m| m.contains("link is not narrow")), "{p:?}");
        // And an incumbent arm that ran layer-major is refused too.
        assert!(!validate(Some(&receipt()), Expect::Incumbent, &Limits::default()).is_empty());
        // The incumbent must not report S1 reserves.
        let mut r = incumbent();
        r["reserve"]["total"] = json!(1);
        assert!(!validate(Some(&r), Expect::Incumbent, &Limits::default()).is_empty());
    }

    #[test]
    fn slowlink_receipt_target_needs_exactly_one_window_and_one_sweep_per_window() {
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 8192}, {"start": 8192, "rows": 8192}]);
        assert!(problems(&r).iter().any(|m| m.contains("one required")));
        r["sweeps"] = json!(2);
        r["submitted_bytes"] = json!(2 * 50_803_507_200u64);
        // Two windows are consistent per se, but never a 16K target pass.
        assert!(problems(&r).iter().any(|m| m.contains("one required")));
        let mut r = receipt();
        r["sweeps"] = json!(2);
        assert!(problems(&r).iter().any(|m| m.contains("once per window")));
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 1000}]);
        assert!(problems(&r).iter().any(|m| m.contains("cover")));
    }

    #[test]
    fn slowlink_receipt_residency_tile_scratch_and_cold_bytes_are_enforced() {
        for (key, value) in [
            // The counts must close on the trunk: 9 resident + 38 host, or 10
            // resident + 39 host, is not a placement of the 48 layers.
            ("resident_layers", json!(9)),
            ("host_layers", json!(39)),
            ("tile_rows", json!(8192)),
            ("retained_scratch_rows", json!(8192)),
            ("host_expert_bytes_total", json!(57_488_179_200u64)),
        ] {
            let mut r = receipt();
            r[key] = value;
            assert!(!problems(&r).is_empty(), "{key} must be refused");
        }
        let mut r = receipt();
        r["submitted_bytes"] = json!(60_000_000_000u64);
        assert!(!problems(&r).is_empty(), "more than one sweep of bytes");
        let mut r = receipt();
        r["useful_bytes"] = json!(51_000_000_000u64);
        assert!(!problems(&r).is_empty(), "useful above submitted");
    }

    /// A layer-major receipt of `resident` resident layers on the 48-layer
    /// trunk: the host layers carry the per-layer bound's bytes.
    fn adaptive_receipt(resident: u64) -> Value {
        let host = TRUNK_LAYERS - resident;
        let bytes = host * COLD_BYTES_PER_HOST_LAYER;
        let mut r = receipt();
        r["resident_layers"] = json!(resident);
        r["host_layers"] = json!(host);
        r["host_expert_bytes_total"] = json!(bytes);
        r["submitted_bytes"] = json!(bytes);
        r["useful_bytes"] = json!(bytes / 2);
        r["unused_prefetched_bytes"] = json!(bytes / 2);
        r
    }

    #[test]
    fn slowlink_receipt_has_no_resident_floor_nor_host_layer_cap() {
        // Nine (fewer than the preferred ten), ten, eleven, zero and one host
        // layer all pass when the counts close on the trunk; ten is data.
        assert_eq!(PREFERRED_RESIDENT_LAYERS, 10);
        for resident in [0u64, 1, 9, 10, 11, 40, 47] {
            let r = adaptive_receipt(resident);
            let bad = problems(&r);
            assert!(bad.is_empty(), "resident {resident}: {bad:?}");
        }
        // All layers resident leaves nothing to stream: not a layer-major claim.
        let all = adaptive_receipt(TRUNK_LAYERS);
        assert!(problems(&all).iter().any(|m| m.contains("host_layers 0")));
        // The per-host-layer byte bound scales with the actual host layers.
        let mut r = adaptive_receipt(9);
        let bound = 39 * COLD_BYTES_PER_HOST_LAYER;
        r["host_expert_bytes_total"] = json!(bound + 1);
        r["submitted_bytes"] = json!(bound + 1);
        assert!(problems(&r).iter().any(|m| m.contains("host layer(s)")));
        // Counts that do not close on the trunk are an accounting error at any N.
        let mut r = adaptive_receipt(9);
        r["host_layers"] = json!(38);
        assert!(problems(&r).iter().any(|m| m.contains("trunk layers")));
    }

    #[test]
    fn slowlink_receipt_unfunded_lookahead_reserve_is_refused() {
        let mut r = receipt();
        r["reserve"]["lookahead"] = json!(0);
        r["reserve"]["ring"] = json!(167_772_160u64);
        assert!(problems(&r).iter().any(|m| m.contains("lookahead")));
        let mut r = receipt();
        r["reserve"]["total"] = json!(10);
        assert!(problems(&r).iter().any(|m| m.contains("sum of its parts")));
    }

    #[test]
    fn slowlink_receipt_overlap_win_needs_cross_layer_event_intersections() {
        let mut r = receipt();
        r["overlap"]["cross_layer_intersections"] = json!(0);
        assert!(!overlap_claim_allowed(&r));
        assert!(problems(&r).iter().any(|m| m.contains("cross-layer")));
        assert!(overlap_claim_allowed(&receipt()));
        // Intersections without measured overlap time are not evidence either.
        let mut r = receipt();
        r["overlap"]["cross_layer_overlap_ms"] = json!(0.0);
        assert!(!overlap_claim_allowed(&r));
    }

    #[test]
    fn slowlink_receipt_unequal_manifests_cannot_claim_bitwise_transport() {
        let s1 = receipt();
        let mut incumbent8192 = incumbent();
        incumbent8192["tile_rows"] = json!(8192);
        incumbent8192["launch_cuts"] = json!([8192]);
        assert!(bitwise_transport_claim(&incumbent8192, &s1).is_err());
        // Even equal manifests on different routes are not transport-only.
        let mut other = incumbent();
        other["launch_cuts"] = s1["launch_cuts"].clone();
        assert!(bitwise_transport_claim(&other, &s1).is_err());
        assert!(bitwise_transport_claim(&s1, &s1.clone()).is_ok());
    }

    #[test]
    fn slowlink_receipt_zero_and_unexecuted_required_checks_are_incomplete() {
        let empty = Checks::new(&[]);
        assert_eq!(empty.status(), "incomplete");
        let mut checks = Checks::new(&["a", "b"]);
        checks.pass("a", Value::Null);
        assert_eq!(checks.status(), "incomplete");
        assert_eq!(checks.missing(), vec!["b".to_string()]);
        checks.skip("b", "not applicable");
        assert_eq!(
            checks.status(),
            "incomplete",
            "a skipped required check is not executed"
        );
        checks.pass("b", Value::Null);
        assert_eq!(checks.status(), "pass");
        checks.fail("b", "later failure", Value::Null);
        assert_eq!(checks.status(), "fail");
    }

    /// One sealed purpose-profile ledger: core = weights + tile census + extras,
    /// owned = core + sequential diag.
    fn profile_ledger(purpose: &str, tile: u64, census: u64, diag: u64) -> Value {
        let (weights, extras) = (20 * GIB, 2 * GIB);
        let core = weights + census + extras;
        json!({
            "purpose": purpose, "tile_rows": tile, "resident_expert_layers": 10,
            "projection_role": "reference",
            "window_rows": 16384, "page_count": 18,
            "device_weights_bytes": weights,
            "s1_tile_census_bytes": census,
            "s1_tile_census_terms": [
                {"name": "f16_weight_shadow_bytes", "bytes": census / 2},
                {"name": "tile_workspace_bytes", "bytes": census - census / 2},
            ],
            "f16_shadow_cold_bound_bytes": census / 2,
            "s1_extras_bytes": extras,
            "s1_extras_terms": [
                {"name": "ring_bytes", "bytes": extras / 2},
                {"name": "route_storage_bytes", "bytes": extras - extras / 2},
            ],
            "device_core_bytes": core, "sequential_diag_bytes": diag,
            "device_owned_bytes": core + diag, "runtime_budget_bytes": 24 * GIB,
            "safety_margin_bytes": GIB / 2,
        })
    }

    fn manifest() -> Value {
        json!({
            "source": {"base": "d8865a16b5d0e01eff712df85d2754122f6edaee", "tip": "a".repeat(40)},
            "binary": {"sha256": "b".repeat(64)},
            "model": {"path": "/m.mq4", "size_bytes": 125288544792u64, "sha256": "c".repeat(64)},
            "ple": {"source": "sidecar", "format": "q4", "sidecars": ["/m.mq4.ple-q4"]},
            "kernel": {"code_object_hashes": {"k": "d"}},
            "host": {"hostname": "hipx", "topology": "walk"},
            "device": {"uuid": "GPU-43390a851e296ee5", "arch": "gfx1100"},
            "link": {"gbps": 7.09, "topology_walk": "x4"},
            "memory_ledger": {
                "device_gib": 23.0, "host_unreclaimable_gib": 90.0,
                "device_owned_bytes": 23 * GIB + GIB / 2,
                "profiles": {
                    "product": profile_ledger("product", 768, GIB, 0),
                    "primitive_selftest": profile_ledger("primitive_selftest", 512, GIB / 2, GIB),
                },
            },
        })
    }

    fn observed() -> Observed {
        Observed {
            binary_sha256: "b".repeat(64),
            model_size_bytes: 125288544792,
            gpu_arch: "gfx1100".into(),
        }
    }

    #[test]
    fn slowlink_receipt_identity_mismatch_and_missing_manifest_keys_fail_closed() {
        assert_eq!(
            manifest_problems(&manifest(), &observed()),
            Vec::<String>::new()
        );
        let mut wrong_binary = observed();
        wrong_binary.binary_sha256 = "e".repeat(64);
        assert!(manifest_problems(&manifest(), &wrong_binary)
            .iter()
            .any(|m| m.contains("binary identity")));
        let mut wrong_model = observed();
        wrong_model.model_size_bytes += 1;
        assert!(manifest_problems(&manifest(), &wrong_model)
            .iter()
            .any(|m| m.contains("model identity")));
        for path in MANIFEST_REQUIRED {
            let mut m = manifest();
            let mut node = &mut m;
            let parts: Vec<&str> = path.split('.').collect();
            for part in &parts[..parts.len() - 1] {
                node = node.get_mut(*part).unwrap();
            }
            node.as_object_mut().unwrap().remove(parts[parts.len() - 1]);
            assert!(
                !manifest_problems(&m, &observed()).is_empty(),
                "{path} must be required"
            );
        }
        let mut short = manifest();
        short["source"]["tip"] = json!("abc123");
        assert!(manifest_problems(&short, &observed())
            .iter()
            .any(|m| m.contains("40-hex")));
    }

    fn kld_receipt() -> Value {
        json!({
            "schema": "hipfire.qwen4.s11ref_klrows.v1", "rows": 6120, "sequences": 24, "route": "decode",
            "n_ctx": 512, "estimator": "e",
            "model_sha256": "1".repeat(64), "ref_sha256": "2".repeat(64), "corpus_sha256": "3".repeat(64),
            "binary_sha256": "4".repeat(64), "output_sha256": "5".repeat(64),
        })
    }

    #[test]
    fn slowlink_receipt_partial_corpus_fails_closed() {
        assert!(kld_receipt_problems(&kld_receipt()).is_empty());
        let mut partial = kld_receipt();
        partial["rows"] = json!(5000);
        assert!(!kld_receipt_problems(&partial).is_empty());
        let mut part = kld_receipt();
        part["schema"] = json!("hipfire.qwen4.s11ref_klrows.partition.v1");
        assert!(!kld_receipt_problems(&part).is_empty());
        let mut seqs = kld_receipt();
        seqs["sequences"] = json!(10);
        assert!(!kld_receipt_problems(&seqs).is_empty());
        let mut other_ref = kld_receipt();
        other_ref["ref_sha256"] = json!("9".repeat(64));
        assert!(!kld_receipts_comparable(&kld_receipt(), &other_ref).is_empty());
        assert!(kld_receipts_comparable(&kld_receipt(), &kld_receipt()).is_empty());
    }

    #[test]
    fn slowlink_receipt_paired_interval_aggregates_per_sequence_before_the_ci() {
        // 4 sequences with very different row counts: the unit is the sequence.
        let base = [0.1, 0.1, 0.1, 0.2, 0.2];
        let cand = [0.1, 0.1, 0.1, 0.2, 0.2];
        let means = sequence_means(&base, &[3, 1, 1]).unwrap();
        assert_eq!(means.len(), 3);
        assert!((means[0] - 0.1).abs() < 1e-12);
        assert!(
            sequence_means(&base, &[3, 1]).is_err(),
            "row counts must cover the rows"
        );
        assert!(
            sequence_means(&base, &[5, 0]).is_err(),
            "an empty sequence is refused"
        );
        let ci = paired_ci(&means, &sequence_means(&cand, &[3, 1, 1]).unwrap()).unwrap();
        assert_eq!((ci.mean, ci.lo, ci.hi), (0.0, 0.0, 0.0));
        assert!(paired_verdict(&ci));
        // A clearly worse candidate with a wide interval above the limit fails.
        let b: Vec<f64> = (0..24).map(|i| 0.05 + 0.001 * i as f64).collect();
        let jitter: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.05 + 0.0001 * (i % 3) as f64)
            .collect();
        let ci = paired_ci(&b, &jitter).unwrap();
        assert!(
            ci.lo > 0.0 && ci.hi > KLD_HI_LIMIT && !paired_verdict(&ci),
            "{ci:?}"
        );
        // Significantly worse but tiny passes via the hi bound.
        let tiny: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.002 + 0.0001 * (i % 3) as f64)
            .collect();
        let ci = paired_ci(&b, &tiny).unwrap();
        assert!(
            ci.lo > 0.0 && ci.hi <= KLD_HI_LIMIT && paired_verdict(&ci),
            "{ci:?}"
        );
        // An interval straddling zero passes.
        let noisy: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { 0.004 } else { -0.004 })
            .collect();
        assert!(paired_verdict(&paired_ci(&b, &noisy).unwrap()));
        assert!(paired_ci(&b[..1], &noisy[..1]).is_err());
        assert!(paired_ci(&b, &noisy[..5]).is_err());
        assert!((t_975(23) - 2.069).abs() < 1e-9);
    }

    const CUTS: [usize; 5] = [3, 4, 1536, 1537, 8192];

    /// A 9000-row boundary prompt: marker id 7 at row `cut - 1` of every cut.
    fn boundary_ids() -> Vec<u32> {
        let mut ids = vec![1500u32; 9000];
        for cut in CUTS {
            ids[cut - 1] = 7;
        }
        ids
    }

    fn fixture() -> Value {
        let mut prompts = Vec::new();
        for rows in REQUIRED_PROMPT_ROWS {
            prompts.push(
                json!({"id": format!("p{rows}"), "kind": "rows", "prompt_ids": vec![7u32; rows]}),
            );
        }
        prompts.push(json!({"id": "irr", "kind": "irregular", "prompt_ids": vec![7u32; 3000], "launch_cuts": [1000, 2500]}));
        prompts.push(
            json!({"id": "res", "kind": "resume", "prompt_ids": vec![7u32; 3000], "entry": 1536}),
        );
        prompts.push(json!({"id": "bnd", "kind": "boundary", "prompt_ids": boundary_ids(), "requested_boundaries": CUTS}));
        prompts.push(json!({"id": "nat", "kind": "natural", "prompt_ids": vec![7u32; 100]}));
        json!({
            "schema": FIXTURE_SCHEMA, "max_seq": 16640, "decode_tokens": 8,
            "repeats": 3, "head_rows": 4, "prompts": prompts, "continuation_ids": [5, 6, 7],
            "failure_injections": FAILURE_POINTS, "boundary_token_id": 7,
        })
    }

    #[test]
    fn slowlink_receipt_fixture_refuses_a_narrowed_case_list_and_unbounded_head_scoring() {
        let parsed = Fixture::parse(&fixture()).unwrap();
        assert_eq!(parsed.prompts.len(), 9);
        for rows in REQUIRED_PROMPT_ROWS {
            let mut f = fixture();
            let prompts = f["prompts"].as_array_mut().unwrap();
            prompts.retain(|p| {
                p["prompt_ids"].as_array().map(|a| a.len()) != Some(rows) || p["kind"] != "rows"
            });
            assert!(Fixture::parse(&f).is_err(), "{rows}-row prompt is required");
        }
        for kind in ["irregular", "resume", "boundary", "natural"] {
            let mut f = fixture();
            f["prompts"]
                .as_array_mut()
                .unwrap()
                .retain(|p| p["kind"] != kind);
            assert!(Fixture::parse(&f).is_err(), "{kind} prompt is required");
        }
        let mut f = fixture();
        f["failure_injections"] = json!(["before_dma"]);
        assert!(Fixture::parse(&f).is_err());
        let mut f = fixture();
        f["head_rows"] = json!(1000);
        assert!(
            Fixture::parse(&f).is_err(),
            "whole-window scoring must stay bounded"
        );
        let mut f = fixture();
        f["repeats"] = json!(1);
        assert!(Fixture::parse(&f).is_err());
        let mut f = fixture();
        f["max_seq"] = json!(16000);
        assert!(Fixture::parse(&f).is_err());
    }

    #[test]
    fn slowlink_receipt_state_required_checks_cover_every_prompt_and_injection() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        let s1 = state_required(&fixture, Expect::LayerMajor);
        for point in FAILURE_POINTS {
            assert!(s1.contains(&format!("state.inject.{point}")));
        }
        assert!(s1.contains(&"state.p16384.route_receipt".to_string()));
        assert!(s1.contains(&"state.irr.launch_cut_chronology".to_string()));
        assert!(!state_required(&fixture, Expect::Incumbent)
            .contains(&"state.inject.before_dma".to_string()));
        assert_eq!(product_required(&fixture).len(), fixture.prompts.len());
        assert!(perf_required(&fixture).contains(&"perf.pp16384".to_string()));
    }

    #[test]
    fn slowlink_receipt_cli_is_frozen_and_requires_absolute_paths() {
        let ok = |extra: &[&str]| {
            let mut v: Vec<String> = [
                "--model",
                "/m",
                "--fixture",
                "/f.json",
                "--output",
                "/o.json",
                "--mode",
                "state",
                "--route",
                "layer-major",
                "--manifest",
                "/mf.json",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            parse_args(v)
        };
        let args = ok(&[]).unwrap();
        assert_eq!((args.mode, args.route), (Mode::State, Expect::LayerMajor));
        assert!(ok(&["--surprise", "x"]).is_err());
        assert!(parse_args(
            [
                "--model",
                "rel",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "state",
                "--route",
                "incumbent",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "state",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "primitives",
                "--route",
                "incumbent",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
        assert!(parse_args(
            [
                "--model",
                "/m",
                "--fixture",
                "/f",
                "--output",
                "/o",
                "--mode",
                "perf",
                "--route",
                "bogus",
                "--manifest",
                "/m"
            ]
            .map(String::from)
        )
        .is_err());
    }

    #[test]
    fn slowlink_receipt_attractor_heuristic_flags_loops_and_collapse() {
        assert!(attractor(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).is_none());
        assert!(attractor(&[5; 20]).is_some());
        let looped: Vec<u32> = (0..40)
            .map(|i| 10 + (i % 2))
            .chain((0..16).map(|i| 100 + i))
            .collect();
        assert!(
            attractor(&looped).is_none(),
            "a loop that ended is not the tail attractor"
        );
        let tail: Vec<u32> = (0..16)
            .map(|i| 100 + i)
            .chain((0..40).map(|i| 10 + (i % 3)))
            .collect();
        assert!(attractor(&tail).is_some());
    }

    #[test]
    fn slowlink_receipt_boundary_fixture_pins_the_marker_at_every_cut_and_nowhere_else() {
        let parsed = Fixture::parse(&fixture()).unwrap();
        assert_eq!(parsed.boundary_token_id, 7);
        assert!(boundary_fixture_problems(&boundary_ids(), &CUTS, 7).is_empty());
        // The marker is required and typed.
        for bad in [json!(null), json!("7"), json!(-1), json!(1u64 << 40)] {
            let mut f = fixture();
            f["boundary_token_id"] = bad;
            assert!(Fixture::parse(&f).is_err());
        }
        let mut f = fixture();
        f.as_object_mut().unwrap().remove("boundary_token_id");
        assert!(Fixture::parse(&f).is_err(), "marker id must be explicit");
        // Inside the row-prompt id range it could collide with real ids.
        let mut f = fixture();
        f["boundary_token_id"] = json!(LCG_ID_MIN);
        assert!(Fixture::parse(&f).is_err());
        // A missing marker at cut - 1, a stray marker and a cut outside the
        // prompt are all refused; the caller's ids are never rewritten.
        for cut in CUTS {
            let mut ids = boundary_ids();
            ids[cut - 1] = 1500;
            let bad = boundary_fixture_problems(&ids, &CUTS, 7);
            assert!(bad.iter().any(|p| p.contains(&format!("boundary {cut}"))));
            let mut f = fixture();
            f["prompts"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .for_each(|p| {
                    if p["kind"] == "boundary" {
                        p["prompt_ids"] = json!(ids);
                    }
                });
            assert!(Fixture::parse(&f).is_err());
        }
        let mut ids = boundary_ids();
        ids[100] = 7;
        assert!(boundary_fixture_problems(&ids, &CUTS, 7)
            .iter()
            .any(|p| p.contains("unrequested boundary 101")));
        assert!(!boundary_fixture_problems(&ids, &[0, 9000], 7).is_empty());
    }

    #[test]
    fn slowlink_receipt_boundary_restore_must_reuse_exactly_the_cut_from_a_snapshot() {
        let receipt = |entry: u64, rows: u64| json!({"entry_pos": entry, "prompt_rows": rows});
        // 1536-row prefix + 300 continuation rows restored at 1536.
        let r = receipt(1536, 300);
        assert!(boundary_restore_problems(1536, 1536, false, Some(&r), 1836).is_empty());
        // A shallower snapshot, a cold recompute and live state never pass.
        assert!(!boundary_restore_problems(1536, 4, false, Some(&r), 1836).is_empty());
        assert!(
            !boundary_restore_problems(1536, 0, false, Some(&receipt(0, 1836)), 1836).is_empty()
        );
        assert!(!boundary_restore_problems(1536, 1536, true, Some(&r), 1836).is_empty());
        // The plan said `cut` but the request recomputed the whole prompt.
        assert!(
            !boundary_restore_problems(1536, 1536, false, Some(&receipt(0, 1836)), 1836).is_empty()
        );
        assert!(
            !boundary_restore_problems(1536, 1536, false, Some(&receipt(1536, 1836)), 1836)
                .is_empty()
        );
        assert!(!boundary_restore_problems(1536, 1536, false, None, 1836).is_empty());
        assert!(boundary_plan_problems(3, 3, false).is_empty());
        assert_eq!(boundary_plan_problems(3, 0, false).len(), 1);
    }

    #[test]
    fn slowlink_receipt_every_requested_boundary_needs_an_executed_exact_restore() {
        let all: Vec<(usize, usize)> = CUTS.iter().map(|&c| (c, c)).collect();
        assert!(boundary_coverage_problems(&CUTS, &all).is_empty());
        for skip in 0..CUTS.len() {
            let mut missing = all.clone();
            missing.remove(skip);
            let bad = boundary_coverage_problems(&CUTS, &missing);
            assert_eq!(bad.len(), 1, "{bad:?}");
            assert!(bad[0].contains("never executed"));
            let mut shallow = all.clone();
            shallow[skip].1 = 0;
            assert!(boundary_coverage_problems(&CUTS, &shallow)[0].contains("reused 0"));
        }
        // An empty case list is not coverage.
        assert_eq!(boundary_coverage_problems(&CUTS, &[]).len(), CUTS.len());
    }

    #[test]
    fn slowlink_receipt_boundary_capture_needs_every_cut_as_an_executed_exit() {
        let mut r = receipt();
        r["windows"] = json!([{"start": 0, "rows": 1537}, {"start": 1537, "rows": 7463}]);
        r["launch_cuts"] = json!([3, 4, 1536, 1537, 8192]);
        assert!(boundary_capture_problems(&CUTS, Some(&r), true).is_empty());
        r["launch_cuts"] = json!([3, 4, 1537, 8192]);
        let bad = boundary_capture_problems(&CUTS, Some(&r), true);
        assert_eq!(bad.len(), 1);
        assert!(bad[0].contains("1536"));
        // The incumbent route's exits are not receipted; only the receipt is required.
        assert!(boundary_capture_problems(&CUTS, Some(&incumbent()), false).is_empty());
        assert!(!boundary_capture_problems(&CUTS, None, false).is_empty());
    }

    /// A full usage literal: device owners 11/22/33/44 plus `pool`, host owners
    /// 101/202/303/404 plus `host_pool`, and distinct counters.
    fn usage_literal(
        pool: u64,
        evictions: u64,
        host_pool: u64,
        host: [u64; 4],
    ) -> SessionCacheUsage {
        SessionCacheUsage {
            pool_bytes: pool,
            budget_bytes: 2 << 30,
            pending_bytes: 11,
            resident_bytes: 22,
            deferred_release_bytes: 33,
            parked_bytes: 44,
            evictions_total: evictions,
            host_budget_bytes: 2 << 30,
            host_pool_bytes: host_pool,
            host_pending_bytes: 101,
            host_resident_bytes: 202,
            host_deferred_release_bytes: 303,
            host_parked_bytes: 404,
            host_captures_total: host[0],
            host_restores_total: host[1],
            host_evictions_total: host[2],
            host_skips_total: host[3],
        }
    }

    #[test]
    fn slowlink_receipt_cache_usage_evidence_reports_the_actual_eviction_delta() {
        let usage =
            |pool: u64, evictions: u64| usage_value(&usage_literal(pool, evictions, 0, [0; 4]));
        let before = usage(4096, 3);
        // Occupancy fell by two snapshots but exactly one eviction ran: the
        // delta is the counter's, not derived from the occupancy change.
        let after = usage(0, 4);
        let evidence = cache_evidence(&before, &after);
        assert_eq!(evidence["evictions_delta"], json!(1));
        assert_eq!(evidence["before"]["pool_bytes"], json!(4096));
        assert_eq!(
            evidence["after"]["device_total_bytes"],
            json!(11 + 22 + 33 + 44)
        );
        assert_eq!(
            cache_evidence(&before, &before)["evictions_delta"],
            json!(0)
        );
        // No cache, or a counter that went backwards, has no delta.
        assert!(cache_evidence(&Value::Null, &after)["evictions_delta"].is_null());
        assert!(cache_evidence(&after, &before)["evictions_delta"].is_null());
        assert!(usage_json(None).is_null());
    }

    #[test]
    fn slowlink_receipt_host_only_captures_count_as_successful_storage() {
        // Every capture went to the pinned-host tier: the device pool is empty.
        let host_only = usage_value(&usage_literal(0, 0, 4096, [5, 0, 0, 0]));
        assert_eq!(host_only["pool_bytes"], json!(0));
        assert_eq!(host_only["host_pool_bytes"], json!(4096));
        assert!(stored_snapshot_problems(&host_only).is_empty());
        // A device-only capture (old tier, no host bytes) still counts.
        let device_only = usage_value(&usage_literal(4096, 0, 0, [0; 4]));
        assert!(stored_snapshot_problems(&device_only).is_empty());
        // A receipt written before the host fields existed keeps passing on its pool.
        let old = json!({"pool_bytes": 4096, "evictions_total": 0});
        assert!(stored_snapshot_problems(&old).is_empty());
        // Host storage never stands in for a requested cut: a plan that reuses
        // fewer rows than the cut, or a request that recomputed from row 0,
        // still fails whatever the host tier holds.
        assert!(!boundary_coverage_problems(&CUTS, &[(CUTS[0], 0)]).is_empty());
        assert!(!boundary_restore_problems(
            1536,
            0,
            false,
            Some(&json!({"entry_pos": 0, "prompt_rows": 1836})),
            1836,
        )
        .is_empty());
        assert!(boundary_restore_problems(
            1536,
            1536,
            false,
            Some(&json!({"entry_pos": 1536, "prompt_rows": 300})),
            1836,
        )
        .is_empty());
    }

    #[test]
    fn slowlink_receipt_empty_or_missing_requested_snapshot_storage_fails() {
        // Both tiers empty after the commit: nothing was captured.
        let empty = usage_value(&usage_literal(0, 0, 0, [0; 4]));
        let problems = stored_snapshot_problems(&empty);
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].contains("no snapshot was published"),
            "{problems:?}"
        );
        // Counters alone are not storage: a capture that was evicted again, or
        // skipped under pressure, leaves nothing to restore.
        let evicted = usage_value(&usage_literal(0, 2, 0, [3, 0, 3, 7]));
        assert!(!stored_snapshot_problems(&evicted).is_empty());
        // Pending, resident and deferred host bytes are not published pool bytes.
        assert_eq!(evicted["host_total_bytes"], json!(101 + 202 + 303 + 404));
        // No usage at all (no cache attached) and a malformed usage fail too.
        assert!(!stored_snapshot_problems(&Value::Null).is_empty());
        assert!(!stored_snapshot_problems(&json!("usage")).is_empty());
        assert!(!stored_snapshot_problems(&json!({})).is_empty());
        // A missing requested boundary snapshot stays a coverage failure.
        assert_eq!(
            boundary_coverage_problems(&CUTS, &[]).len(),
            CUTS.len(),
            "every requested cut needs an executed exact restore"
        );
    }

    #[test]
    fn slowlink_receipt_device_totals_never_add_host_bytes() {
        let usage = usage_value(&usage_literal(4096, 1, 8192, [1, 2, 3, 4]));
        // device_total_bytes: the five DEVICE owners only.
        assert_eq!(usage["device_total_bytes"], json!(4096 + 11 + 22 + 33 + 44));
        // host_total_bytes: the five HOST owners, each once, no device byte.
        assert_eq!(
            usage["host_total_bytes"],
            json!(8192 + 101 + 202 + 303 + 404)
        );
        // The budgets are policy, not occupancy: neither total includes them.
        assert_eq!(usage["budget_bytes"], json!(2u64 << 30));
        assert_eq!(usage["host_budget_bytes"], json!(2u64 << 30));
        // Host occupancy moving never moves the device total, and vice versa.
        let more_host = usage_value(&usage_literal(4096, 1, 1 << 30, [1, 2, 3, 4]));
        assert_eq!(more_host["device_total_bytes"], usage["device_total_bytes"]);
        let more_device = usage_value(&usage_literal(1 << 30, 1, 8192, [1, 2, 3, 4]));
        assert_eq!(more_device["host_total_bytes"], usage["host_total_bytes"]);
    }

    #[test]
    fn slowlink_receipt_additive_host_ownership_counters_all_serialize() {
        let before = usage_value(&usage_literal(0, 1, 4096, [2, 1, 0, 3]));
        let after = usage_value(&usage_literal(0, 4, 8192, [7, 3, 2, 5]));
        for (key, value) in [
            ("host_budget_bytes", 2u64 << 30),
            ("host_pool_bytes", 8192),
            ("host_pending_bytes", 101),
            ("host_resident_bytes", 202),
            ("host_deferred_release_bytes", 303),
            ("host_parked_bytes", 404),
            ("host_captures_total", 7),
            ("host_restores_total", 3),
            ("host_evictions_total", 2),
            ("host_skips_total", 5),
            ("host_total_bytes", 8192 + 101 + 202 + 303 + 404),
        ] {
            assert_eq!(after[key], json!(value), "{key}");
        }
        // The existing device keys are untouched.
        for key in [
            "pool_bytes",
            "budget_bytes",
            "pending_bytes",
            "resident_bytes",
            "deferred_release_bytes",
            "parked_bytes",
            "evictions_total",
            "device_total_bytes",
        ] {
            assert!(after[key].is_u64(), "{key}");
        }
        // The pressure policy is recorded: armed by a nonzero host budget.
        assert_eq!(
            after["pressure_host_policy"],
            json!({"armed": true, "host_budget_bytes": 2u64 << 30})
        );
        let mut off = usage_literal(0, 0, 0, [0; 4]);
        off.host_budget_bytes = 0;
        assert_eq!(
            usage_value(&off)["pressure_host_policy"],
            json!({"armed": false, "host_budget_bytes": 0})
        );
        // Per-request evidence carries the actual counter deltas.
        let evidence = cache_evidence(&before, &after);
        assert_eq!(evidence["evictions_delta"], json!(3));
        assert_eq!(evidence["host_captures_delta"], json!(5));
        assert_eq!(evidence["host_restores_delta"], json!(2));
        assert_eq!(evidence["host_evictions_delta"], json!(2));
        assert_eq!(evidence["host_skips_delta"], json!(2));
        // A counter that went backwards, or a side without it, has no delta.
        assert!(cache_evidence(&after, &before)["host_captures_delta"].is_null());
        assert!(cache_evidence(&json!({}), &after)["host_restores_delta"].is_null());
        // The per-case evidence is built from the before-plan and after samples.
        let mut detail = json!({"cache_usage_before_plan": before, "cache_usage_after": after});
        attach_case_cache_evidence(&mut detail);
        assert_eq!(detail["cache_evidence"]["host_captures_delta"], json!(5));
        assert_eq!(
            detail["cache_evidence"]["after"]["host_pool_bytes"],
            json!(8192)
        );
    }

    #[test]
    fn slowlink_receipt_granule_census_is_additive_and_physical_is_logical_plus_padding() {
        let census = SlowlinkCensus {
            ring: 1_000,
            lookahead: 2_000,
            route_storage: 3_000,
            a4_sidecars: 4_000,
            group_scatter: 5_000,
            arena: 6_000,
            boundary_state: 7_000,
            runtime_overhead: 100,
            diagnostic: 10,
        };
        let mut granule = GranuleCharge {
            granule_bytes: 2 << 20,
            expert_pages: 11,
            route_region: 22,
            hc_arena: 33,
            window_scratch: 915_456,
            boundary_stages: 55,
            layer: 44,
            ..GranuleCharge::default()
        };
        granule.base.weights = 500;
        granule.base.state = 600;
        let logical_owned = 1_000 + 2_000 + 3_000 + 4_000 + 5_000 + 6_000 + 7_000;
        let logical_reserve = logical_owned + 100 + 10;
        let s1_padding = granule.total().unwrap() - granule.base.total().unwrap();
        assert!(s1_padding >= 11 + 22 + 33 + 915_456 + 55);
        let reserve = logical_reserve + granule.total().unwrap();
        let evidence = granule_evidence(&census, &granule, 10, 12345, Ok(reserve));
        // Logical stays payload: the seven owned terms and two budget terms are
        // exactly the census's, with no padding folded in.
        assert_eq!(
            evidence["logical_payload"]["owned_bytes"],
            json!(logical_owned)
        );
        assert_eq!(evidence["logical_payload"]["budget_bytes"], json!(110));
        assert_eq!(
            evidence["logical_payload"]["reserve_bytes"],
            json!(logical_reserve)
        );
        assert_eq!(
            evidence["logical_payload"]["owned_terms"]
                .as_array()
                .unwrap()
                .len(),
            7
        );
        assert_eq!(
            evidence["logical_payload"]["budget_terms"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        // Physical is logical + padding, each once, and matches the profile's reserve.
        assert_eq!(evidence["allocation_granule_bytes"], json!(2u64 << 20));
        assert_eq!(
            evidence["physical"]["owned_s1_bytes"],
            json!(logical_owned + s1_padding)
        );
        assert_eq!(evidence["physical"]["reserved_s1_bytes"], json!(reserve));
        assert_eq!(
            evidence["physical"]["profile_reserve_bytes"],
            json!(reserve)
        );
        assert_eq!(evidence["physical"]["reserve_consistent"], json!(true));
        // The scope is the S1 increment, never total model-owned bytes.
        assert!(evidence["physical"]["scope"]
            .as_str()
            .unwrap()
            .contains("S1 increment"));
        assert!(evidence["physical"].get("owned_bytes").is_none());
        // Every S1 owner term (including window scratch) is listed by name and
        // the list closes onto the derived total, so none can silently vanish.
        let names: Vec<&str> = evidence["backing_padding"]["s1_owner_terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"window scratch") && names.contains(&"boundary stages"));
        assert!(!names.contains(&"weight tensors"), "{names:?}");
        assert_eq!(evidence["backing_padding"]["terms_close"], json!(true));
        // The padding terms sum to the fixed total once; the base owners are
        // reported apart from the S1 owners and the per-layer slack is not a fixed term.
        let padding = &evidence["backing_padding"];
        assert_eq!(
            padding["fixed_total_bytes"],
            json!(granule.total().unwrap())
        );
        assert_eq!(padding["base_reserve_bytes"], json!(1_100));
        assert_eq!(padding["s1_owner_bytes"], json!(s1_padding));
        assert_eq!(
            padding["base_reserve_bytes"].as_u64().unwrap()
                + padding["s1_owner_bytes"].as_u64().unwrap(),
            padding["fixed_total_bytes"].as_u64().unwrap()
        );
        assert_eq!(padding["per_resident_layer_bytes"], json!(44));
        assert_eq!(padding["resident_layers_bytes"], json!(440));
        assert_eq!(evidence["device_snapshot_budget_bytes"], json!(12345));
        assert_eq!(evidence["arena_claim"], json!(false));
        for key in [
            "units",
            "allocation_granule_bytes",
            "logical_payload",
            "backing_padding",
            "physical",
        ] {
            assert!(!evidence[key].is_null(), "{key}");
        }
        // A profile reserve that is not logical + padding is flagged, not hidden.
        let off = granule_evidence(&census, &granule, 10, 0, Ok(reserve + 1));
        assert_eq!(off["physical"]["reserve_consistent"], json!(false));
        let err = granule_evidence(&census, &granule, 10, 0, Err("overflow".into()));
        assert_eq!(err["physical"]["reserve_consistent"], json!(false));
        assert_eq!(err["physical"]["profile_reserve_error"], json!("overflow"));
        // No granule model (payload-only accounting): padding is zero and physical == logical.
        let none = granule_evidence(
            &census,
            &GranuleCharge::default(),
            10,
            0,
            Ok(logical_reserve),
        );
        assert_eq!(none["backing_padding"]["fixed_total_bytes"], json!(0));
        assert_eq!(none["physical"]["owned_s1_bytes"], json!(logical_owned));
        assert_eq!(
            none["physical"]["reserved_s1_bytes"],
            json!(logical_reserve)
        );
        // The cache's device usage fields stay labelled payload, never physical backing.
        let usage = usage_value(&usage_literal(1, 0, 0, [0; 4]));
        assert!(usage["byte_units"]
            .as_str()
            .unwrap()
            .starts_with("payload bytes"));
    }

    /// Fixture plans: nine weight roots and two shadow roots at granule 4096.
    /// The 9+2 shape is only a fixture the tests build; no check selects on it.
    fn arena_plan(weight_members: usize, shadow_members: usize) -> ModelWeightArenaPlan {
        let weights = NamedByteArenaPlan::new(
            (0..weight_members)
                .map(|i| (format!("w{i:02}"), 768usize))
                .collect(),
            256,
            1024,
        )
        .unwrap();
        let f16_shadows = NamedByteArenaPlan::new(
            (0..shadow_members)
                .map(|i| (format!("w{i:02}"), 1024usize))
                .collect(),
            256,
            1024,
        )
        .unwrap();
        ModelWeightArenaPlan {
            weights,
            f16_shadows,
            granule_bytes: 4096,
        }
    }

    fn exact(plan: &NamedByteArenaPlan) -> Vec<Option<usize>> {
        plan.roots.iter().copied().map(Some).collect()
    }

    #[test]
    fn slowlink_receipt_arena_planned_lazy_roots_are_never_reported_as_actual() {
        let plan = arena_plan(9, 2);
        let (evidence, problems) = arena_evidence(Some(&plan), &exact(&plan.weights), &[]);
        assert!(problems.is_empty(), "{problems:?}");
        // Counts are the plan's, and the fixture shape is reported, not required.
        assert_eq!(
            evidence["weights"]["roots"],
            json!(plan.weights.roots.len())
        );
        assert_eq!(evidence["weights"]["roots"], json!(9));
        assert_eq!(evidence["f16_shadows"]["roots"], json!(2));
        assert_eq!(evidence["weights"]["members"], json!(9));
        assert_eq!(evidence["f16_shadows"]["members"], json!(2));
        assert_eq!(evidence["allocation_granule_bytes"], json!(4096));
        // The weight roots are live and exact; the shadow roots are planned only.
        let weights = &evidence["weights"];
        assert_eq!(weights["capacity_proof"], json!("exact"));
        assert_eq!(weights["actual"]["state"], json!("fully_allocated"));
        assert_eq!(weights["actual"]["live_roots"], json!(9));
        assert_eq!(
            weights["actual"]["backing_bytes"],
            weights["planned"]["backing_bytes"]
        );
        assert_eq!(
            weights["planned"]["backing_bytes"],
            json!(plan.weights.backing_bytes(4096).unwrap())
        );
        let shadows = &evidence["f16_shadows"];
        assert_eq!(shadows["actual"]["state"], json!("planned_only"));
        assert_eq!(shadows["actual"]["live_roots"], json!(0));
        assert_eq!(shadows["actual"]["unallocated_roots"], json!(2));
        assert_eq!(shadows["actual"]["capacity_bytes"], json!(0));
        assert_eq!(shadows["actual"]["backing_bytes"], json!(0));
        assert_eq!(shadows["capacity_proof"], json!("none"));
        assert_ne!(shadows["planned"]["backing_bytes"], json!(0));
        assert_eq!(evidence["planned_is_actual"], json!(false));
        assert_eq!(evidence["arena_claim"], json!(false));
        // Capacity is observed only for live owners; the backing is a model and
        // nothing here is a physical or recovery measurement.
        assert_eq!(weights["actual"]["capacity_observed"], json!(true));
        assert_eq!(shadows["actual"]["capacity_observed"], json!(false));
        assert_eq!(shadows["actual"]["capacity_observed_roots"], json!(0));
        for family in ["weights", "f16_shadows"] {
            assert_eq!(
                evidence[family]["actual"]["physical_backing_measured"],
                json!(false)
            );
            assert!(evidence[family]["actual"]["backing_model"]
                .as_str()
                .unwrap()
                .contains("not a per-allocation physical"));
        }
        assert_eq!(evidence["physical_backing_measured"], json!(false));
        assert_eq!(evidence["recovery_measured"], json!(false));
        assert!(evidence["backing_model"]
            .as_str()
            .unwrap()
            .contains("minimum-free"));
        let (none, _) = arena_evidence(None, &[], &[]);
        assert_eq!(none["capacity_observed"], json!(false));
        assert_eq!(none["physical_backing_measured"], json!(false));
        // Additive payload / request / padding / backing chain for each family.
        for family in ["weights", "f16_shadows"] {
            let planned = &evidence[family]["planned"];
            let payload = planned["payload_bytes"].as_u64().unwrap();
            let request = planned["root_request_bytes"].as_u64().unwrap();
            let backing = planned["backing_bytes"].as_u64().unwrap();
            assert!(payload <= request && request <= backing, "{family}");
            assert_eq!(
                planned["member_alignment_padding_bytes"],
                json!(request - payload)
            );
            assert_eq!(planned["granule_padding_bytes"], json!(backing - request));
            assert_eq!(backing % 4096, 0);
        }
        // A lazy shadow root allocated one at a time is partial, still exact per root.
        let one = [Some(plan.f16_shadows.roots[0]), None];
        let (partial, problems) = arena_evidence(Some(&plan), &exact(&plan.weights), &one);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(
            partial["f16_shadows"]["actual"]["state"],
            json!("partial_lazy")
        );
        assert_eq!(partial["f16_shadows"]["capacity_proof"], json!("partial"));
        assert_eq!(partial["f16_shadows"]["actual"]["live_roots"], json!(1));
        // Every shadow root live and exact closes the proof.
        let (full, problems) = arena_evidence(
            Some(&plan),
            &exact(&plan.weights),
            &exact(&plan.f16_shadows),
        );
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(full["f16_shadows"]["capacity_proof"], json!("exact"));
        assert_eq!(
            full["f16_shadows"]["actual"]["state"],
            json!("fully_allocated")
        );
    }

    #[test]
    fn slowlink_receipt_arena_rejects_an_actual_capacity_that_is_not_the_planned_request() {
        let plan = arena_plan(9, 2);
        // An oversized (reused pool) weight buffer is not an exact root.
        let mut weights = exact(&plan.weights);
        weights[3] = Some(plan.weights.roots[3] + 4096);
        let (evidence, problems) = arena_evidence(Some(&plan), &weights, &[]);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("weight arena root 3") && p.contains("oversized")),
            "{problems:?}"
        );
        assert_eq!(evidence["weights"]["capacity_proof"], json!("mismatch"));
        let mismatches = evidence["weights"]["actual"]["capacity_mismatches"]
            .as_array()
            .unwrap();
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0]["root"], json!(3));
        // An undersized shadow root is rejected the same way, even when partial.
        let shadows = [None, Some(plan.f16_shadows.roots[1] - 1)];
        let (evidence, problems) = arena_evidence(Some(&plan), &exact(&plan.weights), &shadows);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("f16 shadow arena root 1")),
            "{problems:?}"
        );
        assert_eq!(evidence["f16_shadows"]["capacity_proof"], json!("mismatch"));
    }

    #[test]
    fn slowlink_receipt_arena_rejects_census_count_mismatch_and_planned_roots_claimed_live() {
        let plan = arena_plan(9, 2);
        // Fewer observed capacities than planned roots.
        let short: Vec<Option<usize>> = exact(&plan.weights).into_iter().take(8).collect();
        let (evidence, problems) = arena_evidence(Some(&plan), &short, &[]);
        assert!(
            problems.iter().any(|p| p.contains("census mismatch")),
            "{problems:?}"
        );
        assert_eq!(
            evidence["weights"]["capacity_proof"],
            json!("census_mismatch")
        );
        assert_eq!(evidence["weights"]["actual"]["capacity_bytes"], json!(0));
        // More shadow capacities than planned shadow roots.
        let long = [Some(1024usize); 3];
        let (_, problems) = arena_evidence(Some(&plan), &exact(&plan.weights), &long);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("f16 shadow arena") && p.contains("census mismatch")),
            "{problems:?}"
        );
        // A planned eager weight root that is not live is planned, not actual.
        for none_live in [Vec::new(), vec![None; 9]] {
            let (evidence, problems) = arena_evidence(Some(&plan), &none_live, &[]);
            assert!(
                problems
                    .iter()
                    .any(|p| p.contains("not live after the load")),
                "{problems:?}"
            );
            assert_eq!(
                evidence["weights"]["actual"]["state"],
                json!("planned_only")
            );
            assert_eq!(evidence["weights"]["actual"]["capacity_bytes"], json!(0));
            assert_ne!(evidence["weights"]["capacity_proof"], json!("exact"));
        }
        let mut partial = exact(&plan.weights);
        partial[8] = None;
        let (evidence, problems) = arena_evidence(Some(&plan), &partial, &[]);
        assert!(
            problems.iter().any(|p| p.contains("1 of 9 planned roots")),
            "{problems:?}"
        );
        assert_eq!(evidence["weights"]["capacity_proof"], json!("partial"));
        // Observed capacities with no plan cannot be attributed to any arena.
        let (evidence, problems) = arena_evidence(None, &exact(&plan.weights), &[]);
        assert_eq!(evidence["present"], json!(false));
        assert!(
            problems.iter().any(|p| p.contains("no arena plan")),
            "{problems:?}"
        );
        let (_, problems) = arena_evidence(None, &[], &[]);
        assert!(problems.is_empty());
        // The plan's presence follows the S1 profile, never the card or the model.
        assert!(arena_presence_problems(true, true).is_empty());
        assert!(arena_presence_problems(false, false).is_empty());
        assert!(arena_presence_problems(false, true)[0].contains("without a model weight arena"));
        assert!(arena_presence_problems(true, false)[0].contains("without the S1 profile"));
    }

    #[test]
    fn slowlink_receipt_arena_counts_are_reported_never_selected_on() {
        // The 9+2 fixture shape is not a gate: other model shapes pass untouched.
        for (weights, shadows) in [(3usize, 1usize), (9, 2), (12, 5), (1, 0)] {
            let plan = arena_plan(weights, shadows);
            let (evidence, problems) = arena_evidence(
                Some(&plan),
                &exact(&plan.weights),
                &exact(&plan.f16_shadows),
            );
            assert!(problems.is_empty(), "{weights}+{shadows}: {problems:?}");
            assert_eq!(
                evidence["weights"]["roots"],
                json!(plan.weights.roots.len())
            );
            assert_eq!(
                evidence["f16_shadows"]["roots"],
                json!(plan.f16_shadows.roots.len())
            );
            assert_eq!(evidence["weights"]["members"], json!(weights));
            assert_eq!(evidence["f16_shadows"]["members"], json!(shadows));
        }
        // An empty family has no roots, no proof to make and no problem.
        let plan = arena_plan(2, 0);
        let (evidence, problems) = arena_evidence(Some(&plan), &exact(&plan.weights), &[]);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(evidence["f16_shadows"]["actual"]["state"], json!("empty"));
        assert_eq!(evidence["f16_shadows"]["capacity_proof"], json!("none"));
    }

    #[test]
    fn slowlink_receipt_arena_keeps_residual_nongating_and_the_ledger_margin_strict() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for mode in [Mode::Primitives, Mode::State, Mode::Product, Mode::Perf] {
            for route in [Expect::LayerMajor, Expect::Incumbent] {
                let ids = required_check_ids(mode, &fixture, route);
                assert!(
                    !ids.iter()
                        .any(|n| n.contains("arena") || n.contains("residual")),
                    "{mode:?}: {ids:?}"
                );
            }
        }
        // Still seven primitives plus identity, placement and durability.
        assert_eq!(PRIMITIVE_CASES.len(), 7);
        let ids = required_check_ids(Mode::Primitives, &fixture, Expect::LayerMajor);
        assert_eq!(ids.len(), 7 + 3);
        // The census residual stays a non-gating diagnostic beside the arena evidence.
        let diag = memory_diagnostic(&good_memory(), Ok(ledger()));
        assert_eq!(diag["gating"], json!(false));
        assert_eq!(diag["required"], json!(false));
        // The 512 MiB safety margin of the ledger is unchanged by arenas.
        assert_eq!(decomposition().safety_margin, 512 << 20);
        assert_eq!(decomposition().allowed_non_weight(), Some(12 * GIB));
    }

    #[test]
    fn slowlink_receipt_state_required_lists_every_boundary_case() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for route in [Expect::LayerMajor, Expect::Incumbent] {
            let names = state_required(&fixture, route);
            assert!(names.contains(&"state.boundary_capture".to_string()));
            for cut in CUTS {
                assert!(names.contains(&format!("state.boundary.{cut}.restore")));
            }
        }
    }

    #[test]
    fn slowlink_receipt_boundary16k_required_checks_and_prompt() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for route in [Expect::LayerMajor, Expect::Incumbent] {
            let names = state_required(&fixture, route);
            for name in boundary16k_required() {
                assert!(names.contains(&name), "{name}");
            }
        }
        assert_eq!(
            boundary16k_required(),
            [
                "state.boundary16k.capture",
                "state.boundary16k.8192.restore",
                "state.boundary16k.16384.restore",
            ]
        );
        let rows: Vec<u32> = (0..16384).map(|i| 1000 + i).collect();
        let ids = boundary16k_prompt(&rows, &[5, 6, 7, 8], 16640, 16).unwrap();
        // 16384 is an interior boundary of the 16387-row prompt, rows unchanged.
        assert_eq!(ids.len(), 16387);
        assert_eq!(&ids[..16384], &rows[..]);
        assert_eq!(&ids[16384..], &[5, 6, 7]);
        assert!(boundary16k_prompt(&rows[..16000], &[5, 6, 7], 16640, 16).is_err());
        assert!(boundary16k_prompt(&rows, &[5, 6], 16640, 16).is_err());
        assert!(boundary16k_prompt(&rows, &[5, 6, 7], 16400, 16).is_err());
    }

    #[test]
    fn slowlink_receipt_largest_verified_restore_cut_ignores_unverified_cases() {
        assert_eq!(largest_verified_cut(&[]), None);
        assert_eq!(
            largest_verified_cut(&[
                (3, true),
                (4, true),
                (1536, true),
                (1537, true),
                (8192, true)
            ]),
            Some(8192)
        );
        // A missed 8192 leaves the deepest verified requested cut visible.
        assert_eq!(
            largest_verified_cut(&[(3, true), (1536, true), (1537, true), (8192, false)]),
            Some(1537)
        );
        assert_eq!(largest_verified_cut(&[(3, false), (8192, false)]), None);
    }

    const GIB: u64 = 1 << 30;
    const TIB: u64 = 1 << 40;

    /// Process counters as one sample reads them: the unreclaimable RSS is
    /// `VmRSS` minus the clean regular-file RSS, derived as production does.
    /// `RssFile` is a TiB of file pages unless a test overrides it.
    fn counters(rss: u64, clean: u64, pin: u64, vram: u64, gtt: u64) -> ProcCounters {
        ProcCounters {
            vm_rss: Some(rss),
            rss_anon: Some(0),
            vm_pin: Some(pin),
            vm_hwm: Some(rss),
            rss_file: Some(TIB),
            rss_shmem: Some(0),
            reclaimable_file_clean: Some(clean),
            host_rss_unreclaimable: Some(rss.saturating_sub(clean)),
            drm_vram: Some(vram),
            drm_gtt: Some(gtt),
        }
    }

    fn mem_row(label: &str, free: u64, c: ProcCounters) -> Value {
        counters_json(label, "test", &c, Some((free, 32 * GIB)))
    }

    fn sampler_row(c: ProcCounters) -> Value {
        let mut row = counters_json(SAMPLER_LABEL, "test", &c, None);
        row["samples"] = json!(5);
        row
    }

    fn decomposition() -> DeviceDecomposition {
        DeviceDecomposition {
            weights: 10 * GIB,
            s1_tile_census: 2 * GIB,
            s1_extras: 7 * GIB,
            sequential_diag: GIB,
            safety_margin: GIB / 2,
        }
    }

    fn ledger_with_host(owned_host: u64) -> MemoryLedger {
        MemoryLedger {
            owned_device_bytes: 20 * GIB,
            owned_host_bytes: owned_host,
            census_basis: "test census".into(),
            decomposition: decomposition(),
        }
    }

    fn ledger() -> MemoryLedger {
        ledger_with_host(80 * GIB)
    }

    /// The phase samples plus the sampler row: `c` at `post_load`,
    /// `request_after` and `end`, `s` as the sampler's maxima.
    fn memory_with(c: ProcCounters, s: ProcCounters) -> Vec<Value> {
        vec![
            // A baseline before any model mapping: no file-backed pages yet,
            // so it cannot raise the high-water RssFile of the cases below.
            mem_row(
                "pre_load",
                31 * GIB,
                ProcCounters {
                    rss_file: Some(0),
                    ..counters(GIB, 0, 0, GIB / 2, 0)
                },
            ),
            mem_row("post_load", 11 * GIB, c),
            mem_row("request_after", 11 * GIB, c),
            mem_row("end", 11 * GIB, c),
            sampler_row(s),
        ]
    }

    /// Owned 20 GiB device / 80 GiB host, a small residual on each.
    fn good_memory() -> Vec<Value> {
        memory_with(
            counters(70 * GIB, 0, 80 * GIB, 20 * GIB, 79 * GIB),
            counters(70 * GIB, 0, 80 * GIB, 20 * GIB + GIB / 2, 80 * GIB),
        )
    }

    fn swap_sampler(samples: &mut [Value], c: ProcCounters) {
        *samples.last_mut().unwrap() = sampler_row(c);
    }

    #[test]
    fn slowlink_receipt_memory_allowances_are_the_fixed_ledger_constants() {
        assert_eq!(UNOWNED_VRAM_ALLOWANCE_BYTES, 1_610_612_736);
        assert_eq!(UNOWNED_VRAM_ALLOWANCE_BYTES, 3 * (1 << 29));
        assert_eq!(UNOWNED_HOST_ALLOWANCE_BYTES, 2_147_483_648);
        assert_eq!(UNOWNED_HOST_ALLOWANCE_BYTES, 2 * GIB);
    }

    #[test]
    fn slowlink_receipt_memory_within_allowances_passes_and_subtracts_only_clean_regular_files() {
        let (bad, receipt) = memory_problems(&good_memory(), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(receipt["residual"]["vram_bytes"], json!(GIB / 2));
        assert_eq!(receipt["residual"]["host_bytes"], json!(0));
        assert_eq!(receipt["high_water"]["rss_file"], json!(TIB));
        assert!(receipt["high_water"]["sampling"]
            .as_str()
            .unwrap()
            .contains("discrete points"));
        // Warm regular-file (HFQ) pages: a TiB of CLEAN file-backed RSS is the
        // explicit reclaimable class and is subtracted, so it passes.
        let warm = counters(TIB + 8 * GIB, TIB, 70 * GIB, 20 * GIB, 70 * GIB);
        let (bad, receipt) = memory_problems(&memory_with(warm, warm), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(
            receipt["high_water"]["host_rss_unreclaimable"],
            json!(8 * GIB)
        );
        // Without the clean class the same RSS is a huge unowned residual.
        let dirty = counters(TIB + 8 * GIB, 0, 70 * GIB, 20 * GIB, 70 * GIB);
        let (bad, _) = memory_problems(&memory_with(dirty, dirty), &ledger());
        assert!(
            bad.iter().any(|p| p.contains("unowned host residual")),
            "{bad:?}"
        );
        // Under-use of the census is not a residual.
        let small = counters(GIB, 0, GIB, GIB, GIB);
        let (bad, receipt) = memory_problems(&memory_with(small, small), &ledger());
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(receipt["residual"]["host_bytes"], json!(0));
    }

    #[test]
    fn slowlink_receipt_memory_pinned_kfd_plus_driver_anon_must_fail_the_host_cap() {
        // 52 GB of pinned KFD maps (RssFile and GTT, zero clean regular-file
        // pages) plus 4 GB of anonymous driver memory, owned census 52 GB.
        let owned = 52 * GIB;
        let kfd = |extra: u64| {
            let mut c = counters(52 * GIB + extra, 0, 52 * GIB, 20 * GIB, 52 * GIB);
            c.rss_file = Some(52 * GIB);
            c
        };
        let (bad, receipt) = memory_problems(
            &memory_with(kfd(4 * GIB), kfd(4 * GIB)),
            &ledger_with_host(owned),
        );
        assert!(
            bad.iter().any(|p| p.contains("unowned host residual")),
            "{bad:?}"
        );
        assert_eq!(receipt["residual"]["host_bytes"], json!(4 * GIB));
        // The whole RssFile is NOT subtracted: the pinned buffer objects stay counted.
        assert_eq!(receipt["high_water"]["rss_file"], json!(52 * GIB));
        // Exactly at the 2 GiB allowance passes, one byte over fails.
        let at = kfd(UNOWNED_HOST_ALLOWANCE_BYTES);
        assert!(
            memory_problems(&memory_with(at, at), &ledger_with_host(owned))
                .0
                .is_empty()
        );
        let over = kfd(UNOWNED_HOST_ALLOWANCE_BYTES + 1);
        assert!(
            !memory_problems(&memory_with(over, over), &ledger_with_host(owned))
                .0
                .is_empty()
        );
        // The pinned maps alone, no extra driver memory, are fully owned.
        let none = kfd(0);
        assert!(
            memory_problems(&memory_with(none, none), &ledger_with_host(owned))
                .0
                .is_empty()
        );
    }

    #[test]
    fn slowlink_receipt_memory_device_residual_over_the_exact_allowance_fails() {
        // Exactly at the allowance passes; one byte over fails (VRAM).
        for (extra, ok) in [
            (UNOWNED_VRAM_ALLOWANCE_BYTES, true),
            (UNOWNED_VRAM_ALLOWANCE_BYTES + 1, false),
        ] {
            let mut s = good_memory();
            swap_sampler(
                &mut s,
                counters(70 * GIB, 0, 80 * GIB, 20 * GIB + extra, 80 * GIB),
            );
            let (bad, _) = memory_problems(&s, &ledger());
            assert_eq!(bad.is_empty(), ok, "{bad:?}");
            assert!(ok || bad.iter().any(|p| p.contains("unowned VRAM residual")));
        }
        // hipMemGetInfo growth alone (small process vram) also fails.
        let host = counters(70 * GIB, 0, 80 * GIB, GIB, 79 * GIB);
        let mut s = memory_with(host, host);
        assert!(memory_problems(&s, &ledger()).0.is_empty());
        s[3]["device_free"] = json!(31 * GIB - 20 * GIB - UNOWNED_VRAM_ALLOWANCE_BYTES - 1);
        let (bad, _) = memory_problems(&s, &ledger());
        assert!(
            bad.iter().any(|p| p.contains("unowned VRAM residual")),
            "{bad:?}"
        );
    }

    #[test]
    fn slowlink_receipt_memory_missing_telemetry_fails_closed() {
        for key in [
            "drm_gtt",
            "drm_vram",
            "vm_pin",
            "vm_rss",
            "reclaimable_regular_file_clean_bytes",
            "host_rss_unreclaimable",
        ] {
            let mut s = good_memory();
            s[1][key] = Value::Null;
            let (bad, _) = memory_problems(&s, &ledger());
            assert!(
                bad.iter()
                    .any(|p| p.contains(&format!("{key} not observable"))),
                "{key}: {bad:?}"
            );
        }
        let mut s = good_memory();
        s[0]["device_free"] = Value::Null;
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        // Process GTT and the smaps class are required of the sampler too.
        for key in ["drm_gtt", "reclaimable_regular_file_clean_bytes"] {
            let mut s = good_memory();
            let mut row = sampler_row(counters(0, 0, 0, 0, 0));
            row[key] = Value::Null;
            *s.last_mut().unwrap() = row;
            assert!(!memory_problems(&s, &ledger()).0.is_empty(), "{key}");
        }
        // Missing phases or sampler are incomplete telemetry, not a pass.
        for label in MEMORY_PHASES {
            let mut s = good_memory();
            s.retain(|r| r["label"] != label);
            assert!(!memory_problems(&s, &ledger()).0.is_empty(), "{label}");
        }
        let mut s = good_memory();
        s.pop();
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        let mut s = good_memory();
        s.last_mut().unwrap()["samples"] = json!(0);
        assert!(!memory_problems(&s, &ledger()).0.is_empty());
        assert!(!memory_problems(&[], &ledger()).0.is_empty());
    }

    #[test]
    fn slowlink_receipt_memory_smaps_counts_only_clean_regular_file_mappings() {
        let map = |range: &str, path: &str, shared: u64, private: u64| {
            format!(
                "{range} r--p 00000000 08:01 99 {path}\nSize: 64 kB\nRss: 64 kB\nShared_Clean: {shared} kB\nPrivate_Clean: {private} kB\nPrivate_Dirty: 5 kB\nVmFlags: rd mr\n"
            )
        };
        let smaps = [
            map("55d0c0a00000-55d0c0a50000", "/models/q.hfq", 100, 60),
            map("7f0000000000-7f0100000000", "/dev/kfd", 0, 52_000),
            map("7f0200000000-7f0200100000", "/dev/dri/renderD128", 0, 11),
            map("7f0300000000-7f0300100000", "[heap]", 0, 7),
            map("7f0400000000-7f0400100000", "/memfd:buf (deleted)", 0, 13),
            map("7f0500000000-7f0500100000", "/dev/zero (deleted)", 0, 17),
            map("7f0600000000-7f0600100000", "/tmp/gone (deleted)", 0, 19),
            map("7f0700000000-7f0700100000", "/data/my model.bin", 4, 0),
            map("7f0800000000-7f0800100000", "", 0, 23),
        ]
        .concat();
        // 100 + 60 + 4 kB: the regular-file mappings' clean RSS only.
        assert_eq!(smaps_regular_file_clean_bytes(&smaps), Some(164 * 1024));
        assert_eq!(smaps_regular_file_clean_bytes(""), None);
        assert_eq!(
            smaps_regular_file_clean_bytes("Rss: 4 kB\nShared_Clean: 4 kB\n"),
            None
        );
        for (path, class) in [
            ("/dev/kfd", MapClass::KfdDri),
            ("/dev/dri/card0", MapClass::KfdDri),
            ("/dev/zero", MapClass::Zero),
            ("", MapClass::Anon),
            ("[stack]", MapClass::Anon),
            ("[anon:scudo]", MapClass::Anon),
            ("/memfd:x", MapClass::Anon),
            ("memfd:x", MapClass::Anon),
            ("/lib/x.so (deleted)", MapClass::Anon),
            ("/lib/x.so", MapClass::File),
        ] {
            assert_eq!(classify_mapping(path), class, "{path}");
        }
    }

    #[test]
    fn slowlink_receipt_memory_ledger_needs_the_owned_census_in_the_manifest() {
        let profile_ledger = |purpose: &str| {
            json!({
                "purpose": purpose, "device_owned_bytes": 20 * GIB,
                "device_weights_bytes": 10 * GIB, "s1_tile_census_bytes": 2 * GIB,
                "s1_extras_bytes": 7 * GIB, "sequential_diag_bytes": GIB,
                "safety_margin_bytes": GIB / 2,
            })
        };
        let manifest = json!({"memory_ledger": {
            "owned_host_bytes": 85899345920u64,
            "census_basis": "A census + 2 GiB cache pool",
            "profiles": {"primitive_selftest": profile_ledger("primitive_selftest")},
        }});
        let profile = PROFILE_PRIMITIVE;
        let ledger = memory_ledger(&manifest, profile).unwrap();
        assert_eq!(ledger.owned_device_bytes, 20 * GIB);
        assert_eq!(ledger.owned_host_bytes, 80 * GIB);
        assert_eq!(ledger.decomposition, decomposition());
        // The product profile is a different ledger: absent here, so it refuses.
        assert!(memory_ledger(&manifest, PROFILE_PRODUCT).is_err());
        for key in [
            "device_owned_bytes",
            "device_weights_bytes",
            "s1_tile_census_bytes",
            "s1_extras_bytes",
            "sequential_diag_bytes",
            "safety_margin_bytes",
        ] {
            let mut m = manifest.clone();
            m["memory_ledger"]["profiles"][profile]
                .as_object_mut()
                .unwrap()
                .remove(key);
            assert!(memory_ledger(&m, profile).is_err(), "{key}");
        }
        for key in ["owned_host_bytes", "census_basis"] {
            let mut m = manifest.clone();
            m["memory_ledger"].as_object_mut().unwrap().remove(key);
            assert!(memory_ledger(&m, profile).is_err(), "{key}");
        }
        // No flat-profile fallback: the old root-level figures are not read.
        let mut flat = manifest.clone();
        flat["memory_ledger"]["profiles"] = json!({});
        for (k, v) in [
            ("device_owned_bytes", 20 * GIB),
            ("device_weights_bytes", 10 * GIB),
            ("s1_tile_census_bytes", 2 * GIB),
            ("s1_extras_bytes", 7 * GIB),
            ("sequential_diag_bytes", GIB),
            ("safety_margin_bytes", GIB / 2),
        ] {
            flat["memory_ledger"][k] = json!(v);
        }
        assert!(memory_ledger(&flat, profile).is_err());
        let mut m = manifest.clone();
        m["memory_ledger"]["owned_host_bytes"] = json!(0);
        assert!(memory_ledger(&m, profile).is_err());
        let mut m = manifest.clone();
        m["memory_ledger"]["profiles"][profile]["safety_margin_bytes"] = json!(0);
        assert!(memory_ledger(&m, profile).is_err(), "zero safety margin");
        // A decomposition whose four-part sum is not the owned bytes refuses.
        let mut m = manifest.clone();
        m["memory_ledger"]["profiles"][profile]["s1_extras_bytes"] = json!(7 * GIB + 1);
        let e = memory_ledger(&m, profile).unwrap_err();
        assert!(e.contains("not device_owned_bytes"), "{e}");
        // The safety margin is not part of the sum.
        let mut m = manifest.clone();
        m["memory_ledger"]["profiles"][profile]["device_owned_bytes"] = json!(20 * GIB + GIB / 2);
        assert!(memory_ledger(&m, profile).is_err());
        // A sum that overflows u64 refuses rather than wrapping.
        let mut m = manifest.clone();
        m["memory_ledger"]["profiles"][profile]["device_weights_bytes"] = json!(u64::MAX);
        assert!(memory_ledger(&m, profile)
            .unwrap_err()
            .contains("overflows"));
        let mut m = manifest;
        m["memory_ledger"]["census_basis"] = json!("  ");
        assert!(memory_ledger(&m, profile).is_err());
    }

    #[test]
    fn slowlink_receipt_memory_non_weight_guard_is_exact_and_keeps_the_residual_check() {
        // allowed = tile 2 + extras 7 + diag 1 + safety 0.5 + driver 1.5 = 12 GiB.
        let d = decomposition();
        assert_eq!(d.committed_sum(), Some(20 * GIB));
        assert_eq!(d.allowed_non_weight(), Some(12 * GIB));
        let at = |observed: u64| {
            let host = counters(70 * GIB, 0, 80 * GIB, observed, 79 * GIB);
            let mut s = memory_with(host, host);
            // Keep hipMemGetInfo growth below the process counter.
            for row in s.iter_mut().filter(|r| r["label"] != SAMPLER_LABEL) {
                row["device_free"] = json!(31 * GIB);
            }
            memory_problems(&s, &ledger())
        };
        // Within the census: the guard passes and reports plain components.
        let (bad, receipt) = at(20 * GIB);
        assert!(bad.is_empty(), "{bad:?}");
        let g = &receipt["device_guard"];
        assert_eq!(g["result"], json!("pass"));
        assert_eq!(g["measured_non_weight_peak_bytes"], json!(10 * GIB));
        assert_eq!(g["allowed_non_weight_bytes"], json!(12 * GIB));
        assert_eq!(g["observed_max_device_bytes"], json!(20 * GIB));
        assert_eq!(g["components"]["s1_tile_census_bytes"], json!(2 * GIB));
        assert_eq!(
            g["components"]["driver_allowance_bytes"],
            json!(UNOWNED_VRAM_ALLOWANCE_BYTES)
        );
        assert!(g["sampling_method"]
            .as_str()
            .unwrap()
            .contains("not a continuous"));
        // Exactly at weights + allowed the guard still passes (the stricter
        // residual check is what fails there).
        let (bad, receipt) = at(22 * GIB);
        assert_eq!(receipt["device_guard"]["result"], json!("pass"));
        assert!(!bad.iter().any(|p| p.contains("non-weight")), "{bad:?}");
        assert!(
            bad.iter().any(|p| p.contains("unowned VRAM residual")),
            "{bad:?}"
        );
        // One byte over fails the guard, with the residual check also failing.
        let (bad, receipt) = at(22 * GIB + 1);
        assert_eq!(receipt["device_guard"]["result"], json!("fail"));
        assert_eq!(
            receipt["device_guard"]["measured_non_weight_peak_bytes"],
            json!(12 * GIB + 1)
        );
        assert!(
            bad.iter().any(|p| p.contains("non-weight device peak")),
            "{bad:?}"
        );
        assert!(
            bad.iter().any(|p| p.contains("unowned VRAM residual")),
            "{bad:?}"
        );
        // Observed below the weights is a lower bound of 0, never a wrap.
        let (_, receipt) = at(GIB);
        assert_eq!(
            receipt["device_guard"]["measured_non_weight_peak_bytes"],
            json!(0)
        );
        assert_eq!(receipt["device_guard"]["result"], json!("pass"));
        // No device observation at all: the guard cannot pass.
        let mut s = good_memory();
        for row in s.iter_mut() {
            row["drm_vram"] = Value::Null;
            row["device_free"] = Value::Null;
        }
        let (bad, receipt) = memory_problems(&s, &ledger());
        assert_eq!(receipt["device_guard"]["result"], json!("unobserved"));
        assert!(
            bad.iter().any(|p| p.contains("non-weight guard")),
            "{bad:?}"
        );
    }

    #[test]
    fn slowlink_receipt_memory_proc_parsers_dedupe_clients_and_fail_closed() {
        let status = "Name:\tx\nVmRSS:\t   100 kB\nRssAnon:\t    40 kB\nRssFile:\t    60 kB\nVmPin:\t     8 kB\nVmPeak:\t 999 kB\n";
        assert_eq!(status_bytes(status, "VmRSS"), Some(100 * 1024));
        assert_eq!(status_bytes(status, "RssFile"), Some(60 * 1024));
        assert_eq!(status_bytes(status, "VmPin"), Some(8 * 1024));
        assert_eq!(status_bytes(status, "VmHWM"), None);
        let client = |id: u32, extra: &str| {
            format!("pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{id}\n{extra}")
        };
        let both = client(7, "drm-memory-vram:\t2048 KiB\ndrm-memory-gtt:\t 512 KiB\n");
        let a = parse_drm_fdinfo(&both).unwrap();
        assert_eq!((a.vram, a.gtt), (Some(2 << 20), Some(512 << 10)));
        // Two fds of one client count once; a second client adds.
        let other = parse_drm_fdinfo(&client(
            9,
            "drm-memory-vram:\t1 MiB\ndrm-memory-gtt:\t1 MiB\n",
        ))
        .unwrap();
        assert_eq!(
            drm_totals(&[a.clone(), a.clone()]),
            Some((2 << 20, 512 << 10))
        );
        assert_eq!(
            drm_totals(&[a.clone(), other]),
            Some((3 << 20, (512 << 10) + (1 << 20)))
        );
        // Not amdgpu / no GTT counter / no client: not observable, never zero.
        assert!(parse_drm_fdinfo("pos:\t0\ndrm-driver:\ti915\ndrm-client-id:\t1\n").is_none());
        assert!(parse_drm_fdinfo("pos:\t0\nflags:\t02\n").is_none());
        let no_gtt = parse_drm_fdinfo(&client(5, "drm-memory-vram:\t4 KiB\n")).unwrap();
        assert_eq!(drm_totals(&[a, no_gtt]), None);
        assert_eq!(drm_totals(&[]), None);
        // Maxima: an observed value replaces an unobserved one.
        let mut max = ProcCounters::default();
        max.merge_max(&ProcCounters {
            drm_gtt: Some(5),
            ..ProcCounters::default()
        });
        max.merge_max(&ProcCounters {
            drm_gtt: Some(3),
            rss_file: Some(9),
            ..ProcCounters::default()
        });
        assert_eq!(
            (max.drm_gtt, max.rss_file, max.vm_pin),
            (Some(5), Some(9), None)
        );
    }

    #[test]
    fn slowlink_receipt_obsolete_residual_criterion_is_not_a_required_check() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for mode in [Mode::Primitives, Mode::State, Mode::Product, Mode::Perf] {
            for route in [Expect::LayerMajor, Expect::Incumbent] {
                let ids = required_check_ids(mode, &fixture, route);
                assert!(
                    !ids.iter().any(|n| n == "memory.unowned_residual"),
                    "{mode:?}: {ids:?}"
                );
                assert!(ids.iter().any(|n| n == "identity.manifest"), "{mode:?}");
                assert!(ids.iter().any(|n| n == "load.placement"), "{mode:?}");
            }
        }
        // The primitive cell: exactly the cases plus identity, load and the
        // sidecar's own durability.
        let ids = required_check_ids(Mode::Primitives, &fixture, Expect::LayerMajor);
        assert_eq!(ids.len(), PRIMITIVE_CASES.len() + 3);
        for case in PRIMITIVE_CASES {
            assert!(ids.contains(&format!("primitive.{case}")), "{case}");
        }
    }

    #[test]
    fn slowlink_receipt_residual_over_allowance_is_info_and_the_verdict_is_the_primitives() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        let ids = required_check_ids(Mode::Primitives, &fixture, Expect::LayerMajor);
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut checks = Checks::new(&refs);
        for id in &ids {
            checks.pass(id, Value::Null);
        }
        // VRAM and host far over every allowance and the device guard.
        let over = counters(200 * GIB, 0, 200 * GIB, 30 * GIB, 200 * GIB);
        let diag = memory_diagnostic(&memory_with(over, over), Ok(ledger()));
        let info = diag["info_errors"].as_array().unwrap();
        assert!(
            info.iter()
                .any(|p| p.as_str().unwrap().contains("unowned VRAM residual")),
            "{info:?}"
        );
        assert!(!info.is_empty());
        // The overrun cannot change the verdict or add a check row.
        assert_eq!(checks.status(), "pass");
        assert!(checks.missing().is_empty());
        assert!(!checks
            .json()
            .to_string()
            .contains("memory.unowned_residual"));
        // The census and residual stay serialized as a labelled non-gating diagnostic.
        assert_eq!(diag["required"], json!(false));
        assert_eq!(diag["gating"], json!(false));
        assert!(diag["note"]
            .as_str()
            .unwrap()
            .contains("not live allocations"));
        assert!(diag["note"]
            .as_str()
            .unwrap()
            .contains("not an actual driver"));
        assert!(diag["observed"].is_object());
        assert!(diag["residual"].is_object());
        assert!(diag["allowances"].is_object());
        assert_eq!(diag["samples"].as_array().unwrap().len(), 5);
        // A primitive failure still fails the verdict.
        checks.fail(&ids[0], "primitive broke", Value::Null);
        assert_eq!(checks.status(), "fail");
    }

    #[test]
    fn slowlink_receipt_residual_within_allowance_has_no_info_errors() {
        let diag = memory_diagnostic(&good_memory(), Ok(ledger()));
        assert_eq!(diag["info_errors"], json!([]));
        assert_eq!(diag["gating"], json!(false));
        assert_eq!(diag["residual"]["vram_bytes"], json!(GIB / 2));
    }

    #[test]
    fn slowlink_receipt_unreadable_ledger_or_telemetry_is_an_info_error_not_a_check() {
        // The census is missing from the manifest: an info_error with the samples kept.
        let samples = good_memory();
        let diag = memory_diagnostic(&samples, memory_ledger(&manifest(), PROFILE_PRIMITIVE));
        assert!(diag["info_error"]
            .as_str()
            .unwrap()
            .contains("manifest key"));
        assert_eq!(diag["samples"].as_array().unwrap().len(), samples.len());
        assert_eq!(diag["required"], json!(false));
        assert_eq!(diag["gating"], json!(false));
        // Incomplete telemetry: the diagnostic is still produced, with info_errors.
        let diag = memory_diagnostic(&[], Ok(ledger()));
        assert!(!diag["info_errors"].as_array().unwrap().is_empty());
        assert_eq!(diag["gating"], json!(false));
        // Neither case touches a verdict.
        let mut checks = Checks::new(&["primitive.x"]);
        checks.pass("primitive.x", Value::Null);
        assert_eq!(checks.status(), "pass");
    }

    // ── durable progress ───────────────────────────────────────────────

    use std::cell::RefCell;
    use std::rc::Rc;

    type SharedLines = Rc<RefCell<Vec<String>>>;

    struct MemSink {
        lines: SharedLines,
        fail_at: Option<usize>,
    }

    impl ProgressSink for MemSink {
        fn append(&mut self, line: &str) -> std::io::Result<()> {
            if Some(self.lines.borrow().len()) == self.fail_at {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "no space left on device",
                ));
            }
            self.lines.borrow_mut().push(line.to_string());
            Ok(())
        }
    }

    /// A log over memory with a deterministic clock: record `n` (1-based) is at
    /// 2024-02-29T12:34:56Z + n s and `7 n` ns monotonic.
    fn memory_log(fail_at: Option<usize>) -> (ProgressLog, SharedLines) {
        let lines: SharedLines = Rc::new(RefCell::new(Vec::new()));
        let mut tick = 0u64;
        let clock: ProgressClock = Box::new(move || {
            tick += 1;
            (1_709_210_096_000 + tick * 1000, tick * 7)
        });
        let sink = MemSink {
            lines: Rc::clone(&lines),
            fail_at,
        };
        let log = ProgressLog::new(Box::new(sink), clock, "run-a".into(), false);
        (log, lines)
    }

    fn log_text(lines: &SharedLines) -> String {
        lines.borrow().concat()
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("slowlink-progress-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn required_names() -> Vec<&'static str> {
        vec![
            "state.p512.route_receipt",
            "state.p512.repeat_determinism",
            "identity.manifest",
            "progress.durability",
        ]
    }

    /// A run killed inside repeat 1 of its first case: one completed repeat,
    /// one in flight, one executed check.
    fn partial_log(final_claims_pass: bool) -> String {
        let (log, lines) = memory_log(None);
        log.record("run_start", json!({"mode": "state"}));
        log.record("plan", json!({"required": required_names()}));
        log.record("memory", json!({"phase": "pre_load"}));
        log.record("case_start", json!({"case": "state.p512", "repeats": 3}));
        log.record("repeat_start", json!({"case": "state.p512", "repeat": 0}));
        log.record(
            "repeat_end",
            json!({"case": "state.p512", "repeat": 0, "status": "ok", "wall_s": 1.5,
                   "state_digest": "aa", "logits_digest": "bb"}),
        );
        log.record(
            "check",
            json!({"name": "identity.manifest", "status": "pass"}),
        );
        log.record("repeat_start", json!({"case": "state.p512", "repeat": 1}));
        if final_claims_pass {
            log.record("final", json!({"status": "pass", "report_sha256": "ff"}));
        }
        log_text(&lines)
    }

    #[test]
    fn slowlink_receipt_progress_utc_and_sidecar_path_are_exact() {
        assert_eq!(utc_rfc3339(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(utc_rfc3339(1_000_000_000_000), "2001-09-09T01:46:40.000Z");
        assert_eq!(utc_rfc3339(1_709_210_096_789), "2024-02-29T12:34:56.789Z");
        assert_eq!(utc_rfc3339(1_782_864_000_000), "2026-07-01T00:00:00.000Z");
        assert_eq!(
            progress_path(Path::new("/r/state.json")),
            PathBuf::from("/r/state.json.progress.jsonl")
        );
    }

    #[test]
    fn slowlink_receipt_progress_record_is_one_timed_line_and_reserved_keys_win() {
        let (log, lines) = memory_log(None);
        log.record(
            "repeat_end",
            json!({"case": "state.p512", "repeat": 0, "seq": 99, "event": "forged", "run_id": "x"}),
        );
        let text = log_text(&lines);
        assert_eq!(text.matches('\n').count(), 1, "one record is one line");
        assert!(text.ends_with('\n'));
        let record: Value = serde_json::from_str(text.trim_end()).unwrap();
        assert_eq!(record["schema"], PROGRESS_SCHEMA);
        assert_eq!(record["seq"], 0);
        assert_eq!(record["event"], "repeat_end");
        assert_eq!(record["run_id"], "run-a");
        assert_eq!(record["utc"], "2024-02-29T12:34:57.000Z");
        assert_eq!(record["unix_ms"], 1_709_210_097_000u64);
        assert_eq!(record["mono_ns"], 7);
        assert_eq!(record["case"], "state.p512");
    }

    #[test]
    fn slowlink_receipt_progress_write_error_is_sticky_and_fails_durability() {
        let (log, lines) = memory_log(Some(1));
        log.record("run_start", json!({}));
        assert!(log.durability_problems().is_empty());
        log.record("repeat_end", json!({"case": "c", "repeat": 0}));
        let error = log.error().expect("the failed write is kept");
        assert!(error.contains("repeat_end"), "{error}");
        assert!(error.contains("no space left"), "{error}");
        // Nothing after a lost record is written as if the log were intact.
        log.record("repeat_end", json!({"case": "c", "repeat": 1}));
        assert_eq!(lines.borrow().len(), 1);
        assert_eq!(log.records(), 1);
        let mut checks = Checks::new(&["progress.durability"]);
        checks.push(
            "progress.durability",
            log.durability_problems(),
            Value::Null,
        );
        assert_eq!(checks.status(), "fail");
    }

    #[test]
    fn slowlink_receipt_progress_checks_are_written_once_each_in_order() {
        let (log, lines) = memory_log(None);
        let mut checks = Checks::new(&["a", "b"]);
        checks.pass("a", json!({"x": 1}));
        log.checks(&checks);
        log.checks(&checks);
        checks.fail("b", "boom", Value::Null);
        log.checks(&checks);
        let records: Vec<Value> = log_text(&lines)
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(records.len(), 2, "each row exactly once");
        assert_eq!(
            (records[0]["name"].as_str(), records[0]["status"].as_str()),
            (Some("a"), Some("pass"))
        );
        assert_eq!(
            (records[1]["name"].as_str(), records[1]["reason"].as_str()),
            (Some("b"), Some("boom"))
        );
        assert!(records.iter().all(|r| r["event"] == "check"));
    }

    #[test]
    fn slowlink_receipt_progress_partial_summary_keeps_progress_and_never_passes() {
        for claims_pass in [false, true] {
            let summary = progress_summary(&partial_log(claims_pass));
            // A killed run is incomplete even if a final record is present.
            assert_eq!(summary["status"], "incomplete", "{claims_pass}");
            assert_eq!(summary["promotable"], false);
            assert_eq!(summary["run_id"], "run-a");
            assert_eq!(summary["torn_tail"], false);
            assert_eq!(summary["seq_contiguous"], true);
            assert_eq!(summary["memory_phases"], json!(["pre_load"]));
            let case = &summary["cases"]["state.p512"];
            assert_eq!(case["repeats_completed"], 1);
            assert_eq!(case["in_flight_repeat"], 1);
            assert_eq!(case["last_completed"]["state_digest"], "aa");
            assert_eq!(case["last_completed"]["wall_s"], 1.5);
            // Unexecuted required checks are named, not filled in.
            assert_eq!(
                summary["missing_required"],
                json!([
                    "state.p512.route_receipt",
                    "state.p512.repeat_determinism",
                    "progress.durability"
                ])
            );
            assert_eq!(summary["failed_checks"], json!([]));
        }
        let with_final = progress_summary(&partial_log(true));
        assert_eq!(with_final["final_recorded"]["status"], "pass");
        assert_eq!(with_final["status"], "incomplete");
    }

    #[test]
    fn slowlink_receipt_progress_torn_tail_is_dropped_and_the_last_repeat_is_kept() {
        let full = partial_log(false);
        for torn in [
            &full[..full.len() - 10],
            full.trim_end_matches('\n'), // newline lost: the record is suspect
        ] {
            let summary = progress_summary(torn);
            assert_eq!(summary["torn_tail"], true);
            assert_eq!(summary["last_event"], "check");
            assert_eq!(summary["cases"]["state.p512"]["repeats_completed"], 1);
            assert_eq!(
                summary["cases"]["state.p512"]["in_flight_repeat"],
                Value::Null
            );
            assert_eq!(summary["status"], "incomplete");
        }
    }

    #[test]
    fn slowlink_receipt_progress_corrupt_middle_line_is_never_interpreted() {
        let mut lines: Vec<String> = partial_log(false).lines().map(String::from).collect();
        lines[4] = "{\"schema\":\"hipfire.qwen4.slowlink_gate.progress.v1\",\"seq\":4,".into();
        let summary = progress_summary(&(lines.join("\n") + "\n"));
        assert_eq!(summary["corrupt_lines"], json!([5]));
        assert_eq!(summary["seq_contiguous"], false);
        assert_ne!(summary["status"], "pass");
        // The corrupted repeat_start is not guessed back: repeat 0 has an end
        // but no valid start, repeat 1 is still the one in flight.
        assert_eq!(summary["cases"]["state.p512"]["in_flight_repeat"], 1);
    }

    #[test]
    fn slowlink_receipt_progress_recorded_failure_is_a_fail_not_an_incomplete() {
        let (log, lines) = memory_log(None);
        log.record("run_start", json!({}));
        log.record("plan", json!({"required": required_names()}));
        let mut checks = Checks::new(&required_names());
        checks.pass("identity.manifest", Value::Null);
        checks.fail(
            "state.p512.route_receipt",
            "repeat 1: hipMalloc failed",
            Value::Null,
        );
        log.checks(&checks);
        let summary = progress_summary(&log_text(&lines));
        assert_eq!(summary["status"], "fail");
        assert_eq!(
            summary["failed_checks"],
            json!(["state.p512.route_receipt"])
        );
        assert_eq!(
            summary["missing_required"],
            json!(["state.p512.repeat_determinism", "progress.durability"])
        );
    }

    #[test]
    fn slowlink_receipt_progress_reopen_seals_a_torn_line_and_summarizes_the_new_run() {
        let dir = scratch_dir("seal");
        let path = dir.join("out.json.progress.jsonl");
        std::fs::write(
            &path,
            "{\"schema\":\"hipfire.qwen4.slowlink_gate.progress.v1\",\"seq\":0,\"ev",
        )
        .unwrap();
        let log = ProgressLog::open_file(&path, "run-b".into(), false).unwrap();
        log.record("run_start", json!({}));
        log.record("abort", json!({"error": "load: out of memory"}));
        assert!(log.error().is_none());
        let summary = progress_summary(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(summary["run_id"], "run-b");
        assert_eq!(summary["corrupt_lines"], json!([1]));
        assert_eq!(summary["torn_tail"], false);
        assert_eq!(summary["run_start_seen"], true);
        assert_eq!(summary["seq_contiguous"], true);
        assert_eq!(summary["records"], 2);
        assert_eq!(summary["error"], "load: out of memory");
        assert_eq!(summary["status"], "incomplete");
        std::fs::remove_dir_all(&dir).unwrap();
        let missing = Path::new("/nonexistent-slowlink-dir/x.progress.jsonl");
        let error = ProgressLog::open_file(missing, "run-c".into(), false)
            .err()
            .expect("an unwritable sidecar is an error before any GPU work");
        assert!(error.contains("open progress sidecar"), "{error}");
    }

    #[test]
    fn slowlink_receipt_write_atomic_replaces_whole_files_and_leaves_no_temporary() {
        let dir = scratch_dir("atomic");
        let path = dir.join("report.json");
        write_atomic(&path, b"old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        // A failed publish (the target is a non-empty directory) is an error
        // and removes its temporary file.
        let blocked = dir.join("blocked");
        std::fs::create_dir_all(blocked.join("inner")).unwrap();
        let error = write_atomic(&blocked, b"x").unwrap_err();
        assert!(error.contains("blocked"), "{error}");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["blocked", "report.json"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn slowlink_receipt_aborted_report_is_incomplete_and_carries_the_sidecar_summary() {
        let dir = scratch_dir("abort");
        let output = dir.join("state.json");
        let arg = |s: &str| s.to_string();
        let args = parse_args([
            arg("--model"),
            arg("/m"),
            arg("--fixture"),
            arg("/f"),
            arg("--manifest"),
            arg("/man"),
            arg("--output"),
            output.to_string_lossy().into_owned(),
            arg("--mode"),
            arg("state"),
            arg("--route"),
            arg("layer-major"),
        ])
        .unwrap();
        // No sidecar yet: the report says so rather than inventing one.
        let bare = aborted_report(&args, "setup failed", None);
        assert_eq!(bare["status"], "incomplete");
        assert!(bare["progress"]["summary"]["unreadable"].is_string());
        std::fs::write(progress_path(&output), partial_log(false)).unwrap();
        let report = aborted_report(&args, "timed out", None);
        assert_eq!(report["status"], "incomplete");
        assert_eq!(report["error"], "timed out");
        assert_eq!(
            report["progress"]["summary"]["cases"]["state.p512"]["repeats_completed"],
            1
        );
        assert!(
            report.get("checks").is_none(),
            "no verdict rows are invented"
        );
        write_report(&output, &report).unwrap();
        let written: Value =
            serde_json::from_str(&std::fs::read_to_string(&output).unwrap()).unwrap();
        assert_eq!(written["status"], "incomplete");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn slowlink_receipt_progress_durability_is_required_and_a_partial_run_is_incomplete() {
        let fixture = Fixture::parse(&fixture()).unwrap();
        for mode in [Mode::Primitives, Mode::State, Mode::Product, Mode::Perf] {
            let ids = required_check_ids(mode, &fixture, Expect::LayerMajor);
            assert!(ids.iter().any(|n| n == "progress.durability"), "{mode:?}");
            let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
            let mut checks = Checks::new(&refs);
            for id in ids.iter().filter(|n| *n != "progress.durability") {
                checks.pass(id, Value::Null);
            }
            assert_eq!(checks.status(), "incomplete", "{mode:?}");
            assert_eq!(checks.missing(), ["progress.durability"]);
            checks.pass("progress.durability", Value::Null);
            assert_eq!(checks.status(), "pass", "{mode:?}");
        }
    }

    #[test]
    fn slowlink_receipt_primitive_later_error_fails_only_its_case_and_leaves_the_rest_unproved() {
        let ids: Vec<String> = PRIMITIVE_CASES
            .iter()
            .map(|case| format!("primitive.{case}"))
            .collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let mut checks = Checks::new(&refs);
        for name in &PRIMITIVE_CASES[..2] {
            judge_primitive(&mut checks, name, &case_report(name, "pass"));
        }
        let third = PRIMITIVE_CASES[2];
        judge_primitive(
            &mut checks,
            third,
            &primitive_error(third, "slow-link primitive could not run: stream lost"),
        );
        assert_eq!(checks.status(), "fail");
        let failed = rows(&checks)
            .into_iter()
            .filter(|r| r["status"] == "fail")
            .count();
        assert_eq!(failed, 1, "the exception is not cloned onto other cases");
        assert_eq!(checks.missing().len(), PRIMITIVE_CASES.len() - 3);
        let r = row(&checks, &format!("primitive.{third}"));
        assert!(r["reason"].as_str().unwrap().contains("stream lost"));
        assert!(r["detail"]["mismatches"].is_null());
    }

    #[test]
    fn slowlink_receipt_progress_window_summary_keeps_only_the_windows_and_never_invents_one() {
        let receipt = json!({
            "route": "layer-major", "entry_pos": 0, "prompt_rows": 16384,
            "windows": [[0, 16384]], "launch_cuts": [], "tile_rows": 1536,
            "spans_ms": {"copy": 1.0}, "cache_usage": {"pool_bytes": 1},
        });
        let summary = window_summary(&[Some(receipt), None]);
        assert_eq!(summary[0]["windows"], json!([[0, 16384]]));
        assert_eq!(summary[0]["tile_rows"], 1536);
        assert!(summary[0].get("spans_ms").is_none());
        assert!(summary[0].get("cache_usage").is_none());
        assert!(summary[1].is_null(), "a missing receipt stays missing");
    }

    fn product_receipt(tile: u64) -> Value {
        let mut r = receipt();
        r["tile_rows"] = json!(tile);
        r["retained_scratch_rows"] = json!(tile);
        r
    }

    #[test]
    fn slowlink_receipt_product_receipt_must_name_the_admitted_tile_not_just_stay_under_1536() {
        let product = purpose_tile_rows(AdmissionPurpose::Product) as u64;
        assert_eq!(product, 768);
        // The exact admitted tile passes.
        let ok = call_problems(
            Expect::LayerMajor,
            product,
            Some(&product_receipt(768)),
            16384,
            true,
            true,
        );
        assert!(ok.is_empty(), "{ok:?}");
        // The old 1,536 tile and the primitive 512 tile are each under the generic
        // 1,536 upper bound and must still fail a product claim.
        for wrong in [1536u64, 512, 1024] {
            let bad = call_problems(
                Expect::LayerMajor,
                product,
                Some(&product_receipt(wrong)),
                16384,
                true,
                true,
            );
            assert!(
                bad.iter().any(|p| p.contains("admitted profile")),
                "tile {wrong}: {bad:?}"
            );
        }
        // A product tile with more retained scratch than the tile fails too.
        let mut roomy = product_receipt(768);
        roomy["retained_scratch_rows"] = json!(1536);
        assert!(
            !call_problems(Expect::LayerMajor, product, Some(&roomy), 16384, true, true).is_empty()
        );
        // The generic historical validator is unchanged: no expected tile, 1,536 passes.
        let generic = Limits {
            require_single_window: true,
            require_overlap: true,
            ..Limits::default()
        };
        assert!(validate(Some(&receipt()), Expect::LayerMajor, &generic).is_empty());
        assert!(validate(Some(&product_receipt(768)), Expect::LayerMajor, &generic).is_empty());
    }

    #[test]
    fn slowlink_receipt_primitive_purpose_expects_512_and_keeps_the_named_short_bypass() {
        let primitive = purpose_tile_rows(AdmissionPurpose::PrimitiveSelftest) as u64;
        assert_eq!(primitive, 512);
        let ok = call_problems(
            Expect::LayerMajor,
            primitive,
            Some(&product_receipt(512)),
            16384,
            true,
            true,
        );
        assert!(ok.is_empty(), "{ok:?}");
        assert!(!call_problems(
            Expect::LayerMajor,
            primitive,
            Some(&product_receipt(768)),
            16384,
            true,
            true
        )
        .is_empty());
        // The approved <=1536-row named incumbent bypass is independent of the tile.
        let mut bypass = incumbent();
        bypass["bypass_reason"] = json!("request at most one tile");
        bypass["prompt_rows"] = json!(1536);
        bypass["windows"] = json!([{"start": 0, "rows": 1536}]);
        bypass["launch_cuts"] = json!([]);
        for tile in [768u64, 512] {
            let bad = call_problems(Expect::LayerMajor, tile, Some(&bypass), 1536, false, false);
            assert!(bad.is_empty(), "tile {tile}: {bad:?}");
            // ...but the bypass without a named reason still fails.
            bypass["bypass_reason"] = Value::Null;
            assert!(
                !call_problems(Expect::LayerMajor, tile, Some(&bypass), 1536, false, false)
                    .is_empty()
            );
            bypass["bypass_reason"] = json!("request at most one tile");
        }
        // A call longer than the bypass bound must run layer-major, whatever the tile.
        let bad = call_problems(
            Expect::LayerMajor,
            768,
            Some(&incumbent()),
            1537,
            false,
            false,
        );
        assert!(!bad.is_empty());
    }

    #[test]
    fn slowlink_receipt_one_manifest_represents_both_purpose_profiles() {
        let m = manifest();
        assert_eq!(PROFILE_PRODUCT, "product");
        assert_eq!(PROFILE_PRIMITIVE, "primitive_selftest");
        assert_eq!(manifest_profile_tile(&m, PROFILE_PRODUCT), Some(768));
        assert_eq!(manifest_profile_tile(&m, PROFILE_PRIMITIVE), Some(512));
        assert_eq!(
            purpose_name(AdmissionPurpose::Product),
            PROFILE_PRODUCT,
            "the product cells read the product profile"
        );
        assert_eq!(
            purpose_name(load_purpose(Mode::Primitives)),
            PROFILE_PRIMITIVE,
            "only the primitive cell reads the primitive profile"
        );
        for mode in [Mode::State, Mode::Product, Mode::Perf] {
            assert_eq!(purpose_name(load_purpose(mode)), PROFILE_PRODUCT);
        }
        assert!(manifest_problems(&m, &observed()).is_empty());
        let admitted = (
            purpose_tile_rows(AdmissionPurpose::Product) as u64,
            purpose_tile_rows(AdmissionPurpose::PrimitiveSelftest) as u64,
        );
        assert!(profile_admission_problems(&m, admitted.0, admitted.1).is_empty());
        // One tile claimed for both purposes, a missing or zero profile, a
        // tile above the historical bound and a manifest that names a tile
        // other than the one this binary admits all fail closed.
        let mut same = manifest();
        for profile in [PROFILE_PRODUCT, PROFILE_PRIMITIVE] {
            same["memory_ledger"]["profiles"][profile]["tile_rows"] = json!(1536);
        }
        assert!(manifest_problems(&same, &observed())
            .iter()
            .any(|p| p.contains("same tile 1536")));
        assert!(!profile_admission_problems(&same, admitted.0, admitted.1).is_empty());
        for broken in [json!(0), json!(1537), json!("768"), Value::Null] {
            let mut m = manifest();
            m["memory_ledger"]["profiles"]["product"]["tile_rows"] = broken;
            assert!(!manifest_problems(&m, &observed()).is_empty());
        }
        let mut swapped = manifest();
        swapped["memory_ledger"]["profiles"]["product"]["tile_rows"] = json!(512);
        swapped["memory_ledger"]["profiles"]["primitive_selftest"]["tile_rows"] = json!(768);
        assert!(manifest_problems(&swapped, &observed()).is_empty());
        let bad = profile_admission_problems(&swapped, admitted.0, admitted.1);
        assert_eq!(bad.len(), 2, "{bad:?}");
    }

    #[test]
    fn slowlink_receipt_profile_ledgers_are_full_and_the_root_owned_bytes_are_the_max() {
        let m = manifest();
        assert!(profile_shape_problems(&m).is_empty());
        // Root device_owned_bytes is the MAX of the per-purpose owned bytes, not
        // the product core plus the primitive diagnostic.
        let product = manifest_profile_u64(&m, PROFILE_PRODUCT, "device_owned_bytes").unwrap();
        let primitive = manifest_profile_u64(&m, PROFILE_PRIMITIVE, "device_owned_bytes").unwrap();
        assert_eq!(
            u64_at(&m, "memory_ledger.device_owned_bytes"),
            Some(product.max(primitive))
        );
        let mut sum = manifest();
        sum["memory_ledger"]["device_owned_bytes"] = json!(product + primitive);
        assert!(profile_shape_problems(&sum)
            .iter()
            .any(|p| p.contains("MAX")));
        // Every profile field is required, there is no flat fallback, and a
        // missing field in either profile fails identity.
        for profile in [PROFILE_PRODUCT, PROFILE_PRIMITIVE] {
            for key in [
                "purpose",
                "tile_rows",
                "resident_expert_layers",
                "window_rows",
                "page_count",
                "device_weights_bytes",
                "s1_tile_census_bytes",
                "s1_tile_census_terms",
                "s1_extras_bytes",
                "s1_extras_terms",
                "device_core_bytes",
                "sequential_diag_bytes",
                "device_owned_bytes",
                "runtime_budget_bytes",
                "safety_margin_bytes",
            ] {
                let mut m = manifest();
                m["memory_ledger"]["profiles"][profile]
                    .as_object_mut()
                    .unwrap()
                    .remove(key);
                assert!(
                    !manifest_problems(&m, &observed()).is_empty(),
                    "{profile}.{key} must be required"
                );
            }
        }
        let mut flat = manifest();
        flat["memory_ledger"]["profiles"] = json!({});
        assert!(!manifest_problems(&flat, &observed()).is_empty());
        // The purpose names its own profile; the product carries no diagnostic
        // reserve; the primitive does; the arithmetic must close.
        let mut wrong = manifest();
        wrong["memory_ledger"]["profiles"]["product"]["purpose"] = json!("primitive_selftest");
        assert!(!profile_shape_problems(&wrong).is_empty());
        let mut diag = manifest();
        diag["memory_ledger"]["profiles"]["product"]["sequential_diag_bytes"] = json!(GIB);
        assert!(profile_shape_problems(&diag)
            .iter()
            .any(|p| p.contains("no diagnostic reserve")));
        let mut nodiag = manifest();
        nodiag["memory_ledger"]["profiles"]["primitive_selftest"]["sequential_diag_bytes"] =
            json!(0);
        assert!(!profile_shape_problems(&nodiag).is_empty());
        let mut core = manifest();
        core["memory_ledger"]["profiles"]["product"]["device_core_bytes"] = json!(GIB);
        assert!(!profile_shape_problems(&core).is_empty());
        // The sealed resident count is reference data: zero, nine or eleven
        // are all valid counts, one that leaves no host layer is not.
        for reference in [0u64, 9, 11] {
            let mut ok = manifest();
            ok["memory_ledger"]["profiles"]["product"]["resident_expert_layers"] = json!(reference);
            assert!(
                profile_shape_problems(&ok).is_empty(),
                "reference {reference}"
            );
        }
        let mut all_resident = manifest();
        all_resident["memory_ledger"]["profiles"]["product"]["resident_expert_layers"] =
            json!(TRUNK_LAYERS);
        assert!(!profile_shape_problems(&all_resident).is_empty());
        let mut unlabelled = manifest();
        unlabelled["memory_ledger"]["profiles"]["product"]["projection_role"] = json!("observed");
        assert!(profile_shape_problems(&unlabelled)
            .iter()
            .any(|p| p.contains("reference data")));
        let mut few = manifest();
        few["memory_ledger"]["profiles"]["product"]
            .as_object_mut()
            .unwrap()
            .remove("projection_role");
        assert!(!profile_shape_problems(&few).is_empty());
        let mut window = manifest();
        window["memory_ledger"]["profiles"]["product"]["window_rows"] = json!(8192);
        assert!(!profile_shape_problems(&window).is_empty());
        // The primitive ledger the memory diagnostic reads is its own profile's.
        let mut with_root = manifest();
        with_root["memory_ledger"]["census_basis"] = json!("basis");
        with_root["memory_ledger"]["owned_host_bytes"] = json!(80 * GIB);
        let ledger = memory_ledger(&with_root, PROFILE_PRIMITIVE).unwrap();
        assert_eq!(ledger.owned_device_bytes, primitive);
        assert_eq!(ledger.decomposition.sequential_diag, GIB);
        assert_eq!(
            memory_ledger(&with_root, PROFILE_PRODUCT)
                .unwrap_err()
                .contains("sequential_diag_bytes"),
            true,
            "the product profile has no diagnostic bytes to decompose"
        );
    }

    #[test]
    fn slowlink_receipt_profile_terms_sum_exactly_and_charge_the_cold_f16_once() {
        let m = manifest();
        for profile in [PROFILE_PRODUCT, PROFILE_PRIMITIVE] {
            assert!(profile_terms_problems(&m, profile, true).is_empty());
            assert!(profile_terms_problems(&m, profile, false).is_empty());
        }
        let terms = |m: &mut Value, profile: &str, key: &str| -> Vec<Value> {
            m["memory_ledger"]["profiles"][profile][key]
                .as_array()
                .unwrap()
                .clone()
        };
        for profile in [PROFILE_PRODUCT, PROFILE_PRIMITIVE] {
            // A term sum that is off by one byte, in either array, fails.
            for (key, census) in [("s1_tile_census_terms", true), ("s1_extras_terms", false)] {
                let mut m = manifest();
                let mut array = terms(&mut m, profile, key);
                let last = array.len() - 1;
                let bytes = array[last]["bytes"].as_u64().unwrap();
                array[last]["bytes"] = json!(bytes + 1);
                m["memory_ledger"]["profiles"][profile][key] = json!(array);
                let bad = profile_terms_problems(&m, profile, census);
                assert!(
                    bad.iter().any(|p| p.contains("sums to")),
                    "{profile} {key}: {bad:?}"
                );
                assert!(!manifest_problems(&m, &observed()).is_empty());
            }
            // The cold F16 term must appear exactly once in the census terms.
            let mut none = manifest();
            let kept: Vec<Value> = terms(&mut none, profile, "s1_tile_census_terms")
                .into_iter()
                .filter(|t| t["name"] != COLD_F16_TERM)
                .collect();
            none["memory_ledger"]["profiles"][profile]["s1_tile_census_terms"] = json!(kept);
            assert!(profile_terms_problems(&none, profile, true)
                .iter()
                .any(|p| p.contains("expected exactly 1")));
            let mut twice = manifest();
            let mut array = terms(&mut twice, profile, "s1_tile_census_terms");
            let cold = array
                .iter()
                .find(|t| t["name"] == COLD_F16_TERM)
                .unwrap()
                .clone();
            array.push(cold);
            twice["memory_ledger"]["profiles"][profile]["s1_tile_census_terms"] = json!(array);
            let bad = profile_terms_problems(&twice, profile, true);
            assert!(
                bad.iter().any(|p| p.contains("repeats the term name")),
                "{bad:?}"
            );
            assert!(
                bad.iter().any(|p| p.contains("expected exactly 1")),
                "{bad:?}"
            );
            // ...and never in the extras (no duplicate physical charge).
            let mut extra = manifest();
            let mut array = terms(&mut extra, profile, "s1_extras_terms");
            array.push(json!({"name": COLD_F16_TERM, "bytes": 0}));
            extra["memory_ledger"]["profiles"][profile]["s1_extras_terms"] = json!(array);
            assert!(profile_terms_problems(&extra, profile, false)
                .iter()
                .any(|p| p.contains("expected exactly 0")));
            // The stated cold bound is that one term, and nothing else.
            let stated = manifest_profile_u64(&m, profile, "f16_shadow_cold_bound_bytes");
            assert!(stated.is_some_and(|n| n > 0));
            let mut wrong = manifest();
            wrong["memory_ledger"]["profiles"][profile]["f16_shadow_cold_bound_bytes"] =
                json!(stated.unwrap() + 1);
            assert!(profile_terms_problems(&wrong, profile, true)
                .iter()
                .any(|p| p.contains("f16_shadow_cold_bound_bytes")));
            // Malformed terms fail: not an array, a missing name or bytes, a
            // negative or fractional byte count, an empty name.
            for bad_terms in [
                json!({"tile": 1}),
                json!([{"bytes": 1}]),
                json!([{"name": "a"}]),
                json!([{"name": "a", "bytes": -1}]),
                json!([{"name": "a", "bytes": 1.5}]),
                json!([{"name": "", "bytes": 1}]),
            ] {
                let mut m = manifest();
                m["memory_ledger"]["profiles"][profile]["s1_extras_terms"] = bad_terms;
                assert!(!profile_terms_problems(&m, profile, false).is_empty());
            }
            // A u64 overflow of the term sum refuses rather than wrapping.
            let mut big = manifest();
            big["memory_ledger"]["profiles"][profile]["s1_extras_terms"] = json!([
                {"name": "a", "bytes": u64::MAX}, {"name": "b", "bytes": 1}
            ]);
            assert!(profile_terms_problems(&big, profile, false)
                .iter()
                .any(|p| p.contains("overflows")));
        }
    }

    #[test]
    fn slowlink_receipt_actual_admitted_profile_must_match_the_sealed_profile() {
        let m = manifest();
        let ok = profile_load_problems(&m, PROFILE_PRODUCT, "product", Some(768), Some(16384));
        assert!(ok.is_empty(), "{ok:?}");
        let ok = profile_load_problems(
            &m,
            PROFILE_PRIMITIVE,
            "primitive_selftest",
            Some(512),
            Some(16384),
        );
        assert!(ok.is_empty(), "{ok:?}");
        // A product cell whose bundle admitted the primitive profile, another
        // tile, another window or no profile at all fails. The resident count
        // is not compared: the sealed count is reference data.
        for (purpose, tile, window) in [
            ("primitive_selftest", Some(768), Some(16384)),
            ("product", Some(512), Some(16384)),
            ("product", Some(1536), Some(16384)),
            ("product", Some(768), Some(8192)),
            ("product", None, None),
        ] {
            let bad = profile_load_problems(&m, PROFILE_PRODUCT, purpose, tile, window);
            assert!(!bad.is_empty(), "{purpose} {tile:?} {window:?}");
        }
    }

    #[test]
    fn slowlink_receipt_profile_evidence_separates_expected_from_bundle_admitted() {
        let m = manifest();
        // Before a successful load nothing is admitted: expected values are
        // never reported as admitted.
        let primitive = s1_profile_evidence(AdmissionPurpose::PrimitiveSelftest, &m, None, None);
        assert_eq!(primitive["expected_purpose"], "primitive_selftest");
        assert_eq!(primitive["expected_tile_rows"], 512);
        assert!(primitive["admitted_purpose"].is_null());
        assert!(primitive["admitted_tile_rows"].is_null());
        assert!(primitive["admission"].is_null());
        assert!(primitive["qualification"]
            .as_str()
            .unwrap()
            .contains("not a product-shape"));
        let product = s1_profile_evidence(AdmissionPurpose::Product, &m, None, None);
        assert_eq!(product["expected_purpose"], "product");
        assert_eq!(product["expected_tile_rows"], 768);
        assert!(product["admitted_purpose"].is_null());
        assert!(product["admitted_tile_rows"].is_null());
        assert_eq!(product["manifest_primitive_tile_rows"], 512);
        assert!(product["qualification"]
            .as_str()
            .unwrap()
            .contains("product-shape claim"));
        // The manifest's resident count is labelled reference, never observed.
        let reference = &product["manifest_resident_expert_layers_reference"];
        assert_eq!(reference["role"], "reference");
        assert_eq!(reference["product"], 10);
        // After a load the admitted values are the bundle's own, whatever was
        // expected (a mismatch is visible, not papered over).
        let admitted = s1_profile_evidence(
            AdmissionPurpose::Product,
            &m,
            Some(("primitive_selftest", 512)),
            None,
        );
        assert_eq!(admitted["expected_purpose"], "product");
        assert_eq!(admitted["admitted_purpose"], "primitive_selftest");
        assert_eq!(admitted["admitted_tile_rows"], 512);
        let witness = admission_witness_json(&quote(POST_INIT, Some(PRE_INIT), 9));
        let ok = s1_profile_evidence(
            AdmissionPurpose::Product,
            &m,
            Some(("product", 768)),
            Some(&witness),
        );
        assert_eq!(ok["admitted_purpose"], "product");
        assert_eq!(ok["admitted_tile_rows"], 768);
        assert_eq!(ok["admission"], witness);
    }

    // ── adaptive admission witness ───────────────────────────────────────
    //
    // SYNTHETIC accounting fixture: the two free samples are the j1811
    // progress samples (before initialization / after it); the layer cost,
    // non-expert bytes and joint reserve are round numbers chosen so that
    // nine resident layers is the maximal count at the post-initialization
    // sample and ten at the reference sample. It is test data for the
    // validator, not the real model's quote (that is the shared Rust emit).

    const PRE_INIT: u64 = 25_713_180_672;
    const POST_INIT: u64 = 25_560_088_576;
    const LAYER_COST: u64 = 1_342_177_280;
    const NON_EXPERT: u64 = 8_589_934_592;
    const JOINT_RESERVE: u64 = 3_598_381_184;

    fn base_needed(n: u64) -> u128 {
        NON_EXPERT as u128 + n as u128 * LAYER_COST as u128 + JOINT_RESERVE as u128
    }

    fn quote(free: u64, before: Option<u64>, n: u64) -> AdmissionScalars {
        let needed = base_needed(n);
        AdmissionScalars {
            free_before_initialization: before,
            free_at_admission: free,
            observed_initialization: before.map(|b| b.saturating_sub(free)),
            non_expert: NON_EXPERT,
            resident_layer: LAYER_COST,
            joint_reserve: JOINT_RESERVE,
            trunk_layers: 48,
            resident_layers: n,
            needed,
            slack: free as i128 - needed as i128,
            preferred_deficit: u64::try_from(base_needed(10).saturating_sub(free as u128)).unwrap(),
        }
    }

    fn actual_of(s: &AdmissionScalars) -> AdmissionActual {
        AdmissionActual {
            resident_layers: Some(s.resident_layers),
            trunk_layers: Some(48),
            snapshot_budget_bytes: u64::try_from(s.slack).ok(),
        }
    }

    fn judged(s: &AdmissionScalars) -> Vec<String> {
        admission_witness_problems(Some(&admission_witness_json(s)), &actual_of(s))
    }

    fn tampered(s: &AdmissionScalars, key: &str, edit: impl FnOnce(&mut Value)) -> Vec<String> {
        let mut w = admission_witness_json(s);
        edit(&mut w[key]);
        admission_witness_problems(Some(&w), &actual_of(s))
    }

    fn base_free(n: u64, extra: u64) -> u64 {
        u64::try_from(base_needed(n)).unwrap() + extra
    }

    #[test]
    fn slowlink_receipt_adaptive_admission_accepts_the_maximal_count_at_n9_n10_and_n0() {
        // N9 at the fresh post-initialization sample: the measured interval
        // loss is explicit, ten is preferred (a 50,000,000 B deficit is
        // information, not a rejection) and nine is maximal.
        let n9 = quote(POST_INIT, Some(PRE_INIT), 9);
        assert_eq!(n9.observed_initialization, Some(153_092_096));
        assert_eq!(n9.preferred_deficit, 50_000_000);
        assert_eq!(judged(&n9), Vec::<String>::new());
        let w = admission_witness_json(&n9);
        assert_eq!(w["memory_basis"], "post_initialization_actual");
        assert_eq!(w["initialization_charged_via_free"], true);
        assert_eq!(w["preferred_resident_layers"], 10);
        assert_eq!(w["resident_layers"], 9);
        assert_eq!(w["free_at_admission_bytes"], POST_INIT);
        assert_eq!(w["free_before_initialization_bytes"], PRE_INIT);
        // N10 at the reference sample (no initialization loss).
        let n10 = quote(PRE_INIT, Some(PRE_INIT), 10);
        assert_eq!(n10.preferred_deficit, 0);
        assert_eq!(n10.observed_initialization, Some(0));
        assert_eq!(judged(&n10), Vec::<String>::new());
        // N0 is a valid count when the base fits and host layers remain.
        let n0 = quote(base_free(0, 100), None, 0);
        assert_eq!(n0.slack, 100);
        assert_eq!(judged(&n0), Vec::<String>::new());
        // Exact and one-byte boundaries at 9: exact fit keeps 9; one byte
        // short of the next layer stays 9; the next layer's exact cost is 10.
        for (extra, n) in [
            (0u64, 9u64),
            (LAYER_COST - 1, 9),
            (LAYER_COST, 10),
            (LAYER_COST + 1, 10),
        ] {
            let s = quote(base_free(9, extra), None, n);
            assert_eq!(judged(&s), Vec::<String>::new(), "extra {extra} n {n}");
        }
        // A larger count than ten is data too, when the actual free supports it.
        assert_eq!(
            judged(&quote(base_free(11, 5), None, 11)),
            Vec::<String>::new()
        );
        // A fresh free above the earlier sample invents no positive loss:
        // the observed interval is 0 (or absent); a positive one is forged.
        let gained = quote(PRE_INIT, Some(POST_INIT), 10);
        assert_eq!(gained.observed_initialization, Some(0));
        assert_eq!(judged(&gained), Vec::<String>::new());
        let mut absent = gained;
        absent.observed_initialization = None;
        assert_eq!(judged(&absent), Vec::<String>::new());
        let mut invented = gained;
        invented.observed_initialization = Some(1);
        assert!(!judged(&invented).is_empty());
    }

    #[test]
    fn slowlink_receipt_adaptive_admission_refuses_nonmaximal_oversubscribed_and_all_resident() {
        // Nine claimed where ten also fits: not maximal.
        let nonmax = quote(base_free(9, LAYER_COST), None, 9);
        assert!(judged(&nonmax).iter().any(|m| m.contains("not maximal")));
        assert!(judged(&quote(base_free(0, LAYER_COST), None, 0))
            .iter()
            .any(|m| m.contains("not maximal")));
        // Ten claimed at the post-initialization sample: oversubscribed.
        let over = quote(POST_INIT, Some(PRE_INIT), 10);
        assert!(over.slack < 0);
        assert!(judged(&over).iter().any(|m| m.contains("oversubscribed")));
        // The base does not fit: no count, not even zero, is a valid fit.
        let no_base = quote(base_free(0, 0) - 1, None, 0);
        assert_eq!(no_base.slack, -1);
        assert!(judged(&no_base)
            .iter()
            .any(|m| m.contains("oversubscribed")));
        // Every layer resident leaves no host layer: an S1 profile is not claimed.
        let all = quote(base_free(48, 5), None, 48);
        assert!(judged(&all).iter().any(|m| m.contains("no host layer")));
        // The bundle's own counts and the snapshot allowance must be the
        // witness's.
        let n9 = quote(POST_INIT, Some(PRE_INIT), 9);
        let w = admission_witness_json(&n9);
        let slack = u64::try_from(n9.slack).unwrap();
        for actual in [
            AdmissionActual {
                resident_layers: Some(10),
                ..actual_of(&n9)
            },
            AdmissionActual {
                resident_layers: None,
                ..actual_of(&n9)
            },
            AdmissionActual {
                trunk_layers: Some(47),
                ..actual_of(&n9)
            },
            AdmissionActual {
                trunk_layers: None,
                ..actual_of(&n9)
            },
            AdmissionActual {
                snapshot_budget_bytes: Some(slack + 1),
                ..actual_of(&n9)
            },
            AdmissionActual {
                snapshot_budget_bytes: None,
                ..actual_of(&n9)
            },
        ] {
            assert!(
                !admission_witness_problems(Some(&w), &actual).is_empty(),
                "{actual:?}"
            );
        }
    }

    #[test]
    fn slowlink_receipt_adaptive_admission_refuses_missing_and_tampered_witnesses() {
        assert!(!admission_witness_problems(None, &AdmissionActual::default()).is_empty());
        assert!(
            !admission_witness_problems(Some(&Value::Null), &AdmissionActual::default()).is_empty()
        );
        assert!(
            !admission_witness_problems(Some(&json!([])), &AdmissionActual::default()).is_empty()
        );
        let n9 = quote(POST_INIT, Some(PRE_INIT), 9);
        let w = admission_witness_json(&n9);
        let keys: Vec<String> = w.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys.len(), 14, "{keys:?}");
        for key in &keys {
            let mut missing = w.clone();
            missing.as_object_mut().unwrap().remove(key);
            assert!(
                !admission_witness_problems(Some(&missing), &actual_of(&n9)).is_empty(),
                "removing {key} must fail"
            );
            assert!(
                !tampered(&n9, key, |v| *v = Value::Null).is_empty(),
                "nulling {key} must fail"
            );
        }
        // Every integer forged one byte (layer) either way is caught by the
        // independent recomputation.
        for key in [
            "free_at_admission_bytes",
            "non_expert_bytes",
            "resident_layer_bytes",
            "joint_reserve_bytes",
            "trunk_layers",
            "resident_layers",
            "preferred_resident_layers",
            "needed_bytes",
            "slack_bytes",
            "preferred_deficit_bytes",
        ] {
            for step in [1i64, -1] {
                let bad = tampered(&n9, key, |v| {
                    let n = v.as_i64().unwrap();
                    *v = json!(n + step);
                });
                assert!(!bad.is_empty(), "{key} {step:+} must fail");
            }
        }
        // A forged zero preferred deficit, a forged positive initialization
        // loss and a forged fresh-free basis all fail.
        assert!(!tampered(&n9, "preferred_deficit_bytes", |v| *v = json!(0)).is_empty());
        assert!(!tampered(&n9, "observed_initialization_bytes", |v| *v = json!(0)).is_empty());
        assert!(!tampered(&n9, "free_before_initialization_bytes", |v| *v =
            json!(PRE_INIT + 1))
        .is_empty());
        assert!(!tampered(&n9, "memory_basis", |v| *v = json!("pre_load")).is_empty());
        assert!(!tampered(&n9, "initialization_charged_via_free", |v| *v =
            json!(false))
        .is_empty());
        assert!(!tampered(&n9, "initialization_charged_via_free", |v| *v =
            json!("true"))
        .is_empty());
        // A witness that claims N10 against a nine-layer quote's free fails.
        let mut forged = admission_witness_json(&n9);
        forged["resident_layers"] = json!(10);
        forged["needed_bytes"] = json!(u64::try_from(base_needed(10)).unwrap());
        forged["slack_bytes"] = json!(0);
        assert!(!admission_witness_problems(Some(&forged), &actual_of(&n9)).is_empty());
    }

    #[test]
    fn slowlink_receipt_manifest_reference_count_never_masquerades_as_the_actual_placement() {
        let m = manifest();
        assert_eq!(
            manifest_profile_u64(&m, PROFILE_PRODUCT, "resident_expert_layers"),
            Some(10)
        );
        // An actual nine-layer admission is valid against a ten-layer reference.
        assert!(
            profile_load_problems(&m, PROFILE_PRODUCT, "product", Some(768), Some(16384))
                .is_empty()
        );
        let payload = LAYER_COST - 4096;
        let acc = placement_accounting(&m, PROFILE_PRODUCT, 9, 48, Some(payload));
        assert_eq!(acc["reference"]["role"], "reference");
        assert_eq!(acc["reference"]["resident_expert_layers"], 10);
        assert_eq!(acc["actual"]["resident_layers"], 9);
        assert_eq!(acc["actual"]["host_layers"], 39);
        assert_eq!(
            acc["actual"]["weights_delta_vs_reference_bytes"],
            -(payload as i64)
        );
        assert_eq!(acc["actual"]["device_weights_bytes"], 20 * GIB - payload);
        let ref_owned = manifest_profile_u64(&m, PROFILE_PRODUCT, "device_owned_bytes").unwrap();
        assert_eq!(acc["actual"]["device_owned_bytes"], ref_owned - payload);
        assert!(acc["actual"]["error"].is_null());
        assert_eq!(acc["reference"]["device_weights_bytes"], 20 * GIB);
        // Equal counts move nothing; a larger actual count moves up.
        let same = placement_accounting(&m, PROFILE_PRODUCT, 10, 48, Some(payload));
        assert_eq!(same["actual"]["device_weights_bytes"], 20 * GIB);
        let up = placement_accounting(&m, PROFILE_PRODUCT, 11, 48, Some(payload));
        assert_eq!(up["actual"]["device_weights_bytes"], 20 * GIB + payload);
        // An unknown layer payload invents no number.
        let unknown = placement_accounting(&m, PROFILE_PRODUCT, 9, 48, None);
        assert!(unknown["actual"]["device_weights_bytes"].is_null());
        assert!(unknown["actual"]["error"].is_string());
        // The memory ledger moves to the actual placement and still sums.
        let mut with_root = manifest();
        with_root["memory_ledger"]["census_basis"] = json!("basis");
        with_root["memory_ledger"]["owned_host_bytes"] = json!(80 * GIB);
        let ledger = memory_ledger(&with_root, PROFILE_PRIMITIVE).unwrap();
        let moved = ledger.at_actual_placement(-(payload as i128)).unwrap();
        assert_eq!(
            moved.owned_device_bytes,
            ledger.owned_device_bytes - payload
        );
        assert_eq!(
            moved.decomposition.weights,
            ledger.decomposition.weights - payload
        );
        assert_eq!(
            moved.decomposition.committed_sum(),
            Some(moved.owned_device_bytes)
        );
        assert!(moved.census_basis.contains("ACTUAL"));
        assert!(ledger
            .at_actual_placement(-(ledger.owned_device_bytes as i128) - 1)
            .is_err());
    }

    fn temp_args(dir: &Path, mode: &str, route: &str) -> Args {
        let arg = |s: &str| s.to_string();
        std::fs::write(dir.join("fixture.json"), b"{\"fixture\":1}\n").unwrap();
        std::fs::write(dir.join("manifest.json"), b"{\"manifest\":1}\n").unwrap();
        parse_args([
            arg("--model"),
            arg("/m.mq4"),
            arg("--fixture"),
            dir.join("fixture.json").to_string_lossy().into_owned(),
            arg("--manifest"),
            dir.join("manifest.json").to_string_lossy().into_owned(),
            arg("--output"),
            dir.join("out.json").to_string_lossy().into_owned(),
            arg("--mode"),
            arg(mode),
            arg("--route"),
            arg(route),
        ])
        .unwrap()
    }

    /// Serialization only: the real `final_report` over temp files and pure
    /// `Checks`; no GPU and no hardware result.
    #[test]
    fn slowlink_receipt_final_report_serializes_the_handoff_contract() {
        let dir = scratch_dir("final");
        for (mode, route, purpose, tile) in [
            (
                "primitives",
                "layer-major",
                AdmissionPurpose::PrimitiveSelftest,
                512u64,
            ),
            ("state", "layer-major", AdmissionPurpose::Product, 768),
        ] {
            let args = temp_args(&dir, mode, route);
            let (log, _lines) = memory_log(None);
            let mut checks = Checks::new(&["identity.manifest", "progress.durability"]);
            checks.pass("identity.manifest", Value::Null);
            checks.pass("progress.durability", Value::Null);
            let mut evidence = serde_json::Map::new();
            evidence.insert(
                "s1_profile".into(),
                s1_profile_evidence(
                    purpose,
                    &manifest(),
                    Some((purpose_name(purpose), tile)),
                    None,
                ),
            );
            let (status, bytes) = final_report(&args, &checks, &evidence, &log).unwrap();
            assert_eq!(status, "pass");
            let report: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(report["schema"], "hipfire.qwen4.slowlink_gate.v1");
            assert_eq!(report["mode"], mode);
            assert_eq!(report["route"], "layer-major");
            assert_eq!(report["status"], "pass");
            for key in ["fixture_sha256", "manifest_sha256"] {
                let hex = report[key].as_str().unwrap();
                assert_eq!(hex.len(), 64, "{key}");
                assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));
            }
            assert_eq!(
                report["fixture_sha256"],
                file_sha256(&dir.join("fixture.json")).unwrap()
            );
            assert_eq!(report["checks"]["status"], "pass");
            assert_eq!(report["checks"]["missing"], json!([]));
            assert_eq!(report["progress"]["schema"], PROGRESS_SCHEMA);
            assert!(report["progress"]["sidecar"]
                .as_str()
                .unwrap()
                .ends_with("out.json.progress.jsonl"));
            let profile = &report["evidence"]["s1_profile"];
            assert_eq!(profile["admitted_purpose"], purpose_name(purpose));
            assert_eq!(profile["admitted_tile_rows"], tile);
            assert_eq!(profile["expected_tile_rows"], tile);
        }
        // An incomplete run (a required check never ran) is never `pass`, and a
        // run whose load never admitted anything reports null admitted fields.
        let args = temp_args(&dir, "state", "layer-major");
        let (log, _lines) = memory_log(None);
        let mut checks = Checks::new(&["identity.manifest", "load.placement"]);
        checks.pass("identity.manifest", Value::Null);
        let mut evidence = serde_json::Map::new();
        evidence.insert(
            "s1_profile".into(),
            s1_profile_evidence(AdmissionPurpose::Product, &manifest(), None, None),
        );
        let (status, bytes) = final_report(&args, &checks, &evidence, &log).unwrap();
        assert_eq!(status, "incomplete");
        let report: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report["status"], "incomplete");
        assert_eq!(report["checks"]["missing"], json!(["load.placement"]));
        assert!(report["evidence"]["s1_profile"]["admitted_purpose"].is_null());
        assert!(report["evidence"]["s1_profile"]["admitted_tile_rows"].is_null());
        assert_eq!(report["evidence"]["s1_profile"]["expected_tile_rows"], 768);
        // A failed check is `fail`.
        let mut failed = Checks::new(&["load.placement"]);
        failed.fail("load.placement", "declined", Value::Null);
        let (status, _) = final_report(&args, &failed, &serde_json::Map::new(), &log).unwrap();
        assert_eq!(status, "fail");
        // An unreadable fixture is an error, never a report with invented hashes.
        std::fs::remove_file(dir.join("fixture.json")).unwrap();
        assert!(final_report(&args, &checks, &evidence, &log).is_err());
        // The aborted report carries no checks, evidence or hashes.
        let aborted = aborted_report(&args, "load failed: hipMalloc", Some(&log));
        assert_eq!(aborted["status"], "incomplete");
        assert_eq!(aborted["mode"], "state");
        assert_eq!(aborted["route"], "layer-major");
        for key in ["checks", "evidence", "fixture_sha256", "manifest_sha256"] {
            assert!(aborted.get(key).is_none(), "{key}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
