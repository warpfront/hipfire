// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure, GPU-free validation shared by the S1 slow-link gate
//! (`examples/qwen4_slowlink_gate.rs`), `tests/slowlink_prefill_hw.rs` and
//! the paired KLD verdict of `examples/qwen4_kld.rs`. Included with
//! `#[path = ...] mod slowlink_receipt;`.
//!
//! Everything fails closed: a missing receipt, a missing key, a partial
//! corpus, an identity mismatch or an empty required-check list is never a
//! pass. The route receipt schema is `V-RECEIPT-API.md` of the S1 plan
//! directory (`Qwen4Bundle::slowlink_receipt`).

#![allow(dead_code)]

use serde_json::{json, Value};

pub const RECEIPT_SCHEMA: &str = "hipfire.qwen4.slowlink_receipt.v1";
pub const FIXTURE_SCHEMA: &str = "hipfire.qwen4.slowlink_fixture.v1";

/// Main's rulings: a mixer tile of at most 1,536 rows, one payload sweep per
/// window. Residency is ADAPTIVE: the loader keeps the largest number of
/// resident layers that actually fits beside the 512 MiB safety margin and the
/// 1 GiB runtime budget. There is no resident-layer floor and no host-layer
/// cap; ten resident layers is only the PREFERRED count (information, never a
/// gate) and the maximal-fits proof is [`admission_witness_problems`].
pub const PREFERRED_RESIDENT_LAYERS: u64 = 10;
/// Trunk layers of the MQ4 fixture model: model data used to tie
/// `resident_layers + host_layers`, not a placement rule.
pub const TRUNK_LAYERS: u64 = 48;
pub const MAX_TILE_ROWS: u64 = 1536;
/// Upper bound of the expert payload bytes one host layer of the MQ4 fixture
/// holds (50,803,507,200 B over 38 host layers when ten stay resident).
pub const COLD_BYTES_PER_HOST_LAYER: u64 = 1_336_934_400;
/// Two 32-expert slots plus one 512-expert lookahead store of MQ4 payload
/// (~1.401 GiB); the 159 MiB ring alone is not the charged reserve.
pub const MIN_RING_LOOKAHEAD_BYTES: u64 = 1_503_238_553;
/// Rows of the one-window target fixture.
pub const TARGET_ROWS: u64 = 16_384;

pub const FAILURE_POINTS: [&str; 4] =
    ["before_dma", "during_panel", "after_layer", "before_commit"];
pub const REQUIRED_PROMPT_ROWS: [usize; 5] = [512, 1536, 1537, 8192, 16384];
pub const PRIMITIVE_CASES: [&str; 7] = [
    "payload_table_identity",
    "repeated_epoch_same_address",
    "delayed_reader_reuse",
    "within_expert_panel_canary",
    "canonical_combine_exact",
    "k1_tail_cohort",
    "zero_row_panel",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    Incumbent,
    LayerMajor,
}

impl Expect {
    pub fn name(self) -> &'static str {
        match self {
            Self::Incumbent => "incumbent",
            Self::LayerMajor => "layer-major",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Limits {
    /// Trunk layers: a layer-major receipt must report
    /// `resident_layers + host_layers == trunk_layers` with at least one host
    /// layer. Model data, never a resident floor.
    pub trunk_layers: u64,
    pub max_tile_rows: u64,
    /// Upper bound of expert payload bytes per host layer.
    pub cold_bytes_per_host_layer: u64,
    pub min_ring_lookahead_bytes: u64,
    /// The request must have run as exactly one admitted window.
    pub require_single_window: bool,
    /// A timing/overlap claim: cross-layer L compute / L+1 H2D intersections
    /// must be observed.
    pub require_overlap: bool,
    /// The admitted profile's exact tile (from the sealed manifest, equal to
    /// the load purpose's tile). `None` keeps only the generic upper bound
    /// `max_tile_rows`, for historical and non-purpose-bound receipts.
    pub expected_tile_rows: Option<u64>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            trunk_layers: TRUNK_LAYERS,
            max_tile_rows: MAX_TILE_ROWS,
            cold_bytes_per_host_layer: COLD_BYTES_PER_HOST_LAYER,
            min_ring_lookahead_bytes: MIN_RING_LOOKAHEAD_BYTES,
            require_single_window: false,
            require_overlap: false,
            expected_tile_rows: None,
        }
    }
}

const U64_KEYS: [&str; 11] = [
    "entry_pos",
    "prompt_rows",
    "tile_rows",
    "retained_scratch_rows",
    "sweeps",
    "resident_layers",
    "host_layers",
    "host_expert_bytes_total",
    "submitted_bytes",
    "useful_bytes",
    "unused_prefetched_bytes",
];

fn get<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(value, |node, key| node.get(key))
}

pub fn u64_at(value: &Value, path: &str) -> Option<u64> {
    get(value, path).and_then(Value::as_u64)
}

pub fn f64_at(value: &Value, path: &str) -> Option<f64> {
    get(value, path).and_then(Value::as_f64)
}

fn structural(receipt: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    if receipt.get("schema").and_then(Value::as_str) != Some(RECEIPT_SCHEMA) {
        bad.push(format!("receipt schema is not {RECEIPT_SCHEMA}"));
    }
    for key in ["route", "selected_by"] {
        if receipt.get(key).and_then(Value::as_str).is_none() {
            bad.push(format!("receipt key {key} missing or not a string"));
        }
    }
    match receipt.get("bypass_reason") {
        Some(Value::Null) | Some(Value::String(_)) => {}
        _ => bad.push("receipt key bypass_reason missing (string or null)".into()),
    }
    for key in U64_KEYS {
        if u64_at(receipt, key).is_none() {
            bad.push(format!(
                "receipt key {key} missing or not an unsigned integer"
            ));
        }
    }
    for key in ["tile", "arena", "ring", "lookahead", "tables", "total"] {
        let path = format!("reserve.{key}");
        if u64_at(receipt, &path).is_none() {
            bad.push(format!(
                "receipt key {path} missing or not an unsigned integer"
            ));
        }
    }
    for key in ["wall", "copy", "compute", "route_barrier", "nonexpert"] {
        let path = format!("spans_ms.{key}");
        match f64_at(receipt, &path) {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            _ => bad.push(format!(
                "receipt key {path} missing or not a finite non-negative number"
            )),
        }
    }
    for key in [
        "cross_layer_intersections",
        "page_full_stalls",
        "copy_starved_stalls",
    ] {
        let path = format!("overlap.{key}");
        if u64_at(receipt, &path).is_none() {
            bad.push(format!(
                "receipt key {path} missing or not an unsigned integer"
            ));
        }
    }
    for key in ["cross_layer_overlap_ms", "copy_idle_ms"] {
        let path = format!("overlap.{key}");
        match f64_at(receipt, &path) {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            _ => bad.push(format!(
                "receipt key {path} missing or not a finite non-negative number"
            )),
        }
    }
    if u64_at(receipt, "peak_vram_bytes").is_none() {
        bad.push("receipt key peak_vram_bytes missing or not an unsigned integer".into());
    }
    for key in ["windows", "launch_cuts"] {
        if receipt.get(key).and_then(Value::as_array).is_none() {
            bad.push(format!("receipt key {key} missing or not an array"));
        }
    }
    bad.extend(sp_problems(receipt));
    bad
}

/// The optional `sp` object of a receipt (present only when the SP short-prompt
/// hook's developer knob was on at load): every problem with its shape. Absent
/// is acceptable: the knob-off receipt has no `sp` key.
///
/// `{knob: true, armed, decline: string|null, decline_reason: string|null,
/// unavailable: string|null, engaged_windows, windows: [{rows, engaged,
/// bypass: string|null, layers, streamed_bytes, streamed_experts, cpu_experts,
/// cpu_slots}]}`. An armed hook has no decline; an unarmed one engaged no
/// window; `engaged_windows` is the count of engaged windows; an incumbent
/// receipt engaged none (SP only ever runs inside a layer-major window).
pub fn sp_problems(receipt: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    let Some(sp) = receipt.get("sp") else {
        return bad;
    };
    let Some(object) = sp.as_object() else {
        return vec!["receipt key sp is not an object".into()];
    };
    if object.get("knob") != Some(&Value::Bool(true)) {
        bad.push("sp.knob must be true: the object exists only with the knob on".into());
    }
    let armed = object.get("armed").and_then(Value::as_bool);
    if armed.is_none() {
        bad.push("sp.armed missing or not a bool".into());
    }
    for key in ["decline", "decline_reason", "unavailable"] {
        match object.get(key) {
            Some(Value::Null) | Some(Value::String(_)) => {}
            _ => bad.push(format!("sp.{key} missing (string or null)")),
        }
    }
    let named = |key: &str| object.get(key).is_some_and(|value| !value.is_null());
    if armed == Some(true) && (named("decline") || named("unavailable")) {
        bad.push("sp is armed but names a decline or an unavailable reason".into());
    }
    if armed == Some(false) && !named("decline") && !named("unavailable") {
        bad.push("sp is not armed but names neither a decline nor an unavailable reason".into());
    }
    let Some(windows) = object.get("windows").and_then(Value::as_array) else {
        bad.push("sp.windows missing or not an array".into());
        return bad;
    };
    let mut engaged = 0u64;
    for (index, window) in windows.iter().enumerate() {
        let flag = window.get("engaged").and_then(Value::as_bool);
        if window.get("rows").and_then(Value::as_u64).is_none() || flag.is_none() {
            bad.push(format!("sp.windows[{index}] has no rows or engaged flag"));
            continue;
        }
        match window.get("bypass") {
            Some(Value::Null) | Some(Value::String(_)) => {}
            _ => bad.push(format!(
                "sp.windows[{index}].bypass missing (string or null)"
            )),
        }
        for key in [
            "layers",
            "streamed_bytes",
            "streamed_experts",
            "cpu_experts",
            "cpu_slots",
        ] {
            if window.get(key).and_then(Value::as_u64).is_none() {
                bad.push(format!(
                    "sp.windows[{index}].{key} missing or not an integer"
                ));
            }
        }
        if flag == Some(true) {
            engaged += 1;
            if armed != Some(true) {
                bad.push(format!("sp.windows[{index}] engaged while sp is not armed"));
            }
            if !window["bypass"].is_null() {
                bad.push(format!("sp.windows[{index}] is engaged but names a bypass"));
            }
            if window.get("layers").and_then(Value::as_u64) == Some(0) {
                bad.push(format!("sp.windows[{index}] is engaged but ran no layer"));
            }
        }
    }
    if object.get("engaged_windows").and_then(Value::as_u64) != Some(engaged) {
        bad.push(format!(
            "sp.engaged_windows is not the {engaged} engaged windows"
        ));
    }
    if engaged > 0 && receipt.get("route").and_then(Value::as_str) != Some("layer-major") {
        bad.push("an SP window cannot run outside a layer-major receipt".into());
    }
    bad
}

/// `(start, rows)` of every admitted window.
fn windows(receipt: &Value) -> Result<Vec<(u64, u64)>, String> {
    receipt["windows"]
        .as_array()
        .ok_or("windows is not an array")?
        .iter()
        .map(|w| {
            Ok((
                w.get("start")
                    .and_then(Value::as_u64)
                    .ok_or("a window has no start")?,
                w.get("rows")
                    .and_then(Value::as_u64)
                    .ok_or("a window has no rows")?,
            ))
        })
        .collect()
}

/// Every reason `receipt` is not an acceptable `expect` route receipt; empty
/// means acceptable. `None` (no receipt at all) is never acceptable.
pub fn validate(receipt: Option<&Value>, expect: Expect, limits: &Limits) -> Vec<String> {
    let Some(r) = receipt else {
        return vec!["no slowlink receipt: the selector verdict is missing".into()];
    };
    let mut bad = structural(r);
    if !bad.is_empty() {
        return bad;
    }
    let route = r["route"].as_str().unwrap_or_default();
    if route != "incumbent" && route != "layer-major" {
        bad.push(format!(
            "route {route:?} is neither incumbent nor layer-major"
        ));
        return bad;
    }
    if !["env0", "env1", "auto"].contains(&r["selected_by"].as_str().unwrap_or_default()) {
        bad.push(format!(
            "selected_by {:?} is not env0|env1|auto",
            r["selected_by"]
        ));
    }
    let prompt_rows = u64_at(r, "prompt_rows").unwrap_or(0);
    match windows(r) {
        Err(e) => bad.push(e),
        Ok(ws) => {
            let mut cursor = 0u64;
            for (index, (start, rows)) in ws.iter().enumerate() {
                if *start != cursor || *rows == 0 {
                    bad.push(format!(
                        "window {index} starts at {start} with {rows} rows; windows must tile the request contiguously"
                    ));
                }
                cursor = start.saturating_add(*rows);
            }
            if cursor != prompt_rows {
                bad.push(format!(
                    "windows cover {cursor} rows, the request has {prompt_rows}"
                ));
            }
            if limits.require_single_window && ws.len() != 1 {
                bad.push(format!("{} windows admitted, one required", ws.len()));
            }
            if route == "layer-major" && u64_at(r, "sweeps") != Some(ws.len() as u64) {
                bad.push(format!(
                    "sweeps {:?} != windows {}: payload must be submitted once per window",
                    u64_at(r, "sweeps"),
                    ws.len()
                ));
            }
        }
    }
    let cuts: Vec<u64> = r["launch_cuts"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    if cuts.len() != r["launch_cuts"].as_array().map_or(0, Vec::len)
        || cuts.windows(2).any(|w| w[0] >= w[1])
        || cuts.last().is_some_and(|&c| c > prompt_rows)
    {
        bad.push("launch_cuts are not strictly ascending row boundaries within the request".into());
    }
    match expect {
        Expect::Incumbent => {
            if route != "incumbent" {
                bad.push(format!("incumbent arm ran route {route}"));
            }
        }
        Expect::LayerMajor => {
            if route != "layer-major" {
                bad.push(format!(
                    "layer-major was forced but the selector ran {route} (bypass_reason {})",
                    r["bypass_reason"]
                ));
            }
        }
    }
    if route == "incumbent" {
        for path in [
            "reserve.total",
            "reserve.ring",
            "reserve.lookahead",
            "overlap.cross_layer_intersections",
            "overlap.page_full_stalls",
            "overlap.copy_starved_stalls",
        ] {
            if u64_at(r, path) != Some(0) {
                bad.push(format!("incumbent route reports nonzero S1 field {path}"));
            }
        }
        return bad;
    }
    // layer-major
    let tile = u64_at(r, "tile_rows").unwrap_or(u64::MAX);
    if tile == 0 || tile > limits.max_tile_rows {
        bad.push(format!(
            "tile_rows {tile} outside 1..={}",
            limits.max_tile_rows
        ));
    }
    // A purpose-bound claim: the receipt must name exactly the admitted
    // profile's tile (and retain no more scratch rows than it); being under
    // the generic upper bound is not enough.
    if let Some(expected) = limits.expected_tile_rows {
        if tile != expected {
            bad.push(format!(
                "tile_rows {tile} != the admitted profile's {expected}-row tile"
            ));
        }
    }
    let scratch = u64_at(r, "retained_scratch_rows").unwrap_or(u64::MAX);
    let scratch_bound = limits.expected_tile_rows.unwrap_or(limits.max_tile_rows);
    if scratch > scratch_bound {
        bad.push(format!(
            "retained_scratch_rows {scratch} > {scratch_bound}: ordinary 8192/full-prompt scratch is retained"
        ));
    }
    let resident = u64_at(r, "resident_layers").unwrap_or(0);
    let host = u64_at(r, "host_layers").unwrap_or(u64::MAX);
    // Placement accounting, not a floor: the two counts must close on the
    // trunk and a host layer must remain (an all-resident load bypasses S1
    // and never claims a layer-major receipt).
    if resident.checked_add(host) != Some(limits.trunk_layers) {
        bad.push(format!(
            "resident_layers {resident} + host_layers {host} != the {} trunk layers",
            limits.trunk_layers
        ));
    }
    if host == 0 {
        bad.push("host_layers 0: no layer streams, so a layer-major route is not claimable".into());
    }
    let host_bytes = u64_at(r, "host_expert_bytes_total").unwrap_or(u64::MAX);
    let submitted = u64_at(r, "submitted_bytes").unwrap_or(u64::MAX);
    let useful = u64_at(r, "useful_bytes").unwrap_or(u64::MAX);
    let sweeps = u64_at(r, "sweeps").unwrap_or(0);
    let host_bound = host.saturating_mul(limits.cold_bytes_per_host_layer);
    if host_bytes > host_bound {
        bad.push(format!(
            "host_expert_bytes_total {host_bytes} > {host_bound} ({host} host layer(s) at most {} B each)",
            limits.cold_bytes_per_host_layer
        ));
    }
    if sweeps == 0 || submitted > host_bytes.saturating_mul(sweeps) {
        bad.push(format!(
            "submitted_bytes {submitted} exceeds {sweeps} sweep(s) of {host_bytes} host bytes"
        ));
    }
    if useful > submitted {
        bad.push(format!(
            "useful_bytes {useful} > submitted_bytes {submitted}"
        ));
    }
    if u64_at(r, "unused_prefetched_bytes").unwrap_or(u64::MAX) > submitted {
        bad.push("unused_prefetched_bytes exceeds submitted_bytes".into());
    }
    let parts: u64 = ["tile", "arena", "ring", "lookahead", "tables"]
        .iter()
        .map(|k| u64_at(r, &format!("reserve.{k}")).unwrap_or(0))
        .sum();
    if u64_at(r, "reserve.total").unwrap_or(0) < parts {
        bad.push("reserve.total is below the sum of its parts".into());
    }
    let ring_look =
        u64_at(r, "reserve.ring").unwrap_or(0) + u64_at(r, "reserve.lookahead").unwrap_or(0);
    if ring_look < limits.min_ring_lookahead_bytes {
        bad.push(format!(
            "ring+lookahead reserve {ring_look} < {}: the cross-layer lookahead store is not charged",
            limits.min_ring_lookahead_bytes
        ));
    }
    if limits.require_overlap
        && (u64_at(r, "overlap.cross_layer_intersections") == Some(0)
            || f64_at(r, "overlap.cross_layer_overlap_ms").unwrap_or(0.0) <= 0.0)
    {
        bad.push(
            "no cross-layer event intersection (copy of layer L+1 overlapping compute of layer L): an overlap win cannot be claimed"
                .into(),
        );
    }
    bad
}

/// A forced layer-major request too short to qualify (at most one tile): the
/// selector may legitimately bypass it, but only explicitly. Accepts a
/// layer-major receipt (full checks) or an incumbent receipt that names its
/// bypass reason.
pub fn validate_forced_short(receipt: Option<&Value>, limits: &Limits) -> Vec<String> {
    match receipt.and_then(|r| r.get("route")).and_then(Value::as_str) {
        Some("layer-major") => validate(receipt, Expect::LayerMajor, limits),
        _ => {
            let mut bad = validate(receipt, Expect::Incumbent, limits);
            let named = receipt
                .and_then(|r| r.get("bypass_reason"))
                .is_some_and(Value::is_string);
            if !named {
                bad.push(
                    "short forced request ran the incumbent without a named bypass_reason".into(),
                );
            }
            bad
        }
    }
}

/// A claim that the schedule overlapped must carry event evidence.
pub fn overlap_claim_allowed(receipt: &Value) -> bool {
    u64_at(receipt, "overlap.cross_layer_intersections").unwrap_or(0) > 0
        && f64_at(receipt, "overlap.cross_layer_overlap_ms").unwrap_or(0.0) > 0.0
}

/// Two arms with different launch manifests or tile sizes are not bitwise
/// transport-only comparisons. `Err` names why a "bitwise transport" claim
/// is refused.
pub fn bitwise_transport_claim(base: &Value, cand: &Value) -> Result<(), String> {
    for key in ["launch_cuts", "windows"] {
        if base.get(key) != cand.get(key) {
            return Err(format!(
                "unequal {key}: not a bitwise transport-only comparison"
            ));
        }
    }
    for key in ["tile_rows", "entry_pos", "prompt_rows"] {
        if u64_at(base, key) != u64_at(cand, key) {
            return Err(format!(
                "unequal {key}: not a bitwise transport-only comparison"
            ));
        }
    }
    if base.get("route") != cand.get("route") {
        return Err("different routes are never a bitwise transport-only comparison".into());
    }
    Ok(())
}

// ── required-check accounting ──────────────────────────────────────────

pub struct Checks {
    required: Vec<String>,
    rows: Vec<Value>,
}

impl Checks {
    pub fn new(required: &[&str]) -> Self {
        Self {
            required: required.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn pass(&mut self, name: &str, detail: Value) {
        eprintln!("qwen4_slowlink_gate: PASS {name}");
        self.rows
            .push(json!({"name": name, "status": "pass", "detail": detail}));
    }

    pub fn fail(&mut self, name: &str, reason: impl Into<String>, detail: Value) {
        let reason = reason.into();
        eprintln!("qwen4_slowlink_gate: FAIL {name}: {reason}");
        self.rows
            .push(json!({"name": name, "status": "fail", "reason": reason, "detail": detail}));
    }

    pub fn skip(&mut self, name: &str, reason: impl Into<String>) {
        let reason = reason.into();
        eprintln!("qwen4_slowlink_gate: SKIP {name}: {reason}");
        self.rows
            .push(json!({"name": name, "status": "skipped", "reason": reason}));
    }

    pub fn push(&mut self, name: &str, problems: Vec<String>, detail: Value) {
        if problems.is_empty() {
            self.pass(name, detail);
        } else {
            self.fail(name, problems.join("; "), detail);
        }
    }

    /// `fail` if any check failed; `incomplete` if no required check was
    /// listed or any required one has no executed (pass/fail) row; else
    /// `pass`.
    pub fn status(&self) -> &'static str {
        let executed = |name: &str| {
            self.rows.iter().any(|row| {
                row["name"] == name && (row["status"] == "pass" || row["status"] == "fail")
            })
        };
        if self.rows.iter().any(|row| row["status"] == "fail") {
            "fail"
        } else if self.required.is_empty() || self.required.iter().any(|n| !executed(n)) {
            "incomplete"
        } else {
            "pass"
        }
    }

    pub fn missing(&self) -> Vec<String> {
        self.required
            .iter()
            .filter(|name| {
                !self.rows.iter().any(|row| {
                    row["name"] == name.as_str()
                        && (row["status"] == "pass" || row["status"] == "fail")
                })
            })
            .cloned()
            .collect()
    }

    /// Every row recorded so far, in order (the durable progress log writes
    /// each exactly once, as it is observed).
    pub fn rows(&self) -> &[Value] {
        &self.rows
    }

    pub fn json(&self) -> Value {
        json!({
            "status": self.status(),
            "required": self.required,
            "missing": self.missing(),
            "checks": self.rows,
        })
    }
}

// ── durable gate progress ──────────────────────────────────────────────
//
// A bounded JSONL sidecar (`<output>.progress.jsonl`) holds the evidence a
// SIGTERM, timeout or crash would otherwise lose: each line is one record,
// written with a single `write_all` and `sync_data` the moment it is
// observed. The sidecar is evidence only: it can never be promoted. Only a
// complete final report (`write_atomic`) carries a verdict, and a summary
// of a partial sidecar is `incomplete` (or `fail` if a failure was already
// observed), never `pass`; a required check that never ran stays listed as
// missing rather than being filled in.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub const PROGRESS_SCHEMA: &str = "hipfire.qwen4.slowlink_gate.progress.v1";
pub const PROGRESS_SUMMARY_SCHEMA: &str = "hipfire.qwen4.slowlink_gate.progress_summary.v1";
pub const PROGRESS_SUFFIX: &str = ".progress.jsonl";
/// Where each record's times come from, recorded in `run_start` so the log can
/// be aligned with the sampler and the untimestamped stderr stream.
pub const PROGRESS_TIME_SOURCES: &str = "utc/unix_ms: SystemTime::now (CLOCK_REALTIME) when the record is built; mono_ns: std::time::Instant (CLOCK_MONOTONIC on Linux) elapsed since the log was opened; every stderr progress echo prints the same utc and mono_ns; the background sampler has no timestamps, its running maxima are embedded in a record and take that record's times";

/// The sidecar next to `output`: `<output file name>.progress.jsonl`.
pub fn progress_path(output: &Path) -> PathBuf {
    let mut name = output
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(PROGRESS_SUFFIX);
    output.with_file_name(name)
}

/// `unix_ms` as `YYYY-MM-DDTHH:MM:SS.mmmZ` (proleptic Gregorian, UTC).
pub fn utc_rfc3339(unix_ms: u64) -> String {
    let (secs, ms) = (unix_ms / 1000, unix_ms % 1000);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Civil-from-days (H. Hinnant): shift the epoch to 0000-03-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// `(unix_ms, mono_ns)`: realtime for UTC, monotonic since the log opened.
pub type ProgressClock = Box<dyn FnMut() -> (u64, u64)>;

pub fn system_clock() -> ProgressClock {
    let opened = std::time::Instant::now();
    Box::new(move || {
        let unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        (unix_ms, opened.elapsed().as_nanos() as u64)
    })
}

/// One complete line (with its `\n`) appended durably, or an error.
pub trait ProgressSink {
    fn append(&mut self, line: &str) -> std::io::Result<()>;
}

struct FileSink(std::fs::File);

impl ProgressSink for FileSink {
    fn append(&mut self, line: &str) -> std::io::Result<()> {
        self.0.write_all(line.as_bytes())?;
        self.0.sync_data()
    }
}

/// fsync the directory holding `path` so a new/renamed entry is durable.
fn sync_parent(path: &Path) -> std::io::Result<()> {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => std::fs::File::open(dir)?.sync_all(),
        _ => Ok(()),
    }
}

struct ProgressInner {
    sink: Box<dyn ProgressSink>,
    clock: ProgressClock,
    run_id: String,
    seq: u64,
    error: Option<String>,
    flushed_rows: usize,
    echo: bool,
}

/// Single-threaded shared handle to the sidecar (the gate's one process-local
/// log; the background sampler never touches it). The FIRST write error is
/// kept, printed and is sticky: later records are not attempted, and the gate
/// turns it into the failed required check `progress.durability`.
#[derive(Clone)]
pub struct ProgressLog(Rc<RefCell<ProgressInner>>);

impl ProgressLog {
    pub fn new(
        sink: Box<dyn ProgressSink>,
        clock: ProgressClock,
        run_id: String,
        echo: bool,
    ) -> Self {
        Self(Rc::new(RefCell::new(ProgressInner {
            sink,
            clock,
            run_id,
            seq: 0,
            error: None,
            flushed_rows: 0,
            echo,
        })))
    }

    /// Open (append, never truncate) the sidecar at `path`. A torn last line
    /// left by an earlier killed run is sealed with a newline first so it can
    /// never swallow this run's first record.
    pub fn open_file(path: &Path, run_id: String, echo: bool) -> Result<Self, String> {
        use std::io::{Read, Seek, SeekFrom};
        let fail = |what: &str, e: std::io::Error| format!("{what} {}: {e}", path.display());
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| fail("open progress sidecar", e))?;
        let len = file
            .metadata()
            .map_err(|e| fail("stat progress sidecar", e))?
            .len();
        if len > 0 {
            let mut last = [0u8; 1];
            file.seek(SeekFrom::End(-1))
                .and_then(|_| file.read_exact(&mut last))
                .map_err(|e| fail("read progress sidecar tail", e))?;
            if last[0] != b'\n' {
                file.write_all(b"\n")
                    .map_err(|e| fail("seal torn progress sidecar line", e))?;
            }
        }
        file.sync_all()
            .and_then(|()| sync_parent(path))
            .map_err(|e| fail("sync progress sidecar", e))?;
        Ok(Self::new(
            Box::new(FileSink(file)),
            system_clock(),
            run_id,
            echo,
        ))
    }

    /// The current `(unix_ms, mono_ns)` of this log's clock.
    pub fn now(&self) -> (u64, u64) {
        (self.0.borrow_mut().clock)()
    }

    /// A `(unix_ms, mono_ns)` as the `{utc, unix_ms, mono_ns}` JSON stamp.
    pub fn stamp((unix_ms, mono_ns): (u64, u64)) -> Value {
        json!({"utc": utc_rfc3339(unix_ms), "unix_ms": unix_ms, "mono_ns": mono_ns})
    }

    /// Append one record `{schema, run_id, seq, event, utc, unix_ms, mono_ns,
    /// ..fields}` durably. On a write error the log goes sticky-failed.
    pub fn record(&self, event: &str, fields: Value) {
        let mut inner = self.0.borrow_mut();
        if inner.error.is_some() {
            return;
        }
        let (unix_ms, mono_ns) = (inner.clock)();
        let mut object = match fields {
            Value::Object(map) => map,
            Value::Null => serde_json::Map::new(),
            other => {
                let mut map = serde_json::Map::new();
                map.insert("value".into(), other);
                map
            }
        };
        let utc = utc_rfc3339(unix_ms);
        let seq = inner.seq;
        object.insert("schema".into(), json!(PROGRESS_SCHEMA));
        object.insert("run_id".into(), json!(inner.run_id));
        object.insert("seq".into(), json!(seq));
        object.insert("event".into(), json!(event));
        object.insert("utc".into(), json!(utc));
        object.insert("unix_ms".into(), json!(unix_ms));
        object.insert("mono_ns".into(), json!(mono_ns));
        let written = serde_json::to_string(&Value::Object(object))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            .and_then(|mut line| {
                line.push('\n');
                inner.sink.append(&line)
            });
        match written {
            Ok(()) => {
                inner.seq += 1;
                if inner.echo {
                    eprintln!(
                        "qwen4_slowlink_gate: progress #{seq} {event} utc={utc} mono_ns={mono_ns}"
                    );
                }
            }
            Err(e) => {
                let message =
                    format!("progress record #{seq} ({event}) was not durably written: {e}");
                eprintln!("qwen4_slowlink_gate: FAIL progress.durability: {message}");
                inner.error = Some(message);
            }
        }
    }

    /// Write every `checks` row not yet written, as a `check` record each.
    pub fn checks(&self, checks: &Checks) {
        let start = self.0.borrow().flushed_rows;
        for row in checks.rows().iter().skip(start) {
            self.record("check", row.clone());
        }
        self.0.borrow_mut().flushed_rows = checks.rows().len();
    }

    /// The first write error, if any.
    pub fn error(&self) -> Option<String> {
        self.0.borrow().error.clone()
    }

    /// Records durably written so far.
    pub fn records(&self) -> u64 {
        self.0.borrow().seq
    }

    /// Problems for the required `progress.durability` check.
    pub fn durability_problems(&self) -> Vec<String> {
        self.error().into_iter().collect()
    }
}

/// Write `bytes` to `path` atomically and durably: temporary sibling, fsync,
/// rename, fsync of the directory. A failure removes the temporary file and
/// leaves `path` as it was; the caller must treat it as a gate failure.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(format!(".tmp-{}", std::process::id()));
    let temporary = path.with_file_name(name);
    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temporary, path)?;
        sync_parent(path)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("write {}: {e}", path.display()));
    }
    Ok(())
}

#[derive(Default)]
struct CaseProgress {
    completed: u64,
    failed: u64,
    in_flight: Option<u64>,
    last_completed: Value,
    last_error: Value,
}

/// What a (possibly truncated, crashed or corrupted) sidecar proves. Only the
/// LAST run in the file is summarized. A torn final line (no newline) is
/// dropped and flagged; a complete line that is not a valid record is
/// `corrupt_lines` and is never interpreted. The status is `fail` only when a
/// failed check row was durably recorded and otherwise `incomplete`: the
/// sidecar can never say `pass`, and the verdict of a finished run is the
/// atomically written report alone (`final_recorded` only names its digest).
pub fn progress_summary(text: &str) -> Value {
    let mut segments: Vec<&str> = text.split('\n').collect();
    let torn_tail = segments.pop().filter(|tail| !tail.is_empty()).is_some();
    let mut records: Vec<Value> = Vec::new();
    let mut corrupt_lines: Vec<usize> = Vec::new();
    for (index, line) in segments.iter().enumerate() {
        match serde_json::from_str::<Value>(line) {
            Ok(value)
                if value["schema"] == PROGRESS_SCHEMA
                    && value["seq"].is_u64()
                    && value["event"].is_string()
                    && value["run_id"].is_string() =>
            {
                records.push(value)
            }
            _ => corrupt_lines.push(index + 1),
        }
    }
    let run_id = records
        .iter()
        .rev()
        .find(|r| r["event"] == "run_start")
        .or_else(|| records.last())
        .map(|r| r["run_id"].clone())
        .unwrap_or(Value::Null);
    let run: Vec<&Value> = records.iter().filter(|r| r["run_id"] == run_id).collect();
    let run_start_seen = run.iter().any(|r| r["event"] == "run_start");
    let seq_contiguous = run
        .iter()
        .enumerate()
        .all(|(i, r)| r["seq"].as_u64() == Some(i as u64));
    let mut cases: BTreeMap<String, CaseProgress> = BTreeMap::new();
    let mut executed: BTreeMap<String, bool> = BTreeMap::new(); // name -> failed
    let mut required: Vec<String> = Vec::new();
    let mut memory_phases: Vec<String> = Vec::new();
    let mut error = Value::Null;
    let mut final_event = Value::Null;
    for record in &run {
        let case_key = record["case"].as_str().map(str::to_string);
        match record["event"].as_str().unwrap_or("") {
            "plan" => {
                required = record["required"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|n| n.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
            }
            "repeat_start" => {
                if let Some(key) = case_key {
                    cases.entry(key).or_default().in_flight = record["repeat"].as_u64();
                }
            }
            "repeat_end" => {
                if let Some(key) = case_key {
                    let case = cases.entry(key).or_default();
                    if case.in_flight == record["repeat"].as_u64() {
                        case.in_flight = None;
                    }
                    if record["status"] == "ok" {
                        case.completed += 1;
                        case.last_completed = json!({
                            "repeat": record["repeat"], "wall_s": record["wall_s"],
                            "state_digest": record["state_digest"],
                            "logits_digest": record["logits_digest"],
                            "utc": record["utc"], "mono_ns": record["mono_ns"],
                        });
                    } else {
                        case.failed += 1;
                        case.last_error =
                            json!({"repeat": record["repeat"], "error": record["error"]});
                    }
                }
            }
            "check" => {
                if let (Some(name), Some(status)) =
                    (record["name"].as_str(), record["status"].as_str())
                {
                    if status == "pass" || status == "fail" {
                        let failed = executed.entry(name.to_string()).or_insert(false);
                        *failed |= status == "fail";
                    }
                }
            }
            "memory" => {
                if let Some(phase) = record["phase"].as_str() {
                    memory_phases.push(phase.to_string());
                }
            }
            "abort" | "load_error" => error = record["error"].clone(),
            "final" => {
                final_event = json!({
                    "status": record["status"], "report_sha256": record["report_sha256"],
                    "utc": record["utc"],
                })
            }
            _ => {}
        }
    }
    let failed_checks: Vec<&String> = executed
        .iter()
        .filter(|(_, failed)| **failed)
        .map(|(name, _)| name)
        .collect();
    let missing_required: Vec<&String> = required
        .iter()
        .filter(|name| !executed.contains_key(*name))
        .collect();
    let cases_json: serde_json::Map<String, Value> = cases
        .into_iter()
        .map(|(key, case)| {
            (
                key,
                json!({
                    "repeats_completed": case.completed, "repeats_failed": case.failed,
                    "in_flight_repeat": case.in_flight,
                    "last_completed": case.last_completed, "last_error": case.last_error,
                }),
            )
        })
        .collect();
    json!({
        "schema": PROGRESS_SUMMARY_SCHEMA,
        "status": if failed_checks.is_empty() { "incomplete" } else { "fail" },
        "promotable": false,
        "run_id": run_id,
        "records": run.len(),
        "run_start_seen": run_start_seen,
        "seq_contiguous": seq_contiguous,
        "corrupt_lines": corrupt_lines,
        "torn_tail": torn_tail,
        "last_event": run.last().map_or(Value::Null, |r| r["event"].clone()),
        "last_utc": run.last().map_or(Value::Null, |r| r["utc"].clone()),
        "last_mono_ns": run.last().map_or(Value::Null, |r| r["mono_ns"].clone()),
        "cases": cases_json,
        "memory_phases": memory_phases,
        "failed_checks": failed_checks,
        "missing_required": missing_required,
        "error": error,
        "final_recorded": final_event,
    })
}

// ── fixture ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct FixturePrompt {
    pub id: String,
    pub kind: String,
    pub prompt_ids: Vec<u32>,
    /// Natural-text prompts: tokenized by the gate after the load
    /// (`prompt_ids` is empty until then).
    pub text: Option<String>,
    /// Resume: the first `entry` ids are a previously prefilled prefix.
    pub entry: usize,
    /// Original launch cuts (row indices) the prompt is driven through.
    pub launch_cuts: Vec<usize>,
    /// Session snapshot boundaries the fixture requests.
    pub requested_boundaries: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub max_seq: usize,
    pub decode_tokens: usize,
    pub repeats: usize,
    pub head_rows: usize,
    /// Session-cache budget of the `state` mode's bundle (bytes).
    pub session_cache_bytes: u64,
    /// Timed prefills per prompt in `perf` mode (default 1).
    pub perf_repeats: usize,
    /// Row-prompt lengths `perf` mode times.
    pub perf_rows: Vec<usize>,
    pub prompts: Vec<FixturePrompt>,
    pub continuation_ids: Vec<u32>,
    pub failure_injections: Vec<String>,
    /// Marker id (`boundary_token_id`) at row `cut - 1` of every requested
    /// boundary of the boundary prompt and nowhere else.
    pub boundary_token_id: u32,
}

fn usize_list(value: Option<&Value>, what: &str) -> Result<Vec<usize>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("{what}: not an unsigned integer list"))
            })
            .collect(),
        Some(_) => Err(format!("{what}: not a list")),
    }
}

fn u32_list(value: Option<&Value>, what: &str) -> Result<Vec<u32>, String> {
    usize_list(value, what)?
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| format!("{what}: id {n} exceeds u32")))
        .collect()
}

/// Row-prompt ids are drawn from `LCG_ID_MIN..LCG_ID_MAX`; the boundary
/// marker id must lie outside it so it occurs only where the fixture put it.
pub const LCG_ID_MIN: u32 = 1000;
pub const LCG_ID_MAX: u32 = 100_000;

/// Problems with a boundary prompt's marker ids: every requested cut is an
/// interior row boundary whose preceding row (`cut - 1`) holds `marker`, and
/// the marker occurs nowhere else (no unrequested snapshot boundary). The
/// caller's ids are checked, never rewritten.
pub fn boundary_fixture_problems(ids: &[u32], cuts: &[usize], marker: u32) -> Vec<String> {
    let mut bad = Vec::new();
    if cuts.is_empty() {
        bad.push("the boundary prompt requests no boundaries".to_string());
    }
    if cuts.windows(2).any(|w| w[0] >= w[1]) {
        bad.push("requested boundaries are not strictly ascending".to_string());
    }
    for &cut in cuts {
        if cut == 0 || cut >= ids.len() {
            bad.push(format!(
                "requested boundary {cut} is outside 1..{}",
                ids.len()
            ));
        } else if ids[cut - 1] != marker {
            bad.push(format!(
                "requested boundary {cut}: row {} holds id {}, not the marker {marker}",
                cut - 1,
                ids[cut - 1]
            ));
        }
    }
    for (row, _) in ids.iter().enumerate().filter(|(_, &id)| id == marker) {
        if !cuts.contains(&(row + 1)) {
            bad.push(format!(
                "marker {marker} at row {row} would add the unrequested boundary {}",
                row + 1
            ));
        }
    }
    bad
}

/// Whole-prompt boundary prefill evidence: a receipt exists and, on the
/// layer-major route, every requested cut is an executed exit (a launch cut
/// or a window edge), since a snapshot at `cut` needs the request to stop
/// there. The incumbent route's exits are not receipted, so only the receipt
/// itself is required of it.
pub fn boundary_capture_problems(
    cuts: &[usize],
    receipt: Option<&Value>,
    layer_major: bool,
) -> Vec<String> {
    let Some(receipt) = receipt else {
        return vec!["no receipt for the whole-prompt boundary prefill".to_string()];
    };
    if !layer_major {
        return Vec::new();
    }
    let mut exits: std::collections::BTreeSet<u64> = receipt["launch_cuts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    for window in receipt["windows"].as_array().into_iter().flatten() {
        if let (Some(start), Some(rows)) = (window["start"].as_u64(), window["rows"].as_u64()) {
            exits.insert(start);
            exits.insert(start.saturating_add(rows));
        }
    }
    cuts.iter()
        .filter(|&&cut| !exits.contains(&(cut as u64)))
        .map(|cut| {
            format!("requested boundary {cut} is not an executed exit (launch cuts / window edges)")
        })
        .collect()
}

/// What the cache planned for one boundary: exactly `cut` rows, from a stored
/// snapshot rather than live state.
pub fn boundary_plan_problems(cut: usize, planned: usize, live: bool) -> Vec<String> {
    let mut bad = Vec::new();
    if planned != cut {
        bad.push(format!(
            "session cache planned {planned} reused rows, not the requested boundary {cut}"
        ));
    }
    if live {
        bad.push("the plan continued live state, not a stored snapshot".to_string());
    }
    bad
}

/// One boundary's restore case: the plan ([`boundary_plan_problems`]) and the
/// restored request, which must have computed only the rows after the cut
/// (`entry_pos == cut`), so a cold recompute or a shallower snapshot never
/// passes as the requested boundary.
pub fn boundary_restore_problems(
    cut: usize,
    planned: usize,
    live: bool,
    receipt: Option<&Value>,
    extended_rows: usize,
) -> Vec<String> {
    let mut bad = boundary_plan_problems(cut, planned, live);
    match receipt {
        None => bad.push("no receipt for the restored request".to_string()),
        Some(r) => {
            let want_rows = extended_rows.saturating_sub(cut) as u64;
            if u64_at(r, "entry_pos") != Some(cut as u64) {
                bad.push(format!(
                    "restored request entry_pos {:?} != {cut}",
                    u64_at(r, "entry_pos")
                ));
            }
            if u64_at(r, "prompt_rows") != Some(want_rows) {
                bad.push(format!(
                    "restored request computed {:?} rows, expected {want_rows}",
                    u64_at(r, "prompt_rows")
                ));
            }
        }
    }
    bad
}

/// Every requested boundary needs an executed case that reused exactly its
/// cut; `reused` is `(cut, reused rows)` per executed case.
pub fn boundary_coverage_problems(requested: &[usize], reused: &[(usize, usize)]) -> Vec<String> {
    requested
        .iter()
        .filter_map(|&cut| match reused.iter().find(|(c, _)| *c == cut) {
            None => Some(format!("requested boundary {cut}: case never executed")),
            Some(&(_, rows)) if rows != cut => Some(format!(
                "requested boundary {cut}: reused {rows} rows, not {cut}"
            )),
            Some(_) => None,
        })
        .collect()
}

/// The 16K natural-cadence boundary case: ONE whole prefill of the 16384-row
/// prompt plus `BOUNDARY16K_TAIL_ROWS` continuation rows, under the unchanged
/// 8192-row snapshot cadence (no marker policy), must leave strict pool
/// snapshots at BOTH cuts.
pub const BOUNDARY16K_CUTS: [usize; 2] = [8192, 16384];
pub const BOUNDARY16K_TAIL_ROWS: usize = 3;

/// Required check names of the 16K boundary case.
pub fn boundary16k_required() -> Vec<String> {
    let mut names = vec!["state.boundary16k.capture".to_string()];
    names.extend(
        BOUNDARY16K_CUTS
            .iter()
            .map(|cut| format!("state.boundary16k.{cut}.restore")),
    );
    names
}

/// The 16K case's whole prompt: the 16384-row prompt ids as given plus the
/// first `BOUNDARY16K_TAIL_ROWS` continuation ids (so 16384 is an interior
/// boundary), refused when it does not fit `max_seq` with decode.
pub fn boundary16k_prompt(
    rows: &[u32],
    continuation: &[u32],
    max_seq: usize,
    decode: usize,
) -> Result<Vec<u32>, String> {
    if rows.len() != 16384 {
        return Err(format!("the 16K prompt has {} rows, not 16384", rows.len()));
    }
    if continuation.len() < BOUNDARY16K_TAIL_ROWS {
        return Err(format!(
            "continuation has {} ids, the 16K case needs {BOUNDARY16K_TAIL_ROWS}",
            continuation.len()
        ));
    }
    let mut ids = rows.to_vec();
    ids.extend_from_slice(&continuation[..BOUNDARY16K_TAIL_ROWS]);
    if ids.len() + decode + 1 > max_seq {
        return Err(format!(
            "{} rows + decode {decode} do not fit max_seq {max_seq}",
            ids.len()
        ));
    }
    Ok(ids)
}

/// The largest requested cut whose case executed and fully verified
/// (`(cut, verified)` per requested case); `None` if none did.
pub fn largest_verified_cut(cases: &[(usize, bool)]) -> Option<usize> {
    cases
        .iter()
        .filter(|(_, verified)| *verified)
        .map(|(cut, _)| *cut)
        .max()
}

impl Fixture {
    /// Parse and validate a fixture. A fixture that omits a required case
    /// (prompt rows 512/1536/1537/8192/16384, an irregular, a resume and a
    /// boundary prompt, a continuation and every named failure injection) is
    /// refused rather than silently narrowed.
    pub fn parse(value: &Value) -> Result<Self, String> {
        if value.get("schema").and_then(Value::as_str) != Some(FIXTURE_SCHEMA) {
            return Err(format!("fixture schema is not {FIXTURE_SCHEMA}"));
        }
        let usize_key = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("fixture key {key} missing or not an unsigned integer"))
        };
        let max_seq = usize_key("max_seq")?;
        let decode_tokens = usize_key("decode_tokens")?;
        let repeats = usize_key("repeats")?;
        let head_rows = usize_key("head_rows")?;
        if repeats < 2 {
            return Err("fixture repeats must be at least 2 (determinism needs a repeat)".into());
        }
        if decode_tokens == 0 || head_rows == 0 || head_rows > 8 {
            return Err(
                "fixture decode_tokens must be > 0 and head_rows in 1..=8 (bounded head scoring)"
                    .into(),
            );
        }
        let session_cache_bytes = value
            .get("session_cache_bytes")
            .map_or(Some(2u64 << 30), Value::as_u64)
            .ok_or("fixture session_cache_bytes is not an unsigned integer")?;
        // In-process timed prefill repetitions of `perf` mode. The delivery
        // observations are fresh processes, so the default is one.
        let perf_repeats = value
            .get("perf_repeats")
            .map_or(Some(1u64), Value::as_u64)
            .filter(|&n| n >= 1)
            .ok_or("fixture perf_repeats is not a positive integer")?
            as usize;
        // Row prompts `perf` mode times (default pp512/pp8192/pp16384). The
        // beta anchor fixture times only the 16384-row synthetic prompt.
        let perf_rows = match value.get("perf_rows") {
            None | Some(Value::Null) => vec![512, 8192, 16384],
            Some(_) => {
                let rows = usize_list(value.get("perf_rows"), "perf_rows")?;
                if rows.is_empty() {
                    return Err("fixture perf_rows is empty".into());
                }
                rows
            }
        };
        let mut prompts = Vec::new();
        for item in value
            .get("prompts")
            .and_then(Value::as_array)
            .ok_or("fixture has no prompts array")?
        {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .ok_or("a fixture prompt has no id")?
                .to_string();
            let kind = item
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("prompt {id}: no kind"))?
                .to_string();
            if !["rows", "irregular", "resume", "boundary", "natural"].contains(&kind.as_str()) {
                return Err(format!("prompt {id}: unknown kind {kind}"));
            }
            let prompt_ids = u32_list(item.get("prompt_ids"), &format!("prompt {id} prompt_ids"))?;
            // A natural-text prompt carries `text` instead of ids: the gate
            // tokenizes it with the artifact's own tokenizer after the load.
            let text = match item.get("text") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
                Some(_) => return Err(format!("prompt {id}: text must be a non-empty string")),
            };
            if kind != "natural" && text.is_some() {
                return Err(format!("prompt {id}: only a natural prompt may carry text"));
            }
            if prompt_ids.is_empty() == text.is_none() {
                return Err(format!(
                    "prompt {id}: give exactly one of prompt_ids and text"
                ));
            }
            let entry = item.get("entry").and_then(Value::as_u64).unwrap_or(0) as usize;
            let launch_cuts =
                usize_list(item.get("launch_cuts"), &format!("prompt {id} launch_cuts"))?;
            let requested_boundaries = usize_list(
                item.get("requested_boundaries"),
                &format!("prompt {id} requested_boundaries"),
            )?;
            if entry >= prompt_ids.len() && kind == "resume" {
                return Err(format!(
                    "prompt {id}: resume entry {entry} leaves no continuation"
                ));
            }
            if launch_cuts.windows(2).any(|w| w[0] >= w[1])
                || launch_cuts.iter().any(|&c| c == 0 || c >= prompt_ids.len())
            {
                return Err(format!(
                    "prompt {id}: launch_cuts must be strictly ascending interior row indices"
                ));
            }
            if !prompt_ids.is_empty()
                && prompt_ids.len() + value["decode_tokens"].as_u64().unwrap_or(0) as usize + 1
                    > max_seq
            {
                return Err(format!(
                    "prompt {id}: {} rows + decode do not fit max_seq {max_seq}",
                    prompt_ids.len()
                ));
            }
            prompts.push(FixturePrompt {
                id,
                kind,
                prompt_ids,
                text,
                entry,
                launch_cuts,
                requested_boundaries,
            });
        }
        let ids: std::collections::BTreeSet<&str> = prompts.iter().map(|p| p.id.as_str()).collect();
        if ids.len() != prompts.len() {
            return Err("fixture prompt ids are not unique".into());
        }
        for rows in REQUIRED_PROMPT_ROWS {
            if !prompts
                .iter()
                .any(|p| p.kind == "rows" && p.prompt_ids.len() == rows)
            {
                return Err(format!("fixture lacks the required {rows}-row prompt"));
            }
        }
        for kind in ["irregular", "resume", "boundary", "natural"] {
            if !prompts.iter().any(|p| p.kind == kind) {
                return Err(format!("fixture lacks a {kind} prompt"));
            }
        }
        // The explicit message-end marker the boundary prompt carries at every
        // requested cut - 1 (`Qwen4Bundle::set_turn_end_token` makes it a
        // snapshot boundary). Pinned by the fixture, outside the row-prompt id
        // range, never inferred from the artifact's own EOS/`<|im_end|>` id.
        let boundary_token_id = value
            .get("boundary_token_id")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or("fixture boundary_token_id missing or not a u32 (the explicit marker id the boundary prompt carries at every requested cut - 1)")?;
        if (LCG_ID_MIN..LCG_ID_MAX).contains(&boundary_token_id) {
            return Err(format!(
                "fixture boundary_token_id {boundary_token_id} lies inside the row-prompt id range {LCG_ID_MIN}..{LCG_ID_MAX}"
            ));
        }
        for prompt in prompts.iter().filter(|p| p.kind == "boundary") {
            let bad = boundary_fixture_problems(
                &prompt.prompt_ids,
                &prompt.requested_boundaries,
                boundary_token_id,
            );
            if !bad.is_empty() {
                return Err(format!("boundary prompt {}: {}", prompt.id, bad.join("; ")));
            }
        }
        let continuation_ids = u32_list(value.get("continuation_ids"), "continuation_ids")?;
        if continuation_ids.is_empty() {
            return Err("fixture has no continuation_ids (divergent continuation)".into());
        }
        let failure_injections: Vec<String> = value
            .get("failure_injections")
            .and_then(Value::as_array)
            .ok_or("fixture has no failure_injections")?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        for point in FAILURE_POINTS {
            if !failure_injections.iter().any(|p| p == point) {
                return Err(format!("fixture lacks the {point} failure injection"));
            }
        }
        if let Some(bad) = failure_injections
            .iter()
            .find(|p| !FAILURE_POINTS.contains(&p.as_str()))
        {
            return Err(format!("unknown failure injection {bad}"));
        }
        Ok(Self {
            max_seq,
            decode_tokens,
            repeats,
            head_rows,
            session_cache_bytes,
            perf_repeats,
            perf_rows,
            prompts,
            continuation_ids,
            failure_injections,
            boundary_token_id,
        })
    }
}

// ── process memory telemetry ───────────────────────────────────────────
//
// Opaque HIP driver/event memory has no prior measurement: Hold 1's first cell
// measures it. The ledger names two ALLOWANCES (not measurements); the gate
// fails when the observed residual over the OWNED census exceeds them, or when
// any required telemetry is not observable. Nothing here is estimated.

/// Allowed unowned driver/event VRAM residual: exactly 1.5 GiB.
pub const UNOWNED_VRAM_ALLOWANCE_BYTES: u64 = 1_610_612_736;
/// Allowed unowned driver/event host residual: exactly 2 GiB.
pub const UNOWNED_HOST_ALLOWANCE_BYTES: u64 = 2_147_483_648;
/// Period of the background `/proc` sampler.
pub const SAMPLER_PERIOD_MS: u64 = 50;
/// Phase samples every run must contain.
pub const MEMORY_PHASES: [&str; 3] = ["pre_load", "post_load", "end"];
/// Label of the background sampler's running maxima.
pub const SAMPLER_LABEL: &str = "sampler_max";

/// One reading of this process's memory counters (bytes). `None` = the
/// counter was not observable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcCounters {
    /// `/proc/self/status` `VmRSS` (includes reclaimable file pages).
    pub vm_rss: Option<u64>,
    /// `RssAnon`.
    pub rss_anon: Option<u64>,
    /// `VmPin`: pages pinned by the driver or `mlock`.
    pub vm_pin: Option<u64>,
    /// `VmHWM`: the kernel's continuous RSS high-water mark.
    pub vm_hwm: Option<u64>,
    /// `RssFile`: file-backed resident pages (HFQ/PLE page cache mappings).
    /// Reclaimable, so recorded but never judged against an allowance.
    pub rss_file: Option<u64>,
    /// `RssShmem`: shared-memory resident pages; recorded only.
    pub rss_shmem: Option<u64>,
    /// Clean RSS (`Shared_Clean + Private_Clean`, `/proc/self/smaps`) of
    /// REGULAR-FILE mappings only: the page-cache-backed, reclaimable part.
    /// Device (`/dev/kfd*`, `/dev/dri/*`), `/dev/zero`, anonymous, memfd and
    /// deleted mappings are never in it, so pinned KFD buffer objects are not
    /// subtracted.
    pub reclaimable_file_clean: Option<u64>,
    /// `VmRSS - reclaimable_file_clean`, both read in the same sample.
    pub host_rss_unreclaimable: Option<u64>,
    /// This process's amdgpu `drm-memory-vram`, summed over unique DRM clients.
    pub drm_vram: Option<u64>,
    /// This process's amdgpu `drm-memory-gtt` (process GTT), same dedup.
    pub drm_gtt: Option<u64>,
}

impl ProcCounters {
    /// Raise every counter to the larger of the two; an observed value
    /// replaces an unobserved one.
    pub fn merge_max(&mut self, other: &ProcCounters) {
        fn raise(a: &mut Option<u64>, b: Option<u64>) {
            if let Some(b) = b {
                *a = Some(a.map_or(b, |a| a.max(b)));
            }
        }
        raise(&mut self.vm_rss, other.vm_rss);
        raise(&mut self.rss_anon, other.rss_anon);
        raise(&mut self.vm_pin, other.vm_pin);
        raise(&mut self.vm_hwm, other.vm_hwm);
        raise(&mut self.rss_file, other.rss_file);
        raise(&mut self.rss_shmem, other.rss_shmem);
        raise(
            &mut self.reclaimable_file_clean,
            other.reclaimable_file_clean,
        );
        raise(
            &mut self.host_rss_unreclaimable,
            other.host_rss_unreclaimable,
        );
        raise(&mut self.drm_vram, other.drm_vram);
        raise(&mut self.drm_gtt, other.drm_gtt);
    }
}

/// Bytes of `key` (`VmRSS`, ...) in `/proc/<pid>/status` text (`kB` lines).
pub fn status_bytes(text: &str, key: &str) -> Option<u64> {
    let rest = text
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))?;
    let mut parts = rest.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => n.checked_mul(1024),
        None => Some(n),
        Some(_) => None,
    }
}

/// One amdgpu DRM client of a process (`/proc/<pid>/fdinfo/<fd>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrmClient {
    pub id: String,
    pub vram: Option<u64>,
    pub gtt: Option<u64>,
}

fn fdinfo_bytes(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let n: u64 = parts.next()?.parse().ok()?;
    let unit: u64 = match parts.next() {
        None => 1,
        Some("KiB") => 1 << 10,
        Some("MiB") => 1 << 20,
        Some("GiB") => 1 << 30,
        Some(_) => return None,
    };
    n.checked_mul(unit)
}

/// The amdgpu client an fdinfo text describes; `None` for any other file or
/// driver, or a client without an id.
pub fn parse_drm_fdinfo(text: &str) -> Option<DrmClient> {
    let (mut driver, mut id, mut vram, mut gtt) = (None, None, None, None);
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "drm-driver" => driver = Some(value.to_string()),
            "drm-client-id" => id = Some(value.to_string()),
            "drm-memory-vram" => vram = fdinfo_bytes(value),
            "drm-memory-gtt" => gtt = fdinfo_bytes(value),
            _ => {}
        }
    }
    if driver.as_deref() != Some("amdgpu") {
        return None;
    }
    Some(DrmClient { id: id?, vram, gtt })
}

/// `(vram, gtt)` summed over unique client ids (several fds may share one
/// client). `None` when there is no amdgpu client or any client lacks either
/// counter: that is "not observable", never zero.
pub fn drm_totals(clients: &[DrmClient]) -> Option<(u64, u64)> {
    let mut unique = std::collections::BTreeMap::new();
    for client in clients {
        unique.entry(client.id.as_str()).or_insert(client);
    }
    if unique.is_empty() {
        return None;
    }
    let (mut vram, mut gtt) = (0u64, 0u64);
    for client in unique.values() {
        vram = vram.checked_add(client.vram?)?;
        gtt = gtt.checked_add(client.gtt?)?;
    }
    Some((vram, gtt))
}

/// Class of one `/proc/<pid>/smaps` mapping; the same rules as the hold
/// sampler `sample_mem.py` (`classify`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapClass {
    KfdDri,
    Zero,
    Anon,
    File,
}

pub fn classify_mapping(path: &str) -> MapClass {
    if path.starts_with("/dev/kfd") || path.starts_with("/dev/dri") {
        MapClass::KfdDri
    } else if path.starts_with("/dev/zero") {
        MapClass::Zero
    } else if path.is_empty()
        || path.starts_with('[')
        || path.starts_with("/memfd:")
        || path.starts_with("memfd:")
        || path.contains("(deleted)")
    {
        MapClass::Anon
    } else if path.starts_with('/') {
        MapClass::File
    } else {
        MapClass::Anon
    }
}

/// Bytes of clean (`Shared_Clean + Private_Clean`) resident pages of the
/// regular-file mappings in `/proc/<pid>/smaps` text. `None` when the text has
/// no mapping at all (unreadable smaps): not observable, never zero.
pub fn smaps_regular_file_clean_bytes(text: &str) -> Option<u64> {
    let mut class = MapClass::Anon;
    let mut mappings = 0usize;
    let mut clean_kb = 0u64;
    for line in text.lines() {
        let first = line.split_whitespace().next().unwrap_or("");
        let is_header = !first.ends_with(':')
            && first.contains('-')
            && first.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
        if is_header {
            // `addr perms offset dev inode [path]`: the path may hold spaces.
            let mut rest = line;
            for _ in 0..5 {
                rest = rest.trim_start();
                rest = rest.split_once(char::is_whitespace).map_or("", |(_, r)| r);
            }
            class = classify_mapping(rest.trim());
            mappings += 1;
        } else if class == MapClass::File {
            if let Some(value) = line
                .strip_prefix("Shared_Clean:")
                .or_else(|| line.strip_prefix("Private_Clean:"))
            {
                let kb: u64 = value.split_whitespace().next()?.parse().ok()?;
                clean_kb = clean_kb.checked_add(kb)?;
            }
        }
    }
    if mappings == 0 {
        return None;
    }
    clean_kb.checked_mul(1024)
}

/// This process's counters right now: `/proc/self/status`, `/proc/self/fdinfo`
/// and, with `with_smaps`, `/proc/self/smaps` (the clean regular-file RSS).
/// A counter that cannot be read stays `None`.
pub fn read_proc_counters(with_smaps: bool) -> ProcCounters {
    let mut counters = ProcCounters::default();
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        counters.vm_rss = status_bytes(&status, "VmRSS");
        counters.rss_anon = status_bytes(&status, "RssAnon");
        counters.vm_pin = status_bytes(&status, "VmPin");
        counters.vm_hwm = status_bytes(&status, "VmHWM");
        counters.rss_file = status_bytes(&status, "RssFile");
        counters.rss_shmem = status_bytes(&status, "RssShmem");
    }
    if with_smaps {
        if let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") {
            counters.reclaimable_file_clean = smaps_regular_file_clean_bytes(&smaps);
        }
        if let (Some(rss), Some(clean)) = (counters.vm_rss, counters.reclaimable_file_clean) {
            counters.host_rss_unreclaimable = Some(rss.saturating_sub(clean));
        }
    }
    let mut clients = Vec::new();
    if let Ok(dir) = std::fs::read_dir("/proc/self/fdinfo") {
        for entry in dir.flatten() {
            if let Some(client) = std::fs::read_to_string(entry.path())
                .ok()
                .and_then(|text| parse_drm_fdinfo(&text))
            {
                clients.push(client);
            }
        }
    }
    if let Some((vram, gtt)) = drm_totals(&clients) {
        counters.drm_vram = Some(vram);
        counters.drm_gtt = Some(gtt);
    }
    counters
}

/// A labeled sample as JSON: `device` is `hipMemGetInfo` `(free, total)`.
pub fn counters_json(
    label: &str,
    method: &str,
    counters: &ProcCounters,
    device: Option<(u64, u64)>,
) -> Value {
    json!({
        "label": label, "method": method,
        "device_free": device.map(|d| d.0), "device_total": device.map(|d| d.1),
        "vm_rss": counters.vm_rss, "rss_anon": counters.rss_anon, "vm_pin": counters.vm_pin,
        "vm_hwm": counters.vm_hwm, "drm_vram": counters.drm_vram, "drm_gtt": counters.drm_gtt,
        "rss_file": counters.rss_file, "rss_shmem": counters.rss_shmem,
        "reclaimable_regular_file_clean_bytes": counters.reclaimable_file_clean,
        "host_rss_unreclaimable": counters.host_rss_unreclaimable,
    })
}

/// The owned-census basis the manifest declares for the cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryLedger {
    /// COMMITTED device census the process owns by ledger (weights, state,
    /// scratch, S1 reserve, diagnostics). It is a positive integer, not a pool
    /// semantic: the session-cache BUDGET cap (2 GiB fixture) and the pending
    /// ceiling are NOT committed census and are not included. The SessionCache
    /// sits outside the S1 reserve; the first primitive cell runs with an empty
    /// cache, so the residual is valid, and cache occupancy/evictions are logged
    /// per case through `cache_usage`, never folded into this figure.
    pub owned_device_bytes: u64,
    /// Host pinned/anonymous bytes the process owns by ledger.
    pub owned_host_bytes: u64,
    /// How the census was derived (non-empty).
    pub census_basis: String,
    /// Device decomposition of `owned_device_bytes` plus the safety margin.
    pub decomposition: DeviceDecomposition,
}

/// Device census decomposition from the manifest; the four committed parts
/// sum to `owned_device_bytes`, the safety margin is on top of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceDecomposition {
    /// Resident model weights (non-expert + resident experts).
    pub weights: u64,
    /// The S1 tile census (`AutoReserve.census_total`).
    pub s1_tile_census: u64,
    /// S1 extras (lookahead/ring/arena/tables beyond the tile).
    pub s1_extras: u64,
    /// Sequential diagnostic scratch.
    pub sequential_diag: u64,
    /// Unchanged auto-reserve safety margin.
    pub safety_margin: u64,
}

impl DeviceDecomposition {
    /// Checked sum of the four committed parts (weights, tile census, extras,
    /// sequential diagnostics); the safety margin is not part of it.
    pub fn committed_sum(&self) -> Option<u64> {
        self.weights
            .checked_add(self.s1_tile_census)?
            .checked_add(self.s1_extras)?
            .checked_add(self.sequential_diag)
    }

    /// Non-weight bytes the guard allows: tile census + extras + sequential
    /// diagnostics + safety margin + the fixed driver allowance (checked).
    pub fn allowed_non_weight(&self) -> Option<u64> {
        self.s1_tile_census
            .checked_add(self.s1_extras)?
            .checked_add(self.sequential_diag)?
            .checked_add(self.safety_margin)?
            .checked_add(UNOWNED_VRAM_ALLOWANCE_BYTES)
    }
}

/// The owned census of one purpose profile: `memory_ledger.{census_basis,
/// owned_host_bytes}` (shared, root) and the profile's
/// `memory_ledger.profiles.<profile>.{device_owned_bytes,device_weights_bytes,
/// s1_tile_census_bytes,s1_extras_bytes,sequential_diag_bytes,
/// safety_margin_bytes}`; any missing or zero figure refuses, as does a
/// decomposition whose checked four-part sum is not the profile's owned device
/// bytes. The primitive cell reads `primitive_selftest` only.
pub fn memory_ledger(manifest: &Value, profile: &str) -> Result<MemoryLedger, String> {
    let bytes = |path: &str| {
        u64_at(manifest, path).filter(|&n| n > 0).ok_or_else(|| {
            format!("manifest key {path} missing or not a positive integer (owned census bytes)")
        })
    };
    let field = |key: &str| bytes(&profile_path(profile, key));
    let census_basis = get(manifest, "memory_ledger.census_basis")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or(
            "manifest key memory_ledger.census_basis missing (how the owned census was derived)",
        )?
        .to_string();
    let owned_device_bytes = field("device_owned_bytes")?;
    let decomposition = DeviceDecomposition {
        weights: field("device_weights_bytes")?,
        s1_tile_census: field("s1_tile_census_bytes")?,
        s1_extras: field("s1_extras_bytes")?,
        sequential_diag: field("sequential_diag_bytes")?,
        safety_margin: field("safety_margin_bytes")?,
    };
    match decomposition.committed_sum() {
        Some(sum) if sum == owned_device_bytes => {}
        Some(sum) => {
            return Err(format!(
                "memory_ledger profile {profile} decomposition sums to {sum} B (weights + tile census + extras + sequential diag), not device_owned_bytes {owned_device_bytes} B"
            ))
        }
        None => return Err("memory_ledger decomposition sum overflows u64".to_string()),
    }
    if decomposition.allowed_non_weight().is_none() {
        return Err("memory_ledger allowed non-weight bytes overflow u64".to_string());
    }
    Ok(MemoryLedger {
        owned_device_bytes,
        owned_host_bytes: bytes("memory_ledger.owned_host_bytes")?,
        census_basis,
        decomposition,
    })
}

/// Judge the run's memory samples against the owned census: every phase
/// sample, the background sampler's maxima and every required counter must be
/// present, and the unowned residual must not exceed the fixed allowances.
///
/// Device observed = max(process `drm-memory-vram`, `hipMemGetInfo` used
/// minus the `pre_load` used). Host observed = max over the single counters
/// `host_rss_unreclaimable` (`VmRSS` minus the CLEAN RSS of regular-file
/// mappings only, from smaps), `VmPin` and process `drm-memory-gtt` (never
/// summed: pinned pages appear in more than one; clean regular-file pages are
/// the only subtracted class, never the whole `RssFile`, which holds pinned
/// KFD buffer objects). Returns `(problems, memory receipt)`.
pub fn memory_problems(samples: &[Value], ledger: &MemoryLedger) -> (Vec<String>, Value) {
    let mut bad = Vec::new();
    for phase in MEMORY_PHASES {
        if !samples.iter().any(|s| s["label"] == phase) {
            bad.push(format!("memory sample {phase} missing"));
        }
    }
    match samples.iter().find(|s| s["label"] == SAMPLER_LABEL) {
        None => bad.push("background memory sampler result missing".to_string()),
        Some(s) if s["samples"].as_u64().unwrap_or(0) == 0 => {
            bad.push("background memory sampler took no samples".to_string())
        }
        Some(_) => {}
    }
    for sample in samples {
        let label = sample["label"].as_str().unwrap_or("?");
        let mut keys = vec![
            "vm_rss",
            "vm_pin",
            "drm_vram",
            "drm_gtt",
            "reclaimable_regular_file_clean_bytes",
            "host_rss_unreclaimable",
        ];
        if label != SAMPLER_LABEL {
            keys.extend(["device_free", "device_total"]);
        }
        for key in keys {
            if sample[key].as_u64().is_none() {
                bad.push(format!("telemetry {key} not observable at sample {label}"));
            }
        }
    }
    let used = |s: &Value| {
        Some(
            s["device_total"]
                .as_u64()?
                .saturating_sub(s["device_free"].as_u64()?),
        )
    };
    let base = samples
        .iter()
        .find(|s| s["label"] == "pre_load")
        .and_then(used);
    let max_key = |key: &str| samples.iter().filter_map(|s| s[key].as_u64()).max();
    let hip_delta_max = base.and_then(|b| {
        samples
            .iter()
            .filter_map(used)
            .map(|u| u.saturating_sub(b))
            .max()
    });
    let (drm_vram, host_rss, vm_pin, drm_gtt) = (
        max_key("drm_vram"),
        max_key("host_rss_unreclaimable"),
        max_key("vm_pin"),
        max_key("drm_gtt"),
    );
    let device_observed = [drm_vram, hip_delta_max].into_iter().flatten().max();
    let host_observed = [host_rss, vm_pin, drm_gtt].into_iter().flatten().max();
    let vram_residual = device_observed.map(|o| o.saturating_sub(ledger.owned_device_bytes));
    let host_residual = host_observed.map(|o| o.saturating_sub(ledger.owned_host_bytes));
    match vram_residual {
        None => bad.push("no device memory usage was observed".to_string()),
        Some(r) if r > UNOWNED_VRAM_ALLOWANCE_BYTES => bad.push(format!(
            "unowned VRAM residual {r} B exceeds the {UNOWNED_VRAM_ALLOWANCE_BYTES} B allowance"
        )),
        Some(_) => {}
    }
    match host_residual {
        None => bad.push("no host memory usage was observed".to_string()),
        Some(r) if r > UNOWNED_HOST_ALLOWANCE_BYTES => bad.push(format!(
            "unowned host residual {r} B exceeds the {UNOWNED_HOST_ALLOWANCE_BYTES} B allowance"
        )),
        Some(_) => {}
    }
    // Non-weight guard: the S1 tile census replaces the incumbent historical
    // floor only if the OBSERVED non-weight device peak stays within the
    // census, the extras, the diagnostics, the safety margin and the driver
    // allowance. Uses the samples above; no new GPU query.
    let d = &ledger.decomposition;
    let measured_non_weight = device_observed.map(|o| o.saturating_sub(d.weights));
    let allowed_non_weight = d.allowed_non_weight();
    let guard_result = match (measured_non_weight, allowed_non_weight) {
        (_, None) => {
            bad.push("allowed non-weight device bytes overflow u64".to_string());
            "invalid"
        }
        (None, _) => {
            bad.push("device non-weight guard: no device usage was observed".to_string());
            "unobserved"
        }
        (Some(m), Some(a)) if m > a => {
            bad.push(format!(
                "measured non-weight device peak {m} B exceeds the allowed {a} B (tile census + extras + diag + safety + driver allowance): the S1 tile census is not proven to cover the historical floor; blocked, floor restore needed"
            ));
            "fail"
        }
        (Some(_), Some(_)) => "pass",
    };
    bad.sort();
    bad.dedup();
    let receipt = json!({
        "samples": samples,
        "ledger": {
            "owned_device_bytes": ledger.owned_device_bytes,
            "owned_host_bytes": ledger.owned_host_bytes,
            "census_basis": ledger.census_basis,
            "device_weights_bytes": d.weights,
            "s1_tile_census_bytes": d.s1_tile_census,
            "s1_extras_bytes": d.s1_extras,
            "sequential_diag_bytes": d.sequential_diag,
            "safety_margin_bytes": d.safety_margin,
        },
        "device_guard": {
            "components": {
                "device_weights_bytes": d.weights,
                "s1_tile_census_bytes": d.s1_tile_census,
                "s1_extras_bytes": d.s1_extras,
                "sequential_diag_bytes": d.sequential_diag,
                "safety_margin_bytes": d.safety_margin,
                "driver_allowance_bytes": UNOWNED_VRAM_ALLOWANCE_BYTES,
            },
            "observed_max_device_bytes": device_observed,
            "sampling_method": "max over discrete point samples (process drm-memory-vram, hipMemGetInfo used minus pre_load used); a measured lower bound of the peak, not a continuous driver peak",
            "measured_non_weight_peak_bytes": measured_non_weight,
            "allowed_non_weight_bytes": allowed_non_weight,
            "result": guard_result,
            "note": "measured_non_weight_peak = observed max device - weights (saturating at 0); allowed = tile census + extras + sequential diag + safety margin + driver allowance; the stricter unowned-residual check is kept",
        },
        "allowances": {
            "unowned_vram_bytes": UNOWNED_VRAM_ALLOWANCE_BYTES,
            "unowned_host_bytes": UNOWNED_HOST_ALLOWANCE_BYTES,
            "note": "named allowances fixed in the gate (1.5 GiB VRAM, 2 GiB host), not measurements",
        },
        "high_water": {
            "drm_vram": drm_vram, "hipmemgetinfo_used_delta": hip_delta_max,
            "host_rss_unreclaimable": host_rss, "vm_pin": vm_pin, "drm_gtt": drm_gtt,
            "rss_anon": max_key("rss_anon"),
            "reclaimable_regular_file_clean_bytes": max_key("reclaimable_regular_file_clean_bytes"),
            "vm_rss": max_key("vm_rss"), "vm_hwm": max_key("vm_hwm"),
            "rss_file": max_key("rss_file"), "rss_shmem": max_key("rss_shmem"),
            "rss_file_note": "RssFile and the clean regular-file RSS are recorded; only the clean regular-file class is subtracted from VmRSS (pinned KFD buffer-object file maps stay counted); RssShmem and RssAnon are recorded only",
            "sampling": format!("maximum over the labeled phase samples and a background /proc sampler ({SAMPLER_PERIOD_MS} ms period, discrete points, not a continuous high-water); vm_hwm is the kernel's continuous RSS high-water"),
        },
        "observed": {
            "device": device_observed,
            "device_basis": "max(process drm-memory-vram, hipMemGetInfo used minus pre_load used)",
            "host": host_observed,
            "host_basis": "max(VmRSS - clean RSS of regular-file mappings, VmPin, process drm-memory-gtt), one counter each, never summed",
        },
        "residual": {
            "vram_bytes": vram_residual, "host_bytes": host_residual,
            "host_by_counter": {
                "host_rss_unreclaimable": host_rss.map(|o| o.saturating_sub(ledger.owned_host_bytes)),
                "vm_pin": vm_pin.map(|o| o.saturating_sub(ledger.owned_host_bytes)),
                "drm_gtt": drm_gtt.map(|o| o.saturating_sub(ledger.owned_host_bytes)),
            },
        },
    });
    (bad, receipt)
}

// ── per-purpose S1 profiles ────────────────────────────────────────────
//
// One H1 manifest represents BOTH admitted purpose profiles, each a FULL source
// profile ledger: `memory_ledger.profiles.product` and
// `memory_ledger.profiles.primitive_selftest` (`purpose`, `tile_rows`,
// `resident_expert_layers`, `window_rows`, `page_count`, `device_weights_bytes`,
// `s1_tile_census_bytes`/`_terms`, `s1_extras_bytes`/`_terms`,
// `device_core_bytes`, `sequential_diag_bytes`, `device_owned_bytes`,
// `runtime_budget_bytes`, `safety_margin_bytes`). The root keeps only the
// shared card/fabric/host figures and `device_owned_bytes`, the MAX of the
// per-purpose owned bytes (the two purposes load sequentially, so the peak is
// not product core plus the primitive diagnostic). There is no flat-profile
// fallback. The product cells (state, headroom, greedy, perf) read
// `product`; the primitive cell reads `primitive_selftest`, which is kernel
// correctness at its own tile only and no product-shape qualification. The
// generic `MAX_TILE_ROWS` stays the historical upper bound of the default
// [`Limits`]; it can never pass a product receipt on its own.

pub const PROFILE_PRODUCT: &str = "product";
pub const PROFILE_PRIMITIVE: &str = "primitive_selftest";

fn profile_path(profile: &str, key: &str) -> String {
    format!("memory_ledger.profiles.{profile}.{key}")
}

/// One u64 field of a sealed purpose profile, if present.
pub fn manifest_profile_u64(manifest: &Value, profile: &str, key: &str) -> Option<u64> {
    u64_at(manifest, &profile_path(profile, key))
}

/// The sealed tile of one purpose profile, if the manifest carries it.
pub fn manifest_profile_tile(manifest: &Value, profile: &str) -> Option<u64> {
    manifest_profile_u64(manifest, profile, "tile_rows")
}

/// Strictly positive integers of every profile ledger. `resident_expert_layers`
/// is deliberately absent: it is the REFERENCE projection's count (an integer,
/// zero is a valid count), never a target the loaded placement must equal.
const PROFILE_POSITIVE_KEYS: [&str; 10] = [
    "tile_rows",
    "window_rows",
    "page_count",
    "device_weights_bytes",
    "s1_tile_census_bytes",
    "s1_extras_bytes",
    "device_core_bytes",
    "device_owned_bytes",
    "runtime_budget_bytes",
    "safety_margin_bytes",
];

/// Shape and arithmetic of both sealed profile ledgers: every field present,
/// the purpose equal to the profile name, tile within the historical bound and
/// distinct between the profiles, the reference placement and the full window,
/// `core = weights + tile census + extras`, `owned = core + sequential diag`,
/// no diagnostic reserve on the product profile and one on the primitive, and
/// the root `device_owned_bytes` equal to the MAX of the two owned figures.
pub fn profile_shape_problems(manifest: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    let mut tiles = Vec::new();
    let mut owned = Vec::new();
    for profile in [PROFILE_PRODUCT, PROFILE_PRIMITIVE] {
        let at = |key: &str| manifest_profile_u64(manifest, profile, key);
        let mut complete = true;
        for key in PROFILE_POSITIVE_KEYS {
            if at(key).filter(|&n| n > 0).is_none() {
                complete = false;
                bad.push(format!(
                    "manifest {} missing or not a positive integer",
                    profile_path(profile, key)
                ));
            }
        }
        if at("sequential_diag_bytes").is_none() {
            complete = false;
            bad.push(format!(
                "manifest {} missing or not an integer",
                profile_path(profile, "sequential_diag_bytes")
            ));
        }
        if at("resident_expert_layers").is_none() {
            complete = false;
            bad.push(format!(
                "manifest {} missing or not an integer (reference projection count)",
                profile_path(profile, "resident_expert_layers")
            ));
        }
        if get(manifest, &profile_path(profile, "projection_role")).and_then(Value::as_str)
            != Some(PROJECTION_ROLE_REFERENCE)
        {
            bad.push(format!(
                "manifest {} is not {PROJECTION_ROLE_REFERENCE:?}: the sealed placement is declared reference data, not a target",
                profile_path(profile, "projection_role")
            ));
        }
        bad.extend(profile_terms_problems(manifest, profile, true));
        bad.extend(profile_terms_problems(manifest, profile, false));
        if get(manifest, &profile_path(profile, "purpose")).and_then(Value::as_str) != Some(profile)
        {
            bad.push(format!(
                "manifest {} is not {profile:?}",
                profile_path(profile, "purpose")
            ));
        }
        if !complete {
            continue;
        }
        let (tile, resident, window) = (
            at("tile_rows").unwrap_or(0),
            at("resident_expert_layers").unwrap_or(0),
            at("window_rows").unwrap_or(0),
        );
        if tile > MAX_TILE_ROWS {
            bad.push(format!(
                "manifest {} {tile} is above the historical {MAX_TILE_ROWS}-row bound",
                profile_path(profile, "tile_rows")
            ));
        }
        if resident >= TRUNK_LAYERS {
            bad.push(format!(
                "manifest {} {resident} leaves no host layer of the {TRUNK_LAYERS}-layer trunk",
                profile_path(profile, "resident_expert_layers")
            ));
        }
        if window != TARGET_ROWS {
            bad.push(format!(
                "manifest {} {window} != the {TARGET_ROWS}-row window",
                profile_path(profile, "window_rows")
            ));
        }
        let part = |key: &str| at(key).unwrap_or(0);
        let core = part("device_weights_bytes")
            .checked_add(part("s1_tile_census_bytes"))
            .and_then(|n| n.checked_add(part("s1_extras_bytes")));
        if core != Some(part("device_core_bytes")) {
            bad.push(format!(
                "manifest {profile} device_core_bytes {} != weights + tile census + extras {core:?}",
                part("device_core_bytes")
            ));
        }
        let total = part("device_core_bytes").checked_add(part("sequential_diag_bytes"));
        if total != Some(part("device_owned_bytes")) {
            bad.push(format!(
                "manifest {profile} device_owned_bytes {} != core + sequential diag {total:?}",
                part("device_owned_bytes")
            ));
        }
        match (profile == PROFILE_PRODUCT, part("sequential_diag_bytes")) {
            (true, 0) | (false, 1..) => {}
            (true, n) => bad.push(format!(
                "manifest {profile} sequential_diag_bytes {n}: the product purpose carries no diagnostic reserve"
            )),
            (false, _) => bad.push(format!(
                "manifest {profile} sequential_diag_bytes is 0: the primitive self-test reserve is missing"
            )),
        }
        tiles.push(tile);
        owned.push(part("device_owned_bytes"));
    }
    if tiles.len() == 2 && tiles[0] == tiles[1] {
        bad.push(format!(
            "manifest memory_ledger.profiles claim the same tile {} for the product and primitive purposes",
            tiles[0]
        ));
    }
    if owned.len() == 2 {
        let peak = owned[0].max(owned[1]);
        if u64_at(manifest, "memory_ledger.device_owned_bytes") != Some(peak) {
            bad.push(format!(
                "manifest memory_ledger.device_owned_bytes {:?} != the MAX {peak} of the per-purpose owned bytes",
                u64_at(manifest, "memory_ledger.device_owned_bytes")
            ));
        }
    }
    bad
}

/// The name of the one census term that carries the model-lifetime cold
/// BF16-to-F16 weight shadow bound (the admission emitter's key).
pub const COLD_F16_TERM: &str = "f16_weight_shadow_bytes";

/// One profile's term array (`[{"name","bytes"}]`, unique names, checked u64
/// sum equal to its total). The cold F16 shadow term appears exactly once in
/// the tile census terms and never in the extras, and when the profile also
/// states `f16_shadow_cold_bound_bytes` it must be that one term, so the cold
/// bound is charged once and never twice.
pub fn profile_terms_problems(manifest: &Value, profile: &str, census: bool) -> Vec<String> {
    let (terms_key, total_key) = if census {
        ("s1_tile_census_terms", "s1_tile_census_bytes")
    } else {
        ("s1_extras_terms", "s1_extras_bytes")
    };
    let path = profile_path(profile, terms_key);
    let Some(array) = get(manifest, &path).and_then(Value::as_array) else {
        return vec![format!(
            "manifest {path} is not an array of {{name, bytes}} terms"
        )];
    };
    let mut bad = Vec::new();
    let mut names: Vec<&str> = Vec::new();
    let mut sum = Some(0u64);
    let mut cold: Vec<u64> = Vec::new();
    for (index, term) in array.iter().enumerate() {
        let name = term
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty());
        let bytes = term.get("bytes").and_then(Value::as_u64);
        let (Some(name), Some(bytes)) = (name, bytes) else {
            bad.push(format!(
                "manifest {path}[{index}] is not {{\"name\": non-empty string, \"bytes\": u64}}"
            ));
            continue;
        };
        if names.contains(&name) {
            bad.push(format!("manifest {path} repeats the term name {name:?}"));
        }
        names.push(name);
        sum = sum.and_then(|s| s.checked_add(bytes));
        if name == COLD_F16_TERM {
            cold.push(bytes);
        }
    }
    match (sum, u64_at(manifest, &profile_path(profile, total_key))) {
        (Some(sum), Some(total)) if sum == total => {}
        (Some(sum), total) => bad.push(format!(
            "manifest {path} sums to {sum} B, not {} {total:?}",
            profile_path(profile, total_key)
        )),
        (None, _) => bad.push(format!("manifest {path} sum overflows u64")),
    }
    let want = usize::from(census);
    if cold.len() != want {
        bad.push(format!(
            "manifest {path} carries {} {COLD_F16_TERM} term(s), expected exactly {want}",
            cold.len()
        ));
    }
    if census {
        let stated = get(
            manifest,
            &profile_path(profile, "f16_shadow_cold_bound_bytes"),
        );
        if let Some(stated) = stated.filter(|v| !v.is_null()) {
            if cold.len() != 1 || stated.as_u64() != Some(cold[0]) {
                bad.push(format!(
                    "manifest {} {stated} != the one {COLD_F16_TERM} term {cold:?}",
                    profile_path(profile, "f16_shadow_cold_bound_bytes")
                ));
            }
        }
    }
    bad
}

/// The sealed profiles must equal what this binary admits for each purpose
/// (`slowlink_admission::purpose_tile_rows`, passed in by the caller): a
/// manifest that names another tile than the one the loader will admit is an
/// identity failure before any GPU work.
pub fn profile_admission_problems(manifest: &Value, product: u64, primitive: u64) -> Vec<String> {
    let mut bad = Vec::new();
    for (profile, admitted) in [(PROFILE_PRODUCT, product), (PROFILE_PRIMITIVE, primitive)] {
        if manifest_profile_tile(manifest, profile) != Some(admitted) {
            bad.push(format!(
                "manifest {} {:?} != the {admitted}-row tile this binary admits for the {profile} purpose",
                profile_path(profile, "tile_rows"),
                manifest_profile_tile(manifest, profile)
            ));
        }
    }
    bad
}

/// The role every sealed profile declares for its placement figures
/// (`resident_expert_layers`, `device_weights_bytes`, `device_owned_bytes`,
/// the census): the projection the manifest generator made at ITS free-memory
/// reference. It is reference data, never observed and never a target that
/// rejects a valid adaptive placement.
pub const PROJECTION_ROLE_REFERENCE: &str = "reference";

/// What the load actually admitted (the bundle's profile) against the sealed
/// profile it was chosen from: the purpose, the tile and the window. The
/// resident count is NOT compared: the sealed count is reference data, and the
/// actual count is proved by [`admission_witness_problems`].
pub fn profile_load_problems(
    manifest: &Value,
    profile: &str,
    admitted_purpose: &str,
    admitted_tile: Option<u64>,
    admitted_window: Option<u64>,
) -> Vec<String> {
    let mut bad = Vec::new();
    if admitted_purpose != profile {
        bad.push(format!(
            "the admitted purpose {admitted_purpose:?} is not the {profile:?} profile this cell selects"
        ));
    }
    for (key, actual) in [
        ("tile_rows", admitted_tile),
        ("window_rows", admitted_window),
    ] {
        let sealed = manifest_profile_u64(manifest, profile, key);
        if actual.is_none() || actual != sealed {
            bad.push(format!(
                "the admitted {key} {actual:?} != the sealed {} {sealed:?}",
                profile_path(profile, key)
            ));
        }
    }
    bad
}

// ── adaptive admission witness ─────────────────────────────────────────
//
// The loader keeps the LARGEST resident layer count that fits (ten is only
// preferred). The load placement detail carries a scalar witness of the
// shared admission quote (`PlacementBudget` at the selected count); this
// module judges it independently from the scalars alone: the selected count
// must fit, one more resident layer must not, at least one host layer must
// remain, and the count and snapshot allowance must be the bundle's own. The
// shared Rust projection generates the witness; no second selection policy
// lives here.

/// Basis of the free-memory sample the quote was made from: the FRESH
/// post-initialization `hipMemGetInfo`, never the gate's earlier `pre_load`.
pub const ADMISSION_MEMORY_BASIS: &str = "post_initialization_actual";

/// The scalars of the shared quote at the selected resident count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionScalars {
    pub free_before_initialization: Option<u64>,
    pub free_at_admission: u64,
    pub observed_initialization: Option<u64>,
    pub non_expert: u64,
    /// Backed device cost of one resident layer (payload + backing slack).
    pub resident_layer: u64,
    pub joint_reserve: u64,
    pub trunk_layers: u64,
    pub resident_layers: u64,
    /// `PlacementBudget::needed_bytes_for(resident_layers)`.
    pub needed: u128,
    /// `PlacementBudget::slack_bytes_at(resident_layers)`.
    pub slack: i128,
    /// `PlacementBudget::preferred_deficit_bytes()`.
    pub preferred_deficit: u64,
}

/// The witness JSON. A figure outside the JSON integer range serializes as
/// `null`, which the validator refuses.
pub fn admission_witness_json(s: &AdmissionScalars) -> Value {
    json!({
        "free_before_initialization_bytes": s.free_before_initialization,
        "free_at_admission_bytes": s.free_at_admission,
        "observed_initialization_bytes": s.observed_initialization,
        "non_expert_bytes": s.non_expert,
        "resident_layer_bytes": s.resident_layer,
        "joint_reserve_bytes": s.joint_reserve,
        "trunk_layers": s.trunk_layers,
        "resident_layers": s.resident_layers,
        "preferred_resident_layers": PREFERRED_RESIDENT_LAYERS,
        "needed_bytes": u64::try_from(s.needed).ok(),
        "slack_bytes": i64::try_from(s.slack).ok(),
        "preferred_deficit_bytes": s.preferred_deficit,
        "memory_basis": ADMISSION_MEMORY_BASIS,
        "initialization_charged_via_free": true,
    })
}

/// The scalar admission witness of a loaded bundle's profile, read from the
/// SAME shared quote the loader admitted with (`profile.placement`, the
/// post-initialization `PlacementBudget` at the selected count): never the
/// gate's `pre_load` sample and never a hand calculation. The independent
/// judge is [`admission_witness_problems`].
pub fn admission_witness(
    profile: &hipfire_arch_qwen4::slowlink_admission::SlowlinkProfile,
) -> Value {
    let budget = &profile.placement;
    let selected = profile.resident_layers;
    admission_witness_json(&AdmissionScalars {
        free_before_initialization: budget.memory.pre_init_free_bytes,
        free_at_admission: budget.memory.free_bytes,
        observed_initialization: budget.memory.observed_initialization_bytes(),
        non_expert: budget.non_expert_bytes,
        resident_layer: budget.resident_layer_bytes,
        joint_reserve: budget.reserve_bytes,
        trunk_layers: budget.trunk_layers as u64,
        resident_layers: selected as u64,
        needed: budget.needed_bytes_for(selected),
        slack: budget.slack_bytes_at(selected),
        preferred_deficit: budget.preferred_deficit_bytes(),
    })
}

/// Payload bytes of one resident layer (the backed layer cost less its own
/// backing slack), the unit the reference weights move by.
pub fn layer_payload_bytes(
    profile: &hipfire_arch_qwen4::slowlink_admission::SlowlinkProfile,
) -> Option<u64> {
    profile
        .placement
        .resident_layer_bytes
        .checked_sub(profile.granule.layer)
}

/// The bundle's own facts, to be tied to the witness.
pub fn admission_actual(
    profile: Option<&hipfire_arch_qwen4::slowlink_admission::SlowlinkProfile>,
    resident_layers: usize,
    trunk_layers: usize,
) -> AdmissionActual {
    AdmissionActual {
        resident_layers: Some(resident_layers as u64),
        trunk_layers: Some(trunk_layers as u64),
        snapshot_budget_bytes: profile.map(|p| p.device_snapshot_budget_bytes),
    }
}

/// What the loaded bundle itself reports, to be tied to the witness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionActual {
    /// The bundle's actual resident layer count.
    pub resident_layers: Option<u64>,
    /// The model's trunk layer count.
    pub trunk_layers: Option<u64>,
    /// The admitted profile's `device_snapshot_budget_bytes`.
    pub snapshot_budget_bytes: Option<u64>,
}

const WITNESS_INTEGERS: [&str; 10] = [
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
];

fn witness_int(w: &Value, key: &str) -> Option<i128> {
    let v = w.get(key)?;
    v.as_u64()
        .map(i128::from)
        .or_else(|| v.as_i64().map(i128::from))
}

/// Every reason the admission witness is not a proof of a valid, maximal,
/// actual placement; empty means proven. A missing, partial or forged witness
/// never passes.
pub fn admission_witness_problems(
    witness: Option<&Value>,
    actual: &AdmissionActual,
) -> Vec<String> {
    let Some(w) = witness.filter(|w| w.is_object()) else {
        return vec![
            "admission witness missing: the actual placement is not proved maximal-fits".into(),
        ];
    };
    let mut bad = Vec::new();
    let mut vals = std::collections::BTreeMap::new();
    for key in WITNESS_INTEGERS {
        match witness_int(w, key) {
            Some(v) if v >= 0 || key == "slack_bytes" => {
                vals.insert(key, v);
            }
            _ => bad.push(format!(
                "admission.{key} missing or not an integer (non-negative except slack_bytes)"
            )),
        }
    }
    for key in [
        "free_before_initialization_bytes",
        "observed_initialization_bytes",
    ] {
        match w.get(key) {
            Some(Value::Null) => {}
            Some(v) if v.as_u64().is_some() => {}
            _ => bad.push(format!(
                "admission.{key} missing (unsigned integer or null)"
            )),
        }
    }
    if w.get("memory_basis").and_then(Value::as_str) != Some(ADMISSION_MEMORY_BASIS) {
        bad.push(format!(
            "admission.memory_basis {:?} is not {ADMISSION_MEMORY_BASIS:?}: the quote must be made from the fresh post-initialization free",
            w.get("memory_basis")
        ));
    }
    if w.get("initialization_charged_via_free") != Some(&Value::Bool(true)) {
        bad.push(
            "admission.initialization_charged_via_free is not true: initialization must be charged once, through the lower actual free"
                .into(),
        );
    }
    if !bad.is_empty() {
        return bad;
    }
    let at = |key: &str| vals[key];
    let (free, non_expert, layer, reserve) = (
        at("free_at_admission_bytes") as u128,
        at("non_expert_bytes") as u128,
        at("resident_layer_bytes") as u128,
        at("joint_reserve_bytes") as u128,
    );
    let (trunk, n) = (at("trunk_layers") as u128, at("resident_layers") as u128);
    if at("preferred_resident_layers") != i128::from(PREFERRED_RESIDENT_LAYERS) {
        bad.push(format!(
            "admission.preferred_resident_layers {} != {PREFERRED_RESIDENT_LAYERS}",
            at("preferred_resident_layers")
        ));
    }
    if layer == 0 {
        bad.push("admission.resident_layer_bytes is 0: maximality is unprovable".into());
        return bad;
    }
    let needed_at = |count: u128| {
        layer
            .checked_mul(count)
            .and_then(|b| b.checked_add(non_expert))
            .and_then(|b| b.checked_add(reserve))
    };
    let (Some(needed), Some(needed_next), Some(needed_preferred)) = (
        needed_at(n),
        needed_at(n + 1),
        needed_at(u128::from(PREFERRED_RESIDENT_LAYERS)),
    ) else {
        bad.push("admission quote arithmetic overflows".into());
        return bad;
    };
    if needed as i128 != at("needed_bytes") {
        bad.push(format!(
            "admission.needed_bytes {} != non_expert + {n} * resident_layer + reserve = {needed}",
            at("needed_bytes")
        ));
    }
    if needed > free {
        bad.push(format!(
            "admission oversubscribed: {n} resident layers need {needed} B but only {free} B were free at admission"
        ));
    } else if free as i128 - needed as i128 != at("slack_bytes") {
        bad.push(format!(
            "admission.slack_bytes {} != free {free} - needed {needed}",
            at("slack_bytes")
        ));
    }
    if n >= trunk {
        bad.push(format!(
            "admission claims {n} resident of {trunk} trunk layers: no host layer remains, so an S1 profile must not be claimed (all-resident bypass)"
        ));
    }
    if needed_next <= free {
        bad.push(format!(
            "admission is not maximal: {} resident layers would also fit ({needed_next} B <= {free} B free)",
            n + 1
        ));
    }
    let deficit = needed_preferred.saturating_sub(free);
    if deficit as i128 != at("preferred_deficit_bytes") {
        bad.push(format!(
            "admission.preferred_deficit_bytes {} != the {PREFERRED_RESIDENT_LAYERS}-layer deficit {deficit} (information only, never a rejection)",
            at("preferred_deficit_bytes")
        ));
    }
    let before = w
        .get("free_before_initialization_bytes")
        .and_then(Value::as_u64);
    let observed = w
        .get("observed_initialization_bytes")
        .and_then(Value::as_u64);
    let observed_ok = match before {
        None => observed.is_none(),
        Some(b) if u128::from(b) >= free => observed == Some(b - free as u64),
        // The fresh free exceeds the earlier sample: the actual free stays
        // authoritative and no positive initialization loss is invented.
        Some(_) => matches!(observed, None | Some(0)),
    };
    if !observed_ok {
        bad.push(format!(
            "admission.observed_initialization_bytes {observed:?} != the measured interval before {before:?} - after {free}"
        ));
    }
    match actual.resident_layers {
        Some(r) if u128::from(r) == n => {}
        other => bad.push(format!(
            "admission.resident_layers {n} != the bundle's actual resident layers {other:?}"
        )),
    }
    match actual.trunk_layers {
        Some(t) if u128::from(t) == trunk => {}
        other => bad.push(format!(
            "admission.trunk_layers {trunk} != the loaded model's trunk layers {other:?}"
        )),
    }
    match actual.snapshot_budget_bytes {
        Some(s) if i128::from(s) == at("slack_bytes") => {}
        other => bad.push(format!(
            "the profile's snapshot allowance {other:?} != the quote's slack {} at the selected count",
            at("slack_bytes")
        )),
    }
    bad
}

/// Device bytes the actual placement holds beyond (or below) the reference
/// projection: `(actual resident - reference resident) * payload bytes of one
/// resident layer`. The census, extras and diagnostics do not depend on the
/// resident count.
pub fn reference_weight_delta(
    manifest: &Value,
    profile: &str,
    actual_resident: u64,
    layer_payload_bytes: u64,
) -> Result<i128, String> {
    let reference =
        manifest_profile_u64(manifest, profile, "resident_expert_layers").ok_or(format!(
            "manifest {} missing",
            profile_path(profile, "resident_expert_layers")
        ))?;
    (i128::from(actual_resident) - i128::from(reference))
        .checked_mul(i128::from(layer_payload_bytes))
        .ok_or_else(|| "placement accounting delta overflows".to_string())
}

fn shifted(base: u64, delta: i128) -> Option<u64> {
    u64::try_from(i128::from(base).checked_add(delta)?).ok()
}

/// The reference projection and the ACTUAL placement, side by side. The
/// reference is manifest data (labelled as such); the actual figures are the
/// reference moved by the measured resident-count difference at the actual
/// layer payload, never presented as observed memory.
pub fn placement_accounting(
    manifest: &Value,
    profile: &str,
    actual_resident: u64,
    trunk_layers: u64,
    layer_payload_bytes: Option<u64>,
) -> Value {
    let reference = |key: &str| manifest_profile_u64(manifest, profile, key);
    let delta = layer_payload_bytes
        .ok_or_else(|| "the actual layer payload bytes are unknown".to_string())
        .and_then(|p| reference_weight_delta(manifest, profile, actual_resident, p));
    let (weights, owned, delta_json, error) = match (
        &delta,
        reference("device_weights_bytes"),
        reference("device_owned_bytes"),
    ) {
        (Ok(d), Some(w), Some(o)) => (shifted(w, *d), shifted(o, *d), i64::try_from(*d).ok(), None),
        (Err(e), _, _) => (None, None, None, Some(e.clone())),
        _ => (
            None,
            None,
            None,
            Some("manifest reference weights/owned bytes missing".to_string()),
        ),
    };
    json!({
        "reference": {
            "role": PROJECTION_ROLE_REFERENCE,
            "note": "the sealed manifest's projection at its own free-memory reference: reference data, not observed and not a target",
            "resident_expert_layers": reference("resident_expert_layers"),
            "device_weights_bytes": reference("device_weights_bytes"),
            "device_owned_bytes": reference("device_owned_bytes"),
        },
        "actual": {
            "basis": "reference moved by (actual - reference resident layers) * actual layer payload bytes; census, extras and diagnostics are independent of the resident count",
            "resident_layers": actual_resident,
            "host_layers": trunk_layers.saturating_sub(actual_resident),
            "resident_layer_payload_bytes": layer_payload_bytes,
            "weights_delta_vs_reference_bytes": delta_json,
            "device_weights_bytes": weights,
            "device_owned_bytes": owned,
            "error": error,
        },
    })
}

impl MemoryLedger {
    /// The ledger of the ACTUAL placement: weights and owned device bytes
    /// moved by `delta` (see [`reference_weight_delta`]). A figure that would
    /// leave the u64 range refuses.
    pub fn at_actual_placement(&self, delta: i128) -> Result<MemoryLedger, String> {
        let mut out = self.clone();
        out.owned_device_bytes = shifted(self.owned_device_bytes, delta)
            .ok_or("actual owned device bytes leave the u64 range")?;
        out.decomposition.weights = shifted(self.decomposition.weights, delta)
            .ok_or("actual device weight bytes leave the u64 range")?;
        out.census_basis = format!(
            "{}; weights and owned device bytes moved to the ACTUAL placement by {delta} B",
            self.census_basis
        );
        Ok(out)
    }
}

// ── manifest identity ──────────────────────────────────────────────────

pub const MANIFEST_REQUIRED: [&str; 20] = [
    "source.base",
    "source.tip",
    "binary.sha256",
    "model.path",
    "model.size_bytes",
    "model.sha256",
    "ple.source",
    "ple.format",
    "ple.sidecars",
    "kernel.code_object_hashes",
    "host.hostname",
    "host.topology",
    "device.uuid",
    "device.arch",
    "link.gbps",
    "link.topology_walk",
    "memory_ledger.device_gib",
    "memory_ledger.host_unreclaimable_gib",
    "memory_ledger.profiles.product.tile_rows",
    "memory_ledger.profiles.primitive_selftest.tile_rows",
];

#[derive(Clone, Debug)]
pub struct Observed {
    pub binary_sha256: String,
    pub model_size_bytes: u64,
    pub gpu_arch: String,
}

fn is_hex(value: Option<&str>, len: usize) -> bool {
    value.is_some_and(|s| s.len() == len && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Missing keys and identity mismatches between a manifest and what the
/// process actually observes. Empty = identity holds.
pub fn manifest_problems(manifest: &Value, observed: &Observed) -> Vec<String> {
    let mut bad = Vec::new();
    for path in MANIFEST_REQUIRED {
        match get(manifest, path) {
            None | Some(Value::Null) => bad.push(format!("manifest key {path} missing")),
            Some(Value::String(s)) if s.is_empty() => {
                bad.push(format!("manifest key {path} empty"))
            }
            _ => {}
        }
    }
    if !bad.is_empty() {
        return bad;
    }
    bad.extend(profile_shape_problems(manifest));
    for path in ["source.base", "source.tip"] {
        if !is_hex(get(manifest, path).and_then(Value::as_str), 40) {
            bad.push(format!("manifest {path} is not a full 40-hex commit"));
        }
    }
    if !is_hex(get(manifest, "binary.sha256").and_then(Value::as_str), 64) {
        bad.push("manifest binary.sha256 is not a 64-hex digest".into());
    } else if get(manifest, "binary.sha256").and_then(Value::as_str)
        != Some(observed.binary_sha256.as_str())
    {
        bad.push(format!(
            "binary identity mismatch: manifest {} != running {}",
            get(manifest, "binary.sha256").unwrap(),
            observed.binary_sha256
        ));
    }
    if !is_hex(get(manifest, "model.sha256").and_then(Value::as_str), 64) {
        bad.push("manifest model.sha256 is not a 64-hex digest".into());
    }
    if u64_at(manifest, "model.size_bytes") != Some(observed.model_size_bytes) {
        bad.push(format!(
            "model identity mismatch: manifest size {:?} != file size {}",
            u64_at(manifest, "model.size_bytes"),
            observed.model_size_bytes
        ));
    }
    if get(manifest, "device.arch").and_then(Value::as_str) != Some(observed.gpu_arch.as_str()) {
        bad.push(format!(
            "device arch mismatch: manifest {:?} != {}",
            get(manifest, "device.arch"),
            observed.gpu_arch
        ));
    }
    for path in [
        "memory_ledger.device_gib",
        "memory_ledger.host_unreclaimable_gib",
        "link.gbps",
    ] {
        match f64_at(manifest, path) {
            Some(v) if v.is_finite() && v > 0.0 => {}
            _ => bad.push(format!("manifest {path} is not a positive number")),
        }
    }
    bad
}

// ── KLD receipts and the paired verdict ────────────────────────────────

pub const KLD_ROWS: u64 = 6120;
pub const KLD_SEQUENCES: u64 = 24;
pub const KLD_HI_LIMIT: f64 = 0.010;

/// A full-gate S11 decode receipt: not a partition, every row, every
/// corpus sequence, the decode route.
pub fn kld_receipt_problems(receipt: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    if receipt.get("schema").and_then(Value::as_str) != Some("hipfire.qwen4.s11ref_klrows.v1") {
        bad.push("not a full-gate s11ref receipt (schema; a partition is never the gate)".into());
    }
    if u64_at(receipt, "rows") != Some(KLD_ROWS) {
        bad.push(format!(
            "rows {:?} != {KLD_ROWS}: partial corpus",
            receipt.get("rows")
        ));
    }
    if u64_at(receipt, "sequences") != Some(KLD_SEQUENCES) {
        bad.push(format!(
            "sequences {:?} != {KLD_SEQUENCES}: partial corpus",
            receipt.get("sequences")
        ));
    }
    if receipt.get("route").and_then(Value::as_str) != Some("decode") {
        bad.push("route is not decode".into());
    }
    for key in [
        "model_sha256",
        "ref_sha256",
        "corpus_sha256",
        "binary_sha256",
        "output_sha256",
    ] {
        if !is_hex(receipt.get(key).and_then(Value::as_str), 64) {
            bad.push(format!("receipt {key} is not a 64-hex digest"));
        }
    }
    bad
}

/// Same reference, corpus, context and route; different weights/binary are
/// expected (candidate versus beta).
pub fn kld_receipts_comparable(base: &Value, cand: &Value) -> Vec<String> {
    let mut bad = Vec::new();
    for key in [
        "ref_sha256",
        "corpus_sha256",
        "n_ctx",
        "route",
        "rows",
        "sequences",
        "estimator",
    ] {
        if base.get(key) != cand.get(key) {
            bad.push(format!(
                "receipts differ in {key}: not the same ordering/reference/config"
            ));
        }
    }
    bad
}

/// Mean KL per corpus sequence: `counts[i]` rows belong to sequence `i`, in
/// canonical row order. Aggregating first means the interval's unit is the
/// sequence, never the token.
pub fn sequence_means(rows: &[f64], counts: &[usize]) -> Result<Vec<f64>, String> {
    let total: usize = counts.iter().sum();
    if total != rows.len() {
        return Err(format!(
            "sequence row counts sum to {total}, the rows are {}",
            rows.len()
        ));
    }
    if counts.iter().any(|&c| c == 0) {
        return Err("a corpus sequence has no scored rows".into());
    }
    let mut at = 0;
    Ok(counts
        .iter()
        .map(|&c| {
            let mean = rows[at..at + c].iter().sum::<f64>() / c as f64;
            at += c;
            mean
        })
        .collect())
}

/// Two-sided 95% Student-t critical value for `df` degrees of freedom
/// (conservative above 30).
pub fn t_975(df: usize) -> f64 {
    const TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    match df {
        0 => f64::INFINITY,
        1..=30 => TABLE[df - 1],
        31..=40 => 2.021,
        41..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedCi {
    pub n: usize,
    pub mean: f64,
    pub se: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Paired candidate-minus-beta interval over per-sequence means.
pub fn paired_ci(base: &[f64], cand: &[f64]) -> Result<PairedCi, String> {
    if base.len() != cand.len() || base.len() < 2 {
        return Err(format!(
            "paired interval needs equal per-sequence vectors of at least 2 ({} vs {})",
            base.len(),
            cand.len()
        ));
    }
    if base.iter().chain(cand).any(|v| !v.is_finite()) {
        return Err("non-finite per-sequence KL".into());
    }
    let diffs: Vec<f64> = cand.iter().zip(base).map(|(c, b)| c - b).collect();
    let n = diffs.len();
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    let half = t_975(n - 1) * se;
    Ok(PairedCi {
        n,
        mean,
        se,
        lo: mean - half,
        hi: mean + half,
    })
}

/// Pass `lo <= 0`; otherwise require `hi <= +0.010`.
pub fn paired_verdict(ci: &PairedCi) -> bool {
    ci.lo <= 0.0 || ci.hi <= KLD_HI_LIMIT
}
