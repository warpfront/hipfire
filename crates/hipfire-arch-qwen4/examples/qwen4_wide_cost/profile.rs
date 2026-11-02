// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-only logic of the `qwen4_wide_verify` cost-curve (`--cost-curve`) and
//! strict-qualification (`--strict-qualify`) profiles: CLI value parsers,
//! timing records, width labels, window composition and the strict summary.
//! Nothing here touches a GPU; the example's `tests` module covers it.
//!
//! Cost model (declared, never fitted): isolated component medians provide
//! only a subtotal, never a full production window price. Composed records
//! always have `window_ms: null` and `complete: false`. Actual full-window
//! receipts preserve the full context and explicit route; their aggregates
//! exclude warmups, missing/incomplete and terminal-censored observations,
//! and keep source, confirmation variant and actual drafted width separate.
//! Differences from component subtotals remain unattributed residuals.
//! GPU timing receipts sum finite, nonnegative per-launch spans in f64 and
//! require a positive total; collector capacity uses checked warmup census math.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub const COST_SCHEMA: &str = "qwen4_wide_verify.cost_curve.v1";
pub const ACTUAL_WINDOWS_SCHEMA: &str = "qwen4_wide_verify.actual_windows.v1";
pub const STRICT_SCHEMA: &str = "qwen4_wide_verify.strict.v1";
/// Parent fabric hard cap is 1800 s; the deadline (loads included) stays under.
pub const MAX_DEADLINE_SECS: u64 = 1750;
pub const DEFAULT_DEADLINE_SECS: u64 = 1500;
pub const DEFAULT_COST_ROWS: &str = "1-16,32,64";
/// Head chain depths of the cost profile: rows 1..16 plus the drafted depths
/// of the 32- and 64-row verify blocks (seed + K drafts).
pub const DEFAULT_HEAD_DEPTHS: &str = "1-16,31,63";
/// Loader MTP reserve of the cost / strict profiles: the physical 64-row
/// verify block (seed + 63 drafts).
pub const PROFILE_MAX_K: usize = 63;
/// Loader MTP reserve of the legacy oracle profile (the shipped default K).
pub const ORACLE_MAX_K: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Profile {
    /// The original bitwise oracle (`n <= 8` hard).
    Oracle,
    /// Timing / pricing only: identity is checked only where claimed.
    Cost,
    /// Fatal bytes + state comparison at every requested row.
    Strict,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Profile::Oracle => "oracle",
            Profile::Cost => "cost-curve",
            Profile::Strict => "strict-qualify",
        }
    }

    pub fn schema(self) -> Option<&'static str> {
        match self {
            Profile::Oracle => None,
            Profile::Cost => Some(COST_SCHEMA),
            Profile::Strict => Some(STRICT_SCHEMA),
        }
    }

    /// Whether a mismatch at verify width `n` is fatal given the arm's
    /// qualified row limit.
    pub fn hard_width(self, n: usize, qualified_rows: usize, few_row_rows: usize) -> bool {
        match self {
            Profile::Oracle => n <= few_row_rows,
            Profile::Cost => n <= qualified_rows,
            Profile::Strict => true,
        }
    }
}

/// Proposal mode of the cost book (`DerivedCostMode`); costs of one mode are
/// never reused for the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CostMode {
    OffPool,
    OnCopy,
}

impl CostMode {
    pub fn key(self) -> &'static str {
        match self {
            CostMode::OffPool => "OffPool",
            CostMode::OnCopy => "OnCopy",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "OffPool" => Ok(CostMode::OffPool),
            "OnCopy" => Ok(CostMode::OnCopy),
            other => Err(format!("--cost-mode takes OffPool|OnCopy, got '{other}'")),
        }
    }

    pub fn of_copy_lane(copy_lane_on: bool) -> Self {
        if copy_lane_on {
            CostMode::OnCopy
        } else {
            CostMode::OffPool
        }
    }
}

pub fn parse_on_off(flag: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!("{flag} takes on|off")),
    }
}

pub fn parse_deadline_secs(value: &str) -> Result<u64, String> {
    let secs: u64 = value
        .parse()
        .map_err(|_| format!("bad --deadline-secs '{value}'"))?;
    if secs == 0 || secs > MAX_DEADLINE_SECS {
        return Err(format!(
            "--deadline-secs must be 1..={MAX_DEADLINE_SECS} (the fabric hard cap is 1800 s and loads count)"
        ));
    }
    Ok(secs)
}

/// Wall-clock budget of the whole process, loads included.
#[derive(Clone, Copy)]
pub struct Deadline {
    start: Instant,
    limit: Duration,
}

impl Deadline {
    pub fn new(start: Instant, secs: u64) -> Self {
        Deadline {
            start,
            limit: Duration::from_secs(secs),
        }
    }

    pub fn exceeded(&self) -> bool {
        self.start.elapsed() >= self.limit
    }

    pub fn json(&self) -> Value {
        json!({
            "limit_s": self.limit.as_secs(),
            "elapsed_s": self.start.elapsed().as_secs_f64(),
            "exceeded": self.exceeded(),
        })
    }
}

pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.get(sorted.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Sum actual per-kernel event spans, not their mean or an outer wall span.
pub fn kernel_span_ms(per_kernel_ms: &[f32], kernel_count: usize) -> Result<f64, String> {
    if kernel_count == 0 || per_kernel_ms.len() != kernel_count {
        return Err("kernel timing requires a nonzero count matching the span length".to_string());
    }
    let mut total = 0.0f64;
    for &span in per_kernel_ms {
        if !span.is_finite() || span < 0.0 {
            return Err("kernel timing spans must be finite and nonnegative".to_string());
        }
        total += f64::from(span);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("kernel timing span sum must be finite and positive".to_string());
    }
    Ok(total)
}

/// Validate one instrumented verify report and return its GPU span: the SUM
/// of the actual per-launch spans, never a mean or a host wall time.
/// Coverage must be exactly `EagerHipLaunchWrappers`; the count must be
/// nonzero, match the span length and fit the prepared event capacity.
#[cfg(feature = "kernel-timing")]
pub fn kernel_report_span_ms(
    report: &hip_bridge::KernelTimingReport,
    capacity: usize,
) -> Result<f64, String> {
    if !matches!(
        report.coverage,
        hip_bridge::KernelTimingCoverage::EagerHipLaunchWrappers
    ) {
        return Err("GPU kernel timing coverage is not eager HIP launch wrappers".into());
    }
    if report.kernel_count > capacity {
        return Err("GPU kernel timing report exceeds prepared event capacity".into());
    }
    kernel_span_ms(&report.per_kernel_ms, report.kernel_count)
}

/// Reserve twice the uninstrumented warmup launch census plus 32 spare slots.
pub fn kernel_timing_capacity(warmup_launches: u64) -> Result<usize, String> {
    if warmup_launches == 0 {
        return Err("kernel timing requires a nonzero warmup launch census".to_string());
    }
    let launches = usize::try_from(warmup_launches)
        .map_err(|_| "kernel timing launch census does not fit usize".to_string())?;
    launches
        .checked_mul(2)
        .and_then(|capacity| capacity.checked_add(32))
        .ok_or_else(|| "kernel timing capacity overflows usize".to_string())
}

/// Raw repeats of one timed boundary. The first sample is the warmup: it is
/// kept apart and never enters the median / min / `all`.
#[derive(Clone, Debug)]
pub struct Timed {
    pub warmup_ms: f64,
    pub repeats: Vec<f64>,
}

impl Timed {
    /// `samples[0]` is the warmup; at least one measured repeat must follow.
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        let (warmup, rest) = samples.split_first()?;
        if rest.is_empty() {
            return None;
        }
        Some(Timed {
            warmup_ms: *warmup,
            repeats: rest.to_vec(),
        })
    }

    pub fn median(&self) -> f64 {
        median(&self.repeats)
    }

    pub fn min(&self) -> f64 {
        self.repeats.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// `median` / `min` / `all` keep the legacy `ms_json` shape.
    pub fn json(&self) -> Value {
        json!({
            "median": self.median(),
            "min": self.min(),
            "all": self.repeats,
            "repeats": self.repeats.len(),
            "warmup_ms": self.warmup_ms,
        })
    }
}

pub fn per_row_ms(total_ms: f64, rows: usize) -> f64 {
    total_ms / rows.max(1) as f64
}

/// Context bucket of an absolute position (`ContextBucket::of`).
pub fn context_bucket(position: usize) -> &'static str {
    if position <= 2048 {
        "Le2K"
    } else if position <= 8192 {
        "Le8K"
    } else if position <= 16384 {
        "Le16K"
    } else {
        "Over16K"
    }
}

pub fn width_label(n: usize, qualified_rows: usize) -> &'static str {
    if n <= qualified_rows {
        "identity-qualified"
    } else {
        "timing-only-beyond-qualified"
    }
}

/// An identity claim needs the width inside the qualified rows AND a checked,
/// fully bitwise result; a width beyond the qualified rows never claims.
pub fn identity_claim(
    n: usize,
    qualified_rows: usize,
    checked: bool,
    all_rows_bitwise: bool,
) -> bool {
    n <= qualified_rows && checked && all_rows_bitwise
}

pub fn card_of_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "gfx1201" => Some("R9700"),
        "gfx1151" => Some("Halo"),
        _ => None,
    }
}

/// Sorted unique list of `a,b,c-d` entries, each `>= min`.
pub fn parse_list(flag: &str, spec: &str, min: usize) -> Result<Vec<usize>, String> {
    let mut values = std::collections::BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let bad = || format!("bad {flag} entry '{part}'");
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().map_err(|_| bad())?;
            let b: usize = b.trim().parse().map_err(|_| bad())?;
            if b < a || a < min {
                return Err(bad());
            }
            values.extend(a..=b);
        } else {
            let n: usize = part.parse().map_err(|_| bad())?;
            if n < min {
                return Err(bad());
            }
            values.insert(n);
        }
    }
    if values.is_empty() {
        return Err(format!("{flag} is empty"));
    }
    Ok(values.into_iter().collect())
}

/// Requested widths with no measured record.
pub fn missing_widths(requested: &[usize], measured: &[usize]) -> Vec<usize> {
    requested
        .iter()
        .copied()
        .filter(|n| !measured.contains(n))
        .collect()
}

/// Measured medians of one verify width.
#[derive(Clone, Debug)]
pub struct WidthCost {
    pub n: usize,
    pub verify_ms: f64,
    /// accept_len -> commit median (`None` = not measured).
    pub commit_ms: BTreeMap<usize, Option<f64>>,
}

/// One actual production window; unsupported observations remain unknown.
#[derive(Clone, Debug)]
pub struct ProdWindow {
    /// Only an actual started request has an identity.
    pub request_id: Option<String>,
    /// Scheduled source, retained even when the observed production source differs.
    pub requested_source: &'static str,
    /// `Mtp`, `Ngram` or `ArRoutedK0`, as observed by the production hook.
    pub source: &'static str,
    /// Ngram requested confirmation mode, not the actual head result.
    /// `unconfirmed` means confirmation was skipped, never that it failed.
    /// Unsupported observations leave the variant unknown.
    pub variant: Option<&'static str>,
    pub requested_confirmation: Option<bool>,
    pub head_confirmation: Option<bool>,
    pub requested_k: usize,
    pub position: usize,
    pub drafts: Option<usize>,
    pub accepted: Option<usize>,
    pub committed: Option<usize>,
    pub wall_ms: Option<f64>,
    /// Resolved EOS/end-of-turn and rejection stay unknown when the hook fails.
    pub terminal: Option<bool>,
    pub terminal_reason: Option<&'static str>,
    pub rejected: Option<bool>,
    pub terminal_censored: bool,
    /// Native MTP causal-margin observations for this window: one entry per
    /// actual native draft (`actual_drafted`), `Some(m)` iff the head-step
    /// margin at that row was finite and nonnegative, else `None` (never
    /// fabricated). Empty for non-Mtp sources. Actual observed features, not
    /// acceptance or prose labels; never priced, exporter-only evidence.
    pub native_draft_margins: Vec<Option<f32>>,
    pub complete: bool,
    pub offer_available: Option<bool>,
    pub missing_reason: Option<String>,
    pub warmup: bool,
    pub verify_route: Value,
}

/// Render one observed native margin as JSON: a finite nonnegative `Some(m)`
/// becomes its number, anything else (`None`, nonfinite, negative) is null.
/// Never fabricates: absence of measurement stays null.
fn native_margin_json(margin: &Option<f32>) -> Value {
    match margin {
        Some(m) if m.is_finite() && *m >= 0.0 => json!(*m),
        _ => Value::Null,
    }
}

/// Render the window's native margin array: one entry per actual native
/// draft for Mtp, `[]` for every other source. Length match against
/// `actual_drafted` is enforced by the caller when it marks completion.
pub fn native_margins_json(source: &str, margins: &[Option<f32>]) -> Value {
    if source != "Mtp" {
        return json!([]);
    }
    json!(margins.iter().map(native_margin_json).collect::<Vec<_>>())
}

impl ProdWindow {
    /// Cost-curve receipt, retaining the original aliases for raw timings.
    pub fn json(&self) -> Value {
        json!({
            "request_id": self.request_id,
            "requested_source": self.requested_source,
            "source": self.source,
            "variant": self.variant,
            "requested_confirmation": self.requested_confirmation,
            "head_confirmation": self.head_confirmation,
            "requested_k": self.requested_k,
            "requested_drafted": self.requested_k,
            "position": self.position,
            "position_start": self.position,
            "context_bucket": context_bucket(self.position),
            "drafts": self.drafts,
            "actual_drafted": self.drafts,
            "accepted": self.accepted,
            "committed": self.committed,
            "wall_ms": self.wall_ms,
            "total_window_us": self.wall_ms.map(|ms| ms * 1000.0),
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "rejected": self.rejected,
            "terminal_censored": self.terminal_censored,
            "native_draft_margins": native_margins_json(self.source, &self.native_draft_margins),
            "complete": self.complete,
            "offer_available": self.offer_available,
            "missing_reason": self.missing_reason,
            "warmup": self.warmup,
            "verify_route": self.verify_route,
        })
    }

    /// Preserve the full context identity, but stamp the observed start position
    /// and its actual bucket instead of carrying a nominal context label.
    pub fn actual_json(&self, prompt_index: usize, context: &Value) -> Value {
        let mut actual = json!({
            "request_id": self.request_id,
            "prompt_index": prompt_index,
            "position_start": self.position,
            "context": context,
            "requested_source": self.requested_source,
            "source": self.source,
            "variant": self.variant,
            "requested_confirmation": self.requested_confirmation,
            "head_confirmation": self.head_confirmation,
            "verify_route": self.verify_route,
            "requested_drafted": self.requested_k,
            "actual_drafted": self.drafts,
            "accepted": self.accepted,
            "committed": self.committed,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "rejected": self.rejected,
            "terminal_censored": self.terminal_censored,
            "native_draft_margins": native_margins_json(self.source, &self.native_draft_margins),
            "total_window_us": self.wall_ms.map(|ms| ms * 1000.0),
            "complete": self.complete,
            "offer_available": self.offer_available,
            "missing_reason": self.missing_reason,
            "warmup": self.warmup,
        });
        actual["context"]["position"] = json!(self.position);
        actual["context"]["bucket"] = json!(context_bucket(self.position));
        actual
    }
}

/// Complete actual full-window totals grouped by `(source, variant, drafts)`.
/// Each invocation belongs to one cost book's proposal mode and context.
/// Requested width never substitutes for an unknown actual drafted width.
pub fn aggregate_production_windows(
    windows: &[ProdWindow],
) -> (Vec<Value>, BTreeMap<(String, Option<String>, usize), f64>) {
    let mut groups: BTreeMap<(String, Option<String>, usize), Vec<(usize, usize, f64)>> =
        BTreeMap::new();
    for window in windows {
        if window.warmup
            || !window.complete
            || window.missing_reason.is_some()
            || window.terminal_censored
            || window.offer_available.is_none()
        {
            continue;
        }
        let (Some(drafts), Some(accepted), Some(committed), Some(wall_ms)) = (
            window.drafts,
            window.accepted,
            window.committed,
            window.wall_ms,
        ) else {
            continue;
        };
        if !wall_ms.is_finite() || wall_ms <= 0.0 {
            continue;
        }
        groups
            .entry((
                window.source.to_string(),
                window.variant.map(str::to_string),
                drafts,
            ))
            .or_default()
            .push((accepted, committed, wall_ms));
    }
    let mut rows = Vec::new();
    let mut medians = BTreeMap::new();
    for ((source, variant, drafts), group) in groups {
        let walls: Vec<f64> = group.iter().map(|&(_, _, wall_ms)| wall_ms).collect();
        let mut histogram: BTreeMap<usize, usize> = BTreeMap::new();
        for &(accepted, _, _) in &group {
            *histogram.entry(accepted).or_default() += 1;
        }
        let committed: f64 = group
            .iter()
            .map(|&(_, committed, _)| committed as f64)
            .sum::<f64>()
            / group.len() as f64;
        let wall_median = median(&walls);
        medians.insert((source.clone(), variant.clone(), drafts), wall_median);
        rows.push(json!({
            "source": source,
            "variant": variant,
            "drafts": drafts,
            "actual_drafted": drafts,
            "windows": group.len(),
            "wall_ms_median": wall_median,
            "wall_ms_all": walls,
            "accepted_histogram": histogram,
            "mean_committed": committed,
        }));
    }
    (rows, medians)
}

/// Compose windows from measured components only. `head_ms` maps a head
/// chain depth to its median step-chain cost. `production` maps
/// `(source, variant, actual drafts)` to the real window median (for the residual).
///
/// Mtp: head(depth k) + verify(k+1) + commit(accept_len).
/// Ngram confirmed (confirmation requested): head(depth 1) + verify + commit.
/// Ngram unconfirmed (confirmation skipped, not failed): verify + commit.
/// The host n-gram lookup is not measured by the isolated components: it is
/// listed as unmeasured and remains part of the unattributed residual.
pub fn compose_windows(
    widths: &[WidthCost],
    head_ms: &BTreeMap<usize, f64>,
    production: &BTreeMap<(String, Option<String>, usize), f64>,
) -> Vec<Value> {
    let mut out = Vec::new();
    for width in widths.iter().filter(|w| w.n >= 2) {
        let k = width.n - 1;
        for (&accept_len, commit) in &width.commit_ms {
            for (source, variant, head_depth) in [
                ("Mtp", None, Some(k)),
                ("Ngram", Some("confirmed"), Some(1)),
                ("Ngram", Some("unconfirmed"), None),
            ] {
                let head = head_depth.map(|depth| head_ms.get(&depth).copied());
                let mut unmeasured: Vec<String> = Vec::new();
                if let Some(None) = head {
                    unmeasured.push(format!("head_chain_depth_{}", head_depth.unwrap_or(0)));
                }
                if commit.is_none() {
                    unmeasured.push(format!("commit_accept_{accept_len}"));
                }
                if source == "Ngram" {
                    unmeasured.push("ngram_lookup_host".to_string());
                }
                let blocking = unmeasured.iter().any(|name| name != "ngram_lookup_host");
                let component_subtotal_ms = if blocking {
                    None
                } else {
                    Some(head.flatten().unwrap_or(0.0) + width.verify_ms + commit.unwrap_or(0.0))
                };
                let key = (source.to_string(), variant.map(str::to_string), k);
                let prod = production.get(&key).copied();
                let residual = match (prod, component_subtotal_ms) {
                    (Some(p), Some(w)) => Some(p - w),
                    _ => None,
                };
                // These isolated boundaries do not include the native head
                // reconciliation, acceptance/pick readback or controller.
                // A subtotal is never a complete production window price.
                unmeasured.extend([
                    "native_head_sync".to_string(),
                    "acceptance_pick_readback".to_string(),
                    "controller".to_string(),
                ]);
                out.push(json!({
                    "source": source,
                    "variant": variant,
                    "k": k,
                    "n": width.n,
                    "accept_len": accept_len,
                    "components": {
                        "head_ms": head.flatten(),
                        "verify_ms": width.verify_ms,
                        "commit_ms": commit,
                    },
                    "unmeasured": unmeasured,
                    "lookup_included": false,
                    "component_subtotal_ms": component_subtotal_ms,
                    "window_ms": null,
                    "complete": false,
                    "basis": "isolated_component_subtotal_not_a_production_window_price",
                    "production_window_ms_median": prod,
                    "unattributed_residual_ms": residual,
                }));
            }
        }
    }
    out
}

/// `(n, clean)` per requested width in request order: whether every width up
/// to the largest clean prefix passed, and the first failing width.
pub fn strict_summary(cells: &[(usize, bool)]) -> Value {
    let mut sorted = cells.to_vec();
    sorted.sort_by_key(|(n, _)| *n);
    let first_failure = sorted.iter().find(|(_, clean)| !clean).map(|(n, _)| *n);
    let mut qualified_through = None;
    for &(n, clean) in &sorted {
        if !clean || n != qualified_through.unwrap_or(0) + 1 {
            break;
        }
        qualified_through = Some(n);
    }
    json!({
        "all_clean": first_failure.is_none() && !sorted.is_empty(),
        "first_failure": first_failure,
        "qualified_through": qualified_through,
        "widths": sorted.iter().map(|(n, clean)| json!({"n": n, "clean": clean})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn kernel_spans_sum_in_f64_instead_of_averaging() {
        assert_eq!(kernel_span_ms(&[1.0, 2.0, 3.0, 0.0], 4), Ok(6.0));
        assert_eq!(
            kernel_span_ms(&[f32::MAX, f32::MAX], 2),
            Ok(2.0 * f64::from(f32::MAX))
        );
    }

    #[test]
    fn kernel_spans_require_a_nonempty_matching_count() {
        assert!(kernel_span_ms(&[], 0).is_err());
        assert!(kernel_span_ms(&[], 1).is_err());
        assert!(kernel_span_ms(&[1.0], 0).is_err());
        assert!(kernel_span_ms(&[1.0], 2).is_err());
        assert!(kernel_span_ms(&[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn kernel_spans_reject_invalid_values_and_zero_totals() {
        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(kernel_span_ms(&[1.0, invalid], 2).is_err());
        }
        assert!(kernel_span_ms(&[0.0, -0.0], 2).is_err());
        assert_eq!(kernel_span_ms(&[-0.0, 1.0], 2), Ok(1.0));
    }

    #[cfg(feature = "kernel-timing")]
    fn timing_report(
        per_kernel_ms: Vec<f32>,
        kernel_count: usize,
    ) -> hip_bridge::KernelTimingReport {
        hip_bridge::KernelTimingReport {
            per_kernel_ms,
            kernel_count,
            coverage: hip_bridge::KernelTimingCoverage::EagerHipLaunchWrappers,
        }
    }

    #[cfg(feature = "kernel-timing")]
    #[test]
    fn kernel_report_sums_spans_and_checks_capacity() {
        let report = timing_report(vec![1.0, 2.0, 3.0], 3);
        assert_eq!(kernel_report_span_ms(&report, 34), Ok(6.0));
        // Exact fit is allowed; anything over capacity is refused.
        assert_eq!(kernel_report_span_ms(&report, 3), Ok(6.0));
        assert_eq!(
            kernel_report_span_ms(&report, 2),
            Err("GPU kernel timing report exceeds prepared event capacity".to_string())
        );
    }

    #[cfg(feature = "kernel-timing")]
    #[test]
    fn kernel_report_rejects_mismatched_counts_and_bad_spans() {
        // Count must be nonzero and match the span length.
        assert!(kernel_report_span_ms(&timing_report(vec![], 0), 34).is_err());
        assert!(kernel_report_span_ms(&timing_report(vec![], 1), 34).is_err());
        assert!(kernel_report_span_ms(&timing_report(vec![1.0], 0), 34).is_err());
        assert!(kernel_report_span_ms(&timing_report(vec![1.0, 2.0], 1), 34).is_err());
        // Every span must be finite and nonnegative; the total positive.
        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(kernel_report_span_ms(&timing_report(vec![1.0, invalid], 2), 34).is_err());
        }
        assert!(kernel_report_span_ms(&timing_report(vec![0.0, -0.0], 2), 34).is_err());
        // Single-variant coverage enum: only the eager-wrapper shape exists,
        // so any future variant is a compile-time refusal, not a runtime pass.
        let report = timing_report(vec![1.0], 1);
        assert!(matches!(
            report.coverage,
            hip_bridge::KernelTimingCoverage::EagerHipLaunchWrappers
        ));
        assert_eq!(kernel_report_span_ms(&report, 34), Ok(1.0));
    }

    #[test]
    fn kernel_timing_capacity_requires_a_positive_census() {
        assert!(kernel_timing_capacity(0).is_err());
        assert_eq!(kernel_timing_capacity(1), Ok(34));
        assert_eq!(kernel_timing_capacity(100), Ok(232));
    }

    #[test]
    fn kernel_timing_capacity_checks_conversion_and_arithmetic_portably() {
        if let Ok(max_count) = u64::try_from(usize::MAX) {
            // Doubling the maximum overflows; doubling half fits, but adding 32 does not.
            assert!(kernel_timing_capacity(max_count).is_err());
            assert!(kernel_timing_capacity(max_count / 2).is_err());
            let largest_safe = (max_count - 32) / 2;
            assert_eq!(
                kernel_timing_capacity(largest_safe),
                Ok(usize::try_from(largest_safe).unwrap() * 2 + 32)
            );
            assert!(kernel_timing_capacity(largest_safe + 1).is_err());
            if let Some(unrepresentable) = max_count.checked_add(1) {
                assert_eq!(
                    kernel_timing_capacity(unrepresentable).unwrap_err(),
                    "kernel timing launch census does not fit usize"
                );
            }
        }
    }

    #[test]
    fn timed_separates_the_warmup_and_requires_a_measured_repeat() {
        assert!(Timed::from_samples(&[]).is_none());
        assert!(Timed::from_samples(&[9.0]).is_none());
        let timed = Timed::from_samples(&[100.0, 3.0, 1.0, 2.0]).unwrap();
        assert_eq!(timed.warmup_ms, 100.0);
        assert_eq!(timed.repeats, vec![3.0, 1.0, 2.0]);
        assert_eq!(timed.median(), 2.0);
        assert_eq!(timed.min(), 1.0);
        let json = timed.json();
        assert_eq!(json["median"], json!(2.0));
        assert_eq!(json["warmup_ms"], json!(100.0));
        assert_eq!(json["repeats"], json!(3));
    }

    #[test]
    fn width_labels_never_claim_identity_beyond_the_qualified_rows() {
        assert_eq!(width_label(8, 8), "identity-qualified");
        assert_eq!(width_label(9, 8), "timing-only-beyond-qualified");
        assert!(identity_claim(8, 8, true, true));
        assert!(!identity_claim(9, 8, true, true));
        assert!(!identity_claim(4, 8, false, true));
        assert!(!identity_claim(4, 8, true, false));
    }

    #[test]
    fn hard_width_rules_per_profile() {
        assert!(Profile::Oracle.hard_width(8, 32, 8));
        assert!(!Profile::Oracle.hard_width(9, 32, 8));
        assert!(Profile::Cost.hard_width(32, 32, 8));
        assert!(!Profile::Cost.hard_width(33, 32, 8));
        assert!(Profile::Strict.hard_width(64, 8, 8));
    }

    #[test]
    fn deadline_and_option_parsers_fail_closed() {
        assert_eq!(parse_deadline_secs("1500"), Ok(1500));
        assert!(parse_deadline_secs("0").is_err());
        assert!(parse_deadline_secs("1751").is_err());
        assert!(parse_deadline_secs("1800").is_err());
        assert!(parse_deadline_secs("x").is_err());
        assert_eq!(parse_on_off("--commit", "on"), Ok(true));
        assert!(parse_on_off("--commit", "yes").is_err());
        assert_eq!(CostMode::parse("OnCopy"), Ok(CostMode::OnCopy));
        assert!(CostMode::parse("copy").is_err());
        assert_eq!(CostMode::of_copy_lane(true).key(), "OnCopy");
        assert_eq!(CostMode::of_copy_lane(false).key(), "OffPool");
        let live = Deadline::new(Instant::now(), 1);
        assert!(!live.exceeded());
        let spent = Deadline::new(Instant::now() - Duration::from_secs(5), 1);
        assert!(spent.exceeded());
        assert_eq!(spent.json()["exceeded"], json!(true));
    }

    #[test]
    fn buckets_and_cards_follow_the_runtime_vocabulary() {
        assert_eq!(context_bucket(2048), "Le2K");
        assert_eq!(context_bucket(2049), "Le8K");
        assert_eq!(context_bucket(16384), "Le16K");
        assert_eq!(context_bucket(16385), "Over16K");
        assert_eq!(card_of_arch("gfx1201"), Some("R9700"));
        assert_eq!(card_of_arch("gfx1151"), Some("Halo"));
        assert_eq!(card_of_arch("gfx942"), None);
        assert_eq!(per_row_ms(32.0, 8), 4.0);
    }

    #[test]
    fn missing_widths_and_strict_summary() {
        assert_eq!(missing_widths(&[1, 2, 32, 64], &[1, 2, 32]), vec![64]);
        assert!(missing_widths(&[1], &[1]).is_empty());
        let summary = strict_summary(&[(2, true), (1, true), (9, false), (16, true)]);
        assert_eq!(summary["all_clean"], json!(false));
        assert_eq!(summary["first_failure"], json!(9));
        assert_eq!(summary["qualified_through"], json!(2));
        let clean = strict_summary(&[(1, true), (2, true)]);
        assert_eq!(clean["all_clean"], json!(true));
        assert_eq!(clean["qualified_through"], json!(2));
        assert_eq!(strict_summary(&[])["all_clean"], json!(false));
        let shard = strict_summary(&[(64, true)]);
        assert_eq!(shard["all_clean"], json!(true));
        assert_eq!(shard["qualified_through"], Value::Null);
        assert_eq!(
            strict_summary(&[(1, true), (3, true)])["qualified_through"],
            json!(1)
        );
    }

    fn windows_fixture() -> Vec<WidthCost> {
        vec![
            WidthCost {
                n: 1,
                verify_ms: 1.0,
                commit_ms: BTreeMap::from([(0, Some(0.1))]),
            },
            WidthCost {
                n: 4,
                verify_ms: 10.0,
                commit_ms: BTreeMap::from([(0, Some(2.0)), (3, None)]),
            },
        ]
    }

    #[test]
    fn composition_sums_only_measured_components_and_never_assumes_zero() {
        let head = BTreeMap::from([(3usize, 5.0f64), (1usize, 1.5f64)]);
        let prod = BTreeMap::from([
            (("Mtp".to_string(), None, 3usize), 19.0f64),
            (
                ("Ngram".to_string(), Some("confirmed".to_string()), 3),
                23.0,
            ),
            (
                ("Ngram".to_string(), Some("unconfirmed".to_string()), 3),
                29.0,
            ),
        ]);
        let windows = compose_windows(&windows_fixture(), &head, &prod);
        // n = 1 has no drafted rows; n = 4 yields 2 accepts x 3 variants.
        assert_eq!(windows.len(), 6);
        let find = |source: &str, variant: Option<&str>, accept: usize| {
            windows
                .iter()
                .find(|w| {
                    w["source"] == json!(source)
                        && w["variant"] == json!(variant)
                        && w["accept_len"] == json!(accept)
                })
                .unwrap()
                .clone()
        };
        let mtp = find("Mtp", None, 0);
        assert_eq!(mtp["k"], json!(3));
        assert_eq!(mtp["component_subtotal_ms"], json!(17.0));
        assert_eq!(mtp["window_ms"], Value::Null);
        assert_eq!(mtp["complete"], json!(false));
        assert_eq!(mtp["production_window_ms_median"], json!(19.0));
        assert_eq!(mtp["unattributed_residual_ms"], json!(2.0));
        let confirmed = find("Ngram", Some("confirmed"), 0);
        assert_eq!(confirmed["component_subtotal_ms"], json!(13.5));
        assert_eq!(confirmed["window_ms"], Value::Null);
        assert_eq!(confirmed["lookup_included"], json!(false));
        assert_eq!(confirmed["production_window_ms_median"], json!(23.0));
        assert_eq!(confirmed["unattributed_residual_ms"], json!(9.5));
        assert!(confirmed["unmeasured"]
            .as_array()
            .unwrap()
            .contains(&json!("ngram_lookup_host")));
        let unconfirmed = find("Ngram", Some("unconfirmed"), 0);
        assert_eq!(unconfirmed["component_subtotal_ms"], json!(12.0));
        assert_eq!(unconfirmed["window_ms"], Value::Null);
        assert_eq!(unconfirmed["production_window_ms_median"], json!(29.0));
        assert_eq!(unconfirmed["unattributed_residual_ms"], json!(17.0));
        assert!(windows
            .iter()
            .all(|w| w["window_ms"].is_null() && w["complete"] == false));
        // Commit not measured: no total, never 0.
        let missing = find("Mtp", None, 3);
        assert_eq!(missing["window_ms"], Value::Null);
        assert_eq!(missing["complete"], json!(false));
        assert!(missing["unmeasured"]
            .as_array()
            .unwrap()
            .contains(&json!("commit_accept_3")));
        // No head chain depth 7: a window needing it has no total.
        let wide = compose_windows(
            &[WidthCost {
                n: 8,
                verify_ms: 20.0,
                commit_ms: BTreeMap::from([(0, Some(1.0))]),
            }],
            &head,
            &BTreeMap::new(),
        );
        let mtp_wide = wide.iter().find(|w| w["source"] == json!("Mtp")).unwrap();
        assert_eq!(mtp_wide["window_ms"], Value::Null);
        assert_eq!(mtp_wide["components"]["head_ms"], Value::Null);
        assert_eq!(mtp_wide["unattributed_residual_ms"], Value::Null);
    }

    fn production_window(
        source: &'static str,
        variant: Option<&'static str>,
        drafts: usize,
        accepted: usize,
        wall_ms: f64,
    ) -> ProdWindow {
        ProdWindow {
            request_id: Some(format!("prompt-0:{source}:K3:repeat0")),
            requested_source: source,
            source,
            variant,
            requested_confirmation: (source == "Ngram").then_some(variant == Some("confirmed")),
            head_confirmation: (source == "Ngram").then_some(false),
            requested_k: 3,
            position: 100,
            drafts: Some(drafts),
            accepted: Some(accepted),
            committed: Some(accepted + 1),
            wall_ms: Some(wall_ms),
            terminal: Some(false),
            terminal_reason: None,
            rejected: Some(accepted < drafts),
            terminal_censored: false,
            native_draft_margins: (source == "Mtp" && drafts > 0)
                .then(|| vec![Some(1.25f32); drafts])
                .unwrap_or_default(),
            complete: true,
            warmup: false,
            // Arithmetic fixtures do not claim an observed execution route.
            verify_route: Value::Null,
            offer_available: Some(drafts > 0),
            missing_reason: None,
        }
    }

    #[test]
    fn production_windows_group_by_source_variant_and_actual_not_requested_width() {
        let mut changed_request = production_window("Mtp", None, 3, 0, 14.0);
        changed_request.requested_k = 63;
        let (rows, medians) = aggregate_production_windows(&[
            production_window("Mtp", None, 3, 3, 10.0),
            changed_request,
            production_window("Mtp", None, 3, 3, 12.0),
            production_window("Mtp", None, 1, 1, 5.0),
            production_window("Ngram", Some("confirmed"), 3, 2, 21.0),
            production_window("Ngram", Some("unconfirmed"), 3, 2, 31.0),
            production_window("Ngram", Some("confirmed"), 8, 5, 41.0),
            production_window("ArRoutedK0", None, 0, 0, 2.0),
        ]);
        assert_eq!(rows.len(), 6);
        assert_eq!(medians[&("Mtp".to_string(), None, 3)], 12.0);
        assert_eq!(medians[&("Mtp".to_string(), None, 1)], 5.0);
        assert!(!medians.contains_key(&("Mtp".to_string(), None, 63)));
        let mtp = rows
            .iter()
            .find(|r| r["source"] == "Mtp" && r["actual_drafted"] == 3)
            .unwrap();
        assert_eq!(mtp["variant"], Value::Null);
        assert_eq!(mtp["windows"], json!(3));
        assert_eq!(mtp["wall_ms_all"], json!([10.0, 14.0, 12.0]));
        assert_eq!(mtp["accepted_histogram"], json!({"0": 1, "3": 2}));
        assert_eq!(mtp["mean_committed"], json!(3.0));
        assert_eq!(
            medians[&("Ngram".to_string(), Some("confirmed".to_string()), 3)],
            21.0
        );
        assert_eq!(
            medians[&("Ngram".to_string(), Some("unconfirmed".to_string()), 3)],
            31.0
        );
        assert_eq!(
            medians[&("Ngram".to_string(), Some("confirmed".to_string()), 8)],
            41.0
        );
        assert_eq!(medians[&("ArRoutedK0".to_string(), None, 0)], 2.0);
    }

    #[test]
    fn production_aggregation_excludes_warmup_incomplete_missing_censored_and_unknown() {
        let good = production_window("Mtp", None, 3, 2, 10.0);
        let mut observations = vec![good.clone()];
        let mut warmup = good.clone();
        warmup.warmup = true;
        warmup.wall_ms = Some(900.0);
        assert_eq!(warmup.json()["warmup"], json!(true));
        observations.push(warmup);
        let mut incomplete = good.clone();
        incomplete.complete = false;
        observations.push(incomplete);
        let mut missing = good.clone();
        missing.missing_reason = Some("unsupported hook".to_string());
        observations.push(missing);
        let mut censored = good.clone();
        censored.terminal_censored = true;
        observations.push(censored);
        let mut unknown = good.clone();
        unknown.drafts = None;
        observations.push(unknown);
        let mut unknown = good.clone();
        unknown.accepted = None;
        observations.push(unknown);
        let mut unknown = good.clone();
        unknown.committed = None;
        observations.push(unknown);
        let mut unknown = good.clone();
        unknown.offer_available = None;
        observations.push(unknown);
        for wall_ms in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            let mut invalid = good.clone();
            invalid.wall_ms = wall_ms;
            observations.push(invalid);
        }
        let (rows, medians) = aggregate_production_windows(&observations);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["windows"], json!(1));
        assert_eq!(rows[0]["wall_ms_all"], json!([10.0]));
        assert_eq!(medians[&("Mtp".to_string(), None, 3)], 10.0);
        let (rows, medians) = aggregate_production_windows(&observations[1..]);
        assert!(rows.is_empty());
        assert!(medians.is_empty());
    }

    #[test]
    fn production_aggregation_stays_scoped_to_each_proposal_mode_book() {
        let books: BTreeMap<_, _> = [(CostMode::OffPool, 11.0), (CostMode::OnCopy, 22.0)]
            .into_iter()
            .map(|(mode, wall_ms)| {
                let (_, medians) =
                    aggregate_production_windows(&[production_window("Mtp", None, 3, 2, wall_ms)]);
                (mode.key(), medians)
            })
            .collect();
        let key = ("Mtp".to_string(), None, 3);
        assert_eq!(books["OffPool"][&key], 11.0);
        assert_eq!(books["OnCopy"][&key], 22.0);
    }

    #[test]
    fn actual_window_json_preserves_exact_schema_context_route_and_units() {
        assert_eq!(ACTUAL_WINDOWS_SCHEMA, "qwen4_wide_verify.actual_windows.v1");
        let mut window = production_window("Ngram", Some("confirmed"), 2, 1, 1.25);
        window.request_id = Some("prompt-7:Ngram:K3:repeat0".to_string());
        window.warmup = true;
        window.verify_route = json!({"kind": "actual", "rows": 3});
        let context = json!({
            "n_ctx": 100, "position": 99999, "bucket": "Over16K",
            "prompt": {"digest": "same-s0"},
            "additional_identity": [1, 2, 3],
        });
        let mut expected_context = context.clone();
        expected_context["position"] = json!(100);
        expected_context["bucket"] = json!("Le2K");
        let actual = window.actual_json(7, &context);
        assert_eq!(
            actual,
            json!({
                "request_id": "prompt-7:Ngram:K3:repeat0",
                "prompt_index": 7,
                "position_start": 100,
                "context": expected_context,
                "requested_source": "Ngram",
                "source": "Ngram",
                "variant": "confirmed",
                "requested_confirmation": true,
                "head_confirmation": false,
                "verify_route": {"kind": "actual", "rows": 3},
                "requested_drafted": 3,
                "actual_drafted": 2,
                "accepted": 1,
                "committed": 2,
                "terminal": false,
                "terminal_reason": null,
                "rejected": true,
                "terminal_censored": false,
                "native_draft_margins": json!([]),
                "total_window_us": 1250.0,
                "complete": true,
                "offer_available": true,
                "missing_reason": null,
                "warmup": true,
            })
        );
        let legacy = window.json();
        assert_eq!(legacy["requested_k"], actual["requested_drafted"]);
        assert_eq!(legacy["drafts"], actual["actual_drafted"]);
        assert_eq!(legacy["wall_ms"], json!(1.25));
        assert_eq!(legacy["total_window_us"], json!(1250.0));
        assert_eq!(legacy["verify_route"], actual["verify_route"]);
        assert_eq!(legacy["requested_confirmation"], json!(true));
        assert_eq!(legacy["head_confirmation"], json!(false));
        for name in [
            "request_id",
            "position_start",
            "terminal",
            "terminal_reason",
            "rejected",
            "native_draft_margins",
        ] {
            assert_eq!(legacy[name], actual[name]);
        }
        let mut changed_context = context.clone();
        changed_context["n_ctx"] = json!(200);
        assert_eq!(actual["context"]["n_ctx"], json!(100));
        assert_eq!(actual["context"]["position"], json!(100));
        assert_eq!(actual["context"]["bucket"], json!("Le2K"));
        assert_eq!(context["position"], json!(99999));
        assert_eq!(context["bucket"], json!("Over16K"));
    }

    #[test]
    fn actual_window_json_keeps_requested_source_separate_from_observed_source() {
        let mut window = production_window("Mtp", None, 0, 0, 1.5);
        window.source = "ArRoutedK0";
        window.complete = false;
        window.missing_reason = Some("requested Mtp, observed ArRoutedK0".to_string());
        let actual = window.actual_json(0, &json!({"n_ctx": 100}));
        let raw = window.json();
        assert_eq!(actual["requested_source"], json!("Mtp"));
        assert_eq!(actual["source"], json!("ArRoutedK0"));
        assert_eq!(actual["request_id"], json!("prompt-0:Mtp:K3:repeat0"));
        assert_eq!(actual["requested_drafted"], json!(3));
        assert_eq!(actual["actual_drafted"], json!(0));
        assert_eq!(actual["accepted"], json!(0));
        assert_eq!(actual["committed"], json!(1));
        assert_eq!(actual["total_window_us"], json!(1500.0));
        assert_eq!(actual["complete"], json!(false));
        assert_eq!(
            actual["missing_reason"],
            json!("requested Mtp, observed ArRoutedK0")
        );
        for name in [
            "requested_source",
            "source",
            "request_id",
            "actual_drafted",
            "accepted",
            "committed",
            "total_window_us",
            "complete",
            "missing_reason",
            "native_draft_margins",
        ] {
            assert_eq!(raw[name], actual[name]);
        }
        assert!(aggregate_production_windows(&[window]).0.is_empty());
    }

    #[test]
    fn actual_window_json_preserves_eos_or_end_of_turn_without_calling_it_a_rejection() {
        let mut window = production_window("Mtp", None, 3, 1, 2.5);
        window.terminal = Some(true);
        window.terminal_reason = Some("eos-or-end-of-turn");
        window.rejected = Some(false);
        window.terminal_censored = true;
        let actual = window.actual_json(0, &json!({"n_ctx": 100}));
        let raw = window.json();
        assert_eq!(actual["request_id"], json!("prompt-0:Mtp:K3:repeat0"));
        assert_eq!(actual["terminal"], json!(true));
        assert_eq!(actual["terminal_reason"], json!("eos-or-end-of-turn"));
        assert_eq!(actual["rejected"], json!(false));
        assert_eq!(actual["terminal_censored"], json!(true));
        for name in [
            "request_id",
            "terminal",
            "terminal_reason",
            "rejected",
            "terminal_censored",
            "native_draft_margins",
        ] {
            assert_eq!(raw[name], actual[name]);
        }
        assert!(aggregate_production_windows(&[window]).0.is_empty());
    }

    #[test]
    fn unsupported_actual_window_json_keeps_observed_counts_and_offer_unknown() {
        let mut window = production_window("Ngram", None, 0, 0, 1.0);
        window.request_id = None;
        window.requested_confirmation = Some(true);
        window.head_confirmation = None;
        window.drafts = None;
        window.accepted = None;
        window.committed = None;
        window.wall_ms = None;
        window.terminal = None;
        window.terminal_reason = None;
        window.rejected = None;
        window.complete = false;
        window.offer_available = None;
        window.missing_reason = Some("unsupported hook".to_string());
        let actual = window.actual_json(0, &json!({"n_ctx": 100}));
        for name in [
            "request_id",
            "variant",
            "head_confirmation",
            "actual_drafted",
            "accepted",
            "committed",
            "total_window_us",
            "offer_available",
            "terminal",
            "terminal_reason",
            "rejected",
        ] {
            assert!(actual.as_object().unwrap().contains_key(name));
            assert_eq!(actual[name], Value::Null);
        }
        assert_eq!(actual["requested_confirmation"], json!(true));
        assert_eq!(actual["request_id"], Value::Null);
        let raw = window.json();
        for name in [
            "request_id",
            "terminal",
            "terminal_reason",
            "rejected",
            "native_draft_margins",
        ] {
            assert_eq!(raw[name], actual[name]);
        }
        assert_eq!(actual["native_draft_margins"], json!([]));
        assert_eq!(actual["requested_drafted"], json!(3));
        assert_eq!(actual["complete"], json!(false));
        assert_eq!(actual["missing_reason"], json!("unsupported hook"));
        assert_eq!(window.json()["actual_drafted"], Value::Null);
        assert!(aggregate_production_windows(&[window]).0.is_empty());
    }

    #[test]
    fn native_margins_render_per_actual_draft_with_nulls_never_fabricated() {
        assert_eq!(
            native_margins_json("Mtp", &[Some(0.25), Some(1.5), None]),
            json!([0.25, 1.5, null])
        );
        // Nonfinite and negative observations stay null, never a zero bin.
        assert_eq!(
            native_margins_json(
                "Mtp",
                &[
                    Some(f32::NAN),
                    Some(f32::INFINITY),
                    Some(-0.5),
                    Some(f32::NEG_INFINITY)
                ]
            ),
            json!([null, null, null, null])
        );
        // Negative zero is a real zero margin, not a negative one.
        assert_eq!(native_margins_json("Mtp", &[Some(-0.0)]), json!([-0.0]));
        // Other sources never carry margins, even if scratch leaked.
        assert_eq!(native_margins_json("Ngram", &[Some(1.0)]), json!([]));
        assert_eq!(native_margins_json("ArRoutedK0", &[Some(1.0)]), json!([]));
    }

    #[test]
    fn window_json_carries_native_margins_matching_actual_drafted() {
        let mtp = production_window("Mtp", None, 3, 2, 10.0);
        assert_eq!(mtp.native_draft_margins.len(), 3);
        let raw = mtp.json();
        assert_eq!(raw["native_draft_margins"], json!([1.25, 1.25, 1.25]));
        let actual = mtp.actual_json(0, &json!({"n_ctx": 100}));
        assert_eq!(actual["native_draft_margins"], json!([1.25, 1.25, 1.25]));
        // Non-Mtp windows emit an empty array, never copied margins.
        for source in ["Ngram", "ArRoutedK0"] {
            let other = production_window(source, None, 0, 0, 2.0);
            assert!(other.native_draft_margins.is_empty());
            assert_eq!(other.json()["native_draft_margins"], json!([]));
            assert_eq!(
                other.actual_json(0, &json!({"n_ctx": 100}))["native_draft_margins"],
                json!([])
            );
        }
        // Pricing never consumes the margin evidence.
        let (rows, _) = aggregate_production_windows(&[mtp.clone()]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].get("native_draft_margins").is_none());
    }
}
