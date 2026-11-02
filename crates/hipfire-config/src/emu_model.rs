//! CPU-only measured derating model; reports only (EMU-DESIGN.md §5).
//!
//! No HIP, no environment reads, no I/O. The model consumes measured kernel
//! observations plus host/target resource rates and produces a *report*; it
//! never rewrites a measured tok/s figure.
//!
//! Per classified kernel (design §5 step 3): with `F` arithmetic ops, `V` local
//! DRAM bytes and `P` host-mapped bytes,
//! `R_h = max(F/C_h, V/B_h, P/L_h)`, `R_t` the same with target rates, and
//! `t' = l + (t - l) * R_t / R_h` where `l = fixed_ns`. Step 4:
//! `T' = unscaled_wall + sum(t')`.
//!
//! Rate mapping (documented choice): `V` uses `read_bytes_s` as `B`, because a
//! [`KernelObservation`] carries no read/write split. `copy_rw_bytes_s` is carried
//! in [`ResourceRates`] (and scaled by [`spec_ratio_rates`]) but is not used by the
//! per-kernel formula.

use std::collections::BTreeSet;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::emu::EmuProfile;

/// Schema version carried by [`DeratingInput`]; checked by [`estimate`].
pub const SCHEMA_VERSION: u32 = 1;

/// Maximum |measured_wall - (unscaled_wall + sum t_i)| / measured_wall (percent).
const COVERAGE_TOLERANCE_PERCENT: u128 = 5;
/// Maximum unclassified kernel time / measured_wall (percent).
const UNCLASSIFIED_TOLERANCE_PERCENT: u128 = 10;

/// Errors for malformed input. Missing/incompatible *evidence* is not an error:
/// it yields an unavailable [`DeratingEstimate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ModelError {
    #[error("unsupported schema_version {found} (expected {expected})")]
    SchemaVersion { found: u32, expected: u32 },
    #[error("measured_wall_ns is zero")]
    ZeroMeasuredWall,
    #[error("checked arithmetic overflow: {0}")]
    Overflow(String),
    #[error("non-finite intermediate value: {0}")]
    NonFinite(String),
    #[error("rate {name} must be finite and > 0, got {value}")]
    InvalidRate { name: String, value: f64 },
    #[error("kernel {symbol}: unknown byte_basis {basis:?} (expected measured_dram | audited_streaming_proxy | unknown)")]
    InvalidByteBasis { symbol: String, basis: String },
    #[error("kernel {symbol}: {detail}")]
    InconsistentKernel { symbol: String, detail: String },
    #[error("identity field {field}: unknown reading without a reason")]
    MissingReason { field: String },
    #[error("identity field {field}: {detail}")]
    InvalidIdentity { field: String, detail: String },
}

/// A measurement that may be unknown. Unknown serializes as `value: null` plus a
/// `reason`; it is never encoded as zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reading<T> {
    pub value: Option<T>,
    #[serde(default)]
    pub reason: Option<String>,
}

impl<T> Reading<T> {
    pub fn known(value: T) -> Self {
        Self {
            value: Some(value),
            reason: None,
        }
    }

    pub fn unknown(reason: impl Into<String>) -> Self {
        Self {
            value: None,
            reason: Some(reason.into()),
        }
    }

    fn check(&self, field: &str) -> Result<(), ModelError> {
        if self.value.is_none() && self.reason.as_deref().map_or(true, |r| r.trim().is_empty()) {
            return Err(ModelError::MissingReason {
                field: field.to_string(),
            });
        }
        Ok(())
    }
}

/// Emulation limits recorded in the identity. `None` = that class is uncapped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmuLimitsRecord {
    pub device_bytes: Option<u64>,
    pub host_pinned_gtt_bytes: Option<u64>,
    pub host_pinned_userptr_bytes: Option<u64>,
    pub host_application_bytes: Option<u64>,
}

/// Route/config fields that select the executed code path and workload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteConfig {
    pub graph: bool,
    pub replay: String,
    pub spec: String,
    pub quant: String,
    pub context_tokens: u64,
    pub rows: u64,
}

/// Point-in-time telemetry. Varies run to run; excluded from identity matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Telemetry {
    pub sclk_mhz: Reading<u32>,
    pub mclk_mhz: Reading<u32>,
    pub power_w: Reading<f64>,
    pub temperature_c: Reading<f64>,
}

/// One identity contract shared by validate and model (design §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub source_revision: String,
    pub source_dirty_note: String,
    /// md5 of the executed binary.
    pub binary_md5: String,
    pub model_md5: Reading<String>,
    pub sidecar_md5: Reading<String>,
    /// `null` + reason for synthetic runs ("no text prompt").
    pub prompt_md5: Reading<String>,
    /// Effective token digest (deterministic input digest for synthetic runs).
    pub token_digest: String,
    pub pci_bdf: String,
    /// Physical (detected) gfx, never the emulated target.
    pub physical_gfx: String,
    pub physical_cus: u32,
    pub hip_mp_count: u32,
    pub rocm_version: String,
    pub driver_version: Reading<String>,
    /// `None` = no emulation profile active (not "unknown").
    pub emu_id: Option<String>,
    pub emu_limits: Option<EmuLimitsRecord>,
    pub route: RouteConfig,
    pub telemetry: Telemetry,
    pub calibration_receipt_digest: Reading<String>,
}

impl RunIdentity {
    /// Structural validation: every unknown reading carries a reason.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.model_md5.check("model_md5")?;
        self.sidecar_md5.check("sidecar_md5")?;
        self.prompt_md5.check("prompt_md5")?;
        self.driver_version.check("driver_version")?;
        self.telemetry.sclk_mhz.check("telemetry.sclk_mhz")?;
        self.telemetry.mclk_mhz.check("telemetry.mclk_mhz")?;
        self.telemetry.power_w.check("telemetry.power_w")?;
        self.telemetry
            .temperature_c
            .check("telemetry.temperature_c")?;
        self.calibration_receipt_digest
            .check("calibration_receipt_digest")?;
        if self.physical_cus == 0 {
            return Err(ModelError::InvalidIdentity {
                field: "physical_cus".into(),
                detail: "must be > 0".into(),
            });
        }
        if self.hip_mp_count == 0 {
            return Err(ModelError::InvalidIdentity {
                field: "hip_mp_count".into(),
                detail: "must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Names (with both values) of every identity field that differs. Telemetry
    /// readings and `calibration_receipt_digest` are excluded: they are
    /// per-run observations, not identity of the calibrated host/config.
    pub fn mismatches(&self, other: &RunIdentity) -> Vec<String> {
        let mut out = Vec::new();
        fn diff<T: PartialEq + Debug>(out: &mut Vec<String>, name: &str, a: &T, b: &T) {
            if a != b {
                out.push(format!("{name}: {a:?} != {b:?}"));
            }
        }
        diff(
            &mut out,
            "source_revision",
            &self.source_revision,
            &other.source_revision,
        );
        diff(
            &mut out,
            "source_dirty_note",
            &self.source_dirty_note,
            &other.source_dirty_note,
        );
        diff(&mut out, "binary_md5", &self.binary_md5, &other.binary_md5);
        diff(&mut out, "model_md5", &self.model_md5, &other.model_md5);
        diff(
            &mut out,
            "sidecar_md5",
            &self.sidecar_md5,
            &other.sidecar_md5,
        );
        diff(&mut out, "prompt_md5", &self.prompt_md5, &other.prompt_md5);
        diff(
            &mut out,
            "token_digest",
            &self.token_digest,
            &other.token_digest,
        );
        diff(&mut out, "pci_bdf", &self.pci_bdf, &other.pci_bdf);
        diff(
            &mut out,
            "physical_gfx",
            &self.physical_gfx,
            &other.physical_gfx,
        );
        diff(
            &mut out,
            "physical_cus",
            &self.physical_cus,
            &other.physical_cus,
        );
        diff(
            &mut out,
            "hip_mp_count",
            &self.hip_mp_count,
            &other.hip_mp_count,
        );
        diff(
            &mut out,
            "rocm_version",
            &self.rocm_version,
            &other.rocm_version,
        );
        diff(
            &mut out,
            "driver_version",
            &self.driver_version,
            &other.driver_version,
        );
        diff(&mut out, "emu_id", &self.emu_id, &other.emu_id);
        diff(&mut out, "emu_limits", &self.emu_limits, &other.emu_limits);
        diff(&mut out, "route", &self.route, &other.route);
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelObservation {
    pub symbol: String,
    /// Includes quant, rows, context, residency and route.
    pub shape_key: String,
    pub calls: u64,
    pub total_ns: u64,
    pub fixed_ns: Option<u64>,
    pub arithmetic_ops: Option<u64>,
    pub local_dram_bytes: Option<u64>,
    pub host_mapped_bytes: Option<u64>,
    /// `measured_dram` | `audited_streaming_proxy` | `unknown`.
    pub byte_basis: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRates {
    pub compute_ops_s: Option<f64>,
    pub read_bytes_s: Option<f64>,
    pub copy_rw_bytes_s: Option<f64>,
    pub host_read_bytes_s: Option<f64>,
}

/// Provenance of `target_rates`. Only `MeasuredOnTarget` can yield
/// [`EstimateQuality::TargetCalibrated`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RatesSource {
    /// Rates derived from specs/ratios ([`spec_ratio_rates`]); no target silicon.
    SpecRatioScenario,
    /// Rates measured on actual target hardware, backed by a calibration receipt.
    MeasuredOnTarget { receipt_digest: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeratingInput {
    pub schema_version: u32,
    /// Identity of the run being derated.
    pub identity: RunIdentity,
    /// Addition: identity of the run/config that produced `host_rates`. Must
    /// match `identity` (telemetry and calibration digest excepted).
    pub host_calibration: RunIdentity,
    pub measured_wall_ns: u64,
    pub unscaled_wall_ns: u64,
    pub host_rates: ResourceRates,
    pub target_rates: ResourceRates,
    /// Addition: provenance of `target_rates`.
    pub target_rates_source: RatesSource,
    pub kernels: Vec<KernelObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateQuality {
    TopologyScenario,
    TargetCalibrated,
}

/// Addition: per-bucket partial diagnostics (design §5 step 6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketDiagnostic {
    pub symbol: String,
    pub shape_key: String,
    pub measured_ns: u64,
    /// `measured_ns / measured_wall_ns`.
    pub measured_time_fraction: f64,
    pub classified: bool,
    /// Resource limiting the host side (`compute` | `local_dram` | `host_mapped`).
    pub host_limiter: Option<String>,
    pub target_limiter: Option<String>,
    /// `R_t / R_h`.
    pub scale_ratio: Option<f64>,
    /// `t'` (with `l = 0` when `fixed_ns` is unknown); `None` if not computable.
    pub estimated_ns: Option<f64>,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeratingEstimate {
    pub quality: EstimateQuality,
    pub estimated_wall_ns: Option<f64>,
    pub unclassified_time_fraction: f64,
    pub assumptions: Vec<String>,
    pub unavailable_reasons: Vec<String>,
    /// Addition: |measured - (unscaled + sum t_i)| / measured; `None` if not computed.
    pub coverage_error_fraction: Option<f64>,
    /// Addition: per-bucket diagnostics (empty when identity mismatch stopped
    /// the model before any arithmetic).
    pub buckets: Vec<BucketDiagnostic>,
}

fn check_rates(side: &str, rates: &ResourceRates) -> Result<(), ModelError> {
    for (field, v) in [
        ("compute_ops_s", rates.compute_ops_s),
        ("read_bytes_s", rates.read_bytes_s),
        ("copy_rw_bytes_s", rates.copy_rw_bytes_s),
        ("host_read_bytes_s", rates.host_read_bytes_s),
    ] {
        if let Some(v) = v {
            if !(v.is_finite() && v > 0.0) {
                return Err(ModelError::InvalidRate {
                    name: format!("{side}.{field}"),
                    value: v,
                });
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Basis {
    Known,
    Unknown,
}

fn parse_basis(k: &KernelObservation) -> Result<Basis, ModelError> {
    match k.byte_basis.as_str() {
        "measured_dram" | "audited_streaming_proxy" => Ok(Basis::Known),
        "unknown" => Ok(Basis::Unknown),
        other => Err(ModelError::InvalidByteBasis {
            symbol: k.symbol.clone(),
            basis: other.to_string(),
        }),
    }
}

fn ns_fraction(ns: u64, wall: u64) -> f64 {
    ns as f64 / wall as f64
}

fn finite(v: f64, what: &str) -> Result<f64, ModelError> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(ModelError::NonFinite(what.to_string()))
    }
}

fn overflow(what: &str) -> ModelError {
    ModelError::Overflow(what.to_string())
}

/// `max` over the resource terms; returns `(R, limiter)`. `None` if a needed
/// rate is absent.
fn max_term(
    terms: &[(&'static str, u64, Option<f64>)],
    what: &str,
) -> Result<Option<(f64, &'static str)>, ModelError> {
    let mut best: (f64, &'static str) = (0.0, "");
    for &(name, amount, rate) in terms {
        let v = if amount == 0 {
            0.0
        } else {
            match rate {
                Some(r) => finite(amount as f64 / r, what)?,
                None => return Ok(None),
            }
        };
        if v > best.0 || best.1.is_empty() {
            best = (v, name);
        }
    }
    Ok(Some(best))
}

/// Step 3–6 of design §5. See module docs.
pub fn estimate(input: &DeratingInput) -> Result<DeratingEstimate, ModelError> {
    if input.schema_version != SCHEMA_VERSION {
        return Err(ModelError::SchemaVersion {
            found: input.schema_version,
            expected: SCHEMA_VERSION,
        });
    }
    input.identity.validate()?;
    input.host_calibration.validate()?;
    if input.measured_wall_ns == 0 {
        return Err(ModelError::ZeroMeasuredWall);
    }
    check_rates("host_rates", &input.host_rates)?;
    check_rates("target_rates", &input.target_rates)?;

    let mut assumptions: Vec<String> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();

    // Quality: only an explicit measured-on-target source with a receipt digest.
    let quality = match &input.target_rates_source {
        RatesSource::SpecRatioScenario => {
            assumptions.push(
                "[INFERENCE] target_rates come from a spec-ratio scenario (no target silicon); \
                 target fidelity is unvalidated"
                    .to_string(),
            );
            EstimateQuality::TopologyScenario
        }
        RatesSource::MeasuredOnTarget { receipt_digest } => {
            let digest_ok = !receipt_digest.trim().is_empty();
            let id_digest = input.identity.calibration_receipt_digest.value.as_deref();
            let id_conflict = matches!(id_digest, Some(d) if d != receipt_digest);
            if !digest_ok {
                reasons.push(
                    "target_rates_source MeasuredOnTarget has an empty receipt_digest".into(),
                );
            }
            if id_conflict {
                reasons.push(format!(
                    "identity.calibration_receipt_digest {:?} != target_rates_source receipt_digest {:?}",
                    id_digest.unwrap_or_default(),
                    receipt_digest
                ));
            }
            if digest_ok && !id_conflict {
                assumptions.push(format!(
                    "target_rates measured on target hardware (receipt {receipt_digest}); \
                     TargetCalibrated also requires the §5-7 held-out validation recorded in that receipt"
                ));
                EstimateQuality::TargetCalibrated
            } else {
                EstimateQuality::TopologyScenario
            }
        }
    };

    // Identity before arithmetic.
    let mism = input.identity.mismatches(&input.host_calibration);
    if !mism.is_empty() {
        reasons.push(format!(
            "host_rates calibration identity differs from input identity: {}",
            mism.join("; ")
        ));
        return Ok(DeratingEstimate {
            quality,
            estimated_wall_ns: None,
            unclassified_time_fraction: 0.0,
            assumptions,
            unavailable_reasons: reasons,
            coverage_error_fraction: None,
            buckets: Vec::new(),
        });
    }

    let measured = input.measured_wall_ns;
    let hr = &input.host_rates;
    let tr = &input.target_rates;

    // Validate kernels and accumulate checked integer sums.
    let mut sum_ns: u64 = 0;
    let mut unclass_ns: u64 = 0;
    for k in &input.kernels {
        let basis = parse_basis(k)?;
        if let Some(l) = k.fixed_ns {
            if l > k.total_ns {
                return Err(ModelError::InconsistentKernel {
                    symbol: k.symbol.clone(),
                    detail: format!("fixed_ns {l} > total_ns {}", k.total_ns),
                });
            }
        }
        if k.calls == 0 && k.total_ns != 0 {
            return Err(ModelError::InconsistentKernel {
                symbol: k.symbol.clone(),
                detail: format!("calls == 0 but total_ns == {}", k.total_ns),
            });
        }
        sum_ns = sum_ns
            .checked_add(k.total_ns)
            .ok_or_else(|| overflow("sum of kernel total_ns"))?;
        if !is_classified(k, basis) {
            unclass_ns = unclass_ns
                .checked_add(k.total_ns)
                .ok_or_else(|| overflow("sum of unclassified total_ns"))?;
        }
    }
    let covered = input
        .unscaled_wall_ns
        .checked_add(sum_ns)
        .ok_or_else(|| overflow("unscaled_wall_ns + sum of kernel total_ns"))?;
    let cov_diff = covered.abs_diff(measured);
    let coverage_error = ns_fraction(cov_diff, measured);
    let unclassified_fraction = ns_fraction(unclass_ns, measured);

    // Per-bucket scaling.
    let mut buckets = Vec::with_capacity(input.kernels.len());
    let mut missing: BTreeSet<String> = BTreeSet::new();
    let mut sum_fixed_known = 0.0_f64; // classified kernels with known l + carried-unscaled kernels
    let mut sum_l0 = 0.0_f64; // unknown-l kernels at l = 0
    let mut sum_lt = 0.0_f64; // unknown-l kernels at l = t
    let mut unknown_fixed_kernels = 0_u64;
    let mut unknown_fixed_ns = 0_u64;
    let mut scalar_possible = true;

    for k in &input.kernels {
        let basis = parse_basis(k)?;
        let t = k.total_ns;
        let mut b = BucketDiagnostic {
            symbol: k.symbol.clone(),
            shape_key: k.shape_key.clone(),
            measured_ns: t,
            measured_time_fraction: ns_fraction(t, measured),
            classified: false,
            host_limiter: None,
            target_limiter: None,
            scale_ratio: None,
            estimated_ns: None,
            note: String::new(),
        };
        if !is_classified(k, basis) {
            b.note =
                "unclassified (byte_basis unknown or a traffic field missing); carried unscaled"
                    .into();
            b.estimated_ns = Some(t as f64);
            sum_fixed_known += t as f64;
            buckets.push(b);
            continue;
        }
        b.classified = true;
        let f = k.arithmetic_ops.unwrap_or(0);
        let v = k.local_dram_bytes.unwrap_or(0);
        let p = k.host_mapped_bytes.unwrap_or(0);
        if f == 0 && v == 0 && p == 0 {
            b.note = "zero resource demand; carried unscaled".into();
            b.estimated_ns = Some(t as f64);
            sum_fixed_known += t as f64;
            buckets.push(b);
            continue;
        }
        let host_terms = [
            ("compute", f, hr.compute_ops_s),
            ("local_dram", v, hr.read_bytes_s),
            ("host_mapped", p, hr.host_read_bytes_s),
        ];
        let target_terms = [
            ("compute", f, tr.compute_ops_s),
            ("local_dram", v, tr.read_bytes_s),
            ("host_mapped", p, tr.host_read_bytes_s),
        ];
        let rh = max_term(&host_terms, "R_h")?;
        let rt = max_term(&target_terms, "R_t")?;
        let (Some((r_h, lim_h)), Some((r_t, lim_t))) = (rh, rt) else {
            for (side, terms) in [("host", &host_terms), ("target", &target_terms)] {
                for &(name, amount, rate) in terms {
                    if amount != 0 && rate.is_none() {
                        let field = match name {
                            "compute" => "compute_ops_s",
                            "local_dram" => "read_bytes_s",
                            _ => "host_read_bytes_s",
                        };
                        missing.insert(format!("{side}_rates.{field}"));
                    }
                }
            }
            b.note = "missing rate for a class with nonzero traffic; not computable".into();
            scalar_possible = false;
            buckets.push(b);
            continue;
        };
        b.host_limiter = Some(lim_h.to_string());
        b.target_limiter = Some(lim_t.to_string());
        let ratio = finite(r_t / r_h, "R_t / R_h")?;
        b.scale_ratio = Some(ratio);
        match k.fixed_ns {
            Some(l) => {
                let tp = l as f64 + (t - l) as f64 * ratio;
                let tp = finite(tp, "t'")?;
                b.estimated_ns = Some(tp);
                sum_fixed_known += tp;
            }
            None => {
                let tp0 = finite(t as f64 * ratio, "t' (l = 0)")?;
                b.estimated_ns = Some(tp0);
                b.note =
                    "fixed_ns unknown; estimated_ns uses l = 0 (sensitivity: l = t gives t' = t)"
                        .into();
                sum_l0 += tp0;
                sum_lt += t as f64;
                unknown_fixed_kernels += 1;
                unknown_fixed_ns = unknown_fixed_ns
                    .checked_add(t)
                    .ok_or_else(|| overflow("sum of unknown-fixed total_ns"))?;
            }
        }
        buckets.push(b);
    }

    for m in &missing {
        reasons.push(format!(
            "{m} is absent but kernels have nonzero traffic in that class"
        ));
    }
    // Coverage / classification gates (integer-exact comparison).
    if (cov_diff as u128) * 100 > (measured as u128) * COVERAGE_TOLERANCE_PERCENT {
        reasons.push(format!(
            "wall coverage error {:.4}% exceeds {COVERAGE_TOLERANCE_PERCENT}% \
             (measured {measured} ns vs unscaled {} + kernels {sum_ns} ns)",
            coverage_error * 100.0,
            input.unscaled_wall_ns
        ));
    }
    if (unclass_ns as u128) * 100 > (measured as u128) * UNCLASSIFIED_TOLERANCE_PERCENT {
        reasons.push(format!(
            "unclassified time fraction {:.4}% exceeds {UNCLASSIFIED_TOLERANCE_PERCENT}%",
            unclassified_fraction * 100.0
        ));
    }
    if unclass_ns > 0 {
        assumptions.push(format!(
            "unclassified kernel time {unclass_ns} ns ({:.4}% of wall) is carried unscaled",
            unclassified_fraction * 100.0
        ));
    }

    let mut estimated: Option<f64> = None;
    if scalar_possible {
        let base = input.unscaled_wall_ns as f64 + sum_fixed_known;
        let central = finite(base + sum_l0, "T'")?;
        if unknown_fixed_kernels > 0 {
            let alt = finite(base + sum_lt, "T' (l = t)")?;
            assumptions.push(format!(
                "[SENSITIVITY] fixed_ns unknown for {unknown_fixed_kernels} classified kernel(s) \
                 ({unknown_fixed_ns} ns measured): estimated wall with l = 0 is {central:.1} ns; \
                 with l = t (those kernels unscaled) is {alt:.1} ns"
            ));
            if quality == EstimateQuality::TargetCalibrated {
                reasons.push(
                    "fixed_ns unknown for classified kernels; a target-calibrated claim must not \
                     assume l = 0 (see [SENSITIVITY] assumption for both bounds)"
                        .into(),
                );
            } else {
                assumptions.push(
                    "scalar estimate uses l = 0 (full scaling of the unknown-fixed kernels)".into(),
                );
            }
        }
        estimated = Some(central);
    }
    if !reasons.is_empty() {
        estimated = None;
    }

    Ok(DeratingEstimate {
        quality,
        estimated_wall_ns: estimated,
        unclassified_time_fraction: unclassified_fraction,
        assumptions,
        unavailable_reasons: reasons,
        coverage_error_fraction: Some(coverage_error),
        buckets,
    })
}

/// Classified = known byte basis and all three traffic fields measured (unknown is
/// never treated as zero).
fn is_classified(k: &KernelObservation, basis: Basis) -> bool {
    basis == Basis::Known
        && k.arithmetic_ops.is_some()
        && k.local_dram_bytes.is_some()
        && k.host_mapped_bytes.is_some()
}

/// Step 5: spec-ratio target rates when no target silicon exists.
///
/// `host_profile = (physical_cus, measured_host_clock_mhz, host_peak_bw_bytes_s)`
/// (bandwidth unit matches `EmuProfile::peak_bw_bytes_s`).
///
/// - `B_t = B_h * BW_peak_target / BW_peak_host` for read and copy (same-efficiency
///   scenario).
/// - `C_t = C_h * (CU_t * clock_t) / (CU_h * clock_h)`; `None` unless
///   `target_clock_mhz` is supplied (explicit clock choice required).
/// - `host_read_bytes_s` stays `None`: the registry carries no target PCIe/fabric link.
///
/// Every derived term is labelled `[INFERENCE]` in the returned assumptions.
pub fn spec_ratio_rates(
    host: &ResourceRates,
    host_profile: (u32, u32, u64),
    target: &EmuProfile,
    target_clock_mhz: Option<u32>,
) -> (ResourceRates, Vec<String>) {
    let (host_cus, host_clock_mhz, host_peak_bw) = host_profile;
    let mut notes: Vec<String> = Vec::new();
    let valid = |v: Option<f64>| v.filter(|x| x.is_finite() && *x > 0.0);

    // Bandwidth ratio.
    let bw_ratio = if host_peak_bw == 0 {
        notes.push(
            "[INFERENCE] host peak bandwidth is 0; target bandwidth rates unavailable".into(),
        );
        None
    } else {
        let r = target.peak_bw_bytes_s as f64 / host_peak_bw as f64;
        if r.is_finite() && r > 0.0 {
            Some(r)
        } else {
            notes.push("[INFERENCE] target/host peak bandwidth ratio is not finite and positive; bandwidth rates unavailable".into());
            None
        }
    };
    let scale_bw = |v: Option<f64>, label: &str, notes: &mut Vec<String>| -> Option<f64> {
        let (Some(v), Some(r)) = (valid(v), bw_ratio) else {
            notes.push(format!(
                "[INFERENCE] target {label} unavailable (host rate or bandwidth ratio missing)"
            ));
            return None;
        };
        let out = v * r;
        out.is_finite().then_some(out)
    };
    let read = scale_bw(host.read_bytes_s, "read_bytes_s", &mut notes);
    let copy = scale_bw(host.copy_rw_bytes_s, "copy_rw_bytes_s", &mut notes);
    if read.is_some() || copy.is_some() {
        notes.push(format!(
            "[INFERENCE] same-efficiency scenario: B_t = B_h * {} / {} (peak external BW target/host)",
            target.peak_bw_bytes_s, host_peak_bw
        ));
    }

    // Compute throughput scenario.
    let compute = match (target_clock_mhz, valid(host.compute_ops_s)) {
        (None, _) => {
            notes.push(
                "[INFERENCE] target compute_ops_s unavailable: throughput scenario requires an \
                 explicit target clock choice"
                    .into(),
            );
            None
        }
        (Some(_), None) => {
            notes.push(
                "[INFERENCE] target compute_ops_s unavailable: host compute rate missing".into(),
            );
            None
        }
        (Some(tc), Some(c_h)) => {
            if host_cus == 0 || host_clock_mhz == 0 || tc == 0 || target.target_cus == 0 {
                notes.push(
                    "[INFERENCE] target compute_ops_s unavailable: CU count or clock is zero"
                        .into(),
                );
                None
            } else {
                let num = target.target_cus as f64 * tc as f64;
                let den = host_cus as f64 * host_clock_mhz as f64;
                let out = c_h * (num / den);
                if out.is_finite() && out > 0.0 {
                    notes.push(format!(
                        "[INFERENCE] throughput scenario: C_t = C_h * ({} CU * {tc} MHz) / ({host_cus} CU * {host_clock_mhz} MHz); \
                         target clock is caller-supplied (if nominal boost, this is a nominal-boost scenario, \
                         not sustained compute); cross-die/cache/VGPR effects not captured",
                        target.target_cus
                    ));
                    Some(out)
                } else {
                    notes.push(
                        "[INFERENCE] target compute_ops_s unavailable: non-finite result".into(),
                    );
                    None
                }
            }
        }
    };

    notes.push(
        "[INFERENCE] target host_read_bytes_s left unset: target PCIe/fabric link is not established"
            .into(),
    );

    (
        ResourceRates {
            compute_ops_s: compute,
            read_bytes_s: read,
            copy_rw_bytes_s: copy,
            host_read_bytes_s: None,
        },
        notes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emu::{Fidelity, MemoryKind};

    fn identity() -> RunIdentity {
        RunIdentity {
            source_revision: "957b3b27b".into(),
            source_dirty_note: "clean".into(),
            binary_md5: "0123456789abcdef0123456789abcdef".into(),
            model_md5: Reading::known("aaaa".into()),
            sidecar_md5: Reading::unknown("no sidecar for this model"),
            prompt_md5: Reading::unknown("no text prompt"),
            token_digest: "tokdigest".into(),
            pci_bdf: "0000:03:00.0".into(),
            physical_gfx: "gfx1201".into(),
            physical_cus: 64,
            hip_mp_count: 32,
            rocm_version: "7.1".into(),
            driver_version: Reading::known("6.14".into()),
            emu_id: Some("rx9070xt".into()),
            emu_limits: Some(EmuLimitsRecord {
                device_bytes: Some(16 << 30),
                host_pinned_gtt_bytes: None,
                host_pinned_userptr_bytes: None,
                host_application_bytes: None,
            }),
            route: RouteConfig {
                graph: true,
                replay: "none".into(),
                spec: "off".into(),
                quant: "q4".into(),
                context_tokens: 4096,
                rows: 1,
            },
            telemetry: Telemetry {
                sclk_mhz: Reading::known(2900),
                mclk_mhz: Reading::unknown("sysfs unreadable"),
                power_w: Reading::known(180.0),
                temperature_c: Reading::known(60.0),
            },
            calibration_receipt_digest: Reading::unknown("host-only calibration"),
        }
    }

    fn kernel(
        symbol: &str,
        total_ns: u64,
        fixed_ns: Option<u64>,
        f: Option<u64>,
        v: Option<u64>,
        p: Option<u64>,
        basis: &str,
    ) -> KernelObservation {
        KernelObservation {
            symbol: symbol.into(),
            shape_key: format!("{symbol}|q4|rows1"),
            calls: 1,
            total_ns,
            fixed_ns,
            arithmetic_ops: f,
            local_dram_bytes: v,
            host_mapped_bytes: p,
            byte_basis: basis.into(),
        }
    }

    // Host: C_h = 1e12 ops/s, B_h = 5e11 B/s, L_h = 5e10 B/s.
    // Target: C_t = 8e11, B_t = 2.5e11, L_t = 4e10.
    fn host_rates() -> ResourceRates {
        ResourceRates {
            compute_ops_s: Some(1.0e12),
            read_bytes_s: Some(5.0e11),
            copy_rw_bytes_s: Some(4.5e11),
            host_read_bytes_s: Some(5.0e10),
        }
    }

    fn target_rates() -> ResourceRates {
        ResourceRates {
            compute_ops_s: Some(8.0e11),
            read_bytes_s: Some(2.5e11),
            copy_rw_bytes_s: Some(2.25e11),
            host_read_bytes_s: Some(4.0e10),
        }
    }

    fn input(kernels: Vec<KernelObservation>, unscaled: u64, measured: u64) -> DeratingInput {
        DeratingInput {
            schema_version: SCHEMA_VERSION,
            identity: identity(),
            host_calibration: identity(),
            measured_wall_ns: measured,
            unscaled_wall_ns: unscaled,
            host_rates: host_rates(),
            target_rates: target_rates(),
            target_rates_source: RatesSource::SpecRatioScenario,
            kernels,
        }
    }

    fn three_kernels() -> Vec<KernelObservation> {
        vec![
            // K1 memory-bound.
            kernel(
                "k1_gemv",
                2_000_000,
                Some(100_000),
                Some(1_000_000_000),
                Some(1_000_000_000),
                Some(0),
                "measured_dram",
            ),
            // K2 compute-bound.
            kernel(
                "k2_gemm",
                4_000_000,
                Some(0),
                Some(4_000_000_000),
                Some(500_000_000),
                Some(0),
                "measured_dram",
            ),
            // K3 host-read-bound.
            kernel(
                "k3_expert",
                4_000_000,
                Some(200_000),
                Some(100_000_000),
                Some(100_000_000),
                Some(200_000_000),
                "audited_streaming_proxy",
            ),
        ]
    }

    // Hand derivation (seconds):
    // K1: F=1e9, V=1e9, P=0.
    //   host:   F/C_h = 1e-3, V/B_h = 1e9/5e11 = 2e-3  -> R_h = 2e-3 (local_dram)
    //   target: F/C_t = 1.25e-3, V/B_t = 1e9/2.5e11 = 4e-3 -> R_t = 4e-3 (local_dram)
    //   ratio = 2; t = 2_000_000, l = 100_000
    //   t' = 100_000 + 1_900_000 * 2 = 3_900_000
    // K2: F=4e9, V=5e8, P=0.
    //   host:   F/C_h = 4e-3, V/B_h = 1e-3 -> R_h = 4e-3 (compute)
    //   target: F/C_t = 4e9/8e11 = 5e-3, V/B_t = 2e-3 -> R_t = 5e-3 (compute)
    //   ratio = 1.25; t = 4_000_000, l = 0
    //   t' = 0 + 4_000_000 * 1.25 = 5_000_000
    // K3: F=1e8, V=1e8, P=2e8.
    //   host:   F/C_h = 1e-4, V/B_h = 2e-4, P/L_h = 2e8/5e10 = 4e-3 -> R_h = 4e-3 (host_mapped)
    //   target: F/C_t = 1.25e-4, V/B_t = 4e-4, P/L_t = 2e8/4e10 = 5e-3 -> R_t = 5e-3 (host_mapped)
    //   ratio = 1.25; t = 4_000_000, l = 200_000
    //   t' = 200_000 + 3_800_000 * 1.25 = 4_950_000
    // sum t = 10_000_000; unscaled = 500_000; measured = 10_500_000 (coverage 0%).
    // T' = 500_000 + 3_900_000 + 5_000_000 + 4_950_000 = 14_350_000
    #[test]
    fn three_kernel_hand_computed() {
        let est = estimate(&input(three_kernels(), 500_000, 10_500_000)).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        assert_eq!(est.quality, EstimateQuality::TopologyScenario);
        let t = est.estimated_wall_ns.unwrap();
        assert!((t - 14_350_000.0).abs() < 1e-3, "{t}");
        assert_eq!(est.unclassified_time_fraction, 0.0);
        assert_eq!(est.coverage_error_fraction, Some(0.0));
        let e: Vec<f64> = est
            .buckets
            .iter()
            .map(|b| b.estimated_ns.unwrap())
            .collect();
        for (got, want) in e.iter().zip([3_900_000.0, 5_000_000.0, 4_950_000.0]) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
        let lim: Vec<_> = est
            .buckets
            .iter()
            .map(|b| b.host_limiter.clone().unwrap())
            .collect();
        assert_eq!(lim, ["local_dram", "compute", "host_mapped"]);
        assert!(est.assumptions.iter().any(|a| a.contains("[INFERENCE]")));
    }

    #[test]
    fn json_round_trip() {
        let inp = input(three_kernels(), 500_000, 10_500_000);
        let s = serde_json::to_string(&inp).unwrap();
        let back: DeratingInput = serde_json::from_str(&s).unwrap();
        assert_eq!(back, inp);
        // Unknown readings serialize as null + reason, not zero.
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        let sidecar = &v["identity"]["sidecar_md5"];
        assert!(sidecar["value"].is_null());
        assert_eq!(sidecar["reason"], "no sidecar for this model");

        let est = estimate(&inp).unwrap();
        let es = serde_json::to_string(&est).unwrap();
        let eb: DeratingEstimate = serde_json::from_str(&es).unwrap();
        // serde_json (without `float_roundtrip`) may parse an f64 one ulp off;
        // compare the fractions with a tolerance and everything else exactly.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(1.0);
        assert!(close(
            eb.unclassified_time_fraction,
            est.unclassified_time_fraction
        ));
        assert_eq!(eb.buckets.len(), est.buckets.len());
        let mut eb_norm = eb.clone();
        for (b, e) in eb_norm.buckets.iter_mut().zip(&est.buckets) {
            assert!(close(b.measured_time_fraction, e.measured_time_fraction));
            b.measured_time_fraction = e.measured_time_fraction;
        }
        eb_norm.unclassified_time_fraction = est.unclassified_time_fraction;
        assert_eq!(eb_norm, est);

        let src = RatesSource::MeasuredOnTarget {
            receipt_digest: "abc".into(),
        };
        let ss = serde_json::to_string(&src).unwrap();
        assert_eq!(serde_json::from_str::<RatesSource>(&ss).unwrap(), src);
    }

    #[test]
    fn schema_version_checked() {
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.schema_version = 2;
        assert_eq!(
            estimate(&inp).unwrap_err(),
            ModelError::SchemaVersion {
                found: 2,
                expected: SCHEMA_VERSION
            }
        );
    }

    #[test]
    fn unknown_reading_without_reason_rejected() {
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.identity.model_md5 = Reading {
            value: None,
            reason: None,
        };
        assert!(matches!(
            estimate(&inp),
            Err(ModelError::MissingReason { .. })
        ));
    }

    #[test]
    fn zero_and_overflow() {
        // Zero measured wall.
        let inp = input(three_kernels(), 500_000, 0);
        assert_eq!(estimate(&inp).unwrap_err(), ModelError::ZeroMeasuredWall);

        // Kernel-sum overflow.
        let ks = vec![
            kernel(
                "a",
                u64::MAX,
                None,
                Some(1),
                Some(1),
                Some(0),
                "measured_dram",
            ),
            kernel("b", 1, None, Some(1), Some(1), Some(0), "measured_dram"),
        ];
        assert!(matches!(
            estimate(&input(ks, 0, 1_000)),
            Err(ModelError::Overflow(_))
        ));

        // unscaled + kernels overflow.
        let ks = vec![kernel(
            "a",
            u64::MAX,
            None,
            Some(1),
            Some(1),
            Some(0),
            "measured_dram",
        )];
        assert!(matches!(
            estimate(&input(ks, 1, 1_000)),
            Err(ModelError::Overflow(_))
        ));

        // Zero / NaN / negative / inf rate.
        for bad in [0.0, f64::NAN, -1.0, f64::INFINITY] {
            let mut inp = input(three_kernels(), 500_000, 10_500_000);
            inp.host_rates.read_bytes_s = Some(bad);
            assert!(
                matches!(estimate(&inp), Err(ModelError::InvalidRate { .. })),
                "{bad}"
            );
        }
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.target_rates.compute_ops_s = Some(0.0);
        assert!(matches!(
            estimate(&inp),
            Err(ModelError::InvalidRate { .. })
        ));

        // fixed_ns > total_ns.
        let ks = vec![kernel(
            "a",
            10,
            Some(11),
            Some(1),
            Some(1),
            Some(0),
            "measured_dram",
        )];
        assert!(matches!(
            estimate(&input(ks, 0, 10)),
            Err(ModelError::InconsistentKernel { .. })
        ));

        // Unknown byte_basis string.
        let ks = vec![kernel("a", 10, None, Some(1), Some(1), Some(0), "guess")];
        assert!(matches!(
            estimate(&input(ks, 0, 10)),
            Err(ModelError::InvalidByteBasis { .. })
        ));
    }

    #[test]
    fn missing_rate_for_nonzero_traffic_is_unavailable() {
        // K3 has host-mapped traffic; drop the host host_read rate.
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.host_rates.host_read_bytes_s = None;
        let est = estimate(&inp).unwrap();
        assert!(est.estimated_wall_ns.is_none());
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("host_rates.host_read_bytes_s")));
        // Partial diagnostics: K1/K2 still computed, K3 not.
        assert!(est.buckets[0].estimated_ns.is_some());
        assert!(est.buckets[1].estimated_ns.is_some());
        assert!(est.buckets[2].estimated_ns.is_none());

        // Target side missing too.
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.target_rates.host_read_bytes_s = None;
        let est = estimate(&inp).unwrap();
        assert!(est.estimated_wall_ns.is_none());
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("target_rates.host_read_bytes_s")));

        // A missing rate for a class with ZERO traffic does not block.
        let mut inp = input(three_kernels()[..2].to_vec(), 500_000, 6_500_000);
        inp.host_rates.host_read_bytes_s = None;
        inp.target_rates.host_read_bytes_s = None;
        let est = estimate(&inp).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        // 500_000 + 3_900_000 + 5_000_000
        assert!((est.estimated_wall_ns.unwrap() - 9_400_000.0).abs() < 1e-3);
    }

    #[test]
    fn coverage_tolerance_five_percent() {
        let k = vec![kernel(
            "a",
            9_500_000,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        )];
        // |10_000_000 - 9_500_000| / 10_000_000 = exactly 5% -> allowed.
        let est = estimate(&input(k.clone(), 0, 10_000_000)).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        assert!(est.estimated_wall_ns.is_some());
        assert!((est.coverage_error_fraction.unwrap() - 0.05).abs() < 1e-12);

        // 9_499_999 -> just over 5% -> unavailable, diagnostics retained.
        let k = vec![kernel(
            "a",
            9_499_999,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        )];
        let est = estimate(&input(k, 0, 10_000_000)).unwrap();
        assert!(est.estimated_wall_ns.is_none());
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("coverage")));
        assert_eq!(est.buckets.len(), 1);
        assert!(est.buckets[0].estimated_ns.is_some());

        // Over-coverage (kernels + unscaled exceed wall by >5%) is also rejected.
        let k = vec![kernel(
            "a",
            11_000_000,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        )];
        let est = estimate(&input(k, 0, 10_000_000)).unwrap();
        assert!(est.estimated_wall_ns.is_none());
    }

    #[test]
    fn unclassified_tolerance_ten_percent() {
        let classified = kernel(
            "c",
            9_000_000,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        );
        // Exactly 10% unclassified -> allowed, carried unscaled.
        let unk = kernel("u", 1_000_000, None, None, None, None, "unknown");
        let est = estimate(&input(vec![classified.clone(), unk], 0, 10_000_000)).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        assert!((est.unclassified_time_fraction - 0.10).abs() < 1e-12);
        assert!(!est.buckets[1].classified);
        assert_eq!(est.buckets[1].estimated_ns, Some(1_000_000.0));

        // 11% -> unavailable.
        let classified = kernel(
            "c",
            8_900_000,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        );
        let unk = kernel("u", 1_100_000, None, None, None, None, "unknown");
        let est = estimate(&input(vec![classified, unk], 0, 10_000_000)).unwrap();
        assert!(est.estimated_wall_ns.is_none());
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("unclassified")));
        assert!((est.unclassified_time_fraction - 0.11).abs() < 1e-12);

        // A known basis with a missing traffic field is unclassified (unknown != zero).
        let partial = kernel(
            "p",
            1_100_000,
            None,
            None,
            Some(5),
            Some(0),
            "measured_dram",
        );
        let classified = kernel(
            "c",
            8_900_000,
            Some(0),
            Some(1),
            Some(1_000_000),
            Some(0),
            "measured_dram",
        );
        let est = estimate(&input(vec![classified, partial], 0, 10_000_000)).unwrap();
        assert!(est.estimated_wall_ns.is_none());
    }

    #[test]
    fn unknown_fixed_ns_reports_two_value_sensitivity() {
        // Single memory-bound kernel (ratio 2), fixed unknown.
        // l = 0: t' = 2 * 8_000_000 = 16_000_000 ; l = t: t' = 8_000_000.
        // unscaled = 2_000_000 -> T' = 18_000_000 (l=0) or 10_000_000 (l=t).
        let k = vec![kernel(
            "a",
            8_000_000,
            None,
            Some(1_000_000_000),
            Some(1_000_000_000),
            Some(0),
            "measured_dram",
        )];
        let est = estimate(&input(k, 2_000_000, 10_000_000)).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        assert!((est.estimated_wall_ns.unwrap() - 18_000_000.0).abs() < 1e-3);
        let s = est
            .assumptions
            .iter()
            .find(|a| a.contains("[SENSITIVITY]"))
            .expect("sensitivity assumption");
        assert!(s.contains("18000000.0"), "{s}");
        assert!(s.contains("10000000.0"), "{s}");
        assert!(s.contains("l = 0") && s.contains("l = t"), "{s}");

        // Known fixed_ns: no sensitivity entry.
        let k = vec![kernel(
            "a",
            8_000_000,
            Some(0),
            Some(1_000_000_000),
            Some(1_000_000_000),
            Some(0),
            "measured_dram",
        )];
        let est = estimate(&input(k, 2_000_000, 10_000_000)).unwrap();
        assert!(!est.assumptions.iter().any(|a| a.contains("[SENSITIVITY]")));

        // Target-calibrated claim must not silently assume l = 0.
        let k = vec![kernel(
            "a",
            8_000_000,
            None,
            Some(1_000_000_000),
            Some(1_000_000_000),
            Some(0),
            "measured_dram",
        )];
        let mut inp = input(k, 2_000_000, 10_000_000);
        inp.target_rates_source = RatesSource::MeasuredOnTarget {
            receipt_digest: "d1".into(),
        };
        let est = estimate(&inp).unwrap();
        assert_eq!(est.quality, EstimateQuality::TargetCalibrated);
        assert!(est.estimated_wall_ns.is_none());
        assert!(est.assumptions.iter().any(|a| a.contains("[SENSITIVITY]")));
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("fixed_ns unknown")));
    }

    #[test]
    fn identity_mismatch_is_unavailable() {
        for mutate in [
            (|i: &mut RunIdentity| i.physical_cus = 96) as fn(&mut RunIdentity),
            |i| i.physical_gfx = "gfx1100".into(),
            |i| i.binary_md5 = "ffff".into(),
            |i| i.route.rows = 8,
            |i| i.route.graph = false,
            |i| i.emu_id = None,
            |i| i.token_digest = "other".into(),
            |i| i.model_md5 = Reading::known("bbbb".into()),
        ] {
            let mut inp = input(three_kernels(), 500_000, 10_500_000);
            mutate(&mut inp.host_calibration);
            let est = estimate(&inp).unwrap();
            assert!(est.estimated_wall_ns.is_none());
            assert!(est.buckets.is_empty());
            assert!(est
                .unavailable_reasons
                .iter()
                .any(|r| r.contains("calibration identity differs")));
        }

        // Telemetry and calibration digest are not identity.
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.host_calibration.telemetry.sclk_mhz = Reading::known(1500);
        inp.host_calibration.telemetry.power_w = Reading::known(10.0);
        inp.host_calibration.calibration_receipt_digest = Reading::known("zzz".into());
        let est = estimate(&inp).unwrap();
        assert!(est.estimated_wall_ns.is_some());
    }

    #[test]
    fn quality_labelling() {
        let inp = input(three_kernels(), 500_000, 10_500_000);
        let est = estimate(&inp).unwrap();
        assert_eq!(est.quality, EstimateQuality::TopologyScenario);

        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.target_rates_source = RatesSource::MeasuredOnTarget {
            receipt_digest: "sha-abc".into(),
        };
        let est = estimate(&inp).unwrap();
        assert_eq!(est.quality, EstimateQuality::TargetCalibrated);
        assert!((est.estimated_wall_ns.unwrap() - 14_350_000.0).abs() < 1e-3);

        // Empty digest: cannot claim calibration.
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.target_rates_source = RatesSource::MeasuredOnTarget {
            receipt_digest: " ".into(),
        };
        let est = estimate(&inp).unwrap();
        assert_eq!(est.quality, EstimateQuality::TopologyScenario);
        assert!(est.estimated_wall_ns.is_none());

        // Identity digest conflicting with the source's digest.
        let mut inp = input(three_kernels(), 500_000, 10_500_000);
        inp.identity.calibration_receipt_digest = Reading::known("other".into());
        inp.target_rates_source = RatesSource::MeasuredOnTarget {
            receipt_digest: "sha-abc".into(),
        };
        let est = estimate(&inp).unwrap();
        assert_eq!(est.quality, EstimateQuality::TopologyScenario);
        assert!(est.estimated_wall_ns.is_none());
    }

    fn profile_9060() -> EmuProfile {
        EmuProfile {
            id: "test-9060",
            host_gfx: "gfx1201",
            target_gfx: "gfx1200",
            target_cus: 32,
            memory: MemoryKind::Discrete {
                capacity_bytes: 16 << 30,
            },
            peak_bw_bytes_s: 320_000_000_000,
            infinity_cache_bytes: 32 << 20,
            l2_bytes: 4 << 20,
            vgpr_file_bytes: 768 << 10,
            fidelity: Fidelity::CrossDieEstimate,
        }
    }

    #[test]
    fn spec_ratio_scenario() {
        let host = ResourceRates {
            compute_ops_s: Some(1.0e12),
            read_bytes_s: Some(638.7e9),
            copy_rw_bytes_s: Some(566.0e9),
            host_read_bytes_s: Some(50.0e9),
        };
        // Same clock 2000 MHz both: C ratio = (32*2000)/(64*2000) = 0.5; BW ratio 320/640 = 0.5.
        let (r, notes) = spec_ratio_rates(
            &host,
            (64, 2000, 640_000_000_000),
            &profile_9060(),
            Some(2000),
        );
        assert_eq!(r.compute_ops_s, Some(5.0e11));
        assert_eq!(r.read_bytes_s, Some(638.7e9 * 0.5));
        assert_eq!(r.copy_rw_bytes_s, Some(566.0e9 * 0.5));
        assert_eq!(r.host_read_bytes_s, None);
        assert!(
            notes.iter().all(|n| n.starts_with("[INFERENCE]")),
            "{notes:?}"
        );
        assert!(notes.iter().any(|n| n.contains("host_read_bytes_s")));

        // Different clocks: (32*2500)/(64*2000) = 0.625.
        let (r, _) = spec_ratio_rates(
            &host,
            (64, 2000, 640_000_000_000),
            &profile_9060(),
            Some(2500),
        );
        assert_eq!(r.compute_ops_s, Some(6.25e11));

        // No explicit clock -> compute unavailable, never invented.
        let (r, notes) =
            spec_ratio_rates(&host, (64, 2000, 640_000_000_000), &profile_9060(), None);
        assert_eq!(r.compute_ops_s, None);
        assert!(r.read_bytes_s.is_some());
        assert!(notes.iter().any(|n| n.contains("explicit target clock")));

        // Missing host rate stays missing.
        let mut h2 = host.clone();
        h2.read_bytes_s = None;
        let (r, _) = spec_ratio_rates(
            &h2,
            (64, 2000, 640_000_000_000),
            &profile_9060(),
            Some(2000),
        );
        assert_eq!(r.read_bytes_s, None);

        // Zero host peak -> no bandwidth scaling.
        let (r, _) = spec_ratio_rates(&host, (64, 2000, 0), &profile_9060(), Some(2000));
        assert_eq!(r.read_bytes_s, None);
        assert_eq!(r.copy_rw_bytes_s, None);
    }

    #[test]
    fn spec_ratio_rates_feed_estimate() {
        // Target rates from spec_ratio_rates: host -> 9060 (BW/2, C/2, no host-read rate).
        let (tr, _) = spec_ratio_rates(
            &host_rates(),
            (64, 2000, 640_000_000_000),
            &profile_9060(),
            Some(2000),
        );
        // Memory-bound kernel only (P=0): ratio = 2 (B/2 dominates, C/2 gives 2x too).
        let k = vec![kernel(
            "a",
            1_000_000,
            Some(0),
            Some(1_000_000),
            Some(1_000_000_000),
            Some(0),
            "measured_dram",
        )];
        let mut inp = input(k, 0, 1_000_000);
        inp.target_rates = tr;
        let est = estimate(&inp).unwrap();
        assert!(
            est.unavailable_reasons.is_empty(),
            "{:?}",
            est.unavailable_reasons
        );
        assert!((est.estimated_wall_ns.unwrap() - 2_000_000.0).abs() < 1e-3);

        // A kernel with host-mapped traffic is unavailable: target host link is unknown.
        let k = vec![kernel(
            "b",
            1_000_000,
            Some(0),
            Some(1),
            Some(1),
            Some(1_000_000),
            "measured_dram",
        )];
        let mut inp = input(k, 0, 1_000_000);
        inp.target_rates = spec_ratio_rates(
            &host_rates(),
            (64, 2000, 640_000_000_000),
            &profile_9060(),
            Some(2000),
        )
        .0;
        let est = estimate(&inp).unwrap();
        assert!(est.estimated_wall_ns.is_none());
        assert!(est
            .unavailable_reasons
            .iter()
            .any(|r| r.contains("target_rates.host_read_bytes_s")));
    }
}
