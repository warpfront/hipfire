// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `hipfire doctor --npu` report. The JSON form is meant to be pasted into
//! Discord or GitHub, so it never carries a hostname, user name, filesystem
//! path or PCI address: only versions, verdicts and measurements.

use serde::Serialize;

/// JSON schema identifier; bump on any field change.
pub(crate) const SCHEMA: &str = "hipfire-doctor-npu-1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Tier {
    /// No blocker, and the host matches a validated row of the profile table.
    Reference,
    /// No blocker, but at least one profile field differs from every validated row.
    CompatibleUnvalidated,
    /// At least one blocking check failed.
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CheckStatus {
    Pass,
    Fail,
    /// Something is wrong that slows hipfire but does not stop it. Never
    /// blocks and never changes the tier.
    Warn,
    /// Reported only; never blocks and never changes the tier by itself.
    Info,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Check {
    /// Stable id: `gpu_arch`, `npu_xdna2`, `accel_node`, `npu_firmware`, `memlock`,
    /// `rocm_runtime`, `daemon`, `model`, `gpu_free`, `npu_trunk_config`, `fclk`, `power`.
    pub id: &'static str,
    pub status: CheckStatus,
    /// What was found, without paths, user or host names.
    pub detail: String,
    /// The exact fix for a failed or warned check; `None` when it passed or is informational.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ProfileDiff {
    pub field: &'static str,
    pub reference: String,
    pub found: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Iommu {
    pub gpu: String,
    pub npu: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Memlock {
    /// `None` = unlimited.
    pub soft: Option<u64>,
    pub hard: Option<u64>,
    /// Bytes the armed trunk pins plus what this process already pins.
    pub required: u64,
    pub cap_ipc_lock: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Fclk {
    /// `None` when the fabric clock state is unreadable.
    pub pinned: Option<bool>,
    /// The attributes are writable directly or through `sudo -n`.
    pub pinnable: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Host {
    pub gpu_arch: Option<String>,
    /// `"xdna2"` when an amdxdna 1022:17f0 NPU is present.
    pub npu: Option<String>,
    pub kernel: String,
    pub iommu: Iommu,
    pub amdxdna: String,
    pub npu_fw: String,
    /// The kernel set the daemon will run: `pack:<tag>@<commit12>` or `jit:<commit12>`.
    pub kernels: String,
    pub memlock: Memlock,
    pub fclk: Fclk,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Model {
    /// Registry tag when the file is a registry model, else `None` (never a path).
    pub tag: Option<String>,
    /// `model_trunk_index_sha256` of the model (lowercase hex): the normalized, PLE-shard-agnostic
    /// NPU trunk index digest, not the raw header-span hash.
    pub model_trunk_index_sha256: String,
    /// `"builtin"` or `"file"`: which calibration record admits this model.
    pub calibration: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Samples {
    pub samples_ms: Vec<f64>,
    pub median_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SizeVerdict {
    Gain,
    Loss,
    NoMeasurableGain,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct SizeResult {
    /// Prompt rows the daemon reported (`prompt_tokens`).
    pub rows: u64,
    pub off: Samples,
    pub auto: Samples,
    /// (median off - median auto) / median off * 100; negative is a loss.
    pub saved_pct: f64,
    /// Larger of the two arms' (max - min) / median * 100.
    pub spread_pct: f64,
    pub verdict: SizeVerdict,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct RateScales {
    pub gpu_x: f64,
    pub npu_x: f64,
    pub frozen: bool,
    /// Rate generation of the last planned request (`rates=gen<N>`).
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct PairWait {
    pub pairs: u64,
    pub median_ms: f64,
    pub max_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ExactnessVerdict {
    Exact,
    Mismatch,
    /// `on` ran no NPU pair, so the comparison proves nothing.
    NotExercised,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Exactness {
    pub verdict: ExactnessVerdict,
    pub prompt_rows: u64,
    pub tokens: u64,
    /// Index of the first differing generated token on a mismatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_difference: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Calibration {
    /// `false` when auto did not arm; `off_reason` then carries the daemon's reason.
    pub armed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub off_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trunk_route: Option<String>,
    pub sizes: Vec<SizeResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_scales: Option<RateScales>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npu_wait_per_pair: Option<PairWait>,
    pub guard_trips: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exactness: Option<Exactness>,
    pub duration_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Action {
    /// Auto measured a gain, or no measurable difference: leave `auto` on.
    KeepAuto,
    /// Measured loss or exactness mismatch: switch the route off.
    SetOff,
    /// The route is configured off, and auto measured a gain.
    SetAuto,
    /// Nothing to recommend (blocked, not calibrated, auto did not arm).
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Recommendation {
    pub action: Action,
    /// One plain sentence.
    pub reason: String,
    /// The command to run, e.g. `hipfire config set npu_trunk off`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Build {
    pub version: &'static str,
    /// First 12 hex digits of the CLI build commit.
    pub commit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Report {
    pub schema: &'static str,
    pub hipfire: Build,
    pub tier: Tier,
    /// Id of the matched (reference) or nearest (compatible) profile row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<&'static str>,
    pub profile_diff: Vec<ProfileDiff>,
    pub host: Host,
    pub checks: Vec<Check>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
    /// Effective `fusions.npu_trunk`.
    pub npu_trunk: String,
    /// `None` when the calibration did not run; `calibration_skipped` says why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_skipped: Option<String>,
    pub recommendation: Recommendation,
    /// `true` only when `--apply` changed `fusions.npu_trunk`.
    pub applied: bool,
}
