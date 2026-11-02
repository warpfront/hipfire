// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Data shared by the `hipfire fit` stages: the machine facts (read from
//! sysfs, `/proc` and the calibration record, never from HIP), the per-fact
//! findings behind the verdict, and the weight placement a load would take.

use serde::Serialize;

pub(crate) const GIB: f64 = (1_u64 << 30) as f64;

/// Bytes as GiB with one decimal, for report text.
pub(crate) fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

/// Whether the model fits, one fact at a time and overall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Verdict {
    Fits,
    /// A machine fact the decision needs is unknown (no GPU detected, an
    /// unreadable free-memory counter), or the load itself decides it.
    Undetermined,
    DoesNotFit,
}

impl Verdict {
    /// The worse of two verdicts: `DoesNotFit` > `Undetermined` > `Fits`.
    pub(crate) fn and(self, other: Verdict) -> Verdict {
        self.max(other)
    }

    fn rank(self) -> u8 {
        match self {
            Verdict::Fits => 0,
            Verdict::Undetermined => 1,
            Verdict::DoesNotFit => 2,
        }
    }

    fn max(self, other: Verdict) -> Verdict {
        if other.rank() > self.rank() {
            other
        } else {
            self
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Verdict::Fits => "FITS",
            Verdict::Undetermined => "UNDETERMINED",
            Verdict::DoesNotFit => "DOES NOT FIT",
        }
    }
}

/// One decided fact behind the verdict, in report order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct Finding {
    /// Stable id: `preflight`, `admission`, `vram`, `context`, `experts`,
    /// `host_ram`, `gtt`, `weights`, `draft`.
    pub id: &'static str,
    pub verdict: Verdict,
    /// What was found, with the numbers it was decided on.
    pub detail: String,
    /// What to change when it does not fit (or cannot be decided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

impl Finding {
    pub(crate) fn fits(id: &'static str, detail: impl Into<String>) -> Self {
        Self {
            id,
            verdict: Verdict::Fits,
            detail: detail.into(),
            fix: None,
        }
    }

    pub(crate) fn fails(
        id: &'static str,
        detail: impl Into<String>,
        fix: impl Into<String>,
    ) -> Self {
        Self {
            id,
            verdict: Verdict::DoesNotFit,
            detail: detail.into(),
            fix: Some(fix.into()),
        }
    }

    pub(crate) fn undetermined(
        id: &'static str,
        detail: impl Into<String>,
        fix: Option<String>,
    ) -> Self {
        Self {
            id,
            verdict: Verdict::Undetermined,
            detail: detail.into(),
            fix,
        }
    }
}

/// The host link of one GPU as the calibration record describes it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct LinkFacts {
    /// A fresh `hipfire calibrate` record covers this GPU.
    pub calibrated: bool,
    /// Stored pinned host-to-device rate, GB/s.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gbps: Option<f64>,
    /// Below the engine's narrow-link threshold
    /// (`hipfire_arch_qwen4::host_link::NARROW_LINK_GBPS`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub narrow: Option<bool>,
    /// One line: the link and its freshness, or why there is no record.
    pub detail: String,
}

/// A GPU a load could use, from the KFD topology and amdgpu sysfs only.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct FitDevice {
    /// Logical index in the set a load would use (0 = the device a
    /// single-GPU load takes).
    pub index: usize,
    pub arch: String,
    pub unified_memory: bool,
    /// `mem_info_vram_total` (the carve-out on an APU).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total: Option<u64>,
    /// Free device bytes as `Gpu::device_mem_info` would report them before
    /// the load, derived without HIP (see `fit/host.rs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<LinkFacts>,
    /// Tokens of context the model being fitted projects on this GPU (`0`: its
    /// weights do not fit the free memory). Filled by the model analysis; absent
    /// where no projection ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity: Option<u64>,
    /// `capacity` capped at the model's trained window (what the report shows).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity_shown: Option<u64>,
    /// `HIP_VISIBLE_DEVICES` value that selects this GPU (UUID, else ROCr
    /// ordinal). Report text only.
    #[serde(skip)]
    pub env_selector: String,
    /// `hardware.devices` value that selects this GPU (UUID, else PCI
    /// address). Report text only.
    #[serde(skip)]
    pub config_selector: String,
}

/// Host memory facts the admission gates read.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct FitHost {
    /// `MemTotal`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_total: Option<u64>,
    /// `MemAvailable` minus the CMA region
    /// (`rdna_compute::uma_memory::mem_available_bytes`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_available: Option<u64>,
    /// TTM page-pool estimate (freed GTT pages the next GTT allocation takes
    /// first).
    pub ttm_pool: u64,
    /// TTM's GTT cap across amdgpu devices; `None` when no cap applies
    /// (userptr paged memory, or no amdgpu GTT counters).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gtt_limit: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gtt_used: Option<u64>,
    pub devices: Vec<FitDevice>,
}

/// Where the weights go.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct PlacementPlan {
    /// One line, e.g. `fully resident` or
    /// `at most 9 of 48 expert layers in VRAM, at least 41.2 GiB host-mapped`.
    pub summary: String,
    /// Routed-expert layers in VRAM: `(resident, total)`. For a bound the
    /// load narrows, `resident` is the upper bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expert_layers_resident: Option<(usize, usize)>,
    /// Host-mapped routed-expert bytes (the lower bound when the split is a
    /// bound).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_mapped_bytes: Option<u64>,
    /// `true` when the exact split is decided at load (it depends on a load
    /// reserve only the GPU can size); the numbers above are then bounds.
    pub decided_at_load: bool,
}

/// One architecture's capacity analysis.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ArchFit {
    pub placement: PlacementPlan,
    pub findings: Vec<Finding>,
    /// Report-only lines a user acts on (inputs the decision used, terms
    /// decided at load).
    pub notes: Vec<String>,
    /// Source-cited derivation lines (which engine terms a bound includes or
    /// omits, with file:line). JSON and `--verbose` only.
    pub details: Vec<String>,
}
