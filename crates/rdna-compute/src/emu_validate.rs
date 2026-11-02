//! Model-free SKU-emulation validator: finite HW_ID topology and traffic checks (EMU-DESIGN.md §6, E8).
//!
//! [`validate`] runs four independent checks on a fresh `Gpu` and returns a
//! schema-v1 [`EmuValidationReceipt`]:
//!
//! - `host_topology`: finite HW_ID1 probe launches (one store per wave into a
//!   unique slot, no atomics), decoded and unioned on the CPU until the
//!   observed CU set stabilises or the launch cap is hit. Pass iff the
//!   observed CU count equals the **host** physical CU count; the target CU
//!   count is never compared.
//! - `capacity`: ledger/adapter checks with small controlled allocations and a
//!   cap-exceeding reservation that is refused without reaching HIP.
//! - `bandwidth`: eager integer read/copy kernels with CPU-oracle checksums,
//!   full destination compare and guard regions.
//! - `graph`: the same kernels captured into HIP graphs and replayed with
//!   identical byte/checksum results.
//!
//! No CU masking of any kind is used anywhere; every kernel is finite. The
//! validator never allocates ballast to hide VRAM and records
//! `cu_emulation = "none"` and `limiter_active = false`: capacity-only mode
//! makes no target-bandwidth claim.

use std::collections::BTreeSet;
use std::path::Path;

use hip_bridge::{Event, HipError, HipResult, KernargBlob};
use hipfire_config::devices::{enumerate_gpus, DeviceRoot, PciBdf};
use hipfire_config::emu::{wgp_to_physical_cus, EmuProfile, MemoryKind};
use hipfire_config::emu_budget::{self, charge_vector, AllocationClass, BudgetUsage};
use hipfire_config::emu_model::{Reading, Telemetry};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{DType, Gpu, GpuTensor};

/// Receipt schema version.
pub const SCHEMA_VERSION: u32 = 1;
/// Largest accepted `buffer_mib` entry.
pub const MAX_BUFFER_MIB: u32 = 2048;
/// Most `buffer_mib` entries per request.
pub const MAX_BUFFER_COUNT: usize = 8;
/// Largest accepted `passes` (timed repetitions).
pub const MAX_PASSES: u32 = 100;
/// Untimed warm-up repetitions before the timed ones.
pub const WARMUP_REPS: u32 = 2;
/// Longest accepted `identity_note`.
pub const MAX_IDENTITY_NOTE_CHARS: usize = 512;
/// Test names accepted in `approved_tests`.
pub const KNOWN_TESTS: [&str; 4] = ["capacity", "topology", "bandwidth", "graph"];
/// Physical arches the validator kernels are guarded for.
pub const SUPPORTED_ARCHES: [&str; 3] = ["gfx1100", "gfx1151", "gfx1201"];

/// Hard cap on topology probe launches.
pub const TOPOLOGY_LAUNCH_CAP: u32 = 64;
/// Stop early after this many consecutive launches add no new CU.
pub const TOPOLOGY_STABLE_WINDOW: u32 = 8;
/// Blocks per physical CU in each topology launch.
pub const TOPOLOGY_BLOCKS_PER_CU: u32 = 16;
/// Threads per block of every validator launch.
pub const BLOCK_THREADS: u32 = 256;
/// Waves per topology block (wave32 on all three supported arches).
pub const TOPOLOGY_WAVES_PER_BLOCK: u32 = 8;
/// Finite dependent-chain length per topology wave (keeps blocks resident briefly).
pub const TOPOLOGY_CHAIN_ITERS: u32 = 200_000;
/// Unwritten-slot / unreadable-register marker the host pre-fills.
pub const HWID_SENTINEL: u32 = 0xFFFF_FFFF;

/// Blocks per physical CU in the traffic kernels.
const TRAFFIC_BLOCKS_PER_CU: u32 = 32;
/// Largest traffic grid.
const TRAFFIC_MAX_BLOCKS: u32 = 65_535;
/// Guard bytes on each side of the copy destination.
const GUARD_BYTES: usize = 4096;
/// Poison byte for destination and guard regions.
const POISON_BYTE: i32 = 0xA5;
/// Host staging chunk for pattern upload / destination readback.
const CHUNK_BYTES: usize = 64 << 20;
/// Free memory kept in reserve when sizing a traffic run.
const FIT_HEADROOM_BYTES: u64 = 64 << 20;
/// Controlled-malloc size for the capacity check.
const CONTROLLED_MALLOC_BYTES: usize = 64 << 20;
/// Committed device bytes above which the Gpu is presumed to hold a model.
const MODEL_FREE_BASELINE_LIMIT_BYTES: u64 = 256 << 20;

const FNV_OFFSET: u32 = 0x811C_9DC5;
const FNV_PRIME: u32 = 0x0100_0193;

const MODULE_NAME: &str = "emu_validate";
const KERNEL_HWID: &str = "emu_hwid_probe";
const KERNEL_READ: &str = "emu_read_checksum";
const KERNEL_COPY: &str = "emu_copy_u32x4";

// ── Request ─────────────────────────────────────────────────────────────────

fn default_buffer_mib() -> Vec<u32> {
    vec![256]
}

fn default_passes() -> u32 {
    5
}

/// `{"profile_id": ..., "buffer_mib": [..], "passes": N, "approved_tests": [..], "identity_note": ..}`.
/// Unknown fields (e.g. the daemon's `"type"`) are ignored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmuValidationRequest {
    /// Profile the caller expects to be active (checked against the Gpu when emu is on).
    pub profile_id: String,
    /// Traffic buffer sizes in MiB, each `1..=MAX_BUFFER_MIB`.
    #[serde(default = "default_buffer_mib")]
    pub buffer_mib: Vec<u32>,
    /// Timed repetitions per size/kernel, `1..=MAX_PASSES`.
    #[serde(default = "default_passes")]
    pub passes: u32,
    /// Subset of [`KNOWN_TESTS`]; empty runs all.
    #[serde(default)]
    pub approved_tests: Vec<String>,
    /// Free-form caller identity (fabric job, host label, ...).
    #[serde(default)]
    pub identity_note: Option<String>,
}

impl EmuValidationRequest {
    /// Bounds validation. Called by [`validate`]; also usable by callers to fail fast.
    pub fn validate(&self) -> Result<(), String> {
        if self.profile_id.trim().is_empty() {
            return Err("profile_id must not be empty".into());
        }
        if self.buffer_mib.len() > MAX_BUFFER_COUNT {
            return Err(format!(
                "buffer_mib has {} entries; at most {MAX_BUFFER_COUNT} are accepted",
                self.buffer_mib.len()
            ));
        }
        for &mib in &self.buffer_mib {
            if mib == 0 || mib > MAX_BUFFER_MIB {
                return Err(format!(
                    "buffer_mib entry {mib} is outside 1..={MAX_BUFFER_MIB}"
                ));
            }
        }
        let unique: BTreeSet<u32> = self.buffer_mib.iter().copied().collect();
        if unique.len() != self.buffer_mib.len() {
            return Err("buffer_mib entries must be distinct".into());
        }
        if self.passes == 0 || self.passes > MAX_PASSES {
            return Err(format!(
                "passes {} is outside 1..={MAX_PASSES}",
                self.passes
            ));
        }
        for test in &self.approved_tests {
            if !KNOWN_TESTS.contains(&test.as_str()) {
                return Err(format!(
                    "unknown approved test `{test}`; known tests: {}",
                    KNOWN_TESTS.join(", ")
                ));
            }
        }
        if let Some(note) = &self.identity_note {
            if note.chars().count() > MAX_IDENTITY_NOTE_CHARS {
                return Err(format!(
                    "identity_note exceeds {MAX_IDENTITY_NOTE_CHARS} characters"
                ));
            }
        }
        if (self.requested("bandwidth") || self.requested("graph")) && self.buffer_mib.is_empty() {
            return Err("bandwidth/graph tests require at least one buffer_mib entry".into());
        }
        Ok(())
    }

    /// Whether `test` runs (an empty `approved_tests` approves every test).
    pub fn requested(&self, test: &str) -> bool {
        self.approved_tests.is_empty() || self.approved_tests.iter().any(|t| t == test)
    }
}

// ── Receipt ─────────────────────────────────────────────────────────────────

/// Outcome of a check or section.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pass,
    Fail,
    Unavailable,
}

/// One named sub-check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckRecord {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

fn check(name: &str, ok: bool, detail: impl Into<String>) -> CheckRecord {
    CheckRecord {
        name: name.to_string(),
        status: if ok { Status::Pass } else { Status::Fail },
        detail: detail.into(),
    }
}

/// Min/median/max/mean/stddev plus the raw samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub n: usize,
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub mean: f64,
    /// Population standard deviation.
    pub stddev: f64,
    pub samples: Vec<f64>,
}

impl Distribution {
    /// `None` for an empty sample set or any non-finite sample.
    pub fn from_samples(samples: Vec<f64>) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let n = sorted.len();
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let var = sorted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        Some(Self {
            n,
            min: sorted[0],
            median,
            max: sorted[n - 1],
            mean,
            stddev: var.sqrt(),
            samples,
        })
    }
}

/// Ledger byte counters (mirror of `BudgetUsage`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageRecord {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub gtt_bytes: u64,
    pub shared_host_bytes: u64,
}

impl From<BudgetUsage> for UsageRecord {
    fn from(u: BudgetUsage) -> Self {
        Self {
            device_bytes: u.device_bytes,
            host_bytes: u.host_bytes,
            gtt_bytes: u.gtt_bytes,
            shared_host_bytes: u.shared_host_bytes,
        }
    }
}

/// Immutable ledger limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitsRecord {
    pub device_bytes: u64,
    pub gtt_bytes: Option<u64>,
    pub shared_host_bytes: Option<u64>,
    pub uma: bool,
}

/// The effective SKU profile (`None` in the receipt when emulation is off).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveProfileRecord {
    pub id: String,
    pub host_gfx: String,
    pub target_gfx: String,
    pub target_cus: u32,
    /// `discrete` or `uma`.
    pub memory: String,
    pub advertised_capacity_bytes: Option<u64>,
    pub peak_bw_bytes_s: u64,
    pub infinity_cache_bytes: u64,
    pub l2_bytes: u64,
    pub fidelity: String,
}

impl EffectiveProfileRecord {
    fn from_profile(p: &EmuProfile) -> Self {
        let (memory, advertised) = match p.memory {
            MemoryKind::Discrete { capacity_bytes } => ("discrete", Some(capacity_bytes)),
            MemoryKind::Uma => ("uma", None),
        };
        Self {
            id: p.id.to_string(),
            host_gfx: p.host_gfx.to_string(),
            target_gfx: p.target_gfx.to_string(),
            target_cus: p.target_cus,
            memory: memory.to_string(),
            advertised_capacity_bytes: advertised,
            peak_bw_bytes_s: p.peak_bw_bytes_s,
            infinity_cache_bytes: p.infinity_cache_bytes,
            l2_bytes: p.l2_bytes,
            fidelity: serde_json::to_value(p.fidelity)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{:?}", p.fidelity)),
        }
    }
}

/// Memory facts at validator entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Profile's advertised capacity (discrete) or the UMA scenario device cap.
    pub advertised_capacity_bytes: Option<u64>,
    /// `hipMemGetInfo` free/total, unadjusted.
    pub raw_real_free_bytes: u64,
    pub raw_real_total_bytes: u64,
    /// `Gpu::device_mem_info` free/total (emu-capped when active).
    pub budgeted_free_bytes: u64,
    pub budgeted_total_bytes: u64,
    pub limits: Option<LimitsRecord>,
    pub committed: Option<UsageRecord>,
    pub pending: Option<UsageRecord>,
    pub pool_parked_bytes: u64,
    /// Always `managed_application_charge`: the ledger counts charges made by
    /// this application's managed allocators, not driver- or other-process use.
    pub accounting_kind: String,
}

/// Physical device identity, kept separate from the emulated target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalRecord {
    pub pci_bdf: String,
    pub physical_gfx: String,
    pub target_gfx: Option<String>,
    pub target_cus: Option<u32>,
    /// Raw `hipDeviceAttributeMultiprocessorCount` (WGPs on RDNA).
    pub hip_mp_count: u32,
    /// `wgp_to_physical_cus(hip_mp_count, gfx)`.
    pub hip_physical_cus: u32,
    /// KFD `simd_count / simd_per_cu` for this BDF.
    pub kfd_physical_cus: Reading<u32>,
    /// Physical CUs captured by the emu install (emu on only).
    pub host_facts_physical_cus: Option<u32>,
    /// The CU count the topology check compares against.
    pub host_physical_cus: Option<u32>,
    /// `host_facts`, `kfd` or `hip_mp_count`.
    pub cu_count_source: Option<String>,
    /// Set when the available sources disagree.
    pub cu_count_disagreement: Option<String>,
    /// Always `none`: no CU/queue/PM4 mask is applied.
    pub cu_emulation: String,
}

/// HW_ID1 bit layout used to decode the probe on the CPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HwIdLayout {
    pub register: String,
    pub simd_shift: u32,
    pub simd_bits: u32,
    pub wgp_shift: u32,
    pub wgp_bits: u32,
    pub sa_shift: u32,
    pub sa_bits: u32,
    pub se_shift: u32,
    pub se_bits: u32,
    /// Always `true`: the layout is assumed for the arch, not read from a spec
    /// at runtime. It is validated only indirectly by the observed-CU-count
    /// match against the host.
    pub assumed: bool,
    pub note: String,
}

/// Decoded HW_ID1 fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HwIdFields {
    pub simd: u32,
    pub wgp: u32,
    pub sa: u32,
    pub se: u32,
}

impl HwIdFields {
    /// Physical CU key: `(se, sa, wgp, simd >> 1)` (two SIMDs per CU, four per WGP).
    pub fn cu_key(&self) -> (u32, u32, u32, u32) {
        (self.se, self.sa, self.wgp, self.simd >> 1)
    }
}

/// CU label `se{}.sa{}.wgp{}.cu{}`.
pub fn cu_label(key: (u32, u32, u32, u32)) -> String {
    format!("se{}.sa{}.wgp{}.cu{}", key.0, key.1, key.2, key.3)
}

/// HW_ID1 layout for `arch`, `None` for an unsupported arch. Same bit fields
/// on all three guarded arches (gfx10+ HW_ID1: simd 9:8, wgp 13:10, sa 16,
/// se 20:18); recorded as assumed.
pub fn hw_id1_layout(arch: &str) -> Option<HwIdLayout> {
    if !SUPPORTED_ARCHES.contains(&arch) {
        return None;
    }
    Some(HwIdLayout {
        register: if arch == "gfx1201" {
            "HW_REG_WAVE_HW_ID1 (id 23)".to_string()
        } else {
            "HW_REG_HW_ID1 (id 23)".to_string()
        },
        simd_shift: 8,
        simd_bits: 2,
        wgp_shift: 10,
        wgp_bits: 4,
        sa_shift: 16,
        sa_bits: 1,
        se_shift: 18,
        se_bits: 3,
        assumed: true,
        note: format!(
            "[INFERENCE] {arch} HW_ID1 field layout assumed equal to the gfx10+ layout \
             (simd 9:8, wgp 13:10, sa 16, se 20:18); not read from the hardware. It is \
             validated only by the observed CU count matching the host physical CU count."
        ),
    })
}

fn field(v: u32, shift: u32, bits: u32) -> u32 {
    (v >> shift) & ((1u32 << bits) - 1)
}

/// Decode one raw HW_ID1 word.
pub fn decode_hw_id1(value: u32, layout: &HwIdLayout) -> HwIdFields {
    HwIdFields {
        simd: field(value, layout.simd_shift, layout.simd_bits),
        wgp: field(value, layout.wgp_shift, layout.wgp_bits),
        sa: field(value, layout.sa_shift, layout.sa_bits),
        se: field(value, layout.se_shift, layout.se_bits),
    }
}

/// `host_topology` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologySection {
    pub status: Status,
    pub requested: bool,
    pub reason: Option<String>,
    pub launches: u32,
    pub launch_cap: u32,
    pub blocks_per_launch: u32,
    pub block_threads: u32,
    pub waves_per_block: u32,
    pub chain_iters: u32,
    /// Cumulative observed CU count after each launch.
    pub cumulative_cus_per_launch: Vec<u32>,
    pub observed_cu_count: u32,
    pub observed_wgp_count: u32,
    /// Observed physical CU identities decoded from HW_ID1.
    pub observed_hw_ids: Vec<String>,
    pub hw_id_layout: Option<HwIdLayout>,
    pub unwritten_slots: u64,
    pub chain_mismatch_slots: u64,
    /// CUs the observation is compared against (host physical, never target).
    pub compared_against_host_cus: Option<u32>,
    /// Always `false`: target CU counts are never compared.
    pub compared_against_target_cus: bool,
}

impl TopologySection {
    fn unavailable(requested: bool, reason: impl Into<String>) -> Self {
        Self {
            status: Status::Unavailable,
            requested,
            reason: Some(reason.into()),
            launches: 0,
            launch_cap: TOPOLOGY_LAUNCH_CAP,
            blocks_per_launch: 0,
            block_threads: BLOCK_THREADS,
            waves_per_block: TOPOLOGY_WAVES_PER_BLOCK,
            chain_iters: TOPOLOGY_CHAIN_ITERS,
            cumulative_cus_per_launch: Vec::new(),
            observed_cu_count: 0,
            observed_wgp_count: 0,
            observed_hw_ids: Vec::new(),
            hw_id_layout: None,
            unwritten_slots: 0,
            chain_mismatch_slots: 0,
            compared_against_host_cus: None,
            compared_against_target_cus: false,
        }
    }
}

/// `capacity` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacitySection {
    pub status: Status,
    pub requested: bool,
    pub reason: Option<String>,
    pub profile_total_bytes: Option<u64>,
    pub observed_total_bytes: Option<u64>,
    pub controlled_malloc_bytes: Option<u64>,
    /// Bytes the over-cap reservation asked for (remaining + 1); never reached HIP.
    pub over_cap_request_bytes: Option<u64>,
    pub over_cap_reached_hip: bool,
    pub checks: Vec<CheckRecord>,
}

impl CapacitySection {
    fn unavailable(requested: bool, reason: impl Into<String>) -> Self {
        Self {
            status: Status::Unavailable,
            requested,
            reason: Some(reason.into()),
            profile_total_bytes: None,
            observed_total_bytes: None,
            controlled_malloc_bytes: None,
            over_cap_request_bytes: None,
            over_cap_reached_hip: false,
            checks: Vec::new(),
        }
    }
}

/// Checksum/guard outcome of one traffic kernel at one size.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChecksumRecord {
    /// What was compared: `per_thread_fnv1a_u32` (read sink) or `dst_full_compare` (copy).
    pub kind: String,
    /// Slots (read) or 32-bit words (copy) compared.
    pub compared: u64,
    pub mismatched: u64,
    pub first_mismatch: Option<u64>,
    pub expected_digest: Option<String>,
    pub observed_digest: Option<String>,
    /// Copy only: both guard regions still hold the poison pattern.
    pub guard_intact: Option<bool>,
    /// Repetitions whose result was verified.
    pub verified_reps: u32,
}

/// One kernel's eager traffic at one size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrafficRecord {
    pub kernel: String,
    pub grid_blocks: u32,
    pub block_threads: u32,
    /// Logical bytes per pass: read = S, copy = S read + S written.
    pub bytes_per_pass: u64,
    pub bytes_basis: String,
    pub time_ms: Distribution,
    pub rate_gb_s: Distribution,
    pub checks: ChecksumRecord,
}

/// One buffer size in the bandwidth section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandwidthSize {
    pub buffer_mib: u32,
    pub bytes: u64,
    pub status: Status,
    pub reason: Option<String>,
    pub read: Option<TrafficRecord>,
    pub copy: Option<TrafficRecord>,
}

/// `bandwidth` section (eager timing and checks).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BandwidthSection {
    pub status: Status,
    pub requested: bool,
    pub reason: Option<String>,
    pub warmup_reps: u32,
    pub timed_reps: u32,
    pub sizes: Vec<BandwidthSize>,
}

/// One graph-replayed kernel at one size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphTraffic {
    pub kernel: String,
    pub replays: u32,
    pub bytes_per_pass: u64,
    pub time_ms: Distribution,
    pub rate_gb_s: Distribution,
    /// Checksums / destination bytes equal the CPU oracle on every verified replay.
    pub matches_cpu_oracle: bool,
    /// Replay results are byte-identical to the eager results.
    pub matches_eager: bool,
    pub checks: ChecksumRecord,
}

/// One buffer size in the graph section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSize {
    pub buffer_mib: u32,
    pub status: Status,
    pub reason: Option<String>,
    pub read: Option<GraphTraffic>,
    pub copy: Option<GraphTraffic>,
}

/// `graph` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSection {
    pub status: Status,
    pub requested: bool,
    pub reason: Option<String>,
    pub sizes: Vec<GraphSize>,
}

/// `estimate` section: capacity-only mode never produces a derating estimate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EstimateSection {
    pub status: Status,
    pub reason: Option<String>,
}

/// Raw host rates measured by the largest passing size (median of eager reps).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawRates {
    pub read_bytes_s: Reading<f64>,
    pub copy_rw_bytes_s: Reading<f64>,
    pub basis: String,
}

/// Scenario goals, informational only (capacity-only mode has no BW acceptance).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioGoals {
    pub target_peak_bw_bytes_s: Option<u64>,
    pub target_cus: Option<u32>,
    pub fidelity: Option<String>,
    pub measured_read_over_target_peak: Reading<f64>,
    pub note: String,
}

/// Run identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdentityRecord {
    pub identity_note: Option<String>,
    pub binary_path: Reading<String>,
    pub binary_sha256: Reading<String>,
    pub binary_md5: Reading<String>,
    pub model_md5: Reading<String>,
    pub sidecar_md5: Reading<String>,
    /// `value = null`, `reason = "no text prompt"`.
    pub prompt_md5: Reading<String>,
    /// Deterministic digest of the synthetic inputs (kernel set, seeds, sizes).
    pub input_digest: String,
    pub emu_id: Option<String>,
}

/// Before/after world-readable telemetry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub source: Reading<String>,
    pub before: Telemetry,
    pub after: Telemetry,
}

/// Process/run context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub pid: u32,
    pub fresh_process: Reading<bool>,
    pub timed_reps_per_kernel: u32,
    /// Order in which the checks actually ran.
    pub order: Vec<String>,
    pub fabric_job_marker: Reading<String>,
    pub hip_runtime: Reading<String>,
    pub driver_version: Reading<String>,
}

/// Schema-v1 receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmuValidationReceipt {
    pub schema_version: u32,
    pub kind: String,
    /// Aggregate of the requested capacity/host_topology/bandwidth/graph sections.
    pub status: Status,
    pub reason: Option<String>,
    pub timestamp_unix_s: u64,
    pub timestamp_utc: String,
    pub accounting_kind: String,
    /// Always `false`: no pacing, CU mask or bandwidth limiter is active.
    pub limiter_active: bool,
    pub profile: Option<EffectiveProfileRecord>,
    pub memory: MemoryRecord,
    pub physical: PhysicalRecord,
    pub capacity: CapacitySection,
    pub host_topology: TopologySection,
    pub bandwidth: BandwidthSection,
    pub graph: GraphSection,
    pub estimate: EstimateSection,
    pub raw_rates: RawRates,
    pub scenario_goals: ScenarioGoals,
    pub identity: IdentityRecord,
    pub telemetry: TelemetryRecord,
    pub run: RunRecord,
    pub fidelity_limits: Vec<String>,
}

impl EmuValidationReceipt {
    /// True when a required section (capacity, host_topology, graph) failed;
    /// the CLI exits nonzero on it. There is no single fidelity boolean.
    pub fn required_failed(&self) -> bool {
        [
            (self.capacity.status, self.capacity.requested),
            (self.host_topology.status, self.host_topology.requested),
            (self.graph.status, self.graph.requested),
        ]
        .iter()
        .any(|&(s, requested)| requested && s == Status::Fail)
    }

    /// Single-line JSON (the daemon response embeds the receipt value instead).
    pub fn to_json_line(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| e.to_string())
    }
}

/// Aggregate section statuses `(status, requested)`: any requested `Fail` →
/// `Fail`; else any requested `Unavailable` (or nothing requested) →
/// `Unavailable`; else `Pass`. Callers must not pass the estimate section.
pub fn aggregate_status(sections: &[(Status, bool)]) -> (Status, Option<String>) {
    let requested: Vec<Status> = sections
        .iter()
        .filter(|(_, r)| *r)
        .map(|(s, _)| *s)
        .collect();
    if requested.is_empty() {
        return (Status::Unavailable, Some("no test requested".to_string()));
    }
    if requested.contains(&Status::Fail) {
        return (Status::Fail, Some("a requested section failed".to_string()));
    }
    if requested.contains(&Status::Unavailable) {
        return (
            Status::Unavailable,
            Some("a requested section is unavailable".to_string()),
        );
    }
    (Status::Pass, None)
}

/// Fold per-size statuses: any `Fail` → `Fail`; no `Pass` → `Unavailable`; else `Pass`.
pub fn aggregate_sizes(statuses: &[Status]) -> Status {
    if statuses.contains(&Status::Fail) {
        Status::Fail
    } else if statuses.contains(&Status::Pass) {
        Status::Pass
    } else {
        Status::Unavailable
    }
}

fn aggregate_checks(checks: &[CheckRecord]) -> Status {
    if checks.is_empty() {
        Status::Unavailable
    } else if checks.iter().any(|c| c.status == Status::Fail) {
        Status::Fail
    } else {
        Status::Pass
    }
}

// ── Pure oracles ────────────────────────────────────────────────────────────

/// Deterministic pattern word at 32-bit index `index` (integer hash).
pub fn pattern_word(index: u64, seed: u32) -> u32 {
    let mut x = (index as u32) ^ ((index >> 32) as u32).wrapping_mul(0x85EB_CA6B) ^ seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// Expected `sink[t]` of `emu_read_checksum` over the pattern buffer.
pub fn thread_checksum(t: u64, stride: u64, n_vec: u64, seed: u32) -> u32 {
    let mut acc = FNV_OFFSET ^ (t as u32);
    let mut v = t;
    while v < n_vec {
        let base = v * 4;
        for k in 0..4 {
            acc = (acc ^ pattern_word(base + k, seed)).wrapping_mul(FNV_PRIME);
        }
        v += stride;
    }
    acc
}

/// Expected whole sink (`threads` slots) of `emu_read_checksum`.
pub fn expected_checksums(n_vec: u64, threads: u64, seed: u32) -> Vec<u32> {
    (0..threads)
        .into_par_iter()
        .map(|t| thread_checksum(t, threads, n_vec, seed))
        .collect()
}

/// Expected per-wave dependent chain of `emu_hwid_probe` (`aux[slot]`).
pub fn hwid_chain(slot: u32, iters: u32) -> u32 {
    let mut x = slot;
    for _ in 0..iters {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    x
}

/// FNV-1a 64 over the little-endian bytes of `words`, as 16 hex digits.
pub fn digest_words(words: &[u32]) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    format!("{h:016x}")
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Fill `buf` (whole 32-bit words) with the pattern starting at word `first_word`.
pub fn fill_pattern(buf: &mut [u8], first_word: u64, seed: u32) {
    buf.par_chunks_mut(4096)
        .enumerate()
        .for_each(|(ci, chunk)| {
            let base = first_word + (ci as u64) * 1024;
            for (j, w) in chunk.chunks_exact_mut(4).enumerate() {
                w.copy_from_slice(&pattern_word(base + j as u64, seed).to_le_bytes());
            }
        });
}

/// `(mismatching words, first mismatching word index)` of `bytes` against the
/// pattern starting at word `first_word`.
pub fn count_pattern_mismatches(bytes: &[u8], first_word: u64, seed: u32) -> (u64, Option<u64>) {
    bytes
        .par_chunks(4096)
        .enumerate()
        .map(|(ci, chunk)| {
            let base = first_word + (ci as u64) * 1024;
            let mut count = 0u64;
            let mut first = None;
            for (j, w) in chunk.chunks_exact(4).enumerate() {
                let got = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                if got != pattern_word(base + j as u64, seed) {
                    count += 1;
                    first.get_or_insert(base + j as u64);
                }
            }
            (count, first)
        })
        .reduce(
            || (0u64, None),
            |a, b| {
                (
                    a.0 + b.0,
                    match (a.1, b.1) {
                        (Some(x), Some(y)) => Some(x.min(y)),
                        (x, None) => x,
                        (None, y) => y,
                    },
                )
            },
        )
}

/// Compare an observed sink with the expected one.
pub fn compare_sinks(observed: &[u32], expected: &[u32]) -> (u64, Option<u64>) {
    let mut count = 0u64;
    let mut first = None;
    for (i, (a, b)) in observed.iter().zip(expected).enumerate() {
        if a != b {
            count += 1;
            first.get_or_insert(i as u64);
        }
    }
    let len_diff = observed.len().abs_diff(expected.len()) as u64;
    (count + len_diff, first)
}

/// Whether a traffic run of `bytes` (source + destination + guards + sink)
/// fits in `free_bytes` with headroom.
pub fn traffic_fits(bytes: u64, sink_bytes: u64, free_bytes: u64) -> bool {
    let need = bytes
        .saturating_mul(2)
        .saturating_add(2 * GUARD_BYTES as u64)
        .saturating_add(sink_bytes)
        .saturating_add(FIT_HEADROOM_BYTES);
    need <= free_bytes
}

/// Field-wise `after - before` (saturating).
pub fn usage_delta(after: &BudgetUsage, before: &BudgetUsage) -> BudgetUsage {
    BudgetUsage {
        device_bytes: after.device_bytes.saturating_sub(before.device_bytes),
        host_bytes: after.host_bytes.saturating_sub(before.host_bytes),
        gtt_bytes: after.gtt_bytes.saturating_sub(before.gtt_bytes),
        shared_host_bytes: after
            .shared_host_bytes
            .saturating_sub(before.shared_host_bytes),
    }
}

/// Device bytes still admissible under the cap (`limit - committed - pending`).
pub fn device_remaining(limit_bytes: u64, committed: &BudgetUsage, pending: &BudgetUsage) -> u64 {
    limit_bytes
        .saturating_sub(committed.device_bytes)
        .saturating_sub(pending.device_bytes)
}

/// Reconcile the physical CU sources. Present sources must agree; priority for
/// the chosen value is host facts, then KFD, then the HIP MP count.
pub fn reconcile_cu_sources(
    hip: Option<u32>,
    kfd: Option<u32>,
    facts: Option<u32>,
) -> Result<(u32, &'static str), String> {
    let present: Vec<(&'static str, u32)> =
        [("host_facts", facts), ("kfd", kfd), ("hip_mp_count", hip)]
            .into_iter()
            .filter_map(|(n, v)| v.filter(|&c| c > 0).map(|c| (n, c)))
            .collect();
    let Some(&(source, count)) = present.first() else {
        return Err("no physical CU count source is available".to_string());
    };
    if present.iter().any(|&(_, c)| c != count) {
        let listing = present
            .iter()
            .map(|(n, c)| format!("{n}={c}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("physical CU count sources disagree: {listing}"));
    }
    Ok((count, source))
}

/// KFD `properties` text → physical CUs (`simd_count / simd_per_cu`, default 2).
pub fn cus_from_kfd_properties(text: &str) -> Option<u32> {
    let prop = |name: &str| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(' ')?;
            (key == name).then(|| value.trim().parse::<u64>().ok())?
        })
    };
    let simds = prop("simd_count")?;
    let per_cu = prop("simd_per_cu").filter(|&v| v > 0).unwrap_or(2);
    if simds == 0 {
        return None;
    }
    u32::try_from(simds / per_cu).ok().filter(|&c| c > 0)
}

fn kfd_physical_cus(bdf: &str, nodes_dir: &Path) -> Result<u32, String> {
    let want = PciBdf::parse(bdf).ok_or_else(|| format!("unparseable PCI BDF `{bdf}`"))?;
    let gpus = enumerate_gpus(nodes_dir).map_err(|e| e.to_string())?;
    let node = gpus
        .iter()
        .find(|g| g.bdf == want)
        .ok_or_else(|| format!("no KFD GPU node for {bdf}"))?;
    let path = nodes_dir.join(node.node.to_string()).join("properties");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    cus_from_kfd_properties(&text)
        .ok_or_else(|| format!("{} has no usable simd_count", path.display()))
}

/// Accumulates decoded HW_ID1 observations across launches.
#[derive(Default)]
pub struct TopologyAccumulator {
    cus: BTreeSet<(u32, u32, u32, u32)>,
    wgps: BTreeSet<(u32, u32, u32)>,
    pub unwritten_slots: u64,
    pub chain_mismatch_slots: u64,
    /// Cumulative CU count after each absorbed launch.
    pub history: Vec<u32>,
}

impl TopologyAccumulator {
    /// Fold one launch: `hwid[slot]` (raw HW_ID1 or sentinel) and `aux[slot]`
    /// (expected `hwid_chain(slot, iters)`).
    pub fn absorb(&mut self, hwid: &[u32], aux: &[u32], iters: u32, layout: &HwIdLayout) {
        for (slot, (&raw, &chain)) in hwid.iter().zip(aux).enumerate() {
            if raw == HWID_SENTINEL {
                self.unwritten_slots += 1;
                continue;
            }
            if chain != hwid_chain(slot as u32, iters) {
                self.chain_mismatch_slots += 1;
            }
            let f = decode_hw_id1(raw, layout);
            self.cus.insert(f.cu_key());
            self.wgps.insert((f.se, f.sa, f.wgp));
        }
        self.history.push(self.cus.len() as u32);
    }

    pub fn cu_count(&self) -> u32 {
        self.cus.len() as u32
    }

    pub fn wgp_count(&self) -> u32 {
        self.wgps.len() as u32
    }

    pub fn labels(&self) -> Vec<String> {
        self.cus.iter().map(|&k| cu_label(k)).collect()
    }

    /// Reached `target`, or the last `window` launches added nothing.
    pub fn done(&self, target: u32, window: u32) -> bool {
        if self.cu_count() >= target {
            return true;
        }
        let n = self.history.len();
        let w = window as usize;
        n > w && self.history[n - 1] == self.history[n - 1 - w]
    }
}

/// Topology verdict. Pass iff observed == host physical CUs; contradictions in
/// the probe itself fail; everything else (missed or surplus CUs, unknown
/// layout, unwritten slots) is unavailable.
pub fn evaluate_topology(
    host_cus: u32,
    observed: u32,
    unwritten_slots: u64,
    chain_mismatch_slots: u64,
    layout_known: bool,
) -> (Status, Option<String>) {
    if !layout_known {
        return (
            Status::Unavailable,
            Some("no HW_ID1 layout for this arch".to_string()),
        );
    }
    if chain_mismatch_slots > 0 {
        return (
            Status::Fail,
            Some(format!(
                "{chain_mismatch_slots} probe waves returned a wrong finite-chain result"
            )),
        );
    }
    if unwritten_slots > 0 {
        return (
            Status::Unavailable,
            Some(format!(
                "{unwritten_slots} wave slots were never written (wave size assumption?)"
            )),
        );
    }
    if observed == host_cus {
        return (Status::Pass, None);
    }
    if observed < host_cus {
        return (
            Status::Unavailable,
            Some(format!(
                "observed {observed} of {host_cus} host CUs within the launch cap (sampling can miss CUs)"
            )),
        );
    }
    (
        Status::Unavailable,
        Some(format!(
            "observed {observed} CUs exceeds host {host_cus}; HW_ID1 layout assumption is inconsistent"
        )),
    )
}

/// `YYYY-MM-DDTHH:MM:SSZ` for a unix timestamp.
pub fn unix_to_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

// ── Telemetry (world-readable sysfs only) ───────────────────────────────────

#[cfg(unix)]
fn world_readable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.permissions().mode() & 0o004 == 0 {
        return Err(format!("{} is not world-readable", path.display()));
    }
    Ok(())
}

#[cfg(not(unix))]
fn world_readable(path: &Path) -> Result<(), String> {
    Err(format!(
        "{}: world-readable check unsupported on this platform",
        path.display()
    ))
}

fn read_world_readable_number(path: &Path) -> Result<f64, String> {
    world_readable(path)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.trim()
        .parse::<f64>()
        .map_err(|e| format!("{}: not a number: {e}", path.display()))
}

fn first_reading(dir: &Path, files: &[String], divisor: f64) -> Reading<f64> {
    let mut reasons = Vec::new();
    for file in files {
        match read_world_readable_number(&dir.join(file)) {
            Ok(v) => return Reading::known(v / divisor),
            Err(e) => reasons.push(e),
        }
    }
    Reading::unknown(reasons.join("; "))
}

/// The hwmon directory of PCI device `bdf` under `sys_root` (`/` in production).
pub fn hwmon_dir_at(sys_root: &Path, bdf: &str) -> Option<std::path::PathBuf> {
    let base = sys_root
        .join("sys/bus/pci/devices")
        .join(bdf.to_ascii_lowercase())
        .join("hwmon");
    let mut dirs: Vec<_> = std::fs::read_dir(base)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("hwmon"))
        .map(|e| e.path())
        .collect();
    dirs.sort();
    dirs.into_iter().next()
}

/// World-readable power/temperature/clock readings of the device; an
/// unreadable or missing file yields an `unknown` reading with a reason.
pub fn sample_telemetry_at(sys_root: &Path, bdf: &str) -> Telemetry {
    let Some(dir) = hwmon_dir_at(sys_root, bdf) else {
        let why = format!("no hwmon directory for {bdf}");
        return Telemetry {
            sclk_mhz: Reading::unknown(why.clone()),
            mclk_mhz: Reading::unknown(why.clone()),
            power_w: Reading::unknown(why.clone()),
            temperature_c: Reading::unknown(why),
        };
    };
    let power = first_reading(
        &dir,
        &["power1_average".to_string(), "power1_input".to_string()],
        1e6,
    );
    // Prefer the `edge` temperature, else temp1.
    let mut temp_files = Vec::new();
    for idx in 1..=8 {
        let label = std::fs::read_to_string(dir.join(format!("temp{idx}_label")))
            .map(|l| l.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if label == "edge" {
            temp_files.push(format!("temp{idx}_input"));
        }
    }
    temp_files.push("temp1_input".to_string());
    let temperature = first_reading(&dir, &temp_files, 1e3);
    let clock = |file: &str| -> Reading<u32> {
        match first_reading(&dir, &[file.to_string()], 1e6) {
            Reading { value: Some(v), .. } => Reading::known(v.round() as u32),
            Reading { reason, .. } => Reading::unknown(reason.unwrap_or_default()),
        }
    };
    Telemetry {
        sclk_mhz: clock("freq1_input"),
        mclk_mhz: clock("freq2_input"),
        power_w: power,
        temperature_c: temperature,
    }
}

// ── GPU plumbing ────────────────────────────────────────────────────────────

fn err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

fn read_tensor_bytes(gpu: &Gpu, t: &GpuTensor, len: usize) -> HipResult<Vec<u8>> {
    let mut out = vec![0u8; len];
    gpu.hip.memcpy_dtoh_at(&mut out, &t.buf, 0)?;
    Ok(out)
}

fn read_tensor_words(gpu: &Gpu, t: &GpuTensor, words: usize) -> HipResult<Vec<u32>> {
    Ok(bytes_to_words(&read_tensor_bytes(gpu, t, words * 4)?))
}

fn memset_tensor(gpu: &Gpu, t: &GpuTensor, value: i32, bytes: usize) -> HipResult<()> {
    gpu.hip.memset(&t.buf, value, bytes)?;
    gpu.hip.device_synchronize()
}

fn alloc_raw(gpu: &mut Gpu, bytes: usize) -> HipResult<GpuTensor> {
    gpu.alloc_tensor(&[bytes], DType::Raw)
}

fn load_kernels(gpu: &mut Gpu) -> HipResult<()> {
    for name in [KERNEL_HWID, KERNEL_READ, KERNEL_COPY] {
        gpu.ensure_kernel(MODULE_NAME, crate::kernels::EMU_VALIDATE_SRC, name)?;
    }
    Ok(())
}

fn args_hwid(hwid: &GpuTensor, aux: &GpuTensor, iters: u32) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(hwid.buf.as_ptr());
    b.push_ptr(aux.buf.as_ptr());
    b.push_u32(iters);
    b.pad_to(16);
    b.into_vec()
}

fn args_read(src: *const std::ffi::c_void, sink: &GpuTensor, n_vec: u64) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(src);
    b.push_ptr(sink.buf.as_ptr());
    b.push_u64(n_vec);
    b.pad_to(16);
    b.into_vec()
}

fn args_copy(dst: *const std::ffi::c_void, src: *const std::ffi::c_void, n_vec: u64) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(dst);
    b.push_ptr(src);
    b.push_u64(n_vec);
    b.pad_to(16);
    b.into_vec()
}

fn launch(gpu: &Gpu, kernel: &str, blocks: u32, args: &mut [u8]) -> HipResult<()> {
    gpu.launch_kernel_blob(kernel, [blocks, 1, 1], [BLOCK_THREADS, 1, 1], 0, args)
}

/// Start/stop events around one launch.
struct EventPair {
    start: Event,
    stop: Event,
}

impl EventPair {
    fn new(gpu: &Gpu) -> HipResult<Self> {
        Ok(Self {
            start: gpu.hip.event_create()?,
            stop: gpu.hip.event_create()?,
        })
    }

    /// Milliseconds the stream spent on `f`'s submitted work.
    fn measure(&self, gpu: &mut Gpu, f: impl FnOnce(&mut Gpu) -> HipResult<()>) -> HipResult<f64> {
        gpu.hip.event_record(&self.start, gpu.stream_ref())?;
        let result = f(gpu);
        gpu.hip.event_record(&self.stop, gpu.stream_ref())?;
        gpu.hip.event_synchronize(&self.stop)?;
        result?;
        Ok(f64::from(
            gpu.hip.event_elapsed_ms(&self.start, &self.stop)?,
        ))
    }

    fn destroy(self, gpu: &Gpu) {
        let _ = gpu.hip.event_destroy(self.start);
        let _ = gpu.hip.event_destroy(self.stop);
    }
}

fn rate_gb_s(bytes: u64, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e-3) / 1e9
}

fn distributions(bytes: u64, ms: Vec<f64>) -> HipResult<(Distribution, Distribution)> {
    let rates: Vec<f64> = ms.iter().map(|&m| rate_gb_s(bytes, m)).collect();
    let t = Distribution::from_samples(ms).ok_or_else(|| err("non-finite timing sample"))?;
    let r = Distribution::from_samples(rates).ok_or_else(|| err("non-finite rate sample"))?;
    Ok((t, r))
}

// ── Topology ────────────────────────────────────────────────────────────────

fn run_topology(gpu: &mut Gpu, host_cus: u32, layout: &HwIdLayout) -> HipResult<TopologySection> {
    let blocks = host_cus
        .checked_mul(TOPOLOGY_BLOCKS_PER_CU)
        .filter(|&b| b > 0 && b <= TRAFFIC_MAX_BLOCKS)
        .ok_or_else(|| err("topology grid size out of range"))?;
    let slots = (blocks * TOPOLOGY_WAVES_PER_BLOCK) as usize;
    let hwid = alloc_raw(gpu, slots * 4)?;
    let aux = alloc_raw(gpu, slots * 4)?;
    let mut args = args_hwid(&hwid, &aux, TOPOLOGY_CHAIN_ITERS);
    let mut acc = TopologyAccumulator::default();
    gpu.ensure_capture_stream()?;
    for _ in 0..TOPOLOGY_LAUNCH_CAP {
        memset_tensor(gpu, &hwid, 0xFF, slots * 4)?;
        memset_tensor(gpu, &aux, 0, slots * 4)?;
        launch(gpu, KERNEL_HWID, blocks, &mut args)?;
        gpu.hip.device_synchronize()?;
        let h = read_tensor_words(gpu, &hwid, slots)?;
        let a = read_tensor_words(gpu, &aux, slots)?;
        acc.absorb(&h, &a, TOPOLOGY_CHAIN_ITERS, layout);
        if acc.done(host_cus, TOPOLOGY_STABLE_WINDOW) {
            break;
        }
    }
    gpu.free_tensor(hwid)?;
    gpu.free_tensor(aux)?;
    let (status, reason) = evaluate_topology(
        host_cus,
        acc.cu_count(),
        acc.unwritten_slots,
        acc.chain_mismatch_slots,
        true,
    );
    Ok(TopologySection {
        status,
        requested: true,
        reason,
        launches: acc.history.len() as u32,
        launch_cap: TOPOLOGY_LAUNCH_CAP,
        blocks_per_launch: blocks,
        block_threads: BLOCK_THREADS,
        waves_per_block: TOPOLOGY_WAVES_PER_BLOCK,
        chain_iters: TOPOLOGY_CHAIN_ITERS,
        cumulative_cus_per_launch: acc.history.clone(),
        observed_cu_count: acc.cu_count(),
        observed_wgp_count: acc.wgp_count(),
        observed_hw_ids: acc.labels(),
        hw_id_layout: Some(layout.clone()),
        unwritten_slots: acc.unwritten_slots,
        chain_mismatch_slots: acc.chain_mismatch_slots,
        compared_against_host_cus: Some(host_cus),
        compared_against_target_cus: false,
    })
}

// ── Capacity ────────────────────────────────────────────────────────────────

fn run_capacity(gpu: &mut Gpu) -> HipResult<CapacitySection> {
    let (Some(_profile), Some(snap0)) = (gpu.emu_profile(), gpu.emu_budget_snapshot()) else {
        return Ok(CapacitySection::unavailable(true, "no profile"));
    };
    let Some(id) = emu_budget::active() else {
        return Ok(CapacitySection::unavailable(true, "no installed budget"));
    };
    let limits = snap0.limits;
    let mut checks: Vec<CheckRecord> = Vec::new();
    let mut sec = CapacitySection::unavailable(true, "pending");
    sec.reason = None;

    // total == profile bytes
    let (free0, total0) = gpu.device_mem_info()?;
    sec.profile_total_bytes = Some(limits.device_bytes);
    sec.observed_total_bytes = Some(total0 as u64);
    checks.push(check(
        "total_equals_profile_bytes",
        total0 as u64 == limits.device_bytes,
        format!(
            "device_mem_info total {total0} vs profile device bytes {}",
            limits.device_bytes
        ),
    ));
    if let Some(p) = gpu.emu_profile() {
        if let MemoryKind::Discrete { capacity_bytes } = p.memory {
            checks.push(check(
                "limit_equals_advertised_capacity",
                limits.device_bytes == capacity_bytes,
                format!(
                    "ledger device limit {} vs advertised {capacity_bytes}",
                    limits.device_bytes
                ),
            ));
        }
    }

    // Controlled small malloc.
    let remaining0 = device_remaining(limits.device_bytes, &snap0.committed, &snap0.pending);
    if (CONTROLLED_MALLOC_BYTES as u64) > remaining0 {
        checks.push(check(
            "controlled_malloc",
            false,
            format!("remaining cap {remaining0} B is below the {CONTROLLED_MALLOC_BYTES} B controlled allocation"),
        ));
    } else {
        sec.controlled_malloc_bytes = Some(CONTROLLED_MALLOC_BYTES as u64);
        match alloc_raw(gpu, CONTROLLED_MALLOC_BYTES) {
            Err(e) => checks.push(check(
                "controlled_malloc",
                false,
                format!("allocation failed: {e}"),
            )),
            Ok(t) => {
                let size = t.buf.size() as u64;
                let snap1 = gpu
                    .emu_budget_snapshot()
                    .ok_or_else(|| err("budget vanished"))?;
                let expected = charge_vector(&limits, AllocationClass::Device, size);
                let delta = usage_delta(&snap1.committed, &snap0.committed);
                checks.push(check(
                    "malloc_moves_ledger_by_charge",
                    delta == expected && snap1.pending == snap0.pending,
                    format!("committed delta {delta:?} vs charge {expected:?} for {size} B"),
                ));
                let (free1, _) = gpu.device_mem_info()?;
                let rem1 = device_remaining(limits.device_bytes, &snap1.committed, &snap1.pending);
                let budget_bound =
                    !limits.uma && free0 as u64 == remaining0 && free1 as u64 == rem1;
                if budget_bound {
                    checks.push(check(
                        "malloc_moves_free_by_charge",
                        free0 as u64 - free1 as u64 == expected.device_bytes,
                        format!("free {free0} -> {free1}, charge {}", expected.device_bytes),
                    ));
                } else {
                    checks.push(check(
                        "malloc_does_not_raise_free",
                        free1 <= free0,
                        format!(
                            "real-free clamp or UMA clamp is tighter than the ledger; free {free0} -> {free1}"
                        ),
                    ));
                }

                // Free to pool: charge stays, bytes are parked.
                gpu.free_tensor(t)?;
                let snap2 = gpu
                    .emu_budget_snapshot()
                    .ok_or_else(|| err("budget vanished"))?;
                checks.push(check(
                    "pool_free_keeps_charge",
                    snap2.committed == snap1.committed && gpu.pool_parked_bytes() as u64 >= size,
                    format!(
                        "committed {:?} -> {:?}; pool parked {} B",
                        snap1.committed,
                        snap2.committed,
                        gpu.pool_parked_bytes()
                    ),
                ));

                // Reuse from the pool: no new HIP allocation, charge unchanged.
                let stats0 = gpu.pool_stats();
                let t2 = alloc_raw(gpu, CONTROLLED_MALLOC_BYTES)?;
                let stats1 = gpu.pool_stats();
                let snap3 = gpu
                    .emu_budget_snapshot()
                    .ok_or_else(|| err("budget vanished"))?;
                checks.push(check(
                    "pool_reuse_keeps_charge",
                    stats1.0 == stats0.0
                        && stats1.1 == stats0.1 + 1
                        && snap3.committed == snap1.committed,
                    format!(
                        "pool (new, reused) {:?}/{:?} -> {:?}/{:?}",
                        stats0.0, stats0.1, stats1.0, stats1.1
                    ),
                ));
                gpu.free_tensor(t2)?;

                // Drain releases the charge of every parked buffer: the controlled one
                // and any buffer an earlier section parked (so compare against the
                // pre-drain parked bytes, not the section's opening snapshot).
                let parked_pre = gpu.pool_parked_bytes() as u64;
                let snap_pre = gpu
                    .emu_budget_snapshot()
                    .ok_or_else(|| err("budget vanished"))?;
                gpu.drain_pool();
                let snap4 = gpu
                    .emu_budget_snapshot()
                    .ok_or_else(|| err("budget vanished"))?;
                let expected_after = snap_pre.committed.device_bytes.checked_sub(parked_pre);
                checks.push(check(
                    "pool_drain_releases_charge",
                    parked_pre >= size
                        && expected_after == Some(snap4.committed.device_bytes)
                        && gpu.pool_parked_bytes() == 0,
                    format!(
                        "device committed {} -> {} after draining {} parked B (controlled {} B)",
                        snap_pre.committed.device_bytes,
                        snap4.committed.device_bytes,
                        parked_pre,
                        size
                    ),
                ));
            }
        }
    }

    // Over-cap reservation: refused by the ledger, never reaching HIP.
    let snap = gpu
        .emu_budget_snapshot()
        .ok_or_else(|| err("budget vanished"))?;
    let remaining = device_remaining(limits.device_bytes, &snap.committed, &snap.pending);
    match remaining.checked_add(1) {
        None => checks.push(check(
            "over_cap_refused",
            false,
            "remaining + 1 overflows u64".to_string(),
        )),
        Some(request) => {
            sec.over_cap_request_bytes = Some(request);
            let stats_before = gpu.pool_stats();
            let preflight = emu_budget::check_additional(id, AllocationClass::Device, request);
            let reserved = emu_budget::reserve(id, AllocationClass::Device, request);
            let refused = matches!(&preflight, Err(e) if e.is_out_of_memory())
                && matches!(&reserved, Err(e) if e.is_out_of_memory());
            drop(reserved);
            let snap_after = gpu
                .emu_budget_snapshot()
                .ok_or_else(|| err("budget vanished"))?;
            let untouched = snap_after.pending == snap.pending
                && snap_after.committed == snap.committed
                && gpu.pool_stats() == stats_before;
            sec.over_cap_reached_hip = false;
            checks.push(check(
                "over_cap_refused",
                refused,
                format!("request {request} B vs remaining {remaining} B: check_additional/reserve both refused as out-of-memory"),
            ));
            checks.push(check(
                "over_cap_refusal_leaves_ledger_and_hip_untouched",
                untouched,
                "ledger committed/pending and pool allocation counters unchanged; no HIP allocation was issued"
                    .to_string(),
            ));
        }
    }

    sec.status = aggregate_checks(&checks);
    sec.checks = checks;
    sec.reason = match sec.status {
        Status::Pass => None,
        _ => Some("a capacity check did not pass".to_string()),
    };
    Ok(sec)
}

// ── Traffic ─────────────────────────────────────────────────────────────────

struct SizeOutcome {
    bandwidth: BandwidthSize,
    graph: GraphSize,
}

fn upload_pattern(gpu: &Gpu, src: &GpuTensor, bytes: usize, seed: u32) -> HipResult<()> {
    let mut chunk = vec![0u8; CHUNK_BYTES.min(bytes)];
    let mut off = 0usize;
    while off < bytes {
        let n = CHUNK_BYTES.min(bytes - off);
        fill_pattern(&mut chunk[..n], (off / 4) as u64, seed);
        gpu.hip.memcpy_htod_offset(&src.buf, off, &chunk[..n])?;
        off += n;
    }
    Ok(())
}

/// Read back the copy destination and compare it to the pattern, plus guards.
fn verify_dst(
    gpu: &Gpu,
    dst: &GpuTensor,
    bytes: usize,
    seed: u32,
    reps: u32,
) -> HipResult<ChecksumRecord> {
    let mut chunk = vec![0u8; CHUNK_BYTES.min(bytes)];
    let mut mismatched = 0u64;
    let mut first = None;
    let mut off = 0usize;
    while off < bytes {
        let n = CHUNK_BYTES.min(bytes - off);
        gpu.hip
            .memcpy_dtoh_at(&mut chunk[..n], &dst.buf, GUARD_BYTES + off)?;
        let (c, f) = count_pattern_mismatches(&chunk[..n], (off / 4) as u64, seed);
        mismatched += c;
        if first.is_none() {
            first = f;
        }
        off += n;
    }
    let mut lo = vec![0u8; GUARD_BYTES];
    let mut hi = vec![0u8; GUARD_BYTES];
    gpu.hip.memcpy_dtoh_at(&mut lo, &dst.buf, 0)?;
    gpu.hip
        .memcpy_dtoh_at(&mut hi, &dst.buf, GUARD_BYTES + bytes)?;
    let guard_ok = lo.iter().chain(hi.iter()).all(|&b| b == POISON_BYTE as u8);
    // Both guard regions must still hold the poison byte.
    Ok(ChecksumRecord {
        kind: "dst_full_compare".to_string(),
        compared: (bytes / 4) as u64,
        mismatched,
        first_mismatch: first,
        expected_digest: None,
        observed_digest: None,
        guard_intact: Some(guard_ok),
        verified_reps: reps,
    })
}

fn sink_record(observed: &[u32], expected: &[u32], reps: u32) -> ChecksumRecord {
    let (mismatched, first) = compare_sinks(observed, expected);
    ChecksumRecord {
        kind: "per_thread_fnv1a_u32".to_string(),
        compared: expected.len() as u64,
        mismatched,
        first_mismatch: first,
        expected_digest: Some(digest_words(expected)),
        observed_digest: Some(digest_words(observed)),
        guard_intact: None,
        verified_reps: reps,
    }
}

fn dst_ok(c: &ChecksumRecord) -> bool {
    c.mismatched == 0 && c.guard_intact == Some(true)
}

/// Eager read/copy then graph read/copy for one buffer size.
fn run_size(
    gpu: &mut Gpu,
    request: &EmuValidationRequest,
    mib: u32,
    host_cus: u32,
    do_eager_report: bool,
    do_graph: bool,
) -> HipResult<SizeOutcome> {
    let bytes = (mib as usize) << 20;
    let n_vec = (bytes / 16) as u64;
    let blocks = (host_cus.max(1) * TRAFFIC_BLOCKS_PER_CU).min(TRAFFIC_MAX_BLOCKS);
    let threads = u64::from(blocks) * u64::from(BLOCK_THREADS);
    let sink_bytes = threads * 4;
    let seed = 0x5EED_0000 ^ mib;
    let reps = request.passes;

    let unavailable = |reason: String| SizeOutcome {
        bandwidth: BandwidthSize {
            buffer_mib: mib,
            bytes: bytes as u64,
            status: Status::Unavailable,
            reason: Some(reason.clone()),
            read: None,
            copy: None,
        },
        graph: GraphSize {
            buffer_mib: mib,
            status: Status::Unavailable,
            reason: Some(reason),
            read: None,
            copy: None,
        },
    };

    let (free, _) = gpu.device_mem_info()?;
    if !traffic_fits(bytes as u64, sink_bytes, free as u64) {
        return Ok(unavailable(format!(
            "{mib} MiB source + destination does not fit in the budgeted free {free} B"
        )));
    }
    let src = match alloc_raw(gpu, bytes) {
        Ok(t) => t,
        Err(e) => return Ok(unavailable(format!("source allocation refused: {e}"))),
    };
    let dst = match alloc_raw(gpu, bytes + 2 * GUARD_BYTES) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(src);
            return Ok(unavailable(format!("destination allocation refused: {e}")));
        }
    };
    let sink = match alloc_raw(gpu, sink_bytes as usize) {
        Ok(t) => t,
        Err(e) => {
            let _ = gpu.free_tensor(src);
            let _ = gpu.free_tensor(dst);
            return Ok(unavailable(format!("sink allocation refused: {e}")));
        }
    };

    let result = run_size_buffers(
        gpu,
        mib,
        bytes,
        n_vec,
        blocks,
        threads,
        seed,
        reps,
        &src,
        &dst,
        &sink,
        do_eager_report,
        do_graph,
    );
    let _ = gpu.free_tensor(src);
    let _ = gpu.free_tensor(dst);
    let _ = gpu.free_tensor(sink);
    gpu.drain_pool();
    result
}

#[allow(clippy::too_many_arguments)]
fn run_size_buffers(
    gpu: &mut Gpu,
    mib: u32,
    bytes: usize,
    n_vec: u64,
    blocks: u32,
    threads: u64,
    seed: u32,
    reps: u32,
    src: &GpuTensor,
    dst: &GpuTensor,
    sink: &GpuTensor,
    do_eager_report: bool,
    do_graph: bool,
) -> HipResult<SizeOutcome> {
    let sink_words = threads as usize;
    let dst_ptr =
        unsafe { (dst.buf.as_ptr() as *mut u8).add(GUARD_BYTES) } as *const std::ffi::c_void;
    let src_ptr = src.buf.as_ptr() as *const std::ffi::c_void;
    let mut read_args = args_read(src_ptr, sink, n_vec);
    let mut copy_args = args_copy(dst_ptr, src_ptr, n_vec);
    let expected = expected_checksums(n_vec, threads, seed);

    upload_pattern(gpu, src, bytes, seed)?;
    gpu.hip.device_synchronize()?;
    let events = EventPair::new(gpu)?;

    // ── Eager read ──
    for _ in 0..WARMUP_REPS {
        launch(gpu, KERNEL_READ, blocks, &mut read_args)?;
    }
    gpu.hip.device_synchronize()?;
    let mut read_ms = Vec::with_capacity(reps as usize);
    let mut read_mismatch = 0u64;
    let mut read_first = None;
    let mut last_sink: Vec<u32> = Vec::new();
    for _ in 0..reps {
        memset_tensor(gpu, sink, 0xFF, sink_words * 4)?;
        let ms = events.measure(gpu, |g| launch(g, KERNEL_READ, blocks, &mut read_args))?;
        read_ms.push(ms);
        last_sink = read_tensor_words(gpu, sink, sink_words)?;
        let (c, f) = compare_sinks(&last_sink, &expected);
        read_mismatch += c;
        if read_first.is_none() {
            read_first = f;
        }
    }
    let mut read_check = sink_record(&last_sink, &expected, reps);
    read_check.mismatched = read_mismatch;
    read_check.first_mismatch = read_first;
    let (read_t, read_r) = distributions(bytes as u64, read_ms)?;

    // ── Eager copy ──
    memset_tensor(gpu, dst, POISON_BYTE, bytes + 2 * GUARD_BYTES)?;
    for _ in 0..WARMUP_REPS {
        launch(gpu, KERNEL_COPY, blocks, &mut copy_args)?;
    }
    gpu.hip.device_synchronize()?;
    let mut copy_ms = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let ms = events.measure(gpu, |g| launch(g, KERNEL_COPY, blocks, &mut copy_args))?;
        copy_ms.push(ms);
    }
    let copy_check = verify_dst(gpu, dst, bytes, seed, reps)?;
    let copy_bytes = 2 * bytes as u64;
    let (copy_t, copy_r) = distributions(copy_bytes, copy_ms)?;

    let read_pass = read_check.mismatched == 0;
    let copy_pass = dst_ok(&copy_check);
    let bw_status = if read_pass && copy_pass {
        Status::Pass
    } else {
        Status::Fail
    };
    let bandwidth = BandwidthSize {
        buffer_mib: mib,
        bytes: bytes as u64,
        status: bw_status,
        reason: (bw_status == Status::Fail)
            .then(|| "checksum, destination or guard mismatch".to_string()),
        read: do_eager_report.then(|| TrafficRecord {
            kernel: KERNEL_READ.to_string(),
            grid_blocks: blocks,
            block_threads: BLOCK_THREADS,
            bytes_per_pass: bytes as u64,
            bytes_basis: "logical bytes read".to_string(),
            time_ms: read_t,
            rate_gb_s: read_r,
            checks: read_check.clone(),
        }),
        copy: do_eager_report.then(|| TrafficRecord {
            kernel: KERNEL_COPY.to_string(),
            grid_blocks: blocks,
            block_threads: BLOCK_THREADS,
            bytes_per_pass: copy_bytes,
            bytes_basis: "logical bytes read + bytes written".to_string(),
            time_ms: copy_t,
            rate_gb_s: copy_r,
            checks: copy_check.clone(),
        }),
    };

    // ── Graph replay ──
    let graph = if !do_graph {
        GraphSize {
            buffer_mib: mib,
            status: Status::Unavailable,
            reason: Some("not requested".to_string()),
            read: None,
            copy: None,
        }
    } else {
        match run_graph_size(
            gpu,
            mib,
            bytes,
            blocks,
            seed,
            reps,
            sink,
            dst,
            &expected,
            &last_sink,
            &mut read_args,
            &mut copy_args,
            &events,
        ) {
            Ok(g) => g,
            Err(e) => GraphSize {
                buffer_mib: mib,
                status: Status::Unavailable,
                reason: Some(format!("graph capture/replay unavailable: {e}")),
                read: None,
                copy: None,
            },
        }
    };
    events.destroy(gpu);
    Ok(SizeOutcome { bandwidth, graph })
}

/// Capture one launch into a graph (no allocation inside the capture).
fn capture_launch(
    gpu: &mut Gpu,
    kernel: &str,
    blocks: u32,
    args: &mut [u8],
) -> HipResult<(hip_bridge::Graph, hip_bridge::GraphExec)> {
    gpu.hip.device_synchronize()?;
    gpu.begin_stream_capture()?;
    let launched = launch(gpu, kernel, blocks, args);
    let graph = gpu.end_stream_capture();
    launched?;
    let graph = graph?;
    match gpu.hip.graph_instantiate(&graph) {
        Ok(exec) => Ok((graph, exec)),
        Err(e) => {
            let _ = gpu.hip.graph_destroy(graph);
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_graph_size(
    gpu: &mut Gpu,
    mib: u32,
    bytes: usize,
    blocks: u32,
    seed: u32,
    reps: u32,
    sink: &GpuTensor,
    dst: &GpuTensor,
    expected: &[u32],
    eager_sink: &[u32],
    read_args: &mut [u8],
    copy_args: &mut [u8],
    events: &EventPair,
) -> HipResult<GraphSize> {
    let sink_words = expected.len();
    gpu.ensure_capture_stream()?;

    // Read graph. `read_args` stays alive until after `graph_destroy`: HIP
    // records the kernarg pointer, not the bytes.
    let (rg, rexec) = capture_launch(gpu, KERNEL_READ, blocks, read_args)?;
    let read_outcome = (|| -> HipResult<GraphTraffic> {
        let mut ms = Vec::with_capacity(reps as usize);
        let mut mismatch = 0u64;
        let mut first = None;
        let mut vs_eager_ok = true;
        let mut last = Vec::new();
        for i in 0..(WARMUP_REPS + reps) {
            memset_tensor(gpu, sink, 0xFF, sink_words * 4)?;
            let t = events.measure(gpu, |g| g.launch_graph(&rexec))?;
            last = read_tensor_words(gpu, sink, sink_words)?;
            let (c, f) = compare_sinks(&last, expected);
            mismatch += c;
            if first.is_none() {
                first = f;
            }
            vs_eager_ok &= last == eager_sink;
            if i >= WARMUP_REPS {
                ms.push(t);
            }
        }
        let mut check = sink_record(&last, expected, WARMUP_REPS + reps);
        check.mismatched = mismatch;
        check.first_mismatch = first;
        let (t, r) = distributions(bytes as u64, ms)?;
        Ok(GraphTraffic {
            kernel: KERNEL_READ.to_string(),
            replays: WARMUP_REPS + reps,
            bytes_per_pass: bytes as u64,
            time_ms: t,
            rate_gb_s: r,
            matches_cpu_oracle: mismatch == 0,
            matches_eager: vs_eager_ok,
            checks: check,
        })
    })();
    let _ = gpu.hip.graph_exec_destroy(rexec);
    let _ = gpu.hip.graph_destroy(rg);
    let read = read_outcome?;

    // Copy graph.
    let (cg, cexec) = capture_launch(gpu, KERNEL_COPY, blocks, copy_args)?;
    let copy_outcome = (|| -> HipResult<GraphTraffic> {
        let mut ms = Vec::with_capacity(reps as usize);
        let mut all_ok = true;
        let mut mismatched = 0u64;
        let mut verified = 0u32;
        let mut last_check = None;
        for i in 0..(WARMUP_REPS + reps) {
            // Poison, replay, verify the full destination every replay.
            memset_tensor(gpu, dst, POISON_BYTE, bytes + 2 * GUARD_BYTES)?;
            let t = events.measure(gpu, |g| g.launch_graph(&cexec))?;
            let c = verify_dst(gpu, dst, bytes, seed, 1)?;
            verified += 1;
            all_ok &= dst_ok(&c);
            mismatched += c.mismatched;
            last_check = Some(c);
            if i >= WARMUP_REPS {
                ms.push(t);
            }
        }
        let mut check = last_check.ok_or_else(|| err("no graph copy replay ran"))?;
        check.mismatched = mismatched;
        check.verified_reps = verified;
        let copy_bytes = 2 * bytes as u64;
        let (t, r) = distributions(copy_bytes, ms)?;
        Ok(GraphTraffic {
            kernel: KERNEL_COPY.to_string(),
            replays: WARMUP_REPS + reps,
            bytes_per_pass: copy_bytes,
            time_ms: t,
            rate_gb_s: r,
            matches_cpu_oracle: all_ok,
            // Eager's full destination compare passed (checked by the caller's
            // bandwidth status); equal-to-oracle on both paths means byte-identical.
            matches_eager: all_ok,
            checks: check,
        })
    })();
    let _ = gpu.hip.graph_exec_destroy(cexec);
    let _ = gpu.hip.graph_destroy(cg);
    let copy = copy_outcome?;

    let ok = read.matches_cpu_oracle
        && read.matches_eager
        && copy.matches_cpu_oracle
        && copy.matches_eager
        && copy.checks.guard_intact == Some(true);
    Ok(GraphSize {
        buffer_mib: mib,
        status: if ok { Status::Pass } else { Status::Fail },
        reason: (!ok)
            .then(|| "graph replay differs from the CPU oracle or eager result".to_string()),
        read: Some(read),
        copy: Some(copy),
    })
}

// ── Identity helpers ────────────────────────────────────────────────────────

fn binary_identity() -> (Reading<String>, Reading<String>) {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            let why = format!("current_exe unavailable: {e}");
            return (Reading::unknown(why.clone()), Reading::unknown(why));
        }
    };
    let path = Reading::known(exe.display().to_string());
    match std::fs::read(&exe) {
        Ok(bytes) => {
            let digest = Sha256::digest(&bytes);
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            (path, Reading::known(hex))
        }
        Err(e) => (path, Reading::unknown(format!("{}: {e}", exe.display()))),
    }
}

/// Digest of the deterministic synthetic inputs (kernel set, seeds, sizes, caps).
pub fn input_digest(request: &EmuValidationRequest) -> String {
    let mut h = Sha256::new();
    h.update(b"emu_validate/v1\n");
    h.update(format!("kernels={KERNEL_HWID},{KERNEL_READ},{KERNEL_COPY}\n").as_bytes());
    h.update(format!("fnv={FNV_OFFSET:08x},{FNV_PRIME:08x}\n").as_bytes());
    h.update(
        format!("topology_cap={TOPOLOGY_LAUNCH_CAP} chain={TOPOLOGY_CHAIN_ITERS}\n").as_bytes(),
    );
    for mib in &request.buffer_mib {
        h.update(format!("size={mib} seed={:08x}\n", 0x5EED_0000u32 ^ mib).as_bytes());
    }
    h.update(format!("passes={} warmup={WARMUP_REPS}\n", request.passes).as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn amdgpu_driver_version() -> Reading<String> {
    let path = Path::new("/sys/module/amdgpu/version");
    match world_readable(path)
        .and_then(|_| std::fs::read_to_string(path).map_err(|e| e.to_string()))
    {
        Ok(v) if !v.trim().is_empty() => Reading::known(v.trim().to_string()),
        Ok(_) => Reading::unknown("/sys/module/amdgpu/version is empty"),
        Err(e) => Reading::unknown(e),
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Entry point ─────────────────────────────────────────────────────────────

/// Run the requested validator checks on `gpu` and build the receipt.
///
/// Errors (rather than a receipt) for: an invalid request, an unsupported
/// physical arch, an active profile different from `request.profile_id`, and a
/// Gpu that already holds a model-sized managed allocation.
pub fn validate(gpu: &mut Gpu, request: &EmuValidationRequest) -> HipResult<EmuValidationReceipt> {
    request
        .validate()
        .map_err(|e| err(format!("emu_validate: invalid request: {e}")))?;
    gpu.bind_thread()?;
    let arch = gpu.arch.clone();
    if !SUPPORTED_ARCHES.contains(&arch.as_str()) {
        return Err(err(format!(
            "emu_validate: unsupported physical arch `{arch}` (supported: {})",
            SUPPORTED_ARCHES.join(", ")
        )));
    }
    let profile = gpu.emu_profile();
    if let Some(p) = profile {
        if p.id != request.profile_id {
            return Err(err(format!(
                "emu_validate: request profile `{}` does not match the active profile `{}`",
                request.profile_id, p.id
            )));
        }
    }
    let snapshot = gpu.emu_budget_snapshot();
    if let Some(s) = &snapshot {
        if s.committed.device_bytes > MODEL_FREE_BASELINE_LIMIT_BYTES {
            return Err(err(format!(
                "emu_validate requires a model-free Gpu: {} B already committed on the device",
                s.committed.device_bytes
            )));
        }
    }

    let bdf = gpu.hip.device_pci_bus_id(gpu.device_id)?;
    let timestamp = now_unix();
    let telemetry_before = sample_telemetry_at(Path::new("/"), &bdf);

    // Memory and physical identity.
    let (raw_free, raw_total) = gpu.hip.get_vram_info()?;
    let (budget_free, budget_total) = gpu.device_mem_info()?;
    let memory = MemoryRecord {
        advertised_capacity_bytes: profile.and_then(|p| match p.memory {
            MemoryKind::Discrete { capacity_bytes } => Some(capacity_bytes),
            MemoryKind::Uma => snapshot.map(|s| s.limits.device_bytes),
        }),
        raw_real_free_bytes: raw_free as u64,
        raw_real_total_bytes: raw_total as u64,
        budgeted_free_bytes: budget_free as u64,
        budgeted_total_bytes: budget_total as u64,
        limits: snapshot.map(|s| LimitsRecord {
            device_bytes: s.limits.device_bytes,
            gtt_bytes: s.limits.gtt_bytes,
            shared_host_bytes: s.limits.shared_host_bytes,
            uma: s.limits.uma,
        }),
        committed: snapshot.map(|s| s.committed.into()),
        pending: snapshot.map(|s| s.pending.into()),
        pool_parked_bytes: gpu.pool_parked_bytes() as u64,
        accounting_kind: "managed_application_charge".to_string(),
    };

    let mp = gpu.hip.get_device_attribute(
        crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        gpu.device_id,
    )?;
    let hip_mp_count = u32::try_from(mp).unwrap_or(0);
    let hip_physical = wgp_to_physical_cus(hip_mp_count, &arch);
    let kfd = kfd_physical_cus(&bdf, &DeviceRoot::system().kfd_topology());
    let facts_cus = gpu.emu_host_facts().map(|f| f.physical_cus);
    let reconciled =
        reconcile_cu_sources(Some(hip_physical), kfd.as_ref().ok().copied(), facts_cus);
    let physical = PhysicalRecord {
        pci_bdf: bdf.clone(),
        physical_gfx: arch.clone(),
        target_gfx: profile.map(|p| p.target_gfx.to_string()),
        target_cus: profile.map(|p| p.target_cus),
        hip_mp_count,
        hip_physical_cus: hip_physical,
        kfd_physical_cus: match &kfd {
            Ok(c) => Reading::known(*c),
            Err(e) => Reading::unknown(e.clone()),
        },
        host_facts_physical_cus: facts_cus,
        host_physical_cus: reconciled.as_ref().ok().map(|&(c, _)| c),
        cu_count_source: reconciled.as_ref().ok().map(|&(_, s)| s.to_string()),
        cu_count_disagreement: reconciled.as_ref().err().cloned(),
        cu_emulation: "none".to_string(),
    };

    let mut order: Vec<String> = Vec::new();
    let want_bw = request.requested("bandwidth");
    let want_graph = request.requested("graph");

    // Topology.
    let host_topology = if !request.requested("topology") {
        TopologySection::unavailable(false, "not requested")
    } else {
        order.push("topology".to_string());
        match &reconciled {
            Err(e) => {
                // Source disagreement is a contradiction in the host facts: fail.
                let mut s = TopologySection::unavailable(true, e.clone());
                if e.contains("disagree") {
                    s.status = Status::Fail;
                }
                s
            }
            Ok((host_cus, _)) => match hw_id1_layout(&arch) {
                None => TopologySection::unavailable(true, "no HW_ID1 layout for this arch"),
                Some(layout) => {
                    load_kernels(gpu)?;
                    run_topology(gpu, *host_cus, &layout)?
                }
            },
        }
    };

    // Capacity.
    let capacity = if !request.requested("capacity") {
        CapacitySection::unavailable(false, "not requested")
    } else {
        order.push("capacity".to_string());
        run_capacity(gpu)?
    };

    // Traffic (eager checks always run when either bandwidth or graph is requested).
    let mut bandwidth = BandwidthSection {
        status: Status::Unavailable,
        requested: want_bw,
        reason: Some("not requested".to_string()),
        warmup_reps: WARMUP_REPS,
        timed_reps: request.passes,
        sizes: Vec::new(),
    };
    let mut graph = GraphSection {
        status: Status::Unavailable,
        requested: want_graph,
        reason: Some("not requested".to_string()),
        sizes: Vec::new(),
    };
    if want_bw || want_graph {
        match &reconciled {
            Err(e) => {
                let why = format!("host CU count unknown: {e}");
                if want_bw {
                    bandwidth.reason = Some(why.clone());
                }
                if want_graph {
                    graph.reason = Some(why);
                }
            }
            Ok((host_cus, _)) => {
                let host_cus = *host_cus;
                load_kernels(gpu)?;
                gpu.ensure_capture_stream()?;
                if want_bw {
                    order.push("bandwidth".to_string());
                }
                if want_graph {
                    order.push("graph".to_string());
                }
                for &mib in &request.buffer_mib {
                    let outcome = run_size(gpu, request, mib, host_cus, want_bw, want_graph)?;
                    bandwidth.sizes.push(outcome.bandwidth);
                    graph.sizes.push(outcome.graph);
                }
                if want_bw {
                    let st: Vec<Status> = bandwidth.sizes.iter().map(|s| s.status).collect();
                    bandwidth.status = aggregate_sizes(&st);
                    bandwidth.reason = (bandwidth.status != Status::Pass)
                        .then(|| "no size completed a passing traffic run".to_string());
                }
                if want_graph {
                    let st: Vec<Status> = graph.sizes.iter().map(|s| s.status).collect();
                    graph.status = aggregate_sizes(&st);
                    graph.reason = (graph.status != Status::Pass)
                        .then(|| "no size completed a passing graph replay".to_string());
                }
            }
        }
    }

    // Final cleanup: leave the pool empty so the ledger returns to baseline.
    gpu.drain_pool();
    let telemetry_after = sample_telemetry_at(Path::new("/"), &bdf);

    // Rates and goals from the largest passing eager size.
    let best = bandwidth
        .sizes
        .iter()
        .filter(|s| s.status == Status::Pass && s.read.is_some() && s.copy.is_some())
        .max_by_key(|s| s.buffer_mib);
    let read_rate = best
        .and_then(|s| s.read.as_ref())
        .map(|r| r.rate_gb_s.median * 1e9);
    let copy_rate = best
        .and_then(|s| s.copy.as_ref())
        .map(|r| r.rate_gb_s.median * 1e9);
    let no_rate = "no passing eager bandwidth size";
    let raw_rates = RawRates {
        read_bytes_s: read_rate.map_or_else(|| Reading::unknown(no_rate), Reading::known),
        copy_rw_bytes_s: copy_rate.map_or_else(|| Reading::unknown(no_rate), Reading::known),
        basis: best.map_or_else(
            || "none".to_string(),
            |s| {
                format!(
                    "median of {} eager reps at buffer_mib={}",
                    request.passes, s.buffer_mib
                )
            },
        ),
    };
    let target_peak = profile.map(|p| p.peak_bw_bytes_s);
    let scenario_goals = ScenarioGoals {
        target_peak_bw_bytes_s: target_peak,
        target_cus: profile.map(|p| p.target_cus),
        fidelity: profile.map(|p| EffectiveProfileRecord::from_profile(p).fidelity),
        measured_read_over_target_peak: match (read_rate, target_peak) {
            (Some(r), Some(t)) if t > 0 => Reading::known(r / t as f64),
            _ => Reading::unknown("needs a profile and a passing eager read size"),
        },
        note: "informational only: capacity-only mode applies no limiter and accepts no target-bandwidth claim"
            .to_string(),
    };

    let estimate = EstimateSection {
        status: Status::Unavailable,
        reason: Some(
            "capacity-only validator run: derating estimates need model-run inputs (emu_model::estimate)"
                .to_string(),
        ),
    };

    // The estimate never gates the overall status (it is unavailable by design).
    let agg_input = [
        (capacity.status, capacity.requested),
        (host_topology.status, host_topology.requested),
        (bandwidth.status, bandwidth.requested),
        (graph.status, graph.requested),
    ];
    let (status, reason) = aggregate_status(&agg_input);

    let (binary_path, binary_sha256) = binary_identity();
    let hip_runtime = match gpu.hip.runtime_version() {
        Ok((maj, min)) => Reading::known(format!("{maj}.{min}")),
        Err(e) => Reading::unknown(e.to_string()),
    };
    let mut fidelity_limits = vec![
        "capacity-only: no CU, queue or PM4 mask and no bandwidth limiter is applied; traffic runs at host speed"
            .to_string(),
        "ledger counts managed application charges only, not driver, other-process or ROCm-internal allocations"
            .to_string(),
        "HW_ID1 field layout is assumed per arch and validated only by CU-count agreement with the host"
            .to_string(),
        "CU census samples placement over a bounded number of launches and can miss CUs (reported unavailable)"
            .to_string(),
        "no derating estimate is produced; target-bandwidth figures are informational".to_string(),
    ];
    if want_bw && request.buffer_mib.iter().any(|&m| m < 256) {
        fidelity_limits.push(
            "buffer sizes below 256 MiB may fit in cache and overstate bandwidth".to_string(),
        );
    }

    Ok(EmuValidationReceipt {
        schema_version: SCHEMA_VERSION,
        kind: "emu_validate".to_string(),
        status,
        reason,
        timestamp_unix_s: timestamp,
        timestamp_utc: unix_to_utc(timestamp),
        accounting_kind: "managed_application_charge".to_string(),
        limiter_active: false,
        profile: profile.map(EffectiveProfileRecord::from_profile),
        memory,
        physical,
        capacity,
        host_topology,
        bandwidth,
        graph,
        estimate,
        raw_rates,
        scenario_goals,
        identity: IdentityRecord {
            identity_note: request.identity_note.clone(),
            binary_path,
            binary_sha256,
            binary_md5: Reading::unknown("md5 not computed in rdna-compute; see binary_sha256"),
            model_md5: Reading::unknown("model-free validator run"),
            sidecar_md5: Reading::unknown("model-free validator run"),
            prompt_md5: Reading::unknown("no text prompt"),
            input_digest: input_digest(request),
            emu_id: profile.map(|p| p.id.to_string()),
        },
        telemetry: TelemetryRecord {
            source: Reading::known(
                "world-readable amdgpu hwmon sysfs, sampled before and after the run".to_string(),
            ),
            before: telemetry_before,
            after: telemetry_after,
        },
        run: RunRecord {
            pid: std::process::id(),
            fresh_process: Reading::unknown(
                "not determinable in-process; the spawning CLI records it",
            ),
            timed_reps_per_kernel: request.passes,
            order,
            fabric_job_marker: match std::env::var("HIPFIRE_FABRIC_ADMISSION") {
                Ok(v) if !v.is_empty() => Reading::known(v),
                _ => Reading::unknown("HIPFIRE_FABRIC_ADMISSION not set"),
            },
            hip_runtime,
            driver_version: amdgpu_driver_version(),
        },
        fidelity_limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> HwIdLayout {
        hw_id1_layout("gfx1100").expect("gfx1100 layout")
    }

    fn req() -> EmuValidationRequest {
        serde_json::from_str(r#"{"profile_id":"rx9070xt"}"#).expect("defaults")
    }

    #[test]
    fn hw_id_decode_extracts_fields() {
        // se=5, sa=1, wgp=9, simd=3, wave_id=17 (ignored).
        let v = (5u32 << 18) | (1 << 16) | (9 << 10) | (3 << 8) | 17;
        let f = decode_hw_id1(v, &layout());
        assert_eq!((f.se, f.sa, f.wgp, f.simd), (5, 1, 9, 3));
        assert_eq!(f.cu_key(), (5, 1, 9, 1));
        assert_eq!(cu_label(f.cu_key()), "se5.sa1.wgp9.cu1");
        // SIMD 0/1 share a CU, 2/3 the other.
        let a = decode_hw_id1(0, &layout());
        let b = decode_hw_id1(1 << 8, &layout());
        let c = decode_hw_id1(2 << 8, &layout());
        assert_eq!(a.cu_key(), b.cu_key());
        assert_ne!(a.cu_key(), c.cu_key());
        // Bits outside the fields do not leak in.
        let g = decode_hw_id1(0xFFFF_FFFF, &layout());
        assert_eq!((g.se, g.sa, g.wgp, g.simd), (7, 1, 15, 3));
    }

    #[test]
    fn layout_only_for_supported_arches_and_marked_assumed() {
        for arch in SUPPORTED_ARCHES {
            let l = hw_id1_layout(arch).expect(arch);
            assert!(l.assumed);
            assert!(l.note.contains("[INFERENCE]"));
        }
        assert!(hw_id1_layout("gfx942").is_none());
        assert!(hw_id1_layout("gfx1030").is_none());
    }

    #[test]
    fn pattern_and_checksum_oracle_are_deterministic() {
        assert_eq!(pattern_word(0, 1), pattern_word(0, 1));
        assert_ne!(pattern_word(0, 1), pattern_word(1, 1));
        assert_ne!(pattern_word(0, 1), pattern_word(0, 2));
        // Oracle equals a direct scalar fold over the vectors a thread owns.
        let (n_vec, threads, seed) = (37u64, 8u64, 7u32);
        let sink = expected_checksums(n_vec, threads, seed);
        assert_eq!(sink.len(), 8);
        for t in 0..threads {
            let mut acc = FNV_OFFSET ^ (t as u32);
            let mut v = t;
            while v < n_vec {
                for k in 0..4 {
                    acc = (acc ^ pattern_word(v * 4 + k, seed)).wrapping_mul(FNV_PRIME);
                }
                v += threads;
            }
            assert_eq!(sink[t as usize], acc);
        }
        // Threads beyond n_vec keep the seeded initial value.
        let tiny = expected_checksums(2, 8, 0);
        assert_eq!(tiny[7], FNV_OFFSET ^ 7);
        // A single flipped word changes the digest.
        let mut bad = sink.clone();
        bad[3] ^= 1;
        assert_ne!(digest_words(&sink), digest_words(&bad));
        assert_eq!(compare_sinks(&bad, &sink), (1, Some(3)));
        assert_eq!(compare_sinks(&sink, &sink), (0, None));
    }

    #[test]
    fn fill_and_mismatch_counting_round_trip() {
        let mut buf = vec![0u8; 4096 * 5 + 64];
        fill_pattern(&mut buf, 1000, 9);
        assert_eq!(count_pattern_mismatches(&buf, 1000, 9), (0, None));
        // Wrong offset or seed mismatches nearly everywhere.
        assert!(count_pattern_mismatches(&buf, 1001, 9).0 > 1000);
        buf[4096 * 3 + 8] ^= 0x80;
        buf[4096 * 4 + 12] ^= 0x01;
        let (n, first) = count_pattern_mismatches(&buf, 1000, 9);
        assert_eq!(n, 2);
        assert_eq!(first, Some(1000 + (4096 * 3 + 8) / 4));
    }

    #[test]
    fn hwid_chain_matches_iterative_definition() {
        assert_eq!(hwid_chain(5, 0), 5);
        assert_eq!(
            hwid_chain(5, 1),
            5u32.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
        );
        assert_ne!(hwid_chain(1, 1000), hwid_chain(2, 1000));
    }

    #[test]
    fn topology_accumulator_unions_and_detects_stability() {
        let l = layout();
        let iters = 10;
        let raw = |se: u32, wgp: u32, simd: u32| (se << 18) | (wgp << 10) | (simd << 8);
        let launch1 = [raw(0, 0, 0), raw(0, 0, 1), raw(0, 1, 2), HWID_SENTINEL];
        let aux = |n: usize| {
            (0..n as u32)
                .map(|s| hwid_chain(s, iters))
                .collect::<Vec<_>>()
        };
        let mut acc = TopologyAccumulator::default();
        acc.absorb(&launch1, &aux(4), iters, &l);
        // (se0,wgp0,simd0|1) is one CU; (se0,wgp1,simd2) another.
        assert_eq!(acc.cu_count(), 2);
        assert_eq!(acc.wgp_count(), 2);
        assert_eq!(acc.unwritten_slots, 1);
        assert_eq!(acc.chain_mismatch_slots, 0);
        assert!(!acc.done(3, 2));
        let launch2 = [raw(1, 0, 0), raw(0, 0, 0), raw(0, 1, 2), raw(0, 1, 1)];
        acc.absorb(&launch2, &aux(4), iters, &l);
        assert_eq!(acc.cu_count(), 4);
        assert_eq!(acc.history, vec![2, 4]);
        assert!(acc.done(4, 8), "target reached");
        // Stability: no growth over the window.
        for _ in 0..3 {
            acc.absorb(&launch2, &aux(4), iters, &l);
        }
        assert!(acc.done(99, 3));
        assert!(!acc.done(99, 5));
        // A wrong chain value is counted.
        let mut bad_aux = aux(4);
        bad_aux[0] ^= 1;
        let mut acc2 = TopologyAccumulator::default();
        acc2.absorb(&launch2, &bad_aux, iters, &l);
        assert_eq!(acc2.chain_mismatch_slots, 1);
    }

    #[test]
    fn topology_verdicts_never_use_target_cus() {
        assert_eq!(evaluate_topology(64, 64, 0, 0, true), (Status::Pass, None));
        assert_eq!(evaluate_topology(64, 60, 0, 0, true).0, Status::Unavailable);
        assert_eq!(evaluate_topology(64, 70, 0, 0, true).0, Status::Unavailable);
        assert_eq!(evaluate_topology(64, 64, 3, 0, true).0, Status::Unavailable);
        assert_eq!(evaluate_topology(64, 64, 0, 1, true).0, Status::Fail);
        assert_eq!(
            evaluate_topology(64, 64, 0, 0, false).0,
            Status::Unavailable
        );
    }

    #[test]
    fn cu_source_reconciliation() {
        assert_eq!(
            reconcile_cu_sources(Some(64), Some(64), Some(64)),
            Ok((64, "host_facts"))
        );
        assert_eq!(
            reconcile_cu_sources(Some(64), Some(64), None),
            Ok((64, "kfd"))
        );
        assert_eq!(
            reconcile_cu_sources(Some(64), None, None),
            Ok((64, "hip_mp_count"))
        );
        assert!(reconcile_cu_sources(Some(32), Some(64), None)
            .unwrap_err()
            .contains("disagree"));
        assert!(reconcile_cu_sources(None, None, None).is_err());
        assert!(reconcile_cu_sources(Some(0), None, None).is_err());
    }

    #[test]
    fn kfd_property_parsing() {
        assert_eq!(
            cus_from_kfd_properties("simd_count 192\nsimd_per_cu 2\n"),
            Some(96)
        );
        assert_eq!(
            cus_from_kfd_properties("cpu_cores_count 4\nsimd_count 128\n"),
            Some(64)
        );
        assert_eq!(cus_from_kfd_properties("simd_count 0\n"), None);
        assert_eq!(cus_from_kfd_properties("nothing\n"), None);
    }

    #[test]
    fn status_aggregation() {
        use Status::*;
        assert_eq!(aggregate_status(&[(Pass, true), (Pass, true)]).0, Pass);
        assert_eq!(
            aggregate_status(&[(Pass, true), (Unavailable, true)]).0,
            Unavailable
        );
        assert_eq!(
            aggregate_status(&[(Fail, true), (Unavailable, true), (Pass, true)]).0,
            Fail
        );
        // Unrequested sections never count.
        assert_eq!(aggregate_status(&[(Pass, true), (Fail, false)]).0, Pass);
        assert_eq!(aggregate_status(&[(Unavailable, false)]).0, Unavailable);
        assert_eq!(aggregate_status(&[]).0, Unavailable);
        assert_eq!(aggregate_sizes(&[Pass, Unavailable]), Pass);
        assert_eq!(aggregate_sizes(&[Pass, Fail]), Fail);
        assert_eq!(aggregate_sizes(&[Unavailable]), Unavailable);
        assert_eq!(aggregate_sizes(&[]), Unavailable);
        assert_eq!(aggregate_checks(&[]), Unavailable);
        assert_eq!(
            aggregate_checks(&[check("a", true, ""), check("b", false, "")]),
            Fail
        );
        assert_eq!(aggregate_checks(&[check("a", true, "")]), Pass);
    }

    #[test]
    fn request_defaults_and_unknown_fields() {
        let r: EmuValidationRequest =
            serde_json::from_str(r#"{"type":"emu_validate","profile_id":"rx7900xt"}"#).unwrap();
        assert_eq!(r.buffer_mib, vec![256]);
        assert_eq!(r.passes, 5);
        assert!(r.approved_tests.is_empty());
        assert!(r.identity_note.is_none());
        assert!(r.validate().is_ok());
        assert!(r.requested("graph") && r.requested("capacity"));
        assert!(serde_json::from_str::<EmuValidationRequest>(r#"{"buffer_mib":[1]}"#).is_err());
    }

    #[test]
    fn request_bounds() {
        let ok = req();
        let mut r = ok.clone();
        r.profile_id = "  ".into();
        assert!(r.validate().is_err());
        for bad in [0u32, MAX_BUFFER_MIB + 1] {
            let mut r = ok.clone();
            r.buffer_mib = vec![bad];
            assert!(r.validate().is_err(), "{bad}");
        }
        let mut r = ok.clone();
        r.buffer_mib = vec![1, MAX_BUFFER_MIB];
        assert!(r.validate().is_ok());
        r.buffer_mib = (1..=(MAX_BUFFER_COUNT as u32 + 1)).collect();
        assert!(r.validate().is_err());
        r.buffer_mib = vec![256, 256];
        assert!(r.validate().is_err());
        let mut r = ok.clone();
        r.passes = 0;
        assert!(r.validate().is_err());
        r.passes = MAX_PASSES + 1;
        assert!(r.validate().is_err());
        let mut r = ok.clone();
        r.approved_tests = vec!["topology".into(), "bogus".into()];
        assert!(r.validate().is_err());
        r.approved_tests = vec!["topology".into()];
        assert!(r.validate().is_ok());
        assert!(r.requested("topology") && !r.requested("graph"));
        let mut r = ok.clone();
        r.identity_note = Some("x".repeat(MAX_IDENTITY_NOTE_CHARS + 1));
        assert!(r.validate().is_err());
        let mut r = ok;
        r.buffer_mib.clear();
        assert!(r.validate().is_err(), "bandwidth needs a size");
        r.approved_tests = vec!["capacity".into()];
        assert!(r.validate().is_ok());
    }

    #[test]
    fn traffic_fit_and_ledger_math() {
        assert!(traffic_fits(256 << 20, 2 << 20, 1 << 30));
        assert!(!traffic_fits(2048 << 20, 2 << 20, 4 << 30));
        assert!(traffic_fits(2048 << 20, 2 << 20, 5 << 30));
        let a = BudgetUsage {
            device_bytes: 10,
            host_bytes: 5,
            gtt_bytes: 0,
            shared_host_bytes: 3,
        };
        let b = BudgetUsage {
            device_bytes: 4,
            host_bytes: 5,
            gtt_bytes: 1,
            shared_host_bytes: 0,
        };
        assert_eq!(
            usage_delta(&a, &b),
            BudgetUsage {
                device_bytes: 6,
                host_bytes: 0,
                gtt_bytes: 0,
                shared_host_bytes: 3
            }
        );
        assert_eq!(device_remaining(100, &a, &b), 86);
        assert_eq!(device_remaining(5, &a, &b), 0);
    }

    #[test]
    fn distribution_statistics() {
        let d = Distribution::from_samples(vec![3.0, 1.0, 2.0, 4.0]).unwrap();
        assert_eq!((d.n, d.min, d.max), (4, 1.0, 4.0));
        assert_eq!(d.median, 2.5);
        assert_eq!(d.mean, 2.5);
        assert!((d.stddev - 1.118033988749895).abs() < 1e-12);
        assert_eq!(d.samples, vec![3.0, 1.0, 2.0, 4.0], "raw order kept");
        assert_eq!(Distribution::from_samples(vec![5.0]).unwrap().median, 5.0);
        assert!(Distribution::from_samples(vec![]).is_none());
        assert!(Distribution::from_samples(vec![f64::NAN]).is_none());
        // Read counts S, copy counts read + write.
        assert!((rate_gb_s(1_000_000_000, 1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn utc_formatting() {
        assert_eq!(unix_to_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(unix_to_utc(1_700_000_000), "2023-11-14T22:13:20Z");
        assert_eq!(unix_to_utc(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn input_digest_is_stable_and_size_sensitive() {
        let a = req();
        assert_eq!(input_digest(&a), input_digest(&a.clone()));
        let mut b = a.clone();
        b.buffer_mib = vec![512];
        assert_ne!(input_digest(&a), input_digest(&b));
    }

    fn blank_receipt() -> EmuValidationReceipt {
        let unknown = |why: &str| Reading::unknown(why.to_string());
        let tel = || Telemetry {
            sclk_mhz: Reading::unknown("x"),
            mclk_mhz: Reading::unknown("x"),
            power_w: Reading::known(212.5),
            temperature_c: Reading::known(61.0),
        };
        EmuValidationReceipt {
            schema_version: SCHEMA_VERSION,
            kind: "emu_validate".into(),
            status: Status::Unavailable,
            reason: Some("a requested section is unavailable".into()),
            timestamp_unix_s: 1_700_000_000,
            timestamp_utc: unix_to_utc(1_700_000_000),
            accounting_kind: "managed_application_charge".into(),
            limiter_active: false,
            profile: Some(EffectiveProfileRecord {
                id: "rx9070xt".into(),
                host_gfx: "gfx1201".into(),
                target_gfx: "gfx1201".into(),
                target_cus: 64,
                memory: "discrete".into(),
                advertised_capacity_bytes: Some(16 << 30),
                peak_bw_bytes_s: 640_000_000_000,
                infinity_cache_bytes: 64 << 20,
                l2_bytes: 8 << 20,
                fidelity: "capacity-only".into(),
            }),
            memory: MemoryRecord {
                advertised_capacity_bytes: Some(16 << 30),
                raw_real_free_bytes: 30 << 30,
                raw_real_total_bytes: 32 << 30,
                budgeted_free_bytes: 16 << 30,
                budgeted_total_bytes: 16 << 30,
                limits: Some(LimitsRecord {
                    device_bytes: 16 << 30,
                    gtt_bytes: None,
                    shared_host_bytes: None,
                    uma: false,
                }),
                committed: Some(UsageRecord::default()),
                pending: Some(UsageRecord::default()),
                pool_parked_bytes: 0,
                accounting_kind: "managed_application_charge".into(),
            },
            physical: PhysicalRecord {
                pci_bdf: "0000:03:00.0".into(),
                physical_gfx: "gfx1201".into(),
                target_gfx: Some("gfx1201".into()),
                target_cus: Some(64),
                hip_mp_count: 32,
                hip_physical_cus: 64,
                kfd_physical_cus: Reading::known(64),
                host_facts_physical_cus: Some(64),
                host_physical_cus: Some(64),
                cu_count_source: Some("host_facts".into()),
                cu_count_disagreement: None,
                cu_emulation: "none".into(),
            },
            capacity: CapacitySection::unavailable(true, "no profile"),
            host_topology: TopologySection::unavailable(true, "x"),
            bandwidth: BandwidthSection {
                status: Status::Unavailable,
                requested: true,
                reason: None,
                warmup_reps: WARMUP_REPS,
                timed_reps: 5,
                sizes: vec![BandwidthSize {
                    buffer_mib: 256,
                    bytes: 256 << 20,
                    status: Status::Pass,
                    reason: None,
                    read: Some(TrafficRecord {
                        kernel: KERNEL_READ.into(),
                        grid_blocks: 2048,
                        block_threads: 256,
                        bytes_per_pass: 256 << 20,
                        bytes_basis: "logical bytes read".into(),
                        // Binary-exact samples: serde_json without `float_roundtrip` can parse a derived f64 one ulp off.
                        time_ms: Distribution::from_samples(vec![1.0, 1.5]).unwrap(),
                        rate_gb_s: Distribution::from_samples(vec![268.0, 244.0]).unwrap(),
                        checks: sink_record(&[1, 2], &[1, 2], 2),
                    }),
                    copy: None,
                }],
            },
            graph: GraphSection {
                status: Status::Unavailable,
                requested: false,
                reason: Some("not requested".into()),
                sizes: vec![],
            },
            estimate: EstimateSection {
                status: Status::Unavailable,
                reason: Some("capacity-only".into()),
            },
            raw_rates: RawRates {
                read_bytes_s: Reading::known(2.4e11),
                copy_rw_bytes_s: Reading::unknown("none"),
                basis: "b".into(),
            },
            scenario_goals: ScenarioGoals {
                target_peak_bw_bytes_s: Some(640_000_000_000),
                target_cus: Some(64),
                fidelity: Some("capacity-only".into()),
                measured_read_over_target_peak: Reading::known(0.375),
                note: "n".into(),
            },
            identity: IdentityRecord {
                identity_note: Some("fabric job 12".into()),
                binary_path: Reading::known("/bin/x".into()),
                binary_sha256: Reading::known("ab".into()),
                binary_md5: unknown("m"),
                model_md5: unknown("model-free"),
                sidecar_md5: unknown("model-free"),
                prompt_md5: unknown("no text prompt"),
                input_digest: "d".into(),
                emu_id: Some("rx9070xt".into()),
            },
            telemetry: TelemetryRecord {
                source: Reading::known("s".into()),
                before: tel(),
                after: tel(),
            },
            run: RunRecord {
                pid: 1,
                fresh_process: Reading::unknown("x"),
                timed_reps_per_kernel: 5,
                order: vec!["topology".into()],
                fabric_job_marker: unknown("none"),
                hip_runtime: Reading::known("6.4".into()),
                driver_version: unknown("none"),
            },
            fidelity_limits: vec!["l".into()],
        }
    }

    #[test]
    fn receipt_round_trips_and_has_frozen_shape() {
        let r = blank_receipt();
        let line = r.to_json_line().unwrap();
        assert!(!line.contains('\n'));
        let back: EmuValidationReceipt = serde_json::from_str(&line).unwrap();
        assert_eq!(back, r);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["status"], "unavailable");
        for section in [
            "capacity",
            "host_topology",
            "bandwidth",
            "graph",
            "estimate",
        ] {
            assert!(v[section]["status"].is_string(), "{section}.status");
        }
        assert_eq!(v["accounting_kind"], "managed_application_charge");
        assert_eq!(v["limiter_active"], false);
        assert_eq!(v["physical"]["cu_emulation"], "none");
        assert_eq!(v["host_topology"]["compared_against_target_cus"], false);
        assert!(v["identity"]["prompt_md5"]["value"].is_null());
        assert_eq!(v["identity"]["prompt_md5"]["reason"], "no text prompt");
        assert!(v.get("fidelity_pass").is_none());
    }

    #[test]
    fn required_failed_only_counts_requested_required_sections() {
        let mut r = blank_receipt();
        assert!(!r.required_failed());
        r.bandwidth.status = Status::Fail;
        assert!(
            !r.required_failed(),
            "bandwidth is not a required-exit section"
        );
        r.graph.status = Status::Fail;
        assert!(!r.required_failed(), "graph not requested");
        r.graph.requested = true;
        assert!(r.required_failed());
        r.graph.requested = false;
        r.capacity.status = Status::Fail;
        assert!(r.required_failed());
        r.capacity.status = Status::Unavailable;
        r.host_topology.status = Status::Fail;
        assert!(r.required_failed());
    }

    #[test]
    #[cfg(unix)]
    fn telemetry_reads_only_world_readable_files() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("emu_validate_tel_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hwmon = root.join("sys/bus/pci/devices/0000:03:00.0/hwmon/hwmon4");
        std::fs::create_dir_all(&hwmon).unwrap();
        let write = |name: &str, text: &str, mode: u32| {
            let p = hwmon.join(name);
            std::fs::write(&p, text).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        };
        write("power1_average", "98000000\n", 0o444);
        write("temp1_input", "70000\n", 0o444);
        write("temp2_label", "edge\n", 0o444);
        write("temp2_input", "61000\n", 0o444);
        write("freq1_input", "2500000000\n", 0o444);
        write("freq2_input", "1000000000\n", 0o600); // not world-readable
        let t = sample_telemetry_at(&root, "0000:03:00.0");
        assert_eq!(t.power_w.value, Some(98.0));
        assert_eq!(t.temperature_c.value, Some(61.0), "edge label preferred");
        assert_eq!(t.sclk_mhz.value, Some(2500));
        assert_eq!(t.mclk_mhz.value, None);
        assert!(t
            .mclk_mhz
            .reason
            .as_deref()
            .unwrap()
            .contains("not world-readable"));
        let none = sample_telemetry_at(&root, "0000:99:00.0");
        assert!(none.power_w.value.is_none() && none.power_w.reason.is_some());
        let _ = std::fs::remove_dir_all(&root);
    }
}
