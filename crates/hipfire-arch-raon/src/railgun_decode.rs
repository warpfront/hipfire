// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon plain-AR decode as a certified railgun program (raon-redline PLAN §3-5).
//!
//! One [`RaonRailgunOwner`] per [`RaonBundle`] (no global or pointer-keyed
//! cache), created by `RaonBundle::new` per `HIPFIRE_RAON_RAILGUN`
//! (developer variable, tri-state, default `auto`):
//!
//! * `0` is the hard kill: no owner is created (no corpus discovery, no
//!   allocation), the decode route is the unchanged HIP graph.
//! * `auto` (default, also unset/empty) creates the owner only when the
//!   structural admission checks pass (exact gfx1201 or gfx1100, whole band, graph route,
//!   no `HIPFIRE_GRAPH=0`, no retained replay, no CPU-executed weights) AND a
//!   receipt corpus is found (`HIPFIRE_RAON_RAILGUN_CORPUS`, else the
//!   `railgun-corpus` directory beside the installed kernel pack). Otherwise
//!   the bundle holds no owner. A step the owner then declines silently runs
//!   the HIP graph step (one log line per bundle for the first
//!   non-routine decline).
//! * `1` forces the owner (whatever the arch or corpus) and logs every
//!   decline once per kind.
//!
//! A step through the owner:
//!
//! 1. **Admission checks** ([`check_admission`]): exact gfx1201 or exact gfx1100
//!    ([`ARCH_POLICIES`]; gfx1100 also needs a resolvable PCI identity), whole band,
//!    graph route, no retained replay, no CPU-executed weights, no census tap,
//!    no MoE route/activation dump, decompressed MLA route, decode graph
//!    warmed, corpus configured. Any failure is a [`RaonRailgunRefusal`]
//!    (`Declined`): nothing was touched and the caller runs the HIP graph step
//!    exactly once.
//! 2. **Admit** the context (`admit_decode_context`, outside any program).
//! 3. **Key** ([`RaonRailgunKey`]): equality over everything the program
//!    bakes in (decode binding, mapped rows, mirror rows, route, scratch
//!    geometry and growth epoch, model and flag digests, every loaded object
//!    digest, every sealed table's generation and content digest). A hit reuses
//!    the program; any difference rebuilds (the previous submission always
//!    completed first: `submit` blocks until the queue is idle).
//! 4. **Build** on a miss: the decode body under the plan sink (graph route
//!    flags, nothing executes) -> launch records with exact-artifact facts from
//!    the receipt corpus -> `railgun::kernel::author` -> sealed-table seal +
//!    checked bind -> witnessed program -> hazard plan (single segment) ->
//!    `Gpu::lower_railgun_window`. Every refusal here is a `Declined`, cached
//!    for the key so a refused binding is not re-planned per token.
//! 5. **Execute**: token staged on the decode stream, stream synchronized, ONE
//!    blocking submit, host length mirror advanced exactly once on success.
//!
//! After a submit that may have executed, the HIP route is never used to retry
//! the token: a proven-quiescent failure aborts the request with `Err`
//! (a lowering self-check failure, which dispatched nothing, is the one
//! `Declined`); unknown quiescence quarantines the owner, and
//! `RaonBundle::free_all` then leaks the tensors the queue may still touch.
//!
//! Everything GPU-free (admission, key, record construction, authoring,
//! sealing/binding, the execution state machine) is plain functions and a
//! generic [`Exec`] over `railgun::window::WindowQueue`, unit-tested without a
//! device.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use railgun::kernel::{
    self as rk, Arg, KernelProgram, LaunchRecord, Lowering, NodeFacts, TableBound,
};
use railgun::plan::{self as rp, CacheTable, G4Table, Pm4Plan, RowPolicy};
use railgun::sealed_table::{
    NullEntries, Pointee, SealedTableRegistry, TableBindInputs, TableContent,
};
use railgun::window::{AllocationIdentity, NodeImage, Quiescence, WindowQueue};
use railgun::{Binding, ResourceId};
use railgun_corpus::sha256_hex;
use rdna_compute::plan_sink::{take_declared_words, PlanSink, PlannedOp};
use rdna_compute::railgun_window::{
    RailgunLoweringContext, RailgunWindowQueue, WindowCorpus, LOWERING_SELF_CHECK,
};
use rdna_compute::Gpu;

use crate::bundle::{DecodeMode, RaonBundle};
use crate::forward::{
    admit_decode_context, decode_binding, decode_body, mla_route, stage_decode_token_on_stream,
};
use crate::mla::MlaRoute;

/// `HIPFIRE_RAON_RAILGUN`: `auto` (default) | `1` | `0` (hard kill).
pub const ENV_ENABLE: &str = "HIPFIRE_RAON_RAILGUN";
/// `HIPFIRE_RAON_RAILGUN_CORPUS`: directory of the railgun JIT receipt corpus
/// (a `railgun-jit-corpus` build's `v2/<host key>` directory).
pub const ENV_CORPUS: &str = "HIPFIRE_RAON_RAILGUN_CORPUS";
/// Receipt trials a G4 row needs to back a cache-table rung.
const G4_MIN_TRIALS: u64 = 1 << 20;

// ----------------------------------------------------------------- arch policy

/// What differs per admitted architecture; everything else (admission checks,
/// key, authoring, execution) is shared.
///
/// * the cache table is `CacheTable::for_arch(arch)` and the lowering
///   parameters `rp::ArchLowering::for_arch(arch)` (applied by the device
///   lowering from the program's arch);
/// * the receipt corpus directory and the optional G4 silicon-receipt table
///   file are read from arch-qualified variables first (a corpus or probe
///   table of another arch can never be reached by accident; the table file
///   must also carry this arch's identity), then the generic ones;
/// * `pci_identity`: the HIP device is matched to its ROCr agent by PCI bus
///   address (gfx1100: the 7900 XTX hosts are multi-GPU and ordinals are not
///   a device identity), so an unresolvable address refuses at admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchPolicy {
    pub arch: &'static str,
    pub pci_identity: bool,
    /// Corpus directory variables, first set wins.
    pub corpus_vars: &'static [&'static str],
    /// G4 cache-table file variables, first set wins.
    pub cache_table_vars: &'static [&'static str],
}

/// The admitted architectures (exact names; every other part, gfx1101 and
/// gfx1151 included, is refused).
pub const ARCH_POLICIES: &[ArchPolicy] = &[
    ArchPolicy {
        arch: "gfx1201",
        pci_identity: false,
        corpus_vars: &["HIPFIRE_RAON_RAILGUN_CORPUS_GFX1201", ENV_CORPUS],
        cache_table_vars: &[
            "HIPFIRE_RAILGUN_CACHE_TABLE_GFX1201",
            "HIPFIRE_RAILGUN_CACHE_TABLE",
        ],
    },
    ArchPolicy {
        arch: "gfx1100",
        pci_identity: true,
        corpus_vars: &["HIPFIRE_RAON_RAILGUN_CORPUS_GFX1100", ENV_CORPUS],
        cache_table_vars: &[
            "HIPFIRE_RAILGUN_CACHE_TABLE_GFX1100",
            "HIPFIRE_RAILGUN_CACHE_TABLE",
        ],
    },
];

/// The policy of exactly `arch`.
pub fn arch_policy(arch: &str) -> Option<&'static ArchPolicy> {
    ARCH_POLICIES.iter().find(|p| p.arch == arch)
}

fn supported_arches() -> String {
    ARCH_POLICIES
        .iter()
        .map(|p| p.arch)
        .collect::<Vec<_>>()
        .join(", ")
}

impl ArchPolicy {
    /// The first of `vars` the process configuration sets to a non-empty
    /// value: `(variable, value)`.
    fn env_value(&self, vars: &[&'static str]) -> Option<(&'static str, String)> {
        vars.iter().find_map(|v| {
            hipfire_config::developer_var(v)
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| (*v, s))
        })
    }

    /// The lowering parameters of this arch (`gfx1201`'s post-dispatch pacing
    /// `NOP`; explicitly unpaced on gfx1100).
    pub fn lowering(&self) -> rp::ArchLowering {
        rp::ArchLowering::for_arch(self.arch)
    }
}

// ------------------------------------------------------------------ refusal

/// Why the railgun route did not run this step. Every variant is a
/// pre-execution decision: the caller runs the HIP-graph step once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RaonRailgunRefusal {
    /// The bundle has no owner (`HIPFIRE_RAON_RAILGUN` is off).
    Disabled,
    /// The arch is not one of [`ARCH_POLICIES`].
    WrongArch { arch: String },
    /// The device's PCI identity cannot be resolved (an arch whose ROCr agent
    /// is matched by PCI address, not by ordinal).
    DeviceIdentity { arch: String },
    /// A band other than the whole single-device model (pipeline/tensor split).
    PartitionedBand { band: String },
    /// `HIPFIRE_GRAPH=0`.
    GraphDisabled,
    /// The decode route is not the HIP graph (eager / retained replay).
    NotGraphMode,
    /// Retained replay (Redline) owns the launch stream.
    RetainedReplay,
    /// CPU-executed or offloaded weights.
    HostWeights,
    /// A census tap is armed on the bundle.
    TapArmed,
    /// `HIPFIRE_MOE_ROUTE_DUMP` / `HIPFIRE_MOE_ACT_CAPTURE` is set.
    DumpArmed,
    /// The context takes the absorbed MLA route (graph only, not certified).
    AbsorbedRoute,
    /// The decode graph's warm-up step has not run (modules not all loaded).
    NotWarm,
    /// `HIPFIRE_RAON_RAILGUN_CORPUS` is unset.
    CorpusUnset,
    /// The corpus could not be opened under the loaded toolchain.
    Corpus(String),
    /// The decode body could not be planned.
    Plan(String),
    /// A node's exact object has no receipt record.
    NoReceipt {
        node: usize,
        symbol: String,
        why: String,
    },
    /// A node's receipt is not tier D (a stub or strict-only record).
    NotTierD {
        node: usize,
        symbol: String,
        tier: String,
    },
    /// A node carries per-window declared words the program cannot supply.
    DeclaredWords { node: usize },
    /// A D2D copy or fill the typed transport nodes cannot express.
    UntypedCopy { node: usize, what: String },
    /// A node that is not PM4 after the witnesses.
    NonPm4 {
        node: usize,
        symbol: String,
        lowering: String,
    },
    /// Sealing a loader table or the checked sealed-table bind failed.
    SealedTable(String),
    /// The G4 cache table could not be built.
    CacheTable(String),
    /// The hazard plan is not one segment.
    Segments(usize),
    /// Lowering through the owned gfx12 transport failed.
    Lower(String),
    /// The pre-submit lowering self-check failed (nothing was dispatched).
    SelfCheck(String),
}

impl RaonRailgunRefusal {
    /// Stable short name (counters, logs).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::WrongArch { .. } => "wrong_arch",
            Self::DeviceIdentity { .. } => "device_identity",
            Self::PartitionedBand { .. } => "partitioned_band",
            Self::GraphDisabled => "graph_disabled",
            Self::NotGraphMode => "not_graph_mode",
            Self::RetainedReplay => "retained_replay",
            Self::HostWeights => "host_weights",
            Self::TapArmed => "tap_armed",
            Self::DumpArmed => "dump_armed",
            Self::AbsorbedRoute => "absorbed_route",
            Self::NotWarm => "not_warm",
            Self::CorpusUnset => "corpus_unset",
            Self::Corpus(_) => "corpus",
            Self::Plan(_) => "plan",
            Self::NoReceipt { .. } => "no_receipt",
            Self::NotTierD { .. } => "not_tier_d",
            Self::DeclaredWords { .. } => "declared_words",
            Self::UntypedCopy { .. } => "untyped_copy",
            Self::NonPm4 { .. } => "non_pm4",
            Self::SealedTable(_) => "sealed_table",
            Self::CacheTable(_) => "cache_table",
            Self::Segments(_) => "segments",
            Self::Lower(_) => "lower",
            Self::SelfCheck(_) => "self_check",
        }
    }
}

impl fmt::Display for RaonRailgunRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(
                f,
                "no railgun owner on this bundle ({ENV_ENABLE}=0, or auto found no admissible \
                 context and receipt corpus)"
            ),
            Self::WrongArch { arch } => {
                write!(f, "arch {arch} is not one of {}", supported_arches())
            }
            Self::DeviceIdentity { arch } => write!(
                f,
                "{arch}: the device's PCI address does not resolve (the ROCr agent is matched by it)"
            ),
            Self::PartitionedBand { band } => {
                write!(f, "band {band} is not the whole single-device model")
            }
            Self::GraphDisabled => write!(f, "HIPFIRE_GRAPH=0 (the HIP graph route is disabled)"),
            Self::NotGraphMode => write!(f, "the decode route is not the HIP graph"),
            Self::RetainedReplay => write!(f, "retained replay owns the launch stream"),
            Self::HostWeights => write!(f, "CPU-executed or offloaded weights"),
            Self::TapArmed => write!(f, "a census tap is armed"),
            Self::DumpArmed => {
                write!(
                    f,
                    "HIPFIRE_MOE_ROUTE_DUMP / HIPFIRE_MOE_ACT_CAPTURE is armed"
                )
            }
            Self::AbsorbedRoute => write!(f, "the absorbed MLA route is not certified"),
            Self::NotWarm => write!(f, "the decode graph warm-up step has not run"),
            Self::CorpusUnset => write!(f, "{ENV_CORPUS} is unset"),
            Self::Corpus(e) => write!(f, "receipt corpus: {e}"),
            Self::Plan(e) => write!(f, "plan sink: {e}"),
            Self::NoReceipt { node, symbol, why } => {
                write!(f, "node {node} {symbol}: no receipt record ({why})")
            }
            Self::NotTierD { node, symbol, tier } => {
                write!(f, "node {node} {symbol}: receipt tier {tier:?} is not D")
            }
            Self::DeclaredWords { node } => {
                write!(
                    f,
                    "node {node}: declared per-window words are not supported"
                )
            }
            Self::UntypedCopy { node, what } => write!(f, "op {node}: {what}"),
            Self::NonPm4 {
                node,
                symbol,
                lowering,
            } => write!(
                f,
                "node {node} {symbol} is not PM4 after witnesses: {lowering}"
            ),
            Self::SealedTable(e) => write!(f, "sealed table: {e}"),
            Self::CacheTable(e) => write!(f, "cache table: {e}"),
            Self::Segments(n) => write!(f, "hazard plan has {n} segments, not one"),
            Self::Lower(e) => write!(f, "lowering: {e}"),
            Self::SelfCheck(e) => write!(f, "lowering self-check: {e}"),
        }
    }
}

/// What a `railgun_decode_try` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RaonRailgunAttempt {
    /// The program executed the step: token consumed, host length advanced
    /// once, logits in the usual buffer.
    Completed,
    /// Nothing was consumed or mutated beyond context admission: the caller
    /// runs the HIP-graph step exactly once.
    Declined(RaonRailgunRefusal),
}

// ------------------------------------------------------------ admission facts

/// What decides, once per bundle and again at every step, whether the context
/// is one the owner can ever serve: read from the device and the bundle, plain
/// data. `retained_replay` is the caller's reading (the step reads the live
/// flag; construction relies on `graph_mode`, which the load-time route
/// resolution already made false when retained replay owned the launch
/// stream).
#[derive(Clone, Debug)]
pub struct Structural<'a> {
    pub arch: &'a str,
    /// The device's PCI address resolved, for an arch whose ROCr agent is
    /// matched by it (always `true` for an arch that needs none).
    pub device_identity: bool,
    pub whole_band: bool,
    pub band: &'a str,
    pub graph_forward_disabled: bool,
    pub graph_mode: bool,
    pub retained_replay: bool,
    pub host_weights: bool,
}

/// What can change from one step to the next.
#[derive(Clone, Debug)]
pub struct StepFacts<'a> {
    pub tap_armed: bool,
    pub dump_armed: bool,
    pub absorbed_route: bool,
    pub warmed: bool,
    pub corpus_dir: Option<&'a str>,
}

/// The first structural reason the context is refused, in a fixed order
/// (cheapest and most structural first); `Ok` when none applies.
pub fn check_structural(a: &Structural<'_>) -> Result<(), RaonRailgunRefusal> {
    use RaonRailgunRefusal as R;
    if arch_policy(a.arch).is_none() {
        return Err(R::WrongArch {
            arch: a.arch.to_owned(),
        });
    }
    if !a.device_identity {
        return Err(R::DeviceIdentity {
            arch: a.arch.to_owned(),
        });
    }
    if !a.whole_band {
        return Err(R::PartitionedBand {
            band: a.band.to_owned(),
        });
    }
    if a.graph_forward_disabled {
        return Err(R::GraphDisabled);
    }
    if a.retained_replay {
        return Err(R::RetainedReplay);
    }
    if a.host_weights {
        return Err(R::HostWeights);
    }
    if !a.graph_mode {
        return Err(R::NotGraphMode);
    }
    Ok(())
}

/// The first reason this step is refused: the structural checks, then the
/// per-step ones (census tap, MoE dump, absorbed route, cold decode graph,
/// no corpus); `Ok` when every cheap check passes.
pub fn check_admission(s: &Structural<'_>, step: &StepFacts<'_>) -> Result<(), RaonRailgunRefusal> {
    use RaonRailgunRefusal as R;
    check_structural(s)?;
    if step.tap_armed {
        return Err(R::TapArmed);
    }
    if step.dump_armed {
        return Err(R::DumpArmed);
    }
    if step.absorbed_route {
        return Err(R::AbsorbedRoute);
    }
    if !step.warmed {
        return Err(R::NotWarm);
    }
    if step.corpus_dir.is_none_or(str::is_empty) {
        return Err(R::CorpusUnset);
    }
    Ok(())
}

// ----------------------------------------------------------- tri-state mode

/// The three states of `HIPFIRE_RAON_RAILGUN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RailgunMode {
    /// `0`: hard kill, no owner.
    Off,
    /// Unset, empty or `auto` (the default).
    Auto,
    /// `1`: the owner always exists and every decline is logged.
    Force,
}

/// Parse the variable (ASCII case-insensitive, trimmed): unset / empty /
/// `auto` -> `Auto`; `1` / `true` / `on` -> `Force`; `0` / `false` / `off` ->
/// `Off`. Any other spelling is an error (the caller fails closed to `Off`).
pub fn parse_mode(raw: Option<&str>) -> Result<RailgunMode, String> {
    let Some(raw) = raw else {
        return Ok(RailgunMode::Auto);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(RailgunMode::Auto),
        "1" | "true" | "on" => Ok(RailgunMode::Force),
        "0" | "false" | "off" => Ok(RailgunMode::Off),
        _ => Err(format!(
            "{ENV_ENABLE}={raw:?} is not auto, 1 or 0; the railgun route is off"
        )),
    }
}

/// How the owner is to behave once created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerPlan {
    pub corpus_dir: Option<String>,
    /// `1`: log every decline once per kind.
    pub loud: bool,
}

/// Why a bundle got no owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoOwner {
    /// `HIPFIRE_RAON_RAILGUN=0`.
    Killed,
    /// `auto`: the context is structurally refused.
    Refused(RaonRailgunRefusal),
    /// `auto`: no usable receipt corpus; names what was looked at.
    NoCorpus(String),
}

impl fmt::Display for NoOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Killed => write!(f, "{ENV_ENABLE}=0"),
            Self::Refused(r) => write!(f, "{r}"),
            Self::NoCorpus(why) => write!(f, "{why}"),
        }
    }
}

/// Decide whether a bundle gets an owner. `env_corpus` is
/// `HIPFIRE_RAON_RAILGUN_CORPUS`, `pack_corpus` the directory beside the
/// installed kernel pack, `has_index` whether a directory holds a corpus.
///
/// `Off` creates nothing and looks at nothing. `Force` always creates the
/// owner, with the explicit corpus, else the pack's when it holds one (an
/// absent corpus then declines every step, loudly). `Auto` creates it only
/// when the structural checks pass and a corpus is found: the explicit
/// directory when set (it must hold a corpus), else the pack's.
pub fn decide_owner(
    mode: RailgunMode,
    s: &Structural<'_>,
    env_corpus: Option<&str>,
    pack_corpus: Option<&str>,
    has_index: &dyn Fn(&str) -> bool,
) -> Result<OwnerPlan, NoOwner> {
    let env_corpus = env_corpus.filter(|d| !d.is_empty());
    match mode {
        RailgunMode::Off => Err(NoOwner::Killed),
        RailgunMode::Force => Ok(OwnerPlan {
            corpus_dir: env_corpus
                .or_else(|| pack_corpus.filter(|d| has_index(d)))
                .map(str::to_owned),
            loud: true,
        }),
        RailgunMode::Auto => {
            check_structural(s).map_err(NoOwner::Refused)?;
            let dir = match (env_corpus, pack_corpus) {
                (Some(d), _) if has_index(d) => d,
                (Some(d), _) => {
                    return Err(NoOwner::NoCorpus(format!(
                        "{ENV_CORPUS}={d} holds no receipt corpus"
                    )))
                }
                (None, Some(d)) if has_index(d) => d,
                (None, Some(d)) => {
                    return Err(NoOwner::NoCorpus(format!(
                        "no receipt corpus beside the installed kernel pack ({d}); set {ENV_CORPUS}"
                    )))
                }
                (None, None) => {
                    return Err(NoOwner::NoCorpus(format!(
                        "no installed kernel pack to pair a receipt corpus with; set {ENV_CORPUS}"
                    )))
                }
            };
            Ok(OwnerPlan {
                corpus_dir: Some(dir.to_owned()),
                loud: false,
            })
        }
    }
}

// ------------------------------------------------------------------------ key

/// The cheap half of the program identity: read from the live bundle every
/// step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseKey {
    /// `decode_binding`: a fast discriminator, never trusted alone.
    pub binding: u64,
    /// Latent rows mapped (baked clamp of the MLA kernels).
    pub mapped_rows: usize,
    /// Expanded K/V mirror rows (0 without a mirror).
    pub mirror_rows: usize,
    /// MLA route of the context (`true` = absorbed).
    pub absorbed: bool,
    pub scratch_rows: usize,
    /// `rdna_compute::scratch::scratch_growth_epoch`: lazily grown `Gpu`
    /// scratch slots are replaced on growth.
    pub scratch_epoch: u64,
    /// Digest of the model identity (config, precision plan, band, weight
    /// anchors): fixed for the bundle's lifetime.
    pub model_digest: String,
    /// Digest of the device feature flags every kernel route reads.
    pub flags_digest: String,
}

/// The full program identity: equality, not a hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaonRailgunKey {
    pub base: BaseKey,
    /// `(symbol, code object sha256)` of every distinct kernel of the
    /// program, sorted: the exact loaded artifacts.
    pub object_digests: Vec<(String, String)>,
    /// `(table allocation base, generation, content sha256)` of every sealed
    /// table the program's witness rests on.
    pub table_generations: Vec<(u64, u64, String)>,
}

impl RaonRailgunKey {
    /// The first field that differs (`None` when equal), for logs and tests.
    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        let (a, b) = (&self.base, &other.base);
        if a.binding != b.binding {
            return Some("binding");
        }
        if a.mapped_rows != b.mapped_rows {
            return Some("mapped_rows");
        }
        if a.mirror_rows != b.mirror_rows {
            return Some("mirror_rows");
        }
        if a.absorbed != b.absorbed {
            return Some("route");
        }
        if a.scratch_rows != b.scratch_rows {
            return Some("scratch_rows");
        }
        if a.scratch_epoch != b.scratch_epoch {
            return Some("scratch_epoch");
        }
        if a.model_digest != b.model_digest {
            return Some("model_digest");
        }
        if a.flags_digest != b.flags_digest {
            return Some("flags_digest");
        }
        if self.object_digests != other.object_digests {
            return Some("object_digests");
        }
        if self.table_generations != other.table_generations {
            return Some("table_generations");
        }
        None
    }
}

// ------------------------------------------------------------- device access

/// What authoring needs from the device: allocation probing and reading a
/// loader-written table. [`GpuView`] over a `Gpu`; a fake in tests.
pub trait DeviceView {
    fn probe(&self, addr: u64) -> Result<Binding, String>;
    fn read(&self, allocation: Binding, len: usize) -> Result<Vec<u8>, String>;
}

/// [`DeviceView`] over the live device.
pub struct GpuView<'a>(pub &'a Gpu);

impl DeviceView for GpuView<'_> {
    fn probe(&self, addr: u64) -> Result<Binding, String> {
        self.0.railgun_probe_allocation(addr)
    }
    fn read(&self, allocation: Binding, len: usize) -> Result<Vec<u8>, String> {
        self.0.railgun_read_allocation(allocation, len)
    }
}

struct ViewContent<'a>(&'a dyn DeviceView);

impl TableContent for ViewContent<'_> {
    fn read_table(&self, allocation: Binding, entries: usize) -> Result<Vec<u8>, String> {
        self.0.read(allocation, entries * 8)
    }
}

// ------------------------------------------------------------ launch records

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The ordered launch records of a planned body: every kernel with its
/// exact-artifact facts and probed pointer allocations, every D2D copy / word
/// fill as the typed railgun transport kernel. A node whose facts are missing
/// keeps the lookup error (`LaunchRecord::facts` is `Err`); [`author_and_plan`]
/// turns that into a refusal. `declared` lists the op indices that carry
/// declared per-window words.
pub fn launch_records(
    ops: &[PlannedOp],
    declared: &[usize],
    copy_sha: &str,
    fill_sha: &str,
    facts: &mut dyn FnMut(&str, &str) -> Result<Arc<NodeFacts>, String>,
    probe: &dyn Fn(u64) -> Result<Binding, String>,
) -> Result<Vec<LaunchRecord>, RaonRailgunRefusal> {
    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if declared.contains(&i) {
            return Err(RaonRailgunRefusal::DeclaredWords { node: i });
        }
        match op {
            PlannedOp::Kernel(k) => {
                let f = facts(&hex(&k.code_object_digest), &k.symbol);
                let mut allocations = Vec::new();
                if let Ok(f) = &f {
                    for off in f.facts.pointer_offsets() {
                        let at = off as usize;
                        if let Some(b) = k.kernarg.get(at..at + 8) {
                            let v = u64::from_le_bytes(b.try_into().expect("8 bytes"));
                            if v != 0 {
                                allocations.push((off, probe(v)));
                            }
                        }
                    }
                }
                out.push(LaunchRecord {
                    symbol: k.symbol.clone(),
                    grid: k.grid,
                    block: k.block,
                    dynamic_lds: k.dynamic_lds,
                    kernarg: k.kernarg.clone(),
                    declared: Vec::new(),
                    facts: f,
                    allocations,
                });
            }
            PlannedOp::Copy { dst, src, bytes } => {
                let n = u32::try_from(*bytes)
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or_else(|| RaonRailgunRefusal::UntypedCopy {
                        node: i,
                        what: format!("copy of {bytes} bytes is not a typed 1..=u32::MAX copy"),
                    })?;
                let l = railgun::copy::copy_launch(*src, *dst, n);
                out.push(LaunchRecord {
                    symbol: l.symbol.to_string(),
                    grid: l.grid,
                    block: l.block,
                    dynamic_lds: 0,
                    kernarg: l.kernarg,
                    declared: Vec::new(),
                    facts: facts(copy_sha, railgun::copy::COPY_SYMBOL),
                    allocations: vec![(0, probe(*src)), (8, probe(*dst))],
                });
            }
            PlannedOp::Fill { dst, value, words } => {
                if *words < 1 || u64::from(*words) > railgun::copy::MAX_FILL_WORDS || dst % 4 != 0 {
                    return Err(RaonRailgunRefusal::UntypedCopy {
                        node: i,
                        what: format!(
                            "fill of {words} words at {dst:#x} is not a typed aligned word fill"
                        ),
                    });
                }
                let l = railgun::copy::fill_launch(*dst, *value, *words);
                out.push(LaunchRecord {
                    symbol: l.symbol.to_string(),
                    grid: l.grid,
                    block: l.block,
                    dynamic_lds: 0,
                    kernarg: l.kernarg,
                    declared: Vec::new(),
                    facts: facts(fill_sha, railgun::copy::FILL_SYMBOL),
                    allocations: vec![(0, probe(*dst))],
                });
            }
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------ authoring

/// An authored, witnessed, planned program (nothing lowered yet).
pub struct Authored {
    pub program: KernelProgram,
    pub plan: Pm4Plan,
    pub images: Vec<Vec<u8>>,
    pub grids: Vec<[u32; 3]>,
    /// The registry holding the sealed tables the witness rests on.
    pub registry: SealedTableRegistry,
    /// `(node, case digest)` the sealed-table bind discharged.
    pub discharged: Vec<(usize, String)>,
    /// Bases of the sealed table allocations (non-vacuous tables only).
    pub table_bases: Vec<u64>,
}

fn describe(l: &Lowering) -> String {
    match l {
        Lowering::Pm4 => "pm4".into(),
        Lowering::Graph { reason } => format!("graph: {reason}"),
        Lowering::HipDirect { reason } => format!("hip_direct: {reason}"),
        Lowering::Conditional { reason } => format!("conditional: {reason}"),
    }
}

fn le32(image: &[u8], at: u32) -> Option<u32> {
    let at = at as usize;
    Some(u32::from_le_bytes(image.get(at..at + 4)?.try_into().ok()?))
}

/// Facts check, author, seal + bind, witness, hazard plan, images.
///
/// Refuses (never guesses) when a node lacks a tier-D receipt, a node is not
/// PM4 after the witnesses (`HipDirect`/`Graph`, an indexed conditional this
/// owner holds no witness for), a sealed table cannot be sealed/bound, or the
/// plan is not one segment. `assume_sealed_table_witness` is applied only
/// after a bind that discharged every sealed node.
pub fn author_and_plan(
    records: &[LaunchRecord],
    table: &CacheTable,
    dev: &dyn DeviceView,
) -> Result<Authored, RaonRailgunRefusal> {
    use RaonRailgunRefusal as R;
    for (i, l) in records.iter().enumerate() {
        match &l.facts {
            Err(why) => {
                return Err(R::NoReceipt {
                    node: i,
                    symbol: l.symbol.clone(),
                    why: why.clone(),
                })
            }
            Ok(f) if f.kref.tier != "D" => {
                return Err(R::NotTierD {
                    node: i,
                    symbol: l.symbol.clone(),
                    tier: f.kref.tier.clone(),
                })
            }
            Ok(_) => {}
        }
    }
    let mut program = rk::author(&table.arch, records);
    let sealed: BTreeSet<usize> = program.sealed_table_nodes().into_iter().collect();
    for (i, n) in program.nodes.iter().enumerate() {
        let ok = n.lowering.is_pm4() || (sealed.contains(&i) && n.indexed.is_none());
        if !ok {
            return Err(R::NonPm4 {
                node: i,
                symbol: n.symbol.clone(),
                lowering: describe(&n.lowering),
            });
        }
    }
    let words = program.authored_words();
    let mut images = Vec::with_capacity(program.nodes.len());
    let mut grids = Vec::with_capacity(program.nodes.len());
    for (i, n) in program.nodes.iter().enumerate() {
        let seg = n
            .segment_bytes
            .ok_or_else(|| R::Lower(format!("node {i} {}: no kernarg segment", n.symbol)))?
            as usize;
        images.push(
            program
                .kernarg_image(i, seg, &words)
                .map_err(|e| R::Lower(format!("node {i} {}: kernarg image: {e}", n.symbol)))?,
        );
        grids.push(n.grid);
    }
    let mut registry = SealedTableRegistry::new();
    let mut discharged = Vec::new();
    let mut table_bases = Vec::new();
    if !sealed.is_empty() {
        let (reg, witness, bases) =
            seal_and_bind(&program, &images, &grids, dev).map_err(R::SealedTable)?;
        if witness.nodes().len() != sealed.len() {
            return Err(R::SealedTable(format!(
                "the bind discharged {} of {} sealed nodes",
                witness.nodes().len(),
                sealed.len()
            )));
        }
        discharged = witness
            .discharged_cases()
            .iter()
            .map(|(n, c, _)| (*n, (*c).to_owned()))
            .collect();
        registry = reg;
        table_bases = bases;
        program = program.assume_sealed_table_witness();
    }
    for (i, n) in program.nodes.iter().enumerate() {
        if !n.lowering.is_pm4() {
            return Err(R::NonPm4 {
                node: i,
                symbol: n.symbol.clone(),
                lowering: describe(&n.lowering),
            });
        }
    }
    let plan = rp::plan(&program, table, RowPolicy::UnreceiptedSystem);
    if !plan.is_single_segment(program.nodes.len()) {
        return Err(R::Segments(plan.segments.len()));
    }
    Ok(Authored {
        program,
        plan,
        images,
        grids,
        registry,
        discharged,
        table_bases,
    })
}

/// Seal every pointer table a certified `sealed_table` node reads and run the
/// checked bind over the whole program.
///
/// * The table allocation and its entry count come from the program (the
///   largest bound any launch carries); a zero bound is vacuous and seals
///   nothing. A table argument pointing into its allocation (`interior != 0`)
///   with a nonzero bound is refused.
/// * Content is read from the live device; every entry is resolved to its
///   allocation and registered as an **immutable pointee**. The owner asserts
///   immutability: the entries point into loaded weights, which the bundle
///   never writes for its lifetime, and the bind additionally proves no node
///   of the program writes any of them.
/// * Null entries are sealed (`NullEntries::Allow`) only when every use of the
///   table has a certificate that proves a null guard.
fn seal_and_bind(
    program: &KernelProgram,
    images: &[Vec<u8>],
    grids: &[[u32; 3]],
    dev: &dyn DeviceView,
) -> Result<
    (
        SealedTableRegistry,
        railgun::sealed_table::SealedTableWitness,
        Vec<u64>,
    ),
    String,
> {
    // resource -> (entries, every use null guarded)
    let mut want: BTreeMap<u32, (u32, bool)> = BTreeMap::new();
    for ni in program.sealed_table_nodes() {
        let node = &program.nodes[ni];
        let case = node
            .table
            .as_ref()
            .ok_or("sealed node without table facts")?;
        for tf in &case.tables {
            let bound = match tf.bound {
                TableBound::Const(c) => c,
                TableBound::Kernarg(at) => le32(&images[ni], at)
                    .ok_or_else(|| format!("node {ni}: bound outside the kernarg image"))?,
            };
            if bound == 0 {
                continue; // `i < 0` never holds: vacuous, nothing to seal
            }
            let (res, interior) = node
                .args
                .iter()
                .find_map(|(o, a)| match a {
                    Arg::Resource { resource, interior } if *o == tf.table_arg_offset => {
                        Some((*resource, *interior))
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    format!(
                        "node {ni}: table argument +{} is not a resource",
                        tf.table_arg_offset
                    )
                })?;
            if interior != 0 {
                return Err(format!(
                    "node {ni}: table argument +{} points {interior} bytes into its allocation",
                    tf.table_arg_offset
                ));
            }
            let e = want.entry(res.0).or_insert((0, true));
            e.0 = e.0.max(bound);
            e.1 &= tf.null_guarded;
        }
    }
    let registry = SealedTableRegistry::new();
    let mut bases = Vec::new();
    for (res, (entries, null_ok)) in want {
        let alloc = *program
            .resources
            .get(res as usize)
            .ok_or_else(|| format!("unknown table resource {res}"))?;
        let bytes = dev.read(alloc, entries as usize * 8)?;
        let mut pointees: BTreeMap<u64, Binding> = BTreeMap::new();
        let mut nulls = 0usize;
        for c in bytes.chunks_exact(8) {
            let v = u64::from_le_bytes(c.try_into().expect("8 bytes"));
            if v == 0 {
                nulls += 1;
                continue;
            }
            let b = dev.probe(v)?;
            pointees.insert(b.base, b);
        }
        if nulls > 0 && !null_ok {
            return Err(format!(
                "table {:#x} holds {nulls} null entries but a node using it has no proved null guard",
                alloc.base
            ));
        }
        let list: Vec<Pointee> = pointees.values().map(|b| Pointee::immutable(*b)).collect();
        let policy = if nulls > 0 {
            NullEntries::Allow
        } else {
            NullEntries::Refuse
        };
        registry
            .seal_with(alloc, entries, &bytes, &list, 1, policy)
            .map_err(|e| format!("seal table {:#x}: {e}", alloc.base))?;
        bases.push(alloc.base);
    }
    let allocations: Vec<AllocationIdentity> = program
        .resources
        .iter()
        .enumerate()
        .map(|(i, b)| AllocationIdentity {
            resource: ResourceId(i as u32),
            base: b.base,
            bytes: b.bytes,
            generation: 1,
        })
        .collect();
    let witness = registry
        .bind_sealed_table(
            &TableBindInputs {
                window: 1,
                shape_generation: 1,
                program,
                steps: &[],
                bindings: &program.resources,
                allocations: &allocations,
                images,
                grids,
            },
            &ViewContent(dev),
        )
        .map_err(|e| format!("bind: {e}"))?;
    Ok((registry, witness, bases))
}

// ------------------------------------------------------------------ execution

/// A lowered program with the identity it was built for.
pub struct Built<Q: WindowQueue> {
    pub key: RaonRailgunKey,
    pub queue: Q,
    pub images: Vec<Vec<u8>>,
    pub grids: Vec<[u32; 3]>,
    /// Distinct kernel symbols of the program (object digests are checked per
    /// symbol every step).
    pub symbols: Vec<String>,
    pub registry: SealedTableRegistry,
    pub table_bases: Vec<u64>,
    pub nodes: usize,
    pub ib_dwords: u32,
}

/// What the slot holds.
pub enum Slot<Q: WindowQueue> {
    Empty,
    Ready(Built<Q>),
    /// This binding was refused: reuse the refusal instead of re-planning.
    Declined {
        key: BaseKey,
        refusal: RaonRailgunRefusal,
    },
}

/// How one submit ended.
#[derive(Debug, PartialEq, Eq)]
pub enum StepOutcome {
    /// Executed: the caller advances the host length once.
    Completed,
    /// Nothing was dispatched (lowering self-check): the HIP route may run.
    Declined(RaonRailgunRefusal),
    /// The queue failed with proven quiescence: the request aborts (the token
    /// may or may not have been consumed; never retried through HIP) and the
    /// program is rebuilt.
    Failed(String),
    /// Quiescence is unknown: the program, its tables and the bundle's tensors
    /// are quarantined for the life of the process.
    Quarantined(String),
}

/// A lookup against the slot.
#[derive(Debug, PartialEq, Eq)]
pub enum Lookup {
    /// A built program with exactly this key.
    Hit,
    /// This binding was refused.
    Declined(RaonRailgunRefusal),
    /// Rebuild; `Some(field)` names what changed from the old program.
    Miss(Option<&'static str>),
}

/// The execution state machine, generic over the queue (unit-tested with a
/// mock). Holds at most one program.
pub struct Exec<Q: WindowQueue> {
    pub slot: Slot<Q>,
    quarantined: Option<ManuallyDrop<Built<Q>>>,
    quarantine_reason: Option<String>,
}

impl<Q: WindowQueue> Default for Exec<Q> {
    fn default() -> Self {
        Self {
            slot: Slot::Empty,
            quarantined: None,
            quarantine_reason: None,
        }
    }
}

impl<Q: WindowQueue> Exec<Q> {
    pub fn quarantine_reason(&self) -> Option<&str> {
        self.quarantine_reason.as_deref()
    }

    pub fn is_quarantined(&self) -> bool {
        self.quarantine_reason.is_some()
    }

    /// Compare the slot with the live key. `extra` yields the object digests
    /// and table generations of the built program (the half that needs the
    /// device); it is called only when a program is held.
    pub fn lookup(
        &self,
        base: &BaseKey,
        extra: &mut dyn FnMut(&Built<Q>) -> (Vec<(String, String)>, Vec<(u64, u64, String)>),
    ) -> Lookup {
        match &self.slot {
            Slot::Empty => Lookup::Miss(None),
            Slot::Declined { key, refusal } => {
                if key == base {
                    Lookup::Declined(refusal.clone())
                } else {
                    Lookup::Miss(None)
                }
            }
            Slot::Ready(b) => {
                let (object_digests, table_generations) = extra(b);
                let live = RaonRailgunKey {
                    base: base.clone(),
                    object_digests,
                    table_generations,
                };
                match b.key.first_difference(&live) {
                    None => Lookup::Hit,
                    Some(field) => Lookup::Miss(Some(field)),
                }
            }
        }
    }

    /// Drop the held program (it is quiescent: `submit` blocks) and remember
    /// that `base` was refused.
    pub fn decline(&mut self, base: BaseKey, refusal: RaonRailgunRefusal) {
        self.slot = Slot::Declined { key: base, refusal };
    }

    pub fn install(&mut self, built: Built<Q>) {
        self.slot = Slot::Ready(built);
    }

    pub fn clear(&mut self) {
        self.slot = Slot::Empty;
    }

    /// One blocking submit of the held program.
    pub fn submit(&mut self) -> StepOutcome {
        if let Some(why) = &self.quarantine_reason {
            return StepOutcome::Failed(format!("the railgun owner is quarantined: {why}"));
        }
        let result = {
            let Slot::Ready(b) = &mut self.slot else {
                return StepOutcome::Failed("no program to submit".into());
            };
            let nodes: Vec<NodeImage<'_>> = b
                .images
                .iter()
                .zip(&b.grids)
                .map(|(k, g)| NodeImage {
                    kernarg: k,
                    grid: *g,
                })
                .collect();
            // `Ok` from `submit` means in flight; `wait` proves quiescence (the
            // owned queue blocks inside `submit`, so this only reads its status).
            let r = b.queue.submit(&nodes, &[]);
            r.and_then(|()| b.queue.wait())
        };
        let Err(f) = result else {
            return StepOutcome::Completed;
        };
        match f.quiescence {
            Quiescence::Unknown => {
                let Slot::Ready(built) = std::mem::replace(&mut self.slot, Slot::Empty) else {
                    unreachable!("a submit only runs on a Ready slot")
                };
                self.quarantined = Some(ManuallyDrop::new(built));
                self.quarantine_reason = Some(f.message.clone());
                StepOutcome::Quarantined(f.message)
            }
            Quiescence::Proven if f.message.starts_with(LOWERING_SELF_CHECK) => {
                let Slot::Ready(built) = std::mem::replace(&mut self.slot, Slot::Empty) else {
                    unreachable!("a submit only runs on a Ready slot")
                };
                let refusal = RaonRailgunRefusal::SelfCheck(f.message);
                self.slot = Slot::Declined {
                    key: built.key.base.clone(),
                    refusal: refusal.clone(),
                };
                StepOutcome::Declined(refusal)
            }
            Quiescence::Proven => {
                self.slot = Slot::Empty;
                StepOutcome::Failed(f.message)
            }
        }
    }
}

// Quarantined programs are never dropped: the queue may still be executing.
// (`ManuallyDrop` keeps the field from running its destructor.)

// ---------------------------------------------------------------------- owner

/// Counters and the last decision, for `RaonBundle::railgun_stats`.
#[derive(Clone, Debug, Default)]
pub struct RaonRailgunStats {
    pub attempts: u64,
    pub completed: u64,
    pub declined: u64,
    /// Programs built (first build + rebuilds).
    pub builds: u64,
    pub rebuilds: u64,
    pub quarantined: bool,
    pub last_refusal: Option<String>,
    pub declined_by_kind: BTreeMap<String, u64>,
    pub nodes: usize,
    pub ib_dwords: u32,
    /// `(symbol, sha256)` of the current program's kernels.
    pub object_digests: Vec<(String, String)>,
    pub last_build_ms: f64,
    /// The field the last rebuild was triggered by.
    pub last_rebuild_cause: Option<String>,
}

/// The per-bundle owner. Created by `RaonBundle::new` per the tri-state
/// `HIPFIRE_RAON_RAILGUN` (see [`decide_owner`]); never exists under `0`.
pub struct RaonRailgunOwner {
    /// The device architecture of the bundle (`Gpu::arch`).
    arch: String,
    /// The per-arch policy; `None` for an arch the owner does not admit.
    policy: Option<&'static ArchPolicy>,
    /// Whether the HIP device's PCI address resolved (arches that need it).
    pci_ok: Option<bool>,
    corpus_dir: Option<String>,
    corpus: Option<Result<WindowCorpus, String>>,
    ctx: Option<RailgunLoweringContext>,
    table: Option<Result<CacheTable, String>>,
    model_digest: String,
    flags: Option<(usize, String)>,
    exec: Exec<RailgunWindowQueue>,
    stats: RaonRailgunStats,
    /// `1`: every decline is logged once per kind.
    loud: bool,
    logged_kinds: BTreeSet<&'static str>,
    /// `auto`: the one line for the first non-routine decline was written.
    auto_logged: bool,
}

/// The structural facts of `b` on `gpu`. `retained_replay` is passed in so
/// the one live read of that state stays in [`RaonRailgunOwner::try_step`].
fn observe_structural<'a>(
    b: &RaonBundle,
    gpu: &'a Gpu,
    band: &'a str,
    retained_replay: bool,
    device_identity: bool,
) -> Structural<'a> {
    Structural {
        arch: &gpu.arch,
        device_identity,
        whole_band: b.weights.band.is_whole(),
        band,
        graph_forward_disabled: gpu.flags.graph_forward == Some(false),
        graph_mode: b.decode_route() == DecodeMode::Graph,
        retained_replay,
        host_weights: gpu.owns_host_cpu_weights(),
    }
}

impl RaonRailgunOwner {
    /// The owner for `b` under `HIPFIRE_RAON_RAILGUN` (read once). `0` returns
    /// before reading anything else. An unrecognised spelling fails closed to
    /// `0` with a message. In `auto` an explicitly set
    /// `HIPFIRE_RAON_RAILGUN_CORPUS` that holds no corpus is reported once;
    /// every other reason for no owner is silent.
    pub(crate) fn from_config(b: &RaonBundle, gpu: &Gpu) -> Option<Box<Self>> {
        let raw = hipfire_config::developer_var(ENV_ENABLE).ok();
        let mode = parse_mode(raw.as_deref()).unwrap_or_else(|e| {
            eprintln!("[raon-railgun] {e}");
            RailgunMode::Off
        });
        if mode == RailgunMode::Off {
            return None;
        }
        let policy = arch_policy(&gpu.arch);
        let env_corpus = match policy {
            Some(p) => p.env_value(p.corpus_vars).map(|(_, v)| v),
            None => hipfire_config::developer_var(ENV_CORPUS)
                .ok()
                .filter(|s| !s.is_empty()),
        };
        let pack_corpus = gpu
            .railgun_pack_corpus_dir()
            .map(|p| p.to_string_lossy().into_owned());
        let band = b.weights.band.label();
        // `graph_mode` already carries a retained-replay load (the route
        // resolution picked eager), so the live replay state is not re-read.
        // The PCI identity resolves here only for an arch that needs it.
        let identity = !policy.is_some_and(|p| p.pci_identity)
            || match gpu.railgun_device_pci_address() {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("[raon-railgun] {} device identity: {e}", gpu.arch);
                    false
                }
            };
        let structural = observe_structural(b, gpu, &band, false, identity);
        let plan = decide_owner(
            mode,
            &structural,
            env_corpus.as_deref(),
            pack_corpus.as_deref(),
            &|dir| Path::new(dir).join(railgun_corpus::CORPUS_INDEX).is_file(),
        );
        match plan {
            Ok(plan) => Some(Box::new(Self::new(plan, &gpu.arch, model_digest(b)))),
            Err(NoOwner::NoCorpus(why)) if env_corpus.is_some() => {
                eprintln!("[raon-railgun] auto: HIP graph route ({why})");
                None
            }
            Err(_) => None,
        }
    }

    fn new(plan: OwnerPlan, arch: &str, model_digest: String) -> Self {
        Self {
            arch: arch.to_owned(),
            policy: arch_policy(arch),
            pci_ok: None,
            corpus_dir: plan.corpus_dir,
            corpus: None,
            ctx: None,
            table: None,
            model_digest,
            flags: None,
            exec: Exec::default(),
            stats: RaonRailgunStats::default(),
            loud: plan.loud,
            logged_kinds: BTreeSet::new(),
            auto_logged: false,
        }
    }

    pub fn stats(&self) -> RaonRailgunStats {
        let mut s = self.stats.clone();
        s.quarantined = self.exec.is_quarantined();
        s
    }

    /// `HIPFIRE_RAON_RAILGUN=1` (owner forced, declines logged per kind);
    /// `false` = `auto`.
    pub fn is_forced(&self) -> bool {
        self.loud
    }

    pub fn is_quarantined(&self) -> bool {
        self.exec.is_quarantined()
    }

    /// Text of the quarantine, if any.
    pub fn quarantine_reason(&self) -> Option<&str> {
        self.exec.quarantine_reason()
    }

    fn note_decline(&mut self, r: &RaonRailgunRefusal) {
        self.stats.declined += 1;
        *self
            .stats
            .declined_by_kind
            .entry(r.kind().to_owned())
            .or_default() += 1;
        let text = r.to_string();
        if self.loud {
            if self.logged_kinds.insert(r.kind()) {
                eprintln!("[raon-railgun] declined ({}): {text}", r.kind());
            }
        } else if !self.auto_logged && *r != RaonRailgunRefusal::NotWarm {
            // `NotWarm` is the routine first step (the HIP graph's warm-up).
            self.auto_logged = true;
            eprintln!("[raon-railgun] auto: HIP graph step ({}): {text}", r.kind());
        }
        self.stats.last_refusal = Some(text);
    }

    fn flags_digest(&mut self, gpu: &Gpu) -> String {
        let ptr = Arc::as_ptr(&gpu.flags) as usize;
        if let Some((p, d)) = &self.flags {
            if *p == ptr {
                return d.clone();
            }
        }
        let d = sha256_hex(format!("{:?}", *gpu.flags).as_bytes());
        self.flags = Some((ptr, d.clone()));
        d
    }

    /// One decode step through the program. See the module docs.
    pub(crate) fn try_step(
        &mut self,
        b: &mut RaonBundle,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<RaonRailgunAttempt, String> {
        self.stats.attempts += 1;
        if let Some(why) = self.exec.quarantine_reason() {
            return Err(format!(
                "raon1_mini: the railgun owner is quarantined after an unknown-quiescence failure: {why}"
            ));
        }
        let pos = b.state.n_tokens();
        let band = b.weights.band.label();
        let identity = if gpu.arch != self.arch {
            eprintln!(
                "[raon-railgun] owner of a {} bundle stepped on a {} device",
                self.arch, gpu.arch
            );
            false
        } else {
            self.device_identity_ok(gpu)
        };
        let structural = observe_structural(b, gpu, &band, gpu.replay.is_enabled(), identity);
        let step = StepFacts {
            tap_armed: b.spec.census_tap.is_some(),
            dump_armed: hipfire_dispatch::pipeline::moe_capture::enabled(),
            absorbed_route: mla_route(&b.work, &b.state, pos + 1) == MlaRoute::Absorbed,
            warmed: b.graph.warmed,
            corpus_dir: self.corpus_dir.as_deref(),
        };
        if let Err(r) = check_admission(&structural, &step) {
            self.note_decline(&r);
            return Ok(RaonRailgunAttempt::Declined(r));
        }
        // Context admission, outside any program (same call the graph route
        // makes; it may grow the mapping or mirror).
        admit_decode_context(
            &b.config,
            &b.weights,
            &mut b.state,
            &b.scratch,
            &b.work,
            gpu,
            pos + 1,
        )?;
        // The decode stream the graph route creates on its first recorded step.
        if gpu.active_stream.is_none() {
            let stream = gpu
                .hip
                .stream_create()
                .map_err(|e| format!("raon1_mini: railgun decode stream: {e:?}"))?;
            gpu.active_stream = Some(stream);
        }
        let base = BaseKey {
            binding: decode_binding(&b.weights, &b.state, &b.scratch, &b.work),
            mapped_rows: b.state.capacity_rows(),
            mirror_rows: b.state.expanded_mirror().map_or(0, |m| m.rows),
            absorbed: false,
            scratch_rows: b.scratch.rows,
            scratch_epoch: rdna_compute::scratch::scratch_growth_epoch(),
            model_digest: self.model_digest.clone(),
            flags_digest: self.flags_digest(gpu),
        };
        let gpu_ref: &Gpu = gpu;
        let lookup = self.exec.lookup(&base, &mut |built| {
            let objects = built
                .symbols
                .iter()
                .map(|s| {
                    (
                        s.clone(),
                        gpu_ref
                            .railgun_loaded_digest(s)
                            .unwrap_or_else(|e| format!("unloaded: {e}")),
                    )
                })
                .collect();
            let tables = built
                .table_bases
                .iter()
                .map(|base| match built.registry.table_at(*base) {
                    Some(t) => (*base, t.generation, hex(&t.content_sha256)),
                    None => (*base, 0, "retired".to_owned()),
                })
                .collect();
            (objects, tables)
        });
        match lookup {
            Lookup::Hit => {}
            Lookup::Declined(r) => {
                self.note_decline(&r);
                return Ok(RaonRailgunAttempt::Declined(r));
            }
            Lookup::Miss(cause) => {
                let rebuild = matches!(self.exec.slot, Slot::Ready(_));
                if rebuild {
                    self.stats.rebuilds += 1;
                    self.stats.last_rebuild_cause = cause.map(str::to_owned);
                }
                // The previous program is quiescent (submits block): release it
                // before planning the new one.
                self.exec.clear();
                let started = Instant::now();
                match self.build(b, gpu, &base) {
                    Ok(built) => {
                        self.stats.builds += 1;
                        self.stats.nodes = built.nodes;
                        self.stats.ib_dwords = built.ib_dwords;
                        self.stats.object_digests = built.key.object_digests.clone();
                        self.stats.last_build_ms = started.elapsed().as_secs_f64() * 1e3;
                        self.exec.install(built);
                    }
                    Err(r) => {
                        self.exec.decline(base, r.clone());
                        self.note_decline(&r);
                        return Ok(RaonRailgunAttempt::Declined(r));
                    }
                }
            }
        }
        // Token on the decode stream, then wait for it before the independent
        // queue reads it.
        stage_decode_token_on_stream(&b.config, &b.scratch, gpu, token)?;
        let stream = gpu
            .active_stream
            .as_ref()
            .ok_or("raon1_mini: railgun decode: no decode stream")?;
        gpu.hip
            .stream_synchronize(stream)
            .map_err(|e| format!("raon1_mini: railgun decode: stage token wait: {e:?}"))?;
        match self.exec.submit() {
            StepOutcome::Completed => {
                self.stats.completed += 1;
                b.state.note_host_tokens(1);
                Ok(RaonRailgunAttempt::Completed)
            }
            StepOutcome::Declined(r) => {
                self.note_decline(&r);
                Ok(RaonRailgunAttempt::Declined(r))
            }
            StepOutcome::Failed(m) => Err(format!(
                "raon1_mini: railgun decode submit failed (quiescent, request aborted): {m}"
            )),
            StepOutcome::Quarantined(m) => {
                self.stats.quarantined = true;
                eprintln!(
                    "[raon-railgun] QUARANTINE: unknown quiescence after a submit ({m}); \
                     tensors, tables and the program are leaked, not freed"
                );
                Err(format!(
                    "raon1_mini: railgun decode submit failed with unknown quiescence \
                     (owner and bundle quarantined): {m}"
                ))
            }
        }
    }

    fn cache_table(&mut self) -> Result<&CacheTable, RaonRailgunRefusal> {
        let policy = self.policy.ok_or_else(|| RaonRailgunRefusal::WrongArch {
            arch: self.arch.clone(),
        })?;
        let t = self.table.get_or_insert_with(|| cache_table(policy));
        t.as_ref()
            .map_err(|e| RaonRailgunRefusal::CacheTable(e.clone()))
    }

    /// Whether the HIP device's PCI address resolves, for an arch whose ROCr
    /// agent is matched by it (always `true` for the others); read once.
    fn device_identity_ok(&mut self, gpu: &Gpu) -> bool {
        if !self.policy.is_some_and(|p| p.pci_identity) {
            return true;
        }
        *self
            .pci_ok
            .get_or_insert_with(|| match gpu.railgun_device_pci_address() {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("[raon-railgun] {} device identity: {e}", gpu.arch);
                    false
                }
            })
    }

    /// Plan, author, witness, hazard-plan and lower the program for the
    /// current binding. Every failure is a refusal.
    fn build(
        &mut self,
        b: &mut RaonBundle,
        gpu: &mut Gpu,
        base: &BaseKey,
    ) -> Result<Built<RailgunWindowQueue>, RaonRailgunRefusal> {
        use RaonRailgunRefusal as R;
        if self.corpus.is_none() {
            let dir = self.corpus_dir.clone().ok_or(R::CorpusUnset)?;
            self.corpus = Some(WindowCorpus::open(gpu, std::path::Path::new(&dir)));
        }
        if let Some(Err(e)) = &self.corpus {
            return Err(R::Corpus(e.clone()));
        }
        let transport = gpu.railgun_transport_digests().map_err(R::Plan)?;
        let (ops, declared) = plan_body(b, gpu).map_err(R::Plan)?;
        let declared: Vec<usize> = declared
            .into_iter()
            .filter(|(_, w)| !w.is_empty())
            .map(|(i, _)| i)
            .collect();
        let corpus = match self.corpus.as_mut() {
            Some(Ok(c)) => c,
            _ => unreachable!("corpus checked above"),
        };
        let records = {
            let gpu_ref: &Gpu = gpu;
            launch_records(
                &ops,
                &declared,
                &hex(&transport.copy),
                &hex(&transport.fill),
                &mut |sha, sym| corpus.facts(sha, sym),
                &|addr| gpu_ref.railgun_probe_allocation(addr),
            )?
        };
        let table = self.cache_table()?.clone();
        let authored = author_and_plan(&records, &table, &GpuView(&*gpu))?;
        if self.ctx.is_none() {
            self.ctx = Some(gpu.railgun_lowering_context().map_err(R::Lower)?);
        }
        let ctx = self.ctx.as_mut().expect("context just ensured");
        let queue = gpu
            .lower_railgun_window(ctx, &authored.program, &authored.plan)
            .map_err(R::Lower)?;
        let mut symbols: Vec<String> = authored
            .program
            .nodes
            .iter()
            .map(|n| n.symbol.clone())
            .collect();
        symbols.sort();
        symbols.dedup();
        let mut object_digests = Vec::with_capacity(symbols.len());
        for s in &symbols {
            object_digests.push((s.clone(), gpu.railgun_loaded_digest(s).map_err(R::Lower)?));
        }
        let table_generations = authored
            .table_bases
            .iter()
            .filter_map(|base| authored.registry.table_at(*base))
            .map(|t| (t.allocation.base, t.generation, hex(&t.content_sha256)))
            .collect();
        let ib_dwords = queue.command_dwords();
        Ok(Built {
            key: RaonRailgunKey {
                base: base.clone(),
                object_digests,
                table_generations,
            },
            queue,
            nodes: authored.program.nodes.len(),
            ib_dwords,
            images: authored.images,
            grids: authored.grids,
            symbols,
            registry: authored.registry,
            table_bases: authored.table_bases,
        })
    }
}

/// The decode body under the plan sink with the HIP-graph route flags, after
/// context admission. Executes nothing; the warm-up step has already loaded
/// every module (plan mode refuses loads).
fn plan_body(
    b: &RaonBundle,
    gpu: &mut Gpu,
) -> Result<
    (
        Vec<PlannedOp>,
        Vec<(usize, Vec<railgun::kernel::DeclaredWord>)>,
    ),
    String,
> {
    let mut sink = PlanSink::default();
    let prior = gpu.graphs.capture_mode;
    gpu.graphs.capture_mode = true;
    let r = gpu.with_plan_sink(&mut sink, |g| {
        decode_body(&b.config, &b.weights, &b.state, &b.scratch, &b.work, g)
            .map_err(|e| hip_bridge::HipError::new(0, &e))
    });
    gpu.graphs.capture_mode = prior;
    r.map_err(|e| format!("{e:?} (refusal: {:?})", sink.refusal))?;
    Ok((sink.ops, take_declared_words()))
}

/// The arch's cache table with the optional G4 silicon receipts attached,
/// exactly as the DFlash window builds it. The receipt file must be the
/// probe table of this arch (`cache-table.<arch>.json`; `attach_receipts`
/// refuses another arch's identity).
fn cache_table(policy: &ArchPolicy) -> Result<CacheTable, String> {
    let arch = policy.arch;
    let mut table = CacheTable::for_arch(arch)
        .ok_or_else(|| format!("railgun has no cache table for {arch}"))?;
    if let Some((var, path)) = policy.env_value(policy.cache_table_vars) {
        let bytes = std::fs::read(&path).map_err(|e| format!("{var}={path}: {e}"))?;
        let attached = attach_g4(&mut table, &path, &bytes).map_err(|e| format!("{var}: {e}"))?;
        eprintln!(
            "[raon-railgun] {arch} G4 receipts {attached}/{} rows backed by {path} sha256={}",
            table.rows.len(),
            sha256_hex(&bytes)
        );
    }
    Ok(table)
}

/// Attach the receipts of the G4 probe table `bytes` (read from `path`) to
/// `table`: the number of rows backed. A probe table of another arch, or one
/// that saw a table rung stale, is an error.
fn attach_g4(table: &mut CacheTable, path: &str, bytes: &[u8]) -> Result<usize, String> {
    let probe: G4Table = serde_json::from_slice(bytes).map_err(|e| format!("{path}: {e}"))?;
    let source = format!("{path} sha256={}", sha256_hex(bytes));
    table.attach_receipts(&probe, &source, G4_MIN_TRIALS)
}

/// Digest of the bundle's model identity: fixed for its lifetime.
pub(crate) fn model_digest(b: &RaonBundle) -> String {
    sha256_hex(
        format!(
            "{:?}|{}|{}|{}",
            b.config,
            b.precision_plan().coverage_line(),
            b.weights.band.label(),
            b.weights.layers.len(),
        )
        .as_bytes(),
    )
}

#[cfg(test)]
mod tests;
