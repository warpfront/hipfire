// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lever1 expert hotset sidecar: parser, validator and enabling policy.
//!
//! A sidecar is an immutable JSON file (`kind = hipfire.qwen4.expert_hotset`,
//! `version = 1`) that ranks every routed expert of the trunk by how often a
//! decode capture routed to it. A global expert id is
//! `layer * experts_per_layer + expert`. The loader fills the longest ranked
//! prefix that fits its runtime byte budget; the budget is never read from
//! the file.
//!
//! # Ranking rule (enforced)
//!
//! `rank` is a permutation of every global id exactly once, ordered by
//! `counts[id]` descending with ties broken by global id ascending.
//! `counts` is indexed by global id and has one entry per expert.
//! `ranking_rule` and `corpus` are provenance strings and are carried, not
//! interpreted.
//!
//! This module hashes only the sidecar bytes ([`HotsetSidecar::sha256`]).
//! `model_sha256`, `counts_sha256`, `format_map_sha256` are validated as full
//! lowercase 64-hex digests and exposed for the loader to compare against its
//! own pins; no model artifact is read here.
//!
//! # Base binding
//!
//! `base_size` and `base_index_sha256` (both required) bind the sidecar to the
//! exact base artifact: the byte length of the HFQ file and the lowercase
//! sha256 of its `[0, data_offset)` header/index span
//! ([`HotsetBase::from_hfq`], the same binding a PLE sidecar uses). There is no
//! whole-artifact hash, so binding never reads a payload.
//!
//! # Default-on policy
//!
//! [`HotsetPolicy`] reads `HIPFIRE_QWEN4_EXPERT_HOTSET`: unset = the
//! `<model>.expert-hotset` file next to the model when it exists, `0` = off,
//! an absolute path = that file. A relative or empty value is the only policy
//! error ([`HotsetPolicy::parse`]). Without a fixed [`HotsetBudget`], every
//! other reason the hotset cannot be used (arch, placement mode, artifact
//! layout, bad or mismatched sidecar, no budget) is a [`HotsetSkip`]: the
//! loader keeps whole-layer expert placement. An explicit fixed budget
//! requires the ranked hotset and fails the load instead of falling back.
//!
//! No GPU access; the only I/O is [`preflight`] statting one file,
//! [`HotsetSidecar::load`] reading one file and [`HotsetBase::from_hfq`]
//! hashing the base index span.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};

use hipfire_config::memory::expert_hotset_sidecar_path;
use hipfire_runtime::hfq::HfqFile;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::expert_residency::EXPERT_VRAM_LAYERS_ENV;

/// Selects the sidecar. Unset: `<model>.expert-hotset` when it exists.
/// `0`: off. Absolute path: that file. Anything else is a policy error.
pub const HOTSET_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOTSET";
/// Developer-only fixed device-expert budget in MiB; unset keeps auto sizing.
pub const HOT_BUDGET_MIB_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB";
/// `kind` of a v1 sidecar.
pub const SIDECAR_KIND: &str = "hipfire.qwen4.expert_hotset";
/// `version` of the sidecar schema this module reads.
pub const SIDECAR_VERSION: u64 = 1;
/// The GPU architecture the default (sibling sidecar) hotset runs on.
pub const HOTSET_ARCH: &str = "gfx1201";

/// Whether a hotset applies on `arch`: the default sibling sidecar only on
/// [`HOTSET_ARCH`]; an explicit [`HOTSET_ENV`] sidecar path on any arch (the
/// MQ4 tier kernels carry no arch guard; the other environment skips and the
/// tier plan's per-arch format check still apply). The single arch rule of the
/// loader's [`decide_environment`]; the CPU tier split has no arch gate of its
/// own, since a tier plan exists only where this held at load.
pub fn hotset_arch_admits(arch: &str, explicit_path: bool) -> bool {
    arch == HOTSET_ARCH || explicit_path
}

/// Log prefix of every hotset line.
pub const LOG_PREFIX: &str = "  qwen4 expert hot set: ";
/// Runtime hot-expert budget: automatic by default, or an exact byte ceiling
/// for the ranked prefix, with the same meaning as lab `--hot-budget-mib`.
/// The fixed ceiling does not include long-prefill staging buffers; those
/// still have to fit alongside it in the loader's available budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HotsetBudget {
    fixed_bytes: Option<u64>,
}

impl HotsetBudget {
    /// Parse positive unsigned decimal MiB with checked conversion to bytes.
    /// Empty, zero, malformed and overflowing explicit values are errors.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        let Some(raw) = raw else {
            return Ok(Self { fixed_bytes: None });
        };
        let invalid = || {
            format!(
                "{HOT_BUDGET_MIB_ENV}={raw:?}: expected positive unsigned decimal MiB \
             that fit in u64 bytes; unset it for automatic sizing"
            )
        };
        if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(invalid());
        }
        let mib = raw.parse::<u64>().map_err(|_| invalid())?;
        let bytes = mib
            .checked_mul(1 << 20)
            .filter(|&bytes| bytes > 0)
            .ok_or_else(invalid)?;
        Ok(Self {
            fixed_bytes: Some(bytes),
        })
    }

    /// Read once from the process configuration snapshot, like [`HotsetPolicy`].
    pub fn from_env() -> Result<Self, String> {
        Self::parse(
            hipfire_config::developer_var(HOT_BUDGET_MIB_ENV)
                .ok()
                .as_deref(),
        )
    }

    /// Whether failure to apply the requested hotset must fail the load.
    pub fn is_fixed(self) -> bool {
        self.fixed_bytes.is_some()
    }

    /// Select the unchanged fixed ceiling only if it fits the available
    /// bytes after all reserves, including long-prefill staging. Never clamp.
    /// Automatic sizing retains the existing positive-budget requirement.
    pub fn select(self, available: u64) -> Result<u64, String> {
        match self.fixed_bytes {
            Some(requested) if requested > available => Err(format!(
                "{HOT_BUDGET_MIB_ENV} requests {requested} bytes ({} MiB), \
                 but only {available} bytes ({} MiB) are available for hot experts \
                 after reserves; refusing to shrink the fixed budget",
                requested >> 20,
                available >> 20,
            )),
            Some(requested) => Ok(requested),
            None => require_budget(Ok(available)).map_err(|skip| skip.to_string()),
        }
    }
}

/// `HIPFIRE_QWEN4_EXPERT_HOTSET`: where the sidecar comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotsetPolicy {
    /// Unset: `<model>.expert-hotset` when it exists.
    Default,
    /// `0`: never use a hotset.
    Off,
    /// An absolute sidecar path.
    Explicit(PathBuf),
}

impl HotsetPolicy {
    /// Parse the raw variable value (`None` = unset). Only a relative or
    /// empty value is an error.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None => Ok(Self::Default),
            Some("0") => Ok(Self::Off),
            Some(value) if value.trim().is_empty() => Err(format!(
                "{HOTSET_ENV} is set but empty; unset it, set 0 to disable, or give an absolute \
                 sidecar path"
            )),
            Some(value) => {
                let path = PathBuf::from(value);
                if path.is_absolute() {
                    Ok(Self::Explicit(path))
                } else {
                    Err(format!(
                        "{HOTSET_ENV}={value:?} is not an absolute path (set 0 to disable)"
                    ))
                }
            }
        }
    }

    /// The policy from the process configuration snapshot.
    pub fn from_env() -> Result<Self, String> {
        Self::parse(hipfire_config::developer_var(HOTSET_ENV).ok().as_deref())
    }

    /// The sidecar this policy names for `model`: `<model>.expert-hotset`
    /// for [`Self::Default`] and [`Self::Off`] (so `0` can still report a
    /// present sidecar it did not apply), the explicit path otherwise.
    pub fn sidecar_path(&self, model: &Path) -> PathBuf {
        match self {
            Self::Default | Self::Off => expert_hotset_sidecar_path(model),
            Self::Explicit(path) => path.clone(),
        }
    }

    fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }
}

/// Selects what the hotset sidecar is allowed to cost. Unset or `ranked`:
/// the ranked-prefix tier plan (the [`HotsetPolicy`] path). `whole-first`:
/// gfx1201 keeps the `hotset=0` whole-layer placement and never builds the
/// tier plan, so decode takes the whole-layer route. Developer variable,
/// independent of [`HOTSET_ENV`] (which only names the sidecar).
pub const PLACEMENT_POLICY_ENV: &str = "HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY";

/// `HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY`: how expert VRAM is spent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlacementPolicy {
    /// Unset or `ranked`: the existing hotset behaviour.
    #[default]
    Ranked,
    /// `whole-first`: whole layers only; no sidecar tier plan.
    WholeFirst,
}

impl PlacementPolicy {
    /// Parse the raw variable value (`None` = unset = [`Self::Ranked`]).
    /// An empty or unknown value is an error.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw.map(str::trim) {
            None | Some("ranked") => Ok(Self::Ranked),
            Some("whole-first") => Ok(Self::WholeFirst),
            Some(value) => Err(format!(
                "{PLACEMENT_POLICY_ENV}={value:?} is not `ranked` or `whole-first`; unset it for \
                 the ranked default"
            )),
        }
    }

    /// The policy from the process configuration snapshot.
    pub fn from_env() -> Result<Self, String> {
        Self::parse(
            hipfire_config::developer_var(PLACEMENT_POLICY_ENV)
                .ok()
                .as_deref(),
        )
    }
}

/// Whether `whole-first` governs a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeFirstDecision {
    /// Not requested, or not gfx1201: the load is exactly the default path.
    Inactive,
    /// gfx1201 load that keeps the `hotset=0` whole-layer placement and does
    /// not consult the sidecar.
    Active,
    /// Requested on gfx1201 but this load's placement is chosen elsewhere;
    /// carries why. Placement is unchanged.
    Ignored(&'static str),
}

/// The pure `whole-first` decision. Effect is strictly [`HOTSET_ARCH`]; on
/// any other architecture the policy is inert and silent.
pub fn decide_whole_first(
    policy: PlacementPolicy,
    env: &HotsetEnv<'_>,
    pipeline_band: bool,
) -> WholeFirstDecision {
    if policy != PlacementPolicy::WholeFirst || env.gpu_arch != HOTSET_ARCH {
        WholeFirstDecision::Inactive
    } else if env.unified_memory {
        WholeFirstDecision::Ignored("unified-memory/use_ranges loading does not use the hotset")
    } else if env.expert_parallel {
        WholeFirstDecision::Ignored(
            "expert-parallel or an agreed expert split chooses expert placement itself",
        )
    } else if pipeline_band {
        WholeFirstDecision::Ignored("a pipeline band keeps every routed expert resident")
    } else if env.explicit_vram_layers {
        WholeFirstDecision::Ignored(
            "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS chooses expert placement itself",
        )
    } else {
        WholeFirstDecision::Active
    }
}

/// Routed-expert layout of the base artifact, as the loader classified it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotsetSource {
    /// Packed Flash-Next aggregate MQ4-XTS routed experts: the only layout
    /// the hotset can back.
    Aggregate,
    /// Per-expert slab artifact (slab arena / K4 route).
    SlabArena,
    /// Stock FP8 routed experts.
    StockFp8,
    /// Any other layout that is not aggregate MQ4-XTS; the text says why.
    Nonaggregate(String),
}

impl fmt::Display for HotsetSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aggregate => f.write_str("aggregate MQ4-XTS experts"),
            Self::SlabArena => f.write_str(
                "per-expert slab artifact (slab arena / K4 route), not aggregate MQ4-XTS",
            ),
            Self::StockFp8 => f.write_str("stock-FP8 routed experts (qt55), not aggregate MQ4-XTS"),
            Self::Nonaggregate(detail) => {
                write!(f, "not an aggregate MQ4-XTS artifact: {detail}")
            }
        }
    }
}

/// Load-time facts the environmental eligibility decision reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotsetEnv<'a> {
    /// GPU architecture name, e.g. `gfx1201`.
    pub gpu_arch: &'a str,
    /// Unified-memory / `use_ranges` (APU) loading.
    pub unified_memory: bool,
    /// Expert-parallel placement.
    pub expert_parallel: bool,
    /// An explicit expert VRAM layer count/split was requested
    /// ([`EXPERT_VRAM_LAYERS_ENV`] or its agreed equivalent).
    pub explicit_vram_layers: bool,
    /// Routed-expert layout of the base artifact.
    pub source: HotsetSource,
}

/// Why the hotset is not used for this load. Never a load failure: the
/// loader keeps whole-layer expert placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotsetSkip {
    /// [`HOTSET_ENV`]`=0`.
    Disabled,
    /// The sidecar file does not exist. `explicit`: the path came from
    /// [`HOTSET_ENV`] rather than the default `<model>.expert-hotset`.
    Missing { explicit: bool },
    /// The GPU is not [`HOTSET_ARCH`] (carries the arch).
    NotGfx1201(String),
    /// Unified-memory / `use_ranges` loading.
    UnifiedMemory,
    /// Expert-parallel placement.
    ExpertParallel,
    /// The artifact is not aggregate MQ4-XTS (never [`HotsetSource::Aggregate`]).
    Source(HotsetSource),
    /// An explicit expert VRAM layer setting decides placement.
    ExplicitVramLayers,
    /// The sidecar cannot be read or does not validate.
    InvalidSidecar(String),
    /// The sidecar is bound to a different base artifact.
    BaseMismatch(String),
    /// No VRAM budget for hot experts (carries why).
    NoBudget(String),
    /// The plan places no expert in VRAM.
    ZeroExperts,
}

impl fmt::Display for HotsetSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => write!(f, "{HOTSET_ENV}=0"),
            Self::Missing { .. } => f.write_str("sidecar not found"),
            Self::NotGfx1201(arch) => write!(
                f,
                "gpu is {arch}; the default hotset runs on {HOTSET_ARCH} only \
                 (other archs need an explicit {HOTSET_ENV} path)"
            ),
            Self::UnifiedMemory => {
                f.write_str("unified-memory/use_ranges loading does not use the hotset")
            }
            Self::ExpertParallel => {
                f.write_str("expert-parallel placement does not use the hotset")
            }
            Self::Source(source) => write!(f, "{source}"),
            Self::ExplicitVramLayers => write!(
                f,
                "{EXPERT_VRAM_LAYERS_ENV} is set and chooses expert placement itself"
            ),
            Self::InvalidSidecar(reason) => write!(f, "invalid sidecar: {reason}"),
            Self::BaseMismatch(reason) => write!(f, "sidecar is for a different base: {reason}"),
            Self::NoBudget(reason) => write!(f, "no VRAM budget for hot experts: {reason}"),
            Self::ZeroExperts => f.write_str("the plan places no expert in VRAM"),
        }
    }
}

impl HotsetSkip {
    /// The single log line (no newline) for a skip at `path`. Any failed
    /// condition reports exactly one line; the only silent skip is an absent
    /// sidecar with [`HOTSET_ENV`] unset.
    pub fn log_line(&self, path: &Path) -> Option<String> {
        match self {
            Self::Missing { explicit: false } => None,
            other => Some(format!(
                "{LOG_PREFIX}{} present, not applied ({other}); whole-layer placement",
                path.display()
            )),
        }
    }
}

/// The pure environmental eligibility decision. `sidecar_exists` is whether
/// the file named by `policy` exists. Order: disabled (`0`, even with no
/// sidecar), missing (silent only for the unset default), GPU arch, unified
/// memory, expert parallel, artifact layout, explicit VRAM layers. `Ok` is
/// the sidecar path to load.
pub fn decide_environment(
    policy: &HotsetPolicy,
    model: &Path,
    sidecar_exists: bool,
    env: &HotsetEnv<'_>,
) -> Result<PathBuf, HotsetSkip> {
    if *policy == HotsetPolicy::Off {
        return Err(HotsetSkip::Disabled);
    }
    if !sidecar_exists {
        return Err(HotsetSkip::Missing {
            explicit: policy.is_explicit(),
        });
    }
    if !hotset_arch_admits(env.gpu_arch, policy.is_explicit()) {
        return Err(HotsetSkip::NotGfx1201(env.gpu_arch.to_owned()));
    }
    if env.unified_memory {
        return Err(HotsetSkip::UnifiedMemory);
    }
    if env.expert_parallel {
        return Err(HotsetSkip::ExpertParallel);
    }
    if env.source != HotsetSource::Aggregate {
        return Err(HotsetSkip::Source(env.source.clone()));
    }
    if env.explicit_vram_layers {
        return Err(HotsetSkip::ExplicitVramLayers);
    }
    Ok(policy.sidecar_path(model))
}

/// [`decide_environment`] after statting the sidecar. A stat error other
/// than not-found is an [`HotsetSkip::InvalidSidecar`].
pub fn preflight(
    policy: &HotsetPolicy,
    model: &Path,
    env: &HotsetEnv<'_>,
) -> Result<PathBuf, HotsetSkip> {
    let exists = match std::fs::metadata(policy.sidecar_path(model)) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            return Err(HotsetSkip::InvalidSidecar(format!(
                "cannot stat the sidecar: {error}"
            )));
        }
    };
    decide_environment(policy, model, exists, env)
}

/// `Ok(budget)` for a usable VRAM budget; an `Err` budget computation or a
/// zero budget is [`HotsetSkip::NoBudget`].
pub fn require_budget(budget: Result<u64, String>) -> Result<u64, HotsetSkip> {
    match budget {
        Ok(0) => Err(HotsetSkip::NoBudget("the budget is 0 bytes".to_owned())),
        Ok(bytes) => Ok(bytes),
        Err(reason) => Err(HotsetSkip::NoBudget(reason)),
    }
}

/// `Ok(n)` when the plan places at least one expert in VRAM; zero is
/// [`HotsetSkip::ZeroExperts`].
pub fn require_experts(vram_experts: usize) -> Result<usize, HotsetSkip> {
    if vram_experts == 0 {
        Err(HotsetSkip::ZeroExperts)
    } else {
        Ok(vram_experts)
    }
}

/// The identity of the base artifact a sidecar is bound to: HFQ file length
/// and sha256 of its `[0, data_offset)` header/index span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotsetBase {
    size: u64,
    index_sha256: String,
}

impl HotsetBase {
    pub fn new(size: u64, index_sha256: [u8; 32]) -> Self {
        Self {
            size,
            index_sha256: hex(&index_sha256),
        }
    }

    /// Read the identity of an open HFQ base (positional reads of the index
    /// span only; no payload).
    pub fn from_hfq(hfq: &HfqFile) -> Result<Self, String> {
        let index = hfq
            .index_sha256()
            .map_err(|e| format!("cannot hash the base index: {e}"))?;
        Ok(Self::new(hfq.file_len(), index))
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Lowercase hex sha256 of the base's header/index span.
    pub fn index_sha256(&self) -> &str {
        &self.index_sha256
    }

    /// Check a sidecar's binding keys (`base_size`, then `base_index_sha256`) name this base.
    fn check_binding(&self, base_size: u64, base_index_sha256: &str) -> Result<(), String> {
        if base_size != self.size {
            return Err(format!(
                "base is {} bytes, sidecar binds {base_size}",
                self.size
            ));
        }
        if base_index_sha256 != self.index_sha256 {
            return Err(format!(
                "base index sha256 {} differs from the sidecar's {}",
                short(&self.index_sha256),
                short(base_index_sha256)
            ));
        }
        Ok(())
    }
}

/// Wire form; unknown keys are refused so v1 stays closed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSidecar {
    version: u64,
    kind: String,
    model_sha256: String,
    base_size: u64,
    base_index_sha256: String,
    counts_sha256: String,
    format_map: String,
    format_map_sha256: String,
    layers: u64,
    experts_per_layer: u64,
    top_k: u64,
    capture_tokens: u64,
    ranking_rule: String,
    corpus: String,
    rank: Vec<u64>,
    counts: Vec<u64>,
}

/// A validated v1 hotset sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotsetSidecar {
    sha256: String,
    model_sha256: String,
    base_size: u64,
    base_index_sha256: String,
    counts_sha256: String,
    format_map: String,
    format_map_sha256: String,
    layers: usize,
    experts_per_layer: usize,
    top_k: usize,
    capture_tokens: u64,
    ranking_rule: String,
    corpus: String,
    rank: Vec<u32>,
    counts: Vec<u64>,
}

impl HotsetSidecar {
    /// Read and validate the sidecar at `path` against the model's trunk
    /// dimensions. Errors name the path.
    pub fn load(
        path: impl AsRef<Path>,
        layers: usize,
        experts_per_layer: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| format!("{HOTSET_ENV}: cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes, layers, experts_per_layer, top_k)
            .map_err(|e| format!("{HOTSET_ENV}: invalid sidecar {}: {e}", path.display()))
    }

    /// Validate sidecar `bytes` (the pure core of [`Self::load`]).
    pub fn from_bytes(
        bytes: &[u8],
        layers: usize,
        experts_per_layer: usize,
        top_k: usize,
    ) -> Result<Self, String> {
        let sha256 = hex(&Sha256::digest(bytes));
        let raw: RawSidecar =
            serde_json::from_slice(bytes).map_err(|e| format!("not a v1 hotset JSON: {e}"))?;

        if raw.kind != SIDECAR_KIND {
            return Err(format!("kind {:?}, expected {SIDECAR_KIND:?}", raw.kind));
        }
        if raw.version != SIDECAR_VERSION {
            return Err(format!(
                "version {}, expected {SIDECAR_VERSION}",
                raw.version
            ));
        }
        for (name, value) in [
            ("model_sha256", &raw.model_sha256),
            ("counts_sha256", &raw.counts_sha256),
            ("format_map_sha256", &raw.format_map_sha256),
        ] {
            check_sha256(name, value)?;
        }
        if raw.base_size == 0 {
            return Err("base_size is 0".to_owned());
        }
        check_sha256("base_index_sha256", &raw.base_index_sha256)?;
        for (name, value) in [
            ("format_map", &raw.format_map),
            ("ranking_rule", &raw.ranking_rule),
            ("corpus", &raw.corpus),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} is empty"));
            }
        }

        check_dim("layers", raw.layers, layers)?;
        check_dim(
            "experts_per_layer",
            raw.experts_per_layer,
            experts_per_layer,
        )?;
        check_dim("top_k", raw.top_k, top_k)?;
        if raw.capture_tokens == 0 {
            return Err("capture_tokens is 0".to_owned());
        }
        let total = layers
            .checked_mul(experts_per_layer)
            .filter(|&total| total > 0 && u32::try_from(total).is_ok())
            .ok_or_else(|| {
                format!("layers {layers} x experts_per_layer {experts_per_layer} is not a valid expert count")
            })?;

        if raw.counts.len() != total {
            return Err(format!(
                "counts has {} entries, expected {total} (layers {layers} x experts_per_layer \
                 {experts_per_layer})",
                raw.counts.len()
            ));
        }

        // Every id exactly once: range and duplicates in one pass, then the
        // first missing id if the list is short.
        let mut seen = vec![false; total];
        let mut rank = Vec::with_capacity(raw.rank.len());
        for (position, &id) in raw.rank.iter().enumerate() {
            let slot = usize::try_from(id)
                .ok()
                .filter(|&slot| slot < total)
                .ok_or_else(|| format!("rank[{position}] = {id} is out of range 0..{total}"))?;
            if std::mem::replace(&mut seen[slot], true) {
                return Err(format!("rank[{position}] = {id} is a duplicate"));
            }
            rank.push(slot as u32);
        }
        if rank.len() != total {
            let missing = seen.iter().position(|&seen| !seen).unwrap_or(total);
            return Err(format!(
                "rank has {} ids, expected {total}; first missing id {missing}",
                rank.len()
            ));
        }

        // Descending count, ties by ascending global id.
        for (position, pair) in rank.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let (count_a, count_b) = (raw.counts[a as usize], raw.counts[b as usize]);
            if count_a < count_b || (count_a == count_b && a > b) {
                return Err(format!(
                    "rank is not (count descending, id ascending) at positions {position}..{}: \
                     id {a} count {count_a} precedes id {b} count {count_b}",
                    position + 1
                ));
            }
        }

        Ok(Self {
            sha256,
            model_sha256: raw.model_sha256,
            base_size: raw.base_size,
            base_index_sha256: raw.base_index_sha256,
            counts_sha256: raw.counts_sha256,
            format_map: raw.format_map,
            format_map_sha256: raw.format_map_sha256,
            layers,
            experts_per_layer,
            top_k,
            capture_tokens: raw.capture_tokens,
            ranking_rule: raw.ranking_rule,
            corpus: raw.corpus,
            rank,
            counts: raw.counts,
        })
    }

    /// Every global expert id (`layer * experts_per_layer + expert`) exactly
    /// once, hottest first (count descending, id ascending).
    pub fn rank(&self) -> &[u32] {
        &self.rank
    }

    /// Decode routing counts indexed by global expert id.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Count of global expert `id`. Panics when `id` is out of range.
    pub fn count(&self, id: u32) -> u64 {
        self.counts[id as usize]
    }

    /// `(layer, expert)` of a global expert id.
    pub fn split_id(&self, id: u32) -> (usize, usize) {
        let id = id as usize;
        (id / self.experts_per_layer, id % self.experts_per_layer)
    }

    /// Lowercase hex SHA-256 of the sidecar file bytes (for the load log).
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Full SHA-256 of the model artifact the counts were captured on. The
    /// loader compares it with its own pin; it is not verified here.
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    /// Byte length of the base HFQ file the sidecar is bound to.
    pub fn base_size(&self) -> u64 {
        self.base_size
    }

    /// Lowercase hex sha256 of the base's `[0, data_offset)` header/index
    /// span the sidecar is bound to.
    pub fn base_index_sha256(&self) -> &str {
        &self.base_index_sha256
    }

    /// Check the sidecar is bound to exactly `base` (size, then index hash).
    pub fn check_base(&self, base: &HotsetBase) -> Result<(), String> {
        base.check_binding(self.base_size, &self.base_index_sha256)
    }

    /// Cheap pre-check used where a sidecar is only a candidate (default discovery): the file at
    /// `path` is a v1 sidecar bound to exactly `base`. Reads the file and its binding keys; the
    /// ranking itself is not validated here (the load's [`Self::load_bound`] does that).
    pub fn peek_bound(path: impl AsRef<Path>, base: &HotsetBase) -> Result<(), HotsetSkip> {
        #[derive(Deserialize)]
        struct Binding {
            version: u64,
            kind: String,
            base_size: u64,
            base_index_sha256: String,
        }
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| {
            HotsetSkip::InvalidSidecar(format!("cannot read {}: {e}", path.display()))
        })?;
        let binding: Binding = serde_json::from_slice(&bytes)
            .map_err(|e| HotsetSkip::InvalidSidecar(format!("not a v1 hotset JSON: {e}")))?;
        if binding.kind != SIDECAR_KIND || binding.version != SIDECAR_VERSION {
            return Err(HotsetSkip::InvalidSidecar(format!(
                "kind {:?} version {}, expected {SIDECAR_KIND:?} version {SIDECAR_VERSION}",
                binding.kind, binding.version
            )));
        }
        base.check_binding(binding.base_size, &binding.base_index_sha256)
            .map_err(HotsetSkip::BaseMismatch)
    }

    /// [`Self::load`] then [`Self::check_base`], as a skip.
    pub fn load_bound(
        path: impl AsRef<Path>,
        layers: usize,
        experts_per_layer: usize,
        top_k: usize,
        base: &HotsetBase,
    ) -> Result<Self, HotsetSkip> {
        let sidecar = Self::load(path, layers, experts_per_layer, top_k)
            .map_err(HotsetSkip::InvalidSidecar)?;
        sidecar.check_base(base).map_err(HotsetSkip::BaseMismatch)?;
        Ok(sidecar)
    }

    /// Full SHA-256 of the source counts the ranking derives from.
    pub fn counts_sha256(&self) -> &str {
        &self.counts_sha256
    }

    /// Human-readable expert format map the sidecar was built for.
    pub fn format_map(&self) -> &str {
        &self.format_map
    }

    /// Full SHA-256 of the format map.
    pub fn format_map_sha256(&self) -> &str {
        &self.format_map_sha256
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn experts_per_layer(&self) -> usize {
        self.experts_per_layer
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Tokens of the decode capture the counts come from.
    pub fn capture_tokens(&self) -> u64 {
        self.capture_tokens
    }

    /// Provenance text describing how `rank` was derived.
    pub fn ranking_rule(&self) -> &str {
        &self.ranking_rule
    }

    /// Provenance text describing the capture corpus.
    pub fn corpus(&self) -> &str {
        &self.corpus
    }
}

fn check_dim(name: &str, got: u64, expected: usize) -> Result<(), String> {
    if usize::try_from(got) == Ok(expected) {
        Ok(())
    } else {
        Err(format!("{name} is {got}, the model has {expected}"))
    }
}

/// The first 12 characters of a digest.
fn short(hex: &str) -> &str {
    hex.get(..12).unwrap_or(hex)
}

fn check_sha256(name: &str, value: &str) -> Result<(), String> {
    let full = value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if full {
        Ok(())
    } else {
        Err(format!(
            "{name} {value:?} is not a full lowercase 64-hex SHA-256"
        ))
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(out, "{byte:02x}").expect("write to String");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const LAYERS: usize = 2;
    const EXPERTS: usize = 3;
    const TOP_K: usize = 2;
    const SHA_A: &str = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";
    const BASE_SIZE: u64 = 125_144_775_192;
    const MODEL: &str = "/m/q.mq4-xts";

    /// Hex of the fixture base's index hash (`[0xab; 32]`).
    fn base_index_hex() -> String {
        "ab".repeat(32)
    }

    fn base() -> HotsetBase {
        HotsetBase::new(BASE_SIZE, [0xab; 32])
    }

    /// Counts with a tie (ids 1 and 4 both 7) and a zero.
    fn counts() -> Vec<u64> {
        vec![5, 7, 0, 9, 7, 1]
    }

    fn ranked(counts: &[u64]) -> Vec<u64> {
        let mut ids: Vec<u64> = (0..counts.len() as u64).collect();
        ids.sort_by_key(|&id| (std::cmp::Reverse(counts[id as usize]), id));
        ids
    }

    fn sidecar() -> Value {
        let counts = counts();
        json!({
            "version": 1,
            "kind": SIDECAR_KIND,
            "model_sha256": SHA_A,
            "counts_sha256": SHA_A,
            "format_map": "all-mq4-xts",
            "format_map_sha256": SHA_A,
            "base_size": BASE_SIZE,
            "base_index_sha256": base_index_hex(),
            "layers": LAYERS,
            "experts_per_layer": EXPERTS,
            "top_k": TOP_K,
            "capture_tokens": 100,
            "ranking_rule": "descending count, ties by global id ascending",
            "corpus": "fixture",
            "rank": ranked(&counts),
            "counts": counts,
        })
    }

    fn parse(value: &Value) -> Result<HotsetSidecar, String> {
        HotsetSidecar::from_bytes(value.to_string().as_bytes(), LAYERS, EXPERTS, TOP_K)
    }

    fn refused(value: &Value, needle: &str) {
        let err = parse(value).expect_err("sidecar must be refused");
        assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hipfire-hotset-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn good_env() -> HotsetEnv<'static> {
        HotsetEnv {
            gpu_arch: "gfx1201",
            unified_memory: false,
            expert_parallel: false,
            explicit_vram_layers: false,
            source: HotsetSource::Aggregate,
        }
    }

    fn decide(
        policy: &HotsetPolicy,
        exists: bool,
        env: &HotsetEnv<'_>,
    ) -> Result<PathBuf, HotsetSkip> {
        decide_environment(policy, Path::new(MODEL), exists, env)
    }

    #[test]
    fn placement_policy_unset_and_ranked_are_the_legacy_default() {
        assert_eq!(PlacementPolicy::default(), PlacementPolicy::Ranked);
        assert_eq!(PlacementPolicy::parse(None), Ok(PlacementPolicy::Ranked));
        assert_eq!(
            PlacementPolicy::parse(Some("ranked")),
            Ok(PlacementPolicy::Ranked)
        );
        assert_eq!(
            PlacementPolicy::parse(Some(" whole-first ")),
            Ok(PlacementPolicy::WholeFirst)
        );
        assert_eq!(PLACEMENT_POLICY_ENV, "HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY");
        assert_ne!(PLACEMENT_POLICY_ENV, HOTSET_ENV);
        for bad in [
            "",
            "  ",
            "whole",
            "Whole-First",
            "1",
            "0",
            "tiered",
            "/tmp/h.json",
        ] {
            let error = PlacementPolicy::parse(Some(bad)).unwrap_err();
            assert!(error.contains(PLACEMENT_POLICY_ENV), "{bad:?}: {error}");
            assert!(error.contains("whole-first"), "{bad:?}: {error}");
        }
    }

    #[test]
    fn whole_first_is_inert_unless_requested_on_gfx1201() {
        let band = false;
        // Unset/ranked: the default path on every arch, even a hotset sidecar.
        assert_eq!(
            decide_whole_first(PlacementPolicy::Ranked, &good_env(), band),
            WholeFirstDecision::Inactive
        );
        for arch in ["gfx1100", "gfx1030", "gfx942", "gfx90a"] {
            let env = HotsetEnv {
                gpu_arch: arch,
                ..good_env()
            };
            assert_eq!(
                decide_whole_first(PlacementPolicy::WholeFirst, &env, band),
                WholeFirstDecision::Inactive,
                "{arch}"
            );
        }
        assert_eq!(
            decide_whole_first(PlacementPolicy::WholeFirst, &good_env(), band),
            WholeFirstDecision::Active
        );
        // Non-aggregate layouts keep the same whole-layer placement.
        for source in [HotsetSource::SlabArena, HotsetSource::StockFp8] {
            let env = HotsetEnv {
                source,
                ..good_env()
            };
            assert_eq!(
                decide_whole_first(PlacementPolicy::WholeFirst, &env, band),
                WholeFirstDecision::Active
            );
        }
    }

    #[test]
    fn whole_first_is_ignored_where_placement_is_chosen_elsewhere() {
        let ignored = |env: &HotsetEnv<'_>, band: bool| {
            matches!(
                decide_whole_first(PlacementPolicy::WholeFirst, env, band),
                WholeFirstDecision::Ignored(_)
            )
        };
        let uma = HotsetEnv {
            unified_memory: true,
            ..good_env()
        };
        let ep = HotsetEnv {
            expert_parallel: true,
            ..good_env()
        };
        let explicit = HotsetEnv {
            explicit_vram_layers: true,
            ..good_env()
        };
        assert!(ignored(&uma, false));
        assert!(ignored(&ep, false));
        assert!(ignored(&explicit, false));
        assert!(ignored(&good_env(), true));
        // A non-gfx1201 card stays silent even when the layout would be ignored.
        let other = HotsetEnv {
            gpu_arch: "gfx1100",
            unified_memory: true,
            ..good_env()
        };
        assert_eq!(
            decide_whole_first(PlacementPolicy::WholeFirst, &other, true),
            WholeFirstDecision::Inactive
        );
    }

    #[test]
    fn valid_sidecar_loads_with_tie_ordered_rank() {
        let parsed = parse(&sidecar()).unwrap();
        assert_eq!(parsed.rank(), &[3, 1, 4, 0, 5, 2]);
        assert_eq!(parsed.counts(), counts().as_slice());
        assert_eq!(parsed.count(3), 9);
        assert_eq!(parsed.split_id(4), (1, 1));
        assert_eq!(parsed.model_sha256(), SHA_A);
        assert_eq!(parsed.capture_tokens(), 100);
        assert_eq!(parsed.base_size(), BASE_SIZE);
        assert_eq!(parsed.base_index_sha256(), base_index_hex());
        let bytes = sidecar().to_string();
        assert_eq!(parsed.sha256(), hex(&Sha256::digest(bytes.as_bytes())));
    }

    #[test]
    fn missing_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5]);
        refused(&value, "first missing id 2");
    }

    #[test]
    fn duplicate_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5, 5]);
        refused(&value, "duplicate");
    }

    #[test]
    fn out_of_range_id_is_refused() {
        let mut value = sidecar();
        value["rank"] = json!([3, 1, 4, 0, 5, 6]);
        refused(&value, "out of range");
    }

    #[test]
    fn wrong_counts_length_is_refused() {
        let mut value = sidecar();
        value["counts"] = json!([5, 7, 0, 9, 7]);
        refused(&value, "counts has 5 entries");
    }

    #[test]
    fn wrong_dimensions_are_refused() {
        for (key, bad) in [("layers", 3), ("experts_per_layer", 4), ("top_k", 3)] {
            let mut value = sidecar();
            value[key] = json!(bad);
            refused(&value, key);
        }
        // Caller-side dimensions that disagree with the file.
        let bytes = sidecar().to_string();
        assert!(HotsetSidecar::from_bytes(bytes.as_bytes(), 48, 512, 10).is_err());
    }

    #[test]
    fn wrong_kind_or_version_is_refused() {
        let mut value = sidecar();
        value["kind"] = json!("hipfire.qwen4.something_else");
        refused(&value, "kind");
        let mut value = sidecar();
        value["version"] = json!(2);
        refused(&value, "version");
    }

    #[test]
    fn rank_inconsistent_with_counts_is_refused() {
        // Higher count after lower count.
        let mut value = sidecar();
        value["rank"] = json!([1, 3, 4, 0, 5, 2]);
        refused(&value, "not (count descending, id ascending)");
        // Tie broken by descending id instead of ascending.
        let mut value = sidecar();
        value["rank"] = json!([3, 4, 1, 0, 5, 2]);
        refused(&value, "not (count descending, id ascending)");
    }

    #[test]
    fn hashes_must_be_full_lowercase_hex() {
        for key in [
            "model_sha256",
            "counts_sha256",
            "format_map_sha256",
            "base_index_sha256",
        ] {
            for bad in ["0d8b5dbc", &SHA_A.to_uppercase(), &SHA_A[..63]] {
                let mut value = sidecar();
                value[key] = json!(bad);
                refused(&value, key);
            }
        }
    }

    #[test]
    fn base_keys_are_required_and_validated() {
        for key in ["base_size", "base_index_sha256"] {
            let mut value = sidecar();
            value.as_object_mut().unwrap().remove(key);
            let err = parse(&value).expect_err("missing base key must be refused");
            assert!(err.contains(key), "{err:?} lacks {key:?}");
        }
        let mut value = sidecar();
        value["base_size"] = json!(0);
        refused(&value, "base_size");
        let mut value = sidecar();
        value["base_size"] = json!(-1);
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["base_size"] = json!("125144775192");
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["base_index_sha256"] = json!(base_index_hex().to_uppercase());
        refused(&value, "base_index_sha256");
    }

    #[test]
    fn missing_unknown_or_non_integer_fields_are_refused() {
        let mut value = sidecar();
        value.as_object_mut().unwrap().remove("capture_tokens");
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["surprise"] = json!(1);
        assert!(parse(&value).is_err());
        let mut value = sidecar();
        value["counts"] = json!([5, 7.5, 0, 9, 7, 1]);
        assert!(parse(&value).is_err());
        assert!(HotsetSidecar::from_bytes(b"not json", LAYERS, EXPERTS, TOP_K).is_err());
    }

    #[test]
    fn load_reads_file_and_names_missing_path() {
        let dir = temp_dir("load");
        let path = dir.join("hotset.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        let parsed = HotsetSidecar::load(&path, LAYERS, EXPERTS, TOP_K).unwrap();
        assert_eq!(parsed.rank().len(), LAYERS * EXPERTS);
        let missing = dir.join("absent.json");
        let err = HotsetSidecar::load(&missing, LAYERS, EXPERTS, TOP_K).unwrap_err();
        assert!(err.contains("absent.json"), "{err}");
    }

    #[test]
    fn binding_accepts_the_matching_base() {
        let parsed = parse(&sidecar()).unwrap();
        assert_eq!(parsed.check_base(&base()), Ok(()));
        let dir = temp_dir("bound");
        let path = dir.join("hotset.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        let bound = HotsetSidecar::load_bound(&path, LAYERS, EXPERTS, TOP_K, &base()).unwrap();
        assert_eq!(bound, parsed);
    }

    #[test]
    fn peek_bound_checks_only_the_binding_of_a_candidate_file() {
        let dir = temp_dir("peek");
        let path = dir.join("candidate.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        assert_eq!(HotsetSidecar::peek_bound(&path, &base()), Ok(()));
        // A different base (size, then index) is a base mismatch.
        assert!(matches!(
            HotsetSidecar::peek_bound(&path, &HotsetBase::new(BASE_SIZE + 1, [0xab; 32])),
            Err(HotsetSkip::BaseMismatch(_))
        ));
        assert!(matches!(
            HotsetSidecar::peek_bound(&path, &HotsetBase::new(BASE_SIZE, [0xcd; 32])),
            Err(HotsetSkip::BaseMismatch(_))
        ));
        // Not a v1 sidecar / unreadable: invalid, never a panic.
        let mut other = sidecar();
        other["kind"] = json!("something.else");
        std::fs::write(&path, other.to_string()).unwrap();
        assert!(matches!(
            HotsetSidecar::peek_bound(&path, &base()),
            Err(HotsetSkip::InvalidSidecar(_))
        ));
        std::fs::write(&path, b"not json").unwrap();
        assert!(matches!(
            HotsetSidecar::peek_bound(&path, &base()),
            Err(HotsetSkip::InvalidSidecar(_))
        ));
        assert!(matches!(
            HotsetSidecar::peek_bound(dir.join("absent"), &base()),
            Err(HotsetSkip::InvalidSidecar(_))
        ));
    }

    #[test]
    fn binding_size_mismatch_is_a_base_mismatch_skip() {
        let parsed = parse(&sidecar()).unwrap();
        let other = HotsetBase::new(BASE_SIZE + 1, [0xab; 32]);
        let err = parsed.check_base(&other).unwrap_err();
        assert!(
            err.contains("125144775193") && err.contains("125144775192"),
            "{err}"
        );
        let dir = temp_dir("size-mismatch");
        let path = dir.join("hotset.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        match HotsetSidecar::load_bound(&path, LAYERS, EXPERTS, TOP_K, &other) {
            Err(HotsetSkip::BaseMismatch(reason)) => assert_eq!(reason, err),
            other => panic!("expected BaseMismatch, got {other:?}"),
        }
    }

    #[test]
    fn binding_index_mismatch_is_a_base_mismatch_skip() {
        let parsed = parse(&sidecar()).unwrap();
        let other = HotsetBase::new(BASE_SIZE, [0xcd; 32]);
        let err = parsed.check_base(&other).unwrap_err();
        assert!(
            err.contains("cdcdcdcdcdcd") && err.contains("abababababab"),
            "{err}"
        );
        let dir = temp_dir("index-mismatch");
        let path = dir.join("hotset.json");
        std::fs::write(&path, sidecar().to_string()).unwrap();
        assert!(matches!(
            HotsetSidecar::load_bound(&path, LAYERS, EXPERTS, TOP_K, &other),
            Err(HotsetSkip::BaseMismatch(_))
        ));
    }

    #[test]
    fn load_bound_reports_invalid_sidecars_as_skips() {
        let dir = temp_dir("invalid");
        let absent = dir.join("absent.json");
        assert!(matches!(
            HotsetSidecar::load_bound(&absent, LAYERS, EXPERTS, TOP_K, &base()),
            Err(HotsetSkip::InvalidSidecar(_))
        ));
        let mut value = sidecar();
        value.as_object_mut().unwrap().remove("base_size");
        let path = dir.join("no-base.json");
        std::fs::write(&path, value.to_string()).unwrap();
        match HotsetSidecar::load_bound(&path, LAYERS, EXPERTS, TOP_K, &base()) {
            Err(HotsetSkip::InvalidSidecar(reason)) => {
                assert!(reason.contains("base_size"), "{reason}")
            }
            other => panic!("expected InvalidSidecar, got {other:?}"),
        }
    }

    /// A K0-shaped ranking (48 layers x 512 experts, top_k 10, as written by
    /// `scripts/k0rank.py`) loads bound from a file for the EP2 tier: the
    /// parser needs no EP-specific input, the rank is a permutation, and the
    /// static parity ownership (`expert % 2`) splits it into two disjoint
    /// halves of 12,288 units each. The default environment decision still
    /// skips expert-parallel placement; that gate belongs to the caller.
    #[test]
    fn k0_shaped_ranking_loads_bound_and_splits_by_parity_for_ep2() {
        const L: usize = 48;
        const E: usize = 512;
        let counts: Vec<u64> = (0..L * E)
            .map(|id| ((id * 2_654_435_761) % 997) as u64)
            .collect();
        let mut value = sidecar();
        let o = value.as_object_mut().unwrap();
        o.insert("layers".into(), json!(L));
        o.insert("experts_per_layer".into(), json!(E));
        o.insert("top_k".into(), json!(10));
        o.insert("rank".into(), json!(ranked(&counts)));
        o.insert("counts".into(), json!(counts));
        let dir = temp_dir("k0-shaped");
        let path = dir.join("k0.expert-hotset");
        std::fs::write(&path, value.to_string()).unwrap();

        let parsed = HotsetSidecar::load_bound(&path, L, E, 10, &base()).unwrap();
        assert_eq!(parsed.rank().len(), L * E);
        let mut halves = [Vec::new(), Vec::new()];
        for &id in parsed.rank() {
            let (_, expert) = parsed.split_id(id);
            halves[expert % 2].push(id);
        }
        assert_eq!((halves[0].len(), halves[1].len()), (L * E / 2, L * E / 2));
        for half in &halves {
            assert!(half
                .windows(2)
                .all(|w| parsed.count(w[0]) >= parsed.count(w[1])));
        }

        let mut env = good_env();
        env.expert_parallel = true;
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &env),
            Err(HotsetSkip::ExpertParallel)
        );
    }

    #[test]
    fn policy_parses_the_documented_values() {
        assert_eq!(HotsetPolicy::parse(None), Ok(HotsetPolicy::Default));
        assert_eq!(HotsetPolicy::parse(Some("0")), Ok(HotsetPolicy::Off));
        assert_eq!(
            HotsetPolicy::parse(Some("/tmp/hotset.json")),
            Ok(HotsetPolicy::Explicit(PathBuf::from("/tmp/hotset.json")))
        );
        // Empty, blank, relative and near-miss values are the only policy errors.
        for bad in [
            "",
            "  ",
            "hotset.json",
            "./hotset.json",
            "1",
            " 0",
            "off",
            "auto",
        ] {
            let err = HotsetPolicy::parse(Some(bad)).unwrap_err();
            assert!(err.contains(HOTSET_ENV), "{bad:?}: {err}");
        }
        let empty = HotsetPolicy::parse(Some("")).unwrap_err();
        assert!(empty.contains("empty"), "{empty}");
        let relative = HotsetPolicy::parse(Some("hotset.json")).unwrap_err();
        assert!(relative.contains("absolute"), "{relative}");
    }

    #[test]
    fn sidecar_paths_follow_the_policy() {
        let model = Path::new(MODEL);
        let default = PathBuf::from("/m/q.mq4-xts.expert-hotset");
        assert_eq!(HotsetPolicy::Default.sidecar_path(model), default);
        // `0` still names the default sidecar so a present one can be reported.
        assert_eq!(HotsetPolicy::Off.sidecar_path(model), default);
        assert_eq!(
            HotsetPolicy::Explicit(PathBuf::from("/x/h.json")).sidecar_path(model),
            PathBuf::from("/x/h.json")
        );
    }

    #[test]
    fn eligible_environment_returns_the_sidecar_path() {
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &good_env()),
            Ok(PathBuf::from("/m/q.mq4-xts.expert-hotset"))
        );
        assert_eq!(
            decide(
                &HotsetPolicy::Explicit(PathBuf::from("/x/h.json")),
                true,
                &good_env()
            ),
            Ok(PathBuf::from("/x/h.json"))
        );
    }

    #[test]
    fn every_environment_skip_reason_is_reported() {
        let policy = HotsetPolicy::Default;
        assert_eq!(
            decide(&HotsetPolicy::Off, true, &good_env()),
            Err(HotsetSkip::Disabled)
        );
        assert_eq!(
            decide(&policy, false, &good_env()),
            Err(HotsetSkip::Missing { explicit: false })
        );
        assert_eq!(
            decide(
                &HotsetPolicy::Explicit(PathBuf::from("/x/h.json")),
                false,
                &good_env()
            ),
            Err(HotsetSkip::Missing { explicit: true })
        );
        let mut env = good_env();
        env.gpu_arch = "gfx1100";
        assert_eq!(
            decide(&policy, true, &env),
            Err(HotsetSkip::NotGfx1201("gfx1100".to_owned()))
        );
        // A non-gfx1201 arch is opt-in: an explicit sidecar path applies there.
        let explicit = HotsetPolicy::Explicit(PathBuf::from("/x/h.json"));
        assert_eq!(
            decide(&explicit, true, &env),
            Ok(PathBuf::from("/x/h.json"))
        );
        env.gpu_arch = "gfx1151";
        assert_eq!(
            decide(&explicit, true, &env),
            Ok(PathBuf::from("/x/h.json"))
        );
        assert_eq!(
            decide(&policy, true, &env),
            Err(HotsetSkip::NotGfx1201("gfx1151".to_owned()))
        );
        let mut env = good_env();
        env.unified_memory = true;
        assert_eq!(decide(&policy, true, &env), Err(HotsetSkip::UnifiedMemory));
        let mut env = good_env();
        env.expert_parallel = true;
        assert_eq!(decide(&policy, true, &env), Err(HotsetSkip::ExpertParallel));
        for source in [
            HotsetSource::SlabArena,
            HotsetSource::StockFp8,
            HotsetSource::Nonaggregate("per-expert entry".to_owned()),
        ] {
            let mut env = good_env();
            env.source = source.clone();
            assert_eq!(decide(&policy, true, &env), Err(HotsetSkip::Source(source)));
        }
        let mut env = good_env();
        env.explicit_vram_layers = true;
        assert_eq!(
            decide(&policy, true, &env),
            Err(HotsetSkip::ExplicitVramLayers)
        );
    }

    #[test]
    fn hotset_arch_truth_table() {
        // (arch, explicit path, admitted): the default sibling sidecar runs on
        // gfx1201 only; an explicit path runs on any arch.
        for (arch, explicit, admitted) in [
            ("gfx1201", false, true),
            ("gfx1201", true, true),
            ("gfx1100", false, false),
            ("gfx1100", true, true),
            ("gfx1151", false, false),
            ("gfx1151", true, true),
            ("gfx1200", false, false),
            ("gfx1200", true, true),
            ("gfx942", false, false),
            ("gfx942", true, true),
            ("gfx12010", false, false),
        ] {
            assert_eq!(
                hotset_arch_admits(arch, explicit),
                admitted,
                "{arch} explicit={explicit}"
            );
            let mut env = good_env();
            env.gpu_arch = arch;
            let policy = if explicit {
                HotsetPolicy::Explicit(PathBuf::from("/x/h.json"))
            } else {
                HotsetPolicy::Default
            };
            assert_eq!(
                decide(&policy, true, &env).is_ok(),
                admitted,
                "decide_environment follows the same rule: {arch} explicit={explicit}"
            );
        }
    }

    #[test]
    fn environment_skip_precedence_is_disabled_missing_arch_then_placement() {
        let bad = HotsetEnv {
            gpu_arch: "gfx1100",
            unified_memory: true,
            expert_parallel: true,
            explicit_vram_layers: true,
            source: HotsetSource::StockFp8,
        };
        // `0` is reported whether or not a sidecar exists.
        assert_eq!(
            decide(&HotsetPolicy::Off, false, &bad),
            Err(HotsetSkip::Disabled)
        );
        assert_eq!(
            decide(&HotsetPolicy::Off, true, &bad),
            Err(HotsetSkip::Disabled)
        );
        assert_eq!(
            decide(&HotsetPolicy::Default, false, &bad),
            Err(HotsetSkip::Missing { explicit: false })
        );
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &bad),
            Err(HotsetSkip::NotGfx1201("gfx1100".to_owned()))
        );
        let mut env = bad.clone();
        env.gpu_arch = "gfx1201";
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &env),
            Err(HotsetSkip::UnifiedMemory)
        );
        env.unified_memory = false;
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &env),
            Err(HotsetSkip::ExpertParallel)
        );
        env.expert_parallel = false;
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &env),
            Err(HotsetSkip::Source(HotsetSource::StockFp8))
        );
        env.source = HotsetSource::Aggregate;
        assert_eq!(
            decide(&HotsetPolicy::Default, true, &env),
            Err(HotsetSkip::ExplicitVramLayers)
        );
    }

    #[test]
    fn preflight_stats_the_policy_path() {
        let dir = temp_dir("preflight");
        let model = dir.join("model.mq4-xts");
        assert_eq!(
            preflight(&HotsetPolicy::Default, &model, &good_env()),
            Err(HotsetSkip::Missing { explicit: false })
        );
        let sidecar_path = dir.join("model.mq4-xts.expert-hotset");
        std::fs::write(&sidecar_path, b"{}").unwrap();
        assert_eq!(
            preflight(&HotsetPolicy::Default, &model, &good_env()),
            Ok(sidecar_path.clone())
        );
        assert_eq!(
            preflight(&HotsetPolicy::Off, &model, &good_env()),
            Err(HotsetSkip::Disabled)
        );
        // `0` with no sidecar at all still reports (stat finds nothing).
        assert_eq!(
            preflight(&HotsetPolicy::Off, &dir.join("no-such-model"), &good_env()),
            Err(HotsetSkip::Disabled)
        );
        let explicit = HotsetPolicy::Explicit(dir.join("absent.json"));
        assert_eq!(
            preflight(&explicit, &model, &good_env()),
            Err(HotsetSkip::Missing { explicit: true })
        );
    }

    #[test]
    fn budget_and_plan_skips() {
        assert_eq!(require_budget(Ok(1 << 30)), Ok(1 << 30));
        assert!(matches!(
            require_budget(Ok(0)),
            Err(HotsetSkip::NoBudget(_))
        ));
        assert_eq!(
            require_budget(Err("short of VRAM".to_owned())),
            Err(HotsetSkip::NoBudget("short of VRAM".to_owned()))
        );
        assert_eq!(require_experts(7), Ok(7));
        assert_eq!(require_experts(0), Err(HotsetSkip::ZeroExperts));
    }

    #[test]
    fn fixed_budget_parses_checked_mib() {
        let auto = HotsetBudget::parse(None).unwrap();
        assert!(!auto.is_fixed());
        assert_eq!(auto.select(17).unwrap(), 17);
        assert!(auto.select(0).is_err());
        let fixed = HotsetBudget::parse(Some("1024")).unwrap();
        assert!(fixed.is_fixed());
        assert_eq!(fixed.select(2 << 30).unwrap(), 1 << 30);
        let max_mib = u64::MAX >> 20;
        let max_raw = max_mib.to_string();
        assert_eq!(
            HotsetBudget::parse(Some(&max_raw))
                .unwrap()
                .select(u64::MAX)
                .unwrap(),
            max_mib << 20,
        );
    }

    #[test]
    fn invalid_or_unavailable_fixed_budget_fails_closed() {
        for raw in ["", " ", "0", "-1", "+1", "1.5", "1MiB", "auto", " 1", "1 "] {
            let error = HotsetBudget::parse(Some(raw)).unwrap_err();
            assert!(error.contains(HOT_BUDGET_MIB_ENV), "{error}");
        }
        for raw in [(u64::MAX >> 20) + 1, u64::MAX] {
            assert!(HotsetBudget::parse(Some(&raw.to_string())).is_err());
        }
        assert!(HotsetBudget::parse(Some("18446744073709551616")).is_err());
        let fixed = HotsetBudget::parse(Some("2")).unwrap();
        assert_eq!(fixed.select(2 << 20).unwrap(), 2 << 20);
        for available in [0, (2 << 20) - 1] {
            let error = fixed.select(available).unwrap_err();
            assert!(error.contains(HOT_BUDGET_MIB_ENV), "{error}");
            assert!(error.contains("refusing to shrink"), "{error}");
        }
    }

    #[test]
    fn fixed_budget_keeps_ranked_prefix_across_different_available_budgets() {
        use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
        use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};
        use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, Placement};

        // Synthetic spans only: each expert occupies 512 KiB. A 2 MiB
        // ceiling admits three, not four, because the owner also has tail pad.
        let count = (EXPERTS_PER_LAYER as usize) * (TRUNK_LAYERS as usize);
        let slabs: Vec<_> = (0..count)
            .map(|id| ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: ExpertSlabSpan {
                    offset: id * (1 << 19),
                    len: 1 << 18,
                },
                down: ExpertSlabSpan {
                    offset: id * (1 << 19) + (1 << 18),
                    len: 1 << 18,
                },
            })
            .collect();
        let rank: Vec<u32> = (0..count as u32).rev().collect();
        let fixed = HotsetBudget::parse(Some("2")).unwrap();
        // Different auto headroom (e.g. AR versus MTP) and stage reserves
        // cannot change the selected ceiling or the logical partition.
        let ar =
            plan_expert_tiers_from_slabs(&slabs, &rank, fixed.select(8 << 20).unwrap()).unwrap();
        let mtp =
            plan_expert_tiers_from_slabs(&slabs, &rank, fixed.select(3 << 20).unwrap()).unwrap();
        assert_eq!(ar.device_experts(), 3);
        assert_eq!(ar.logical_digest(), mtp.logical_digest());
        assert_eq!(ar, mtp);
        for (index, &id) in rank.iter().enumerate() {
            assert_eq!(
                ar.entry(id as usize / 512, id as usize % 512).unwrap().tier,
                if index < 3 {
                    Placement::Device
                } else {
                    Placement::HostMapped
                },
            );
        }
        assert!(fixed.select((2 << 20) - 1).is_err());
    }

    #[test]
    fn log_lines_name_the_path_and_fall_back_to_whole_layer_placement() {
        let path = Path::new("/m/q.mq4-xts.expert-hotset");
        // An absent sidecar with HOTSET unset is the only silent skip.
        assert_eq!(HotsetSkip::Missing { explicit: false }.log_line(path), None);
        assert_eq!(
            HotsetSkip::Disabled.log_line(path).unwrap(),
            "  qwen4 expert hot set: /m/q.mq4-xts.expert-hotset present, not applied \
             (HIPFIRE_QWEN4_EXPERT_HOTSET=0); whole-layer placement"
        );
        assert_eq!(
            HotsetSkip::Missing { explicit: true }
                .log_line(path)
                .unwrap(),
            "  qwen4 expert hot set: /m/q.mq4-xts.expert-hotset present, not applied \
             (sidecar not found); whole-layer placement"
        );
        assert_eq!(
            HotsetSkip::NotGfx1201("gfx1100".to_owned())
                .log_line(path)
                .unwrap(),
            "  qwen4 expert hot set: /m/q.mq4-xts.expert-hotset present, not applied (gpu is \
             gfx1100; the default hotset runs on gfx1201 only (other archs need an explicit \
             HIPFIRE_QWEN4_EXPERT_HOTSET path)); whole-layer placement"
        );
        for skip in [
            HotsetSkip::UnifiedMemory,
            HotsetSkip::ExpertParallel,
            HotsetSkip::Source(HotsetSource::SlabArena),
            HotsetSkip::Source(HotsetSource::StockFp8),
            HotsetSkip::Source(HotsetSource::Nonaggregate("qt mismatch".to_owned())),
            HotsetSkip::ExplicitVramLayers,
            HotsetSkip::InvalidSidecar("bad".to_owned()),
            HotsetSkip::BaseMismatch("size".to_owned()),
            HotsetSkip::NoBudget("none".to_owned()),
            HotsetSkip::ZeroExperts,
        ] {
            let line = skip.log_line(path).expect("non-silent skip");
            assert!(
                line.starts_with(
                    "  qwen4 expert hot set: /m/q.mq4-xts.expert-hotset present, not applied ("
                ) && line.ends_with("); whole-layer placement"),
                "{line}"
            );
            assert!(!line.contains("HOTCACHE"), "{line}");
        }
    }
}
