//! Admission and bounded resources for the opt-in exact expert cache.
//!
//! The user surface is the typed `memory.moe_expert_cache` (`off|auto|on`, compat env
//! `HIPFIRE_MOE_EXPERT_CACHE`) and `memory.moe_expert_cache_tail_bytes` (compat env
//! `HIPFIRE_MOE_EXPERT_CACHE_TAIL_BYTES`), resolved by
//! [`hipfire_config::memory::moe_expert_cache_policy`]:
//!
//! * `off` (default): static placement; no cache.
//! * `auto`: no Qwen4 route has passed a measured qualification yet, so `auto` resolves to
//!   static placement and logs one line naming why (never an error).
//! * `on`: requests the exact cache; every unqualified route below is a named load error
//!   ([`admit`], [`ExpertCacheConfig::from_env`]).
//!
//! `memory.moe_expert_cache_tail_bytes` is an upper bound, in BYTES, on the evictable donor
//! bytes. [`ExpertCacheConfig::tail_cap_bytes`] is the sizing rule and the tier planner
//! ([`crate::expert_tier_plan::ExpertTierPlan::with_expert_cache_tail`]) rounds the cap DOWN to
//! whole expert spans: unset means [`DEFAULT_DONOR_PERCENT`] percent of the ranked device
//! owner's bytes, `0` is a valid static-only plan (no evictable slots, nothing attached).
//!
//! `HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT` stays a developer knob (target for paired fills in flight;
//! the shared core's per-layer pools each take `ceil(cap / pools)`, see `expert_cache`).
//!
//! [`record_catalog_identity`] builds the runtime [`ExpertCatalog`] from the admitted tier plan
//! and records its canonical identity digest on the plan before any weight is materialized.

use hipfire_config::memory::{MoeExpertCacheMode, MoeExpertCachePolicy};
use hipfire_runtime::expert_cache::catalog::{
    BundleSpec, GroupSpec, OwnerExtent, OwnerRegion, Projection, ProjectionPart, SegmentSpec,
};
use hipfire_runtime::expert_cache::{CatalogError, ExpertCatalog, ExpertKey, KernelAbi};
use rdna_compute::railgun_check::CheckMode;
use sha2::{Digest, Sha256};

use crate::expert_slab_table::ExpertSlabSpan;
use crate::expert_slabs::{
    expert_entry_name, ExpertFormat, ExpertProj, EXPERTS_PER_LAYER, TRUNK_LAYERS,
};
use crate::expert_tier_plan::{ExpertTierPlan, Placement};

pub const EXPERT_CACHE_INFLIGHT_ENV: &str = "HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT";

/// Percent of the ranked device owner's bytes that bounds the evictable tail when
/// `memory.moe_expert_cache_tail_bytes` is unset.
pub const DEFAULT_DONOR_PERCENT: u64 = 50;

/// Kernel ABI of the Qwen4 tier-stage expert layout (fused gate/up segment, then down).
pub const KERNEL_ABI: &str = "qwen4.flash-next.tier-expert.v1";

const SOURCE_FINGERPRINT_DOMAIN: &[u8] = b"hipfire.qwen4.expert-cache.source.v1";

const IMMUTABLE_WEIGHTS_CHECK_REASON: &str =
    "is incompatible with HIPFIRE_RAILGUN_CHECK=always: physical donor-slot payloads are mutable, but the check compares a fixed Weights digest baseline";

/// Read the same resolved process snapshot and parser as CheckState, including
/// TOML developer.railgun_check. Raw environment access would miss that setting.
pub fn railgun_check_always() -> Result<bool, String> {
    let resolved = hipfire_config::developer_var("HIPFIRE_RAILGUN_CHECK").ok();
    let mode = resolved
        .as_deref()
        .map(CheckMode::parse)
        .transpose()?
        .unwrap_or(CheckMode::Off);
    Ok(mode == CheckMode::Always)
}

fn admit_railgun_check(cache_enabled: bool, always: bool) -> Result<(), String> {
    if cache_enabled && always {
        Err(format!(
            "exact expert cache {IMMUTABLE_WEIGHTS_CHECK_REASON}"
        ))
    } else {
        Ok(())
    }
}

/// Resource policy only. Routing scores and selected expert ids never change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCacheConfig {
    /// `memory.moe_expert_cache_tail_bytes`: cap on evictable donor bytes (`None` = default
    /// sizing, `Some(0)` = static-only plan).
    pub tail_bytes_cap: Option<u64>,
    pub max_inflight: usize,
}

/// The immutable ranked load decision, used only to seed cache recency.
#[derive(Clone, Debug)]
pub struct ExpertCacheLoad {
    pub config: ExpertCacheConfig,
    pub warm_rank: Vec<u32>,
}

impl ExpertCacheConfig {
    /// The cache request of this process: `Some` only for `memory.moe_expert_cache=on`.
    /// `auto` resolves to static placement (one log line per process, not an error).
    pub fn from_env() -> Result<Option<Self>, String> {
        let policy =
            hipfire_config::memory::moe_expert_cache_policy().map_err(|error| error.to_string())?;
        let cache = Self::resolve(
            policy,
            hipfire_config::developer_var(EXPERT_CACHE_INFLIGHT_ENV)
                .ok()
                .as_deref(),
        )?;
        if policy.mode == MoeExpertCacheMode::Auto {
            static AUTO_NOTE: std::sync::Once = std::sync::Once::new();
            AUTO_NOTE.call_once(|| {
                eprintln!(
                    "  qwen4 expert cache: memory.moe_expert_cache=auto resolves to static \
                     placement: no Qwen4 route has passed its measured qualification yet \
                     (set memory.moe_expert_cache=on to request the experimental exact cache)"
                );
            });
        }
        // Both preload planning and product loading call this before weight
        // fulfillment. Leave railgun behavior unchanged when the cache is off.
        if cache.is_some() {
            admit_railgun_check(true, railgun_check_always()?)?;
        }
        Ok(cache)
    }

    /// Pure mapping of the typed policy (and the developer in-flight knob) to the Qwen4 request.
    /// Invalid explicit in-flight settings fail even while the cache is off or `auto`.
    pub fn resolve(
        policy: MoeExpertCachePolicy,
        inflight: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let max_inflight = {
            const MAX: usize = 4096;
            let value = inflight.map_or(Ok(64), |raw| {
                raw.parse::<usize>().map_err(|_| {
                    format!("{EXPERT_CACHE_INFLIGHT_ENV}: expected an integer in 1..={MAX}")
                })
            })?;
            if value == 0 || value > MAX {
                return Err(format!(
                    "{EXPERT_CACHE_INFLIGHT_ENV}: expected an integer in 1..={MAX}"
                ));
            }
            value
        };
        Ok(match policy.mode {
            MoeExpertCacheMode::Off | MoeExpertCacheMode::Auto => None,
            MoeExpertCacheMode::On => Some(Self {
                tail_bytes_cap: policy.tail_bytes_cap,
                max_inflight,
            }),
        })
    }

    /// `memory.moe_expert_cache_tail_bytes = 0`: admitted but no evictable slot, so nothing is
    /// attached and no cache resource is charged.
    pub fn is_static_only(self) -> bool {
        self.tail_bytes_cap == Some(0)
    }

    /// The cap, in bytes, on the evictable donor tail of a ranked device owner of
    /// `device_owner_bytes`: the configured cap, else [`DEFAULT_DONOR_PERCENT`] percent of the
    /// owner. The planner rounds it DOWN to whole expert spans.
    pub fn tail_cap_bytes(self, device_owner_bytes: u64) -> Result<u64, String> {
        match self.tail_bytes_cap {
            Some(cap) => Ok(cap),
            None => Ok(device_owner_bytes
                .checked_mul(DEFAULT_DONOR_PERCENT)
                .ok_or("expert-cache donor size overflows")?
                / 100),
        }
    }

    /// Device tables, retained route ids and a conservative extra queue/metadata
    /// reserve. The 32 MiB headroom is an admission margin, not a measured peak.
    /// Zero for a static-only plan (nothing is attached).
    pub fn device_reserve_bytes(
        self,
        layers: usize,
        experts: usize,
        topk: usize,
        max_rows: usize,
    ) -> Result<u64, String> {
        if self.is_static_only() {
            return Ok(0);
        }
        let tables = layers
            .checked_mul(experts)
            .and_then(|x| x.checked_mul(16))
            .ok_or("expert-cache table budget overflows")?;
        let routes = layers
            .checked_mul(topk)
            .and_then(|x| x.checked_mul(max_rows))
            .and_then(|x| x.checked_mul(4))
            .ok_or("expert-cache route budget overflows")?;
        u64::try_from(tables)
            .ok()
            .and_then(|x| x.checked_add(routes as u64))
            .and_then(|x| x.checked_add(32 << 20))
            .ok_or_else(|| "expert-cache device reserve overflows".into())
    }

    /// Conservative margin for both additional runtimes (decode and immutable
    /// short prefill), their CPU projection/resource and binding metadata, LRU
    /// state, upload mirrors and route landing storage. Current layouts motivate
    /// this allowance; it is not measured usage or a bound for all future
    /// layouts/allocators. The payload mirror is charged by the actual tier plan.
    /// Zero for a static-only plan (nothing is attached).
    pub fn host_reserve_bytes(self) -> u64 {
        if self.is_static_only() {
            0
        } else {
            128 << 20
        }
    }
}

pub struct ExpertCacheAdmission<'a> {
    pub arch: &'a str,
    pub flash_next: bool,
    pub single_device: bool,
    pub ranked_hotset: bool,
    pub uniform_mq4: bool,
    /// DFlash, n-gram or DSpark drafting. Native MTP is not here: its verify and repair
    /// windows are n-row cache windows over the same sealed QT44/QT53 route.
    pub other_speculation: bool,
    pub continuous_batch: bool,
    pub cpu_experts: bool,
    pub decode_share: bool,
    pub npu: bool,
    pub slowlink: bool,
    pub expert_keep: bool,
    pub railgun_check_always: bool,
}

pub fn admit(facts: &ExpertCacheAdmission<'_>) -> Result<(), String> {
    let reason = if facts.arch != "gfx1201" {
        Some("requires exact gfx1201")
    } else if !facts.flash_next {
        Some("requires Flash-Next")
    } else if !facts.single_device {
        Some("requires one GPU without PP/TP/EP")
    } else if !facts.ranked_hotset {
        Some("requires an applied ranked expert hotset")
    } else if !facts.uniform_mq4 {
        Some("requires uniform QT44/QT53 experts")
    } else if facts.other_speculation {
        Some("requires DFlash, n-gram and DSpark speculation off (native MTP is admitted)")
    } else if facts.continuous_batch {
        Some("does not support continuous batching/lanes")
    } else if facts.cpu_experts {
        Some("requires PCIe expert execution without CPU/D2 splice")
    } else if facts.decode_share {
        Some("requires decode-share off")
    } else if facts.npu {
        Some("does not support the NPU trunk")
    } else if facts.slowlink {
        Some("does not support the S1 slow-link path")
    } else if facts.expert_keep {
        Some("does not support expert-keep pruning")
    } else if facts.railgun_check_always {
        Some(IMMUTABLE_WEIGHTS_CHECK_REASON)
    } else {
        None
    };
    reason.map_or(Ok(()), |why| Err(format!("exact expert cache {why}")))
}

/// Hex of a digest, for the catalog's source fingerprint string.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to a String");
    }
    out
}

fn segment_of(
    key: ExpertKey,
    layer: usize,
    expert: usize,
    proj: ExpertProj,
    span: ExpertSlabSpan,
    format: ExpertFormat,
    fingerprint: &str,
) -> Result<SegmentSpec, CatalogError> {
    let (expect, qt, dtype) = match proj {
        ExpertProj::GateUp => (format.gate_up_bytes(), format.pair().0, format.dtypes().0),
        ExpertProj::Down => (format.down_bytes(), format.pair().1, format.dtypes().1),
    };
    let source_name = expert_entry_name(layer, expert, proj);
    if span.len != expect {
        return Err(CatalogError::LayoutMismatch {
            key,
            why: "a source span differs from its format's exact payload bytes",
        });
    }
    let [rows, k] = proj.shape().map(|dim| dim as usize);
    let len = span.len as u64;
    let rows_u64 = rows as u64;
    if len % rows_u64 != 0 {
        return Err(CatalogError::InvalidSegment {
            source: source_name,
            why: "the payload is not a whole number of rows",
        });
    }
    let parts = match proj {
        ExpertProj::GateUp => {
            if len % 2 != 0 || rows % 2 != 0 {
                return Err(CatalogError::InvalidSegment {
                    source: source_name,
                    why: "the fused gate/up payload does not split into equal halves",
                });
            }
            vec![
                ProjectionPart {
                    projection: Projection::Gate,
                    logical_shape: vec![rows / 2, k],
                    bytes: len / 2,
                },
                ProjectionPart {
                    projection: Projection::Up,
                    logical_shape: vec![rows / 2, k],
                    bytes: len / 2,
                },
            ]
        }
        ExpertProj::Down => vec![ProjectionPart {
            projection: Projection::Down,
            logical_shape: vec![rows, k],
            bytes: len,
        }],
    };
    Ok(SegmentSpec {
        source_name,
        fingerprint: fingerprint.to_owned(),
        extent: OwnerExtent {
            region: 0,
            offset: span.offset as u64,
            len,
        },
        alias: None,
        dtype,
        quant_tag: format!("qt{qt}"),
        basis: format!("{format:?}"),
        row_stride: len / rows_u64,
        alignment: ExpertTierPlan::PAYLOAD_ALIGN,
        parts,
        sidecar_names: Vec::new(),
    })
}

/// Build the runtime [`ExpertCatalog`] of one admitted Qwen4 tier plan: one group per trunk
/// layer, one bundle per routed expert (a fused gate/up segment and a down segment), the
/// artifact's payload extents as the single canonical owner region. `pinned` marks the experts
/// the plan keeps resident (device-placed and not a donor of the cache tail); everything else
/// is cacheable. MTP-head payloads (host windows) are not catalog members.
///
/// The source fingerprint and owner id derive only from canonical facts (the pinned artifact
/// digest when present, every expert's format and artifact span), never from placement, so a
/// different tail keeps the identity and any canonical change breaks it. The owner generation is
/// 0: the live owner generation is bound by the latch, not at admission.
pub fn build_expert_catalog(plan: &ExpertTierPlan) -> Result<ExpertCatalog, CatalogError> {
    let layers = TRUNK_LAYERS as usize;
    let experts = EXPERTS_PER_LAYER as usize;
    let missing = || CatalogError::PlanMismatch {
        why: "the tier plan lacks a trunk expert",
    };

    // Pass 1: canonical fingerprint and the artifact extent every segment must fit in.
    let mut hash = Sha256::new();
    hash.update(SOURCE_FINGERPRINT_DOMAIN);
    match plan.artifact_sha256() {
        Some(sha) => {
            hash.update([1u8]);
            hash.update(sha);
        }
        None => {
            hash.update([0u8]);
            hash.update([0u8; 32]);
        }
    }
    let mut region_len = 0u64;
    for layer in 0..layers {
        for expert in 0..experts {
            let entry = plan.entry(layer, expert).ok_or_else(missing)?;
            let (gate_up, down) = plan.source(layer, expert).ok_or_else(missing)?;
            let (gate_up_qt, down_qt) = entry.format.pair();
            hash.update([gate_up_qt, down_qt]);
            for span in [gate_up, down] {
                hash.update((span.offset as u64).to_le_bytes());
                hash.update((span.len as u64).to_le_bytes());
                let end = span.end().and_then(|end| u64::try_from(end).ok()).ok_or(
                    CatalogError::Overflow {
                        what: "expert source span end",
                    },
                )?;
                region_len = region_len.max(end);
            }
        }
    }
    let digest: [u8; 32] = hash.finalize().into();
    let fingerprint = hex(&digest);
    let owner_id = u64::from_le_bytes(digest[..8].try_into().expect("8 of 32 digest bytes"));
    let owner_generation = 0u64;
    let donors: &[u32] = plan
        .expert_cache_tail()
        .map_or(&[][..], |tail| tail.donors.as_slice());

    // Pass 2: groups and bundles.
    let mut groups = Vec::with_capacity(layers);
    for layer in 0..layers {
        let group_id = layer as u32;
        let mut bundles = Vec::with_capacity(experts);
        for expert in 0..experts {
            let entry = plan.entry(layer, expert).ok_or_else(missing)?;
            let (gate_up, down) = plan.source(layer, expert).ok_or_else(missing)?;
            let id = (layer * experts + expert) as u32;
            let key = ExpertKey {
                owner_id,
                owner_generation,
                group_id,
                expert_id: expert as u32,
            };
            bundles.push(BundleSpec {
                expert_id: expert as u32,
                pinned: entry.tier == Placement::Device && donors.binary_search(&id).is_err(),
                segments: vec![
                    segment_of(
                        key,
                        layer,
                        expert,
                        ExpertProj::GateUp,
                        gate_up,
                        entry.format,
                        &fingerprint,
                    )?,
                    segment_of(
                        key,
                        layer,
                        expert,
                        ExpertProj::Down,
                        down,
                        entry.format,
                        &fingerprint,
                    )?,
                ],
            });
        }
        groups.push(GroupSpec {
            group_id,
            label: format!("qwen4 trunk layer {layer}"),
            source_fingerprint: fingerprint.clone(),
            kernel_abi: KernelAbi(KERNEL_ABI.to_owned()),
            aux: Vec::new(),
            bundles,
        });
    }
    ExpertCatalog::new(
        owner_id,
        owner_generation,
        vec![OwnerRegion {
            name: "qwen4.artifact.expert_payloads".to_owned(),
            len: region_len,
        }],
        groups,
    )
}

/// Cache admission of an already laid-out tier plan, before any materialization: build the
/// catalog (refusing with its [`CatalogError`] as a named load error) and record its canonical
/// identity digest on the plan for the later latch. Table extents do not exist yet, so the
/// recorded digest covers the canonical catalog alone; the latch recomputes it with the tables.
pub fn record_catalog_identity(plan: ExpertTierPlan) -> Result<ExpertTierPlan, String> {
    let named = |error: CatalogError| format!("exact expert cache catalog: {error}");
    let catalog = build_expert_catalog(&plan).map_err(named)?;
    let identity = catalog.identity(&[]).map_err(named)?;
    Ok(plan.with_expert_cache_identity(*identity.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_slab_table::ExpertSlab;
    use crate::expert_tier_plan::plan_expert_tiers_from_slabs;
    use hipfire_config::memory::{resolve_moe_expert_cache, OffloadBudget};

    fn policy(mode: Option<&str>, tail: Option<&str>) -> MoeExpertCachePolicy {
        resolve_moe_expert_cache(mode, tail, OffloadBudget::Full).unwrap()
    }

    #[test]
    fn config_maps_off_auto_on_to_the_admission_request() {
        let resolve =
            |mode, tail, inflight| ExpertCacheConfig::resolve(policy(mode, tail), inflight);
        assert_eq!(resolve(None, None, None).unwrap(), None);
        assert_eq!(resolve(Some("off"), None, None).unwrap(), None);
        // `auto`: no qualified route, so static placement (with or without a cap).
        assert_eq!(resolve(Some("auto"), None, None).unwrap(), None);
        assert_eq!(resolve(Some("auto"), Some("4096"), None).unwrap(), None);
        let on = resolve(Some("on"), None, None).unwrap().unwrap();
        assert_eq!((on.tail_bytes_cap, on.max_inflight), (None, 64));
        assert!(!on.is_static_only());
        let capped = resolve(Some("on"), Some("123456"), Some("8"))
            .unwrap()
            .unwrap();
        assert_eq!(
            (capped.tail_bytes_cap, capped.max_inflight),
            (Some(123456), 8)
        );
        let zero = resolve(Some("on"), Some("0"), None).unwrap().unwrap();
        assert_eq!(zero.tail_bytes_cap, Some(0));
        assert!(zero.is_static_only());
        // A cap with the cache off is the config crate's named error, before any Qwen4 reader.
        assert!(resolve_moe_expert_cache(Some("off"), Some("5"), OffloadBudget::Full).is_err());
        // Invalid explicit in-flight settings fail even while the cache is off or auto.
        for mode in [None, Some("off"), Some("auto"), Some("on")] {
            for raw in ["0", "4097", "-1", "NaN", ""] {
                assert!(
                    resolve(mode, None, Some(raw)).is_err(),
                    "mode {mode:?} inflight {raw:?}"
                );
            }
        }
    }

    #[test]
    fn tail_cap_defaults_to_half_the_device_owner_and_zero_charges_nothing() {
        let on = ExpertCacheConfig {
            tail_bytes_cap: None,
            max_inflight: 64,
        };
        assert_eq!(on.tail_cap_bytes(1000).unwrap(), 500);
        assert_eq!(on.tail_cap_bytes(0).unwrap(), 0);
        assert!(on.tail_cap_bytes(u64::MAX).is_err());
        assert_eq!(on.host_reserve_bytes(), 128 << 20);
        assert!(on.device_reserve_bytes(usize::MAX, 512, 10, 64).is_err());
        assert_eq!(
            on.device_reserve_bytes(48, 512, 10, 64).unwrap(),
            (48 * 512 * 16) + (48 * 10 * 64 * 4) + (32 << 20),
        );
        // One row is the AR-only footprint.
        assert_eq!(
            on.device_reserve_bytes(48, 512, 10, 1).unwrap(),
            (48 * 512 * 16) + (48 * 10 * 4) + (32 << 20),
        );
        assert!(on.device_reserve_bytes(1, usize::MAX, 10, 64).is_err());
        assert!(on.device_reserve_bytes(1, 512, usize::MAX, 64).is_err());
        assert!(on.device_reserve_bytes(1, 512, 10, usize::MAX).is_err());
        // An explicit cap is used verbatim, even above the owner's bytes.
        let capped = ExpertCacheConfig {
            tail_bytes_cap: Some(7 << 40),
            ..on
        };
        assert_eq!(capped.tail_cap_bytes(1 << 30).unwrap(), 7 << 40);
        // Static-only attaches nothing, so it reserves nothing.
        let zero = ExpertCacheConfig {
            tail_bytes_cap: Some(0),
            ..on
        };
        assert_eq!(zero.tail_cap_bytes(1 << 30).unwrap(), 0);
        assert_eq!(zero.device_reserve_bytes(48, 512, 10, 64).unwrap(), 0);
        assert_eq!(zero.host_reserve_bytes(), 0);
    }

    /// Real Mq4Xts payload sizes laid out back to back; `shift` moves the whole artifact.
    fn real_slabs(shift: usize) -> Vec<ExpertSlab> {
        let format = ExpertFormat::Mq4Xts;
        let mut cursor = shift;
        (0..(TRUNK_LAYERS * EXPERTS_PER_LAYER) as usize)
            .map(|_| {
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: format.gate_up_bytes(),
                };
                cursor += format.gate_up_bytes();
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: format.down_bytes(),
                };
                cursor += format.down_bytes();
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    fn rank() -> Vec<u32> {
        (0..16).collect()
    }

    /// Padded device bytes of one expert (both projections 256-aligned).
    const SPAN: u64 = 2_611_200;

    #[test]
    fn tail_bytes_round_down_to_whole_expert_spans_and_clamp() {
        let rank = rank();
        let plain = plan_expert_tiers_from_slabs(&real_slabs(0), &rank, u64::MAX).unwrap();
        let device = plain.owner_bytes(Placement::Device);
        let on = ExpertCacheConfig {
            tail_bytes_cap: None,
            max_inflight: 64,
        };
        // Default: 50% of the device owner, rounded down to whole experts.
        let cap = on.tail_cap_bytes(device).unwrap();
        let (cached, tail) = plain.clone().with_expert_cache_tail(&rank, cap).unwrap();
        assert!(!tail.donors.is_empty());
        assert_eq!(tail.donor_bytes % SPAN, 0);
        assert!(tail.donor_bytes <= cap && tail.donor_bytes + SPAN > cap);
        assert_eq!(cached.expert_cache_tail(), Some(&tail));
        // An explicit cap rounds down as well (one byte short of three spans keeps two).
        let (_, tail) = plain
            .clone()
            .with_expert_cache_tail(&rank, 3 * SPAN - 1)
            .unwrap();
        assert_eq!(tail.donors.len(), 2);
        let (_, tail) = plain
            .clone()
            .with_expert_cache_tail(&rank, 3 * SPAN)
            .unwrap();
        assert_eq!(tail.donors.len(), 3);
        // Zero (and anything under one span) is a static-only plan, not an error.
        for cap in [0, SPAN - 1] {
            let (static_only, tail) = plain.clone().with_expert_cache_tail(&rank, cap).unwrap();
            assert_eq!(static_only, plain);
            assert!(tail.donors.is_empty() && static_only.expert_cache_tail().is_none());
        }
        // A cap above every donor byte clamps to all eligible experts.
        let (_, tail) = plain.with_expert_cache_tail(&rank, u64::MAX).unwrap();
        assert_eq!(tail.donors.len(), rank.len());
    }

    #[test]
    fn catalog_covers_every_expert_and_marks_only_the_core_pinned() {
        let rank = rank();
        let plain = plan_expert_tiers_from_slabs(&real_slabs(0), &rank, u64::MAX).unwrap();
        let (cached, tail) = plain
            .clone()
            .with_expert_cache_tail(&rank, 5 * SPAN)
            .unwrap();
        assert_eq!(tail.donors.len(), 5);
        let catalog = build_expert_catalog(&cached).unwrap();
        let facts = catalog.facts();
        assert_eq!(facts.groups, TRUNK_LAYERS as usize);
        assert_eq!(facts.bundles, (TRUNK_LAYERS * EXPERTS_PER_LAYER) as usize);
        // Device experts that are not donors are the pinned core; hosts and donors are cacheable.
        assert_eq!(facts.pinned, rank.len() - 5);
        assert_eq!(facts.pools, 1);
        // Without a tail every device expert is pinned.
        let facts = build_expert_catalog(&plain).unwrap().facts();
        assert_eq!(facts.pinned, rank.len());
    }

    #[test]
    fn recorded_identity_survives_a_remap_and_breaks_on_canonical_change() {
        let rank = rank();
        let plain = plan_expert_tiers_from_slabs(&real_slabs(0), &rank, u64::MAX).unwrap();
        assert_eq!(plain.expert_cache_identity(), None);
        let (cached, _) = plain
            .clone()
            .with_expert_cache_tail(&rank, 5 * SPAN)
            .unwrap();
        let a = record_catalog_identity(plain.clone()).unwrap();
        let b = record_catalog_identity(cached).unwrap();
        let identity = a
            .expert_cache_identity()
            .copied()
            .expect("identity recorded");
        // Placement (a different tail, pinned flags, device offsets) is not identity.
        assert_eq!(b.expert_cache_identity(), Some(&identity));
        assert_eq!(a.logical_digest(), plain.logical_digest());
        // A canonical change (the artifact's payload extents) is.
        let moved = plan_expert_tiers_from_slabs(&real_slabs(256), &rank, u64::MAX).unwrap();
        let moved = record_catalog_identity(moved).unwrap();
        assert_ne!(moved.expert_cache_identity(), Some(&identity));
        // So is the pinned artifact digest.
        let pinned = record_catalog_identity(plain.with_artifact_sha256([5u8; 32])).unwrap();
        assert_ne!(pinned.expert_cache_identity(), Some(&identity));
    }

    #[test]
    fn catalog_errors_are_named_refusals() {
        let rank = rank();
        // Two experts sharing artifact bytes overlap in the single owner region.
        let mut slabs = real_slabs(0);
        slabs[1] = slabs[0];
        let plan = plan_expert_tiers_from_slabs(&slabs, &rank, u64::MAX).unwrap();
        assert!(matches!(
            build_expert_catalog(&plan),
            Err(CatalogError::OwnerExtentOverlap { .. })
        ));
        let error = record_catalog_identity(plan).unwrap_err();
        assert!(error.starts_with("exact expert cache catalog: "), "{error}");
        assert!(error.contains("overlap"), "{error}");
        // A span that is not its format's exact payload size is a layout mismatch.
        let mut slabs = real_slabs(0);
        slabs[3].gate_up.len -= 256;
        slabs[3].down.offset -= 256;
        let plan = plan_expert_tiers_from_slabs(&slabs, &rank, u64::MAX).unwrap();
        assert!(matches!(
            build_expert_catalog(&plan),
            Err(CatalogError::LayoutMismatch { .. })
        ));
        let error = record_catalog_identity(plan).unwrap_err();
        assert!(error.contains("exact payload bytes"), "{error}");
    }

    fn facts() -> ExpertCacheAdmission<'static> {
        ExpertCacheAdmission {
            arch: "gfx1201",
            flash_next: true,
            single_device: true,
            ranked_hotset: true,
            uniform_mq4: true,
            other_speculation: false,
            continuous_batch: false,
            cpu_experts: false,
            decode_share: false,
            npu: false,
            slowlink: false,
            expert_keep: false,
            railgun_check_always: false,
        }
    }

    #[test]
    fn expert_cache_admission_refuses_each_unqualified_route() {
        assert!(admit(&facts()).is_ok());
        let mut f = facts();
        f.arch = "gfx1200";
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.single_device = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.flash_next = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.ranked_hotset = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.uniform_mq4 = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.other_speculation = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.continuous_batch = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.cpu_experts = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.decode_share = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.npu = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.slowlink = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.expert_keep = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.railgun_check_always = true;
        let error = admit(&f).unwrap_err();
        assert!(error.contains("physical donor-slot payloads"));
        assert!(error.contains("fixed Weights digest baseline"));
    }

    #[test]
    fn expert_cache_railgun_check_compatibility_truth_table() {
        // Cache-off preserves the checker's own behavior. Only Always has the
        // immutable physical-Weights baseline that conflicts with donor reuse.
        for (cache_enabled, setting, allowed) in [
            (false, "off", true),
            (false, "sample(1)", true),
            (false, "always", true),
            (true, "off", true),
            (true, "sample(1)", true),
            (true, "sample(64)", true),
            (true, "always", false),
            (true, " always ", false),
        ] {
            let mode = CheckMode::parse(setting).unwrap();
            assert_eq!(
                admit_railgun_check(cache_enabled, mode == CheckMode::Always).is_ok(),
                allowed,
                "cache={cache_enabled} check={setting:?}",
            );
        }
    }
}
