// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The runtime expert catalog (slice S2a, CPU only): what a cache may copy, where its canonical
//! bytes live, which slots may host it, and a stable identity of all of that.
//!
//! # Concepts (PLAN §2.2)
//!
//! * [`ExpertCatalog`]: the load-time checked description of every expert BUNDLE
//!   ([`ExpertKey`]) of one owner generation: its encoded source segments (a
//!   [`SegmentSpec`] each: exact owner extent, logical shapes, projection layout, quant and
//!   basis identities, optional read-only alias), the group-shared immutable auxiliary bindings
//!   ([`AuxSpec`], the sidecars) and the planned residency of every bundle (`pinned` = resident
//!   by the admitted plan, otherwise cacheable). It derives from the existing sealed metadata
//!   ([`GroupSpec::from_execution_plan`] over `sealed_moe::ExpertExecutionPlan`) or is built
//!   directly from plain descriptors ([`ExpertCatalog::new`]); it imports no architecture crate.
//! * [`LayoutClass`]: the exact physical layout of one bundle in a slot: the kernel ABI, every
//!   segment's slot offset, length, alignment, codec / basis / row stride / projection parts,
//!   the auxiliary formats it reads and which auxiliary each segment references. A slot hosts
//!   only a bundle of an EQUAL class; two bundles that differ in any of those fields (padding
//!   included: offsets and lengths are exact) are different classes and never share a pool.
//!   A mixed-format layer therefore yields one pool per class ([`ExpertCatalog::pool_of`]),
//!   and bundles of identical layout share a pool across groups.
//! * [`ExpertCatalog::identity`] (A2.3): a canonical digest over the catalog (owner id and
//!   generation, owner regions, per group the source fingerprint, kernel ABI, auxiliary
//!   bindings and per bundle the segments, extents, aliases and class) plus the caller's table
//!   addresses and extents. It deliberately contains NO per-expert slot pointer and not the
//!   planned residency, so a remap (a table's contents change, a bundle moves between slots,
//!   a pinned bundle becomes cacheable) keeps it, while any canonical change (source, owner
//!   generation, layout, extent, table address or size) breaks it.
//!
//! # Derivation (file:line of the existing metadata, at this worktree's base)
//!
//! * `sealed_moe.rs:166` `ExpertExecutionPlan` (records, source fingerprint, bases, dimensions),
//!   `:128` `ExpertRecord`, `:103` `ExpertProjectionBinding` (per projection source name,
//!   fingerprint, shape, dtype, bytes, row stride, alignment, quant tag, basis, sidecar names,
//!   alias owner and offset); `:37` `ExpertSourceMetadata` is its input.
//! * `weight_manifest.rs:865` `ExpertProjectionResources` (the exact per-expert byte counts the
//!   plan checked each binding against).
//! * `hipfire-arch-qwen4/src/expert_ep_units.rs:167` `EpUnitPlan` (+ `:58` `EpUnitGeometry`): the
//!   D2 per-expert unit plan; [`UnitPlanDescriptor`] is its plain mirror (geometry bytes and the
//!   `[rank][layer][expert]` residency bitmap) and supplies the `pinned` flags.
//!
//! # Allocation
//!
//! Construction and [`ExpertCatalog::identity`] are load-time and allocate freely. Every lookup
//! used on a hot path ([`ExpertCatalog::bundle`], [`ExpertCatalog::pool_of`],
//! [`ExpertCatalog::check_slot_class`], [`ExpertCatalog::copy_extents`]) is a binary search over
//! immutable arrays and allocates nothing.

use std::collections::HashMap;
use std::fmt;

use rdna_compute::DType;
use sha2::{Digest, Sha256};

use super::{ExpertKey, PoolId};
use crate::sealed_moe::{ExpertExecutionPlan, ExpertProjectionBinding};

/// The identity of the kernel ABI a layout is consumed by (kernel family, entry points and the
/// argument / table conventions). Equal layouts under different ABIs are different classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelAbi(pub String);

/// The logical projection a part of a segment holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Projection {
    Gate,
    Up,
    Down,
}

impl Projection {
    fn tag(self) -> u8 {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }
}

/// One canonical backing allocation (a pinned-host / GTT / file-mapped owner): index = region id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerRegion {
    pub name: String,
    pub len: u64,
}

/// A byte range of one [`OwnerRegion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerExtent {
    pub region: u32,
    pub offset: u64,
    pub len: u64,
}

impl OwnerExtent {
    fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }
}

/// A segment (or auxiliary binding) that is a read-only VIEW of another source's bytes.
/// Alias extents are range checked but exempt from the overlap check (they overlap their owner
/// by definition).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentAlias {
    pub owner_source: String,
    pub byte_offset: u64,
}

/// One logical projection inside a segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionPart {
    pub projection: Projection,
    pub logical_shape: Vec<usize>,
    /// Exact encoded bytes of this projection.
    pub bytes: u64,
}

/// One encoded source segment of a bundle: a contiguous byte range of the canonical owner that
/// the cache copies verbatim into the slot.
///
/// `parts` is the projection layout of the segment in byte order and must cover `extent.len`
/// exactly. The allowed groupings are `[Gate]`, `[Up]`, `[Down]` and the fused `[Gate, Up]`
/// (gate slice first, as the sealed plan's fused carriers store it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentSpec {
    pub source_name: String,
    pub fingerprint: String,
    pub extent: OwnerExtent,
    pub alias: Option<SegmentAlias>,
    pub dtype: DType,
    pub quant_tag: String,
    pub basis: String,
    pub row_stride: u64,
    pub alignment: u64,
    pub parts: Vec<ProjectionPart>,
    /// Names of the group auxiliary bindings (sidecars) this segment's kernels read.
    pub sidecar_names: Vec<String>,
}

/// A group-shared immutable auxiliary binding (a sidecar: rotation signs, a codebook, ...). It is
/// bound once per group and never copied into a slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuxSpec {
    pub source_name: String,
    pub fingerprint: String,
    pub extent: OwnerExtent,
    pub alias: Option<SegmentAlias>,
    pub dtype: DType,
    pub quant_tag: String,
    pub basis: String,
    pub logical_shape: Vec<usize>,
    pub row_stride: u64,
    pub alignment: u64,
}

/// One logical expert bundle of a group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleSpec {
    pub expert_id: u32,
    /// Resident by the admitted plan (never evictable); `false` = a cacheable bundle.
    pub pinned: bool,
    /// The slot layout order: segments are laid out in this order.
    pub segments: Vec<SegmentSpec>,
}

/// One group (trunk layer, MTP head, artifact namespace, rank-local shard).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSpec {
    pub group_id: u32,
    /// Human label (diagnostics only; not part of the identity).
    pub label: String,
    /// The source generation every segment and auxiliary binding of the group came from.
    pub source_fingerprint: String,
    pub kernel_abi: KernelAbi,
    pub aux: Vec<AuxSpec>,
    pub bundles: Vec<BundleSpec>,
}

/// Where a manifest source starts in the canonical owner (supplied by the loader, which owns
/// the allocations; the sealed plan carries sizes but no addresses).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcePlacement {
    pub source_name: String,
    pub region: u32,
    pub offset: u64,
}

impl GroupSpec {
    /// Build one group from a sealed execution plan for one logical rank.
    ///
    /// * The bundles are the experts `plan.rank_ownership()[logical_rank]` owns (global expert
    ///   ids; for Single / TP plans every expert).
    /// * Fused gate/up carriers become one `[Gate, Up]` segment, separate ones two segments.
    /// * Packed sources (every expert shares the projection source names, or there is one
    ///   expert) place expert `e` at `placement + alias offset + e x per-expert bytes`; per-expert
    ///   sources at `placement + alias offset`.
    /// * Sidecars are the group's auxiliary bindings (whole source bytes, `projection_bytes`).
    ///   A projection reads its own `sidecar_source_names`, or the layout's sidecars when it
    ///   has none (the rule the dispatch adapter applies).
    /// * An alias source resolves through its owner's placement.
    /// * `pinned[e]` (indexed by global expert id, e.g. [`UnitPlanDescriptor::layer_resident`])
    ///   marks planned-resident experts; `None` = every bundle cacheable.
    pub fn from_execution_plan(
        plan: &ExpertExecutionPlan,
        logical_rank: usize,
        group_id: u32,
        kernel_abi: KernelAbi,
        placements: &[SourcePlacement],
        pinned: Option<&[bool]>,
    ) -> Result<Self, CatalogError> {
        let mismatch = |why: &'static str| CatalogError::PlanMismatch { why };
        let ownership = plan
            .rank_ownership()
            .get(logical_rank)
            .ok_or_else(|| mismatch("logical rank is outside the plan's rank ownership"))?;
        if let Some(flags) = pinned {
            if flags.len() != plan.n_experts() {
                return Err(mismatch("pinned flags do not cover the plan's experts"));
            }
        }
        let records = plan.experts();
        let same = |a: &ExpertProjectionBinding, b: &ExpertProjectionBinding| {
            a.source_name == b.source_name
        };
        let packed = match records.first() {
            None => true,
            Some(first) => records.iter().all(|record| {
                same(&record.gate, &first.gate)
                    && same(&record.up, &first.up)
                    && same(&record.down, &first.down)
            }),
        };

        // The group's auxiliary bindings: the layout sidecars of the first owned expert, every
        // other owned expert must list the same ones.
        let mut aux = Vec::new();
        let mut aux_names: Option<Vec<&str>> = None;
        let mut bundles = Vec::with_capacity(ownership.global_expert_ids.len());
        for &expert in &ownership.global_expert_ids {
            let record = records
                .get(expert)
                .ok_or_else(|| mismatch("rank ownership names an expert the plan lacks"))?;
            let names: Vec<&str> = record
                .sidecars
                .iter()
                .map(|sidecar| sidecar.source_name.as_str())
                .collect();
            let uniform = aux_names.as_ref().map(|first| *first == names);
            match uniform {
                None => {
                    for sidecar in &record.sidecars {
                        let (extent, alias) = resolve_extent(
                            sidecar,
                            placements,
                            0,
                            to_u64(sidecar.projection_bytes, "sidecar bytes")?,
                        )?;
                        aux.push(AuxSpec {
                            source_name: sidecar.source_name.clone(),
                            fingerprint: sidecar.fingerprint.clone(),
                            extent,
                            alias,
                            dtype: sidecar.dtype,
                            quant_tag: sidecar.quant_tag.clone(),
                            basis: sidecar.basis.clone(),
                            logical_shape: sidecar.logical_shape.clone(),
                            row_stride: to_u64(sidecar.row_stride, "sidecar row stride")?,
                            alignment: to_u64(sidecar.alignment, "sidecar alignment")?,
                        });
                    }
                    aux_names = Some(names);
                }
                Some(true) => {}
                Some(false) => {
                    return Err(mismatch("owned experts list different sidecars"));
                }
            }

            let sidecars_of = |binding: &ExpertProjectionBinding| -> Vec<String> {
                if binding.sidecar_source_names.is_empty() {
                    record
                        .sidecars
                        .iter()
                        .map(|sidecar| sidecar.source_name.clone())
                        .collect()
                } else {
                    binding.sidecar_source_names.clone()
                }
            };
            let within = |binding: &ExpertProjectionBinding| -> Result<u64, CatalogError> {
                if packed {
                    to_u64(binding.encoded_bytes, "expert slice bytes")?
                        .checked_mul(expert as u64)
                        .ok_or(CatalogError::Overflow {
                            what: "packed expert offset",
                        })
                } else {
                    Ok(0)
                }
            };
            let part = |projection: Projection, binding: &ExpertProjectionBinding| {
                to_u64(binding.projection_bytes, "projection bytes").map(|bytes| ProjectionPart {
                    projection,
                    logical_shape: binding.logical_shape.clone(),
                    bytes,
                })
            };
            let segment = |binding: &ExpertProjectionBinding,
                           parts: Vec<ProjectionPart>|
             -> Result<SegmentSpec, CatalogError> {
                let len = to_u64(binding.encoded_bytes, "segment bytes")?;
                let (extent, alias) = resolve_extent(binding, placements, within(binding)?, len)?;
                Ok(SegmentSpec {
                    source_name: binding.source_name.clone(),
                    fingerprint: binding.fingerprint.clone(),
                    extent,
                    alias,
                    dtype: binding.dtype,
                    quant_tag: binding.quant_tag.clone(),
                    basis: binding.basis.clone(),
                    row_stride: to_u64(binding.row_stride, "row stride")?,
                    alignment: to_u64(binding.alignment, "alignment")?,
                    parts,
                    sidecar_names: sidecars_of(binding),
                })
            };

            let mut segments = Vec::with_capacity(3);
            if record.gate.source_name == record.up.source_name {
                let format_matches = record.gate.dtype == record.up.dtype
                    && record.gate.quant_tag == record.up.quant_tag
                    && record.gate.basis == record.up.basis
                    && record.gate.fingerprint == record.up.fingerprint
                    && record.gate.row_stride == record.up.row_stride
                    && record.gate.alignment == record.up.alignment
                    && record.gate.encoded_bytes == record.up.encoded_bytes;
                if !format_matches {
                    return Err(mismatch("a fused gate/up carrier's halves disagree"));
                }
                segments.push(segment(
                    &record.gate,
                    vec![
                        part(Projection::Gate, &record.gate)?,
                        part(Projection::Up, &record.up)?,
                    ],
                )?);
            } else {
                segments.push(segment(
                    &record.gate,
                    vec![part(Projection::Gate, &record.gate)?],
                )?);
                segments.push(segment(
                    &record.up,
                    vec![part(Projection::Up, &record.up)?],
                )?);
            }
            segments.push(segment(
                &record.down,
                vec![part(Projection::Down, &record.down)?],
            )?);

            bundles.push(BundleSpec {
                expert_id: u32::try_from(expert).map_err(|_| CatalogError::Overflow {
                    what: "expert id exceeds u32",
                })?,
                pinned: pinned.map(|flags| flags[expert]).unwrap_or(false),
                segments,
            });
        }
        Ok(Self {
            group_id,
            label: format!("{} layer {:?}", plan.group(), plan.layer()),
            source_fingerprint: plan.source_fingerprint().to_string(),
            kernel_abi,
            aux,
            bundles,
        })
    }
}

fn to_u64(value: usize, what: &'static str) -> Result<u64, CatalogError> {
    u64::try_from(value).map_err(|_| CatalogError::Overflow { what })
}

/// Resolve a binding's canonical extent: its alias owner's placement (or its own) plus the alias
/// byte offset plus `within`.
fn resolve_extent(
    binding: &ExpertProjectionBinding,
    placements: &[SourcePlacement],
    within: u64,
    len: u64,
) -> Result<(OwnerExtent, Option<SegmentAlias>), CatalogError> {
    let base_name = binding
        .alias_owner
        .as_deref()
        .unwrap_or(binding.source_name.as_str());
    let placement = placements
        .iter()
        .find(|placement| placement.source_name == base_name)
        .ok_or_else(|| CatalogError::MissingPlacement {
            source: base_name.to_string(),
        })?;
    let alias_offset = match binding.alias_byte_offset {
        Some(offset) => to_u64(offset, "alias byte offset")?,
        None => 0,
    };
    let offset = placement
        .offset
        .checked_add(alias_offset)
        .and_then(|offset| offset.checked_add(within))
        .ok_or(CatalogError::Overflow {
            what: "source extent offset",
        })?;
    let alias = binding.alias_owner.as_ref().map(|owner| SegmentAlias {
        owner_source: owner.clone(),
        byte_offset: alias_offset,
    });
    Ok((
        OwnerExtent {
            region: placement.region,
            offset,
            len,
        },
        alias,
    ))
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Every refusal of the catalog. Construction errors name the offending group / key / source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogError {
    /// The catalog describes no group.
    NoGroups,
    /// A group has no expert bundle.
    ZeroExperts {
        group_id: u32,
    },
    DuplicateGroup {
        group_id: u32,
    },
    /// Two bundles of one group claim the same expert id.
    DuplicateKey {
        key: ExpertKey,
    },
    /// An owner region is unusable (zero length).
    InvalidRegion {
        region: u32,
        why: &'static str,
    },
    UnknownRegion {
        source: String,
        region: u32,
    },
    /// An extent reaches past its owner region.
    OwnerExtentOutOfRange {
        source: String,
        region: u32,
        offset: u64,
        len: u64,
        region_len: u64,
    },
    /// Two non-alias extents share bytes of one region.
    OwnerExtentOverlap {
        first: String,
        second: String,
        region: u32,
        offset: u64,
    },
    /// A segment reads a sidecar the group does not bind.
    MissingSidecar {
        key: ExpertKey,
        sidecar: String,
    },
    /// A sidecar / auxiliary binding is malformed (empty or duplicate name, not unique, ...).
    InvalidSidecar {
        group_id: u32,
        sidecar: String,
        why: &'static str,
    },
    /// A segment or auxiliary binding of another source generation than its group.
    FingerprintMismatch {
        group_id: u32,
        source: String,
    },
    /// A segment (or auxiliary binding) is malformed on its own.
    InvalidSegment {
        source: String,
        why: &'static str,
    },
    /// The bundle's projections, a segment's parts or a slot / pool class do not fit together.
    LayoutMismatch {
        key: ExpertKey,
        why: &'static str,
    },
    /// A checked size computation overflowed.
    Overflow {
        what: &'static str,
    },
    UnknownKey {
        key: ExpertKey,
    },
    UnknownPool(PoolId),
    UnknownGroup {
        group_id: u32,
    },
    /// A sealed plan names a source the loader gave no placement for.
    MissingPlacement {
        source: String,
    },
    /// The sealed plan (or its pinned flags) is internally inconsistent.
    PlanMismatch {
        why: &'static str,
    },
    /// A caller-supplied table extent is empty or overflows the address space.
    TableExtentInvalid {
        group_id: u32,
        table: u32,
        address: u64,
        bytes: u64,
    },
    DuplicateTable {
        group_id: u32,
        table: u32,
    },
    TableExtentOverlap {
        first: (u32, u32),
        second: (u32, u32),
    },
    /// A unit plan descriptor's dimensions are inconsistent.
    UnitPlanInconsistent {
        why: &'static str,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGroups => write!(f, "the expert catalog describes no group"),
            Self::ZeroExperts { group_id } => write!(f, "group {group_id} has no expert bundle"),
            Self::DuplicateGroup { group_id } => write!(f, "group {group_id} is declared twice"),
            Self::DuplicateKey { key } => write!(
                f,
                "duplicate expert key (group {}, expert {})",
                key.group_id, key.expert_id
            ),
            Self::InvalidRegion { region, why } => write!(f, "owner region {region}: {why}"),
            Self::UnknownRegion { source, region } => {
                write!(f, "source '{source}' names unknown owner region {region}")
            }
            Self::OwnerExtentOutOfRange {
                source,
                region,
                offset,
                len,
                region_len,
            } => write!(
                f,
                "source '{source}' extent [{offset}, +{len}) exceeds owner region {region} \
                 ({region_len} bytes)"
            ),
            Self::OwnerExtentOverlap {
                first,
                second,
                region,
                offset,
            } => write!(
                f,
                "sources '{first}' and '{second}' overlap in owner region {region} at {offset}"
            ),
            Self::MissingSidecar { key, sidecar } => write!(
                f,
                "expert (group {}, expert {}) reads sidecar '{sidecar}' the group does not bind",
                key.group_id, key.expert_id
            ),
            Self::InvalidSidecar {
                group_id,
                sidecar,
                why,
            } => write!(f, "group {group_id} sidecar '{sidecar}': {why}"),
            Self::FingerprintMismatch { group_id, source } => write!(
                f,
                "group {group_id}: source '{source}' is from another source generation"
            ),
            Self::InvalidSegment { source, why } => write!(f, "source '{source}': {why}"),
            Self::LayoutMismatch { key, why } => write!(
                f,
                "expert (group {}, expert {}) layout: {why}",
                key.group_id, key.expert_id
            ),
            Self::Overflow { what } => write!(f, "{what} overflows"),
            Self::UnknownKey { key } => write!(
                f,
                "no expert (owner {} gen {}, group {}, expert {}) in the catalog",
                key.owner_id, key.owner_generation, key.group_id, key.expert_id
            ),
            Self::UnknownPool(pool) => write!(f, "the catalog has no pool {}", pool.0),
            Self::UnknownGroup { group_id } => write!(f, "the catalog has no group {group_id}"),
            Self::MissingPlacement { source } => {
                write!(f, "no owner placement was supplied for source '{source}'")
            }
            Self::PlanMismatch { why } => write!(f, "sealed plan: {why}"),
            Self::TableExtentInvalid {
                group_id,
                table,
                address,
                bytes,
            } => write!(
                f,
                "group {group_id} table {table}: invalid extent {address:#x} + {bytes}"
            ),
            Self::DuplicateTable { group_id, table } => {
                write!(f, "group {group_id} table {table} is supplied twice")
            }
            Self::TableExtentOverlap { first, second } => write!(
                f,
                "tables (group {}, {}) and (group {}, {}) overlap",
                first.0, first.1, second.0, second.1
            ),
            Self::UnitPlanInconsistent { why } => write!(f, "unit plan: {why}"),
        }
    }
}

impl std::error::Error for CatalogError {}

// ---------------------------------------------------------------------------------------------
// Layout class
// ---------------------------------------------------------------------------------------------

/// One projection part as the class records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassPart {
    pub projection: Projection,
    pub logical_shape: Vec<usize>,
    pub bytes: u64,
}

/// One segment of a slot, exactly as the kernels see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassSegment {
    /// Byte offset of the segment inside the slot.
    pub slot_offset: u64,
    pub len: u64,
    pub dtype: DType,
    pub quant_tag: String,
    pub basis: String,
    pub row_stride: u64,
    pub alignment: u64,
    pub parts: Vec<ClassPart>,
    /// Indices into [`LayoutClass::aux`] this segment's kernels read.
    pub sidecars: Vec<u32>,
}

/// The format of one auxiliary binding a class reads (identity names are not part of the class:
/// the table binds the group's own aux, the class fixes only what the kernels may assume).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassAux {
    pub dtype: DType,
    pub quant_tag: String,
    pub basis: String,
    pub logical_shape: Vec<usize>,
    pub row_stride: u64,
    pub alignment: u64,
    pub len: u64,
}

/// The exact physical layout class of a bundle. Equal classes are slot-compatible; nothing else
/// is (no padding-based reinterpretation: offsets and lengths are part of the class).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutClass {
    kernel_abi: KernelAbi,
    segments: Vec<ClassSegment>,
    aux: Vec<ClassAux>,
    alignment: u64,
    slot_bytes: u64,
    slot_stride: u64,
    digest: [u8; 32],
}

impl LayoutClass {
    pub fn kernel_abi(&self) -> &KernelAbi {
        &self.kernel_abi
    }

    pub fn segments(&self) -> &[ClassSegment] {
        &self.segments
    }

    pub fn aux(&self) -> &[ClassAux] {
        &self.aux
    }

    /// Alignment of a slot start (the largest segment alignment).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Bytes from the slot start to the end of the last segment.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Distance between two slots of this class in a block: `slot_bytes` rounded up to
    /// `alignment`.
    pub fn slot_stride(&self) -> u64 {
        self.slot_stride
    }

    /// SHA-256 of the canonical class encoding.
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The block-relative write extent `(offset, bytes)` of segment `segment` of slot `slot`.
    pub fn slot_extent(&self, slot: u32, segment: usize) -> Option<(u64, u64)> {
        let segment = self.segments.get(segment)?;
        let base = u64::from(slot).checked_mul(self.slot_stride)?;
        let offset = base.checked_add(segment.slot_offset)?;
        offset.checked_add(segment.len)?;
        Some((offset, segment.len))
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut canon = Canon::new("hipfire.expert-cache.layout-class.v1");
        canon.str(&self.kernel_abi.0);
        canon.u64(self.alignment);
        canon.u64(self.slot_bytes);
        canon.u64(self.slot_stride);
        canon.u64(self.segments.len() as u64);
        for segment in &self.segments {
            canon.u64(segment.slot_offset);
            canon.u64(segment.len);
            canon.dtype(segment.dtype);
            canon.str(&segment.quant_tag);
            canon.str(&segment.basis);
            canon.u64(segment.row_stride);
            canon.u64(segment.alignment);
            canon.u64(segment.parts.len() as u64);
            for part in &segment.parts {
                canon.u8(part.projection.tag());
                canon.shape(&part.logical_shape);
                canon.u64(part.bytes);
            }
            canon.u64(segment.sidecars.len() as u64);
            for &sidecar in &segment.sidecars {
                canon.u32(sidecar);
            }
        }
        canon.u64(self.aux.len() as u64);
        for aux in &self.aux {
            canon.dtype(aux.dtype);
            canon.str(&aux.quant_tag);
            canon.str(&aux.basis);
            canon.shape(&aux.logical_shape);
            canon.u64(aux.row_stride);
            canon.u64(aux.alignment);
            canon.u64(aux.len);
        }
        canon.finish()
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment.checked_sub(1)?;
    value.checked_add(mask).map(|value| value & !mask)
}

/// Canonical, length-prefixed SHA-256 writer (no concatenation ambiguity).
struct Canon(Sha256);

impl Canon {
    fn new(domain: &str) -> Self {
        let mut canon = Self(Sha256::new());
        canon.str(domain);
        canon
    }

    fn u8(&mut self, value: u8) {
        self.0.update([value]);
    }

    fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
    }

    fn dtype(&mut self, dtype: DType) {
        self.str(&format!("{dtype:?}"));
    }

    fn shape(&mut self, shape: &[usize]) {
        self.u64(shape.len() as u64);
        for &dim in shape {
            self.u64(dim as u64);
        }
    }

    fn extent(&mut self, extent: &OwnerExtent) {
        self.u32(extent.region);
        self.u64(extent.offset);
        self.u64(extent.len);
    }

    fn alias(&mut self, alias: &Option<SegmentAlias>) {
        match alias {
            None => self.u8(0),
            Some(alias) => {
                self.u8(1);
                self.str(&alias.owner_source);
                self.u64(alias.byte_offset);
            }
        }
    }

    fn finish(self) -> [u8; 32] {
        let out = self.0.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&out);
        digest
    }
}

// ---------------------------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------------------------

/// A validated bundle (immutable).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogBundle {
    expert_id: u32,
    pinned: bool,
    pool: PoolId,
    payload_bytes: u64,
    segments: Vec<SegmentSpec>,
}

impl CatalogBundle {
    pub fn expert_id(&self) -> u32 {
        self.expert_id
    }

    /// Resident by the admitted plan (never evictable).
    pub fn pinned(&self) -> bool {
        self.pinned
    }

    /// The pool (layout class) that hosts this bundle.
    pub fn pool(&self) -> PoolId {
        self.pool
    }

    /// Encoded bytes a fill copies (the sum of the segment lengths).
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// The encoded source segments, in slot layout order.
    pub fn segments(&self) -> &[SegmentSpec] {
        &self.segments
    }

    /// Encoded bytes of one projection (`Gate` / `Up` / `Down`).
    pub fn projection_bytes(&self, projection: Projection) -> u64 {
        self.segments
            .iter()
            .flat_map(|segment| segment.parts.iter())
            .filter(|part| part.projection == projection)
            .map(|part| part.bytes)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogGroup {
    group_id: u32,
    label: String,
    source_fingerprint: String,
    kernel_abi: KernelAbi,
    /// Sorted by source name.
    aux: Vec<AuxSpec>,
    /// Sorted by expert id.
    bundles: Vec<CatalogBundle>,
}

/// One copy a fill performs: canonical source bytes into a slot offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyExtent {
    pub source: OwnerExtent,
    /// Byte offset inside the slot.
    pub slot_offset: u64,
}

/// What one pool (layout class) must hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolDemand {
    pub pool: PoolId,
    /// Bundles of the class.
    pub bundles: usize,
    /// Of those, resident by the admitted plan.
    pub pinned: usize,
    /// Of those, evictable (they need cache slots).
    pub cacheable: usize,
    pub slot_stride: u64,
}

/// Plan facts of a whole catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CatalogFacts {
    pub groups: usize,
    pub bundles: usize,
    pub pinned: usize,
    pub cacheable: usize,
    pub pinned_bytes: u64,
    pub cacheable_bytes: u64,
    /// Layout classes (= pools).
    pub pools: usize,
}

impl CatalogFacts {
    /// Nothing is cacheable: every bundle is resident by the plan. A zero-tail (static-only) plan
    /// is valid; no pool needs a slot.
    pub fn is_static_only(&self) -> bool {
        self.cacheable == 0
    }

    /// Every bundle is cacheable (nothing is pinned).
    pub fn is_fully_cacheable(&self) -> bool {
        self.pinned == 0
    }
}

/// A caller-supplied table image address and size. `table` orders the tables of one group
/// (e.g. 0 = gate/up, 1 = down); a swap of two tables is a canonical change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableExtent {
    pub group_id: u32,
    pub table: u32,
    pub address: u64,
    pub bytes: u64,
}

/// The stable A2.3 identity of a catalog plus its table extents (SHA-256).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CatalogIdentity(pub [u8; 32]);

impl CatalogIdentity {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for CatalogIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for CatalogIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CatalogIdentity({self})")
    }
}

/// The load-time checked expert catalog of one owner generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertCatalog {
    owner_id: u64,
    owner_generation: u64,
    regions: Vec<OwnerRegion>,
    /// Sorted by group id.
    groups: Vec<CatalogGroup>,
    /// `classes[i]` is the class of `PoolId(i)`, in first-appearance order over the groups sorted
    /// by id and bundles by expert id.
    classes: Vec<LayoutClass>,
}

impl ExpertCatalog {
    /// Validate and build the catalog from plain descriptors. Group and bundle declaration order
    /// does not matter (both are sorted).
    pub fn new(
        owner_id: u64,
        owner_generation: u64,
        regions: Vec<OwnerRegion>,
        mut groups: Vec<GroupSpec>,
    ) -> Result<Self, CatalogError> {
        if u32::try_from(regions.len()).is_err() {
            return Err(CatalogError::Overflow {
                what: "owner region count",
            });
        }
        for (index, region) in regions.iter().enumerate() {
            if region.len == 0 {
                return Err(CatalogError::InvalidRegion {
                    region: index as u32,
                    why: "owner region has no bytes",
                });
            }
        }
        if groups.is_empty() {
            return Err(CatalogError::NoGroups);
        }
        groups.sort_by_key(|group| group.group_id);
        if let Some(pair) = groups
            .windows(2)
            .find(|pair| pair[0].group_id == pair[1].group_id)
        {
            return Err(CatalogError::DuplicateGroup {
                group_id: pair[0].group_id,
            });
        }

        let key_of = |group_id: u32, expert_id: u32| ExpertKey {
            owner_id,
            owner_generation,
            group_id,
            expert_id,
        };
        let mut classes: Vec<LayoutClass> = Vec::new();
        let mut class_index: HashMap<[u8; 32], u16> = HashMap::new();
        let mut out_groups = Vec::with_capacity(groups.len());
        let mut total_payload = 0u64;

        for mut group in groups {
            let group_id = group.group_id;
            if group.bundles.is_empty() {
                return Err(CatalogError::ZeroExperts { group_id });
            }
            if group.source_fingerprint.is_empty() {
                return Err(CatalogError::FingerprintMismatch {
                    group_id,
                    source: String::new(),
                });
            }
            group.aux.sort_by(|a, b| a.source_name.cmp(&b.source_name));
            validate_aux(group_id, &group.source_fingerprint, &group.aux)?;
            group.bundles.sort_by_key(|bundle| bundle.expert_id);
            if let Some(pair) = group
                .bundles
                .windows(2)
                .find(|pair| pair[0].expert_id == pair[1].expert_id)
            {
                return Err(CatalogError::DuplicateKey {
                    key: key_of(group_id, pair[0].expert_id),
                });
            }

            let mut bundles = Vec::with_capacity(group.bundles.len());
            for bundle in group.bundles {
                let key = key_of(group_id, bundle.expert_id);
                validate_bundle(group_id, &group.source_fingerprint, &bundle, key)?;
                let class = derive_class(&group.kernel_abi, &group.aux, &bundle.segments, key)?;
                let found = class_index.get(class.digest()).copied();
                let pool = match found {
                    Some(index) => index,
                    None => {
                        let index =
                            u16::try_from(classes.len()).map_err(|_| CatalogError::Overflow {
                                what: "layout classes exceed the PoolId range",
                            })?;
                        class_index.insert(*class.digest(), index);
                        classes.push(class);
                        index
                    }
                };
                let mut payload_bytes = 0u64;
                for segment in &bundle.segments {
                    payload_bytes = payload_bytes.checked_add(segment.extent.len).ok_or(
                        CatalogError::Overflow {
                            what: "bundle payload bytes",
                        },
                    )?;
                }
                total_payload =
                    total_payload
                        .checked_add(payload_bytes)
                        .ok_or(CatalogError::Overflow {
                            what: "catalog payload bytes",
                        })?;
                bundles.push(CatalogBundle {
                    expert_id: bundle.expert_id,
                    pinned: bundle.pinned,
                    pool: PoolId(pool),
                    payload_bytes,
                    segments: bundle.segments,
                });
            }
            out_groups.push(CatalogGroup {
                group_id,
                label: group.label,
                source_fingerprint: group.source_fingerprint,
                kernel_abi: group.kernel_abi,
                aux: group.aux,
                bundles,
            });
        }

        check_owner_extents(&regions, &out_groups)?;
        Ok(Self {
            owner_id,
            owner_generation,
            regions,
            groups: out_groups,
            classes,
        })
    }

    pub fn owner_id(&self) -> u64 {
        self.owner_id
    }

    pub fn owner_generation(&self) -> u64 {
        self.owner_generation
    }

    pub fn regions(&self) -> &[OwnerRegion] {
        &self.regions
    }

    /// Group ids, ascending.
    pub fn group_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.groups.iter().map(|group| group.group_id)
    }

    pub fn group_label(&self, group_id: u32) -> Option<&str> {
        self.group(group_id).map(|group| group.label.as_str())
    }

    pub fn group_fingerprint(&self, group_id: u32) -> Option<&str> {
        self.group(group_id)
            .map(|group| group.source_fingerprint.as_str())
    }

    pub fn group_kernel_abi(&self, group_id: u32) -> Option<&KernelAbi> {
        self.group(group_id).map(|group| &group.kernel_abi)
    }

    /// The group's immutable auxiliary bindings, sorted by source name.
    pub fn group_aux(&self, group_id: u32) -> Option<&[AuxSpec]> {
        self.group(group_id).map(|group| group.aux.as_slice())
    }

    fn group(&self, group_id: u32) -> Option<&CatalogGroup> {
        self.groups
            .binary_search_by_key(&group_id, |group| group.group_id)
            .ok()
            .map(|index| &self.groups[index])
    }

    /// The bundle `key` names; `None` for another owner generation or an unknown expert.
    pub fn bundle(&self, key: &ExpertKey) -> Option<&CatalogBundle> {
        if key.owner_id != self.owner_id || key.owner_generation != self.owner_generation {
            return None;
        }
        let group = self.group(key.group_id)?;
        let index = group
            .bundles
            .binary_search_by_key(&key.expert_id, |bundle| bundle.expert_id)
            .ok()?;
        Some(&group.bundles[index])
    }

    /// Every `(key, bundle)`, by group id then expert id.
    pub fn entries(&self) -> impl Iterator<Item = (ExpertKey, &CatalogBundle)> + '_ {
        let owner_id = self.owner_id;
        let owner_generation = self.owner_generation;
        self.groups.iter().flat_map(move |group| {
            let group_id = group.group_id;
            group.bundles.iter().map(move |bundle| {
                (
                    ExpertKey {
                        owner_id,
                        owner_generation,
                        group_id,
                        expert_id: bundle.expert_id,
                    },
                    bundle,
                )
            })
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = ExpertKey> + '_ {
        self.entries().map(|(key, _)| key)
    }

    /// The keys hosted by `pool`.
    pub fn keys_in_pool(&self, pool: PoolId) -> impl Iterator<Item = ExpertKey> + '_ {
        self.entries()
            .filter(move |(_, bundle)| bundle.pool == pool)
            .map(|(key, _)| key)
    }

    /// The evictable keys hosted by `pool` (the pinned ones never need a slot).
    pub fn cacheable_keys_in_pool(&self, pool: PoolId) -> impl Iterator<Item = ExpertKey> + '_ {
        self.entries()
            .filter(move |(_, bundle)| bundle.pool == pool && !bundle.pinned)
            .map(|(key, _)| key)
    }

    /// The pool (layout class) `key` belongs to.
    pub fn pool_of(&self, key: &ExpertKey) -> Option<PoolId> {
        self.bundle(key).map(|bundle| bundle.pool)
    }

    /// The layout classes; `classes()[pool.0]` is the class of `pool`.
    pub fn classes(&self) -> &[LayoutClass] {
        &self.classes
    }

    pub fn class_of(&self, pool: PoolId) -> Option<&LayoutClass> {
        self.classes.get(usize::from(pool.0))
    }

    /// Prove `key` may live in a slot of `pool`: the key's class is the pool's class.
    pub fn check_slot_class(&self, key: &ExpertKey, pool: PoolId) -> Result<(), CatalogError> {
        if self.class_of(pool).is_none() {
            return Err(CatalogError::UnknownPool(pool));
        }
        let bundle = self
            .bundle(key)
            .ok_or(CatalogError::UnknownKey { key: *key })?;
        if bundle.pool != pool {
            return Err(CatalogError::LayoutMismatch {
                key: *key,
                why: "the key's layout class is not the pool's class",
            });
        }
        Ok(())
    }

    /// The source-to-slot copies of one fill of `key`, in slot layout order.
    pub fn copy_extents(&self, key: &ExpertKey) -> Option<impl Iterator<Item = CopyExtent> + '_> {
        let bundle = self.bundle(key)?;
        let class = self.class_of(bundle.pool)?;
        Some(
            bundle
                .segments
                .iter()
                .zip(class.segments.iter())
                .map(|(segment, class_segment)| CopyExtent {
                    source: segment.extent,
                    slot_offset: class_segment.slot_offset,
                }),
        )
    }

    /// Per-pool demand, indexed by `PoolId`.
    pub fn pool_demands(&self) -> Vec<PoolDemand> {
        let mut demands: Vec<PoolDemand> = self
            .classes
            .iter()
            .enumerate()
            .map(|(index, class)| PoolDemand {
                pool: PoolId(index as u16),
                bundles: 0,
                pinned: 0,
                cacheable: 0,
                slot_stride: class.slot_stride,
            })
            .collect();
        for (_, bundle) in self.entries() {
            let demand = &mut demands[usize::from(bundle.pool.0)];
            demand.bundles += 1;
            if bundle.pinned {
                demand.pinned += 1;
            } else {
                demand.cacheable += 1;
            }
        }
        demands
    }

    /// The pools that have at least one cacheable key (a zero-tail or full-resident plan has
    /// none, and then needs no `PoolInit`): construct one S1a policy pool per entry.
    pub fn pools_needing_slots(&self) -> Vec<PoolDemand> {
        self.pool_demands()
            .into_iter()
            .filter(|demand| demand.cacheable > 0)
            .collect()
    }

    pub fn facts(&self) -> CatalogFacts {
        let mut facts = CatalogFacts {
            groups: self.groups.len(),
            bundles: 0,
            pinned: 0,
            cacheable: 0,
            pinned_bytes: 0,
            cacheable_bytes: 0,
            pools: self.classes.len(),
        };
        for (_, bundle) in self.entries() {
            facts.bundles += 1;
            // The per-bundle sums were checked against the catalog total at construction.
            if bundle.pinned {
                facts.pinned += 1;
                facts.pinned_bytes += bundle.payload_bytes;
            } else {
                facts.cacheable += 1;
                facts.cacheable_bytes += bundle.payload_bytes;
            }
        }
        facts
    }

    /// Check every bundle against a D2 unit plan's geometry: gate + up bytes and down bytes of
    /// every bundle are exactly the unit bytes the plan charged.
    pub fn check_unit_geometry(&self, plan: &UnitPlanDescriptor) -> Result<(), CatalogError> {
        for (key, bundle) in self.entries() {
            let gate_up =
                bundle.projection_bytes(Projection::Gate) + bundle.projection_bytes(Projection::Up);
            if gate_up != plan.gate_up_bytes {
                return Err(CatalogError::LayoutMismatch {
                    key,
                    why: "gate/up bytes differ from the unit plan's geometry",
                });
            }
            if bundle.projection_bytes(Projection::Down) != plan.down_bytes {
                return Err(CatalogError::LayoutMismatch {
                    key,
                    why: "down bytes differ from the unit plan's geometry",
                });
            }
        }
        Ok(())
    }

    /// The A2.3 stable identity: canonical digest of the catalog plus `tables` (the caller's
    /// per-group table addresses and extents). No slot pointer and no planned residency is in
    /// it: a remap keeps it, any canonical change breaks it.
    pub fn identity(&self, tables: &[TableExtent]) -> Result<CatalogIdentity, CatalogError> {
        let mut tables = tables.to_vec();
        tables.sort_by_key(|table| (table.group_id, table.table));
        for table in &tables {
            if self.group(table.group_id).is_none() {
                return Err(CatalogError::UnknownGroup {
                    group_id: table.group_id,
                });
            }
            if table.bytes == 0 || table.address.checked_add(table.bytes).is_none() {
                return Err(CatalogError::TableExtentInvalid {
                    group_id: table.group_id,
                    table: table.table,
                    address: table.address,
                    bytes: table.bytes,
                });
            }
        }
        if let Some(pair) = tables
            .windows(2)
            .find(|pair| (pair[0].group_id, pair[0].table) == (pair[1].group_id, pair[1].table))
        {
            return Err(CatalogError::DuplicateTable {
                group_id: pair[0].group_id,
                table: pair[0].table,
            });
        }
        let mut by_address: Vec<&TableExtent> = tables.iter().collect();
        by_address.sort_by_key(|table| table.address);
        for pair in by_address.windows(2) {
            // Both ends were checked not to overflow above.
            if pair[0].address + pair[0].bytes > pair[1].address {
                return Err(CatalogError::TableExtentOverlap {
                    first: (pair[0].group_id, pair[0].table),
                    second: (pair[1].group_id, pair[1].table),
                });
            }
        }

        let mut canon = Canon::new("hipfire.expert-cache.catalog-identity.v1");
        canon.u64(self.owner_id);
        canon.u64(self.owner_generation);
        canon.u64(self.regions.len() as u64);
        for region in &self.regions {
            canon.str(&region.name);
            canon.u64(region.len);
        }
        canon.u64(self.groups.len() as u64);
        for group in &self.groups {
            canon.u32(group.group_id);
            canon.str(&group.source_fingerprint);
            canon.str(&group.kernel_abi.0);
            canon.u64(group.aux.len() as u64);
            for aux in &group.aux {
                canon.str(&aux.source_name);
                canon.str(&aux.fingerprint);
                canon.extent(&aux.extent);
                canon.alias(&aux.alias);
                canon.dtype(aux.dtype);
                canon.str(&aux.quant_tag);
                canon.str(&aux.basis);
                canon.shape(&aux.logical_shape);
                canon.u64(aux.row_stride);
                canon.u64(aux.alignment);
            }
            canon.u64(group.bundles.len() as u64);
            for bundle in &group.bundles {
                canon.u32(bundle.expert_id);
                canon
                    .0
                    .update(self.classes[usize::from(bundle.pool.0)].digest());
                canon.u64(bundle.segments.len() as u64);
                for segment in &bundle.segments {
                    canon.str(&segment.source_name);
                    canon.str(&segment.fingerprint);
                    canon.extent(&segment.extent);
                    canon.alias(&segment.alias);
                }
            }
        }
        canon.u64(tables.len() as u64);
        for table in &tables {
            canon.u32(table.group_id);
            canon.u32(table.table);
            canon.u64(table.address);
            canon.u64(table.bytes);
        }
        Ok(CatalogIdentity(canon.finish()))
    }
}

fn nonempty_distinct<'a>(
    group_id: u32,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), CatalogError> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if name.is_empty() {
            return Err(CatalogError::InvalidSidecar {
                group_id,
                sidecar: String::new(),
                why: "empty sidecar name",
            });
        }
        if seen.contains(&name) {
            return Err(CatalogError::InvalidSidecar {
                group_id,
                sidecar: name.to_string(),
                why: "duplicate sidecar name",
            });
        }
        seen.push(name);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    source_name: &str,
    fingerprint: &str,
    quant_tag: &str,
    basis: &str,
    row_stride: u64,
    alignment: u64,
    extent: &OwnerExtent,
    alias: &Option<SegmentAlias>,
) -> Result<(), CatalogError> {
    let bad = |why: &'static str| CatalogError::InvalidSegment {
        source: source_name.to_string(),
        why,
    };
    if source_name.is_empty() {
        return Err(bad("empty source name"));
    }
    if fingerprint.is_empty() {
        return Err(bad("empty source fingerprint"));
    }
    if quant_tag.is_empty() {
        return Err(bad("empty quant tag"));
    }
    if basis.is_empty() {
        return Err(bad("empty basis"));
    }
    if extent.len == 0 {
        return Err(bad("zero-length extent"));
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(bad("alignment is not a power of two"));
    }
    if row_stride == 0 {
        return Err(bad("zero row stride"));
    }
    if let Some(alias) = alias {
        if alias.owner_source.is_empty() || alias.owner_source == source_name {
            return Err(bad("alias owner is empty or the source itself"));
        }
    }
    Ok(())
}

fn validate_aux(group_id: u32, fingerprint: &str, aux: &[AuxSpec]) -> Result<(), CatalogError> {
    nonempty_distinct(group_id, aux.iter().map(|aux| aux.source_name.as_str()))?;
    for aux in aux {
        validate_common(
            &aux.source_name,
            &aux.fingerprint,
            &aux.quant_tag,
            &aux.basis,
            aux.row_stride,
            aux.alignment,
            &aux.extent,
            &aux.alias,
        )?;
        if aux.logical_shape.is_empty() || aux.logical_shape.contains(&0) {
            return Err(CatalogError::InvalidSidecar {
                group_id,
                sidecar: aux.source_name.clone(),
                why: "empty or zero-sized logical shape",
            });
        }
        if aux.fingerprint != fingerprint {
            return Err(CatalogError::FingerprintMismatch {
                group_id,
                source: aux.source_name.clone(),
            });
        }
    }
    Ok(())
}

fn validate_bundle(
    group_id: u32,
    fingerprint: &str,
    bundle: &BundleSpec,
    key: ExpertKey,
) -> Result<(), CatalogError> {
    let mismatch = |why: &'static str| CatalogError::LayoutMismatch { key, why };
    if bundle.segments.is_empty() {
        return Err(mismatch("the bundle has no segment"));
    }
    let mut seen = [0usize; 3];
    for segment in &bundle.segments {
        validate_common(
            &segment.source_name,
            &segment.fingerprint,
            &segment.quant_tag,
            &segment.basis,
            segment.row_stride,
            segment.alignment,
            &segment.extent,
            &segment.alias,
        )?;
        if segment.fingerprint != fingerprint {
            return Err(CatalogError::FingerprintMismatch {
                group_id,
                source: segment.source_name.clone(),
            });
        }
        nonempty_distinct(group_id, segment.sidecar_names.iter().map(String::as_str))?;
        if segment.parts.is_empty() {
            return Err(mismatch("a segment holds no projection"));
        }
        let mut covered = 0u64;
        for part in &segment.parts {
            if part.bytes == 0 || part.logical_shape.is_empty() || part.logical_shape.contains(&0) {
                return Err(CatalogError::InvalidSegment {
                    source: segment.source_name.clone(),
                    why: "a projection part has no bytes or an empty / zero logical shape",
                });
            }
            covered = covered
                .checked_add(part.bytes)
                .ok_or(CatalogError::Overflow {
                    what: "segment part bytes",
                })?;
            seen[usize::from(part.projection.tag())] += 1;
        }
        if covered != segment.extent.len {
            return Err(mismatch(
                "a segment's projection parts do not cover its extent exactly",
            ));
        }
        let allowed = matches!(
            segment
                .parts
                .iter()
                .map(|part| part.projection)
                .collect::<Vec<_>>()
                .as_slice(),
            [Projection::Gate]
                | [Projection::Up]
                | [Projection::Down]
                | [Projection::Gate, Projection::Up]
        );
        if !allowed {
            return Err(mismatch("unsupported projection grouping in a segment"));
        }
    }
    if seen != [1, 1, 1] {
        return Err(mismatch(
            "the bundle must hold exactly one gate, one up and one down projection",
        ));
    }
    Ok(())
}

fn derive_class(
    kernel_abi: &KernelAbi,
    aux: &[AuxSpec],
    segments: &[SegmentSpec],
    key: ExpertKey,
) -> Result<LayoutClass, CatalogError> {
    let mut class_segments = Vec::with_capacity(segments.len());
    let mut cursor = 0u64;
    let mut alignment = 1u64;
    for segment in segments {
        let offset = align_up(cursor, segment.alignment).ok_or(CatalogError::Overflow {
            what: "slot layout offset",
        })?;
        cursor = offset
            .checked_add(segment.extent.len)
            .ok_or(CatalogError::Overflow {
                what: "slot layout end",
            })?;
        alignment = alignment.max(segment.alignment);
        let mut sidecars = Vec::with_capacity(segment.sidecar_names.len());
        for name in &segment.sidecar_names {
            let index = aux
                .binary_search_by(|aux| aux.source_name.as_str().cmp(name.as_str()))
                .map_err(|_| CatalogError::MissingSidecar {
                    key,
                    sidecar: name.clone(),
                })?;
            sidecars.push(index as u32);
        }
        class_segments.push(ClassSegment {
            slot_offset: offset,
            len: segment.extent.len,
            dtype: segment.dtype,
            quant_tag: segment.quant_tag.clone(),
            basis: segment.basis.clone(),
            row_stride: segment.row_stride,
            alignment: segment.alignment,
            parts: segment
                .parts
                .iter()
                .map(|part| ClassPart {
                    projection: part.projection,
                    logical_shape: part.logical_shape.clone(),
                    bytes: part.bytes,
                })
                .collect(),
            sidecars,
        });
    }
    let slot_stride = align_up(cursor, alignment).ok_or(CatalogError::Overflow {
        what: "slot stride",
    })?;
    let mut class = LayoutClass {
        kernel_abi: kernel_abi.clone(),
        segments: class_segments,
        aux: aux
            .iter()
            .map(|aux| ClassAux {
                dtype: aux.dtype,
                quant_tag: aux.quant_tag.clone(),
                basis: aux.basis.clone(),
                logical_shape: aux.logical_shape.clone(),
                row_stride: aux.row_stride,
                alignment: aux.alignment,
                len: aux.extent.len,
            })
            .collect(),
        alignment,
        slot_bytes: cursor,
        slot_stride,
        digest: [0; 32],
    };
    class.digest = class.compute_digest();
    Ok(class)
}

/// Range-check every extent against its owner region and check that no two non-alias extents
/// share bytes.
fn check_owner_extents<'a>(
    regions: &[OwnerRegion],
    groups: &'a [CatalogGroup],
) -> Result<(), CatalogError> {
    let mut solid: Vec<(u32, u64, u64, &str)> = Vec::new();
    let mut visit = |name: &'a str,
                     extent: &OwnerExtent,
                     alias: &Option<SegmentAlias>|
     -> Result<(), CatalogError> {
        let region =
            regions
                .get(extent.region as usize)
                .ok_or_else(|| CatalogError::UnknownRegion {
                    source: name.to_string(),
                    region: extent.region,
                })?;
        let end = extent.end().ok_or(CatalogError::Overflow {
            what: "owner extent end",
        })?;
        if end > region.len {
            return Err(CatalogError::OwnerExtentOutOfRange {
                source: name.to_string(),
                region: extent.region,
                offset: extent.offset,
                len: extent.len,
                region_len: region.len,
            });
        }
        if alias.is_none() {
            solid.push((extent.region, extent.offset, end, name));
        }
        Ok(())
    };
    for group in groups {
        for aux in &group.aux {
            visit(&aux.source_name, &aux.extent, &aux.alias)?;
        }
        for bundle in &group.bundles {
            for segment in &bundle.segments {
                visit(&segment.source_name, &segment.extent, &segment.alias)?;
            }
        }
    }
    solid.sort_by_key(|&(region, start, end, _)| (region, start, end));
    let mut previous: Option<(u32, u64, &str)> = None;
    for &(region, start, end, name) in &solid {
        if let Some((previous_region, previous_end, previous_name)) = previous {
            if previous_region == region && start < previous_end {
                return Err(CatalogError::OwnerExtentOverlap {
                    first: previous_name.to_string(),
                    second: name.to_string(),
                    region,
                    offset: start,
                });
            }
        }
        // No overlap so far: this extent ends past every earlier one of its region.
        previous = Some((region, end, name));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// D2 unit plan descriptor
// ---------------------------------------------------------------------------------------------

/// Plain mirror of the Qwen4 D2 per-expert unit plan (`EpUnitPlan`, `expert_ep_units.rs:167`):
/// the geometry's byte sizes and the `[rank][layer][expert]` residency bitmap (true only for
/// experts the rank owns and keeps in VRAM). It feeds the catalog's `pinned` flags
/// ([`Self::layer_resident`]) and its geometry check ([`ExpertCatalog::check_unit_geometry`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitPlanDescriptor {
    layers: usize,
    experts_per_layer: usize,
    ranks: usize,
    gate_up_bytes: u64,
    down_bytes: u64,
    resident: Vec<bool>,
}

impl UnitPlanDescriptor {
    pub fn new(
        layers: usize,
        experts_per_layer: usize,
        ranks: usize,
        gate_up_bytes: u64,
        down_bytes: u64,
        resident: Vec<bool>,
    ) -> Result<Self, CatalogError> {
        let bad = |why: &'static str| CatalogError::UnitPlanInconsistent { why };
        if layers == 0 || experts_per_layer == 0 || ranks == 0 {
            return Err(bad("layers, experts per layer and ranks must be non-zero"));
        }
        if gate_up_bytes == 0 || down_bytes == 0 {
            return Err(bad("an expert has no bytes"));
        }
        if gate_up_bytes.checked_add(down_bytes).is_none() {
            return Err(CatalogError::Overflow { what: "unit bytes" });
        }
        let total = ranks
            .checked_mul(layers)
            .and_then(|units| units.checked_mul(experts_per_layer))
            .ok_or(CatalogError::Overflow {
                what: "unit plan size",
            })?;
        if resident.len() != total {
            return Err(bad("the residency bitmap is not [rank][layer][expert]"));
        }
        Ok(Self {
            layers,
            experts_per_layer,
            ranks,
            gate_up_bytes,
            down_bytes,
            resident,
        })
    }

    pub fn unit_bytes(&self) -> u64 {
        self.gate_up_bytes + self.down_bytes
    }

    /// One layer's residency bitmap of `rank` over all `experts_per_layer` experts: the
    /// `pinned` argument of [`GroupSpec::from_execution_plan`].
    pub fn layer_resident(&self, rank: usize, layer: usize) -> Option<&[bool]> {
        if rank >= self.ranks || layer >= self.layers {
            return None;
        }
        let at = (rank * self.layers + layer) * self.experts_per_layer;
        self.resident.get(at..at + self.experts_per_layer)
    }

    /// Units `rank` keeps resident.
    pub fn resident_units(&self, rank: usize) -> Option<usize> {
        if rank >= self.ranks {
            return None;
        }
        let per_rank = self.layers * self.experts_per_layer;
        Some(
            self.resident[rank * per_rank..(rank + 1) * per_rank]
                .iter()
                .filter(|&&kept| kept)
                .count(),
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_mesh::{DeviceMesh, DimKind};
    use crate::sealed_moe::{plan_expert_execution, ExpertSourceMetadata};
    use crate::tp_shard::ExpertAssign;
    use crate::weight_manifest::{
        ExpertGroupSpec, ExpertParallelism, ExpertProjectionResources, ExpertResourceRequirements,
        ExpertSourceLayout, ShardPolicy, WeightEntry,
    };

    const OWNER: u64 = 7;
    const GEN: u64 = 3;
    const HOST: u32 = 0;
    const AUXR: u32 = 1;

    fn key(group_id: u32, expert_id: u32) -> ExpertKey {
        ExpertKey {
            owner_id: OWNER,
            owner_generation: GEN,
            group_id,
            expert_id,
        }
    }

    /// Format of one fixture bundle.
    #[derive(Clone)]
    struct Fmt {
        dtype: DType,
        quant: &'static str,
        gate_up: u64,
        down: u64,
    }

    fn mq4() -> Fmt {
        Fmt {
            dtype: DType::MQ4G256V2,
            quant: "qt44",
            gate_up: 64,
            down: 32,
        }
    }

    fn lloyd() -> Fmt {
        Fmt {
            dtype: DType::MQ2G256LloydU,
            quant: "mq2-lloyd",
            gate_up: 48,
            down: 32,
        }
    }

    /// Canonical offset of one fixture bundle (groups never share bytes).
    fn base(group_id: u32, expert: u32) -> u64 {
        u64::from(group_id) * 0x2000 + u64::from(expert) * 256
    }

    fn gate_up_segment(group_id: u32, expert: u32, fmt: &Fmt, sidecars: &[&str]) -> SegmentSpec {
        SegmentSpec {
            source_name: "gate_up".into(),
            fingerprint: "gen-1".into(),
            extent: OwnerExtent {
                region: HOST,
                offset: base(group_id, expert),
                len: fmt.gate_up,
            },
            alias: None,
            dtype: fmt.dtype,
            quant_tag: fmt.quant.into(),
            basis: "fwht-g256".into(),
            row_stride: 8,
            alignment: 16,
            parts: vec![
                ProjectionPart {
                    projection: Projection::Gate,
                    logical_shape: vec![4, 8],
                    bytes: fmt.gate_up / 2,
                },
                ProjectionPart {
                    projection: Projection::Up,
                    logical_shape: vec![4, 8],
                    bytes: fmt.gate_up / 2,
                },
            ],
            sidecar_names: sidecars.iter().map(|name| name.to_string()).collect(),
        }
    }

    fn down_segment(group_id: u32, expert: u32, fmt: &Fmt) -> SegmentSpec {
        SegmentSpec {
            source_name: "down".into(),
            fingerprint: "gen-1".into(),
            extent: OwnerExtent {
                region: HOST,
                offset: base(group_id, expert) + 128,
                len: fmt.down,
            },
            alias: None,
            dtype: fmt.dtype,
            quant_tag: fmt.quant.into(),
            basis: "fwht-g128".into(),
            row_stride: 4,
            alignment: 16,
            parts: vec![ProjectionPart {
                projection: Projection::Down,
                logical_shape: vec![8, 4],
                bytes: fmt.down,
            }],
            sidecar_names: Vec::new(),
        }
    }

    fn bundle(group_id: u32, expert: u32, fmt: &Fmt, pinned: bool) -> BundleSpec {
        BundleSpec {
            expert_id: expert,
            pinned,
            segments: vec![
                gate_up_segment(group_id, expert, fmt, &["rot"]),
                down_segment(group_id, expert, fmt),
            ],
        }
    }

    fn rot(group_id: u32) -> AuxSpec {
        AuxSpec {
            source_name: "rot".into(),
            fingerprint: "gen-1".into(),
            extent: OwnerExtent {
                region: AUXR,
                offset: u64::from(group_id) * 64,
                len: 64,
            },
            alias: None,
            dtype: DType::F32,
            quant_tag: "f32".into(),
            basis: "natural".into(),
            logical_shape: vec![16],
            row_stride: 64,
            alignment: 4,
        }
    }

    fn group(group_id: u32, experts: &[(u32, Fmt, bool)]) -> GroupSpec {
        GroupSpec {
            group_id,
            label: format!("layer {group_id}"),
            source_fingerprint: "gen-1".into(),
            kernel_abi: KernelAbi("qwen4.mq4.v1".into()),
            aux: vec![rot(group_id)],
            bundles: experts
                .iter()
                .map(|(expert, fmt, pinned)| bundle(group_id, *expert, fmt, *pinned))
                .collect(),
        }
    }

    fn regions() -> Vec<OwnerRegion> {
        vec![
            OwnerRegion {
                name: "host".into(),
                len: 1 << 20,
            },
            OwnerRegion {
                name: "aux".into(),
                len: 4096,
            },
        ]
    }

    fn build(groups: Vec<GroupSpec>) -> Result<ExpertCatalog, CatalogError> {
        ExpertCatalog::new(OWNER, GEN, regions(), groups)
    }

    fn base_groups() -> Vec<GroupSpec> {
        vec![
            group(0, &[(0, mq4(), true), (1, mq4(), false), (2, mq4(), false)]),
            group(1, &[(0, mq4(), false), (1, mq4(), false)]),
        ]
    }

    fn tables() -> Vec<TableExtent> {
        vec![
            TableExtent {
                group_id: 0,
                table: 0,
                address: 0x7000_0000,
                bytes: 0x400,
            },
            TableExtent {
                group_id: 0,
                table: 1,
                address: 0x7000_1000,
                bytes: 0x400,
            },
            TableExtent {
                group_id: 1,
                table: 0,
                address: 0x7000_2000,
                bytes: 0x400,
            },
        ]
    }

    // --- validation errors ----------------------------------------------------------------

    #[test]
    fn zero_experts_and_no_groups_are_refused() {
        assert_eq!(build(Vec::new()).unwrap_err(), CatalogError::NoGroups);
        let mut empty = base_groups();
        empty[1].bundles.clear();
        assert_eq!(
            build(empty).unwrap_err(),
            CatalogError::ZeroExperts { group_id: 1 }
        );
    }

    #[test]
    fn duplicate_key_and_group_are_refused() {
        let mut dup = base_groups();
        dup[0].bundles.push(bundle(0, 1, &mq4(), false));
        assert_eq!(
            build(dup).unwrap_err(),
            CatalogError::DuplicateKey { key: key(0, 1) }
        );
        let mut dup_group = base_groups();
        dup_group[1].group_id = 0;
        assert_eq!(
            build(dup_group).unwrap_err(),
            CatalogError::DuplicateGroup { group_id: 0 }
        );
    }

    #[test]
    fn missing_and_invalid_sidecars_are_refused() {
        let mut missing = base_groups();
        missing[0].aux.clear();
        assert_eq!(
            build(missing).unwrap_err(),
            CatalogError::MissingSidecar {
                key: key(0, 0),
                sidecar: "rot".into()
            }
        );

        let mut unbound = base_groups();
        unbound[1].bundles[1].segments[0].sidecar_names = vec!["codebook".into()];
        assert_eq!(
            build(unbound).unwrap_err(),
            CatalogError::MissingSidecar {
                key: key(1, 1),
                sidecar: "codebook".into()
            }
        );

        let mut duplicate_aux = base_groups();
        duplicate_aux[0].aux.push(rot(0));
        assert!(matches!(
            build(duplicate_aux).unwrap_err(),
            CatalogError::InvalidSidecar {
                group_id: 0,
                why: "duplicate sidecar name",
                ..
            }
        ));

        let mut empty_name = base_groups();
        empty_name[0].aux[0].source_name = String::new();
        assert!(matches!(
            build(empty_name).unwrap_err(),
            CatalogError::InvalidSidecar {
                why: "empty sidecar name",
                ..
            }
        ));

        let mut twice = base_groups();
        twice[0].bundles[0].segments[0].sidecar_names = vec!["rot".into(), "rot".into()];
        assert!(matches!(
            build(twice).unwrap_err(),
            CatalogError::InvalidSidecar {
                why: "duplicate sidecar name",
                ..
            }
        ));

        let mut zero_shape = base_groups();
        zero_shape[0].aux[0].logical_shape = vec![0];
        assert!(matches!(
            build(zero_shape).unwrap_err(),
            CatalogError::InvalidSidecar { .. }
        ));

        let mut other_generation = base_groups();
        other_generation[0].aux[0].fingerprint = "gen-2".into();
        assert_eq!(
            build(other_generation).unwrap_err(),
            CatalogError::FingerprintMismatch {
                group_id: 0,
                source: "rot".into()
            }
        );
    }

    #[test]
    fn segment_from_another_source_generation_is_refused() {
        let mut stale = base_groups();
        stale[1].bundles[0].segments[1].fingerprint = "gen-0".into();
        assert_eq!(
            build(stale).unwrap_err(),
            CatalogError::FingerprintMismatch {
                group_id: 1,
                source: "down".into()
            }
        );
    }

    #[test]
    fn out_of_range_and_unknown_region_extents_are_refused() {
        let mut past_end = base_groups();
        past_end[0].bundles[0].segments[1].extent.offset = (1 << 20) - 16;
        assert_eq!(
            build(past_end).unwrap_err(),
            CatalogError::OwnerExtentOutOfRange {
                source: "down".into(),
                region: HOST,
                offset: (1 << 20) - 16,
                len: 32,
                region_len: 1 << 20,
            }
        );
        let mut aux_past = base_groups();
        aux_past[0].aux[0].extent.offset = 4096 - 32;
        assert!(matches!(
            build(aux_past).unwrap_err(),
            CatalogError::OwnerExtentOutOfRange { region: AUXR, .. }
        ));
        let mut unknown = base_groups();
        unknown[0].bundles[0].segments[0].extent.region = 9;
        assert_eq!(
            build(unknown).unwrap_err(),
            CatalogError::UnknownRegion {
                source: "gate_up".into(),
                region: 9
            }
        );
    }

    #[test]
    fn overlapping_owner_extents_are_refused_but_aliases_are_not() {
        // Expert 1's gate_up starts inside expert 0's.
        let mut overlap = base_groups();
        overlap[0].bundles[1].segments[0].extent.offset = 32;
        assert!(matches!(
            build(overlap).unwrap_err(),
            CatalogError::OwnerExtentOverlap { region: HOST, .. }
        ));

        // The same overlap is legal for a read-only alias of the overlapped source.
        let mut alias = base_groups();
        alias[0].bundles[1].segments[0].extent.offset = 32;
        alias[0].bundles[1].segments[0].alias = Some(SegmentAlias {
            owner_source: "gate_up.base".into(),
            byte_offset: 32,
        });
        build(alias).unwrap();

        // An alias is still range checked.
        let mut alias_out = base_groups();
        alias_out[0].bundles[1].segments[0].extent.offset = (1 << 20) - 8;
        alias_out[0].bundles[1].segments[0].alias = Some(SegmentAlias {
            owner_source: "gate_up.base".into(),
            byte_offset: 0,
        });
        assert!(matches!(
            build(alias_out).unwrap_err(),
            CatalogError::OwnerExtentOutOfRange { .. }
        ));

        // An aux binding sharing bytes with a segment overlaps too.
        let mut aux_overlap = base_groups();
        aux_overlap[0].aux[0].extent = OwnerExtent {
            region: HOST,
            offset: 16,
            len: 64,
        };
        assert!(matches!(
            build(aux_overlap).unwrap_err(),
            CatalogError::OwnerExtentOverlap { .. }
        ));
    }

    #[test]
    fn size_overflow_is_refused() {
        let mut overflow = base_groups();
        overflow[0].bundles[0].segments[1].extent.offset = u64::MAX - 8;
        assert_eq!(
            build(overflow).unwrap_err(),
            CatalogError::Overflow {
                what: "owner extent end"
            }
        );
        let mut slot = base_groups();
        slot[0].bundles[0].segments[0].alignment = 1 << 63;
        slot[0].bundles[0].segments[1].alignment = 1 << 63;
        assert_eq!(
            build(slot).unwrap_err(),
            CatalogError::Overflow {
                what: "slot stride"
            }
        );
        // A table at the top of the address space cannot be hashed.
        let catalog = build(base_groups()).unwrap();
        let bad = [TableExtent {
            group_id: 0,
            table: 0,
            address: u64::MAX - 4,
            bytes: 64,
        }];
        assert!(matches!(
            catalog.identity(&bad).unwrap_err(),
            CatalogError::TableExtentInvalid { .. }
        ));
    }

    #[test]
    fn layout_mismatches_are_refused() {
        // Parts that do not cover the extent: no padding-based reinterpretation.
        let mut padded = base_groups();
        padded[0].bundles[0].segments[1].extent.len = 48;
        assert_eq!(
            build(padded).unwrap_err(),
            CatalogError::LayoutMismatch {
                key: key(0, 0),
                why: "a segment's projection parts do not cover its extent exactly"
            }
        );
        // No down projection.
        let mut no_down = base_groups();
        no_down[0].bundles[2].segments.pop();
        assert_eq!(
            build(no_down).unwrap_err(),
            CatalogError::LayoutMismatch {
                key: key(0, 2),
                why: "the bundle must hold exactly one gate, one up and one down projection"
            }
        );
        // Up before gate in a fused carrier.
        let mut swapped = base_groups();
        swapped[0].bundles[0].segments[0].parts.reverse();
        assert!(matches!(
            build(swapped).unwrap_err(),
            CatalogError::LayoutMismatch {
                why: "unsupported projection grouping in a segment",
                ..
            }
        ));
        // A second gate.
        let mut twice = base_groups();
        let extra = gate_up_segment(0, 9, &mq4(), &["rot"]);
        twice[0].bundles[1].segments.push(extra);
        assert!(matches!(
            build(twice).unwrap_err(),
            CatalogError::LayoutMismatch { .. }
        ));
        // A slot of another class refuses the key.
        let catalog = build(vec![group(0, &[(0, mq4(), false), (1, lloyd(), false)])]).unwrap();
        let pool0 = catalog.pool_of(&key(0, 0)).unwrap();
        let pool1 = catalog.pool_of(&key(0, 1)).unwrap();
        catalog.check_slot_class(&key(0, 0), pool0).unwrap();
        assert!(matches!(
            catalog.check_slot_class(&key(0, 0), pool1).unwrap_err(),
            CatalogError::LayoutMismatch { .. }
        ));
        assert_eq!(
            catalog.check_slot_class(&key(0, 0), PoolId(9)).unwrap_err(),
            CatalogError::UnknownPool(PoolId(9))
        );
        assert_eq!(
            catalog.check_slot_class(&key(5, 0), pool0).unwrap_err(),
            CatalogError::UnknownKey { key: key(5, 0) }
        );
    }

    #[test]
    fn malformed_segments_and_regions_are_refused() {
        let mut zero_len = base_groups();
        zero_len[0].bundles[0].segments[1].extent.len = 0;
        assert!(matches!(
            build(zero_len).unwrap_err(),
            CatalogError::InvalidSegment {
                why: "zero-length extent",
                ..
            }
        ));
        let mut alignment = base_groups();
        alignment[0].bundles[0].segments[0].alignment = 12;
        assert!(matches!(
            build(alignment).unwrap_err(),
            CatalogError::InvalidSegment { .. }
        ));
        let mut empty_region = regions();
        empty_region[1].len = 0;
        assert_eq!(
            ExpertCatalog::new(OWNER, GEN, empty_region, base_groups()).unwrap_err(),
            CatalogError::InvalidRegion {
                region: 1,
                why: "owner region has no bytes"
            }
        );
    }

    // --- classes and pools ------------------------------------------------------------------

    #[test]
    fn mixed_layer_is_partitioned_into_class_pools() {
        // Layer 0: experts 0, 2 are MQ4, expert 1 Lloyd; layer 1 is all MQ4. Declared out of
        // order on purpose.
        let catalog = build(vec![
            group(1, &[(1, mq4(), false), (0, mq4(), true)]),
            group(
                0,
                &[(2, mq4(), false), (1, lloyd(), false), (0, mq4(), true)],
            ),
        ])
        .unwrap();
        assert_eq!(catalog.classes().len(), 2);
        let mq4_pool = catalog.pool_of(&key(0, 0)).unwrap();
        let lloyd_pool = catalog.pool_of(&key(0, 1)).unwrap();
        assert_ne!(mq4_pool, lloyd_pool);
        // First appearance over groups 0, 1 / experts ascending: expert 0 is MQ4 => PoolId(0).
        assert_eq!(mq4_pool, PoolId(0));
        assert_eq!(lloyd_pool, PoolId(1));
        // Identical layout shares the pool across groups and experts.
        assert_eq!(catalog.pool_of(&key(0, 2)), Some(mq4_pool));
        assert_eq!(catalog.pool_of(&key(1, 0)), Some(mq4_pool));
        assert_eq!(catalog.pool_of(&key(1, 1)), Some(mq4_pool));
        // Another owner generation never hits.
        let mut stale = key(0, 0);
        stale.owner_generation = GEN + 1;
        assert_eq!(catalog.pool_of(&stale), None);

        assert_eq!(
            catalog.keys_in_pool(lloyd_pool).collect::<Vec<_>>(),
            vec![key(0, 1)]
        );
        assert_eq!(
            catalog.keys_in_pool(mq4_pool).collect::<Vec<_>>(),
            vec![key(0, 0), key(0, 2), key(1, 0), key(1, 1)]
        );
        assert_eq!(
            catalog.cacheable_keys_in_pool(mq4_pool).collect::<Vec<_>>(),
            vec![key(0, 2), key(1, 1)]
        );
        let demands = catalog.pool_demands();
        assert_eq!(
            demands,
            vec![
                PoolDemand {
                    pool: mq4_pool,
                    bundles: 4,
                    pinned: 2,
                    cacheable: 2,
                    slot_stride: 96,
                },
                PoolDemand {
                    pool: lloyd_pool,
                    bundles: 1,
                    pinned: 0,
                    cacheable: 1,
                    slot_stride: 80,
                },
            ]
        );
        assert_eq!(catalog.pools_needing_slots().len(), 2);

        // Exact slot layout of each class: segments in order, each at its alignment.
        let class = catalog.class_of(mq4_pool).unwrap();
        assert_eq!(class.kernel_abi().0, "qwen4.mq4.v1");
        assert_eq!(class.alignment(), 16);
        assert_eq!(class.slot_bytes(), 96);
        assert_eq!(class.slot_stride(), 96);
        assert_eq!(class.segments()[0].slot_offset, 0);
        assert_eq!(class.segments()[1].slot_offset, 64);
        assert_eq!(class.segments()[0].sidecars, vec![0]);
        assert_eq!(class.aux().len(), 1);
        let lloyd_class = catalog.class_of(lloyd_pool).unwrap();
        // 48 -> next 16-aligned offset 48, down 32: 80 bytes; no shared reinterpretation.
        assert_eq!(lloyd_class.segments()[1].slot_offset, 48);
        assert_eq!(lloyd_class.slot_bytes(), 80);
        assert_ne!(class.digest(), lloyd_class.digest());
        assert_eq!(class.slot_extent(2, 1), Some((2 * 96 + 64, 32)));
        assert_eq!(class.slot_extent(0, 2), None);
        assert_eq!(
            class.slot_extent(u32::MAX, 0).map(|(o, _)| o),
            Some(u64::from(u32::MAX) * 96)
        );

        // The copies of one fill: canonical extents into slot offsets.
        let copies: Vec<_> = catalog.copy_extents(&key(0, 2)).unwrap().collect();
        assert_eq!(
            copies,
            vec![
                CopyExtent {
                    source: OwnerExtent {
                        region: HOST,
                        offset: 512,
                        len: 64
                    },
                    slot_offset: 0
                },
                CopyExtent {
                    source: OwnerExtent {
                        region: HOST,
                        offset: 640,
                        len: 32
                    },
                    slot_offset: 64
                },
            ]
        );
        assert!(catalog.copy_extents(&stale).is_none());
    }

    #[test]
    fn classes_differ_by_abi_alignment_padding_shape_and_sidecar_wiring() {
        let base = build(vec![group(0, &[(0, mq4(), false)])]).unwrap();
        let base_digest = *base.class_of(PoolId(0)).unwrap().digest();
        let variants: Vec<Box<dyn Fn(&mut GroupSpec)>> = vec![
            Box::new(|g: &mut GroupSpec| g.kernel_abi = KernelAbi("qwen4.mq4.v2".into())),
            Box::new(|g: &mut GroupSpec| g.bundles[0].segments[0].alignment = 32),
            Box::new(|g: &mut GroupSpec| g.bundles[0].segments[0].row_stride = 16),
            Box::new(|g: &mut GroupSpec| g.bundles[0].segments[0].basis = "fwht-g128".into()),
            Box::new(|g: &mut GroupSpec| {
                g.bundles[0].segments[0].parts[0].logical_shape = vec![8, 4]
            }),
            Box::new(|g: &mut GroupSpec| g.bundles[0].segments[0].sidecar_names = Vec::new()),
            Box::new(|g: &mut GroupSpec| g.aux[0].quant_tag = "f16".into()),
            Box::new(|g: &mut GroupSpec| g.aux[0].extent.len = 32),
            // The same payload but a longer down segment (padding): a different class.
            Box::new(|g: &mut GroupSpec| {
                g.bundles[0].segments[1].extent.len = 48;
                g.bundles[0].segments[1].parts[0].bytes = 48;
            }),
        ];
        let mut seen = vec![base_digest];
        for (index, vary) in variants.iter().enumerate() {
            let mut groups = vec![group(0, &[(0, mq4(), false)])];
            vary(&mut groups[0]);
            let catalog = build(groups).unwrap_or_else(|e| panic!("variant {index}: {e}"));
            let digest = *catalog.class_of(PoolId(0)).unwrap().digest();
            assert!(!seen.contains(&digest), "variant {index} aliases a class");
            seen.push(digest);
        }
        // The auxiliary NAME and fingerprint are identity, not class: a layer with the same aux
        // format under another name shares the pool.
        let mut renamed = group(1, &[(0, mq4(), false)]);
        renamed.aux[0].source_name = "rot.layer1".into();
        renamed.bundles[0].segments[0].sidecar_names = vec!["rot.layer1".into()];
        let catalog = build(vec![group(0, &[(0, mq4(), false)]), renamed]).unwrap();
        assert_eq!(catalog.classes().len(), 1);
        assert_eq!(catalog.pool_of(&key(1, 0)), Some(PoolId(0)));
    }

    // --- plan facts ---------------------------------------------------------------------------

    #[test]
    fn zero_tail_and_full_resident_plan_facts() {
        // Every bundle pinned: a static-only plan, no pool needs a slot.
        let resident = build(vec![
            group(0, &[(0, mq4(), true), (1, mq4(), true)]),
            group(1, &[(0, mq4(), true)]),
        ])
        .unwrap();
        let facts = resident.facts();
        assert!(facts.is_static_only());
        assert!(!facts.is_fully_cacheable());
        assert_eq!(
            facts,
            CatalogFacts {
                groups: 2,
                bundles: 3,
                pinned: 3,
                cacheable: 0,
                pinned_bytes: 3 * 96,
                cacheable_bytes: 0,
                pools: 1,
            }
        );
        assert!(resident.pools_needing_slots().is_empty());
        assert_eq!(resident.cacheable_keys_in_pool(PoolId(0)).count(), 0);

        // Nothing pinned: every bundle cacheable.
        let cold = build(base_groups()).unwrap();
        // base has one pinned (group 0 expert 0)
        let facts = cold.facts();
        assert_eq!((facts.pinned, facts.cacheable), (1, 4));
        assert_eq!(facts.cacheable_bytes, 4 * 96);
        assert!(!facts.is_static_only());
        let none_pinned = build(vec![group(0, &[(0, mq4(), false)])]).unwrap();
        assert!(none_pinned.facts().is_fully_cacheable());
        assert_eq!(none_pinned.pools_needing_slots().len(), 1);
    }

    #[test]
    fn unit_plan_descriptor_supplies_pinned_flags_and_geometry() {
        // 1 rank, 2 layers x 3 experts: layer 0 fully resident, layer 1 only expert 2.
        let plan =
            UnitPlanDescriptor::new(2, 3, 1, 64, 32, vec![true, true, true, false, false, true])
                .unwrap();
        assert_eq!(plan.unit_bytes(), 96);
        assert_eq!(plan.layer_resident(0, 1), Some(&[false, false, true][..]));
        assert_eq!(plan.layer_resident(1, 0), None);
        assert_eq!(plan.resident_units(0), Some(4));
        assert_eq!(plan.resident_units(1), None);

        let pinned_of = |layer: usize| plan.layer_resident(0, layer).unwrap().to_vec();
        let groups = (0..2u32)
            .map(|layer| {
                let flags = pinned_of(layer as usize);
                group(
                    layer,
                    &(0..3u32)
                        .map(|e| (e, mq4(), flags[e as usize]))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let catalog = build(groups).unwrap();
        assert_eq!(catalog.facts().pinned, plan.resident_units(0).unwrap());
        catalog.check_unit_geometry(&plan).unwrap();
        // A plan that charges other bytes disagrees with the catalog.
        let other = UnitPlanDescriptor::new(2, 3, 1, 80, 32, vec![false; 6]).unwrap();
        assert!(matches!(
            catalog.check_unit_geometry(&other).unwrap_err(),
            CatalogError::LayoutMismatch { .. }
        ));
        // Malformed descriptors.
        assert!(matches!(
            UnitPlanDescriptor::new(2, 3, 1, 64, 32, vec![false; 5]).unwrap_err(),
            CatalogError::UnitPlanInconsistent { .. }
        ));
        assert!(matches!(
            UnitPlanDescriptor::new(0, 3, 1, 64, 32, Vec::new()).unwrap_err(),
            CatalogError::UnitPlanInconsistent { .. }
        ));
        assert!(matches!(
            UnitPlanDescriptor::new(1, 1, 1, u64::MAX, 1, vec![false]).unwrap_err(),
            CatalogError::Overflow { .. }
        ));
    }

    // --- identity (A2.3) ----------------------------------------------------------------------

    fn identity_of(groups: Vec<GroupSpec>, tables: &[TableExtent]) -> CatalogIdentity {
        build(groups).unwrap().identity(tables).unwrap()
    }

    #[test]
    fn identity_is_stable_under_remap() {
        let base = build(base_groups()).unwrap();
        let identity = base.identity(&tables()).unwrap();
        assert_eq!(identity, base.identity(&tables()).unwrap());
        // Declaration order (groups, bundles, aux, tables) is not identity.
        let mut shuffled = base_groups();
        shuffled.reverse();
        shuffled[1].bundles.reverse();
        let mut shuffled_tables = tables();
        shuffled_tables.reverse();
        assert_eq!(identity_of(shuffled, &shuffled_tables), identity);
        // A remap moves experts between slots and changes which bundles are resident: neither is
        // visible to the catalog identity (no slot pointer, no planned residency in it).
        let mut remapped = base_groups();
        remapped[0].bundles[0].pinned = false;
        remapped[0].bundles[1].pinned = true;
        remapped[1].bundles[0].pinned = true;
        assert_eq!(identity_of(remapped, &tables()), identity);
        // Labels are diagnostics.
        let mut relabelled = base_groups();
        relabelled[0].label = "renamed".into();
        assert_eq!(identity_of(relabelled, &tables()), identity);
        assert_eq!(identity.to_string().len(), 64);
    }

    #[test]
    fn identity_changes_with_any_canonical_change() {
        let base = identity_of(base_groups(), &tables());
        let mut seen = vec![base];
        let mut check = |label: &str, identity: CatalogIdentity| {
            assert!(
                !seen.contains(&identity),
                "{label} did not change the identity"
            );
            seen.push(identity);
        };

        // Owner identity and generation.
        let catalog = ExpertCatalog::new(OWNER + 1, GEN, regions(), base_groups()).unwrap();
        check("owner id", catalog.identity(&tables()).unwrap());
        let catalog = ExpertCatalog::new(OWNER, GEN + 1, regions(), base_groups()).unwrap();
        check("owner generation", catalog.identity(&tables()).unwrap());
        // Region size and name.
        let mut bigger = regions();
        bigger[0].len += 4096;
        let catalog = ExpertCatalog::new(OWNER, GEN, bigger, base_groups()).unwrap();
        check("region size", catalog.identity(&tables()).unwrap());
        let mut renamed = regions();
        renamed[1].name = "aux2".into();
        let catalog = ExpertCatalog::new(OWNER, GEN, renamed, base_groups()).unwrap();
        check("region name", catalog.identity(&tables()).unwrap());

        // Source generation of the whole group.
        let mut regenerated = base_groups();
        regenerated[1].source_fingerprint = "gen-2".into();
        for bundle in &mut regenerated[1].bundles {
            for segment in &mut bundle.segments {
                segment.fingerprint = "gen-2".into();
            }
        }
        regenerated[1].aux[0].fingerprint = "gen-2".into();
        check("source fingerprint", identity_of(regenerated, &tables()));

        // One canonical extent moves.
        let mut moved = base_groups();
        moved[0].bundles[1].segments[1].extent.offset += 16;
        check("segment extent", identity_of(moved, &tables()));
        // One source name.
        let mut rename = base_groups();
        rename[0].bundles[2].segments[1].source_name = "down2".into();
        check("source name", identity_of(rename, &tables()));
        // A codec / basis / shape / abi change.
        let mut codec = base_groups();
        codec[1].bundles[0].segments[0].quant_tag = "qt45".into();
        check("quant tag", identity_of(codec, &tables()));
        let mut basis = base_groups();
        basis[0].bundles[0].segments[1].basis = "fwht-g256".into();
        check("basis", identity_of(basis, &tables()));
        let mut shape = base_groups();
        shape[0].bundles[0].segments[0].parts[1].logical_shape = vec![8, 4];
        check("logical shape", identity_of(shape, &tables()));
        let mut abi = base_groups();
        abi[1].kernel_abi = KernelAbi("qwen4.mq4.v2".into());
        check("kernel abi", identity_of(abi, &tables()));
        // The auxiliary binding.
        let mut aux = base_groups();
        aux[0].aux[0].extent.offset = 128;
        check("aux extent", identity_of(aux, &tables()));
        let mut alias = base_groups();
        alias[0].aux[0].alias = Some(SegmentAlias {
            owner_source: "rot_owner".into(),
            byte_offset: 0,
        });
        check("aux alias", identity_of(alias, &tables()));
        // An expert appears.
        let mut more = base_groups();
        more[1].bundles.push(bundle(1, 2, &mq4(), false));
        check("expert added", identity_of(more, &tables()));
        // A group appears.
        let mut group_added = base_groups();
        group_added.push(group(2, &[(0, mq4(), false)]));
        check("group added", identity_of(group_added, &tables()));

        // Table address, size, presence and ordering.
        let mut address = tables();
        address[0].address += 0x100;
        check("table address", identity_of(base_groups(), &address));
        let mut bytes = tables();
        bytes[1].bytes += 8;
        check("table bytes", identity_of(base_groups(), &bytes));
        check("table removed", identity_of(base_groups(), &tables()[..2]));
        let mut swapped = tables();
        let (a, b) = (swapped[0].address, swapped[1].address);
        swapped[0].address = b;
        swapped[1].address = a;
        check("table swap", identity_of(base_groups(), &swapped));
        let mut regroup = tables();
        regroup[2].group_id = 0;
        regroup[2].table = 2;
        check("table group", identity_of(base_groups(), &regroup));
    }

    #[test]
    fn identity_refuses_bad_table_extents() {
        let catalog = build(base_groups()).unwrap();
        let mut unknown = tables();
        unknown[0].group_id = 9;
        assert_eq!(
            catalog.identity(&unknown).unwrap_err(),
            CatalogError::UnknownGroup { group_id: 9 }
        );
        let mut duplicate = tables();
        duplicate[1].table = 0;
        assert_eq!(
            catalog.identity(&duplicate).unwrap_err(),
            CatalogError::DuplicateTable {
                group_id: 0,
                table: 0
            }
        );
        let mut empty = tables();
        empty[0].bytes = 0;
        assert!(matches!(
            catalog.identity(&empty).unwrap_err(),
            CatalogError::TableExtentInvalid { .. }
        ));
        let mut overlap = tables();
        overlap[1].address = overlap[0].address + 0x200;
        assert_eq!(
            catalog.identity(&overlap).unwrap_err(),
            CatalogError::TableExtentOverlap {
                first: (0, 0),
                second: (0, 1)
            }
        );
        // No tables at all is a valid (catalog-only) identity.
        catalog.identity(&[]).unwrap();
    }

    // --- derivation from the sealed execution plan -------------------------------------------

    fn source(
        name: &str,
        shape: Vec<usize>,
        bytes: usize,
        stride: usize,
        align: usize,
    ) -> ExpertSourceMetadata {
        ExpertSourceMetadata::new(
            name,
            "fused-v1",
            shape,
            DType::F32,
            bytes,
            stride,
            align,
            "f32",
            "natural",
        )
    }

    /// A Single-device packed fused plan with a `rotation` sidecar (the sealed_moe fused fixture).
    fn packed_plan() -> ExpertExecutionPlan {
        let mesh = DeviceMesh::single().unwrap();
        let manifest = vec![
            WeightEntry::layer("router", 0, vec![2, 1], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "gate_up",
                0,
                vec![2, 4, 2],
                DType::F32,
                ShardPolicy::Replicate,
            ),
            WeightEntry::layer("down", 0, vec![2, 2, 2], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer(
                "rotation",
                0,
                vec![2, 1],
                DType::F32,
                ShardPolicy::Replicate,
            ),
        ];
        let spec = ExpertGroupSpec {
            group: "fused".into(),
            layer: Some(0),
            n_experts: 2,
            parallelism: ExpertParallelism::Single,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PackedFused {
                gate_up: "gate_up".into(),
                down: "down".into(),
                sidecars: vec!["rotation".into()],
            },
            resources: ExpertResourceRequirements::new(vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 8,
                };
                2
            ]),
            router: "router".into(),
            execution: "cpu.fused".into(),
        };
        let sources = vec![
            source("router", vec![2, 1], 8, 8, 8),
            source("gate_up", vec![2, 4, 2], 64, 8, 8),
            source("down", vec![2, 2, 2], 32, 8, 8),
            source("rotation", vec![2, 1], 8, 4, 4),
        ];
        let placement = crate::mesh_placement::BandPlacement::uniform(&mesh, 1).unwrap();
        let manifest_plan =
            crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1, &placement).unwrap();
        plan_expert_execution(&manifest, &manifest_plan, &[spec], &sources, &mesh, &[9])
            .unwrap()
            .pop()
            .unwrap()
    }

    fn placement(name: &str, region: u32, offset: u64) -> SourcePlacement {
        SourcePlacement {
            source_name: name.into(),
            region,
            offset,
        }
    }

    #[test]
    fn catalog_derives_from_a_packed_fused_execution_plan() {
        let plan = packed_plan();
        let placements = [
            placement("gate_up", 0, 0),
            placement("down", 0, 64),
            placement("rotation", 1, 0),
        ];
        let spec = GroupSpec::from_execution_plan(
            &plan,
            0,
            5,
            KernelAbi("fixture.v1".into()),
            &placements,
            Some(&[true, false][..]),
        )
        .unwrap();
        assert_eq!(spec.source_fingerprint, "fused-v1");
        assert_eq!(spec.aux.len(), 1);
        assert_eq!(
            spec.aux[0].extent,
            OwnerExtent {
                region: 1,
                offset: 0,
                len: 8
            }
        );
        assert_eq!(spec.bundles.len(), 2);
        // Fused gate/up: one segment, gate then up, 16 bytes each; down 16.
        let first = &spec.bundles[0];
        assert!(first.pinned);
        assert_eq!(first.segments.len(), 2);
        assert_eq!(first.segments[0].parts.len(), 2);
        assert_eq!(
            first.segments[0].extent,
            OwnerExtent {
                region: 0,
                offset: 0,
                len: 32
            }
        );
        assert_eq!(
            first.segments[1].extent,
            OwnerExtent {
                region: 0,
                offset: 64,
                len: 16
            }
        );
        assert_eq!(
            first.segments[0].sidecar_names,
            vec!["rotation".to_string()]
        );
        // Packed: expert 1 sits one per-expert slice further.
        let second = &spec.bundles[1];
        assert!(!second.pinned);
        assert_eq!(second.segments[0].extent.offset, 32);
        assert_eq!(second.segments[1].extent.offset, 80);

        let host = vec![
            OwnerRegion {
                name: "host".into(),
                len: 96,
            },
            OwnerRegion {
                name: "aux".into(),
                len: 8,
            },
        ];
        let catalog = ExpertCatalog::new(OWNER, GEN, host, vec![spec]).unwrap();
        assert_eq!(catalog.classes().len(), 1);
        assert_eq!(catalog.pool_of(&key(5, 0)), Some(PoolId(0)));
        assert_eq!(catalog.pool_of(&key(5, 1)), Some(PoolId(0)));
        let class = catalog.class_of(PoolId(0)).unwrap();
        assert_eq!(class.slot_bytes(), 48);
        assert_eq!(class.alignment(), 8);
        assert_eq!(class.segments()[1].slot_offset, 32);
        assert_eq!(catalog.group_label(5), Some("fused layer Some(0)"));
        let bundle = catalog.bundle(&key(5, 0)).unwrap();
        assert_eq!(bundle.projection_bytes(Projection::Gate), 16);
        assert_eq!(bundle.projection_bytes(Projection::Up), 16);
        assert_eq!(bundle.projection_bytes(Projection::Down), 16);
        assert_eq!(bundle.payload_bytes(), 48);
        let facts = catalog.facts();
        assert_eq!((facts.pinned, facts.cacheable), (1, 1));
        assert_eq!(facts.pinned_bytes, 48);

        // Placement / pinned-flag / rank errors.
        let missing = [placement("gate_up", 0, 0), placement("down", 0, 64)];
        assert_eq!(
            GroupSpec::from_execution_plan(
                &plan,
                0,
                5,
                KernelAbi("fixture.v1".into()),
                &missing,
                None
            )
            .unwrap_err(),
            CatalogError::MissingPlacement {
                source: "rotation".into()
            }
        );
        assert!(matches!(
            GroupSpec::from_execution_plan(
                &plan,
                0,
                5,
                KernelAbi("fixture.v1".into()),
                &placements,
                Some(&[true][..])
            )
            .unwrap_err(),
            CatalogError::PlanMismatch { .. }
        ));
        assert!(matches!(
            GroupSpec::from_execution_plan(
                &plan,
                3,
                5,
                KernelAbi("fixture.v1".into()),
                &placements,
                None
            )
            .unwrap_err(),
            CatalogError::PlanMismatch { .. }
        ));
        // A plan whose sidecar the loader placed past its region fails at catalog build.
        let tiny = vec![
            OwnerRegion {
                name: "host".into(),
                len: 96,
            },
            OwnerRegion {
                name: "aux".into(),
                len: 4,
            },
        ];
        let spec = GroupSpec::from_execution_plan(
            &plan,
            0,
            5,
            KernelAbi("fixture.v1".into()),
            &placements,
            None,
        )
        .unwrap();
        assert!(matches!(
            ExpertCatalog::new(OWNER, GEN, tiny, vec![spec]).unwrap_err(),
            CatalogError::OwnerExtentOutOfRange { .. }
        ));
    }

    /// A two-rank EP plan with per-expert separate gate/up/down sources and a shared sidecar
    /// (the sealed_moe per-expert fixture plus `rot`).
    fn per_expert_plan() -> ExpertExecutionPlan {
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).unwrap();
        let policy = ShardPolicy::ExpertSharded {
            n_experts: 2,
            assign: ExpertAssign::Stride,
        };
        let mut manifest = vec![
            WeightEntry::layer("router", 0, vec![2, 2], DType::F32, ShardPolicy::Replicate),
            WeightEntry::layer("rot", 0, vec![2, 1], DType::F32, ShardPolicy::Replicate),
        ];
        for name in ["gate0", "gate1", "up0", "up1", "down0", "down1"] {
            manifest.push(WeightEntry::layer(
                name,
                0,
                vec![2, 2, 2],
                DType::F32,
                policy.clone(),
            ));
        }
        let spec = ExpertGroupSpec {
            group: "per-expert".into(),
            layer: Some(0),
            n_experts: 2,
            parallelism: ExpertParallelism::ExpertParallel,
            assignment: ExpertAssign::Stride,
            source_layout: ExpertSourceLayout::PerExpertSeparate {
                gate: vec!["gate0".into(), "gate1".into()],
                up: vec!["up0".into(), "up1".into()],
                down: vec!["down0".into(), "down1".into()],
                sidecars: vec!["rot".into()],
            },
            resources: ExpertResourceRequirements::new(vec![
                ExpertProjectionResources {
                    gate_bytes: 16,
                    up_bytes: 16,
                    down_bytes: 16,
                    alignment: 8,
                };
                2
            ]),
            router: "router".into(),
            execution: "cpu.test".into(),
        };
        let mut sources: Vec<_> = ["gate0", "gate1", "up0", "up1", "down0", "down1", "router"]
            .into_iter()
            .map(|name| source(name, vec![2, 2], 16, 8, 8))
            .collect();
        sources.push(source("rot", vec![2, 1], 8, 4, 4));
        let placement = crate::mesh_placement::BandPlacement::uniform(&mesh, 1).unwrap();
        let manifest_plan =
            crate::weight_manifest::plan_manifest(&manifest, &[], &mesh, 1, &placement).unwrap();
        plan_expert_execution(&manifest, &manifest_plan, &[spec], &sources, &mesh, &[7, 2])
            .unwrap()
            .pop()
            .unwrap()
    }

    #[test]
    fn catalog_derives_a_rank_local_group_from_per_expert_sources() {
        let plan = per_expert_plan();
        let placements = [
            placement("gate0", 0, 0),
            placement("gate1", 0, 16),
            placement("up0", 0, 32),
            placement("up1", 0, 48),
            placement("down0", 0, 64),
            placement("down1", 0, 80),
            placement("rot", 1, 0),
        ];
        // Rank 1 owns expert 1 only (stride ownership).
        let spec = GroupSpec::from_execution_plan(
            &plan,
            1,
            2,
            KernelAbi("fixture.v1".into()),
            &placements,
            None,
        )
        .unwrap();
        assert_eq!(spec.bundles.len(), 1);
        let bundle = &spec.bundles[0];
        assert_eq!(bundle.expert_id, 1);
        assert!(!bundle.pinned);
        // Separate sources: gate, up, down are three segments at their own placements.
        let extents: Vec<_> = bundle.segments.iter().map(|s| s.extent.offset).collect();
        assert_eq!(extents, vec![16, 48, 80]);
        assert!(bundle.segments.iter().all(|s| s.parts.len() == 1));
        // The sidecar is a whole-source auxiliary binding.
        assert_eq!(
            spec.aux[0].extent,
            OwnerExtent {
                region: 1,
                offset: 0,
                len: 8
            }
        );
        let host = vec![
            OwnerRegion {
                name: "host".into(),
                len: 96,
            },
            OwnerRegion {
                name: "aux".into(),
                len: 8,
            },
        ];
        let catalog = ExpertCatalog::new(OWNER, GEN, host, vec![spec]).unwrap();
        assert_eq!(catalog.keys().collect::<Vec<_>>(), vec![key(2, 1)]);
        let class = catalog.class_of(PoolId(0)).unwrap();
        assert_eq!(class.segments().len(), 3);
        assert_eq!(class.slot_bytes(), 48);
        // Rank 0 holds expert 0: same layout, so the same class digest.
        let rank0 = GroupSpec::from_execution_plan(
            &plan,
            0,
            3,
            KernelAbi("fixture.v1".into()),
            &placements,
            None,
        )
        .unwrap();
        assert_eq!(rank0.bundles[0].expert_id, 0);
        let both = ExpertCatalog::new(
            OWNER,
            GEN,
            vec![
                OwnerRegion {
                    name: "host".into(),
                    len: 96,
                },
                OwnerRegion {
                    name: "aux".into(),
                    len: 8,
                },
            ],
            vec![
                GroupSpec::from_execution_plan(
                    &plan,
                    1,
                    2,
                    KernelAbi("fixture.v1".into()),
                    &placements,
                    None,
                )
                .unwrap(),
                rank0,
            ],
        );
        // Both groups bind the same `rot` bytes: the overlap check refuses the double binding.
        assert!(matches!(
            both.unwrap_err(),
            CatalogError::OwnerExtentOverlap { .. }
        ));
    }
}
