// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DS4 canonical routed-expert source of the shared expert cache (slice S6a, CPU only).
//!
//! DeepSeek V4 Flash keeps no host-mapped owner of its routed experts: the loader reads each
//! layer's `w1`/`w3`/`w2` tensors out of the HFQ into a temporary host `Vec` and uploads one
//! device blob per layer (`upload_layer_routed_experts*` in `arch.rs`), so a cache has nothing to
//! fill from. This module describes the canonical source the cache fills from and the admission
//! that charges its host RAM. The decision behind it (pinned-host mapped owner, not pread
//! demand) is `expert-cache/s6/S6a-DECISION.md`.
//!
//! # What it builds
//!
//! [`Ds4ExpertSource::build`] reads the HFQ tensor index (never a payload byte) and returns:
//!
//! * the **file map**: for every routed expert of every group, the exact file extents of its
//!   `w1`, `w3` and `w2` tensors (`FileExtent`: file index, absolute offset, length), each
//!   checked against the geometry's byte count, the routed quant type (19, MQ2-Lloyd) and the
//!   file length. It feeds initial materialization of the canonical owner and an explicit pread
//!   mode ([`Ds4ExpertSource::file_copies`], the segments `PreadSource::fetch_into_batch` takes);
//! * the **catalog**: an [`ExpertCatalog`] over the canonical pinned-host owner regions the
//!   loader allocates (two per group, exactly the two blobs the resident loader packs): region
//!   `2i` holds `n_experts` fused `w1 || w3` gate/up carriers of `2 x w1` bytes, region `2i + 1`
//!   holds `n_experts` `w2` slices. A bundle is one fused `[Gate, Up]` segment plus one `[Down]`
//!   segment, so a slot is the exact `gate_up || down` image the routed kernels read through
//!   their two pointer tables;
//! * the **host charge** ([`Ds4ExpertSource::host_charge`]) and admission
//!   ([`expert_cache_support`], [`check_host_ram`] on the shared
//!   [`hipfire_runtime::offload::PinnableHostRam`] measure, [`Ds4ExpertSource::admit`]).
//!
//! # Groups and keys
//!
//! `ExpertKey { owner_id, owner_generation, group_id, expert_id }`: trunk layer `l` is group
//! `l`; the appended MTP head (`mtp.0`) is group `trunk_layers`, so its label can never be
//! mistaken for a trunk index. The expert id is the router's id; with no REAP keep-map (refused)
//! it is also the HFQ tensor index. A key of another owner id or generation never resolves
//! ([`Ds4SourceError::StaleOwner`]).
//!
//! # Refusals (named, never a degraded source)
//!
//! A missing tensor, a wrong quant type, a length that is not the geometry's, an extent that
//! reaches past the file or overlaps another expert's, an HFQ overlay (its tensors live in a
//! second file this index does not name), a keep-map, expert parallelism, a heterogeneous load,
//! unified memory, tp/pp > 1 and a host without room are each a distinct error that names the
//! file, group, expert and tensor. [`hipfire_config::memory::spill_support`] is deliberately
//! untouched: it gates dense-layer spill and still refuses arch 9, so enabling expert caching
//! cannot admit arbitrary dense offload.
//!
//! No forward-path wiring lives here: binding the catalog to `RoutedSeal`, the tables and the
//! window protocol is S6b.

use std::fmt;

use hipfire_runtime::expert_cache::catalog::{
    BundleSpec, CatalogError, ExpertCatalog, GroupSpec, KernelAbi, OwnerExtent, OwnerRegion,
    Projection, ProjectionPart, SegmentSpec, TableExtent,
};
use hipfire_runtime::expert_cache::ExpertKey;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::offload::{HostRamShortfall, PinnableHostRam};
use rdna_compute::DType;

/// `arch_id` of DeepSeek V4 Flash.
pub const DS4_ARCH_ID: u32 = 9;
/// HFQ `quant_type` of a routed expert (MQ2G256Lloyd, `QT_MQ2_LLOYD` in `arch.rs`). Identical in
/// the MQ2-Lloyd and MQ2R SKUs: MQ2R's E8 tensors are dense, its routed experts stay qt 19.
pub const ROUTED_QUANT_TYPE: u8 = 19;
/// Dtype the routed kernels read (`routed_seal.rs` `DTYPE`).
pub const ROUTED_DTYPE: DType = DType::MQ2G256Lloyd;
/// Group size of the dtype: a row's `k` must be a multiple of it.
const GROUP_K: usize = 256;
/// Slot segment alignment: `hipMalloc` granularity, which the resident blobs already satisfy
/// (both strides are multiples of 256).
const SLOT_ALIGNMENT: u64 = 256;
/// Kernel ABI identity of the DS4 routed MQ2-Lloyd gate/up + down pointer-table kernels.
pub const KERNEL_ABI: &str = "deepseek4.routed-mq2-lloyd.gate-up-down.v1";
const QUANT_TAG: &str = "mq2g256-lloyd";
const BASIS: &str = "mq-fwht256-rotated";
/// Slack `hipHostMalloc` mappings carry past their logical size (`HOST_TAIL_PAD_BYTES` in
/// `rdna-compute/src/dispatch.rs`, private there; mirrored, charged per blob).
pub const HOST_TAIL_PAD_BYTES: u64 = 1 << 20;
/// Pointer table index of the gate/up table in [`Ds4ExpertSource::table_extent`].
pub const TABLE_GATE_UP: u32 = 0;
/// Pointer table index of the down table.
pub const TABLE_DOWN: u32 = 1;

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// The expert dimensions a source is built for (from `DeepseekV4Config`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ds4Geometry {
    pub hidden: usize,
    pub moe_intermediate: usize,
    /// Routed experts per layer (`n_routed_experts`).
    pub n_experts: usize,
    /// Trunk layers whose experts the source covers (`num_hidden_layers`).
    pub trunk_layers: usize,
}

/// Exact encoded bytes of one expert's projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertBytes {
    /// `w1` (gate): `[mi, hidden]`. `w3` (up) is the same size.
    pub gate: u64,
    pub up: u64,
    /// `w2` (down): `[hidden, mi]`.
    pub down: u64,
}

impl ExpertBytes {
    /// The fused gate/up carrier (`w1 || w3`).
    pub fn gate_up(&self) -> u64 {
        self.gate + self.up
    }

    /// One whole bundle: `w1 + w3 + w2`.
    pub fn bundle(&self) -> u64 {
        self.gate + self.up + self.down
    }
}

impl Ds4Geometry {
    fn check(&self) -> Result<(), Ds4SourceError> {
        let bad = |why: &'static str| Err(Ds4SourceError::Geometry { why });
        if self.n_experts == 0 || self.trunk_layers == 0 {
            return bad("no routed experts or no layers");
        }
        if u32::try_from(self.n_experts).is_err() || u32::try_from(self.trunk_layers + 1).is_err() {
            return bad("expert or layer count exceeds u32");
        }
        if self.hidden == 0
            || self.moe_intermediate == 0
            || !self.hidden.is_multiple_of(GROUP_K)
            || !self.moe_intermediate.is_multiple_of(GROUP_K)
        {
            return bad(
                "hidden / moe_intermediate must be non-zero multiples of the 256-wide MQ2 group",
            );
        }
        Ok(())
    }

    /// Encoded bytes of one expert at [`ROUTED_DTYPE`].
    pub fn expert_bytes(&self) -> Result<ExpertBytes, Ds4SourceError> {
        self.check()?;
        let overflow = || Ds4SourceError::Overflow {
            what: "expert projection bytes",
        };
        let row = |k: usize| ROUTED_DTYPE.row_bytes(k).ok_or_else(overflow);
        let gate = (self.moe_intermediate as u64)
            .checked_mul(row(self.hidden)? as u64)
            .ok_or_else(overflow)?;
        let down = (self.hidden as u64)
            .checked_mul(row(self.moe_intermediate)? as u64)
            .ok_or_else(overflow)?;
        Ok(ExpertBytes {
            gate,
            up: gate,
            down,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Tensor index (the HFQ, or a fake)
// ---------------------------------------------------------------------------------------------

/// One tensor's header facts (never its payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorFacts {
    /// Absolute file offset of the payload.
    pub offset: u64,
    pub len: u64,
    pub quant_type: u8,
}

/// The tensor index of one model file.
pub trait TensorIndex {
    /// Diagnostic name of the file.
    fn label(&self) -> String;
    /// Identity of the file's content (the source generation of every segment read from it).
    fn fingerprint(&self) -> String;
    /// Sealed length of the file.
    fn file_len(&self) -> u64;
    /// An attached overlay shadows tensors by name from a second file: not described by one
    /// file index.
    fn is_overlaid(&self) -> bool;
    fn tensor(&self, name: &str) -> Option<TensorFacts>;
}

/// [`TensorIndex`] over an open [`HfqFile`].
pub struct HfqIndex<'a>(pub &'a HfqFile);

impl TensorIndex for HfqIndex<'_> {
    fn label(&self) -> String {
        self.0.path().display().to_string()
    }

    fn fingerprint(&self) -> String {
        use fmt::Write;
        self.0
            .content_digest()
            .iter()
            .fold(String::with_capacity(64), |mut out, byte| {
                let _ = write!(out, "{byte:02x}");
                out
            })
    }

    fn file_len(&self) -> u64 {
        self.0.file_len()
    }

    fn is_overlaid(&self) -> bool {
        self.0.has_overlay()
    }

    fn tensor(&self, name: &str) -> Option<TensorFacts> {
        let info = self.0.find_tensor_info(name)?;
        Some(TensorFacts {
            offset: u64::try_from(info.data_offset).ok()?,
            len: u64::try_from(info.data_size).ok()?,
            quant_type: info.quant_type,
        })
    }
}

/// Where the MTP head's routed experts come from.
pub enum MtpSource<'a> {
    /// No MTP head.
    None,
    /// In-band: `mtp.0` lives in the trunk file.
    InBand,
    /// The `-mtp` addon HFQ (its own file, its own source generation).
    Addon(&'a dyn TensorIndex),
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Every refusal of building or using a source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ds4SourceError {
    Geometry {
        why: &'static str,
    },
    /// An HFQ overlay is attached: its tensors shadow the base's by name from another file.
    Overlay {
        file: String,
    },
    /// A routed-expert tensor is absent from the file.
    MissingTensor {
        file: String,
        group_id: u32,
        expert_id: u32,
        name: String,
    },
    WrongQuant {
        file: String,
        group_id: u32,
        expert_id: u32,
        name: String,
        found: u8,
        expected: u8,
    },
    /// The tensor's byte count is not the geometry's (a ragged or truncated expert).
    WrongLength {
        file: String,
        group_id: u32,
        expert_id: u32,
        name: String,
        found: u64,
        expected: u64,
    },
    /// The tensor's extent reaches past the sealed file (a short or truncated file).
    PastEndOfFile {
        file: String,
        group_id: u32,
        expert_id: u32,
        name: String,
        offset: u64,
        len: u64,
        file_len: u64,
    },
    /// Two tensors of the source share bytes of one file.
    Overlap {
        file: String,
        first: String,
        second: String,
        offset: u64,
    },
    Catalog(CatalogError),
    /// A key the source does not describe (unknown group or expert).
    UnknownKey {
        key: ExpertKey,
    },
    /// A key of another owner id or generation.
    StaleOwner {
        key: ExpertKey,
        owner_id: u64,
        owner_generation: u64,
    },
    Overflow {
        what: &'static str,
    },
}

impl fmt::Display for Ds4SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry { why } => write!(f, "deepseek4 expert source: bad geometry: {why}"),
            Self::Overlay { file } => write!(
                f,
                "deepseek4 expert source: {file} has an HFQ overlay attached; its tensors shadow \
                 the base from a second file, which a single-file source cannot describe"
            ),
            Self::MissingTensor {
                file,
                group_id,
                expert_id,
                name,
            } => write!(
                f,
                "deepseek4 expert source: group {group_id} expert {expert_id}: tensor '{name}' \
                 is missing from {file}"
            ),
            Self::WrongQuant {
                file,
                group_id,
                expert_id,
                name,
                found,
                expected,
            } => write!(
                f,
                "deepseek4 expert source: group {group_id} expert {expert_id}: '{name}' in {file} \
                 has quant type {found}, routed experts are quant type {expected} (MQ2-Lloyd)"
            ),
            Self::WrongLength {
                file,
                group_id,
                expert_id,
                name,
                found,
                expected,
            } => write!(
                f,
                "deepseek4 expert source: group {group_id} expert {expert_id}: '{name}' in {file} \
                 is {found} bytes, the geometry requires {expected}"
            ),
            Self::PastEndOfFile {
                file,
                group_id,
                expert_id,
                name,
                offset,
                len,
                file_len,
            } => write!(
                f,
                "deepseek4 expert source: group {group_id} expert {expert_id}: '{name}' \
                 ({len} bytes at offset {offset}) reaches past the end of {file} \
                 ({file_len} bytes): truncated or corrupt file"
            ),
            Self::Overlap {
                file,
                first,
                second,
                offset,
            } => write!(
                f,
                "deepseek4 expert source: '{first}' and '{second}' share bytes of {file} at \
                 offset {offset}"
            ),
            Self::Catalog(error) => write!(f, "deepseek4 expert source: {error}"),
            Self::UnknownKey { key } => write!(
                f,
                "deepseek4 expert source: no expert for group {} expert {}",
                key.group_id, key.expert_id
            ),
            Self::StaleOwner {
                key,
                owner_id,
                owner_generation,
            } => write!(
                f,
                "deepseek4 expert source: group {} expert {} is bound to owner {} generation {}, \
                 this source is owner {owner_id} generation {owner_generation}",
                key.group_id, key.expert_id, key.owner_id, key.owner_generation
            ),
            Self::Overflow { what } => write!(f, "deepseek4 expert source: {what} overflows"),
        }
    }
}

impl std::error::Error for Ds4SourceError {}

impl From<CatalogError> for Ds4SourceError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

// ---------------------------------------------------------------------------------------------
// The source
// ---------------------------------------------------------------------------------------------

/// A byte range of one model file (`file` indexes [`Ds4ExpertSource::files`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileExtent {
    pub file: u32,
    pub offset: u64,
    pub len: u64,
}

/// The file extents of one expert's three tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundleFiles {
    /// `w1` (gate).
    pub gate: FileExtent,
    /// `w3` (up).
    pub up: FileExtent,
    /// `w2` (down).
    pub down: FileExtent,
}

/// Which projection a [`FileCopy`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Gate,
    Up,
    Down,
}

/// One file-to-slot copy of a fill: the file bytes and where they land in a slot image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileCopy {
    pub part: Part,
    pub extent: FileExtent,
    /// Byte offset inside the slot (`gate_up || down`, the catalog's layout class).
    pub slot_offset: u64,
}

/// One model file of the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceFile {
    pub label: String,
    pub fingerprint: String,
    pub len: u64,
}

/// One group (trunk layer or MTP head) of the source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupSource {
    group_id: u32,
    prefix: String,
    file: u32,
    gate_up_region: u32,
    down_region: u32,
    /// Indexed by expert id.
    bundles: Vec<BundleFiles>,
}

impl GroupSource {
    pub fn group_id(&self) -> u32 {
        self.group_id
    }

    /// HFQ tensor prefix (`layers.{l}` / `mtp.0`).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Index of the file the group's tensors live in.
    pub fn file(&self) -> u32 {
        self.file
    }

    /// Catalog owner region holding the group's fused gate/up carriers.
    pub fn gate_up_region(&self) -> u32 {
        self.gate_up_region
    }

    /// Catalog owner region holding the group's down slices.
    pub fn down_region(&self) -> u32 {
        self.down_region
    }

    pub fn bundle(&self, expert_id: u32) -> Option<&BundleFiles> {
        self.bundles.get(expert_id as usize)
    }

    pub fn experts(&self) -> usize {
        self.bundles.len()
    }
}

/// Pinned host RAM the canonical owner takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCharge {
    /// Logical blob bytes (gate/up and down blobs of every group).
    pub blob_bytes: u64,
    /// `HOST_TAIL_PAD_BYTES` per blob.
    pub tail_pad_bytes: u64,
    /// Host allocations (two per group).
    pub allocations: usize,
}

impl HostCharge {
    pub fn total(&self) -> u64 {
        self.blob_bytes.saturating_add(self.tail_pad_bytes)
    }
}

/// The canonical source of every routed expert of one DS4 owner generation.
#[derive(Clone, Debug)]
pub struct Ds4ExpertSource {
    geometry: Ds4Geometry,
    bytes: ExpertBytes,
    files: Vec<SourceFile>,
    groups: Vec<GroupSource>,
    catalog: ExpertCatalog,
}

impl Ds4ExpertSource {
    /// Check the tensor index of every expert of every group and build the source. Reads no
    /// payload byte. Group `l` is trunk layer `l`; the MTP head, when present, is group
    /// `geometry.trunk_layers`.
    pub fn build(
        owner_id: u64,
        owner_generation: u64,
        geometry: Ds4Geometry,
        trunk: &dyn TensorIndex,
        mtp: MtpSource<'_>,
    ) -> Result<Self, Ds4SourceError> {
        let bytes = geometry.expert_bytes()?;
        let mut indexes: Vec<&dyn TensorIndex> = vec![trunk];
        let mut plan: Vec<(u32, String, u32)> = (0..geometry.trunk_layers)
            .map(|layer| (layer as u32, format!("layers.{layer}"), 0u32))
            .collect();
        let mtp_group = geometry.trunk_layers as u32;
        match mtp {
            MtpSource::None => {}
            MtpSource::InBand => plan.push((mtp_group, "mtp.0".to_string(), 0)),
            MtpSource::Addon(addon) => {
                indexes.push(addon);
                plan.push((mtp_group, "mtp.0".to_string(), 1));
            }
        }
        let mut files = Vec::with_capacity(indexes.len());
        for index in &indexes {
            if index.is_overlaid() {
                return Err(Ds4SourceError::Overlay {
                    file: index.label(),
                });
            }
            files.push(SourceFile {
                label: index.label(),
                fingerprint: index.fingerprint(),
                len: index.file_len(),
            });
        }

        let n = geometry.n_experts;
        let mut groups = Vec::with_capacity(plan.len());
        for (position, (group_id, prefix, file)) in plan.into_iter().enumerate() {
            let index = indexes[file as usize];
            let source_file = &files[file as usize];
            let mut bundles = Vec::with_capacity(n);
            for expert in 0..n {
                let expert_id = expert as u32;
                let part = |part: &str, expected: u64| {
                    resolve(
                        index,
                        source_file,
                        file,
                        group_id,
                        expert_id,
                        format!("{prefix}.ffn.experts.{expert}.{part}.weight"),
                        expected,
                    )
                };
                bundles.push(BundleFiles {
                    gate: part("w1", bytes.gate)?,
                    up: part("w3", bytes.up)?,
                    down: part("w2", bytes.down)?,
                });
            }
            groups.push(GroupSource {
                group_id,
                prefix,
                file,
                gate_up_region: 2 * position as u32,
                down_region: 2 * position as u32 + 1,
                bundles,
            });
        }
        check_file_overlap(&files, &groups)?;

        let (regions, specs) = catalog_inputs(&geometry, &bytes, &files, &groups)?;
        let catalog = ExpertCatalog::new(owner_id, owner_generation, regions, specs)?;
        Ok(Self {
            geometry,
            bytes,
            files,
            groups,
            catalog,
        })
    }

    pub fn geometry(&self) -> &Ds4Geometry {
        &self.geometry
    }

    pub fn expert_bytes(&self) -> &ExpertBytes {
        &self.bytes
    }

    /// The model files the extents index (0 = the trunk file; 1 = the MTP addon, if any).
    pub fn files(&self) -> &[SourceFile] {
        &self.files
    }

    pub fn groups(&self) -> &[GroupSource] {
        &self.groups
    }

    pub fn group(&self, group_id: u32) -> Option<&GroupSource> {
        self.groups.iter().find(|group| group.group_id == group_id)
    }

    /// The group id of the MTP head, if the source has one.
    pub fn mtp_group_id(&self) -> Option<u32> {
        let id = self.geometry.trunk_layers as u32;
        self.group(id).map(|_| id)
    }

    pub fn catalog(&self) -> &ExpertCatalog {
        &self.catalog
    }

    pub fn owner_id(&self) -> u64 {
        self.catalog.owner_id()
    }

    pub fn owner_generation(&self) -> u64 {
        self.catalog.owner_generation()
    }

    /// The key of `(group_id, expert_id)` under this source's owner generation.
    pub fn key(&self, group_id: u32, expert_id: u32) -> ExpertKey {
        ExpertKey {
            owner_id: self.owner_id(),
            owner_generation: self.owner_generation(),
            group_id,
            expert_id,
        }
    }

    fn check_owner(&self, key: &ExpertKey) -> Result<(), Ds4SourceError> {
        if key.owner_id != self.owner_id() || key.owner_generation != self.owner_generation() {
            return Err(Ds4SourceError::StaleOwner {
                key: *key,
                owner_id: self.owner_id(),
                owner_generation: self.owner_generation(),
            });
        }
        Ok(())
    }

    /// The file extents of `key`'s three tensors. Refuses another owner generation.
    pub fn bundle_files(&self, key: &ExpertKey) -> Result<&BundleFiles, Ds4SourceError> {
        self.check_owner(key)?;
        self.group(key.group_id)
            .and_then(|group| group.bundle(key.expert_id))
            .ok_or(Ds4SourceError::UnknownKey { key: *key })
    }

    /// The three file-to-slot copies that fill `key`'s slot: `w1` and `w3` back to back at the
    /// gate/up segment, `w2` at the down segment (offsets from the catalog's layout class), the
    /// segments a pread demand fill hands `PreadSource::fetch_into_batch`.
    pub fn file_copies(&self, key: &ExpertKey) -> Result<[FileCopy; 3], Ds4SourceError> {
        let files = self.bundle_files(key)?;
        let unknown = || Ds4SourceError::UnknownKey { key: *key };
        let bundle = self.catalog.bundle(key).ok_or_else(unknown)?;
        let class = self.catalog.class_of(bundle.pool()).ok_or_else(unknown)?;
        let segments = class.segments();
        let (gate_up, down) = match segments {
            [gate_up, down] => (gate_up.slot_offset, down.slot_offset),
            _ => return Err(unknown()),
        };
        Ok([
            FileCopy {
                part: Part::Gate,
                extent: files.gate,
                slot_offset: gate_up,
            },
            FileCopy {
                part: Part::Up,
                extent: files.up,
                slot_offset: gate_up + files.gate.len,
            },
            FileCopy {
                part: Part::Down,
                extent: files.down,
                slot_offset: down,
            },
        ])
    }

    /// The catalog identity table extent of `group_id`'s pointer table at device `address`
    /// (`table` is [`TABLE_GATE_UP`] or [`TABLE_DOWN`]): one `u64` per expert.
    pub fn table_extent(
        &self,
        group_id: u32,
        table: u32,
        address: u64,
    ) -> Result<TableExtent, Ds4SourceError> {
        let group =
            self.group(group_id)
                .ok_or(Ds4SourceError::Catalog(CatalogError::UnknownGroup {
                    group_id,
                }))?;
        Ok(TableExtent {
            group_id,
            table,
            address,
            bytes: group.bundles.len() as u64 * 8,
        })
    }

    /// Pinned host RAM the canonical owner takes: every group's two blobs plus their tail pads.
    pub fn host_charge(&self) -> HostCharge {
        let per_group = self.geometry.n_experts as u64 * self.bytes.bundle();
        let allocations = 2 * self.groups.len();
        HostCharge {
            blob_bytes: per_group * self.groups.len() as u64,
            tail_pad_bytes: HOST_TAIL_PAD_BYTES * allocations as u64,
            allocations,
        }
    }

    /// Admit this source on `topology` against `ram`, the host RAM a pinning load can take
    /// ([`hipfire_runtime::offload::pinnable_host_ram`]): the topology must be one the canonical
    /// owner supports and the pinned charge plus headroom must fit.
    pub fn admit(
        &self,
        topology: &Topology,
        ram: PinnableHostRam,
    ) -> Result<HostCharge, Ds4Refusal> {
        expert_cache_support(topology)?;
        let charge = self.host_charge();
        check_host_ram(&charge, ram)?;
        Ok(charge)
    }
}

fn resolve(
    index: &dyn TensorIndex,
    file: &SourceFile,
    file_no: u32,
    group_id: u32,
    expert_id: u32,
    name: String,
    expected: u64,
) -> Result<FileExtent, Ds4SourceError> {
    let label = || file.label.clone();
    let facts = index
        .tensor(&name)
        .ok_or_else(|| Ds4SourceError::MissingTensor {
            file: label(),
            group_id,
            expert_id,
            name: name.clone(),
        })?;
    if facts.quant_type != ROUTED_QUANT_TYPE {
        return Err(Ds4SourceError::WrongQuant {
            file: label(),
            group_id,
            expert_id,
            name,
            found: facts.quant_type,
            expected: ROUTED_QUANT_TYPE,
        });
    }
    if facts.len != expected {
        return Err(Ds4SourceError::WrongLength {
            file: label(),
            group_id,
            expert_id,
            name,
            found: facts.len,
            expected,
        });
    }
    match facts.offset.checked_add(facts.len) {
        Some(end) if end <= file.len => Ok(FileExtent {
            file: file_no,
            offset: facts.offset,
            len: facts.len,
        }),
        _ => Err(Ds4SourceError::PastEndOfFile {
            file: label(),
            group_id,
            expert_id,
            name,
            offset: facts.offset,
            len: facts.len,
            file_len: file.len,
        }),
    }
}

/// No two expert tensors of one file share a byte: a duplicated or aliased offset would fill two
/// experts from one payload.
fn check_file_overlap(files: &[SourceFile], groups: &[GroupSource]) -> Result<(), Ds4SourceError> {
    let mut extents: Vec<(u32, u64, u64, u32, u32, &'static str)> = Vec::new();
    for group in groups {
        for (expert, bundle) in group.bundles.iter().enumerate() {
            for (part, extent) in [("w1", bundle.gate), ("w3", bundle.up), ("w2", bundle.down)] {
                extents.push((
                    extent.file,
                    extent.offset,
                    extent.offset + extent.len,
                    group.group_id,
                    expert as u32,
                    part,
                ));
            }
        }
    }
    extents.sort_by_key(|&(file, start, end, ..)| (file, start, end));
    let name = |group_id: u32, expert: u32, part: &str| {
        let prefix = groups
            .iter()
            .find(|group| group.group_id == group_id)
            .map(|group| group.prefix.as_str())
            .unwrap_or("?");
        format!("{prefix}.ffn.experts.{expert}.{part}.weight")
    };
    for pair in extents.windows(2) {
        let (file, _, previous_end, g0, e0, p0) = pair[0];
        let (file1, start, _, g1, e1, p1) = pair[1];
        if file == file1 && start < previous_end {
            return Err(Ds4SourceError::Overlap {
                file: files[file as usize].label.clone(),
                first: name(g0, e0, p0),
                second: name(g1, e1, p1),
                offset: start,
            });
        }
    }
    Ok(())
}

/// The owner regions and group specs of the canonical host owner.
fn catalog_inputs(
    geometry: &Ds4Geometry,
    bytes: &ExpertBytes,
    files: &[SourceFile],
    groups: &[GroupSource],
) -> Result<(Vec<OwnerRegion>, Vec<GroupSpec>), Ds4SourceError> {
    let overflow = |what: &'static str| Ds4SourceError::Overflow { what };
    let n = geometry.n_experts as u64;
    let gate_up_len = n
        .checked_mul(bytes.gate_up())
        .ok_or_else(|| overflow("gate/up blob bytes"))?;
    let down_len = n
        .checked_mul(bytes.down)
        .ok_or_else(|| overflow("down blob bytes"))?;
    let gate_row = ROUTED_DTYPE
        .row_bytes(geometry.hidden)
        .ok_or_else(|| overflow("gate row bytes"))? as u64;
    let down_row = ROUTED_DTYPE
        .row_bytes(geometry.moe_intermediate)
        .ok_or_else(|| overflow("down row bytes"))? as u64;
    let (mi, hidden) = (geometry.moe_intermediate, geometry.hidden);

    let mut regions = Vec::with_capacity(2 * groups.len());
    let mut specs = Vec::with_capacity(groups.len());
    for group in groups {
        regions.push(OwnerRegion {
            name: format!("{}.expert_gate_up_blob", group.prefix),
            len: gate_up_len,
        });
        regions.push(OwnerRegion {
            name: format!("{}.expert_w2_blob", group.prefix),
            len: down_len,
        });
        let fingerprint = files[group.file as usize].fingerprint.clone();
        let mut bundles = Vec::with_capacity(group.bundles.len());
        for expert in 0..group.bundles.len() as u64 {
            let gate_up = SegmentSpec {
                source_name: format!("{}.ffn.experts.{expert}.w1+w3", group.prefix),
                fingerprint: fingerprint.clone(),
                extent: OwnerExtent {
                    region: group.gate_up_region,
                    offset: expert * bytes.gate_up(),
                    len: bytes.gate_up(),
                },
                alias: None,
                dtype: ROUTED_DTYPE,
                quant_tag: QUANT_TAG.to_string(),
                basis: BASIS.to_string(),
                row_stride: gate_row,
                alignment: SLOT_ALIGNMENT,
                parts: vec![
                    ProjectionPart {
                        projection: Projection::Gate,
                        logical_shape: vec![mi, hidden],
                        bytes: bytes.gate,
                    },
                    ProjectionPart {
                        projection: Projection::Up,
                        logical_shape: vec![mi, hidden],
                        bytes: bytes.up,
                    },
                ],
                sidecar_names: Vec::new(),
            };
            let down = SegmentSpec {
                source_name: format!("{}.ffn.experts.{expert}.w2", group.prefix),
                fingerprint: fingerprint.clone(),
                extent: OwnerExtent {
                    region: group.down_region,
                    offset: expert * bytes.down,
                    len: bytes.down,
                },
                alias: None,
                dtype: ROUTED_DTYPE,
                quant_tag: QUANT_TAG.to_string(),
                basis: BASIS.to_string(),
                row_stride: down_row,
                alignment: SLOT_ALIGNMENT,
                parts: vec![ProjectionPart {
                    projection: Projection::Down,
                    logical_shape: vec![hidden, mi],
                    bytes: bytes.down,
                }],
                sidecar_names: Vec::new(),
            };
            bundles.push(BundleSpec {
                expert_id: expert as u32,
                pinned: false,
                segments: vec![gate_up, down],
            });
        }
        specs.push(GroupSpec {
            group_id: group.group_id,
            label: group.prefix.clone(),
            source_fingerprint: fingerprint,
            kernel_abi: KernelAbi(KERNEL_ABI.to_string()),
            aux: Vec::new(),
            bundles,
        });
    }
    Ok((regions, specs))
}

// ---------------------------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------------------------

/// The load facts expert caching is admitted on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    pub arch_id: u32,
    pub tp: usize,
    pub pp: usize,
    /// `Some(n)` with `n > 0` on an expert-parallel route.
    pub ep_experts: Option<usize>,
    /// A REAP keep-map is attached (compact expert ids, overlay tensors).
    pub reap_keep: bool,
    /// The mixed-GPU G2 base load (routed experts on a second device).
    pub heterogeneous: bool,
    /// Unified-memory device (Strix Halo class): the canonical host owner is the same memory the
    /// device reads, so a cache has no copy to hide and the owner would double the footprint.
    pub unified_memory: bool,
}

/// Why expert caching cannot run on this load. Stable names; callers own the wording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ds4Refusal {
    Arch(u32),
    Tp(usize),
    Pp(usize),
    ExpertParallel(usize),
    ReapKeepMap,
    Heterogeneous,
    UnifiedMemory,
    /// The pinned canonical owner (plus headroom) does not fit the host RAM available.
    HostRam(HostRamShortfall),
}

impl fmt::Display for Ds4Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "load refused: DeepSeek V4 expert caching ")?;
        match self {
            Self::Arch(id) => write!(f, "is implemented for arch id {DS4_ARCH_ID} only; this source is arch id {id}"),
            Self::Tp(tp) => write!(f, "supports the single-GPU topology only; tp={tp}"),
            Self::Pp(pp) => write!(f, "supports the single-GPU topology only; pp={pp}"),
            Self::ExpertParallel(n) => write!(
                f,
                "cannot be combined with expert parallelism ({n} experts pinned per rank)"
            ),
            Self::ReapKeepMap => write!(
                f,
                "cannot be combined with a REAP keep-map: compact expert ids and overlay tensors have no canonical host source"
            ),
            Self::Heterogeneous => write!(
                f,
                "cannot be combined with the mixed-GPU base load, which places routed experts on a second device"
            ),
            Self::UnifiedMemory => write!(
                f,
                "is not admitted on unified memory: the canonical host owner is the memory the device reads, so a cache hides no copy and the owner doubles the footprint"
            ),
            Self::HostRam(shortfall) => write!(
                f,
                "does not fit host RAM: {}",
                shortfall.describe("its pinned routed experts", "free host memory")
            ),
        }
    }
}

impl std::error::Error for Ds4Refusal {}

/// The expert-cache eligibility truth table of a DS4 load. First failing dimension wins (arch,
/// tp, pp, EP, REAP, heterogeneous, unified memory), the order of
/// [`hipfire_config::memory::spill_support`]. Pure: GPU-free.
///
/// This is the DS4 expert-only analogue of `spill_support`, which stays untouched: a dense-layer
/// spill (`memory.gpu_layer_budget`) is still refused for arch 9 and enabling expert caching
/// admits nothing else.
pub fn expert_cache_support(topology: &Topology) -> Result<(), Ds4Refusal> {
    if topology.arch_id != DS4_ARCH_ID {
        return Err(Ds4Refusal::Arch(topology.arch_id));
    }
    if topology.tp > 1 {
        return Err(Ds4Refusal::Tp(topology.tp));
    }
    if topology.pp > 1 {
        return Err(Ds4Refusal::Pp(topology.pp));
    }
    if let Some(experts) = topology.ep_experts {
        if experts > 0 {
            return Err(Ds4Refusal::ExpertParallel(experts));
        }
    }
    if topology.reap_keep {
        return Err(Ds4Refusal::ReapKeepMap);
    }
    if topology.heterogeneous {
        return Err(Ds4Refusal::Heterogeneous);
    }
    if topology.unified_memory {
        return Err(Ds4Refusal::UnifiedMemory);
    }
    Ok(())
}

/// Refuse before any allocation unless the pinned canonical owner leaves
/// [`hipfire_runtime::offload::HOST_RAM_HEADROOM_BYTES`] of `ram`
/// ([`PinnableHostRam::admit`]: `MemAvailable - CMA` plus the TTM page pool a GTT-backed pin
/// takes first). Pinned pages are not reclaimable, so over-committing them starves the host. An
/// unreadable `MemAvailable` refuses.
pub fn check_host_ram(charge: &HostCharge, ram: PinnableHostRam) -> Result<(), Ds4Refusal> {
    ram.admit(charge.total()).map_err(Ds4Refusal::HostRam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::offload::HOST_RAM_HEADROOM_BYTES;
    use std::collections::BTreeMap;

    const OWNER: u64 = 0xd54;
    const GEN: u64 = 7;
    /// Smallest legal geometry: one 256-wide group per row.
    const SMALL: Ds4Geometry = Ds4Geometry {
        hidden: 256,
        moe_intermediate: 256,
        n_experts: 6,
        trunk_layers: 3,
    };
    /// DeepSeek V4 Flash: `deepseek4.rs` test config (`hidden_size 4096`, `n_routed_experts 256`,
    /// `moe_intermediate_size 2048`, 43 layers).
    const FLASH: Ds4Geometry = Ds4Geometry {
        hidden: 4096,
        moe_intermediate: 2048,
        n_experts: 256,
        trunk_layers: 43,
    };

    struct FakeIndex {
        label: String,
        fingerprint: String,
        len: u64,
        overlaid: bool,
        tensors: BTreeMap<String, TensorFacts>,
    }

    impl TensorIndex for FakeIndex {
        fn label(&self) -> String {
            self.label.clone()
        }
        fn fingerprint(&self) -> String {
            self.fingerprint.clone()
        }
        fn file_len(&self) -> u64 {
            self.len
        }
        fn is_overlaid(&self) -> bool {
            self.overlaid
        }
        fn tensor(&self, name: &str) -> Option<TensorFacts> {
            self.tensors.get(name).copied()
        }
    }

    /// A file holding `prefixes` x `n` experts x (w1, w3, w2) laid out back to back after a
    /// header, with the geometry's exact sizes.
    fn fake(label: &str, geometry: &Ds4Geometry, prefixes: &[String]) -> FakeIndex {
        let bytes = geometry.expert_bytes().unwrap();
        let mut tensors = BTreeMap::new();
        let mut offset = 4096u64;
        for prefix in prefixes {
            for e in 0..geometry.n_experts {
                for (part, len) in [("w1", bytes.gate), ("w3", bytes.up), ("w2", bytes.down)] {
                    tensors.insert(
                        format!("{prefix}.ffn.experts.{e}.{part}.weight"),
                        TensorFacts {
                            offset,
                            len,
                            quant_type: ROUTED_QUANT_TYPE,
                        },
                    );
                    offset += len;
                }
            }
        }
        FakeIndex {
            label: label.to_string(),
            fingerprint: format!("fp-{label}"),
            len: offset,
            overlaid: false,
            tensors,
        }
    }

    fn trunk_prefixes(geometry: &Ds4Geometry) -> Vec<String> {
        (0..geometry.trunk_layers)
            .map(|l| format!("layers.{l}"))
            .collect()
    }

    fn trunk(geometry: &Ds4Geometry) -> FakeIndex {
        fake("base.hfq", geometry, &trunk_prefixes(geometry))
    }

    fn build_with(
        geometry: Ds4Geometry,
        trunk: &FakeIndex,
        mtp: MtpSource<'_>,
    ) -> Result<Ds4ExpertSource, Ds4SourceError> {
        Ds4ExpertSource::build(OWNER, GEN, geometry, trunk, mtp)
    }

    fn built(mtp: MtpSource<'_>) -> Ds4ExpertSource {
        // An in-band MTP head lives in the trunk file.
        let mut prefixes = trunk_prefixes(&SMALL);
        if matches!(mtp, MtpSource::InBand) {
            prefixes.push("mtp.0".to_string());
        }
        let index = fake("base.hfq", &SMALL, &prefixes);
        build_with(SMALL, &index, mtp).unwrap()
    }

    // ---- sizes -----------------------------------------------------------------------------

    #[test]
    fn flash_expert_bytes_are_the_shipped_checkpoint_strides() {
        // MQ2G256Lloyd: 72 bytes per 256-wide group. w1 = w3 = 2048 rows x 4096/256 x 72 and
        // w2 = 4096 rows x 2048/256 x 72, both 2_359_296 (the stride #870's tests pin).
        let bytes = FLASH.expert_bytes().unwrap();
        assert_eq!(bytes.gate, 2_359_296);
        assert_eq!(bytes.up, 2_359_296);
        assert_eq!(bytes.down, 2_359_296);
        assert_eq!(bytes.gate_up(), 4_718_592);
        assert_eq!(bytes.bundle(), 7_077_888);
        assert_eq!(256 * bytes.bundle(), 1_811_939_328, "one layer of experts");
        assert_eq!(43 * 256 * bytes.bundle(), 77_913_391_104, "trunk experts");
    }

    #[test]
    fn geometry_refuses_unaligned_or_empty_shapes() {
        for geometry in [
            Ds4Geometry { hidden: 0, ..SMALL },
            Ds4Geometry {
                moe_intermediate: 300,
                ..SMALL
            },
            Ds4Geometry {
                n_experts: 0,
                ..SMALL
            },
            Ds4Geometry {
                trunk_layers: 0,
                ..SMALL
            },
        ] {
            assert!(
                matches!(
                    geometry.expert_bytes(),
                    Err(Ds4SourceError::Geometry { .. })
                ),
                "{geometry:?}"
            );
        }
    }

    // ---- coverage --------------------------------------------------------------------------

    #[test]
    fn every_routed_expert_of_every_layer_appears_exactly_once() {
        let source = built(MtpSource::InBand);
        let groups = SMALL.trunk_layers + 1;
        let keys: Vec<ExpertKey> = source.catalog().keys().collect();
        assert_eq!(keys.len(), groups * SMALL.n_experts);
        let mut expected = Vec::new();
        for group in 0..groups as u32 {
            for expert in 0..SMALL.n_experts as u32 {
                expected.push(source.key(group, expert));
            }
        }
        // `keys()` is ordered by group then expert; an exact match proves no key is missing,
        // duplicated or extra.
        assert_eq!(keys, expected);
        assert_eq!(source.groups().len(), groups);
        // The file map covers the same set once: 3 tensors per expert, all distinct extents.
        let mut extents: Vec<(u32, u64)> = source
            .groups()
            .iter()
            .flat_map(|group| {
                (0..SMALL.n_experts as u32).flat_map(move |expert| {
                    let b = group.bundle(expert).unwrap();
                    [b.gate, b.up, b.down].map(|e| (e.file, e.offset))
                })
            })
            .collect();
        assert_eq!(extents.len(), groups * SMALL.n_experts * 3);
        extents.sort_unstable();
        extents.dedup();
        assert_eq!(
            extents.len(),
            groups * SMALL.n_experts * 3,
            "no tensor twice"
        );
        let facts = source.catalog().facts();
        assert_eq!(facts.bundles, groups * SMALL.n_experts);
        assert_eq!(facts.pinned, 0);
        assert_eq!(facts.pools, 1, "every DS4 bundle shares one layout class");
        assert_eq!(
            facts.cacheable_bytes,
            (groups * SMALL.n_experts) as u64 * source.expert_bytes().bundle()
        );
    }

    #[test]
    fn mtp_head_is_its_own_group_after_the_trunk() {
        let none = built(MtpSource::None);
        assert_eq!(none.mtp_group_id(), None);
        assert_eq!(none.groups().len(), SMALL.trunk_layers);
        let inband = built(MtpSource::InBand);
        assert_eq!(inband.mtp_group_id(), Some(SMALL.trunk_layers as u32));
        let head = inband.group(SMALL.trunk_layers as u32).unwrap();
        assert_eq!(head.prefix(), "mtp.0");
        assert_eq!(head.file(), 0, "in-band reads the trunk file");
        assert!(inband
            .catalog()
            .group_label(SMALL.trunk_layers as u32)
            .is_some_and(|label| label == "mtp.0"));
    }

    #[test]
    fn mtp_addon_is_a_second_file_with_its_own_source_generation() {
        let trunk = trunk(&SMALL);
        let addon = fake("mtp-addon.hfq", &SMALL, &["mtp.0".to_string()]);
        let source = build_with(SMALL, &trunk, MtpSource::Addon(&addon)).unwrap();
        assert_eq!(source.files().len(), 2);
        let head = source.group(SMALL.trunk_layers as u32).unwrap();
        assert_eq!(head.file(), 1);
        assert_eq!(head.bundle(2).unwrap().down.file, 1);
        assert_eq!(source.groups()[0].bundle(2).unwrap().down.file, 0);
        let catalog = source.catalog();
        assert_eq!(catalog.group_fingerprint(0), Some("fp-base.hfq"));
        assert_eq!(
            catalog.group_fingerprint(SMALL.trunk_layers as u32),
            Some("fp-mtp-addon.hfq")
        );
    }

    #[test]
    fn file_extents_are_the_index_extents() {
        let trunk = trunk(&SMALL);
        let source = build_with(SMALL, &trunk, MtpSource::None).unwrap();
        let facts = |name: &str| trunk.tensors[name];
        let b = source.bundle_files(&source.key(2, 4)).unwrap();
        for (extent, name) in [
            (b.gate, "layers.2.ffn.experts.4.w1.weight"),
            (b.up, "layers.2.ffn.experts.4.w3.weight"),
            (b.down, "layers.2.ffn.experts.4.w2.weight"),
        ] {
            assert_eq!(
                (extent.file, extent.offset, extent.len),
                (0, facts(name).offset, facts(name).len),
                "{name}"
            );
        }
    }

    #[test]
    fn file_copies_describe_the_exact_slot_image() {
        let source = built(MtpSource::None);
        let key = source.key(1, 3);
        let copies = source.file_copies(&key).unwrap();
        let bytes = source.expert_bytes();
        let class = source
            .catalog()
            .class_of(source.catalog().pool_of(&key).unwrap())
            .unwrap();
        assert_eq!(class.slot_bytes(), bytes.bundle());
        // w1 || w3 back to back, then w2: the gate_up carrier and the down slice the kernels read.
        assert_eq!(copies[0].part, Part::Gate);
        assert_eq!(copies[0].slot_offset, 0);
        assert_eq!(copies[1].part, Part::Up);
        assert_eq!(copies[1].slot_offset, bytes.gate);
        assert_eq!(copies[2].part, Part::Down);
        assert_eq!(copies[2].slot_offset, bytes.gate_up());
        // Disjoint and exactly covering the slot.
        let mut spans: Vec<(u64, u64)> = copies
            .iter()
            .map(|c| (c.slot_offset, c.slot_offset + c.extent.len))
            .collect();
        spans.sort_unstable();
        assert_eq!(spans[0].0, 0);
        assert_eq!(spans[0].1, spans[1].0);
        assert_eq!(spans[1].1, spans[2].0);
        assert_eq!(spans[2].1, class.slot_bytes());
        // The catalog's own copy plan agrees on the canonical-owner side.
        let extents: Vec<_> = source.catalog().copy_extents(&key).unwrap().collect();
        assert_eq!(extents.len(), 2);
        assert_eq!(extents[0].slot_offset, 0);
        assert_eq!(
            extents[0].source.region,
            source.group(1).unwrap().gate_up_region()
        );
        assert_eq!(extents[0].source.offset, 3 * bytes.gate_up());
        assert_eq!(extents[1].slot_offset, bytes.gate_up());
        assert_eq!(
            extents[1].source.region,
            source.group(1).unwrap().down_region()
        );
        assert_eq!(extents[1].source.offset, 3 * bytes.down);
    }

    // ---- refusals --------------------------------------------------------------------------

    #[test]
    fn missing_tensor_is_refused_and_named() {
        let mut index = trunk(&SMALL);
        index
            .tensors
            .remove("layers.1.ffn.experts.4.w3.weight")
            .unwrap();
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        assert_eq!(
            error,
            Ds4SourceError::MissingTensor {
                file: "base.hfq".into(),
                group_id: 1,
                expert_id: 4,
                name: "layers.1.ffn.experts.4.w3.weight".into(),
            }
        );
        let text = error.to_string();
        assert!(text.contains("group 1 expert 4"), "{text}");
        assert!(text.contains("layers.1.ffn.experts.4.w3.weight"), "{text}");
        assert!(text.contains("base.hfq"), "{text}");
    }

    #[test]
    fn missing_mtp_head_tensor_is_refused() {
        let trunk = trunk(&SMALL);
        // In-band MTP requires `mtp.0.*` in the trunk file; there are none.
        let error = build_with(SMALL, &trunk, MtpSource::InBand).unwrap_err();
        assert!(
            matches!(&error, Ds4SourceError::MissingTensor { group_id: 3, expert_id: 0, name, .. } if name == "mtp.0.ffn.experts.0.w1.weight"),
            "{error:?}"
        );
    }

    #[test]
    fn short_or_ragged_segment_is_refused_with_both_lengths() {
        let mut index = trunk(&SMALL);
        let bytes = SMALL.expert_bytes().unwrap();
        index
            .tensors
            .get_mut("layers.0.ffn.experts.2.w2.weight")
            .unwrap()
            .len = bytes.down - 128;
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        assert_eq!(
            error,
            Ds4SourceError::WrongLength {
                file: "base.hfq".into(),
                group_id: 0,
                expert_id: 2,
                name: "layers.0.ffn.experts.2.w2.weight".into(),
                found: bytes.down - 128,
                expected: bytes.down,
            }
        );
        let text = error.to_string();
        assert!(
            text.contains(&format!("{} bytes", bytes.down - 128)),
            "{text}"
        );
        assert!(text.contains(&format!("requires {}", bytes.down)), "{text}");
    }

    #[test]
    fn wrong_quant_type_is_refused() {
        let mut index = trunk(&SMALL);
        index
            .tensors
            .get_mut("layers.2.ffn.experts.0.w1.weight")
            .unwrap()
            .quant_type = 13;
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        assert!(
            matches!(
                error,
                Ds4SourceError::WrongQuant {
                    group_id: 2,
                    expert_id: 0,
                    found: 13,
                    expected: 19,
                    ..
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn extent_past_the_end_of_a_short_file_is_refused() {
        let mut index = trunk(&SMALL);
        let last = index.len;
        index.len = last - 1; // the final tensor now ends one byte past the file
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        match &error {
            Ds4SourceError::PastEndOfFile {
                group_id,
                expert_id,
                name,
                file_len,
                offset,
                len,
                ..
            } => {
                assert_eq!((*group_id, *expert_id), (2, 5));
                assert_eq!(name, "layers.2.ffn.experts.5.w2.weight");
                assert_eq!(*file_len, last - 1);
                assert_eq!(offset + len, last);
            }
            other => panic!("{other:?}"),
        }
        let text = error.to_string();
        assert!(text.contains("past the end"), "{text}");
        assert!(text.contains("group 2 expert 5"), "{text}");
    }

    #[test]
    fn offset_overflow_is_a_past_the_end_refusal_not_a_wrap() {
        let mut index = trunk(&SMALL);
        index
            .tensors
            .get_mut("layers.0.ffn.experts.0.w1.weight")
            .unwrap()
            .offset = u64::MAX - 8;
        assert!(matches!(
            build_with(SMALL, &index, MtpSource::None),
            Err(Ds4SourceError::PastEndOfFile {
                group_id: 0,
                expert_id: 0,
                ..
            })
        ));
    }

    #[test]
    fn two_experts_sharing_a_payload_are_refused() {
        let mut index = trunk(&SMALL);
        let alias = index.tensors["layers.0.ffn.experts.1.w1.weight"];
        *index
            .tensors
            .get_mut("layers.0.ffn.experts.3.w1.weight")
            .unwrap() = alias;
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        match &error {
            Ds4SourceError::Overlap {
                file,
                first,
                second,
                offset,
            } => {
                assert_eq!(file, "base.hfq");
                assert_eq!(*offset, alias.offset);
                let mut pair = [first.as_str(), second.as_str()];
                pair.sort_unstable();
                assert_eq!(
                    pair,
                    [
                        "layers.0.ffn.experts.1.w1.weight",
                        "layers.0.ffn.experts.3.w1.weight"
                    ]
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_hfq_overlay_is_refused() {
        let mut index = trunk(&SMALL);
        index.overlaid = true;
        let error = build_with(SMALL, &index, MtpSource::None).unwrap_err();
        assert_eq!(
            error,
            Ds4SourceError::Overlay {
                file: "base.hfq".into()
            }
        );
        assert!(error.to_string().contains("overlay"));
        // An overlaid MTP addon is refused too.
        let mut addon = fake("addon.hfq", &SMALL, &["mtp.0".to_string()]);
        addon.overlaid = true;
        assert!(matches!(
            build_with(SMALL, &trunk(&SMALL), MtpSource::Addon(&addon)),
            Err(Ds4SourceError::Overlay { file }) if file == "addon.hfq"
        ));
    }

    // ---- owner-generation binding ----------------------------------------------------------

    #[test]
    fn keys_bind_the_owner_generation() {
        let trunk = trunk(&SMALL);
        let old = build_with(SMALL, &trunk, MtpSource::None).unwrap();
        let new = Ds4ExpertSource::build(OWNER, GEN + 1, SMALL, &trunk, MtpSource::None).unwrap();
        let stale = old.key(1, 2);
        let fresh = new.key(1, 2);
        assert_ne!(stale, fresh);
        // The same expert under the old generation never resolves against the new owner.
        assert!(new.catalog().bundle(&stale).is_none());
        assert!(old.catalog().bundle(&stale).is_some());
        assert_eq!(
            new.bundle_files(&stale).unwrap_err(),
            Ds4SourceError::StaleOwner {
                key: stale,
                owner_id: OWNER,
                owner_generation: GEN + 1,
            }
        );
        assert!(matches!(
            new.file_copies(&stale),
            Err(Ds4SourceError::StaleOwner { .. })
        ));
        // Another owner id is stale for the same reason.
        let other = ExpertKey {
            owner_id: OWNER + 1,
            ..fresh
        };
        assert!(matches!(
            new.bundle_files(&other),
            Err(Ds4SourceError::StaleOwner { .. })
        ));
        assert!(new.bundle_files(&fresh).is_ok());
        // Unknown experts are named, not defaulted.
        assert_eq!(
            new.bundle_files(&new.key(9, 0)).unwrap_err(),
            Ds4SourceError::UnknownKey { key: new.key(9, 0) }
        );
        assert!(matches!(
            new.bundle_files(&new.key(0, SMALL.n_experts as u32)),
            Err(Ds4SourceError::UnknownKey { .. })
        ));
    }

    #[test]
    fn catalog_identity_binds_generation_and_table_address() {
        let trunk = trunk(&SMALL);
        let identity = |generation: u64, address: u64| {
            let source =
                Ds4ExpertSource::build(OWNER, generation, SMALL, &trunk, MtpSource::None).unwrap();
            let tables = [
                source.table_extent(0, TABLE_GATE_UP, address).unwrap(),
                source
                    .table_extent(0, TABLE_DOWN, address + 0x1000)
                    .unwrap(),
            ];
            source.catalog().identity(&tables).unwrap()
        };
        let base = identity(GEN, 0x10_0000);
        assert_eq!(base, identity(GEN, 0x10_0000));
        assert_ne!(base, identity(GEN + 1, 0x10_0000), "owner generation");
        assert_ne!(base, identity(GEN, 0x20_0000), "table address");
        let source = built(MtpSource::None);
        let table = source.table_extent(1, TABLE_GATE_UP, 0x8000).unwrap();
        assert_eq!(table.bytes, SMALL.n_experts as u64 * 8);
        assert!(matches!(
            source.table_extent(99, TABLE_GATE_UP, 0x8000),
            Err(Ds4SourceError::Catalog(CatalogError::UnknownGroup {
                group_id: 99
            }))
        ));
    }

    // ---- host RAM --------------------------------------------------------------------------

    #[test]
    fn host_charge_is_every_blob_plus_its_tail_pad() {
        let source = built(MtpSource::InBand);
        let groups = (SMALL.trunk_layers + 1) as u64;
        let charge = source.host_charge();
        assert_eq!(charge.allocations as u64, 2 * groups);
        assert_eq!(
            charge.blob_bytes,
            groups * SMALL.n_experts as u64 * source.expert_bytes().bundle()
        );
        assert_eq!(charge.tail_pad_bytes, 2 * groups * HOST_TAIL_PAD_BYTES);
        assert_eq!(charge.total(), charge.blob_bytes + charge.tail_pad_bytes);
        // The regions the catalog names are exactly those blobs.
        let region_bytes: u64 = source.catalog().regions().iter().map(|r| r.len).sum();
        assert_eq!(region_bytes, charge.blob_bytes);
        assert_eq!(source.catalog().regions().len(), charge.allocations);
    }

    #[test]
    fn flash_host_charge_matches_the_checkpoint() {
        let index = fake("flash.hfq", &FLASH, &{
            // Only the geometry matters for the charge; build it from a one-layer index and
            // scale by the closed form instead of materializing 33k fake tensors twice.
            trunk_prefixes(&Ds4Geometry {
                trunk_layers: 1,
                ..FLASH
            })
        });
        let one = Ds4ExpertSource::build(
            OWNER,
            GEN,
            Ds4Geometry {
                trunk_layers: 1,
                ..FLASH
            },
            &index,
            MtpSource::None,
        )
        .unwrap();
        let charge = one.host_charge();
        assert_eq!(charge.blob_bytes, 1_811_939_328);
        assert_eq!(charge.allocations, 2);
        // 43 trunk layers + the MTP head.
        assert_eq!(44 * charge.blob_bytes, 79_725_330_432);
        assert_eq!(44 * 2 * HOST_TAIL_PAD_BYTES, 92_274_688);
    }

    fn ram(mem_available: Option<u64>, ttm_pool: u64) -> PinnableHostRam {
        PinnableHostRam {
            mem_available,
            ttm_pool,
        }
    }

    #[test]
    fn host_ram_check_refuses_below_headroom_and_counts_the_pool() {
        let charge = built(MtpSource::None).host_charge();
        let needed = charge.total() + HOST_RAM_HEADROOM_BYTES;
        assert!(check_host_ram(&charge, ram(Some(needed), 0)).is_ok());
        let short = ram(Some(needed - 1), 0);
        let error = check_host_ram(&charge, short).unwrap_err();
        assert_eq!(
            error,
            Ds4Refusal::HostRam(HostRamShortfall {
                host_bytes: charge.total(),
                headroom: HOST_RAM_HEADROOM_BYTES,
                ram: short,
            })
        );
        assert!(error.to_string().contains("pinned host RAM"), "{error}");
        assert!(error.to_string().contains("TTM's page pool"), "{error}");
        // Freed GTT pages parked in TTM's page pool count, byte for byte.
        assert!(
            check_host_ram(&charge, ram(Some(charge.total()), HOST_RAM_HEADROOM_BYTES)).is_ok()
        );
        // Unreadable MemAvailable refuses rather than guessing.
        let unreadable = check_host_ram(&charge, ram(None, u64::MAX)).unwrap_err();
        assert!(unreadable.to_string().contains("unreadable"));
        let empty = HostCharge {
            blob_bytes: 0,
            tail_pad_bytes: 0,
            allocations: 0,
        };
        assert!(check_host_ram(&empty, ram(None, 0)).is_ok());
        // Saturating, never wrapping.
        let huge = HostCharge {
            blob_bytes: u64::MAX - 1,
            tail_pad_bytes: 1,
            allocations: 2,
        };
        assert!(check_host_ram(&huge, ram(Some(u64::MAX), 0)).is_err());
    }

    // ---- topology --------------------------------------------------------------------------

    fn single() -> Topology {
        Topology {
            arch_id: DS4_ARCH_ID,
            tp: 1,
            pp: 1,
            ep_experts: None,
            reap_keep: false,
            heterogeneous: false,
            unified_memory: false,
        }
    }

    #[test]
    fn support_truth_table_refuses_each_dimension_in_priority_order() {
        assert_eq!(expert_cache_support(&single()), Ok(()));
        let with = |edit: fn(&mut Topology)| {
            let mut topology = single();
            edit(&mut topology);
            expert_cache_support(&topology)
        };
        assert_eq!(with(|t| t.arch_id = 6), Err(Ds4Refusal::Arch(6)));
        assert_eq!(with(|t| t.tp = 2), Err(Ds4Refusal::Tp(2)));
        assert_eq!(with(|t| t.pp = 2), Err(Ds4Refusal::Pp(2)));
        assert_eq!(
            with(|t| t.ep_experts = Some(4)),
            Err(Ds4Refusal::ExpertParallel(4))
        );
        assert_eq!(with(|t| t.ep_experts = Some(0)), Ok(()), "zero is not EP");
        assert_eq!(with(|t| t.reap_keep = true), Err(Ds4Refusal::ReapKeepMap));
        assert_eq!(
            with(|t| t.heterogeneous = true),
            Err(Ds4Refusal::Heterogeneous)
        );
        assert_eq!(
            with(|t| t.unified_memory = true),
            Err(Ds4Refusal::UnifiedMemory)
        );
        // tp wins over the EP pinning behind it; arch wins over everything.
        assert_eq!(
            with(|t| {
                t.tp = 2;
                t.ep_experts = Some(4);
            }),
            Err(Ds4Refusal::Tp(2))
        );
        assert_eq!(
            with(|t| {
                t.arch_id = 5;
                t.tp = 2;
            }),
            Err(Ds4Refusal::Arch(5))
        );
    }

    #[test]
    fn dense_spill_stays_refused_for_arch_9() {
        // Expert caching is its own admission; it does not admit `memory.gpu_layer_budget` /
        // `memory.moe_expert_budget` spill on DS4.
        use hipfire_config::memory::{spill_support, SpillRefusal, SpillSupport};
        assert_eq!(
            spill_support(DS4_ARCH_ID, 1, 1, None, false, false, false),
            SpillSupport::Refused(SpillRefusal::Arch(DS4_ARCH_ID))
        );
    }

    #[test]
    fn admit_checks_topology_before_host_ram_and_returns_the_charge() {
        let source = built(MtpSource::InBand);
        let charge = source.host_charge();
        let room = ram(Some(charge.total() + HOST_RAM_HEADROOM_BYTES), 0);
        assert_eq!(source.admit(&single(), room), Ok(charge));
        // No host RAM figure at all: the topology refusal still wins.
        let mut ep = single();
        ep.ep_experts = Some(8);
        assert_eq!(
            source.admit(&ep, ram(None, 0)),
            Err(Ds4Refusal::ExpertParallel(8))
        );
        assert!(matches!(
            source.admit(&single(), ram(Some(charge.total()), 0)),
            Err(Ds4Refusal::HostRam(_))
        ));
    }

    #[test]
    fn refusals_read_as_load_sentences() {
        let text = Ds4Refusal::ReapKeepMap.to_string();
        assert!(
            text.starts_with("load refused: DeepSeek V4 expert caching"),
            "{text}"
        );
        assert!(Ds4Refusal::Arch(6).to_string().contains("arch id 6"));
        assert!(Ds4Refusal::UnifiedMemory
            .to_string()
            .contains("unified memory"));
    }
}
