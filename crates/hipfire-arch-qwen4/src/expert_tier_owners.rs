// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Immutable model-level expert payload owners and borrowed layer tables.
//!
//! Materialization streams bounded source spans into at most one device and
//! one pinned, device-mapped host allocation. The lease retains ownership via
//! a mutex, while its immutable CPU binding is read without locking.
//!
//! B5-C dependency: `ExpertTierReplayIdentity` and `route_binding()` must be
//! added only after `rdna_compute::route_binding` lands. This module's generation
//! identifies allocations; it is not a replay authorization token.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use hip_bridge::{DeviceBuffer, HipRuntime, HIP_HOST_MALLOC_MAPPED};
use hipfire_runtime::hfq_parallel::{
    self, DeviceRangeCopy, DirectSource, HostRangeCopy, HFQ_READER_LANES,
};
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};
use crate::expert_tier_plan::{DecodeShareTail, ExpertTierEntry, ExpertTierPlan, Placement};
use sha2::{Digest, Sha256};

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const TOTAL_EXPERTS: usize = TRUNK_LAYERS as usize * EXPERTS;
const COPY_CHUNK_BYTES: u64 = 8 * 1024 * 1024;
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// A materialization failure; no partially built lease is published.
#[derive(Debug)]
pub enum ExpertTierOwnerError {
    Runtime(String),
    Source(String),
    SourceLength {
        expected: u64,
        got: usize,
    },
    MissingBase(Placement),
    Overflow,
    Extent {
        tier: Placement,
        offset: u64,
        len: u64,
        owner_bytes: u64,
    },
    Alignment {
        address: u64,
    },
    TailPad {
        tier: Placement,
    },
    GenerationExhausted,
    /// A decode-share view invariant of the plan does not hold.
    DecodeShare(String),
}

impl fmt::Display for ExpertTierOwnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(message) => write!(f, "expert tier HIP operation: {message}"),
            Self::Source(message) => write!(f, "expert tier source: {message}"),
            Self::SourceLength { expected, got } => write!(
                f,
                "expert tier source returned {got} bytes, expected {expected}"
            ),
            Self::MissingBase(tier) => write!(f, "missing {tier:?} owner base"),
            Self::Overflow => write!(f, "expert tier address or extent overflow"),
            Self::Extent {
                tier,
                offset,
                len,
                owner_bytes,
            } => write!(
                f,
                "{tier:?} extent {offset}+{len} exceeds owner size {owner_bytes}"
            ),
            Self::Alignment { address } => {
                write!(f, "expert tier address {address:#x} is not 256-aligned")
            }
            Self::TailPad { tier } => {
                write!(f, "{tier:?} owner has insufficient readable tail padding")
            }
            Self::GenerationExhausted => write!(f, "expert tier generation exhausted"),
            Self::DecodeShare(message) => write!(f, "expert tier decode-share: {message}"),
        }
    }
}

impl std::error::Error for ExpertTierOwnerError {}

fn issue_generation(counter: &AtomicU64) -> Result<u64, ExpertTierOwnerError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
            last.checked_add(1)
        })
        .map(|last| last + 1)
        .map_err(|_| ExpertTierOwnerError::GenerationExhausted)
}

#[derive(Debug)]
struct Tables {
    gate_up: Vec<usize>,
    down: Vec<usize>,
    codecs: Vec<u8>,
}

fn tier_index(tier: Placement) -> usize {
    match tier {
        Placement::Device => 0,
        Placement::HostMapped => 1,
    }
}

fn checked_extent(
    tier: Placement,
    offset: u64,
    len: u64,
    owner_bytes: u64,
) -> Result<(), ExpertTierOwnerError> {
    let end = offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    if len == 0 || end > owner_bytes {
        return Err(ExpertTierOwnerError::Extent {
            tier,
            offset,
            len,
            owner_bytes,
        });
    }
    Ok(())
}

fn checked_address(base: u64, offset: u64) -> Result<usize, ExpertTierOwnerError> {
    let address = base
        .checked_add(offset)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    if address % ExpertTierPlan::PAYLOAD_ALIGN != 0 {
        return Err(ExpertTierOwnerError::Alignment { address });
    }
    usize::try_from(address).map_err(|_| ExpertTierOwnerError::Overflow)
}

fn tail_pad_bytes() -> u64 {
    let align = ExpertTierPlan::PAYLOAD_ALIGN;
    ((ARENA_TAIL_PAD_BYTES as u64).max(align) + align - 1) & !(align - 1)
}

/// Pure construction of the exact CPU mirrors uploaded to the device tables
/// for the canonical (Resident) view.
fn build_tables(
    plan: &ExpertTierPlan,
    device_base: Option<u64>,
    host_device_alias: Option<u64>,
) -> Result<Tables, ExpertTierOwnerError> {
    build_tables_for(plan, plan.entries(), device_base, host_device_alias)
}

/// [`build_tables`] over an explicit entry list (the plan's own entries for
/// the Resident view, `DecodeShareTail::borrow_entries` for BorrowPrefill).
fn build_tables_for(
    plan: &ExpertTierPlan,
    entries: &[ExpertTierEntry],
    device_base: Option<u64>,
    host_device_alias: Option<u64>,
) -> Result<Tables, ExpertTierOwnerError> {
    // The HIP pointer-table ABI is 64-bit, not a dtype/quantization tag table.
    if std::mem::size_of::<usize>() != 8 {
        return Err(ExpertTierOwnerError::Overflow);
    }
    if entries.len() != TOTAL_EXPERTS {
        return Err(ExpertTierOwnerError::Overflow);
    }
    let bases = [device_base, host_device_alias];
    for tier in [Placement::Device, Placement::HostMapped] {
        let bytes = plan.owner_bytes(tier);
        if bytes != 0 {
            let base = bases[tier_index(tier)].ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            checked_address(base, 0)?;
            let end = base
                .checked_add(bytes)
                .ok_or(ExpertTierOwnerError::Overflow)?;
            usize::try_from(end).map_err(|_| ExpertTierOwnerError::Overflow)?;
        }
    }
    let mut tables = Tables {
        gate_up: Vec::with_capacity(TOTAL_EXPERTS),
        down: Vec::with_capacity(TOTAL_EXPERTS),
        codecs: Vec::with_capacity(TOTAL_EXPERTS),
    };
    let mut used = [0u64; 2];
    for layer in 0..TRUNK_LAYERS as usize {
        for eid in 0..EXPERTS {
            let entry = entries
                .get(layer * EXPERTS + eid)
                .ok_or(ExpertTierOwnerError::Overflow)?;
            let t = tier_index(entry.tier);
            let bytes = plan.owner_bytes(entry.tier);
            let base = bases[t].ok_or(ExpertTierOwnerError::MissingBase(entry.tier))?;
            for (offset, len) in [
                (entry.gate_up_offset, entry.gate_up_len),
                (entry.down_offset, entry.down_len),
            ] {
                checked_extent(entry.tier, offset, len, bytes)?;
                used[t] = used[t].max(offset + len);
            }
            tables
                .gate_up
                .push(checked_address(base, entry.gate_up_offset)?);
            tables.down.push(checked_address(base, entry.down_offset)?);
            tables.codecs.push(match entry.format {
                ExpertFormat::Mq4Xts => 0,
                ExpertFormat::Mq3I4 => 1,
                ExpertFormat::Mq2I4 => 2,
                ExpertFormat::E8pI4 => 3,
            });
        }
    }
    for window in plan.host_windows() {
        for (span, offset) in &window.spans {
            let tier = Placement::HostMapped;
            checked_extent(tier, *offset, span.len as u64, plan.owner_bytes(tier))?;
            let base = host_device_alias.ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            checked_address(base, *offset)?;
            used[1] = used[1].max(*offset + span.len as u64);
        }
    }
    for tier in [Placement::Device, Placement::HostMapped] {
        let bytes = plan.owner_bytes(tier);
        if bytes != 0 && bytes.checked_sub(used[tier_index(tier)]).unwrap_or(0) < tail_pad_bytes() {
            return Err(ExpertTierOwnerError::TailPad { tier });
        }
    }
    Ok(tables)
}

/// Domain of the BorrowPrefill view digest.
const BORROW_DIGEST_DOMAIN: &[u8] = b"hipfire.expert_tier_plan.borrow_prefill";

/// SHA-256 over the borrow domain, the plan's logical digest, the class index
/// and its dirtied-donor count, and every entry's `(tier, offsets, lens)` in
/// id order (the class view's entries).
fn borrow_view_digest(
    logical_digest: [u8; 32],
    class: usize,
    donors: usize,
    entries: &[ExpertTierEntry],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(BORROW_DIGEST_DOMAIN);
    hash.update(logical_digest);
    hash.update((class as u64).to_le_bytes());
    hash.update((donors as u64).to_le_bytes());
    for entry in entries {
        hash.update([tier_index(entry.tier) as u8]);
        hash.update(entry.gate_up_offset.to_le_bytes());
        hash.update(entry.gate_up_len.to_le_bytes());
        hash.update(entry.down_offset.to_le_bytes());
        hash.update(entry.down_len.to_le_bytes());
    }
    hash.finalize().into()
}

/// Immutable device-visible and CPU ranges of a decode-share lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeShareRanges {
    /// `(device address, bytes)` of the contiguous donor tail.
    pub donor_device: (usize, usize),
    /// `(CPU address, bytes)` of the pinned mirror window.
    pub mirror_cpu: (usize, usize),
    /// Device alias address of the mirror window start.
    pub mirror_alias: usize,
    /// Number of donor experts.
    pub donors: usize,
    /// Bytes at the start of the donor tail that hold the borrowed weight
    /// shadows ([`crate::expert_tier_plan::DecodeShareTail::shadow_bytes`]);
    /// the scratch backing starts right after them. 0 without borrowed shadows.
    pub shadow_bytes: usize,
}

/// Cache restore spans. Unlike DecodeShareRanges, these never authorize a
/// scratch loan; all donor bytes remain expert payload slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCacheRanges {
    pub donor_device: (usize, usize),
    pub mirror_cpu: (usize, usize),
    pub mirror_alias: usize,
}

/// Checked addresses for one original donor expert. GU/down lengths exclude
/// alignment padding; CPU ranges and host aliases name immutable mirror bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCacheSlotRanges {
    pub layer: usize,
    pub original_expert: usize,
    pub gate_up_device: (usize, usize),
    pub down_device: (usize, usize),
    pub gate_up_host: (usize, usize),
    pub down_host: (usize, usize),
    pub gate_up_cpu: (usize, usize),
    pub down_cpu: (usize, usize),
}

#[derive(Clone)]
pub struct ExpertCacheSlot {
    pub ranges: ExpertCacheSlotRanges,
    lease: ExpertTierLease,
}

impl ExpertCacheSlot {
    pub fn lease(&self) -> ExpertTierLease {
        self.lease.clone()
    }
    pub fn generation(&self) -> u64 {
        self.lease.generation()
    }
}

/// Mutable table storage has its own lifetime, independently of GpuForward.
/// This is ownership metadata, not a dispatch or replay authorization.
/// Descriptors returned below are borrowed; retire every GPU user before
/// dropping its last keepalive or changing entries used by that user.
#[derive(Clone)]
pub struct ExpertCacheTableLease {
    owner: Arc<Mutex<ExpertCacheTableOwner>>,
    bases: [usize; 2],
    generation: u64,
    layer: usize,
}

impl ExpertCacheTableLease {
    pub fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        self.owner.clone()
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn gate_up_table(&self) -> GpuTensor {
        borrowed_table(self.bases[0], 2 * EXPERTS, DType::F32)
    }
    pub fn down_table(&self) -> GpuTensor {
        borrowed_table(self.bases[1], 2 * EXPERTS, DType::F32)
    }
}

struct ExpertCacheTableOwner {
    hip: Arc<HipRuntime>,
    device_id: i32,
    tables: [Option<GpuTensor>; 2],
    _tier: ExpertTierLease,
}

impl Drop for ExpertCacheTableOwner {
    fn drop(&mut self) {
        let Ok(previous) = self.hip.current_device() else {
            return;
        };
        if self.hip.set_device(self.device_id).is_err() {
            return;
        }
        for table in &mut self.tables {
            if let Some(table) = table.take() {
                let _ = self.hip.free(table.buf);
            }
        }
        if previous != self.device_id {
            let _ = self.hip.set_device(previous);
        }
    }
}

fn cache_slot_specs(
    plan: &ExpertTierPlan,
    layer: usize,
    device_base: u64,
    host_cpu: usize,
    host_alias: u64,
) -> Result<Vec<ExpertCacheSlotRanges>, ExpertTierOwnerError> {
    if layer >= TRUNK_LAYERS as usize {
        return Err(decode_share_error("cache layer out of range".to_string()));
    }
    let tail = plan
        .expert_cache_tail()
        .ok_or_else(|| decode_share_error("plan has no expert cache".to_string()))?;
    let cpu_address = |offset: u64| -> Result<usize, ExpertTierOwnerError> {
        host_cpu
            .checked_add(usize::try_from(offset).map_err(|_| ExpertTierOwnerError::Overflow)?)
            .ok_or(ExpertTierOwnerError::Overflow)
    };
    let mut slots = Vec::new();
    for &id in &tail.donors {
        if id as usize / EXPERTS != layer {
            continue;
        }
        let resident = &plan.entries()[id as usize];
        let mirror = &tail.borrow_entries[id as usize];
        let gu =
            usize::try_from(resident.gate_up_len).map_err(|_| ExpertTierOwnerError::Overflow)?;
        let dn = usize::try_from(resident.down_len).map_err(|_| ExpertTierOwnerError::Overflow)?;
        slots.push(ExpertCacheSlotRanges {
            layer,
            original_expert: id as usize % EXPERTS,
            gate_up_device: (checked_address(device_base, resident.gate_up_offset)?, gu),
            down_device: (checked_address(device_base, resident.down_offset)?, dn),
            gate_up_host: (checked_address(host_alias, mirror.gate_up_offset)?, gu),
            down_host: (checked_address(host_alias, mirror.down_offset)?, dn),
            gate_up_cpu: (cpu_address(mirror.gate_up_offset)?, gu),
            down_cpu: (cpu_address(mirror.down_offset)?, dn),
        });
    }
    Ok(slots)
}

/// CPU-built state of one BorrowPrefill class view, before its tables are
/// uploaded: the Resident entries except donors `[0, donors)` (physical
/// order), which point at their mirror.
struct BorrowViewCpu {
    entries: Vec<ExpertTierEntry>,
    tables: Tables,
    donors: usize,
    digest: [u8; 32],
}

/// CPU-built state of the BorrowPrefill views, before their tables are uploaded.
struct DecodeShareCpu {
    views: Vec<BorrowViewCpu>,
    ranges: DecodeShareRanges,
    donor_offset: u64,
    class_spans: Vec<u64>,
}

/// One published BorrowPrefill class view. Its device pointer tables are owned
/// (and freed once) by `ExpertTierOwners`; only their base addresses live here.
struct BorrowView {
    entries: Vec<ExpertTierEntry>,
    tables: Tables,
    gate_up_table_base: usize,
    down_table_base: usize,
    donors: usize,
    digest: [u8; 32],
}

/// The published BorrowPrefill views, one per prefill capacity class (nested:
/// a larger class reads more donors from the mirror).
struct DecodeShareBinding {
    views: Vec<BorrowView>,
    ranges: DecodeShareRanges,
    /// Device-owner offset of the first donor byte.
    donor_offset: u64,
    /// Dirty byte prefix of each prefill class (`views[c]` reads the mirror for
    /// the donors that prefix overlaps).
    class_spans: Vec<u64>,
}

fn decode_share_error(message: String) -> ExpertTierOwnerError {
    ExpertTierOwnerError::DecodeShare(message)
}

/// Pure CPU construction of the BorrowPrefill view and its ranges. `None` for
/// a plan without donors. Verifies the plan's own promises (donor offsets map
/// one-to-one onto the mirror, non-donors are untouched, codecs are shared).
fn build_decode_share(
    plan: &ExpertTierPlan,
    resident: &Tables,
    device_base: Option<u64>,
    host_cpu: Option<usize>,
    host_device_alias: Option<u64>,
) -> Result<Option<DecodeShareCpu>, ExpertTierOwnerError> {
    build_donor_tail(
        plan,
        plan.decode_share_tail(),
        resident,
        device_base,
        host_cpu,
        host_device_alias,
    )
}

fn build_donor_tail(
    plan: &ExpertTierPlan,
    tail: Option<&DecodeShareTail>,
    resident: &Tables,
    device_base: Option<u64>,
    host_cpu: Option<usize>,
    host_device_alias: Option<u64>,
) -> Result<Option<DecodeShareCpu>, ExpertTierOwnerError> {
    let Some(tail) = tail else {
        return Ok(None);
    };
    if tail.donors.is_empty() {
        return Ok(None);
    }
    let device_base = device_base.ok_or(ExpertTierOwnerError::MissingBase(Placement::Device))?;
    let host_alias =
        host_device_alias.ok_or(ExpertTierOwnerError::MissingBase(Placement::HostMapped))?;
    let host_cpu = host_cpu.ok_or(ExpertTierOwnerError::MissingBase(Placement::HostMapped))?;
    if tail.borrow_entries.len() != TOTAL_EXPERTS {
        return Err(decode_share_error(format!(
            "borrow view has {} entries, expected {TOTAL_EXPERTS}",
            tail.borrow_entries.len()
        )));
    }
    let donor_end = tail
        .donor_offset
        .checked_add(tail.donor_bytes)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    let mirror_end = tail
        .mirror_offset
        .checked_add(tail.donor_bytes)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    checked_extent(
        Placement::Device,
        tail.donor_offset,
        tail.donor_bytes,
        plan.owner_bytes(Placement::Device),
    )?;
    checked_extent(
        Placement::HostMapped,
        tail.mirror_offset,
        tail.donor_bytes,
        plan.owner_bytes(Placement::HostMapped),
    )?;
    for (id, borrow) in tail.borrow_entries.iter().enumerate() {
        let resident_entry = plan
            .entry(id / EXPERTS, id % EXPERTS)
            .ok_or(ExpertTierOwnerError::Overflow)?;
        if tail.donors.binary_search(&(id as u32)).is_err() {
            if borrow != resident_entry {
                return Err(decode_share_error(format!(
                    "non-donor expert {id} differs between the resident and borrow views"
                )));
            }
            continue;
        }
        if resident_entry.tier != Placement::Device || borrow.tier != Placement::HostMapped {
            return Err(decode_share_error(format!(
                "donor {id} is not Device in the resident view and HostMapped in the borrow view"
            )));
        }
        for (device_offset, host_offset, len, borrow_len) in [
            (
                resident_entry.gate_up_offset,
                borrow.gate_up_offset,
                resident_entry.gate_up_len,
                borrow.gate_up_len,
            ),
            (
                resident_entry.down_offset,
                borrow.down_offset,
                resident_entry.down_len,
                borrow.down_len,
            ),
        ] {
            let device_rel = device_offset.checked_sub(tail.donor_offset);
            let host_rel = host_offset.checked_sub(tail.mirror_offset);
            let end_ok = device_offset
                .checked_add(len)
                .is_some_and(|end| end <= donor_end)
                && host_offset
                    .checked_add(len)
                    .is_some_and(|end| end <= mirror_end);
            if len != borrow_len || device_rel.is_none() || device_rel != host_rel || !end_ok {
                return Err(decode_share_error(format!(
                    "donor {id} does not map one-to-one onto the mirror window"
                )));
            }
        }
    }
    let classes = &tail.class_donors;
    if classes.is_empty()
        || tail.class_spans.len() != classes.len()
        || classes.windows(2).any(|pair| pair[0] > pair[1])
        || classes.last() != Some(&tail.donors.len())
    {
        return Err(decode_share_error(format!(
            "class donor counts {classes:?} are not nested and ending at the {} donors",
            tail.donors.len()
        )));
    }
    let mut views = Vec::with_capacity(classes.len());
    for (class, &donors) in classes.iter().enumerate() {
        // Class view: the Resident entries except donors `[0, donors)`
        // (physical order), which read their mirror.
        let mut entries = plan.entries().to_vec();
        for &id in &tail.donors[..donors] {
            entries[id as usize] = tail.borrow_entries[id as usize].clone();
        }
        let tables = build_tables_for(plan, &entries, Some(device_base), Some(host_alias))?;
        if tables.codecs != resident.codecs {
            return Err(decode_share_error(
                "borrow and resident codec tables differ".to_string(),
            ));
        }
        views.push(BorrowViewCpu {
            digest: borrow_view_digest(plan.logical_digest(), class, donors, &entries),
            entries,
            tables,
            donors,
        });
    }
    let donor_bytes =
        usize::try_from(tail.donor_bytes).map_err(|_| ExpertTierOwnerError::Overflow)?;
    let ranges = DecodeShareRanges {
        donor_device: (
            checked_address(device_base, tail.donor_offset)?,
            donor_bytes,
        ),
        mirror_cpu: (
            host_cpu
                .checked_add(
                    usize::try_from(tail.mirror_offset)
                        .map_err(|_| ExpertTierOwnerError::Overflow)?,
                )
                .ok_or(ExpertTierOwnerError::Overflow)?,
            donor_bytes,
        ),
        mirror_alias: checked_address(host_alias, tail.mirror_offset)?,
        donors: tail.donors.len(),
        shadow_bytes: usize::try_from(tail.shadow_bytes)
            .map_err(|_| ExpertTierOwnerError::Overflow)?,
    };
    Ok(Some(DecodeShareCpu {
        views,
        ranges,
        donor_offset: tail.donor_offset,
        class_spans: tail.class_spans.clone(),
    }))
}

/// Compare the device donor bytes with `mirror` (whose length is the verified
/// prefix) in bounded chunks. `read_device(chunk, offset)` fills `chunk` with
/// the device donor bytes starting `offset` bytes into the donor tail.
fn verify_tail_bytes(
    mirror: &[u8],
    mut read_device: impl FnMut(&mut [u8], usize) -> Result<(), String>,
) -> Result<[u8; 32], String> {
    let mut hash = Sha256::new();
    let mut scratch = Vec::new();
    let mut done = 0usize;
    while done < mirror.len() {
        let n = (mirror.len() - done).min(COPY_CHUNK_BYTES as usize);
        scratch.resize(n, 0u8);
        read_device(&mut scratch, done)?;
        let expected = &mirror[done..done + n];
        if scratch != expected {
            let at = scratch
                .iter()
                .zip(expected)
                .position(|(device, host)| device != host)
                .unwrap_or(0);
            return Err(format!(
                "donor tail differs from its mirror at tail byte {}: device {:#04x}, mirror {:#04x}",
                done + at,
                scratch[at],
                expected[at]
            ));
        }
        hash.update(&scratch);
        done += n;
    }
    Ok(hash.finalize().into())
}

/// The pinned host address is retained only for CPU copies and `hipHostFree`.
/// Storing opaque addresses as integers follows Gpu's host-mapped registry and
/// leaves ownership Send without adding an unsafe auto-trait implementation.
struct HostPayload {
    cpu_address: usize,
    device_alias: u64,
    bytes: usize,
}

/// Owns all expert allocations. Destruction requires GPU quiescence; retained
/// launches must hold a keepalive until their retirement proves that condition.
pub struct ExpertTierOwners {
    device: Option<DeviceBuffer>,
    host: Option<HostPayload>,
    gate_up_table: Option<GpuTensor>,
    down_table: Option<GpuTensor>,
    codec_table: Option<GpuTensor>,
    /// Device pointer tables of every BorrowPrefill class view: gate/up then
    /// down, class by class.
    borrow_tables: Vec<GpuTensor>,
    binding: Option<Arc<ExpertTierBinding>>,
    plan: Arc<ExpertTierPlan>,
    hip: Arc<HipRuntime>,
    device_id: i32,
    generation: u64,
}

/// Immutable CPU state, including the exact mirrors used for table uploads.
/// No GPU buffer handles are stored here: DeviceBuffer is deliberately !Sync.
pub struct ExpertTierBinding {
    plan: Arc<ExpertTierPlan>,
    tables: Tables,
    gate_up_table_base: usize,
    down_table_base: usize,
    codec_table_base: usize,
    /// `(device-visible base, bytes)` per tier, indexed by `tier_index`; the
    /// host entry is the mapped device alias. `None` for an empty owner.
    owner_ranges: [Option<(usize, usize)>; 2],
    /// `(CPU address, bytes)` of the one pinned host owner, for CPU reads and
    /// staging; `None` when no host payload exists.
    host_cpu_range: Option<(usize, usize)>,
    generation: u64,
    /// The BorrowPrefill view, present when the plan carries donors.
    decode_share: Option<DecodeShareBinding>,
    expert_cache: Option<DecodeShareBinding>,
    /// Raw SHA-256 (hex) of the hot-set sidecar, identity metadata bound once
    /// by the loader after materialization.
    hotset_sidecar_sha256: std::sync::OnceLock<String>,
    logical_digest: [u8; 32],
    /// Set while one forward owns the donor tail's borrow protocol
    /// ([`ExpertTierLease::claim_decode_share`]).
    decode_share_claimed: std::sync::atomic::AtomicBool,
}

/// Cheap shared retention of a frozen materialized model binding.
#[derive(Clone)]
pub struct ExpertTierLease {
    binding: Arc<ExpertTierBinding>,
    owners: Arc<Mutex<ExpertTierOwners>>,
}

// A compile-time check, not an unsafe implementation. The mutex supplies Sync
// for the Send-only DeviceBuffer owners; the CPU binding is independently Sync.
const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Arc<Mutex<ExpertTierOwners>>>();
    send_sync::<ExpertTierLease>();
    send_sync::<ExpertCacheTableLease>();
};

impl ExpertTierLease {
    pub fn layer(&self, layer: usize) -> Option<LayerExpertTierView<'_>> {
        self.layer_view(layer, ExpertReadView::Resident)
    }

    /// One layer through a chosen read view; `None` for an out-of-range layer
    /// or for `BorrowPrefill` on a lease without donor backing.
    pub fn layer_view(
        &self,
        layer: usize,
        view: ExpertReadView,
    ) -> Option<LayerExpertTierView<'_>> {
        Some(LayerExpertTierView {
            lease: self,
            core: self.binding.layer_view(layer, view)?,
        })
    }

    pub fn owner_bytes(&self, tier: Placement) -> u64 {
        self.binding.plan.owner_bytes(tier)
    }

    /// Immutable `(device-visible base, bytes)` of the tier's one owner, read
    /// without locking. The host range is the mapped device alias. `None`
    /// when the tier holds no payload.
    pub fn owner_range(&self, tier: Placement) -> Option<(usize, usize)> {
        self.binding.owner_ranges[tier_index(tier)]
    }

    /// Immutable `(CPU address, bytes)` of the pinned host owner, read without
    /// locking; valid while this lease (or its keepalive) is retained. `None`
    /// when the plan places nothing in host memory.
    pub fn host_cpu_range(&self) -> Option<(usize, usize)> {
        self.binding.host_cpu_range
    }

    pub fn keepalive(&self) -> Arc<dyn Any + Send + Sync> {
        self.owners.clone()
    }

    pub fn generation(&self) -> u64 {
        self.binding.generation
    }

    pub fn logical_digest(&self) -> [u8; 32] {
        self.binding.logical_digest
    }

    /// Record the raw SHA-256 (lowercase hex) of the hot-set sidecar this plan
    /// was built from. The loader calls it once after `materialize`; it is
    /// identity metadata only (never authorization) and never alters the
    /// plan. Recording the same value again is accepted; a different value or a
    /// malformed digest is refused.
    pub fn bind_hotset_sidecar_sha256(&self, sha256_hex: &str) -> Result<(), String> {
        if sha256_hex.len() != 64
            || !sha256_hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "hot-set sidecar digest must be 64 lowercase hex digits, got {sha256_hex:?}"
            ));
        }
        let bound = self
            .binding
            .hotset_sidecar_sha256
            .get_or_init(|| sha256_hex.to_string());
        if bound == sha256_hex {
            Ok(())
        } else {
            Err(format!(
                "hot-set sidecar digest already bound to {bound}, refusing {sha256_hex}"
            ))
        }
    }

    /// The sidecar SHA-256 bound by [`Self::bind_hotset_sidecar_sha256`], if any.
    pub fn hotset_sidecar_sha256(&self) -> Option<&str> {
        self.binding.hotset_sidecar_sha256.get().map(String::as_str)
    }

    /// Donors each BorrowPrefill class view reads from the mirror (nested,
    /// ascending; the last is the donor count); empty without donors.
    pub fn borrow_class_donors(&self) -> Vec<usize> {
        self.binding
            .decode_share
            .as_ref()
            .map(|share| share.views.iter().map(|view| view.donors).collect())
            .unwrap_or_default()
    }

    /// Number of BorrowPrefill class views (0 without donors).
    pub fn borrow_classes(&self) -> usize {
        self.binding
            .decode_share
            .as_ref()
            .map_or(0, |share| share.views.len())
    }

    /// Dirty byte prefix of each BorrowPrefill class view (ascending; empty
    /// without donors).
    pub fn borrow_class_spans(&self) -> Vec<u64> {
        self.binding
            .decode_share
            .as_ref()
            .map(|share| share.class_spans.clone())
            .unwrap_or_default()
    }

    /// The decode-share donor/mirror ranges, read without locking; `None`
    /// unless the plan carries a donor tail with at least one donor.
    pub fn decode_share(&self) -> Option<DecodeShareRanges> {
        self.binding.decode_share.as_ref().map(|share| share.ranges)
    }

    /// Canonical cache reads: core experts stay on device; donors and cold
    /// experts read immutable host backing. Resident remains the frozen warm
    /// prefill view. This accessor never claims or lends the donor tail.
    pub fn expert_cache_layer(&self, layer: usize) -> Option<LayerExpertTierView<'_>> {
        self.binding.expert_cache.as_ref()?;
        self.layer_view(layer, ExpertReadView::BorrowPrefill(0))
    }

    /// Exact contiguous range to restore before using Resident prefill.
    /// Restore completion and warm-table publication require caller-managed
    /// quiescence; the immutable mirror must never be overwritten.
    pub fn expert_cache_ranges(&self) -> Option<ExpertCacheRanges> {
        let ranges = self.binding.expert_cache.as_ref()?.ranges;
        Some(ExpertCacheRanges {
            donor_device: ranges.donor_device,
            mirror_cpu: ranges.mirror_cpu,
            mirror_alias: ranges.mirror_alias,
        })
    }

    /// Slot metadata retains the payload owner and its generation. Core
    /// experts have no slot and no mirror. Slots are in physical tail order.
    pub fn expert_cache_slots(&self, layer: usize) -> Option<Vec<ExpertCacheSlot>> {
        self.binding.expert_cache.as_ref()?;
        let specs = cache_slot_specs(
            &self.binding.plan,
            layer,
            self.owner_range(Placement::Device)?.0 as u64,
            self.host_cpu_range()?.0,
            self.owner_range(Placement::HostMapped)?.0 as u64,
        )
        .ok()?;
        Some(
            specs
                .into_iter()
                .map(|ranges| ExpertCacheSlot {
                    ranges,
                    lease: self.clone(),
                })
                .collect(),
        )
    }

    /// Allocate independent mutable GU/DN pointer tables for one layer,
    /// initialized from the canonical view. Their lease, not this tier lease
    /// alone, must be retained by every recorded/retained dispatch capability.
    /// Returned tensor descriptors are borrowed and must never be freed by a
    /// runtime's owns_expert_tables path.
    pub fn allocate_expert_cache_tables(
        &self,
        gpu: &mut Gpu,
        layer: usize,
    ) -> Result<ExpertCacheTableLease, ExpertTierOwnerError> {
        let canonical = self
            .expert_cache_layer(layer)
            .ok_or_else(|| decode_share_error("no canonical cache view for layer".to_string()))?;
        gpu.bind_thread().map_err(runtime_error)?;
        let (hip, device_id) = {
            let owners = self.owners.lock().map_err(|_| {
                ExpertTierOwnerError::Runtime("expert payload owner mutex poisoned".to_string())
            })?;
            (owners.hip.clone(), owners.device_id)
        };
        if gpu.device_id != device_id {
            return Err(ExpertTierOwnerError::Runtime(
                "cache tables requested on another device".to_string(),
            ));
        }
        let mut owner = ExpertCacheTableOwner {
            hip,
            device_id,
            tables: [None, None],
            _tier: self.clone(),
        };
        let mut bases = [0usize; 2];
        for (index, pointers) in [canonical.core.gate_up_ptrs(), canonical.core.down_ptrs()]
            .into_iter()
            .enumerate()
        {
            owner.tables[index] = Some(GpuTensor {
                buf: owner.hip.malloc(EXPERTS * 8).map_err(runtime_error)?,
                shape: vec![2 * EXPERTS],
                dtype: DType::F32,
            });
            let table = owner.tables[index]
                .as_ref()
                .expect("installed before upload");
            owner
                .hip
                .memcpy_htod(&table.buf, pointer_bytes(pointers))
                .map_err(runtime_error)?;
            bases[index] = table.buf.as_ptr() as usize;
        }
        if let Err(error) = owner.hip.device_synchronize() {
            // Failed visibility cannot authorize freeing either table or its
            // canonical backing. Retain the complete owner, including the tier
            // lease that keeps the pointer-upload source mirrors alive. A later
            // outer cleanup must not drop these allocations after this failure.
            std::mem::forget(owner);
            return Err(runtime_error(error));
        }
        Ok(ExpertCacheTableLease {
            owner: Arc::new(Mutex::new(owner)),
            bases,
            generation: self.generation(),
            layer,
        })
    }

    /// Digest of one read view: the plan's logical digest for `Resident`, the
    /// BorrowPrefill digest for `BorrowPrefill` (`None` on a non-sharing lease).
    pub fn view_digest(&self, view: ExpertReadView) -> Option<[u8; 32]> {
        self.binding.view_digest(view)
    }

    /// Claim the donor tail for one forward: while the claim lives, no other
    /// forward may borrow it (two forwards would dirty the same bytes under
    /// each other's loan state). `Err` when the lease carries no donor tail or
    /// is already claimed. Dropping the claim releases it and retains the
    /// lease, so the owners outlive every GPU use the claimant made.
    pub fn claim_decode_share(&self) -> Result<DecodeShareClaim, String> {
        if self.binding.decode_share.is_none() {
            return Err("the tier lease carries no decode-share donor tail".to_string());
        }
        if !self.binding.try_claim_decode_share() {
            return Err(
                "the decode-share donor tail is already claimed by another forward".to_string(),
            );
        }
        Ok(DecodeShareClaim {
            lease: self.clone(),
        })
    }

    /// Debug/gate oracle: synchronize the device, copy the first
    /// `prefix_bytes` of the donor tail back in bounded chunks and compare
    /// them byte-for-byte with the pinned mirror. Returns the SHA-256 of the
    /// donor bytes, or an error naming the first differing offset.
    pub fn verify_donor_tail(
        &self,
        gpu: &mut Gpu,
        prefix_bytes: usize,
    ) -> Result<[u8; 32], String> {
        let share = self
            .binding
            .decode_share
            .as_ref()
            .ok_or_else(|| "lease carries no decode-share donor tail".to_string())?;
        let (mirror_cpu, donor_bytes) = share.ranges.mirror_cpu;
        if prefix_bytes > donor_bytes {
            return Err(format!(
                "donor tail prefix {prefix_bytes} exceeds the {donor_bytes}-byte donor extent"
            ));
        }
        gpu.bind_thread().map_err(|error| error.to_string())?;
        let owners = self
            .owners
            .lock()
            .map_err(|_| "expert tier owners lock poisoned".to_string())?;
        owners
            .hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let device = owners
            .device
            .as_ref()
            .ok_or_else(|| "decode-share lease has no device owner".to_string())?;
        let donor_offset =
            usize::try_from(share.donor_offset).map_err(|_| "donor offset overflow".to_string())?;
        // SAFETY: `mirror_cpu` designates `donor_bytes` bytes of the pinned
        // host owner kept alive by `owners` (locked above); nothing writes the
        // mirror after materialization.
        let mirror = unsafe { std::slice::from_raw_parts(mirror_cpu as *const u8, prefix_bytes) };
        verify_tail_bytes(mirror, |chunk, offset| {
            owners
                .hip
                .memcpy_dtoh_at(chunk, device, donor_offset + offset)
                .map_err(|error| error.to_string())
        })
    }
}

/// Exclusive ownership of a lease's donor tail ([`ExpertTierLease::claim_decode_share`]).
pub struct DecodeShareClaim {
    lease: ExpertTierLease,
}

impl DecodeShareClaim {
    pub fn lease(&self) -> &ExpertTierLease {
        &self.lease
    }
}

impl Drop for DecodeShareClaim {
    fn drop(&mut self) {
        self.lease.binding.release_decode_share_claim();
    }
}

/// Which expert table set a read goes through.
///
/// `Resident` is the canonical placement: every table entry points at the
/// expert's own tier owner. On a decode-share lease,
/// `BorrowPrefill(class)` maps the first `n_class` donors (physical order; the donors
/// that prefill capacity class `class` can overwrite) point at the pinned host
/// mirror, every other entry is identical to `Resident`. Class views are
/// nested; the last class is the all-donors view. A cache plan exposes only
/// the canonical all-donors class 0, without enabling scratch sharing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertReadView {
    Resident,
    BorrowPrefill(usize),
}

impl ExpertTierBinding {
    fn borrowed_view(&self, class: usize) -> Option<&BorrowView> {
        self.decode_share
            .as_ref()
            .or(self.expert_cache.as_ref())?
            .views
            .get(class)
    }

    /// Take the donor-tail claim; `false` when it is already held.
    fn try_claim_decode_share(&self) -> bool {
        self.decode_share_claimed
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
    }

    fn release_decode_share_claim(&self) {
        self.decode_share_claimed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Digest of one read view; `None` for `BorrowPrefill` without donors or
    /// past the last class.
    fn view_digest(&self, view: ExpertReadView) -> Option<[u8; 32]> {
        match view {
            ExpertReadView::Resident => Some(self.logical_digest),
            ExpertReadView::BorrowPrefill(class) => {
                self.borrowed_view(class).map(|view| view.digest)
            }
        }
    }

    /// One layer of one view; `None` for an out-of-range layer or for
    /// `BorrowPrefill(class)` on a binding without donors or past its classes.
    fn layer_view(&self, layer: usize, view: ExpertReadView) -> Option<BindingLayerView<'_>> {
        if layer >= TRUNK_LAYERS as usize {
            return None;
        }
        if let ExpertReadView::BorrowPrefill(class) = view {
            if self.borrowed_view(class).is_none() {
                return None;
            }
        }
        Some(BindingLayerView {
            binding: self,
            layer,
            view,
        })
    }
}

/// The CPU-only core of a layer view: every table read goes through the
/// selected view's data. [`LayerExpertTierView`] adds the lease retention.
#[derive(Clone, Copy)]
struct BindingLayerView<'a> {
    binding: &'a ExpertTierBinding,
    layer: usize,
    view: ExpertReadView,
}

impl<'a> BindingLayerView<'a> {
    fn entry(&self, eid: usize) -> Option<&'a ExpertTierEntry> {
        match self.view {
            ExpertReadView::Resident => self.binding.plan.entry(self.layer, eid),
            ExpertReadView::BorrowPrefill(_) => {
                if eid >= EXPERTS {
                    return None;
                }
                self.borrow().entries.get(self.layer * EXPERTS + eid)
            }
        }
    }

    /// The class view a `BorrowPrefill(class)` layer reads.
    fn borrow(&self) -> &'a BorrowView {
        let ExpertReadView::BorrowPrefill(class) = self.view else {
            unreachable!("borrow() is only called on a BorrowPrefill view");
        };
        self.binding
            .borrowed_view(class)
            .expect("a BorrowPrefill view exists only with checked donor backing")
    }

    fn tables(&self) -> &'a Tables {
        match self.view {
            ExpertReadView::Resident => &self.binding.tables,
            ExpertReadView::BorrowPrefill(_) => &self.borrow().tables,
        }
    }

    fn gate_up_table_base(&self) -> usize {
        match self.view {
            ExpertReadView::Resident => self.binding.gate_up_table_base,
            ExpertReadView::BorrowPrefill(_) => self.borrow().gate_up_table_base,
        }
    }

    fn down_table_base(&self) -> usize {
        match self.view {
            ExpertReadView::Resident => self.binding.down_table_base,
            ExpertReadView::BorrowPrefill(_) => self.borrow().down_table_base,
        }
    }

    fn gate_up_ptrs(&self) -> &'a [usize] {
        let start = self.layer * EXPERTS;
        &self.tables().gate_up[start..start + EXPERTS]
    }

    fn down_ptrs(&self) -> &'a [usize] {
        let start = self.layer * EXPERTS;
        &self.tables().down[start..start + EXPERTS]
    }

    fn codec_table(&self) -> &'a [u8] {
        // Codecs are identical in both views (asserted at materialization).
        let start = self.layer * EXPERTS;
        &self.binding.tables.codecs[start..start + EXPERTS]
    }

    fn any_host(&self) -> bool {
        (0..EXPERTS)
            .any(|eid| self.entry(eid).map(|entry| entry.tier) == Some(Placement::HostMapped))
    }
}

/// Borrowed frozen tables for one layer, in original eid order. GPU sub-views
/// are non-owning: never free them, and retain `lease()` across pending GPU work.
pub struct LayerExpertTierView<'a> {
    lease: &'a ExpertTierLease,
    core: BindingLayerView<'a>,
}

impl<'a> LayerExpertTierView<'a> {
    pub fn layer(&self) -> usize {
        self.core.layer
    }

    pub fn view(&self) -> ExpertReadView {
        self.core.view
    }

    /// Digest of this view: the plan's logical digest for `Resident`, the
    /// BorrowPrefill digest for `BorrowPrefill`.
    pub fn view_digest(&self) -> [u8; 32] {
        self.lease
            .binding
            .view_digest(self.core.view)
            .expect("a layer view exists only for a view its lease carries")
    }

    pub fn entry(&self, eid: usize) -> Option<&'a ExpertTierEntry> {
        self.core.entry(eid)
    }

    pub fn placement(&self, eid: usize) -> Option<Placement> {
        self.entry(eid).map(|entry| entry.tier)
    }

    pub fn any_host(&self) -> bool {
        self.core.any_host()
    }

    pub fn gate_up_ptrs(&self) -> &'a [usize] {
        self.core.gate_up_ptrs()
    }

    pub fn down_ptrs(&self) -> &'a [usize] {
        self.core.down_ptrs()
    }

    pub fn codec_table(&self) -> &'a [u8] {
        self.core.codec_table()
    }

    pub fn gate_up_ptrs_device(&self) -> GpuTensor {
        self.pointer_table_view(self.core.gate_up_table_base())
    }

    pub fn down_ptrs_device(&self) -> GpuTensor {
        self.pointer_table_view(self.core.down_table_base())
    }

    pub fn codec_table_device(&self) -> GpuTensor {
        borrowed_table(
            self.lease.binding.codec_table_base,
            TOTAL_EXPERTS,
            DType::Raw,
        )
        .sub_offset(self.core.layer * EXPERTS, EXPERTS)
    }

    pub fn lease(&self) -> ExpertTierLease {
        self.lease.clone()
    }

    /// Forwards [`ExpertTierLease::owner_range`].
    pub fn owner_range(&self, tier: Placement) -> Option<(usize, usize)> {
        self.lease.owner_range(tier)
    }

    /// Forwards [`ExpertTierLease::host_cpu_range`].
    pub fn host_cpu_range(&self) -> Option<(usize, usize)> {
        self.lease.host_cpu_range()
    }

    fn pointer_table_view(&self, base: usize) -> GpuTensor {
        // Existing MoE pointer tensors use two F32 storage elements per u64
        // address. F32 is metadata only: kernels read absolute 64-bit pointers.
        borrowed_table(base, 2 * TOTAL_EXPERTS, DType::F32)
            .sub_offset(2 * self.core.layer * EXPERTS, 2 * EXPERTS)
    }
}

fn borrowed_table(base: usize, elements: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: only frozen table addresses from a live lease reach here.
        // The returned descriptor is borrowed and must not outlive the lease.
        buf: unsafe {
            DeviceBuffer::from_raw(base as *mut std::ffi::c_void, elements * dtype.size())
        },
        shape: vec![elements],
        dtype,
    }
}

fn runtime_error(error: hip_bridge::HipError) -> ExpertTierOwnerError {
    ExpertTierOwnerError::Runtime(error.to_string())
}

fn pointer_bytes(entries: &[usize]) -> &[u8] {
    // SAFETY: usize has no padding or uninitialized bytes. build_tables rejects
    // non-64-bit platforms; the HIP host/device pointer ABI is native-endian.
    unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr().cast::<u8>(),
            std::mem::size_of_val(entries),
        )
    }
}

/// A logical payload span, never including table bytes or owner tail padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PayloadSpan {
    file_offset: u64,
    len: u64,
    tier: Placement,
    dst_offset: u64,
}

/// Preserve copy order, merging only runs contiguous in both address spaces.
fn coalesce_spans(spans: &[PayloadSpan]) -> Result<Vec<PayloadSpan>, ExpertTierOwnerError> {
    let mut runs: Vec<PayloadSpan> = Vec::with_capacity(spans.len());
    for span in spans {
        span.file_offset
            .checked_add(span.len)
            .ok_or(ExpertTierOwnerError::Overflow)?;
        span.dst_offset
            .checked_add(span.len)
            .ok_or(ExpertTierOwnerError::Overflow)?;
        if let Some(last) = runs.last_mut() {
            if last.tier == span.tier
                && last.file_offset.checked_add(last.len) == Some(span.file_offset)
                && last.dst_offset.checked_add(last.len) == Some(span.dst_offset)
            {
                last.len = last
                    .len
                    .checked_add(span.len)
                    .ok_or(ExpertTierOwnerError::Overflow)?;
                continue;
            }
        }
        runs.push(*span);
    }
    Ok(runs)
}

fn payload_spans(plan: &ExpertTierPlan) -> Result<Vec<PayloadSpan>, ExpertTierOwnerError> {
    let mut spans = Vec::new();
    for tier in [Placement::Device, Placement::HostMapped] {
        for copy in plan.copies(tier) {
            checked_extent(
                tier,
                copy.dst_offset,
                copy.src.len as u64,
                plan.owner_bytes(tier),
            )?;
            spans.push(PayloadSpan {
                file_offset: copy.src.offset as u64,
                len: copy.src.len as u64,
                tier,
                dst_offset: copy.dst_offset,
            });
        }
    }
    for window in plan.host_windows() {
        for (span, offset) in &window.spans {
            checked_extent(
                Placement::HostMapped,
                *offset,
                span.len as u64,
                plan.owner_bytes(Placement::HostMapped),
            )?;
            spans.push(PayloadSpan {
                file_offset: span.offset as u64,
                len: span.len as u64,
                tier: Placement::HostMapped,
                dst_offset: *offset,
            });
        }
    }
    coalesce_spans(&spans)
}

fn record_direct_stats(stats: hfq_parallel::DirectReadStats) {
    for (phase, ns, bytes) in [
        (
            "expert_tiers.direct_read",
            stats.direct_read_ns,
            stats.bytes_direct,
        ),
        (
            "expert_tiers.buffered_read",
            stats.buffered_read_ns,
            stats.bytes_buffered,
        ),
    ] {
        if bytes != 0 {
            rdna_compute::load_phases::add(phase, ns, bytes);
        }
    }
}

fn copy_span<'src>(
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
    source_offset: u64,
    len: u64,
    dst_offset: u64,
    mut write_chunk: impl FnMut(u64, &[u8]) -> Result<(), ExpertTierOwnerError>,
) -> Result<(), ExpertTierOwnerError> {
    source_offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    dst_offset
        .checked_add(len)
        .ok_or(ExpertTierOwnerError::Overflow)?;
    let mut copied = 0;
    while copied < len {
        let n = (len - copied).min(COPY_CHUNK_BYTES);
        let bytes = src(source_offset + copied, n).map_err(ExpertTierOwnerError::Source)?;
        if bytes.len() as u64 != n {
            return Err(ExpertTierOwnerError::SourceLength {
                expected: n,
                got: bytes.len(),
            });
        }
        write_chunk(dst_offset + copied, bytes)?;
        copied += n;
    }
    Ok(())
}

/// The direct path's spans: [`payload_spans`] with every source extent checked
/// against the artifact's EOF before anything is allocated.
///
/// Expert slab offsets are already ABSOLUTE file offsets (the sealed HFQ index
/// offsets, which the mmap path reads through `HfqFile::data_range` over the
/// whole-file mapping). The direct reader takes them unchanged: rebasing on
/// the HFQ data section shifts every expert (Q4L identity gate, 526f3b6ba).
fn direct_payload_spans(
    plan: &ExpertTierPlan,
    source: &DirectSource,
) -> Result<Vec<PayloadSpan>, ExpertTierOwnerError> {
    let spans = payload_spans(plan)?;
    let file_len = source.file_identity().len;
    for span in &spans {
        let end = span
            .file_offset
            .checked_add(span.len)
            .ok_or(ExpertTierOwnerError::Overflow)?;
        if end > file_len {
            return Err(ExpertTierOwnerError::Source(format!(
                "expert span {}+{} exceeds the {file_len}-byte artifact",
                span.file_offset, span.len
            )));
        }
    }
    Ok(spans)
}

/// Fill the host-mapped tier's spans of `spans` (from
/// [`direct_payload_spans`]) at `base` with direct parallel reads.
///
/// # Safety
/// `base` must designate a writable allocation covering every host span's
/// destination extent, exclusive to this call.
unsafe fn fill_host_spans(
    source: &DirectSource,
    spans: &[PayloadSpan],
    base: *mut u8,
) -> Result<hfq_parallel::DirectReadStats, ExpertTierOwnerError> {
    let copies = spans
        .iter()
        .filter(|span| span.tier == Placement::HostMapped)
        .map(|span| {
            let offset =
                usize::try_from(span.dst_offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
            Ok(HostRangeCopy {
                file_offset: span.file_offset,
                len: usize::try_from(span.len).map_err(|_| ExpertTierOwnerError::Overflow)?,
                dst: (base as usize)
                    .checked_add(offset)
                    .ok_or(ExpertTierOwnerError::Overflow)? as *mut u8,
            })
        })
        .collect::<Result<Vec<_>, ExpertTierOwnerError>>()?;
    // SAFETY: forwarded from the caller's contract.
    unsafe { hfq_parallel::read_ranges_into_host(source, &copies, HFQ_READER_LANES) }
        .map_err(|error| ExpertTierOwnerError::Source(error.to_string()))
}

impl ExpertTierOwners {
    fn allocate_device(&self, bytes: u64) -> Result<Option<DeviceBuffer>, ExpertTierOwnerError> {
        if bytes == 0 {
            return Ok(None);
        }
        let bytes = usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?;
        self.hip.malloc(bytes).map(Some).map_err(runtime_error)
    }

    fn allocate_host(&mut self, bytes: u64) -> Result<(), ExpertTierOwnerError> {
        if bytes == 0 {
            return Ok(());
        }
        let bytes = usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?;
        let alloc_started = std::time::Instant::now();
        let _alloc_phase = AllocPhase(alloc_started, bytes as u64);
        let cpu = self
            .hip
            .host_malloc(bytes, HIP_HOST_MALLOC_MAPPED)
            .map_err(runtime_error)?;
        // Record the owner immediately, so alias lookup failure also frees it.
        self.host = Some(HostPayload {
            cpu_address: cpu as usize,
            device_alias: 0,
            bytes,
        });
        let alias = self
            .hip
            .host_get_device_pointer(cpu, 0)
            .map_err(runtime_error)?;
        if alias.is_null() {
            return Err(ExpertTierOwnerError::Runtime(
                "hipHostGetDevicePointer returned null".into(),
            ));
        }
        self.host
            .as_mut()
            .expect("host allocated above")
            .device_alias = alias as usize as u64;
        Ok(())
    }

    fn copy_payload<'src>(
        &self,
        tier: Placement,
        source_offset: u64,
        len: u64,
        dst_offset: u64,
        src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
    ) -> Result<(), ExpertTierOwnerError> {
        checked_extent(tier, dst_offset, len, self.plan.owner_bytes(tier))?;
        let copy_started = std::time::Instant::now();
        let copied = copy_span(src, source_offset, len, dst_offset, |offset, chunk| {
            let offset = usize::try_from(offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
            match tier {
                Placement::Device => {
                    let owner = self
                        .device
                        .as_ref()
                        .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                    self.hip
                        .memcpy_htod_offset(owner, offset, chunk)
                        .map_err(runtime_error)
                }
                Placement::HostMapped => {
                    let owner = self
                        .host
                        .as_ref()
                        .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                    // SAFETY: the entire span was checked against the owner;
                    // chunk offsets are inside it and the allocation is private
                    // until all payloads and frozen tables have been published.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk.as_ptr(),
                            (owner.cpu_address as *mut u8).add(offset),
                            chunk.len(),
                        );
                    }
                    Ok(())
                }
            }
        });
        rdna_compute::load_phases::add(
            match tier {
                Placement::Device => "expert_tiers.h2d",
                Placement::HostMapped => "expert_tiers.host_copy",
            },
            rdna_compute::load_phases::ns_since(copy_started),
            len,
        );
        copied
    }

    fn read_payloads(
        &self,
        gpu: &mut Gpu,
        source: &DirectSource,
        spans: &[PayloadSpan],
    ) -> Result<(), ExpertTierOwnerError> {
        self.zero_tail(Placement::Device)?;
        if let Some(owner) = &self.device {
            let copies = spans
                .iter()
                .filter(|span| span.tier == Placement::Device)
                .map(|span| {
                    Ok(DeviceRangeCopy {
                        file_offset: span.file_offset,
                        len: usize::try_from(span.len)
                            .map_err(|_| ExpertTierOwnerError::Overflow)?,
                        dst: owner,
                        dst_offset: usize::try_from(span.dst_offset)
                            .map_err(|_| ExpertTierOwnerError::Overflow)?,
                    })
                })
                .collect::<Result<Vec<_>, ExpertTierOwnerError>>()?;
            let stats =
                hfq_parallel::upload_ranges_to_device(gpu, source, &copies, HFQ_READER_LANES)
                    .map_err(ExpertTierOwnerError::Source)?;
            record_direct_stats(stats);
            rdna_compute::load_phases::add(
                "expert_tiers.h2d",
                stats.h2d_ns,
                copies.iter().map(|copy| copy.len as u64).sum(),
            );
        }
        self.zero_tail(Placement::HostMapped)?;
        if let Some(owner) = &self.host {
            // SAFETY: payload_spans checked every destination extent against
            // this private pinned allocation. Plan copies do not overlap.
            let stats = unsafe { fill_host_spans(source, spans, owner.cpu_address as *mut u8) }?;
            record_direct_stats(stats);
        }
        Ok(())
    }

    fn write_payload_digest(
        &self,
        path: &std::path::Path,
        spans: &[PayloadSpan],
    ) -> Result<(), ExpertTierOwnerError> {
        use hipfire_runtime::weight_store::{
            append_weight_digest_line, Sha256Stream, WeightDigestRow,
        };
        // Hash destinations, not mmap sources: this detects read/copy errors.
        // A single bounded scratch is reused across all logical device ranges.
        let mut scratch = Vec::new();
        for (tier, name, residency) in [
            (Placement::Device, "qwen4.expert_tiers.device", "device"),
            (
                Placement::HostMapped,
                "qwen4.expert_tiers.host",
                "host_mapped",
            ),
        ] {
            let mut hash = Sha256Stream::new();
            let mut logical_bytes = 0u64;
            for span in spans.iter().filter(|span| span.tier == tier) {
                logical_bytes = logical_bytes
                    .checked_add(span.len)
                    .ok_or(ExpertTierOwnerError::Overflow)?;
                let offset =
                    usize::try_from(span.dst_offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
                let len = usize::try_from(span.len).map_err(|_| ExpertTierOwnerError::Overflow)?;
                match tier {
                    Placement::HostMapped => {
                        let owner = self
                            .host
                            .as_ref()
                            .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                        // SAFETY: spans contain only checked logical extents
                        // within the private pinned owner, now fully filled.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                (owner.cpu_address as *const u8).add(offset),
                                len,
                            )
                        };
                        hash.update(bytes);
                    }
                    Placement::Device => {
                        let owner = self
                            .device
                            .as_ref()
                            .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                        let mut copied = 0;
                        while copied < len {
                            let n = (len - copied).min(COPY_CHUNK_BYTES as usize);
                            scratch.resize(n, 0);
                            self.hip
                                .memcpy_dtoh_at(&mut scratch, owner, offset + copied)
                                .map_err(runtime_error)?;
                            hash.update(&scratch);
                            copied += n;
                        }
                    }
                }
            }
            if logical_bytes != 0 {
                let sha256 = hash.finish_hex();
                append_weight_digest_line(
                    path,
                    &WeightDigestRow {
                        name,
                        layer: None,
                        residency,
                        dtype: "expert_payload",
                        logical_bytes,
                        sha256: &sha256,
                    },
                )
                .map_err(|error| ExpertTierOwnerError::Source(error.to_string()))?;
            }
        }
        Ok(())
    }

    fn zero_tail(&self, tier: Placement) -> Result<(), ExpertTierOwnerError> {
        let bytes = self.plan.owner_bytes(tier);
        if bytes == 0 {
            return Ok(());
        }
        let tail = tail_pad_bytes();
        let offset = bytes
            .checked_sub(tail)
            .ok_or(ExpertTierOwnerError::TailPad { tier })?;
        let offset = usize::try_from(offset).map_err(|_| ExpertTierOwnerError::Overflow)?;
        let tail = usize::try_from(tail).map_err(|_| ExpertTierOwnerError::Overflow)?;
        match tier {
            Placement::Device => {
                let owner = self
                    .device
                    .as_ref()
                    .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                self.hip
                    .memset(&owner.byte_view(offset, tail), 0, tail)
                    .map_err(runtime_error)
            }
            Placement::HostMapped => {
                let owner = self
                    .host
                    .as_ref()
                    .ok_or(ExpertTierOwnerError::MissingBase(tier))?;
                debug_assert_eq!(offset + tail, owner.bytes);
                // SAFETY: the checked tail range is inside the sole pinned
                // allocation and no consumer has access during materialization.
                unsafe {
                    std::ptr::write_bytes((owner.cpu_address as *mut u8).add(offset), 0, tail)
                };
                Ok(())
            }
        }
    }
}

/// Records one `expert_tiers.host_alloc` call (pinned host allocation plus its
/// device alias lookup) when the allocating scope ends, error paths included.
struct AllocPhase(std::time::Instant, u64);

impl Drop for AllocPhase {
    fn drop(&mut self) {
        rdna_compute::load_phases::add(
            "expert_tiers.host_alloc",
            rdna_compute::load_phases::ns_since(self.0),
            self.1,
        );
    }
}

impl Drop for ExpertTierOwners {
    fn drop(&mut self) {
        // A keepalive may be retired on another thread without a Gpu. HIP frees
        // must target the owning device. A failed bind cannot safely free here.
        let previous_device = match self.hip.current_device() {
            Ok(device) => device,
            Err(_) => return,
        };
        if self.hip.set_device(self.device_id).is_err() {
            return;
        }
        for table in [
            &mut self.gate_up_table,
            &mut self.down_table,
            &mut self.codec_table,
        ] {
            if let Some(table) = table.take() {
                let _ = self.hip.free(table.buf);
            }
        }
        for table in self.borrow_tables.drain(..) {
            let _ = self.hip.free(table.buf);
        }
        if let Some(owner) = self.device.take() {
            let _ = self.hip.free(owner);
        }
        if let Some(owner) = self.host.take() {
            let _ = self
                .hip
                .host_free(owner.cpu_address as *mut std::ffi::c_void);
        }
        // Do not invalidate rdna-compute's thread-local device-binding cache.
        if previous_device != self.device_id {
            let _ = self.hip.set_device(previous_device);
        }
    }
}

/// Allocate, fill and freeze a validated placement plan using bounded mmap
/// copies. The source mapping need only remain valid until this call returns.
pub fn materialize<'src>(
    gpu: &mut Gpu,
    plan: Arc<ExpertTierPlan>,
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
) -> Result<ExpertTierLease, ExpertTierOwnerError> {
    materialize_impl(gpu, plan, src, None)
}

/// Fill the same owners directly from the artifact, without faulting mmap
/// payloads. Allocation, table publication and rollback match `materialize`.
pub fn materialize_direct<'src>(
    gpu: &mut Gpu,
    plan: Arc<ExpertTierPlan>,
    source: &DirectSource,
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
) -> Result<ExpertTierLease, ExpertTierOwnerError> {
    materialize_impl(gpu, plan, src, Some(source))
}

fn materialize_impl<'src>(
    gpu: &mut Gpu,
    plan: Arc<ExpertTierPlan>,
    src: &dyn Fn(u64, u64) -> Result<&'src [u8], String>,
    direct_source: Option<&DirectSource>,
) -> Result<ExpertTierLease, ExpertTierOwnerError> {
    let digest_path = hipfire_runtime::weight_store::weight_digest_path();
    let spans = match direct_source {
        Some(source) => Some(direct_payload_spans(&plan, source)?),
        None if digest_path.is_some() => Some(payload_spans(&plan)?),
        None => None,
    };
    gpu.bind_thread().map_err(runtime_error)?;
    // Gpu owns HipRuntime by value and exposes no shared handle. A second
    // dlopen of the same HIP library supplies a standalone lifetime-safe handle
    // without taking Gpu's pool or host registry ownership. No runtime hook is
    // required; a future shared-runtime accessor can avoid this load.
    let hip = Arc::new(HipRuntime::load().map_err(runtime_error)?);
    let mut owners = ExpertTierOwners {
        device: None,
        host: None,
        gate_up_table: None,
        down_table: None,
        codec_table: None,
        binding: None,
        borrow_tables: Vec::new(),
        plan,
        hip,
        device_id: gpu.device_id,
        generation: issue_generation(&GENERATION)?,
    };
    owners.device = owners.allocate_device(owners.plan.owner_bytes(Placement::Device))?;
    owners.allocate_host(owners.plan.owner_bytes(Placement::HostMapped))?;
    let tables = build_tables(
        &owners.plan,
        owners
            .device
            .as_ref()
            .map(|owner| owner.as_ptr() as usize as u64),
        owners.host.as_ref().map(|owner| owner.device_alias),
    )?;
    let decode_share_cpu = build_decode_share(
        &owners.plan,
        &tables,
        owners
            .device
            .as_ref()
            .map(|owner| owner.as_ptr() as usize as u64),
        owners.host.as_ref().map(|owner| owner.cpu_address),
        owners.host.as_ref().map(|owner| owner.device_alias),
    )?;
    let expert_cache_cpu = build_donor_tail(
        &owners.plan,
        owners.plan.expert_cache_tail(),
        &tables,
        owners
            .device
            .as_ref()
            .map(|owner| owner.as_ptr() as usize as u64),
        owners.host.as_ref().map(|owner| owner.cpu_address),
        owners.host.as_ref().map(|owner| owner.device_alias),
    )?;
    if let Some(source) = direct_source {
        owners.read_payloads(
            gpu,
            source,
            spans.as_deref().expect("direct spans planned above"),
        )?;
    } else {
        for tier in [Placement::Device, Placement::HostMapped] {
            owners.zero_tail(tier)?;
            for copy in owners.plan.copies(tier) {
                owners.copy_payload(
                    tier,
                    copy.src.offset as u64,
                    copy.src.len as u64,
                    copy.dst_offset,
                    src,
                )?;
            }
        }
        for window in owners.plan.host_windows() {
            for (span, offset) in &window.spans {
                owners.copy_payload(
                    Placement::HostMapped,
                    span.offset as u64,
                    span.len as u64,
                    *offset,
                    src,
                )?;
            }
        }
    }
    if let Some(path) = digest_path {
        owners
            .write_payload_digest(&path, spans.as_deref().expect("digest spans planned above"))?;
    }
    // Install each allocation in the RAII owner before its upload can fail.
    owners.gate_up_table = Some(GpuTensor {
        buf: owners
            .hip
            .malloc(TOTAL_EXPERTS * 8)
            .map_err(runtime_error)?,
        shape: vec![2 * TOTAL_EXPERTS],
        dtype: DType::F32,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.gate_up_table.as_ref().expect("allocated above").buf,
            pointer_bytes(&tables.gate_up),
        )
        .map_err(runtime_error)?;
    owners.down_table = Some(GpuTensor {
        buf: owners
            .hip
            .malloc(TOTAL_EXPERTS * 8)
            .map_err(runtime_error)?,
        shape: vec![2 * TOTAL_EXPERTS],
        dtype: DType::F32,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.down_table.as_ref().expect("allocated above").buf,
            pointer_bytes(&tables.down),
        )
        .map_err(runtime_error)?;
    owners.codec_table = Some(GpuTensor {
        buf: owners.hip.malloc(TOTAL_EXPERTS).map_err(runtime_error)?,
        shape: vec![TOTAL_EXPERTS],
        dtype: DType::Raw,
    });
    owners
        .hip
        .memcpy_htod(
            &owners.codec_table.as_ref().expect("allocated above").buf,
            &tables.codecs,
        )
        .map_err(runtime_error)?;
    owners.hip.device_synchronize().map_err(runtime_error)?;
    let mut owner_ranges = [None; 2];
    for (tier, base) in [
        (
            Placement::Device,
            owners
                .device
                .as_ref()
                .map(|owner| owner.as_ptr() as usize as u64),
        ),
        (
            Placement::HostMapped,
            owners.host.as_ref().map(|owner| owner.device_alias),
        ),
    ] {
        let bytes = owners.plan.owner_bytes(tier);
        if bytes != 0 {
            let base = base.ok_or(ExpertTierOwnerError::MissingBase(tier))?;
            owner_ranges[tier_index(tier)] = Some((
                usize::try_from(base).map_err(|_| ExpertTierOwnerError::Overflow)?,
                usize::try_from(bytes).map_err(|_| ExpertTierOwnerError::Overflow)?,
            ));
        }
    }
    let mut publish =
        |cpu: Option<DecodeShareCpu>| -> Result<Option<DecodeShareBinding>, ExpertTierOwnerError> {
            Ok(match cpu {
                Some(cpu) => {
                    let mut views = Vec::with_capacity(cpu.views.len());
                    for view in cpu.views {
                        // Each table is owned by `owners` as soon as it is allocated: an
                        // error below frees every table with the owners.
                        let mut bases = [0usize; 2];
                        for (base, pointers) in bases
                            .iter_mut()
                            .zip([&view.tables.gate_up, &view.tables.down])
                        {
                            owners.borrow_tables.push(GpuTensor {
                                buf: owners
                                    .hip
                                    .malloc(TOTAL_EXPERTS * 8)
                                    .map_err(runtime_error)?,
                                shape: vec![2 * TOTAL_EXPERTS],
                                dtype: DType::F32,
                            });
                            let table = owners.borrow_tables.last().expect("pushed above");
                            owners
                                .hip
                                .memcpy_htod(&table.buf, pointer_bytes(pointers))
                                .map_err(runtime_error)?;
                            *base = table.buf.as_ptr() as usize;
                        }
                        views.push(BorrowView {
                            entries: view.entries,
                            tables: view.tables,
                            gate_up_table_base: bases[0],
                            down_table_base: bases[1],
                            donors: view.donors,
                            digest: view.digest,
                        });
                    }
                    owners.hip.device_synchronize().map_err(runtime_error)?;
                    Some(DecodeShareBinding {
                        views,
                        ranges: cpu.ranges,
                        donor_offset: cpu.donor_offset,
                        class_spans: cpu.class_spans,
                    })
                }
                None => None,
            })
        };
    let decode_share = publish(decode_share_cpu)?;
    let expert_cache = publish(expert_cache_cpu)?;
    let binding = Arc::new(ExpertTierBinding {
        hotset_sidecar_sha256: std::sync::OnceLock::new(),
        decode_share_claimed: std::sync::atomic::AtomicBool::new(false),
        owner_ranges,
        host_cpu_range: owners
            .host
            .as_ref()
            .map(|host| (host.cpu_address, host.bytes)),
        logical_digest: owners.plan.logical_digest(),
        plan: owners.plan.clone(),
        tables,
        decode_share,
        expert_cache,
        gate_up_table_base: owners
            .gate_up_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        down_table_base: owners
            .down_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        codec_table_base: owners
            .codec_table
            .as_ref()
            .expect("allocated above")
            .buf
            .as_ptr() as usize,
        generation: owners.generation,
    });
    owners.binding = Some(binding.clone());
    Ok(ExpertTierLease {
        binding,
        owners: Arc::new(Mutex::new(owners)),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
    use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
    use crate::expert_slabs::ExpertFormat;
    use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, ExpertTierPlan, Placement};
    use std::sync::atomic::{AtomicU64, Ordering};

    const TOTAL: usize = 48 * 512;
    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;

    // 256-aligned, nonzero, far apart; host alias is deliberately NOT device + owner.
    const DEV_BASE: u64 = 0x0000_7000_0000_0000;
    const HOST_ALIAS: u64 = 0x0000_5000_1000_0100;

    /// Same arithmetic as the planner's private `tail_pad()`.
    fn tail() -> u64 {
        ((ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255
    }

    fn slabs_with(f: impl Fn(usize) -> (ExpertFormat, usize, usize)) -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..TOTAL)
            .map(|i| {
                let (format, gu, dn) = f(i);
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: gu,
                };
                cursor += gu;
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: dn,
                };
                cursor += dn;
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    fn uniform() -> Vec<ExpertSlab> {
        slabs_with(|_| (ExpertFormat::Mq4Xts, GU, DN))
    }

    fn format_cycle() -> Vec<ExpertSlab> {
        slabs_with(|i| match i % 3 {
            0 => (ExpertFormat::Mq4Xts, GU, DN),
            1 => (ExpertFormat::Mq3I4, GU, DN),
            _ => (ExpertFormat::Mq2I4, GU, DN),
        })
    }

    fn all_host_plan() -> ExpertTierPlan {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        assert_eq!(plan.owner_bytes(Placement::Device), 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        plan
    }

    fn all_device_plan() -> ExpertTierPlan {
        let rank: Vec<u32> = (0..TOTAL as u32).collect();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, u64::MAX).unwrap();
        assert_eq!(plan.device_experts(), TOTAL);
        assert!(plan.owner_bytes(Placement::Device) > 0);
        assert_eq!(plan.owner_bytes(Placement::HostMapped), 0);
        plan
    }

    fn mixed_plan() -> ExpertTierPlan {
        let rank = [0u32, 5, 600, 24_575];
        let budget = rank.len() as u64 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&format_cycle(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), 4);
        assert!(plan.owner_bytes(Placement::Device) > 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        plan
    }

    fn err<T>(r: Result<T, ExpertTierOwnerError>) -> ExpertTierOwnerError {
        match r {
            Ok(_) => panic!("expected Err, got Ok"),
            Err(e) => e,
        }
    }

    /// Verify every pointer of every one of the 24,576 experts against the plan.
    fn assert_tables(
        plan: &ExpertTierPlan,
        t: &Tables,
        device_base: Option<u64>,
        host_alias: Option<u64>,
    ) {
        assert_eq!(t.gate_up.len(), TOTAL);
        assert_eq!(t.down.len(), TOTAL);
        assert_eq!(t.codecs.len(), TOTAL);
        for layer in 0..48usize {
            for expert in 0..512usize {
                let i = layer * 512 + expert;
                let e = plan.entry(layer, expert).unwrap();
                let (base, owner) = match e.tier {
                    Placement::Device => {
                        (device_base.unwrap(), plan.owner_bytes(Placement::Device))
                    }
                    Placement::HostMapped => {
                        (host_alias.unwrap(), plan.owner_bytes(Placement::HostMapped))
                    }
                };
                let gu = t.gate_up[i] as u64;
                let dn = t.down[i] as u64;
                assert_eq!(gu, base + e.gate_up_offset, "gate_up {layer}/{expert}");
                assert_eq!(dn, base + e.down_offset, "down {layer}/{expert}");
                assert_eq!(gu % 256, 0, "gate_up align {layer}/{expert}");
                assert_eq!(dn % 256, 0, "down align {layer}/{expert}");
                // Inside the owner, leaving the zeroed tail pad after the payload.
                assert!(gu >= base && gu + e.gate_up_len <= base + owner - tail());
                assert!(dn >= base && dn + e.down_len <= base + owner - tail());
                // Projections of one expert never overlap.
                assert!(gu + e.gate_up_len <= dn || dn + e.down_len <= gu);
                let want = match e.format {
                    ExpertFormat::Mq4Xts => 0u8,
                    ExpertFormat::Mq3I4 => 1u8,
                    ExpertFormat::Mq2I4 => 2u8,
                    ExpertFormat::E8pI4 => 3u8,
                };
                assert_eq!(t.codecs[i], want, "codec {layer}/{expert}");
            }
        }
    }

    // ---- placement mixes -------------------------------------------------

    #[test]
    fn all_host_plan_needs_only_host_alias() {
        let plan = all_host_plan();
        let t = build_tables(&plan, None, Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, None, Some(HOST_ALIAS));
        for i in 0..TOTAL {
            assert!(t.gate_up[i] as u64 >= HOST_ALIAS);
            assert!(t.down[i] as u64 >= HOST_ALIAS);
        }
    }

    #[test]
    fn all_device_plan_needs_only_device_base() {
        let plan = all_device_plan();
        let t = build_tables(&plan, Some(DEV_BASE), None).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), None);
        for i in 0..TOTAL {
            assert!(t.gate_up[i] as u64 >= DEV_BASE);
            assert!(t.down[i] as u64 >= DEV_BASE);
        }
    }

    #[test]
    fn mixed_plan_fills_all_24576_pointers_from_the_right_owner() {
        let plan = mixed_plan();
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));

        let dev_end = DEV_BASE + plan.owner_bytes(Placement::Device);
        let host_end = HOST_ALIAS + plan.owner_bytes(Placement::HostMapped);
        let mut dev = 0usize;
        let mut host = 0usize;
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            let p = t.gate_up[i] as u64;
            match e.tier {
                Placement::Device => {
                    dev += 1;
                    assert!((DEV_BASE..dev_end).contains(&p));
                }
                Placement::HostMapped => {
                    host += 1;
                    assert!((HOST_ALIAS..host_end).contains(&p));
                }
            }
        }
        assert_eq!(dev, 4);
        assert_eq!(dev + host, TOTAL);
        // The four device experts are exactly the ranked ids.
        for id in [0usize, 5, 600, 24_575] {
            assert_eq!(
                plan.entry(id / 512, id % 512).unwrap().tier,
                Placement::Device
            );
        }
    }

    #[test]
    fn host_pointers_use_the_distinct_alias_not_the_device_base() {
        let plan = mixed_plan();
        let a = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let other_alias = HOST_ALIAS + 0x10_0000;
        let b = build_tables(&plan, Some(DEV_BASE), Some(other_alias)).unwrap();
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            match e.tier {
                Placement::Device => {
                    // Device pointers are independent of the host alias.
                    assert_eq!(a.gate_up[i], b.gate_up[i]);
                    assert_eq!(a.down[i], b.down[i]);
                }
                Placement::HostMapped => {
                    assert_eq!(b.gate_up[i] - a.gate_up[i], 0x10_0000);
                    assert_eq!(b.down[i] - a.down[i], 0x10_0000);
                    assert_eq!(a.gate_up[i] as u64, HOST_ALIAS + e.gate_up_offset);
                }
            }
        }
        // Changing the device base never moves host pointers.
        let c = build_tables(&plan, Some(DEV_BASE + 0x1000), Some(HOST_ALIAS)).unwrap();
        for i in 0..TOTAL {
            let e = plan.entry(i / 512, i % 512).unwrap();
            if e.tier == Placement::HostMapped {
                assert_eq!(a.gate_up[i], c.gate_up[i]);
                assert_eq!(a.down[i], c.down[i]);
            } else {
                assert_eq!(c.gate_up[i] - a.gate_up[i], 0x1000);
            }
        }
    }

    #[test]
    fn codec_bytes_are_mq4_0_w3_1_w2_2() {
        let plan = mixed_plan();
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_eq!(t.codecs[0], 0); // i % 3 == 0 -> Mq4Xts
        assert_eq!(t.codecs[1], 1); // Mq3I4
        assert_eq!(t.codecs[2], 2); // Mq2I4
        assert_eq!(t.codecs[TOTAL - 1], ((TOTAL - 1) % 3) as u8);
        let mut counts = [0usize; 3];
        for &c in &t.codecs {
            counts[c as usize] += 1;
        }
        assert_eq!(counts, [TOTAL / 3; 3]);
    }

    // ---- base presence ---------------------------------------------------

    #[test]
    fn empty_tier_needs_no_base() {
        // All-host: no device owner, so no device base is required.
        let t = build_tables(&all_host_plan(), None, Some(HOST_ALIAS));
        assert!(t.is_ok());
        // All-device: no host owner, so no host alias is required.
        let t = build_tables(&all_device_plan(), Some(DEV_BASE), None);
        assert!(t.is_ok());
    }

    #[test]
    fn populated_tier_without_base_is_rejected() {
        assert!(matches!(
            err(build_tables(&all_host_plan(), None, None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        assert!(matches!(
            err(build_tables(&all_host_plan(), Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        assert!(matches!(
            err(build_tables(&all_device_plan(), None, None)),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        assert!(matches!(
            err(build_tables(&all_device_plan(), None, Some(HOST_ALIAS))),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, None, Some(HOST_ALIAS))),
            ExpertTierOwnerError::MissingBase(Placement::Device)
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
    }

    // ---- alignment / address overflow -------------------------------------

    #[test]
    fn misaligned_base_is_rejected() {
        for bad in [DEV_BASE + 1, DEV_BASE + 128, DEV_BASE + 255] {
            match err(build_tables(&all_device_plan(), Some(bad), None)) {
                ExpertTierOwnerError::Alignment { address } => assert_ne!(address % 256, 0),
                other => panic!("device base {bad:#x}: {other:?}"),
            }
        }
        for bad in [HOST_ALIAS + 1, HOST_ALIAS + 64] {
            match err(build_tables(&all_host_plan(), None, Some(bad))) {
                ExpertTierOwnerError::Alignment { address } => assert_ne!(address % 256, 0),
                other => panic!("host alias {bad:#x}: {other:?}"),
            }
        }
        // Mixed plan: either misaligned base fails even when the other is fine.
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE + 4), Some(HOST_ALIAS))),
            ExpertTierOwnerError::Alignment { .. }
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), Some(HOST_ALIAS + 4))),
            ExpertTierOwnerError::Alignment { .. }
        ));
    }

    #[test]
    fn aligned_nonzero_base_at_page_and_256_boundaries_is_accepted() {
        for base in [256u64, 0x1000, 0x20_0000, DEV_BASE] {
            let t = build_tables(&all_device_plan(), Some(base), None).unwrap();
            assert_eq!(t.gate_up[0] as u64, base);
        }
    }

    #[test]
    fn address_overflow_is_rejected() {
        // Highest 256-aligned address: base + any nonzero offset or owner size wraps.
        let top = u64::MAX & !255;
        assert!(matches!(
            err(build_tables(&all_device_plan(), Some(top), None)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(build_tables(&all_host_plan(), None, Some(top))),
            ExpertTierOwnerError::Overflow
        ));
        // Base fits but base + owner_bytes does not.
        let plan = all_device_plan();
        let near = (u64::MAX - plan.owner_bytes(Placement::Device) / 2) & !255;
        assert!(matches!(
            err(build_tables(&plan, Some(near), None)),
            ExpertTierOwnerError::Overflow
        ));
        // Mixed: overflow on either side.
        let mixed = mixed_plan();
        assert!(matches!(
            err(build_tables(&mixed, Some(top), Some(HOST_ALIAS))),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(build_tables(&mixed, Some(DEV_BASE), Some(top))),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- checked_extent ----------------------------------------------------

    #[test]
    fn checked_extent_accepts_in_bounds_and_exact_end() {
        assert!(checked_extent(Placement::Device, 0, 1, 4096).is_ok());
        assert!(checked_extent(Placement::Device, 0, 4096, 4096).is_ok());
        assert!(checked_extent(Placement::HostMapped, 4095, 1, 4096).is_ok());
        assert!(checked_extent(Placement::HostMapped, 256, 3840, 4096).is_ok());
        // > 4 GiB offsets stay exact.
        let big = 12u64 << 30;
        assert!(checked_extent(Placement::HostMapped, big, 4096, big + 8192).is_ok());
    }

    #[test]
    fn checked_extent_rejects_out_of_bounds() {
        match err(checked_extent(Placement::Device, 1, 4096, 4096)) {
            ExpertTierOwnerError::Extent {
                tier,
                offset,
                len,
                owner_bytes,
            } => {
                assert_eq!(tier, Placement::Device);
                assert_eq!((offset, len, owner_bytes), (1, 4096, 4096));
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 0, 4097, 4096)),
            ExpertTierOwnerError::Extent {
                tier: Placement::HostMapped,
                ..
            }
        ));
        // Starts at the owner end / beyond it.
        assert!(matches!(
            err(checked_extent(Placement::Device, 4096, 1, 4096)),
            ExpertTierOwnerError::Extent { .. }
        ));
        assert!(matches!(
            err(checked_extent(Placement::Device, 9000, 1, 4096)),
            ExpertTierOwnerError::Extent { .. }
        ));
        // Empty owner holds nothing.
        assert!(matches!(
            err(checked_extent(Placement::Device, 0, 1, 0)),
            ExpertTierOwnerError::Extent { owner_bytes: 0, .. }
        ));
    }

    #[test]
    fn checked_extent_rejects_zero_len() {
        for owner in [0u64, 4096] {
            match err(checked_extent(Placement::Device, 0, 0, owner)) {
                ExpertTierOwnerError::Extent { len, .. } => assert_eq!(len, 0),
                other => panic!("owner {owner}: {other:?}"),
            }
        }
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 128, 0, 4096)),
            ExpertTierOwnerError::Extent { len: 0, .. }
        ));
    }

    #[test]
    fn checked_extent_rejects_arithmetic_overflow() {
        assert!(matches!(
            err(checked_extent(Placement::Device, u64::MAX, 1, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(Placement::Device, u64::MAX - 3, 4, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(Placement::HostMapped, 1, u64::MAX, u64::MAX)),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(checked_extent(
                Placement::HostMapped,
                u64::MAX,
                u64::MAX,
                u64::MAX
            )),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- host windows ------------------------------------------------------

    fn mtp_spans() -> [ExpertSlabSpan; 2] {
        [
            ExpertSlabSpan {
                offset: 5_000_000_000,
                len: 1000,
            },
            ExpertSlabSpan {
                offset: 6_000_000_000,
                len: 513,
            },
        ]
    }

    #[test]
    fn host_window_on_mixed_plan_is_validated_with_tail_present() {
        let plan = mixed_plan().with_host_window("mtp", &mtp_spans()).unwrap();
        let owner = plan.owner_bytes(Placement::HostMapped);
        // Planner contract the validator relies on: every window extent is
        // inside the host owner and leaves the full tail pad after it.
        let w = &plan.host_windows()[0];
        assert_eq!(w.spans.len(), 2);
        for (span, dst) in &w.spans {
            assert_eq!(dst % 256, 0);
            assert!(checked_extent(Placement::HostMapped, *dst, span.len as u64, owner).is_ok());
            assert!(dst + span.len as u64 + tail() <= owner);
        }
        assert_eq!(owner % 256, 0);
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));
    }

    #[test]
    fn host_window_on_all_host_plan_does_not_need_device_base() {
        let plan = all_host_plan()
            .with_host_window("mtp", &mtp_spans())
            .unwrap();
        let t = build_tables(&plan, None, Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, None, Some(HOST_ALIAS));
        assert!(matches!(
            err(build_tables(&plan, None, None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
    }

    #[test]
    fn host_window_on_all_device_plan_requires_host_alias() {
        let device = all_device_plan();
        assert_eq!(device.owner_bytes(Placement::HostMapped), 0);
        // Without a window the host alias is not needed.
        assert!(build_tables(&device, Some(DEV_BASE), None).is_ok());

        let plan = device.with_host_window("mtp", &mtp_spans()).unwrap();
        let host = plan.owner_bytes(Placement::HostMapped);
        // The window allocates the previously empty host tier: one tail pad.
        assert_eq!(host, 1024 + 768 + tail());
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), None)),
            ExpertTierOwnerError::MissingBase(Placement::HostMapped)
        ));
        let t = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_tables(&plan, &t, Some(DEV_BASE), Some(HOST_ALIAS));
        // Window start must also respect alias alignment and overflow.
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS + 8))),
            ExpertTierOwnerError::Alignment { .. }
        ));
        assert!(matches!(
            err(build_tables(&plan, Some(DEV_BASE), Some(u64::MAX & !255))),
            ExpertTierOwnerError::Overflow
        ));
    }

    // ---- generation counter ----------------------------------------------

    #[test]
    fn generation_is_monotonic_from_one() {
        let c = AtomicU64::new(0);
        let mut prev = 0u64;
        for want in 1..=64u64 {
            let g = issue_generation(&c).unwrap();
            assert_eq!(g, want);
            assert!(g > prev);
            prev = g;
        }
        assert_eq!(c.load(Ordering::SeqCst), 64);
    }

    #[test]
    fn generation_accepts_u64_max_then_exhausts_without_modifying() {
        let c = AtomicU64::new(u64::MAX - 2);
        assert_eq!(issue_generation(&c).unwrap(), u64::MAX - 1);
        assert_eq!(issue_generation(&c).unwrap(), u64::MAX);
        assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
        for _ in 0..3 {
            assert!(matches!(
                err(issue_generation(&c)),
                ExpertTierOwnerError::GenerationExhausted
            ));
            assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
        }
    }

    #[test]
    fn generation_exhausted_counter_stays_exhausted_from_max_directly() {
        let c = AtomicU64::new(u64::MAX);
        assert!(matches!(
            err(issue_generation(&c)),
            ExpertTierOwnerError::GenerationExhausted
        ));
        assert_eq!(c.load(Ordering::SeqCst), u64::MAX);
    }

    #[test]
    fn process_generation_static_is_strictly_increasing() {
        // Other tests may issue concurrently, so only strict increase is asserted.
        let a = issue_generation(&GENERATION).unwrap();
        let b = issue_generation(&GENERATION).unwrap();
        assert!(a >= 1);
        assert!(b > a);
    }

    #[test]
    fn source_copies_are_bounded_and_keep_offsets() {
        let chunk = vec![0xa5; COPY_CHUNK_BYTES as usize];
        let requests = std::cell::RefCell::new(Vec::new());
        let source = |offset, len| {
            requests.borrow_mut().push((offset, len));
            Ok(&chunk[..len as usize])
        };
        let mut writes = Vec::new();
        copy_span(
            &source,
            123,
            COPY_CHUNK_BYTES * 2 + 17,
            256,
            |offset, bytes| {
                assert!(bytes.iter().all(|byte| *byte == 0xa5));
                writes.push((offset, bytes.len() as u64));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            *requests.borrow(),
            vec![
                (123, COPY_CHUNK_BYTES),
                (123 + COPY_CHUNK_BYTES, COPY_CHUNK_BYTES),
                (123 + 2 * COPY_CHUNK_BYTES, 17)
            ]
        );
        assert_eq!(
            writes,
            vec![
                (256, COPY_CHUNK_BYTES),
                (256 + COPY_CHUNK_BYTES, COPY_CHUNK_BYTES),
                (256 + 2 * COPY_CHUNK_BYTES, 17)
            ]
        );
    }

    #[test]
    fn pointer_mirrors_are_the_exact_upload_bytes() {
        let tables = build_tables(&mixed_plan(), Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        for entries in [&tables.gate_up, &tables.down] {
            let bytes = pointer_bytes(entries);
            assert_eq!(bytes.len(), TOTAL * 8);
            for (entry, bytes) in entries.iter().zip(bytes.chunks_exact(8)) {
                assert_eq!(bytes, entry.to_ne_bytes());
            }
        }
    }

    // ---- payload span coalescing ------------------------------------------

    fn span(file_offset: u64, len: u64, tier: Placement, dst_offset: u64) -> PayloadSpan {
        PayloadSpan {
            file_offset,
            len,
            tier,
            dst_offset,
        }
    }

    fn span_bytes(spans: &[PayloadSpan]) -> u64 {
        spans.iter().map(|s| s.len).sum()
    }

    /// Logical payload bytes the plan asks to copy: tier copies plus host windows.
    fn plan_payload_bytes(plan: &ExpertTierPlan) -> u64 {
        let copies: u64 = [Placement::Device, Placement::HostMapped]
            .into_iter()
            .flat_map(|tier| plan.copies(tier))
            .map(|copy| copy.src.len as u64)
            .sum();
        let windows: u64 = plan
            .host_windows()
            .iter()
            .flat_map(|window| window.spans.iter())
            .map(|(span, _)| span.len as u64)
            .sum();
        copies + windows
    }

    #[test]
    fn coalesce_merges_contiguous_same_tier_runs_in_both_address_spaces() {
        let input = [
            span(100, 10, Placement::Device, 1000),
            span(110, 20, Placement::Device, 1010),
            span(130, 5, Placement::Device, 1030),
        ];
        let out = coalesce_spans(&input).unwrap();
        assert_eq!(out, vec![span(100, 35, Placement::Device, 1000)]);
        assert_eq!(span_bytes(&out), span_bytes(&input));

        let host = [
            span(0, 256, Placement::HostMapped, 512),
            span(256, 256, Placement::HostMapped, 768),
        ];
        assert_eq!(
            coalesce_spans(&host).unwrap(),
            vec![span(0, 512, Placement::HostMapped, 512)]
        );
    }

    #[test]
    fn coalesce_splits_on_source_gap() {
        let input = [
            span(100, 10, Placement::Device, 1000),
            span(111, 10, Placement::Device, 1010),
        ];
        let out = coalesce_spans(&input).unwrap();
        assert_eq!(out, input.to_vec());
        assert_eq!(span_bytes(&out), 20);
    }

    #[test]
    fn coalesce_splits_on_destination_gap() {
        let input = [
            span(100, 10, Placement::HostMapped, 1000),
            span(110, 10, Placement::HostMapped, 1011),
        ];
        let out = coalesce_spans(&input).unwrap();
        assert_eq!(out, input.to_vec());
        assert_eq!(span_bytes(&out), 20);
    }

    #[test]
    fn coalesce_splits_on_tier_change_even_when_contiguous() {
        let input = [
            span(100, 10, Placement::Device, 1000),
            span(110, 10, Placement::HostMapped, 1010),
            span(120, 10, Placement::Device, 1020),
        ];
        let out = coalesce_spans(&input).unwrap();
        assert_eq!(out, input.to_vec());
        assert_eq!(span_bytes(&out), 30);
    }

    #[test]
    fn coalesce_empty_input_is_empty() {
        assert_eq!(coalesce_spans(&[]).unwrap(), Vec::new());
    }

    #[test]
    fn coalesce_rejects_file_end_overflow() {
        assert!(matches!(
            err(coalesce_spans(&[span(u64::MAX, 1, Placement::Device, 0)])),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(coalesce_spans(&[
                span(0, 1, Placement::Device, 0),
                span(u64::MAX - 3, 4, Placement::HostMapped, 0),
            ])),
            ExpertTierOwnerError::Overflow
        ));
    }

    #[test]
    fn coalesce_rejects_destination_end_overflow() {
        assert!(matches!(
            err(coalesce_spans(&[span(
                0,
                1,
                Placement::HostMapped,
                u64::MAX
            )])),
            ExpertTierOwnerError::Overflow
        ));
        assert!(matches!(
            err(coalesce_spans(&[
                span(0, 1, Placement::Device, 0),
                span(1, 4, Placement::Device, u64::MAX - 3),
            ])),
            ExpertTierOwnerError::Overflow
        ));
    }

    #[test]
    fn payload_spans_all_host_plan_preserves_logical_bytes() {
        let plan = all_host_plan();
        let spans = payload_spans(&plan).unwrap();
        assert!(!spans.is_empty());
        assert!(spans
            .iter()
            .all(|s| s.tier == Placement::HostMapped && s.len > 0));
        assert_eq!(span_bytes(&spans), plan_payload_bytes(&plan));
        assert_eq!(span_bytes(&spans), TOTAL as u64 * (GU + DN) as u64);
        // Already maximal: a second pass changes nothing.
        assert_eq!(coalesce_spans(&spans).unwrap(), spans);
    }

    #[test]
    fn payload_spans_all_device_plan_preserves_logical_bytes() {
        let plan = all_device_plan();
        let spans = payload_spans(&plan).unwrap();
        assert!(!spans.is_empty());
        assert!(spans
            .iter()
            .all(|s| s.tier == Placement::Device && s.len > 0));
        assert_eq!(span_bytes(&spans), plan_payload_bytes(&plan));
        assert_eq!(span_bytes(&spans), TOTAL as u64 * (GU + DN) as u64);
        assert_eq!(coalesce_spans(&spans).unwrap(), spans);
    }

    #[test]
    fn payload_spans_mixed_plan_preserves_logical_bytes_per_tier() {
        let plan = mixed_plan();
        let spans = payload_spans(&plan).unwrap();
        assert_eq!(span_bytes(&spans), plan_payload_bytes(&plan));
        for tier in [Placement::Device, Placement::HostMapped] {
            let want: u64 = plan.copies(tier).map(|c| c.src.len as u64).sum();
            let got: u64 = spans.iter().filter(|s| s.tier == tier).map(|s| s.len).sum();
            assert!(want > 0);
            assert_eq!(got, want, "{tier:?}");
        }
        assert_eq!(coalesce_spans(&spans).unwrap(), spans);
    }

    #[test]
    fn payload_spans_include_host_windows_in_every_plan_mix() {
        let window_bytes: u64 = mtp_spans().iter().map(|s| s.len as u64).sum();
        for base in [all_host_plan(), all_device_plan(), mixed_plan()] {
            let without = payload_spans(&base).unwrap();
            let plan = base.with_host_window("mtp", &mtp_spans()).unwrap();
            let spans = payload_spans(&plan).unwrap();
            assert_eq!(span_bytes(&spans), plan_payload_bytes(&plan));
            assert_eq!(span_bytes(&spans), span_bytes(&without) + window_bytes);
            // The far-apart window sources are never merged away.
            for (src, dst) in &plan.host_windows()[0].spans {
                assert!(
                    spans.iter().any(|s| s.tier == Placement::HostMapped
                        && s.file_offset <= src.offset as u64
                        && src.offset as u64 + src.len as u64 <= s.file_offset + s.len
                        && s.dst_offset + (src.offset as u64 - s.file_offset) == *dst),
                    "window span {src:?} -> {dst} missing"
                );
            }
            assert_eq!(coalesce_spans(&spans).unwrap(), spans);
        }
    }

    /// Regression (Q4L identity gate, 526f3b6ba): slab offsets come from the
    /// sealed HFQ index and are ABSOLUTE file offsets. The direct fill must
    /// place exactly the bytes the mmap path reads through
    /// `HfqFile::data_range`, on a real HFQ file whose data section starts at
    /// a nonzero offset (rebasing on `data_offset` shifts every expert).
    #[test]
    fn direct_host_fill_matches_mmap_on_nonzero_data_offset_hfq() {
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};
        const FGU: usize = 40;
        const FDN: usize = 24;
        let path = std::env::temp_dir().join(format!(
            "q4l-abs-offsets-{}-{:?}.hfq",
            std::process::id(),
            std::thread::current().id()
        ));
        let tensors: Vec<HfqMemTensor> = (0..TOTAL)
            .flat_map(|i| {
                [("gate_up", FGU, 0usize), ("down", FDN, 1)].map(|(proj, len, p)| {
                    HfqMemTensor {
                        name: format!("expert.{i}.{proj}"),
                        quant_type: 0,
                        shape: vec![len as u32],
                        group_size: 0,
                        // Distinct per tensor and position: any shifted read differs.
                        data: (0..len)
                            .map(|j| {
                                (((i * 2 + p) * 4096 + j) as u32)
                                    .wrapping_mul(2_654_435_761)
                                    .rotate_right(13) as u8
                            })
                            .collect(),
                    }
                })
            })
            .collect();
        write_hfqm_package_mem(&path, 0, "{}", &tensors).unwrap();
        let hfq = HfqFile::open(&path).unwrap();
        assert!(
            hfq.data_offset() > 0,
            "fixture must have a nonzero data section offset"
        );
        let infos = hfq.tensors();
        let slabs: Vec<ExpertSlab> = (0..TOTAL)
            .map(|i| {
                let (gu, dn) = (&infos[2 * i], &infos[2 * i + 1]);
                ExpertSlab {
                    format: ExpertFormat::Mq4Xts,
                    gate_up: ExpertSlabSpan {
                        offset: gu.data_offset,
                        len: gu.data_size,
                    },
                    down: ExpertSlabSpan {
                        offset: dn.data_offset,
                        len: dn.data_size,
                    },
                }
            })
            .collect();
        // The fixture detects the rebasing bug: a data_offset-shifted span
        // reads different bytes (or falls outside the file).
        let first = slabs[0].gate_up;
        assert_ne!(
            hfq.data_range(first.offset + hfq.data_offset() as usize, first.len),
            hfq.data_range(first.offset, first.len)
        );
        let source = DirectSource::open(&hfq).unwrap();
        let rank = [0u32, 7, 513, 24_575];
        for plan in [
            plan_expert_tiers_from_slabs(&slabs, &[], 0).unwrap(),
            plan_expert_tiers_from_slabs(&slabs, &rank, u64::MAX).unwrap(),
        ] {
            let spans = direct_payload_spans(&plan, &source).unwrap();
            let mut owner = vec![0u8; plan.owner_bytes(Placement::HostMapped) as usize];
            // SAFETY: `owner` covers the whole host tier, exclusive to the call.
            unsafe { fill_host_spans(&source, &spans, owner.as_mut_ptr()) }.unwrap();
            // Device spans carry the file offsets `upload_ranges_to_device`
            // reads; read them into a host image of the device owner.
            let mut device = vec![0u8; plan.owner_bytes(Placement::Device) as usize];
            let device_reads: Vec<HostRangeCopy> = spans
                .iter()
                .filter(|span| span.tier == Placement::Device)
                .map(|span| HostRangeCopy {
                    file_offset: span.file_offset,
                    len: span.len as usize,
                    dst: device[span.dst_offset as usize..].as_mut_ptr(),
                })
                .collect();
            // SAFETY: every extent was checked inside the device owner image.
            unsafe { hfq_parallel::read_ranges_into_host(&source, &device_reads, 4) }.unwrap();
            for (tier, image) in [
                (Placement::HostMapped, &owner),
                (Placement::Device, &device),
            ] {
                for copy in plan.copies(tier) {
                    let dst = copy.dst_offset as usize;
                    assert_eq!(
                        &image[dst..dst + copy.src.len],
                        hfq.data_range(copy.src.offset, copy.src.len).unwrap(),
                        "{tier:?} expert {}/{} {:?}",
                        copy.layer,
                        copy.expert,
                        copy.proj
                    );
                }
            }
        }
        // A span past EOF is refused before any allocation.
        let file_len = hfq.file_len() as usize;
        let past_eof: Vec<ExpertSlab> = slabs
            .iter()
            .map(|slab| ExpertSlab {
                format: slab.format,
                gate_up: ExpertSlabSpan {
                    offset: slab.gate_up.offset + file_len,
                    ..slab.gate_up
                },
                down: ExpertSlabSpan {
                    offset: slab.down.offset + file_len,
                    ..slab.down
                },
            })
            .collect();
        let plan = plan_expert_tiers_from_slabs(&past_eof, &[], 0).unwrap();
        assert!(matches!(
            direct_payload_spans(&plan, &source),
            Err(ExpertTierOwnerError::Source(_))
        ));
        drop(source);
        drop(hfq);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod decode_share_tests {
    use super::*;
    use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
    use crate::expert_tier_plan::plan_expert_tiers_from_slabs;

    const TOTAL: usize = 48 * 512;
    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;
    const DEV_BASE: u64 = 0x0000_7000_0000_0000;
    const HOST_ALIAS: u64 = 0x0000_5000_1000_0100;
    const HOST_CPU: usize = 0x0000_6000_0000_0000;

    fn tail() -> u64 {
        ((ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255
    }

    fn format_cycle() -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..TOTAL)
            .map(|i| {
                let format = match i % 3 {
                    0 => ExpertFormat::Mq4Xts,
                    1 => ExpertFormat::Mq3I4,
                    _ => ExpertFormat::Mq2I4,
                };
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: GU,
                };
                cursor += GU;
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: DN,
                };
                cursor += DN;
                ExpertSlab {
                    format,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    /// Four resident experts of different layers (all layers keep host experts).
    fn plain_plan() -> (ExpertTierPlan, Vec<u32>) {
        let rank = vec![0u32, 5, 600, 24_575];
        let budget = rank.len() as u64 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&format_cycle(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), 4);
        (plan, rank)
    }

    fn shared_plan() -> ExpertTierPlan {
        let (plan, rank) = plain_plan();
        let (shared, tail_info) = plan.with_decode_share(&rank, 2 * PADDED).unwrap();
        assert_eq!(tail_info.donors, vec![600, 24_575]);
        shared
    }

    fn cpu_share(plan: &ExpertTierPlan) -> DecodeShareCpu {
        let resident = build_tables(plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        build_decode_share(
            plan,
            &resident,
            Some(DEV_BASE),
            Some(HOST_CPU),
            Some(HOST_ALIAS),
        )
        .unwrap()
        .expect("sharing plan")
    }

    #[test]
    fn decode_share_borrow_tables_map_donors_to_the_mirror_and_nothing_else() {
        let plan = shared_plan();
        let tail_info = plan.decode_share_tail().unwrap().clone();
        let resident = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let share = cpu_share(&plan);
        assert_eq!(share.views[0].tables.codecs, resident.codecs);
        for id in 0..TOTAL {
            let is_donor = tail_info.donors.contains(&(id as u32));
            if is_donor {
                let b = &tail_info.borrow_entries[id];
                assert_eq!(b.tier, Placement::HostMapped);
                assert_eq!(
                    share.views[0].tables.gate_up[id] as u64,
                    HOST_ALIAS + b.gate_up_offset
                );
                assert_eq!(
                    share.views[0].tables.down[id] as u64,
                    HOST_ALIAS + b.down_offset
                );
                // Mirror offsets equal the device offsets shifted by the window.
                let r = plan.entry(id / 512, id % 512).unwrap();
                assert_eq!(
                    b.gate_up_offset - tail_info.mirror_offset,
                    r.gate_up_offset - tail_info.donor_offset
                );
                assert_eq!(
                    b.down_offset - tail_info.mirror_offset,
                    r.down_offset - tail_info.donor_offset
                );
                assert_ne!(share.views[0].tables.gate_up[id], resident.gate_up[id]);
            } else {
                assert_eq!(
                    share.views[0].tables.gate_up[id], resident.gate_up[id],
                    "{id}"
                );
                assert_eq!(share.views[0].tables.down[id], resident.down[id], "{id}");
            }
        }
        // Layers with a host expert keep their `any_host` bit in both views.
        for layer in 0..48usize {
            let any = |entries: &dyn Fn(usize) -> Placement| {
                (0..512).any(|e| entries(layer * 512 + e) == Placement::HostMapped)
            };
            let resident_any = any(&|i| plan.entry(i / 512, i % 512).unwrap().tier);
            let borrow_any = any(&|i| tail_info.borrow_entries[i].tier);
            assert_eq!(resident_any, borrow_any, "layer {layer}");
        }
        // Mirror alias / donor range arithmetic.
        let ranges = share.ranges;
        assert_eq!(ranges.donors, 2);
        assert_eq!(
            ranges.donor_device,
            (
                (DEV_BASE + tail_info.donor_offset) as usize,
                tail_info.donor_bytes as usize
            )
        );
        assert_eq!(
            ranges.mirror_cpu,
            (
                HOST_CPU + tail_info.mirror_offset as usize,
                tail_info.donor_bytes as usize
            )
        );
        assert_eq!(
            ranges.mirror_alias,
            (HOST_ALIAS + tail_info.mirror_offset) as usize
        );
        assert_eq!(share.donor_offset, tail_info.donor_offset);
    }

    #[test]
    fn decode_share_borrow_digest_differs_from_resident_and_is_stable() {
        let plan = shared_plan();
        let a = cpu_share(&plan).views[0].digest;
        let b = cpu_share(&plan).views[0].digest;
        assert_eq!(a, b);
        assert_ne!(a, plan.logical_digest());
        // A different donor set changes the digest.
        let (base, rank) = plain_plan();
        let (other, _) = base.with_decode_share(&rank, PADDED).unwrap();
        assert_ne!(cpu_share(&other).views[0].digest, a);
        // The digest covers the entries, so it is a function of them.
        let tail_info = plan.decode_share_tail().unwrap();
        assert_eq!(
            a,
            borrow_view_digest(plan.logical_digest(), 0, 2, &tail_info.borrow_entries)
        );
        assert_ne!(
            a,
            borrow_view_digest(plan.logical_digest(), 0, 2, plan.entries()),
            "the resident entries must not reproduce the borrow digest"
        );
    }

    #[test]
    fn decode_share_non_sharing_plan_has_no_borrow_view() {
        let (plan, _) = plain_plan();
        let resident = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert!(build_decode_share(
            &plan,
            &resident,
            Some(DEV_BASE),
            Some(HOST_CPU),
            Some(HOST_ALIAS)
        )
        .unwrap()
        .is_none());
        // Resident build over the plan's own entries is the same call.
        let again =
            build_tables_for(&plan, plan.entries(), Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert_eq!(again.gate_up, resident.gate_up);
        assert_eq!(again.down, resident.down);
        assert_eq!(again.codecs, resident.codecs);
        // A need-0 share is an empty tail: still no borrow view.
        let (plan, rank) = plain_plan();
        let (zero, _) = plan.with_decode_share(&rank, 0).unwrap();
        let resident = build_tables(&zero, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        assert!(build_decode_share(
            &zero,
            &resident,
            Some(DEV_BASE),
            Some(HOST_CPU),
            Some(HOST_ALIAS)
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn decode_share_build_requires_bases_and_rejects_short_entry_lists() {
        let plan = shared_plan();
        let resident = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        for (device, cpu, alias, tier) in [
            (None, Some(HOST_CPU), Some(HOST_ALIAS), Placement::Device),
            (
                Some(DEV_BASE),
                None,
                Some(HOST_ALIAS),
                Placement::HostMapped,
            ),
            (Some(DEV_BASE), Some(HOST_CPU), None, Placement::HostMapped),
        ] {
            match build_decode_share(&plan, &resident, device, cpu, alias) {
                Err(ExpertTierOwnerError::MissingBase(t)) => assert_eq!(t, tier),
                other => panic!("expected MissingBase({tier:?}), got {:?}", other.err()),
            }
        }
        assert!(matches!(
            build_tables_for(
                &plan,
                &plan.entries()[1..],
                Some(DEV_BASE),
                Some(HOST_ALIAS)
            ),
            Err(ExpertTierOwnerError::Overflow)
        ));
    }

    #[test]
    fn decode_share_verify_tail_bytes_compares_hashes_and_names_the_first_difference() {
        let n = (COPY_CHUNK_BYTES as usize) * 2 + 4097;
        let mirror: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        let device = mirror.clone();
        let read = |data: &Vec<u8>| {
            let data = data.clone();
            move |chunk: &mut [u8], offset: usize| {
                chunk.copy_from_slice(&data[offset..offset + chunk.len()]);
                Ok(())
            }
        };
        let digest = verify_tail_bytes(&mirror, read(&device)).unwrap();
        let want: [u8; 32] = Sha256::digest(&mirror).into();
        assert_eq!(digest, want);
        // Prefix verification hashes only the prefix.
        let prefix = &mirror[..1000];
        let want: [u8; 32] = Sha256::digest(prefix).into();
        assert_eq!(verify_tail_bytes(prefix, read(&device)).unwrap(), want);
        assert_eq!(
            verify_tail_bytes(&[], read(&device)).unwrap(),
            <[u8; 32]>::from(Sha256::digest([]))
        );
        // A corrupted byte in the second chunk is reported by its tail offset.
        let bad_at = COPY_CHUNK_BYTES as usize + 123;
        let mut bad = device.clone();
        bad[bad_at] ^= 0xff;
        let error = verify_tail_bytes(&mirror, read(&bad)).unwrap_err();
        assert!(error.contains(&format!("tail byte {bad_at}:")), "{error}");
        // A read failure propagates.
        let error =
            verify_tail_bytes(&mirror, |_: &mut [u8], _| Err("boom".to_string())).unwrap_err();
        assert_eq!(error, "boom");
    }

    /// A frozen binding over fake (never dereferenced) table addresses: the
    /// view logic is GPU-free, so no lease or HIP runtime is needed.
    fn fake_binding(plan: ExpertTierPlan) -> ExpertTierBinding {
        let plan = Arc::new(plan);
        let tables = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let cache_mode = plan.expert_cache_tail().is_some();
        let tail_binding = build_donor_tail(
            &plan,
            plan.decode_share_tail().or(plan.expert_cache_tail()),
            &tables,
            Some(DEV_BASE),
            Some(HOST_CPU),
            Some(HOST_ALIAS),
        )
        .unwrap()
        .map(|cpu| {
            let views = cpu
                .views
                .into_iter()
                .enumerate()
                .map(|(class, view)| {
                    // Class 0 keeps the single-view addresses; each class owns its own pair.
                    let bases = (
                        0x1000_0000usize + class * 0x0010_0000,
                        0x2000_0000usize + class * 0x0010_0000,
                    );
                    BorrowView {
                        entries: view.entries,
                        tables: view.tables,
                        gate_up_table_base: bases.0,
                        down_table_base: bases.1,
                        donors: view.donors,
                        digest: view.digest,
                    }
                })
                .collect();
            DecodeShareBinding {
                views,
                ranges: cpu.ranges,
                donor_offset: cpu.donor_offset,
                class_spans: cpu.class_spans,
            }
        });
        let (decode_share, expert_cache) = if cache_mode {
            (None, tail_binding)
        } else {
            (tail_binding, None)
        };
        ExpertTierBinding {
            logical_digest: plan.logical_digest(),
            plan,
            tables,
            decode_share,
            expert_cache,
            gate_up_table_base: 0x3000_0000,
            down_table_base: 0x4000_0000,
            codec_table_base: 0x5000_0000,
            owner_ranges: [None; 2],
            host_cpu_range: None,
            generation: 1,
            hotset_sidecar_sha256: std::sync::OnceLock::new(),
            decode_share_claimed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[test]
    fn decode_share_claim_is_exclusive_and_released() {
        let binding = fake_binding(shared_plan());
        assert!(binding.try_claim_decode_share());
        assert!(!binding.try_claim_decode_share(), "a second claim refuses");
        binding.release_decode_share_claim();
        assert!(
            binding.try_claim_decode_share(),
            "released claims can be retaken"
        );
    }

    #[test]
    fn decode_share_layer_views_select_their_own_tables() {
        let plan = shared_plan();
        let tail_info = plan.decode_share_tail().unwrap().clone();
        let binding = fake_binding(plan.clone());
        assert!(binding.decode_share.is_some());
        assert_eq!(
            binding.view_digest(ExpertReadView::Resident),
            Some(plan.logical_digest())
        );
        let borrow_digest = binding
            .view_digest(ExpertReadView::BorrowPrefill(0))
            .unwrap();
        assert_ne!(borrow_digest, plan.logical_digest());
        assert!(binding.layer_view(48, ExpertReadView::Resident).is_none());
        assert!(binding
            .layer_view(48, ExpertReadView::BorrowPrefill(0))
            .is_none());
        let donor = 24_575usize; // layer 47, expert 511
        let (layer, expert) = (donor / 512, donor % 512);
        let resident = binding.layer_view(layer, ExpertReadView::Resident).unwrap();
        let borrow = binding
            .layer_view(layer, ExpertReadView::BorrowPrefill(0))
            .unwrap();
        assert_eq!(resident.view, ExpertReadView::Resident);
        assert_eq!(borrow.view, ExpertReadView::BorrowPrefill(0));
        assert_eq!(
            resident.entry(expert).map(|entry| entry.tier),
            Some(Placement::Device)
        );
        assert_eq!(
            borrow.entry(expert).map(|entry| entry.tier),
            Some(Placement::HostMapped)
        );
        assert_eq!(borrow.entry(expert), Some(&tail_info.borrow_entries[donor]));
        assert_eq!(resident.entry(expert), plan.entry(layer, expert));
        assert!(borrow.entry(512).is_none());
        assert_eq!(resident.any_host(), borrow.any_host());
        assert_eq!(resident.codec_table(), borrow.codec_table());
        assert_ne!(
            resident.gate_up_ptrs()[expert],
            borrow.gate_up_ptrs()[expert]
        );
        assert_eq!(
            borrow.gate_up_ptrs()[expert] as u64,
            HOST_ALIAS + tail_info.borrow_entries[donor].gate_up_offset
        );
        assert_eq!(
            borrow.down_ptrs()[expert] as u64,
            HOST_ALIAS + tail_info.borrow_entries[donor].down_offset
        );
        // A non-donor expert reads the same bytes in both views.
        assert_eq!(resident.gate_up_ptrs()[0], borrow.gate_up_ptrs()[0]);
        assert_eq!(resident.down_ptrs()[0], borrow.down_ptrs()[0]);
        assert_eq!(resident.entry(0), borrow.entry(0));
        // Each view owns its own device pointer tables; codecs are shared.
        assert_eq!(resident.gate_up_table_base(), 0x3000_0000);
        assert_eq!(borrow.gate_up_table_base(), 0x1000_0000);
        assert_eq!(resident.down_table_base(), 0x4000_0000);
        assert_eq!(borrow.down_table_base(), 0x2000_0000);
        // The `any_host` bit of every layer is identical between the views.
        for layer in 0..48 {
            let r = binding.layer_view(layer, ExpertReadView::Resident).unwrap();
            let b = binding
                .layer_view(layer, ExpertReadView::BorrowPrefill(0))
                .unwrap();
            assert_eq!(r.any_host(), b.any_host(), "layer {layer}");
        }
    }

    /// The n_c table: donors whose device extent starts below the class span,
    /// nested, ending at every donor, a partial overlap counting its donor.
    #[test]
    fn decode_share_class_donor_counts_are_nested_and_count_a_partial_overlap() {
        let (plan, rank) = plain_plan();
        let (shared, tail_info) = plan.with_decode_share(&rank, 2 * PADDED).unwrap();
        // Without classes: one all-donors view.
        assert_eq!(tail_info.class_spans, vec![2 * PADDED]);
        assert_eq!(tail_info.class_donors, vec![2]);
        // Donor k starts at k * PADDED in the tail.
        for (span, want) in [
            (0, 0),
            (1, 1),
            (PADDED, 1),
            (PADDED + 1, 2),
            (2 * PADDED, 2),
        ] {
            assert_eq!(tail_info.donors_dirtied_by(span), want, "span {span}");
        }
        let classed = shared
            .clone()
            .with_decode_share_classes(&[PADDED / 2, PADDED, PADDED + 1, 2 * PADDED])
            .unwrap();
        let tail_info = classed.decode_share_tail().unwrap();
        assert_eq!(tail_info.class_donors, vec![1, 1, 2, 2]);
        assert_eq!(
            tail_info.class_spans,
            vec![PADDED / 2, PADDED, PADDED + 1, 2 * PADDED]
        );
        // The class set never changes the plan itself.
        assert_eq!(classed.logical_digest(), shared.logical_digest());
        for bad in [&[][..], &[PADDED, 1][..], &[3 * PADDED][..], &[PADDED][..]] {
            assert!(
                matches!(
                    shared.clone().with_decode_share_classes(bad),
                    Err(crate::expert_tier_plan::DecodeShareError::BadClasses(_))
                ),
                "{bad:?}"
            );
        }
        // A plan without donors ignores classes.
        let (plain, _) = plain_plan();
        assert!(plain
            .with_decode_share_classes(&[PADDED])
            .unwrap()
            .decode_share_tail()
            .is_none());
    }

    /// Class views differ from the Resident view exactly at donors `[0, n_c)`;
    /// the largest is the all-donors borrow view; digests are distinct; every
    /// layer's host bit is identical in all of them.
    #[test]
    fn decode_share_class_views_differ_from_resident_exactly_at_their_donors() {
        let (plan, rank) = plain_plan();
        let (shared, _) = plan.with_decode_share(&rank, 2 * PADDED).unwrap();
        let classed = shared
            .with_decode_share_classes(&[PADDED, 2 * PADDED])
            .unwrap();
        let tail_info = classed.decode_share_tail().unwrap().clone();
        assert_eq!(tail_info.donors, vec![600, 24_575]);
        assert_eq!(tail_info.class_donors, vec![1, 2]);
        let resident = build_tables(&classed, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let share = cpu_share(&classed);
        assert_eq!(share.views.len(), 2);
        assert_eq!(share.class_spans, vec![PADDED, 2 * PADDED]);
        for (class, view) in share.views.iter().enumerate() {
            assert_eq!(view.donors, tail_info.class_donors[class]);
            for id in 0..TOTAL {
                let dirty = tail_info.donors[..view.donors].contains(&(id as u32));
                let want = if dirty {
                    &tail_info.borrow_entries[id]
                } else {
                    &classed.entries()[id]
                };
                assert_eq!(&view.entries[id], want, "class {class} expert {id}");
                assert_eq!(
                    view.tables.gate_up[id] != resident.gate_up[id],
                    dirty,
                    "class {class} expert {id}"
                );
                assert_eq!(view.tables.down[id] != resident.down[id], dirty);
            }
            assert_eq!(view.tables.codecs, resident.codecs);
        }
        // The largest class is today's all-donors view.
        assert_eq!(share.views[1].entries, tail_info.borrow_entries);
        // Digests differ per class and from the resident digest.
        let (d0, d1) = (share.views[0].digest, share.views[1].digest);
        assert_ne!(d0, d1);
        assert_ne!(d0, classed.logical_digest());
        assert_ne!(d1, classed.logical_digest());
        assert_eq!(
            d1,
            borrow_view_digest(classed.logical_digest(), 1, 2, &tail_info.borrow_entries)
        );

        // Through the binding: layer views select their class and share the host bit.
        let binding = fake_binding(classed.clone());
        assert_eq!(
            binding.view_digest(ExpertReadView::BorrowPrefill(0)),
            Some(d0)
        );
        assert_eq!(
            binding.view_digest(ExpertReadView::BorrowPrefill(1)),
            Some(d1)
        );
        assert!(binding
            .view_digest(ExpertReadView::BorrowPrefill(2))
            .is_none());
        assert!(binding
            .layer_view(0, ExpertReadView::BorrowPrefill(2))
            .is_none());
        // Donor 600 (layer 1, expert 88) is dirty in both; donor 24,575 only in class 1.
        let c0 = binding
            .layer_view(47, ExpertReadView::BorrowPrefill(0))
            .unwrap();
        let c1 = binding
            .layer_view(47, ExpertReadView::BorrowPrefill(1))
            .unwrap();
        assert_eq!(c0.entry(511).map(|e| e.tier), Some(Placement::Device));
        assert_eq!(c1.entry(511).map(|e| e.tier), Some(Placement::HostMapped));
        assert_ne!(c0.gate_up_table_base(), c1.gate_up_table_base());
        assert_ne!(c0.down_table_base(), c1.down_table_base());
        for layer in 0..48 {
            let r = binding.layer_view(layer, ExpertReadView::Resident).unwrap();
            for class in 0..2 {
                let b = binding
                    .layer_view(layer, ExpertReadView::BorrowPrefill(class))
                    .unwrap();
                assert_eq!(r.any_host(), b.any_host(), "layer {layer} class {class}");
                assert_eq!(r.codec_table(), b.codec_table());
            }
        }
    }

    #[test]
    fn decode_share_borrow_view_is_none_on_a_non_sharing_binding() {
        let (plan, _) = plain_plan();
        let binding = fake_binding(plan.clone());
        assert!(binding.decode_share.is_none());
        assert!(binding
            .view_digest(ExpertReadView::BorrowPrefill(0))
            .is_none());
        assert!(binding
            .layer_view(0, ExpertReadView::BorrowPrefill(0))
            .is_none());
        assert_eq!(
            binding.view_digest(ExpertReadView::Resident),
            Some(plan.logical_digest())
        );
        // The Resident view is exactly the plan.
        let resident = binding.layer_view(3, ExpertReadView::Resident).unwrap();
        assert_eq!(resident.view, ExpertReadView::Resident);
        assert_eq!(
            resident.gate_up_ptrs(),
            &binding.tables.gate_up[3 * 512..4 * 512]
        );
        assert_eq!(resident.down_ptrs(), &binding.tables.down[3 * 512..4 * 512]);
        for expert in 0..512 {
            assert_eq!(resident.entry(expert), plan.entry(3, expert));
        }
        assert!(binding.layer_view(48, ExpertReadView::Resident).is_none());
    }
    #[test]
    fn expert_cache_canonical_view_has_complete_fallback_without_scratch_share() {
        let (plain, rank) = plain_plan();
        let (plan, tail) = plain.with_expert_cache_tail(&rank, 2 * PADDED).unwrap();
        let binding = fake_binding(plan.clone());
        assert!(binding.decode_share.is_none());
        assert!(binding.expert_cache.is_some());
        assert_eq!(binding.expert_cache.as_ref().unwrap().views.len(), 1);
        assert!(binding
            .layer_view(0, ExpertReadView::BorrowPrefill(1))
            .is_none());
        for layer in 0..48 {
            let warm = binding.layer_view(layer, ExpertReadView::Resident).unwrap();
            let canonical = binding
                .layer_view(layer, ExpertReadView::BorrowPrefill(0))
                .unwrap();
            for expert in 0..512 {
                let id = layer * 512 + expert;
                assert_eq!(warm.entry(expert), plan.entry(layer, expert));
                assert_eq!(canonical.entry(expert), Some(&tail.borrow_entries[id]));
                let entry = canonical.entry(expert).unwrap();
                let base = if entry.tier == Placement::Device {
                    DEV_BASE
                } else {
                    HOST_ALIAS
                };
                assert_eq!(
                    canonical.gate_up_ptrs()[expert] as u64,
                    base + entry.gate_up_offset
                );
                assert_eq!(
                    canonical.down_ptrs()[expert] as u64,
                    base + entry.down_offset
                );
            }
        }
        // A short prefill may overlap slot DMA only because its immutable
        // tables cannot address any byte of the writable donor pool.
        let donor_start = (DEV_BASE + tail.donor_offset) as usize;
        let donor_end = donor_start + tail.donor_bytes as usize;
        let overlaps = |ptr: usize, len: u64| ptr < donor_end && donor_start < ptr + len as usize;
        let mut resident_donor_reads = 0;
        for layer in 0..48 {
            let warm = binding.layer_view(layer, ExpertReadView::Resident).unwrap();
            let canonical = binding
                .layer_view(layer, ExpertReadView::BorrowPrefill(0))
                .unwrap();
            assert_ne!(canonical.gate_up_table_base(), warm.gate_up_table_base());
            assert_ne!(canonical.down_table_base(), warm.down_table_base());
            for expert in 0..EXPERTS {
                let entry = canonical.entry(expert).unwrap();
                assert!(!overlaps(
                    canonical.gate_up_ptrs()[expert],
                    entry.gate_up_len
                ));
                assert!(!overlaps(canonical.down_ptrs()[expert], entry.down_len));
                let resident = warm.entry(expert).unwrap();
                resident_donor_reads +=
                    usize::from(overlaps(warm.gate_up_ptrs()[expert], resident.gate_up_len));
                resident_donor_reads +=
                    usize::from(overlaps(warm.down_ptrs()[expert], resident.down_len));
            }
        }
        // Negative control: using Resident without restoration reads every
        // original donor from mutable VRAM, so that view is unsafe here.
        assert_eq!(resident_donor_reads, 2 * tail.donors.len());
        assert_ne!(
            binding.view_digest(ExpertReadView::Resident),
            binding.view_digest(ExpertReadView::BorrowPrefill(0))
        );
    }

    #[test]
    fn expert_cache_slot_ranges_are_disjoint_and_restore_exact_donor_bytes() {
        let (plain, rank) = plain_plan();
        let host_before = plain.owner_bytes(Placement::HostMapped);
        let (plan, tail) = plain.with_expert_cache_tail(&rank, 2 * PADDED).unwrap();
        let resident = build_tables(&plan, Some(DEV_BASE), Some(HOST_ALIAS)).unwrap();
        let cpu = build_donor_tail(
            &plan,
            plan.expert_cache_tail(),
            &resident,
            Some(DEV_BASE),
            Some(HOST_CPU),
            Some(HOST_ALIAS),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            plan.owner_bytes(Placement::HostMapped),
            host_before + tail.donor_bytes
        );
        assert_eq!(
            cpu.ranges.donor_device,
            (
                (DEV_BASE + tail.donor_offset) as usize,
                tail.donor_bytes as usize
            )
        );
        assert_eq!(
            cpu.ranges.mirror_cpu,
            (
                HOST_CPU + tail.mirror_offset as usize,
                tail.donor_bytes as usize
            )
        );
        let mut spans = Vec::new();
        let mut ids = Vec::new();
        for layer in 0..48 {
            for slot in cache_slot_specs(&plan, layer, DEV_BASE, HOST_CPU, HOST_ALIAS).unwrap() {
                ids.push((layer * EXPERTS + slot.original_expert) as u32);
                for (device, alias, cpu_range) in [
                    (slot.gate_up_device, slot.gate_up_host, slot.gate_up_cpu),
                    (slot.down_device, slot.down_host, slot.down_cpu),
                ] {
                    let relative = device.0 - cpu.ranges.donor_device.0;
                    assert_eq!(alias.0 - cpu.ranges.mirror_alias, relative);
                    assert_eq!(cpu_range.0 - cpu.ranges.mirror_cpu.0, relative);
                    assert_eq!(device.1, alias.1);
                    assert_eq!(device.1, cpu_range.1);
                    assert!(relative + device.1 <= tail.donor_bytes as usize);
                    spans.push((device.0, device.0 + device.1));
                }
            }
        }
        assert_eq!(ids, tail.donors);
        spans.sort_unstable();
        assert!(spans.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        assert!(cache_slot_specs(&plan, 48, DEV_BASE, HOST_CPU, HOST_ALIAS).is_err());
        // Bulk restore includes alignment bytes but excludes guard and owner pad.
        assert_eq!(
            cpu.ranges.donor_device.0 as u64,
            DEV_BASE + tail.donor_offset
        );
        assert_eq!(
            cpu.ranges.donor_device.0 as u64 + tail.donor_bytes,
            DEV_BASE + plan.owner_bytes(Placement::Device)
                - (((ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255)
        );
    }
}
