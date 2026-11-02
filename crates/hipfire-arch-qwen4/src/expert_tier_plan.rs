// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure CPU per-expert placement plan (B5-C1): which experts live in the
//! device-resident owner and which in the host-mapped owner.
//!
//! One mechanism serves pure-MQ4 HotCache, Consumer24 B5 (MQ4 + MQ2-I4) and
//! sub-4 K4 (MQ4 + MQ3-I4). Placement never changes format, numerics, slot
//! order or ids. The **model id** of an expert is `layer * 512 + expert`.
//!
//! * At most two model-level payload owners exist: one [`Placement::Device`]
//!   and one [`Placement::HostMapped`]. An empty tier allocates nothing
//!   (`owner_bytes == 0`).
//! * Every projection starts 256-byte aligned inside its owner
//!   ([`ExpertTierPlan::PAYLOAD_ALIGN`]); every non-empty owner carries
//!   [`ARENA_TAIL_PAD_BYTES`] (at least 256 B) of zeroed tail pad.
//! * Offsets and lengths are checked `u64`; model-level owners exceed 4 GiB.
//! * Destination layout inside a tier is `(layer, expert)` ascending, `gate_up`
//!   then `down`, independent of the rank order that chose the set.
//!
//! No GPU, environment or file access. Route/arch admission of the tier route
//! as a whole belongs to C2 admission, not to [`ExpertTierPlan::check_executable`],
//! which only refuses expert *formats* that have no kernel on an arch.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan, ExpertSlabTable};
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};

/// Codec id of an MQ2-I4 (qt133/qt134) expert. Reserved; refused by
/// [`ExpertTierPlan::check_executable`] until W2 kernels and route admission land.
pub const CODEC_W2: u8 = 2;

const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
const TOTAL_EXPERTS: usize = TRUNK_LAYERS as usize * EXPERTS;
const DIGEST_DOMAIN: &[u8] = b"hipfire.expert_tier_plan";

/// Which model-level owner holds an expert's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Device-resident owner.
    Device,
    /// Pinned, device-mapped host owner.
    HostMapped,
}

/// One expert's format and its destination inside its tier's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertTierEntry {
    /// From the artifact; never altered by placement.
    pub format: ExpertFormat,
    pub tier: Placement,
    /// Relative to the base of THIS entry's tier owner.
    pub gate_up_offset: u64,
    pub gate_up_len: u64,
    /// Relative to the base of THIS entry's tier owner.
    pub down_offset: u64,
    pub down_len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertTierError {
    /// A rank id is `>= 24,576`.
    RankIdOutOfRange { id: u32 },
    /// A rank id appears twice.
    RankDuplicate { id: u32 },
    /// A source span is empty, overflows, or the expert's two spans overlap.
    SourceSpan { layer: usize, expert: usize },
    /// Checked `u64` arithmetic overflowed.
    Overflow,
    /// The expert format has no kernel on `arch`.
    UnsupportedFormat { format: ExpertFormat, arch: String },
    /// The slab list is not exactly `48 * 512 = 24,576` entries.
    SlabCount { got: usize },
    /// An aggregate (legacy) artifact projection is malformed
    /// ([`crate::expert_slab_aggregate`]).
    Aggregate { layer: usize, reason: &'static str },
}

impl fmt::Display for ExpertTierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankIdOutOfRange { id } => {
                write!(
                    f,
                    "rank id {id} is out of range (model ids are < {TOTAL_EXPERTS})"
                )
            }
            Self::RankDuplicate { id } => write!(f, "rank id {id} appears more than once"),
            Self::SourceSpan { layer, expert } => write!(
                f,
                "layer {layer} expert {expert} has an empty, overflowing or overlapping source span"
            ),
            Self::Overflow => write!(f, "expert tier plan arithmetic overflowed u64"),
            Self::UnsupportedFormat { format, arch } => {
                write!(
                    f,
                    "expert format {format:?} has no kernels on arch {arch:?}"
                )
            }
            Self::SlabCount { got } => write!(
                f,
                "expected exactly {TOTAL_EXPERTS} expert slabs, got {got}"
            ),
            Self::Aggregate { layer, reason } => {
                write!(f, "aggregate expert tensors of layer {layer}: {reason}")
            }
        }
    }
}

impl std::error::Error for ExpertTierError {}

/// Which projection of an expert a [`TierCopy`] moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TierProj {
    GateUp,
    Down,
}

/// One bounded source-to-owner copy of the materialization pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierCopy {
    pub layer: usize,
    pub expert: usize,
    pub proj: TierProj,
    /// Source span inside the artifact.
    pub src: ExpertSlabSpan,
    /// Destination offset inside the owner of the copy's tier.
    pub dst_offset: u64,
}

/// Extra payload spans (loaded MTP expert payloads) placed in the HOST owner
/// after the trunk host experts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostWindow {
    pub label: String,
    /// `(artifact span, destination offset inside the host owner)`.
    pub spans: Vec<(ExpertSlabSpan, u64)>,
}

/// Label of the decode-share donor mirror host window.
pub const DECODE_SHARE_MIRROR_LABEL: &str = "decode-share-donor-mirror";

/// The donor tail of a decode-share plan ([`ExpertTierPlan::with_decode_share`]).
///
/// Device layout: non-donor device experts from offset 0, then an always-zero
/// guard of one tail pad, then the donors back to back (`donor_offset ..
/// donor_offset + donor_bytes`), then the usual owner tail pad. The donors'
/// source bytes are mirrored in the host owner so that mirror byte `i` equals
/// device donor byte `i`: one contiguous H2D copy restores any prefix of the
/// tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeShareTail {
    /// Donor model ids (layer*512+expert), (layer, expert) ascending == physical order.
    pub donors: Vec<u32>,
    /// Device-owner offset of the always-zero guard placed between the last non-donor and the first donor.
    pub guard_offset: u64,
    /// Device-owner offset of the first donor byte (256-aligned). Donor bytes are contiguous to donor_offset + donor_bytes.
    pub donor_offset: u64,
    /// Contiguous donor extent B (sum of the donors' 256-padded projections; no gaps).
    pub donor_bytes: u64,
    /// Host-owner offset of the mirror window start; mirror byte i == device donor byte i for i < donor_bytes.
    pub mirror_offset: u64,
    /// The BorrowPrefill view: identical to the plan's entries except donors, which are
    /// `Placement::HostMapped` with offsets into the host owner (inside the mirror window).
    pub borrow_entries: Vec<ExpertTierEntry>,
    /// Dirty byte prefix of each prefill capacity class, ascending: class `c`
    /// overwrites `[donor_offset, donor_offset + class_spans[c])` while it is
    /// the largest class of a borrow epoch. One entry (the whole tail) until
    /// [`ExpertTierPlan::with_decode_share_classes`] sets the real spans; empty
    /// without donors.
    pub class_spans: Vec<u64>,
    /// `class_donors[c]`: donors (physical order) whose device extent starts
    /// below `donor_offset + class_spans[c]`, i.e. the donors class `c` dirties
    /// (a partially overlapped donor counts). Nested ascending; the last is
    /// `donors.len()`.
    pub class_donors: Vec<usize>,
    /// Bytes at the START of the donor tail (`donor_offset ..`) that hold the
    /// borrowed F16 / INT8 weight shadows, rebuilt at the first acquire of
    /// every borrow epoch ([`ExpertTierPlan::with_decode_share_shadows`]); the
    /// scratch backing starts right after them. 0 when the shadows stay
    /// resident. Every class span includes these bytes (the shadow region is
    /// dirtied by every class).
    pub shadow_bytes: u64,
}

impl DecodeShareTail {
    /// Donors (physical order) whose device extent starts below the first
    /// `span` bytes of the donor tail: the donors a scratch span of `span`
    /// bytes dirties. Donor `k` starts at the mirror-relative offset of its
    /// gate/up payload (mirror byte `i` == device donor byte `i`).
    pub fn donors_dirtied_by(&self, span: u64) -> usize {
        self.donors.partition_point(|&id| {
            self.borrow_entries[id as usize]
                .gate_up_offset
                .saturating_sub(self.mirror_offset)
                < span
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeShareError {
    /// The eligible resident experts cannot supply `need` bytes.
    Insufficient {
        need: u64,
        available: u64,
    },
    Overflow,
    /// The plan already carries a donor tail; a plan is re-laid out once.
    AlreadyShared,
    /// The class spans are empty, descending, or do not end at the last donor.
    BadClasses(String),
}

impl fmt::Display for DecodeShareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Insufficient { need, available } => write!(
                f,
                "decode-share needs {need} donor bytes but the eligible resident experts hold {available}"
            ),
            Self::Overflow => write!(f, "decode-share plan arithmetic overflowed u64"),
            Self::AlreadyShared => write!(f, "plan already carries a decode-share donor tail"),
            Self::BadClasses(why) => write!(f, "decode-share class spans are invalid: {why}"),
        }
    }
}

impl std::error::Error for DecodeShareError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertTierPlan {
    entries: Vec<ExpertTierEntry>,
    source: Vec<(ExpertSlabSpan, ExpertSlabSpan)>,
    device_bytes: u64,
    host_bytes: u64,
    device_experts: usize,
    host_windows: Vec<HostWindow>,
    decode_share: Option<DecodeShareTail>,
    artifact_sha256: Option<[u8; 32]>,
    digest: [u8; 32],
}

fn align_up(x: u64) -> Option<u64> {
    let a = ExpertTierPlan::PAYLOAD_ALIGN;
    Some(x.checked_add(a - 1)? & !(a - 1))
}

fn tail_pad() -> u64 {
    // `max(ARENA_TAIL_PAD_BYTES, 256)` rounded to 256; the constant is far from u64::MAX.
    align_up((ARENA_TAIL_PAD_BYTES as u64).max(ExpertTierPlan::PAYLOAD_ALIGN))
        .expect("tail pad fits u64")
}

fn tier_index(tier: Placement) -> usize {
    match tier {
        Placement::Device => 0,
        Placement::HostMapped => 1,
    }
}

fn digest_of(
    entries: &[ExpertTierEntry],
    source: &[(ExpertSlabSpan, ExpertSlabSpan)],
    device_bytes: u64,
    host_bytes: u64,
    windows: &[HostWindow],
    artifact_sha256: Option<[u8; 32]>,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(DIGEST_DOMAIN);
    h.update(ExpertTierPlan::VERSION.to_le_bytes());
    match artifact_sha256 {
        Some(sha) => {
            h.update([1u8]);
            h.update(sha);
        }
        None => {
            h.update([0u8]);
            h.update([0u8; 32]);
        }
    }
    for (entry, (gu, dn)) in entries.iter().zip(source) {
        let (gu_qt, dn_qt) = entry.format.pair();
        h.update([gu_qt, dn_qt]);
        h.update((gu.offset as u64).to_le_bytes());
        h.update((gu.len as u64).to_le_bytes());
        h.update((dn.offset as u64).to_le_bytes());
        h.update((dn.len as u64).to_le_bytes());
        h.update([tier_index(entry.tier) as u8]);
        h.update(entry.gate_up_offset.to_le_bytes());
        h.update(entry.gate_up_len.to_le_bytes());
        h.update(entry.down_offset.to_le_bytes());
        h.update(entry.down_len.to_le_bytes());
    }
    h.update(device_bytes.to_le_bytes());
    h.update(host_bytes.to_le_bytes());
    for window in windows {
        h.update((window.label.len() as u64).to_le_bytes());
        h.update(window.label.as_bytes());
        h.update((window.spans.len() as u64).to_le_bytes());
        for (span, dst) in &window.spans {
            h.update((span.offset as u64).to_le_bytes());
            h.update((span.len as u64).to_le_bytes());
            h.update(dst.to_le_bytes());
        }
    }
    h.finalize().into()
}

impl ExpertTierPlan {
    /// Version of the plan layout and digest stream.
    pub const VERSION: u32 = 1;
    /// Alignment of every projection start inside an owner.
    pub const PAYLOAD_ALIGN: u64 = 256;

    fn index(layer: usize, expert: usize) -> Option<usize> {
        (layer < TRUNK_LAYERS as usize && expert < EXPERTS).then_some(layer * EXPERTS + expert)
    }

    pub fn entry(&self, layer: usize, expert: usize) -> Option<&ExpertTierEntry> {
        self.entries.get(Self::index(layer, expert)?)
    }

    /// Every entry in model id order (`layer * 512 + expert`).
    pub fn entries(&self) -> &[ExpertTierEntry] {
        &self.entries
    }

    /// `(gate_up, down)` source spans in the artifact.
    pub fn source(&self, layer: usize, expert: usize) -> Option<(ExpertSlabSpan, ExpertSlabSpan)> {
        self.source.get(Self::index(layer, expert)?).copied()
    }

    /// Padded owner size including the tail pad; 0 means no allocation.
    pub fn owner_bytes(&self, tier: Placement) -> u64 {
        match tier {
            Placement::Device => self.device_bytes,
            Placement::HostMapped => self.host_bytes,
        }
    }

    pub fn device_experts(&self) -> usize {
        self.device_experts
    }

    /// SHA-256 of the canonical little-endian plan stream (receipts and
    /// diagnostics only, never authorization).
    pub fn logical_digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn host_windows(&self) -> &[HostWindow] {
        &self.host_windows
    }

    /// Refuse expert formats the tier route does not serve on `arch`: each entry's exact quant
    /// pair must be [`rdna_compute::expert_capabilities::Route::TierStage`]-admitted. Beta: W2
    /// has no kernel anywhere, E8P-I4 is held everywhere (`Hold::TierPlanE8pRefused`), W3 is
    /// admitted on gfx1201 only (gfx1151 holds it, other arches have no kernel), MQ4 is accepted
    /// on every arch (arch-neutral aggregate route). Route/arch admission of the tier route as a
    /// whole belongs to C2 admission, not here.
    pub fn check_executable(&self, arch: &str) -> Result<(), ExpertTierError> {
        use rdna_compute::expert_capabilities::{admitted, ExpertQt, Route};
        for entry in &self.entries {
            let (gate_up, down) = entry.format.pair();
            if !admitted(arch, ExpertQt::new(gate_up, down), Route::TierStage) {
                return Err(ExpertTierError::UnsupportedFormat {
                    format: entry.format,
                    arch: arch.to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Copies of `tier` in deterministic `(layer, expert, gate_up, down)` order.
    pub fn copies(&self, tier: Placement) -> impl Iterator<Item = TierCopy> + '_ {
        self.entries
            .iter()
            .zip(&self.source)
            .enumerate()
            .filter(move |(_, (entry, _))| entry.tier == tier)
            .flat_map(|(i, (entry, (gu, dn)))| {
                let layer = i / EXPERTS;
                let expert = i % EXPERTS;
                [
                    TierCopy {
                        layer,
                        expert,
                        proj: TierProj::GateUp,
                        src: *gu,
                        dst_offset: entry.gate_up_offset,
                    },
                    TierCopy {
                        layer,
                        expert,
                        proj: TierProj::Down,
                        src: *dn,
                        dst_offset: entry.down_offset,
                    },
                ]
            })
    }

    /// Append `spans` (e.g. loaded MTP expert payloads) to the host owner after
    /// the trunk host experts. The host owner grows; its tail pad is counted once.
    pub fn with_host_window(
        mut self,
        label: &str,
        spans: &[ExpertSlabSpan],
    ) -> Result<Self, ExpertTierError> {
        let tail = tail_pad();
        // End of the host payloads, before the tail pad.
        let mut cursor = self.host_bytes.saturating_sub(tail);
        let mut placed = Vec::with_capacity(spans.len());
        for span in spans {
            if span.len == 0 || span.end().is_none() {
                return Err(ExpertTierError::Overflow);
            }
            let dst = align_up(cursor).ok_or(ExpertTierError::Overflow)?;
            cursor = dst
                .checked_add(span.len as u64)
                .ok_or(ExpertTierError::Overflow)?;
            placed.push((*span, dst));
        }
        self.host_bytes = if cursor == 0 {
            0
        } else {
            align_up(cursor)
                .and_then(|c| c.checked_add(tail))
                .ok_or(ExpertTierError::Overflow)?
        };
        self.host_windows.push(HostWindow {
            label: label.to_owned(),
            spans: placed,
        });
        self.digest = self.recompute_digest();
        Ok(self)
    }

    /// Pin the artifact SHA-256 into the plan digest.
    pub fn with_artifact_sha256(mut self, sha: [u8; 32]) -> Self {
        self.artifact_sha256 = Some(sha);
        self.digest = self.recompute_digest();
        self
    }

    /// The donor tail of a plan built by [`Self::with_decode_share`].
    pub fn decode_share_tail(&self) -> Option<&DecodeShareTail> {
        self.decode_share.as_ref()
    }

    /// Re-lay out this (already admitted) plan with a donor tail of at least
    /// `need_bytes`.
    ///
    /// Donors are the coldest `Device` experts of `rank` (walked from its end)
    /// whose layer has at least one `HostMapped` expert in this plan, taken
    /// until their padded payloads sum to `>= need_bytes`. Membership never
    /// changes: the `Device` set and `device_experts` are identical before and
    /// after; only offsets move (see [`DecodeShareTail`] for the layout). The
    /// donors' source spans are appended as a host window labelled
    /// [`DECODE_SHARE_MIRROR_LABEL`], in physical order, so the mirror is
    /// byte-aligned with the device donor extent.
    ///
    /// `need_bytes == 0` returns the plan unchanged with an empty tail (no
    /// donors, no guard, no window). Ids of `rank` that are out of range or
    /// repeat are skipped (the plan builder already rejected them).
    pub fn with_decode_share(
        mut self,
        rank: &[u32],
        need_bytes: u64,
    ) -> Result<(Self, DecodeShareTail), DecodeShareError> {
        use DecodeShareError::{AlreadyShared, Insufficient, Overflow};
        if self.decode_share.is_some() {
            return Err(AlreadyShared);
        }
        if need_bytes == 0 {
            let tail = DecodeShareTail {
                donors: Vec::new(),
                guard_offset: 0,
                donor_offset: 0,
                donor_bytes: 0,
                mirror_offset: 0,
                borrow_entries: self.entries.clone(),
                class_spans: Vec::new(),
                class_donors: Vec::new(),
                shadow_bytes: 0,
            };
            return Ok((self, tail));
        }
        let padded = |e: &ExpertTierEntry| -> Result<u64, DecodeShareError> {
            align_up(e.gate_up_len)
                .zip(align_up(e.down_len))
                .and_then(|(g, d)| g.checked_add(d))
                .ok_or(Overflow)
        };

        // Eligibility: Device here, in a layer that has a host expert here.
        let mut layer_has_host = vec![false; TRUNK_LAYERS as usize];
        for (i, e) in self.entries.iter().enumerate() {
            if e.tier == Placement::HostMapped {
                layer_has_host[i / EXPERTS] = true;
            }
        }

        // Selection: coldest eligible first.
        let mut is_donor = vec![false; TOTAL_EXPERTS];
        let mut got = 0u64;
        for &id in rank.iter().rev() {
            let idx = id as usize;
            let Some(e) = self.entries.get(idx) else {
                continue;
            };
            if e.tier != Placement::Device || !layer_has_host[idx / EXPERTS] || is_donor[idx] {
                continue;
            }
            got = got.checked_add(padded(e)?).ok_or(Overflow)?;
            is_donor[idx] = true;
            if got >= need_bytes {
                break;
            }
        }
        if got < need_bytes {
            return Err(Insufficient {
                need: need_bytes,
                available: got,
            });
        }

        // Device layout: non-donors, guard, donors.
        let place = |e: &mut ExpertTierEntry, cursor: &mut u64| -> Result<(), DecodeShareError> {
            let gate_up_offset = align_up(*cursor).ok_or(Overflow)?;
            let after_gu = gate_up_offset.checked_add(e.gate_up_len).ok_or(Overflow)?;
            let down_offset = align_up(after_gu).ok_or(Overflow)?;
            *cursor = down_offset.checked_add(e.down_len).ok_or(Overflow)?;
            e.gate_up_offset = gate_up_offset;
            e.down_offset = down_offset;
            Ok(())
        };
        let mut cursor = 0u64;
        for (i, e) in self.entries.iter_mut().enumerate() {
            if e.tier == Placement::Device && !is_donor[i] {
                place(e, &mut cursor)?;
            }
        }
        let guard_offset = align_up(cursor).ok_or(Overflow)?;
        let donor_offset =
            align_up(guard_offset.checked_add(tail_pad()).ok_or(Overflow)?).ok_or(Overflow)?;
        cursor = donor_offset;
        let mut donors = Vec::new();
        for (i, e) in self.entries.iter_mut().enumerate() {
            if is_donor[i] {
                place(e, &mut cursor)?;
                donors.push(u32::try_from(i).map_err(|_| Overflow)?);
            }
        }
        let donor_end = align_up(cursor).ok_or(Overflow)?;
        let donor_bytes = donor_end - donor_offset;
        assert_eq!(
            donor_bytes, got,
            "donor extent must equal the sum of padded donor payloads"
        );
        self.device_bytes = donor_end.checked_add(tail_pad()).ok_or(Overflow)?;

        // Mirror window: the donors' source spans in physical order.
        let mut spans = Vec::with_capacity(donors.len() * 2);
        for &id in &donors {
            let (gu, dn) = self.source[id as usize];
            spans.push(gu);
            spans.push(dn);
        }
        let mut plan = self
            .with_host_window(DECODE_SHARE_MIRROR_LABEL, &spans)
            .map_err(|_| Overflow)?;
        let window = plan.host_windows.last().expect("window just appended");
        let mirror_offset = window.spans.first().ok_or(Overflow)?.1;

        // One contiguous copy must restore the whole tail: mirror byte i == device byte i.
        let mut borrow_entries = plan.entries.clone();
        for (k, &id) in donors.iter().enumerate() {
            let e = &plan.entries[id as usize];
            let (gu_dst, dn_dst) = (window.spans[2 * k].1, window.spans[2 * k + 1].1);
            for (host_dst, device_off) in [(gu_dst, e.gate_up_offset), (dn_dst, e.down_offset)] {
                let host_rel = host_dst.checked_sub(mirror_offset).ok_or(Overflow)?;
                let device_rel = device_off.checked_sub(donor_offset).ok_or(Overflow)?;
                if host_rel != device_rel {
                    return Err(Overflow);
                }
            }
            let b = &mut borrow_entries[id as usize];
            b.tier = Placement::HostMapped;
            b.gate_up_offset = gu_dst;
            b.down_offset = dn_dst;
        }
        let mirror_end = mirror_offset.checked_add(donor_bytes).ok_or(Overflow)?;
        if mirror_end > plan.host_bytes {
            return Err(Overflow);
        }

        let donors_len = donors.len();
        let tail = DecodeShareTail {
            donors,
            guard_offset,
            donor_offset,
            donor_bytes,
            mirror_offset,
            borrow_entries,
            class_spans: vec![donor_bytes],
            class_donors: vec![donors_len],
            shadow_bytes: 0,
        };
        plan.decode_share = Some(tail.clone());
        Ok((plan, tail))
    }

    /// Set the prefill capacity classes of this plan's donor tail: `spans[c]` is
    /// the byte prefix of the tail class `c` dirties (scratch span of the class,
    /// ascending). Computes `class_donors` (the donors each class dirties; a
    /// donor partially overlapped counts) and requires the last class to dirty
    /// every donor and no span to exceed the tail. A plan without donors is
    /// returned unchanged.
    pub fn with_decode_share_classes(mut self, spans: &[u64]) -> Result<Self, DecodeShareError> {
        let Some(tail) = self.decode_share.as_mut() else {
            return Ok(self);
        };
        if tail.donors.is_empty() {
            return Ok(self);
        }
        let bad = |why: &str| Err(DecodeShareError::BadClasses(why.to_string()));
        if spans.is_empty() {
            return bad("no classes");
        }
        if spans.windows(2).any(|pair| pair[0] > pair[1]) {
            return bad("spans descend");
        }
        if spans.last().is_some_and(|&last| last > tail.donor_bytes) {
            return bad("a span exceeds the donor tail");
        }
        let class_donors: Vec<usize> = spans
            .iter()
            .map(|&span| tail.donors_dirtied_by(span))
            .collect();
        if class_donors.last() != Some(&tail.donors.len()) {
            return bad("the largest class does not dirty every donor");
        }
        tail.class_spans = spans.to_vec();
        tail.class_donors = class_donors;
        Ok(self)
    }

    /// Declare the first `shadow_bytes` of this plan's donor tail to be the
    /// borrowed shadow region (every class span already includes it, so
    /// `class_spans[0] >= shadow_bytes`; the region is 256-aligned). A plan
    /// without donors is returned unchanged (and `shadow_bytes` must be 0).
    pub fn with_decode_share_shadows(
        mut self,
        shadow_bytes: u64,
    ) -> Result<Self, DecodeShareError> {
        let Some(tail) = self.decode_share.as_mut() else {
            return if shadow_bytes == 0 {
                Ok(self)
            } else {
                Err(DecodeShareError::BadClasses(
                    "shadow bytes without a donor tail".to_string(),
                ))
            };
        };
        if shadow_bytes % 256 != 0 {
            return Err(DecodeShareError::BadClasses(
                "the shadow region is not 256-aligned".to_string(),
            ));
        }
        if tail
            .class_spans
            .first()
            .is_some_and(|&first| first < shadow_bytes)
        {
            return Err(DecodeShareError::BadClasses(
                "a class span is smaller than the shadow region".to_string(),
            ));
        }
        if shadow_bytes > tail.donor_bytes {
            return Err(DecodeShareError::BadClasses(
                "the shadow region exceeds the donor tail".to_string(),
            ));
        }
        tail.shadow_bytes = shadow_bytes;
        Ok(self)
    }

    fn recompute_digest(&self) -> [u8; 32] {
        digest_of(
            &self.entries,
            &self.source,
            self.device_bytes,
            self.host_bytes,
            &self.host_windows,
            self.artifact_sha256,
        )
    }
}

/// Plan placement from the K4-validated span table.
pub fn plan_expert_tiers(
    slabs: &ExpertSlabTable,
    rank: &[u32],
    vram_budget: u64,
) -> Result<ExpertTierPlan, ExpertTierError> {
    plan_expert_tiers_from_slabs(slabs.slabs(), rank, vram_budget)
}

/// `rank`: unique model ids in priority order (it may be a subset); omitted ids
/// are host-mapped. The device set is the LONGEST rank PREFIX whose padded
/// payloads plus tail pad fit `vram_budget`; the walk stops at the first id
/// that does not fit (no skipping). `slabs` must hold exactly 24,576 entries in
/// `layer * 512 + expert` order.
pub fn plan_expert_tiers_from_slabs(
    slabs: &[ExpertSlab],
    rank: &[u32],
    vram_budget: u64,
) -> Result<ExpertTierPlan, ExpertTierError> {
    if slabs.len() != TOTAL_EXPERTS {
        return Err(ExpertTierError::SlabCount { got: slabs.len() });
    }

    // 1. Validate sources and compute each expert's padded size.
    let mut padded = Vec::with_capacity(TOTAL_EXPERTS);
    for (i, slab) in slabs.iter().enumerate() {
        let bad = || ExpertTierError::SourceSpan {
            layer: i / EXPERTS,
            expert: i % EXPERTS,
        };
        let (gu, dn) = (slab.gate_up, slab.down);
        if gu.len == 0 || dn.len == 0 {
            return Err(bad());
        }
        let gu_end = gu.end().ok_or_else(bad)?;
        let dn_end = dn.end().ok_or_else(bad)?;
        if !(gu_end <= dn.offset || dn_end <= gu.offset) {
            return Err(bad());
        }
        let gu_pad = align_up(gu.len as u64).ok_or(ExpertTierError::Overflow)?;
        let dn_pad = align_up(dn.len as u64).ok_or(ExpertTierError::Overflow)?;
        padded.push(
            gu_pad
                .checked_add(dn_pad)
                .ok_or(ExpertTierError::Overflow)?,
        );
    }

    // 2. Validate rank.
    let mut seen = vec![false; TOTAL_EXPERTS];
    for &id in rank {
        let idx = id as usize;
        if idx >= TOTAL_EXPERTS {
            return Err(ExpertTierError::RankIdOutOfRange { id });
        }
        if std::mem::replace(&mut seen[idx], true) {
            return Err(ExpertTierError::RankDuplicate { id });
        }
    }

    // 3./4. Longest rank prefix that fits, tail pad included.
    let tail = tail_pad();
    let mut on_device = vec![false; TOTAL_EXPERTS];
    let mut used = 0u64;
    let mut device_experts = 0usize;
    for &id in rank {
        let idx = id as usize;
        let next = used
            .checked_add(padded[idx])
            .ok_or(ExpertTierError::Overflow)?;
        let with_tail = next.checked_add(tail).ok_or(ExpertTierError::Overflow)?;
        if with_tail > vram_budget {
            break;
        }
        used = next;
        on_device[idx] = true;
        device_experts += 1;
    }

    // 5. Layout, (layer, expert) ascending within each tier.
    let mut cursor = [0u64; 2];
    let mut count = [0usize; 2];
    let mut entries = Vec::with_capacity(TOTAL_EXPERTS);
    let mut source = Vec::with_capacity(TOTAL_EXPERTS);
    for (i, slab) in slabs.iter().enumerate() {
        let tier = if on_device[i] {
            Placement::Device
        } else {
            Placement::HostMapped
        };
        let t = tier_index(tier);
        let gate_up_len = slab.gate_up.len as u64;
        let down_len = slab.down.len as u64;
        let gate_up_offset = align_up(cursor[t]).ok_or(ExpertTierError::Overflow)?;
        let after_gu = gate_up_offset
            .checked_add(gate_up_len)
            .ok_or(ExpertTierError::Overflow)?;
        let down_offset = align_up(after_gu).ok_or(ExpertTierError::Overflow)?;
        cursor[t] = down_offset
            .checked_add(down_len)
            .ok_or(ExpertTierError::Overflow)?;
        count[t] += 1;
        entries.push(ExpertTierEntry {
            format: slab.format,
            tier,
            gate_up_offset,
            gate_up_len,
            down_offset,
            down_len,
        });
        source.push((slab.gate_up, slab.down));
    }
    let owner = |t: usize| -> Result<u64, ExpertTierError> {
        if count[t] == 0 {
            return Ok(0);
        }
        align_up(cursor[t])
            .and_then(|c| c.checked_add(tail))
            .ok_or(ExpertTierError::Overflow)
    };
    let device_bytes = owner(0)?;
    let host_bytes = owner(1)?;
    assert_eq!(
        device_bytes,
        if device_experts == 0 { 0 } else { used + tail },
        "device owner layout must equal the admitted prefix size"
    );

    let digest = digest_of(&entries, &source, device_bytes, host_bytes, &[], None);
    Ok(ExpertTierPlan {
        entries,
        source,
        device_bytes,
        host_bytes,
        device_experts,
        host_windows: Vec::new(),
        decode_share: None,
        artifact_sha256: None,
        digest,
    })
}

/// Model ids ordered by descending count, ties by model id ascending, keeping
/// only ids for which `eligible` is true. Deterministic.
pub fn rank_from_counts(counts: &[u64], eligible: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..counts.len())
        .filter_map(|i| u32::try_from(i).ok())
        .filter(|&id| eligible(id))
        .collect();
    ids.sort_unstable_by(|&a, &b| {
        counts[b as usize]
            .cmp(&counts[a as usize])
            .then_with(|| a.cmp(&b))
    });
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;

    fn tail() -> u64 {
        tail_pad()
    }

    fn slabs_with(f: impl Fn(usize) -> (ExpertFormat, usize, usize)) -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..TOTAL_EXPERTS)
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

    #[test]
    fn budget_zero_is_all_host() {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[0, 1, 2], 0).unwrap();
        assert_eq!(plan.device_experts(), 0);
        assert_eq!(plan.owner_bytes(Placement::Device), 0);
        assert!(plan.owner_bytes(Placement::HostMapped) > 0);
        for layer in 0..48 {
            for expert in 0..512 {
                assert_eq!(
                    plan.entry(layer, expert).unwrap().tier,
                    Placement::HostMapped
                );
            }
        }
        assert_eq!(plan.copies(Placement::Device).count(), 0);
    }

    #[test]
    fn tier_stage_bytes_is_the_largest_host_payload_per_parity_plus_pad() {
        use crate::expert_residency::expert_tier_stage_bytes;
        let pad = crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64;
        // Nothing in VRAM: both parity buffers hold a full host layer.
        let all_host = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        assert_eq!(expert_tier_stage_bytes(&all_host), 2 * (512 * PADDED + pad));
        // Experts 0..256 of every layer resident: each buffer holds 256 host experts.
        let rank: Vec<u32> = (0..48u32)
            .flat_map(|layer| (0..256).map(move |e| layer * 512 + e))
            .collect();
        let half = plan_expert_tiers_from_slabs(&uniform(), &rank, u64::MAX).unwrap();
        assert_eq!(expert_tier_stage_bytes(&half), 2 * (256 * PADDED + pad));
        // Layer 1 additionally all resident: layer 3 still sizes the odd buffer.
        let mut more = rank.clone();
        more.extend((256..512).map(|e| 512 + e));
        let more = plan_expert_tiers_from_slabs(&uniform(), &more, u64::MAX).unwrap();
        assert_eq!(expert_tier_stage_bytes(&more), 2 * (256 * PADDED + pad));
    }

    #[test]
    fn exact_fit_and_one_byte_short() {
        let rank = [10u32, 5, 700, 3];
        let n = rank.len() as u64;
        let exact = n * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, exact).unwrap();
        assert_eq!(plan.device_experts(), 4);
        assert_eq!(plan.owner_bytes(Placement::Device), exact);

        let short = plan_expert_tiers_from_slabs(&uniform(), &rank, exact - 1).unwrap();
        assert_eq!(short.device_experts(), 3);
        assert_eq!(
            short.owner_bytes(Placement::Device),
            (n - 1) * PADDED + tail()
        );
        for &id in &rank[..3] {
            let (l, e) = (id as usize / 512, id as usize % 512);
            assert_eq!(short.entry(l, e).unwrap().tier, Placement::Device);
        }
        let last = rank[3] as usize;
        assert_eq!(
            short.entry(last / 512, last % 512).unwrap().tier,
            Placement::HostMapped
        );
    }

    #[test]
    fn rank_prefix_stops_at_first_non_fit() {
        // id 5 is huge; id 6 is small and would fit, but must not be admitted.
        let slabs = slabs_with(|i| {
            if i == 5 {
                (ExpertFormat::Mq4Xts, 100_000, 100_000)
            } else {
                (ExpertFormat::Mq4Xts, GU, DN)
            }
        });
        let budget = 2 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&slabs, &[0, 5, 6], budget).unwrap();
        assert_eq!(plan.device_experts(), 1);
        assert_eq!(plan.entry(0, 0).unwrap().tier, Placement::Device);
        assert_eq!(plan.entry(0, 5).unwrap().tier, Placement::HostMapped);
        assert_eq!(plan.entry(0, 6).unwrap().tier, Placement::HostMapped);
    }

    #[test]
    fn alignment_padding_with_odd_lengths() {
        let slabs = slabs_with(|_| (ExpertFormat::Mq4Xts, 1001, 3));
        let plan = plan_expert_tiers_from_slabs(&slabs, &[0, 1], 1 << 20).unwrap();
        // 1001 -> 1024, 3 -> 256.
        assert_eq!(plan.owner_bytes(Placement::Device), 2 * 1280 + tail());
        let e0 = plan.entry(0, 0).unwrap();
        let e1 = plan.entry(0, 1).unwrap();
        assert_eq!((e0.gate_up_offset, e0.down_offset), (0, 1024));
        assert_eq!((e1.gate_up_offset, e1.down_offset), (1280, 1280 + 1024));
        assert_eq!((e1.gate_up_len, e1.down_len), (1001, 3));
        // Host tier starts at its own base.
        let h = plan.entry(0, 2).unwrap();
        assert_eq!(h.tier, Placement::HostMapped);
        assert_eq!((h.gate_up_offset, h.down_offset), (0, 1024));
        for entry in [e0, e1, h] {
            assert_eq!(entry.gate_up_offset % 256, 0);
            assert_eq!(entry.down_offset % 256, 0);
        }
    }

    #[test]
    fn host_owner_beyond_4_gib_keeps_u64_offsets() {
        // Synthetic spans: nothing is allocated, only numbers.
        const L: usize = 1 << 28; // 256 MiB per projection
        let slabs = slabs_with(|_| (ExpertFormat::Mq4Xts, L, L));
        let plan = plan_expert_tiers_from_slabs(&slabs, &[], 0).unwrap();
        let per = 2 * L as u64;
        let last = plan.entry(47, 511).unwrap();
        let idx = (TOTAL_EXPERTS - 1) as u64;
        assert_eq!(last.gate_up_offset, idx * per);
        assert_eq!(last.down_offset, idx * per + L as u64);
        assert!(last.gate_up_offset > u64::from(u32::MAX));
        assert!(last.down_offset > u64::from(u32::MAX));
        assert_eq!(
            plan.owner_bytes(Placement::HostMapped),
            TOTAL_EXPERTS as u64 * per + tail()
        );
        let first_over = plan
            .copies(Placement::HostMapped)
            .find(|c| c.dst_offset > u64::from(u32::MAX))
            .unwrap();
        assert_eq!(first_over.dst_offset % 256, 0);
    }

    #[test]
    fn invalid_rank_is_rejected() {
        let slabs = uniform();
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[1, 24_576], 1 << 20),
            Err(ExpertTierError::RankIdOutOfRange { id: 24_576 })
        );
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[1, 2, 1], 1 << 20),
            Err(ExpertTierError::RankDuplicate { id: 1 })
        );
        // Validation covers ids past the budget stop point too.
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[4, 4], 0),
            Err(ExpertTierError::RankDuplicate { id: 4 })
        );
    }

    #[test]
    fn slab_count_and_source_spans_are_validated() {
        let mut slabs = uniform();
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs[..10], &[], 0),
            Err(ExpertTierError::SlabCount { got: 10 })
        );
        slabs.push(slabs[0]);
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[], 0),
            Err(ExpertTierError::SlabCount {
                got: TOTAL_EXPERTS + 1
            })
        );
        slabs.pop();

        let mut empty = slabs.clone();
        empty[513].down.len = 0;
        assert_eq!(
            plan_expert_tiers_from_slabs(&empty, &[], 0),
            Err(ExpertTierError::SourceSpan {
                layer: 1,
                expert: 1
            })
        );

        let mut overlap = slabs.clone();
        overlap[7].down.offset = overlap[7].gate_up.offset + 1;
        assert_eq!(
            plan_expert_tiers_from_slabs(&overlap, &[], 0),
            Err(ExpertTierError::SourceSpan {
                layer: 0,
                expert: 7
            })
        );

        let mut wrap = slabs.clone();
        wrap[3].gate_up.offset = usize::MAX - 5;
        assert_eq!(
            plan_expert_tiers_from_slabs(&wrap, &[], 0),
            Err(ExpertTierError::SourceSpan {
                layer: 0,
                expert: 3
            })
        );
    }

    #[test]
    fn u64_overflow_is_reported() {
        let mut slabs = uniform();
        slabs[0].gate_up = ExpertSlabSpan {
            offset: 0,
            len: usize::MAX - 10,
        };
        slabs[0].down = ExpertSlabSpan {
            offset: usize::MAX - 5,
            len: 1,
        };
        assert_eq!(
            plan_expert_tiers_from_slabs(&slabs, &[], 0),
            Err(ExpertTierError::Overflow)
        );
    }

    #[test]
    fn layout_is_layer_expert_order_regardless_of_rank_order() {
        let budget = 3 * PADDED + tail();
        let a = plan_expert_tiers_from_slabs(&uniform(), &[600, 2, 100], budget).unwrap();
        let b = plan_expert_tiers_from_slabs(&uniform(), &[2, 100, 600], budget).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.logical_digest(), b.logical_digest());
        // Ascending model-id order inside the device owner.
        assert_eq!(a.entry(0, 2).unwrap().gate_up_offset, 0);
        assert_eq!(a.entry(0, 100).unwrap().gate_up_offset, PADDED);
        assert_eq!(a.entry(1, 88).unwrap().gate_up_offset, 2 * PADDED);
        let order: Vec<_> = a
            .copies(Placement::Device)
            .map(|c| (c.layer, c.expert, c.proj))
            .collect();
        assert_eq!(
            order,
            vec![
                (0, 2, TierProj::GateUp),
                (0, 2, TierProj::Down),
                (0, 100, TierProj::GateUp),
                (0, 100, TierProj::Down),
                (1, 88, TierProj::GateUp),
                (1, 88, TierProj::Down),
            ]
        );
    }

    #[test]
    fn digest_covers_placement_format_source_artifact_and_window() {
        let budget = 2 * PADDED + tail();
        let base = plan_expert_tiers_from_slabs(&uniform(), &[0, 1], budget).unwrap();
        let d0 = base.logical_digest();
        assert_eq!(
            d0,
            plan_expert_tiers_from_slabs(&uniform(), &[0, 1], budget)
                .unwrap()
                .logical_digest()
        );

        // One placement bit.
        let placement = plan_expert_tiers_from_slabs(&uniform(), &[0, 2], budget).unwrap();
        assert_ne!(d0, placement.logical_digest());

        // One format (same lengths).
        let mut fmt_slabs = uniform();
        fmt_slabs[9].format = ExpertFormat::Mq3I4;
        let fmt = plan_expert_tiers_from_slabs(&fmt_slabs, &[0, 1], budget).unwrap();
        assert_ne!(d0, fmt.logical_digest());

        // One source span.
        let mut src_slabs = uniform();
        src_slabs[20].gate_up.offset -= 1;
        let src = plan_expert_tiers_from_slabs(&src_slabs, &[0, 1], budget).unwrap();
        assert_ne!(d0, src.logical_digest());

        let sha = base.clone().with_artifact_sha256([7u8; 32]);
        assert_ne!(d0, sha.logical_digest());
        assert_ne!(
            sha.logical_digest(),
            base.clone()
                .with_artifact_sha256([8u8; 32])
                .logical_digest()
        );

        let win = base
            .clone()
            .with_host_window(
                "mtp",
                &[ExpertSlabSpan {
                    offset: 1 << 40,
                    len: 100,
                }],
            )
            .unwrap();
        assert_ne!(d0, win.logical_digest());
    }

    #[test]
    fn mixed_formats_are_preserved_and_check_executable_gates_by_arch() {
        let slabs = slabs_with(|i| match i % 3 {
            0 => (ExpertFormat::Mq4Xts, 1000, 500),
            1 => (ExpertFormat::Mq3I4, 800, 400),
            _ => (ExpertFormat::Mq2I4, 600, 300),
        });
        let plan = plan_expert_tiers_from_slabs(&slabs, &[1, 2, 3], 1 << 20).unwrap();
        for (i, slab) in slabs.iter().enumerate() {
            let e = plan.entry(i / 512, i % 512).unwrap();
            assert_eq!(e.format, slab.format);
            assert_eq!(
                (e.gate_up_len, e.down_len),
                (slab.gate_up.len as u64, slab.down.len as u64)
            );
            assert_eq!(
                plan.source(i / 512, i % 512),
                Some((slab.gate_up, slab.down))
            );
        }
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            assert!(matches!(
                plan.check_executable(arch),
                Err(ExpertTierError::UnsupportedFormat {
                    format: ExpertFormat::Mq3I4 | ExpertFormat::Mq2I4,
                    ..
                })
            ));
        }
        assert_eq!(
            plan.check_executable("gfx1201"),
            Err(ExpertTierError::UnsupportedFormat {
                format: ExpertFormat::Mq2I4,
                arch: "gfx1201".into()
            })
        );

        let w3 = slabs_with(|i| {
            if i % 2 == 0 {
                (ExpertFormat::Mq4Xts, GU, DN)
            } else {
                (ExpertFormat::Mq3I4, GU, DN)
            }
        });
        let plan = plan_expert_tiers_from_slabs(&w3, &[], 0).unwrap();
        assert_eq!(plan.check_executable("gfx1201"), Ok(()));
        for arch in ["gfx1100", "gfx1151"] {
            assert_eq!(
                plan.check_executable(arch),
                Err(ExpertTierError::UnsupportedFormat {
                    format: ExpertFormat::Mq3I4,
                    arch: arch.into()
                })
            );
        }

        let mq4 = plan_expert_tiers_from_slabs(&uniform(), &[], 0).unwrap();
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            assert_eq!(mq4.check_executable(arch), Ok(()));
        }
    }

    #[test]
    fn tail_pad_follows_the_last_copy_in_every_owner() {
        let plan = plan_expert_tiers_from_slabs(&uniform(), &[0, 1, 2], 1 << 20).unwrap();
        assert!(tail() >= ARENA_TAIL_PAD_BYTES as u64);
        assert_eq!(tail() % 256, 0);
        for tier in [Placement::Device, Placement::HostMapped] {
            let last_end = plan
                .copies(tier)
                .map(|c| c.dst_offset + c.src.len as u64)
                .max()
                .unwrap();
            assert!(
                plan.owner_bytes(tier) >= last_end + ARENA_TAIL_PAD_BYTES as u64,
                "{tier:?}"
            );
            assert_eq!(plan.owner_bytes(tier) % 256, 0);
        }
    }

    #[test]
    fn copy_census_is_exact_and_non_overlapping() {
        let rank: Vec<u32> = (0..24_576).step_by(7).collect();
        let budget = 1000 * PADDED + tail();
        let plan = plan_expert_tiers_from_slabs(&uniform(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), 1000);

        let mut seen = vec![0u8; TOTAL_EXPERTS * 2];
        for tier in [Placement::Device, Placement::HostMapped] {
            let bound = plan.owner_bytes(tier) - tail();
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for c in plan.copies(tier) {
                let slot = (c.layer * 512 + c.expert) * 2 + usize::from(c.proj == TierProj::Down);
                seen[slot] += 1;
                assert_eq!(plan.entry(c.layer, c.expert).unwrap().tier, tier);
                ranges.push((c.dst_offset, c.dst_offset + c.src.len as u64));
            }
            ranges.sort_unstable();
            for w in ranges.windows(2) {
                assert!(w[0].1 <= w[1].0, "overlap {w:?}");
            }
            assert!(ranges.last().unwrap().1 <= bound);
        }
        assert!(seen.iter().all(|&n| n == 1));
    }

    #[test]
    fn host_window_appends_after_trunk_host_end_with_single_tail() {
        let base = plan_expert_tiers_from_slabs(&uniform(), &[0], PADDED + tail()).unwrap();
        let host_before = base.owner_bytes(Placement::HostMapped);
        let host_end = host_before - tail();
        let spans = [
            ExpertSlabSpan {
                offset: 5_000_000_000,
                len: 1000,
            },
            ExpertSlabSpan {
                offset: 6_000_000_000,
                len: 513,
            },
        ];
        let plan = base.clone().with_host_window("mtp", &spans).unwrap();
        let w = &plan.host_windows()[0];
        assert_eq!(w.label, "mtp");
        assert_eq!(w.spans[0], (spans[0], host_end));
        assert_eq!(w.spans[1], (spans[1], host_end + 1024));
        assert_eq!(
            plan.owner_bytes(Placement::HostMapped),
            host_end + 1024 + 768 + tail()
        );
        assert_eq!(
            plan.owner_bytes(Placement::Device),
            base.owner_bytes(Placement::Device)
        );

        // Window into an otherwise empty host tier allocates it, tail once.
        let all_dev =
            plan_expert_tiers_from_slabs(&uniform(), &(0..24_576).collect::<Vec<_>>(), u64::MAX)
                .unwrap();
        assert_eq!(all_dev.owner_bytes(Placement::HostMapped), 0);
        let w = all_dev.with_host_window("mtp", &spans[..1]).unwrap();
        assert_eq!(w.owner_bytes(Placement::HostMapped), 1024 + tail());
        assert_eq!(w.host_windows()[0].spans[0].1, 0);

        assert_eq!(
            base.clone()
                .with_host_window("x", &[ExpertSlabSpan { offset: 0, len: 0 }]),
            Err(ExpertTierError::Overflow)
        );
        assert_eq!(
            base.with_host_window(
                "x",
                &[ExpertSlabSpan {
                    offset: usize::MAX,
                    len: 2
                }]
            ),
            Err(ExpertTierError::Overflow)
        );
    }

    #[test]
    fn rank_from_counts_breaks_ties_by_id_and_filters() {
        let counts = [5u64, 9, 5, 0, 9, 5];
        assert_eq!(rank_from_counts(&counts, |_| true), vec![1, 4, 0, 2, 5, 3]);
        assert_eq!(rank_from_counts(&counts, |id| id % 2 == 0), vec![4, 0, 2]);
        assert!(rank_from_counts(&counts, |_| false).is_empty());
        assert!(rank_from_counts(&[], |_| true).is_empty());
    }

    #[test]
    fn error_display_is_informative() {
        let e = ExpertTierError::UnsupportedFormat {
            format: ExpertFormat::Mq2I4,
            arch: "gfx1100".into(),
        };
        assert!(e.to_string().contains("gfx1100"));
        let _: &dyn std::error::Error = &e;
        assert_eq!(CODEC_W2, 2);
    }

    // ---- decode-share donor tail ----

    fn model_ids(layer: usize, experts: std::ops::Range<usize>) -> Vec<u32> {
        experts.map(|e| (layer * EXPERTS + e) as u32).collect()
    }

    fn device_plan(rank: &[u32]) -> ExpertTierPlan {
        plan_expert_tiers_from_slabs(&uniform(), rank, u64::MAX).unwrap()
    }

    /// Every device projection as `(offset, padded_len)`.
    fn device_ranges(plan: &ExpertTierPlan) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for id in 0..TOTAL_EXPERTS {
            let e = plan.entry(id / EXPERTS, id % EXPERTS).unwrap();
            if e.tier == Placement::Device {
                out.push((e.gate_up_offset, align_up(e.gate_up_len).unwrap()));
                out.push((e.down_offset, align_up(e.down_len).unwrap()));
            }
        }
        out.sort_unstable();
        out
    }

    /// Every structural promise of `with_decode_share`, checked against the
    /// pre-sharing plan.
    fn check_shared(
        before: &ExpertTierPlan,
        after: &ExpertTierPlan,
        tail_info: &DecodeShareTail,
        rank: &[u32],
        need: u64,
    ) {
        let pad = tail();
        assert_eq!(after.decode_share_tail(), Some(tail_info));
        // Membership and counts unchanged.
        assert_eq!(after.device_experts(), before.device_experts());
        let mut layer_has_host = [false; 48];
        for id in 0..TOTAL_EXPERTS {
            let (l, e) = (id / EXPERTS, id % EXPERTS);
            let (b, a) = (before.entry(l, e).unwrap(), after.entry(l, e).unwrap());
            assert_eq!(b.tier, a.tier, "tier of {id}");
            assert_eq!(
                (b.format, b.gate_up_len, b.down_len),
                (a.format, a.gate_up_len, a.down_len)
            );
            assert_eq!(before.source(l, e), after.source(l, e));
            layer_has_host[l] |= b.tier == Placement::HostMapped;
        }
        // Donors: ascending, Device, in the admitted rank, eligible layer.
        let ids = &tail_info.donors;
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        let mut donor_sum = 0;
        for &id in ids {
            let (l, e) = (id as usize / EXPERTS, id as usize % EXPERTS);
            assert_eq!(before.entry(l, e).unwrap().tier, Placement::Device);
            assert!(rank.contains(&id));
            assert!(layer_has_host[l], "donor {id} sits in an all-device layer");
            donor_sum += PADDED;
        }
        assert_eq!(tail_info.donor_bytes, donor_sum);
        assert!(tail_info.donor_bytes >= need);
        // Coldest first: dropping the hottest donor must fall short of `need`.
        assert!(tail_info.donor_bytes - PADDED < need);
        // Layout.
        let end = tail_info.donor_offset + tail_info.donor_bytes;
        assert_eq!(tail_info.donor_offset % 256, 0);
        assert_eq!(tail_info.guard_offset % 256, 0);
        assert_eq!(tail_info.donor_offset, tail_info.guard_offset + pad);
        assert_eq!(after.owner_bytes(Placement::Device), end + pad);
        let ranges = device_ranges(after);
        for (off, len) in &ranges {
            assert_eq!(off % 256, 0);
            assert!(len > &0);
        }
        for w in ranges.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "overlap {w:?}");
        }
        let donor_set: std::collections::HashSet<u32> = ids.iter().copied().collect();
        for id in 0..TOTAL_EXPERTS {
            let e = after.entry(id / EXPERTS, id % EXPERTS).unwrap();
            if e.tier != Placement::Device {
                continue;
            }
            let projs = [
                (e.gate_up_offset, align_up(e.gate_up_len).unwrap()),
                (e.down_offset, align_up(e.down_len).unwrap()),
            ];
            for (off, len) in projs {
                if donor_set.contains(&(id as u32)) {
                    assert!(off >= tail_info.donor_offset && off + len <= end);
                } else {
                    // Non-donors lie wholly below the guard.
                    assert!(
                        off + len <= tail_info.guard_offset,
                        "non-donor {id} past guard"
                    );
                }
                // Guard disjoint.
                assert!(off + len <= tail_info.guard_offset || off >= tail_info.guard_offset + pad);
            }
        }
        // Mirror window: exactly the donors' source spans in donor order.
        let w = after.host_windows().last().unwrap();
        assert_eq!(w.label, DECODE_SHARE_MIRROR_LABEL);
        let expect: Vec<ExpertSlabSpan> = ids
            .iter()
            .flat_map(|&id| {
                let (gu, dn) = before
                    .source(id as usize / EXPERTS, id as usize % EXPERTS)
                    .unwrap();
                [gu, dn]
            })
            .collect();
        assert_eq!(w.spans.iter().map(|(s, _)| *s).collect::<Vec<_>>(), expect);
        assert_eq!(w.spans[0].1, tail_info.mirror_offset);
        // Rule 5: host_dst - mirror_offset == device_offset - donor_offset.
        for (k, &id) in ids.iter().enumerate() {
            let e = after
                .entry(id as usize / EXPERTS, id as usize % EXPERTS)
                .unwrap();
            assert_eq!(
                w.spans[2 * k].1 - tail_info.mirror_offset,
                e.gate_up_offset - tail_info.donor_offset
            );
            assert_eq!(
                w.spans[2 * k + 1].1 - tail_info.mirror_offset,
                e.down_offset - tail_info.donor_offset
            );
        }
        assert!(
            tail_info.mirror_offset + tail_info.donor_bytes
                <= after.owner_bytes(Placement::HostMapped)
        );
        // borrow_entries differ from the plan's entries only at donors.
        assert_eq!(tail_info.borrow_entries.len(), TOTAL_EXPERTS);
        for id in 0..TOTAL_EXPERTS {
            let plan_entry = after.entry(id / EXPERTS, id % EXPERTS).unwrap();
            let borrow = &tail_info.borrow_entries[id];
            if let Some(k) = ids.iter().position(|&d| d as usize == id) {
                assert_eq!(borrow.tier, Placement::HostMapped);
                assert_eq!(borrow.gate_up_offset, w.spans[2 * k].1);
                assert_eq!(borrow.down_offset, w.spans[2 * k + 1].1);
                assert_eq!(
                    (borrow.gate_up_len, borrow.down_len),
                    (plan_entry.gate_up_len, plan_entry.down_len)
                );
                assert_eq!(borrow.format, plan_entry.format);
            } else {
                assert_eq!(borrow, plan_entry, "entry {id}");
            }
        }
        // Digest covers the new layout.
        assert_ne!(after.logical_digest(), before.logical_digest());
    }

    #[test]
    fn decode_share_exact_layout_and_size() {
        let rank = model_ids(0, 0..8);
        let before = device_plan(&rank);
        let (after, t) = before.clone().with_decode_share(&rank, 3 * PADDED).unwrap();
        assert_eq!(t.donors, vec![5, 6, 7]);
        assert_eq!(t.donor_bytes, 3 * PADDED);
        // Five non-donors: 4 * 1536 + 1024 + 500 payload bytes end.
        let non_donor_end = 4 * PADDED + 1024 + DN as u64;
        assert_eq!(t.guard_offset, align_up(non_donor_end).unwrap());
        assert_eq!(t.donor_offset, t.guard_offset + tail());
        assert_eq!(
            after.owner_bytes(Placement::Device),
            t.donor_offset + 3 * PADDED + tail()
        );
        let e5 = after.entry(0, 5).unwrap();
        assert_eq!(
            (e5.gate_up_offset, e5.down_offset),
            (t.donor_offset, t.donor_offset + 1024)
        );
        let e7 = after.entry(0, 7).unwrap();
        assert_eq!(e7.gate_up_offset, t.donor_offset + 2 * PADDED);
        // Non-donors keep the dense (layer, expert) prefix layout.
        let e4 = after.entry(0, 4).unwrap();
        assert_eq!(e4.gate_up_offset, 4 * PADDED);
        check_shared(&before, &after, &t, &rank, 3 * PADDED);
    }

    #[test]
    fn decode_share_layout_with_odd_lengths_and_several_layers() {
        let slabs = slabs_with(|_| (ExpertFormat::Mq4Xts, 1001, 3));
        let rank: Vec<u32> = [model_ids(2, 0..30), model_ids(5, 100..140)].concat();
        let before = plan_expert_tiers_from_slabs(&slabs, &rank, u64::MAX).unwrap();
        let need = 40 * 1280;
        let (after, t) = before.clone().with_decode_share(&rank, need).unwrap();
        // 1001 -> 1024, 3 -> 256: 1280 padded per expert; the 40 coldest are layer 5's.
        assert_eq!(t.donor_bytes, need);
        assert_eq!(t.donors, model_ids(5, 100..140));
        assert_eq!(after.device_experts(), 70);
        assert_eq!(
            after.owner_bytes(Placement::Device),
            t.donor_offset + need + tail()
        );
        assert_eq!(
            after.decode_share_tail().unwrap().borrow_entries[(5 * EXPERTS + 100)].tier,
            Placement::HostMapped
        );
    }

    #[test]
    fn decode_share_invariants_over_mixed_ranks() {
        // Resident experts spread over several layers, hot to cold.
        let mut rank = Vec::new();
        for e in 0..40 {
            for l in [0usize, 3, 7, 47] {
                rank.push((l * EXPERTS + e) as u32);
            }
        }
        let before = device_plan(&rank);
        for need in [1, PADDED, 7 * PADDED - 3, 50 * PADDED, 160 * PADDED] {
            let (after, t) = before.clone().with_decode_share(&rank, need).unwrap();
            check_shared(&before, &after, &t, &rank, need);
        }
    }

    #[test]
    fn decode_share_skips_experts_of_all_device_layers() {
        // Layer 0 is entirely resident and coldest; layer 1 has 3 resident experts
        // (and 509 host experts). Layer 0 never donates.
        let mut rank = model_ids(1, 0..3);
        rank.extend(model_ids(0, 0..512));
        let before = device_plan(&rank);
        assert_eq!(before.device_experts(), 515);

        let (after, t) = before.clone().with_decode_share(&rank, 2 * PADDED).unwrap();
        assert_eq!(t.donors, model_ids(1, 1..3));
        check_shared(&before, &after, &t, &rank, 2 * PADDED);
        // Layer 0's tier is untouched in both views.
        for e in 0..512 {
            assert_eq!(after.entry(0, e).unwrap().tier, Placement::Device);
            assert_eq!(t.borrow_entries[e].tier, Placement::Device);
        }

        // Exactly three eligible experts exist; a fourth is not borrowable.
        assert!(before.clone().with_decode_share(&rank, 3 * PADDED).is_ok());
        assert_eq!(
            before.with_decode_share(&rank, 3 * PADDED + 1),
            Err(DecodeShareError::Insufficient {
                need: 3 * PADDED + 1,
                available: 3 * PADDED
            })
        );
    }

    #[test]
    fn decode_share_insufficient_budget() {
        // Nothing resident.
        let none = device_plan(&[]);
        assert_eq!(
            none.with_decode_share(&[], 1),
            Err(DecodeShareError::Insufficient {
                need: 1,
                available: 0
            })
        );
        // A fully resident machine has no host expert anywhere: nothing donates.
        let all: Vec<u32> = (0..TOTAL_EXPERTS as u32).collect();
        let plan = device_plan(&all);
        assert_eq!(
            plan.with_decode_share(&all, 1),
            Err(DecodeShareError::Insufficient {
                need: 1,
                available: 0
            })
        );
        // Short by one byte.
        let rank = model_ids(0, 0..4);
        let plan = device_plan(&rank);
        assert_eq!(
            plan.with_decode_share(&rank, 4 * PADDED + 1),
            Err(DecodeShareError::Insufficient {
                need: 4 * PADDED + 1,
                available: 4 * PADDED
            })
        );
    }

    #[test]
    fn decode_share_one_expert_and_exact_fit_boundary() {
        let rank = model_ids(0, 0..8);
        let before = device_plan(&rank);
        // need == one expert's padded bytes: exactly the coldest expert.
        let (after, t) = before.clone().with_decode_share(&rank, PADDED).unwrap();
        assert_eq!(t.donors, vec![7]);
        assert_eq!(t.donor_bytes, PADDED);
        check_shared(&before, &after, &t, &rank, PADDED);
        // One byte more needs the second-coldest too.
        let (after, t) = before.clone().with_decode_share(&rank, PADDED + 1).unwrap();
        assert_eq!(t.donors, vec![6, 7]);
        assert_eq!(t.donor_bytes, 2 * PADDED);
        check_shared(&before, &after, &t, &rank, PADDED + 1);
        // One byte less than one expert is still one expert.
        let (_, t) = before.clone().with_decode_share(&rank, PADDED - 1).unwrap();
        assert_eq!(t.donors, vec![7]);
        // Donate every resident expert of the only host-bearing layer.
        let (after, t) = before.with_decode_share(&rank, 8 * PADDED).unwrap();
        assert_eq!(t.donors, model_ids(0, 0..8));
        assert_eq!(t.guard_offset, 0);
        assert_eq!(
            after.owner_bytes(Placement::Device),
            tail() + 8 * PADDED + tail()
        );
    }

    #[test]
    fn decode_share_need_zero_is_identity() {
        let rank = model_ids(0, 0..8);
        let before = device_plan(&rank);
        let (after, t) = before.clone().with_decode_share(&rank, 0).unwrap();
        assert_eq!(after, before);
        assert_eq!(after.logical_digest(), before.logical_digest());
        assert!(after.host_windows().is_empty());
        assert!(after.decode_share_tail().is_none());
        assert!(t.donors.is_empty());
        assert_eq!(
            (
                t.guard_offset,
                t.donor_offset,
                t.donor_bytes,
                t.mirror_offset
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            t.borrow_entries,
            (0..TOTAL_EXPERTS)
                .map(|id| *before.entry(id / EXPERTS, id % EXPERTS).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn decode_share_digest_changes_only_with_sharing() {
        let rank = model_ids(0, 0..8);
        let before = device_plan(&rank);
        let twin = device_plan(&rank);
        assert_eq!(before.logical_digest(), twin.logical_digest());
        let (a, _) = before.clone().with_decode_share(&rank, PADDED).unwrap();
        let (b, _) = before.clone().with_decode_share(&rank, 2 * PADDED).unwrap();
        let (a2, _) = twin.with_decode_share(&rank, PADDED).unwrap();
        assert_ne!(a.logical_digest(), before.logical_digest());
        assert_ne!(a.logical_digest(), b.logical_digest());
        assert_eq!(a.logical_digest(), a2.logical_digest());
        assert_eq!(a.logical_digest(), a.recompute_digest());
        // A sharing plan cannot be re-shared.
        assert_eq!(
            a.with_decode_share(&rank, PADDED).map(|_| ()),
            Err(DecodeShareError::AlreadyShared)
        );
    }

    #[test]
    fn decode_share_keeps_existing_host_windows_and_rejects_duplicate_ranks() {
        let rank = model_ids(0, 0..8);
        let mtp = [ExpertSlabSpan {
            offset: 9_000_000,
            len: 777,
        }];
        let before = device_plan(&rank).with_host_window("mtp", &mtp).unwrap();
        let (after, t) = before.clone().with_decode_share(&rank, 2 * PADDED).unwrap();
        assert_eq!(after.host_windows().len(), 2);
        assert_eq!(after.host_windows()[0], before.host_windows()[0]);
        assert!(t.mirror_offset >= 1024);
        check_shared(&before, &after, &t, &rank, 2 * PADDED);
        // The plan builder, not the sharing pass, rejects a duplicate rank.
        assert_eq!(
            plan_expert_tiers_from_slabs(&uniform(), &[1, 2, 1], u64::MAX),
            Err(ExpertTierError::RankDuplicate { id: 1 })
        );
    }
}
