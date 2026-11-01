// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EP reserve decoupling, slice P2: the scratch-plan adapter and the two scratch
//! banks of an `HIPFIRE_EP_RESERVE_DECOUPLE=1` load (`ep-reserve/DECISION.md`
//! §4, §5, §7.1, §7.2).
//!
//! A child module of `gpu_forward`, so it reuses the scratch internals
//! ([`Qwen4GpuForwardScratch::plan_scoped`], `bind_views`, `from_tensors`)
//! instead of keeping a second list of dimensions.
//!
//! # Memory model
//!
//! * `TokenIds` (prompt rows) and `MoeYDownGrouped` (prompt rows) are
//!   **persistent**: one allocation each, owned by [`EpReserveBanks`], viewed
//!   (never owned) by both banks. The registered tile-exchange address
//!   (`ep_tile_registration` -> `&scratch.moe_y_down_grouped`) is therefore the
//!   same in the resident and the borrowed bank and across every swap. A tensor
//!   root cannot span two donor allocations (the 475 MiB grouped-down root is
//!   larger than either an EP2 gate/up (425 MiB) or down (212.5 MiB) donor
//!   segment), which is why it is not packed into donor memory.
//! * The resident bank (512 rows) owns, through [`EpReserveBanks`], every other
//!   root of its plan; its grouped-down view spans the whole persistent root,
//!   like `TokenIds` (the tile exchange rejects a buffer whose size differs
//!   from its registration; kernels use explicit row counts).
//! * The borrowed bank (prompt rows) allocates nothing: every non-persistent
//!   root is a non-owning view at `segment.address + offset` of a donor
//!   segment, placed by the plan (`EpRootPlacement`), no root straddling two
//!   segments.
//! * Both banks are built with `ScratchBackingKind::Borrowed`, so
//!   [`Qwen4GpuForwardScratch::free_gpu`] frees nothing of either. Every
//!   device allocation of the load's scratch has exactly one owner:
//!   [`EpReserveBanks::release`] frees each allocation once.

use super::{
    invalid, DType, Gpu, GpuTensor, Qwen4BandScope, Qwen4Config, Qwen4ForwardScratchMode,
    Qwen4GpuForwardError, Qwen4GpuForwardScratch, ScratchBackingKind, ScratchField, ScratchSlot,
    DECODE_SHARE_ROOT_ALIGN, MOE_GROUPED_BLOCK_M, QWEN4_DECODE_SHARE_RESIDENT_ROWS,
};

/// Rows of the resident bank (decode, verify, every call of at most this many rows).
pub(crate) const EP_RESERVE_RESIDENT_ROWS: usize = QWEN4_DECODE_SHARE_RESIDENT_ROWS;

/// One root of a scratch plan: `(field, bytes, persistent)`.
pub(crate) type EpRootSpec = (ScratchField, u64, bool);

/// Whether `field` is a persistent root: one allocation sized for the prompt rows,
/// owned by [`EpReserveBanks`], never packed into donor memory.
pub(crate) const fn ep_reserve_persistent_root(field: ScratchField) -> bool {
    matches!(
        field,
        ScratchField::TokenIds | ScratchField::MoeYDownGrouped
    )
}

/// Every owning non-empty root of `slots` in [`ScratchField`] order; `None` on
/// overflow. Borrowed alias views (`owner.is_some()`) are not roots.
fn plan_roots(slots: &[ScratchSlot]) -> Option<Vec<EpRootSpec>> {
    let mut roots = Vec::with_capacity(slots.len());
    for (index, slot) in slots.iter().enumerate() {
        if slot.owner.is_some() {
            continue;
        }
        let bytes = u64::try_from(slot.bytes().ok()?).ok()?;
        if bytes == 0 {
            continue;
        }
        let field = ScratchField::ALL[index];
        roots.push((field, bytes, ep_reserve_persistent_root(field)));
    }
    Some(roots)
}

/// The root specs of `Qwen4GpuForwardScratch::plan_scoped(config, rows, mode,
/// single scope)`: every owning non-empty root in [`ScratchField`] order with its
/// byte size and whether it is persistent (`TokenIds` and `MoeYDownGrouped` only).
/// `None` on arithmetic overflow or a geometry the plan refuses. CPU-only.
pub(crate) fn ep_reserve_scratch(
    config: &Qwen4Config,
    mode: Qwen4ForwardScratchMode,
    rows: usize,
) -> Option<Vec<EpRootSpec>> {
    let (slots, _) =
        Qwen4GpuForwardScratch::plan_scoped(config, rows, mode, &Qwen4BandScope::single(config))
            .ok()?;
    plan_roots(&slots)
}

/// Bytes of the speculative logits and argmax buffers of `rows` rows (separate
/// persistent allocations, never donor-packed): the sum of
/// `qwen4_spec_buffer_bytes`. `None` on overflow.
pub(crate) fn ep_reserve_spec_bytes(config: &Qwen4Config, rows: usize) -> Option<u64> {
    let [logits, argmax] = super::qwen4_spec_buffer_bytes(config, rows)?;
    logits.checked_add(argmax)
}

/// Check that every alias view of `slots` (a plan of `rows` rows) lies inside its
/// owner's root, in bytes, including the `rows * pitch` reservation of a pitched
/// member. `Full` plans have no aliases; a compact plan does.
pub(crate) fn check_aliases_inside_roots(
    rows: usize,
    slots: &[ScratchSlot],
) -> Result<(), Qwen4GpuForwardError> {
    let overflow = || invalid("scratch alias extent overflows");
    for (index, slot) in slots.iter().enumerate() {
        let Some(owner) = slot.owner else { continue };
        if slot.elements == 0 {
            continue;
        }
        let field = ScratchField::ALL[index];
        let owner_slot = slots
            .get(owner as usize)
            .filter(|owner_slot| owner_slot.owner.is_none() && owner_slot.elements != 0)
            .ok_or_else(|| invalid(format!("{field:?} views {owner:?}, which owns no root")))?;
        if slot.dtype != owner_slot.dtype && slot.dtype != DType::Raw {
            return Err(invalid(format!(
                "{field:?} retypes its owner {owner:?} to something other than raw bytes"
            )));
        }
        let root_bytes = u64::try_from(owner_slot.bytes()?).map_err(|_| overflow())?;
        let owner_size = owner_slot.dtype.size() as u64;
        let first = (slot.offset as u64)
            .checked_mul(owner_size)
            .ok_or_else(overflow)?;
        let view_bytes = (slot.elements as u64)
            .checked_mul(slot.dtype.size() as u64)
            .ok_or_else(overflow)?;
        let mut end = first.checked_add(view_bytes).ok_or_else(overflow)?;
        if let Some(pitch) = slot.pitch {
            let pitched = (rows as u64)
                .checked_mul(pitch as u64)
                .and_then(|elements| elements.checked_mul(owner_size))
                .and_then(|bytes| first.checked_add(bytes))
                .ok_or_else(overflow)?;
            end = end.max(pitched);
        }
        if end > root_bytes {
            return Err(invalid(format!(
                "{field:?} ends at byte {end} of its owner {owner:?}'s {root_bytes}-byte root"
            )));
        }
    }
    Ok(())
}

/// One donor segment: live device memory the loan lends to the borrowed bank.
/// `address` is 256-aligned (a donor expert tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpDonorSegment {
    pub address: usize,
    pub bytes: u64,
}

/// Where one non-persistent root of the borrowed bank lives: `offset` bytes into
/// segment `segment`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpRootPlacement {
    pub field: ScratchField,
    pub segment: usize,
    pub offset: u64,
    pub bytes: u64,
}

/// Check `placements` against the borrowed plan's non-persistent roots and the
/// donor `segments`: every such root placed exactly once at its exact size, no
/// persistent root placed, every start 256-aligned, no root straddling a
/// segment, no two roots overlapping, segments 256-aligned and mutually
/// disjoint in the address space. All arithmetic is checked.
pub(crate) fn validate_placements(
    roots: &[EpRootSpec],
    segments: &[EpDonorSegment],
    placements: &[EpRootPlacement],
) -> Result<(), Qwen4GpuForwardError> {
    let overflow = || invalid("EP reserve placement arithmetic overflows");
    let align = DECODE_SHARE_ROOT_ALIGN;
    // Segments: aligned, representable, mutually disjoint.
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        if segment.address as u64 % align != 0 {
            return Err(invalid(format!(
                "donor segment {index} base {:#x} is not {align}-byte aligned",
                segment.address
            )));
        }
        let bytes = usize::try_from(segment.bytes).map_err(|_| overflow())?;
        let end = segment.address.checked_add(bytes).ok_or_else(overflow)?;
        ranges.push((segment.address, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
        return Err(invalid("donor segments overlap in the address space"));
    }
    // Roots: exactly the non-persistent roots, each placed once at its size.
    let expected: Vec<(ScratchField, u64)> = roots
        .iter()
        .filter(|(_, _, persistent)| !persistent)
        .map(|&(field, bytes, _)| (field, bytes))
        .collect();
    if placements.len() != expected.len() {
        return Err(invalid(format!(
            "{} root placements for {} donor-packed roots",
            placements.len(),
            expected.len()
        )));
    }
    let mut seen = [false; ScratchField::COUNT];
    for placement in placements {
        let Some(&(_, bytes)) = expected.iter().find(|(field, _)| *field == placement.field) else {
            return Err(invalid(format!(
                "{:?} is not a donor-packed root of the plan (persistent roots are never placed)",
                placement.field
            )));
        };
        if std::mem::replace(&mut seen[placement.field as usize], true) {
            return Err(invalid(format!("{:?} is placed twice", placement.field)));
        }
        if placement.bytes != bytes {
            return Err(invalid(format!(
                "{:?} is placed with {} bytes, its root has {bytes}",
                placement.field, placement.bytes
            )));
        }
        let segment = segments.get(placement.segment).ok_or_else(|| {
            invalid(format!(
                "{:?} names donor segment {} of {}",
                placement.field,
                placement.segment,
                segments.len()
            ))
        })?;
        if placement.offset % align != 0 {
            return Err(invalid(format!(
                "{:?} offset {} is not {align}-byte aligned",
                placement.field, placement.offset
            )));
        }
        let end = placement
            .offset
            .checked_add(placement.bytes)
            .ok_or_else(overflow)?;
        if end > segment.bytes {
            return Err(invalid(format!(
                "{:?} [{}, {end}) leaves its {}-byte donor segment {} (a root cannot straddle segments)",
                placement.field, placement.offset, segment.bytes, placement.segment
            )));
        }
    }
    let mut ordered: Vec<&EpRootPlacement> = placements.iter().collect();
    ordered.sort_unstable_by_key(|placement| (placement.segment, placement.offset));
    for pair in ordered.windows(2) {
        if pair[0].segment == pair[1].segment && pair[1].offset < pair[0].offset + pair[0].bytes {
            return Err(invalid(format!(
                "{:?} overlaps {:?} in donor segment {}",
                pair[1].field, pair[0].field, pair[0].segment
            )));
        }
    }
    Ok(())
}

/// The device operations the bank owner performs, so address/ownership logic runs
/// without a GPU. [`Gpu`] is the production implementation.
pub(crate) trait EpScratchDevice {
    /// A zeroed owning allocation of `elements` of `dtype`.
    fn alloc_zeroed(&mut self, elements: usize, dtype: DType) -> hip_bridge::HipResult<GpuTensor>;
    /// Free an owning allocation. The tensor is consumed whatever the result.
    fn release_tensor(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()>;
}

impl EpScratchDevice for Gpu {
    fn alloc_zeroed(&mut self, elements: usize, dtype: DType) -> hip_bridge::HipResult<GpuTensor> {
        Gpu::zeros(self, &[elements], dtype)
    }

    fn release_tensor(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
        self.free_tensor(tensor)
    }
}

/// Which bank is swapped into the forward's `scratch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpBankKind {
    /// The 512-row bank (decode, verify, lanes, calls of at most 512 rows).
    Resident,
    /// The prompt-row bank over donor segments (513..=prompt rows).
    Borrowed,
}

/// The sealed route of the call being admitted ([`EpReserveBanks::select`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpBankRoute {
    /// Compact owner-routed tile exchange: peers read the persistent grouped-down
    /// root, never donor weights, so a wide call may borrow.
    TileExchange,
    /// Full peer tables: they read the canonical donor-layer expert bytes, which
    /// a borrowed epoch overwrites. A call that needs more rows than the
    /// resident bank holds cannot run on this route.
    FullTable,
}

/// Result of [`EpReserveBanks::select`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpBankSelection {
    pub kind: EpBankKind,
    /// A bank swap happened (false: the requested bank was already active).
    pub swapped: bool,
}

/// Counters of the bank owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EpBankStats {
    /// Selections that ran a call on the borrowed bank.
    pub tiles_borrowed: u64,
    /// Swaps into the borrowed bank (one per borrowed epoch start).
    pub swaps_in: u64,
    /// Swaps back to the resident bank.
    pub swaps_out: u64,
}

/// The tile-exchange capacity of the prompt plan (not of the selected bank).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpExchangeCapacity {
    /// Prompt rows.
    pub rows: usize,
    /// Elements of the persistent grouped-down root.
    pub grouped_down_elements: usize,
    /// Bytes of the persistent grouped-down root.
    pub grouped_down_bytes: u64,
    /// `i32` tile ids the plan holds (`MoeExpertTileIds`).
    pub tile_ids: usize,
    /// Grouped rows the plan bounds (`tile_ids * MOE_GROUPED_BLOCK_M`).
    pub grouped_rows: usize,
}

/// The owners and banks of an EP reserve-decoupled load's forward scratch.
///
/// `Qwen4GpuForward::scratch` is the resident bank at every call boundary; a call
/// of more than [`EP_RESERVE_RESIDENT_ROWS`] rows on the tile-exchange route
/// swaps the borrowed bank in ([`Self::select`]) and a later resident call swaps
/// it back ([`Self::swap_out`]). Neither allocates, builds a table or grows a
/// vector, and neither moves the persistent roots.
pub(crate) struct EpReserveBanks {
    prompt_rows: usize,
    resident_rows: usize,
    /// Plan of the borrowed bank, in [`ScratchField`] order.
    prompt_slots: Vec<ScratchSlot>,
    /// Plan of the resident bank, in [`ScratchField`] order.
    resident_slots: Vec<ScratchSlot>,
    /// Every device allocation of this scratch, in allocation order: the two
    /// persistent roots, then the resident bank's other roots. Each is freed
    /// once, by [`Self::release`]; every scratch tensor is a non-owning view.
    owned: Vec<(ScratchField, GpuTensor)>,
    /// Donor memory the borrowed bank views, and where each root lives.
    segments: Vec<EpDonorSegment>,
    placements: Vec<EpRootPlacement>,
    /// Address of the persistent grouped-down root (registered with the tile
    /// exchange once) and of the persistent token ids.
    registered_y_down: usize,
    registered_ids: usize,
    /// The borrowed bank while the resident one is selected.
    borrowed: Option<Qwen4GpuForwardScratch>,
    /// The resident bank while the borrowed one is selected.
    parked: Option<Qwen4GpuForwardScratch>,
    active: EpBankKind,
    device_bytes: u64,
    /// Host staging sized for the largest bank.
    host_token_ids: Vec<u8>,
    host_ple: Vec<u8>,
    stats: EpBankStats,
}

/// Every allocation made so far; rolled back on a construction failure.
#[derive(Default)]
struct Allocation {
    owned: Vec<(ScratchField, GpuTensor)>,
}

impl Allocation {
    fn alloc<D: EpScratchDevice>(
        &mut self,
        device: &mut D,
        field: ScratchField,
        slot: &ScratchSlot,
    ) -> Result<(), Qwen4GpuForwardError> {
        let tensor = device.alloc_zeroed(slot.elements, slot.dtype)?;
        self.owned.push((field, tensor));
        Ok(())
    }

    /// Free everything allocated so far, once each. Rollback failures cannot be
    /// retried (the tensor is consumed) and never replace the original error.
    fn rollback<D: EpScratchDevice>(self, device: &mut D) {
        for (_, tensor) in self.owned {
            let _ = device.release_tensor(tensor);
        }
    }
}

impl EpReserveBanks {
    /// Build the resident bank (returned, to be installed as the forward's
    /// `scratch`) and the bank owner.
    ///
    /// The persistent roots and the resident bank's roots are the only device
    /// allocations; the borrowed bank views `segments` per `placements`
    /// (`plan_ep_reserve`'s deterministic placement of the prompt plan's
    /// non-persistent roots). Everything fallible that needs no device is checked
    /// before the first allocation; an allocation failure frees every earlier
    /// allocation once and returns the original error. A full dedicated
    /// prompt-row bank is never allocated, not even transiently.
    pub(crate) fn build<D: EpScratchDevice>(
        device: &mut D,
        config: &Qwen4Config,
        mode: Qwen4ForwardScratchMode,
        prompt_rows: usize,
        segments: &[EpDonorSegment],
        placements: &[EpRootPlacement],
    ) -> Result<(Qwen4GpuForwardScratch, Self), Qwen4GpuForwardError> {
        let resident_rows = EP_RESERVE_RESIDENT_ROWS;
        if prompt_rows <= resident_rows {
            return Err(invalid(format!(
                "a {prompt_rows}-row prompt does not exceed the {resident_rows}-row resident bank"
            )));
        }
        let scope = Qwen4BandScope::single(config);
        let (prompt_slots, prompt_ple) =
            Qwen4GpuForwardScratch::plan_scoped(config, prompt_rows, mode, &scope)?;
        let (resident_slots, resident_ple) =
            Qwen4GpuForwardScratch::plan_scoped(config, resident_rows, mode, &scope)?;
        check_aliases_inside_roots(prompt_rows, &prompt_slots)?;
        check_aliases_inside_roots(resident_rows, &resident_slots)?;
        for field in [ScratchField::TokenIds, ScratchField::MoeYDownGrouped] {
            let (prompt, resident) = (
                &prompt_slots[field as usize],
                &resident_slots[field as usize],
            );
            if prompt.owner.is_some() || prompt.elements == 0 || resident.owner.is_some() {
                return Err(invalid(format!(
                    "persistent {field:?} is not an owning root of the {mode:?} plan"
                )));
            }
            if prompt.dtype != resident.dtype || resident.elements > prompt.elements {
                return Err(invalid(format!(
                    "the resident {field:?} ({:?} x {}) does not fit inside the persistent prompt root ({:?} x {})",
                    resident.dtype, resident.elements, prompt.dtype, prompt.elements
                )));
            }
        }
        let prompt_roots = plan_roots(&prompt_slots)
            .ok_or_else(|| invalid("prompt scratch root sizes overflow"))?;
        validate_placements(&prompt_roots, segments, placements)?;
        let resident_roots = plan_roots(&resident_slots)
            .ok_or_else(|| invalid("resident scratch root sizes overflow"))?;
        let device_bytes = prompt_roots
            .iter()
            .filter(|root| root.2)
            .chain(resident_roots.iter().filter(|root| !root.2))
            .try_fold(0u64, |total, root| total.checked_add(root.1))
            .ok_or_else(|| invalid("EP reserve scratch device bytes overflow"))?;

        // Allocate: persistent roots first, then the resident bank's roots.
        let mut allocation = Allocation::default();
        let allocated = (|| {
            for field in [ScratchField::TokenIds, ScratchField::MoeYDownGrouped] {
                allocation.alloc(device, field, &prompt_slots[field as usize])?;
            }
            for &(field, _, persistent) in &resident_roots {
                if !persistent {
                    allocation.alloc(device, field, &resident_slots[field as usize])?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = allocated {
            allocation.rollback(device);
            return Err(error);
        }

        // Assemble the two banks from non-owning views. Nothing below allocates.
        let assembled = Self::assemble(
            mode,
            prompt_rows,
            resident_rows,
            &prompt_slots,
            &resident_slots,
            prompt_ple,
            resident_ple,
            segments,
            placements,
            &allocation.owned,
        );
        match assembled {
            Ok((resident, borrowed, host_token_ids, host_ple)) => {
                let registered_y_down = ep_owned(&allocation.owned, ScratchField::MoeYDownGrouped)
                    .buf
                    .as_ptr() as usize;
                let registered_ids = ep_owned(&allocation.owned, ScratchField::TokenIds)
                    .buf
                    .as_ptr() as usize;
                let banks = Self {
                    prompt_rows,
                    resident_rows,
                    prompt_slots,
                    resident_slots,
                    owned: allocation.owned,
                    segments: segments.to_vec(),
                    placements: placements.to_vec(),
                    registered_y_down,
                    registered_ids,
                    borrowed: Some(borrowed),
                    parked: None,
                    active: EpBankKind::Resident,
                    device_bytes,
                    host_token_ids,
                    host_ple,
                    stats: EpBankStats::default(),
                };
                Ok((resident, banks))
            }
            Err(error) => {
                allocation.rollback(device);
                Err(error)
            }
        }
    }

    /// Both banks over `owned` and the donor `segments`; verifies the stable
    /// roots are the same addresses in both.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        mode: Qwen4ForwardScratchMode,
        prompt_rows: usize,
        resident_rows: usize,
        prompt_slots: &[ScratchSlot],
        resident_slots: &[ScratchSlot],
        prompt_ple: usize,
        resident_ple: usize,
        segments: &[EpDonorSegment],
        placements: &[EpRootPlacement],
        owned: &[(ScratchField, GpuTensor)],
    ) -> Result<
        (
            Qwen4GpuForwardScratch,
            Qwen4GpuForwardScratch,
            Vec<u8>,
            Vec<u8>,
        ),
        Qwen4GpuForwardError,
    > {
        let ids = ep_owned(owned, ScratchField::TokenIds);
        let y_down = ep_owned(owned, ScratchField::MoeYDownGrouped);
        // The whole persistent token ids (the prompt rows) in both banks: the ids
        // a prompt forward uploaded are read through whichever bank is active.
        let persistent = |field: ScratchField| match field {
            ScratchField::TokenIds => ids.sub_offset(0, ids.numel()),
            // The whole registered extent: the tile exchange rejects a grouped-down
            // buffer whose size differs from its registration. Kernels use explicit
            // row counts, so the resident bank's smaller plan needs no narrower view.
            _ => y_down.sub_offset(0, y_down.numel()),
        };

        // Resident bank: its own roots are views of the owned allocations.
        let mut tensors: Vec<Option<GpuTensor>> = (0..resident_slots.len()).map(|_| None).collect();
        for (index, slot) in resident_slots.iter().enumerate() {
            let field = ScratchField::ALL[index];
            if slot.owner.is_some() || slot.elements == 0 {
                continue;
            }
            tensors[index] = Some(if ep_reserve_persistent_root(field) {
                persistent(field)
            } else {
                ep_owned(owned, field).shallow_clone()
            });
        }
        let (resident, resident_ids, resident_ple_bytes) =
            Self::finish_bank(mode, resident_rows, resident_slots, tensors, resident_ple)?;

        // Borrowed bank: every non-persistent root is a view into donor memory.
        let mut tensors: Vec<Option<GpuTensor>> = (0..prompt_slots.len()).map(|_| None).collect();
        for (index, slot) in prompt_slots.iter().enumerate() {
            let field = ScratchField::ALL[index];
            if slot.owner.is_some() || slot.elements == 0 {
                continue;
            }
            tensors[index] = Some(if ep_reserve_persistent_root(field) {
                persistent(field)
            } else {
                let placement = placements
                    .iter()
                    .find(|placement| placement.field == field)
                    .ok_or_else(|| invalid(format!("no donor placement for {field:?}")))?;
                let segment = &segments[placement.segment];
                let address = usize::try_from(placement.offset)
                    .ok()
                    .and_then(|offset| segment.address.checked_add(offset))
                    .ok_or_else(|| invalid("borrowed scratch root address overflows"))?;
                let bytes = usize::try_from(placement.bytes)
                    .map_err(|_| invalid("borrowed scratch root bytes overflow"))?;
                // SAFETY: `validate_placements` proved `[offset, offset + bytes)`
                // lies inside the donor segment, which the caller guarantees is
                // live device memory for the bank's life. The tensor is a
                // non-owning view: both banks are `ScratchBackingKind::Borrowed`
                // and the bank owner never frees a view.
                let buf = unsafe {
                    hip_bridge::DeviceBuffer::from_raw(address as *mut std::ffi::c_void, bytes)
                };
                GpuTensor {
                    buf,
                    shape: vec![slot.elements],
                    dtype: slot.dtype,
                }
            });
        }
        let (borrowed, borrowed_ids, borrowed_ple_bytes) =
            Self::finish_bank(mode, prompt_rows, prompt_slots, tensors, prompt_ple)?;

        for field in [ScratchField::TokenIds, ScratchField::MoeYDownGrouped] {
            let (a, b) = (
                resident.field_tensor(field).buf.as_ptr(),
                borrowed.field_tensor(field).buf.as_ptr(),
            );
            if a != b || a != ep_owned(owned, field).buf.as_ptr() {
                return Err(invalid(format!(
                    "persistent {field:?} differs between the resident and borrowed banks"
                )));
            }
        }
        let host_token_ids = if borrowed_ids.len() >= resident_ids.len() {
            borrowed_ids
        } else {
            resident_ids
        };
        let host_ple = if borrowed_ple_bytes.len() >= resident_ple_bytes.len() {
            borrowed_ple_bytes
        } else {
            resident_ple_bytes
        };
        Ok((resident, borrowed, host_token_ids, host_ple))
    }

    /// Bind alias views and absent fields, then assemble a `Borrowed`-backed bank.
    fn finish_bank(
        mode: Qwen4ForwardScratchMode,
        rows: usize,
        slots: &[ScratchSlot],
        mut tensors: Vec<Option<GpuTensor>>,
        max_ple_bytes: usize,
    ) -> Result<(Qwen4GpuForwardScratch, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_none() && slot.elements != 0 && tensors[index].is_none() {
                return Err(invalid(format!(
                    "no tensor bound for scratch root {:?}",
                    ScratchField::ALL[index]
                )));
            }
        }
        Qwen4GpuForwardScratch::bind_views(slots, &mut tensors);
        Ok(Qwen4GpuForwardScratch::from_tensors(
            rows,
            mode,
            None,
            ScratchBackingKind::Borrowed,
            slots.to_vec(),
            tensors,
            max_ple_bytes,
        ))
    }

    /// Prompt rows: the authority for every prompt loop and the exchange capacity.
    pub(crate) fn prompt_rows(&self) -> usize {
        self.prompt_rows
    }

    pub(crate) fn resident_rows(&self) -> usize {
        self.resident_rows
    }

    pub(crate) fn active(&self) -> EpBankKind {
        self.active
    }

    pub(crate) fn stats(&self) -> EpBankStats {
        self.stats
    }

    /// Plan (slots in [`ScratchField`] order) of a bank.
    pub(crate) fn bank_slots(&self, kind: EpBankKind) -> &[ScratchSlot] {
        match kind {
            EpBankKind::Resident => &self.resident_slots,
            EpBankKind::Borrowed => &self.prompt_slots,
        }
    }

    /// Donor segments and root placements the borrowed bank views.
    pub(crate) fn donor_layout(&self) -> (&[EpDonorSegment], &[EpRootPlacement]) {
        (&self.segments, &self.placements)
    }

    /// Device bytes this scratch allocates (persistent roots plus the resident
    /// bank's other roots); the borrowed bank allocates none.
    pub(crate) fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// The registered tile-exchange root: its address and byte size. Constant for
    /// the load's life.
    pub(crate) fn registered_grouped_down(&self) -> (usize, usize) {
        (
            self.registered_y_down,
            ep_owned(&self.owned, ScratchField::MoeYDownGrouped)
                .buf
                .size(),
        )
    }

    /// The persistent token ids' address.
    pub(crate) fn registered_token_ids(&self) -> usize {
        self.registered_ids
    }

    /// Exchange capacity of the prompt plan (never the selected bank's).
    pub(crate) fn exchange_capacity(&self) -> EpExchangeCapacity {
        let y_down = &self.prompt_slots[ScratchField::MoeYDownGrouped as usize];
        let tile_ids = self.prompt_slots[ScratchField::MoeExpertTileIds as usize].elements
            / std::mem::size_of::<i32>();
        EpExchangeCapacity {
            rows: self.prompt_rows,
            grouped_down_elements: y_down.elements,
            grouped_down_bytes: (y_down.elements * y_down.dtype.size()) as u64,
            tile_ids,
            grouped_rows: tile_ids * MOE_GROUPED_BLOCK_M,
        }
    }

    /// Host staging vectors sized for the largest bank (token-id bytes, PLE bytes).
    pub(crate) fn host_staging(&mut self) -> (&mut Vec<u8>, &mut Vec<u8>) {
        (&mut self.host_token_ids, &mut self.host_ple)
    }

    /// Move the host staging vectors out (once, into the forward's own fields).
    pub(crate) fn take_host_staging(&mut self) -> (Vec<u8>, Vec<u8>) {
        (
            std::mem::take(&mut self.host_token_ids),
            std::mem::take(&mut self.host_ple),
        )
    }

    /// The bank a call of `rows` rows will run on when it is not yet the forward's
    /// `scratch`: the parked borrowed bank for more than the resident rows while
    /// the resident bank is active. `None`: use the forward's own `scratch`.
    pub(crate) fn bank_for_rows(&self, rows: usize) -> Option<&Qwen4GpuForwardScratch> {
        if rows > self.resident_rows && self.active == EpBankKind::Resident {
            self.borrowed.as_ref()
        } else {
            None
        }
    }

    /// `scratch` still exposes the registered exchange root and token ids.
    pub(crate) fn check_stable(
        &self,
        scratch: &Qwen4GpuForwardScratch,
    ) -> Result<(), Qwen4GpuForwardError> {
        if scratch.moe_y_down_grouped.buf.as_ptr() as usize != self.registered_y_down
            || scratch.moe_y_down_grouped.buf.size() != self.registered_grouped_down().1
        {
            return Err(invalid(
                "the selected scratch bank does not expose the registered grouped-down root",
            ));
        }
        if scratch.token_ids.buf.as_ptr() as usize != self.registered_ids {
            return Err(invalid(
                "the selected scratch bank does not expose the persistent token ids",
            ));
        }
        Ok(())
    }

    /// Admit a call of `rows` rows on `route` and make its bank the forward's
    /// `scratch`. At most [`EP_RESERVE_RESIDENT_ROWS`] rows: the resident bank
    /// (swapping back from a borrowed epoch). More: the borrowed bank, only on the
    /// tile-exchange route (a wide Full-table call, or more than the prompt rows,
    /// is an error and swaps nothing). The registered grouped-down root and token
    /// ids are identical in both banks, so a swap never changes them.
    pub(crate) fn select(
        &mut self,
        scratch: &mut Qwen4GpuForwardScratch,
        rows: usize,
        route: EpBankRoute,
    ) -> Result<EpBankSelection, Qwen4GpuForwardError> {
        if rows == 0 {
            return Err(invalid("a scratch bank cannot serve zero rows"));
        }
        if rows > self.prompt_rows {
            return Err(invalid(format!(
                "{rows} rows exceed the {}-row prompt capacity of the EP reserve banks",
                self.prompt_rows
            )));
        }
        let wanted = if rows <= self.resident_rows {
            EpBankKind::Resident
        } else {
            if route == EpBankRoute::FullTable {
                return Err(invalid(format!(
                    "a {rows}-row call on the Full-table route needs more than the {}-row resident bank, \
                     and Full tables read the donor layers a borrowed bank overwrites",
                    self.resident_rows
                )));
            }
            EpBankKind::Borrowed
        };
        let swapped = if wanted == self.active {
            self.check_active(scratch)?;
            false
        } else if wanted == EpBankKind::Borrowed {
            self.swap_in(scratch)?;
            true
        } else {
            self.swap_out(scratch)?;
            true
        };
        if wanted == EpBankKind::Borrowed {
            self.stats.tiles_borrowed += 1;
        }
        Ok(EpBankSelection {
            kind: wanted,
            swapped,
        })
    }

    /// `scratch` is the bank the owner believes is active.
    fn check_active(&self, scratch: &Qwen4GpuForwardScratch) -> Result<(), Qwen4GpuForwardError> {
        let rows = match self.active {
            EpBankKind::Resident => self.resident_rows,
            EpBankKind::Borrowed => self.prompt_rows,
        };
        if scratch.max_chunk != rows {
            return Err(invalid(format!(
                "scratch holds a {}-row bank but the {:?} bank ({rows} rows) is active",
                scratch.max_chunk, self.active
            )));
        }
        Ok(())
    }

    /// Swap the borrowed bank into `scratch`, parking the resident one.
    fn swap_in(
        &mut self,
        scratch: &mut Qwen4GpuForwardScratch,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.check_active(scratch)?;
        let bank = self
            .borrowed
            .take()
            .ok_or_else(|| invalid("the borrowed scratch bank is already swapped in"))?;
        self.parked = Some(std::mem::replace(scratch, bank));
        self.active = EpBankKind::Borrowed;
        self.stats.swaps_in += 1;
        debug_assert!(self.check_stable(scratch).is_ok());
        Ok(())
    }

    /// Swap the resident bank back into `scratch`; `Ok(false)` when it already is.
    /// Whether the donors may be restored before this is the caller's protocol.
    pub(crate) fn swap_out(
        &mut self,
        scratch: &mut Qwen4GpuForwardScratch,
    ) -> Result<bool, Qwen4GpuForwardError> {
        self.check_active(scratch)?;
        if self.active == EpBankKind::Resident {
            return Ok(false);
        }
        let resident = self
            .parked
            .take()
            .ok_or_else(|| invalid("the resident scratch bank is not parked"))?;
        self.borrowed = Some(std::mem::replace(scratch, resident));
        self.active = EpBankKind::Resident;
        self.stats.swaps_out += 1;
        debug_assert!(self.check_stable(scratch).is_ok());
        Ok(true)
    }

    /// Free every device allocation exactly once and return the first error (a
    /// failed free does not stop the rest). The banks are views and free nothing;
    /// this is valid whichever bank is swapped in. The caller must have drained
    /// every peer read of the persistent roots first.
    pub(crate) fn release<D: EpScratchDevice>(
        self,
        device: &mut D,
    ) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for (_, tensor) in self.owned {
            if let Err(error) = device.release_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

/// The allocation of `field` in `owned`. Every caller names a field the build
/// allocated (`TokenIds`, `MoeYDownGrouped` or a resident root).
fn ep_owned(owned: &[(ScratchField, GpuTensor)], field: ScratchField) -> &GpuTensor {
    owned
        .iter()
        .find(|(owned_field, _)| *owned_field == field)
        .map(|(_, tensor)| tensor)
        .expect("the EP reserve build allocated this root")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use std::collections::BTreeMap;

    type Mode = Qwen4ForwardScratchMode;
    type F = ScratchField;
    const MIB: u64 = 1 << 20;
    const PROMPT: usize = 4096;

    /// Gate/up and down allocations of one EP2 donor layer (256 experts/rank).
    const GATE_UP_SEGMENT: u64 = 256 * 1_740_800;
    const DOWN_SEGMENT: u64 = 256 * 870_400;

    /// An address-simulating device: records live allocations, fails on demand,
    /// and refuses double or foreign frees.
    #[derive(Default)]
    struct FakeDevice {
        next: usize,
        live: BTreeMap<usize, u64>,
        allocs: usize,
        frees: usize,
        double_frees: usize,
        fail_alloc_at: Option<usize>,
        fail_free_at: Option<usize>,
    }

    impl FakeDevice {
        fn new() -> Self {
            Self {
                next: 0x4_0000_0000,
                ..Self::default()
            }
        }
    }

    impl EpScratchDevice for FakeDevice {
        fn alloc_zeroed(
            &mut self,
            elements: usize,
            dtype: DType,
        ) -> hip_bridge::HipResult<GpuTensor> {
            let index = self.allocs;
            if self.fail_alloc_at == Some(index) {
                return Err(hip_bridge::HipError::new(
                    hip_bridge::HIP_ERROR_OUT_OF_MEMORY,
                    "fake allocation failure",
                ));
            }
            self.allocs += 1;
            let bytes = elements * dtype.size();
            let address = self.next;
            self.next = (address + bytes + 255) / 256 * 256 + 256;
            self.live.insert(address, bytes as u64);
            Ok(GpuTensor {
                // SAFETY: a CPU-only address; never dereferenced.
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(address as *mut std::ffi::c_void, bytes)
                },
                shape: vec![elements],
                dtype,
            })
        }

        fn release_tensor(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
            let index = self.frees;
            self.frees += 1;
            if self.live.remove(&(tensor.buf.as_ptr() as usize)).is_none() {
                self.double_frees += 1;
                return Err(hip_bridge::HipError::new(
                    1,
                    "free of a non-live allocation",
                ));
            }
            if self.fail_free_at == Some(index) {
                return Err(hip_bridge::HipError::new(1, "fake free failure"));
            }
            Ok(())
        }
    }

    fn config() -> Qwen4Config {
        compact_test_config()
    }

    fn mib(bytes: u64) -> f64 {
        bytes as f64 / MIB as f64
    }

    /// `layers` EP2 donor layers as `[gate_up, down]` segments, 1 GiB apart.
    fn segments(layers: usize) -> Vec<EpDonorSegment> {
        (0..layers)
            .flat_map(|layer| {
                let base = 0x20_0000_0000usize + layer * (2 << 30);
                [
                    EpDonorSegment {
                        address: base,
                        bytes: GATE_UP_SEGMENT,
                    },
                    EpDonorSegment {
                        address: base + (1 << 30),
                        bytes: DOWN_SEGMENT,
                    },
                ]
            })
            .collect()
    }

    fn donor_roots() -> Vec<(ScratchField, u64)> {
        ep_reserve_scratch(&config(), Mode::Full, PROMPT)
            .unwrap()
            .into_iter()
            .filter(|(_, _, persistent)| !persistent)
            .map(|(field, bytes, _)| (field, bytes))
            .collect()
    }

    fn pack(layers: usize) -> Option<Vec<EpRootPlacement>> {
        let caps: Vec<u64> = segments(layers).iter().map(|s| s.bytes).collect();
        crate::ep_reserve::pack_roots(&donor_roots(), &caps).map(|placed| {
            placed
                .into_iter()
                .map(|(field, segment, offset, bytes)| EpRootPlacement {
                    field,
                    segment,
                    offset,
                    bytes,
                })
                .collect()
        })
    }

    fn build(
        device: &mut FakeDevice,
    ) -> Result<(Qwen4GpuForwardScratch, EpReserveBanks), Qwen4GpuForwardError> {
        EpReserveBanks::build(
            device,
            &config(),
            Mode::Full,
            PROMPT,
            &segments(7),
            &pack(7).expect("seven donor layers pack"),
        )
    }

    fn live_bank_bytes(banks: &EpReserveBanks) -> u64 {
        banks
            .owned
            .iter()
            .map(|(_, tensor)| tensor.buf.size() as u64)
            .sum()
    }

    #[test]
    fn adapter_reproduces_the_decision_arithmetic() {
        let config = config();
        let roots = ep_reserve_scratch(&config, Mode::Full, PROMPT).unwrap();
        // ScratchField order, strictly ascending, only the two persistent roots flagged.
        assert!(roots
            .windows(2)
            .all(|w| (w[0].0 as usize) < (w[1].0 as usize)));
        let persistent: Vec<F> = roots.iter().filter(|r| r.2).map(|r| r.0).collect();
        assert_eq!(persistent, vec![F::TokenIds, F::MoeYDownGrouped]);
        let bytes = |field: F| roots.iter().find(|r| r.0 == field).unwrap().1;
        assert_eq!(bytes(F::TokenIds), PROMPT as u64 * 4);
        // 475 MiB at 4096 rows, 125 MiB at 512 (G(4096)=48640, G(512)=12800, x2560x4).
        assert_eq!(bytes(F::MoeYDownGrouped), 475 * MIB);
        let small = ep_reserve_scratch(&config, Mode::Full, EP_RESERVE_RESIDENT_ROWS).unwrap();
        assert_eq!(
            small.iter().find(|r| r.0 == F::MoeYDownGrouped).unwrap().1,
            125 * MIB
        );
        // Roots are exactly the owning allocations of the production plan.
        let total: u64 = roots.iter().map(|r| r.1).sum();
        assert_eq!(
            total,
            Qwen4GpuForwardScratch::device_bytes(&config, PROMPT, Mode::Full).unwrap()
        );
        // Reconciliation with DECISION.md §4.2 (independent integer arithmetic).
        let without_ids = total - bytes(F::TokenIds);
        let donor_packed = without_ids - bytes(F::MoeYDownGrouped);
        assert!(
            (mib(without_ids) - 4328.238).abs() < 0.001,
            "{}",
            mib(without_ids)
        );
        assert!(
            (mib(donor_packed) - 3853.238).abs() < 0.001,
            "{}",
            mib(donor_packed)
        );
        // Spec buffers are separate persistent allocations, not roots.
        assert_eq!(
            ep_reserve_spec_bytes(&config, PROMPT).unwrap(),
            super::super::qwen4_spec_buffer_bytes(&config, PROMPT)
                .unwrap()
                .iter()
                .sum::<u64>()
        );
        // The largest donor-packed root fits a gate/up segment (400 MiB < 425 MiB), and
        // the persistent grouped-down root fits neither segment kind.
        assert_eq!(bytes(F::MoeDownExpanded), 400 * MIB);
        assert!(bytes(F::MoeYDownGrouped) > GATE_UP_SEGMENT);
        assert!(bytes(F::MoeDownExpanded) <= GATE_UP_SEGMENT);
        // A refused geometry is None, not a partial list.
        assert!(ep_reserve_scratch(&config, Mode::Full, 0).is_none());
    }

    #[test]
    fn six_donor_layers_refuse_and_seven_pack() {
        assert!(pack(6).is_none(), "six layers hold only 3825 MiB");
        let six: u64 = segments(6).iter().map(|s| s.bytes).sum();
        assert_eq!(six, 3825 * MIB);
        let placements = pack(7).expect("seven layers pack");
        let roots = ep_reserve_scratch(&config(), Mode::Full, PROMPT).unwrap();
        validate_placements(&roots, &segments(7), &placements).unwrap();
        // No root crosses a segment boundary; ~609 MiB of the 4462.5 MiB is spare.
        let used: u64 = placements.iter().map(|p| p.bytes).sum();
        let capacity: u64 = segments(7).iter().map(|s| s.bytes).sum();
        assert!(capacity - used > 600 * MIB && capacity - used < 620 * MIB);
        // Deterministic.
        assert_eq!(placements, pack(7).unwrap());
    }

    #[test]
    fn placements_that_straddle_overlap_or_misplace_are_refused() {
        let roots = ep_reserve_scratch(&config(), Mode::Full, PROMPT).unwrap();
        let segs = segments(7);
        let good = pack(7).unwrap();
        validate_placements(&roots, &segs, &good).unwrap();
        let broken = |edit: &dyn Fn(&mut Vec<EpRootPlacement>)| {
            let mut placements = good.clone();
            edit(&mut placements);
            validate_placements(&roots, &segs, &placements)
                .unwrap_err()
                .to_string()
        };
        // Straddle: a root pushed past the end of its segment.
        let last = good.len() - 1;
        assert!(
            broken(&|p| p[last].offset = segs[p[last].segment].bytes - 256).contains("straddle")
        );
        // Overlap inside a segment: the later of two roots of one segment moves onto the earlier.
        let (lo, hi) = (0..good.len())
            .flat_map(|i| (0..good.len()).map(move |j| (i, j)))
            .find(|&(i, j)| good[i].segment == good[j].segment && good[i].offset < good[j].offset)
            .expect("seven donor layers hold several roots in one segment");
        let target = good[lo].offset;
        assert!(broken(&|p| p[hi].offset = target).contains("overlaps"));
        // Unaligned start, unknown segment, wrong size, duplicate, missing, persistent root.
        assert!(broken(&|p| p[0].offset += 1).contains("aligned"));
        assert!(broken(&|p| p[0].segment = 99).contains("donor segment"));
        assert!(broken(&|p| p[0].bytes += 256).contains("bytes"));
        assert!(broken(&|p| p[1].field = p[0].field).contains("twice"));
        assert!(broken(&|p| {
            p.pop();
        })
        .contains("placements"));
        assert!(broken(&|p| p[0].field = F::MoeYDownGrouped).contains("persistent"));
        assert!(broken(&|p| p[0].field = F::TokenIds).contains("persistent"));
        // Segments that are unaligned or overlap in the address space.
        let mut unaligned = segs.clone();
        unaligned[0].address += 8;
        assert!(validate_placements(&roots, &unaligned, &good).is_err());
        let mut aliased = segs.clone();
        aliased[1].address = aliased[0].address + 4096;
        assert!(validate_placements(&roots, &aliased, &good).is_err());
        // Arithmetic overflow fails closed.
        let mut huge = good.clone();
        huge[0].offset = u64::MAX - 255;
        assert!(validate_placements(&roots, &segs, &huge).is_err());
    }

    #[test]
    fn every_alias_stays_inside_its_root() {
        let config = config();
        let scope = Qwen4BandScope::single(&config);
        for mode in [Mode::Full, Mode::FnSymGfx1201] {
            for rows in [EP_RESERVE_RESIDENT_ROWS, 1024, PROMPT] {
                let (slots, _) =
                    Qwen4GpuForwardScratch::plan_scoped(&config, rows, mode, &scope).unwrap();
                check_aliases_inside_roots(rows, &slots).unwrap();
            }
        }
        // A view past its root is refused.
        let (mut slots, _) =
            Qwen4GpuForwardScratch::plan_scoped(&config, 1024, Mode::FnSymGfx1201, &scope).unwrap();
        let (index, _) = slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.owner.is_some() && slot.elements != 0)
            .expect("the compact plan has alias views");
        slots[index].elements = usize::MAX / 8;
        assert!(check_aliases_inside_roots(1024, &slots).is_err());
    }

    #[test]
    fn banks_share_one_exchange_root_and_one_token_id_owner() {
        let mut device = FakeDevice::new();
        let (resident, banks) = build(&mut device).unwrap();
        let borrowed = banks.borrowed.as_ref().unwrap();
        let (address, bytes) = banks.registered_grouped_down();
        // True 4096 capacity: the persistent root is the whole 475 MiB.
        assert_eq!(bytes as u64, 475 * MIB);
        assert_eq!(banks.exchange_capacity().grouped_down_bytes, 475 * MIB);
        assert_eq!(banks.exchange_capacity().grouped_rows, 48_640);
        assert_eq!(banks.exchange_capacity().rows, PROMPT);
        // External root pointers are identical in both banks.
        for scratch in [&resident, borrowed] {
            banks.check_stable(scratch).unwrap();
            assert_eq!(scratch.moe_y_down_grouped.buf.as_ptr() as usize, address);
            assert_eq!(
                scratch.token_ids.buf.as_ptr() as usize,
                banks.registered_token_ids()
            );
            // Full token-id upload capacity in both banks.
            assert_eq!(scratch.token_ids.numel(), PROMPT * 4);
        }
        // Both banks expose the whole registered extent (the tile exchange rejects any
        // other size); the resident plan's smaller grouped-down need is a row count.
        assert_eq!(resident.moe_y_down_grouped.buf.size(), bytes);
        assert_eq!(borrowed.moe_y_down_grouped.buf.size(), bytes);
        assert_eq!(
            resident.moe_y_down_grouped.numel(),
            borrowed.moe_y_down_grouped.numel()
        );
        assert_eq!(resident.max_chunk, EP_RESERVE_RESIDENT_ROWS);
        assert_eq!(borrowed.max_chunk, PROMPT);
        // Both banks are non-owning: neither would free anything.
        assert!(resident.backing == ScratchBackingKind::Borrowed);
        assert!(borrowed.backing == ScratchBackingKind::Borrowed);
        // Allocated: the persistent roots and the resident bank's other roots, nothing prompt-sized.
        assert_eq!(live_bank_bytes(&banks), banks.device_bytes());
        assert_eq!(
            device.live.values().sum::<u64>(),
            banks.device_bytes(),
            "the fake device sees exactly the charged bytes"
        );
        let full_prompt =
            Qwen4GpuForwardScratch::device_bytes(&config(), PROMPT, Mode::Full).unwrap();
        assert!(banks.device_bytes() < full_prompt / 2);
        // Resident roots are the owned allocations, not the donor memory.
        for &(field, _, persistent) in
            &ep_reserve_scratch(&config(), Mode::Full, EP_RESERVE_RESIDENT_ROWS).unwrap()
        {
            if persistent {
                continue;
            }
            let ptr = resident.field_tensor(field).buf.as_ptr() as usize;
            assert_eq!(ptr, ep_owned(&banks.owned, field).buf.as_ptr() as usize);
        }
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn borrowed_roots_sit_at_their_placed_donor_addresses() {
        let mut device = FakeDevice::new();
        let (_resident, banks) = build(&mut device).unwrap();
        let borrowed = banks.borrowed.as_ref().unwrap();
        let (segs, placements) = banks.donor_layout();
        let mut spans: Vec<(usize, usize, F)> = Vec::new();
        for placement in placements {
            let tensor = borrowed.field_tensor(placement.field);
            let address = tensor.buf.as_ptr() as usize;
            let segment = &segs[placement.segment];
            assert_eq!(address, segment.address + placement.offset as usize);
            assert_eq!(tensor.buf.size() as u64, placement.bytes);
            // Inside exactly one segment, never touching owned memory.
            assert!(address + tensor.buf.size() <= segment.address + segment.bytes as usize);
            assert!(!device.live.contains_key(&address));
            spans.push((address, address + tensor.buf.size(), placement.field));
        }
        spans.sort_unstable_by_key(|s| s.0);
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "roots overlap");
        // Each root has exactly the plan's element count and dtype.
        for (index, slot) in banks.prompt_slots.iter().enumerate() {
            let tensor = borrowed.field_tensor(ScratchField::ALL[index]);
            assert_eq!(tensor.dtype, slot.dtype);
            if !ep_reserve_persistent_root(ScratchField::ALL[index]) {
                assert_eq!(tensor.numel(), slot.elements);
            }
        }
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn select_matrix_and_swaps_preserve_the_registered_root() {
        let mut device = FakeDevice::new();
        let (mut scratch, mut banks) = build(&mut device).unwrap();
        let address = banks.registered_grouped_down();
        let ids = banks.registered_token_ids();
        let (ids_ptr, ple_ptr) = {
            let (a, b) = banks.host_staging();
            (a.as_ptr(), b.as_ptr())
        };
        let (ids_len, ple_len) = {
            let (a, b) = banks.host_staging();
            (a.len(), b.len())
        };
        assert_eq!(ids_len, PROMPT * 4, "staging is sized for the largest bank");
        let allocs = device.allocs;
        for (rows, borrowed) in [
            (1, false),
            (8, false),
            (9, false),
            (511, false),
            (512, false),
            (513, true),
            (1024, true),
            (2047, true),
            (2048, true),
            (4095, true),
            (4096, true),
            (12, false),
            (4096, true),
            (512, false),
        ] {
            let selection = banks
                .select(&mut scratch, rows, EpBankRoute::TileExchange)
                .unwrap();
            assert_eq!(
                selection.kind == EpBankKind::Borrowed,
                borrowed,
                "{rows} rows"
            );
            assert_eq!(banks.active() == EpBankKind::Borrowed, borrowed);
            assert_eq!(scratch.max_chunk, if borrowed { PROMPT } else { 512 });
            assert!(scratch.max_chunk >= rows);
            banks.check_stable(&scratch).unwrap();
            assert_eq!(banks.registered_grouped_down(), address);
            assert_eq!(banks.registered_token_ids(), ids);
        }
        // [4096, 4096] stays borrowed with no swap in between.
        banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        let before = banks.stats();
        let again = banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        assert!(!again.swapped);
        assert_eq!(banks.stats().swaps_in, before.swaps_in);
        assert_eq!(banks.stats().tiles_borrowed, before.tiles_borrowed + 1);
        // swap_out restores the resident bank; a second one is a no-op.
        assert!(banks.swap_out(&mut scratch).unwrap());
        assert!(!banks.swap_out(&mut scratch).unwrap());
        assert_eq!(scratch.max_chunk, 512);
        banks.check_stable(&scratch).unwrap();
        let stats = banks.stats();
        assert_eq!(stats.swaps_in, stats.swaps_out);
        // Bank switching allocated and grew nothing.
        assert_eq!(device.allocs, allocs);
        let (a, b) = banks.host_staging();
        assert_eq!((a.as_ptr(), b.as_ptr()), (ids_ptr, ple_ptr));
        assert_eq!((a.len(), b.len()), (ids_len, ple_len));
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn full_table_and_oversized_calls_fail_closed_without_swapping() {
        let mut device = FakeDevice::new();
        let (mut scratch, mut banks) = build(&mut device).unwrap();
        // A wide call on the Full-table route cannot borrow; nothing swaps.
        for rows in [513, 4096] {
            assert!(banks
                .select(&mut scratch, rows, EpBankRoute::FullTable)
                .is_err());
            assert_eq!(banks.active(), EpBankKind::Resident);
            assert_eq!(scratch.max_chunk, 512);
        }
        // A resident-sized Full-table call uses the resident bank.
        let ok = banks
            .select(&mut scratch, 12, EpBankRoute::FullTable)
            .unwrap();
        assert_eq!(ok.kind, EpBankKind::Resident);
        // Zero rows and more than the prompt rows are errors on any route.
        for rows in [0, PROMPT + 1] {
            assert!(banks
                .select(&mut scratch, rows, EpBankRoute::TileExchange)
                .is_err());
        }
        // A tail after a borrowed epoch swaps back; Full-table tail runs resident.
        banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        let tail = banks
            .select(&mut scratch, 12, EpBankRoute::FullTable)
            .unwrap();
        assert_eq!((tail.kind, tail.swapped), (EpBankKind::Resident, true));
        // A wide Full-table call during a borrowed epoch is an error and leaves it active.
        banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        assert!(banks
            .select(&mut scratch, 4096, EpBankRoute::FullTable)
            .is_err());
        assert_eq!(banks.active(), EpBankKind::Borrowed);
        assert_eq!(scratch.max_chunk, PROMPT);
        // A mismatched `scratch` is refused rather than corrupting the swap.
        let mut other_device = FakeDevice::new();
        let (mut other, _other_banks) = build(&mut other_device).unwrap();
        assert!(banks.swap_out(&mut other).is_err());
        assert_eq!(banks.active(), EpBankKind::Borrowed);
        assert!(banks.swap_out(&mut scratch).unwrap());
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn release_frees_every_allocation_exactly_once() {
        let mut device = FakeDevice::new();
        let (mut scratch, mut banks) = build(&mut device).unwrap();
        let expected = device.allocs;
        assert_eq!(device.live.len(), expected);
        // Valid with the borrowed bank swapped in.
        banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        assert!(banks.release(&mut device).is_none());
        assert_eq!(device.frees, expected);
        assert_eq!(device.double_frees, 0);
        assert!(device.live.is_empty());
    }

    #[test]
    fn a_failed_free_does_not_stop_the_rest() {
        let mut probe = FakeDevice::new();
        let (_s, banks) = build(&mut probe).unwrap();
        let total = probe.allocs;
        assert!(banks.release(&mut probe).is_none());
        for fail in [0, 1, total / 2, total - 1] {
            let mut device = FakeDevice::new();
            let (_s, banks) = build(&mut device).unwrap();
            device.fail_free_at = Some(fail);
            assert!(banks.release(&mut device).is_some());
            assert_eq!(
                device.frees, total,
                "every allocation is still attempted once"
            );
            assert!(device.live.is_empty());
            assert_eq!(device.double_frees, 0);
        }
    }

    #[test]
    fn construction_failure_at_every_allocation_frees_exactly_what_was_allocated() {
        let mut probe = FakeDevice::new();
        let (_s, banks) = build(&mut probe).unwrap();
        let total = probe.allocs;
        // TokenIds + MoeYDownGrouped + the resident bank's other roots.
        assert!(total > 40, "{total}");
        assert!(banks.release(&mut probe).is_none());
        for fail in 0..total {
            let mut device = FakeDevice::new();
            device.fail_alloc_at = Some(fail);
            assert!(build(&mut device).is_err(), "failure at allocation {fail}");
            assert_eq!(device.allocs, fail, "allocations before the failure");
            assert_eq!(device.frees, fail, "each earlier allocation freed once");
            assert!(device.live.is_empty(), "failure at {fail} leaked");
            assert_eq!(device.double_frees, 0);
        }
    }

    #[test]
    fn invalid_inputs_are_refused_before_any_allocation() {
        let cfg = config();
        let good = pack(7).unwrap();
        let segs = segments(7);
        let run = |mode: Mode, rows: usize, segs: &[EpDonorSegment], p: &[EpRootPlacement]| {
            let mut device = FakeDevice::new();
            let result = EpReserveBanks::build(&mut device, &cfg, mode, rows, segs, p);
            assert_eq!(device.allocs, 0, "nothing allocated before validation");
            assert!(device.live.is_empty());
            result.is_err()
        };
        // The compact plan aliases the grouped-down root: it is not persistent-capable.
        assert!(run(Mode::FnSymGfx1201, PROMPT, &segs, &good));
        // A prompt that does not exceed the resident bank has nothing to borrow.
        assert!(run(Mode::Full, EP_RESERVE_RESIDENT_ROWS, &segs, &good));
        assert!(run(Mode::Full, 0, &segs, &good));
        // Missing / insufficient donor memory.
        assert!(run(Mode::Full, PROMPT, &segs, &good[..good.len() - 1]));
        assert!(run(Mode::Full, PROMPT, &segments(6), &good));
        // And a valid build succeeds.
        let mut device = FakeDevice::new();
        let (_s, banks) =
            EpReserveBanks::build(&mut device, &cfg, Mode::Full, PROMPT, &segs, &good).unwrap();
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn bank_for_rows_names_only_the_parked_borrowed_bank() {
        let mut device = FakeDevice::new();
        let (mut scratch, mut banks) = build(&mut device).unwrap();
        for rows in [1, 512] {
            assert!(banks.bank_for_rows(rows).is_none());
        }
        for rows in [513, 4096] {
            let bank = banks.bank_for_rows(rows).expect("parked borrowed bank");
            assert_eq!(bank.max_chunk, PROMPT);
        }
        // Borrowed already active: the forward's own scratch is the bank.
        banks
            .select(&mut scratch, 4096, EpBankRoute::TileExchange)
            .unwrap();
        for rows in [1, 512, 513, 4096] {
            assert!(banks.bank_for_rows(rows).is_none());
        }
        banks.swap_out(&mut scratch).unwrap();
        assert_eq!(banks.bank_for_rows(513).unwrap().max_chunk, PROMPT);
        assert!(banks.release(&mut device).is_none());
    }

    #[test]
    fn token_upload_survives_swaps() {
        let mut device = FakeDevice::new();
        let (mut scratch, mut banks) = build(&mut device).unwrap();
        // The bytes a prompt forward uploads are read through the bank active
        // afterwards: one persistent tensor, the same address and extent in
        // every bank and after every swap.
        let ids = (banks.registered_token_ids(), scratch.token_ids.buf.size());
        for rows in [4096, 512, 2048, 12, 4096] {
            banks
                .select(&mut scratch, rows, EpBankRoute::TileExchange)
                .unwrap();
            assert_eq!(
                (
                    scratch.token_ids.buf.as_ptr() as usize,
                    scratch.token_ids.buf.size()
                ),
                ids
            );
            // The upload range of `rows` ids fits the persistent extent.
            assert!(rows * 4 <= scratch.token_ids.buf.size());
        }
        banks.swap_out(&mut scratch).unwrap();
        assert!(banks.release(&mut device).is_none());
    }
}
