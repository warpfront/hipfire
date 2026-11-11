// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The EP reserve-sharing donor loan (`HIPFIRE_EP_RESERVE_DECOUPLE=1`).
//!
//! While a 513..=4096-row prompt forward runs, its borrowed scratch bank lives
//! inside the weights of a fixed suffix of whole-layer donors
//! (`Q/ep_reserve.rs`: `EpReservePlan::{segments, touched}`). The scratch
//! overwrites only the *touched extents* of those donor tensors (the first
//! `len` bytes of each used segment); everything else stays valid.
//!
//! This file is the **local** memory owner of that loan:
//!
//! * pinned host mirrors of the donor tensors' *actual loaded bytes*, read
//!   back once at load (the `Qwen4ExpertStage` D2H mirror precedent), whole
//!   tensor each, because the borrowed expert tables and S2 need complete
//!   tensors ([`EpReserveLoan::mirror`] exposes them);
//! * the restore-completion event;
//! * a [`LoanController`] (`decode_share_loan.rs`, reused unchanged, no
//!   shadows, `sync_restore = true`: a successful prompt restores before it
//!   returns) driven through [`ExtentLoanOps`], which maps the controller's
//!   *virtual dirty prefix* over the ordered touched extents. A virtual offset
//!   is never added to a donor address: `[v, v + n)` is split at every extent
//!   boundary into per-tensor pieces ([`ExtentMap::pieces`]), and zero fill,
//!   asynchronous restore, blocking recovery and the verify oracle each
//!   touch exactly those pieces.
//!
//! It owns no tier lease and implements no collective barrier. Cross-rank
//! rendezvous (before the first overwrite, after the last restore), mesh abort
//! and request state belong to `Qwen4EpMesh`/`RankPool`; a local restore or
//! recovery proves only that THIS rank's donor bytes are whole.
//!
//! The device side is the small [`LoanDevice`] trait (one real operation per
//! method, per piece); [`GpuLoanDevice`] is the real implementation over one
//! `Gpu`, and every transition above it is unit-tested on the CPU with a
//! recording fake that also simulates the donor/mirror bytes.
//!
//! Release is typed: [`EpReserveLoan::release`] needs the
//! [`EpLoanTeardownPermit`] that only a successful [`EpReserveLoan::teardown`]
//! returns, so a loan whose restore could not be proven complete is never
//! released under a possibly pending DMA (the caller leaks it; the struct has
//! no `Drop`, like every other pinned/device owner here).

use super::*;
use crate::decode_share_loan::{LoanController, LoanOps};
use crate::ep_reserve::{EpDonorSlot, EpReserveTouchedExtent};

/// Largest device read of the verify oracle at once.
const VERIFY_CHUNK_BYTES: usize = 32 << 20;

/// One contiguous byte range of one donor tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LoanPiece {
    /// Index of the donor tensor in the loan's region list.
    pub region: usize,
    /// Byte offset inside that tensor.
    pub offset: u64,
    pub len: u64,
}

/// The ordered touched extents of the donor tensors: the geometry that turns
/// the controller's virtual byte offsets into per-tensor pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtentMap {
    /// Full byte size of each donor tensor (and of its mirror).
    region_bytes: Vec<u64>,
    /// Touched extents in plan order; consecutive contiguous extents of one
    /// tensor are merged.
    extents: Vec<LoanPiece>,
    /// Virtual start of each extent.
    starts: Vec<u64>,
    total: u64,
}

impl ExtentMap {
    /// `region_bytes[i]` is the byte size of donor tensor `i`; `extents` are
    /// the touched ranges in virtual order. Every extent must be non-empty and
    /// inside its tensor, and no two may overlap (a byte would be counted twice
    /// in the virtual prefix).
    pub(crate) fn new(region_bytes: &[u64], extents: &[LoanPiece]) -> Result<Self, String> {
        let mut merged: Vec<LoanPiece> = Vec::with_capacity(extents.len());
        for (index, extent) in extents.iter().enumerate() {
            let Some(&size) = region_bytes.get(extent.region) else {
                return Err(format!(
                    "touched extent {index} names donor tensor {} of {}",
                    extent.region,
                    region_bytes.len()
                ));
            };
            if extent.len == 0 {
                return Err(format!("touched extent {index} is empty"));
            }
            if extent
                .offset
                .checked_add(extent.len)
                .is_none_or(|end| end > size)
            {
                return Err(format!(
                    "touched extent {index} [{}, +{}) exceeds the {size}-byte donor tensor {}",
                    extent.offset, extent.len, extent.region
                ));
            }
            let contiguous = merged.last().is_some_and(|last| {
                last.region == extent.region && last.offset + last.len == extent.offset
            });
            if contiguous {
                if let Some(last) = merged.last_mut() {
                    last.len += extent.len;
                }
            } else {
                merged.push(*extent);
            }
        }
        // Overlap check over the merged list, per tensor.
        let mut by_region: Vec<(usize, u64, u64)> = merged
            .iter()
            .map(|extent| (extent.region, extent.offset, extent.len))
            .collect();
        by_region.sort_unstable();
        for pair in by_region.windows(2) {
            let ((r0, o0, l0), (r1, o1, _)) = (pair[0], pair[1]);
            if r0 == r1 && o0 + l0 > o1 {
                return Err(format!(
                    "touched extents overlap inside donor tensor {r0} at byte {o1}"
                ));
            }
        }
        let mut starts = Vec::with_capacity(merged.len());
        let mut total = 0u64;
        for extent in &merged {
            starts.push(total);
            total = total
                .checked_add(extent.len)
                .ok_or_else(|| "touched extent total overflows".to_string())?;
        }
        Ok(Self {
            region_bytes: region_bytes.to_vec(),
            extents: merged,
            starts,
            total,
        })
    }

    /// Bytes of the virtual dirty space: the sum of the touched extents.
    pub(crate) fn total_bytes(&self) -> u64 {
        self.total
    }

    pub(crate) fn region_count(&self) -> usize {
        self.region_bytes.len()
    }

    /// Full byte size of donor tensor `region` (its mirror's size).
    pub(crate) fn region_bytes(&self, region: usize) -> Option<u64> {
        self.region_bytes.get(region).copied()
    }

    /// The merged touched extents in virtual order.
    pub(crate) fn extents(&self) -> &[LoanPiece] {
        &self.extents
    }

    /// The per-tensor pieces covering the virtual range `[offset, offset + len)`.
    pub(crate) fn pieces(&self, offset: u64, len: u64) -> Result<Pieces<'_>, String> {
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= self.total)
            .ok_or_else(|| {
                format!(
                    "virtual donor range [{offset}, +{len}) exceeds the {} touched bytes",
                    self.total
                )
            })?;
        let (index, within) = if len == 0 {
            (self.extents.len(), 0)
        } else {
            // The last extent whose virtual start is <= offset.
            let index = self.starts.partition_point(|&start| start <= offset) - 1;
            (index, offset - self.starts[index])
        };
        Ok(Pieces {
            map: self,
            index,
            within,
            remaining: end - offset,
        })
    }

    /// The pieces of the first `bytes` virtual bytes (a dirty prefix).
    pub(crate) fn prefix(&self, bytes: u64) -> Result<Pieces<'_>, String> {
        self.pieces(0, bytes)
    }
}

/// Iterator over the pieces of one virtual range; allocation-free.
pub(crate) struct Pieces<'a> {
    map: &'a ExtentMap,
    index: usize,
    within: u64,
    remaining: u64,
}

impl Iterator for Pieces<'_> {
    type Item = LoanPiece;

    fn next(&mut self) -> Option<LoanPiece> {
        if self.remaining == 0 {
            return None;
        }
        let extent = self.map.extents[self.index];
        let take = (extent.len - self.within).min(self.remaining);
        let piece = LoanPiece {
            region: extent.region,
            offset: extent.offset + self.within,
            len: take,
        };
        self.remaining -= take;
        self.within += take;
        if self.within == extent.len {
            self.index += 1;
            self.within = 0;
        }
        Some(piece)
    }
}

/// Resolve the plan's `(layer, slot)` touched extents against the loaded
/// donor tensors `(layer, slot, actual bytes)` (region index = position).
pub(crate) fn build_map(
    donors: &[(usize, EpDonorSlot, u64)],
    touched: &[EpReserveTouchedExtent],
) -> Result<ExtentMap, String> {
    for (index, donor) in donors.iter().enumerate() {
        if donors[..index]
            .iter()
            .any(|seen| seen.0 == donor.0 && seen.1 == donor.1)
        {
            return Err(format!(
                "donor tensor (layer {}, {:?}) is listed twice",
                donor.0, donor.1
            ));
        }
    }
    let region_bytes: Vec<u64> = donors.iter().map(|donor| donor.2).collect();
    let mut extents = Vec::with_capacity(touched.len());
    for extent in touched {
        let region = donors
            .iter()
            .position(|donor| donor.0 == extent.layer && donor.1 == extent.slot)
            .ok_or_else(|| {
                format!(
                    "touched extent names layer {} {:?}, which is not a loaded donor tensor",
                    extent.layer, extent.slot
                )
            })?;
        extents.push(LoanPiece {
            region,
            offset: extent.offset,
            len: extent.len,
        });
    }
    ExtentMap::new(&region_bytes, &extents)
}

/// Device effects of the loan, per tensor piece. Every method is one real
/// operation; errors are plain text.
pub(crate) trait LoanDevice {
    fn capturing(&mut self) -> Result<bool, String>;
    /// Enqueue a zero fill of `piece` on the compute stream.
    fn zero(&mut self, piece: LoanPiece) -> Result<(), String>;
    /// Enqueue the H2D restore of `piece` from its mirror on the compute stream.
    fn copy_async(&mut self, piece: LoanPiece) -> Result<(), String>;
    /// Record the restore-completion event on the compute stream.
    fn record_done(&mut self) -> Result<(), String>;
    /// Host-wait the completion event and check the async error state.
    fn wait_done(&mut self) -> Result<(), String>;
    fn device_sync(&mut self) -> Result<(), String>;
    /// Blocking H2D restore of `piece`.
    fn copy_blocking(&mut self, piece: LoanPiece) -> Result<(), String>;
    /// Compare the device bytes of `piece` with its mirror; `Err` names the
    /// first difference.
    fn verify_piece(&mut self, piece: LoanPiece) -> Result<(), String>;
}

/// Receipt counters of the piece-level traffic (successful operations only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EpLoanStats {
    pub zero_pieces: u64,
    pub zeroed_bytes: u64,
    pub async_pieces: u64,
    pub async_restore_bytes: u64,
    pub blocking_pieces: u64,
    pub blocking_restore_bytes: u64,
    pub verified_bytes: u64,
}

/// [`LoanOps`] over the touched extents: the controller's virtual offsets
/// become per-tensor pieces for a [`LoanDevice`].
pub(crate) struct ExtentLoanOps<'a, D: LoanDevice + ?Sized> {
    pub map: &'a ExtentMap,
    pub device: &'a mut D,
    pub stats: &'a mut EpLoanStats,
    /// Run the byte oracle after every restore.
    pub verify: bool,
}

impl<D: LoanDevice + ?Sized> LoanOps for ExtentLoanOps<'_, D> {
    fn capturing(&mut self) -> Result<bool, String> {
        self.device.capturing()
    }

    fn memset_zero(&mut self, offset: u64, bytes: u64) -> Result<(), String> {
        for piece in self.map.pieces(offset, bytes)? {
            self.device.zero(piece)?;
            self.stats.zero_pieces += 1;
            self.stats.zeroed_bytes += piece.len;
        }
        Ok(())
    }

    fn enqueue_restore(&mut self, bytes: u64) -> Result<(), String> {
        for piece in self.map.prefix(bytes)? {
            self.device.copy_async(piece)?;
            self.stats.async_pieces += 1;
            self.stats.async_restore_bytes += piece.len;
        }
        self.device.record_done()
    }

    fn wait_restore(&mut self) -> Result<(), String> {
        self.device.wait_done()
    }

    fn device_sync(&mut self) -> Result<(), String> {
        self.device.device_sync()
    }

    fn blocking_restore(&mut self, bytes: u64) -> Result<(), String> {
        for piece in self.map.prefix(bytes)? {
            self.device.copy_blocking(piece)?;
            self.stats.blocking_pieces += 1;
            self.stats.blocking_restore_bytes += piece.len;
        }
        Ok(())
    }

    fn verify(&mut self, bytes: u64) -> Result<(), String> {
        if !self.verify {
            return Ok(());
        }
        for piece in self.map.prefix(bytes)? {
            self.device.verify_piece(piece)?;
            self.stats.verified_bytes += piece.len;
        }
        Ok(())
    }
}

/// Proof that a teardown restore completed (or a recovery copy completed
/// after a drain): the donors are whole and the restore resources may be
/// released. Only [`EpReserveLoan::teardown`] mints it.
#[derive(Debug)]
pub(crate) struct EpLoanTeardownPermit(());

/// The controller's teardown as a permit: `Err` means the restore could not
/// be proven complete and the owners must be retained, not freed.
pub(crate) fn teardown_permit(
    controller: &mut LoanController,
    ops: &mut dyn LoanOps,
) -> Result<EpLoanTeardownPermit, String> {
    controller.teardown(ops).map(|()| EpLoanTeardownPermit(()))
}

/// One donor tensor of the loan: the non-owning device view and its pinned
/// mirror's CPU address.
#[derive(Clone, Copy, Debug)]
struct LoanRegion {
    layer: usize,
    slot: EpDonorSlot,
    /// Device address of the donor tensor (owned by `Qwen4Weights`).
    base: usize,
    bytes: usize,
    /// Pinned CPU address of the mirror of those `bytes`.
    mirror_cpu: usize,
}

/// One donor tensor as the loader owns it.
pub(crate) struct EpLoanDonor<'a> {
    pub layer: usize,
    pub slot: EpDonorSlot,
    /// The owning device tensor (never a host-mapped one); the mirror covers
    /// all of `owner.buf.size()`.
    pub owner: &'a GpuTensor,
}

/// A complete, immutable source view of one donor tensor's bytes: the pinned
/// mirror (the owning allocation) for S2 / borrowed-runtime construction.
/// The mirror is the host-mapped owner; the donor's device address is
/// reported so a borrowed table can name the donor slot it replaces, but a
/// device view is never re-tagged as host memory.
#[derive(Clone, Copy)]
pub(crate) struct EpLoanMirror<'a> {
    pub layer: usize,
    pub slot: EpDonorSlot,
    /// The pinned host-mapped owner of the mirror bytes.
    pub tensor: &'a GpuTensor,
    /// Full mirror length (the donor tensor's loaded byte size).
    pub bytes: usize,
    /// Device address of the donor tensor the mirror restores.
    pub donor_device_base: usize,
}

/// The device operations over one `Gpu`.
struct GpuLoanDevice<'a> {
    gpu: &'a mut Gpu,
    regions: &'a [LoanRegion],
    done: &'a hip_bridge::Event,
}

/// `(region, byte offset, byte length)` of `piece`, checked against its tensor.
fn checked_span(
    regions: &[LoanRegion],
    piece: LoanPiece,
) -> Result<(&LoanRegion, usize, usize), String> {
    let region = regions
        .get(piece.region)
        .ok_or_else(|| format!("donor piece names unknown tensor {}", piece.region))?;
    let offset = usize::try_from(piece.offset).map_err(|_| "donor offset overflows".to_string())?;
    let len = usize::try_from(piece.len).map_err(|_| "donor length overflows".to_string())?;
    if offset.checked_add(len).is_none_or(|end| end > region.bytes) {
        return Err(format!(
            "donor range [{offset}, +{len}) exceeds the {}-byte tensor of layer {} {:?}",
            region.bytes, region.layer, region.slot
        ));
    }
    Ok((region, offset, len))
}

impl GpuLoanDevice<'_> {
    fn compute(&self) -> Result<&hip_bridge::Stream, String> {
        self.gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "EP reserve loan needs the compute stream".to_string())
    }

    /// A non-owning view of the donor range and the matching mirror bytes.
    fn view(
        &self,
        piece: LoanPiece,
    ) -> Result<(hip_bridge::DeviceBuffer, &'static [u8], &LoanRegion), String> {
        let (region, offset, len) = checked_span(self.regions, piece)?;
        // SAFETY: the range lies inside the donor tensor `Qwen4Weights` owns
        // for this forward's life; the buffer is a `Borrowed` alias that no
        // free path receives.
        let dst = unsafe {
            hip_bridge::DeviceBuffer::from_raw((region.base + offset) as *mut std::ffi::c_void, len)
        };
        // SAFETY: `mirror_cpu` designates `region.bytes` bytes of the pinned
        // host owner this loan keeps alive until `release`; nothing writes the
        // mirror after materialization. The slice never outlives the
        // enclosing operation.
        let src =
            unsafe { std::slice::from_raw_parts((region.mirror_cpu + offset) as *const u8, len) };
        Ok((dst, src, region))
    }
}

impl LoanDevice for GpuLoanDevice<'_> {
    fn capturing(&mut self) -> Result<bool, String> {
        match self.gpu.active_stream.as_ref() {
            Some(stream) => self
                .gpu
                .hip
                .stream_is_capturing(stream)
                .map_err(|error| error.to_string()),
            None => Ok(false),
        }
    }

    fn zero(&mut self, piece: LoanPiece) -> Result<(), String> {
        let (buf, _, _) = self.view(piece)?;
        let compute = self.compute()?;
        self.gpu
            .hip
            .memset_async(&buf, 0, buf.size(), compute)
            .map_err(|error| error.to_string())
    }

    fn copy_async(&mut self, piece: LoanPiece) -> Result<(), String> {
        let (dst, src, _) = self.view(piece)?;
        let compute = self.compute()?;
        // Stream order puts the copy after every scratch consumer of the
        // prompt and before any later compute work.
        self.gpu
            .hip
            .memcpy_htod_async(&dst, src, compute)
            .map_err(|error| error.to_string())
    }

    fn record_done(&mut self) -> Result<(), String> {
        let compute = self.compute()?;
        self.gpu
            .hip
            .event_record(self.done, Some(compute))
            .map_err(|error| error.to_string())
    }

    fn wait_done(&mut self) -> Result<(), String> {
        self.gpu
            .hip
            .event_synchronize(self.done)
            .map_err(|error| error.to_string())?;
        self.gpu
            .take_device_wait_error()
            .map_err(|error| error.to_string())
    }

    fn device_sync(&mut self) -> Result<(), String> {
        self.gpu
            .hip
            .device_synchronize()
            .map_err(|error| error.to_string())
    }

    fn copy_blocking(&mut self, piece: LoanPiece) -> Result<(), String> {
        let (dst, src, _) = self.view(piece)?;
        self.gpu
            .hip
            .memcpy_htod(&dst, src)
            .map_err(|error| error.to_string())
    }

    fn verify_piece(&mut self, piece: LoanPiece) -> Result<(), String> {
        let (_, _, region) = self.view(piece)?;
        let (layer, slot, base, mirror) =
            (region.layer, region.slot, region.base, region.mirror_cpu);
        let (_, offset, len) = checked_span(self.regions, piece)?;
        let mut host = vec![0u8; len.min(VERIFY_CHUNK_BYTES)];
        let mut done = 0usize;
        while done < len {
            let n = (len - done).min(VERIFY_CHUNK_BYTES);
            // SAFETY: inside the checked donor range (see `view`).
            let src = unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    (base + offset + done) as *mut std::ffi::c_void,
                    n,
                )
            };
            self.gpu
                .memcpy_dtoh_auto(&mut host[..n], &src)
                .map_err(|error| error.to_string())?;
            // SAFETY: inside the checked mirror range (see `view`).
            let want =
                unsafe { std::slice::from_raw_parts((mirror + offset + done) as *const u8, n) };
            if let Some(at) = host[..n]
                .iter()
                .zip(want)
                .position(|(got, want)| got != want)
            {
                return Err(format!(
                    "donor layer {layer} {slot:?} byte {} holds {:#04x}, its mirror {:#04x}",
                    offset + done + at,
                    host[at],
                    want[at]
                ));
            }
            done += n;
        }
        Ok(())
    }
}

/// The local owner of the EP reserve donor loan.
pub(crate) struct EpReserveLoan {
    controller: LoanController,
    map: ExtentMap,
    regions: Vec<LoanRegion>,
    /// Owning pinned mirrors, parallel to `regions`.
    mirrors: Vec<GpuTensor>,
    done: hip_bridge::Event,
    verify: bool,
    stats: EpLoanStats,
}

/// Read one donor tensor back once into a pinned host-mapped mirror; returns
/// the mirror and its CPU address.
fn mirror_donor(
    gpu: &mut Gpu,
    owner: &GpuTensor,
) -> Result<(GpuTensor, usize), Qwen4GpuForwardError> {
    let bytes = owner.buf.size();
    let mut host = vec![0u8; bytes];
    // The mirror reads the loaded bytes back once, at load.
    gpu.memcpy_dtoh_auto(&mut host, &owner.buf)?;
    let mirror = gpu.upload_raw_host_mapped(&host, &[bytes])?;
    drop(host);
    match gpu
        .host_bytes(&mirror)
        .map(|host| (host.as_ptr() as usize, host.len()))
    {
        Some((cpu, len)) if len == bytes => Ok((mirror, cpu)),
        _ => {
            let _ = gpu.free_tensor(mirror);
            Err(invalid(
                "an EP reserve donor mirror has no pinned host bytes of the donor size",
            ))
        }
    }
}

/// Free `mirrors`, returning the first failure.
fn free_mirrors(gpu: &mut Gpu, mirrors: Vec<GpuTensor>) -> Option<hip_bridge::HipError> {
    let mut first = None;
    for mirror in mirrors {
        if let Err(error) = gpu.free_tensor(mirror) {
            first.get_or_insert(error);
        }
    }
    first
}

impl EpReserveLoan {
    /// Mirror the loaded bytes of every donor tensor and bind the touched
    /// extents. Call at load, before any scratch writes a donor byte. All
    /// validation runs before the first allocation; a failure after it frees
    /// every mirror and the event created so far.
    pub(crate) fn materialize(
        gpu: &mut Gpu,
        donors: &[EpLoanDonor<'_>],
        touched: &[EpReserveTouchedExtent],
        verify: bool,
        log_epochs: bool,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let mut specs = Vec::with_capacity(donors.len());
        for donor in donors {
            // Host addresses come only from owning tensors, and a donor is
            // device memory by definition (G1: a borrowed view is never
            // re-tagged as host-mapped).
            if donor.owner.buf.is_host_mapped() {
                return Err(invalid(format!(
                    "EP reserve donor layer {} {:?} is host-mapped, not device memory",
                    donor.layer, donor.slot
                )));
            }
            specs.push((donor.layer, donor.slot, donor.owner.buf.size() as u64));
        }
        let map = build_map(&specs, touched).map_err(invalid)?;
        gpu.ensure_capture_stream()?;
        let mut mirrors = Vec::with_capacity(donors.len());
        let mut regions = Vec::with_capacity(donors.len());
        for donor in donors {
            match mirror_donor(gpu, donor.owner) {
                Ok((mirror, mirror_cpu)) => {
                    regions.push(LoanRegion {
                        layer: donor.layer,
                        slot: donor.slot,
                        base: donor.owner.buf.as_ptr() as usize,
                        bytes: donor.owner.buf.size(),
                        mirror_cpu,
                    });
                    mirrors.push(mirror);
                }
                Err(error) => {
                    let _ = free_mirrors(gpu, mirrors);
                    return Err(error);
                }
            }
        }
        let done = match gpu
            .hip
            .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)
        {
            Ok(done) => done,
            Err(error) => {
                let _ = free_mirrors(gpu, mirrors);
                return Err(error.into());
            }
        };
        Ok(Self {
            controller: LoanController::new(true, log_epochs),
            map,
            regions,
            mirrors,
            done,
            verify,
            stats: EpLoanStats::default(),
        })
    }

    /// Run `f` over the controller and the extent operations of `gpu`.
    fn with_ops<R>(
        &mut self,
        gpu: &mut Gpu,
        f: impl FnOnce(&mut LoanController, &mut dyn LoanOps) -> R,
    ) -> R {
        let mut device = GpuLoanDevice {
            gpu,
            regions: &self.regions,
            done: &self.done,
        };
        let mut ops = ExtentLoanOps {
            map: &self.map,
            device: &mut device,
            stats: &mut self.stats,
            verify: self.verify,
        };
        f(&mut self.controller, &mut ops)
    }

    /// Bytes of the virtual dirty space (what a full borrowed epoch dirties).
    pub(crate) fn touched_bytes(&self) -> u64 {
        self.map.total_bytes()
    }

    /// Total pinned mirror bytes this loan owns.
    pub(crate) fn mirror_bytes(&self) -> u64 {
        self.regions.iter().map(|region| region.bytes as u64).sum()
    }

    pub(crate) fn stats(&self) -> EpLoanStats {
        self.stats
    }

    pub(crate) fn is_resident(&self) -> bool {
        self.controller.is_resident()
    }

    /// Bytes the current epoch has dirtied (virtual prefix); `None` outside one.
    pub(crate) fn dirty_extent(&self) -> Option<u64> {
        self.controller.dirty_extent()
    }

    /// `Err` naming the reason once poisoned.
    pub(crate) fn check(&self) -> Result<(), String> {
        self.controller.check()
    }

    /// The immutable full mirror of donor tensor `(layer, slot)`.
    pub(crate) fn mirror(&self, layer: usize, slot: EpDonorSlot) -> Option<EpLoanMirror<'_>> {
        let index = self
            .regions
            .iter()
            .position(|region| region.layer == layer && region.slot == slot)?;
        Some(self.mirror_at(index))
    }

    fn mirror_at(&self, index: usize) -> EpLoanMirror<'_> {
        let region = &self.regions[index];
        EpLoanMirror {
            layer: region.layer,
            slot: region.slot,
            tensor: &self.mirrors[index],
            bytes: region.bytes,
            donor_device_base: region.base,
        }
    }

    /// Every donor tensor's full mirror, in the loan's region order.
    pub(crate) fn mirrors(&self) -> impl Iterator<Item = EpLoanMirror<'_>> {
        (0..self.regions.len()).map(|index| self.mirror_at(index))
    }

    /// A borrowed tile whose scratch spans the first `span` virtual bytes is
    /// about to run (`span <= touched_bytes()`).
    pub(crate) fn acquire(&mut self, gpu: &mut Gpu, span: u64) -> Result<(), String> {
        if span > self.map.total_bytes() {
            return Err(format!(
                "EP reserve tile needs {span} donor bytes; the plan touches {}",
                self.map.total_bytes()
            ));
        }
        self.with_ops(gpu, |controller, ops| controller.acquire(ops, span))
    }

    /// A borrowed tile over the whole fixed touched layout (the one borrowed
    /// bank).
    pub(crate) fn acquire_all(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        let span = self.map.total_bytes();
        self.acquire(gpu, span)
    }

    /// A resident-bank or Full-table call is about to start: restore dirty
    /// extents and host-wait this rank's copy. The caller then crosses the
    /// mesh barrier.
    pub(crate) fn require_resident(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.with_ops(gpu, |controller, ops| controller.require_resident(ops))
    }

    /// After a tile: a failed tile recovers the dirty extents even if the
    /// caller abandons the prompt.
    pub(crate) fn after_tile(&mut self, gpu: &mut Gpu, ok: bool) -> Result<(), String> {
        self.with_ops(gpu, |controller, ops| controller.after_tile(ops, ok))
    }

    /// The outermost prompt ended: a successful prompt restores synchronously
    /// (`sync_restore`), an abandoned one takes the blocking recovery.
    pub(crate) fn finish_prompt(&mut self, gpu: &mut Gpu, ok: bool) -> Result<(), String> {
        self.with_ops(gpu, |controller, ops| controller.finish_prompt(ops, ok))
    }

    /// Restore everything; the permit authorizes [`Self::release`]. `Err`: the
    /// restore could not be proven complete: retain (leak) the loan, never
    /// release it under a possibly pending DMA.
    pub(crate) fn teardown(&mut self, gpu: &mut Gpu) -> Result<EpLoanTeardownPermit, String> {
        self.with_ops(gpu, |controller, ops| teardown_permit(controller, ops))
    }

    /// Free the mirrors and destroy the completion event. Donor tensors are
    /// not freed (the weights own them).
    pub(crate) fn release(
        self,
        gpu: &mut Gpu,
        _permit: EpLoanTeardownPermit,
    ) -> Option<hip_bridge::HipError> {
        let mut first = free_mirrors(gpu, self.mirrors);
        if let Err(error) = gpu.hip.event_destroy(self.done) {
            first.get_or_insert(error);
        }
        first
    }

    /// One summary line for the free log.
    pub(crate) fn summary(&self) -> String {
        let stats = self.stats;
        format!(
            "{}; {} MiB mirrored, {} MiB touched; pieces: {} zero ({} MiB), {} async restore ({} MiB), {} blocking ({} MiB)",
            self.controller.summary(),
            self.mirror_bytes() >> 20,
            self.map.total_bytes() >> 20,
            stats.zero_pieces,
            stats.zeroed_bytes >> 20,
            stats.async_pieces,
            stats.async_restore_bytes >> 20,
            stats.blocking_pieces,
            stats.blocking_restore_bytes >> 20,
        )
    }
}

#[cfg(test)]
mod ep_reserve_loan_tests {
    use super::*;
    use crate::decode_share_loan::LoanState;
    use std::collections::BTreeMap;

    /// Donor tensors: (layer 5 gate/up, layer 5 down, layer 6 gate/up, layer 6 down).
    const REGION_BYTES: [u64; 4] = [100, 60, 100, 60];

    /// Touched extents, nonadjacent across tensors: virtual [0,64) -> r0,
    /// [64,96) -> r1, [96,144) -> r2; r3 is a donor with nothing touched.
    fn extents() -> Vec<LoanPiece> {
        vec![
            LoanPiece {
                region: 0,
                offset: 0,
                len: 64,
            },
            LoanPiece {
                region: 1,
                offset: 0,
                len: 32,
            },
            LoanPiece {
                region: 2,
                offset: 0,
                len: 48,
            },
        ]
    }

    fn map() -> ExtentMap {
        ExtentMap::new(&REGION_BYTES, &extents()).unwrap()
    }

    fn piece(region: usize, offset: u64, len: u64) -> LoanPiece {
        LoanPiece {
            region,
            offset,
            len,
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Op {
        Capturing,
        Zero(LoanPiece),
        CopyAsync(LoanPiece),
        RecordDone,
        WaitDone,
        Sync,
        CopyBlocking(LoanPiece),
        Verify(LoanPiece),
    }

    /// Recording fake that also simulates the donor and mirror bytes.
    struct Fake {
        device: Vec<Vec<u8>>,
        mirror: Vec<Vec<u8>>,
        log: Vec<Op>,
        counts: BTreeMap<&'static str, usize>,
        /// `(op name, nth call)` that fails once.
        fail_once: Vec<(&'static str, usize)>,
        /// Op names that fail on every call.
        sticky: Vec<&'static str>,
        /// nth async copy that reports success but never lands (lost DMA).
        lose_async_copy: Option<usize>,
        capturing: bool,
    }

    impl Fake {
        fn new() -> Self {
            let mirror: Vec<Vec<u8>> = REGION_BYTES
                .iter()
                .enumerate()
                .map(|(r, &n)| {
                    (0..n)
                        .map(|i| ((r as u64 * 37 + i) % 251 + 1) as u8)
                        .collect()
                })
                .collect();
            Self {
                device: mirror.clone(),
                mirror,
                log: Vec::new(),
                counts: BTreeMap::new(),
                fail_once: Vec::new(),
                sticky: Vec::new(),
                lose_async_copy: None,
                capturing: false,
            }
        }

        fn hit(&mut self, name: &'static str, op: Op) -> Result<(), String> {
            self.log.push(op);
            let count = self.counts.entry(name).or_insert(0);
            *count += 1;
            let count = *count;
            if self.sticky.contains(&name) || self.fail_once.contains(&(name, count)) {
                Err(format!("{name} #{count} failed"))
            } else {
                Ok(())
            }
        }

        /// What the forward's scratch does while borrowed: scribble over the
        /// touched extents.
        fn scribble(&mut self, bytes: u64) {
            for piece in map().prefix(bytes).unwrap() {
                let (o, n) = (piece.offset as usize, piece.len as usize);
                self.device[piece.region][o..o + n].fill(0xEE);
            }
        }

        fn whole(&self) -> bool {
            self.device == self.mirror
        }

        fn range(piece: LoanPiece) -> std::ops::Range<usize> {
            piece.offset as usize..(piece.offset + piece.len) as usize
        }

        fn pieces_of(&self, want: &str) -> Vec<LoanPiece> {
            self.log
                .iter()
                .filter_map(|op| match (op, want) {
                    (Op::Zero(p), "zero")
                    | (Op::CopyAsync(p), "async")
                    | (Op::CopyBlocking(p), "blocking") => Some(*p),
                    _ => None,
                })
                .collect()
        }
    }

    impl LoanDevice for Fake {
        fn capturing(&mut self) -> Result<bool, String> {
            self.log.push(Op::Capturing);
            Ok(self.capturing)
        }
        fn zero(&mut self, p: LoanPiece) -> Result<(), String> {
            self.hit("zero", Op::Zero(p))?;
            self.device[p.region][Self::range(p)].fill(0);
            Ok(())
        }
        fn copy_async(&mut self, p: LoanPiece) -> Result<(), String> {
            self.hit("async", Op::CopyAsync(p))?;
            if self.lose_async_copy == Some(self.counts["async"]) {
                return Ok(());
            }
            let src = self.mirror[p.region][Self::range(p)].to_vec();
            self.device[p.region][Self::range(p)].copy_from_slice(&src);
            Ok(())
        }
        fn record_done(&mut self) -> Result<(), String> {
            self.hit("record", Op::RecordDone)
        }
        fn wait_done(&mut self) -> Result<(), String> {
            self.hit("wait", Op::WaitDone)
        }
        fn device_sync(&mut self) -> Result<(), String> {
            self.hit("sync", Op::Sync)
        }
        fn copy_blocking(&mut self, p: LoanPiece) -> Result<(), String> {
            self.hit("blocking", Op::CopyBlocking(p))?;
            let src = self.mirror[p.region][Self::range(p)].to_vec();
            self.device[p.region][Self::range(p)].copy_from_slice(&src);
            Ok(())
        }
        fn verify_piece(&mut self, p: LoanPiece) -> Result<(), String> {
            self.hit("verify", Op::Verify(p))?;
            if self.device[p.region][Self::range(p)] != self.mirror[p.region][Self::range(p)] {
                return Err(format!("region {} differs from its mirror", p.region));
            }
            Ok(())
        }
    }

    struct Rig {
        map: ExtentMap,
        fake: Fake,
        stats: EpLoanStats,
        controller: LoanController,
        verify: bool,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                map: map(),
                fake: Fake::new(),
                stats: EpLoanStats::default(),
                controller: LoanController::new(true, false),
                verify: false,
            }
        }

        fn run<R>(&mut self, f: impl FnOnce(&mut LoanController, &mut dyn LoanOps) -> R) -> R {
            let mut ops = ExtentLoanOps {
                map: &self.map,
                device: &mut self.fake,
                stats: &mut self.stats,
                verify: self.verify,
            };
            f(&mut self.controller, &mut ops)
        }

        fn acquire(&mut self, span: u64) -> Result<(), String> {
            self.run(|c, ops| c.acquire(ops, span))
        }
        fn require_resident(&mut self) -> Result<(), String> {
            self.run(|c, ops| c.require_resident(ops))
        }
        fn finish(&mut self, ok: bool) -> Result<(), String> {
            self.run(|c, ops| c.finish_prompt(ops, ok))
        }
        fn after_tile(&mut self, ok: bool) -> Result<(), String> {
            self.run(|c, ops| c.after_tile(ops, ok))
        }
        fn teardown(&mut self) -> Result<EpLoanTeardownPermit, String> {
            self.run(|c, ops| teardown_permit(c, ops))
        }
    }

    #[test]
    fn map_pieces_cross_nonadjacent_segments() {
        let map = map();
        assert_eq!(map.total_bytes(), 144);
        assert_eq!(
            map.pieces(0, 144).unwrap().collect::<Vec<_>>(),
            vec![piece(0, 0, 64), piece(1, 0, 32), piece(2, 0, 48)]
        );
        // Starts inside r0, spans r1, ends inside r2: the virtual offset is
        // never added to one tensor's base.
        assert_eq!(
            map.pieces(50, 60).unwrap().collect::<Vec<_>>(),
            vec![piece(0, 50, 14), piece(1, 0, 32), piece(2, 0, 14)]
        );
        // Begins exactly on an extent boundary.
        assert_eq!(
            map.pieces(64, 40).unwrap().collect::<Vec<_>>(),
            vec![piece(1, 0, 32), piece(2, 0, 8)]
        );
        assert_eq!(map.pieces(100, 0).unwrap().count(), 0);
        assert_eq!(map.pieces(144, 0).unwrap().count(), 0);
        assert!(map.pieces(100, 45).is_err());
        assert!(map.pieces(u64::MAX, 2).is_err());
        // Mirrors stay whole-tensor: the touched space is smaller than them.
        assert_eq!(map.region_bytes(1), Some(60));
        assert_eq!(map.region_bytes(3), Some(60));
        assert!(map.total_bytes() < REGION_BYTES.iter().sum::<u64>());
    }

    #[test]
    fn map_merges_contiguous_and_validates() {
        // Adjacent extents of one tensor merge; plan order is preserved.
        let merged = ExtentMap::new(
            &REGION_BYTES,
            &[piece(2, 0, 16), piece(2, 16, 16), piece(0, 8, 8)],
        )
        .unwrap();
        assert_eq!(merged.extents(), &[piece(2, 0, 32), piece(0, 8, 8)]);
        assert_eq!(merged.total_bytes(), 40);
        assert_eq!(
            merged.pieces(30, 4).unwrap().collect::<Vec<_>>(),
            vec![piece(2, 30, 2), piece(0, 8, 2)]
        );
        // Rejections.
        assert!(ExtentMap::new(&REGION_BYTES, &[piece(4, 0, 1)]).is_err());
        assert!(ExtentMap::new(&REGION_BYTES, &[piece(0, 0, 0)]).is_err());
        assert!(ExtentMap::new(&REGION_BYTES, &[piece(0, 90, 11)]).is_err());
        assert!(ExtentMap::new(&REGION_BYTES, &[piece(0, u64::MAX, 2)]).is_err());
        assert!(ExtentMap::new(
            &REGION_BYTES,
            &[piece(0, 0, 40), piece(1, 0, 8), piece(0, 30, 10)]
        )
        .is_err());
    }

    #[test]
    fn build_map_resolves_layer_slot_and_refuses_unknown() {
        let donors = [
            (5, EpDonorSlot::GateUp, 100u64),
            (5, EpDonorSlot::Down, 60),
            (6, EpDonorSlot::GateUp, 100),
        ];
        let touched = [
            EpReserveTouchedExtent {
                layer: 6,
                slot: EpDonorSlot::GateUp,
                offset: 0,
                len: 48,
            },
            EpReserveTouchedExtent {
                layer: 5,
                slot: EpDonorSlot::Down,
                offset: 0,
                len: 32,
            },
        ];
        let map = build_map(&donors, &touched).unwrap();
        assert_eq!(
            map.extents(),
            &[piece(2, 0, 48), piece(1, 0, 32)],
            "plan order, tensor index by (layer, slot)"
        );
        let unknown = [EpReserveTouchedExtent {
            layer: 6,
            slot: EpDonorSlot::Down,
            offset: 0,
            len: 8,
        }];
        assert!(build_map(&donors, &unknown).is_err());
        let duplicate = [donors[0], donors[0]];
        assert!(build_map(&duplicate, &[]).is_err());
        // A touched extent past the ACTUAL loaded size refuses.
        let oversize = [EpReserveTouchedExtent {
            layer: 5,
            slot: EpDonorSlot::Down,
            offset: 0,
            len: 64,
        }];
        assert!(build_map(&donors, &oversize).is_err());
    }

    #[test]
    fn zero_fill_splits_across_segments_and_only_grows() {
        let mut rig = Rig::new();
        rig.acquire(80).unwrap();
        assert_eq!(
            rig.fake.pieces_of("zero"),
            vec![piece(0, 0, 64), piece(1, 0, 16)]
        );
        rig.fake.log.clear();
        // Growth within the epoch zeroes only [80, 120).
        rig.acquire(120).unwrap();
        assert_eq!(
            rig.fake.pieces_of("zero"),
            vec![piece(1, 16, 16), piece(2, 0, 24)]
        );
        rig.fake.log.clear();
        // A smaller tile is already zeroed: no memset, still the same epoch.
        rig.acquire(40).unwrap();
        assert!(rig.fake.pieces_of("zero").is_empty());
        assert_eq!(rig.controller.dirty_extent(), Some(120));
        assert_eq!(rig.controller.counters().epochs, 1);
        assert_eq!(rig.stats.zeroed_bytes, 120);
        // r3 (nothing touched) was never written.
        assert!(rig.fake.device[3] == rig.fake.mirror[3]);
    }

    #[test]
    fn restore_writes_only_touched_extents_and_leaves_donors_whole() {
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        // Mid-epoch the touched extents hold scratch; the rest is untouched.
        assert!(!rig.fake.whole());
        assert_eq!(rig.fake.device[3], rig.fake.mirror[3]);
        assert_eq!(rig.fake.device[0][64..], rig.fake.mirror[0][64..]);
        rig.fake.log.clear();
        rig.finish(true).unwrap();
        assert!(rig.fake.whole(), "sync finish restored every touched byte");
        assert!(rig.controller.is_resident());
        assert_eq!(
            rig.fake.pieces_of("async"),
            vec![piece(0, 0, 64), piece(1, 0, 32), piece(2, 0, 48)],
            "only the touched extents are copied, not the 100/60/100/60-byte tensors"
        );
        assert!(rig.fake.pieces_of("blocking").is_empty());
        assert_eq!(rig.stats.async_restore_bytes, 144);
        // The completion event is recorded after the last copy, then waited.
        let tail: Vec<_> = rig.fake.log.iter().rev().take(2).cloned().collect();
        assert_eq!(tail, vec![Op::WaitDone, Op::RecordDone]);
        // A clean epoch is a no-op: nothing is enqueued again.
        rig.fake.log.clear();
        rig.require_resident().unwrap();
        rig.finish(true).unwrap();
        assert!(rig.fake.log.is_empty());
        assert_eq!(rig.controller.counters().restores, 1);
    }

    #[test]
    fn partial_prefix_restores_only_the_dirtied_prefix() {
        let mut rig = Rig::new();
        rig.acquire(70).unwrap();
        rig.fake.scribble(70);
        rig.finish(true).unwrap();
        assert_eq!(
            rig.fake.pieces_of("async"),
            vec![piece(0, 0, 64), piece(1, 0, 6)]
        );
        assert!(rig.fake.whole());
    }

    #[test]
    fn dirty_is_recorded_before_the_first_write() {
        // The SECOND zero piece fails: the first already landed, the extent
        // is dirty before any write, so recovery restores the whole span
        // (including the piece the failed memset never reached).
        let mut rig = Rig::new();
        rig.fake.fail_once.push(("zero", 2));
        let error = rig.acquire(80).unwrap_err();
        assert!(error.contains("zero donor scratch"), "{error}");
        assert_eq!(
            rig.fake.pieces_of("blocking"),
            vec![piece(0, 0, 64), piece(1, 0, 16)],
            "recovery covers the full dirty prefix"
        );
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());
        assert_eq!(rig.controller.counters().recoveries, 1);
        // The Dirty marker preceded every device write: the first op after the
        // capture query is the failing zero, and recovery drained first.
        assert!(matches!(
            rig.fake.log.as_slice(),
            [
                Op::Capturing,
                Op::Zero(_),
                Op::Zero(_),
                Op::Capturing,
                Op::Sync,
                ..
            ]
        ));
    }

    #[test]
    fn partial_copy_failure_recovers_the_whole_prefix() {
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.fail_once.push(("async", 2));
        let error = rig.finish(true).unwrap_err();
        assert!(error.contains("enqueue donor restore"), "{error}");
        // r0 landed asynchronously, r1 failed, r2 never started; the blocking
        // recovery repairs all three extents.
        assert_eq!(
            rig.fake.pieces_of("blocking"),
            vec![piece(0, 0, 64), piece(1, 0, 32), piece(2, 0, 48)]
        );
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());
        assert!(rig.controller.check().is_ok());
    }

    #[test]
    fn event_failures_recover_or_poison() {
        // record_done fails after every copy was enqueued: recover.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.fail_once.push(("record", 1));
        assert!(rig
            .finish(true)
            .unwrap_err()
            .contains("enqueue donor restore"));
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());

        // wait_done fails once: the blocking copy repairs it.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.fail_once.push(("wait", 1));
        assert!(rig.finish(true).unwrap_err().contains("wait donor restore"));
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());

        // The wait fails and the recovery copy fails too: poisoned, and every
        // later call names the reason without touching the device.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.fail_once.push(("wait", 1));
        rig.fake.sticky.push("blocking");
        let error = rig.finish(true).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(matches!(rig.controller.state(), LoanState::Poisoned(_)));
        rig.fake.log.clear();
        assert!(rig.acquire(8).unwrap_err().contains("poisoned"));
        assert!(rig.require_resident().unwrap_err().contains("poisoned"));
        assert!(rig.fake.log.is_empty());
    }

    #[test]
    fn verify_mismatch_poisons() {
        let mut rig = Rig::new();
        rig.verify = true;
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        // The second restore copy "succeeds" but its DMA never landed.
        rig.fake.lose_async_copy = Some(2);
        let error = rig.finish(true).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(error.contains("verification after restore"), "{error}");
        assert!(matches!(rig.controller.state(), LoanState::Poisoned(_)));
        assert!(
            rig.stats.verified_bytes <= 64,
            "stops at the first bad piece"
        );
        assert!(rig.acquire(8).is_err());

        // With the oracle off the same lost copy is not looked for, and a
        // good run verifies exactly the restored prefix.
        let mut rig = Rig::new();
        rig.verify = true;
        rig.acquire(100).unwrap();
        rig.fake.scribble(100);
        rig.finish(true).unwrap();
        assert_eq!(rig.stats.verified_bytes, 100);
        assert_eq!(
            rig.fake
                .log
                .iter()
                .filter_map(|op| match op {
                    Op::Verify(p) => Some(*p),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![piece(0, 0, 64), piece(1, 0, 32), piece(2, 0, 4)]
        );
    }

    #[test]
    fn stream_capture_is_refused() {
        let mut rig = Rig::new();
        rig.fake.capturing = true;
        let error = rig.acquire(64).unwrap_err();
        assert!(error.contains("stream capture"), "{error}");
        assert!(rig.fake.pieces_of("zero").is_empty(), "no byte was written");
        assert!(rig.controller.is_resident());

        // A resident call that finds the donors dirty under capture poisons.
        let mut rig = Rig::new();
        rig.acquire(64).unwrap();
        rig.fake.capturing = true;
        let error = rig.require_resident().unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        assert!(rig.fake.pieces_of("async").is_empty());
        assert!(rig.fake.pieces_of("blocking").is_empty());
    }

    #[test]
    fn failed_tile_and_abandoned_prompt_recover_locally() {
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.after_tile(true).unwrap();
        assert!(!rig.fake.whole(), "an ok tile leaves the epoch dirty");
        rig.after_tile(false).unwrap();
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());

        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.finish(false).unwrap();
        assert!(rig.fake.whole());
        assert_eq!(rig.fake.pieces_of("blocking").len(), 3);
    }

    #[test]
    fn teardown_permit_only_after_proven_restore() {
        // Dirty at teardown: restored and waited, permit issued.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        assert!(rig.teardown().is_ok());
        assert!(rig.fake.whole());

        // The pending DMA cannot be proven complete (wait fails, recovery
        // fails): no permit, so the owners are never released.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.sticky.push("wait");
        rig.fake.sticky.push("blocking");
        let error = rig.teardown().unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
        // A poisoned loan stays refused.
        assert!(rig.teardown().is_err());

        // A wait failure whose blocking recovery completes IS whole: permit.
        let mut rig = Rig::new();
        rig.acquire(144).unwrap();
        rig.fake.scribble(144);
        rig.fake.fail_once.push(("wait", 1));
        let _ = rig.teardown();
        assert!(rig.fake.whole());
        assert!(rig.controller.is_resident());

        // Already resident: a plain permit with no device work.
        let mut rig = Rig::new();
        assert!(rig.teardown().is_ok());
        assert!(rig.fake.log.is_empty());
    }

    #[test]
    fn two_epochs_reuse_the_same_extents() {
        let mut rig = Rig::new();
        for _ in 0..2 {
            rig.acquire(144).unwrap();
            rig.fake.scribble(144);
            rig.acquire(144).unwrap(); // a second full tile of the same epoch
            rig.fake.scribble(144);
            rig.finish(true).unwrap();
            assert!(rig.fake.whole());
        }
        let counters = rig.controller.counters();
        assert_eq!(
            (counters.epochs, counters.tiles, counters.restores),
            (2, 4, 2)
        );
        // Zero filled once per epoch (no re-zero between full tiles).
        assert_eq!(rig.stats.zeroed_bytes, 288);
        assert_eq!(rig.stats.async_restore_bytes, 288);
    }
}
