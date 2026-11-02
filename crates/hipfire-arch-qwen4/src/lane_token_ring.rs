// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pinned, fenced staging of a lane forward's token ids.
//!
//! The lane forward used to upload its ids with a blocking pageable
//! `hipMemcpy`, which waits for every kernel already queued on the legacy
//! stream, i.e. for the preceding group's compute. A pinned host-asynchronous
//! copy on the active stream returns at once, but the device reads the host
//! bytes any time until the stream reaches the copy, so the bytes must not be
//! rewritten, reallocated or freed meanwhile.
//!
//! [`LaneTokenRing`] owns that hazard. It holds [`LANE_TOKEN_RING_SLOTS`]
//! pinned slots, each with its own completion event recorded right behind its
//! copy on the stream the copy ran on. A forward takes the slot at the cursor
//! ([`LaneTokenRing::stage`]), which first waits for that slot's own event (the
//! copy only, never the compute queued after it), and uploads it
//! ([`LaneTokenRing::upload`]). The cursor moves on only once the event is
//! recorded, so the next forward writes a different slot and enqueues its group
//! without waiting behind the previous one; a slot is waited on only when the
//! ring wraps. A forward that fails after the enqueue leaves the slot pending
//! (`Unfenced` when the event record failed too), so the next write drains the
//! device instead of trusting an event, and [`Drop`] leaks a pending slot's
//! bytes rather than free memory a copy may still read.

use crate::ple_stage::{PleDrain, PleUpload};
use hip_bridge::{DeviceBuffer, Event, HipError, HipResult, PinnedHostBuffer};
use rdna_compute::Gpu;

/// Pinned slots in flight. One lane forward uploads one slot; with enqueue-ahead
/// the next forward writes the next slot while the device still reads the last.
pub(crate) const LANE_TOKEN_RING_SLOTS: usize = 4;

/// The reuse state of the ring, free of any device handle.
#[derive(Debug)]
struct RingState {
    slots: [PleUpload; LANE_TOKEN_RING_SLOTS],
    cursor: usize,
    /// The cursor slot was drained and handed out, and its upload has not
    /// been enqueued yet.
    staged: bool,
}

impl RingState {
    fn new() -> Self {
        Self {
            slots: [PleUpload::Idle; LANE_TOKEN_RING_SLOTS],
            cursor: 0,
            staged: false,
        }
    }

    fn cursor(&self) -> usize {
        self.cursor
    }

    /// The wait that retires `slot`.
    fn drain(&self, slot: usize) -> PleDrain {
        self.slots[slot].drain()
    }

    fn is_pending(&self, slot: usize) -> bool {
        self.slots[slot] != PleUpload::Idle
    }

    /// `slot` no longer has a reader.
    fn retired(&mut self, slot: usize) {
        self.slots[slot] = PleUpload::Idle;
    }

    /// The cursor slot was drained and its bytes handed to the host fill.
    fn staged(&mut self) {
        debug_assert!(!self.is_pending(self.cursor), "staged an undrained slot");
        self.staged = true;
    }

    /// Just before the copy of the staged slot is enqueued; pessimistic (the
    /// copy may already read when the call fails). `None` without a staged
    /// slot.
    fn begin_enqueue(&mut self) -> Option<usize> {
        if !std::mem::take(&mut self.staged) {
            return None;
        }
        let slot = self.cursor;
        self.slots[slot] = self.slots[slot].enqueue();
        Some(slot)
    }

    /// The event recorded right behind `slot`'s copy. Only a fenced slot moves
    /// the cursor; a slot left `Unfenced` is drained by its next writer.
    fn fence_recorded(&mut self, slot: usize) {
        self.slots[slot] = self.slots[slot].fence_recorded();
        if self.slots[slot] == PleUpload::Fenced {
            self.cursor = (slot + 1) % LANE_TOKEN_RING_SLOTS;
        }
    }
}

struct Slot {
    bytes: Option<PinnedHostBuffer>,
    fence: Option<Event>,
}

/// Pinned token-id staging of one forward: [`LANE_TOKEN_RING_SLOTS`] slots of
/// `capacity` bytes, allocated on first use of each slot.
pub(crate) struct LaneTokenRing {
    slots: Vec<Slot>,
    state: RingState,
    capacity: usize,
}

// SAFETY: the ring is the unique owner of its `hipHostMalloc` allocations
// (no host alias escapes: the only mutable view is the borrow `stage` returns)
// and its events. The raw pointer inside `PinnedHostBuffer` is the only reason
// it is not `Send`; its free function and library handle are thread-agnostic.
// Every use goes through `&mut self`, and the owning forward drains each pending
// slot before the bytes are freed (`free_gpu`, or `Drop` leaks them). Like
// `PinnedTransferBuffer`, deliberately not `Sync`.
unsafe impl Send for LaneTokenRing {}

impl LaneTokenRing {
    /// `capacity`: bytes of the widest upload (non-zero). No device memory is
    /// taken until a slot is first staged.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            slots: (0..LANE_TOKEN_RING_SLOTS)
                .map(|_| Slot {
                    bytes: None,
                    fence: None,
                })
                .collect(),
            state: RingState::new(),
            capacity,
        }
    }

    /// Retire `index`'s pending copy. On failure the state is unchanged, so
    /// the bytes stay unwritable and the next call retries.
    fn drain(&mut self, gpu: &Gpu, index: usize) -> HipResult<()> {
        match (self.state.drain(index), self.slots[index].fence.as_ref()) {
            (PleDrain::Nothing, _) => {}
            (PleDrain::Fence, Some(fence)) => gpu.hip.event_synchronize(fence)?,
            // A Fenced slot without an event cannot arise; drain the whole
            // device rather than trust it.
            (PleDrain::Fence, None) | (PleDrain::Device, _) => gpu.hip.device_synchronize()?,
        }
        self.state.retired(index);
        Ok(())
    }

    /// The first `len` bytes of the next slot for the host fill. It is the only
    /// mutable access to the pinned bytes and waits (on the slot's own copy
    /// event, not on later compute) until no enqueued copy reads them.
    pub(crate) fn stage(&mut self, gpu: &Gpu, len: usize) -> HipResult<&mut [u8]> {
        if len > self.capacity {
            return Err(HipError::new(
                0,
                "lane token upload exceeds the pinned staging capacity",
            ));
        }
        gpu.bind_thread()?;
        let index = self.state.cursor();
        if self.slots[index].bytes.is_none() {
            self.slots[index].bytes = Some(gpu.hip.host_buffer(self.capacity)?);
        }
        if self.slots[index].fence.is_none() {
            self.slots[index].fence = Some(
                gpu.hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            );
        }
        self.drain(gpu, index)?;
        self.state.staged();
        let bytes = self.slots[index]
            .bytes
            .as_mut()
            .expect("slot allocated above")
            .as_bytes_mut();
        Ok(&mut bytes[..len])
    }

    /// Enqueue the staged slot's first `len` bytes into `dst` on the active
    /// stream (the legacy stream when none is set) without blocking the host,
    /// and fence the copy on that same stream. Every `?` after the enqueue
    /// leaves the slot pending, so a failing forward cannot release it early.
    pub(crate) fn upload(&mut self, gpu: &Gpu, dst: &DeviceBuffer, len: usize) -> HipResult<()> {
        if len > self.capacity {
            return Err(HipError::new(
                0,
                "lane token upload exceeds the pinned staging capacity",
            ));
        }
        gpu.bind_thread()?;
        let Some(index) = self.state.begin_enqueue() else {
            return Err(HipError::new(0, "lane token upload without a staged slot"));
        };
        let slot = &mut self.slots[index];
        let src = &slot
            .bytes
            .as_mut()
            .expect("staged slot is allocated")
            .as_bytes_mut()[..len];
        let stream = gpu.active_stream.as_ref();
        match stream {
            Some(stream) => gpu.hip.memcpy_htod_async(dst, src, stream)?,
            None => gpu.hip.memcpy_htod_async_default(dst, src)?,
        }
        // Same stream as the copy, so the event completes after it.
        let fence = slot.fence.as_ref().expect("staged slot has a fence");
        gpu.hip.event_record(fence, stream)?;
        self.state.fence_recorded(index);
        Ok(())
    }

    /// Teardown: retire every pending copy, then destroy the events. The first
    /// failure is returned; a slot that could not be drained keeps its bytes
    /// leaked by [`Drop`].
    pub(crate) fn free_gpu(mut self, gpu: &Gpu) -> Option<HipError> {
        let mut first = None;
        for index in 0..LANE_TOKEN_RING_SLOTS {
            if let Err(error) = self.drain(gpu, index) {
                first.get_or_insert(error);
            }
            if let Some(fence) = self.slots[index].fence.take() {
                if let Err(error) = gpu.hip.event_destroy(fence) {
                    first.get_or_insert(error);
                }
            }
        }
        first
    }
}

impl Drop for LaneTokenRing {
    fn drop(&mut self) {
        // A slot dropped with a copy pending (no `free_gpu`, or a failed drain)
        // keeps its pinned bytes alive forever instead of freeing memory the
        // device may still read.
        for (slot, state) in self.slots.iter_mut().zip(self.state.slots) {
            if state != PleUpload::Idle {
                std::mem::forget(slot.bytes.take());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One full stage + enqueue + fence step on the cursor slot; returns the
    /// slot used and the drain the stage owed.
    fn step(state: &mut RingState) -> (usize, PleDrain) {
        let slot = state.cursor();
        let drain = state.drain(slot);
        state.retired(slot);
        state.staged();
        assert_eq!(state.begin_enqueue(), Some(slot));
        state.fence_recorded(slot);
        (slot, drain)
    }

    #[test]
    fn consecutive_forwards_use_distinct_slots_without_a_wait() {
        let mut state = RingState::new();
        for expected in 0..LANE_TOKEN_RING_SLOTS {
            assert_eq!(step(&mut state), (expected, PleDrain::Nothing));
        }
    }

    #[test]
    fn a_wrapped_slot_waits_on_its_own_fence_only() {
        let mut state = RingState::new();
        for _ in 0..LANE_TOKEN_RING_SLOTS {
            step(&mut state);
        }
        for lap in 0..2 * LANE_TOKEN_RING_SLOTS {
            assert_eq!(
                step(&mut state),
                (lap % LANE_TOKEN_RING_SLOTS, PleDrain::Fence)
            );
        }
    }

    #[test]
    fn a_slot_stays_pending_until_it_is_retired() {
        let mut state = RingState::new();
        step(&mut state);
        assert!(state.is_pending(0));
        assert!(!state.is_pending(1));
        state.retired(0);
        assert!(!state.is_pending(0));
    }

    #[test]
    fn a_failed_fence_record_keeps_the_slot_and_demands_a_device_drain() {
        let mut state = RingState::new();
        let slot = state.cursor();
        state.staged();
        assert_eq!(state.begin_enqueue(), Some(slot));
        // The enqueue or the record failed: `fence_recorded` is never called.
        assert!(state.is_pending(slot));
        assert_eq!(state.cursor(), slot);
        assert_eq!(state.drain(slot), PleDrain::Device);
        state.retired(slot);
        assert_eq!(step(&mut state), (slot, PleDrain::Nothing));
    }

    #[test]
    fn an_unstaged_slot_cannot_be_enqueued() {
        let mut state = RingState::new();
        assert_eq!(state.begin_enqueue(), None);
        state.staged();
        assert_eq!(state.begin_enqueue(), Some(0));
        assert_eq!(state.begin_enqueue(), None);
    }

    #[test]
    fn a_stage_without_an_upload_keeps_the_cursor_and_stays_idle() {
        let mut state = RingState::new();
        step(&mut state);
        let slot = state.cursor();
        state.retired(slot);
        state.staged();
        // The forward failed before enqueueing: the next stage reuses the slot
        // with nothing to wait for.
        assert_eq!(state.cursor(), slot);
        assert_eq!(state.drain(slot), PleDrain::Nothing);
    }
}
