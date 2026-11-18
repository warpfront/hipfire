// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device I/O of the expert cache (slice S1b): a copy stream plus a fixed-size event pool that
//! executes the policy's [`FillOp`] tickets and turns retired events into the completion tokens
//! [`ExpertCachePolicy::deliver_completion`] accepts.
//!
//! # Contract
//!
//! * One ticket = one batch = ONE event. [`DeviceIo::issue`] copies every segment of the
//!   ticket's bundle (host bytes -> the slot's write extents) on the copy stream, then records
//!   one event after the last copy. The bank's [`FenceBackend`] protocol is the only way a
//!   [`FillComplete`] token exists, so the event pool is exposed to the bank as the private
//!   `Fences` adapter: its `poll` / `wait` return `Complete` only after `hipEventQuery` reported
//!   the event recorded and complete (or `hipEventSynchronize` returned success). No caller bool
//!   or epoch is ever completion proof; nothing outside that adapter can retire an event.
//! * Events are created once ([`HipCopyBackend::new`]). An event is `Idle`, `InFlight` (recorded,
//!   not proven complete) or `Retired` (proven complete, token not yet consumed); only an `Idle`
//!   event is handed out again, and the bank's `recycle` (called exactly when it consumes a
//!   `Complete` fence) is the only `Retired -> Idle` edge. Running out of idle events is the named
//!   [`IoError::EventPoolExhausted`], never growth.
//! * Bundles are described by plain [`Segment`]s (host pointer + length, destination span index
//!   + offset) supplied through [`BundleSource`]; no architecture type is involved. A bundle must
//!   cover every write extent of the slot exactly, without overlap
//!   ([`validate_bundle`]); a partially described bundle is refused before any copy is enqueued,
//!   so "gate/up only" can never publish.
//! * Failure is sticky. A copy that fails after an earlier segment was enqueued, a record /
//!   query / synchronize error, a backend that cannot prove quiescence, or a drop with an
//!   unretired event POISONS the [`DeviceIo`]: every later call is [`IoError::Poisoned`] and the
//!   backend is told to [`CopyDevice::quarantine`] (the HIP backend then leaks its stream and
//!   events instead of destroying handles work may still use). A poisoned fill never completes
//!   and nothing is published. A refusal before any copy (no source, bad bundle) poisons nothing
//!   and abandons the ticket.
//! * Completion tokens reach the policy only through
//!   [`ExpertCachePolicy::deliver_completion`], whose buffer is bounded: the io refuses to
//!   retire a batch while [`ExpertCachePolicy::completion_room`] is zero
//!   ([`IoError::CompletionRoomExhausted`]) so a token is never lost to an overflow.
//!
//! # Allocation
//!
//! The event book, the in-flight table and the issue scratch are sized from the event count at
//! construction. The io's own steady state allocates nothing; railgun's batch tokens
//! (`issue_fills`, `complete_fills`) allocate inside railgun, as the policy documents.
//!
//! # HIP primitives reused
//!
//! The same calls D2 (`hipfire-arch-qwen4::expert_unit_tail_rank`) uses: a non-blocking copy
//! stream (`stream_create_non_blocking`), timing-free events (`event_create_with_flags(
//! HIP_EVENT_DISABLE_TIMING)`), `memcpy_htod_async_at`, `event_record`, `event_query`,
//! `event_synchronize`.

use std::borrow::Borrow;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use hip_bridge::{DeviceBuffer, Event, HipRuntime, Stream, HIP_EVENT_DISABLE_TIMING};
use railgun::sealed_table::generation::{
    FenceBackend, FenceFailure, FenceState, FillTicket, FillsNotDone, PendingFills,
};
use railgun::window::Quiescence;
use railgun::Binding;

use super::{CacheError, ExpertCachePolicy, ExpertKey, FillOp, PoolId};

/// Most segments one bundle may have.
pub const MAX_SEGMENTS: usize = 16;
/// Most write extents one slot has (the bank's limit).
pub const MAX_SPANS: usize = 4;

// ---------------------------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------------------------

/// A byte range of canonical host memory (a plain address, never dereferenced by the validation
/// logic). It is the source of exactly one async copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostRange {
    addr: usize,
    len: usize,
}

impl HostRange {
    /// # Safety
    ///
    /// `[addr, addr + len)` must be readable host memory that nothing writes and nothing frees
    /// from the moment a copy of it is enqueued until the fill's event is retired (canonical
    /// pinned model bytes satisfy this for the model's lifetime). Pinned memory makes the copy a
    /// plain stream-ordered DMA; pageable memory is correct but the driver may stage it
    /// synchronously.
    pub unsafe fn new(addr: usize, len: usize) -> Self {
        Self { addr, len }
    }

    pub fn addr(&self) -> usize {
        self.addr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One projection segment of a bundle: `host` is copied to `write_spans()[span].base + offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub host: HostRange,
    /// Index into the ticket's write spans.
    pub span: u8,
    /// Byte offset inside that span.
    pub offset: u64,
}

impl Segment {
    pub fn new(host: HostRange, span: u8, offset: u64) -> Self {
        Self { host, span, offset }
    }
}

/// The caller's canonical source map: the segments of one expert bundle. Plain descriptors only.
pub trait BundleSource {
    /// The segments of `key`'s bundle for a slot of `pool`, `None` if the key has no canonical
    /// source.
    fn bundle(&self, key: &ExpertKey, pool: PoolId) -> Option<&[Segment]>;
}

/// Why a bundle cannot be copied into a slot. Checked before any copy is enqueued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BundleError {
    Empty,
    TooManySegments {
        count: usize,
        max: usize,
    },
    TooManySpans {
        spans: usize,
        max: usize,
    },
    /// A slot extent whose end wraps the address space.
    AddressOverflow {
        span: usize,
    },
    NullHost {
        segment: usize,
    },
    ZeroLength {
        segment: usize,
    },
    SpanOutOfRange {
        segment: usize,
        span: usize,
        spans: usize,
    },
    OutOfBounds {
        segment: usize,
        offset: u64,
        len: u64,
        span_bytes: u64,
    },
    Overlap {
        first: usize,
        second: usize,
    },
    /// The segments of a span do not cover it exactly: part of the slot would hold stale bytes.
    Uncovered {
        span: usize,
        covered: u64,
        bytes: u64,
    },
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Empty => write!(f, "the bundle has no segments"),
            Self::TooManySegments { count, max } => {
                write!(f, "{count} segments, at most {max}")
            }
            Self::TooManySpans { spans, max } => write!(f, "{spans} write spans, at most {max}"),
            Self::AddressOverflow { span } => {
                write!(f, "write span {span} wraps the address space")
            }
            Self::NullHost { segment } => write!(f, "segment {segment} has a null host address"),
            Self::ZeroLength { segment } => write!(f, "segment {segment} is empty"),
            Self::SpanOutOfRange {
                segment,
                span,
                spans,
            } => write!(
                f,
                "segment {segment} names span {span}, the slot has {spans}"
            ),
            Self::OutOfBounds {
                segment,
                offset,
                len,
                span_bytes,
            } => write!(
                f,
                "segment {segment} [{offset}, +{len}) leaves its {span_bytes}-byte span"
            ),
            Self::Overlap { first, second } => {
                write!(f, "segments {first} and {second} overlap")
            }
            Self::Uncovered {
                span,
                covered,
                bytes,
            } => write!(
                f,
                "segments cover {covered} of the {bytes} bytes of span {span}"
            ),
        }
    }
}

impl std::error::Error for BundleError {}

/// Check that `segments` fill every write span of the slot exactly: in range, non-overlapping,
/// nonzero, with every span fully covered. Pure; allocates nothing.
pub fn validate_bundle(segments: &[Segment], spans: &[Binding]) -> Result<(), BundleError> {
    if segments.is_empty() {
        return Err(BundleError::Empty);
    }
    if segments.len() > MAX_SEGMENTS {
        return Err(BundleError::TooManySegments {
            count: segments.len(),
            max: MAX_SEGMENTS,
        });
    }
    if spans.len() > MAX_SPANS {
        return Err(BundleError::TooManySpans {
            spans: spans.len(),
            max: MAX_SPANS,
        });
    }
    for (i, span) in spans.iter().enumerate() {
        if span.base.checked_add(span.bytes).is_none() {
            return Err(BundleError::AddressOverflow { span: i });
        }
    }
    let mut covered = [0u64; MAX_SPANS];
    for (i, seg) in segments.iter().enumerate() {
        if seg.host.addr == 0 {
            return Err(BundleError::NullHost { segment: i });
        }
        if seg.host.len == 0 {
            return Err(BundleError::ZeroLength { segment: i });
        }
        if seg.host.addr.checked_add(seg.host.len).is_none() {
            return Err(BundleError::NullHost { segment: i });
        }
        let s = usize::from(seg.span);
        let Some(span) = spans.get(s) else {
            return Err(BundleError::SpanOutOfRange {
                segment: i,
                span: s,
                spans: spans.len(),
            });
        };
        let len = seg.host.len as u64;
        let in_bounds = seg
            .offset
            .checked_add(len)
            .is_some_and(|end| end <= span.bytes);
        if !in_bounds {
            return Err(BundleError::OutOfBounds {
                segment: i,
                offset: seg.offset,
                len,
                span_bytes: span.bytes,
            });
        }
        for (j, other) in segments[..i].iter().enumerate() {
            if other.span == seg.span {
                let other_len = other.host.len as u64;
                if seg.offset < other.offset + other_len && other.offset < seg.offset + len {
                    return Err(BundleError::Overlap {
                        first: j,
                        second: i,
                    });
                }
            }
        }
        // In bounds and pairwise disjoint: the sum cannot exceed the span.
        covered[s] += len;
    }
    for (i, span) in spans.iter().enumerate() {
        if covered[i] != span.bytes {
            return Err(BundleError::Uncovered {
                span: i,
                covered: covered[i],
                bytes: span.bytes,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Every refusal of the device I/O. Capacity overflow, validation, poison and failed copies are
/// distinct, named variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IoError {
    /// Construction: the backend owns no events.
    NoEvents,
    /// Construction or a backend call failed.
    Backend(String),
    /// Sticky quarantine: a copy or fence failed, or work was dropped unretired.
    Poisoned { reason: String },
    /// No idle event (an event is reused only after its fence was proven complete AND consumed).
    EventPoolExhausted {
        capacity: usize,
        in_flight: usize,
        requested: usize,
    },
    /// The policy's completion buffer is full: plan the policy (staging its completions),
    /// then poll again. The batch stays pending.
    CompletionRoomExhausted { capacity: usize },
    /// The key has no canonical source.
    NoSource { key: ExpertKey },
    /// The bundle does not fit the slot's write extents.
    Bundle { key: ExpertKey, error: BundleError },
    /// Issuing the fill failed. `poisoned` says whether copies may have been enqueued.
    IssueFailed {
        key: ExpertKey,
        slot: u32,
        message: String,
        poisoned: bool,
    },
    /// The fill's event failed: the fill never completes.
    FillFailed {
        key: ExpertKey,
        slot: u32,
        fill_generation: u64,
        message: String,
        quiescence: Quiescence,
    },
    /// The policy refused a completion token.
    Policy(CacheError),
    /// An internal invariant of the event protocol broke (the io is poisoned).
    Protocol(&'static str),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoEvents => write!(f, "the copy backend owns no events"),
            Self::Backend(m) => write!(f, "copy backend: {m}"),
            Self::Poisoned { reason } => write!(f, "device io poisoned: {reason}"),
            Self::EventPoolExhausted {
                capacity,
                in_flight,
                requested,
            } => write!(
                f,
                "event pool exhausted: {requested} fill(s) need events, capacity {capacity}, \
                 {in_flight} in use"
            ),
            Self::CompletionRoomExhausted { capacity } => write!(
                f,
                "completion buffer may be full ({capacity} tokens): stage them first"
            ),
            Self::NoSource { key } => write!(f, "no canonical source for {key:?}"),
            Self::Bundle { key, error } => write!(f, "bundle of {key:?}: {error}"),
            Self::IssueFailed {
                key,
                slot,
                message,
                poisoned,
            } => write!(
                f,
                "fill of {key:?} into slot {slot} failed{}: {message}",
                if *poisoned { " (poisoned)" } else { "" }
            ),
            Self::FillFailed {
                key,
                slot,
                fill_generation,
                message,
                quiescence,
            } => write!(
                f,
                "fill of {key:?} slot {slot} generation {fill_generation} failed \
                 (quiescence {quiescence:?}): {message}"
            ),
            Self::Policy(e) => write!(f, "policy refused a completion: {e}"),
            Self::Protocol(m) => write!(f, "event protocol violated: {m}"),
        }
    }
}

impl std::error::Error for IoError {}

/// A refused [`DeviceIo::issue`]. `op` is handed back only when nothing was done and the fill
/// may be retried (no idle event, poisoned io); in every other case the ticket was abandoned.
#[derive(Debug)]
pub struct Rejected {
    pub error: IoError,
    pub op: Option<FillOp>,
}

// ---------------------------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------------------------

/// The device side: one copy stream and a fixed set of events, indexed `0..events()`.
///
/// Methods take `&self` (the HIP calls are `&self`; a fake uses interior mutability) so the copy
/// closure and the fence adapter can share the backend.
pub trait CopyDevice {
    /// Changes whenever the stream / event handles are recreated.
    fn incarnation(&self) -> u64;
    /// Events the backend owns; fixed for its lifetime.
    fn events(&self) -> usize;
    /// Enqueue one async host -> device copy of `src` to the absolute device address `dst`,
    /// inside `pool`'s allocation, on the copy stream. `Ok` means enqueued, not complete.
    fn enqueue_copy(&self, pool: PoolId, dst: u64, src: HostRange) -> Result<(), String>;
    /// Record `event` on the copy stream after everything enqueued so far.
    fn record(&self, event: usize) -> Result<(), String>;
    /// `Ok(true)` only when the event was recorded and every earlier copy finished.
    fn query(&self, event: usize) -> Result<bool, String>;
    /// Block until the event completed.
    fn synchronize(&self, event: usize) -> Result<(), String>;
    /// Work may still be running: never destroy the stream or events (leak them).
    fn quarantine(&self);
}

// ---------------------------------------------------------------------------------------------
// Event book (device-free)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventState {
    /// Free to record.
    Idle,
    /// Recorded under this fence id; not proven complete.
    InFlight(u64),
    /// Proven complete under this fence id; its token not yet consumed.
    Retired(u64),
}

/// The event slots and their generations (fence ids, never reused).
#[derive(Debug)]
struct EventBook {
    states: Vec<EventState>,
    next_fence: u64,
}

impl EventBook {
    fn new(events: usize) -> Self {
        Self {
            states: vec![EventState::Idle; events],
            next_fence: 1,
        }
    }

    fn capacity(&self) -> usize {
        self.states.len()
    }

    fn free(&self) -> usize {
        self.states
            .iter()
            .filter(|s| **s == EventState::Idle)
            .count()
    }

    fn occupied(&self) -> usize {
        self.capacity() - self.free()
    }

    fn exhausted(&self, requested: usize) -> IoError {
        IoError::EventPoolExhausted {
            capacity: self.capacity(),
            in_flight: self.occupied(),
            requested,
        }
    }

    /// Take an idle event for a new generation. An `InFlight` or unconsumed `Retired` event is
    /// never handed out.
    fn acquire(&mut self) -> Result<(usize, u64), IoError> {
        let Some(idx) = self.states.iter().position(|s| *s == EventState::Idle) else {
            return Err(self.exhausted(1));
        };
        let fence = self.next_fence;
        self.next_fence += 1;
        self.states[idx] = EventState::InFlight(fence);
        Ok((idx, fence))
    }

    fn find(&self, fence: u64) -> Option<(usize, EventState)> {
        self.states.iter().enumerate().find_map(|(i, s)| match *s {
            EventState::InFlight(f) | EventState::Retired(f) if f == fence => Some((i, *s)),
            _ => None,
        })
    }

    /// `InFlight -> Retired`: the caller holds a queried / synchronized proof.
    fn retire(&mut self, fence: u64) -> Result<(), IoError> {
        match self.find(fence) {
            Some((idx, EventState::InFlight(_))) => {
                self.states[idx] = EventState::Retired(fence);
                Ok(())
            }
            Some((_, _)) => Ok(()),
            None => Err(IoError::Protocol("a fence that names no event was retired")),
        }
    }

    /// `Retired -> Idle`: the token was consumed. Refused unless the event retired.
    fn consume(&mut self, fence: u64) -> Result<(), IoError> {
        match self.find(fence) {
            Some((idx, EventState::Retired(_))) => {
                self.states[idx] = EventState::Idle;
                Ok(())
            }
            Some((_, _)) => Err(IoError::Protocol("an event was consumed before it retired")),
            None => Err(IoError::Protocol(
                "a fence that names no event was consumed",
            )),
        }
    }

    fn is_idle(&self, idx: usize) -> bool {
        self.states.get(idx) == Some(&EventState::Idle)
    }

    /// `InFlight | Retired -> Idle` after a successful host synchronize of the event (shutdown).
    fn release_synchronized(&mut self, idx: usize) {
        self.states[idx] = EventState::Idle;
    }
}

/// The io's mutable core: the event book and the sticky poison.
#[derive(Debug)]
struct IoState {
    book: EventBook,
    poison: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Fence adapter
// ---------------------------------------------------------------------------------------------

/// The bank's view of the copy stream: one event per fence. Borrowed for one call.
struct Fences<'a, B: CopyDevice> {
    backend: &'a B,
    state: &'a mut IoState,
    /// The event the last `record` used.
    recorded: Option<usize>,
}

impl<B: CopyDevice> Fences<'_, B> {
    fn unknown(message: impl Into<String>) -> FenceFailure {
        FenceFailure {
            message: message.into(),
            quiescence: Quiescence::Unknown,
        }
    }
}

// SAFETY: a fence is one event recorded on the copy stream by `record()`, called by the bank
// after the last copy of the batch was enqueued on that stream. `poll` returns `Complete` only
// when `CopyDevice::query` reported the event recorded and complete (or the book already holds
// that proof as `Retired`), and `wait` only when `CopyDevice::synchronize` returned success; in
// stream order that means every earlier copy finished. A query / synchronize error is `Failed`
// with `Unknown` quiescence (never `Proven`). The incarnation is the backend's. `recycle`
// consumes the retired event; the book hands an event out again only from `Idle`.
unsafe impl<B: CopyDevice> FenceBackend for Fences<'_, B> {
    fn incarnation(&self) -> u64 {
        self.backend.incarnation()
    }

    fn record(&mut self) -> Result<u64, FenceFailure> {
        let (idx, fence) = self.state.book.acquire().map_err(|e| FenceFailure {
            message: e.to_string(),
            quiescence: Quiescence::Proven,
        })?;
        if let Err(message) = self.backend.record(idx) {
            return Err(Self::unknown(format!(
                "event {idx} record failed: {message}"
            )));
        }
        self.recorded = Some(idx);
        Ok(fence)
    }

    fn poll(&mut self, fence: u64) -> FenceState {
        let Some((idx, state)) = self.state.book.find(fence) else {
            return FenceState::Failed(Self::unknown(format!("fence {fence} is not recorded")));
        };
        if matches!(state, EventState::Retired(_)) {
            return FenceState::Complete;
        }
        match self.backend.query(idx) {
            Ok(true) => match self.state.book.retire(fence) {
                Ok(()) => FenceState::Complete,
                Err(e) => FenceState::Failed(Self::unknown(e.to_string())),
            },
            Ok(false) => FenceState::Pending,
            Err(message) => FenceState::Failed(Self::unknown(format!(
                "event {idx} query failed: {message}"
            ))),
        }
    }

    fn wait(&mut self, fence: u64) -> FenceState {
        let Some((idx, state)) = self.state.book.find(fence) else {
            return FenceState::Failed(Self::unknown(format!("fence {fence} is not recorded")));
        };
        if matches!(state, EventState::Retired(_)) {
            return FenceState::Complete;
        }
        match self.backend.synchronize(idx) {
            Ok(()) => match self.state.book.retire(fence) {
                Ok(()) => FenceState::Complete,
                Err(e) => FenceState::Failed(Self::unknown(e.to_string())),
            },
            Err(message) => FenceState::Failed(Self::unknown(format!(
                "event {idx} synchronize failed: {message}"
            ))),
        }
    }

    fn recycle(&mut self, fence: u64) {
        if let Err(e) = self.state.book.consume(fence) {
            // `recycle` cannot return an error: poison, the next call reports it.
            if self.state.poison.is_none() {
                self.state.poison = Some(e.to_string());
            }
            self.backend.quarantine();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// DeviceIo
// ---------------------------------------------------------------------------------------------

/// A fill whose event is recorded and not yet retired.
struct Inflight {
    pool: PoolId,
    key: ExpertKey,
    slot: u32,
    fill_generation: u64,
    batch: PendingFills,
}

/// Cumulative counters of the io.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Fills issued (one event each).
    pub fills_issued: u64,
    /// Segment copies enqueued.
    pub copies_enqueued: u64,
    pub bytes_enqueued: u64,
    /// Completion tokens delivered to the policy.
    pub completions_delivered: u64,
    /// Fills refused before any copy (no source, bad bundle).
    pub refused: u64,
    /// Fills whose copy or event failed.
    pub failed: u64,
}

/// What one [`DeviceIo::poll`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollSummary {
    /// Completion tokens delivered.
    pub delivered: usize,
    /// Fills still in flight after the poll.
    pub pending: usize,
}

/// The copy engine of one expert cache: executes [`FillOp`]s on the backend's copy stream and
/// delivers their completions. Not `Sync`; one owner thread.
pub struct DeviceIo<B: CopyDevice> {
    backend: B,
    state: IoState,
    /// `pending[event]`: the fill recorded on that event (ticket <-> event is 1:1).
    pending: Vec<Option<Inflight>>,
    /// Drain buffer of `issue_pending` (capacity = events).
    scratch: Vec<FillOp>,
    stats: IoStats,
}

impl<B: CopyDevice> DeviceIo<B> {
    pub fn new(backend: B) -> Result<Self, IoError> {
        let events = backend.events();
        if events == 0 {
            return Err(IoError::NoEvents);
        }
        let mut pending = Vec::with_capacity(events);
        pending.resize_with(events, || None);
        Ok(Self {
            backend,
            state: IoState {
                book: EventBook::new(events),
                poison: None,
            },
            pending,
            scratch: Vec::with_capacity(events),
            stats: IoStats::default(),
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Events the pool holds.
    pub fn capacity(&self) -> usize {
        self.state.book.capacity()
    }

    /// Events not in use: how many fills [`Self::issue_pending`] can take now.
    pub fn free_events(&self) -> usize {
        self.state.book.free()
    }

    /// Fills recorded and not yet delivered.
    pub fn in_flight(&self) -> usize {
        self.state.book.occupied()
    }

    pub fn is_poisoned(&self) -> bool {
        self.state.poison.is_some()
    }

    pub fn poison_reason(&self) -> Option<&str> {
        self.state.poison.as_deref()
    }

    pub fn stats(&self) -> IoStats {
        self.stats
    }

    fn check_live(&self) -> Result<(), IoError> {
        match &self.state.poison {
            Some(reason) => Err(IoError::Poisoned {
                reason: reason.clone(),
            }),
            None => Ok(()),
        }
    }

    fn poison(&mut self, reason: impl Into<String>) {
        if self.state.poison.is_none() {
            self.state.poison = Some(reason.into());
        }
        self.backend.quarantine();
    }

    /// Issue every fill the policy decided, one event each. Refused up front (nothing taken
    /// from the policy) with [`IoError::EventPoolExhausted`] when the fills outnumber the idle
    /// events. If one fill fails, the fills not yet issued are abandoned. Returns the fills
    /// issued.
    pub fn issue_pending(
        &mut self,
        policy: &mut ExpertCachePolicy,
        source: &dyn BundleSource,
    ) -> Result<usize, IoError> {
        self.check_live()?;
        let wanted = policy.fills_pending();
        if wanted == 0 {
            return Ok(0);
        }
        if wanted > self.state.book.free() {
            return Err(self.state.book.exhausted(wanted));
        }
        // `wanted <= free <= capacity`: the extend below never grows the scratch.
        let mut ops = std::mem::take(&mut self.scratch);
        ops.extend(policy.take_fills());
        let mut issued = 0usize;
        let mut failure = None;
        let mut drain = ops.drain(..);
        for op in drain.by_ref() {
            match self.issue(policy, op, source) {
                Ok(()) => issued += 1,
                Err(rejected) => {
                    if let Some(op) = rejected.op {
                        policy.abandon_fills(op.pool, vec![op.ticket]);
                    }
                    failure = Some(rejected.error);
                    break;
                }
            }
        }
        for op in drain {
            policy.abandon_fills(op.pool, vec![op.ticket]);
        }
        self.scratch = ops;
        match failure {
            Some(error) => Err(error),
            None => Ok(issued),
        }
    }

    /// Copy every segment of `op`'s bundle into its slot extents on the copy stream and record
    /// one event after the last. See [`Rejected`] for what a refusal hands back.
    pub fn issue(
        &mut self,
        policy: &mut ExpertCachePolicy,
        op: FillOp,
        source: &dyn BundleSource,
    ) -> Result<(), Rejected> {
        if let Err(error) = self.check_live() {
            return Err(Rejected {
                error,
                op: Some(op),
            });
        }
        if self.state.book.free() == 0 {
            return Err(Rejected {
                error: self.state.book.exhausted(1),
                op: Some(op),
            });
        }
        let FillOp {
            key,
            pool,
            slot,
            fill_generation,
            ticket,
        } = op;
        let Some(segments) = source.bundle(&key, pool) else {
            policy.abandon_fills(pool, vec![ticket]);
            self.stats.refused += 1;
            return Err(Rejected {
                error: IoError::NoSource { key },
                op: None,
            });
        };
        if let Err(error) = validate_bundle(segments, ticket.write_spans()) {
            policy.abandon_fills(pool, vec![ticket]);
            self.stats.refused += 1;
            return Err(Rejected {
                error: IoError::Bundle { key, error },
                op: None,
            });
        }

        let copied = Cell::new(0u64);
        let bytes = Cell::new(0u64);
        let backend = &self.backend;
        let mut enqueue = |t: &FillTicket| -> Result<(), String> {
            let spans = t.write_spans();
            for (i, seg) in segments.iter().enumerate() {
                // Validated: the span exists and `base + offset` stays inside it.
                let dst = spans[usize::from(seg.span)].base + seg.offset;
                backend
                    .enqueue_copy(pool, dst, seg.host)
                    .map_err(|m| format!("segment {i}: {m}"))?;
                copied.set(copied.get() + 1);
                bytes.set(bytes.get() + seg.host.len() as u64);
            }
            Ok(())
        };
        let mut fences = Fences {
            backend: &self.backend,
            state: &mut self.state,
            recorded: None,
        };
        let issued = policy.issue_fills(pool, vec![ticket], &mut fences, &mut enqueue);
        let recorded = fences.recorded;
        self.stats.copies_enqueued += copied.get();
        self.stats.bytes_enqueued += bytes.get();
        match issued {
            Err((tickets, failure)) => {
                // Copies may be enqueued without a fence: unproven quiescence poisons.
                let poisoned = copied.get() > 0 || failure.quiescence == Quiescence::Unknown;
                policy.abandon_fills(pool, tickets);
                self.stats.failed += 1;
                if poisoned {
                    self.poison(format!(
                        "fill of {key:?} slot {slot} failed after {} copy(ies): {}",
                        copied.get(),
                        failure.message
                    ));
                }
                Err(Rejected {
                    error: IoError::IssueFailed {
                        key,
                        slot,
                        message: failure.message,
                        poisoned,
                    },
                    op: None,
                })
            }
            Ok(batch) => {
                let Some(idx) = recorded.filter(|i| self.pending[*i].is_none()) else {
                    self.poison("a fill was issued without a free event slot");
                    return Err(Rejected {
                        error: IoError::Protocol("a fill was issued without a free event slot"),
                        op: None,
                    });
                };
                self.pending[idx] = Some(Inflight {
                    pool,
                    key,
                    slot,
                    fill_generation,
                    batch,
                });
                self.stats.fills_issued += 1;
                Ok(())
            }
        }
    }

    /// Retire every fill whose event completed (non-blocking), or wait for all of them
    /// (`wait`), and deliver one completion token per retired fill to `policy`.
    ///
    /// A failed event poisons the io and is reported; its fill never completes. Fills whose
    /// event is still running stay pending.
    pub fn poll(
        &mut self,
        policy: &mut ExpertCachePolicy,
        wait: bool,
    ) -> Result<PollSummary, IoError> {
        self.check_live()?;
        let mut summary = PollSummary::default();
        for idx in 0..self.pending.len() {
            if self.pending[idx].is_none() {
                continue;
            }
            if policy.completion_room() == 0 {
                return Err(IoError::CompletionRoomExhausted {
                    capacity: policy.config().completions,
                });
            }
            let Some(Inflight {
                pool,
                key,
                slot,
                fill_generation,
                batch,
            }) = self.pending[idx].take()
            else {
                continue;
            };
            let mut fences = Fences {
                backend: &self.backend,
                state: &mut self.state,
                recorded: None,
            };
            match policy.complete_fills(pool, batch, &mut fences, wait) {
                Ok(tokens) => {
                    let mut tokens = tokens.into_iter();
                    let (Some(token), None) = (tokens.next(), tokens.next()) else {
                        let m = "a one-ticket batch did not yield exactly one completion";
                        self.poison(m);
                        return Err(IoError::Protocol(m));
                    };
                    if token.slot() != slot || token.fill_generation() != fill_generation {
                        let m = "a completion names another slot or generation";
                        self.poison(m);
                        return Err(IoError::Protocol(m));
                    }
                    if !self.state.book.is_idle(idx) {
                        let m = "a completed fill's event was not consumed";
                        self.poison(m);
                        return Err(IoError::Protocol(m));
                    }
                    policy
                        .deliver_completion(pool, token)
                        .map_err(IoError::Policy)?;
                    self.stats.completions_delivered += 1;
                    summary.delivered += 1;
                }
                Err(FillsNotDone::Pending(batch)) => {
                    self.pending[idx] = Some(Inflight {
                        pool,
                        key,
                        slot,
                        fill_generation,
                        batch,
                    });
                    summary.pending += 1;
                }
                Err(FillsNotDone::Failed { failure, .. }) => {
                    self.stats.failed += 1;
                    self.poison(format!(
                        "fill of {key:?} slot {slot} generation {fill_generation}: {}",
                        failure.message
                    ));
                    return Err(IoError::FillFailed {
                        key,
                        slot,
                        fill_generation,
                        message: failure.message,
                        quiescence: failure.quiescence,
                    });
                }
            }
        }
        Ok(summary)
    }

    /// Wait for every in-flight event, then release the backend. The dropped batches publish
    /// nothing. A poisoned io, or an event that cannot be synchronized, leaves the backend
    /// quarantined and is an error.
    pub fn shutdown(mut self) -> Result<(), IoError> {
        self.check_live()?;
        for idx in 0..self.pending.len() {
            if self.pending[idx].is_none() {
                continue;
            }
            if let Err(message) = self.backend.synchronize(idx) {
                let reason = format!("event {idx} did not synchronize at shutdown: {message}");
                self.poison(reason.clone());
                return Err(IoError::Poisoned { reason });
            }
            self.pending[idx] = None;
            self.state.book.release_synchronized(idx);
        }
        Ok(())
    }
}

impl<B: CopyDevice> Drop for DeviceIo<B> {
    fn drop(&mut self) {
        // An unretired fill (or an earlier failure) means the stream may still read host bytes
        // and write slots: never let the backend destroy its handles.
        if self.state.book.occupied() > 0 || self.state.poison.is_some() {
            self.backend.quarantine();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// HIP backend
// ---------------------------------------------------------------------------------------------

static NEXT_INCARNATION: AtomicU64 = AtomicU64::new(1);

/// A device allocation the copies may write: one pool's block.
pub struct PoolDest {
    base: u64,
    bytes: u64,
    buffer: DeviceBuffer,
}

impl PoolDest {
    /// # Safety
    ///
    /// `[base, base + bytes)` must be live device memory (the pool's block) that outlives the
    /// backend and every copy enqueued through it.
    pub unsafe fn new(base: u64, bytes: u64) -> Self {
        // SAFETY: the caller guarantees the range is live device memory for the backend's life.
        let buffer = unsafe {
            DeviceBuffer::from_raw(base as usize as *mut std::ffi::c_void, bytes as usize)
        };
        Self {
            base,
            bytes,
            buffer,
        }
    }
}

/// The HIP implementation of [`CopyDevice`]: a non-blocking copy stream and a fixed set of
/// timing-free events, all created at construction. `H` is any handle to the runtime
/// (`&HipRuntime`, `Arc<HipRuntime>`).
pub struct HipCopyBackend<H: Borrow<HipRuntime>> {
    hip: H,
    stream: Option<Stream>,
    events: Vec<Event>,
    pools: Vec<PoolDest>,
    incarnation: u64,
    quarantined: Cell<bool>,
}

fn backend_error(error: impl std::fmt::Display) -> IoError {
    IoError::Backend(error.to_string())
}

impl<H: Borrow<HipRuntime>> HipCopyBackend<H> {
    /// Create the copy stream and `events` events. `pools[i]` is the destination of `PoolId(i)`.
    pub fn new(hip: H, pools: Vec<PoolDest>, events: usize) -> Result<Self, IoError> {
        if events == 0 {
            return Err(IoError::NoEvents);
        }
        let (stream, created) = {
            let rt: &HipRuntime = hip.borrow();
            let stream = rt.stream_create_non_blocking().map_err(backend_error)?;
            let mut created = Vec::with_capacity(events);
            for _ in 0..events {
                match rt.event_create_with_flags(HIP_EVENT_DISABLE_TIMING) {
                    Ok(event) => created.push(event),
                    Err(error) => {
                        for event in created.drain(..) {
                            let _ = rt.event_destroy(event);
                        }
                        let _ = rt.stream_destroy(stream);
                        return Err(backend_error(error));
                    }
                }
            }
            (stream, created)
        };
        Ok(Self {
            hip,
            stream: Some(stream),
            events: created,
            pools,
            incarnation: NEXT_INCARNATION.fetch_add(1, Ordering::Relaxed),
            quarantined: Cell::new(false),
        })
    }

    fn rt(&self) -> &HipRuntime {
        self.hip.borrow()
    }

    fn event(&self, event: usize) -> Result<&Event, String> {
        self.events
            .get(event)
            .ok_or_else(|| format!("no event {event}"))
    }

    fn stream(&self) -> Result<&Stream, String> {
        self.stream
            .as_ref()
            .ok_or_else(|| "the copy stream is gone".to_string())
    }
}

impl<H: Borrow<HipRuntime>> CopyDevice for HipCopyBackend<H> {
    fn incarnation(&self) -> u64 {
        self.incarnation
    }

    fn events(&self) -> usize {
        self.events.len()
    }

    fn enqueue_copy(&self, pool: PoolId, dst: u64, src: HostRange) -> Result<(), String> {
        let dest = self
            .pools
            .get(usize::from(pool.0))
            .ok_or_else(|| format!("no destination for pool {}", pool.0))?;
        let offset = dst
            .checked_sub(dest.base)
            .filter(|offset| {
                offset
                    .checked_add(src.len as u64)
                    .is_some_and(|end| end <= dest.bytes)
            })
            .ok_or_else(|| {
                format!(
                    "copy of {} bytes to {dst:#x} leaves pool {} [{:#x}, +{})",
                    src.len, pool.0, dest.base, dest.bytes
                )
            })?;
        let stream = self.stream()?;
        // SAFETY: `HostRange::new` obliges its creator to keep these bytes readable and
        // unmodified until the fill's event retires.
        let bytes = unsafe { std::slice::from_raw_parts(src.addr as *const u8, src.len) };
        self.rt()
            .memcpy_htod_async_at(&dest.buffer, offset as usize, bytes, stream)
            .map_err(|e| e.to_string())
    }

    fn record(&self, event: usize) -> Result<(), String> {
        let event = self.event(event)?;
        self.rt()
            .event_record(event, Some(self.stream()?))
            .map_err(|e| e.to_string())
    }

    fn query(&self, event: usize) -> Result<bool, String> {
        self.rt()
            .event_query(self.event(event)?)
            .map_err(|e| e.to_string())
    }

    fn synchronize(&self, event: usize) -> Result<(), String> {
        self.rt()
            .event_synchronize(self.event(event)?)
            .map_err(|e| e.to_string())
    }

    fn quarantine(&self) {
        self.quarantined.set(true);
    }
}

impl<H: Borrow<HipRuntime>> Drop for HipCopyBackend<H> {
    fn drop(&mut self) {
        if self.quarantined.get() {
            // Work may still use the stream and events: leak them.
            return;
        }
        let Some(stream) = self.stream.take() else {
            return;
        };
        let rt: &HipRuntime = self.hip.borrow();
        if rt.stream_synchronize(&stream).is_err() {
            // Cannot prove the stream idle: leak.
            return;
        }
        for event in self.events.drain(..) {
            let _ = rt.event_destroy(event);
        }
        let _ = rt.stream_destroy(stream);
    }
}

// ---------------------------------------------------------------------------------------------
// Tests (CPU only, fake backend)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use railgun::sealed_table::generation::{BlockRegistry, Expert, SlotSpan};

    use super::*;
    use crate::expert_cache::{PolicyConfig, PoolConfig, PoolInit, SlotPhase};

    const BLOCK: u64 = 0x1000_0000;
    const TABLE: u64 = 0x7000_0000;
    const SLOT_BYTES: u64 = 0x2000;
    const HOST: usize = 0x9000_0000;
    const P0: PoolId = PoolId(0);

    // -- the fake device ------------------------------------------------------------------

    #[derive(Default)]
    struct Shared {
        /// `(pool, dst, host addr, len)` per enqueued copy.
        copies: RefCell<Vec<(PoolId, u64, usize, usize)>>,
        copy_calls: Cell<usize>,
        fail_copy_at: Cell<Option<usize>>,
        records: RefCell<Vec<usize>>,
        fail_record: Cell<bool>,
        /// New records stay pending until released.
        hold: Cell<bool>,
        done: RefCell<Vec<bool>>,
        fail_query: Cell<bool>,
        fail_sync: Cell<bool>,
        quarantined: Cell<bool>,
    }

    struct Fake {
        events: usize,
        shared: Rc<Shared>,
    }

    fn fake(events: usize) -> (Fake, Rc<Shared>) {
        let shared = Rc::new(Shared::default());
        *shared.done.borrow_mut() = vec![false; events];
        (
            Fake {
                events,
                shared: shared.clone(),
            },
            shared,
        )
    }

    impl CopyDevice for Fake {
        fn incarnation(&self) -> u64 {
            7
        }
        fn events(&self) -> usize {
            self.events
        }
        fn enqueue_copy(&self, pool: PoolId, dst: u64, src: HostRange) -> Result<(), String> {
            let s = &self.shared;
            let n = s.copy_calls.get();
            s.copy_calls.set(n + 1);
            if s.fail_copy_at.get() == Some(n) {
                return Err("injected copy failure".into());
            }
            s.copies
                .borrow_mut()
                .push((pool, dst, src.addr(), src.len()));
            Ok(())
        }
        fn record(&self, event: usize) -> Result<(), String> {
            let s = &self.shared;
            if s.fail_record.get() {
                return Err("injected record failure".into());
            }
            s.records.borrow_mut().push(event);
            s.done.borrow_mut()[event] = !s.hold.get();
            Ok(())
        }
        fn query(&self, event: usize) -> Result<bool, String> {
            if self.shared.fail_query.get() {
                return Err("injected query failure".into());
            }
            Ok(self.shared.done.borrow()[event])
        }
        fn synchronize(&self, event: usize) -> Result<(), String> {
            if self.shared.fail_sync.get() {
                return Err("injected synchronize failure".into());
            }
            self.shared.done.borrow_mut()[event] = true;
            Ok(())
        }
        fn quarantine(&self) {
            self.shared.quarantined.set(true);
        }
    }

    // -- the policy and the source -----------------------------------------------------------

    fn key(expert: u32) -> ExpertKey {
        ExpertKey {
            owner_id: 1,
            owner_generation: 1,
            group_id: 0,
            expert_id: expert,
        }
    }

    /// A pool of `slots` slots, each two 0x1000-byte write spans, seeded with `seeds` keys.
    fn policy(slots: usize, seeds: usize, completions: usize) -> ExpertCachePolicy {
        let registry = BlockRegistry::new(0);
        let block = registry
            .adopt::<Expert>(BLOCK, slots as u64 * SLOT_BYTES, 1)
            .expect("the pool block");
        let spans = (0..slots as u64)
            .map(|s| {
                vec![
                    SlotSpan {
                        offset: s * SLOT_BYTES,
                        bytes: 0x1000,
                    },
                    SlotSpan {
                        offset: s * SLOT_BYTES + 0x1000,
                        bytes: 0x1000,
                    },
                ]
            })
            .collect();
        let init = PoolInit {
            block,
            spans,
            tables: vec![Binding {
                base: TABLE,
                bytes: 0x100,
            }],
            seed: (0..seeds as u32).map(|i| (key(i), i)).collect(),
        };
        let config = PolicyConfig {
            use_log: 512,
            completions,
            parked: 64,
            consumers: 8,
            queues: 2,
            ..PolicyConfig::default()
        };
        ExpertCachePolicy::new(
            config,
            vec![(
                PoolConfig {
                    slots,
                    fills_per_window: 2,
                },
                init,
            )],
        )
        .expect("the policy builds")
    }

    fn host(expert: u32, part: usize, len: usize) -> HostRange {
        // SAFETY: the fake backend never dereferences the address.
        unsafe { HostRange::new(HOST + expert as usize * 0x10000 + part * 0x4000, len) }
    }

    /// Three segments per bundle: span 0 in two halves, span 1 whole.
    fn bundle(expert: u32) -> Vec<Segment> {
        vec![
            Segment::new(host(expert, 0, 0x800), 0, 0),
            Segment::new(host(expert, 1, 0x800), 0, 0x800),
            Segment::new(host(expert, 2, 0x1000), 1, 0),
        ]
    }

    struct Src(HashMap<ExpertKey, Vec<Segment>>);

    impl Src {
        fn for_experts(n: u32) -> Self {
            Self((0..n).map(|e| (key(e), bundle(e))).collect())
        }
    }

    impl BundleSource for Src {
        fn bundle(&self, key: &ExpertKey, _pool: PoolId) -> Option<&[Segment]> {
            self.0.get(key).map(Vec::as_slice)
        }
    }

    fn span(base: u64, bytes: u64) -> Binding {
        Binding { base, bytes }
    }

    fn two_spans() -> [Binding; 2] {
        [span(0x1000, 0x1000), span(0x3000, 0x1000)]
    }

    fn io(events: usize) -> (DeviceIo<Fake>, Rc<Shared>) {
        let (backend, shared) = fake(events);
        (DeviceIo::new(backend).expect("the io builds"), shared)
    }

    /// A compute-side fence that is complete at once (the retirement protocol's last-read
    /// fence; the copy engine under test is the other one).
    struct Instant(u64);

    // SAFETY: the test enqueues no compute work, so every fence is trivially complete.
    unsafe impl FenceBackend for Instant {
        fn incarnation(&self) -> u64 {
            1
        }
        fn record(&mut self) -> Result<u64, FenceFailure> {
            self.0 += 1;
            Ok(self.0)
        }
        fn poll(&mut self, _fence: u64) -> FenceState {
            FenceState::Complete
        }
        fn wait(&mut self, _fence: u64) -> FenceState {
            FenceState::Complete
        }
    }

    /// Retire, finish and seal the pool if a publication is pending, then stage every delivered
    /// completion; returns the completions published.
    fn published(p: &mut ExpertCachePolicy) -> u32 {
        if p.pool_needs_retirement(P0) {
            p.retire_generation(P0, &mut Instant(0))
                .expect("the generation retires");
            p.finish_plan(P0).expect("the plan finishes");
            p.seal(P0).expect("the next generation seals");
        }
        p.plan_group(0, (0, 0))
            .expect("the publication plan")
            .published
    }

    // -- bundle validation -------------------------------------------------------------------

    #[test]
    fn a_whole_bundle_validates() {
        assert_eq!(validate_bundle(&bundle(0), &two_spans()), Ok(()));
    }

    #[test]
    fn bounds_are_checked_per_segment() {
        let spans = two_spans();
        // Past the end of span 1.
        let mut b = bundle(0);
        b[2] = Segment::new(host(0, 2, 0x1000), 1, 1);
        assert!(matches!(
            validate_bundle(&b, &spans),
            Err(BundleError::OutOfBounds { segment: 2, .. })
        ));
        // An offset that wraps u64.
        let mut b = bundle(0);
        b[2] = Segment::new(host(0, 2, 0x1000), 1, u64::MAX);
        assert!(matches!(
            validate_bundle(&b, &spans),
            Err(BundleError::OutOfBounds { segment: 2, .. })
        ));
        // A span the slot does not have.
        let mut b = bundle(0);
        b[2] = Segment::new(host(0, 2, 0x1000), 2, 0);
        assert!(matches!(
            validate_bundle(&b, &spans),
            Err(BundleError::SpanOutOfRange {
                segment: 2,
                span: 2,
                spans: 2
            })
        ));
    }

    #[test]
    fn a_partial_bundle_is_refused_before_any_copy() {
        let spans = two_spans();
        // Span 1 (the "down") never described: gate/up alone must not publish.
        let b = bundle(0)[..2].to_vec();
        assert_eq!(
            validate_bundle(&b, &spans),
            Err(BundleError::Uncovered {
                span: 1,
                covered: 0,
                bytes: 0x1000
            })
        );
        // Half of span 0.
        let b = vec![bundle(0)[0], bundle(0)[2]];
        assert!(matches!(
            validate_bundle(&b, &spans),
            Err(BundleError::Uncovered { span: 0, .. })
        ));
    }

    #[test]
    fn overlap_empty_null_and_count_are_named() {
        let spans = two_spans();
        let mut b = bundle(0);
        b[1] = Segment::new(host(0, 1, 0x800), 0, 0x400);
        assert_eq!(
            validate_bundle(&b, &spans),
            Err(BundleError::Overlap {
                first: 0,
                second: 1
            })
        );
        assert_eq!(validate_bundle(&[], &spans), Err(BundleError::Empty));
        let mut b = bundle(0);
        b[0] = Segment::new(host(0, 0, 0), 0, 0);
        assert_eq!(
            validate_bundle(&b, &spans),
            Err(BundleError::ZeroLength { segment: 0 })
        );
        let mut b = bundle(0);
        // SAFETY: never dereferenced.
        b[0].host = unsafe { HostRange::new(0, 0x800) };
        assert_eq!(
            validate_bundle(&b, &spans),
            Err(BundleError::NullHost { segment: 0 })
        );
        let many: Vec<Segment> = (0..=MAX_SEGMENTS)
            .map(|i| Segment::new(host(0, i, 1), 0, i as u64))
            .collect();
        assert_eq!(
            validate_bundle(&many, &spans),
            Err(BundleError::TooManySegments {
                count: MAX_SEGMENTS + 1,
                max: MAX_SEGMENTS
            })
        );
        let five = [span(0x1000, 1); 5];
        assert!(matches!(
            validate_bundle(&bundle(0), &five),
            Err(BundleError::TooManySpans { spans: 5, max: 4 })
        ));
        let wrap = [span(u64::MAX, 2)];
        assert_eq!(
            validate_bundle(&bundle(0), &wrap),
            Err(BundleError::AddressOverflow { span: 0 })
        );
    }

    // -- the event book ----------------------------------------------------------------------

    #[test]
    fn an_event_is_not_reused_before_it_retired_and_was_consumed() {
        let mut book = EventBook::new(1);
        let (idx, fence) = book.acquire().expect("the one event");
        assert_eq!(idx, 0);
        // In flight: refused.
        assert!(matches!(
            book.acquire(),
            Err(IoError::EventPoolExhausted {
                capacity: 1,
                in_flight: 1,
                requested: 1
            })
        ));
        // Consuming an event that never retired is refused.
        assert!(matches!(book.consume(fence), Err(IoError::Protocol(_))));
        book.retire(fence).expect("retires");
        // Retired but not consumed: still refused.
        assert!(matches!(
            book.acquire(),
            Err(IoError::EventPoolExhausted { .. })
        ));
        book.consume(fence).expect("consumes");
        // A consumed fence is gone: it cannot be consumed or retired twice.
        assert!(matches!(book.consume(fence), Err(IoError::Protocol(_))));
        assert!(matches!(book.retire(fence), Err(IoError::Protocol(_))));
        let (again, next) = book.acquire().expect("the idle event is reusable");
        assert_eq!(again, 0);
        assert_ne!(next, fence, "a new generation has a new fence id");
    }

    // -- issue / poll ------------------------------------------------------------------------

    #[test]
    fn issue_copies_every_segment_records_one_event_per_ticket_and_completes_on_the_event() {
        let mut p = policy(4, 2, 16);
        let src = Src::for_experts(2);
        let (mut io, dev) = io(4);
        dev.hold.set(true);

        assert_eq!(io.issue_pending(&mut p, &src), Ok(2));
        assert_eq!(p.fills_pending(), 0);
        assert_eq!(io.in_flight(), 2);
        assert_eq!(io.free_events(), 2);
        // Slot s: span 0 at BLOCK + s*0x2000, span 1 at +0x1000.
        let base = |s: u64| BLOCK + s * SLOT_BYTES;
        let copies: Vec<(u64, usize, usize)> = dev
            .copies
            .borrow()
            .iter()
            .map(|c| (c.1, c.2, c.3))
            .collect();
        assert_eq!(
            copies,
            vec![
                (base(0), HOST, 0x800),
                (base(0) + 0x800, HOST + 0x4000, 0x800),
                (base(0) + 0x1000, HOST + 0x8000, 0x1000),
                (base(1), HOST + 0x10000, 0x800),
                (base(1) + 0x800, HOST + 0x10000 + 0x4000, 0x800),
                (base(1) + 0x1000, HOST + 0x10000 + 0x8000, 0x1000),
            ]
        );
        // One event per ticket, recorded after that ticket's copies.
        assert_eq!(*dev.records.borrow(), vec![0, 1]);
        assert_eq!(io.stats().copies_enqueued, 6);
        assert_eq!(io.stats().bytes_enqueued, 2 * 0x2000);

        // The events are still running: nothing completes, nothing is publishable.
        assert_eq!(
            io.poll(&mut p, false),
            Ok(PollSummary {
                delivered: 0,
                pending: 2
            })
        );
        assert_eq!(published(&mut p), 0);
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Filling));

        // Only the event decides: release slot 0's event alone.
        dev.done.borrow_mut()[0] = true;
        assert_eq!(
            io.poll(&mut p, false),
            Ok(PollSummary {
                delivered: 1,
                pending: 1
            })
        );
        assert_eq!(io.in_flight(), 1);
        assert_eq!(published(&mut p), 1);
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Live));
        assert_eq!(p.slot_phase(P0, 1), Some(SlotPhase::Filling));

        dev.done.borrow_mut()[1] = true;
        assert_eq!(
            io.poll(&mut p, false),
            Ok(PollSummary {
                delivered: 1,
                pending: 0
            })
        );
        assert_eq!(io.in_flight(), 0);
        assert_eq!(published(&mut p), 1);
        assert_eq!(p.slot_phase(P0, 1), Some(SlotPhase::Live));
        assert_eq!(io.stats().completions_delivered, 2);
        assert!(!io.is_poisoned());
        drop(io);
        assert!(!dev.quarantined.get(), "a clean drop quarantines nothing");
    }

    #[test]
    fn a_waiting_poll_synchronizes_instead_of_querying() {
        let mut p = policy(2, 1, 16);
        let src = Src::for_experts(1);
        let (mut io, dev) = io(2);
        dev.hold.set(true);
        io.issue_pending(&mut p, &src).expect("issues");
        assert_eq!(
            io.poll(&mut p, true),
            Ok(PollSummary {
                delivered: 1,
                pending: 0
            })
        );
        assert_eq!(published(&mut p), 1);
    }

    #[test]
    fn a_reused_event_is_refused_until_its_fill_completed_and_was_consumed() {
        let mut p = policy(4, 2, 16);
        let src = Src::for_experts(2);
        let (mut io, dev) = io(1);
        dev.hold.set(true);
        let mut ops: Vec<FillOp> = p.take_fills().collect();
        let second = ops.pop().expect("two fills");
        let first = ops.pop().expect("two fills");

        io.issue(&mut p, first, &src).expect("the one event");
        // The only event is in flight: the second fill is refused and handed back.
        let rejected = io.issue(&mut p, second, &src).expect_err("no idle event");
        assert!(matches!(
            rejected.error,
            IoError::EventPoolExhausted {
                capacity: 1,
                in_flight: 1,
                requested: 1
            }
        ));
        let second = rejected.op.expect("the fill is handed back for a retry");
        assert_eq!(dev.records.borrow().len(), 1, "no second record");
        assert!(!io.is_poisoned());

        // Still running: polling does not free the event.
        assert_eq!(io.poll(&mut p, false).map(|s| s.pending), Ok(1));
        assert!(io.issue(&mut p, second, &src).is_err());
    }

    #[test]
    fn a_completed_and_consumed_event_serves_the_next_generation() {
        let mut p = policy(4, 2, 16);
        let src = Src::for_experts(2);
        let (mut io, dev) = io(1);
        dev.hold.set(true);
        let mut ops: Vec<FillOp> = p.take_fills().collect();
        let second = ops.pop().expect("two fills");
        let first = ops.pop().expect("two fills");

        io.issue(&mut p, first, &src).expect("issues");
        let second = io
            .issue(&mut p, second, &src)
            .expect_err("no idle event")
            .op
            .expect("handed back");
        dev.done.borrow_mut()[0] = true;
        assert_eq!(io.poll(&mut p, false).map(|s| s.delivered), Ok(1));
        io.issue(&mut p, second, &src)
            .expect("the event was consumed");
        assert_eq!(
            *dev.records.borrow(),
            vec![0, 0],
            "the same event, a new generation"
        );
    }

    #[test]
    fn capacity_overflow_is_named_and_takes_nothing_from_the_policy() {
        let mut p = policy(4, 3, 16);
        let src = Src::for_experts(3);
        let (mut io, dev) = io(2);
        assert_eq!(
            io.issue_pending(&mut p, &src),
            Err(IoError::EventPoolExhausted {
                capacity: 2,
                in_flight: 0,
                requested: 3
            })
        );
        assert_eq!(p.fills_pending(), 3, "nothing was drained");
        assert!(dev.copies.borrow().is_empty());
        assert!(!io.is_poisoned());
        assert_eq!(DeviceIo::new(fake(0).0).err(), Some(IoError::NoEvents));
    }

    #[test]
    fn a_partial_segment_failure_poisons_and_never_completes() {
        let mut p = policy(4, 2, 16);
        let src = Src::for_experts(2);
        let (mut io, dev) = io(4);
        // The second segment copy of the first bundle fails after the first was enqueued.
        dev.fail_copy_at.set(Some(1));
        let error = io.issue_pending(&mut p, &src).expect_err("the copy fails");
        assert!(matches!(
            error,
            IoError::IssueFailed {
                slot: 0,
                poisoned: true,
                ..
            }
        ));
        assert!(io.is_poisoned());
        assert!(dev.quarantined.get(), "the backend is quarantined");
        assert!(
            dev.records.borrow().is_empty(),
            "no event after a failed bundle"
        );
        assert_eq!(io.in_flight(), 0);
        // The second ticket was abandoned, not issued.
        assert_eq!(dev.copies.borrow().len(), 1);
        // Sticky: nothing more is issued or polled, nothing completes.
        assert!(matches!(
            io.poll(&mut p, true),
            Err(IoError::Poisoned { .. })
        ));
        assert!(matches!(
            io.issue_pending(&mut p, &src),
            Err(IoError::Poisoned { .. })
        ));
        assert_eq!(published(&mut p), 0);
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Filling));
        assert_eq!(io.stats().completions_delivered, 0);
    }

    #[test]
    fn a_failed_record_poisons_and_never_completes() {
        let mut p = policy(2, 1, 16);
        let src = Src::for_experts(1);
        let (mut io, dev) = io(2);
        dev.fail_record.set(true);
        let error = io
            .issue_pending(&mut p, &src)
            .expect_err("the record fails");
        assert!(matches!(error, IoError::IssueFailed { poisoned: true, .. }));
        assert!(io.is_poisoned());
        assert!(dev.quarantined.get());
        assert_eq!(published(&mut p), 0);
    }

    #[test]
    fn a_failed_event_poisons_and_never_completes() {
        let mut p = policy(2, 1, 16);
        let src = Src::for_experts(1);
        let (mut io, dev) = io(2);
        dev.hold.set(true);
        io.issue_pending(&mut p, &src).expect("issues");
        dev.fail_query.set(true);
        let error = io.poll(&mut p, false).expect_err("the query fails");
        assert!(matches!(
            error,
            IoError::FillFailed {
                slot: 0,
                quiescence: Quiescence::Unknown,
                ..
            }
        ));
        assert!(io.is_poisoned());
        assert!(dev.quarantined.get());
        // Even if the device later reports done, a poisoned io delivers nothing.
        dev.fail_query.set(false);
        dev.done.borrow_mut()[0] = true;
        assert!(matches!(
            io.poll(&mut p, false),
            Err(IoError::Poisoned { .. })
        ));
        assert_eq!(published(&mut p), 0);
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Filling));
    }

    #[test]
    fn dropping_an_unretired_fill_poisons_the_backend() {
        let mut p = policy(2, 1, 16);
        let src = Src::for_experts(1);
        let (mut io, dev) = io(2);
        dev.hold.set(true);
        io.issue_pending(&mut p, &src).expect("issues");
        assert!(!dev.quarantined.get());
        drop(io);
        assert!(dev.quarantined.get(), "an unretired drop quarantines");
        assert_eq!(published(&mut p), 0, "and publishes nothing");
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Filling));
    }

    #[test]
    fn shutdown_synchronizes_every_event_or_quarantines() {
        let mut p = policy(2, 1, 16);
        let src = Src::for_experts(1);
        let (mut io, dev) = io(2);
        dev.hold.set(true);
        io.issue_pending(&mut p, &src).expect("issues");
        io.shutdown().expect("the event synchronizes");
        assert!(
            !dev.quarantined.get(),
            "a proven-idle shutdown releases the backend"
        );
        assert_eq!(published(&mut p), 0, "the dropped batch published nothing");

        let mut p = policy(2, 1, 16);
        let (mut io, dev) = io_held(2);
        io.issue_pending(&mut p, &src).expect("issues");
        dev.fail_sync.set(true);
        assert!(matches!(io.shutdown(), Err(IoError::Poisoned { .. })));
        assert!(dev.quarantined.get());
    }

    fn io_held(events: usize) -> (DeviceIo<Fake>, Rc<Shared>) {
        let (io, dev) = io(events);
        dev.hold.set(true);
        (io, dev)
    }

    #[test]
    fn a_bad_bundle_is_refused_before_any_copy_without_poison() {
        let mut p = policy(2, 1, 16);
        // Expert 0 described by gate/up only.
        let src = Src(HashMap::from([(key(0), bundle(0)[..2].to_vec())]));
        let (mut io, dev) = io(2);
        assert!(matches!(
            io.issue_pending(&mut p, &src),
            Err(IoError::Bundle {
                error: BundleError::Uncovered { span: 1, .. },
                ..
            })
        ));
        assert!(dev.copies.borrow().is_empty(), "no copy was enqueued");
        assert!(dev.records.borrow().is_empty());
        assert!(!io.is_poisoned());
        assert!(!dev.quarantined.get());
        assert_eq!(io.in_flight(), 0);
        assert_eq!(io.stats().refused, 1);
        assert_eq!(published(&mut p), 0);
    }

    #[test]
    fn a_key_without_a_source_is_named() {
        let mut p = policy(2, 1, 16);
        let src = Src(HashMap::new());
        let (mut io, dev) = io(2);
        assert_eq!(
            io.issue_pending(&mut p, &src),
            Err(IoError::NoSource { key: key(0) })
        );
        assert!(dev.copies.borrow().is_empty());
        assert!(!io.is_poisoned());
    }

    #[test]
    fn a_full_completion_buffer_never_loses_a_token() {
        // The policy's buffer holds one token.
        let mut p = policy(4, 2, 1);
        let src = Src::for_experts(2);
        let (mut io, dev) = io(4);
        dev.hold.set(true);
        io.issue_pending(&mut p, &src).expect("issues");
        dev.done.borrow_mut().iter_mut().for_each(|d| *d = true);

        // The first token fits; the second batch stays pending, its event untouched.
        assert_eq!(
            io.poll(&mut p, false),
            Err(IoError::CompletionRoomExhausted { capacity: 1 })
        );
        assert_eq!(io.stats().completions_delivered, 1);
        assert_eq!(io.in_flight(), 1);
        assert!(!io.is_poisoned());

        assert_eq!(published(&mut p), 1);
        assert_eq!(
            io.poll(&mut p, false),
            Ok(PollSummary {
                delivered: 1,
                pending: 0
            })
        );
        assert_eq!(published(&mut p), 1);
        assert_eq!(p.slot_phase(P0, 0), Some(SlotPhase::Live));
        assert_eq!(p.slot_phase(P0, 1), Some(SlotPhase::Live));
    }
}
