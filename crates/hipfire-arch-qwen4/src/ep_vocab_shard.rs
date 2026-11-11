// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The vocabulary-sharded target LM head of the EP2 mesh (`HIPFIRE_QWEN4_EP_VOCAB_SHARD`, X1):
//! switch, admission and the cross-rank exchange.
//!
//! # What is sharded
//!
//! Today both EP ranks project the whole target LM head and scan whole logits rows, yet only
//! rank 0 uses the result. With the shard on, rank `r` projects and scans the head rows
//! `[r * V/2, (r + 1) * V/2)` ([`VocabExchange::view`]) into PACKED `[rows][V/2]` logits and
//! shard-local top-1 ids (`LaneOutputs::vocab_shard`); rank 0 then merges, per row and in a fixed
//! order, exactly the picks and the sampled draws the whole head would have produced
//! (`argmax_f32_pair_merge`, `lane_leviathan_verify` select / merged modes). Both ranks keep the
//! whole head resident (the MTP rescore reads the target rows of any candidate id).
//!
//! # The exchange
//!
//! Rank 0 reads three things of rank 1 through UVA peer pointers: its packed logits (one value a
//! row), its top-1 ids and its candidate table. [`VocabExchange`] owns those buffers (allocated at
//! mesh attach, BEFORE peer access is enabled: ROCm does not map later allocations, and the lane
//! store that holds the unsharded step logits is staged after the mesh attaches), one HIP event
//! per rank and a host barrier. A rank's own buffers are written only by that rank's stream; a
//! peer only reads them.
//!
//! # Ordering and reuse safety (DESIGN §4)
//!
//! ```text
//! rank 1 stream : shard head -> argmax -> select -> record E1
//! rank 1 host   :                                   publish -> barrier -> D2H, receipts
//! rank 0 host   :                                             barrier -> wait_peer(E1)
//! rank 0 stream : shard head -> argmax -> select ------------------ wait E1 -> pair-merge
//!                                                                     -> merged verdict -> D2H
//! ```
//!
//! * The barrier only orders "E1 is recorded" before "rank 0's wait on E1 is enqueued". Neither
//!   host thread waits for GPU work; rank 0's stream waits for E1 on the device
//!   ([`VocabExchange::wait_peer`], never `event_synchronize`). The event is created
//!   `RELEASE_TO_SYSTEM` like the row gather's, so rank 1's writes are visible to rank 0's
//!   peer reads.
//! * The barrier is released by the mesh's own failure path: [`VocabExchange::abort`] is
//!   registered as a `RankPool` abort callback, so a rank that fails (or panics) before the
//!   barrier releases the other one instead of hanging it; a timeout aborts too.
//! * Reuse: rank 1's logits, top-1 and candidate table are next written by the NEXT verify
//!   step's head. That step starts only after the root decided from rank 0's top-1 / draw D2H,
//!   which follows rank 0's merge kernels on rank 0's stream, which are the only readers of
//!   those buffers and complete before that D2H returns. Host order alone therefore guarantees
//!   no read of the previous step is still in flight (no write-after-read), and no event is
//!   needed in that direction. Every sharded verify calls [`VocabExchange::barrier`] exactly
//!   once per rank on every path (a step without sampled lanes included), so the generations
//!   of the two ranks never drift.
//!
//! The registry of extra slots ([`VocabExchange::register`]) lets a later slice (the MQ2 draft
//! head's partial blocks) publish more buffers through the same event and barrier.

use crate::gpu_forward_lanes::{VocabShardView, LANE_MAX_ROWS};
use crate::weights::trunk_shard::{default_applies, parse_switch, requested_for, ShardSwitch};
use hip_bridge::{
    Event, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM,
};
use rdna_compute::lane_leviathan::LLV_CAND_WORDS;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Developer switch of the vocabulary shard.
pub const VOCAB_SHARD_ENV: &str = "HIPFIRE_QWEN4_EP_VOCAB_SHARD";

/// The shard is a two-rank split.
pub const VOCAB_SHARD_RANKS: usize = 2;

/// Verify rows the exchange's buffers hold: [`LANE_MAX_ROWS`] rows of each of eight lanes. A step
/// with more rows keeps the whole head (the root decides, [`crate::mesh_lane_protocol`]).
pub const VOCAB_SHARD_ROWS: usize = LANE_MAX_ROWS * 8;

/// The shard boundary is a multiple of this many head rows (the projection kernels' tile).
const SHARD_ALIGN: usize = 256;

/// Host-barrier timeout (the EP row gather's budget).
const BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

/// Slot names of the target side's buffers ([`VocabExchange::register`]).
pub const SLOT_LOGITS: &str = "logits";
pub const SLOT_TOP1: &str = "top1";
pub const SLOT_CAND: &str = "cand";
/// The MQ2 draft head's partial-block scratch of a rank (`draft_rows * TOPK8_PARTIAL_BYTES`).
pub const SLOT_PARTIAL: &str = "partial";

/// Developer switch of the vocabulary-sharded MQ2 draft head.
pub const DRAFT_VOCAB_SHARD_ENV: &str = "HIPFIRE_QWEN4_EP_DRAFT_VOCAB_SHARD";

/// Draft rows per head round the partial scratch holds: one per lane of a batched head round.
pub const DRAFT_SHARD_ROWS: usize = crate::mtp_gpu::MTP_DRAFT_LANES_MAX;

/// What the raw `HIPFIRE_QWEN4_EP_DRAFT_VOCAB_SHARD` value asks for (same grammar).
pub fn draft_vocab_shard_switch() -> ShardSwitch {
    parse_switch(
        hipfire_config::developer_var(DRAFT_VOCAB_SHARD_ENV)
            .ok()
            .as_deref(),
    )
}

/// [`AttachDecision`] of the draft head's shard: it rides the target shard's exchange, so it
/// needs it attached (`exchange_attached`). Unset or `=1` with an exchange: attach; `=0`: off;
/// unset without an exchange: off (the target switch decided); `=1` without an exchange: an
/// error (the attach refuses, loudly).
pub fn draft_attach_decision(
    switch: ShardSwitch,
    exchange_attached: bool,
) -> Result<AttachDecision, String> {
    match (switch, exchange_attached) {
        (ShardSwitch::Off, _) => Ok(AttachDecision::Off),
        (_, true) => Ok(AttachDecision::Attach),
        (ShardSwitch::On, false) => Err(format!(
            "{DRAFT_VOCAB_SHARD_ENV}=1 refused: it needs the vocabulary-shard exchange \
             ({VOCAB_SHARD_ENV}), which is not attached on this load"
        )),
        (ShardSwitch::Default, false) => Ok(AttachDecision::Off),
    }
}

/// What the raw `HIPFIRE_QWEN4_EP_VOCAB_SHARD` value asks for (the trunk shard's switch grammar:
/// unset = the default, `0` / `off` / `false` or any unrecognised value = off, `1` / `true` /
/// `on` = forced).
pub fn vocab_shard_switch() -> ShardSwitch {
    parse_switch(
        hipfire_config::developer_var(VOCAB_SHARD_ENV)
            .ok()
            .as_deref(),
    )
}

/// Whether the switch asks for the shard in this process: forced, or the default (EP2 over two
/// exact gfx1201 cards).
pub fn vocab_shard_requested() -> bool {
    requested_for(vocab_shard_switch(), default_applies())
}

/// The shard geometry a load can run: two ranks and a vocabulary whose half is a whole number of
/// projection tiles. `Some(reason)` when it cannot.
pub fn geometry_refusal(ranks: usize, vocab: usize) -> Option<String> {
    if ranks != VOCAB_SHARD_RANKS {
        return Some(format!(
            "the vocabulary shard splits the head across exactly {VOCAB_SHARD_RANKS} ranks, \
             not {ranks}"
        ));
    }
    if vocab == 0 || vocab % (VOCAB_SHARD_RANKS * SHARD_ALIGN) != 0 {
        return Some(format!(
            "the {vocab}-row head does not split into {VOCAB_SHARD_RANKS} halves of whole \
             {SHARD_ALIGN}-row tiles"
        ));
    }
    None
}

/// Something else on this load that owns the LM head's rows (`Some(reason)`): the shard needs the
/// whole head resident and unsplit on every rank.
pub fn blocker(trunk_split: bool) -> Option<&'static str> {
    if crate::weights::trunk_shard::SHARD_LM_HEAD {
        Some("the LM head is row-cut across the ranks (the shard needs it whole on every rank)")
    } else if trunk_split {
        Some("HIPFIRE_EP_TRUNK_SPLIT splits the LM head's output rows itself")
    } else {
        None
    }
}

/// The mesh-attach decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttachDecision {
    /// Not requested (the kill switch, or the default does not apply): nothing is allocated.
    Off,
    /// Attach the exchange.
    Attach,
    /// The default applies but this load cannot run the shard: nothing is allocated, the reason
    /// is logged.
    Declined(String),
}

/// [`AttachDecision`] of a load: `switch` / `default_applies` (see
/// [`requested_for`]), the mesh's `ranks`, the model's `vocab` and any `blocker`. A FORCED switch
/// (`=1`) on a load that cannot run the shard is an error (the attach refuses, loudly); the
/// default only declines.
pub fn attach_decision(
    switch: ShardSwitch,
    default_applies: bool,
    ranks: usize,
    vocab: usize,
    blocker: Option<&str>,
) -> Result<AttachDecision, String> {
    if !requested_for(switch, default_applies) {
        return Ok(AttachDecision::Off);
    }
    let reason = geometry_refusal(ranks, vocab).or_else(|| blocker.map(str::to_string));
    match (reason, switch) {
        (None, _) => Ok(AttachDecision::Attach),
        (Some(reason), ShardSwitch::On) => Err(format!("{VOCAB_SHARD_ENV}=1 refused: {reason}")),
        (Some(reason), _) => Ok(AttachDecision::Declined(reason)),
    }
}

/// The addresses of a peer rank's buffers, as device addresses valid on this rank's device once
/// peer access is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerAddrs {
    /// The peer's packed `[rows][width]` F32 logits.
    pub logits: u64,
    /// The peer's `rows` shard-local top-1 ids (i32).
    pub top1: u64,
    /// The peer's candidate table (`rows * LLV_CAND_WORDS` words).
    pub cand: u64,
}

/// Generation-counting spin barrier with an abort flag and a timeout (the row gather's
/// `SpinBarrier`, which is private to its crate).
struct ShardBarrier {
    parties: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
    aborted: AtomicBool,
    timeout: Duration,
}

impl ShardBarrier {
    fn new(parties: usize, timeout: Duration) -> Self {
        Self {
            parties,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            aborted: AtomicBool::new(false),
            timeout,
        }
    }

    fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }

    fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    /// Block until every party arrived. `Err` when aborted (before or while waiting) or on
    /// timeout; a timeout aborts the barrier so the other waiter fails promptly too.
    fn wait(&self) -> Result<(), &'static str> {
        if self.is_aborted() {
            return Err("vocab-shard barrier aborted");
        }
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.parties {
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return Ok(());
        }
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            if self.generation.load(Ordering::Acquire) != generation {
                return Ok(());
            }
            if self.is_aborted() {
                return Err("vocab-shard barrier aborted");
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            if start.elapsed() >= self.timeout {
                self.abort();
                return Err("vocab-shard barrier timed out");
            }
            std::thread::yield_now();
        }
    }
}

/// One rank's side of the exchange.
#[derive(Default)]
struct RankSide {
    device_id: Option<i32>,
    event: Option<Event>,
    /// Buffers this exchange allocated for the rank (freed by [`VocabExchange::release`]).
    tensors: BTreeMap<&'static str, GpuTensor>,
    /// Device addresses by slot name: the owned tensors' and anything registered.
    addrs: BTreeMap<&'static str, u64>,
    peers_enabled: bool,
}

/// The two-rank exchange of a vocabulary-sharded verify (see the module docs).
pub struct VocabExchange {
    vocab: usize,
    rows: usize,
    /// Draft rows of the MQ2 partial scratch ([`Self::with_draft_rows`]); `0`: none.
    draft_rows: usize,
    sides: [Mutex<RankSide>; VOCAB_SHARD_RANKS],
    barrier: ShardBarrier,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn err(message: impl AsRef<str>) -> HipError {
    HipError::new(0, message.as_ref())
}

impl VocabExchange {
    /// An exchange for a `vocab`-row head holding up to `rows` verify rows. Allocates nothing
    /// ([`Self::register_rank`] does, on each rank's thread).
    pub fn new(vocab: usize, rows: usize) -> Result<Self, String> {
        if let Some(reason) = geometry_refusal(VOCAB_SHARD_RANKS, vocab) {
            return Err(reason);
        }
        if rows == 0 {
            return Err("the vocabulary shard exchange needs at least one row".to_string());
        }
        Ok(Self {
            vocab,
            rows,
            sides: [
                Mutex::new(RankSide::default()),
                Mutex::new(RankSide::default()),
            ],
            barrier: ShardBarrier::new(VOCAB_SHARD_RANKS, BARRIER_TIMEOUT),
            draft_rows: 0,
        })
    }

    /// Also allocate, per rank, the MQ2 draft head's partial scratch of `rows` draft rows
    /// ([`SLOT_PARTIAL`], `rows * Gpu::TOPK8_PARTIAL_BYTES` bytes) for the vocabulary-sharded
    /// draft head (`HIPFIRE_QWEN4_EP_DRAFT_VOCAB_SHARD`). `0`: none.
    pub fn with_draft_rows(mut self, rows: usize) -> Self {
        self.draft_rows = rows;
        self
    }

    /// Verify rows the buffers hold.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Head rows of the whole vocabulary.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Head rows of one rank's shard (the packed logits row width).
    pub fn width(&self) -> usize {
        self.vocab / VOCAB_SHARD_RANKS
    }

    /// MQ2 draft-head rows the partial scratch holds per rank (`0`: no draft exchange).
    pub fn draft_rows(&self) -> usize {
        self.draft_rows
    }

    /// Rank `rank`'s shard of the head: global rows `[rank * width, (rank + 1) * width)`.
    pub fn view(&self, rank: usize) -> VocabShardView {
        VocabShardView {
            base: rank * self.width(),
            width: self.width(),
        }
    }

    fn check_rank(&self, rank: usize, gpu: &Gpu) -> HipResult<()> {
        if rank >= VOCAB_SHARD_RANKS {
            return Err(err(format!(
                "vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks"
            )));
        }
        if let Some(device) = lock(&self.sides[rank]).device_id {
            if device != gpu.device_id {
                return Err(err(format!(
                    "vocab shard: rank {rank} is device {device} but was given a Gpu on device {}",
                    gpu.device_id
                )));
            }
        }
        Ok(())
    }

    /// Allocate rank `rank`'s buffers and event on `gpu` and register their addresses. Call on the
    /// rank's thread BEFORE peer access is enabled on any rank.
    ///
    /// Every rank gets a candidate table (`rows * LLV_CAND_WORDS` words: the select mode's
    /// output; rank 0 merges its own with the peer's). A rank other than 0 also gets the packed
    /// shard logits and top-1 ids rank 0 reads: the rank-0 verify keeps using the lane store's
    /// own buffers, which no peer ever reads.
    pub fn register_rank(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        if rank >= VOCAB_SHARD_RANKS {
            return Err(err(format!(
                "vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks"
            )));
        }
        gpu.bind_thread()?;
        if lock(&self.sides[rank]).device_id.is_some() {
            return Err(err(format!("vocab shard: rank {rank} registered twice")));
        }
        let mut side = RankSide {
            device_id: Some(gpu.device_id),
            ..RankSide::default()
        };
        let result = (|| -> HipResult<()> {
            let flags = HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM;
            side.event = Some(gpu.hip.event_create_with_flags(flags)?);
            let mut own = |slot: &'static str, tensor: GpuTensor| {
                side.addrs.insert(slot, tensor.buf.as_ptr() as usize as u64);
                side.tensors.insert(slot, tensor);
            };
            own(
                SLOT_CAND,
                gpu.zeros(&[self.rows * LLV_CAND_WORDS], DType::F32)?,
            );
            if rank != 0 {
                own(
                    SLOT_LOGITS,
                    gpu.zeros(&[self.rows * self.width()], DType::F32)?,
                );
                own(
                    SLOT_TOP1,
                    gpu.zeros(&[self.rows * std::mem::size_of::<i32>()], DType::Raw)?,
                );
            }
            if self.draft_rows > 0 {
                own(
                    SLOT_PARTIAL,
                    gpu.zeros(&[self.draft_rows * Gpu::TOPK8_PARTIAL_BYTES], DType::Raw)?,
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Nothing is published: free what this call made.
            if let Some(event) = side.event.take() {
                let _ = gpu.hip.event_destroy(event);
            }
            for (_, tensor) in std::mem::take(&mut side.tensors) {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        *lock(&self.sides[rank]) = side;
        Ok(())
    }

    /// Publish an extra device buffer of rank `rank` under `slot` (a later slice's: the MQ2 head's
    /// partial blocks). The address must stay valid until [`Self::release`], and be allocated
    /// before peer access is enabled.
    pub fn register(&self, rank: usize, slot: &'static str, addr: u64) -> Result<(), String> {
        if rank >= VOCAB_SHARD_RANKS || addr == 0 {
            return Err(format!(
                "vocab shard: cannot register slot {slot:?} at {addr:#x} of rank {rank}"
            ));
        }
        let mut side = lock(&self.sides[rank]);
        if side.addrs.insert(slot, addr).is_some() {
            return Err(format!(
                "vocab shard: rank {rank} registered {slot:?} twice"
            ));
        }
        Ok(())
    }

    /// The device address rank `rank` registered under `slot`.
    pub fn slot(&self, rank: usize, slot: &'static str) -> Result<u64, String> {
        let side = lock(self.sides.get(rank).ok_or_else(|| {
            format!("vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks")
        })?);
        side.addrs
            .get(slot)
            .copied()
            .ok_or_else(|| format!("vocab shard: rank {rank} has no {slot:?} slot"))
    }

    /// A non-owning view of the buffer this exchange allocated for `rank` under `slot`
    /// ([`SLOT_CAND`] on every rank; [`SLOT_LOGITS`] / [`SLOT_TOP1`] on ranks other than 0).
    pub fn own_view(&self, rank: usize, slot: &'static str) -> Result<GpuTensor, String> {
        let side = lock(self.sides.get(rank).ok_or_else(|| {
            format!("vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks")
        })?);
        side.tensors
            .get(slot)
            .map(|tensor| tensor.sub_offset(0, tensor.numel()))
            .ok_or_else(|| format!("vocab shard: rank {rank} owns no {slot:?} buffer"))
    }

    /// Enable P2P from `rank`'s device to the peer's (idempotent), after both ranks registered.
    /// The row gather enables the same device pair; this checks it and records it, so a read
    /// through [`Self::peer`] is refused until it happened.
    pub fn enable_peers(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check_rank(rank, gpu)?;
        gpu.bind_thread()?;
        let peer = 1 - rank;
        let peer_device = lock(&self.sides[peer])
            .device_id
            .ok_or_else(|| err(format!("vocab shard: peer rank {peer} never registered")))?;
        // Same physical device (logical emulation): no peer mapping needed.
        if peer_device != gpu.device_id {
            if !gpu.hip.can_access_peer(gpu.device_id, peer_device)? {
                return Err(err(format!(
                    "vocab shard: device {} cannot access peer device {peer_device} (rank {rank} \
                     -> {peer})",
                    gpu.device_id
                )));
            }
            match gpu.hip.enable_peer_access(peer_device) {
                Ok(()) => {}
                Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                Err(e) => return Err(e),
            }
        }
        lock(&self.sides[rank]).peers_enabled = true;
        Ok(())
    }

    /// The addresses of the peer of `rank` (valid on `rank`'s device). `Err` before `rank` enabled
    /// peer access or when the peer did not register its target buffers.
    pub fn peer(&self, rank: usize) -> Result<PeerAddrs, String> {
        if rank >= VOCAB_SHARD_RANKS {
            return Err(format!(
                "vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks"
            ));
        }
        if !lock(&self.sides[rank]).peers_enabled {
            return Err(format!(
                "vocab shard: rank {rank} reads its peer before peer access is enabled"
            ));
        }
        let peer = 1 - rank;
        Ok(PeerAddrs {
            logits: self.slot(peer, SLOT_LOGITS)?,
            top1: self.slot(peer, SLOT_TOP1)?,
            cand: self.slot(peer, SLOT_CAND)?,
        })
    }

    /// The device address of the peer of `rank`'s `slot` buffer (valid on `rank`'s device). `Err`
    /// before `rank` enabled peer access or when the peer registered no such slot.
    pub fn peer_slot(&self, rank: usize, slot: &'static str) -> Result<u64, String> {
        if rank >= VOCAB_SHARD_RANKS {
            return Err(format!(
                "vocab shard: rank {rank} outside {VOCAB_SHARD_RANKS} ranks"
            ));
        }
        if !lock(&self.sides[rank]).peers_enabled {
            return Err(format!(
                "vocab shard: rank {rank} reads its peer before peer access is enabled"
            ));
        }
        self.slot(1 - rank, slot)
    }

    /// The HIP device id rank `rank` registered on (`None` before [`Self::register_rank`]).
    pub fn device_of(&self, rank: usize) -> Option<i32> {
        lock(self.sides.get(rank)?).device_id
    }

    /// Record `rank`'s event on `gpu`'s launch stream (`gpu.active_stream`, or the null stream
    /// when none: it MUST be the stream the rank's sharded work ran on).
    pub fn publish(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check_rank(rank, gpu)?;
        gpu.bind_thread()?;
        let side = lock(&self.sides[rank]);
        let event = side
            .event
            .as_ref()
            .ok_or_else(|| err("vocab shard: publish before register or after release"))?;
        gpu.hip.event_record(event, gpu.active_stream.as_ref())
    }

    /// Make `rank`'s launch stream wait, on the device, for the peer's last
    /// [`Self::publish`]. Never blocks the host. The peer must have published (and the barrier
    /// ordered that before this call).
    pub fn wait_peer(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check_rank(rank, gpu)?;
        gpu.bind_thread()?;
        let side = lock(&self.sides[1 - rank]);
        let event = side
            .event
            .as_ref()
            .ok_or_else(|| err("vocab shard: wait before the peer registered or after release"))?;
        gpu.hip
            .stream_wait_event_on(gpu.active_stream.as_ref(), event)
    }

    /// The host barrier of the two ranks: orders the peer's [`Self::publish`] before this rank's
    /// [`Self::wait_peer`]. Released by [`Self::abort`] (the mesh's failure path), and by a
    /// timeout, with `Err`.
    pub fn barrier(&self) -> Result<(), String> {
        self.barrier.wait().map_err(str::to_string)
    }

    /// Poison the barrier: every current and future waiter returns `Err`. Idempotent; registered
    /// as a `RankPool` abort callback so a failed rank releases the other.
    pub fn abort(&self) {
        self.barrier.abort();
    }

    /// Whether [`Self::abort`] was called (or a barrier wait timed out).
    pub fn is_aborted(&self) -> bool {
        self.barrier.is_aborted()
    }

    /// Free `rank`'s buffers and event. Call once per rank on its own thread, after its stream and
    /// the peer's stream have drained (the mesh's close / teardown drain first).
    pub fn release(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        self.check_rank(rank, gpu)?;
        let mut side = std::mem::take(&mut *lock(&self.sides[rank]));
        if side.device_id.is_none() {
            return Ok(());
        }
        gpu.bind_thread()?;
        let mut first = Ok(());
        if let Some(event) = side.event.take() {
            first = gpu.hip.event_destroy(event);
        }
        for (_, tensor) in std::mem::take(&mut side.tensors) {
            let freed = gpu.free_tensor(tensor);
            if first.is_ok() {
                first = freed;
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn switch_grammar_is_the_trunk_shards() {
        assert_eq!(parse_switch(None), ShardSwitch::Default);
        assert_eq!(parse_switch(Some("")), ShardSwitch::Default);
        assert_eq!(parse_switch(Some("1")), ShardSwitch::On);
        assert_eq!(parse_switch(Some("on")), ShardSwitch::On);
        assert_eq!(parse_switch(Some("0")), ShardSwitch::Off);
        assert_eq!(parse_switch(Some("off")), ShardSwitch::Off);
        assert_eq!(parse_switch(Some("false")), ShardSwitch::Off);
        assert_eq!(parse_switch(Some("bogus")), ShardSwitch::Off);
    }

    #[test]
    fn geometry_needs_two_ranks_and_whole_tiles() {
        assert_eq!(geometry_refusal(2, 248_320), None);
        assert!(geometry_refusal(1, 248_320).is_some());
        assert!(geometry_refusal(4, 248_320).is_some());
        assert!(geometry_refusal(2, 0).is_some());
        // Half of 248,321 is not a whole number of tiles; 248,064 / 2 = 124,032 = 484.5 tiles.
        assert!(geometry_refusal(2, 248_321).is_some());
        assert!(geometry_refusal(2, 248_064).is_some());
        assert_eq!(248_320 / 2 % SHARD_ALIGN, 0);
    }

    #[test]
    fn attach_decision_default_declines_and_forced_refuses() {
        use AttachDecision::*;
        let go = |switch, default, ranks, vocab, blocker| {
            attach_decision(switch, default, ranks, vocab, blocker)
        };
        // The kill switch: nothing, whatever else holds.
        assert_eq!(go(ShardSwitch::Off, true, 2, 248_320, None), Ok(Off));
        // Default on a load the default does not cover: off, quietly.
        assert_eq!(go(ShardSwitch::Default, false, 2, 248_320, None), Ok(Off));
        // The shipped shape.
        assert_eq!(go(ShardSwitch::Default, true, 2, 248_320, None), Ok(Attach));
        assert_eq!(go(ShardSwitch::On, false, 2, 248_320, None), Ok(Attach));
        // The default applies but the load cannot run it: declined with a reason, never an error.
        match go(ShardSwitch::Default, true, 2, 248_320, Some("split")) {
            Ok(Declined(reason)) => assert!(reason.contains("split")),
            other => panic!("{other:?}"),
        }
        match go(ShardSwitch::Default, true, 2, 248_321, None) {
            Ok(Declined(reason)) => assert!(reason.contains("tiles")),
            other => panic!("{other:?}"),
        }
        // Forced on an ineligible load: refused loudly.
        let error = go(ShardSwitch::On, false, 4, 248_320, None).unwrap_err();
        assert!(
            error.contains(VOCAB_SHARD_ENV) && error.contains("refused"),
            "{error}"
        );
        assert!(go(ShardSwitch::On, true, 2, 248_320, Some("split")).is_err());
    }

    #[test]
    fn blocker_names_what_owns_the_head() {
        // `SHARD_LM_HEAD` is false: only the trunk split blocks.
        assert_eq!(blocker(false), None);
        assert!(blocker(true).is_some());
    }

    #[test]
    fn the_draft_switch_rides_the_attached_exchange() {
        use AttachDecision::*;
        // The kill switch.
        assert_eq!(draft_attach_decision(ShardSwitch::Off, true), Ok(Off));
        assert_eq!(draft_attach_decision(ShardSwitch::Off, false), Ok(Off));
        // On by default iff the exchange is attached; forced the same.
        assert_eq!(
            draft_attach_decision(ShardSwitch::Default, true),
            Ok(Attach)
        );
        assert_eq!(draft_attach_decision(ShardSwitch::On, true), Ok(Attach));
        // No exchange: the default is off quietly, a forced switch refuses loudly.
        assert_eq!(draft_attach_decision(ShardSwitch::Default, false), Ok(Off));
        let error = draft_attach_decision(ShardSwitch::On, false).unwrap_err();
        assert!(
            error.contains(DRAFT_VOCAB_SHARD_ENV) && error.contains("refused"),
            "{error}"
        );
    }

    #[test]
    fn the_draft_scratch_is_opt_in_and_sized_by_rows() {
        let x = VocabExchange::new(248_320, 8).unwrap();
        assert_eq!(x.draft_rows(), 0, "no draft scratch unless asked for");
        let x = x.with_draft_rows(DRAFT_SHARD_ROWS);
        assert_eq!(x.draft_rows(), 8);
        // No device registered yet: no peer slot, no device id.
        assert!(x.peer_slot(0, SLOT_PARTIAL).is_err());
        assert_eq!(x.device_of(1), None);
    }

    #[test]
    fn the_shards_tile_the_vocabulary() {
        let x = VocabExchange::new(248_320, VOCAB_SHARD_ROWS).unwrap();
        assert_eq!(x.width(), 124_160);
        assert_eq!(
            x.view(0),
            VocabShardView {
                base: 0,
                width: 124_160
            }
        );
        assert_eq!(
            x.view(1),
            VocabShardView {
                base: 124_160,
                width: 124_160
            }
        );
        assert_eq!(x.view(0).width + x.view(1).width, x.vocab());
        assert!(VocabExchange::new(248_321, 8).is_err());
        assert!(VocabExchange::new(248_320, 0).is_err());
    }

    #[test]
    fn a_peer_read_is_refused_until_registered_and_enabled() {
        let x = VocabExchange::new(248_320, 8).unwrap();
        assert!(x.peer(0).unwrap_err().contains("peer access"));
        assert!(x.peer(2).is_err());
        assert!(x.slot(1, SLOT_LOGITS).is_err());
        assert!(x.own_view(1, SLOT_LOGITS).is_err());
        // The generic registry: a later slice's slot.
        x.register(1, "mq2_partial", 0x1000).unwrap();
        assert_eq!(x.slot(1, "mq2_partial"), Ok(0x1000));
        assert!(x.register(1, "mq2_partial", 0x2000).is_err(), "twice");
        assert!(x.register(1, "other", 0).is_err(), "null address");
        assert!(x.register(2, "other", 8).is_err(), "rank out of range");
    }

    #[test]
    fn barrier_passes_many_generations_of_two_ranks() {
        let x = Arc::new(VocabExchange::new(248_320, 8).unwrap());
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let x = x.clone();
                std::thread::spawn(move || {
                    for _ in 0..500 {
                        x.barrier().expect("barrier");
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread");
        }
        assert!(!x.is_aborted());
    }

    #[test]
    fn abort_releases_a_waiter_and_poisons_the_barrier() {
        let x = Arc::new(VocabExchange::new(248_320, 8).unwrap());
        let waiter = {
            let x = x.clone();
            std::thread::spawn(move || x.barrier())
        };
        std::thread::sleep(Duration::from_millis(50));
        x.abort();
        let started = Instant::now();
        assert!(waiter.join().expect("thread").is_err());
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(x.barrier().is_err(), "a later waiter fails at once");
        assert!(x.is_aborted());
    }

    #[test]
    fn a_timeout_aborts_the_barrier() {
        let barrier = ShardBarrier::new(2, Duration::from_millis(100));
        let started = Instant::now();
        assert_eq!(barrier.wait(), Err("vocab-shard barrier timed out"));
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert!(barrier.is_aborted());
        assert!(barrier.wait().is_err());
    }

    #[test]
    fn the_exchange_is_shareable_across_rank_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VocabExchange>();
    }
}
