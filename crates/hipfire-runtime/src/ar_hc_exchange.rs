// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-scoped mailboxes and dispatch of the fused TP2 all-reduce + HC write
//! (`rdna_compute::ar_hc_write`, kernel `ar_hc_write_gfx1201`) for the Flash-Next EP2 trunk
//! shard (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`).
//!
//! Each rank owns one fine-grained VRAM mailbox that the PEER writes (`hipDeviceMallocFinegrained`):
//! a TAGGED region (two banks sized for 32-byte tagged granules of four f32 partials; bf16
//! partials use 16-byte granules of the same slots) and, unless `HIPFIRE_EP_AR_WIRE_DENSE=0`, a
//! DENSE region (two banks sized for 16-byte untagged f32 granules, 8 bytes for bf16 partials,
//! EMPTY = all ones; launches of at least [`AR_HC_DENSE_MIN_ROWS`] rows use it, see the kernel
//! header for the argument; the SP all-gather uses it with 8-byte granules), and one 256-byte block holding the
//! device-resident generation word and arrival ticket. Mailboxes are registered per rank
//! (any thread, that rank's `Gpu`) BEFORE peer access is enabled
//! ([`ArHcExchange::enable_peers`]), then [`ArHcExchange::qualify`] runs a short eager
//! exchange on both ranks compared bit for bit with the host model
//! (`ar_hc_write_model` / [`ArHcPartial::fold`]); a failed qualification turns the route off for
//! the process ([`ArHcExchange::enabled`]) and the caller keeps the incumbent EP2 path.
//!
//! Protocol constraints: both ranks issue the same sequence of [`ArHcExchange::write`]
//! calls with the same row counts (the generation words advance in lock step); a wait error
//! (`Gpu::take_device_wait_error`) leaves the generations unequal, so the host must call
//! [`ArHcExchange::reset`] on both drained ranks before the next launch.
//!
//! Host arrival gate: every launch (each slice of [`ArHcExchange::write`], of the SP launches)
//! first counts the rank's host arrival and waits, on the host, until the peer's host has
//! arrived at the same launch. The kernel's device wait is bounded at
//! `rdna_compute::device_wait::DEVICE_WAIT_TIMEOUT_MS` (500 ms) from the moment this rank's
//! GPU reaches it; without the gate a rank whose GPU runs ahead starts that clock while the
//! peer's host has not even issued its launch, so a host-side stall of the peer (a first-use
//! kernel load or bind on its thread, a page-cache or `mmap_lock` stall, a profiler) poisons
//! the mesh although nothing failed. With the gate the device bound covers only the GPU-side
//! lag behind an issued launch, and the host wait carries the EP host-barrier budget
//! (`ep_row_gather::DEFAULT_BARRIER_TIMEOUT`, 30 s) with the same abort contract: a failing
//! rank's pool abort ([`ArHcExchange::abort`]) releases a waiting peer at once, and a timed-out
//! wait aborts the gate for both ranks. The gate couples the two ranks' host timelines at
//! every launch; the retained / persistent path will replace it with device-side boundaries.
//! `HIPFIRE_EP_AR_HOST_GATE=0` ([`AR_HOST_GATE_ENV`]) is the kill switch (launch ungated, as
//! before).

use crate::ep_row_gather::{DEFAULT_BARRIER_TIMEOUT, SLOW_BARRIER_REPORT};
use hip_bridge::{DeviceBuffer, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED};
use rdna_compute::ar_hc_write::{
    ar_hc_bank_bytes_for, ar_hc_launch_geometry, ar_hc_write_model, ar_hc_write_model_f32,
    ArHcPartial, ArHcWire, StreamKind, AR_HC_GEN_BYTES, AR_HC_HIDDEN, AR_HC_ROW_CAP,
};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

fn err(message: String) -> HipError {
    HipError::new(0, &message)
}

struct RankMail {
    device_id: i32,
    mailbox: DeviceBuffer,
    gen: DeviceBuffer,
}

// SAFETY: the device buffers are only touched under the per-rank mutex, by calls the owning
// rank's thread makes with its own `Gpu` (register, reset, release, the launch); the shared
// pointers are plain addresses behind atomics.
unsafe impl Send for ArHcExchange {}
unsafe impl Sync for ArHcExchange {}

/// The two-rank fused all-reduce transport.
pub struct ArHcExchange {
    max_rows: usize,
    mode: ArHcMode,
    ranks: [Mutex<Option<RankMail>>; 2],
    /// Mailbox / generation base pointers, 0 until registered (lock-free launch reads).
    mailbox_ptr: [AtomicUsize; 2],
    gen_ptr: [AtomicUsize; 2],
    peers_enabled: [AtomicBool; 2],
    enabled: AtomicBool,
    /// Host arrival gate: launches each rank's host has arrived at since the last reset.
    arrivals: [AtomicU64; 2],
    /// Set by [`ArHcExchange::abort`] (a failed rank) or a timed-out gate wait.
    gate_aborted: AtomicBool,
    /// [`AR_HOST_GATE_ENV`] at construction: `false` launches ungated (the kill switch).
    host_gate_on: bool,
}

/// Which mailbox wires a transport uses (see the kernel header for the two formats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArHcMode {
    /// Every launch on the tagged wire (the kill-switch format).
    Tagged,
    /// Every launch on the dense wire (oracles and benches).
    Dense,
    /// Dense for launches of at least [`AR_HC_DENSE_MIN_ROWS`] rows, tagged below (the product
    /// default: the dense re-arm costs ~1 us of fixed latency that decode-sized launches
    /// do not earn back, the byte saving is only worth it at prefill sizes).
    Hybrid,
}

impl ArHcMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Tagged => "tagged",
            Self::Dense => "dense",
            Self::Hybrid => "hybrid",
        }
    }
}

/// First row count of a launch the hybrid mode sends on the dense wire (the measured
/// crossover on gfx1201 is between 8 and 32 rows; see arwire/RESULT.md).
// selector-receipt: receipts/selectors/ar_hc_dense_min_rows.json
pub const AR_HC_DENSE_MIN_ROWS: usize = 16;

/// `HIPFIRE_EP_AR_WIRE_DENSE` (developer, default on): `0` selects the tagged granule wire for
/// every launch (the kill switch); anything else the hybrid default.
pub const AR_WIRE_DENSE_ENV: &str = "HIPFIRE_EP_AR_WIRE_DENSE";

/// The mode the process selects (`HIPFIRE_EP_AR_WIRE_DENSE`).
pub fn ar_hc_mode_from_env() -> ArHcMode {
    if hipfire_config::developer_bool(AR_WIRE_DENSE_ENV, true) {
        ArHcMode::Hybrid
    } else {
        ArHcMode::Tagged
    }
}

/// `HIPFIRE_EP_AR_HOST_GATE` (developer, default on): `0` launches every all-reduce / SP launch
/// without the host arrival gate (the kill switch: the device wait alone, as before).
pub const AR_HOST_GATE_ENV: &str = "HIPFIRE_EP_AR_HOST_GATE";

impl ArHcExchange {
    /// Transport for launches of up to `max_rows` rows each (<= [`AR_HC_ROW_CAP`]) in the mode
    /// `HIPFIRE_EP_AR_WIRE_DENSE` selects (hybrid unless killed).
    pub fn new(max_rows: usize) -> HipResult<Self> {
        Self::with_mode(max_rows, ar_hc_mode_from_env())
    }

    /// [`Self::new`] in an explicit mode (oracles and benches).
    pub fn with_mode(max_rows: usize, mode: ArHcMode) -> HipResult<Self> {
        if max_rows == 0 || max_rows > AR_HC_ROW_CAP {
            return Err(err(format!(
                "ar_hc_exchange: max_rows {max_rows} outside 1..={AR_HC_ROW_CAP}"
            )));
        }
        Ok(Self {
            max_rows,
            mode,
            ranks: [Mutex::new(None), Mutex::new(None)],
            mailbox_ptr: [AtomicUsize::new(0), AtomicUsize::new(0)],
            gen_ptr: [AtomicUsize::new(0), AtomicUsize::new(0)],
            peers_enabled: [AtomicBool::new(false), AtomicBool::new(false)],
            enabled: AtomicBool::new(true),
            arrivals: [AtomicU64::new(0), AtomicU64::new(0)],
            gate_aborted: AtomicBool::new(false),
            host_gate_on: hipfire_config::developer_bool(AR_HOST_GATE_ENV, true),
        })
    }

    /// Host arrival gate of `rank`'s next launch (see the module docs): count this rank's
    /// arrival, then wait until the peer's host has arrived at the same launch. Spins, then
    /// yields (the [`crate::ep_row_gather`] host-barrier loop); Err when the gate is aborted
    /// before or while waiting, or after [`DEFAULT_BARRIER_TIMEOUT`] (which aborts the gate so
    /// the peer fails at once too). A wait that passes [`SLOW_BARRIER_REPORT`] is reported
    /// once while still waiting, so a stall that later resolves stays visible.
    fn host_gate(&self, rank: usize) -> HipResult<()> {
        if !self.host_gate_on {
            return Ok(());
        }
        let peer = 1 - rank;
        let ticket = self.arrivals[rank].fetch_add(1, Ordering::AcqRel) + 1;
        let span = hip_bridge::site_ledger::begin();
        let start = Instant::now();
        let mut spins = 0u32;
        let mut reported = false;
        loop {
            // Abort before arrivals, as `SpinBarrier::wait_reporting`: a peer that arrived and
            // then aborted still reads as arrived.
            let aborted = self.gate_aborted.load(Ordering::Acquire);
            if self.arrivals[peer].load(Ordering::Acquire) >= ticket {
                hip_bridge::site_ledger::end_tagged(
                    span,
                    hip_bridge::site_ledger::Api::HostWait,
                    "ar_hc_exchange::host_gate",
                );
                return Ok(());
            }
            if aborted {
                return Err(err(format!(
                    "ar_hc_exchange: rank {rank} launch {ticket} aborted while waiting for rank \
                     {peer}'s host (the mesh failed elsewhere)"
                )));
            }
            if spins < 2048 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let waited = start.elapsed();
            if waited >= DEFAULT_BARRIER_TIMEOUT {
                self.abort();
                return Err(err(format!(
                    "ar_hc_exchange: rank {rank} waited {} s on the host for rank {peer} to \
                     arrive at launch {ticket} (rank {peer} is at {}); the gate is aborted",
                    waited.as_secs(),
                    self.arrivals[peer].load(Ordering::Acquire)
                )));
            }
            if !reported && waited >= SLOW_BARRIER_REPORT {
                reported = true;
                eprintln!(
                    "ar_hc_exchange: rank {rank} has waited {:.1} s on the host for rank {peer} \
                     to arrive at launch {ticket} (rank {peer} is at {}); still waiting",
                    waited.as_secs_f64(),
                    self.arrivals[peer].load(Ordering::Acquire)
                );
            }
            std::thread::yield_now();
        }
    }

    /// Release every host-gate waiter with an error, now and until [`Self::reset`] (the rank
    /// pool's abort callback: a failed rank never arrives).
    pub fn abort(&self) {
        self.gate_aborted.store(true, Ordering::Release);
    }

    /// Fault oracles only: `rank`'s host arrives at its next launch without issuing it, so the
    /// peer's matching launch passes the host gate and its device wait runs into the bound.
    pub fn skip_launch(&self, rank: usize) {
        self.arrivals[rank].fetch_add(1, Ordering::AcqRel);
    }

    pub fn mode(&self) -> ArHcMode {
        self.mode
    }

    /// The wire a launch of `rows` rows (a slice of at most `max_rows`) uses.
    pub fn wire_for(&self, rows: usize) -> ArHcWire {
        match self.mode {
            ArHcMode::Tagged => ArHcWire::Tagged,
            ArHcMode::Dense => ArHcWire::Dense,
            ArHcMode::Hybrid if rows >= AR_HC_DENSE_MIN_ROWS => ArHcWire::Dense,
            ArHcMode::Hybrid => ArHcWire::Tagged,
        }
    }

    /// Whether launches of this transport can use `wire` (its kernels are resolved at register).
    fn uses_wire(&self, wire: ArHcWire) -> bool {
        match wire {
            ArHcWire::Tagged => self.mode != ArHcMode::Dense,
            ArHcWire::Dense => self.mode != ArHcMode::Tagged,
        }
    }

    /// The dense region exists iff some launch can use it. The tagged region always exists.
    fn has_dense_region(&self) -> bool {
        self.uses_wire(ArHcWire::Dense)
    }

    /// Bytes of one bank of `wire`'s region.
    pub fn bank_stride(&self, wire: ArHcWire) -> usize {
        ar_hc_bank_bytes_for(wire, self.max_rows)
    }

    /// Byte offset of `wire`'s region in the mailbox: the tagged region first, then the dense.
    pub fn region_offset(&self, wire: ArHcWire) -> usize {
        match wire {
            ArHcWire::Tagged => 0,
            ArHcWire::Dense => 2 * self.bank_stride(ArHcWire::Tagged),
        }
    }

    /// Bytes of the whole mailbox of one rank.
    pub fn mailbox_bytes(&self) -> usize {
        let tagged = 2 * self.bank_stride(ArHcWire::Tagged);
        if self.has_dense_region() {
            tagged + 2 * self.bank_stride(ArHcWire::Dense)
        } else {
            tagged
        }
    }

    /// Base device address of rank `rank`'s mailbox (0 until registered). Oracle / negative
    /// tests only: the peer writes it, the owner polls it.
    pub fn mailbox_addr(&self, rank: usize) -> usize {
        self.mailbox_ptr[rank].load(Ordering::Acquire)
    }

    /// Device bytes a rank spends on the mailbox and generation block.
    pub fn rank_bytes(&self) -> usize {
        self.mailbox_bytes() + AR_HC_GEN_BYTES
    }

    /// Fill every region with its idle pattern (tagged 0x00, dense 0xFF): one host copy of the
    /// whole mailbox (load / reset time only).
    fn idle_fill_mailbox(&self, gpu: &mut Gpu, mailbox: &DeviceBuffer) -> HipResult<()> {
        let mut pattern = vec![ArHcWire::Tagged.idle_fill(); self.mailbox_bytes()];
        if self.has_dense_region() {
            let from = self.region_offset(ArHcWire::Dense);
            pattern[from..].fill(ArHcWire::Dense.idle_fill());
        }
        gpu.hip.memcpy_htod(mailbox, &pattern)
    }

    /// Allocate rank `rank`'s mailbox (fine-grained, every region idle-filled) and generation block on `gpu`
    /// and resolve the kernel (JIT is not capture-safe). Call on the rank's thread before
    /// [`Self::enable_peers`] on any rank.
    pub fn register(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        if rank >= 2 {
            return Err(err(format!("ar_hc_exchange: rank {rank} outside 2 ranks")));
        }
        gpu.bind_thread()?;
        for kind in StreamKind::ALL {
            for wire in [ArHcWire::Tagged, ArHcWire::Dense] {
                if self.uses_wire(wire) {
                    for partial in ArHcPartial::ALL {
                        gpu.ensure_ar_hc_write_gfx1201_kernel(kind, wire, partial)?;
                    }
                }
            }
        }
        gpu.ensure_device_wait_block()?;
        let mut slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_some() {
            return Err(err(format!("ar_hc_exchange: rank {rank} registered twice")));
        }
        let mailbox = gpu.hip.malloc_peer_finegrained(self.mailbox_bytes())?;
        if let Err(e) = self.idle_fill_mailbox(gpu, &mailbox) {
            let _ = gpu.hip.free(mailbox);
            return Err(e);
        }
        let gen = match gpu.hip.malloc(AR_HC_GEN_BYTES) {
            Ok(g) => g,
            Err(e) => {
                let _ = gpu.hip.free(mailbox);
                return Err(e);
            }
        };
        if let Err(e) = gpu.hip.memset(&gen, 0, AR_HC_GEN_BYTES) {
            let _ = gpu.hip.free(mailbox);
            let _ = gpu.hip.free(gen);
            return Err(e);
        }
        self.mailbox_ptr[rank].store(mailbox.as_ptr() as usize, Ordering::Release);
        self.gen_ptr[rank].store(gen.as_ptr() as usize, Ordering::Release);
        *slot = Some(RankMail {
            device_id: gpu.device_id,
            mailbox,
            gen,
        });
        Ok(())
    }

    /// Enable P2P from this rank's device to the peer's (idempotent; after BOTH mailboxes
    /// exist: ROCm does not map later allocations).
    pub fn enable_peers(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        let peer = 1 - rank.min(1);
        let peer_device = {
            let slot = self.ranks[peer].lock().unwrap_or_else(|p| p.into_inner());
            slot.as_ref()
                .ok_or_else(|| {
                    err(format!(
                        "ar_hc_exchange: peer rank {peer} is not registered"
                    ))
                })?
                .device_id
        };
        gpu.bind_thread()?;
        if peer_device != gpu.device_id {
            if !gpu.hip.can_access_peer(gpu.device_id, peer_device)? {
                return Err(err(format!(
                    "ar_hc_exchange: device {} cannot access peer device {peer_device}",
                    gpu.device_id
                )));
            }
            match gpu.hip.enable_peer_access(peer_device) {
                Ok(()) => {}
                Err(e) if e.code == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
                Err(e) => return Err(e),
            }
        }
        self.peers_enabled[rank].store(true, Ordering::Release);
        Ok(())
    }

    /// False after a failed qualification or [`Self::disable`].
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Zero the rank's generation block and idle-fill its mailbox (tagged region: tag 0 never
    /// matches a live generation; dense region: every slot EMPTY), zero its host arrival count
    /// and clear a gate abort.
    /// BOTH ranks must be drained and reset before either launches again.
    pub fn reset(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: reset before register".into()))?;
        gpu.bind_thread()?;
        gpu.hip.memset(&mail.gen, 0, AR_HC_GEN_BYTES)?;
        self.idle_fill_mailbox(gpu, &mail.mailbox)?;
        self.arrivals[rank].store(0, Ordering::Release);
        self.gate_aborted.store(false, Ordering::Release);
        Ok(())
    }

    /// The rank's device generation word (launches completed since the last reset). The next
    /// launch's tag is this plus one (0xFFFF_FFFF wraps to 2) and its bank is `tag & 1`.
    /// Oracle / negative tests only; the rank must be drained.
    pub fn generation(&self, rank: usize, gpu: &mut Gpu) -> HipResult<u32> {
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: generation before register".into()))?;
        gpu.bind_thread()?;
        let mut word = [0u8; 4];
        gpu.hip.memcpy_dtoh(&mut word, &mail.gen)?;
        Ok(u32::from_le_bytes(word))
    }

    /// Copy of the rank's mailbox region of `wire` (both banks). Oracle / negative tests only;
    /// the rank must be drained.
    pub fn download_mailbox(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        wire: ArHcWire,
    ) -> HipResult<Vec<u8>> {
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: download before register".into()))?;
        gpu.bind_thread()?;
        let mut bytes = vec![0u8; 2 * self.bank_stride(wire)];
        let whole = self.mailbox_bytes();
        let mut all = vec![0u8; whole];
        gpu.hip.memcpy_dtoh(&mut all, &mail.mailbox)?;
        let off = self.region_offset(wire);
        bytes.copy_from_slice(&all[off..off + 2 * self.bank_stride(wire)]);
        Ok(bytes)
    }

    /// Free this rank's buffers (the rank must be drained).
    pub fn release(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        let mut slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        self.mailbox_ptr[rank].store(0, Ordering::Release);
        self.gen_ptr[rank].store(0, Ordering::Release);
        self.peers_enabled[rank].store(false, Ordering::Release);
        if let Some(mail) = slot.take() {
            gpu.bind_thread()?;
            let a = gpu.hip.free(mail.mailbox);
            let b = gpu.hip.free(mail.gen);
            a.and(b)?;
        }
        Ok(())
    }

    /// One fused all-reduce + HC write of `rows` rows on `rank`: `partial` `[rows, 2560]` f32
    /// sent in the `partial_fmt` wire format, `streams` `[rows, 4, 2560]` of `kind` (BF16 bits or
    /// F32 words, in place), `gates` `[rows, 4]` f32 (device pointers; rows beyond
    /// [`AR_HC_ROW_CAP`] or `max_rows` are launched in slices, each its own generation, in order,
    /// identically on both ranks; both ranks pass the same `partial_fmt`).
    ///
    /// # Safety
    /// The pointers must address live device memory of those extents on `gpu`'s device.
    pub unsafe fn write(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        partial: *mut c_void,
        streams: *mut c_void,
        kind: StreamKind,
        partial_fmt: ArHcPartial,
        gates: *const c_void,
        rows: usize,
        mixed_dbg: *mut c_void,
    ) -> HipResult<()> {
        if !self.enabled() {
            return Err(err("ar_hc_exchange: route is disabled".into()));
        }
        if rank >= 2 || !self.peers_enabled[rank].load(Ordering::Acquire) {
            return Err(err(format!(
                "ar_hc_exchange: rank {rank} has no peer access"
            )));
        }
        let peer = 1 - rank;
        let (peer_mail, local_mail, gen) = (
            self.mailbox_ptr[peer].load(Ordering::Acquire),
            self.mailbox_ptr[rank].load(Ordering::Acquire),
            self.gen_ptr[rank].load(Ordering::Acquire),
        );
        if peer_mail == 0 || local_mail == 0 || gen == 0 {
            return Err(err(
                "ar_hc_exchange: write before both ranks registered".into()
            ));
        }
        // The generation buffer is only borrowed for its pointer; hold the lock-free view.
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: write after release".into()))?;
        gpu.bind_thread()?;
        let slice = self.max_rows.min(AR_HC_ROW_CAP);
        let mut done = 0usize;
        while done < rows {
            let n = slice.min(rows - done);
            let wire = self.wire_for(n);
            let region = self.region_offset(wire);
            let geometry = ar_hc_launch_geometry(n, gpu.flags.ep_ar_wide_grid_enabled())
                .ok_or_else(|| err(format!("ar_hc_exchange: no launch geometry for {n} rows")))?;
            let dbg = if mixed_dbg.is_null() {
                mixed_dbg
            } else {
                mixed_dbg
                    .cast::<u8>()
                    .add(done * AR_HC_HIDDEN * 4)
                    .cast::<c_void>()
            };
            self.host_gate(rank)?;
            // SAFETY: forwarded from this function's contract; offsets stay inside the
            // `rows`-row extents.
            gpu.ar_hc_write_gfx1201(
                partial
                    .cast::<u8>()
                    .add(done * AR_HC_HIDDEN * 4)
                    .cast::<c_void>(),
                streams
                    .cast::<u8>()
                    .add(done * kind.row_bytes())
                    .cast::<c_void>(),
                kind,
                wire,
                partial_fmt,
                gates.cast::<u8>().add(done * 4 * 4).cast::<c_void>(),
                (peer_mail + region) as *mut c_void,
                (local_mail + region) as *const c_void,
                self.bank_stride(wire),
                &mail.gen,
                rank == 0,
                n,
                geometry,
                dbg,
            )?;
            done += n;
        }
        Ok(())
    }

    /// Load-time qualification of rank `rank` over BOTH stream kinds (BF16 bits, then F32
    /// words), both partial formats and every wire this transport can use (the launch of `rows`
    /// rows, and under the hybrid mode also one of [`AR_HC_DENSE_MIN_ROWS`] rows that takes the
    /// dense wire), run CONCURRENTLY on both ranks' threads (the kernels rendezvous): see
    /// [`Self::qualify_kind`].
    pub fn qualify(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        rows: usize,
        exchanges: usize,
    ) -> HipResult<()> {
        let mut sizes = vec![rows];
        let dense_rows = rows.max(AR_HC_DENSE_MIN_ROWS);
        if self.mode == ArHcMode::Hybrid
            && self.wire_for(rows) != ArHcWire::Dense
            && self.wire_for(dense_rows.min(self.max_rows)) == ArHcWire::Dense
        {
            sizes.push(dense_rows.min(self.max_rows));
        }
        for kind in StreamKind::ALL {
            for partial in ArHcPartial::ALL {
                for &r in &sizes {
                    self.qualify_kind(rank, gpu, kind, partial, r, exchanges)?;
                }
            }
        }
        Ok(())
    }

    /// Qualification of one stream kind and partial format: `exchanges` eager fused launches
    /// over `rows` rows with deterministic partials, the folded `m` and the HC streams (gates 0,
    /// so the gate is
    /// exactly 1 whatever libm computes) compared bit for bit with the host model. A failure
    /// (including a bounded-wait timeout) returns `Err`, [`Self::disable`]s the route and
    /// leaves the generations unequal, so the caller must [`Self::reset`] both ranks before
    /// reuse.
    pub fn qualify_kind(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        kind: StreamKind,
        partial_fmt: ArHcPartial,
        rows: usize,
        exchanges: usize,
    ) -> HipResult<()> {
        let n = rows * AR_HC_HIDDEN;
        let alloc = |gpu: &mut Gpu, bytes: usize| gpu.hip.malloc(bytes);
        let partial = alloc(gpu, n * 4)?;
        let streams = alloc(gpu, rows * kind.row_bytes())?;
        let gates = alloc(gpu, rows * 16)?;
        let dbg = alloc(gpu, n * 4)?;
        let result = (|| -> HipResult<()> {
            gpu.hip.memset(&gates, 0, rows * 16)?;
            for exchange in 0..exchanges {
                let p0: Vec<f32> = (0..n).map(|i| qualify_partial(exchange, 0, i)).collect();
                let p1: Vec<f32> = (0..n).map(|i| qualify_partial(exchange, 1, i)).collect();
                let mine = if rank == 0 { &p0 } else { &p1 };
                let gates_host = vec![0f32; rows * 4];
                let want_m;
                let mut got_m = vec![0f32; n];
                match kind {
                    StreamKind::Bf16 => {
                        let mut streams_host = vec![0u16; rows * 4 * AR_HC_HIDDEN];
                        for (i, s) in streams_host.iter_mut().enumerate() {
                            *s = rdna_compute::ar_hc_write::bf16_bits(
                                qualify_partial(exchange, 7, i) * 4.0,
                            );
                        }
                        gpu.hip.memcpy_htod(&partial, f32_bytes(mine))?;
                        gpu.hip.memcpy_htod(&streams, u16_bytes(&streams_host))?;
                        self.qualify_write(
                            rank,
                            gpu,
                            &partial,
                            &streams,
                            kind,
                            partial_fmt,
                            &gates,
                            rows,
                            &dbg,
                        )?;
                        let mut want_streams = streams_host.clone();
                        want_m = ar_hc_write_model(
                            partial_fmt,
                            &p0,
                            &p1,
                            &mut want_streams,
                            &gates_host,
                            rows,
                        );
                        gpu.hip.memcpy_dtoh(f32_bytes_mut(&mut got_m), &dbg)?;
                        let mut got_streams = vec![0u16; want_streams.len()];
                        gpu.hip
                            .memcpy_dtoh(u16_bytes_mut(&mut got_streams), &streams)?;
                        if let Some(i) = got_streams
                            .iter()
                            .zip(&want_streams)
                            .position(|(a, b)| a != b)
                        {
                            return Err(err(format!(
                                "ar_hc_exchange: qualification (bf16 streams, {} partials) exchange {exchange}: stream differs at {i}: {:#06x} vs {:#06x}",
                                partial_fmt.name(),
                                got_streams[i], want_streams[i]
                            )));
                        }
                    }
                    StreamKind::F32 => {
                        // Mostly bf16-exact words (what the forward stores), every 7th
                        // unrounded: the kernel rounds on read like `hyper_write_f32`.
                        let mut streams_host = vec![0f32; rows * 4 * AR_HC_HIDDEN];
                        for (i, s) in streams_host.iter_mut().enumerate() {
                            let v = qualify_partial(exchange, 7, i) * 4.0;
                            *s = if i % 7 == 0 {
                                v
                            } else {
                                rdna_compute::ar_hc_write::bf16_round(v)
                            };
                        }
                        gpu.hip.memcpy_htod(&partial, f32_bytes(mine))?;
                        gpu.hip.memcpy_htod(&streams, f32_bytes(&streams_host))?;
                        self.qualify_write(
                            rank,
                            gpu,
                            &partial,
                            &streams,
                            kind,
                            partial_fmt,
                            &gates,
                            rows,
                            &dbg,
                        )?;
                        let mut want_streams = streams_host.clone();
                        want_m = ar_hc_write_model_f32(
                            partial_fmt,
                            &p0,
                            &p1,
                            &mut want_streams,
                            &gates_host,
                            rows,
                        );
                        gpu.hip.memcpy_dtoh(f32_bytes_mut(&mut got_m), &dbg)?;
                        let mut got_streams = vec![0f32; want_streams.len()];
                        gpu.hip
                            .memcpy_dtoh(f32_bytes_mut(&mut got_streams), &streams)?;
                        if let Some(i) = got_streams
                            .iter()
                            .zip(&want_streams)
                            .position(|(a, b)| a.to_bits() != b.to_bits())
                        {
                            return Err(err(format!(
                                "ar_hc_exchange: qualification (f32 streams, {} partials) exchange {exchange}: stream differs at {i}: {} vs {}",
                                partial_fmt.name(),
                                got_streams[i], want_streams[i]
                            )));
                        }
                    }
                }
                for (i, (a, b)) in got_m.iter().zip(&want_m).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        return Err(err(format!(
                            "ar_hc_exchange: qualification ({} streams, {} partials) exchange {exchange}: folded m differs at {i}: {a} vs {b}",
                            kind.name(),
                            partial_fmt.name()
                        )));
                    }
                    debug_assert_eq!(b.to_bits(), partial_fmt.fold(p0[i], p1[i]).to_bits());
                }
            }
            Ok(())
        })();
        let freed = [partial, streams, gates, dbg]
            .into_iter()
            .map(|b| gpu.hip.free(b))
            .fold(Ok(()), |a, b| a.and(b));
        if result.is_err() {
            self.disable();
        }
        result.and(freed)
    }
}

impl ArHcExchange {
    /// One synchronous qualification launch (write, drain, wait-error check).
    #[allow(clippy::too_many_arguments)]
    fn qualify_write(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        partial: &DeviceBuffer,
        streams: &DeviceBuffer,
        kind: StreamKind,
        partial_fmt: ArHcPartial,
        gates: &DeviceBuffer,
        rows: usize,
        dbg: &DeviceBuffer,
    ) -> HipResult<()> {
        // SAFETY: the buffers are live on this device with exactly the extents `kind` and
        // `rows` imply (allocated by `qualify_kind`).
        unsafe {
            self.write(
                rank,
                gpu,
                partial.as_ptr(),
                streams.as_ptr(),
                kind,
                partial_fmt,
                gates.as_ptr(),
                rows,
                dbg.as_ptr(),
            )?;
        }
        gpu.hip.device_synchronize()?;
        gpu.take_device_wait_error()
    }
}

use rdna_compute::ar_hc_write::{
    sp_ag_geometry, sp_ag_rows_per_launch, sp_rs_geometry, sp_slices, SpAgElem, SP_ROW_CAP,
};

/// Sequence-parallel launches (`rdna_compute::ar_hc_write::sp_*`) on the DENSE region of the same
/// mailboxes and the same generation words as [`ArHcExchange::write`], so SP and all-reduce
/// launches interleave freely as long as both ranks issue mirrored launches in the same order.
/// The transport must be in [`ArHcMode::Dense`] or [`ArHcMode::Hybrid`] (it has a dense region).
impl ArHcExchange {
    /// Whether this transport has the dense region the SP kernels need.
    pub fn sp_available(&self) -> bool {
        self.uses_wire(ArHcWire::Dense)
    }

    /// Most rows of each range one SP launch of this transport serves.
    pub fn sp_row_cap(&self) -> usize {
        self.max_rows.min(SP_ROW_CAP)
    }

    /// Resolve every SP kernel on `gpu` (JIT is not capture-safe); call at load next to
    /// [`Self::register`].
    pub fn sp_ensure_kernels(&self, gpu: &mut Gpu) -> HipResult<()> {
        for kind in StreamKind::ALL {
            for partial in ArHcPartial::ALL {
                gpu.ensure_sp_hc_rs_gfx1201_kernel(kind, partial)?;
            }
        }
        for elem in SpAgElem::ALL {
            gpu.ensure_sp_ag_rows_gfx1201_kernel(elem)?;
        }
        Ok(())
    }

    /// Dense-region addresses `(peer mailbox, local mailbox)` of `rank` after the usual launch
    /// checks.
    fn sp_endpoints(&self, rank: usize) -> HipResult<(usize, usize)> {
        if !self.enabled() {
            return Err(err("ar_hc_exchange: route is disabled".into()));
        }
        if !self.sp_available() {
            return Err(err(
                "ar_hc_exchange: the SP launches need a dense mailbox region (mode dense or hybrid)"
                    .into(),
            ));
        }
        if rank >= 2 || !self.peers_enabled[rank].load(Ordering::Acquire) {
            return Err(err(format!(
                "ar_hc_exchange: rank {rank} has no peer access"
            )));
        }
        let peer = 1 - rank;
        let (peer_mail, local_mail, gen) = (
            self.mailbox_ptr[peer].load(Ordering::Acquire),
            self.mailbox_ptr[rank].load(Ordering::Acquire),
            self.gen_ptr[rank].load(Ordering::Acquire),
        );
        if peer_mail == 0 || local_mail == 0 || gen == 0 {
            return Err(err(
                "ar_hc_exchange: SP launch before both ranks registered".into(),
            ));
        }
        let region = self.region_offset(ArHcWire::Dense);
        Ok((peer_mail + region, local_mail + region))
    }

    /// Reduce-scatter fused into the HC write of `rank`: the own rows `own = (row0, rows)` of
    /// the full-tile `partial` / `streams` / `gates` are folded and written, the peer rows
    /// `peer` are pushed to the peer rank (whose own range it is) in the `partial_fmt` wire
    /// format. Ranges longer than [`Self::sp_row_cap`] launch in mirrored slices ([`sp_slices`]),
    /// each its own generation. The own rows equal those of [`Self::write`] over the same
    /// partials in the same format. `mixed_dbg` (oracle, nullable) receives `m` of the own rows
    /// at the full-tile offset.
    ///
    /// # Safety
    /// The pointers must address live device memory of the full tile on `gpu`'s device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sp_rs_write(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        partial: *mut c_void,
        streams: *mut c_void,
        kind: StreamKind,
        partial_fmt: ArHcPartial,
        gates: *const c_void,
        own: (usize, usize),
        peer: (usize, usize),
        mixed_dbg: *mut c_void,
    ) -> HipResult<()> {
        let (peer_mail, local_mail) = self.sp_endpoints(rank)?;
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: SP launch after release".into()))?;
        gpu.bind_thread()?;
        let stride = self.bank_stride(ArHcWire::Dense);
        for ((own_off, own_n), (peer_off, peer_n)) in sp_slices(own.1, peer.1, self.sp_row_cap()) {
            let geometry = sp_rs_geometry(own_n, peer_n).ok_or_else(|| {
                err(format!(
                    "ar_hc_exchange: no SP launch geometry for {own_n} own / {peer_n} peer rows"
                ))
            })?;
            self.host_gate(rank)?;
            // SAFETY: forwarded from this function's contract; the row ranges lie inside the
            // caller's tile.
            gpu.sp_hc_rs_write_gfx1201(
                partial,
                streams,
                kind,
                partial_fmt,
                gates,
                peer_mail as *mut c_void,
                local_mail as *const c_void,
                stride,
                &mail.gen,
                rank == 0,
                (own.0 + own_off, own_n),
                (peer.0 + peer_off, peer_n),
                geometry,
                mixed_dbg,
            )?;
        }
        Ok(())
    }

    /// All-gather of rows of `buf` (`[rows][row_elems]` of `elem`): the own rows are sent, the
    /// peer rows received in place. Longer ranges launch in mirrored slices.
    ///
    /// # Safety
    /// `buf` must address the live full buffer on `gpu`'s device.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sp_ag_rows(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        buf: *mut c_void,
        elem: SpAgElem,
        row_elems: usize,
        own: (usize, usize),
        peer: (usize, usize),
    ) -> HipResult<()> {
        let (peer_mail, local_mail) = self.sp_endpoints(rank)?;
        let stride = self.bank_stride(ArHcWire::Dense);
        let cap = sp_ag_rows_per_launch(row_elems, stride).min(self.sp_row_cap());
        if cap == 0 {
            return Err(err(format!(
                "ar_hc_exchange: row_elems {row_elems} does not fit a dense bank of {stride} bytes"
            )));
        }
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: SP launch after release".into()))?;
        gpu.bind_thread()?;
        for ((own_off, own_n), (peer_off, peer_n)) in sp_slices(own.1, peer.1, cap) {
            let geometry = sp_ag_geometry(own_n, peer_n, row_elems).ok_or_else(|| {
                err(format!(
                    "ar_hc_exchange: no SP all-gather geometry for {own_n} own / {peer_n} peer rows of {row_elems}"
                ))
            })?;
            self.host_gate(rank)?;
            // SAFETY: as above.
            gpu.sp_ag_rows_gfx1201(
                buf,
                elem,
                row_elems,
                peer_mail as *mut c_void,
                local_mail as *const c_void,
                stride,
                &mail.gen,
                (own.0 + own_off, own_n),
                (peer.0 + peer_off, peer_n),
                geometry,
            )?;
        }
        Ok(())
    }

    /// [`Self::sp_ag_rows`] for ranges of different lengths (a work-balanced split). Every
    /// launch carries rows in both directions ([`sp_lockstep_slices`]), so each launch waits
    /// for the peer's matching launch and the two ranks stay in lockstep.
    ///
    /// [`sp_slices`] pads the shorter range with empty chunks. A launch with no peer rows only
    /// pushes and does not wait, so the rank with the longer range runs ahead. Its push two
    /// launches later reuses a mailbox bank the peer may still be polling, which overwrites
    /// granules not yet consumed or leaves the peer waiting on a re-armed slot. Mirrored on
    /// the peer (it computes the swapped list).
    ///
    /// # Safety
    /// As [`Self::sp_ag_rows`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sp_ag_rows_lockstep(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        buf: *mut c_void,
        elem: SpAgElem,
        row_elems: usize,
        own: (usize, usize),
        peer: (usize, usize),
    ) -> HipResult<()> {
        let cap = sp_ag_rows_per_launch(row_elems, self.bank_stride(ArHcWire::Dense))
            .min(self.sp_row_cap());
        let slices = sp_lockstep_slices(own.1, peer.1, cap).ok_or_else(|| {
            err(format!(
                "ar_hc_exchange: no lockstep all-gather of {own:?} / {peer:?} rows of {row_elems} \
                 (launch cap {cap} rows)"
            ))
        })?;
        for ((own_off, own_n), (peer_off, peer_n)) in slices {
            // SAFETY: forwarded from this function's contract; each slice lies inside its range
            // and fits one launch (`own_n`, `peer_n` <= cap).
            unsafe {
                self.sp_ag_rows(
                    rank,
                    gpu,
                    buf,
                    elem,
                    row_elems,
                    (own.0 + own_off, own_n),
                    (peer.0 + peer_off, peer_n),
                )?;
            }
        }
        Ok(())
    }
}

/// Lockstep slices of an all-gather between an own range of `own_rows` and a peer range of
/// `peer_rows` rows: `k = max(ceil(own_rows / cap), ceil(peer_rows / cap))` launches, each range
/// cut into `k` near-equal non-empty chunks (`floor(i * len / k)` boundaries). Items are
/// `((own_off, own_n), (peer_off, peer_n))` relative to the range starts, every `n` in
/// `1..=cap`. The list is a function of the two lengths only, so the peer's list is this one
/// with the pairs swapped. `None` when `cap == 0`, a range is empty, or a range is shorter than
/// `k`.
pub fn sp_lockstep_slices(
    own_rows: usize,
    peer_rows: usize,
    cap: usize,
) -> Option<Vec<((usize, usize), (usize, usize))>> {
    if cap == 0 || own_rows == 0 || peer_rows == 0 {
        return None;
    }
    let k = own_rows.div_ceil(cap).max(peer_rows.div_ceil(cap));
    if own_rows < k || peer_rows < k {
        return None;
    }
    let chunk = |len: usize, i: usize| {
        let (a, b) = (i * len / k, (i + 1) * len / k);
        (a, b - a)
    };
    Some(
        (0..k)
            .map(|i| (chunk(own_rows, i), chunk(peer_rows, i)))
            .collect(),
    )
}

/// Host mirror of one qualification partial element: an fp32 value in [-1, 1) that differs
/// per (exchange, rank, index).
fn qualify_partial(exchange: usize, rank: usize, i: usize) -> f32 {
    let mut h = (i as u32).wrapping_mul(0x9E37_79B1)
        ^ (exchange as u32).wrapping_mul(0x85EB_CA6B)
        ^ (rank as u32 + 1).wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    ((h >> 8) as f32) * f32::from_bits(0x3400_0000) - 1.0
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is read as plain bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above, exclusive.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
fn u16_bytes(v: &[u16]) -> &[u8] {
    // SAFETY: u16 has no padding.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
fn u16_bytes_mut(v: &mut [u16]) -> &mut [u8] {
    // SAFETY: as above, exclusive.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A transport with the gate forced on or off whatever the process environment says.
    fn gated(on: bool) -> Arc<ArHcExchange> {
        let mut x = ArHcExchange::with_mode(1, ArHcMode::Tagged).unwrap();
        x.host_gate_on = on;
        Arc::new(x)
    }

    /// The kill switch launches ungated: no arrival is counted and no peer is awaited.
    #[test]
    fn host_gate_off_never_waits() {
        let x = gated(false);
        for _ in 0..3 {
            x.host_gate(0).unwrap();
        }
        assert_eq!(x.arrivals[0].load(Ordering::Acquire), 0);
    }

    /// The host gate lets a launch through only once the peer's host arrived at the same
    /// launch: a rank that runs ahead waits out a stalled peer (here 300 ms) instead of
    /// launching, and mirrored sequences cross in order.
    #[test]
    fn host_gate_waits_for_a_stalled_peer_and_crosses_in_order() {
        let x = gated(true);
        let peer = {
            let x = x.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(300));
                for _ in 0..1000 {
                    x.host_gate(1).unwrap();
                }
            })
        };
        let start = Instant::now();
        x.host_gate(0).unwrap();
        assert!(start.elapsed() >= std::time::Duration::from_millis(250));
        for _ in 1..1000 {
            x.host_gate(0).unwrap();
            // never ahead of the peer's arrival
            assert!(x.arrivals[1].load(Ordering::Acquire) >= x.arrivals[0].load(Ordering::Acquire));
        }
        peer.join().unwrap();
    }

    /// A failed peer never arrives: the pool's abort releases the waiter with an error, and
    /// `skip_launch` (fault oracles) arrives without a launch.
    #[test]
    fn host_gate_abort_releases_and_skip_launch_arrives() {
        let x = gated(true);
        let aborter = {
            let x = x.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                x.abort();
            })
        };
        let start = Instant::now();
        let e = x.host_gate(0).unwrap_err().to_string();
        assert!(e.contains("aborted"), "{e}");
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
        aborter.join().unwrap();
        // `reset` state: counts zeroed, abort cleared (no device here: set directly)
        x.arrivals[0].store(0, Ordering::Release);
        x.gate_aborted.store(false, Ordering::Release);
        x.skip_launch(1);
        x.host_gate(0).unwrap();
    }

    #[test]
    fn sizes_and_bounds() {
        assert!(ArHcExchange::new(0).is_err());
        assert!(ArHcExchange::new(AR_HC_ROW_CAP + 1).is_err());
        let t = 32 * 640 * 32usize;
        let d16 = 32 * 640 * 16usize;
        let x = ArHcExchange::with_mode(32, ArHcMode::Tagged).unwrap();
        assert_eq!(x.bank_stride(ArHcWire::Tagged), t);
        assert_eq!(x.rank_bytes(), 2 * t + 256);
        for rows in [1usize, 16, 32] {
            assert_eq!(x.wire_for(rows), ArHcWire::Tagged);
        }
        let d = ArHcExchange::with_mode(32, ArHcMode::Dense).unwrap();
        assert_eq!(d.bank_stride(ArHcWire::Dense), d16);
        assert_eq!(d.region_offset(ArHcWire::Dense), 2 * t);
        assert_eq!(d.rank_bytes(), 2 * t + 2 * d16 + 256);
        assert_eq!(d.wire_for(1), ArHcWire::Dense);
        let h = ArHcExchange::with_mode(32, ArHcMode::Hybrid).unwrap();
        assert_eq!(h.rank_bytes(), d.rank_bytes());
        assert_eq!(h.wire_for(AR_HC_DENSE_MIN_ROWS - 1), ArHcWire::Tagged);
        assert_eq!(h.wire_for(AR_HC_DENSE_MIN_ROWS), ArHcWire::Dense);
        assert_eq!(ArHcWire::Dense.idle_fill(), 0xFF);
        assert_eq!(ArHcWire::Tagged.idle_fill(), 0x00);
        assert!(x.enabled());
        x.disable();
        assert!(!x.enabled());
    }

    /// The qualification partials stay in [-1, 1) and differ between ranks and exchanges.
    #[test]
    fn qualification_partials_are_distinct_and_bounded() {
        let mut seen = std::collections::HashSet::new();
        for e in 0..4 {
            for r in 0..2 {
                for i in 0..64 {
                    let v = qualify_partial(e, r, i);
                    assert!((-1.0..1.0).contains(&v));
                    seen.insert(v.to_bits());
                }
            }
        }
        assert!(seen.len() > 400);
    }

    /// Lockstep slices: both directions carry rows in every launch, every chunk fits the cap,
    /// the chunks tile both ranges in order, and the peer's list is the swapped one.
    #[test]
    fn lockstep_slices_tile_both_ranges_in_every_launch() {
        for (a, b, cap) in [
            (2896, 1200, 256),
            (2048, 2048, 256),
            (1916, 1915, 256),
            (257, 2, 256),
            (5793, 2399, 256),
            (7, 4, 2),
            (1, 1, 1),
        ] {
            let s = sp_lockstep_slices(a, b, cap).unwrap();
            let k = a.div_ceil(cap).max(b.div_ceil(cap));
            assert_eq!(s.len(), k, "{a}/{b}/{cap}");
            let (mut na, mut nb) = (0, 0);
            for &((oa, ca), (ob, cb)) in &s {
                assert!(
                    (1..=cap).contains(&ca) && (1..=cap).contains(&cb),
                    "{a}/{b}/{cap}"
                );
                assert_eq!((oa, ob), (na, nb));
                na += ca;
                nb += cb;
            }
            assert_eq!((na, nb), (a, b));
            let swapped: Vec<_> = sp_lockstep_slices(b, a, cap)
                .unwrap()
                .into_iter()
                .map(|(x, y)| (y, x))
                .collect();
            assert_eq!(swapped, s);
        }
        // A range shorter than the launch count, an empty range or a zero cap has none.
        assert!(sp_lockstep_slices(600, 2, 256).is_none());
        assert!(sp_lockstep_slices(0, 5, 256).is_none());
        assert!(sp_lockstep_slices(5, 5, 0).is_none());
    }
}
