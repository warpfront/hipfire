// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-scoped mailboxes and dispatch of the fused TP2 all-reduce + HC write
//! (`rdna_compute::ar_hc_write`, kernel `ar_hc_write_gfx1201`) for the Flash-Next EP2 trunk
//! shard (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`).
//!
//! Each rank owns one fine-grained VRAM mailbox that the PEER writes (`hipDeviceMallocFinegrained`):
//! a TAGGED region (two banks of 16-byte tagged granules) and, unless `HIPFIRE_EP_AR_WIRE_DENSE=0`,
//! a DENSE region (two banks of 8-byte untagged granules, EMPTY = all ones; launches of at
//! least [`AR_HC_DENSE_MIN_ROWS`] rows use it, see the kernel header for the argument), and one 256-byte block holding the
//! device-resident generation word and arrival ticket. Mailboxes are registered per rank
//! (any thread, that rank's `Gpu`) BEFORE peer access is enabled
//! ([`ArHcExchange::enable_peers`]), then [`ArHcExchange::qualify`] runs a short eager
//! exchange on both ranks compared bit for bit with the host model
//! (`ar_hc_write_model` / `ar_hc_fold`); a failed qualification turns the route off for
//! the process ([`ArHcExchange::enabled`]) and the caller keeps the incumbent EP2 path.
//!
//! Protocol constraints: both ranks issue the same sequence of [`ArHcExchange::write`]
//! calls with the same row counts (the generation words advance in lock step); a wait error
//! (`Gpu::take_device_wait_error`) leaves the generations unequal, so the host must call
//! [`ArHcExchange::reset`] on both drained ranks before the next launch.

use hip_bridge::{DeviceBuffer, HipError, HipResult, HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED};
use rdna_compute::ar_hc_write::{
    ar_hc_bank_bytes_for, ar_hc_fold, ar_hc_geometry, ar_hc_write_model, ar_hc_write_model_f32,
    ArHcWire, StreamKind, AR_HC_GEN_BYTES, AR_HC_HIDDEN, AR_HC_ROW_CAP,
};
use rdna_compute::Gpu;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

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
}

/// Which mailbox wires a transport uses (see the kernel header for the two formats).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArHcMode {
    /// Every launch on the 16-byte tagged wire (the kill-switch format).
    Tagged,
    /// Every launch on the dense 8-byte wire (oracles and benches).
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
pub const AR_HC_DENSE_MIN_ROWS: usize = 16;

/// `HIPFIRE_EP_AR_WIRE_DENSE` (developer, default on): `0` selects the original 16-byte tagged
/// granule wire for every launch (the kill switch); anything else the hybrid default.
pub const AR_WIRE_DENSE_ENV: &str = "HIPFIRE_EP_AR_WIRE_DENSE";

/// The mode the process selects (`HIPFIRE_EP_AR_WIRE_DENSE`).
pub fn ar_hc_mode_from_env() -> ArHcMode {
    if hipfire_config::developer_bool(AR_WIRE_DENSE_ENV, true) {
        ArHcMode::Hybrid
    } else {
        ArHcMode::Tagged
    }
}

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
        })
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
                    gpu.ensure_ar_hc_write_gfx1201_kernel(kind, wire)?;
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
    /// matches a live generation; dense region: every slot EMPTY).
    /// BOTH ranks must be drained and reset before either launches again.
    pub fn reset(&self, rank: usize, gpu: &mut Gpu) -> HipResult<()> {
        let slot = self.ranks[rank].lock().unwrap_or_else(|p| p.into_inner());
        let mail = slot
            .as_ref()
            .ok_or_else(|| err("ar_hc_exchange: reset before register".into()))?;
        gpu.bind_thread()?;
        gpu.hip.memset(&mail.gen, 0, AR_HC_GEN_BYTES)?;
        self.idle_fill_mailbox(gpu, &mail.mailbox)
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
    pub fn download_mailbox(&self, rank: usize, gpu: &mut Gpu, wire: ArHcWire) -> HipResult<Vec<u8>> {
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

    /// One fused all-reduce + HC write of `rows` rows on `rank`: `partial` `[rows, 2560]` f32,
    /// `streams` `[rows, 4, 2560]` of `kind` (BF16 bits or F32 words, in place), `gates` `[rows, 4]` f32 (device
    /// pointers; rows beyond [`AR_HC_ROW_CAP`] or `max_rows` are launched in slices, each
    /// its own generation, in order, identically on both ranks).
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
            let geometry = ar_hc_geometry(n)
                .ok_or_else(|| err(format!("ar_hc_exchange: no launch geometry for {n} rows")))?;
            let dbg = if mixed_dbg.is_null() {
                mixed_dbg
            } else {
                mixed_dbg
                    .cast::<u8>()
                    .add(done * AR_HC_HIDDEN * 4)
                    .cast::<c_void>()
            };
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
    /// words) and every wire this transport can use (the launch of `rows` rows, and under the
    /// hybrid mode also one of [`AR_HC_DENSE_MIN_ROWS`] rows that takes the dense wire), run
    /// CONCURRENTLY on both ranks' threads (the kernels rendezvous): see
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
            for &r in &sizes {
                self.qualify_kind(rank, gpu, kind, r, exchanges)?;
            }
        }
        Ok(())
    }

    /// Qualification of one stream kind: `exchanges` eager fused launches over `rows` rows with
    /// deterministic partials, the folded `m` and the HC streams (gates 0, so the gate is
    /// exactly 1 whatever libm computes) compared bit for bit with the host model. A failure
    /// (including a bounded-wait timeout) returns `Err`, [`Self::disable`]s the route and
    /// leaves the generations unequal, so the caller must [`Self::reset`] both ranks before
    /// reuse.
    pub fn qualify_kind(
        &self,
        rank: usize,
        gpu: &mut Gpu,
        kind: StreamKind,
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
                            rank, gpu, &partial, &streams, kind, &gates, rows, &dbg,
                        )?;
                        let mut want_streams = streams_host.clone();
                        want_m = ar_hc_write_model(&p0, &p1, &mut want_streams, &gates_host, rows);
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
                                "ar_hc_exchange: qualification (bf16) exchange {exchange}: stream differs at {i}: {:#06x} vs {:#06x}",
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
                            rank, gpu, &partial, &streams, kind, &gates, rows, &dbg,
                        )?;
                        let mut want_streams = streams_host.clone();
                        want_m =
                            ar_hc_write_model_f32(&p0, &p1, &mut want_streams, &gates_host, rows);
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
                                "ar_hc_exchange: qualification (f32) exchange {exchange}: stream differs at {i}: {} vs {}",
                                got_streams[i], want_streams[i]
                            )));
                        }
                    }
                }
                for (i, (a, b)) in got_m.iter().zip(&want_m).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        return Err(err(format!(
                            "ar_hc_exchange: qualification ({}) exchange {exchange}: folded m differs at {i}: {a} vs {b}",
                            kind.name()
                        )));
                    }
                    debug_assert_eq!(b.to_bits(), ar_hc_fold(p0[i], p1[i]).to_bits());
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
                gates.as_ptr(),
                rows,
                dbg.as_ptr(),
            )?;
        }
        gpu.hip.device_synchronize()?;
        gpu.take_device_wait_error()
    }
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

    #[test]
    fn sizes_and_bounds() {
        assert!(ArHcExchange::new(0).is_err());
        assert!(ArHcExchange::new(AR_HC_ROW_CAP + 1).is_err());
        let t = 32 * 640 * 16usize;
        let d8 = 32 * 640 * 8usize;
        let x = ArHcExchange::with_mode(32, ArHcMode::Tagged).unwrap();
        assert_eq!(x.bank_stride(ArHcWire::Tagged), t);
        assert_eq!(x.rank_bytes(), 2 * t + 256);
        for rows in [1usize, 16, 32] {
            assert_eq!(x.wire_for(rows), ArHcWire::Tagged);
        }
        let d = ArHcExchange::with_mode(32, ArHcMode::Dense).unwrap();
        assert_eq!(d.bank_stride(ArHcWire::Dense), d8);
        assert_eq!(d.region_offset(ArHcWire::Dense), 2 * t);
        assert_eq!(d.rank_bytes(), 2 * t + 2 * d8 + 256);
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
}
