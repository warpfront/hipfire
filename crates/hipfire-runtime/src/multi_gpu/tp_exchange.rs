// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq
// hipfire — see LICENSE and NOTICE in the project root.

//! 16-bit partial-sum exchange for the dense tensor-parallel residual
//! reduction (`HIPFIRE_TP_EXCHANGE`).
//!
//! A row-parallel stage (attention out-projection, FFN down-projection)
//! leaves one f32 partial per rank; the residual stream `x` is f32 and
//! replicated. The exchange moves each partial between ranks as 16 bits:
//!
//! 1. the producing rank converts its f32 partial to 16 bits once
//!    (`tp_pack16_*`, round-to-nearest-even);
//! 2. peers read that 16-bit copy (direct P2P read for TP2, gather copy for
//!    the rooted TP>2 fold);
//! 3. every rank, the producer included, sums the rounded, rank-ordered
//!    partials in f32 and adds the sum to its f32 residual:
//!    `x + (((r(p0) + r(p1)) + r(p2)) + r(p3))`, `r` = the 16-bit round trip
//!    (`tp_fold16_*`, `tp_graph_signal_reduce16_*`).
//!
//! The residual stream, the sum and the broadcast of the rooted route stay
//! f32. Every rank adds identical bytes, so the replicated residual stays
//! bit-identical across ranks, and the hand and generic dense TP routes (which
//! both reduce through [`Gpus::all_reduce_sum_f32_peer_add_routed`] or the
//! executor's `peer_reduce`) stay byte-identical to each other. The result is
//! NOT byte-identical to the f32 exchange: it is a numeric route judged by
//! KLD (see `docs/multi-gpu.md`).
//!
//! Scope: the peer-access transport between distinct devices with 2..=4 ranks
//! and a row count x dim that is a multiple of 8. The host-staged transport,
//! same-device emulated ranks, other rank counts and the non-add all-reduces
//! (expert-parallel, leased) keep the f32 exchange.

use std::ffi::c_void;
use std::sync::LazyLock;

use hip_bridge::{DeviceBuffer, HipError, HipResult};
pub use rdna_compute::norm::TpExchange16;

use super::{tp_graph_payload_offset, tp_peer_direct_enabled, Gpus};

/// Developer knob selecting the dense TP partial exchange format.
pub const TP_EXCHANGE_ENV: &str = "HIPFIRE_TP_EXCHANGE";

/// Largest rank count the 16-bit fold kernels take (four sources; the rooted
/// fold's root contributes its own partial as an f32 first term).
const MAX_EXCHANGE_RANKS: usize = 4;

/// What `HIPFIRE_TP_EXCHANGE` selects: a fixed exchange for every reduction
/// (`None`: f32) or the measured policy ([`auto_exchange`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpExchangeMode {
    Auto,
    Fixed(Option<TpExchange16>),
}

/// What `HIPFIRE_TP_EXCHANGE` resolves to when unset.
const DEFAULT_MODE: TpExchangeMode = TpExchangeMode::Auto;

/// The `auto` policy for a reduction of `rows` rows over `tp` ranks: bf16 for
/// multi-row exchanges (prefill, verify) at any degree and for single-row
/// decode at TP>=3; f32 for single-row decode at TP2, where the extra pack
/// launch costs more than the halved peer read saves (measured -1.4%).
pub fn auto_exchange(rows: usize, tp: usize) -> Option<TpExchange16> {
    if rows <= 1 && tp <= 2 {
        None
    } else {
        Some(TpExchange16::Bf16)
    }
}

impl TpExchangeMode {
    /// The format for a `rows`-row reduction over `tp` ranks (`None`: f32).
    pub fn resolve(self, rows: usize, tp: usize) -> Option<TpExchange16> {
        match self {
            Self::Auto => auto_exchange(rows, tp),
            Self::Fixed(fmt) => fmt,
        }
    }

    /// The widest format any reduction may use, for kernel preparation.
    pub fn widest(self) -> Option<TpExchange16> {
        match self {
            Self::Auto => Some(TpExchange16::Bf16),
            Self::Fixed(fmt) => fmt,
        }
    }
}

/// Parse `HIPFIRE_TP_EXCHANGE`: `auto`, `f32`, `bf16`, `f16`; unset or empty
/// takes the default (`auto`). Anything else is an error.
pub fn parse_tp_exchange(raw: Option<&str>) -> Result<TpExchangeMode, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(DEFAULT_MODE),
        Some("auto") => Ok(TpExchangeMode::Auto),
        Some("f32") => Ok(TpExchangeMode::Fixed(None)),
        Some("bf16") => Ok(TpExchangeMode::Fixed(Some(TpExchange16::Bf16))),
        Some("f16") => Ok(TpExchangeMode::Fixed(Some(TpExchange16::F16))),
        Some(other) => Err(format!(
            "{TP_EXCHANGE_ENV}={other:?} is not one of auto, f32, bf16, f16"
        )),
    }
}

/// The requested exchange mode, resolved once per process. An unrecognised
/// value is reported on stderr and falls back to f32, the lossless route.
pub fn tp_exchange_mode() -> TpExchangeMode {
    static MODE: LazyLock<TpExchangeMode> = LazyLock::new(|| {
        let raw = hipfire_config::developer_var(TP_EXCHANGE_ENV).ok();
        parse_tp_exchange(raw.as_deref()).unwrap_or_else(|error| {
            eprintln!("  {error}; using the f32 exchange");
            TpExchangeMode::Fixed(None)
        })
    });
    *MODE
}

/// The widest format the process may exchange (`None`: f32 only); what kernel
/// preparation compiles.
pub fn tp_exchange_format() -> Option<TpExchange16> {
    tp_exchange_mode().widest()
}

/// What the dense TP load log reports for the partial exchange: the requested
/// mode when the transport is peer access, `f32` otherwise (the host-staged
/// transport never uses the 16-bit route).
pub fn tp_exchange_label(peer_transport: bool) -> &'static str {
    match (peer_transport, tp_exchange_mode()) {
        (true, TpExchangeMode::Auto) => "auto (bf16; f32 for TP2 single-row decode)",
        (true, TpExchangeMode::Fixed(Some(TpExchange16::Bf16))) => "bf16",
        (true, TpExchangeMode::Fixed(Some(TpExchange16::F16))) => "f16",
        _ => "f32",
    }
}

/// Bytes between the two slots of a rank's pack buffer: a `count`-element
/// 16-bit partial rounded to the 256-byte segment alignment.
fn exch16_slot_bytes(count: usize) -> HipResult<usize> {
    count
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_next_multiple_of(256))
        .ok_or_else(|| HipError::new(0, "TP 16-bit exchange byte count overflow"))
}

impl Gpus {
    /// The 16-bit exchange format for a `rows`-row, `count`-element
    /// reduction on this mesh, `None` when the f32 exchange applies (mode,
    /// rank count, shared devices, no peer access, missing rank streams or a
    /// count that is not a multiple of 8).
    pub fn tp_exchange16_for(&self, rows: usize, count: usize) -> Option<TpExchange16> {
        let fmt = tp_exchange_mode().resolve(rows, self.devices.len())?;
        self.tp_exchange16_admitted(count).then_some(fmt)
    }

    fn tp_exchange16_admitted(&self, count: usize) -> bool {
        let n = self.devices.len();
        count > 0
            && count.is_multiple_of(8)
            && (2..=MAX_EXCHANGE_RANKS).contains(&n)
            && self.peer_access_enabled
            && self.devices.iter().all(|d| d.active_stream.is_some())
            && self.devices_are_distinct()
    }

    /// The generic executor's peer transport, which does not depend on
    /// `peer_access_enabled`: the format for `rows` rows / `count` elements
    /// over `n` ranks that live on distinct devices.
    pub(crate) fn tp_exchange16_executor(&self, rows: usize, count: usize) -> Option<TpExchange16> {
        let n = self.devices.len();
        let fmt = tp_exchange_mode().resolve(rows, n)?;
        (count > 0
            && count.is_multiple_of(8)
            && (2..=MAX_EXCHANGE_RANKS).contains(&n)
            && self.devices.iter().all(|d| d.active_stream.is_some())
            && self.devices_are_distinct())
        .then_some(fmt)
    }

    fn devices_are_distinct(&self) -> bool {
        self.devices.iter().enumerate().all(|(i, a)| {
            self.devices[i + 1..]
                .iter()
                .all(|b| a.device_id != b.device_id)
        })
    }

    /// Ensure every rank owns a pack buffer of two slots of at least
    /// `slot_bytes`. Grows by freeing the old set after draining the devices;
    /// never call it while a graph is capturing.
    pub(super) fn ensure_tp_exch16(&mut self, slot_bytes: usize) -> HipResult<()> {
        if !self.tp_exch16.is_empty() && self.tp_exch16_slot_bytes >= slot_bytes {
            return Ok(());
        }
        self.free_tp_exch16()?;
        let total = slot_bytes
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "TP 16-bit exchange buffer overflow"))?;
        let mut buffers: Vec<DeviceBuffer> = Vec::with_capacity(self.devices.len());
        for rank in 0..self.devices.len() {
            let allocated = self.devices[rank]
                .bind_thread()
                .and_then(|_| self.devices[rank].hip.malloc(total));
            match allocated {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for (owner, buffer) in buffers.into_iter().enumerate() {
                        let _ = self.devices[owner].bind_thread();
                        let _ = self.devices[owner].hip.free(buffer);
                    }
                    return Err(error);
                }
            }
        }
        self.tp_exch16 = buffers;
        self.tp_exch16_slot_bytes = slot_bytes;
        Ok(())
    }

    /// Drain every device, then free the pack buffers. Idempotent.
    pub(super) fn free_tp_exch16(&mut self) -> HipResult<()> {
        if self.tp_exch16.is_empty() {
            self.tp_exch16_slot_bytes = 0;
            return Ok(());
        }
        for rank in 0..self.tp_exch16.len() {
            self.devices[rank].bind_thread()?;
            self.devices[rank].hip.device_synchronize()?;
        }
        let mut first_error: Option<HipError> = None;
        for (rank, buffer) in std::mem::take(&mut self.tp_exch16).into_iter().enumerate() {
            let freed = self.devices[rank]
                .bind_thread()
                .and_then(|_| self.devices[rank].hip.free(buffer));
            if let Err(error) = freed {
                first_error.get_or_insert(error);
            }
        }
        self.tp_exch16_slot_bytes = 0;
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn exch16_ptr(&self, rank: usize, slot: usize) -> *mut c_void {
        // SAFETY: `slot` is 0 or 1 and the buffer holds two `slot_bytes` slots.
        unsafe {
            self.tp_exch16[rank]
                .as_ptr()
                .cast::<u8>()
                .add(slot * self.tp_exch16_slot_bytes)
                .cast::<c_void>()
        }
    }

    /// The 16-bit twin of the shared residual reduction
    /// [`Gpus::all_reduce_sum_f32_peer_add_routed`]: `residual[r] +=
    /// r(partial[0]) + r(partial[1]) + ...` in rank order on every rank.
    /// Two ranks take the symmetric direct-peer read (kill switch
    /// `HIPFIRE_TP_PEER_DIRECT=0`, alternating slots / whole-token graph
    /// capture as for the f32 route); other rank counts the rooted
    /// gather/fold/broadcast.
    pub(super) fn all_reduce_peer_add16(
        &mut self,
        fmt: TpExchange16,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
        partial_slot: Option<usize>,
    ) -> HipResult<()> {
        if partial_slot.is_some_and(|slot| slot > 1) {
            return Err(HipError::new(
                0,
                "all_reduce_peer_add16: partial_slot must be 0 or 1",
            ));
        }
        if self.devices.len() == 2 && tp_peer_direct_enabled() {
            let alternating = partial_slot.filter(|_| self.tp_peer_alternating_effective());
            return self.direct_add16(fmt, partials, residuals, count, alternating);
        }
        self.rooted_add16(fmt, partials, residuals, count)
    }

    fn check_add16_args(
        &self,
        operation: &str,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let n = self.devices.len();
        if partials.len() != n || residuals.len() != n {
            return Err(HipError::new(
                0,
                &format!(
                    "{operation}: partials.len()={} residuals.len()={} != n_devices={n}",
                    partials.len(),
                    residuals.len()
                ),
            ));
        }
        if self.active_peer_lease.is_some()
            || self.peer_lease_quarantined
            || !self.peer_lease_buffers.is_empty()
        {
            return Err(HipError::new(
                0,
                &format!("{operation}: peer scratch is leased — use the leased reduction API"),
            ));
        }
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::new(0, &format!("{operation}: byte count overflow")))?;
        for (what, buffers) in [("partial", partials), ("residual", residuals)] {
            for (rank, buffer) in buffers.iter().enumerate() {
                if buffer.size() < bytes {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "{operation}: {what} rank {rank} has {} bytes, needs {bytes}",
                            buffer.size()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Best-effort drain of every active rank stream after a failed enqueue
    /// (the caller may already have producer work queued).
    fn drain_rank_streams(&self) {
        for gpu in &self.devices {
            let _ = gpu.bind_thread();
            if let Some(stream) = gpu.active_stream.as_ref() {
                let _ = gpu.hip.stream_synchronize(stream);
            }
        }
    }

    /// TP2 symmetric direct-peer read with a 16-bit exchange. Per rank r:
    /// pack the f32 partial into the rank's slot, record the producer event,
    /// wait the peer's, then `x_r += r(p0) + r(p1)` reading both 16-bit
    /// slots (the peer's through its P2P mapping). `alternating_slot` selects
    /// the producer-only protocol of the f32 route; inside whole-token graph
    /// capture the fused signal + push-to-peer-shadow + reduce kernel is
    /// recorded instead.
    fn direct_add16(
        &mut self,
        fmt: TpExchange16,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
        alternating_slot: Option<usize>,
    ) -> HipResult<()> {
        let operation = if alternating_slot.is_some() {
            "all_reduce_peer_direct_add16_alternating"
        } else {
            "all_reduce_peer_direct_add16"
        };
        self.check_add16_args(operation, partials, residuals, count)?;
        let n = self.devices.len();
        debug_assert_eq!(n, 2);
        let slot_bytes = exch16_slot_bytes(count)?;
        let bytes16 = count * 2;

        if let Some(slot) = alternating_slot
            .filter(|_| self.devices.iter().all(|device| device.graphs.capture_mode))
        {
            if self.tp_graph_signals.len() != n || !self.tp_graph_signal_topology_supported() {
                return Err(HipError::new(
                    0,
                    &format!("{operation}: graph capture has no admitted TP signal tape"),
                ));
            }
            let epoch_index = self.tp_graph_capture_epoch;
            if epoch_index >= self.tp_graph_barrier_count {
                return Err(HipError::new(
                    0,
                    &format!(
                        "TP graph reduce epoch {epoch_index} exceeds prepared capacity {}",
                        self.tp_graph_barrier_count
                    ),
                ));
            }
            if self.tp_graph_payloads.len() != n || self.tp_graph_payload_bytes < bytes16 {
                return Err(HipError::new(
                    0,
                    &format!("{operation}: TP graph peer shadows were not prepared before capture"),
                ));
            }
            let segment = self.tp_graph_payload_bytes;
            for rank in 0..n {
                let base = self.tp_graph_payloads[rank].as_ptr().cast::<u8>();
                // Same segment protocol as the f32 graph reduce: rank writes
                // the segment its peer consumes for this slot and reads its
                // own; the segment now holds `2 * count` bytes.
                let peer_shadow = unsafe {
                    base.add(tp_graph_payload_offset(segment, 1 - rank, slot))
                        .cast::<c_void>()
                };
                let local_shadow = unsafe {
                    base.add(tp_graph_payload_offset(segment, rank, slot))
                        .cast::<c_void>() as *const c_void
                };
                self.devices[rank].tp_graph_signal_reduce16_gfx1100(
                    fmt,
                    &self.tp_graph_signals[rank],
                    &self.tp_graph_signals[1 - rank],
                    residuals[rank],
                    partials[rank],
                    peer_shadow,
                    local_shadow,
                    rank == 0,
                    count,
                )?;
            }
            self.tp_graph_capture_epoch += 1;
            return Ok(());
        }

        let result = (|| -> HipResult<()> {
            self.ensure_tp_exch16(slot_bytes)?;
            self.ensure_rank_barrier_events()?;
            self.ensure_rank_done_events()?;
            let slot = alternating_slot.unwrap_or(0);

            // (a) Pack each rank's partial into its own slot, then record the
            // producer event after the pack.
            for rank in 0..n {
                self.devices[rank].bind_thread()?;
                let dst = self.exch16_ptr(rank, slot);
                self.devices[rank].tp_pack16(
                    fmt,
                    partials[rank].as_ptr() as *const c_void,
                    dst,
                    count,
                )?;
                let gpu = &self.devices[rank];
                let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("{operation}: device {rank} lost its active_stream"),
                    )
                })?;
                let producer_event = if alternating_slot == Some(1) {
                    &self.rank_done_events[rank]
                } else {
                    &self.rank_barrier_events[rank]
                };
                gpu.hip.event_record(producer_event, Some(stream))?;
            }

            // (b)+(c) Wait the peer's pack, then fold both 16-bit partials in
            // rank order into the local residual.
            let sources = [
                self.exch16_ptr(0, slot) as *const c_void,
                self.exch16_ptr(1, slot) as *const c_void,
            ];
            for rank in 0..n {
                let peer = 1 - rank;
                self.devices[rank].bind_thread()?;
                {
                    let gpu = &self.devices[rank];
                    let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("{operation}: device {rank} lost its active_stream"),
                        )
                    })?;
                    let producer_event = if alternating_slot == Some(1) {
                        &self.rank_done_events[peer]
                    } else {
                        &self.rank_barrier_events[peer]
                    };
                    gpu.hip.stream_wait_event(stream, producer_event)?;
                }
                self.devices[rank].tp_fold16(
                    fmt,
                    std::ptr::null_mut(),
                    residuals[rank].as_ptr(),
                    std::ptr::null(),
                    &sources,
                    count,
                )?;
                if alternating_slot.is_none() {
                    let gpu = &self.devices[rank];
                    let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("{operation}: device {rank} lost its active_stream"),
                        )
                    })?;
                    gpu.hip
                        .event_record(&self.rank_done_events[rank], Some(stream))?;
                }
            }

            // (e) The ordinary protocol: do not repack a slot the peer may
            // still be reading. The alternating protocol gets this from its
            // slot alternation (a rank cannot pack the next use of a slot
            // before consuming the other slot, which waits the peer's pack).
            if alternating_slot.is_none() {
                for rank in 0..n {
                    let peer = 1 - rank;
                    let gpu = &self.devices[rank];
                    gpu.bind_thread()?;
                    let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                        HipError::new(
                            0,
                            &format!("{operation}: device {rank} lost its active_stream"),
                        )
                    })?;
                    gpu.hip
                        .stream_wait_event(stream, &self.rank_done_events[peer])?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.drain_rank_streams();
            return Err(error);
        }
        Ok(())
    }

    /// Rooted gather / fold / broadcast with a 16-bit gather (`2..=4` ranks).
    /// Each non-root rank packs its partial; the root copies the packed
    /// partials into its scratch, folds `r(p0) + r(p1) + ...` (its own
    /// partial rounded in register), adds the sum to its residual and
    /// broadcasts the f32 residual to every peer.
    fn rooted_add16(
        &mut self,
        fmt: TpExchange16,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let operation = "all_reduce_peer_rooted_add16";
        self.check_add16_args(operation, partials, residuals, count)?;
        let n = self.devices.len();
        let bytes = count * std::mem::size_of::<f32>();
        let bytes16 = count * 2;
        let slot_bytes = exch16_slot_bytes(count)?;

        let result = (|| -> HipResult<()> {
            // `ensure_peer_ar_tmp` may drain and free the pack buffers when
            // it grows, so it comes first.
            self.ensure_peer_ar_tmp(bytes16)?;
            self.ensure_tp_exch16(slot_bytes)?;
            self.ensure_rank_barrier_events()?;

            // 1. Non-root ranks pack their partial, then record E[r].
            for rank in 1..n {
                self.devices[rank].bind_thread()?;
                let dst = self.exch16_ptr(rank, 0);
                self.devices[rank].tp_pack16(
                    fmt,
                    partials[rank].as_ptr() as *const c_void,
                    dst,
                    count,
                )?;
                let gpu = &self.devices[rank];
                let stream = gpu.active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("{operation}: device {rank} lost its active_stream"),
                    )
                })?;
                gpu.hip
                    .event_record(&self.rank_barrier_events[rank], Some(stream))?;
            }

            // 2. Root waits each CURRENT E[r] and pulls the packed partial.
            let root_dev_id = self.devices[0].device_id;
            self.devices[0].bind_thread()?;
            for rank in 1..n {
                let src_dev_id = self.devices[rank].device_id;
                let root_stream = self.devices[0].active_stream.as_ref().ok_or_else(|| {
                    HipError::new(0, &format!("{operation}: rank 0 lost its active_stream"))
                })?;
                self.devices[0]
                    .hip
                    .stream_wait_event(root_stream, &self.rank_barrier_events[rank])?;
                self.devices[0].hip.memcpy_peer_async(
                    &self.peer_ar_tmp[0][rank - 1],
                    root_dev_id,
                    &self.tp_exch16[rank],
                    src_dev_id,
                    bytes16,
                    root_stream,
                )?;
            }

            // 3. Rank-order fold on the root: residual += r(p0) + r(p1) + ...
            let mut sources = [std::ptr::null::<c_void>(); MAX_EXCHANGE_RANKS];
            for (slot, source) in sources.iter_mut().take(n - 1).enumerate() {
                *source = self.peer_ar_tmp[0][slot].as_ptr() as *const c_void;
            }
            self.devices[0].tp_fold16(
                fmt,
                std::ptr::null_mut(),
                residuals[0].as_ptr(),
                partials[0].as_ptr() as *const c_void,
                &sources[..n - 1],
                count,
            )?;
            {
                let root_stream = self.devices[0].active_stream.as_ref().ok_or_else(|| {
                    HipError::new(0, &format!("{operation}: rank 0 lost its active_stream"))
                })?;
                self.devices[0]
                    .hip
                    .event_record(&self.rank_barrier_events[0], Some(root_stream))?;
            }

            // 4. Each destination waits CURRENT E[0], peer-copies the root
            //    residual on its own stream, then records E[r].
            for rank in 1..n {
                let dst_gpu = &self.devices[rank];
                let dst_dev_id = dst_gpu.device_id;
                dst_gpu.bind_thread()?;
                let dst_stream = dst_gpu.active_stream.as_ref().ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!("{operation}: device {rank} lost its active_stream"),
                    )
                })?;
                dst_gpu
                    .hip
                    .stream_wait_event(dst_stream, &self.rank_barrier_events[0])?;
                dst_gpu.hip.memcpy_peer_async(
                    residuals[rank],
                    dst_dev_id,
                    residuals[0],
                    root_dev_id,
                    bytes,
                    dst_stream,
                )?;
                dst_gpu
                    .hip
                    .event_record(&self.rank_barrier_events[rank], Some(dst_stream))?;
            }

            // 5. Reverse source reuse: the root waits every E[r] so its next
            //    overwrite cannot race a destination's read.
            self.devices[0].bind_thread()?;
            let root_stream = self.devices[0].active_stream.as_ref().ok_or_else(|| {
                HipError::new(0, &format!("{operation}: rank 0 lost its active_stream"))
            })?;
            for rank in 1..n {
                self.devices[0]
                    .hip
                    .stream_wait_event(root_stream, &self.rank_barrier_events[rank])?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.drain_rank_streams();
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchange_env_parses_each_mode_and_rejects_unknown() {
        use TpExchangeMode::{Auto, Fixed};
        assert_eq!(parse_tp_exchange(None), Ok(Auto));
        assert_eq!(parse_tp_exchange(Some("")), Ok(Auto));
        assert_eq!(parse_tp_exchange(Some("auto")), Ok(Auto));
        assert_eq!(parse_tp_exchange(Some("f32")), Ok(Fixed(None)));
        assert_eq!(
            parse_tp_exchange(Some("bf16")),
            Ok(Fixed(Some(TpExchange16::Bf16)))
        );
        assert_eq!(
            parse_tp_exchange(Some(" f16 ")),
            Ok(Fixed(Some(TpExchange16::F16)))
        );
        assert!(parse_tp_exchange(Some("fp8")).is_err());
        assert!(parse_tp_exchange(Some("BF16")).is_err());
    }

    #[test]
    fn auto_policy_table_by_rows_and_tp() {
        let bf16 = Some(TpExchange16::Bf16);
        // (rows, tp, expected)
        let table = [
            (1, 2, None),
            (1, 3, bf16),
            (1, 4, bf16),
            (2, 2, bf16),
            (16, 2, bf16),
            (2048, 2, bf16),
            (2, 3, bf16),
            (2048, 4, bf16),
        ];
        for (rows, tp, want) in table {
            assert_eq!(auto_exchange(rows, tp), want, "rows={rows} tp={tp}");
            assert_eq!(TpExchangeMode::Auto.resolve(rows, tp), want);
        }
        // Explicit overrides ignore rows and tp.
        for (rows, tp) in [(1, 2), (1, 4), (2048, 2)] {
            assert_eq!(TpExchangeMode::Fixed(None).resolve(rows, tp), None);
            assert_eq!(TpExchangeMode::Fixed(bf16).resolve(rows, tp), bf16);
        }
        assert_eq!(TpExchangeMode::Auto.widest(), bf16);
        assert_eq!(TpExchangeMode::Fixed(None).widest(), None);
    }

    #[test]
    fn exchange_slot_is_256_aligned_and_checked() {
        assert_eq!(exch16_slot_bytes(8).unwrap(), 256);
        assert_eq!(exch16_slot_bytes(128).unwrap(), 256);
        assert_eq!(exch16_slot_bytes(129).unwrap(), 512);
        assert_eq!(exch16_slot_bytes(5120).unwrap(), 10240);
        assert!(exch16_slot_bytes(usize::MAX).is_err());
    }

    fn host_bits(fmt: TpExchange16, v: f32) -> u16 {
        match fmt {
            TpExchange16::Bf16 => half::bf16::from_f32(v).to_bits(),
            TpExchange16::F16 => half::f16::from_f32(v).to_bits(),
        }
    }

    fn host_round(fmt: TpExchange16, v: f32) -> f32 {
        match fmt {
            TpExchange16::Bf16 => half::bf16::from_f32(v).to_f32(),
            TpExchange16::F16 => half::f16::from_f32(v).to_f32(),
        }
    }

    /// Deterministic values over many magnitudes and both signs; `finite`
    /// keeps them inside the f16 range so sums stay finite.
    fn values(seed: u64, n: usize, finite: bool) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mantissa = ((state >> 33) as f32) / (1u64 << 31) as f32 + 0.5;
                let exponent = ((state >> 8) % 36) as i32 - 20;
                let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                let v = sign * mantissa * (2.0f32).powi(exponent);
                if finite {
                    v.clamp(-60000.0, 60000.0)
                } else {
                    v
                }
            })
            .collect()
    }

    /// `tp_pack16_*` and `tp_fold16_*` against the host round-to-nearest-even
    /// codec: every packed bit pattern, then a fold with an f32 first term,
    /// three 16-bit sources, the sum output and the residual add. The count
    /// is not a multiple of 4, so the vector loop and the scalar tail both run.
    #[test]
    #[ignore = "GPU: one card"]
    fn pack_and_fold_match_the_host_codec() {
        use rdna_compute::Gpu;
        const N: usize = 4099;
        let mut gpu = Gpu::init().unwrap();
        for fmt in [TpExchange16::Bf16, TpExchange16::F16] {
            // Specials ride at the front of the pack input.
            let mut input = values(1, N, false);
            let specials = [
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                65504.0,
                65519.0,
                65520.0,
                1.0e5,
                f32::MIN_POSITIVE,
                1.0e-8,
                3.4e38,
            ];
            input[..specials.len()].copy_from_slice(&specials);
            let src = gpu.upload_f32(&input, &[N]).unwrap();
            let dst = gpu.hip.malloc(N * 2).unwrap();
            gpu.tp_pack16(fmt, src.buf.as_ptr(), dst.as_ptr(), N)
                .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let mut packed = vec![0u8; N * 2];
            gpu.hip.memcpy_dtoh(&mut packed, &dst).unwrap();
            for (i, (v, bytes)) in input.iter().zip(packed.chunks_exact(2)).enumerate() {
                let got = u16::from_le_bytes([bytes[0], bytes[1]]);
                let want = host_bits(fmt, *v);
                if v.is_nan() {
                    let back = match fmt {
                        TpExchange16::Bf16 => half::bf16::from_bits(got).is_nan(),
                        TpExchange16::F16 => half::f16::from_bits(got).is_nan(),
                    };
                    assert!(back, "{fmt:?} NaN lane {i}: got {got:#06x}");
                } else {
                    assert_eq!(got, want, "{fmt:?} lane {i}: {v:e}");
                }
            }

            // Fold: first = f32 term, three packed sources, sum_out, x += sum.
            let (a, b, c, d, e) = (
                values(2, N, true),
                values(3, N, true),
                values(4, N, true),
                values(5, N, true),
                values(6, N, true),
            );
            let mut srcs = Vec::new();
            for host in [&a, &b, &c] {
                let t = gpu.upload_f32(host, &[N]).unwrap();
                let buf = gpu.hip.malloc(N * 2).unwrap();
                gpu.tp_pack16(fmt, t.buf.as_ptr(), buf.as_ptr(), N).unwrap();
                srcs.push(buf);
            }
            let first = gpu.upload_f32(&d, &[N]).unwrap();
            let x = gpu.upload_f32(&e, &[N]).unwrap();
            let sum = gpu.upload_f32(&vec![0.0; N], &[N]).unwrap();
            let ptrs: Vec<*const c_void> = srcs.iter().map(|b| b.as_ptr() as *const _).collect();
            gpu.tp_fold16(
                fmt,
                sum.buf.as_ptr(),
                x.buf.as_ptr(),
                first.buf.as_ptr() as *const c_void,
                &ptrs,
                N,
            )
            .unwrap();
            gpu.hip.device_synchronize().unwrap();
            let (got_sum, got_x) = (
                gpu.download_f32(&sum).unwrap(),
                gpu.download_f32(&x).unwrap(),
            );
            for i in 0..N {
                let want_sum = ((host_round(fmt, d[i]) + host_round(fmt, a[i]))
                    + host_round(fmt, b[i]))
                    + host_round(fmt, c[i]);
                let want_x = e[i] + want_sum;
                assert_eq!(got_sum[i].to_bits(), want_sum.to_bits(), "{fmt:?} sum {i}");
                assert_eq!(got_x[i].to_bits(), want_x.to_bits(), "{fmt:?} x {i}");
            }
        }
    }
}
