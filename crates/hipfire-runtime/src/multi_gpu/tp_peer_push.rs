// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-scoped resources and dispatch of the gfx1201 tagged peer-VRAM residual
//! reduction (`rdna_compute::tp_peer_push`): the TP2 f32 route
//! (`kernels/src/tp2_peer_push.gfx1201.hip`, opt-in) and the TP4 bf16
//! all-to-all route (`kernels/src/tp_peer_pushn.gfx1201.hip`), which is what
//! makes the TP4 whole-token graph possible: one kernel per rank per
//! collective, every cross-rank dependency a device-resident generation.
//!
//! Each rank owns one fine-grained VRAM mailbox (two banks of 16-byte tagged
//! granules, `hipDeviceMallocFinegrained`) that the PEER writes, and one
//! device-resident generation word. The mailboxes are allocated in
//! [`Gpus::prepare_tp_graph_peer_shadows`], before peer access is enabled, and
//! [`Gpus::enable_peer_all`] qualifies them with a short eager exchange that
//! is compared bit for bit with a host replay; a qualification failure turns
//! the route off for the process and the incumbent coherent-shadow protocol
//! serves every reduction. Opt-in (`HIPFIRE_TP_PEER_PUSH=1`), default off: on
//! the R9700 pair the 27B decode gain is +0.6% (54.5 -> 54.9 tok/s), below the
//! bar for a default, so the incumbent stays the default route.

use super::tp_exchange::{tp_exchange_mode, TpExchange16};
use super::Gpus;
use hip_bridge::{DeviceBuffer, HipError, HipResult};
use rdna_compute::tp_peer_push::{
    TP_PEER_PUSHN_MAX_COUNT, TP_PEER_PUSH_GRANULE_BYTES, TP_PEER_PUSH_MAX_COUNT,
};

/// Eager exchanges the load-time qualification runs (each one a fresh
/// partial pair on both ranks, accumulated into one residual).
const QUALIFY_EXCHANGES: usize = 64;

pub(super) struct TpPeerPush {
    /// One mailbox per rank (rank r's mailbox is written by the other ranks).
    mailboxes: Vec<DeviceBuffer>,
    /// One 256-byte block per rank; word 0 is the rank's device generation.
    gens: Vec<DeviceBuffer>,
    /// Bytes between the two banks of a mailbox (256-byte aligned).
    bank_stride: usize,
    /// Bytes between the sender slots of a bank (TP4 only; 0 for TP2).
    slot_stride: usize,
    /// Largest reduction (floats) the mailboxes hold.
    max_count: usize,
    /// Rank count (2: f32 exchange, 3 or 4: bf16 exchange).
    ranks: usize,
    /// Collectives recorded into the graph being captured (TP4).
    captured: usize,
    /// `HIPFIRE_TP_PEER_PUSH_FAULT=rank:replay[:ms]` (developer-only, read once at
    /// prepare): stall `rank`'s stream for `ms` (default 900, at most 1500)
    /// before the `replay`-th whole-token graph launch, once.
    fault: Option<FaultSpec>,
    /// Whole-token graph launches so far (for `fault`).
    launches: usize,
    /// False after a failed qualification: every reduction uses the incumbent.
    enabled: bool,
}

/// `HIPFIRE_TP_PEER_PUSH`: tri-state. `1` forces the peer-VRAM reduction on,
/// `0` off. Unset: off for a gfx1201 pair (+0.6% decode, not worth a default),
/// on for four ranks, where it replaces the event/copy route that cannot be
/// captured into a graph (+43% TP4 decode).
fn tp_peer_push_requested(ranks: usize) -> bool {
    match hipfire_config::developer_var("HIPFIRE_TP_PEER_PUSH")
        .ok()
        .as_deref()
    {
        Some("1") => true,
        Some("0") => false,
        _ => ranks == 4,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaultSpec {
    rank: usize,
    replay: usize,
    ms: u32,
}

/// `rank:replay[:ms]`; anything malformed or out of range is ignored (loudly).
fn parse_fault(raw: Option<&str>, ranks: usize) -> Option<FaultSpec> {
    let raw = raw?;
    let parts: Vec<&str> = raw.split(':').collect();
    let parsed = (|| {
        let rank = parts.first()?.parse::<usize>().ok()?;
        let replay = parts.get(1)?.parse::<usize>().ok()?;
        let ms = match parts.get(2) {
            Some(ms) => ms.parse::<u32>().ok()?,
            None => 900,
        };
        (parts.len() <= 3 && rank < ranks && ms > 0).then_some(FaultSpec {
            rank,
            replay,
            ms: ms.min(rdna_compute::tp_peer_push::TP_PEER_PUSH_FAULT_MAX_MS),
        })
    })();
    if parsed.is_none() {
        eprintln!("[tp-peer-push] ignoring malformed HIPFIRE_TP_PEER_PUSH_FAULT={raw:?}");
    } else {
        eprintln!("[tp-peer-push] fault injection armed: {raw}");
    }
    parsed
}

/// bf16 round trip of the exchange codec (`tp_exchange16.h`: round to nearest
/// even, NaN canonicalised), the value every rank adds for a partial.
fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let half = if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        ((bits >> 16) & 0x8000) | 0x7fc0
    } else {
        bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
    };
    f32::from_bits(half << 16)
}

/// Host mirror of one qualification partial element: an exact fp32 value in
/// [-1, 1) that differs per (exchange, rank, index).
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

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is read as plain bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

impl Gpus {
    /// Allocate the peer mailboxes and generation words for reductions of up
    /// to `payload_bytes / 4` floats and resolve the kernel (JIT is not
    /// capture-safe). Best effort: any condition or allocation failure leaves
    /// `tp_peer_push` empty and the incumbent route in force. Must run before
    /// [`Self::enable_peer_all`].
    pub(super) fn prepare_tp_peer_push(&mut self, payload_bytes: usize) {
        let n = self.devices.len();
        if self.tp_peer_push.is_some() || !tp_peer_push_requested(n) {
            return;
        }
        if !matches!(n, 2 | 4) || self.devices.iter().any(|gpu| gpu.arch != "gfx1201") {
            return;
        }
        let max_count = payload_bytes / std::mem::size_of::<f32>();
        let fits = if n == 2 {
            max_count != 0 && max_count.is_multiple_of(2) && max_count <= TP_PEER_PUSH_MAX_COUNT
        } else {
            // The bf16 all-to-all route serves exactly the bf16 decode exchange.
            max_count != 0
                && max_count.is_multiple_of(4)
                && max_count <= TP_PEER_PUSHN_MAX_COUNT
                && tp_exchange_mode().resolve(1, n) == Some(TpExchange16::Bf16)
        };
        if !fits {
            return;
        }
        match self.allocate_tp_peer_push(max_count) {
            Ok(push) => self.tp_peer_push = Some(push),
            Err(error) => eprintln!(
                "[tp-peer-push] disabled: could not prepare the peer-VRAM mailboxes: {error}"
            ),
        }
    }

    fn allocate_tp_peer_push(&mut self, max_count: usize) -> HipResult<TpPeerPush> {
        let n = self.devices.len();
        let (slot_stride, bank_stride) = if n == 2 {
            (
                0,
                (max_count / 2 * TP_PEER_PUSH_GRANULE_BYTES).next_multiple_of(256),
            )
        } else {
            let slot = (max_count / 4 * TP_PEER_PUSH_GRANULE_BYTES).next_multiple_of(256);
            (slot, n * slot)
        };
        for gpu in &mut self.devices {
            if n == 2 {
                gpu.ensure_tp_peer_push_gfx1201_kernel()?;
            } else {
                gpu.ensure_tp_peer_pushn_gfx1201_kernel()?;
            }
            gpu.ensure_device_wait_block()?;
            if hipfire_config::developer_var("HIPFIRE_TP_PEER_PUSH_FAULT").is_ok() {
                gpu.ensure_tp_peer_push_fault_kernel()?;
            }
        }
        let mut push = TpPeerPush {
            mailboxes: Vec::with_capacity(n),
            gens: Vec::with_capacity(n),
            bank_stride,
            slot_stride,
            max_count,
            ranks: n,
            captured: 0,
            fault: parse_fault(
                hipfire_config::developer_var("HIPFIRE_TP_PEER_PUSH_FAULT")
                    .ok()
                    .as_deref(),
                n,
            ),
            launches: 0,
            enabled: true,
        };
        for rank in 0..n {
            let hip = &self.devices[rank].hip;
            self.devices[rank].bind_thread()?;
            let allocated = (|| -> HipResult<()> {
                let mailbox = hip.malloc_peer_finegrained(2 * bank_stride)?;
                if let Err(error) = hip.memset(&mailbox, 0, 2 * bank_stride) {
                    let _ = hip.free(mailbox);
                    return Err(error);
                }
                push.mailboxes.push(mailbox);
                let generation = hip.malloc(256)?;
                if let Err(error) = hip.memset(&generation, 0, 256) {
                    let _ = hip.free(generation);
                    return Err(error);
                }
                push.gens.push(generation);
                Ok(())
            })();
            if let Err(error) = allocated {
                self.release_tp_peer_push(push);
                return Err(error);
            }
        }
        Ok(push)
    }

    fn release_tp_peer_push(&self, push: TpPeerPush) -> Option<HipError> {
        let mut first = None;
        for (rank, buffer) in push.mailboxes.into_iter().enumerate() {
            if let Err(error) = self.devices[rank]
                .bind_thread()
                .and_then(|()| self.devices[rank].hip.free(buffer))
            {
                first.get_or_insert(error);
            }
        }
        for (rank, buffer) in push.gens.into_iter().enumerate() {
            if let Err(error) = self.devices[rank]
                .bind_thread()
                .and_then(|()| self.devices[rank].hip.free(buffer))
            {
                first.get_or_insert(error);
            }
        }
        first
    }

    /// Free the mailboxes and generation words after every captured rank graph
    /// has been dropped and the ranks are idle.
    pub(super) fn free_tp_peer_push(&mut self) -> HipResult<()> {
        match self.tp_peer_push.take() {
            Some(push) => match self.release_tp_peer_push(push) {
                Some(error) => Err(error),
                None => Ok(()),
            },
            None => Ok(()),
        }
    }

    /// Zero every generation word and mailbox (tags 0 never match a live
    /// generation). Both ranks must be drained: the generations must restart
    /// together, so this belongs with the TP signal reset.
    pub(super) fn reset_tp_peer_push(&mut self) -> HipResult<()> {
        let Some(push) = self.tp_peer_push.as_ref() else {
            return Ok(());
        };
        for rank in 0..push.ranks {
            let hip = &self.devices[rank].hip;
            self.devices[rank].bind_thread()?;
            hip.memset(&push.gens[rank], 0, 256)?;
            hip.memset(&push.mailboxes[rank], 0, 2 * push.bank_stride)?;
        }
        Ok(())
    }

    /// True when a reduction of `count` floats runs on the peer-push kernel.
    pub(super) fn tp_peer_push_serves(&self, count: usize) -> bool {
        self.tp_peer_push.as_ref().is_some_and(|push| {
            let granule = if push.ranks == 2 { 2 } else { 4 };
            push.enabled && count != 0 && count.is_multiple_of(granule) && count <= push.max_count
        })
    }

    /// True when the TP4 bf16 route is prepared and has passed its
    /// qualification (the whole-token graph's only cross-rank dependency).
    pub(super) fn tp_peer_pushn_active(&self) -> bool {
        self.tp_peer_push
            .as_ref()
            .is_some_and(|push| push.enabled && push.ranks >= 3)
    }

    /// Developer-only fault injection: count this whole-token graph launch and,
    /// when it is the armed one, stall the chosen rank's stream (once) so the
    /// peers' bounded waits time out. Call before the rank graphs launch.
    pub(super) fn tp_peer_push_inject_fault(&mut self) -> HipResult<()> {
        let Some(push) = self.tp_peer_push.as_mut() else {
            return Ok(());
        };
        push.launches += 1;
        let Some(fault) = push.fault.filter(|fault| fault.replay == push.launches) else {
            return Ok(());
        };
        push.fault = None;
        eprintln!(
            "[tp-peer-push] FAULT INJECTION: stalling rank {} for {} ms before graph launch {}",
            fault.rank, fault.ms, fault.replay
        );
        self.devices[fault.rank].bind_thread()?;
        self.devices[fault.rank].tp_peer_push_fault_spin_gfx1201(fault.ms)
    }

    /// Rewind the capture-time collective counter before a whole-token capture.
    pub(super) fn begin_tp_peer_push_capture(&mut self) -> HipResult<()> {
        let push = self
            .tp_peer_push
            .as_mut()
            .ok_or_else(|| HipError::new(0, "TP peer push is not prepared"))?;
        push.captured = 0;
        Ok(())
    }

    /// Collectives the capture in progress has recorded.
    pub(super) fn tp_peer_push_captured_count(&self) -> usize {
        self.tp_peer_push.as_ref().map_or(0, |push| push.captured)
    }

    /// Record one captured collective: counted once per logical reduction.
    pub(super) fn tp_peer_push_reduce_captured(
        &mut self,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        self.tp_peer_push_reduce(partials, residuals, count)?;
        if let Some(push) = self.tp_peer_push.as_mut() {
            push.captured += 1;
        }
        Ok(())
    }

    /// Enqueue the peer-push reduction on every rank (eager or captured).
    pub(super) fn tp_peer_push_reduce(
        &mut self,
        partials: &[&DeviceBuffer],
        residuals: &[&DeviceBuffer],
        count: usize,
    ) -> HipResult<()> {
        let push = self
            .tp_peer_push
            .as_ref()
            .ok_or_else(|| HipError::new(0, "tp_peer_push_reduce: peer push is not prepared"))?;
        let n = push.ranks;
        if partials.len() != n || residuals.len() != n {
            return Err(HipError::new(
                0,
                "tp_peer_push_reduce: partial/residual count differs from the rank count",
            ));
        }
        if n == 2 {
            let stride = push.bank_stride;
            for rank in 0..2 {
                let peer_mailbox = push.mailboxes[1 - rank].as_ptr();
                let local_mailbox = push.mailboxes[rank].as_ptr() as *const std::ffi::c_void;
                self.devices[rank].tp_peer_push_reduce_gfx1201(
                    residuals[rank],
                    partials[rank],
                    peer_mailbox,
                    local_mailbox,
                    stride,
                    &push.gens[rank],
                    rank == 0,
                    count,
                )?;
            }
            return Ok(());
        }
        let mut mailboxes = [std::ptr::null_mut::<std::ffi::c_void>(); 4];
        for (slot, mailbox) in mailboxes.iter_mut().zip(&push.mailboxes) {
            *slot = mailbox.as_ptr();
        }
        for rank in 0..n {
            self.devices[rank].tp_peer_pushn_reduce_bf16_gfx1201(
                residuals[rank],
                partials[rank],
                mailboxes,
                push.slot_stride,
                push.bank_stride,
                &push.gens[rank],
                rank,
                n,
                count,
            )?;
        }
        Ok(())
    }

    /// Load-time qualification, run once peer access is enabled:
    /// `QUALIFY_EXCHANGES` eager peer-push exchanges on fresh partials, every
    /// rank's residual compared bit for bit with the host replay (`x + (p0 +
    /// p1)` for TP2, `x + (r(p0) + r(p1) + ...)` with the bf16 round trip for
    /// TP4). Any mismatch, device wait error or HIP error disables the
    /// route for the process (loudly) and leaves the incumbent protocol in force.
    pub(super) fn qualify_tp_peer_push(&mut self) {
        let Some(push) = self.tp_peer_push.as_ref() else {
            return;
        };
        if !push.enabled {
            return;
        }
        let count = push.max_count;
        let outcome = self.run_tp_peer_push_qualification(count);
        let verdict = match outcome {
            Ok(None) => None,
            Ok(Some(message)) => Some(message),
            Err(error) => Some(format!("HIP error: {error}")),
        };
        // Whatever happened, restart the generations together.
        let drained = self.devices.iter().try_for_each(|gpu| {
            gpu.bind_thread()?;
            gpu.hip.device_synchronize()
        });
        for gpu in &self.devices {
            // A timeout record belongs to this qualification, not to a decode step.
            let _ = gpu.take_device_wait_error();
        }
        let reset = drained.and_then(|()| self.reset_tp_peer_push());
        match (verdict, reset) {
            (None, Ok(())) => {}
            (verdict, reset) => {
                if let Some(push) = self.tp_peer_push.as_mut() {
                    push.enabled = false;
                }
                eprintln!(
                    "[tp-peer-push] disabled, using the incumbent reduction: {}{}",
                    verdict.unwrap_or_else(|| "qualification passed".to_string()),
                    match reset {
                        Ok(()) => String::new(),
                        Err(error) => format!("; mailbox reset failed: {error}"),
                    }
                );
            }
        }
    }

    fn run_tp_peer_push_qualification(&mut self, count: usize) -> HipResult<Option<String>> {
        let n = self.devices.len();
        let bf16 = n >= 3;
        let bytes = count * std::mem::size_of::<f32>();
        let mut partials = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);
        for rank in 0..n {
            self.devices[rank].bind_thread()?;
            partials.push(self.devices[rank].hip.malloc(bytes)?);
            residuals.push(self.devices[rank].hip.malloc(bytes)?);
        }
        let result = (|| -> HipResult<Option<String>> {
            let mut expected: Vec<f32> = (0..count).map(|i| (i % 13) as f32 - 6.0).collect();
            for rank in 0..n {
                self.devices[rank].bind_thread()?;
                self.devices[rank]
                    .hip
                    .memcpy_htod(&residuals[rank], f32_bytes(&expected))?;
            }
            let mut p = vec![vec![0f32; count]; n];
            for exchange in 0..QUALIFY_EXCHANGES {
                for rank in 0..n {
                    for (i, v) in p[rank].iter_mut().enumerate() {
                        *v = qualify_partial(exchange, rank, i);
                    }
                    self.devices[rank].bind_thread()?;
                    self.devices[rank]
                        .hip
                        .memcpy_htod(&partials[rank], f32_bytes(&p[rank]))?;
                }
                let partial_refs: Vec<&DeviceBuffer> = partials.iter().collect();
                let residual_refs: Vec<&DeviceBuffer> = residuals.iter().collect();
                self.tp_peer_push_reduce(&partial_refs, &residual_refs, count)?;
                for gpu in &self.devices {
                    gpu.bind_thread()?;
                    gpu.hip.device_synchronize()?;
                    gpu.take_device_wait_error()?;
                }
                for (i, e) in expected.iter_mut().enumerate() {
                    let term = |rank: usize| {
                        if bf16 {
                            bf16_round(p[rank][i])
                        } else {
                            p[rank][i]
                        }
                    };
                    let mut sum = term(0);
                    for rank in 1..n {
                        sum += term(rank);
                    }
                    *e += sum;
                }
                for rank in 0..n {
                    self.devices[rank].bind_thread()?;
                    let mut raw = vec![0u8; bytes];
                    self.devices[rank]
                        .hip
                        .memcpy_dtoh(&mut raw, &residuals[rank])?;
                    for (i, chunk) in raw.chunks_exact(4).enumerate() {
                        let got = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        if got.to_bits() != expected[i].to_bits() {
                            return Ok(Some(format!(
                                "rank {rank} exchange {exchange} element {i}: got {got:e}, expected {:e}",
                                expected[i]
                            )));
                        }
                    }
                }
            }
            Ok(None)
        })();
        for rank in 0..n {
            self.devices[rank].bind_thread()?;
            let _ = self.devices[rank].hip.device_synchronize();
        }
        for (rank, buffer) in partials.into_iter().chain(residuals).enumerate() {
            let rank = rank % n;
            self.devices[rank].bind_thread()?;
            let _ = self.devices[rank].hip.free(buffer);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_matches_the_device_codec() {
        assert_eq!(bf16_round(1.0).to_bits(), 1.0f32.to_bits());
        // 1 + 2^-8 is a tie between bf16 neighbours: round to even keeps 1.0.
        assert_eq!(
            bf16_round(f32::from_bits(0x3f80_8000)).to_bits(),
            0x3f80_0000
        );
        assert_eq!(
            bf16_round(f32::from_bits(0x3f81_8000)).to_bits(),
            0x3f82_0000
        );
        assert!(bf16_round(f32::NAN).is_nan());
        assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn qualification_partials_are_exact_and_distinct() {
        let a = qualify_partial(0, 0, 0);
        assert!((-1.0..1.0).contains(&a));
        assert_ne!(a.to_bits(), qualify_partial(0, 1, 0).to_bits());
        assert_ne!(a.to_bits(), qualify_partial(1, 0, 0).to_bits());
        assert_ne!(a.to_bits(), qualify_partial(0, 0, 1).to_bits());
    }
}
