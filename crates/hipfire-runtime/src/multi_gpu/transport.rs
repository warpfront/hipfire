// SPDX-License-Identifier: MIT
// Copyright (c) 2026 alpineq

//! Default-off, load-scoped edge resources owned by `multi_gpu`.
//!
//! Callers supply indices into `Gpus::devices`, not HIP ordinals. A lease binds
//! those indices to the two supplied devices once. Submission uses their active
//! streams only; drain synchronizes whole devices, including auxiliary streams.
//! Generations count transfers, never token positions. Drop deliberately leaks
//! undrained resources: only a successful explicit drain permits reclamation.
//! Host aliases are borrowed addresses (HIP has no separate alias-free API).

use crate::mesh_placement::EdgeTransport;
use hip_bridge::{
    DeviceBuffer, Event, MemoryType, HIP_ERROR_PEER_ACCESS_UNSUPPORTED, HIP_EVENT_DISABLE_TIMING,
    HIP_EVENT_RELEASE_TO_SYSTEM, HIP_HOST_MALLOC_MAPPED,
};
use rdna_compute::{Gpu, GpuTensor};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

type Result<T, E = String> = std::result::Result<T, E>;
const NEVER: u64 = u64::MAX;
const WAIT_LIMIT: Duration = Duration::from_secs(30);
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
static PEERS: LazyLock<Mutex<HashSet<(i32, i32)>>> = LazyLock::new(Default::default);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId {
    pub src_device: usize,
    pub dst_device: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeSpec {
    pub edge: EdgeId,
    pub transport: EdgeTransport,
    pub max_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeDescriptor {
    pub epoch: u64,
    pub op: u64,
    pub rows: usize,
    pub dtype_bytes: usize,
    pub bytes: usize,
}

impl EdgeDescriptor {
    fn validate(self, epoch: u64, capacity: usize, extent: usize, address: usize) -> Result<()> {
        if self.epoch != epoch {
            return Err("edge descriptor has a stale transport epoch".into());
        }
        if self.rows == 0
            || !matches!(self.dtype_bytes, 2 | 4)
            || self.bytes == 0
            || self.bytes > capacity
            || self.bytes > extent
            || self.bytes % self.rows != 0
            || (self.bytes / self.rows) % 16 != 0
            || address == 0
            || address % 16 != 0
            || self.bytes / 16 > i32::MAX as usize
        {
            return Err("edge descriptor alignment, dtype, or live extent is invalid".into());
        }
        Ok(())
    }
}

fn parity(k: u64) -> Result<usize> {
    if k == NEVER {
        Err("edge generation exhausted".into())
    } else {
        Ok((k & 1) as usize)
    }
}

struct Protocol {
    ready_recorded: [AtomicU64; 2],
    done_recorded: [AtomicU64; 2],
    producer: AtomicU64,
    consumer: AtomicU64,
    poisoned: AtomicBool,
}

impl Protocol {
    fn new() -> Self {
        Self {
            ready_recorded: std::array::from_fn(|_| AtomicU64::new(NEVER)),
            done_recorded: std::array::from_fn(|_| AtomicU64::new(NEVER)),
            producer: AtomicU64::new(0),
            consumer: AtomicU64::new(0),
            poisoned: AtomicBool::new(false),
        }
    }
    fn check(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Acquire) {
            Err("edge lease poisoned or stopped".into())
        } else {
            Ok(())
        }
    }
    fn observe(&self, word: &AtomicU64, generation: u64, deadline: Instant) -> Result<()> {
        loop {
            self.check()?;
            let seen = word.load(Ordering::Acquire);
            if seen == generation {
                return Ok(());
            }
            if seen != NEVER && seen > generation {
                return Err("edge generation skipped or stale event reuse".into());
            }
            if Instant::now() >= deadline {
                return Err("edge record publication deadline exceeded".into());
            }
            std::thread::yield_now();
        }
    }
    fn reuse(&self, k: u64, deadline: Instant) -> Result<usize> {
        self.check()?;
        let p = parity(k)?;
        if k >= 2 {
            self.observe(&self.done_recorded[p], k - 2, deadline)?;
        }
        Ok(p)
    }
}

#[derive(Default)]
struct Bank {
    host: usize,
    aliases: [usize; 2],
    peer: Option<DeviceBuffer>,
    ready: Option<Event>,
    done: Option<Event>,
    descriptor: Mutex<Option<EdgeDescriptor>>,
}

/// Fault injection is lab-only and never participates in production routing.
#[cfg(feature = "lab")]
#[derive(Clone, Copy, Debug)]
#[repr(u64)]
pub enum FaultPoint {
    BeforeReuseWait = 1,
    AfterReuseWait,
    AfterPublishCopy,
    AfterReadyRecord,
    AfterReadyPublication,
    AfterReadyWait,
    AfterReceiveCopy,
    AfterDoneRecord,
    AfterDonePublication,
}

pub struct EdgeLease {
    spec: EdgeSpec,
    epoch: u64,
    devices: [i32; 2],
    banks: [Bank; 2],
    owners: [Option<GpuTensor>; 2],
    protocol: Protocol,
    drained: bool,
    quarantined: bool,
    #[cfg(feature = "lab")]
    fault: AtomicU64,
}

fn hip<T>(value: hip_bridge::HipResult<T>) -> Result<T> {
    value.map_err(|e| e.to_string())
}

fn record_then_publish(
    word: &AtomicU64,
    generation: u64,
    record: impl FnOnce() -> Result<()>,
) -> Result<()> {
    record()?;
    word.store(generation, Ordering::Release);
    Ok(())
}

/// Flatten validated whole rows into one uint4 row. The single owner entry and
/// kernel are prepared at provision, so submission allocates no host storage.
/// This is the copy `qualify_peer` byte-qualifies; every cross-device read of
/// the mesh executor (edge banks and TP partials) goes through it.
pub(super) fn copy_bytes(
    gpu: &mut Gpu,
    owners: &GpuTensor,
    src: usize,
    dst: usize,
    bytes: usize,
) -> Result<()> {
    let mut sources = [0; 8];
    sources[0] = src;
    hip(gpu.ep_gather_rows_by_slot_owner(owners, sources, dst, 1, 2, 1, bytes / 16))
}

/// Mint a transport epoch. Edge leases and the mesh executor's TP group share
/// this namespace, so a descriptor for one can never name the other.
pub(super) fn next_epoch() -> Result<u64> {
    NEXT_EPOCH
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            (v < NEVER - 1).then_some(v + 1)
        })
        .map_err(|_| "transport epoch exhausted".into())
}

/// Provision exactly the admitted route. Peer requires prior byte qualification;
/// HostMapped never queries or enables peer access and never substitutes aliases.
pub fn provision(src: &mut Gpu, dst: &mut Gpu, spec: EdgeSpec) -> Result<EdgeLease> {
    hip(src.bind_thread())?;
    hip(dst.bind_thread())?;
    if src.device_id == dst.device_id
        || spec.edge.src_device == spec.edge.dst_device
        || spec.max_bytes == 0
        || spec.max_bytes % 16 != 0
        || spec.max_bytes / 16 > i32::MAX as usize
        || src.active_stream.is_none()
        || dst.active_stream.is_none()
    {
        return Err("invalid edge devices, capacity, or missing active streams".into());
    }
    if spec.transport == EdgeTransport::Peer
        && !PEERS
            .lock()
            .map_err(|_| "peer registry poisoned")?
            .contains(&(src.device_id, dst.device_id))
    {
        return Err("peer edge has not passed byte qualification".into());
    }
    let epoch = next_epoch()?;
    let mut lease = EdgeLease {
        spec,
        epoch,
        devices: [src.device_id, dst.device_id],
        banks: std::array::from_fn(|_| Bank::default()),
        owners: [None, None],
        protocol: Protocol::new(),
        drained: false,
        quarantined: false,
        #[cfg(feature = "lab")]
        fault: AtomicU64::new(0),
    };
    let prepared = (|| {
        hip(src.bind_thread())?;
        lease.owners[0] = Some(hip(src.upload_raw(&[0; 16], &[16]))?);
        hip(dst.bind_thread())?;
        lease.owners[1] = Some(hip(dst.upload_raw(&[0; 16], &[16]))?);
        for bank in &mut lease.banks {
            hip(src.bind_thread())?;
            match spec.transport {
                EdgeTransport::HostMapped => {
                    bank.host =
                        hip(src.hip.host_malloc(spec.max_bytes, HIP_HOST_MALLOC_MAPPED))? as usize;
                    bank.aliases[0] =
                        hip(src.hip.host_get_device_pointer(bank.host as *mut c_void, 0))? as usize;
                    hip(dst.bind_thread())?;
                    bank.aliases[1] =
                        hip(dst.hip.host_get_device_pointer(bank.host as *mut c_void, 0))? as usize;
                }
                EdgeTransport::Peer => {
                    let buffer = hip(src.hip.malloc(spec.max_bytes))?;
                    bank.aliases = [buffer.as_ptr() as usize; 2];
                    bank.peer = Some(buffer);
                }
            }
            if bank.aliases.iter().any(|p| *p == 0 || *p % 16 != 0) {
                return Err("edge bank has an invalid device alias".into());
            }
            hip(src.bind_thread())?;
            bank.ready = Some(hip(src.hip.event_create_with_flags(
                HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM,
            ))?);
            hip(dst.bind_thread())?;
            bank.done = Some(hip(dst.hip.event_create_with_flags(
                HIP_EVENT_DISABLE_TIMING | HIP_EVENT_RELEASE_TO_SYSTEM,
            ))?);
        }
        // Load/JIT once, outside the per-transfer path, using the local 16-byte
        // zero owner table as scratch. In particular, never warm via peer writes.
        for (i, gpu) in [(0, &mut *src), (1, &mut *dst)] {
            hip(gpu.bind_thread())?;
            let owners = lease.owners[i].as_ref().unwrap();
            let address = owners.buf.as_ptr() as usize;
            copy_bytes(gpu, owners, address, address, 16)?;
            hip(gpu.hip.device_synchronize())?;
        }
        Ok(())
    })();
    if let Err(error) = prepared {
        return match lease.drain(src, dst) {
            Ok(()) => Err(error),
            Err(drain) => Err(format!("{error}; provision quarantined: {drain}")),
        };
    }
    Ok(lease)
}

impl EdgeLease {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn edge(&self) -> EdgeId {
        self.spec.edge
    }
    pub fn poison(&self) {
        self.protocol.poisoned.store(true, Ordering::Release);
    }

    #[cfg(feature = "lab")]
    pub fn inject_fault(&self, point: FaultPoint) {
        self.fault.store(point as u64, Ordering::Release);
    }

    fn handoff(&self, point: u64) -> Result<()> {
        #[cfg(feature = "lab")]
        if self
            .fault
            .compare_exchange(point, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.poison();
        }
        #[cfg(not(feature = "lab"))]
        let _ = point;
        self.protocol.check()
    }
    fn fail_closed(&self, result: Result<()>) -> Result<()> {
        if result.is_err() {
            self.poison();
        }
        result
    }
    fn validate_local(
        &self,
        gpu: &Gpu,
        side: usize,
        view: &GpuTensor,
        desc: EdgeDescriptor,
    ) -> Result<()> {
        self.protocol.check()?;
        if gpu.device_id != self.devices[side] || gpu.active_stream.is_none() {
            return Err("edge participant or active stream mismatch".into());
        }
        desc.validate(
            self.epoch,
            self.spec.max_bytes,
            view.buf.size(),
            view.buf.as_ptr() as usize,
        )?;
        let attributes = hip(gpu.hip.pointer_get_attributes(&view.buf))?;
        if attributes.device != gpu.device_id
            || MemoryType::from_raw(attributes.mem_type) != Some(MemoryType::Device)
            || attributes.is_managed != 0
        {
            return Err(
                "edge activation must be local device memory owned by its participant".into(),
            );
        }
        Ok(())
    }

    pub fn publish(&self, src: &mut Gpu, view: &GpuTensor, desc: EdgeDescriptor) -> Result<()> {
        self.fail_closed(hip(src.bind_thread()))?;
        let result = (|| {
            self.validate_local(src, 0, view, desc)?;
            let k = self.protocol.producer.load(Ordering::Acquire);
            self.handoff(1)?;
            let p = self.protocol.reuse(k, Instant::now() + WAIT_LIMIT)?;
            let bank = &self.banks[p];
            if k >= 2 {
                hip(src.hip.stream_wait_event(
                    src.active_stream.as_ref().unwrap(),
                    bank.done.as_ref().unwrap(),
                ))?;
            }
            self.handoff(2)?;
            *bank
                .descriptor
                .lock()
                .map_err(|_| "edge descriptor lock poisoned")? = Some(desc);
            copy_bytes(
                src,
                self.owners[0].as_ref().unwrap(),
                view.buf.as_ptr() as usize,
                bank.aliases[0],
                desc.bytes,
            )?;
            self.handoff(3)?;
            record_then_publish(&self.protocol.ready_recorded[p], k, || {
                hip(src
                    .hip
                    .event_record(bank.ready.as_ref().unwrap(), src.active_stream.as_ref()))?;
                self.handoff(4)
            })?;
            self.protocol.producer.store(k + 1, Ordering::Release);
            self.handoff(5)
        })();
        self.fail_closed(result)
    }

    pub fn receive(&self, dst: &mut Gpu, view: &GpuTensor, desc: EdgeDescriptor) -> Result<()> {
        self.fail_closed(hip(dst.bind_thread()))?;
        let result = (|| {
            self.validate_local(dst, 1, view, desc)?;
            let k = self.protocol.consumer.load(Ordering::Acquire);
            let p = parity(k)?;
            self.protocol.observe(
                &self.protocol.ready_recorded[p],
                k,
                Instant::now() + WAIT_LIMIT,
            )?;
            let bank = &self.banks[p];
            if *bank
                .descriptor
                .lock()
                .map_err(|_| "edge descriptor lock poisoned")?
                != Some(desc)
            {
                return Err("edge descriptor identity mismatch".into());
            }
            hip(dst.hip.stream_wait_event(
                dst.active_stream.as_ref().unwrap(),
                bank.ready.as_ref().unwrap(),
            ))?;
            self.handoff(6)?;
            copy_bytes(
                dst,
                self.owners[1].as_ref().unwrap(),
                bank.aliases[1],
                view.buf.as_ptr() as usize,
                desc.bytes,
            )?;
            self.handoff(7)?;
            record_then_publish(&self.protocol.done_recorded[p], k, || {
                hip(dst
                    .hip
                    .event_record(bank.done.as_ref().unwrap(), dst.active_stream.as_ref()))?;
                self.handoff(8)
            })?;
            self.protocol.consumer.store(k + 1, Ordering::Release);
            self.handoff(9)
        })();
        self.fail_closed(result)
    }

    /// Stop submissions, drain *all* streams on both devices, invalidate aliases,
    /// free banks, then events. Any failed drain permanently quarantines resources.
    pub fn drain(&mut self, src: &mut Gpu, dst: &mut Gpu) -> Result<()> {
        self.fail_closed(hip(src.bind_thread()))?;
        self.poison();
        if self.drained {
            return Ok(());
        }
        if self.quarantined {
            return Err("edge resources quarantined after failed drain".into());
        }
        if [src.device_id, dst.device_id] != self.devices {
            return Err("cannot drain edge with different participants".into());
        }
        let result = drain_then_release(
            &mut (&mut *self, src, dst),
            |context, side| {
                let gpu = if side == 0 {
                    &mut *context.1
                } else {
                    &mut *context.2
                };
                hip(gpu.bind_thread())?;
                hip(gpu.hip.device_synchronize())
            },
            |context| context.0.release(context.1, context.2),
        );
        if result.is_err() {
            self.quarantined = true;
        }
        result
    }

    fn release(&mut self, src: &mut Gpu, dst: &mut Gpu) -> Result<()> {
        // No resource is touched before both device-wide drains succeeded.
        for bank in &mut self.banks {
            bank.aliases = [0; 2];
        }
        hip(src.bind_thread())?;
        for bank in &mut self.banks {
            if bank.host != 0 {
                let host = std::mem::take(&mut bank.host);
                hip(src.hip.host_free(host as *mut c_void))?;
            }
            if let Some(buffer) = bank.peer.take() {
                hip(src.hip.free(buffer))?;
            }
        }
        for (i, gpu) in [(0, &mut *src), (1, &mut *dst)] {
            hip(gpu.bind_thread())?;
            if let Some(owner) = self.owners[i].take() {
                hip(gpu.hip.free(owner.buf))?;
            }
        }
        for bank in &mut self.banks {
            hip(src.bind_thread())?;
            if let Some(event) = bank.ready.take() {
                hip(src.hip.event_destroy(event))?;
            }
            hip(dst.bind_thread())?;
            if let Some(event) = bank.done.take() {
                hip(dst.hip.event_destroy(event))?;
            }
        }
        self.drained = true;
        Ok(())
    }
}

/// Shared drain gate, also exercised with fake streams in host-only tests.
fn drain_then_release<C>(
    context: &mut C,
    mut synchronize: impl FnMut(&mut C, usize) -> Result<()>,
    release: impl FnOnce(&mut C) -> Result<()>,
) -> Result<()> {
    let source = synchronize(context, 0);
    let destination = synchronize(context, 1);
    source.and(destination)?;
    release(context)
}

/// Query and enable peer access only for query-true edges, then exercise actual
/// peer-pointer kernel reads in BOTH directions. A lying query is `Ok(false)`,
/// never a runtime transport fallback. Qualification is separate from provisioning.
pub fn qualify_peer(a: &mut Gpu, b: &mut Gpu) -> Result<bool> {
    hip(a.bind_thread())?;
    if a.device_id == b.device_id {
        return Err("peer qualification needs distinct devices".into());
    }
    let ab = hip(a.hip.can_access_peer(a.device_id, b.device_id))?;
    hip(b.bind_thread())?;
    let ba = hip(b.hip.can_access_peer(b.device_id, a.device_id))?;
    if !ab || !ba {
        return Ok(false);
    }
    let devices = [a.device_id, b.device_id];
    for (gpu, peer) in [(&mut *a, devices[1]), (&mut *b, devices[0])] {
        hip(gpu.bind_thread())?;
        if let Err(error) = gpu.hip.enable_peer_access(peer) {
            if error.code == HIP_ERROR_PEER_ACCESS_UNSUPPORTED {
                return Ok(false);
            }
            return Err(error.to_string());
        }
    }
    let ab = probe_direction(a, b)?;
    let ba = probe_direction(b, a)?;
    if ab && ba {
        let mut registry = PEERS.lock().map_err(|_| "peer registry poisoned")?;
        registry.insert((a.device_id, b.device_id));
        registry.insert((b.device_id, a.device_id));
    }
    Ok(ab && ba)
}

fn probe_direction(src: &mut Gpu, dst: &mut Gpu) -> Result<bool> {
    hip(src.bind_thread())?;
    const BYTES: usize = 4096;
    let pattern: Vec<u8> = (0..BYTES)
        .map(|i| {
            let v = (i as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (src.device_id as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                ^ (dst.device_id as u64).wrapping_mul(0x94d049bb133111eb);
            ((v ^ (v >> 31) ^ (v >> 17)) % 255 + 1) as u8
        })
        .collect();
    let mut source = None;
    let mut output = None;
    let mut owners = None;
    let result = (|| {
        source = Some(hip(src.upload_raw(&pattern, &[BYTES]))?);
        hip(src.hip.device_synchronize())?;
        hip(dst.bind_thread())?;
        output = Some(hip(dst.upload_raw(&[0; BYTES], &[BYTES]))?);
        owners = Some(hip(dst.upload_raw(&[0; 4], &[4]))?);
        copy_bytes(
            dst,
            owners.as_ref().unwrap(),
            source.as_ref().unwrap().buf.as_ptr() as usize,
            output.as_ref().unwrap().buf.as_ptr() as usize,
            BYTES,
        )?;
        hip(dst.hip.device_synchronize())?;
        let mut got = [0; BYTES];
        hip(dst.hip.memcpy_dtoh(&mut got, &output.as_ref().unwrap().buf))?;
        Ok(got.as_slice() == pattern)
    })();
    // Even an enqueue failure may leave work in flight. Failed drain means leak,
    // not best-effort free; descriptors have no implicit GPU-free destructor.
    let src_drain = hip(src.bind_thread()).and_then(|_| hip(src.hip.device_synchronize()));
    let dst_drain = hip(dst.bind_thread()).and_then(|_| hip(dst.hip.device_synchronize()));
    src_drain.and(dst_drain)?;
    hip(src.bind_thread())?;
    if let Some(tensor) = source {
        hip(src.hip.free(tensor.buf))?;
    }
    hip(dst.bind_thread())?;
    for tensor in [output, owners].into_iter().flatten() {
        hip(dst.hip.free(tensor.buf))?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_parity_and_exhaustion() {
        for k in 0..100 {
            assert_eq!(parity(k).unwrap(), (k % 2) as usize);
        }
        assert_eq!(parity(NEVER - 1).unwrap(), 0);
        assert!(parity(NEVER).is_err());
    }

    fn descriptor() -> EdgeDescriptor {
        EdgeDescriptor {
            epoch: 7,
            op: 42,
            rows: 8,
            dtype_bytes: 4,
            bytes: 8 * 40960,
        }
    }

    #[test]
    fn descriptor_checks_epoch_alignment_extent_and_identity() {
        let d = descriptor();
        assert!(d.validate(7, d.bytes, d.bytes, 16).is_ok());
        assert!(d.validate(8, d.bytes, d.bytes, 16).is_err());
        assert!(d.validate(7, d.bytes - 1, d.bytes, 16).is_err());
        assert!(d.validate(7, d.bytes, d.bytes - 1, 16).is_err());
        assert!(d.validate(7, d.bytes, d.bytes, 17).is_err());
        for invalid in [
            EdgeDescriptor { rows: 0, ..d },
            EdgeDescriptor { rows: 7, ..d },
            EdgeDescriptor {
                dtype_bytes: 1,
                ..d
            },
            EdgeDescriptor { bytes: 17, ..d },
        ] {
            assert!(invalid.validate(7, d.bytes, d.bytes, 16).is_err());
        }
        for different in [
            EdgeDescriptor { epoch: 8, ..d },
            EdgeDescriptor { op: 43, ..d },
            EdgeDescriptor { rows: 4, ..d },
            EdgeDescriptor {
                dtype_bytes: 2,
                ..d
            },
            EdgeDescriptor { bytes: 16, ..d },
        ] {
            assert_ne!(d, different);
        }
    }

    #[test]
    fn symmetric_record_before_wait_and_k_plus_two_reuse() {
        let state = Protocol::new();
        let mut events = Vec::new();
        for k in 0..20 {
            let deadline = Instant::now() + Duration::from_millis(10);
            let p = state.reuse(k, deadline).unwrap();
            if k >= 2 {
                assert!(events.contains(&("done_record", k - 2)));
                events.push(("done_wait", k - 2));
            }
            // Host publication happens only after a successful fake record.
            events.push(("ready_record", k));
            state.ready_recorded[p].store(k, Ordering::Release);
            state
                .observe(&state.ready_recorded[p], k, deadline)
                .unwrap();
            assert!(events.contains(&("ready_record", k)));
            events.push(("ready_wait", k));
            events.push(("done_record", k));
            state.done_recorded[p].store(k, Ordering::Release);
        }
        assert_eq!(
            events
                .iter()
                .filter(|(name, _)| *name == "done_wait")
                .count(),
            18
        );
    }

    #[test]
    fn acquire_release_handshake_with_delayed_fake_events() {
        let state = Protocol::new();
        let payload = [AtomicU64::new(NEVER), AtomicU64::new(NEVER)];
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for k in 0..100 {
                    let p = state
                        .reuse(k, Instant::now() + Duration::from_secs(2))
                        .unwrap();
                    payload[p].store(k ^ 0x5eed, Ordering::Relaxed);
                    record_then_publish(&state.ready_recorded[p], k, || {
                        std::thread::yield_now();
                        Ok(())
                    })
                    .unwrap();
                }
            });
            for k in 0..100 {
                let p = parity(k).unwrap();
                state
                    .observe(
                        &state.ready_recorded[p],
                        k,
                        Instant::now() + Duration::from_secs(2),
                    )
                    .unwrap();
                assert_eq!(payload[p].load(Ordering::Relaxed), k ^ 0x5eed);
                record_then_publish(&state.done_recorded[p], k, || {
                    std::thread::yield_now();
                    Ok(())
                })
                .unwrap();
            }
        });
    }

    #[test]
    fn unrecorded_events_time_out_and_poison_wakes_waiters() {
        let state = Protocol::new();
        assert!(state
            .observe(&state.ready_recorded[0], 0, Instant::now())
            .is_err());
        assert!(state.reuse(2, Instant::now()).is_err());
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                state.observe(
                    &state.ready_recorded[0],
                    0,
                    Instant::now() + Duration::from_secs(2),
                )
            });
            state.poisoned.store(true, Ordering::Release);
            assert!(waiter.join().unwrap().unwrap_err().contains("poisoned"));
        });
    }

    #[test]
    fn old_publications_never_satisfy_future_generation() {
        let state = Protocol::new();
        state.ready_recorded[0].store(0, Ordering::Release);
        assert!(state
            .observe(&state.ready_recorded[0], 2, Instant::now())
            .is_err());
        state.done_recorded[0].store(2, Ordering::Release);
        assert!(state.reuse(2, Instant::now()).is_err());
    }

    #[test]
    fn failed_record_is_never_published() {
        let state = Protocol::new();
        for word in [&state.ready_recorded[0], &state.done_recorded[0]] {
            assert!(
                record_then_publish(word, 0, || Err("injected record failure".into())).is_err()
            );
            assert_eq!(word.load(Ordering::Acquire), NEVER);
            record_then_publish(word, 0, || Ok(())).unwrap();
            assert_eq!(word.load(Ordering::Acquire), 0);
        }
        state.poisoned.store(true, Ordering::Release);
        assert!(state
            .observe(&state.ready_recorded[0], 0, Instant::now())
            .is_err());
    }

    #[test]
    fn drain_ordering_and_failed_drain_quarantine() {
        for failed_side in [None, Some(0), Some(1)] {
            let mut log = Vec::new();
            let result = drain_then_release(
                &mut log,
                |log, side| {
                    log.push(if side == 0 {
                        "drain_source_all_streams"
                    } else {
                        "drain_destination_all_streams"
                    });
                    if failed_side == Some(side) {
                        Err("injected drain failure".into())
                    } else {
                        Ok(())
                    }
                },
                |log| {
                    log.extend(["invalidate_aliases", "free_banks", "free_events"]);
                    Ok(())
                },
            );
            assert_eq!(
                &log[..2],
                &["drain_source_all_streams", "drain_destination_all_streams"]
            );
            if failed_side.is_some() {
                assert!(result.is_err());
                assert_eq!(
                    log.len(),
                    2,
                    "failed drain must never reach resource release"
                );
            } else {
                assert!(result.is_ok());
                assert_eq!(
                    &log[2..],
                    &["invalidate_aliases", "free_banks", "free_events"]
                );
            }
        }
    }
}
