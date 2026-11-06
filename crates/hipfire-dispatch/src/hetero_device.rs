//! One layer's attachment to the device-side hetero split (design v3).
//!
//! Sealed MoE decode calls whose routed experts expose a [`HeteroDeviceSplit`]
//! (via `RoutedExpertWeights::hetero_split`) lower `PublishCold` before the
//! routed GPU tail and `AwaitCold` + `InstallCold` before the canonical combine,
//! in place of the host-synchronous D2 `PrepareSplit`/`JoinSplit`. Every stage
//! is a device kernel, so the call is capturable and replayable.

use rdna_compute::{Gpu, GpuTensor, HeteroSlotPtrs};

use crate::types::DispatchError;

/// Device state one trunk layer's split stages need.
pub struct HeteroDeviceSplit {
    pub layer: u32,
    pub layers: u32,
    /// This layer's mailbox slot (pinned host memory).
    pub slot: HeteroSlotPtrs,
    /// The load's device token counter (shared by every layer; a view).
    pub counter: GpuTensor,
    /// 1 = the expert is cold (host tier, computed by the expert device).
    pub cold_bitmap: GpuTensor,
    /// The primary's routed ids with cold slots masked to -1.
    pub masked_ids: GpuTensor,
    pub n_experts: i32,
    /// `hetero_await_cold` poll bound.
    pub max_polls: u32,
    /// Rows the slot holds (`masked_ids` is `max_rows x top_k`).
    pub max_rows: usize,
    /// `PublishCold` first advances `counter`: a slot whose calls are not
    /// one per forward (the MTP head drafts several times between trunk
    /// forwards) owns its counter, so every request carries a fresh sequence.
    pub advance: bool,
}

fn hip(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(format!("hetero device split: {e}"))
}

impl HeteroDeviceSplit {
    /// `PublishCold`: route into the mailbox, device-masked ids, request word.
    pub fn publish(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        top_k: usize,
        dim: usize,
        x_rot: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.advance {
            gpu.hetero_token_advance(&self.counter).map_err(hip)?;
        }
        gpu.hetero_publish_cold(
            &self.counter,
            self.layer,
            self.layers,
            x_rot,
            ids,
            weights,
            &self.cold_bitmap,
            self.n_experts,
            rows as i32,
            top_k as i32,
            dim as i32,
            &self.slot,
            &self.masked_ids,
        )
        .map_err(hip)
    }

    /// `AwaitCold`: bounded wait for the expert device's response.
    pub fn await_cold(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        gpu.hetero_await_cold(
            &self.counter,
            self.layer,
            self.layers,
            &self.slot,
            self.max_polls,
        )
        .map_err(hip)
    }

    /// `InstallCold`: the cold slots' rows into the expanded down buffer.
    pub fn install(
        &self,
        gpu: &mut Gpu,
        slots: usize,
        dim: usize,
        ids: &GpuTensor,
        down_expanded: &GpuTensor,
    ) -> Result<(), DispatchError> {
        gpu.hetero_install_cold(
            &self.slot,
            ids,
            &self.cold_bitmap,
            self.n_experts,
            slots as u32,
            dim as i32,
            down_expanded,
        )
        .map_err(hip)
    }
}

// ───────────────────────────── prefill ─────────────────────────────

use std::sync::{Arc, Mutex};

use crate::hetero_mailbox::{int4_row_bytes, PrefillGeometry, PrefillOffsets, ROWS_F16, ROWS_INT4};

/// One prefill slot of the mailbox and the primary's buffers for it: a chunk
/// (or a layer-major tile) in flight owns one slot until its join.
struct PrefillSlot {
    offsets: PrefillOffsets,
    /// F16 rows staged for the copy out, the chunk's int4 sidecar snapshot
    /// (the primary's own IU4 scratch is reused by later stages while the out
    /// stream still reads), and the BF16 partial copied back.
    x_f16: GpuTensor,
    xq: GpuTensor,
    partial: GpuTensor,
    published: hip_bridge::Event,
    returned: hip_bridge::Event,
    /// A publish into this slot awaits its join.
    pending: std::sync::atomic::AtomicBool,
    /// The slot's error word copied into primary memory by the back stream
    /// after its waits: the add reads it per thread, and a remote word would
    /// cost every thread a read across the link.
    err_local: GpuTensor,
    /// What `publish` staged for `send`: rows and the sidecar snapshot.
    staged: Mutex<Option<(usize, Option<u64>)>>,
    /// The chunk's route (i32 ids, F32 weights), staged in primary memory and
    /// copied across by the out stream: remote stores from the main stream
    /// would make its next event wait for their flush across the link.
    route_local: GpuTensor,
}

/// Load-wide prefill exchange state, shared by every layer of the load: the
/// mailbox's prefill slots with the primary's buffers, the two side streams,
/// the slot the next publish uses and whether its join is deferred to the
/// caller (the layer-major schedule keeps several tiles in flight).
pub struct HeteroPrefillShared {
    pub mailbox: u64,
    /// `sub_chunks` is a slot's count; a chunk uses
    /// [`PrefillGeometry::sub_chunks_for`] of them.
    pub geom: PrefillGeometry,
    pub n_experts: i32,
    /// The load's device token counter (a view).
    pub counter: GpuTensor,
    /// `hetero_await_cold` poll bound per sub-chunk.
    pub max_polls: u32,
    slots: Vec<PrefillSlot>,
    current: std::sync::atomic::AtomicUsize,
    defer: std::sync::atomic::AtomicBool,
    out: Mutex<Option<hip_bridge::Stream>>,
    back: Mutex<Option<hip_bridge::Stream>>,
    /// `HIPFIRE_HETERO_PREFILL_TRACE=1`: per-layer stream times (synchronizes
    /// at every join; diagnosis only; immediate joins only).
    trace: Option<PrefillTrace>,
}

struct PrefillTrace {
    start: hip_bridge::Event,
    hot: hip_bridge::Event,
    /// (layers, rows, publish->hot-combine ms, publish->exchange-returned ms)
    sums: Mutex<(u64, u64, f64, f64)>,
    /// Sub-chunks sent as int4 / F16, and calls with no sidecar offered.
    int4: std::sync::atomic::AtomicU64,
    f16: std::sync::atomic::AtomicU64,
    no_xq: std::sync::atomic::AtomicU64,
}

// SAFETY: plain device handles used only from the forward that owns the load.
unsafe impl Send for HeteroPrefillShared {}
unsafe impl Sync for HeteroPrefillShared {}

fn word(addr: u64) -> &'static std::sync::atomic::AtomicU32 {
    // SAFETY: a word of the live pinned mailbox.
    unsafe { &*(addr as usize as *const std::sync::atomic::AtomicU32) }
}

impl HeteroPrefillShared {
    /// Allocate the primary's buffers, streams and events on `gpu`, one set per
    /// slot of `slot_offsets`.
    pub fn new(
        gpu: &mut Gpu,
        mailbox: u64,
        slot_offsets: &[PrefillOffsets],
        geom: PrefillGeometry,
        n_experts: usize,
        counter: GpuTensor,
        max_polls: u32,
    ) -> Result<Self, DispatchError> {
        if slot_offsets.is_empty() {
            return Err(DispatchError::Hip("hetero prefill: no slot".into()));
        }
        let words = (geom.max_rows * geom.dim).div_ceil(2);
        let xq_words = (geom.max_rows * int4_row_bytes(geom.dim)).div_ceil(4);
        let mut slots = Vec::with_capacity(slot_offsets.len());
        for &offsets in slot_offsets {
            slots.push(PrefillSlot {
                offsets,
                x_f16: gpu
                    .alloc_tensor(&[words], rdna_compute::DType::F32)
                    .map_err(hip)?,
                xq: gpu
                    .alloc_tensor(&[xq_words], rdna_compute::DType::F32)
                    .map_err(hip)?,
                partial: gpu
                    .alloc_tensor(&[words], rdna_compute::DType::F32)
                    .map_err(hip)?,
                // Dependency-only events (no timestamps), per hip_bridge's guidance.
                published: gpu
                    .hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)
                    .map_err(hip)?,
                returned: gpu
                    .hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)
                    .map_err(hip)?,
                pending: Default::default(),
                err_local: gpu
                    .alloc_tensor(&[64], rdna_compute::DType::F32)
                    .map_err(hip)?,
                staged: Mutex::new(None),
                route_local: gpu
                    .alloc_tensor(&[2 * geom.max_rows * geom.top_k], rdna_compute::DType::F32)
                    .map_err(hip)?,
            });
        }
        Ok(Self {
            mailbox,
            geom,
            n_experts: n_experts as i32,
            counter,
            max_polls,
            slots,
            current: Default::default(),
            defer: Default::default(),
            out: Mutex::new(Some(gpu.hip.stream_create_non_blocking().map_err(hip)?)),
            back: Mutex::new(Some(gpu.hip.stream_create_non_blocking().map_err(hip)?)),
            trace: if hipfire_config::developer_var("HIPFIRE_HETERO_PREFILL_TRACE").is_ok_and(|v| v == "1") {
                Some(PrefillTrace {
                    start: gpu.hip.event_create().map_err(hip)?,
                    hot: gpu.hip.event_create().map_err(hip)?,
                    sums: Mutex::new((0, 0, 0.0, 0.0)),
                    int4: Default::default(),
                    f16: Default::default(),
                    no_xq: Default::default(),
                })
            } else {
                None
            },
        })
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The slot the next publish uses (the layer-major schedule's tile).
    pub fn set_slot(&self, slot: usize) -> Result<(), DispatchError> {
        if slot >= self.slots.len() {
            return Err(DispatchError::Hip(format!(
                "hetero prefill: slot {slot} of {}",
                self.slots.len()
            )));
        }
        self.current
            .store(slot, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn slot(&self) -> usize {
        self.current.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// While set, sealed calls leave `JoinColdRows` to the caller
    /// ([`HeteroPrefillSplit::join_slot`]).
    pub fn set_defer_join(&self, on: bool) {
        self.defer.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn defer_join(&self) -> bool {
        self.defer.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The expert device's int4 floor (0 = F16 only), published at its start.
    fn int4_min_rows(&self) -> usize {
        word(self.mailbox + self.slots[0].offsets.int4_min_rows as u64)
            .load(std::sync::atomic::Ordering::Acquire) as usize
    }

    /// Forget every owed join (an aborted request; its partials are dropped).
    pub fn clear_pending(&self) {
        for s in &self.slots {
            s.pending.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The first nonzero error word of any slot (a timed-out wait).
    pub fn error_word(&self) -> u32 {
        self.slots
            .iter()
            .map(|s| {
                word(self.mailbox + s.offsets.error as u64)
                    .load(std::sync::atomic::Ordering::Acquire)
            })
            .find(|&e| e != 0)
            .unwrap_or(0)
    }
}

/// One trunk layer's prefill attachment to the hetero split.
pub struct HeteroPrefillSplit {
    pub layer: u32,
    pub layers: u32,
    pub shared: Arc<HeteroPrefillShared>,
    /// 1 = cold (computed by the expert device).
    pub cold_bitmap: GpuTensor,
    /// The primary's routed ids with cold slots masked to -1 (`max_rows x top_k`).
    pub masked_ids: GpuTensor,
}

/// Run `f` with `slot`'s stream as the active stream, restoring both.
fn on_stream<T>(
    gpu: &mut Gpu,
    slot: &Mutex<Option<hip_bridge::Stream>>,
    f: impl FnOnce(&mut Gpu) -> Result<T, DispatchError>,
) -> Result<T, DispatchError> {
    let mut held = slot
        .lock()
        .map_err(|_| DispatchError::Hip("hetero prefill: stream lock poisoned".into()))?;
    let side = held
        .take()
        .ok_or_else(|| DispatchError::Hip("hetero prefill: side stream in use".into()))?;
    let main = std::mem::replace(&mut gpu.active_stream, Some(side));
    let out = f(gpu);
    *held = std::mem::replace(&mut gpu.active_stream, main);
    out
}

impl HeteroPrefillSplit {
    pub fn max_rows(&self) -> usize {
        self.shared.geom.max_rows
    }

    /// `PublishColdRows`: stage the chunk (F16 rows, the route into the slot,
    /// device-masked ids), then queue per sub-chunk on the out stream the copy
    /// of its rows (the int4 sidecar `xq` when given and the expert device
    /// admits the block, else F16) and its release, and on the back stream
    /// its bounded wait and the copy back of its BF16 partial. Nothing waits
    /// on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn publish(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        top_k: usize,
        x_norm: &GpuTensor,
        ids: &GpuTensor,
        weights: &GpuTensor,
        xq: Option<*mut std::ffi::c_void>,
    ) -> Result<(), DispatchError> {
        let sh = &*self.shared;
        let g = sh.geom;
        if rows == 0 || rows > g.max_rows || top_k != g.top_k {
            return Err(DispatchError::Hip(format!(
                "hetero prefill: {rows} rows x top {top_k} outside the slot"
            )));
        }
        let ps = &sh.slots[sh.slot()];
        if ps.pending.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return Err(DispatchError::Hip(
                "hetero prefill: publish into a slot whose join is still owed".into(),
            ));
        }
        let (mb, o, dim) = (sh.mailbox, ps.offsets, g.dim);
        if let Some(t) = &sh.trace {
            gpu.hip
                .event_record(&t.start, gpu.active_stream.as_ref())
                .map_err(hip)?;
            if xq.is_none() {
                t.no_xq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        gpu.hetero_publish_rows(
            x_norm,
            ids,
            weights,
            &self.cold_bitmap,
            sh.n_experts,
            rows as i32,
            top_k as i32,
            dim as i32,
            &ps.x_f16,
            ps.route_local.buf.as_ptr() as u64,
            ps.route_local.buf.as_ptr() as u64 + (g.max_rows * g.top_k * 4) as u64,
            &self.masked_ids,
        )
        .map_err(hip)?;
        let i4 = int4_row_bytes(dim);
        // Snapshot the sidecar on the main stream, before any later stage of
        // this call can reuse the primary's IU4 scratch.
        let xq = match xq {
            Some(ptr) => {
                // A plain copy kernel on the main stream (the runtime's blit path
                // left ~0.9 ms of idle before the next kernel).
                // SAFETY: the chunk's live int4 sidecar (`rows` rows).
                let src = GpuTensor {
                    buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, rows * i4) },
                    shape: vec![rows * i4 / 4],
                    dtype: rdna_compute::DType::F32,
                };
                gpu.copy_f32_buffer(&ps.xq, &src, rows * i4 / 4)
                    .map_err(hip)?;
                Some(ps.xq.buf.as_ptr())
            }
            None => None,
        };
        gpu.hip
            .event_record(&ps.published, gpu.active_stream.as_ref())
            .map_err(hip)?;
        *ps.staged
            .lock()
            .map_err(|_| DispatchError::Hip("hetero prefill: staged lock poisoned".into()))? =
            Some((rows, xq.map(|p| p as u64)));
        Ok(())
    }

    /// `SendColdRows`: enqueue the exchange `publish` staged, per sub-chunk on
    /// the out stream the copy of its rows (int4 when the sidecar exists and
    /// the expert device admits the block, else F16) and its release, and on
    /// the back stream its bounded wait and the copy back of its BF16 partial,
    /// then the error word. Nothing waits on the host.
    pub fn send(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let sh = &*self.shared;
        let g = sh.geom;
        let ps = &sh.slots[sh.slot()];
        let (rows, xq) = ps
            .staged
            .lock()
            .map_err(|_| DispatchError::Hip("hetero prefill: staged lock poisoned".into()))?
            .take()
            .ok_or_else(|| DispatchError::Hip("hetero prefill: send without a publish".into()))?;
        let xq = xq.map(|p| p as usize as *mut std::ffi::c_void);
        let (mb, o, dim) = (sh.mailbox, ps.offsets, g.dim);
        let i4 = int4_row_bytes(dim);
        let subs = g.sub_chunks_for(rows);
        let split = PrefillGeometry {
            sub_chunks: subs,
            ..g
        };
        let floor = sh.int4_min_rows();
        // SAFETY: views of the live pinned prefill slot.
        let (mb_x, mb_part) = unsafe {
            (
                hip_bridge::DeviceBuffer::from_raw(
                    (mb + o.x_norm as u64) as *mut std::ffi::c_void,
                    o.ids - o.x_norm,
                ),
                hip_bridge::DeviceBuffer::from_raw(
                    (mb + o.partial as u64) as *mut std::ffi::c_void,
                    g.max_rows * dim * 2,
                ),
            )
        };
        on_stream(gpu, &sh.out, |gpu| {
            gpu.hip
                .stream_wait_event_on(gpu.active_stream.as_ref(), &ps.published)
                .map_err(hip)?;
            // The route first (every sub-chunk's release follows it in order).
            let route = rows * g.top_k * 4;
            // SAFETY: the slot's ids and weights fields in the live pinned mailbox.
            let (mb_ids, mb_w) = unsafe {
                (
                    hip_bridge::DeviceBuffer::from_raw(
                        (mb + o.ids as u64) as *mut std::ffi::c_void,
                        route,
                    ),
                    hip_bridge::DeviceBuffer::from_raw(
                        (mb + o.weights as u64) as *mut std::ffi::c_void,
                        route,
                    ),
                )
            };
            let st = gpu.active_stream.as_ref().expect("side stream installed");
            gpu.hip
                .memcpy_dtod_async_at(&mb_ids, 0, &ps.route_local.buf, 0, route, st)
                .map_err(hip)?;
            gpu.hip
                .memcpy_dtod_async_at(
                    &mb_w,
                    0,
                    &ps.route_local.buf,
                    g.max_rows * g.top_k * 4,
                    route,
                    st,
                )
                .map_err(hip)?;
            for i in 0..subs {
                let (off, n) = split.sub_range(rows, i);
                if n == 0 {
                    continue;
                }
                let int4 = xq.filter(|_| floor > 0 && n >= floor);
                let format = if let Some(ptr) = int4 {
                    // Block-major sidecar: one run per K128 block, gathered
                    // into the block's own [K/128][n] layout.
                    // SAFETY: the split's own sidecar snapshot (`rows` rows).
                    let src = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, rows * i4) };
                    gpu.hip
                        .memcpy_dtod_2d_async_at(
                            &mb_x,
                            off * i4,
                            n * 72,
                            &src,
                            off * 72,
                            rows * 72,
                            n * 72,
                            dim / 128,
                            gpu.active_stream.as_ref(),
                        )
                        .map_err(hip)?;
                    ROWS_INT4
                } else {
                    let st = gpu.active_stream.as_ref().expect("side stream installed");
                    gpu.hip
                        .memcpy_dtod_async_at(
                            &mb_x,
                            off * dim * 2,
                            &ps.x_f16.buf,
                            off * dim * 2,
                            n * dim * 2,
                            st,
                        )
                        .map_err(hip)?;
                    ROWS_F16
                };
                if let Some(t) = &sh.trace {
                    let c = if format == ROWS_INT4 { &t.int4 } else { &t.f16 };
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let w = o.sub[i];
                gpu.hetero_release_rows(
                    &sh.counter,
                    self.layer,
                    self.layers,
                    off as i32,
                    n as i32,
                    format,
                    mb + w.layer as u64,
                    mb + w.req_seq as u64,
                )
                .map_err(hip)?;
            }
            Ok(())
        })?;
        on_stream(gpu, &sh.back, |gpu| {
            gpu.hip
                .stream_wait_event_on(gpu.active_stream.as_ref(), &ps.published)
                .map_err(hip)?;
            for i in 0..subs {
                let (off, n) = split.sub_range(rows, i);
                if n == 0 {
                    continue;
                }
                let w = o.sub[i];
                let slot = HeteroSlotPtrs {
                    req_seq: mb + w.req_seq as u64,
                    resp_seq: mb + w.resp_seq as u64,
                    error: mb + o.error as u64,
                    rows: 0,
                    x_rot: 0,
                    ids: 0,
                    weights: 0,
                    cold_out: 0,
                };
                gpu.hetero_await_cold(&sh.counter, self.layer, self.layers, &slot, sh.max_polls)
                    .map_err(hip)?;
                let st = gpu.active_stream.as_ref().expect("side stream installed");
                gpu.hip
                    .memcpy_dtod_async_at(
                        &ps.partial.buf,
                        off * dim * 2,
                        &mb_part,
                        off * dim * 2,
                        n * dim * 2,
                        st,
                    )
                    .map_err(hip)?;
            }
            // The error word, after every wait of this chunk, into primary memory.
            // SAFETY: the slot's error word in the live pinned mailbox.
            let err = unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    (mb + o.error as u64) as *mut std::ffi::c_void,
                    4,
                )
            };
            let st = gpu.active_stream.as_ref().expect("side stream installed");
            gpu.hip
                .memcpy_dtod_async_at(&ps.err_local.buf, 0, &err, 0, 4, st)
                .map_err(hip)?;
            gpu.hip
                .event_record(&ps.returned, gpu.active_stream.as_ref())
                .map_err(hip)
        })
    }

    /// `JoinColdRows`: the main stream waits for the exchange, then adds the
    /// BF16 cold partial into `x_batch` (skipped after a timed-out wait).
    pub fn join(
        &self,
        gpu: &mut Gpu,
        rows: usize,
        x_batch: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let sh = &*self.shared;
        let s = sh.slot();
        if let Some(t) = &sh.trace {
            let slot = &sh.slots[s];
            gpu.hip
                .event_record(&t.hot, gpu.active_stream.as_ref())
                .map_err(hip)?;
            gpu.hip.event_synchronize(&t.hot).map_err(hip)?;
            gpu.hip.event_synchronize(&slot.returned).map_err(hip)?;
            let hot = gpu.hip.event_elapsed_ms(&t.start, &t.hot).map_err(hip)? as f64;
            let cold = gpu
                .hip
                .event_elapsed_ms(&t.start, &slot.returned)
                .map_err(hip)? as f64;
            if let Ok(mut sums) = t.sums.lock() {
                *sums = (
                    sums.0 + 1,
                    sums.1 + rows as u64,
                    sums.2 + hot,
                    sums.3 + cold,
                );
                if sums.0 % self.layers as u64 == 0 {
                    let n = sums.0 as f64;
                    eprintln!(
                        "hetero prefill trace: {} layers, {:.0} rows/layer avg: publish->hot combine {:.2} ms, publish->cold returned {:.2} ms; blocks int4 {} f16 {}, calls without a sidecar {} (floor {})",
                        sums.0,
                        sums.1 as f64 / n,
                        sums.2 / n,
                        sums.3 / n,
                        t.int4.swap(0, std::sync::atomic::Ordering::Relaxed),
                        t.f16.swap(0, std::sync::atomic::Ordering::Relaxed),
                        t.no_xq.swap(0, std::sync::atomic::Ordering::Relaxed),
                        sh.int4_min_rows()
                    );
                    *sums = (0, 0, 0.0, 0.0);
                }
            }
        }
        self.join_slot(gpu, s, rows, x_batch)
    }

    /// The join of slot `slot` (the layer-major schedule's deferred join):
    /// the main stream waits for that slot's exchange, then adds its BF16 cold
    /// partial into `x_batch` (skipped after any timed-out wait).
    pub fn join_slot(
        &self,
        gpu: &mut Gpu,
        slot: usize,
        rows: usize,
        x_batch: &GpuTensor,
    ) -> Result<(), DispatchError> {
        let sh = &*self.shared;
        let st = sh.slots.get(slot).ok_or_else(|| {
            DispatchError::Hip(format!("hetero prefill: slot {slot} of {}", sh.slots.len()))
        })?;
        if !st.pending.swap(false, std::sync::atomic::Ordering::Relaxed) {
            // The call did not take the exchange (its own combine carried the
            // cold experts): nothing to add.
            return Ok(());
        }
        gpu.hip
            .stream_wait_event_on(gpu.active_stream.as_ref(), &st.returned)
            .map_err(hip)?;
        gpu.hetero_add_partial(
            st.err_local.buf.as_ptr() as u64,
            &st.partial,
            (rows * sh.geom.dim) as i32,
            x_batch,
        )
        .map_err(hip)
    }
}
