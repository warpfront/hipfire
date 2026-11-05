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
        gpu.hetero_await_cold(&self.counter, self.layer, self.layers, &self.slot, self.max_polls)
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

/// Load-wide prefill exchange state, shared by every layer of the load (layers
/// run one at a time): the prefill slot of the mailbox, the primary's staging
/// and partial buffers, and the two side streams with their events.
pub struct HeteroPrefillShared {
    pub mailbox: u64,
    pub offsets: PrefillOffsets,
    /// `sub_chunks` is the slot's count; a chunk uses
    /// [`PrefillGeometry::sub_chunks_for`] of them.
    pub geom: PrefillGeometry,
    pub n_experts: i32,
    /// The load's device token counter (a view).
    pub counter: GpuTensor,
    /// `hetero_await_cold` poll bound per sub-chunk.
    pub max_polls: u32,
    /// F16 rows staged for the copy out, the chunk's int4 sidecar snapshot
    /// (the primary's own IU4 scratch is reused by later stages while the out
    /// stream still reads), and the BF16 partial copied back.
    x_f16: GpuTensor,
    xq: GpuTensor,
    partial: GpuTensor,
    out: Mutex<Option<hip_bridge::Stream>>,
    back: Mutex<Option<hip_bridge::Stream>>,
    published: hip_bridge::Event,
    returned: hip_bridge::Event,
    /// `HIPFIRE_HETERO_PREFILL_TRACE=1`: per-layer stream times (synchronizes
    /// at every join; diagnosis only).
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

impl HeteroPrefillShared {
    /// Allocate the primary's buffers, streams and events on `gpu`.
    pub fn new(
        gpu: &mut Gpu,
        mailbox: u64,
        offsets: PrefillOffsets,
        geom: PrefillGeometry,
        n_experts: usize,
        counter: GpuTensor,
        max_polls: u32,
    ) -> Result<Self, DispatchError> {
        let words = (geom.max_rows * geom.dim).div_ceil(2);
        let x_f16 = gpu.alloc_tensor(&[words], rdna_compute::DType::F32).map_err(hip)?;
        let partial = gpu.alloc_tensor(&[words], rdna_compute::DType::F32).map_err(hip)?;
        let xq = gpu
            .alloc_tensor(&[(geom.max_rows * int4_row_bytes(geom.dim)).div_ceil(4)], rdna_compute::DType::F32)
            .map_err(hip)?;
        Ok(Self {
            mailbox,
            offsets,
            geom,
            n_experts: n_experts as i32,
            counter,
            max_polls,
            x_f16,
            xq,
            partial,
            out: Mutex::new(Some(gpu.hip.stream_create_non_blocking().map_err(hip)?)),
            back: Mutex::new(Some(gpu.hip.stream_create_non_blocking().map_err(hip)?)),
            published: gpu.hip.event_create().map_err(hip)?,
            returned: gpu.hip.event_create().map_err(hip)?,
            trace: if std::env::var("HIPFIRE_HETERO_PREFILL_TRACE").is_ok_and(|v| v == "1") {
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

    /// The expert device's int4 floor (0 = F16 only), published at its start.
    fn int4_min_rows(&self) -> usize {
        let addr = self.mailbox as usize + self.offsets.int4_min_rows;
        // SAFETY: a word of the live pinned mailbox.
        unsafe { &*(addr as *const std::sync::atomic::AtomicU32) }.load(std::sync::atomic::Ordering::Acquire) as usize
    }

    /// The shared error word (any sub-chunk's timed-out wait).
    pub fn error_word(&self) -> u32 {
        let addr = self.mailbox as usize + self.offsets.error;
        // SAFETY: a word of the live pinned mailbox.
        unsafe { &*(addr as *const std::sync::atomic::AtomicU32) }.load(std::sync::atomic::Ordering::Acquire)
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
    let mut held = slot.lock().map_err(|_| DispatchError::Hip("hetero prefill: stream lock poisoned".into()))?;
    let side = held.take().ok_or_else(|| DispatchError::Hip("hetero prefill: side stream in use".into()))?;
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
            return Err(DispatchError::Hip(format!("hetero prefill: {rows} rows x top {top_k} outside the slot")));
        }
        let (mb, o, dim) = (sh.mailbox, sh.offsets, g.dim);
        if let Some(t) = &sh.trace {
            gpu.hip.event_record(&t.start, gpu.active_stream.as_ref()).map_err(hip)?;
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
            &sh.x_f16,
            mb + o.ids as u64,
            mb + o.weights as u64,
            &self.masked_ids,
        )
        .map_err(hip)?;
        let i4 = int4_row_bytes(dim);
        // Snapshot the sidecar on the main stream, before any later stage of
        // this call can reuse the primary's IU4 scratch.
        let xq = match xq {
            Some(ptr) => {
                // SAFETY: the chunk's live int4 sidecar (`rows` rows).
                let src = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, rows * i4) };
                match gpu.active_stream.as_ref() {
                    Some(st) => gpu.hip.memcpy_dtod_async_at(&sh.xq.buf, 0, &src, 0, rows * i4, st),
                    None => gpu.hip.memcpy_dtod_async_default_at(&sh.xq.buf, 0, &src, 0, rows * i4),
                }
                .map_err(hip)?;
                Some(sh.xq.buf.as_ptr())
            }
            None => None,
        };
        gpu.hip.event_record(&sh.published, gpu.active_stream.as_ref()).map_err(hip)?;
        let subs = g.sub_chunks_for(rows);
        let split = PrefillGeometry { sub_chunks: subs, ..g };
        let floor = sh.int4_min_rows();
        // SAFETY: views of the live pinned prefill slot.
        let (mb_x, mb_part) = unsafe {
            (
                hip_bridge::DeviceBuffer::from_raw((mb + o.x_norm as u64) as *mut std::ffi::c_void, o.ids - o.x_norm),
                hip_bridge::DeviceBuffer::from_raw((mb + o.partial as u64) as *mut std::ffi::c_void, g.max_rows * dim * 2),
            )
        };
        on_stream(gpu, &sh.out, |gpu| {
            gpu.hip.stream_wait_event_on(gpu.active_stream.as_ref(), &sh.published).map_err(hip)?;
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
                        .memcpy_dtod_2d_async_at(&mb_x, off * i4, n * 72, &src, off * 72, rows * 72, n * 72, dim / 128, gpu.active_stream.as_ref())
                        .map_err(hip)?;
                    ROWS_INT4
                } else {
                    let st = gpu.active_stream.as_ref().expect("side stream installed");
                    gpu.hip
                        .memcpy_dtod_async_at(&mb_x, off * dim * 2, &sh.x_f16.buf, off * dim * 2, n * dim * 2, st)
                        .map_err(hip)?;
                    ROWS_F16
                };
                if let Some(t) = &sh.trace {
                    let c = if format == ROWS_INT4 { &t.int4 } else { &t.f16 };
                    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                let w = o.sub[i];
                gpu.hetero_release_rows(&sh.counter, self.layer, self.layers, off as i32, n as i32, format, mb + w.layer as u64, mb + w.req_seq as u64)
                    .map_err(hip)?;
            }
            Ok(())
        })?;
        on_stream(gpu, &sh.back, |gpu| {
            gpu.hip.stream_wait_event_on(gpu.active_stream.as_ref(), &sh.published).map_err(hip)?;
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
                    x_rot: 0,
                    ids: 0,
                    weights: 0,
                    cold_out: 0,
                };
                gpu.hetero_await_cold(&sh.counter, self.layer, self.layers, &slot, sh.max_polls).map_err(hip)?;
                let st = gpu.active_stream.as_ref().expect("side stream installed");
                gpu.hip
                    .memcpy_dtod_async_at(&sh.partial.buf, off * dim * 2, &mb_part, off * dim * 2, n * dim * 2, st)
                    .map_err(hip)?;
            }
            gpu.hip.event_record(&sh.returned, gpu.active_stream.as_ref()).map_err(hip)
        })
    }

    /// `JoinColdRows`: the main stream waits for the exchange, then adds the
    /// BF16 cold partial into `x_batch` (skipped after a timed-out wait).
    pub fn join(&self, gpu: &mut Gpu, rows: usize, x_batch: &GpuTensor) -> Result<(), DispatchError> {
        let sh = &*self.shared;
        if let Some(t) = &sh.trace {
            gpu.hip.event_record(&t.hot, gpu.active_stream.as_ref()).map_err(hip)?;
            gpu.hip.event_synchronize(&t.hot).map_err(hip)?;
            gpu.hip.event_synchronize(&sh.returned).map_err(hip)?;
            let hot = gpu.hip.event_elapsed_ms(&t.start, &t.hot).map_err(hip)? as f64;
            let cold = gpu.hip.event_elapsed_ms(&t.start, &sh.returned).map_err(hip)? as f64;
            if let Ok(mut s) = t.sums.lock() {
                *s = (s.0 + 1, s.1 + rows as u64, s.2 + hot, s.3 + cold);
                if s.0 % self.layers as u64 == 0 {
                    let n = s.0 as f64;
                    eprintln!(
                        "hetero prefill trace: {} layers, {:.0} rows/layer avg: publish->hot combine {:.2} ms, publish->cold returned {:.2} ms; blocks int4 {} f16 {}, calls without a sidecar {} (floor {})",
                        s.0,
                        s.1 as f64 / n,
                        s.2 / n,
                        s.3 / n,
                        t.int4.swap(0, std::sync::atomic::Ordering::Relaxed),
                        t.f16.swap(0, std::sync::atomic::Ordering::Relaxed),
                        t.no_xq.swap(0, std::sync::atomic::Ordering::Relaxed),
                        sh.int4_min_rows()
                    );
                    *s = (0, 0, 0.0, 0.0);
                }
            }
        }
        gpu.hip.stream_wait_event_on(gpu.active_stream.as_ref(), &sh.returned).map_err(hip)?;
        gpu.hetero_add_partial(sh.mailbox + sh.offsets.error as u64, &sh.partial, (rows * sh.geom.dim) as i32, x_batch)
            .map_err(hip)
    }
}
