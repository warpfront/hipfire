//! Strix responder of the device-side hetero split (design v3).
//!
//! One gfx1151 `Gpu` on a worker thread. At start it records one HIP graph per
//! trunk layer:
//! `hetero_mask_hot -> qt44 gate/up -> SiLU+rotate -> qt53 down -> hetero_store_seq`.
//! The graph reads the layer's mailbox slot (activation and ids) and writes its
//! cold rows and response word. The pointer tables of the layer's host-tier
//! experts and its cold bitmap are uploaded once.
//!
//! The thread then scans every slot's request word and launches the graph of
//! any layer whose request changed since it last answered. It never predicts
//! sequence numbers: the graph answers with the request's own value, so the
//! responder cannot drift from the primary's token counter. While idle it backs
//! off from spinning to sleeping (at most 1 ms). Waits on the primary side are
//! bounded there (`hetero_await_cold`); this thread only stops on shutdown or a
//! HIP failure, which it records for the host.
//!
//! With a prefill slot configured it also watches each prefill sub-chunk's
//! request word: a change carries a layer, a first row and a row count, and the
//! thread queues that block's cold prefill (`hetero_prefill::run_cold_prefill`,
//! eager: row counts vary) followed by `hetero_store_seq` on the same stream. Prefill and decode never
//! interleave (the primary waits on each), so one thread serves both.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::hetero_mailbox::{
    int4_row_bytes, MailboxGeometry, PrefillGeometry, PrefillOffsets, MIN_SUB_CHUNK_ROWS, ROWS_F16, ROWS_INT4,
};
use crate::hetero_prefill::{
    run_cold_prefill, ColdPrefillConfig, ColdPrefillIo, ColdPrefillLayer, ColdPrefillScratch,
};
use crate::igpu_split::check_device_accessible;
use crate::types::DispatchError;

fn err(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("hetero responder: {message}"))
}

/// One trunk layer's cold experts: full-layer pointer tables of the host-tier
/// experts (0 for hot ones) and the cold bitmap (1 = cold).
#[derive(Clone, Debug)]
pub struct ColdLayer {
    pub gate_up: Vec<u64>,
    pub down: Vec<u64>,
    pub cold_bitmap: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ResponderConfig {
    /// HIP logical device of the gfx1151.
    pub ordinal: i32,
    pub geom: MailboxGeometry,
    pub mailbox_base: u64,
    pub n_experts: usize,
    pub mi: usize,
    /// The prefill slot and the cold-prefill geometry, when prefill is served.
    /// The prefill slots (one per chunk or layer-major tile in flight), all of
    /// one cold-prefill geometry; empty when prefill is not served.
    pub prefill: Vec<PrefillServe>,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillServe {
    /// Offsets of the prefill slot from the mailbox base.
    pub offsets: PrefillOffsets,
    /// Live sub-chunks (`offsets.sub[..sub_chunks]`).
    pub sub_chunks: usize,
    pub cold: ColdPrefillConfig,
}

impl PrefillServe {
    fn geometry(&self) -> PrefillGeometry {
        PrefillGeometry { max_rows: self.cold.max_rows, top_k: self.cold.top_k, dim: self.cold.dim, sub_chunks: self.sub_chunks }
    }
}

pub struct HeteroResponder {
    stop: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    answered: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

/// The slot's request word, read from the CPU (coherent pinned memory).
fn req_word(cfg: &ResponderConfig, layer: usize) -> &'static AtomicU32 {
    let addr = cfg.mailbox_base as usize + cfg.geom.slot(layer).req_seq;
    // SAFETY: a 4-byte aligned word inside the live pinned mailbox, which the
    // caller keeps alive for the responder's lifetime.
    unsafe { &*(addr as *const AtomicU32) }
}

/// A word of the mailbox, read from the CPU (coherent pinned memory).
fn word(cfg: &ResponderConfig, offset: usize) -> &'static AtomicU32 {
    // SAFETY: a 4-byte aligned word inside the live pinned mailbox, which the
    // caller keeps alive for the responder's lifetime.
    unsafe { &*((cfg.mailbox_base as usize + offset) as *const AtomicU32) }
}

struct Prefill {
    serves: Vec<PrefillServe>,
    layers: Vec<ColdPrefillLayer>,
    scratch: ColdPrefillScratch,
}

struct Graphs {
    /// Its `active_stream` is the one stream every graph is captured on and
    /// launched on (and every prefill job queued on).
    gpu: rdna_compute::Gpu,
    execs: Vec<hip_bridge::GraphExec>,
    prefill: Option<Prefill>,
    _keep: Vec<rdna_compute::GpuTensor>,
}

impl Prefill {
    /// The slot's fields for the block starting at row `row_off`, its rows in
    /// `format`.
    fn io(&self, base: u64, slot: usize, row_off: usize, format: u32) -> ColdPrefillIo {
        let o = self.serves[slot].offsets;
        let c = self.serves[slot].cold;
        let row_bytes = if format == ROWS_INT4 { int4_row_bytes(c.dim) } else { c.dim * 2 };
        let (row16, route) = ((row_off * c.dim * 2) as u64, (row_off * c.top_k * 4) as u64);
        ColdPrefillIo {
            x_norm: base + o.x_norm as u64 + (row_off * row_bytes) as u64,
            format,
            ids: base + o.ids as u64 + route,
            weights: base + o.weights as u64 + route,
            partial: base + o.partial as u64 + row16,
        }
    }
}

/// Queue one prefill sub-chunk job and its answer.
fn serve_prefill(
    gr: &mut Graphs,
    cfg: &ResponderConfig,
    slot: usize,
    sub: usize,
    layer: u32,
    row_off: u32,
    rows: u32,
    format: u32,
) -> Result<(), String> {
    let p = gr.prefill.as_ref().ok_or("prefill request without a prefill slot")?;
    let (layer, row_off, rows) = (layer as usize, row_off as usize, rows as usize);
    if layer >= p.layers.len() || rows == 0 || row_off + rows > p.scratch.max_rows() {
        return Err(format!("prefill request for layer {layer}, rows {row_off}+{rows} is out of range"));
    }
    let io = p.io(cfg.mailbox_base, slot, row_off, format);
    run_cold_prefill(&mut gr.gpu, &p.layers[layer], &p.scratch, rows, io).map_err(|e| format!("{e:?}"))?;
    let w = p.serves[slot].offsets.sub[sub];
    let base = cfg.mailbox_base;
    gr.gpu
        .hetero_store_seq(base + w.req_seq as u64, base + w.resp_seq as u64)
        .map_err(|e| e.to_string())
}

/// Build every layer's prefill binding and compile every kernel the cold
/// prefill can select (one call per row-count class), so no request pays a
/// JIT compile inside the primary's bounded wait. The warm-up writes a
/// synthetic route into the idle prefill slot before any primary uses it.
fn build_prefill(gpu: &mut rdna_compute::Gpu, cfg: &ResponderConfig, serves: &[PrefillServe], layers: &[ColdLayer]) -> Result<Prefill, String> {
    let serve = serves[0];
    let c = serve.cold;
    if serves.iter().any(|s| s.cold.max_rows != c.max_rows || s.sub_chunks != serve.sub_chunks) {
        return Err("prefill slots differ in geometry".into());
    }
    if c.n_experts != cfg.n_experts || c.mi != cfg.mi || c.dim != cfg.geom.dim || c.top_k != cfg.geom.top_k {
        return Err(format!("prefill geometry {c:?} disagrees with the decode mailbox {:?}", cfg.geom));
    }
    let bound = layers
        .iter()
        .map(|l| ColdPrefillLayer::new(gpu, c, l))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{e:?}"))?;
    let scratch = ColdPrefillScratch::new(gpu, c).map_err(|e| format!("{e:?}"))?;
    let p = Prefill { serves: serves.to_vec(), layers: bound, scratch };
    let warm = layers
        .iter()
        .enumerate()
        .find_map(|(l, cl)| cl.cold_bitmap.iter().position(|&b| b != 0).map(|e| (l, e)));
    let mut int4_min_rows = 0u32;
    if let Some((layer, expert)) = warm {
        let io = p.io(cfg.mailbox_base, 0, 0, ROWS_F16);
        let slots = c.max_rows * c.top_k;
        // SAFETY: the idle prefill slot's ids/weights/x fields, sized for max_rows.
        unsafe {
            std::slice::from_raw_parts_mut(io.ids as *mut i32, slots).fill(expert as i32);
            std::slice::from_raw_parts_mut(io.weights as *mut f32, slots).fill(0.0);
            std::slice::from_raw_parts_mut(io.x_norm as *mut u16, c.max_rows * c.dim).fill(0);
        }
        let g = serve.geometry();
        let mut rows = vec![1usize, 2, 8, 16, 64, 256, c.max_rows, g.sub_range(c.max_rows, 0).1];
        rows.retain(|&r| r <= c.max_rows);
        for r in rows {
            run_cold_prefill(gpu, &p.layers[layer], &p.scratch, r, io).map_err(|e| format!("warm-up {r} rows: {e:?}"))?;
        }
        let st = gpu.active_stream.as_ref().expect("responder stream is installed");
        gpu.hip.stream_synchronize(st).map_err(|e| e.to_string())?;
        // The int4 floor: blocks from MIN_SUB_CHUNK_ROWS to max_rows must
        // take the symmetric IU4 arm (a zero sidecar is a valid activation).
        // Any refusal publishes 0 and the primary sends F16 only.
        if c.max_rows >= MIN_SUB_CHUNK_ROWS {
            let io4 = p.io(cfg.mailbox_base, 0, 0, ROWS_INT4);
            // SAFETY: the idle prefill slot's row area holds max_rows int4 rows.
            unsafe {
                std::slice::from_raw_parts_mut(io4.x_norm as *mut u8, c.max_rows * int4_row_bytes(c.dim)).fill(0);
            }
            // Every layer must take the symmetric route (its own header check),
            // since the floor is one value for the whole load.
            let symmetric = |l: &ColdPrefillLayer| {
                l.route_policy().is_some_and(|p| {
                    p.capability == crate::families::moe::MoeRouteCapability::Qt44Qt53GroupedSymmetric
                })
            };
            let asym = layers
                .iter()
                .zip(&p.layers)
                .filter(|(cl, l)| cl.cold_bitmap.iter().any(|&b| b != 0) && !symmetric(l))
                .count();
            let mut admits = asym == 0;
            if !admits {
                eprintln!("  hetero responder: {asym} layers' cold experts failed the symmetric header check; the primary sends F16");
            }
            for r in [MIN_SUB_CHUNK_ROWS, c.max_rows].into_iter().filter(|_| admits) {
                if let Err(e) = run_cold_prefill(gpu, &p.layers[layer], &p.scratch, r, io4) {
                    eprintln!("  hetero responder: int4 rows refused at {r} rows ({e:?}); the primary sends F16");
                    admits = false;
                    break;
                }
            }
            eprintln!(
                "  hetero responder: prefill route {:?}, int4 floor {}",
                p.layers[layer].route_policy().map(|p| p.capability),
                if admits { MIN_SUB_CHUNK_ROWS } else { 0 }
            );
            let st = gpu.active_stream.as_ref().expect("responder stream is installed");
            gpu.hip.stream_synchronize(st).map_err(|e| e.to_string())?;
            if admits {
                int4_min_rows = MIN_SUB_CHUNK_ROWS as u32;
            }
        }
    }
    for s in serves {
        word(cfg, s.offsets.int4_min_rows).store(int4_min_rows, Ordering::Release);
    }
    Ok(p)
}

/// A non-owning tensor over `size` bytes at a device address.
///
/// # Safety
/// `addr` must stay valid for every launch of the graphs that capture it.
unsafe fn view(addr: u64, size: usize) -> rdna_compute::GpuTensor {
    rdna_compute::GpuTensor {
        buf: hip_bridge::DeviceBuffer::from_raw(addr as *mut std::ffi::c_void, size),
        shape: vec![size / 4],
        dtype: rdna_compute::DType::F32,
    }
}

fn build(cfg: &ResponderConfig, layers: &[ColdLayer]) -> Result<Graphs, String> {
    let g = cfg.geom;
    if g.max_rows != 1 || g.top_k != 10 || layers.len() != g.layers {
        return Err(format!("geometry {g:?} with {} layers: v1 serves 1-row decode, top-10", layers.len()));
    }
    let mut gpu = rdna_compute::Gpu::init_with_device(cfg.ordinal).map_err(|e| e.to_string())?;
    if gpu.arch != "gfx1151" {
        return Err(format!("device {} is {}, not gfx1151", cfg.ordinal, gpu.arch));
    }
    let s = |e: hip_bridge::HipError| e.to_string();
    {
        let hip = &gpu.hip;
        let lookup = |p: u64| {
            // SAFETY: one-byte non-owning view, attribute query only.
            let probe = unsafe { hip_bridge::DeviceBuffer::from_raw(p as *mut std::ffi::c_void, 1) };
            hip.pointer_get_attributes(&probe).ok().map(|a| a.device_pointer as u64)
        };
        check_device_accessible(&[cfg.mailbox_base], lookup).map_err(|e| format!("{e:?}"))?;
        for l in layers {
            check_device_accessible(&l.gate_up, lookup)
                .and_then(|()| check_device_accessible(&l.down, lookup))
                .map_err(|e| format!("{e:?}"))?;
        }
    }
    let slots = g.max_rows * g.top_k;
    let f32s = |gpu: &mut rdna_compute::Gpu, n: usize| {
        gpu.alloc_tensor(&[n], rdna_compute::DType::F32).map_err(|e| e.to_string())
    };
    let masked = f32s(&mut gpu, slots)?;
    let gate = f32s(&mut gpu, slots * cfg.mi)?;
    let up = f32s(&mut gpu, slots * cfg.mi)?;
    let rot = f32s(&mut gpu, slots * cfg.mi)?;
    let mut tables = Vec::with_capacity(layers.len());
    for l in layers {
        let as_bytes = |v: &[u64]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
        let gu = f32s(&mut gpu, 2 * cfg.n_experts)?;
        let dn = f32s(&mut gpu, 2 * cfg.n_experts)?;
        let bm = f32s(&mut gpu, cfg.n_experts.div_ceil(4))?;
        gpu.memcpy_htod_auto(&gu.buf, &as_bytes(&l.gate_up)).map_err(s)?;
        gpu.memcpy_htod_auto(&dn.buf, &as_bytes(&l.down)).map_err(s)?;
        gpu.memcpy_htod_auto(&bm.buf, &l.cold_bitmap).map_err(s)?;
        tables.push((gu, dn, bm));
    }
    gpu.active_stream = Some(gpu.hip.stream_create_non_blocking().map_err(s)?);
    fn stream(gpu: &rdna_compute::Gpu) -> &hip_bridge::Stream {
        gpu.active_stream.as_ref().expect("responder stream is installed")
    }
    let record = |gpu: &mut rdna_compute::Gpu, l: usize| -> Result<(), String> {
        let o = g.slot(l);
        let base = cfg.mailbox_base;
        let (gu, dn, bm) = &tables[l];
        // SAFETY: mailbox fields of slot `l`, live for the responder's lifetime.
        let (x, out) = unsafe {
            (view(base + o.x_rot as u64, g.dim * 4), view(base + o.cold_out as u64, slots * g.dim * 4))
        };
        gpu.hetero_mask_hot(base + o.ids as u64, bm, cfg.n_experts as i32, slots as i32, &masked).map_err(s)?;
        gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(gu, &masked, &x, &gate, &up, 2 * cfg.mi, g.dim, 1)
            .map_err(s)?;
        gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2_shared(&gate, &up, &rot, cfg.mi, slots, None).map_err(s)?;
        gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            dn, &masked, &rot, &out, g.dim, cfg.mi, 1, cfg.n_experts, g.top_k,
        )
        .map_err(s)?;
        gpu.hetero_store_seq(base + o.req_seq as u64, base + o.resp_seq as u64).map_err(s)
    };
    // Warm-up outside capture: compiles and loads every kernel. A zeroed slot
    // answers 0 ("never written"), which no primary waits for.
    record(&mut gpu, 0)?;
    gpu.hip.stream_synchronize(stream(&gpu)).map_err(s)?;
    let mut execs = Vec::with_capacity(layers.len());
    for l in 0..layers.len() {
        gpu.hip.stream_begin_capture(stream(&gpu), 0).map_err(s)?;
        let recorded = record(&mut gpu, l);
        let graph = gpu.hip.stream_end_capture(stream(&gpu)).map_err(s)?;
        recorded?;
        execs.push(gpu.hip.graph_instantiate(&graph).map_err(s)?);
        let _ = gpu.hip.graph_destroy(graph);
    }
    let mut keep = vec![masked, gate, up, rot];
    for (gu, dn, bm) in tables {
        keep.extend([gu, dn, bm]);
    }
    let prefill = if cfg.prefill.is_empty() {
        None
    } else {
        Some(build_prefill(&mut gpu, cfg, &cfg.prefill, layers)?)
    };
    Ok(Graphs { gpu, execs, prefill, _keep: keep })
}

impl HeteroResponder {
    /// Record the per-layer graphs on gfx1151 device `cfg.ordinal` and start
    /// answering. Returns once the graphs exist, or the setup error.
    pub fn start(cfg: ResponderConfig, layers: Vec<ColdLayer>) -> Result<Self, DispatchError> {
        let stop = Arc::new(AtomicBool::new(false));
        let failure = Arc::new(Mutex::new(None::<String>));
        let answered = Arc::new(AtomicU64::new(0));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let (stop2, failure2, answered2) = (Arc::clone(&stop), Arc::clone(&failure), Arc::clone(&answered));
        let worker = std::thread::Builder::new()
            .name("hetero-responder".into())
            .spawn(move || {
                let mut graphs = match build(&cfg, &layers) {
                    Ok(gr) => {
                        let _ = ready_tx.send(Ok(()));
                        gr
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let n = cfg.geom.layers;
                let mut last: Vec<u32> = (0..n).map(|l| req_word(&cfg, l).load(Ordering::Acquire)).collect();
                let prefill_words: Vec<_> = cfg
                    .prefill
                    .iter()
                    .enumerate()
                    .flat_map(|(slot, p)| {
                        p.offsets.sub[..p.sub_chunks].iter().enumerate().map(move |(sub, w)| (slot, sub, *w))
                    })
                    .map(|(slot, sub, w)| {
                        (slot, sub, word(&cfg, w.req_seq), word(&cfg, w.layer), word(&cfg, w.row_off), word(&cfg, w.rows), word(&cfg, w.format))
                    })
                    .collect();
                let mut last_prefill: Vec<u32> =
                    prefill_words.iter().map(|(_, _, req, ..)| req.load(Ordering::Acquire)).collect();
                let mut idle = 0u32;
                while !stop2.load(Ordering::Acquire) {
                    let mut busy = false;
                    // Sub-chunks in order: the primary releases them in order.
                    for (i, (slot, sub, req, layer, row_off, rows, format)) in prefill_words.iter().enumerate() {
                        let r = req.load(Ordering::Acquire);
                        if r != last_prefill[i] {
                            last_prefill[i] = r;
                            busy = true;
                            // The primary stores the meta before its request
                            // word (release); this Acquire load of the
                            // request orders the meta reads after it.
                            let (l, off, rw, f) = (
                                layer.load(Ordering::Acquire),
                                row_off.load(Ordering::Acquire),
                                rows.load(Ordering::Acquire),
                                format.load(Ordering::Acquire),
                            );
                            if let Err(e) = serve_prefill(&mut graphs, &cfg, *slot, *sub, l, off, rw, f) {
                                if let Ok(mut f) = failure2.lock() {
                                    *f = Some(format!("prefill layer {l} slot {slot} sub {sub}: {e}"));
                                }
                                return;
                            }
                            answered2.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    for l in 0..n {
                        let r = req_word(&cfg, l).load(Ordering::Acquire);
                        if r != last[l] {
                            last[l] = r;
                            busy = true;
                            let st = graphs.gpu.active_stream.as_ref().expect("responder stream");
                            if let Err(e) = graphs.gpu.hip.graph_launch(&graphs.execs[l], st) {
                                if let Ok(mut f) = failure2.lock() {
                                    *f = Some(format!("layer {l}: graph launch: {e}"));
                                }
                                return;
                            }
                            answered2.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    if busy {
                        idle = 0;
                    } else {
                        idle = idle.saturating_add(1);
                        if idle < 20_000 {
                            std::hint::spin_loop();
                        } else {
                            let us = 50u64 << ((idle - 20_000) / 1_000).min(4);
                            std::thread::sleep(Duration::from_micros(us.min(1_000)));
                        }
                    }
                }
                if let Some(st) = graphs.gpu.active_stream.as_ref() {
                    let _ = graphs.gpu.hip.stream_synchronize(st);
                }
                for e in graphs.execs {
                    let _ = graphs.gpu.hip.graph_exec_destroy(e);
                }
            })
            .map_err(|e| err(format!("spawn: {e}")))?;
        ready_rx
            .recv()
            .map_err(|_| err("worker exited before it was ready"))?
            .map_err(err)?;
        Ok(Self { stop, failure, answered, worker: Some(worker) })
    }

    /// The HIP failure that stopped the responder, if any.
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|f| f.clone())
    }

    /// Requests answered so far.
    pub fn answered(&self) -> u64 {
        self.answered.load(Ordering::Relaxed)
    }
}

impl Drop for HeteroResponder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}
