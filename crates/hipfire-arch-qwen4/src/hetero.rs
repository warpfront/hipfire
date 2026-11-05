//! `topology.hetero_experts`: the Qwen4 heterogeneous expert split.
//!
//! Experimental and opt-in, and not yet admitted by default (prefill still
//! moves the cold expert weights across the link; see the design in
//! `docs/superpowers/specs/2026-11-04-qwen4-hetero-expert-server-design.md`).
//!
//! With `topology.hetero_experts = "gfx1151"` and `hardware.devices =
//! "gfx1201,gfx1151"`, the gfx1201 primary runs the trunk and the hot routed
//! experts. Every D2 split executor then computes its cold slots on one shared
//! gfx1151 iGPU arm ([`hipfire_dispatch::igpu_split::IgpuColdArm`], HIP logical
//! device 1) instead of host CPU cores. The arm is process-wide: a daemon holds
//! one model.

use std::sync::{Arc, Mutex};

use hipfire_dispatch::igpu_split::{ColdArmDims, IgpuColdArm};

/// HIP logical device of the expert iGPU (`hardware.devices = "gfx1201,gfx1151"`).
const EXPERT_DEVICE: i32 = 1;

/// The primary (HIP logical device 0) must be the gfx1201.
pub(crate) fn check_primary(primary_arch: &str) -> Result<(), String> {
    if primary_arch == "gfx1201" {
        Ok(())
    } else {
        Err(format!(
            "topology.hetero_experts: the primary device is {primary_arch}; set hardware.devices = \"gfx1201,gfx1151\" (the gfx1201 first)"
        ))
    }
}

/// `topology.hetero_experts` for this process: `Some(arch)` when the split is
/// requested and its combination rules hold.
pub(crate) fn requested() -> Result<Option<String>, String> {
    use hipfire_config::topology;
    let value = hipfire_config::process_value(topology::HETERO_EXPERTS_ENV);
    topology::resolve_hetero_experts(
        value.as_deref(),
        hipfire_runtime::config::get().allow_mixed_arch,
        topology::qwen4_ep()?,
        topology::pp()?,
        1,
    )
}

static ARM: Mutex<Option<Arc<IgpuColdArm>>> = Mutex::new(None);

/// The shared iGPU arm when the split is requested, opened on first use; `None`
/// when it is not. `dims.max_rows` is raised to the D2 storage cap so every
/// executor of the load (wider verify reservations included) fits one arm.
pub(crate) fn arm_for(primary_arch: &str, dims: ColdArmDims) -> Result<Option<Arc<IgpuColdArm>>, String> {
    if requested()?.is_none() {
        return Ok(None);
    }
    check_primary(primary_arch)?;
    let mut slot = ARM.lock().map_err(|_| "topology.hetero_experts: arm lock poisoned".to_owned())?;
    if let Some(arm) = slot.as_ref() {
        return Ok(Some(Arc::clone(arm)));
    }
    let dims = ColdArmDims {
        max_rows: hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS,
        ..dims
    };
    let arm = Arc::new(
        IgpuColdArm::new(EXPERT_DEVICE, dims)
            .map_err(|e| format!("topology.hetero_experts: {e:?}"))?,
    );
    eprintln!(
        "  qwen4 hetero experts: igpu arm on gfx1151 (HIP device {EXPERT_DEVICE}), cold D2 slots run there"
    );
    *slot = Some(Arc::clone(&arm));
    Ok(Some(arm))
}

// ---- Device-side split (design v3) ---------------------------------------

use hipfire_dispatch::hetero_device::{HeteroDeviceSplit, HeteroPrefillShared, HeteroPrefillSplit};
use hipfire_dispatch::hetero_mailbox::{MailboxGeometry, PrefillGeometry, MAX_SUB_CHUNKS};
use hipfire_dispatch::hetero_prefill::ColdPrefillConfig;
use hipfire_dispatch::hetero_responder::{ColdLayer, HeteroResponder, PrefillServe, ResponderConfig};

/// The prefill chunk under the split: the mailbox's prefill slot and the expert
/// device's scratch are sized for it, and the load's prefill chunk is clamped
/// to it (`gpu_forward::qwen4_prefill_chunk_requested`).
pub const HETERO_PREFILL_MAX_ROWS: usize = 2048;

/// Prefill slots of the mailbox: the tiles the layer-major hetero schedule
/// keeps in flight (`slowlink_prefill::try_run_hetero_prompt`); the
/// chunk-major path uses slot 0.
pub const HETERO_PREFILL_SLOTS: usize = 8;

/// Layer-major tile rows for a `rows`-row prompt (`HIPFIRE_HETERO_TILE_ROWS`,
/// whole 256-row steps, at least 512 so every tile keeps the expert device's
/// grouped arm), at most the forward's chunk.
pub(crate) fn layer_major_tile_rows(max_chunk: usize, rows: usize) -> usize {
    std::env::var("HIPFIRE_HETERO_TILE_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|r| r / 256 * 256)
        .filter(|&r| r >= 512)
        // Measured (R9700 + gfx1151, 4 tiles in flight): 2048-row tiles run
        // 4K/8K at 1,902/1,868 tok/s vs 1,654/1,611 at 1024 rows (the
        // primary's per-row cost falls with the batch); a prompt below two
        // full tiles splits into 1024-row tiles (2K: 1,741 vs 1,660 one chunk).
        .unwrap_or(if rows >= 2 * max_chunk { max_chunk } else { 1024 })
        .min(max_chunk)
}

/// Tiles the layer-major schedule keeps in flight (`HIPFIRE_HETERO_TILES`,
/// default 4), at most the mailbox's slots.
pub(crate) fn layer_major_tiles(slots: usize) -> usize {
    std::env::var("HIPFIRE_HETERO_TILES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&t| t >= 2)
        .unwrap_or(4)
        .min(slots)
}

/// `hetero_await_cold` poll bound: each poll is a PCIe read (~1-2 us) plus a
/// short sleep, so this bounds a dead or stalled expert device at ~2 s.
const AWAIT_MAX_POLLS: u32 = 1_000_000;

/// Load-wide state of the device-side split: the pinned mailbox, the device
/// token counter, each layer's cold tables until the last layer attaches, and
/// then the responder.
struct DeviceContext {
    mailbox: usize,
    geom: MailboxGeometry,
    counter: rdna_compute::GpuTensor,
    n_experts: usize,
    mi: usize,
    pending: Vec<Option<ColdLayer>>,
    responder: Option<HeteroResponder>,
    prefill: Arc<HeteroPrefillShared>,
    prefill_cold: ColdPrefillConfig,
    prefill_offsets: Vec<hipfire_dispatch::hetero_mailbox::PrefillOffsets>,
}

// SAFETY: the raw mailbox address and the counter's device pointer are plain
// addresses; every access goes through HIP or atomics on pinned memory.
unsafe impl Send for DeviceContext {}

static DEVICE: Mutex<Option<DeviceContext>> = Mutex::new(None);

fn lock_device() -> Result<std::sync::MutexGuard<'static, Option<DeviceContext>>, String> {
    DEVICE.lock().map_err(|_| "topology.hetero_experts: device context lock poisoned".to_owned())
}

/// Whether every layer has attached (the responder can start).
pub(crate) fn all_attached<T>(pending: &[Option<T>]) -> bool {
    !pending.is_empty() && pending.iter().all(Option::is_some)
}

/// Attach trunk layer `layer` of `layers` to the device-side split: allocate
/// the load's mailbox and counter on first use, return the layer's
/// [`HeteroDeviceSplit`], and start the responder once every layer attached.
/// `placement[e] == 1` marks a cold (host-tier) expert; `tables` are the cold
/// experts' (gate/up, down) byte addresses (0 for hot ones).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attach_device_layer(
    gpu: &mut rdna_compute::Gpu,
    layer: usize,
    layers: usize,
    dim: usize,
    mi: usize,
    top_k: usize,
    placement: &[u8],
    tables: (Vec<u64>, Vec<u64>),
    prefill_cold: ColdPrefillConfig,
) -> Result<(HeteroDeviceSplit, HeteroPrefillSplit), String> {
    check_primary(&gpu.arch)?;
    let hip = |e: hip_bridge::HipError| format!("topology.hetero_experts: {e}");
    let mut guard = lock_device()?;
    if guard.is_none() {
        let geom = MailboxGeometry { layers, max_rows: 1, top_k, dim };
        let pgeom = PrefillGeometry { max_rows: prefill_cold.max_rows, top_k, dim, sub_chunks: MAX_SUB_CHUNKS };
        let mut poffs = Vec::with_capacity(HETERO_PREFILL_SLOTS);
        let mut end = geom.total_bytes();
        for _ in 0..HETERO_PREFILL_SLOTS {
            let o = pgeom.offsets(end);
            end = o.error + pgeom.bytes();
            poffs.push(o);
        }
        let total = end;
        let flags = hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT;
        let mailbox = gpu.hip.host_malloc(total, flags).map_err(hip)?.cast::<u8>();
        // SAFETY: fresh pinned allocation of `total` bytes: zero = never written.
        unsafe { std::ptr::write_bytes(mailbox, 0, total) };
        let counter = gpu.alloc_tensor(&[1], rdna_compute::DType::F32).map_err(hip)?;
        gpu.memcpy_htod_auto(&counter.buf, &0u32.to_ne_bytes()).map_err(hip)?;
        // SAFETY: a non-owning view of the load's counter, which outlives the load.
        let counter_view = unsafe {
            rdna_compute::GpuTensor {
                buf: hip_bridge::DeviceBuffer::from_raw(counter.buf.as_ptr(), 4),
                shape: vec![1],
                dtype: rdna_compute::DType::F32,
            }
        };
        let prefill = HeteroPrefillShared::new(gpu, mailbox as u64, &poffs, pgeom, placement.len(), counter_view, AWAIT_MAX_POLLS)
            .map_err(|e| format!("topology.hetero_experts: {e:?}"))?;
        *guard = Some(DeviceContext {
            mailbox: mailbox as usize,
            geom,
            counter,
            n_experts: placement.len(),
            mi,
            pending: (0..layers).map(|_| None).collect(),
            responder: None,
            prefill: Arc::new(prefill),
            prefill_cold,
            prefill_offsets: poffs,
        });
        eprintln!(
            "  qwen4 hetero experts: device split, {layers} layers, mailbox {} MiB pinned ({HETERO_PREFILL_SLOTS} prefill slots)",
            total >> 20
        );
    }
    let ctx = guard.as_mut().expect("initialized above");
    if ctx.geom.layers != layers || layer >= layers || ctx.n_experts != placement.len() {
        return Err(format!("topology.hetero_experts: layer {layer} of {layers} does not match the load's geometry"));
    }
    ctx.pending[layer] = Some(ColdLayer { gate_up: tables.0, down: tables.1, cold_bitmap: placement.to_vec() });
    let bitmap = gpu.alloc_tensor(&[placement.len().div_ceil(4)], rdna_compute::DType::F32).map_err(hip)?;
    gpu.memcpy_htod_auto(&bitmap.buf, placement).map_err(hip)?;
    let masked_ids = gpu.alloc_tensor(&[top_k], rdna_compute::DType::F32).map_err(hip)?;
    let o = ctx.geom.slot(layer);
    let b = ctx.mailbox as u64;
    let slot = rdna_compute::HeteroSlotPtrs {
        req_seq: b + o.req_seq as u64,
        resp_seq: b + o.resp_seq as u64,
        error: b + o.error as u64,
        x_rot: b + o.x_rot as u64,
        ids: b + o.ids as u64,
        weights: b + o.weights as u64,
        cold_out: b + o.cold_out as u64,
    };
    // SAFETY: a non-owning view of the load's counter, which outlives the load.
    let counter = unsafe {
        rdna_compute::GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(ctx.counter.buf.as_ptr(), 4),
            shape: vec![1],
            dtype: rdna_compute::DType::F32,
        }
    };
    if all_attached(&ctx.pending) && ctx.responder.is_none() {
        let layers_v: Vec<ColdLayer> = ctx.pending.iter().map(|l| l.clone().expect("all attached")).collect();
        let cfg = ResponderConfig {
            ordinal: EXPERT_DEVICE,
            geom: ctx.geom,
            mailbox_base: b,
            n_experts: ctx.n_experts,
            mi: ctx.mi,
            prefill: ctx
                .prefill_offsets
                .iter()
                .map(|&offsets| PrefillServe { offsets, sub_chunks: MAX_SUB_CHUNKS, cold: ctx.prefill_cold })
                .collect(),
        };
        let responder = HeteroResponder::start(cfg, layers_v).map_err(|e| format!("topology.hetero_experts: {e:?}"))?;
        eprintln!("  qwen4 hetero experts: responder on gfx1151 (HIP device {EXPERT_DEVICE}) recorded {layers} layer graphs");
        ctx.responder = Some(responder);
    }
    // SAFETY: a non-owning view of the layer's bitmap, owned by its decode split.
    let prefill_bitmap = unsafe {
        rdna_compute::GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(bitmap.buf.as_ptr(), bitmap.buf.size()),
            shape: bitmap.shape.clone(),
            dtype: bitmap.dtype,
        }
    };
    let prefill_masked = gpu
        .alloc_tensor(&[ctx.prefill.geom.max_rows * top_k], rdna_compute::DType::F32)
        .map_err(hip)?;
    let prefill = HeteroPrefillSplit {
        layer: layer as u32,
        layers: layers as u32,
        shared: Arc::clone(&ctx.prefill),
        cold_bitmap: prefill_bitmap,
        masked_ids: prefill_masked,
    };
    Ok((
        HeteroDeviceSplit {
            layer: layer as u32,
            layers: layers as u32,
            slot,
            counter,
            cold_bitmap: bitmap,
            masked_ids,
            n_experts: placement.len() as i32,
            max_polls: AWAIT_MAX_POLLS,
        },
        prefill,
    ))
}

/// Start of every forward (decode token or prefill chunk): fail closed on an
/// earlier timeout or a stopped responder, then advance the device token
/// counter (an eager launch on the primary's stream, ahead of the forward's
/// split stages), so every request of the forward carries a fresh sequence.
/// No-op without a device split.
pub(crate) fn begin_token(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let guard = lock_device()?;
    let Some(ctx) = guard.as_ref() else {
        return Ok(());
    };
    let Some(responder) = ctx.responder.as_ref() else {
        return Err("topology.hetero_experts: not every layer attached the device split; no responder".into());
    };
    if let Some(f) = responder.failure() {
        return Err(format!("topology.hetero_experts: responder stopped: {f}"));
    }
    for l in 0..ctx.geom.layers {
        let addr = ctx.mailbox + ctx.geom.slot(l).error;
        // SAFETY: an aligned word in the live pinned mailbox.
        let e = unsafe { &*(addr as *const std::sync::atomic::AtomicU32) }.load(std::sync::atomic::Ordering::Acquire);
        if e != 0 {
            return Err(format!(
                "topology.hetero_experts: layer {l} timed out waiting for the expert device (seq {e}); the split is poisoned"
            ));
        }
    }
    let e = ctx.prefill.error_word();
    if e != 0 {
        return Err(format!(
            "topology.hetero_experts: a prefill chunk timed out waiting for the expert device (seq {e}); the split is poisoned"
        ));
    }
    gpu.hetero_token_advance(&ctx.counter).map_err(|e| format!("topology.hetero_experts: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responder_starts_only_when_every_layer_attached() {
        assert!(!all_attached::<u8>(&[]));
        assert!(!all_attached(&[Some(1), None]));
        assert!(all_attached(&[Some(1), Some(2)]));
    }

    #[test]
    fn hetero_requires_primary_gfx1201_then_gfx1151() {
        assert!(check_primary("gfx1201").is_ok());
        let e = check_primary("gfx1151").unwrap_err();
        assert!(e.contains("topology.hetero_experts") && e.contains("gfx1201,gfx1151"), "{e}");
    }
}

/// The layer-major prompt schedule's resources (see
/// `slowlink_prefill::try_run_hetero_prompt`): the load's prefill exchange,
/// the HC-stream arena (one `tile_rows x wide` F32 region per slot) and one
/// FFN-output buffer per slot (a tile's MoE result held until its join).
pub(crate) struct LayerMajor {
    pub shared: Arc<HeteroPrefillShared>,
    pub streams: rdna_compute::GpuTensor,
    pub ffn: Vec<rdna_compute::GpuTensor>,
}

/// (tile rows, wide, hidden) the arena was sized for, and its owners.
static ARENA: Mutex<Option<((usize, usize, usize), rdna_compute::GpuTensor, Vec<rdna_compute::GpuTensor>)>> =
    Mutex::new(None);

fn view_of(t: &rdna_compute::GpuTensor) -> rdna_compute::GpuTensor {
    // SAFETY: a non-owning view of an arena tensor that lives for the process.
    rdna_compute::GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(t.buf.as_ptr(), t.buf.size()) },
        shape: t.shape.clone(),
        dtype: t.dtype,
    }
}

/// `Some` when this load runs the hetero split with its prefill exchange and
/// `HIPFIRE_HETERO_LAYER_MAJOR` is not `0`. The arena is allocated on first use
/// (once per process: a daemon holds one model).
pub(crate) fn layer_major(
    gpu: &mut rdna_compute::Gpu,
    tile_rows: usize,
    wide: usize,
    hidden: usize,
) -> Result<Option<LayerMajor>, String> {
    if std::env::var("HIPFIRE_HETERO_LAYER_MAJOR").is_ok_and(|v| v == "0") {
        return Ok(None);
    }
    let shared = {
        let guard = lock_device()?;
        match guard.as_ref() {
            Some(ctx) if ctx.responder.is_some() => Arc::clone(&ctx.prefill),
            _ => return Ok(None),
        }
    };
    if tile_rows > shared.geom.max_rows {
        return Ok(None);
    }
    let slots = shared.slots();
    let hip = |e: hip_bridge::HipError| format!("topology.hetero_experts: layer-major arena: {e}");
    let mut arena = ARENA.lock().map_err(|_| "topology.hetero_experts: arena lock poisoned".to_owned())?;
    let shape = (tile_rows, wide, hidden);
    if arena.as_ref().is_none_or(|(s, _, _)| *s != shape) {
        if let Some((_, streams, ffn)) = arena.take() {
            let _ = gpu.free_tensor(streams);
            for t in ffn {
                let _ = gpu.free_tensor(t);
            }
        }
        let streams = gpu.alloc_tensor(&[slots * tile_rows * wide], rdna_compute::DType::F32).map_err(hip)?;
        let ffn = (0..slots)
            .map(|_| gpu.alloc_tensor(&[tile_rows * hidden], rdna_compute::DType::F32))
            .collect::<Result<Vec<_>, _>>()
            .map_err(hip)?;
        eprintln!(
            "  qwen4 hetero experts: layer-major prompt arena {} MiB ({slots} tiles of {tile_rows} rows)",
            (slots * tile_rows * (wide + hidden) * 4) >> 20
        );
        *arena = Some((shape, streams, ffn));
    }
    let (_, streams, ffn) = arena.as_ref().expect("allocated above");
    Ok(Some(LayerMajor { shared, streams: view_of(streams), ffn: ffn.iter().map(view_of).collect() }))
}
