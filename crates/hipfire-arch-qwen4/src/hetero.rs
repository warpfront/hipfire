//! `topology.hetero_experts`: the Qwen4 heterogeneous expert split.
//!
//! Experimental and opt-in, and not yet admitted by default (prefill still
//! moves the cold expert weights across the link; see the design in
//! `docs/superpowers/specs/2026-11-04-qwen4-hetero-expert-server-design.md`).
//!
//! With `topology.hetero_experts = "gfx1151"` and `hardware.devices =
//! "gfx1201,gfx1151"`, the gfx1201 primary runs the trunk and the hot routed
//! experts; the cold experts are computed on the gfx1151 (HIP logical device
//! 1) by a responder that answers each layer through a pinned mailbox (the
//! device-side split below). Its state is load-owned: [`release`] tears it
//! down at unload, and a reload starts a fresh one.

use std::sync::{Arc, Mutex};

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
/// requested, its combination rules hold and the process's cards are the
/// pair it runs on (a gfx1201 primary, then the expert arch). A key set on
/// any other device set is an error, not a load that silently changes its
/// dispatch and prefill settings and never engages the split.
pub(crate) fn requested() -> Result<Option<String>, String> {
    use hipfire_config::topology;
    let value = hipfire_config::process_value(topology::HETERO_EXPERTS_ENV);
    let Some(arch) = topology::resolve_hetero_experts(
        value.as_deref(),
        hipfire_runtime::config::get().allow_mixed_arch,
        topology::qwen4_ep()?,
        topology::pp()?,
        // tp is a load parameter, not process configuration; Qwen4's mesh
        // admission refuses any tp > 1 for every load
        // (`EffectiveMesh::is_qwen4_admitted`), the split's included.
        1,
    )?
    else {
        return Ok(None);
    };
    // `None`: the cards cannot be told before the runtime loads (no KFD
    // topology); the attach's `check_primary` still holds the primary.
    if let Some(devices) = hipfire_config::devices::startup_devices() {
        check_devices(&devices.iter().map(|d| d.arch.as_str()).collect::<Vec<_>>(), &arch)?;
    }
    Ok(Some(arch))
}

/// The split's card pair: HIP logical device 0 the gfx1201 primary, device
/// [`EXPERT_DEVICE`] the expert arch.
fn check_devices(archs: &[&str], expert: &str) -> Result<(), String> {
    match archs {
        ["gfx1201", second, ..] if *second == expert => Ok(()),
        _ => Err(format!(
            "topology.hetero_experts = {expert:?} needs hardware.devices = \"gfx1201,{expert}\" (the gfx1201 first); this process has {archs:?}"
        )),
    }
}

/// Whether this process requests the split (a valid `topology.hetero_experts`).
/// Needs the process and runtime configuration installed. An invalid key is
/// reported here once and reads as not requested; the load itself refuses it
/// (`Qwen4GpuForward::new_inner_with_reserve`).
pub fn split_requested() -> bool {
    match requested() {
        Ok(found) => found.is_some(),
        Err(error) => {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| eprintln!("  qwen4 hetero experts: {error}"));
            false
        }
    }
}

/// Whether a layer attached this load's device split.
pub(crate) fn attached() -> bool {
    lock_device().is_ok_and(|guard| guard.is_some())
}

// ---- Device-side split (design v3) ---------------------------------------

use hipfire_dispatch::hetero_device::{HeteroDeviceSplit, HeteroPrefillShared, HeteroPrefillSplit};
use hipfire_dispatch::hetero_mailbox::{MailboxGeometry, PrefillGeometry, MAX_SUB_CHUNKS};
use hipfire_dispatch::hetero_prefill::ColdPrefillConfig;
use hipfire_dispatch::hetero_responder::{
    ColdLayer, HeteroResponder, PrefillServe, ResponderConfig,
};

/// Rows of the decode exchange: single-token decode and MTP verify windows up
/// to this width take the decode slot (one round trip per layer, no staging);
/// wider calls take the prefill exchange. The default of [`hetero_decode_rows`].
pub const HETERO_DECODE_ROWS: usize = 8;

/// Widest decode slot [`hetero_decode_rows`] admits: the wide verify cap.
pub const HETERO_DECODE_ROWS_MAX: usize = rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS;

/// The load's decode slot width: `HIPFIRE_HETERO_DECODE_ROWS`
/// (`HETERO_DECODE_ROWS..=HETERO_DECODE_ROWS_MAX`) or [`HETERO_DECODE_ROWS`].
/// A wider slot lets one lane step's routed MoE over every row of several
/// lanes' verify windows take a single exchange per layer; the mailbox and the
/// responder's per-row-count graphs grow with it. Read once.
pub fn hetero_decode_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_HETERO_DECODE_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|r| (HETERO_DECODE_ROWS..=HETERO_DECODE_ROWS_MAX).contains(r))
            .unwrap_or(HETERO_DECODE_ROWS)
    })
}

/// The prefill chunk under the split: the mailbox's prefill slot and the expert
/// device's scratch are sized for it, and the load's prefill chunk is clamped
/// to it (`gpu_forward::qwen4_prefill_chunk_requested`).
pub const HETERO_PREFILL_MAX_ROWS: usize = 4096;

/// The load's hetero prefill chunk: `HIPFIRE_HETERO_PREFILL_ROWS` (whole
/// 256-row steps, 512..=8192) or [`HETERO_PREFILL_MAX_ROWS`]. Read once.
pub fn hetero_prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        hipfire_config::developer_var("HIPFIRE_HETERO_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|r| r / 256 * 256)
            .filter(|r| (512..=8192).contains(r))
            .unwrap_or(HETERO_PREFILL_MAX_ROWS)
    })
}

/// Prefill slots of the mailbox: the tiles the layer-major hetero schedule
/// keeps in flight (`slowlink_prefill::try_run_hetero_prompt`); the
/// chunk-major path uses slot 0.
pub const HETERO_PREFILL_SLOTS: usize = 8;

/// Layer-major tile rows for a `rows`-row prompt (`HIPFIRE_HETERO_TILE_ROWS`,
/// whole 256-row steps, at least 512 so every tile keeps the expert device's
/// grouped arm), at most the forward's chunk.
pub(crate) fn layer_major_tile_rows(max_chunk: usize, rows: usize) -> usize {
    hipfire_config::developer_var("HIPFIRE_HETERO_TILE_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|r| r / 256 * 256)
        .filter(|&r| r >= 512)
        // The fewest tiles that fit the chunk, never one (the exchange must
        // overlap another tile's work), each a whole number of 256-row steps.
        // Measured (R9700 + gfx1151): bigger tiles cut the primary's per-row
        // cost (8K: 2 x 4096 2,729 tok/s vs 4 x 2048 2,616; 4K: 2 x 2048
        // 2,417 vs 4 x 1024 1,654).
        .unwrap_or_else(|| {
            let tiles = rows.div_ceil(max_chunk).max(2);
            rows.div_ceil(tiles).div_ceil(256) * 256
        })
        .min(max_chunk)
}

/// Tiles the layer-major schedule keeps in flight (`HIPFIRE_HETERO_TILES`,
/// default 4), at most the mailbox's slots.
pub(crate) fn layer_major_tiles(slots: usize) -> usize {
    hipfire_config::developer_var("HIPFIRE_HETERO_TILES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&t| t >= 2)
        .unwrap_or(4)
        .min(slots)
}

/// `hetero_await_cold` poll bound: each poll is a PCIe read (~1-2 us) plus a
/// short sleep (`s_sleep(8)`, ~0.2 us), so this bounds a dead or stalled
/// expert device at ~2 s.
const AWAIT_MAX_POLLS: u32 = 1_000_000;

/// Load-wide state of the device-side split: the pinned mailbox, the device
/// token counter, each layer's cold tables until the last layer attaches, and
/// then the responder.
struct DeviceContext {
    mailbox: usize,
    /// Address of the load-wide poison word (in the mailbox allocation):
    /// nonzero once a wait timed out or the responder stopped.
    poison: usize,
    geom: MailboxGeometry,
    counter: rdna_compute::GpuTensor,
    n_experts: usize,
    mi: usize,
    pending: Vec<Option<ColdLayer>>,
    responder: Option<HeteroResponder>,
    /// The MTP head's own request counter, once its slot is served.
    head_counter: Option<rdna_compute::GpuTensor>,
    prefill: Arc<HeteroPrefillShared>,
    prefill_cold: ColdPrefillConfig,
    prefill_offsets: Vec<hipfire_dispatch::hetero_mailbox::PrefillOffsets>,
}

// SAFETY: the raw mailbox address and the counter's device pointer are plain
// addresses; every access goes through HIP or atomics on pinned memory.
unsafe impl Send for DeviceContext {}

static DEVICE: Mutex<Option<DeviceContext>> = Mutex::new(None);

fn lock_device() -> Result<std::sync::MutexGuard<'static, Option<DeviceContext>>, String> {
    DEVICE
        .lock()
        .map_err(|_| "topology.hetero_experts: device context lock poisoned".to_owned())
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
        // One slot per trunk layer, then the MTP head's (served once a head
        // attaches: `attach_head_layer`).
        let geom = MailboxGeometry {
            layers: layers + HEAD_SLOTS,
            max_rows: hetero_decode_rows(),
            top_k,
            dim,
        };
        let pgeom = PrefillGeometry {
            max_rows: prefill_cold.max_rows,
            top_k,
            dim,
            sub_chunks: MAX_SUB_CHUNKS,
        };
        let mut poffs = Vec::with_capacity(HETERO_PREFILL_SLOTS);
        let mut end = geom.total_bytes();
        for _ in 0..HETERO_PREFILL_SLOTS {
            let o = pgeom.offsets(end);
            end = o.error + pgeom.bytes();
            poffs.push(o);
        }
        // The load-wide poison word, on its own cache line after the slots.
        let poison_off = end.next_multiple_of(64);
        let total = poison_off + 64;
        let flags = hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT;
        let mailbox = gpu.hip.host_malloc(total, flags).map_err(hip)?.cast::<u8>();
        // SAFETY: fresh pinned allocation of `total` bytes: zero = never written.
        unsafe { std::ptr::write_bytes(mailbox, 0, total) };
        let counter = gpu
            .alloc_tensor(&[1], rdna_compute::DType::F32)
            .map_err(hip)?;
        gpu.memcpy_htod_auto(&counter.buf, &0u32.to_ne_bytes())
            .map_err(hip)?;
        // SAFETY: a non-owning view of the load's counter, which outlives the load.
        let counter_view = unsafe {
            rdna_compute::GpuTensor {
                buf: hip_bridge::DeviceBuffer::from_raw(counter.buf.as_ptr(), 4),
                shape: vec![1],
                dtype: rdna_compute::DType::F32,
            }
        };
        let prefill = HeteroPrefillShared::new(
            gpu,
            mailbox as u64,
            mailbox as u64 + poison_off as u64,
            &poffs,
            pgeom,
            placement.len(),
            counter_view,
            AWAIT_MAX_POLLS,
        )
        .map_err(|e| format!("topology.hetero_experts: {e:?}"))?;
        *guard = Some(DeviceContext {
            mailbox: mailbox as usize,
            poison: mailbox as usize + poison_off,
            geom,
            counter,
            n_experts: placement.len(),
            mi,
            pending: (0..layers).map(|_| None).collect(),
            responder: None,
            head_counter: None,
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
    // A context whose responder runs, or that already holds this layer,
    // belongs to an earlier load that was never released ([`release`]): its
    // responder's graphs bake that load's cold expert addresses.
    if ctx.responder.is_some() || ctx.pending.get(layer).is_some_and(Option::is_some) {
        return Err(format!(
            "topology.hetero_experts: layer {layer} attached while an earlier load's device split is still live (not released)"
        ));
    }
    if ctx.pending.len() != layers || layer >= layers || ctx.n_experts != placement.len() {
        return Err(format!(
            "topology.hetero_experts: layer {layer} of {layers} does not match the load's geometry"
        ));
    }
    ctx.pending[layer] = Some(ColdLayer {
        gate_up: tables.0,
        down: tables.1,
        cold_bitmap: placement.to_vec(),
    });
    let bitmap = gpu
        .alloc_tensor(&[placement.len().div_ceil(4)], rdna_compute::DType::F32)
        .map_err(hip)?;
    gpu.memcpy_htod_auto(&bitmap.buf, placement).map_err(hip)?;
    let masked_ids = gpu
        .alloc_tensor(&[hetero_decode_rows() * top_k], rdna_compute::DType::F32)
        .map_err(hip)?;
    let slot = slot_ptrs(ctx, layer);
    let b = ctx.mailbox as u64;
    // SAFETY: a non-owning view of the load's counter, which outlives the load.
    let counter = unsafe {
        rdna_compute::GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(ctx.counter.buf.as_ptr(), 4),
            shape: vec![1],
            dtype: rdna_compute::DType::F32,
        }
    };
    if all_attached(&ctx.pending) && ctx.responder.is_none() {
        let layers_v: Vec<ColdLayer> = ctx
            .pending
            .iter()
            .map(|l| l.clone().expect("all attached"))
            .collect();
        let cfg = ResponderConfig {
            ordinal: EXPERT_DEVICE,
            geom: ctx.geom,
            mailbox_base: b,
            poison: ctx.poison as u64,
            n_experts: ctx.n_experts,
            mi: ctx.mi,
            prefill: ctx
                .prefill_offsets
                .iter()
                .map(|&offsets| PrefillServe {
                    offsets,
                    sub_chunks: MAX_SUB_CHUNKS,
                    cold: ctx.prefill_cold,
                })
                .collect(),
        };
        let responder = HeteroResponder::start(cfg, layers_v)
            .map_err(|e| format!("topology.hetero_experts: {e:?}"))?;
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
        .alloc_tensor(
            &[ctx.prefill.geom.max_rows * top_k],
            rdna_compute::DType::F32,
        )
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
            max_rows: hetero_decode_rows(),
            advance: false,
        },
        prefill,
    ))
}

/// Mailbox slots after the trunk layers': the MTP head's.
const HEAD_SLOTS: usize = 1;

fn slot_ptrs(ctx: &DeviceContext, index: usize) -> rdna_compute::HeteroSlotPtrs {
    let o = ctx.geom.slot(index);
    let b = ctx.mailbox as u64;
    rdna_compute::HeteroSlotPtrs {
        req_seq: b + o.req_seq as u64,
        resp_seq: b + o.resp_seq as u64,
        error: b + o.error as u64,
        rows: b + o.rows as u64,
        x_rot: b + o.x_rot as u64,
        ids: b + o.ids as u64,
        weights: b + o.weights as u64,
        cold_out: b + o.cold_out as u64,
        poison: ctx.poison as u64,
    }
}

/// Attach the native MTP head's MoE to the device split, after every trunk
/// layer attached (the responder runs): the responder records graphs for the
/// head's slot, and the returned split advances its own counter on every
/// publish (the head drafts several times between trunk forwards).
/// `placement`/`tables` as for [`attach_device_layer`].
pub(crate) fn attach_head_layer(
    gpu: &mut rdna_compute::Gpu,
    dim: usize,
    top_k: usize,
    placement: &[u8],
    tables: (Vec<u64>, Vec<u64>),
) -> Result<HeteroDeviceSplit, String> {
    check_primary(&gpu.arch)?;
    let hip = |e: hip_bridge::HipError| format!("topology.hetero_experts: MTP head: {e}");
    let mut guard = lock_device()?;
    let ctx = guard.as_mut().ok_or_else(|| {
        "topology.hetero_experts: MTP head attached before the trunk layers".to_owned()
    })?;
    let index = ctx.pending.len();
    if ctx.head_counter.is_some()
        || ctx.n_experts != placement.len()
        || ctx.geom.dim != dim
        || ctx.geom.top_k != top_k
        || index >= ctx.geom.layers
    {
        return Err("topology.hetero_experts: the MTP head does not match the load's split (or attached twice)".into());
    }
    let responder = ctx.responder.as_ref().ok_or_else(|| {
        "topology.hetero_experts: MTP head attached before the responder started".to_owned()
    })?;
    responder
        .add_slot(
            index,
            ColdLayer {
                gate_up: tables.0,
                down: tables.1,
                cold_bitmap: placement.to_vec(),
            },
        )
        .map_err(|e| format!("topology.hetero_experts: MTP head: {e:?}"))?;
    let counter = gpu
        .alloc_tensor(&[1], rdna_compute::DType::F32)
        .map_err(hip)?;
    gpu.memcpy_htod_auto(&counter.buf, &0u32.to_ne_bytes())
        .map_err(hip)?;
    // SAFETY: a non-owning view of the head's counter, which the context keeps.
    let counter_view = unsafe {
        rdna_compute::GpuTensor {
            buf: hip_bridge::DeviceBuffer::from_raw(counter.buf.as_ptr(), 4),
            shape: vec![1],
            dtype: rdna_compute::DType::F32,
        }
    };
    ctx.head_counter = Some(counter);
    let bitmap = gpu
        .alloc_tensor(&[placement.len().div_ceil(4)], rdna_compute::DType::F32)
        .map_err(hip)?;
    gpu.memcpy_htod_auto(&bitmap.buf, placement).map_err(hip)?;
    let masked_ids = gpu
        .alloc_tensor(&[hetero_decode_rows() * top_k], rdna_compute::DType::F32)
        .map_err(hip)?;
    eprintln!(
        "  qwen4 hetero experts: MTP head on the device split (slot {index}, {} cold experts)",
        placement.iter().filter(|&&c| c != 0).count()
    );
    Ok(HeteroDeviceSplit {
        layer: 0,
        layers: 1,
        slot: slot_ptrs(ctx, index),
        counter: counter_view,
        cold_bitmap: bitmap,
        masked_ids,
        n_experts: placement.len() as i32,
        max_polls: AWAIT_MAX_POLLS,
        max_rows: hetero_decode_rows(),
        advance: true,
    })
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
        return Err(
            "topology.hetero_experts: not every layer attached the device split; no responder"
                .into(),
        );
    };
    poisoned(ctx, responder)?;
    gpu.hetero_token_advance(&ctx.counter)
        .map_err(|e| format!("topology.hetero_experts: {e}"))
}

/// After a forward's host readback (its work is complete): fail the forward
/// whose cold rows a timed-out wait skipped, so a token computed without its
/// cold experts (an EOS or a request's last token included) never reaches the
/// client. No-op without a device split.
pub(crate) fn check_forward() -> Result<(), String> {
    let guard = lock_device()?;
    match guard.as_ref() {
        Some(ctx) => match ctx.responder.as_ref() {
            Some(responder) => poisoned(ctx, responder),
            None => Ok(()),
        },
        None => Ok(()),
    }
}

/// `Err` when the split is poisoned: the responder stopped, a layer's or a
/// prefill chunk's wait timed out, or the poison word is set.
fn poisoned(ctx: &DeviceContext, responder: &HeteroResponder) -> Result<(), String> {
    if let Some(f) = responder.failure() {
        return Err(format!("topology.hetero_experts: responder stopped: {f}"));
    }
    for l in 0..ctx.geom.layers {
        let addr = ctx.mailbox + ctx.geom.slot(l).error;
        // SAFETY: an aligned word in the live pinned mailbox.
        let e = unsafe { &*(addr as *const std::sync::atomic::AtomicU32) }
            .load(std::sync::atomic::Ordering::Acquire);
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
    // SAFETY: an aligned word in the live pinned mailbox.
    let p = unsafe { &*(ctx.poison as *const std::sync::atomic::AtomicU32) }
        .load(std::sync::atomic::Ordering::Acquire);
    if p != 0 {
        return Err(format!(
            "topology.hetero_experts: the split is poisoned (word {p:#x}): a wait timed out or the responder stopped"
        ));
    }
    Ok(())
}

/// Before each MoE call of an fn-lanes step: advance the device token counter
/// so the call's requests carry a fresh sequence. A lane step makes one MoE
/// call per lane (or row group) per layer, all into the same layer slot; with
/// one advance per step ([`begin_token`]) every call after the first would
/// repeat the sequence the responder already answered and install the
/// earlier call's cold rows. No-op without a device split.
pub(crate) fn advance_call(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let guard = lock_device()?;
    let Some(ctx) = guard.as_ref() else {
        return Ok(());
    };
    gpu.hetero_token_advance(&ctx.counter)
        .map_err(|e| format!("topology.hetero_experts: {e}"))
}

/// Tear down the load's split at unload (or after a load that failed once
/// it attached a layer): drain the primary, stop and join the responder
/// (its captured graphs read the load's cold expert pages), free the token
/// counters, the prefill exchange, the layer-major arena and the pinned
/// mailbox. Every layer's split must already be gone:
/// a prefill exchange still shared is leaked with a warning, never freed
/// under a live reference. No-op without a split. The next load starts a
/// fresh context.
pub(crate) fn release(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let hip = |e: hip_bridge::HipError| format!("topology.hetero_experts: release: {e}");
    let ctx = lock_device()?.take();
    let arena = ARENA
        .lock()
        .map_err(|_| "topology.hetero_experts: arena lock poisoned".to_owned())?
        .take();
    if ctx.is_none() && arena.is_none() {
        return Ok(());
    }
    let mut first: Option<String> = gpu.hip.device_synchronize().err().map(hip);
    if let Some((_, streams, ffn)) = arena {
        for t in std::iter::once(streams).chain(ffn) {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(hip(e));
            }
        }
    }
    if let Some(ctx) = ctx {
        let DeviceContext {
            mailbox,
            counter,
            responder,
            head_counter,
            prefill,
            ..
        } = ctx;
        // The responder polls the mailbox and reads the cold pages: it stops
        // before either goes.
        drop(responder);
        match Arc::try_unwrap(prefill) {
            Ok(prefill) => {
                if let Err(e) = prefill.free_gpu(gpu) {
                    first.get_or_insert(format!("topology.hetero_experts: release: {e:?}"));
                }
            }
            Err(_) => eprintln!(
                "  qwen4 hetero experts: WARNING prefill exchange still shared at release; leaked"
            ),
        }
        for t in std::iter::once(counter).chain(head_counter) {
            if let Err(e) = gpu.free_tensor(t) {
                first.get_or_insert(hip(e));
            }
        }
        if let Err(e) = gpu.hip.host_free(mailbox as *mut std::ffi::c_void) {
            first.get_or_insert(hip(e));
        }
        eprintln!("  qwen4 hetero experts: device split released");
    }
    first.map_or(Ok(()), Err)
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
    fn hetero_needs_the_gfx1201_then_expert_card_pair() {
        assert!(check_devices(&["gfx1201", "gfx1151"], "gfx1151").is_ok());
        for archs in [
            &["gfx1151"][..],
            &["gfx1201"],
            &["gfx1151", "gfx1201"],
            &["gfx1201", "gfx1100"],
            &[],
        ] {
            let e = check_devices(archs, "gfx1151").unwrap_err();
            assert!(e.contains("gfx1201,gfx1151"), "{e}");
        }
    }

    #[test]
    fn hetero_requires_primary_gfx1201_then_gfx1151() {
        assert!(check_primary("gfx1201").is_ok());
        let e = check_primary("gfx1151").unwrap_err();
        assert!(
            e.contains("topology.hetero_experts") && e.contains("gfx1201,gfx1151"),
            "{e}"
        );
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
static ARENA: Mutex<
    Option<(
        (usize, usize, usize),
        rdna_compute::GpuTensor,
        Vec<rdna_compute::GpuTensor>,
    )>,
> = Mutex::new(None);

fn view_of(t: &rdna_compute::GpuTensor) -> rdna_compute::GpuTensor {
    // SAFETY: a non-owning view of an arena tensor that lives until [`release`].
    rdna_compute::GpuTensor {
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(t.buf.as_ptr(), t.buf.size()) },
        shape: t.shape.clone(),
        dtype: t.dtype,
    }
}

/// `Some` when this load runs the hetero split with its prefill exchange and
/// `HIPFIRE_HETERO_LAYER_MAJOR` is not `0`. The arena is allocated on first use
/// and freed with the load's split ([`release`]).
pub(crate) fn layer_major(
    gpu: &mut rdna_compute::Gpu,
    tile_rows: usize,
    wide: usize,
    hidden: usize,
) -> Result<Option<LayerMajor>, String> {
    if hipfire_config::developer_var("HIPFIRE_HETERO_LAYER_MAJOR").is_ok_and(|v| v == "0") {
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
    let mut arena = ARENA
        .lock()
        .map_err(|_| "topology.hetero_experts: arena lock poisoned".to_owned())?;
    let shape = (tile_rows, wide, hidden);
    if arena.as_ref().is_none_or(|(s, _, _)| *s != shape) {
        if let Some((_, streams, ffn)) = arena.take() {
            let _ = gpu.free_tensor(streams);
            for t in ffn {
                let _ = gpu.free_tensor(t);
            }
        }
        let streams = gpu
            .alloc_tensor(&[slots * tile_rows * wide], rdna_compute::DType::F32)
            .map_err(hip)?;
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
    Ok(Some(LayerMajor {
        shared,
        streams: view_of(streams),
        ffn: ffn.iter().map(view_of).collect(),
    }))
}

/// Primary (gfx1201) device bytes the split adds for a `chunk_rows` prefill
/// chunk when `topology.hetero_experts` is requested (0 otherwise): every
/// prefill slot's staging (F16 rows, int4 sidecar snapshot, BF16 partial,
/// route), the per-layer masked ids, and the layer-major arena (HC streams and
/// an FFN buffer per tile in flight). Charged in the placement reserve's
/// `forward` term, so the hot set leaves room for them.
pub(crate) fn primary_device_bytes(config: &crate::config::Qwen4Config, chunk_rows: usize) -> u64 {
    if !matches!(requested(), Ok(Some(_))) {
        return 0;
    }
    let rows = chunk_rows.min(hetero_prefill_rows()) as u64;
    let (hidden, top_k) = (config.hidden_size as u64, config.num_experts_per_tok as u64);
    let wide = config.hc_count as u64 * hidden;
    let int4 = hipfire_dispatch::hetero_mailbox::int4_row_bytes(config.hidden_size) as u64;
    let slot = rows * hidden * 2 * 2 + rows * int4 + rows * top_k * 4 * 2 + 256;
    let masked = config.num_hidden_layers as u64 * rows * top_k * 4;
    let tiles = layer_major_tiles(HETERO_PREFILL_SLOTS) as u64;
    let arena = tiles * rows * (wide + hidden) * 4;
    HETERO_PREFILL_SLOTS as u64 * slot + masked + arena
}
