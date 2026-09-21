// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scratch buffer state extracted from the Gpu god object.
//! Owns all per-GPU scratch allocations for FWHT rotation, FP16/FP8/INT8
//! activation conversion, and ParoQuant activation copies.

use crate::kernels;
use crate::{DType, GpuTensor};
use hip_bridge::{
    DeviceBuffer, Function, HipResult, HipRuntime, KernargBlob, Module, Stream,
    HIP_ERROR_INVALID_IMAGE,
};
use std::collections::HashMap;
use std::ffi::c_void;

// ── ScratchState ─────────────────────────────────────────────────────────

/// Device pointers and extents produced by `ScratchState::prepare_mq4v2_fp8_x`.
/// Pointer views belong to the originating `Gpu` and remain valid only until
/// its next `prepare_mq4v2_fp8_x` call or teardown.
pub struct Mq4v2Fp8Prepared {
    pub x_fp8: *mut c_void,
    pub half_sums: *mut c_void,
    pub row_scales: *mut c_void,
    pub x_fp8_bytes: usize,
    pub half_sums_bytes: usize,
    pub row_scales_bytes: usize,
    pub n: usize,
    pub k: usize,
    pub scale_mode: i32,
    /// `true` when `x_fp8` uses 16x16 WMMA-fragment order rather than [N,K].
    pub fragment_order: bool,
}

/// Opaque reservation of the shared `int4_mmq_x_scratch` buffer for a
/// producer-emitted IU4 sidecar (C2). Obtained from
/// [`ScratchState::reserve_int4_mmq`] / [`crate::Gpu::reserve_int4_mmq`].
/// Does not launch a quantizer — the producer writes `block_i4_128` in place.
/// Converted to [`Int4MmqPrepared`] only after a successful producer launch.
#[derive(Debug)]
pub struct Int4MmqReservation {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int4MmqReservation {
    #[inline]
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Frozen producer-emitted IU4 handle. Fields are intentionally private so
/// consumers cannot reconstruct a fake handle or bypass generation checks.
/// Valid only until the next `reserve_int4_mmq` on the same Gpu (generation
/// bump) or teardown.
#[derive(Debug)]
pub struct Int4MmqPrepared {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int4MmqPrepared {
    /// Seal a reservation after the producer wrote the sidecar. Host only —
    /// no device work.
    pub fn from_reservation(res: Int4MmqReservation) -> Self {
        Self {
            ptr: res.ptr,
            k: res.k,
            n: res.n,
            generation: res.generation,
        }
    }

    /// Validate generation / (k,n) / pointer against the live scratch and
    /// return the device pointer for the IU4 consumer. Fails closed on any
    /// mismatch so a stale handle cannot silently re-enter the standalone
    /// quantizer path.
    pub fn checked_ptr(
        &self,
        live_generation: u64,
        live_ptr: *mut c_void,
        k: usize,
        n: usize,
    ) -> HipResult<*mut c_void> {
        if self.ptr.is_null()
            || live_ptr.is_null()
            || self.ptr != live_ptr
            || self.generation != live_generation
            || self.k != k
            || self.n != n
        {
            return Err(hip_bridge::HipError::new(
                0,
                "Int4MmqPrepared: stale or mismatched IU4 sidecar handle",
            ));
        }
        Ok(self.ptr)
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

pub struct ScratchState {
    pub mq_signs1: Option<GpuTensor>,
    pub mq_signs2: Option<GpuTensor>,
    pub mq_signs1_128: Option<GpuTensor>,
    pub mq_signs2_128: Option<GpuTensor>,
    pub mq_x_rot: Option<GpuTensor>,
    pub mq_x_rot_fp8: Option<DeviceBuffer>,
    pub mq_x_rot_fp8_bytes: usize,
    pub mq_x_q8: Option<DeviceBuffer>,
    pub mq_x_scales: Option<DeviceBuffer>,
    /// Persistent gfx1100 K=2048 RMSNorm+MQ state shared by the split and
    /// wavegrid experiments. The wavegrid layout is eight f32 partials, one
    /// f32 RMS value, and three u32 epoch counters, padded to 64 bytes; the
    /// split path uses its first f32 as the RMS handoff.
    pub mq_rmsnorm_wavegrid_scratch: Option<DeviceBuffer>,
    /// Dedicated F32 temporary for the unfused GEMV-residual alias fallback.
    /// Lazily allocated and grown on demand; no other scratch path uses it.
    pub gemv_residual_tmp: Option<GpuTensor>,
    pub paro_x_scratch: Option<GpuTensor>,
    /// Rotation scratch buffers for PARO fused-kernel dispatch. 4 × [k] F32
    /// buffers, lazily allocated and grown on demand. Used by
    /// `fused_qkvza_paro4g128t` (4 explicit) and `fused_gate_up_paro4g128t`
    /// (1 explicit + `mq_x_rot` internal). `ensure_paro_fused_scratch`
    /// allocates/grows; `DeviceBuffer::alias()` builds per-call descriptors.
    pub paro_fused_scratch: Option<Vec<GpuTensor>>,
    pub fp16_x_scratch: Option<DeviceBuffer>,
    pub fp16_x_scratch_bytes: usize,
    pub fp16_x_source_ptr: *mut c_void,
    pub fp8_x_scratch: Option<DeviceBuffer>,
    pub fp8_x_scratch_bytes: usize,
    pub fp8_x_source_ptr: *mut c_void,
    pub q8_1_mmq_x_scratch: Option<DeviceBuffer>,
    pub q8_1_mmq_x_scratch_bytes: usize,
    /// Dedicated iu4-direct MMQ pre-pass X buffer (`block_i4_128`, 72 B per
    /// [K/128 block, batch]). Not shared with `q8_1_mmq_x_scratch` (144 B
    /// blocks): the iu4 consumer reads nibble headers the Q8_1 prelude
    /// never writes, so aliasing would corrupt both routes.
    pub int4_mmq_x_scratch: Option<DeviceBuffer>,
    pub int4_mmq_x_scratch_bytes: usize,
    /// Generation bumped on every `reserve_int4_mmq` so a prepared handle
    /// cannot outlive a later re-reservation of the same scratch slot.
    pub int4_mmq_generation: u64,
    /// Dedicated MQ4v2 FP8 pre-pass X buffer (E4M3 bytes, [N,K]). Not shared
    /// with `fp8_x_scratch` and never pointer-cached — always overwritten.
    pub mq4v2_fp8_x_scratch: Option<DeviceBuffer>,
    pub mq4v2_fp8_x_scratch_bytes: usize,
    /// Half-row sums [N, K/256, 2] f32 for the MQ4v2 FP8 zero-point correction.
    pub mq4v2_fp8_half_sums_scratch: Option<DeviceBuffer>,
    pub mq4v2_fp8_half_sums_scratch_bytes: usize,
    /// Per-row power-of-two (or unit) scales [N] f32 for MQ4v2 FP8 activations.
    pub mq4v2_fp8_row_scales_scratch: Option<DeviceBuffer>,
    pub mq4v2_fp8_row_scales_scratch_bytes: usize,
    /// Partials buffer for the deterministic K-split GEMM (ksplit_det):
    /// [K_SPLITS][batch_size][M] fp32, grows-never-shrinks.
    pub ksplit_det_partials: Option<DeviceBuffer>,
    pub ksplit_det_partials_bytes: usize,
    /// Partials buffer for the multi-workgroup parallel sampler
    /// (`sample_topk_partial` → `sample_topk_finalize`). Holds
    /// `[n_blocks*TOP_K]` f32 values followed by `[n_blocks*TOP_K]` i32 indices
    /// in one allocation; grows-never-shrinks.
    pub sample_partials: Option<DeviceBuffer>,
    pub sample_partials_bytes: usize,
    /// F4b: f16 Q scratch for the gfx11 FA2 pair (`attention_fa2_q_preconvert_gfx11`
    /// writes it, the FA2 body reads it): [batch, 24, 256] f16, grows-never-shrinks.
    /// Every valid cell is written by the pre-convert launch before the body reads
    /// it (same-stream ordering), so no init is needed.
    pub fa2_q16_scratch: Option<DeviceBuffer>,
    pub fa2_q16_scratch_bytes: usize,
    /// Stage-b fp8 Q scratch (the stage-b Q pre-convert writes it, the
    /// stage-b FA2 body reads it): [batch, 24, 256] e4m3 codes followed by
    /// [batch, 24] f32 `sq`, followed (route Q only) by the 64-key scale
    /// plane (256 B per (blockIdx.x, kv_h, split) slot: 64 f16 K scales +
    /// 64 f16 V scales). Grows-never-shrinks with the same pre-growth
    /// capture invalidation as `fa2_q16_scratch`. Route N never sizes the
    /// scale plane (it reads scales from the native row header). Every
    /// valid cell is written by the pre-convert/fill before the body reads
    /// it (same-stream ordering), so no init is needed.
    pub fa2_fp8_q_scratch: Option<DeviceBuffer>,
    pub fa2_fp8_q_scratch_bytes: usize,
}

// ── Shared kernel dispatch helpers ──────────────────────────────────────

/// Compile and load a kernel, caching the result in `modules`/`functions`.
pub(crate) fn compile_and_load_kernel(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    module_name: &str,
    source: &str,
    func_name: &str,
) -> HipResult<()> {
    if functions.contains_key(func_name) {
        return Ok(());
    }
    let obj_path = compiler.compile(module_name, source)?;
    let obj_path_str = obj_path.to_str().unwrap().to_string();
    // Alias the launched function name to this arch's compiled artifact so the
    // retained-PM4 capture can resolve func_name -> owning .hsaco even when the
    // arch-selected module name differs (e.g. gemv_hfq4g256_residual launched vs
    // module gemv_hfq4g256_residual_rdna3 on RDNA3). Additive; no-op when equal.
    if func_name != module_name {
        compiler.register_func_artifact(func_name, std::path::PathBuf::from(&obj_path_str));
    }
    if !modules.contains_key(module_name) {
        let module = module_load_or_recompile(hip, compiler, module_name, source, &obj_path_str)?;
        modules.insert(module_name.to_string(), module);
    }
    let module = &modules[module_name];
    let func = hip.module_get_function(module, func_name).map_err(|error| {
        let context = format!(
            "hipModuleGetFunction failed for symbol {func_name:?} in module {module_name:?}: {error}"
        );
        hip_bridge::HipError::new(error.code, &context)
    })?;
    functions.insert(func_name.to_string(), func);
    Ok(())
}

/// Load a compiled module, self-healing a stale/invalid cached image. If
/// `hipModuleLoad` rejects the `.hsaco` as an invalid device image
/// (`HIP_ERROR_INVALID_IMAGE`) — e.g. a cross-build blob left in a shared
/// `.hipfire_kernels` cache — evict it, recompile from source, and retry once.
/// Any other error propagates unchanged. (Fix for the bench/run "device kernel
/// image is invalid" crash when two daemon builds share a cwd kernel cache.)
pub(crate) fn module_load_or_recompile(
    hip: &HipRuntime,
    compiler: &mut crate::compiler::KernelCompiler,
    module_name: &str,
    source: &str,
    obj_path: &str,
) -> HipResult<Module> {
    match hip.module_load(obj_path) {
        Ok(m) => Ok(m),
        Err(e) if e.code == HIP_ERROR_INVALID_IMAGE => {
            eprintln!(
                "  {module_name}: cached kernel image invalid (HIP {}); recompiling from source",
                e.code
            );
            let fresh = compiler.recompile(module_name, source)?;
            hip.module_load(fresh.to_str().unwrap())
        }
        Err(e) => Err(e),
    }
}

/// Launch a kernel, routing through the blob path when graph capture, replay
/// recording, or force_blob is active. Shared between `Gpu::launch_maybe_blob`
/// and `ScratchState` methods so the branching logic stays in one place.
///
/// Invariant: for any body, `capture_blobs.len()` after a HipGraph capture
/// equals `replay.recorded_launches().len()` after a ReplayController capture.
/// A divergence means a helper bypassed the replay recorder.
pub(crate) fn launch_maybe_blob(
    hip: &HipRuntime,
    compiler: Option<&crate::compiler::KernelCompiler>,
    functions: &HashMap<String, Function>,
    stream: Option<&Stream>,
    capture_blobs: &mut Vec<Vec<u8>>,
    capture_mode: bool,
    force_blob_path: bool,
    mut replay: Option<&mut crate::replay::ReplayController>,
    func_name: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    params: &mut [*mut c_void],
    blob_builder: impl FnOnce() -> KernargBlob,
) -> HipResult<()> {
    let record = replay.as_ref().map_or(false, |r| r.is_recording());
    let result: HipResult<()> = if record || capture_mode || force_blob_path {
        let mut blob = blob_builder();
        blob.pad_to(16);
        if record {
            // Single decision point for how a launch is recorded: same
            // artifact lookup shape as `Gpu::launch_maybe_blob_bound`.
            let artifact = compiler
                .as_ref()
                .and_then(|c| {
                    c
                .compiled_kernels()
                .get(func_name)
                .or_else(|| match func_name {
                    "mq_rotate_x" => c.compiled_kernels().get("gemv_mq4g256"),
                    "deinterleave_f32_batched" => {
                        c.compiled_kernels().get("deinterleave_batched")
                    }
                    name if name.starts_with("gemv_hfq4g256_residual_sigmoid_scaled_gpu") => {
                        c
                            .compiled_kernels()
                            .get("gemv_hfq4g256_residual_scaled")
                    }
                    "gemv_hfq4g256_moe_gate_up_k8_indexed" => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_moe_gate_up_indexed"),
                    name if name.starts_with("gemv_hfq4g256_multirow_r") => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_multirow_default")
                        .or_else(|| {
                            c
                                .compiled_kernels()
                                .get("gemv_hfq4g256_multirow_rdna3")
                        }),
                    name if name.starts_with("gemv_hfq4g256_residual_multirow_r") => c
                        .compiled_kernels()
                        .get("gemv_hfq4g256_residual_multirow_default")
                        .or_else(|| {
                            c
                                .compiled_kernels()
                                .get("gemv_hfq4g256_residual_multirow_rdna3")
                        }),
                    _ => None,
                })
                .or_else(|| {
                    func_name
                        .strip_suffix("_f32")
                        .and_then(|name| c.compiled_kernels().get(name))
                })
                        .cloned()
                });
            replay.as_mut().unwrap().record_hip_launch_typed_bound(
                hip,
                func_name,
                artifact,
                grid,
                block,
                shared_mem,
                blob.as_bytes(),
                None,
            );
        }
        if capture_mode {
            capture_blobs.push(blob.into_vec());
            let buf = capture_blobs.last_mut().unwrap();
            let func = &functions[func_name];
            unsafe { hip.launch_kernel_blob(func, grid, block, shared_mem, stream, buf.as_mut_slice()) }
        } else {
            let mut bytes = blob.into_vec();
            let func = &functions[func_name];
            unsafe { hip.launch_kernel_blob(func, grid, block, shared_mem, stream, bytes.as_mut_slice()) }
        }
    } else {
        let func = &functions[func_name];
        unsafe { hip.launch_kernel(func, grid, block, shared_mem, stream, params) }
    };
    // Scratch converts share the dispatch stream: a failure here names the
    // kernel the same way the dispatch funnel does. Deliberately no
    // last-kernel recording — this helper has no `Gpu` to record into.
    result.map_err(|e| e.with_kernel(func_name))
}

/// Predicate for the FP16/FP8 scratch fast path. The convert kernel must run
/// iff either recorder is active (`is_recording || capture_mode`) or the
/// cached source pointer differs. This is the same predicate
/// `Gpu::launch_maybe_blob_bound` uses for deciding whether to record, so
/// the skip and the record stay coupled: if the kernel does not run it is
/// not recorded, and if a recorder is active the kernel always runs.
#[inline]
pub(crate) fn scratch_must_convert(
    capture_mode: bool,
    is_recording: bool,
    cached_ptr: *mut c_void,
    src_ptr: *mut c_void,
) -> bool {
    is_recording || capture_mode || cached_ptr != src_ptr
}

/// Unified predicate for routing through the blob path. Mirrors
/// `Gpu::launch_maybe_blob_bound`'s `record || capture_mode || force_blob_path`.
#[inline]
pub(crate) fn use_blob_path(is_recording: bool, capture_mode: bool, force_blob_path: bool) -> bool {
    is_recording || capture_mode || force_blob_path
}

// ── FWHT sign table generation (deterministic LCG) ──────────────────────

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

// ── ScratchState helpers ────────────────────────────────────────────────

/// Grow a `Option<DeviceBuffer>` scratch slot to at least `needed` bytes,
/// RELEASING the previous allocation.
///
/// `DeviceBuffer` has no `Drop` impl, so the natural-looking
///
/// ```ignore
/// if self.foo_bytes < needed {
///     self.foo = Some(hip.malloc(needed)?);
///     self.foo_bytes = needed;
/// }
/// ```
///
/// silently leaks the old allocation on every growth step, for every
/// architecture. These scratches are keyed on batch/context, so a serving
/// session that sees a sequence of increasing shapes leaks one buffer per
/// distinct larger shape.
///
/// Measured on gfx1100 (25.8 GB) before the leak fix, issuing requests with
/// growing prompts and sampling VRAM between them: repeating the SAME shape
/// cost +0.0 MB (the buffer is correctly reused), while each larger shape
/// added its FULL size rather than the increment --
/// +199.6, +411.1, +754.9, +1459.9, +2906.8 MB across five shapes, 5.7 GB
/// retained. That is what pushed a multi-turn DFlash session into
/// `hipMalloc: out of memory` on turn 3.
///
/// The old buffer may still be referenced by kernels already enqueued on the
/// stream, and `HipRuntime::free` documents that the caller must ensure the
/// buffer is idle, so synchronise before releasing. Growth is monotonic and
/// bounded by the largest shape ever seen, so this sync is rare and its cost
/// is irrelevant next to the allocation it replaces.
///
/// A captured hipGraph embeds the pointers live at capture time, so this
/// function MUST NOT free a replaced buffer while a stale graph could still
/// replay it: freeing here once broke qwen35 outright, every turn empty with
/// `spec_step: HipError(700) ... reset_recurrent`. The previous revision
/// handled that with `std::mem::forget(old)` once any graph had been
/// captured — correct output, but the retained buffers were the 5.7 GB leak
/// above. The fix is invalidation, not retention: every `Gpu` caller checks
/// [`scratch_will_grow`] before delegating and drops all captured execution
/// state first (`Gpu::invalidate_for_scratch_growth`, which reuses the
/// model-swap teardown `invalidate_for_layout_growth`: AR graph + verify /
/// replay graphs + retained Redline route). Graphs re-capture lazily on the
/// next replay, so each growth event costs at most one re-capture — rare, and
/// bounded by the largest shape ever seen. Per-pointer attribution is
/// impossible (captured graphs store only baked device pointers, no slot
/// provenance), so invalidation is wholesale by design.
///
/// Frees BEFORE allocating: the whole point is to run when memory is tight,
/// and holding both at once is what we are trying to avoid. On allocation
/// failure the slot is left empty with a zero byte count, so the `?` in the
/// caller returns before any `unwrap`, and a later call retries cleanly.
fn grow_scratch_buffer(
    hip: &HipRuntime,
    slot: &mut Option<DeviceBuffer>,
    have_bytes: &mut usize,
    needed: usize,
) -> HipResult<()> {
    if !scratch_will_grow(*have_bytes, slot.is_some(), needed) {
        return Ok(());
    }
    if let Some(old) = slot.take() {
        // The caller invalidated captured graphs before delegating (see the
        // doc comment): no live graph embeds `old`, so sync + free is safe.
        hip.device_synchronize()?;
        let _ = hip.free(old);
    }
    *have_bytes = 0;
    let fresh = hip.malloc(needed)?;
    *slot = Some(fresh);
    *have_bytes = needed;
    Ok(())
}

/// True when [`grow_scratch_buffer`] would replace the slot: the early-return
/// condition factored out so `Gpu` callers can invalidate captured graphs
/// BEFORE delegating. Single source of truth — the two sites cannot drift.
///
/// `slot_is_some` matters: a never-allocated slot with a stale non-zero byte
/// count must still allocate.
#[inline]
pub(crate) fn scratch_will_grow(
    have_bytes: usize,
    slot_is_some: bool,
    needed: usize,
) -> bool {
    !(have_bytes >= needed && slot_is_some)
}

/// Whether a scratch growth must invalidate captured execution state first:
/// a graph has been captured in this process, and no capture/record is in
/// flight (mid-flight invalidation would clear the `capture_blobs` the
/// in-flight capture is filling). Pure so the contract is unit-testable;
/// `Gpu::invalidate_for_scratch_growth` feeds it live state.
#[inline]
pub(crate) fn scratch_growth_invalidates(
    graph_captured: bool,
    capture_mode: bool,
    recording: bool,
) -> bool {
    graph_captured && !capture_mode && !recording
}

/// Byte size of the `q8_1_mmq_x_scratch` slot for `(k, batch_size)`: one 144 B
/// `block_q8_1_mmq` per [K/128 block, batch]. Shared by
/// `ensure_q8_1_mmq_x` / `ensure_q8_1_mmq_x128` and their `Gpu` callers, which
/// need the same number for the pre-growth invalidation check.
#[inline]
pub(crate) fn q8_1_mmq_x_needed(k: usize, batch_size: usize) -> usize {
    let blocks_k = (k + 127) / 128;
    let block_q8_1_mmq_bytes = 144usize;
    blocks_k * batch_size * block_q8_1_mmq_bytes
}
/// Byte geometry of the stage-b fp8 Q scratch slot for
/// `(batch_size, n_splits, with_scale_plane)`: returns
/// `(total_bytes, sq_offset, scale_offset)` where `sq_offset` starts the
/// `[batch, 24]` f32 `sq` section after the `[batch, 24, 256]` e4m3 codes,
/// and `scale_offset` starts the route-Q 64-key scale plane (256 B per
/// `(blockIdx.x, kv_h, split)` slot: 64 f16 K scales + 64 f16 V scales).
/// Route N passes `with_scale_plane = false` (scales come from the native
/// row header, never sized in). All three sections are 16-byte aligned at
/// every admitted batch (codes are batch*6144 B, sq batch*96 B, plane
/// slots*256 B). Shared by `ensure_fa2_fp8_q_scratch` and the stage-b
/// launchers, which need the same number for the pre-growth invalidation
/// check. Callers must have validated `batch_size`/`n_splits` first.
#[inline]
pub(crate) fn fa2_fp8_q_needed(
    batch_size: usize,
    n_splits: usize,
    with_scale_plane: bool,
) -> (usize, usize, usize) {
    let codes_bytes = batch_size * 24 * 256;
    let sq_bytes = batch_size * 24 * 4;
    let sq_offset = codes_bytes;
    let scale_offset = codes_bytes + sq_bytes;
    let scale_bytes = if with_scale_plane {
        batch_size.div_ceil(8) * 4 * n_splits * 256
    } else {
        0
    };
    (scale_offset + scale_bytes, sq_offset, scale_offset)
}

/// Byte size of the `int4_mmq_x_scratch` slot for `(k, batch_size)`: one 72 B
/// `block_i4_128` per [K/128 block, batch]. Shared by `ensure_int4_mmq_x`
/// and its `Gpu` caller.
#[inline]
pub(crate) fn int4_mmq_x_needed(k: usize, batch_size: usize) -> usize {
    let blocks_k = (k + 127) / 128;
    let block_i4_128_bytes = 72usize;
    blocks_k * batch_size * block_i4_128_bytes
}

/// Byte size of the `int4_mmq_x_scratch` slot for a producer reservation
/// `(k, n)`. Shared by `reserve_int4_mmq` and its `Gpu` caller. The caller
/// must have checked `k % 256 == 0 && n > 0` first (same as `reserve_int4_mmq`).
#[inline]
pub(crate) fn int4_mmq_reserve_needed(k: usize, n: usize) -> usize {
    let blocks_k = k / 128;
    blocks_k * n * 72
}

/// Byte sizes of the three MQ4v2 FP8 pre-pass buffers `(x_fp8, half_sums,
/// row_scales)` for `(n, k)`: X8 = `n*k` bytes, half_sums =
/// `n*(k/256)*2` f32, row_scales = `n` f32. Shared by
/// `prepare_mq4v2_fp8_x` / `prepare_mq4v2_fp8_x_f32` and their `Gpu` caller.
#[inline]
pub(crate) fn mq4v2_fp8_needed(n: usize, k: usize) -> (usize, usize, usize) {
    let x_fp8_bytes = n.checked_mul(k).expect("mq4v2 fp8 x extent overflow");
    let groups = k / 256;
    let half_sums_bytes = n
        .checked_mul(groups)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
        .expect("mq4v2 fp8 half_sums extent overflow");
    let row_scales_bytes = n
        .checked_mul(std::mem::size_of::<f32>())
        .expect("mq4v2 fp8 row_scales extent overflow");
    (x_fp8_bytes, half_sums_bytes, row_scales_bytes)
}

/// Producer layout extents. Fragment-order A pads only the byte plane to a
/// complete 256-row GEMM tile; decoded sums and scales keep their true-N ABI.
#[inline]
pub(crate) fn mq4v2_fp8_needed_for_layout(
    n: usize,
    k: usize,
    fragment_order: bool,
) -> (usize, usize, usize) {
    let storage_n = if fragment_order {
        n.checked_add(255).expect("mq4v2 fp8 row padding overflow") & !255
    } else {
        n
    };
    let (x_fp8_bytes, _, _) = mq4v2_fp8_needed(storage_n, k);
    let (_, half_sums_bytes, row_scales_bytes) = mq4v2_fp8_needed(n, k);
    (x_fp8_bytes, half_sums_bytes, row_scales_bytes)
}

impl ScratchState {
    /// Ensure the ksplit_det partials scratch is at least `n_bytes`, growing
    /// (never shrinking). Returns the device pointer. No init needed: every
    /// valid output cell is written exactly once per K-split before finalize.
    pub fn ensure_ksplit_det_partials(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.ksplit_det_partials,
            &mut self.ksplit_det_partials_bytes,
            n_bytes,
        )?;
        Ok(self.ksplit_det_partials.as_ref().unwrap().as_ptr())
    }

    /// Ensure the parallel-sampler partials scratch is at least `n_bytes`,
    /// growing (never shrinking). Returns the device base pointer. No init
    /// needed: every valid cell is written by `sample_topk_partial` before
    /// `sample_topk_finalize` reads it.
    pub fn ensure_sample_partials(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.sample_partials,
            &mut self.sample_partials_bytes,
            n_bytes,
        )?;
        Ok(self.sample_partials.as_ref().unwrap().as_ptr())
    }

    /// Ensure the F4b FA2 f16 Q scratch holds at least `n_bytes`
    /// (batch*24*256*2 for the gfx11 FA2 pair), growing (never shrinking).
    /// Returns the device base pointer. No init needed: the pre-convert
    /// launch writes every valid cell before the FA2 body reads it
    /// (same-stream ordering).
    pub fn ensure_fa2_q16_scratch(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.fa2_q16_scratch,
            &mut self.fa2_q16_scratch_bytes,
            n_bytes,
        )?;
        Ok(self.fa2_q16_scratch.as_ref().unwrap().as_ptr())
    }
    /// Ensure the stage-b FA2 fp8 Q scratch holds at least `n_bytes`
    /// (`fa2_fp8_q_needed(batch, n_splits, with_scale_plane).0`), growing
    /// (never shrinking). Returns the device base pointer: e4m3 codes at
    /// +0, f32 `sq` at +batch*24*256, route-Q scale plane after that.
    /// No init needed: the stage-b pre-convert/fill writes every valid
    /// cell before the stage-b body reads it (same-stream ordering).
    pub fn ensure_fa2_fp8_q_scratch(
        &mut self,
        hip: &HipRuntime,
        n_bytes: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.fa2_fp8_q_scratch,
            &mut self.fa2_fp8_q_scratch_bytes,
            n_bytes,
        )?;
        Ok(self.fa2_fp8_q_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure the dedicated GEMV-residual temporary can hold at least
    /// `min_elems` F32 values, growing on demand and never shrinking.
    pub fn ensure_gemv_residual_tmp(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        min_elems: usize,
    ) -> HipResult<&GpuTensor> {
        crate::graph::bind_thread(hip, device_id)?;
        let needed_bytes = min_elems * 4;
        let needs_grow = self
            .gemv_residual_tmp
            .as_ref()
            .map_or(true, |tmp| tmp.buf.size() < needed_bytes);
        if needs_grow {
            // `GpuTensor` has no `Drop` impl: overwrite would strand the old
            // device allocation. The `Gpu` caller invalidated captured graphs
            // first (same contract as `grow_scratch_buffer`), so sync + free.
            if let Some(old) = self.gemv_residual_tmp.take() {
                hip.device_synchronize()?;
                let _ = hip.free(old.buf);
            }
            self.gemv_residual_tmp = Some(GpuTensor {
                buf: hip.malloc(needed_bytes)?,
                shape: vec![min_elems],
                dtype: DType::F32,
            });
        }
        Ok(self.gemv_residual_tmp.as_ref().unwrap())
    }

    /// Lazily initialize MagnumQuant FWHT sign tables (256 floats each, seeds 42 and 1042).
    pub fn ensure_mq_signs(
        &mut self,
        hip: &HipRuntime,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_signs1.is_some() {
            return Ok(());
        }
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = alloc_tensor_on(hip, pool, device_id, &[256], DType::F32)?;
        let s2t = alloc_tensor_on(hip, pool, device_id, &[256], DType::F32)?;
        hip.memcpy_htod(&s1t.buf, &s1b)?;
        hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = alloc_tensor_on(hip, pool, device_id, &[32768], DType::F32)?;
        let x_q8 = hip.malloc(32768)?; // INT8 buffer for dp4a
        let x_scales = hip.malloc(128 * 4)?; // up to 128 groups × f32
        self.mq_signs1 = Some(s1t);
        self.mq_signs2 = Some(s2t);
        self.mq_x_rot = Some(x_rot);
        self.mq_x_q8 = Some(x_q8);
        self.mq_x_scales = Some(x_scales);
        Ok(())
    }

    /// Lazily allocate and zero the persistent gfx1100 RMSNorm state.
    pub fn ensure_mq_rmsnorm_wavegrid_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_rmsnorm_wavegrid_scratch.is_none() {
            let scratch = hip.malloc(64)?;
            hip.memset(&scratch, 0, 64)?;
            self.mq_rmsnorm_wavegrid_scratch = Some(scratch);
        }
        Ok(())
    }

    /// Lazily initialize MagnumQuant FWHT sign tables for G128 (128 floats each,
    /// seeds 43 and 1043). Also allocates the shared `mq_x_rot` scratch if not
    /// already present — the G256 path (`ensure_mq_signs`) normally owns that
    /// allocation, but the G128 path must be self-sufficient so models that carry
    /// only MQ4G128 weights still get the scratch buffer.
    pub fn ensure_mq_signs_128(
        &mut self,
        hip: &HipRuntime,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_signs1_128.is_some() && self.mq_x_rot.is_some() {
            return Ok(());
        }
        if self.mq_signs1_128.is_none() {
            let signs1 = gen_fwht_signs(43, 128);
            let signs2 = gen_fwht_signs(1043, 128);
            let s1b: Vec<u8> = signs1.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = signs2.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1t = alloc_tensor_on(hip, pool, device_id, &[128], DType::F32)?;
            let s2t = alloc_tensor_on(hip, pool, device_id, &[128], DType::F32)?;
            hip.memcpy_htod(&s1t.buf, &s1b)?;
            hip.memcpy_htod(&s2t.buf, &s2b)?;
            self.mq_signs1_128 = Some(s1t);
            self.mq_signs2_128 = Some(s2t);
        }
        // Allocate shared rotation scratch if ensure_mq_signs (G256 path) has not run yet.
        if self.mq_x_rot.is_none() {
            let x_rot = alloc_tensor_on(hip, pool, device_id, &[32768], DType::F32)?;
            self.mq_x_rot = Some(x_rot);
        }
        Ok(())
    }

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).
    pub fn ensure_paro_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        dim: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if let Some(ref s) = self.paro_x_scratch {
            if s.buf.size() >= dim * 4 {
                return Ok(());
            }
        }
        // Free-before-alloc: `GpuTensor` has no `Drop`; see `ensure_gemv_residual_tmp`.
        if let Some(old) = self.paro_x_scratch.take() {
            hip.device_synchronize()?;
            let _ = hip.free(old.buf);
        }
        let buf = hip.malloc(dim * 4)?; // F32
        self.paro_x_scratch = Some(GpuTensor {
            buf,
            shape: vec![dim],
            dtype: DType::F32,
        });
        Ok(())
    }

    /// Ensure 4 rotation scratch buffers for Paro fused-kernel dispatch.
    /// Each buffer is sized `[k]` F32. On first call, allocates all 4;
    /// on subsequent calls, grows any buffer whose size is < k.
    /// Separate from `paro_x_scratch` (single activation buffer) because
    /// the fused kernels rotate weights internally and need multiple
    /// independent rotation output buffers.
    pub fn ensure_paro_fused_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        let needed_bytes = k * 4; // F32
        match &mut self.paro_fused_scratch {
            Some(bufs) => {
                // Grow any buffer that's too small (never shrinks).
                for buf in bufs.iter_mut() {
                    if buf.buf.size() < needed_bytes {
                        // Alloc-first (not free-first): a malloc failure leaves
                        // the old buffer live in the slot via `?`, and these
                        // four F32 rows are small enough that briefly holding
                        // both is harmless. `GpuTensor`/`DeviceBuffer` have no
                        // `Drop`, so free the replaced buffer explicitly. The
                        // `Gpu` caller invalidated captured graphs first (same
                        // contract as `grow_scratch_buffer`), so sync + free.
                        let old = std::mem::replace(
                            buf,
                            GpuTensor {
                                buf: hip.malloc(needed_bytes)?,
                                shape: vec![k],
                                dtype: DType::F32,
                            },
                        );
                        hip.device_synchronize()?;
                        let _ = hip.free(old.buf);
                    }
                }
            }
            None => {
                let mut vec = Vec::with_capacity(4);
                for _ in 0..4 {
                    vec.push(GpuTensor {
                        buf: hip.malloc(needed_bytes)?,
                        shape: vec![k],
                        dtype: DType::F32,
                    });
                }
                self.paro_fused_scratch = Some(vec);
            }
        }
        Ok(())
    }

    /// Ensure the FP16 X scratch contains the conversion of `x`. Skips the
    /// convert kernel if `x.buf.as_ptr()` matches the last converted source.
    /// When either recorder is active (`capture_mode` or `replay.is_recording()`)
    /// the kernel always runs so the tape stays complete; the skip only applies
    /// to the live non-recording path. Returns the FP16 device pointer.
    pub fn ensure_fp16_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems * 2;

        // Grow scratch if needed (never shrinks), releasing the old buffer.
        if self.fp16_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp16_x_scratch,
                &mut self.fp16_x_scratch_bytes,
                needed,
            )?;
            self.fp16_x_source_ptr = std::ptr::null_mut(); // force reconversion after realloc
        }

        let must_convert = scratch_must_convert(capture_mode, replay.is_recording(), self.fp16_x_source_ptr, src_ptr);
        if must_convert {
            let in_ptr = src_ptr;
            let out_ptr = self.fp16_x_scratch.as_ref().unwrap().as_ptr();
            let n_val = n_elems as i32;
            let mut in_ptr_m = in_ptr;
            let mut out_ptr_m = out_ptr;
            let mut n_val_m = n_val;
            let mut conv_params: Vec<*mut c_void> = vec![
                &mut in_ptr_m as *mut _ as *mut c_void,
                &mut out_ptr_m as *mut _ as *mut c_void,
                &mut n_val_m as *mut _ as *mut c_void,
            ];
            let grid = ((n_elems + 255) / 256) as u32;
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "convert_f32_to_f16",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = KernargBlob::new();
                    b.push_ptr(in_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(n_val);
                    b
                },
            )?;
            self.fp16_x_source_ptr = src_ptr;
        }

        Ok(self.fp16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Convert F32 to F16 without caching. Used when the same x tensor
    /// pointer is reused with different contents across layers (e.g.
    /// DeepSeek V4 prefill reuses the same x_in pointer with new contents
    /// every layer), where pointer-keyed caching would read stale FP16.
    /// Always launches; both recorders observe the same launch via
    /// `launch_maybe_blob`'s unified `record || capture_mode || force_blob` gate.
    pub fn convert_fp16_x_uncached(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let needed = n_elems * 2;
        if self.fp16_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp16_x_scratch,
                &mut self.fp16_x_scratch_bytes,
                needed,
            )?;
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }

        let in_ptr = x.buf.as_ptr();
        let out_ptr = self.fp16_x_scratch.as_ref().unwrap().as_ptr();
        let n_val = n_elems as i32;
        let mut in_ptr_m = in_ptr;
        let mut out_ptr_m = out_ptr;
        let mut n_val_m = n_val;
        let mut conv_params: Vec<*mut c_void> = vec![
            &mut in_ptr_m as *mut _ as *mut c_void,
            &mut out_ptr_m as *mut _ as *mut c_void,
            &mut n_val_m as *mut _ as *mut c_void,
        ];
        let grid = ((n_elems + 255) / 256) as u32;
        launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "convert_f32_to_f16",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut conv_params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(in_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(n_val);
                b
            },
        )?;
        Ok(self.fp16_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure the FP8 (E4M3) X scratch contains the conversion of `x`
    /// (an F32 GpuTensor). Returns the FP8 device pointer. gfx12 only —
    /// uses cvt_pk_fp8_f32. Caches by `x.buf.as_ptr()` like its FP16
    /// sibling so back-to-back same-X GEMM dispatches skip reconversion.
    /// The cache is bypassed when either recorder is active, matching
    /// `ensure_fp16_x` and `scratch_must_convert`.
    pub fn ensure_fp8_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "pack_f32_to_fp8_gfx12",
            kernels::PACK_F32_TO_FP8_GFX12_SRC,
            "pack_f32_to_fp8_gfx12",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = n_elems; // 1 byte per element

        if self.fp8_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp8_x_scratch,
                &mut self.fp8_x_scratch_bytes,
                needed,
            )?;
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }

        let must_convert = scratch_must_convert(capture_mode, replay.is_recording(), self.fp8_x_source_ptr, src_ptr);
        if must_convert {
            let in_ptr = src_ptr;
            let out_ptr = self.fp8_x_scratch.as_ref().unwrap().as_ptr();
            let n_val = n_elems as i32;
            let mut in_ptr_m = in_ptr;
            let mut out_ptr_m = out_ptr;
            let mut n_val_m = n_val;
            let mut conv_params: Vec<*mut c_void> = vec![
                &mut in_ptr_m as *mut _ as *mut c_void,
                &mut out_ptr_m as *mut _ as *mut c_void,
                &mut n_val_m as *mut _ as *mut c_void,
            ];
            let grid = ((n_elems + 4095) / 4096) as u32;
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "pack_f32_to_fp8_gfx12",
                [grid, 1, 1],
                [256, 1, 1],
                0,
                &mut conv_params,
                || {
                    let mut b = KernargBlob::new();
                    b.push_ptr(in_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(n_val);
                    b
                },
            )?;
            self.fp8_x_source_ptr = src_ptr;
        }

        Ok(self.fp8_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Grow dedicated MQ4v2 FP8 pre-pass buffers and always pack F16 X into
    /// E4M3 with half-row sums and row scales. Deliberately omits pointer
    /// caching (see `invalidate_x_caches_for`): a stable source pointer with
    /// changed contents would otherwise return stale prepared extents.
    ///
    /// Geometry: grid `[N,1,1]`, block `[256,1,1]` (eight wave32s per row).
    /// Preconditions `n>0`,
    /// `k>0`, `k % 256 == 0` are caller-owned; sizes are
    /// X8=`n*k` bytes, half_sums=`n*(k/256)*2` f32, row_scales=`n` f32.
    pub(crate) fn prepare_mq4v2_fp8_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x_f16: *mut c_void,
        n: usize,
        k: usize,
        scale_mode: i32,
    ) -> HipResult<Mq4v2Fp8Prepared> {
        self.prepare_mq4v2_fp8_x_impl(
            hip, compiler, modules, functions, stream, capture_blobs, capture_mode,
            force_blob_path, replay,
            "pack_f16_to_fp8_mq4v2_gfx12",
            kernels::PACK_F16_TO_FP8_MQ4V2_GFX12_SRC,
            "pack_f16_to_fp8_mq4v2_gfx12",
            x_f16, n, k, scale_mode,
        )
    }

    /// F32-input MQ4v2 FP8 pre-pass: same outputs/geometry as
    /// [`Self::prepare_mq4v2_fp8_x`] but packs F32 `x` directly, skipping the
    /// `convert_f32_to_f16` hop. Single F32->E4M3 rounding instead of the
    /// F16 path's double rounding — last-ulp differences accepted (see the
    /// kernel comment; quality covered by the WT2 KLD gate).
    pub(crate) fn prepare_mq4v2_fp8_x_f32(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        x_f32: *mut c_void,
        n: usize,
        k: usize,
        scale_mode: i32,
    ) -> HipResult<Mq4v2Fp8Prepared> {
        self.prepare_mq4v2_fp8_x_impl(
            hip, compiler, modules, functions, stream, capture_blobs, capture_mode,
            force_blob_path, replay,
            "pack_f32_to_fp8_mq4v2_gfx12",
            kernels::PACK_F32_TO_FP8_MQ4V2_GFX12_SRC,
            "pack_f32_to_fp8_mq4v2_gfx12",
            x_f32, n, k, scale_mode,
        )
    }

    /// Shared pre-pass body behind [`Self::prepare_mq4v2_fp8_x`] (F16 input)
    /// and [`Self::prepare_mq4v2_fp8_x_f32`] (F32 input). `module`/`symbol`
    /// select the pack entry; `x_ptr` is that entry's input pointer.
    #[allow(clippy::too_many_arguments)]
    fn prepare_mq4v2_fp8_x_impl(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        module: &str,
        ksrc: &str,
        symbol: &str,
        x_ptr: *mut c_void,
        n: usize,
        k: usize,
        scale_mode: i32,
    ) -> HipResult<Mq4v2Fp8Prepared> {
        compile_and_load_kernel(compiler, hip, modules, functions, module, ksrc, symbol)?;

        let (x_fp8_bytes, half_sums_bytes, row_scales_bytes) = mq4v2_fp8_needed(n, k);

        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_x_scratch,
            &mut self.mq4v2_fp8_x_scratch_bytes,
            x_fp8_bytes,
        )?;
        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_half_sums_scratch,
            &mut self.mq4v2_fp8_half_sums_scratch_bytes,
            half_sums_bytes,
        )?;
        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_row_scales_scratch,
            &mut self.mq4v2_fp8_row_scales_scratch_bytes,
            row_scales_bytes,
        )?;

        // Always overwrite valid extents — no source_ptr cache.
        let out_x = self.mq4v2_fp8_x_scratch.as_ref().unwrap().as_ptr();
        let out_sums = self.mq4v2_fp8_half_sums_scratch.as_ref().unwrap().as_ptr();
        let out_scales = self.mq4v2_fp8_row_scales_scratch.as_ref().unwrap().as_ptr();
        let mut in_ptr_m = x_ptr;
        let mut out_x_m = out_x;
        let mut out_sums_m = out_sums;
        let mut out_scales_m = out_scales;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut scale_mode_m = scale_mode;
        let mut params: Vec<*mut c_void> = vec![
            &mut in_ptr_m as *mut _ as *mut c_void,
            &mut out_x_m as *mut _ as *mut c_void,
            &mut out_sums_m as *mut _ as *mut c_void,
            &mut out_scales_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut scale_mode_m as *mut _ as *mut c_void,
        ];
        // Profile bytes: input row read (F16×2 B or F32×4 B per elem) + FP8
        // bytes, half-sums and row-scales writes. Bandwidth attribution only;
        // the timer is what makes this launch visible in HIPFIRE_PROFILE.
        let bytes = x_fp8_bytes
            + half_sums_bytes
            + row_scales_bytes
            + n * k
                * if symbol == "pack_f32_to_fp8_mq4v2_gfx12" {
                    4
                } else {
                    2
                };
        let timer_name: &'static str = if symbol == "pack_f32_to_fp8_mq4v2_gfx12" {
            "pack_f32_to_fp8_mq4v2_gfx12"
        } else {
            "pack_f16_to_fp8_mq4v2_gfx12"
        };
        let timer = crate::profile::begin_timer(hip, "gemm", timer_name, bytes);
        let result = launch_maybe_blob(
            hip,
            Some(&*compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            symbol,
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x_ptr);
                b.push_ptr(out_x);
                b.push_ptr(out_sums);
                b.push_ptr(out_scales);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(scale_mode);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        result?;

        Ok(Mq4v2Fp8Prepared {
            x_fp8: out_x,
            half_sums: out_sums,
            row_scales: out_scales,
            x_fp8_bytes,
            half_sums_bytes,
            row_scales_bytes,
            n,
            k,
            scale_mode,
            fragment_order: false,
        })
    }


    /// Ensure prefill activations are quantized into a llama.cpp-style
    /// `block_q8_1_mmq` layout. The scratch is ordered by [K/128 block, batch]
    /// so a 128-column batch tile is contiguous for each K tile. Always
    /// launches; both recorders observe it via the unified blob gate.
    pub fn ensure_q8_1_mmq_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        device_id: i32,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "gemm_hfq4g256_residual_mmq",
            kernels::GEMM_HFQ4G256_RESIDUAL_MMQ_SRC,
            "quantize_q8_1_mmq_ds4",
        )?;

        let needed = q8_1_mmq_x_needed(k, batch_size);
        grow_scratch_buffer(
            hip,
            &mut self.q8_1_mmq_x_scratch,
            &mut self.q8_1_mmq_x_scratch_bytes,
            needed,
        )?;

        let src_ptr = x.buf.as_ptr();
        let must_convert = true;
        if must_convert {
            let out_ptr = self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr();
            let mut xp = src_ptr;
            let mut yp = out_ptr;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut xp as *mut _ as *mut c_void,
                &mut yp as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];
            let grid_x = ((k + 1023) / 1024) as u32;
            let grid_y = batch_size as u32;
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "quantize_q8_1_mmq_ds4",
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = KernargBlob::new();
                    b.push_ptr(src_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            )?;
        }

        Ok(self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure prefill activations are quantized at per-128 granularity
    /// (`quantize_q8_1_mmq_ds4_x128`: one (d, s) per 128-K half replicated
    /// to all ds4 slots). Same 144 B block layout and [K/128, batch] order
    /// as [`Self::ensure_q8_1_mmq_x`], same scratch buffer — the consumer
    /// side (`gemm_mq4g256v2_mmq_prequant`) reads the same
    /// `HIPFIRE_GFX11_MMQ_X128` flag, so prelude and consumer always agree.
    /// Always launches.
    pub fn ensure_q8_1_mmq_x128(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        device_id: i32,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "gemm_mq4g256v2_residual_mmq",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_SRC,
            "quantize_q8_1_mmq_ds4_x128",
        )?;

        let needed = q8_1_mmq_x_needed(k, batch_size);
        grow_scratch_buffer(
            hip,
            &mut self.q8_1_mmq_x_scratch,
            &mut self.q8_1_mmq_x_scratch_bytes,
            needed,
        )?;

        let src_ptr = x.buf.as_ptr();
        let must_convert = true;
        if must_convert {
            let out_ptr = self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr();
            let mut xp = src_ptr;
            let mut yp = out_ptr;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut xp as *mut _ as *mut c_void,
                &mut yp as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];
            let grid_x = ((k + 1023) / 1024) as u32;
            let grid_y = batch_size as u32;
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "quantize_q8_1_mmq_ds4_x128",
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = KernargBlob::new();
                    b.push_ptr(src_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            )?;
        }

        Ok(self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Ensure prefill activations are quantized to int4 (`block_i4_128`, 72 B
    /// per [K/128 block, batch]: f32 d, i32 s, 64 B nibbles) for the
    /// iu4-direct MMQ consumer (`gemm_mq4g256v2_mmq_prequant_iu4`), gated by
    /// `HIPFIRE_IU4_PREFILL`. Same [K/128, batch] order and always-launch
    /// lifetime contract as [`Self::ensure_q8_1_mmq_x128`], but a dedicated
    /// `int4_mmq_x_scratch` buffer — the layouts differ (72 B vs 144 B) and
    /// the iu4 consumer reads nibble headers the Q8_1 prelude never writes.
    /// Bumps `int4_mmq_generation` so any prior prepared handle fails closed.
    pub fn ensure_int4_mmq_x(
        &mut self,
        hip: &HipRuntime,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        device_id: i32,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "gemm_mq4g256v2_residual_mmq_iu4",
            kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_SRC,
            "quantize_int4_mmq_ds128",
        )?;

        let needed = int4_mmq_x_needed(k, batch_size);
        grow_scratch_buffer(
            hip,
            &mut self.int4_mmq_x_scratch,
            &mut self.int4_mmq_x_scratch_bytes,
            needed,
        )?;
        // Invalidate any outstanding prepared producer handle.
        self.int4_mmq_generation = self.int4_mmq_generation.wrapping_add(1);

        let src_ptr = x.buf.as_ptr();
        let must_convert = true;
        if must_convert {
            let out_ptr = self.int4_mmq_x_scratch.as_ref().unwrap().as_ptr();
            let mut xp = src_ptr;
            let mut yp = out_ptr;
            let mut k_val = k as i32;
            let mut n_val = batch_size as i32;
            let mut params: Vec<*mut c_void> = vec![
                &mut xp as *mut _ as *mut c_void,
                &mut yp as *mut _ as *mut c_void,
                &mut k_val as *mut _ as *mut c_void,
                &mut n_val as *mut _ as *mut c_void,
            ];
            let grid_x = ((k + 1023) / 1024) as u32;
            let grid_y = batch_size as u32;
            let bytes = batch_size * k * 4 + needed;
            let timer =
                crate::profile::begin_timer(hip, "quantize", "quantize_int4_mmq_ds128", bytes);
            launch_maybe_blob(
                hip,
                Some(&*compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
                "quantize_int4_mmq_ds128",
                [grid_x, grid_y, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = KernargBlob::new();
                    b.push_ptr(src_ptr);
                    b.push_ptr(out_ptr);
                    b.push_i32(k_val);
                    b.push_i32(n_val);
                    b
                },
            )?;
            if let Some(t) = timer {
                t.finish(hip);
            }
        }

        Ok(self.int4_mmq_x_scratch.as_ref().unwrap().as_ptr())
    }

    /// Grow `int4_mmq_x_scratch` for a producer-emitted IU4 sidecar and bump
    /// the generation. Does **not** launch `quantize_int4_mmq_ds128` — the
    /// RMSNorm/FWHT or SwiGLU/FWHT producer writes the 72-byte blocks.
    pub fn reserve_int4_mmq(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<Int4MmqReservation> {
        if k == 0 || n == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "reserve_int4_mmq: need k%256==0 and n>0",
            ));
        }
        let needed = int4_mmq_reserve_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.int4_mmq_x_scratch,
            &mut self.int4_mmq_x_scratch_bytes,
            needed,
        )?;
        self.int4_mmq_generation = self.int4_mmq_generation.wrapping_add(1);
        let ptr = self.int4_mmq_x_scratch.as_ref().unwrap().as_ptr();
        Ok(Int4MmqReservation {
            ptr,
            k,
            n,
            generation: self.int4_mmq_generation,
        })
    }
    /// Grow the three MQ4v2 FP8 pre-pass buffers for a producer-emitted FP8
    /// stream and return their device pointers `(x_fp8, half_sums,
    /// row_scales)`. Does **not** launch `pack_f32_to_fp8_mq4v2_gfx12` — the
    /// fused `_mq4v2_fp8_gfx12` producer writes all three planes in place
    /// with byte-identical outputs. Same always-overwrite contract as
    /// `prepare_mq4v2_fp8_x_impl` (never pointer-cached); the caller seals
    /// the pointers into [`Mq4v2Fp8Prepared`] only after a successful
    /// producer launch.
    pub fn grow_mq4v2_fp8_for_producer(
        &mut self,
        hip: &HipRuntime,
        n: usize,
        k: usize,
        fragment_order: bool,
    ) -> HipResult<(*mut c_void, *mut c_void, *mut c_void)> {
        if k == 0 || n == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "grow_mq4v2_fp8_for_producer: need k%256==0 and n>0",
            ));
        }
        let (x_fp8_bytes, half_sums_bytes, row_scales_bytes) =
            mq4v2_fp8_needed_for_layout(n, k, fragment_order);
        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_x_scratch,
            &mut self.mq4v2_fp8_x_scratch_bytes,
            x_fp8_bytes,
        )?;
        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_half_sums_scratch,
            &mut self.mq4v2_fp8_half_sums_scratch_bytes,
            half_sums_bytes,
        )?;
        grow_scratch_buffer(
            hip,
            &mut self.mq4v2_fp8_row_scales_scratch,
            &mut self.mq4v2_fp8_row_scales_scratch_bytes,
            row_scales_bytes,
        )?;
        Ok((
            self.mq4v2_fp8_x_scratch.as_ref().unwrap().as_ptr(),
            self.mq4v2_fp8_half_sums_scratch.as_ref().unwrap().as_ptr(),
            self.mq4v2_fp8_row_scales_scratch.as_ref().unwrap().as_ptr(),
        ))
    }

    /// Live generation + pointer for [`Int4MmqPrepared::checked_ptr`].
    #[inline]
    pub fn int4_mmq_live(&self) -> (u64, *mut c_void) {
        let ptr = self
            .int4_mmq_x_scratch
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        (self.int4_mmq_generation, ptr)
    }

    /// Invalidate the FP16/FP8 activation scratch caches. Must be called
    /// whenever the scratch buffer used by MagnumQuant rotation is
    /// written — the scratch pointer is stable but the DATA changes per
    /// rotation; without this invalidation, FP8/FP16 activation scratch
    /// returns stale data on every call after the first within a forward
    /// pass (silent correctness bug).
    pub fn invalidate_x_caches_for(&mut self, dst_ptr: *mut c_void) {
        if self.fp16_x_source_ptr == dst_ptr {
            self.fp16_x_source_ptr = std::ptr::null_mut();
        }
        if self.fp8_x_source_ptr == dst_ptr {
            self.fp8_x_source_ptr = std::ptr::null_mut();
        }
    }

    // ── Rotation methods ────────────────────────────────────────────────

    /// Standalone FWHT rotation for MagnumQuant (MQ4). Writes K floats into x_rot.
    /// Exposed so callers can batch one rotation across multiple GEMVs that share x
    /// (e.g., Q/K/V projections all consume the same post-RMSNorm x).
    ///
    /// NOTE: caller must have ensured the kernel (`mq_rotate_x` in module
    /// `gemv_mq4g256`) before calling this method.
    pub fn rotate_x_mq(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Batched `rotate_x_mq`. Grid.y is the batch dim.
    pub fn rotate_x_mq_batched(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut xp = x.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) * batch_size;
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_batched", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x",
            [n_groups * batch_size as u32, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// FWHT-128 standalone rotation for MQ4G128 activations.
    pub fn rotate_x_mq_128(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs_128(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 128) as u32;
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k);
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_128", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x_128",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    pub fn rotate_x_mq_awq(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let xp = x.buf.as_ptr();
        let awp = awq_scale.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &awp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = k * 4 * 3 + 2 * 256 * 4;
        let timer = crate::profile::begin_timer(hip, "fwht", "rotate_x_mq_awq", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "rotate_x_mq_awq",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(awp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Phase A Stage A — F2 batched AWQ variant of `rotate_x_mq`.
    /// Grid.y is the batch dim — processes [N × K] x/x_rot.
    pub fn rotate_x_mq_awq_batched(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        awq_scale: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
        batch_size: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;
        let mut xp = x.buf.as_ptr();
        let mut awp = awq_scale.buf.as_ptr();
        let mut xrp = x_rot.buf.as_ptr();
        let mut s1 = s1_ptr;
        let mut s2 = s2_ptr;
        let mut kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut xrp as *mut _ as *mut c_void,
            &mut awp as *mut _ as *mut c_void,
            &mut s1 as *mut _ as *mut c_void,
            &mut s2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
        ];
        let bytes = (k * 4 * 3 + 2 * 256 * 4) * batch_size;
        let timer = crate::profile::begin_timer(hip, "fwht", "rotate_x_mq_awq_batched", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "rotate_x_mq_awq",
            [n_groups, batch_size as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(awp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result
    }

    /// Fused FWHT rotation + FP8 pack for the decode FP8 path.
    /// Writes both F32 (into `x_rot`) and FP8 (into `mq_x_rot_fp8`
    /// sibling scratch) in one kernel launch. Returns the FP8 buffer's
    /// device pointer for the caller to feed directly to the FP8 GEMV.
    /// gfx12-only — uses cvt_pk_fp8_f32.
    pub fn rotate_x_mq_dual_fp8(
        &mut self,
        hip: &HipRuntime,
        functions: &mut HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        compiler: &mut crate::compiler::KernelCompiler,
        modules: &mut HashMap<String, Module>,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        x_rot: &GpuTensor,
        k: usize,
    ) -> HipResult<*mut c_void> {
        self.ensure_mq_signs(hip, pool, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            "mq_rotate_x_dual_fp8_gfx12",
            kernels::MQ_ROTATE_X_DUAL_FP8_GFX12_SRC,
            "mq_rotate_x_dual_fp8_gfx12",
        )?;
        // Lazily allocate the FP8 sibling scratch sized to match k bytes.
        grow_scratch_buffer(hip, &mut self.mq_x_rot_fp8, &mut self.mq_x_rot_fp8_bytes, k)?;
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let xp = x.buf.as_ptr();
        let xrp = x_rot.buf.as_ptr();
        let xfp = self.mq_x_rot_fp8.as_ref().unwrap().as_ptr();
        let n_groups = (k / 256) as u32;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xrp as *const _ as *mut c_void,
            &xfp as *const _ as *mut c_void,
            &s1_ptr as *const _ as *mut c_void,
            &s2_ptr as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) + k;
        let timer = crate::profile::begin_timer(hip, "fwht", "mq_rotate_x_dual_fp8", bytes);
        let result = launch_maybe_blob(
                hip,
                Some(compiler),
                functions,
                stream,
                capture_blobs,
                capture_mode,
                force_blob_path,
                Some(replay),
            "mq_rotate_x_dual_fp8_gfx12",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xrp);
                b.push_ptr(xfp);
                b.push_ptr(s1_ptr);
                b.push_ptr(s2_ptr);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        self.invalidate_x_caches_for(xrp);
        result?;
        Ok(xfp)
    }

    /// Standalone MQ8 rotate + INT8 quantize of x into internal `mq_x_q8`/`mq_x_scales`.
    /// After this, `gemv_mq8g256_prerotated` can be called multiple times with the same x.
    pub fn rotate_quantize_x_mq8(
        &mut self,
        hip: &HipRuntime,
        compiler: &crate::compiler::KernelCompiler,
        functions: &HashMap<String, Function>,
        stream: Option<&Stream>,
        capture_blobs: &mut Vec<Vec<u8>>,
        capture_mode: bool,
        force_blob_path: bool,
        replay: &mut crate::replay::ReplayController,
        pool: &mut crate::pool::GpuPool,
        device_id: i32,
        x: &GpuTensor,
        k: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        self.ensure_mq_signs(hip, pool, device_id)?;

        let xq_ptr = self.mq_x_q8.as_ref().unwrap().as_ptr();
        let xs_ptr = self.mq_x_scales.as_ref().unwrap().as_ptr();
        let s1_ptr = self.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2_ptr = self.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let n_groups = (k / 256) as u32;

        let xp = x.buf.as_ptr();
        let xq = xq_ptr;
        let xs = xs_ptr;
        let s1 = s1_ptr;
        let s2 = s2_ptr;
        let kv = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &xp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
        ];
        let bytes = crate::profile::mq_rotate_bytes(k) + (k / 256) * 4 + k;
        let timer = crate::profile::begin_timer(hip, "fwht", "mq8_rotate_quantize_x", bytes);
        let result = launch_maybe_blob(
            hip,
            Some(compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            "mq8_rotate_quantize_x",
            [n_groups, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(xq);
                b.push_ptr(xs);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        // Invalidate caches for the internal q8 buffers' output? The output
        // is internal scratch, not x_rot, so no fp16/fp8 cache invalidation needed.
        // But keep symmetric: if any future caller aliases, this remains safe.
        result
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

fn alloc_tensor_on(
    hip: &HipRuntime,
    pool: &mut crate::pool::GpuPool,
    device_id: i32,
    shape: &[usize],
    dtype: DType,
) -> HipResult<GpuTensor> {
    crate::graph::bind_thread(hip, device_id)?;
    let numel: usize = shape.iter().product();
    let byte_size = numel * dtype.size();
    let buf = pool.alloc(hip, byte_size)?;
    Ok(GpuTensor {
        buf,
        shape: shape.to_vec(),
        dtype,
    })
}

#[cfg(test)]
mod scratch_growth_tests {
    use super::*;

    /// The will-grow predicate is the single source of truth shared by
    /// `grow_scratch_buffer` and every `Gpu` pre-growth invalidation guard.
    /// These two sites must agree, or a growth frees a buffer a live graph
    /// embeds (HipError 700) or an invalidation fires without a growth
    /// (needless re-capture every decode token).
    #[test]
    fn will_grow_matches_replacement_condition() {
        // Empty slot must allocate, even against a stale byte count.
        assert!(scratch_will_grow(0, false, 100));
        assert!(scratch_will_grow(10_000, false, 100));
        // Occupied slot: only when short.
        assert!(scratch_will_grow(0, true, 100));
        assert!(scratch_will_grow(99, true, 100));
        assert!(!scratch_will_grow(100, true, 100));
        assert!(!scratch_will_grow(10_000, true, 100));
        // Zero-size requests never grow an occupied slot.
        assert!(!scratch_will_grow(100, true, 0));
    }

    /// Invalidation fires exactly when a stale graph could exist and no
    /// capture/record is in flight. Mid-flight invalidation would clear the
    /// `capture_blobs` the in-flight capture is filling, corrupting it.
    #[test]
    fn invalidation_contract() {
        // Nothing captured: never invalidate (keeps pre-capture prefill and
        // the never-capturing archs on the fast path).
        assert!(!scratch_growth_invalidates(false, false, false));
        assert!(!scratch_growth_invalidates(false, true, false));
        assert!(!scratch_growth_invalidates(false, false, true));
        // Captured, steady state: invalidate (the leak fix).
        assert!(scratch_growth_invalidates(true, false, false));
        // Captured but mid-flight: never invalidate.
        assert!(!scratch_growth_invalidates(true, true, false));
        assert!(!scratch_growth_invalidates(true, false, true));
        assert!(!scratch_growth_invalidates(true, true, true));
    }

    /// Geometry helpers shared by the ensure bodies and the `Gpu` guards:
    /// both sides must compute the same byte counts.
    #[test]
    fn needed_helpers_match_slot_geometry() {
        // q8_1: 144 B per [K/128 block, batch].
        assert_eq!(q8_1_mmq_x_needed(2048, 8192), 16 * 8192 * 144);
        assert_eq!(q8_1_mmq_x_needed(2048, 65535), 16 * 65535 * 144);
        // Non-multiple K rounds up a block.
        assert_eq!(q8_1_mmq_x_needed(2049, 1), 17 * 1 * 144);
        // int4: 72 B per [K/128 block, batch].
        assert_eq!(int4_mmq_x_needed(2048, 1024), 16 * 1024 * 72);
        assert_eq!(int4_mmq_reserve_needed(2048, 1024), 16 * 1024 * 72);
        // mq4v2 fp8: n*k bytes, n*(k/256)*2 f32 half-sums, n f32 row scales.
        assert_eq!(mq4v2_fp8_needed(32, 2048), (65536, 32 * 8 * 2 * 4, 32 * 4));
    }
    /// Stage-b fp8 Q scratch layout (§3–§4 of the stage-b plan): sections
    /// 16-byte aligned and mutually disjoint for every admitted
    /// batch × splits combination, with the route-Q scale plane never
    /// aliasing `sq` and route N sizing no plane at all.
    #[test]
    fn fa2_fp8_q_layout_aligned_and_disjoint() {
        for &batch in &[1usize, 7, 8, 9, 512] {
            for &splits in &[1usize, 8] {
                for &with_plane in &[false, true] {
                    let (total, sq_off, scale_off) =
                        fa2_fp8_q_needed(batch, splits, with_plane);
                    let codes_bytes = batch * 24 * 256;
                    let sq_bytes = batch * 24 * 4;
                    // Sections tile the buffer exactly, in order.
                    assert_eq!(sq_off, codes_bytes, "b={batch} s={splits}");
                    assert_eq!(scale_off, codes_bytes + sq_bytes, "b={batch} s={splits}");
                    // Every section start (and the end) is 16-byte aligned.
                    assert_eq!(sq_off % 16, 0, "b={batch}");
                    assert_eq!(scale_off % 16, 0, "b={batch}");
                    assert_eq!(total % 16, 0, "b={batch} s={splits}");
                    // Non-empty codes/sq for every admitted batch.
                    assert!(codes_bytes > 0 && sq_bytes > 0);
                    let grid_x = batch.div_ceil(8);
                    let n_slots = grid_x * 4 * splits;
                    if with_plane {
                        // 256 B per (blockIdx.x, kv_h, split) slot: 64 f16
                        // K scales + 64 f16 V scales.
                        assert_eq!(total - scale_off, n_slots * 256);
                        // First and last slots' K/V halves are disjoint and
                        // inside the plane.
                        for &slot in &[0, n_slots - 1] {
                            let base = scale_off + slot * 256;
                            let k_end = base + 128;
                            let v_end = base + 256;
                            assert!(base >= scale_off && v_end <= total);
                            assert!(k_end <= base + 128 && base + 128 < v_end);
                        }
                    } else {
                        // Route N: no scale plane sized in, nothing aliases sq.
                        assert_eq!(total, scale_off);
                    }
                }
                // Route N never pays for the route-Q plane.
                let (with, _, _) = fa2_fp8_q_needed(batch, splits, true);
                let (without, _, _) = fa2_fp8_q_needed(batch, splits, false);
                assert!(with > without);
            }
        }
        // Pin the §13 total: batch 512 / 8 splits with plane = 3,719,168 B.
        assert_eq!(fa2_fp8_q_needed(512, 8, true).0, 3_719_168);
        assert_eq!(
            fa2_fp8_q_needed(512, 8, true),
            (3_719_168, 512 * 24 * 256, 512 * 24 * 256 + 512 * 24 * 4)
        );
    }
}
