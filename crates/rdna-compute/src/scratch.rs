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
    /// Grouped-position down plane: records per K128 group (`m_total`); 0 = flat-slot sidecar.
    plane_rows: usize,
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
    /// Grouped-position down plane (record `g * plane_rows + position`); 0 = flat-slot sidecar.
    plane_rows: usize,
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
            plane_rows: res.plane_rows,
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

    /// Seal the CURRENT live `int4_mmq_x_scratch` (generation + pointer from
    /// [`ScratchState::int4_mmq_live`]) after a successful standalone
    /// `ensure_int4_mmq_x` quantize. Host only; crate-private so external
    /// callers cannot forge a handle.
    pub(crate) fn from_live(generation: u64, ptr: *mut c_void, k: usize, n: usize) -> Self {
        Self {
            ptr,
            k,
            n,
            generation,
            plane_rows: 0,
        }
    }

    /// Records per K128 group when this is a grouped-position down plane, `None` for the flat-slot sidecar
    /// (consumed only by the position-indexed `..._gp` down entries).
    #[inline]
    pub fn grouped_plane_rows(&self) -> Option<usize> {
        (self.plane_rows != 0).then_some(self.plane_rows)
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

    /// Records per K128 group of a grouped-position down plane (`m_total`), 0 for the flat-slot layout.
    #[inline]
    pub fn plane_rows(&self) -> usize {
        self.plane_rows
    }
}

/// Opaque reservation of the dedicated `int4_mmq_down_scratch` buffer for the
/// A4-fused gate/up SiLU epilogue, which writes the w_down `block_i4_128`
/// sidecar directly. Never aliases `int4_mmq_x_scratch`. Obtained from
/// [`ScratchState::reserve_int4_mmq_down`] / [`crate::Gpu::reserve_int4_mmq_down`].
/// Converted to [`Int4MmqDownPrepared`] only after a successful launch.
#[derive(Debug)]
pub struct Int4MmqDownReservation {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int4MmqDownReservation {
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

/// Frozen handle to a down-proj sidecar written by the A4-fused gate/up
/// epilogue. Fields are private so consumers cannot forge a handle or bypass
/// the generation check. Valid only until the next `reserve_int4_mmq_down`
/// on the same Gpu or teardown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int4MmqDownPrepared {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int4MmqDownPrepared {
    /// Seal a reservation after the fused producer was launched. Host only —
    /// no device work.
    pub fn from_reservation(res: Int4MmqDownReservation) -> Self {
        Self {
            ptr: res.ptr,
            k: res.k,
            n: res.n,
            generation: res.generation,
        }
    }

    /// Validate generation / (k,n) / pointer against the live down scratch and
    /// return the device pointer for the IU4 consumer. Fails closed on any
    /// mismatch.
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
                "Int4MmqDownPrepared: stale or mismatched IU4 down-sidecar handle",
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

/// Reservation for one producer-emitted `block_i8_128` sidecar.
#[derive(Debug)]
pub struct Int8MmqReservation {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int8MmqReservation {
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Valid only until the next int8 sidecar reservation or quantization.
#[derive(Debug)]
pub struct Int8MmqPrepared {
    ptr: *mut c_void,
    k: usize,
    n: usize,
    generation: u64,
}

impl Int8MmqPrepared {
    pub fn from_reservation(res: Int8MmqReservation) -> Self {
        Self {
            ptr: res.ptr,
            k: res.k,
            n: res.n,
            generation: res.generation,
        }
    }

    pub fn checked_ptr(
        &self,
        generation: u64,
        ptr: *mut c_void,
        k: usize,
        n: usize,
    ) -> HipResult<*mut c_void> {
        if ptr.is_null()
            || ptr != self.ptr
            || generation != self.generation
            || k != self.k
            || n != self.n
        {
            return Err(hip_bridge::HipError::new(
                0,
                "Int8MmqPrepared: stale or mismatched A8 sidecar",
            ));
        }
        Ok(ptr)
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
    /// Four-byte result of `tensor_ops::argmax_f32_host`.
    pub argmax_host: Option<GpuTensor>,
    /// `(x, x_rot, k)` pointers of an `mq_rotate_x` output a fused kernel
    /// already wrote (`Gpu::hyper_read_projected_rotate`); the next matching
    /// `Gpu::rotate_x_mq` consumes it instead of launching. The step executor
    /// clears it after the consuming step.
    pub prerotated: Option<(usize, usize, usize)>,
    /// Zeroed head-pair arrival counters of the fused GDN step's rotation
    /// epilogue (each launch leaves them zero).
    pub gdn_pair_counters: Option<DeviceBuffer>,
    /// Persistent gfx1100 AWQ RMSNorm wavegrid state: 256 f32 per-thread
    /// partials, one f32 RMS, three u32 epoch counters, padded to 2048 bytes.
    pub mq_rmsnorm_awq_wavegrid_scratch: Option<DeviceBuffer>,
    /// Dedicated F32 temporary for the unfused GEMV-residual alias fallback.
    /// Lazily allocated and grown on demand; no other scratch path uses it.
    pub gemv_residual_tmp: Option<GpuTensor>,
    /// GGUF-native route scratch (q8_1 activation row, prefill dequantisation
    /// region); see [`crate::ggml_ops`].
    pub ggml: crate::ggml_ops::GgmlScratch,
    /// Int8 activation scratch `(xq, dx)` of the opt-in exact-gfx1100
    /// small-batch verify GEMV (`gemv_q8x.rs`), allocated on first use.
    pub q8x_act: Option<(GpuTensor, GpuTensor)>,
    /// Dedicated F32 `x_out` operand for `Gpu::mq6_fp8_prepare_x` (the MQ4v2 FP8
    /// producer's rotated-row output, `[batch * k]`; the in-register producer
    /// never writes it). Lazily allocated, grown on demand, never shrunk.
    pub mq6_fp8_x_rot: Option<GpuTensor>,
    pub paro_x_scratch: Option<GpuTensor>,
    /// Rotation scratch buffers for PARO fused-kernel dispatch. 4 × [k] F32
    /// buffers, lazily allocated and grown on demand. Used by
    /// `fused_qkvza_paro4g128t` (4 explicit) and `fused_gate_up_paro4g128t`
    /// (1 explicit + `mq_x_rot` internal). `ensure_paro_fused_scratch`
    /// allocates/grows; `DeviceBuffer::alias()` builds per-call descriptors.
    pub paro_fused_scratch: Option<Vec<GpuTensor>>,
    pub fp16_x_scratch: Option<DeviceBuffer>,
    pub fp16_x_scratch_bytes: usize,
    /// Global score rows for the QSA block selector when an arena's pooled
    /// block count exceeds its dynamic-LDS row (contexts past 61,440 tokens):
    /// `QSA_SELECT_GLOBAL_ROWS` rows of the arena's padded block count.
    pub qsa_select_scores: Option<DeviceBuffer>,
    pub qsa_select_scores_bytes: usize,
    /// F32 dequantization of one Q8 / Q8Ef GDN state for the chunked prefill route.
    pub gdn_state_f32: Option<DeviceBuffer>,
    pub gdn_state_f32_bytes: usize,
    /// F16 K and block-transposed V of the QSA cache rows the gathered WMMA
    /// attention reads (`tensor_ops::qsa_gathered_wmma`), legacy backing.
    /// Its own slot, not the shared FP16 X scratch: Qwen4 reserves it for the
    /// whole context at load (`tensor_ops::reserve_qsa_gathered_wmma_scratch`),
    /// so no captured graph or recorded tape ever sees it move. Used where
    /// the VMM workspace is not admitted (`Gpu::qsa_gather_vmm_capable`:
    /// everything but gfx1151/gfx1201 on a VMM-certified platform); kept for
    /// the process.
    pub qsa_gather_f16: Option<DeviceBuffer>,
    pub qsa_gather_f16_bytes: usize,
    /// The same workspace as a registered VMM owner (gfx1151 or gfx1201 on a
    /// certified platform): one stable virtual reservation for the whole context
    /// (`tensor_ops::reserve_qsa_gathered_wmma_workspace`), physical pages
    /// mapped in place over the prefix a forward ending at `end` touches
    /// (`tensor_ops::qsa_gathered_wmma_scratch_bytes(heads, end)`). Growth is
    /// refused while a capture or record is armed. Gpu-owned, not
    /// model-owned: released by `Gpu::invalidate_weight_caches` and the
    /// `Gpu::ensure_vmm_cleaned` gate. At most one of the two slots is set.
    pub qsa_gather_vmm: Option<GpuTensor>,
    /// Mapped bytes of `qsa_gather_vmm`; its reserved virtual extent is the
    /// tensor's byte size.
    pub qsa_gather_vmm_mapped: usize,
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
    /// Dedicated down-proj `block_i4_128` sidecar written by the A4-fused
    /// gate/up SiLU epilogue (`HIPFIRE_V2B_A4_EPI=1`). Never aliases
    /// `int4_mmq_x_scratch` and never bumps `int4_mmq_generation`.
    pub int4_mmq_down_scratch: Option<DeviceBuffer>,
    pub int4_mmq_down_scratch_bytes: usize,
    /// Generation bumped on every `reserve_int4_mmq_down`.
    pub int4_mmq_down_generation: u64,
    /// Generation bumped on every `reserve_int4_mmq` so a prepared handle
    /// cannot outlive a later re-reservation of the same scratch slot.
    pub int4_mmq_generation: u64,
    /// Generation whose `int4_mmq_x_scratch` contents a slab producer twin
    /// wrote (`HIPFIRE_IU4_SLAB` planes, read only by the `_b1s` GEMMs).
    /// Every other generation holds token-order `block_i4_128` records.
    pub int4_mmq_slab_generation: Option<u64>,
    /// W8A8 Q8_0 sidecar: int8 [N, K] followed by F32 scales [K/32, N]
    /// (`quantize_x_q8_blk32`), laid out by [`q8_a8_x_layout`].
    pub q8_a8_x_scratch: Option<DeviceBuffer>,
    pub q8_a8_x_scratch_bytes: usize,
    /// A8 `block_i8_128` sidecar, [K/128, N] of 136-byte blocks.
    pub int8_mmq_x_scratch: Option<DeviceBuffer>,
    pub int8_mmq_x_scratch_bytes: usize,
    pub int8_mmq_generation: u64,
    /// Qwen4 symmetric IU4 MoE stable-grouping scratch: `[L]` local ranks,
    /// `[E, C]` per-chunk counts and `[E, C + 1]` chunk prefixes (`C =
    /// ceil(L / 256)`), one allocation laid out by [`qwen4_moe_group_layout`].
    /// Every word a grouping reads is written earlier in the same invocation,
    /// so it needs no init. Grows-never-shrinks; reserved before capture by
    /// [`ScratchState::reserve_qwen4_moe_group`].
    pub qwen4_moe_group_scratch: Option<DeviceBuffer>,
    pub qwen4_moe_group_scratch_bytes: usize,
    /// Qwen4 symmetric IU4 MoE down sidecar: compact flat-slot `[K/128, L]`
    /// `block_i4_128`. K = 640 is not a multiple of 256, so it cannot share
    /// `int4_mmq_x_scratch`; generation-checked like it.
    pub qwen4_moe_down_i4_scratch: Option<DeviceBuffer>,
    pub qwen4_moe_down_i4_scratch_bytes: usize,
    /// Sealed expert-slab route: the decode lane-union descriptor words
    /// (`Gpu::qwen4_slab_union_plan`) and the two codec-masked tile-id arrays
    /// of the K2 prefill. Fixed or grows-never-shrinks; reserved before capture
    /// by `Gpu::reserve_qwen4_slab`.
    pub qwen4_slab_union_scratch: Option<DeviceBuffer>,
    pub qwen4_slab_union_scratch_bytes: usize,
    pub qwen4_slab_tiles_scratch: Option<DeviceBuffer>,
    pub qwen4_slab_tiles_scratch_bytes: usize,
    pub qwen4_moe_down_i4_generation: u64,
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
    /// Dedicated FP8 verify activation bytes (`n*k` E4M3, [N,K] row-major) of
    /// the PeaceMaker FP8 verify family. Never pointer-cached: every
    /// `fp8_verify_pack_x` repacks. Grown only through
    /// [`Self::pack_fp8_verify_rows`] behind `scratch_will_grow`.
    pub fp8_verify_x8_scratch: Option<DeviceBuffer>,
    pub fp8_verify_x8_scratch_bytes: usize,
    /// Per-row activation scales (`n` f32) paired with `fp8_verify_x8_scratch`.
    pub fp8_verify_scales_scratch: Option<DeviceBuffer>,
    pub fp8_verify_scales_scratch_bytes: usize,
    /// One reusable F2 weight workspace: Wf followed by Rw and Ew. Each
    /// repack overwrites all valid planes before its same-stream GEMM reads them.
    pub fp8_f2_weights: Option<DeviceBuffer>,
    pub fp8_f2_weights_bytes: usize,
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
    /// Planned-route module barrier (`Gpu::begin_route_kernel_load`). Lives
    /// here so the scratch helpers' `compile_and_load_kernel` calls drain it
    /// too; `Idle` keeps today's lazy loading exactly.
    pub(crate) route_load: RouteLoad,
    /// `FeatureFlags::pm_decode` snapshot for the load funnel: accepted
    /// [`crate::pm_decode_twins`] replace their HIP modules (exact gfx1201).
    pub(crate) pm_decode: bool,
}

// ── Planned-route kernel barrier ─────────────────────────────────────────

/// Counters of one begun route's kernel preload, for the loader's log. A
/// module counts as compiled when its object is in the kernel cache before
/// ready; `lazy_during_load` counts kernels the route did not cover that a
/// dispatch requested between begin and ready.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteKernelStats {
    /// Predicted-family modules requested for compilation.
    pub predicted_modules: usize,
    /// Of those, modules with a compiled object before ready.
    pub predicted_compiled: usize,
    /// Modules with at least one symbol bound.
    pub predicted_bound_modules: usize,
    /// Symbols bound from predicted modules (conflict-free subset only).
    pub predicted_bound_symbols: usize,
    /// Unplanned kernels requested while the model was loading.
    pub lazy_during_load: usize,
}

/// Load-time kernel state of a planner-closed route.
///
/// `Pending` holds the frozen plan while its missing HIP modules compile on
/// CPU workers; nothing of it is on the GPU yet. The first kernel request or
/// launch drains it ([`prepare_route_modules`]) into `Prepared`: every
/// planned HIP and embedded image loaded and every planned symbol bound.
/// After a successful model load a route without an `open` label is
/// `Sealed`: an unplanned module request then fails instead of compiling or
/// loading between dispatches. A route with one (walked plans, and every
/// predicted-family fallback) is `Open`: an unplanned kernel still loads
/// lazily, with one warning line naming it and the route, and is counted.
#[derive(Default)]
pub(crate) enum RouteLoad {
    #[default]
    Idle,
    Pending {
        plan: crate::kernel_registry::RouteKernelPlan,
        batch: Option<crate::compiler::KernelBatch>,
        /// Predicted-family modules compiled in the same batch as `plan`
        /// and bound after it. Their compile or bind failure fails the load
        /// exactly like a planned module's. Empty for a plan-only route; only
        /// an `open` route has any.
        predicted: Vec<crate::kernel_registry::PredictedModule>,
        open: Option<String>,
    },
    Prepared {
        open: Option<String>,
        stats: RouteKernelStats,
    },
    Sealed,
    Open {
        route: String,
        unplanned: usize,
        stats: RouteKernelStats,
    },
}

/// What an unplanned kernel request does, so the warning never claims a
/// compile that did not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LazyKind {
    /// A native embedded image: loaded and bound, never compiled.
    EmbeddedImage,
    /// A HIP module whose object the compiler already holds from before the
    /// request (planned/predicted compile, or an earlier load): bound only.
    HipPrecompiled,
    /// A HIP module with no pre-request module record under this name. It
    /// compiles only if its cache entry is missing; a module compiled
    /// earlier under a compiler-family alias name only loads and binds.
    HipNoRecord,
}

impl LazyKind {
    fn what(self) -> &'static str {
        match self {
            Self::EmbeddedImage => "native image loaded lazily (embedded; no compile)",
            Self::HipPrecompiled => "bound lazily (module already compiled)",
            Self::HipNoRecord => {
                "loaded lazily (no pre-ready module record; compiled now only if not in the kernel cache)"
            }
        }
    }
}

impl RouteLoad {
    /// Planned modules are not yet loaded: raw launches must refuse.
    pub(crate) fn pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }

    /// Gate a kernel request the preloaded plan did not bind: a sealed route
    /// refuses it, an open route counts it and warns once per request with
    /// the module, route, phase (during load / after ready) and what the
    /// request does ([`LazyKind`]), and any other state loads it silently.
    /// The counters count unplanned requests, not compiles: a bind of an
    /// already compiled or embedded module counts like any other miss.
    pub(crate) fn admit_unplanned(
        &mut self,
        module_name: &str,
        func_name: &str,
        kind: LazyKind,
    ) -> HipResult<()> {
        let what = kind.what();
        match self {
            Self::Sealed => Err(unplanned_route_kernel(module_name, func_name)),
            Self::Open {
                route, unplanned, ..
            } => {
                *unplanned += 1;
                eprintln!(
                    "  kernels: WARNING unplanned kernel {func_name} (module {module_name}) {what} after \
                     ready on the {route} route plan [{unplanned} unplanned after ready]"
                );
                Ok(())
            }
            Self::Prepared { open, stats } => {
                stats.lazy_during_load += 1;
                if let Some(route) = open {
                    eprintln!(
                        "  kernels: WARNING unplanned kernel {func_name} (module {module_name}) {what} during \
                         load on the {route} route plan [{} unplanned during load]",
                        stats.lazy_during_load
                    );
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// A begun route. Predicted-family modules are only ever speculative
    /// coverage, never a closure, so they require an `open` label: a
    /// fallback route can not seal.
    pub(crate) fn pending_route(
        plan: crate::kernel_registry::RouteKernelPlan,
        batch: Option<crate::compiler::KernelBatch>,
        predicted: Vec<crate::kernel_registry::PredictedModule>,
        open: Option<String>,
    ) -> HipResult<Self> {
        if !predicted.is_empty() && open.is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                "predicted kernels need an open route (a predicted fallback can never be sealed)",
            ));
        }
        Ok(Self::Pending {
            plan,
            batch,
            predicted,
            open,
        })
    }

    /// The state a drained route takes once its model loaded: `Open` (counting
    /// from zero) when it has an `open` label, else `Sealed`.
    pub(crate) fn finished(self) -> Self {
        match self {
            Self::Prepared {
                open: Some(route),
                stats,
            } => Self::Open {
                route,
                unplanned: 0,
                stats,
            },
            _ => Self::Sealed,
        }
    }

    /// Complete a drained route as `Open` under `route` whatever its plan's
    /// label was: a route whose model left the closed plan keeps every
    /// predictable module compiled and bound and its counters, and can no
    /// longer seal. Any other state is unchanged.
    pub(crate) fn finished_open(self, route: String) -> Self {
        match self {
            Self::Prepared { stats, .. } => Self::Open {
                route,
                unplanned: 0,
                stats,
            },
            other => other,
        }
    }

    /// Counters of the begun route (zero when none was begun).
    pub(crate) fn stats(&self) -> RouteKernelStats {
        match self {
            Self::Prepared { stats, .. } | Self::Open { stats, .. } => *stats,
            Self::Pending { predicted, .. } => RouteKernelStats {
                predicted_modules: predicted.len(),
                ..RouteKernelStats::default()
            },
            _ => RouteKernelStats::default(),
        }
    }
}

/// Drain a pending route plan: join the whole compile batch, then load every
/// planned HIP and embedded module and bind every planned symbol to it. All
/// on the GPU owner thread, before the first dispatch. Planned bindings
/// replace any same-named function cached by an earlier load, so one entry
/// name resolves to the one image this plan selected. Predicted-family
/// modules then bind only their proven conflict-free symbols and never
/// replace an existing binding. A compile or bind failure of a planned or a
/// predicted module is a load error: a known module is predictable work, not
/// a lazy miss. Idempotent; a no-op unless `Pending`. On error the route
/// returns to `Idle` (the caller's load fails and unwinds).
pub(crate) fn prepare_route_modules(
    route: &mut RouteLoad,
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    pm_decode: bool,
) -> HipResult<()> {
    if !route.pending() {
        return Ok(());
    }
    let RouteLoad::Pending {
        plan,
        batch,
        predicted,
        open,
    } = std::mem::take(route)
    else {
        unreachable!("checked pending");
    };
    if let Some(batch) = batch {
        compiler.finish_batch(batch)?;
    }
    for entry in &plan.entries {
        match entry {
            crate::kernel_registry::PlannedKernel::Hip(entry) => {
                for &symbol in entry.symbols {
                    load_kernel_module(
                        compiler,
                        hip,
                        modules,
                        functions,
                        pm_decode,
                        entry.module,
                        entry.source(),
                        symbol,
                    )?;
                }
            }
            crate::kernel_registry::PlannedKernel::Embedded {
                module,
                image,
                radiowave_json,
                symbols,
            } => {
                for &symbol in *symbols {
                    load_embedded_module(
                        compiler,
                        hip,
                        modules,
                        functions,
                        module,
                        image,
                        *radiowave_json,
                        symbol,
                    )?;
                }
            }
        }
    }
    let stats = bind_predicted_modules(compiler, hip, modules, functions, pm_decode, &predicted)?;
    *route = RouteLoad::Prepared { open, stats };
    Ok(())
}

/// Bind the proven conflict-free symbols of each predicted module. Every
/// predicted module must have a compiled object after the batch joined; a
/// symbol already in the function cache keeps its binding.
fn bind_predicted_modules(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    pm_decode: bool,
    predicted: &[crate::kernel_registry::PredictedModule],
) -> HipResult<RouteKernelStats> {
    let mut stats = RouteKernelStats {
        predicted_modules: predicted.len(),
        ..RouteKernelStats::default()
    };
    for module in predicted {
        let name = module.entry.module;
        if !compiler.compiled_kernels().contains_key(name) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "predicted kernel module {name} has no compiled object after the route compile"
                ),
            ));
        }
        stats.predicted_compiled += 1;
        let mut bound = 0;
        for &symbol in &module.bind {
            if functions.contains_key(symbol) {
                continue;
            }
            load_kernel_module(
                compiler,
                hip,
                modules,
                functions,
                pm_decode,
                name,
                module.entry.source(),
                symbol,
            )?;
            bound += 1;
        }
        if bound > 0 {
            stats.predicted_bound_modules += 1;
            stats.predicted_bound_symbols += bound;
        }
    }
    Ok(stats)
}

// ── Shared kernel dispatch helpers ──────────────────────────────────────

/// Compile and load a kernel, caching the result in `modules`/`functions`.
///
/// A pending planned route is drained first, before the cached-function
/// early return, so no request can run ahead of the whole-plan barrier. A
/// sealed route refuses an unplanned module instead of compiling or loading
/// it between dispatches; an open one loads it with a warning
/// ([`RouteLoad::admit_unplanned`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_and_load_kernel(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    route: &mut RouteLoad,
    pm_decode: bool,
    module_name: &str,
    source: &str,
    func_name: &str,
) -> HipResult<()> {
    prepare_route_modules(route, compiler, hip, modules, functions, pm_decode)?;
    if functions.contains_key(func_name) {
        return Ok(());
    }
    route.admit_unplanned(
        module_name,
        func_name,
        if compiler.compiled_kernels().contains_key(module_name) {
            LazyKind::HipPrecompiled
        } else {
            LazyKind::HipNoRecord
        },
    )?;
    load_kernel_module(
        compiler,
        hip,
        modules,
        functions,
        pm_decode,
        module_name,
        source,
        func_name,
    )
}

/// Error for a kernel outside a sealed route plan.
pub(crate) fn unplanned_route_kernel(module_name: &str, func_name: &str) -> hip_bridge::HipError {
    hip_bridge::HipError::new(
        0,
        &format!(
            "kernel {func_name:?} (module {module_name:?}) is outside this load's planned kernel \
             route; refusing a post-ready compile/module load"
        ),
    )
}

/// Resolve, load and bind one symbol, always (re)binding `func_name`. With
/// `pm_decode` on exact gfx1201 an accepted [`crate::pm_decode_twins`] module
/// loads its embedded image instead: verified once at module admission
/// (fail closed), and hipcc is never invoked for it.
#[allow(clippy::too_many_arguments)]
fn load_kernel_module(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    pm_decode: bool,
    module_name: &str,
    source: &str,
    func_name: &str,
) -> HipResult<()> {
    if let Some(twin) =
        crate::pm_decode_twins::pm_decode_module(compiler.arch(), pm_decode, module_name)
    {
        if !modules.contains_key(module_name) {
            twin.verify(source, func_name)
                .map_err(|reason| hip_bridge::HipError::new(0, &reason))?;
            // `load_embedded_module` re-checks the image digest at admission.
        } else if !twin.symbols.contains(&func_name) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "pm_decode twin {module_name}: symbol {func_name:?} is not an accepted export"
                ),
            ));
        }
        return load_embedded_module(
            compiler,
            hip,
            modules,
            functions,
            twin.module,
            twin.image,
            Some(twin.radiowave_json),
            func_name,
        );
    }
    let obj_path = compiler.compile_for_symbol(module_name, source, func_name)?;
    let obj_path_str = obj_path.to_str().unwrap().to_string();
    if !modules.contains_key(module_name) {
        let module =
            module_load_or_recompile(hip, compiler, module_name, source, func_name, &obj_path_str)?;
        modules.insert(module_name.to_string(), module);
    }
    let module = &modules[module_name];
    let func = hip.module_get_function(module, func_name).map_err(|error| {
        let context = format!(
            "hipModuleGetFunction failed for symbol {func_name:?} in module {module_name:?}: {error}"
        );
        hip_bridge::HipError::new(error.code, &context)
    })?;
    // Bind the launched function name to its module's admitted image so the
    // recorder resolves it even when the arch-selected module name differs
    // (e.g. gemv_hfq4g256_residual launched from gemv_hfq4g256_residual_rdna3).
    bind_loaded_function(compiler, module_name, func_name)?;
    functions.insert(func_name.to_string(), func);
    Ok(())
}

/// Load an embedded native image as `module_name` (once, admitted as that
/// module's in-memory code object; no file is written), then resolve and
/// (re)bind `func_name` to it. A module already bound to a different image
/// is refused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_embedded_module(
    compiler: &mut crate::compiler::KernelCompiler,
    hip: &HipRuntime,
    modules: &mut HashMap<String, Module>,
    functions: &mut HashMap<String, Function>,
    module_name: &'static str,
    image: &'static [u8],
    radiowave_json: Option<&str>,
    func_name: &str,
) -> HipResult<()> {
    if !modules.contains_key(module_name) {
        // An accepted pm_decode twin is re-verified against its pinned digest
        // at admission, whichever path (lazy funnel or route preload) loads it.
        let twin = crate::pm_decode_twins::GFX1201_TWINS
            .iter()
            .find(|t| t.image.as_ptr() == image.as_ptr());
        if let Some(twin) = twin {
            twin.verify_image()
                .map_err(|reason| hip_bridge::HipError::new(0, &reason))?;
        }
        if hipfire_config::developer_var("HIPFIRE_MODULE_LOAD_AUDIT").as_deref() == Ok("1") {
            // Developer-only route-closure audit: every embedded image load,
            // with the accepted pm_decode identities when it is a twin.
            eprintln!(
                "HIPFIRE_MODULE_LOAD_AUDIT\tembedded\t{}",
                serde_json::json!({
                    "arch": compiler.arch(),
                    "module": module_name,
                    "symbol": func_name,
                    "origin": "native_embedded",
                    "image_sha256": crate::code_object::CodeObjectId::of(image).to_hex(),
                    "pm_decode_accepted_elf_sha256": twin.map(|t| crate::pm_decode_twins::hex(&t.accepted_elf_sha256)),
                    "pm_decode_source_sha256": twin.map(|t| crate::pm_decode_twins::hex(&t.source_sha256)),
                })
            );
        }
        let module = hip.module_load_data(image)?;
        qsa_code_check_register(compiler.arch(), &module, module_name, image);
        compiler
            .code_objects_mut()
            .admit_module(
                module_name,
                crate::code_object::CodeObjectArtifact::native_embedded(
                    module_name,
                    image.into(),
                    radiowave_json,
                ),
            )
            .map_err(|reason| {
                qsa_code_check_unregister(&module);
                hip_bridge::HipError::new(0, &reason)
            })?;
        modules.insert(module_name.to_owned(), module);
    } else if let Some(bound) = compiler.code_objects().module(module_name) {
        if bound.id() != crate::code_object::CodeObjectId::of(image) {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("module {module_name:?} is bound to image {} and a different image was requested", bound.id()),
            ));
        }
    }
    let func = hip.module_get_function(&modules[module_name], func_name)?;
    bind_loaded_function(compiler, module_name, func_name)?;
    functions.insert(func_name.to_owned(), func);
    Ok(())
}

/// Bind `func_name` to the artifact already admitted for `module_name`.
pub(crate) fn bind_loaded_function(
    compiler: &mut crate::compiler::KernelCompiler,
    module_name: &str,
    func_name: &str,
) -> HipResult<()> {
    let registry = compiler.code_objects_mut();
    let artifact = registry.module(module_name).cloned().ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!("module {module_name:?} has no admitted code object"),
        )
    })?;
    registry
        .bind_function(func_name, &artifact)
        .map_err(|reason| hip_bridge::HipError::new(0, &reason))
}

/// Process-wide `qwen4_qsa_code_check` switch, set once at `Gpu` init from the
/// parsed feature flag. Default off: every site below is one relaxed load.
static QSA_CODE_CHECK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static QSA_CODE_CHECK_HOOK: std::sync::Once = std::sync::Once::new();

/// Enable the loaded-code check for this process when `flags` asks for it and
/// install the bridge's first-illegal-address capture hook exactly once. Must
/// run before the first module load. A disabled flag touches nothing.
pub(crate) fn enable_qsa_code_check(flags: &crate::FeatureFlags) {
    if !flags.qwen4_qsa_code_check {
        return;
    }
    QSA_CODE_CHECK.store(true, std::sync::atomic::Ordering::Relaxed);
    QSA_CODE_CHECK_HOOK.call_once(|| {
        hip_bridge::register_first_illegal_address_hook(|| {
            crate::loaded_code_check::capture_after_700();
        });
    });
}

#[inline]
fn qsa_code_check_enabled() -> bool {
    QSA_CODE_CHECK.load(std::sync::atomic::Ordering::Relaxed)
}

/// The device code object HIP loads from `image`: a raw ELF as is, or the
/// `arch` entry (`hipv4-amdgcn-amd-amdhsa--<arch>[:features]`) of an
/// uncompressed clang offload bundle. `None` for anything else (compressed
/// bundle, missing arch entry, malformed header).
pub(crate) fn device_code_object<'a>(image: &'a [u8], arch: &str) -> Option<&'a [u8]> {
    const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
    if image.starts_with(&ELF_MAGIC) {
        return Some(image);
    }
    if !image.starts_with(BUNDLE_MAGIC) {
        return None;
    }
    let u64_at = |at: usize| -> Option<usize> {
        let bytes: [u8; 8] = image.get(at..at.checked_add(8)?)?.try_into().ok()?;
        usize::try_from(u64::from_le_bytes(bytes)).ok()
    };
    let wanted = format!("hipv4-amdgcn-amd-amdhsa--{arch}");
    let mut cursor = BUNDLE_MAGIC.len() + 8;
    let mut found: Option<&[u8]> = None;
    for _ in 0..u64_at(BUNDLE_MAGIC.len())? {
        let (offset, size, len) = (u64_at(cursor)?, u64_at(cursor + 8)?, u64_at(cursor + 16)?);
        let triple = image.get(cursor + 24..cursor.checked_add(24)?.checked_add(len)?)?;
        cursor += 24 + len;
        let matches = triple == wanted.as_bytes()
            || (triple.starts_with(wanted.as_bytes()) && triple[wanted.len()] == b':');
        if matches {
            let payload = image.get(offset..offset.checked_add(size)?)?;
            if found.replace(payload).is_some() {
                return None;
            }
        }
    }
    found.filter(|elf| elf.starts_with(&ELF_MAGIC))
}

/// Register a just-loaded module with the loaded-code registry (flag on only).
/// `image` is the exact byte slice passed to `hipModuleLoadData`; the caller
/// invokes this immediately after that call succeeds, at every load site in
/// this crate. The registry receives the `arch` device ELF extracted from a
/// bundle; when no ELF can be extracted (compressed or malformed bundle,
/// missing arch entry) it receives the original image unchanged so it records
/// the module as unavailable rather than dropping it.
pub(crate) fn qsa_code_check_register(arch: &str, module: &Module, name: &str, image: &[u8]) {
    if !qsa_code_check_enabled() {
        return;
    }
    let image = device_code_object(image, arch).unwrap_or(image);
    crate::loaded_code_check::register_module(module.raw_ptr() as usize, name, image);
}

/// Unregister one module handle (flag on only), for a load whose admission
/// failed after it was registered.
pub(crate) fn qsa_code_check_unregister(module: &Module) {
    if !qsa_code_check_enabled() {
        return;
    }
    crate::loaded_code_check::unregister_module(module.raw_ptr() as usize);
}

/// Unregister every cached module handle (flag on only). `Gpu` teardown calls
/// this before the runtime and module map drop. HIP modules are never
/// unloaded in this crate, so this is the only lifetime event.
pub(crate) fn qsa_code_check_unregister_all(modules: &HashMap<String, Module>) {
    if !qsa_code_check_enabled() {
        return;
    }
    for module in modules.values() {
        crate::loaded_code_check::unregister_module(module.raw_ptr() as usize);
    }
}

/// Load a compiled module from its immutable image, self-healing a
/// stale/invalid cached image. The cache file is read once; HIP loads those
/// exact bytes and the same bytes are admitted as the module's code object.
/// If HIP rejects the image as invalid (`HIP_ERROR_INVALID_IMAGE`) — e.g. a
/// cross-build blob left in a shared `.hipfire_kernels` cache — evict it,
/// recompile from source, and retry once. Any other error propagates
/// unchanged. (Fix for the bench/run "device kernel image is invalid" crash
/// when two daemon builds share a cwd kernel cache.)
pub(crate) fn module_load_or_recompile(
    hip: &HipRuntime,
    compiler: &mut crate::compiler::KernelCompiler,
    module_name: &str,
    source: &str,
    symbol: &str,
    obj_path: &str,
) -> HipResult<Module> {
    if hipfire_config::developer_var("HIPFIRE_MODULE_LOAD_AUDIT").as_deref() == Ok("1") {
        // Developer-only route-closure audit (default off): source/recipe
        // identity of every HIP module, whichever loader reached it; the
        // hip-bridge lines that follow carry the handle and resolved symbols.
        use sha2::{Digest, Sha256};
        let recipe = crate::compiler::KernelCompiler::recipe_for_source(
            compiler.arch(),
            module_name,
            source,
            &compiler.extra_flags,
        );
        let digest = Sha256::digest(source.as_bytes());
        let source_sha256: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        eprintln!(
            "HIPFIRE_MODULE_LOAD_AUDIT\thip\t{}",
            serde_json::json!({
                "arch": compiler.arch(),
                "module": module_name,
                "symbol": symbol,
                "source_sha256": source_sha256,
                "flags": recipe.flags,
                "scheduler_profile": recipe.scheduler_profile,
                "object": obj_path,
            })
        );
    }
    let read = |path: &std::path::Path| -> HipResult<std::sync::Arc<[u8]>> {
        std::fs::read(path).map(Into::into).map_err(|error| {
            hip_bridge::HipError::new(0, &format!("read {}: {error}", path.display()))
        })
    };
    let mut path = std::path::PathBuf::from(obj_path);
    let mut image = read(&path)?;
    let module = match hip.module_load_data(&image) {
        Ok(m) => m,
        Err(e) if e.code == HIP_ERROR_INVALID_IMAGE => {
            eprintln!(
                "  {module_name}: cached kernel image invalid (HIP {}); recompiling from source",
                e.code
            );
            path = compiler.recompile(module_name, source)?;
            image = read(&path)?;
            hip.module_load_data(&image)?
        }
        Err(e) => return Err(e),
    };
    qsa_code_check_register(compiler.arch(), &module, module_name, &image);
    compiler
        .code_objects_mut()
        .admit_module(
            module_name,
            crate::code_object::CodeObjectArtifact::hip_file(&path, image),
        )
        .map_err(|reason| {
            qsa_code_check_unregister(&module);
            hip_bridge::HipError::new(0, &reason)
        })?;
    Ok(module)
}

/// Launch a kernel, routing through the blob path when graph capture, replay
/// recording, a G0 observation, or force_blob is active. Shared between
/// `Gpu::launch_maybe_blob` and `ScratchState` methods so the branching logic
/// stays in one place.
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
    // Fail closed before the recorder / capture tapes see the launch.
    hip.validate_launch_grid(grid)
        .map_err(|e| e.with_kernel(func_name))?;
    let record = replay.as_ref().map_or(false, |r| r.is_recording());
    let observe = replay.as_ref().is_some_and(|r| r.is_g0_observing());
    let result: HipResult<()> = if record
        || observe
        || capture_mode
        || force_blob_path
        || hip.plan_active()
    {
        let mut blob = blob_builder();
        blob.pad_to(16);
        if observe {
            replay.as_mut().unwrap().observe_g0_launch(
                func_name,
                grid,
                block,
                shared_mem,
                blob.as_bytes(),
            );
        }
        if record {
            // Single decision point for how a launch is recorded: the artifact
            // alias table and the record shape are shared with
            // `Gpu::launch_maybe_blob_bound`.
            let artifact = compiler
                .as_ref()
                .and_then(|c| crate::dispatch::recorded_launch_artifact(c, func_name));
            replay.as_mut().unwrap().record_hip_launch_typed_bound(
                hip,
                func_name,
                artifact,
                grid,
                block,
                shared_mem,
                blob.as_bytes(),
                None,
                &[],
                compiler,
                &[],
            );
        }
        if capture_mode {
            capture_blobs.push(blob.into_vec());
            let buf = capture_blobs.last_mut().unwrap();
            let func = &functions[func_name];
            unsafe {
                hip.launch_kernel_blob(func, grid, block, shared_mem, stream, buf.as_mut_slice())
            }
        } else {
            let mut bytes = blob.into_vec();
            let func = &functions[func_name];
            unsafe {
                hip.launch_kernel_blob(func, grid, block, shared_mem, stream, bytes.as_mut_slice())
            }
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

/// Fold a `(blocks_per_row, rows)` launch into a one-dimensional grid.
///
/// Row/token kernels that used to read the row from `blockIdx.y` decode it as
/// `blockIdx.x / blocks_per_row` (and the in-row block as
/// `blockIdx.x % blocks_per_row`), so the row count is not bound by the
/// device `gridDim.y` limit (65535 on gfx1201). Only overflow of the flat
/// count is an error; a row is never dropped or clamped.
pub(crate) fn fold_rows_into_x(blocks_per_row: usize, rows: usize) -> HipResult<[u32; 3]> {
    match blocks_per_row.checked_mul(rows) {
        Some(n) if n <= i32::MAX as usize => Ok([n as u32, 1, 1]),
        _ => Err(hip_bridge::HipError::new(
            1,
            &format!(
                "folded launch grid {blocks_per_row} blocks/row x {rows} rows exceeds the \
                 one-dimensional grid.x limit {}; refusing to truncate rows",
                i32::MAX
            ),
        )),
    }
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
/// Bumped every time a scratch slot's old buffer is freed (growth replaces
/// it) or a growth is announced by [`Gpu::invalidate_for_scratch_growth`].
/// Anything that caches device pointers into scratch slots (the certified
/// railgun window's prepared skeletons) must treat a changed epoch as "every
/// pointer I hold into scratch may be freed": a captured hipGraph and the
/// retained Redline route are invalidated or re-resolved the same way, but a
/// railgun skeleton's bound images would otherwise keep pointing at the freed
/// buffer (whose address range is then reused, e.g. by the retained kernarg
/// pool) and write through it.
static SCRATCH_GROWTH_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The current scratch growth epoch (see [`SCRATCH_GROWTH_EPOCH`]).
pub fn scratch_growth_epoch() -> u64 {
    SCRATCH_GROWTH_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

pub(crate) fn note_scratch_growth() {
    SCRATCH_GROWTH_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

fn grow_scratch_buffer(
    hip: &HipRuntime,
    slot: &mut Option<DeviceBuffer>,
    have_bytes: &mut usize,
    needed: usize,
) -> HipResult<()> {
    if !scratch_will_grow(*have_bytes, slot.is_some(), needed) {
        return Ok(());
    }
    if slot.is_some() {
        note_scratch_growth();
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

/// [`grow_scratch_buffer`] for slots owned outside this module's helpers.
/// Words of the sealed expert-slab lane-union descriptor of `lanes` decode
/// lanes (`S = 10 * lanes` slots): two buckets of `S` expert ids, `S + 1`
/// offsets and `S` slot ids.
pub const fn qwen4_slab_union_words(lanes: usize) -> usize {
    match qwen4_slab_union_words_checked(lanes) {
        Some(words) => words,
        None => panic!("qwen4_slab_union_words: lane count overflows"),
    }
}

/// [`qwen4_slab_union_words`] with checked arithmetic; `None` on overflow.
pub const fn qwen4_slab_union_words_checked(lanes: usize) -> Option<usize> {
    let slots = match lanes.checked_mul(10) {
        Some(slots) => slots,
        None => return None,
    };
    match slots.checked_mul(6) {
        Some(words) => words.checked_add(2),
        None => None,
    }
}

/// Words of the native eight-row slab descriptors: `ceil(lanes / 8) * 482`.
/// Each descriptor reserves all 80 slots, including an incomplete tail tile.
/// Zero lanes need no descriptors; `None` on overflow.
pub const fn qwen4_slab_rowtile_words_checked(lanes: usize) -> Option<usize> {
    let tiles = match (lanes / 8).checked_add(if lanes % 8 == 0 { 0 } else { 1 }) {
        Some(tiles) => tiles,
        None => return None,
    };
    tiles.checked_mul(482)
}

pub(crate) fn grow_scratch_slot(
    hip: &HipRuntime,
    slot: &mut Option<DeviceBuffer>,
    have_bytes: &mut usize,
    needed: usize,
) -> HipResult<()> {
    grow_scratch_buffer(hip, slot, have_bytes, needed)
}

/// True when [`grow_scratch_buffer`] would replace the slot: the early-return
/// condition factored out so `Gpu` callers can invalidate captured graphs
/// BEFORE delegating. Single source of truth — the two sites cannot drift.
///
/// `slot_is_some` matters: a never-allocated slot with a stale non-zero byte
/// count must still allocate.
#[inline]
pub(crate) fn scratch_will_grow(have_bytes: usize, slot_is_some: bool, needed: usize) -> bool {
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

/// Bytes a stable-VA workspace (`ScratchState::qsa_gather_f16`) whose
/// mapped prefix is `mapped` bytes must map next so the prefix covers
/// `needed`: `needed` rounded up to the driver `granularity`, less `mapped`
/// (always a granularity multiple); 0 when already covered. `None` when
/// `needed` exceeds the `reserved` logical extent or `granularity` is 0.
#[inline]
pub(crate) fn vmm_prefix_growth(
    mapped: usize,
    reserved: usize,
    granularity: usize,
    needed: usize,
) -> Option<usize> {
    if granularity == 0 || needed > reserved {
        return None;
    }
    if needed <= mapped {
        return Some(0);
    }
    needed
        .checked_next_multiple_of(granularity)?
        .checked_sub(mapped)
}

/// What [`crate::Gpu::qsa_gather_reserve`] does for a VMM gather workspace
/// asking for `requested` bytes when `reserved` bytes of virtual extent are
/// held (`None`: nothing reserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QsaGatherReserve {
    /// The held reservation already covers the request; nothing is touched.
    Keep,
    /// Nothing is held: reserve a fresh VA.
    Reserve,
    /// A smaller reservation is held. Its pages are never unmapped and
    /// remapped in place (gfx1100: a remapped VA keeps translating to the old
    /// backing): the old arena is released (VA retired, never reused) and a
    /// fresh VA is reserved.
    ReplaceRetiring,
}

/// See [`QsaGatherReserve`]. A reservation is never shrunk and never grown in
/// place.
pub(crate) fn qsa_gather_reserve_action(
    reserved: Option<usize>,
    requested: usize,
) -> QsaGatherReserve {
    match reserved {
        Some(held) if held >= requested => QsaGatherReserve::Keep,
        Some(_) => QsaGatherReserve::ReplaceRetiring,
        None => QsaGatherReserve::Reserve,
    }
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

/// Checked byte size of the IU4 sidecar of a producer reservation `(k, rows)`:
/// the extent `reserve_int4_mmq` / `reserve_int4_mmq_down` allocate (one 72 B
/// `block_i4_128` per [K/128 block, row]). Applies the allocators' admission
/// (`k % 256 == 0`, `k > 0`, `rows > 0`); `None` for a rejected shape or on
/// overflow.
pub fn int4_mmq_reservation_bytes(k: usize, rows: usize) -> Option<usize> {
    if k == 0 || rows == 0 || k % 256 != 0 {
        return None;
    }
    int4_mmq_blocks_bytes(k, rows)
}

/// [`int4_mmq_reservation_bytes`] for the Qwen4 symmetric MoE down sidecar,
/// whose allocator (`reserve_qwen4_moe_down_i4`) admits `k % 128 == 0`.
pub fn qwen4_moe_down_i4_bytes(k: usize, rows: usize) -> Option<usize> {
    if k == 0 || rows == 0 || k % 128 != 0 {
        return None;
    }
    int4_mmq_blocks_bytes(k, rows)
}

/// `k / 128 * rows * 72` with checked arithmetic (same value as
/// [`int4_mmq_reserve_needed`] for admitted shapes).
fn int4_mmq_blocks_bytes(k: usize, rows: usize) -> Option<usize> {
    (k / 128).checked_mul(rows)?.checked_mul(72)
}

/// Byte offsets of the stable-grouping scratch for `slots` routed slots over
/// `experts` experts: `[slots]` ranks, `[experts, chunks]` counts, then
/// `[experts, chunks + 1]` chunk prefixes, `chunks = ceil(slots / 256)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4MoeGroupLayout {
    pub chunks: usize,
    pub partial_offset: usize,
    pub prefix_offset: usize,
    pub bytes: usize,
}

pub fn qwen4_moe_group_layout(slots: usize, experts: usize) -> Qwen4MoeGroupLayout {
    qwen4_moe_group_layout_checked(slots, experts)
        .expect("qwen4_moe_group_layout: slot/expert counts overflow the layout")
}

/// [`qwen4_moe_group_layout`] with checked arithmetic; `None` on `usize` overflow.
pub fn qwen4_moe_group_layout_checked(slots: usize, experts: usize) -> Option<Qwen4MoeGroupLayout> {
    let chunks = slots.div_ceil(256);
    let partial_offset = slots.checked_mul(4)?;
    let prefix_offset = partial_offset.checked_add(experts.checked_mul(chunks)?.checked_mul(4)?)?;
    let bytes = prefix_offset.checked_add(
        experts
            .checked_mul(chunks.checked_add(1)?)?
            .checked_mul(4)?,
    )?;
    Some(Qwen4MoeGroupLayout {
        chunks,
        partial_offset,
        prefix_offset,
        bytes,
    })
}

/// Upper bound on the grouped (per-expert 16-row padded) row count of `slots`
/// routed slots over `experts` experts: every slot, plus at most 15 pad rows
/// for each non-empty expert (`min(slots, experts)` of them), rounded up to a
/// 16-row tile. `None` on overflow.
pub fn qwen4_grouped_rows_bound(slots: usize, experts: usize) -> Option<usize> {
    let padded = slots
        .checked_add(slots.min(experts).checked_mul(15)?)?
        .checked_add(15)?;
    Some(padded / 16 * 16)
}

/// Bytes of the Qwen4 slab route's two codec-masked tile-id arrays for
/// `tiles` grouped tiles (`2 * tiles` u32), the extent `reserve_qwen4_slab`
/// allocates. `None` on overflow.
pub fn qwen4_slab_tiles_bytes(tiles: usize) -> Option<usize> {
    tiles.checked_mul(2)?.checked_mul(4)
}

/// Bytes of the slab lane-union descriptor for `lanes` decode lanes
/// ([`qwen4_slab_union_words_checked`] words of u32). `None` on overflow.
pub fn qwen4_slab_union_bytes(lanes: usize) -> Option<usize> {
    qwen4_slab_union_words_checked(lanes)?.checked_mul(4)
}

/// Bytes of the native eight-row slab descriptors; zero for zero lanes and
/// `None` on overflow. Each physical tile occupies 1928 bytes.
pub fn qwen4_slab_rowtile_bytes(lanes: usize) -> Option<usize> {
    qwen4_slab_rowtile_words_checked(lanes)?.checked_mul(4)
}

/// Shared descriptor allocation required by both the legacy lane union and
/// native row tiles. Both representations must fit; `None` on either overflow.
/// Reservation, replay-growth checks and admission must use this same extent.
pub fn qwen4_slab_descriptor_bytes(lanes: usize) -> Option<usize> {
    let union = qwen4_slab_union_bytes(lanes)?;
    let rowtiles = qwen4_slab_rowtile_bytes(lanes)?;
    Some(union.max(rowtiles))
}

/// Bytes of the FP16 X scratch for `n_elems` F16 elements; `None` on overflow.
pub fn fp16_x_bytes(n_elems: usize) -> Option<usize> {
    n_elems.checked_mul(2)
}

/// FWHT sign-table lengths (f32 elements each, two tables per group size).
pub const MQ_FWHT_SIGNS_G256: usize = 256;
pub const MQ_FWHT_SIGNS_G128: usize = 128;
/// Shared F32 rotation scratch elements (covers K up to 32768).
pub const MQ_X_ROT_ELEMS: usize = 32768;
/// INT8 dp4a activation scratch bytes.
pub const MQ_X_Q8_BYTES: usize = 32768;
/// Per-group F32 activation scales (up to 128 groups).
pub const MQ_X_SCALES_BYTES: usize = 128 * 4;
/// The fixed MagnumQuant allocations once both `ensure_mq_signs` and
/// `ensure_mq_signs_128` have run: two G256 and two G128 sign tables, the
/// shared rotation scratch, the INT8 scratch and the scales.
pub const MQ_FIXED_WORKSPACE_BYTES: usize =
    (2 * MQ_FWHT_SIGNS_G256 + 2 * MQ_FWHT_SIGNS_G128 + MQ_X_ROT_ELEMS) * std::mem::size_of::<f32>()
        + MQ_X_Q8_BYTES
        + MQ_X_SCALES_BYTES;

/// Whether the exact-gfx1201 dense GDN scan (`fn_gdn_dense`) admits a GDN state stored in
/// `format`: plain Q8 and live-EF Q8Ef; never F32 and never another architecture (gfx942 is
/// F32-only). Geometry, row floor and flags are the route's own checks.
pub fn qwen4_dense_gdn_admits(arch: &str, format: crate::tensor_ops::GdnStateFormat) -> bool {
    arch == "gfx1201"
        && matches!(
            format,
            crate::tensor_ops::GdnStateFormat::Q8 | crate::tensor_ops::GdnStateFormat::Q8Ef
        )
}

/// Model dimensions the Qwen4 prefill workspace census reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4PrefillWorkspaceDims {
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    /// GDN state format whose dense-scan scratch is reachable (exact gfx1201 route,
    /// [`qwen4_dense_gdn_admits`]); `None` when the route is not. `Q8` and `Q8Ef` charge the A
    /// scratch; only `Q8` also charges the thread-local zero-entry EF scratch (`Q8Ef` scans
    /// against the owner's live residual, which the state owner charges, not this census). Needs
    /// the `deltanet` feature: without it the scratch cannot be sized and `Some` yields `None`.
    pub gdn_format: Option<crate::tensor_ops::GdnStateFormat>,
    /// Include the GDN head-pair counters the short-tail remainder steps
    /// allocate.
    pub include_short_tails: bool,
    /// The routed-expert IU4 path exists (MoE profile whose `hidden_size` and
    /// `moe_intermediate_size` the IU4 allocators admit). The `int4_xq`,
    /// `moe_group`, `down_i4q`, `slab_tiles` and `slab_union` terms are also 0
    /// whenever `num_experts == 0` or `top_k == 0` (no MoE owner reachable).
    /// When they are 0 none of their dimension checks can make the query
    /// return `None`.
    pub routed_i4: bool,
}

/// Reusable prefill workspace terms for `rows` rows, in allocation-census
/// order: `fp16_x`, `ksplit_det`, `int4_xq`, `moe_group`, `down_i4q`,
/// `slab_tiles`, `slab_union`, `mq_fixed`, `gdn_dense_a`, `gdn_dense_ef`,
/// `gdn_pair_counters`. Every term is the owning allocator's size query, not a
/// copy of its formula. Excludes F16 weight shadows, `State::device_bytes`, the
/// QSA selector / gather scratch and stage buffers. `None` on overflow, a
/// dimension an owning allocator rejects, or `gdn_format` without `deltanet`.
pub fn qwen4_prefill_workspace_terms(
    rows: usize,
    dims: &Qwen4PrefillWorkspaceDims,
) -> Option<[(&'static str, u64); 11]> {
    let hc_wide = dims.hc_count.checked_mul(dims.hidden_size)?;
    let fp16_x = fp16_x_bytes(rows.checked_mul(hc_wide.max(dims.hidden_size))?)?;
    let ksplit_det = if hc_wide == 0 || dims.hc_lowrank == 0 {
        0
    } else {
        crate::Gpu::qwen4_prefill_ksplit_bytes(rows, dims.hc_lowrank, hc_wide)?
    };
    let routed = dims.routed_i4 && dims.num_experts != 0 && dims.top_k != 0;
    let (int4_xq, moe_group, down_i4q, slab_tiles) = if routed {
        let slots = rows.checked_mul(dims.top_k)?;
        (
            int4_mmq_reservation_bytes(dims.hidden_size, rows)?,
            qwen4_moe_group_layout_checked(slots, dims.num_experts)?.bytes,
            qwen4_moe_down_i4_bytes(dims.moe_intermediate_size, slots)?,
            qwen4_slab_tiles_bytes(
                qwen4_grouped_rows_bound(slots, dims.num_experts)?.div_ceil(16),
            )?,
        )
    } else {
        (0, 0, 0, 0)
    };
    let slab_union = if routed {
        qwen4_slab_descriptor_bytes(crate::slab_moe::SLAB_MAX_LANES)?
    } else {
        0
    };
    #[cfg(feature = "deltanet")]
    let (gdn_a, gdn_ef) = match dims.gdn_format {
        Some(format) => (
            crate::fn_gdn_dense::workspace_a_bytes_for(format, rows)?,
            crate::fn_gdn_dense::workspace_ef_bytes_for(format, rows),
        ),
        None => (0, 0),
    };
    #[cfg(not(feature = "deltanet"))]
    let (gdn_a, gdn_ef) = if dims.gdn_format.is_some() {
        return None;
    } else {
        (0usize, 0usize)
    };
    let pair_counters = if dims.include_short_tails {
        crate::tensor_ops::GDN_PAIR_COUNTERS_BYTES
    } else {
        0
    };
    let term = |name: &'static str, bytes: usize| Some((name, u64::try_from(bytes).ok()?));
    Some([
        term("fp16_x", fp16_x)?,
        term("ksplit_det", ksplit_det)?,
        term("int4_xq", int4_xq)?,
        term("moe_group", moe_group)?,
        term("down_i4q", down_i4q)?,
        term("slab_tiles", slab_tiles)?,
        term("slab_union", slab_union)?,
        term("mq_fixed", MQ_FIXED_WORKSPACE_BYTES)?,
        term("gdn_dense_a", gdn_a)?,
        term("gdn_dense_ef", gdn_ef)?,
        term("gdn_pair_counters", pair_counters)?,
    ])
}

/// Byte layout of the W8A8 Q8_0 activation sidecar for `(k, n)`:
/// `(scales_offset, total_bytes)`. int8 `[n, k]` at 0, F32 scales
/// `[k / 32, n]` at a 256-byte aligned offset.
#[inline]
pub(crate) fn q8_a8_x_layout(k: usize, n: usize) -> (usize, usize) {
    let scales_off = (n * k).next_multiple_of(256);
    (scales_off, scales_off + n * (k / 32) * 4)
}

#[inline]
pub(crate) fn int8_mmq_x_needed(k: usize, n: usize) -> usize {
    k.div_ceil(128) * n * 136
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

/// Wf (Mp*K bytes), Rw (Mp*K/128 f32), Ew (Mp f32).
pub(crate) fn fp8_f2_weight_extents(m: usize, k: usize) -> Option<(usize, usize, usize)> {
    let mp = m.checked_add(255)? / 256 * 256;
    let wf = mp.checked_mul(k)?;
    let rw = mp.checked_mul(k / 128)?.checked_mul(4)?;
    let ew = mp.checked_mul(4)?;
    Some((wf, rw, ew))
}

impl ScratchState {
    /// Grow the single ordered-stream F2 workspace; no caller may retain
    /// its contents after a later repack, even when the allocation did not grow.
    pub(crate) fn ensure_fp8_f2_weights(
        &mut self,
        hip: &HipRuntime,
        needed: usize,
    ) -> HipResult<*mut c_void> {
        grow_scratch_buffer(
            hip,
            &mut self.fp8_f2_weights,
            &mut self.fp8_f2_weights_bytes,
            needed,
        )?;
        Ok(self.fp8_f2_weights.as_ref().unwrap().as_ptr())
    }

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
                note_scratch_growth();
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

    /// Ensure the MQ6 FP8 producer `x_out` scratch holds at least `min_elems` F32
    /// values (same grow-on-demand contract as [`Self::ensure_gemv_residual_tmp`]:
    /// the `Gpu` caller invalidated captured graphs first when it will grow).
    pub fn ensure_mq6_fp8_x_rot(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
        min_elems: usize,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        let needed_bytes = min_elems * 4;
        let needs_grow = self
            .mq6_fp8_x_rot
            .as_ref()
            .map_or(true, |t| t.buf.size() < needed_bytes);
        if needs_grow {
            if let Some(old) = self.mq6_fp8_x_rot.take() {
                hip.device_synchronize()?;
                note_scratch_growth();
                let _ = hip.free(old.buf);
            }
            self.mq6_fp8_x_rot = Some(GpuTensor {
                buf: hip.malloc(needed_bytes)?,
                shape: vec![min_elems],
                dtype: DType::F32,
            });
        }
        Ok(())
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
        let s1 = gen_fwht_signs(42, MQ_FWHT_SIGNS_G256);
        let s2 = gen_fwht_signs(1042, MQ_FWHT_SIGNS_G256);
        let s1b: Vec<u8> = s1.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s2b: Vec<u8> = s2.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let s1t = alloc_tensor_on(hip, pool, device_id, &[MQ_FWHT_SIGNS_G256], DType::F32)?;
        let s2t = alloc_tensor_on(hip, pool, device_id, &[MQ_FWHT_SIGNS_G256], DType::F32)?;
        hip.memcpy_htod(&s1t.buf, &s1b)?;
        hip.memcpy_htod(&s2t.buf, &s2b)?;
        // Allocate scratch buffers — 32K elements covers K up to 32768
        let x_rot = alloc_tensor_on(hip, pool, device_id, &[MQ_X_ROT_ELEMS], DType::F32)?;
        let x_q8 = hip.malloc(MQ_X_Q8_BYTES)?; // INT8 buffer for dp4a
        let x_scales = hip.malloc(MQ_X_SCALES_BYTES)?; // up to 128 groups × f32
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

    /// Lazily allocate and zero the persistent gfx1100 AWQ RMSNorm wavegrid
    /// state: 256 f32 per-thread partials, one f32 RMS, three u32 epoch
    /// counters, padded to 2048 bytes.
    pub fn ensure_mq_rmsnorm_awq_wavegrid_scratch(
        &mut self,
        hip: &HipRuntime,
        device_id: i32,
    ) -> HipResult<()> {
        crate::graph::bind_thread(hip, device_id)?;
        if self.mq_rmsnorm_awq_wavegrid_scratch.is_none() {
            let scratch = hip.malloc(2048)?;
            if let Err(err) = hip.memset(&scratch, 0, 2048) {
                note_scratch_growth();
                let _ = hip.free(scratch);
                return Err(err);
            }
            self.mq_rmsnorm_awq_wavegrid_scratch = Some(scratch);
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
            let signs1 = gen_fwht_signs(43, MQ_FWHT_SIGNS_G128);
            let signs2 = gen_fwht_signs(1043, MQ_FWHT_SIGNS_G128);
            let s1b: Vec<u8> = signs1.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s2b: Vec<u8> = signs2.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let s1t = alloc_tensor_on(hip, pool, device_id, &[MQ_FWHT_SIGNS_G128], DType::F32)?;
            let s2t = alloc_tensor_on(hip, pool, device_id, &[MQ_FWHT_SIGNS_G128], DType::F32)?;
            hip.memcpy_htod(&s1t.buf, &s1b)?;
            hip.memcpy_htod(&s2t.buf, &s2b)?;
            self.mq_signs1_128 = Some(s1t);
            self.mq_signs2_128 = Some(s2t);
        }
        // Allocate shared rotation scratch if ensure_mq_signs (G256 path) has not run yet.
        if self.mq_x_rot.is_none() {
            let x_rot = alloc_tensor_on(hip, pool, device_id, &[MQ_X_ROT_ELEMS], DType::F32)?;
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
            note_scratch_growth();
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
                        note_scratch_growth();
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
            &mut self.route_load,
            self.pm_decode,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let src_ptr = x.buf.as_ptr();
        let needed = fp16_x_bytes(n_elems)
            .ok_or_else(|| hip_bridge::HipError::new(0, "ensure_fp16_x: extent overflow"))?;

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

        let must_convert = scratch_must_convert(
            capture_mode,
            replay.is_recording(),
            self.fp16_x_source_ptr,
            src_ptr,
        );
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

    /// The FP16 X scratch, grown to `n_elems`, for a caller that writes it
    /// directly.  It no longer holds the cached source's conversion, so the
    /// pointer-keyed `ensure_fp16_x` cache is dropped.
    pub fn fp16_x_scratch_writable(
        &mut self,
        hip: &HipRuntime,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        let needed = fp16_x_bytes(n_elems).ok_or_else(|| {
            hip_bridge::HipError::new(0, "fp16_x_scratch_writable: extent overflow")
        })?;
        if self.fp16_x_scratch_bytes < needed {
            grow_scratch_buffer(
                hip,
                &mut self.fp16_x_scratch,
                &mut self.fp16_x_scratch_bytes,
                needed,
            )?;
        }
        self.fp16_x_source_ptr = std::ptr::null_mut();
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
            &mut self.route_load,
            self.pm_decode,
            "convert_f32_to_f16",
            kernels::GEMM_HFQ4G256_RESIDUAL_FP16_SRC,
            "convert_f32_to_f16",
        )?;

        let in_ptr = x.buf.as_ptr();
        let out_ptr = self.fp16_x_scratch_writable(hip, n_elems)?;
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
            &mut self.route_load,
            self.pm_decode,
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

        let must_convert = scratch_must_convert(
            capture_mode,
            replay.is_recording(),
            self.fp8_x_source_ptr,
            src_ptr,
        );
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
            hip,
            compiler,
            modules,
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            replay,
            "pack_f16_to_fp8_mq4v2_gfx12",
            kernels::PACK_F16_TO_FP8_MQ4V2_GFX12_SRC,
            "pack_f16_to_fp8_mq4v2_gfx12",
            x_f16,
            n,
            k,
            scale_mode,
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
            hip,
            compiler,
            modules,
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            replay,
            "pack_f32_to_fp8_mq4v2_gfx12",
            kernels::PACK_F32_TO_FP8_MQ4V2_GFX12_SRC,
            "pack_f32_to_fp8_mq4v2_gfx12",
            x_f32,
            n,
            k,
            scale_mode,
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
        let mut row_scale_shift = crate::gemv::fp8_row_scale_shift()?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            &mut self.route_load,
            self.pm_decode,
            module,
            ksrc,
            symbol,
        )?;

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
            &mut row_scale_shift as *mut _ as *mut c_void,
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
                b.push_i32(row_scale_shift);
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
        })
    }

    /// FP8 verify activation pre-pass: pack F32 `x_f32` `[n, k]` into the
    /// dedicated E4M3 + per-row-scale slots with
    /// `pack_f32_to_fp8_verify_rows_gfx1201` (bytes and scales bit-identical
    /// to `pack_f32_to_fp8_mq4v2_gfx12` scale_mode 1). Always overwrites; no
    /// source-pointer cache. The caller owns validation (`n > 0`,
    /// `k % 256 == 0`, `k <= 32768`, `x_f32` 16-byte aligned) and MUST have
    /// run the scratch-growth protocol for both slots. Launched through
    /// `launch_maybe_blob` so capture and replay recording observe it.
    /// Returns `(x8, scales)` device pointers, valid until the next pack.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pack_fp8_verify_rows(
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
        row_scale_shift: i32,
    ) -> HipResult<(*mut c_void, *mut c_void)> {
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            &mut self.route_load,
            self.pm_decode,
            "pack_fp8_verify_rows_gfx1201",
            kernels::PACK_FP8_VERIFY_ROWS_GFX1201_SRC,
            "pack_f32_to_fp8_verify_rows_gfx1201",
        )?;
        let x8_bytes = n * k;
        let scales_bytes = n * 4;
        grow_scratch_buffer(
            hip,
            &mut self.fp8_verify_x8_scratch,
            &mut self.fp8_verify_x8_scratch_bytes,
            x8_bytes,
        )?;
        grow_scratch_buffer(
            hip,
            &mut self.fp8_verify_scales_scratch,
            &mut self.fp8_verify_scales_scratch_bytes,
            scales_bytes,
        )?;
        let out_x8 = self.fp8_verify_x8_scratch.as_ref().unwrap().as_ptr();
        let out_scales = self.fp8_verify_scales_scratch.as_ref().unwrap().as_ptr();
        let mut in_ptr_m = x_f32;
        let mut out_x8_m = out_x8;
        let mut out_scales_m = out_scales;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut shift_m = row_scale_shift;
        let mut params: [*mut c_void; 6] = [
            &mut in_ptr_m as *mut _ as *mut c_void,
            &mut out_x8_m as *mut _ as *mut c_void,
            &mut out_scales_m as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut shift_m as *mut _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            hip,
            "gemm",
            "pack_f32_to_fp8_verify_rows_gfx1201",
            n * k * 4 + x8_bytes + scales_bytes,
        );
        let result = launch_maybe_blob(
            hip,
            Some(&*compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            "pack_f32_to_fp8_verify_rows_gfx1201",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x_f32);
                b.push_ptr(out_x8);
                b.push_ptr(out_scales);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b.push_i32(row_scale_shift);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(hip);
        }
        result?;
        Ok((out_x8, out_scales))
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
            &mut self.route_load,
            self.pm_decode,
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
            let grid = fold_rows_into_x(k.div_ceil(1024), batch_size)?;
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
                grid,
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
            &mut self.route_load,
            self.pm_decode,
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
            let grid = fold_rows_into_x(k.div_ceil(1024), batch_size)?;
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
                grid,
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
            &mut self.route_load,
            self.pm_decode,
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
            let grid = fold_rows_into_x(k.div_ceil(1024), batch_size)?;
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
                grid,
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
            plane_rows: 0,
        })
    }

    /// Grow `int4_mmq_down_scratch` for the A4-fused gate/up epilogue's
    /// down-proj sidecar (`[k/128, n]` 72 B records) and bump its generation.
    /// Same validity gate as [`Self::reserve_int4_mmq`]; never touches
    /// `int4_mmq_x_scratch` or its generation.
    pub fn reserve_int4_mmq_down(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<Int4MmqDownReservation> {
        if k == 0 || n == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "reserve_int4_mmq_down: need k%256==0 and n>0",
            ));
        }
        let needed = int4_mmq_reserve_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.int4_mmq_down_scratch,
            &mut self.int4_mmq_down_scratch_bytes,
            needed,
        )?;
        self.int4_mmq_down_generation = self.int4_mmq_down_generation.wrapping_add(1);
        let ptr = self.int4_mmq_down_scratch.as_ref().unwrap().as_ptr();
        Ok(Int4MmqDownReservation {
            ptr,
            k,
            n,
            generation: self.int4_mmq_down_generation,
        })
    }

    /// Grow the Qwen4 symmetric IU4 MoE grouping scratch to hold the layout
    /// of `max_slots` routed slots over `experts` experts (every smaller
    /// slot count fits). Callers reserve the forward's maximum before graph
    /// capture so a captured grouping never reallocates.
    pub fn reserve_qwen4_moe_group(
        &mut self,
        hip: &HipRuntime,
        max_slots: usize,
        experts: usize,
    ) -> HipResult<()> {
        let needed = qwen4_moe_group_layout(max_slots, experts).bytes;
        grow_scratch_buffer(
            hip,
            &mut self.qwen4_moe_group_scratch,
            &mut self.qwen4_moe_group_scratch_bytes,
            needed,
        )
    }

    /// Reserve the sealed expert-slab scratch: enough descriptor space for both
    /// representations of `max_lanes` lanes and `tiles` masked tile ids per codec.
    /// Preflight all size arithmetic before either allocation can change.
    pub fn reserve_qwen4_slab(
        &mut self,
        hip: &HipRuntime,
        max_lanes: usize,
        tiles: usize,
    ) -> HipResult<()> {
        let union_bytes = qwen4_slab_descriptor_bytes(max_lanes).ok_or_else(|| {
            hip_bridge::HipError::new(0, "reserve_qwen4_slab: union descriptor extent overflow")
        })?;
        let tile_bytes = qwen4_slab_tiles_bytes(tiles).ok_or_else(|| {
            hip_bridge::HipError::new(0, "reserve_qwen4_slab: tile mask extent overflow")
        })?;
        grow_scratch_buffer(
            hip,
            &mut self.qwen4_slab_union_scratch,
            &mut self.qwen4_slab_union_scratch_bytes,
            union_bytes,
        )?;
        if tiles != 0 {
            grow_scratch_buffer(
                hip,
                &mut self.qwen4_slab_tiles_scratch,
                &mut self.qwen4_slab_tiles_scratch_bytes,
                tile_bytes,
            )?;
        }
        Ok(())
    }

    /// Grow the Qwen4 symmetric IU4 MoE down sidecar for `[k/128, n]`
    /// `block_i4_128` and bump its generation (`k % 128 == 0`). The producer
    /// writes every live slot's blocks; the reservation launches nothing.
    pub fn reserve_qwen4_moe_down_i4(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<Int4MmqReservation> {
        self.reserve_qwen4_moe_down_i4_plane(hip, k, n, 0)
    }

    /// [`Self::reserve_qwen4_moe_down_i4`] for either layout: `plane_rows == 0` is the flat-slot
    /// sidecar `[k/128, n]`; otherwise the grouped-position plane `[k/128, plane_rows]` (`plane_rows`
    /// = the grouped row count), whose handle still reports `n` real slots.
    pub fn reserve_qwen4_moe_down_i4_plane(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
        plane_rows: usize,
    ) -> HipResult<Int4MmqReservation> {
        if k == 0 || n == 0 || k % 128 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "reserve_qwen4_moe_down_i4: need k%128==0 and n>0",
            ));
        }
        let needed = qwen4_moe_down_i4_bytes(k, if plane_rows == 0 { n } else { plane_rows })
            .ok_or_else(|| {
                hip_bridge::HipError::new(0, "reserve_qwen4_moe_down_i4: extent overflow")
            })?;
        grow_scratch_buffer(
            hip,
            &mut self.qwen4_moe_down_i4_scratch,
            &mut self.qwen4_moe_down_i4_scratch_bytes,
            needed,
        )?;
        self.qwen4_moe_down_i4_generation = self.qwen4_moe_down_i4_generation.wrapping_add(1);
        Ok(Int4MmqReservation {
            ptr: self.qwen4_moe_down_i4_scratch.as_ref().unwrap().as_ptr(),
            k,
            n,
            generation: self.qwen4_moe_down_i4_generation,
            plane_rows,
        })
    }

    /// Live generation + pointer of the down sidecar, for
    /// [`Int4MmqPrepared::checked_ptr`].
    #[inline]
    pub fn qwen4_moe_down_i4_live(&self) -> (u64, *mut c_void) {
        let ptr = self
            .qwen4_moe_down_i4_scratch
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        (self.qwen4_moe_down_i4_generation, ptr)
    }

    /// Grow the W8A8 Q8_0 activation sidecar to hold `(k, n)` and return
    /// `(xq, xs)` device pointers (int8 `[n, k]` and F32 scales `[k/32, n]`).
    /// Callers invalidate graphs before growth (see
    /// `Gpu::ensure_q8_a8_x`); the producer overwrites every byte it reads.
    pub(crate) fn reserve_q8_a8_x(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<(*mut c_void, *mut c_void)> {
        let (scales_off, needed) = q8_a8_x_layout(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.q8_a8_x_scratch,
            &mut self.q8_a8_x_scratch_bytes,
            needed,
        )?;
        let base = self.q8_a8_x_scratch.as_ref().unwrap().as_ptr();
        Ok((
            base,
            (base as *mut u8).wrapping_add(scales_off) as *mut c_void,
        ))
    }

    /// Always re-quantize the f32 activation row into the dedicated A8 sidecar.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_int8_mmq_x(
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
        n: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        // The quantizer ships in each arch's A8 GEMM module (one compile).
        let (module, source) = if compiler.arch() == "gfx1201" {
            (
                "gemm_mq4g256v2_residual_mmq_i8_gfx12",
                kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_I8_GFX12_SRC,
            )
        } else {
            (
                "gemm_mq4g256v2_residual_mmq_i8_gfx11",
                kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_I8_GFX11_SRC,
            )
        };
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            &mut self.route_load,
            self.pm_decode,
            module,
            source,
            "quantize_int8_mmq_ds128",
        )?;
        let needed = int8_mmq_x_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.int8_mmq_x_scratch,
            &mut self.int8_mmq_x_scratch_bytes,
            needed,
        )?;
        self.int8_mmq_generation = self.int8_mmq_generation.wrapping_add(1);
        let xp = x.buf.as_ptr();
        let yp = self.int8_mmq_x_scratch.as_ref().unwrap().as_ptr();
        let mut x_ptr = xp;
        let mut y_ptr = yp;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let grid = fold_rows_into_x(k.div_ceil(1024), n)?;
        let timer = crate::profile::begin_timer(
            hip,
            "quantize",
            "quantize_int8_mmq_ds128",
            n * k * 4 + needed,
        );
        launch_maybe_blob(
            hip,
            Some(&*compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            "quantize_int8_mmq_ds128",
            grid,
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        )?;
        if let Some(t) = timer {
            t.finish(hip);
        }
        Ok(yp)
    }

    /// Re-quantize f32 activations into the gfx1201 `block_fp8_128` (e4m3) sidecar. The sidecar
    /// shares `int8_mmq_x_scratch` (same `[K/128, n] x 136 B` extent) and bumps the same
    /// generation, so it overwrites any live int8 sidecar.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_fp8_mmq_x(
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
        n: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        crate::graph::bind_thread(hip, device_id)?;
        compile_and_load_kernel(
            compiler,
            hip,
            modules,
            functions,
            &mut self.route_load,
            self.pm_decode,
            crate::moe_grouped_qt44_iu4::FP8_X_MODULE,
            crate::moe_grouped_qt44_iu4::FP8_X_SRC,
            "quantize_fp8_mmq_ds128",
        )?;
        let needed = int8_mmq_x_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.int8_mmq_x_scratch,
            &mut self.int8_mmq_x_scratch_bytes,
            needed,
        )?;
        self.int8_mmq_generation = self.int8_mmq_generation.wrapping_add(1);
        let xp = x.buf.as_ptr();
        let yp = self.int8_mmq_x_scratch.as_ref().unwrap().as_ptr();
        let mut x_ptr = xp;
        let mut y_ptr = yp;
        let mut k_val = k as i32;
        let mut n_val = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut x_ptr as *mut _ as *mut c_void,
            &mut y_ptr as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
        ];
        let grid = fold_rows_into_x(k.div_ceil(1024), n)?;
        let timer = crate::profile::begin_timer(
            hip,
            "quantize",
            "quantize_fp8_mmq_ds128",
            n * k * 4 + needed,
        );
        launch_maybe_blob(
            hip,
            Some(&*compiler),
            functions,
            stream,
            capture_blobs,
            capture_mode,
            force_blob_path,
            Some(replay),
            "quantize_fp8_mmq_ds128",
            grid,
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(xp);
                b.push_ptr(yp);
                b.push_i32(k_val);
                b.push_i32(n_val);
                b
            },
        )?;
        if let Some(t) = timer {
            t.finish(hip);
        }
        Ok(yp)
    }

    /// Reserve the existing 144-byte DS4 slot without issuing a quantizer.
    /// Caller must invalidate graphs before growth and overwrite every block.
    pub(crate) fn reserve_packed_mq4(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<*mut c_void> {
        let needed = q8_1_mmq_x_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.q8_1_mmq_x_scratch,
            &mut self.q8_1_mmq_x_scratch_bytes,
            needed,
        )?;
        Ok(self.q8_1_mmq_x_scratch.as_ref().unwrap().as_ptr())
    }

    pub fn reserve_int8_mmq(
        &mut self,
        hip: &HipRuntime,
        k: usize,
        n: usize,
    ) -> HipResult<Int8MmqReservation> {
        if k == 0 || n == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "reserve_int8_mmq: need k%256==0 and n>0",
            ));
        }
        let needed = int8_mmq_x_needed(k, n);
        grow_scratch_buffer(
            hip,
            &mut self.int8_mmq_x_scratch,
            &mut self.int8_mmq_x_scratch_bytes,
            needed,
        )?;
        self.int8_mmq_generation = self.int8_mmq_generation.wrapping_add(1);
        Ok(Int8MmqReservation {
            ptr: self.int8_mmq_x_scratch.as_ref().unwrap().as_ptr(),
            k,
            n,
            generation: self.int8_mmq_generation,
        })
    }

    pub fn int8_mmq_live(&self) -> (u64, *mut c_void) {
        (
            self.int8_mmq_generation,
            self.int8_mmq_x_scratch
                .as_ref()
                .map_or(std::ptr::null_mut(), |b| b.as_ptr()),
        )
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
    ) -> HipResult<(*mut c_void, *mut c_void, *mut c_void)> {
        if k == 0 || n == 0 || k % 256 != 0 {
            return Err(hip_bridge::HipError::new(
                0,
                "grow_mq4v2_fp8_for_producer: need k%256==0 and n>0",
            ));
        }
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

    /// Live generation + pointer for [`Int4MmqDownPrepared::checked_ptr`].
    #[inline]
    pub fn int4_mmq_down_live(&self) -> (u64, *mut c_void) {
        let ptr = self
            .int4_mmq_down_scratch
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        (self.int4_mmq_down_generation, ptr)
    }

    /// Record that a slab producer twin wrote the live generation sealed
    /// into `prepared` (host only). Fails closed on a stale handle.
    pub fn mark_int4_mmq_slab(&mut self, prepared: &Int4MmqPrepared) -> HipResult<()> {
        let (generation, ptr) = self.int4_mmq_live();
        prepared.checked_ptr(generation, ptr, prepared.k(), prepared.n())?;
        self.int4_mmq_slab_generation = Some(generation);
        Ok(())
    }

    /// True when `ptr` is the live IU4 scratch and its contents are the slab
    /// layout; a later reservation or standalone quantize bumps the
    /// generation, so this can never describe overwritten contents.
    #[inline]
    pub fn int4_mmq_slab_at(&self, ptr: *mut c_void) -> bool {
        let (generation, live) = self.int4_mmq_live();
        !ptr.is_null() && ptr == live && self.int4_mmq_slab_generation == Some(generation)
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
    /// Folded 1-D grid `[(K/256) * N, 1, 1]` (batch = blockIdx.x / (K/256)) — processes [N × K] x/x_rot.
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
        let grid = fold_rows_into_x(n_groups as usize, batch_size)?;
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
            grid,
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
            &mut self.route_load,
            self.pm_decode,
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
mod route_load_tests {
    use super::*;

    /// A kernel a walked (open) route did not plan loads lazily and is
    /// counted; a sealed route refuses it; an unplanned route loads silently.
    #[test]
    fn unwalked_kernel_warns_and_loads_on_an_open_route() {
        let mut open = RouteLoad::Open {
            route: "Flash-Next gfx1201 NativeMtp".into(),
            unplanned: 0,
            stats: RouteKernelStats::default(),
        };
        assert!(open
            .admit_unplanned("gemv_mq2g256v2", "gemv_mq2g256v2", LazyKind::HipNoRecord)
            .is_ok());
        assert!(open
            .admit_unplanned(
                "topk8_rescore_mq6g256v2",
                "topk8_rescore_mq6g256v2_k2560",
                LazyKind::EmbeddedImage
            )
            .is_ok());
        assert!(matches!(open, RouteLoad::Open { unplanned: 2, .. }));
        let mut sealed = RouteLoad::Sealed;
        let error = sealed
            .admit_unplanned("gemv_mq2g256v2", "gemv_mq2g256v2", LazyKind::HipNoRecord)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("outside this load's planned kernel route"));
        for mut lazy in [
            RouteLoad::Idle,
            RouteLoad::Prepared {
                open: None,
                stats: RouteKernelStats::default(),
            },
        ] {
            assert!(lazy
                .admit_unplanned("gemv_mq2g256v2", "gemv_mq2g256v2", LazyKind::HipNoRecord)
                .is_ok());
        }
    }

    fn predicted_module(module: &'static str) -> crate::kernel_registry::PredictedModule {
        let entry = crate::kernel_registry::entries("gfx1151", "")
            .unwrap()
            .into_iter()
            .find(|e| e.module == module)
            .unwrap();
        crate::kernel_registry::PredictedModule {
            bind: entry.symbols.to_vec(),
            entry,
        }
    }

    /// A predicted-family fallback can never be sealed: predicted modules
    /// need an open label, and only an open `Prepared` route finishes `Open`.
    #[test]
    fn predicted_route_is_always_open() {
        let empty = || crate::kernel_registry::RouteKernelPlan {
            entries: Vec::new(),
        };
        let refused = RouteLoad::pending_route(empty(), None, vec![predicted_module("add")], None);
        assert!(refused.err().unwrap().to_string().contains("open route"));
        let route = RouteLoad::pending_route(
            empty(),
            None,
            vec![predicted_module("add")],
            Some("qwen3.5 gfx1151 predicted".into()),
        )
        .unwrap();
        assert!(route.pending());
        assert_eq!(route.stats().predicted_modules, 1);
        // Sealing is only reachable for a plan without predicted modules.
        let sealed = RouteLoad::pending_route(empty(), None, Vec::new(), None).unwrap();
        assert!(matches!(sealed, RouteLoad::Pending { open: None, .. }));
        let stats = RouteKernelStats {
            predicted_modules: 3,
            predicted_compiled: 2,
            ..Default::default()
        };
        let prepared = RouteLoad::Prepared {
            open: Some("fallback".into()),
            stats,
        };
        assert!(matches!(
            prepared.finished(),
            RouteLoad::Open { unplanned: 0, stats: s, .. } if s == stats
        ));
        assert!(matches!(
            RouteLoad::Prepared { open: None, stats }.finished(),
            RouteLoad::Sealed
        ));
    }

    /// Kernels requested between begin and ready and after ready are
    /// counted in separate phases; the open counter is unchanged by load.
    #[test]
    fn lazy_misses_count_per_phase() {
        let mut route = RouteLoad::Prepared {
            open: Some("qwen3.8-flash-next gfx1100 predicted".into()),
            stats: RouteKernelStats::default(),
        };
        route
            .admit_unplanned("a", "a_f32", LazyKind::HipNoRecord)
            .unwrap();
        route
            .admit_unplanned("b", "b_f32", LazyKind::HipPrecompiled)
            .unwrap();
        route
            .admit_unplanned("n", "n_f32", LazyKind::EmbeddedImage)
            .unwrap();
        assert_eq!(route.stats().lazy_during_load, 3);
        let mut open = route.finished();
        assert!(matches!(open, RouteLoad::Open { unplanned: 0, .. }));
        open.admit_unplanned("c", "c_f32", LazyKind::HipNoRecord)
            .unwrap();
        assert!(matches!(open, RouteLoad::Open { unplanned: 1, .. }));
        assert_eq!(open.stats().lazy_during_load, 3);
    }

    /// A model that left its closed plan finishes open with the preload's
    /// counters kept (never sealed, never reset), even from a sealed plan.
    #[test]
    fn model_leaving_the_plan_finishes_open_and_keeps_stats() {
        let stats = RouteKernelStats {
            predicted_modules: 4,
            predicted_compiled: 4,
            ..Default::default()
        };
        for open in [None, Some("walked".to_owned())] {
            let route =
                RouteLoad::Prepared { open, stats }.finished_open("left the planned route".into());
            assert!(matches!(
                &route,
                RouteLoad::Open { route, unplanned: 0, stats: s } if route == "left the planned route" && *s == stats
            ));
        }
        assert!(matches!(
            RouteLoad::Sealed.finished_open("x".into()),
            RouteLoad::Sealed
        ));
    }
}

#[cfg(test)]
mod int4_mmq_from_live_tests {
    use super::*;

    #[test]
    fn from_live_token_is_fail_closed() {
        let ptr = 0x1000usize as *mut c_void;
        let other = 0x2000usize as *mut c_void;
        let token = Int4MmqPrepared::from_live(7, ptr, 2048, 128);
        assert_eq!(token.checked_ptr(7, ptr, 2048, 128).unwrap(), ptr);
        assert!(token.checked_ptr(8, ptr, 2048, 128).is_err());
        assert!(token.checked_ptr(7, other, 2048, 128).is_err());
        assert!(token
            .checked_ptr(7, std::ptr::null_mut(), 2048, 128)
            .is_err());
        assert!(token.checked_ptr(7, ptr, 1024, 128).is_err());
        assert!(token.checked_ptr(7, ptr, 2048, 64).is_err());
    }
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

    /// The gathered QSA workspace maps exactly the granularity-rounded prefix
    /// a launch touches: nothing when covered, only the missing pages past
    /// the current prefix, and never past the reservation.
    #[test]
    fn vmm_prefix_growth_maps_only_the_missing_pages() {
        const G: usize = 2 << 20;
        // Covered (including the exact page end): no map.
        assert_eq!(vmm_prefix_growth(G, 64 * G, G, 1), Some(0));
        assert_eq!(vmm_prefix_growth(G, 64 * G, G, G), Some(0));
        // One byte past a page maps one more page.
        assert_eq!(vmm_prefix_growth(G, 64 * G, G, G + 1), Some(G));
        // From empty: the rounded prefix.
        assert_eq!(vmm_prefix_growth(0, 64 * G, G, 3 * G - 7), Some(3 * G));
        // Up to an unaligned logical reservation end (the arena rounds it up).
        assert_eq!(vmm_prefix_growth(2 * G, 5 * G / 2, G, 5 * G / 2), Some(G));
        // Past the reservation, or without a granularity: refused.
        assert_eq!(vmm_prefix_growth(2 * G, 5 * G / 2, G, 5 * G / 2 + 1), None);
        assert_eq!(vmm_prefix_growth(0, 64 * G, 0, 1), None);
    }

    /// gfx1100 map-only discipline for the gather workspace: a held
    /// reservation is kept when it covers the request, never shrunk, and a
    /// larger request replaces it with a fresh VA (the old one is released and
    /// retired by `VmmArena::release`), never grown or remapped in place.
    #[test]
    fn qsa_gather_reservation_is_never_shrunk_or_grown_in_place() {
        use QsaGatherReserve::{Keep, ReplaceRetiring, Reserve};
        assert_eq!(qsa_gather_reserve_action(None, 1), Reserve);
        assert_eq!(qsa_gather_reserve_action(Some(100), 100), Keep);
        assert_eq!(qsa_gather_reserve_action(Some(100), 1), Keep);
        assert_eq!(qsa_gather_reserve_action(Some(100), 101), ReplaceRetiring);
        // The two fixed decode reservations (F32 split-K 48 splits, q8 split-KV
        // 32 splits) in either arrival order: at most one replacement, and the
        // larger one then serves both.
        let (small, large) = (32 * 1028 * 24 * 64, 48 * 1032 * 24 * 64);
        let mut held = None;
        let mut replacements = 0;
        for request in [small, large, small, large, small] {
            match qsa_gather_reserve_action(held, request) {
                Keep => {}
                Reserve => held = Some(request),
                ReplaceRetiring => {
                    replacements += 1;
                    held = Some(request);
                }
            }
        }
        assert_eq!((held, replacements), (Some(large), 1));
    }

    /// Mapping the gather workspace (and, by the same rule, a QSA arena) only
    /// advances the mapped frontier: every growth is a whole number of pages,
    /// the prefix after it covers the request, and nothing ever asks for the
    /// mapped prefix to shrink.
    #[test]
    fn vmm_prefix_growth_only_moves_the_frontier_forward() {
        const G: usize = 2 << 20;
        let reserved = 64 * G;
        let mut mapped = 0usize;
        for needed in [1, G, 5 * G + 3, 3 * G, 7, 64 * G, 64 * G - 1] {
            let grow = vmm_prefix_growth(mapped, reserved, G, needed).unwrap();
            assert_eq!(grow % G, 0, "growth must be whole pages");
            let next = mapped + grow;
            assert!(next >= mapped && next >= needed && next <= reserved);
            if needed <= mapped {
                assert_eq!(grow, 0, "a covered request maps nothing");
            }
            mapped = next;
        }
    }

    /// Source audit (CPU) of the gfx1100 VMM growth path in `dispatch.rs`:
    /// growth only calls the arena's forward map; the gather reserve's one
    /// replacement path releases (retires) the old reservation before it
    /// allocates a fresh VA; neither unmaps, remaps or frees on any other path.
    /// hipMemUnmap is not called from this crate at all.
    #[test]
    fn gather_and_vmm_growth_paths_never_unmap_or_remap_a_live_va() {
        fn code(source: &str, signature: &str, end: &str) -> String {
            let start = source
                .find(signature)
                .unwrap_or_else(|| panic!("{signature} not found"));
            let rest = &source[start..];
            let body = &rest[..rest.find(end).expect("end of function")];
            body.lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n")
        }
        let src = include_str!("dispatch.rs");
        let unmap = format!("{}{}", "mem_", "unmap");
        let never = |body: &str, what: &str, extra: &[&str]| {
            for token in [unmap.as_str(), "mem_release", "free_tensor", "release("]
                .into_iter()
                .chain(extra.iter().copied())
            {
                assert!(!body.contains(token), "{what} contains `{token}`");
            }
        };

        let grow = code(src, "fn grow_vmm_tensor_impl(", "\n    }\n");
        never(&grow, "grow_vmm_tensor_impl", &["alloc_vmm_tensor"]);
        assert!(grow.contains(".map_next(") && grow.contains(".map_next_granules("));

        let prefix = code(src, "fn map_qsa_gather_prefix(", "\n    }\n");
        never(&prefix, "map_qsa_gather_prefix", &["alloc_vmm_tensor"]);
        assert!(prefix.contains("grow_vmm_tensor("));

        let scratch = code(src, "pub(crate) fn qsa_gather_scratch(", "\n    }\n");
        never(&scratch, "qsa_gather_scratch", &["alloc_vmm_tensor"]);
        assert!(scratch.contains("map_qsa_gather_prefix("));

        let reserve = code(src, "pub(crate) fn qsa_gather_reserve(", "\n    }\n");
        never(&reserve, "qsa_gather_reserve", &[]);
        let replace = reserve
            .find("QsaGatherReserve::ReplaceRetiring")
            .expect("replacement branch");
        let release = reserve
            .find("release_qsa_gather_workspace()")
            .expect("old reservation is released");
        let alloc = reserve.find("alloc_vmm_tensor(").expect("fresh VA");
        assert!(
            replace < release && release < alloc,
            "a replaced reservation must be released (VA retired) before the fresh one is reserved"
        );
        assert_eq!(reserve.matches("release_qsa_gather_workspace()").count(), 1);

        // The release wrapper only frees the arena owner (VmmArena::release
        // unmaps its own segments and retires the VA).
        let wrapper = code(
            src,
            "pub(crate) fn release_qsa_gather_workspace(",
            "\n    }\n",
        );
        assert!(wrapper.contains("free_tensor(workspace)"));
        assert!(!wrapper.contains(unmap.as_str()));

        // No hipMemUnmap in rdna-compute at all: the only unmapping is inside
        // hip-bridge's VmmArena teardown.
        fn scan(dir: &std::path::Path, needle: &str, hits: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("read_dir") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    scan(&path, needle, hits);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    let text = std::fs::read_to_string(&path).expect("read source");
                    if text.contains(needle) {
                        hits.push(path.display().to_string());
                    }
                }
            }
        }
        let mut hits = Vec::new();
        scan(
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
            &format!(".{unmap}("),
            &mut hits,
        );
        assert!(
            hits.is_empty(),
            "direct hipMemUnmap in rdna-compute: {hits:?}"
        );
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
                    let (total, sq_off, scale_off) = fa2_fp8_q_needed(batch, splits, with_plane);
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

#[cfg(test)]
mod prefill_workspace_tests {
    use super::*;

    fn flash_next(
        gdn_format: Option<crate::tensor_ops::GdnStateFormat>,
        include_short_tails: bool,
    ) -> Qwen4PrefillWorkspaceDims {
        Qwen4PrefillWorkspaceDims {
            hidden_size: 2560,
            moe_intermediate_size: 640,
            num_experts: 512,
            top_k: 10,
            hc_count: 4,
            hc_lowrank: 320,
            gdn_format,
            include_short_tails,
            routed_i4: true,
        }
    }

    #[test]
    fn size_queries_match_allocator_geometry() {
        assert_eq!(int4_mmq_reservation_bytes(2560, 4096), Some(5_898_240));
        assert_eq!(
            int4_mmq_reservation_bytes(2560, 4096),
            Some(int4_mmq_reserve_needed(2560, 4096))
        );
        assert_eq!(int4_mmq_reservation_bytes(640, 4096), None); // k % 256 != 0
        assert_eq!(int4_mmq_reservation_bytes(2560, 0), None);
        assert_eq!(qwen4_moe_down_i4_bytes(640, 40960), Some(14_745_600));
        assert_eq!(qwen4_moe_down_i4_bytes(641, 40960), None);
        assert_eq!(
            qwen4_moe_group_layout_checked(40960, 512).map(|l| l.bytes),
            Some(821_248)
        );
        assert_eq!(
            qwen4_moe_group_layout_checked(40960, 512),
            Some(qwen4_moe_group_layout(40960, 512))
        );
        assert_eq!(qwen4_moe_group_layout_checked(usize::MAX, 512), None);
        assert_eq!(qwen4_grouped_rows_bound(40960, 512), Some(48_640));
        assert_eq!(qwen4_slab_tiles_bytes(3040), Some(24_320));
        assert_eq!(qwen4_slab_union_bytes(16), Some(3848));
        assert_eq!(qwen4_slab_union_bytes(32), Some(7688));
        assert_eq!(qwen4_slab_union_bytes(64), Some(15_368));
        assert_eq!(
            qwen4_slab_descriptor_bytes(crate::slab_moe::SLAB_MAX_LANES),
            Some(15_424)
        );
        assert_eq!(qwen4_slab_union_bytes(usize::MAX), None);
        assert_eq!(fp16_x_bytes(4096 * 10240), Some(83_886_080));
        assert_eq!(fp16_x_bytes(usize::MAX), None);
        assert_eq!(MQ_FIXED_WORKSPACE_BYTES, 167_424);
    }

    #[test]
    fn slab_rowtile_sizes_cover_every_width_and_fixed_tail() {
        const ONE_TILE_WORDS: Option<usize> = qwen4_slab_rowtile_words_checked(1);
        assert_eq!(ONE_TILE_WORDS, Some(482));
        assert_eq!(qwen4_slab_rowtile_words_checked(0), Some(0));
        assert_eq!(qwen4_slab_rowtile_bytes(0), Some(0));
        // The legacy representation retains its two terminal offsets at zero.
        assert_eq!(qwen4_slab_union_words_checked(0), Some(2));
        assert_eq!(qwen4_slab_union_bytes(0), Some(8));
        assert_eq!(qwen4_slab_descriptor_bytes(0), Some(8));
        for lanes in 1usize..=64 {
            let tiles = lanes.div_ceil(8);
            let words = tiles * 482;
            assert_eq!(qwen4_slab_rowtile_words_checked(lanes), Some(words));
            assert_eq!(qwen4_slab_rowtile_bytes(lanes), Some(words * 4));
            assert_eq!(
                qwen4_slab_rowtile_bytes(lanes),
                qwen4_slab_rowtile_bytes(tiles * 8),
                "tail descriptor must remain full-sized at R={lanes}"
            );
            assert_eq!(qwen4_slab_union_words_checked(lanes), Some(lanes * 60 + 2));
            assert_eq!(qwen4_slab_union_words(lanes), lanes * 60 + 2);
            assert_eq!(qwen4_slab_union_bytes(lanes), Some((lanes * 60 + 2) * 4));
            assert_eq!(
                qwen4_slab_descriptor_bytes(lanes),
                Some(((lanes * 60 + 2) * 4).max(words * 4))
            );
            if lanes % 8 == 1 && lanes > 1 {
                assert_eq!(
                    qwen4_slab_rowtile_bytes(lanes).unwrap()
                        - qwen4_slab_rowtile_bytes(lanes - 1).unwrap(),
                    1928
                );
            }
        }
        assert_eq!(qwen4_slab_rowtile_words_checked(64), Some(3856));
        assert_eq!(qwen4_slab_rowtile_bytes(64), Some(15_424));
        assert_eq!(qwen4_slab_descriptor_bytes(64), Some(15_424));
    }

    #[test]
    fn slab_descriptor_size_overflows_are_checked() {
        let max_word_tiles = usize::MAX / 482;
        let max_word_lanes = max_word_tiles * 8;
        assert_eq!(
            qwen4_slab_rowtile_words_checked(max_word_lanes),
            Some(max_word_tiles * 482)
        );
        assert_eq!(qwen4_slab_rowtile_words_checked(max_word_lanes + 1), None);
        let max_byte_tiles = usize::MAX / 1928;
        let max_byte_lanes = max_byte_tiles * 8;
        assert_eq!(
            qwen4_slab_rowtile_bytes(max_byte_lanes),
            Some(max_byte_tiles * 1928)
        );
        // Word sizing still fits when conversion to bytes first overflows.
        assert!(qwen4_slab_rowtile_words_checked(max_byte_lanes + 1).is_some());
        assert_eq!(qwen4_slab_rowtile_bytes(max_byte_lanes + 1), None);
        assert_eq!(qwen4_slab_descriptor_bytes(max_byte_lanes + 1), None);
        let max_union_lanes = (usize::MAX - 2) / 60;
        assert!(qwen4_slab_union_words_checked(max_union_lanes).is_some());
        assert_eq!(qwen4_slab_union_words_checked(max_union_lanes + 1), None);
        let max_union_byte_lanes = (usize::MAX / 4 - 2) / 60;
        assert!(qwen4_slab_union_bytes(max_union_byte_lanes).is_some());
        assert_eq!(qwen4_slab_union_bytes(max_union_byte_lanes + 1), None);
        assert_eq!(qwen4_slab_descriptor_bytes(max_union_byte_lanes + 1), None);
        assert_eq!(qwen4_slab_rowtile_words_checked(usize::MAX), None);
        assert_eq!(qwen4_slab_rowtile_bytes(usize::MAX), None);
        assert_eq!(qwen4_slab_descriptor_bytes(usize::MAX), None);
        assert_eq!(qwen4_slab_tiles_bytes(0), Some(0));
        let max_mask_tiles = usize::MAX / 8;
        assert_eq!(
            qwen4_slab_tiles_bytes(max_mask_tiles),
            Some(max_mask_tiles * 8)
        );
        assert_eq!(qwen4_slab_tiles_bytes(max_mask_tiles + 1), None);
        assert_eq!(qwen4_slab_tiles_bytes(usize::MAX), None);
    }

    #[cfg(feature = "deltanet")]
    #[test]
    fn flash_next_4096_workspace_terms() {
        let sum = |terms: [(&str, u64); 11]| terms.iter().map(|t| t.1).sum::<u64>();
        let terms = qwen4_prefill_workspace_terms(
            4096,
            &flash_next(Some(crate::tensor_ops::GdnStateFormat::Q8), true),
        )
        .unwrap();
        let names: Vec<&str> = terms.iter().map(|t| t.0).collect();
        assert_eq!(
            names,
            [
                "fp16_x",
                "ksplit_det",
                "int4_xq",
                "moe_group",
                "down_i4q",
                "slab_tiles",
                "slab_union",
                "mq_fixed",
                "gdn_dense_a",
                "gdn_dense_ef",
                "gdn_pair_counters",
            ]
        );
        let values: Vec<u64> = terms.iter().map(|t| t.1).collect();
        assert_eq!(
            values,
            [
                83_886_080, 20_971_520, 5_898_240, 821_248, 14_745_600, 24_320, 15_424, 167_424,
                25_165_824, 1_572_864, 1_024,
            ]
        );
        assert_eq!(sum(terms), 153_269_568);
        let no_tails = qwen4_prefill_workspace_terms(
            4096,
            &flash_next(Some(crate::tensor_ops::GdnStateFormat::Q8), false),
        )
        .unwrap();
        assert_eq!(sum(no_tails), 153_268_544);
        let no_gdn = qwen4_prefill_workspace_terms(4096, &flash_next(None, false)).unwrap();
        assert_eq!(sum(no_gdn), 153_268_544 - 25_165_824 - 1_572_864);
    }

    /// The dense scan's A scratch is charged for every admitted GDN state (Q8, Q8Ef); the
    /// 1.5 MiB thread-local zero-entry EF scratch only for plain Q8 (Q8Ef scans against the
    /// live residual, charged by the state owner); F32 never reaches the route.
    #[cfg(feature = "deltanet")]
    #[test]
    fn dense_gdn_workspace_terms_charge_by_format() {
        use crate::tensor_ops::GdnStateFormat;
        let terms =
            |format| qwen4_prefill_workspace_terms(4096, &flash_next(format, true)).unwrap();
        let gdn = |format| {
            let t = terms(format);
            assert_eq!(t[8].0, "gdn_dense_a");
            assert_eq!(t[9].0, "gdn_dense_ef");
            (t[8].1, t[9].1)
        };
        assert_eq!(gdn(Some(GdnStateFormat::Q8)), (25_165_824, 1_572_864));
        assert_eq!(gdn(Some(GdnStateFormat::Q8Ef)), (25_165_824, 0));
        assert_eq!(gdn(Some(GdnStateFormat::F32)), (0, 0));
        assert_eq!(gdn(None), (0, 0));
        // Every other term is format-independent.
        for format in [
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
            GdnStateFormat::F32,
        ] {
            let (with, without) = (terms(Some(format)), terms(None));
            for index in (0..11).filter(|index| *index != 8 && *index != 9) {
                assert_eq!(with[index], without[index], "{format:?} term {index}");
            }
        }
    }

    #[test]
    fn dense_gdn_admission_is_exact_gfx1201_q8_family() {
        use crate::tensor_ops::GdnStateFormat::{Q8Ef, F32, Q8};
        for arch in ["gfx1100", "gfx1151", "gfx942", "gfx1200", "gfx12010", ""] {
            for format in [F32, Q8, Q8Ef] {
                assert!(!qwen4_dense_gdn_admits(arch, format), "{arch} {format:?}");
            }
        }
        assert!(qwen4_dense_gdn_admits("gfx1201", Q8));
        assert!(qwen4_dense_gdn_admits("gfx1201", Q8Ef));
        assert!(!qwen4_dense_gdn_admits("gfx1201", F32));
    }

    #[test]
    fn workspace_terms_overflow_and_invalid_dims_are_none() {
        assert!(qwen4_prefill_workspace_terms(usize::MAX, &flash_next(None, false)).is_none());
        let mut dims = flash_next(None, false);
        dims.hidden_size = 2561; // int4 sidecar needs k % 256 == 0
        assert!(qwen4_prefill_workspace_terms(4096, &dims).is_none());
    }

    #[test]
    fn dense_profile_without_routed_i4_charges_only_generic_terms() {
        let mut dims = flash_next(None, false);
        dims.num_experts = 0;
        dims.top_k = 0;
        dims.moe_intermediate_size = 0;
        dims.routed_i4 = false;
        let terms = qwen4_prefill_workspace_terms(4096, &dims).unwrap();
        for (name, bytes) in terms {
            if [
                "int4_xq",
                "moe_group",
                "down_i4q",
                "slab_tiles",
                "slab_union",
            ]
            .contains(&name)
            {
                assert_eq!(bytes, 0, "{name}");
            }
        }
        assert_eq!(terms[1], ("ksplit_det", 20_971_520));
        assert_eq!(terms[0], ("fp16_x", 83_886_080));
        assert_eq!(terms[7], ("mq_fixed", 167_424));
        // Routed dims that the i4 allocators reject stay ignored when the gate is off.
        dims.hidden_size = 2561;
        dims.hc_count = 0;
        dims.hc_lowrank = 0;
        assert!(qwen4_prefill_workspace_terms(4096, &dims).is_some());
    }

    #[test]
    fn dense_ffn_profile_is_not_none_even_with_routed_i4_set() {
        // Dense FFN / synthetic: no experts, small non-256 hidden, routed_i4 left true.
        let dims = Qwen4PrefillWorkspaceDims {
            hidden_size: 64,
            moe_intermediate_size: 0,
            num_experts: 0,
            top_k: 0,
            hc_count: 4,
            hc_lowrank: 32,
            gdn_format: None,
            include_short_tails: true,
            routed_i4: true,
        };
        let terms = qwen4_prefill_workspace_terms(4096, &dims).unwrap();
        for index in [2, 3, 4, 5, 6] {
            assert_eq!(terms[index].1, 0, "{}", terms[index].0);
        }
        assert_eq!(terms[0], ("fp16_x", 4096 * 256 * 2));
        assert_eq!(terms[7], ("mq_fixed", 167_424));
        assert_eq!(terms[10], ("gdn_pair_counters", 1024));
    }
}

#[cfg(test)]
mod device_code_object_tests {
    use super::device_code_object;

    const ELF: &[u8] = &[0x7f, b'E', b'L', b'F', 2, 1, 1, 0x40];
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";

    /// Uncompressed clang offload bundle: header, entry table, payloads.
    fn bundle(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let table: usize = entries.iter().map(|(triple, _)| 24 + triple.len()).sum();
        let mut payload_at = MAGIC.len() + 8 + table;
        let mut out = MAGIC.to_vec();
        out.extend((entries.len() as u64).to_le_bytes());
        for (triple, payload) in entries {
            out.extend((payload_at as u64).to_le_bytes());
            out.extend((payload.len() as u64).to_le_bytes());
            out.extend((triple.len() as u64).to_le_bytes());
            out.extend(triple.as_bytes());
            payload_at += payload.len();
        }
        for (_, payload) in entries {
            out.extend(*payload);
        }
        out
    }

    #[test]
    fn raw_elf_is_returned_unchanged() {
        assert_eq!(device_code_object(ELF, "gfx1201"), Some(ELF));
    }

    #[test]
    fn bundle_yields_the_requested_arch_payload() {
        let other = [0x7f, b'E', b'L', b'F', 9, 9];
        let image = bundle(&[
            ("host-x86_64-unknown-linux-gnu-", &b"HOST"[..]),
            ("hipv4-amdgcn-amd-amdhsa--gfx1100", &other[..]),
            ("hipv4-amdgcn-amd-amdhsa--gfx1201", ELF),
        ]);
        assert_eq!(device_code_object(&image, "gfx1201"), Some(&ELF[..]));
        assert_eq!(device_code_object(&image, "gfx1100"), Some(&other[..]));
    }

    #[test]
    fn bundle_arch_entry_with_feature_suffix_matches_but_longer_arch_does_not() {
        let image = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1201:xnack-", ELF)]);
        assert_eq!(device_code_object(&image, "gfx1201"), Some(&ELF[..]));
        let image = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx12010", ELF)]);
        assert_eq!(device_code_object(&image, "gfx1201"), None);
    }

    #[test]
    fn wrong_arch_bundle_is_none() {
        let image = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1100", ELF)]);
        assert_eq!(device_code_object(&image, "gfx1201"), None);
    }

    #[test]
    fn duplicate_arch_entries_are_none() {
        let image = bundle(&[
            ("hipv4-amdgcn-amd-amdhsa--gfx1201", ELF),
            ("hipv4-amdgcn-amd-amdhsa--gfx1201", ELF),
        ]);
        assert_eq!(device_code_object(&image, "gfx1201"), None);
    }

    #[test]
    fn arch_payload_that_is_not_an_elf_is_none() {
        let image = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1201", &b"not an elf"[..])]);
        assert_eq!(device_code_object(&image, "gfx1201"), None);
    }

    #[test]
    fn malformed_inputs_are_none() {
        assert_eq!(device_code_object(&[], "gfx1201"), None);
        assert_eq!(device_code_object(b"junk", "gfx1201"), None);
        assert_eq!(device_code_object(MAGIC, "gfx1201"), None);
        let image = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1201", ELF)]);
        for cut in [
            MAGIC.len() + 4,
            MAGIC.len() + 8,
            MAGIC.len() + 20,
            image.len() - 1,
        ] {
            assert_eq!(
                device_code_object(&image[..cut], "gfx1201"),
                None,
                "truncated at {cut}"
            );
        }
    }

    #[test]
    fn overflowing_fields_are_none_without_panicking() {
        let base = bundle(&[("hipv4-amdgcn-amd-amdhsa--gfx1201", ELF)]);
        let header = MAGIC.len();
        let entry = header + 8;
        for (at, label) in [
            (header, "count"),
            (entry, "offset"),
            (entry + 8, "size"),
            (entry + 16, "triple len"),
        ] {
            for value in [u64::MAX, u64::MAX - 7, 1 << 63] {
                let mut image = base.clone();
                image[at..at + 8].copy_from_slice(&value.to_le_bytes());
                assert_eq!(
                    device_code_object(&image, "gfx1201"),
                    None,
                    "{label} = {value:#x}"
                );
            }
        }
        let mut image = base;
        image[entry..entry + 8].copy_from_slice(&(u64::MAX - 3).to_le_bytes());
        image[entry + 8..entry + 16].copy_from_slice(&8u64.to_le_bytes());
        assert_eq!(
            device_code_object(&image, "gfx1201"),
            None,
            "offset + size wraps"
        );
    }

    #[test]
    fn compressed_bundle_is_none() {
        let mut image = b"CCOB".to_vec();
        image.extend([2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        image.extend(ELF);
        assert_eq!(device_code_object(&image, "gfx1201"), None);
    }
}

#[cfg(test)]
mod scratch_growth_epoch_tests {
    use super::{note_scratch_growth, scratch_growth_epoch};

    /// Every announced or performed scratch growth moves the epoch the railgun
    /// window caches key on.
    #[test]
    fn scratch_growth_moves_the_epoch() {
        let before = scratch_growth_epoch();
        note_scratch_growth();
        assert!(scratch_growth_epoch() > before);
        let mid = scratch_growth_epoch();
        note_scratch_growth();
        assert!(scratch_growth_epoch() > mid);
    }
}
