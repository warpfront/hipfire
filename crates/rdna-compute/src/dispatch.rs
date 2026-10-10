// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! High-level GPU dispatch interface.
//! Manages compiled kernels, provides typed tensor operations.

use crate::compiler::KernelCompiler;
use crate::feature_flags::FeatureFlags;
use crate::kernels;
use hip_bridge::{
    DeviceBuffer, HipError, HipMemAllocationProp, HipMemGenericAllocationHandle, HipResult,
    HipRuntime, Rocblas, VmmArena, VmmPhysicalId, VmmSharedPrefix,
    HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Per-group byte size of the MQ3-Lloyd quantization layout.
///
/// 16 B fp16 codebook (8 entries) + 96 B 3-bit packed indices = 112 B.
/// Compare to HFQ3 / uniform MQ3's 104 B/group (8 B affine header).
///
/// Every Lloyd-MQ3 dispatch arm references this constant; **never use a
/// literal 112 in dispatch.rs** — keeping the named constant lets a
/// future review grep `\* 1(04|12)` and find any Lloyd-related hits as
/// stride-mismatch bugs (followup discipline from
/// docs/plans/mq-lloyd-batched-prefill-followup.md).
pub const LLOYD_MQ3_GROUP_BYTES: usize = 112;

/// Per-group byte size of the MQ4-Lloyd quantization layout.
///
/// 32 B fp16 codebook (16 entries) + 128 B 4-bit nibble-pair indices = 160 B.
/// Compare to HFQ4 / uniform MQ4's 136 B/group (8 B affine header).
///
/// Every Lloyd-MQ4 dispatch arm references this constant; **never use a
/// literal 160 in dispatch.rs** — keeping the named constant lets a
/// future review grep `\* 1(36|60)` and find any Lloyd-related hits as
/// stride-mismatch bugs (followup discipline from
/// docs/plans/mq-lloyd-batched-prefill-followup.md).
pub const LLOYD_MQ4_GROUP_BYTES: usize = 160;

/// HIP `hipDeviceAttribute_t` ordinal for `hipDeviceAttributeIntegrated`
/// ("Device is integrated GPU"). Pinned by enumerating the CUDA-compatible
/// block in the local `hip_runtime_api.h`; the same enumeration reproduces
/// the repo's existing pins (`HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` =
/// 63 in `profiler.rs`). Ordinals HAVE shifted between ROCm releases before
/// (`ReservedSharedMemPerBlock` was inserted ahead of
/// `MaxSharedMemoryPerBlock`, moving it 74 → 75), so re-verify on any ROCm
/// bump — see [`Gpu::is_uma`], the only consumer.
const HIP_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 16;

/// Process-wide host→device upload counter, read through [`Gpu::htod_uploads`].
/// Diagnostics only: a per-call constant upload in a hot loop is invisible in a
/// kernel budget but shows up here.
static HTOD_UPLOADS: AtomicUsize = AtomicUsize::new(0);

// ── MQ*-GL ("global Lloyd") format constants ────────────────────────────
//
// GL = one codebook shared by the whole tensor plus a per-block fp16 scale,
// laid out as two SoA regions rather than an interleaved per-group header:
//
//   MQ2G256GL (qt 38): 64 B indices/group, then fp16 scales
//   MQ3G256GL (qt 39): 96 B indices/group, then fp16 scales
//
// Both use gpr = K/256. There is no inline per-group header.

/// Per-group INDEX bytes for MQ2-G256-GL (2 bits × 256 weights). The fp16
/// per-block scale lives in a SEPARATE trailing region, NOT in the group.
pub const GL_MQ2_GROUP_IDX_BYTES: usize = 64;

/// Per-group INDEX bytes for MQ3-G256-GL (3 bits × 256 weights). Same SoA
/// split as MQ2-GL — the scale is in the trailing region, not inline.
pub const GL_MQ3_GROUP_IDX_BYTES: usize = 96;

/// Per-group bytes for MQ4-G256 v2 (qt=44): 136 B/group, byte-identical to
/// MQ4G256 (qt=13) except the 8 header bytes. Payload unchanged: 128 B of
/// 4-bit nibbles at offset 8, lane `t` reading the u32 at `8 + 4*t`,
/// covering weights `8t..8t+7`. Header layout:
/// `[0..2)` fp16 scale half0 (weights 0-127), `[2..4)` fp16 zero half0,
/// `[4..6)` fp16 scale half1 (weights 128-255), `[6..8)` fp16 zero half1.
/// Little-endian, low 16 bits scale, high 16 zero within each dword,
/// half-wave uniform, lane-invariant scalar loads. `K % 256 == 0`.
pub const MQ4V2_GROUP_BYTES: usize = 136;

/// Per-group bytes for MQ4-G128 v2 (qt=53), a row-local format admitted by
/// typed Qwen4 sealed/dense consumers.
///
/// Generic GPU consumers still reject qt=53. Keep this constant here so
/// runtime extent checks share the wire size without aliasing it to the legacy
/// 72-byte `MQ4G128` layout.
pub const MQ4G128V2_GROUP_BYTES: usize = 68;

/// Per-group bytes for MQ4-G256-C (qt=45): 136 B/group, 4.25 bpw, byte-identical
/// payload to MQ4G256 (qt=13) at the same offset. Pad layout (NOT the earlier
/// 132 B planar layout):
/// per group, 136 B stride: `[0..4)` fp16 header, `[4..8)` zero padding, `[8..136)` 128 B nibbles.
/// Header is ONE packed dword, low 16 bits fp16 scale, high 16 bits fp16 zero,
/// governing all 256 weights (`w = q * scale + zero`), unlike qt=44's dual half-grids.
/// The 4 padding bytes are the deliberate price of putting the payload at +8 where
/// v1 has it: a 132 B stride left the payload 4-byte aligned half the time and cost
/// 7-11% prefill on `global_load_b128`. The pad layout is the same size as v1
/// (136 B/group, `m*gpr*136` per tensor), so the 2.43% size win is deliberately given up.
/// Little-endian, lane-invariant scalar loads. See `kernels/src/gemv_mq4cpad_residual.hip`
/// and `kernels/src/gemm_mq4cpad_residual_wmma_gfx12_bt.hip`.
pub const MQ4C_GROUP_BYTES: usize = 136;

/// Per-group bytes for MQ6-G256 v2 (qt=47): 200 B/group, 6.25 bpw.
/// Neutral-size Magnum V2 layout: LE `[0..2)` fp16 s0, `[2..4)` fp16 z0,
/// `[4..6)` fp16 s1, `[6..8)` fp16 z1, `[8..200)` 192 B legacy 6-bit payload
/// (4 weights per 3 bytes, `4/3 B`). Half 0 governs q[0..128), half 1
/// q[128..256); reconstruction `q * f32(s[h]) + f32(z[h])`. `K % 256 == 0`.
pub const MQ6G256V2_GROUP_BYTES: usize = 200;

/// Per-group bytes for MQ5-G256 v2 (qt=48): 168 B/group, 5.25 bpw.
/// Neutral-size Magnum V2 layout: LE `[0..2)` fp16 s0, `[2..4)` fp16 z0,
/// `[4..6)` fp16 s1, `[6..8)` fp16 z1, `[8..168)` 160 B legacy 5-bit payload
/// (8 weights per 5 bytes, `8/5 B`). Half semantics as MQ6G256V2.
pub const MQ5G256V2_GROUP_BYTES: usize = 168;

/// Per-group bytes for MQ3-G256 v2 (qt=49): 104 B/group, 3.25 bpw.
/// Neutral-size Magnum V2 layout: LE `[0..2)` fp16 s0, `[2..4)` fp16 z0,
/// `[4..6)` fp16 s1, `[6..8)` fp16 z1, `[8..104)` 96 B legacy 3-bit payload
/// (8 weights per 3 bytes, `8/3 B`). Half semantics as MQ6G256V2.
pub const MQ3G256V2_GROUP_BYTES: usize = 104;

/// Per-group bytes for MQ2-G256 v2 (qt=50): 72 B/group, 2.25 bpw.
/// Neutral-size Magnum V2 layout: LE `[0..2)` fp16 s0, `[2..4)` fp16 z0,
/// `[4..6)` fp16 s1, `[6..8)` fp16 z1, `[8..72)` 64 B legacy 2-bit payload
/// (4 weights per byte). Half semantics as MQ6G256V2.
pub const MQ2G256V2_GROUP_BYTES: usize = 72;

/// Bytes per group in the trailing fp16 scale region (both GL dtypes).
pub const GL_GROUP_SCALE_BYTES: usize = 2;

/// **MUST STAY BIT-IDENTICAL TO `hipfire-quantize::main::GL_CB2`.**
///
/// The MQ2-GL GEMV kernels take the codebook as four SCALAR FLOAT KERNEL ARGS
/// (`cb0..cb3`) — it is *not* stored in the `.hfq` file — so the runtime must
/// reproduce the exact levels the encoder quantized against. Encoder side:
/// `crates/hipfire-quantize/src/main.rs` (`GL_CB2`, consumed by
/// `gl_encode_block` from `quantize_mq2g256gl`).
///
/// A silent drift between the two arrays is a **silent accuracy failure**, not a
/// crash: every weight decodes to a plausible-but-wrong level and the model
/// degrades without any error. The coupling is machine-checked by
/// `gl_codebooks_match_runtime` in `hipfire-quantize/src/main.rs`; if you change
/// one array you MUST change the other and that test will tell you if you didn't.
///
/// Values are the textbook Lloyd–Max reconstruction levels for a unit Gaussian
/// (2-bit, MSE 0.1175), reproduced to 3 decimals by fitting on 28.3 M real a3b
/// post-FWHT expert weights (2026-08-04).
pub const GL_CB2: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];

/// **MUST STAY BIT-IDENTICAL TO `hipfire-quantize::main::GL_CB3`.**
///
/// 3-bit sibling of [`GL_CB2`]; passed to the MQ3-GL GEMV kernels as eight
/// scalar float kernel args (`cb0..cb7`, ascending). Same drift hazard, same
/// machine check (`gl_codebooks_match_runtime`). Textbook Lloyd–Max 3-bit
/// Gaussian levels (MSE 0.03454).
pub const GL_CB3: [f32; 8] = [
    -2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520,
];

/// Current layer index, set by the qwen35 forward_prefill_chunk at the
/// start of each layer iteration. Used by `hfq3_mmq_layer_gate_pass` to
/// support per-layer MMQ-on/off experiments (see issue #302 — KLD
/// attribution sweep). Default 0; no semantic meaning outside an
/// instrumented sweep.
pub static MMQ_CURRENT_LAYER: AtomicUsize = AtomicUsize::new(0);

fn pm4_dynamic_grid_enabled() -> bool {
    // Multi-spelling ("1"/"true"/"yes"/"on") predicate preserved verbatim;
    // only the process-global cache is gone (the snapshot is the cache now).
    hipfire_config::process_value("HIPFIRE_REPLAY_PM4_DYNAMIC_GRID")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

/// Dynamic fields a launch declares to the retained-replay recorder.
///
/// A declaration is how the lowering *names* a value that changes between
/// forwards instead of leaving it to be discovered by differencing two
/// recordings. The recorded bytes always hold the capture-position values; a
/// grid binding narrows the recorded maximum grid for the replay position, and
/// each kernarg binding re-derives one 4-byte scalar from that position.
#[derive(Clone, Copy, Default)]
pub struct ReplayLaunchBindings<'a> {
    pub grid: Option<crate::replay::ReplayGridBinding>,
    pub kernargs: &'a [crate::replay::ReplayKernargBinding],
}

impl ReplayLaunchBindings<'static> {
    /// No dynamic fields: replay the launch exactly as captured.
    pub const NONE: Self = Self {
        grid: None,
        kernargs: &[],
    };
}

/// A by-value kernel argument. Its kernarg-blob encoding is its native bytes at
/// natural alignment, which is the layout HIP builds from a `void**` params
/// array for the same kernel signature.
pub(crate) trait KernargValue: Copy {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob);
}

impl KernargValue for *mut c_void {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob) {
        blob.push_ptr(self);
    }
}
impl KernargValue for *const c_void {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob) {
        blob.push_ptr(self);
    }
}
impl KernargValue for i32 {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob) {
        blob.push_i32(self);
    }
}
impl KernargValue for u32 {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob) {
        blob.push_u32(self);
    }
}
impl KernargValue for f32 {
    fn push_to(self, blob: &mut hip_bridge::KernargBlob) {
        blob.push_f32(self);
    }
}

/// Native bytes of a kernel argument value (debug kernarg cross-check).
#[cfg(debug_assertions)]
pub(crate) fn kernarg_value_bytes<T: KernargValue>(v: &T) -> &[u8] {
    // SAFETY: `T` is one of the plain-old-data impls above; reading its
    // `size_of::<T>()` bytes through a shared reference is sound.
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Debug cross-check for [`launch_params_blob!`]: the `void**` params array and
/// the blob argument list name the same values in the same order, so HIP's
/// params-built kernargs and the captured blob are byte-identical.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_params_match(params: &[*mut c_void], values: &[&[u8]]) {
    assert_eq!(params.len(), values.len(), "kernarg count: params vs blob values");
    for (i, (p, v)) in params.iter().zip(values).enumerate() {
        // SAFETY: each params entry points at a live local at least as large
        // as the matching blob value (both lists name the same locals).
        let got = unsafe { std::slice::from_raw_parts(*p as *const u8, v.len()) };
        assert_eq!(got, *v, "kernarg {i}: params bytes differ from blob value");
    }
}

/// Launch through [`Gpu::launch_maybe_blob`] with the blob built from the same
/// locals the `params` array points at, listed in params order. Debug builds
/// assert both lists agree byte for byte at every launch.
macro_rules! launch_params_blob {
    ($gpu:expr, $name:expr, $grid:expr, $block:expr, $smem:expr, $params:ident; $($v:ident),+ $(,)?) => {{
        #[cfg(debug_assertions)]
        $crate::dispatch::debug_assert_params_match(
            &$params,
            &[$($crate::dispatch::kernarg_value_bytes(&$v)),+],
        );
        $gpu.launch_maybe_blob($name, $grid, $block, $smem, &mut $params, || {
            let mut b = hip_bridge::KernargBlob::new();
            $($crate::dispatch::KernargValue::push_to($v, &mut b);)+
            b
        })
    }};
}
pub(crate) use launch_params_blob;

/// Resolve the compiled artifact that owns a launched symbol.
///
/// `KernelCompiler::compiled_kernels()` is keyed by module name while the
/// recorder stores the launched *function* name, so runtime-specialized
/// variants resolve through this one alias table. `None` means the artifact is
/// unknown and preparation will reject the tape rather than guess.
pub(crate) fn recorded_launch_artifact(
    compiler: &KernelCompiler,
    func_name: &str,
) -> Option<std::path::PathBuf> {
    let compiled = compiler.compiled_kernels();
    compiled
        .get(func_name)
        .or_else(|| match func_name {
            "mq_rotate_x" => compiled.get("gemv_mq4g256"),
            "deinterleave_f32_batched" => compiled.get("deinterleave_batched"),
            name if name.starts_with("gemv_hfq4g256_residual_sigmoid_scaled_gpu") => {
                compiled.get("gemv_hfq4g256_residual_scaled")
            }
            "gemv_hfq4g256_moe_gate_up_k8_indexed" => {
                compiled.get("gemv_hfq4g256_moe_gate_up_indexed")
            }
            name if name.starts_with("gemv_hfq4g256_multirow_r") => compiled
                .get("gemv_hfq4g256_multirow_default")
                .or_else(|| compiled.get("gemv_hfq4g256_multirow_rdna3")),
            name if name.starts_with("gemv_hfq4g256_residual_multirow_r") => compiled
                .get("gemv_hfq4g256_residual_multirow_default")
                .or_else(|| compiled.get("gemv_hfq4g256_residual_multirow_rdna3")),
            _ => None,
        })
        .or_else(|| {
            func_name
                .strip_suffix("_f32")
                .and_then(|name| compiled.get(name))
        })
        .cloned()
}

/// Minimum batch size at which the FP8 WMMA prefill path is enabled.
/// Below this, the FP16 WMMA path wins on gfx1201 (measured 0.71-0.94×
/// at N ≤ 512, 0.82-1.26× only at N ≥ 2048 with high DPM variance —
/// see project_fp8_wmma_hfp4g32_2026_05_10.md). Decode (batch_size=1)
/// must never hit FP8 WMMA. Threshold tuned conservatively; A/B against
/// FP16 WMMA on the production prefill bench can lower it later.
pub(crate) const FP8_WMMA_MIN_BATCH: usize = 1024;

/// Slack mapped past the end of every host-located weight tensor.
///
/// Forward-path GEMV/attention kernels read a few bytes beyond each tensor blob —
/// a phantom group header or a vector tail load. The device path absorbs that into
/// `hipMalloc` slack, but an exact-fit host reservation makes the first overread
/// the first unmapped byte and faults with "Page not present" (every real MQ3 blob
/// is an exact 4096-byte multiple, so the slack is exactly zero precisely when it
/// matters). The overread provably does not affect output: device-path
/// byte-identity runs produce identical results with or without it.
///
/// 1 MiB is generous relative to the observed overread (bytes to one group header)
/// and now costs host RAM only, so there is no reason to shave it.
const HOST_TAIL_PAD_BYTES: usize = 1024 * 1024;

// AR-forward hipGraph policy (2026-05-15, after `<think>\n!!!!!` attractor
// debug on Qwen3.5-27B mq4 gfx1100):
//
//   - `ar_forward_kernel_dirty`: true on init / after kernel module change.
//     Forces direct dispatch on the very first call so any inline JIT or
//     lazy hipMalloc happens outside a captured region.
//   - `ar_forward_replay_enabled`: true only after the caller has signalled
//     `end_decode_turn()` AND a capture exists AND kernels are not dirty.
//     Until then, every forward call captures a fresh graph and launches it
//     (correct output per call; cheaper than full direct on amortization).
//
// Why caller-driven commit instead of auto-enable: empirically, captured
// graphs on this codebase + ROCm 7.2.2 sometimes snapshot stale kernarg
// state mid-decode, producing a token-0 attractor on every replay. Gating
// replay until a FULL decode turn completes via the captured-launch path
// gives the captured graph the longest possible runway to be invalidated
// by JIT recompilation; if a turn finishes coherently with capture+launch,
// the same graph is more likely to replay coherently on the next turn.

/// Minimum output dimension M at which the FP8-dot4 decode GEMV path
/// is enabled. Below this, the fallback wins or ties on gfx1201
/// (measured 0.92-1.03× on wo M=2048 K=2048 vs 1.17-1.21× on FFN
/// shapes M ≥ 4096 — see mq_rotate_x_dual_fp8 bench, 2026-05-11).
/// This is the empirical embodiment of "Option α" mixed-precision
/// routing — choose the kernel that wins for the actual shape rather
/// than uniformly applying FP8 everywhere.
pub(crate) const FP8_GEMV_MIN_M: usize = 4096;

/// Tensor stored on the GPU. Tracks shape and element type.
pub struct GpuTensor {
    pub buf: DeviceBuffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// A `GpuTensor` whose buffer is a null pointer of size 0, for CPU-only unit
    /// tests in **dependent crates** that read only tensor metadata (shape/dtype/op)
    /// and never touch the device.
    ///
    /// CONTRACT: the returned tensor must NEVER be passed to a HIP call — its buffer
    /// is null and dereferencing it on the GPU is undefined behavior. It exists only
    /// so cross-crate tests can borrow a `&GpuTensor` for metadata-only logic.
    ///
    /// Not `#[cfg(test)]`-gated on purpose: `#[cfg(test)]` here would only be active
    /// when `rdna-compute`'s own tests build, making this invisible to dependent
    /// crates' tests (e.g. `hipfire-dispatch`). `#[doc(hidden)]` keeps it out of the
    /// public API surface while remaining reachable cross-crate, matching the
    /// `FeatureFlags::for_test` precedent.
    #[doc(hidden)]
    pub fn null_for_test() -> Self {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(std::ptr::null_mut::<std::ffi::c_void>(), 0)
            },
            shape: vec![0],
            dtype: crate::DType::F32,
        }
    }

    /// Create a non-owning sub-view at a byte offset. For F32 tensors,
    /// `offset_elems` is the number of f32 elements to skip.
    /// The returned tensor is a view — do NOT free it.
    pub fn sub_offset(&self, offset_elems: usize, len_elems: usize) -> GpuTensor {
        let element_size = self.dtype.size();
        let byte_off = offset_elems
            .checked_mul(element_size)
            .expect("tensor sub-view offset overflow");
        let byte_len = len_elems
            .checked_mul(element_size)
            .expect("tensor sub-view length overflow");
        let byte_end = byte_off
            .checked_add(byte_len)
            .expect("tensor sub-view range overflow");
        assert!(
            byte_end <= self.buf.size(),
            "tensor sub-view exceeds accessible buffer prefix"
        );
        let ptr = unsafe { (self.buf.as_ptr() as *mut u8).add(byte_off) as *mut std::ffi::c_void };
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, byte_len) },
            shape: vec![len_elems],
            dtype: self.dtype,
        }
    }

    /// Full-buffer non-owning alias (the whole-tensor form of `sub_offset`).
    /// The returned tensor shares the source's device pointer; it is a VIEW —
    /// do NOT pass it to `free_tensor`. `DeviceBuffer::from_raw` has no
    /// Drop-time free, so the alias and source coexist safely until the OWNER
    /// is freed exactly once.
    pub fn shallow_clone(&self) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(self.buf.as_ptr(), self.buf.size()) },
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

/// Byte alignment of every frontier entry inside a [`VmmPrefixSet`] slab.
const VMM_PREFIX_SLAB_ALIGN: usize = 256;

/// One registered VMM owner to checkpoint with [`Gpu::capture_vmm_prefixes`].
///
/// `valid_bytes` is the live prefix of the tensor (`<=` its mapped bytes);
/// `writable_from` is the first byte that a restore may still rewrite in place
/// (`<= valid_bytes`). Bytes below `writable_from` that fill whole leading
/// physical granules are aliased instead of copied.
pub struct VmmPrefixSpec<'a> {
    pub tensor: &'a GpuTensor,
    pub valid_bytes: usize,
    pub writable_from: usize,
}

/// Per-arena record of a [`VmmPrefixSet`].
struct VmmPrefixEntry {
    shape: Vec<usize>,
    dtype: DType,
    logical_bytes: usize,
    reserved_bytes: usize,
    valid_bytes: usize,
    writable_from: usize,
    alias_bytes: usize,
    slab_offset: usize,
    prefix: VmmSharedPrefix,
}

/// A retained prefix of several VMM owners: leases on the aliased leading
/// granules plus ONE packed slab holding the byte-exact frontier copies.
///
/// Release it with [`Gpu::free_vmm_prefixes`]; dropping it leaks the leases.
#[must_use = "a VmmPrefixSet holds physical granule leases; release it with Gpu::free_vmm_prefixes"]
pub struct VmmPrefixSet {
    entries: Vec<VmmPrefixEntry>,
    slab: Option<GpuTensor>,
    slab_bytes: usize,
}

impl VmmPrefixSet {
    /// Number of arenas in the set.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sum of aliased (shared, uncopied) bytes.
    pub fn alias_bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.alias_bytes).sum()
    }

    /// Sum of copied frontier bytes.
    pub fn frontier_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| entry.valid_bytes - entry.alias_bytes)
            .sum()
    }

    /// Actual slab allocation, including alignment padding (0 when nothing was copied).
    pub fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }

    /// Valid bytes recorded for arena `arena`. Panics when `arena >= len()`.
    pub fn valid_bytes(&self, arena: usize) -> usize {
        self.entries[arena].valid_bytes
    }

    /// Aliased bytes recorded for arena `arena`. Panics when `arena >= len()`.
    pub fn arena_alias_bytes(&self, arena: usize) -> usize {
        self.entries[arena].alias_bytes
    }

    /// First in-place-writable byte recorded for arena `arena`. Panics when `arena >= len()`.
    pub fn arena_writable_from(&self, arena: usize) -> usize {
        self.entries[arena].writable_from
    }

    /// Every aliased physical granule, arena by arena, in offset order.
    pub fn physical_granules(&self) -> impl Iterator<Item = (VmmPhysicalId, usize)> + '_ {
        self.entries
            .iter()
            .flat_map(|entry| entry.prefix.granules().iter().copied())
    }
}

/// Re-pointing of one VMM owner to its replacement for [`Gpu::relocate_qsa_resources`].
/// `reserved_bytes`, `mapped_bytes` and `owner_generation` describe the NEW owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmmResourceMove {
    pub old_base: usize,
    pub new_base: usize,
    pub reserved_bytes: usize,
    pub mapped_bytes: usize,
    pub owner_generation: u64,
}

/// Largest granule boundary `<= limit` over consecutive granule sizes in
/// offset order. A granule that would cross `limit` (or has no bytes) ends the walk.
fn vmm_alias_boundary(granule_sizes: impl IntoIterator<Item = usize>, limit: usize) -> usize {
    let mut end = 0usize;
    for size in granule_sizes {
        match end.checked_add(size) {
            Some(next) if size != 0 && next <= limit => end = next,
            _ => break,
        }
    }
    end
}

/// Packed slab layout: per-entry offsets (each [`VMM_PREFIX_SLAB_ALIGN`]-aligned)
/// and the total slab bytes. `None` when the layout overflows `usize`.
fn vmm_slab_layout(lens: &[usize]) -> Option<(Vec<usize>, usize)> {
    let mut offsets = Vec::with_capacity(lens.len());
    let mut cursor = 0usize;
    for &len in lens {
        offsets.push(cursor);
        cursor = cursor
            .checked_add(len)?
            .checked_next_multiple_of(VMM_PREFIX_SLAB_ALIGN)?;
    }
    Some((offsets, cursor))
}

/// Host-only shape checks for a relocation batch: non-empty extents, mapped
/// within reserved, no overflow, and no duplicate or overlapping old ranges
/// (or new ranges).
fn validate_vmm_move_ranges(moves: &[VmmResourceMove]) -> Result<(), String> {
    let mut old_ranges: Vec<(usize, usize)> = Vec::with_capacity(moves.len());
    let mut new_ranges: Vec<(usize, usize)> = Vec::with_capacity(moves.len());
    for (index, mv) in moves.iter().enumerate() {
        if mv.reserved_bytes == 0 {
            return Err(format!("VMM move {index}: reserved_bytes is zero"));
        }
        if mv.mapped_bytes > mv.reserved_bytes {
            return Err(format!(
                "VMM move {index}: mapped {} exceeds reserved {}",
                mv.mapped_bytes, mv.reserved_bytes
            ));
        }
        let old_end = mv.old_base.checked_add(mv.reserved_bytes).ok_or_else(|| {
            format!("VMM move {index}: old range 0x{:x}+{} overflows", mv.old_base, mv.reserved_bytes)
        })?;
        let new_end = mv.new_base.checked_add(mv.reserved_bytes).ok_or_else(|| {
            format!("VMM move {index}: new range 0x{:x}+{} overflows", mv.new_base, mv.reserved_bytes)
        })?;
        old_ranges.push((mv.old_base, old_end));
        new_ranges.push((mv.new_base, new_end));
    }
    for (label, ranges) in [("old", &mut old_ranges), ("new", &mut new_ranges)] {
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "VMM move batch has duplicate or overlapping {label} ranges: 0x{:x}..0x{:x} and 0x{:x}..0x{:x}",
                    pair[0].0, pair[0].1, pair[1].0, pair[1].1
                ));
            }
        }
    }
    Ok(())
}

/// Capture-time plan for one [`VmmPrefixSpec`].
struct VmmPrefixPlan {
    key: usize,
    mapped: usize,
    alias: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,     // 2 bytes; native bf16 reference (KLD oracle). Widen→f32 = high-16-bit shift.
    Q4K,      // 144 bytes per 256 elements
    Q6K,      // 210 bytes per 256 elements
    Q8_0,     // 34 bytes per 32 elements
    Q4F16G64, // 36 bytes per 64 elements (RDNA-native FP16 dequant)
    Q4F16G32, // 20 bytes per 32 elements (RDNA-native FP16 dequant)
    Q8HFQ,    // split-metadata: scales contiguous then values contiguous, 128B-aligned rows
    HFQ4G256, // 136 bytes per 256 elements (flat 4-bit, f32 scale+zero, 18 VGPRs)
    HFQ4G128, // 72 bytes per 128 elements (flat 4-bit, f32 scale+zero, 14 VGPRs)
    HFQ3G256, // 104 bytes per 256 elements (flat 3-bit, f32 scale+zero)
    HFQ3G128, // 56 bytes per 128 elements (flat 3-bit, f32 scale+zero)
    MQ4G256,  // MagnumQuant: FWHT-rotated HFQ4-G256 (136 bytes/group, same as HFQ4G256)
    /// MQ4-G256 v2 (qt=44): FWHT-rotated, 136 B/group, byte-identical payload to
    /// MQ4G256 except the 8 header bytes: `[0..2)` fp16 scale half0 (w 0-127),
    /// `[2..4)` fp16 zero half0, `[4..6)` fp16 scale half1 (w 128-255),
    /// `[6..8)` fp16 zero half1, `[8..136)` 128 B nibbles, identical to qt=13.
    /// Little-endian, low 16 scale / high 16 zero per dword, half-wave uniform,
    /// lane-invariant scalar loads. `K % 256 == 0`, 4.25 bpw.
    MQ4G256V2,
    /// MQ4-G256 v2 Lloyd (qt=52): FWHT-rotated, 136 B/group, wire layout
    /// BYTE-IDENTICAL to MQ4G256V2 (dual fp16 half-grids + 128 B nibbles).
    /// Nibble q decodes through a PER-TENSOR 16-level centered codebook
    /// `w = sc·(C[q]) + zp'` with `C[q] = L[q]−7.5` (sidecar `lloyd_levels`,
    /// f32[16] in [0,15] units, E4M3-snapped) and rewritten headers
    /// `zp' = fp16(zp+7.5·sc)` per half (loader-applied; the file keeps the
    /// uncentered zp). `K % 256 == 0`, 4.25 bpw. NEVER decoded by a uniform
    /// kernel: every dispatch arm must check the LUT variant / fail closed.
    MQ4G256V2Lloyd,
    /// MQ4-G256-C (qt=45): FWHT-rotated, 136 B/group, 4.25 bpw, pad layout:
    /// per group 136 B: `[0..4)` fp16 header (low scale, high zero), `[4..8)` zero padding,
    /// `[8..136)` 128 B nibbles at same offset as v1 (MQ4G256). ONE affine grid per 256
    /// (`w = q * scale + zero`), half-wave uniform, lane-invariant scalar loads.
    /// `K % 256 == 0`.
    MQ4CG256,
    /// MQ6-G256 v2 (qt=47): FWHT-rotated, 200 B/group, neutral Magnum V2.
    /// Per-group 200 B: `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
    /// `[6..8)` fp16 z1, `[8..200)` 192 B 6-bit payload (4/3 B). Half 0
    /// q[0..128), half 1 q[128..256); `w = q*f32(s[h])+f32(z[h])`.
    /// `K % 256 == 0`, 6.25 bpw.
    MQ6G256V2,
    /// MQ5-G256 v2 (qt=48): FWHT-rotated, 168 B/group, neutral Magnum V2.
    /// Per-group 168 B: `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
    /// `[6..8)` fp16 z1, `[8..168)` 160 B 5-bit payload (8/5 B). Same half
    /// semantics as MQ6G256V2. `K % 256 == 0`, 5.25 bpw.
    MQ5G256V2,
    /// MQ3-G256 v2 (qt=49): FWHT-rotated, 104 B/group, neutral Magnum V2.
    /// Per-group 104 B: `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
    /// `[6..8)` fp16 z1, `[8..104)` 96 B 3-bit payload (8/3 B). Same half
    /// MQ4-G128 v2 (qt=53): row-local FWHT-128, 68 B per `ceil(K/128)` group.
    /// Typed Qwen4 sealed/dense consumers use dedicated kernels; generic GPU
    /// consumers must reject it rather than treating it as legacy `MQ4G128`
    /// (72 B/group).
    MQ4G128V2,
    MQ4G128, // MagnumQuant: FWHT-128-rotated INT4 (72 bytes/group, same layout as HFQ4G128)
    MQ3G256V2,
    /// MQ2-G256 v2 (qt=50): FWHT-rotated, 72 B/group, neutral Magnum V2.
    /// Per-group 72 B: `[0..2)` fp16 s0, `[2..4)` fp16 z0, `[4..6)` fp16 s1,
    /// `[6..8)` fp16 z1, `[8..72)` 64 B 2-bit payload (4/B). Same half
    /// semantics as MQ6G256V2. `K % 256 == 0`, 2.25 bpw.
    MQ2G256V2,
    MQ8G256,      // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target (258 bytes/group)
    MQ6G256,      // MagnumQuant: FWHT-rotated HFQ6-G256 (200 bytes/group, same as HFQ6G256)
    MQ5G256,      // MagnumQuant: FWHT-rotated 5-bit (168 bytes/group, 5.25 bpw)
    MQ3G256,      // MagnumQuant: FWHT-rotated HFQ3-G256 (104 bytes/group, same as HFQ3G256)
    MQ2G256,      // MagnumQuant: FWHT-rotated HFQ2-G256 (72 bytes/group, same as HFQ2G256)
    MQ2G256Lloyd, // MagnumQuant 2-bit + Lloyd-Max 4-entry fp16 codebook (72 bytes/group)
    /// Unrotated MQ2-Lloyd (qt=51). Byte-identical to `MQ2G256Lloyd`
    /// (72 B/group: 4-entry fp16 codebook + 64 B of 2-bit indices), so the same
    /// kernels bind — but NOT FWHT-rotated. It consumes x in the natural basis,
    /// so `needs_x_rot_local` is false for it and it must never be added to the
    /// rotation chain. Carries native-ternary checkpoints (Maple-Preview)
    /// losslessly: rotation would destroy the three-value structure that lets a
    /// K=3 codebook be exact.
    MQ2G256LloydU,
    MQ3G256Lloyd, // MagnumQuant 3-bit + Lloyd-Max 8-entry fp16 codebook (112 bytes/group)
    MQ4G256Lloyd, // MagnumQuant 4-bit + Lloyd-Max 16-entry fp16 codebook (160 bytes/group)
    MQ2G256GL,    // MagnumQuant 2-bit + TENSOR-GLOBAL 4-entry codebook (GL_CB2), SoA:
    // [M*gpr*64 B indices][M*gpr*2 B fp16 per-block scales] = 2.0625 bpw. NOT
    // interleaved — no per-group header. Codebook is a compile-time constant
    // passed as scalar kernel args, not stored in the file. MoE-routed-expert
    // only (indexed gate_up + atomic self-combining down); there is no dense /
    // plain / prerotated single-weight GEMV kernel for this dtype.
    MQ3G256GL, // MagnumQuant 3-bit + TENSOR-GLOBAL 8-entry codebook (GL_CB3), SoA:
    // [M*gpr*96 B indices][M*gpr*2 B fp16 per-block scales] = 3.0625 bpw. Same
    // MoE-only scope + scalar-arg codebook as MQ2G256GL.
    HFP4G32, // HFP4: E2M1 element + UE8M0 g32 block scale + FP16 row scale.
    // Per-row header 16 B; per-block payload 17 B (UE8M0 + 16 packed nibbles).
    MFP4G32, // MFP4: HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same byte layout
    // as HFP4G32; format_flags bit 0 + bits 2-3 = 01 stamps the rotation kind.
    // Runtime applies the matching FWHT to x via mq_rotate_x; the kernel itself
    // is shared with HFP4G32.
    MFP4G32Lloyd, // mfp4 + per-tensor 16-entry fp16 Lloyd codebook. Same per-row byte
    // layout as MFP4G32, plus a 32-B codebook prefix before row 0. format_flags=0x05
    // (same FWHT rotation as MFP4G32); recon uses codebook[nibble] not E2M1_LUT.
    MFP4G32P, // mfp4+P — mfp4 (E2M1 + FP16 row scale + offline FWHT) with the per-32-block
    // UE8M0 scale promoted to E4M3 (FP8, non-power-of-2). Byte layout BYTE-IDENTICAL to
    // MFP4G32 (NO prefix); only the per-block scale byte's decode differs (E4M3 vs UE8M0).
    MFP4G32E8, // mfp4-E8: mfp4+P container (E4M3 block scale, NO prefix, same row_bytes)
    // with the per-32-block 16 E2M1 nibbles replaced by 4x32-bit E8-lattice codewords
    // (8 weights/codeword, QUANT_STEP=0.88). 4.25 bpw, byte-IDENTICAL footprint to MFP4G32P.
    MFP4G32E8SOA, // mfp4-E8 SoA: same E8 data as MFP4G32E8 permuted for coalesced reads.
    // Per-row: [16B hdr: row_scale_a:f16@0, n_blocks:u16@4, flag:0x06@6]
    //   + [n_blocks B: E4M3 scales, pad to 16B boundary]
    //   + [n_blocks*16 B: 4xu32 E8 codewords/block, 16B-aligned].
    // Pure byte-permutation of MFP4G32E8 => dequant result IDENTICAL.
    MFP3G32E8, // mfp3-E8: MFP4G32E8 frame, 3-bit lattice (center 4), 13 B/blk, 104 B/grp, 3.25 bpw. Drop-in for MQ3G256Lloyd.
    MFP2G32E8, // mfp2-E8: MFP4G32E8 frame, 2-bit lattice (center 2),  9 B/blk,  72 B/grp, 2.25 bpw. Drop-in for MQ2G256Lloyd.
    MFP4G32E8G128, // qt=42 mfp4-E8 with a 128-wide FWHT rotation group: the MFP4G32E8
    // wire layout BYTE-FOR-BYTE (16-B hdr + (K/32) x 17 B blocks, no prefix, no padding),
    // but the encoder rotates 128-element segments (sign seeds 43/1043) instead of 256.
    // That is the only thing 256 was doing for this family, and it is what excludes
    // K=640 = 5x128, so this tier covers qwen4's expert `down_proj` axis. The rotation
    // width is NOT in the bytes: the row stride is identical to MFP4G32E8, and the
    // decoder is the same E8-lattice body. Activation side must use the G128 basis
    // (`rotate_x_mq_128`), the same one MQ4G128V2 consumes.
    HFQ2G256, // 72 bytes per 256 elements (flat 2-bit, f32 scale+zero, ~19 VGPRs)
    HFQ2G128, // 40 bytes per 128 elements (flat 2-bit, f32 scale+zero)
    TQ2G128,  // ternary Bonsai-27B: 34 bytes per 128 elements (flat 2-bit ternary, group 128)
    // Phase 4: ternary kernels wire GPU decode/dispatch; this Task 7 slice is
    // CPU-foundation only (variant + byte-size + RawCodec load mapping).
    BQ1G128,    // binary Bonsai-27B: 18 bytes per 128 elements (flat 1-bit sign, group 128)
    HFQ6G256,   // 200 bytes per 256 elements (6-bit, f32 scale+zero)
    ParoQ4G128, // ParoQuant: AWQ-packed INT4 G128 repacked to HFQ4G128 layout at load.
    // Weights are standard HFQ4G128 (72 bytes/group); the ParoQuant distinction
    // is that weight_gemv applies Givens rotation to activations before GEMV.
    // Rotation metadata (pairs, theta, channel_scales) lives on WeightTensor::paro.
    Raw, // raw bytes, no element interpretation
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4K
            | DType::Q6K
            | DType::Q8_0
            | DType::Q4F16G64
            | DType::Q4F16G32
            | DType::Q8HFQ
            | DType::HFQ4G256
            | DType::HFQ4G128
            | DType::HFQ3G256
            | DType::HFQ3G128
            | DType::HFQ2G256
            | DType::HFQ2G128
            | DType::TQ2G128
            | DType::BQ1G128
            | DType::HFQ6G256
            | DType::MQ4G256
            | DType::MQ4G256V2
            | DType::MQ4G128V2
            | DType::MQ4G256V2Lloyd
            | DType::MQ4CG256
            | DType::MQ6G256V2
            | DType::MQ5G256V2
            | DType::MQ3G256V2
            | DType::MQ2G256V2
            | DType::MQ4G128
            | DType::MQ6G256
            | DType::MQ5G256
            | DType::MQ8G256
            | DType::MQ3G256
            | DType::MQ2G256
            | DType::MQ2G256Lloyd
            | DType::MQ2G256LloydU
            | DType::MQ3G256Lloyd
            | DType::MQ4G256Lloyd
            | DType::MQ2G256GL
            | DType::MQ3G256GL
            | DType::HFP4G32
            | DType::MFP4G32
            | DType::MFP4G32Lloyd
            | DType::MFP4G32P
            | DType::MFP4G32E8
            | DType::MFP4G32E8SOA
            | DType::MFP3G32E8
            | DType::MFP2G32E8
            | DType::MFP4G32E8G128
            | DType::ParoQ4G128
            | DType::Raw => 1, // byte-level
        }
    }

    /// Encoded bytes of one row of `k` logical values, or `None` for a dtype
    /// without a fixed per-row layout here (or `k == 0`).
    ///
    /// Grouped formats store whole groups: a trailing partial group occupies a
    /// full group. Whether a caller accepts such a row (external rows and
    /// several kernels require `k` to be group-aligned) is the caller's
    /// policy, not the layout's.
    pub fn row_bytes(self, k: usize) -> Option<usize> {
        if k == 0 {
            return None;
        }
        let grouped = |group: usize, group_bytes: usize| k.div_ceil(group).checked_mul(group_bytes);
        let blocks = k.div_ceil(32);
        match self {
            DType::F32 | DType::F16 | DType::BF16 => k.checked_mul(self.size()),
            DType::Q8_0 => grouped(32, 34),
            DType::HFQ4G256 | DType::MQ4G256 | DType::MQ4G256V2 => grouped(256, 136),
            DType::MQ4G128V2 => grouped(128, 68),
            DType::HFQ6G256 | DType::MQ6G256 | DType::MQ6G256V2 => grouped(256, 200),
            DType::MQ3G256 => grouped(256, 104),
            DType::MQ2G256 | DType::MQ2G256Lloyd => grouped(256, 72),
            DType::MQ3G256Lloyd => grouped(256, 112),
            DType::MQ4G256Lloyd => grouped(256, 160),
            // HFQ4-G128 layout, see the variant.
            DType::ParoQ4G128 => grouped(128, 72),
            // E8 formats: a 16-byte row header, then per-32 blocks.
            DType::MFP4G32E8 => blocks.checked_mul(17)?.checked_add(16),
            DType::MFP3G32E8 => blocks.checked_mul(13)?.checked_add(16),
            // SoA: header, one scale byte per block padded to 16, 16 bytes of
            // nibbles per block.
            DType::MFP4G32E8SOA => blocks
                .checked_add(15)
                .map(|scales| scales & !15)?
                .checked_add(blocks.checked_mul(16)?)?
                .checked_add(16),
            _ => None,
        }
    }

    /// Whether a `WeightTensor` of this dtype should have the
    /// `<weight>.awq_scale.weight` F16 sidecar attached at load time.
    ///
    /// Centralizes the gate that previously lived inline at every
    /// loader call site (qwen35.rs `load_weight_tensor`, etc.). The
    /// motivation is the May 2026 regression where `qwen35.rs:907`
    /// gated on `matches!(wt.gpu_dtype, DType::MQ4G256)` and silently
    /// dropped AWQ sidecars for `MQ3G256`-quantized Qwen3.5 weights,
    /// producing fluent-but-nonsensical token soup for ~5 hours
    /// before the missing arm was traced. Adding a new AWQ-eligible
    /// dtype is now a one-line edit here instead of two scattered
    /// edits per loader.
    ///
    /// Current allow-list = the empirical truth of which dtypes ship
    /// AWQ sidecars from the quantizer AND have an AWQ-aware forward
    /// path (`rotate_x_mq_for` etc., wired through `awq_scale.is_some()`).
    ///
    /// **Forward-path-ready candidates not currently in the allow-list**
    /// (forward kernels exist but no `.hfq` file in tree ships sidecars
    /// for them — widen only after the quantizer side is verified to
    /// emit sidecars and at least one coherence-gate row exercises the
    /// combination):
    /// - `MQ6G256`
    /// - `MQ2G256`, `MQ2G256Lloyd`
    /// - `MQ3G256Lloyd`
    /// - `MFP4G32` (forward path has explicit `awq_scale.is_some()`
    ///   branching at llama.rs:609 but the quantizer comment says
    ///   "AWQ is gated to MQ4G256 today" — confirm before widening)
    ///
    /// `MQ8G256` is explicitly **not** a candidate: it uses its own
    /// INT8-quantized scratch path (`gemv_mq8g256_with_rotate`,
    /// `rotate_quantize_x_mq8`) and does not flow through
    /// `rotate_x_mq_for`, so there is no AWQ-aware kernel to dispatch
    /// to.
    ///
    /// **lm_head / embed_tokens callers:** as of the lm_head-AWQ
    /// runtime PR, this helper IS safe for the `output` weight in
    /// `qwen35.rs::load_weights` / `load_weights_vl`. Both dispatch
    /// paths that consume `weights.output` now route through
    /// AWQ-aware rotations when a sidecar is attached:
    /// - Decode: `weight_gemv` → `rotate_x_mq_for` (llama.rs)
    /// - Spec-decode verify: `speculative.rs::rotate_x_mq_batched_for`
    ///
    /// Pre-runtime-fix, attaching a sidecar on lm_head would have
    /// produced `(W·s)·x ≠ W·x` via the spec-verify path's plain
    /// `rotate_x_mq_batched` and driven the KLD 0.67 → 13.5
    /// corruption documented at `docs/plans/awq_fix_claude.md`. The
    /// quantizer-side `awq_eligible` whitelist
    /// (`hipfire-quantize/src/main.rs:3849`) still gates which
    /// tensors actually receive `W' = W·s` pre-multiplication at
    /// quant time — this helper governs only whether the loader
    /// attaches an already-emitted sidecar.
    pub fn supports_awq_sidecar(self) -> bool {
        // MQ3G256Lloyd / MQ2G256Lloyd added 2026-05-28: they are "forward-path-ready"
        // (flow through rotate_x_mq_for, which applies x/=awq_scale when a sidecar is
        // attached) — see the doc block above. Enables AWQ×Lloyd composition once the
        // quantizer emits sidecars for the Lloyd arms.
        matches!(
            self,
            DType::MQ4G256
                // qt=44 shares qt=13's AWQ contract exactly: the quantizer
                // pre-scales weights by `s` and emits the sidecar, and the forward
                // path divides x by `s` in the rotate step, so `(W·s)·(x/s) = W·x`.
                // Omitting it here does not fail — it silently drops the sidecar and
                // computes `(W·s)·x`, a per-channel scale error on every projection.
                // That is the May 2026 regression this predicate was centralised to
                // prevent; qt=44's artifact carries 496 sidecars.
                | DType::MQ4G256V2
                // qt=52 shares qt=44's AWQ contract exactly (same pipeline,
                // same rotate-step x/s division). Omitting it silently drops
                // the sidecar → (W·s)·x scale error on every projection.
                | DType::MQ4G256V2Lloyd
                // qt=45 shares qt=13/qt=44's AWQ contract exactly — same failure mode
                // if omitted: silent sidecar drop → (W·s)·x. Include it.
                | DType::MQ4CG256
                // Neutral V2 family (qt47-50): same per-half fp16 scale+zero header,
                // same AWQ pre-scale contract as qt44. All four must be listed or
                // their sidecars are silently dropped → same May-2026 regression.
                | DType::MQ6G256V2
                | DType::MQ5G256V2
                | DType::MQ3G256V2
                | DType::MQ2G256V2
                | DType::MQ3G256
                | DType::MQ2G256
                | DType::MQ3G256Lloyd
                | DType::MQ2G256Lloyd
        )
    }

    /// Per-row byte stride for split-metadata layouts. Q8HFQ packs `n_groups*2`
    /// scale bytes + `k` value bytes per row, padded to a 128-byte boundary; all
    /// other dtypes encode their layout internally and ignore this value (returns 0).
    /// Single source of truth for the formula previously inlined at hfq.rs:748.
    pub fn row_stride(self, k: usize) -> usize {
        match self {
            DType::Q8HFQ => {
                let raw_row = (k / 32) * 2 + k;
                (raw_row + 127) & !127
            }
            _ => 0,
        }
    }

    /// Whether this format's GEMV kernel requires K%256==0 (HFP4 family: the
    pub fn requires_k_mod_256(self) -> bool {
        matches!(
            self,
            DType::HFP4G32
                | DType::MFP4G32
                | DType::MQ2G256GL
                | DType::MQ3G256GL
                | DType::MQ4G256V2
                | DType::MQ4G256V2Lloyd
                | DType::MQ4CG256
                | DType::MQ6G256V2
                | DType::MQ5G256V2
                | DType::MQ3G256V2
                | DType::MQ2G256V2
        )
    }
}

/// Activation-capture hook for the Tier 1 hipfire-native calibration path.
///
/// Foundation scaffold (2026-05-19) — the field on `Gpu` is set by
/// `collect_imatrix` / `collect_hessian` (see
/// `crates/hipfire-runtime/src/bin/`) and called from each linear-layer
/// dispatch site to feed activations into an on-GPU reduction
/// (per-channel `Σ act²` for imatrix, K×K outer-product for the GPTQ
/// Hessian).
///
/// `tensor_name` is the canonical hipfire tensor identifier (the same
/// string the .hfq loader uses, e.g. `model.layers.0.self_attn.q_proj`)
/// so the reduction kernel can key its on-GPU accumulator dictionary
/// by name without ambiguity across MoE expert indices.
///
/// `input_ptr` / `numel` / `dtype` describe the activation tensor in
/// HBM at the moment of the linear-layer dispatch. The capture
/// implementation is responsible for launching its own reduction
/// kernel on the same stream as the producing GEMM (so ordering is
/// preserved without an extra `hipDeviceSynchronize`). The hook MUST
/// NOT free or reallocate the input tensor.
///
/// `Send + Sync` lets the same handler be shared across multi-GPU
/// dispatch threads (one `Gpu` instance per device, all pointing at
/// the same Arc'd handler that funnels into a per-tensor accumulator).
pub trait ActivationCapture: Send + Sync {
    /// Called by linear-layer dispatch arms when calibration is active.
    ///
    /// `gpu`         — the dispatcher, so the collector can run its on-GPU
    ///                 reduction kernels (`calib_sumsq_reduce_f32` /
    ///                 `calib_hessian_outer_f32`). Safe to take `&mut Gpu`:
    ///                 the dispatch site clones the collector `Arc` before
    ///                 calling, so `gpu.active_capture` is not aliased here.
    /// `tensor_name` — canonical .hfq / GGUF tensor name (resolved from the
    ///                 weight buffer pointer via `gpu.capture_names`).
    /// `input`       — the input-activation buffer (borrowed; do NOT retain past
    ///                 the call). NOTE its `.shape` may be a shared scratch sized
    ///                 to `max(dim, hidden)`, so it is NOT a reliable source of
    ///                 `k`/`n` — use the passed `n`/`k` instead.
    /// `n`           — number of activation rows (tokens / batch) this call.
    /// `k`           — the linear's input dim (the meaningful width of each row).
    /// Interior mutability (`&self`) lets the collector accumulate without an
    /// exclusive borrow.
    fn capture(&self, gpu: &mut Gpu, tensor_name: &str, input: &GpuTensor, n: usize, k: usize);
}

/// Per-weight MMQ screening state (issue #87).
pub struct MmqScreenState {
    pub cache: HashMap<usize, bool>,
    pub enabled: bool,
    pub threshold: f32,
}

/// Delta buffer offered to the next residual GEMM by [`Gpu::arm_residual_fold`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualFoldArm {
    pub delta: *mut c_void,
    /// Capacity of `delta` in f32 elements.
    pub len: usize,
}

/// A residual add deferred by the gfx1151 V2B GEMM: `y[n][m] += delta[n][m]`
/// is owed, and the next IU4 RMSNorm over `y` performs it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualFold {
    pub y: *mut c_void,
    pub delta: *mut c_void,
    pub m: usize,
    pub n: usize,
}

/// High-level GPU context. Owns the HIP runtime, compiler, and loaded kernels.
///
/// Tape completeness invariant: for any body executed via `Gpu`,
/// `self.graphs.capture_blobs.len()` after a HipGraph capture must equal
/// `self.replay.recorded_launches().len()` after a `ReplayController` capture
/// of the same body. Every kernel launch reachable from the four
/// `self.scratch.*` helpers (`ensure_fp16_x`, `convert_fp16_x_uncached`,
/// `ensure_fp8_x`, `ensure_q8_1_mmq_x`) is recorded through the unified
/// `launch_maybe_blob` gate so the two tapes stay in lockstep. A future
/// helper that appends to `capture_blobs` without also recording into
/// `self.replay` will silently truncate a retained tape — use
/// `debug_assert_tape_parity` or compare the two counts in a test.
pub struct Gpu {
    pub hip: HipRuntime,
    pub arch: String,
    pub flags: Arc<FeatureFlags>,
    pub arch_caps: crate::arch_caps::ArchCaps,
    pub device_id: i32,
    pub(crate) compiler: KernelCompiler,
    pub(crate) modules: HashMap<String, hip_bridge::Module>,
    pub(crate) functions: HashMap<String, hip_bridge::Function>,
    pub(crate) pool: crate::pool::GpuPool,
    /// VMM owners keyed by their base virtual address. VMM-backed tensors use
    /// non-owning DeviceBuffer views and must bypass GpuPool/hipFree teardown.
    vmm_arenas: HashMap<usize, VmmArena>,
    /// Monotonic owner generation of every registered VMM owner, keyed like
    /// `vmm_arenas` and recorded at registration (alloc and fork). A base
    /// address is never reused for a live owner, but the generation lets
    /// relocation prove a new owner is the one it was told about.
    vmm_generations: HashMap<usize, u64>,
    /// Last generation handed out; owner generations start at 1.
    vmm_generation_counter: u64,
    /// Arenas whose cleanup failed before they could enter the owner map.
    /// They have no tensor owner and are retried during explicit/Gpu teardown.
    orphan_vmm_arenas: Vec<VmmArena>,
    /// `hipHostMalloc(hipHostMallocMapped)` owners, keyed by the **device-visible**
    /// base address kernels are handed. The value is the *host* pointer that
    /// `hipHostFree` needs, which `hipHostGetDevicePointer` may alias differently
    /// on non-large-BAR systems — so the pair must be remembered rather than
    /// reconstructed. Stored as `usize` (not a raw pointer) to leave `Gpu`'s
    /// auto-traits exactly as they were for the VMM arena map.
    host_mapped: HashMap<usize, usize>,
    /// When set, all kernel launches go to this stream instead of null stream.
    pub active_stream: Option<hip_bridge::Stream>,
    /// Name of the most recently launched kernel on this `Gpu`'s stream.
    /// Recorded in `launch_maybe_blob_bound` (the shared dispatch funnel) on
    /// every successful launch so `sync_with_deadline` can name the suspect
    /// when a sync times out. Scratch-helper converts and the optional CK
    /// path bypass this funnel and are deliberately not tracked here.
    last_kernel: Option<String>,
    /// Scratch buffers for FWHT rotation, FP16/FP8 activation conversion, etc.
    pub scratch: crate::scratch::ScratchState,
    /// Model-scoped Redline warmup recorder and fail-closed backend gate.
    pub replay: crate::replay::ReplayController,
    /// Artifact-level MQ4V2 grid contract. Set by the model loader from HFQ
    /// metadata; false for legacy/asymmetric artifacts.
    pub mq4v2_symmetric: bool,
    /// Qwen4 symmetric MQ4 trunk verified for the dense IU4 route
    /// (`HIPFIRE_QWEN4_TRUNK_IU4`); set by the Qwen4 forward after every
    /// trunk header passed [`Gpu::qwen4_trunk_sym_check`]. See
    /// [`Gpu::qwen4_trunk_iu4_applies`].
    pub qwen4_trunk_iu4: bool,
    /// ADD-epilogue fold (gfx1151 V2B, `HIPFIRE_V2B_ADDEPI`): delta buffer
    /// armed by the caller for the next residual GEMM, and the deferred add
    /// that GEMM left for the following IU4 RMSNorm. See
    /// [`Gpu::arm_residual_fold`].
    pub(crate) residual_fold_arm: Option<ResidualFoldArm>,
    pub(crate) residual_fold_pending: Option<ResidualFold>,

    /// Process-pinned optional CK runtime. Loading is explicit and fail-closed;
    /// individual attention families still decide whether a capability cell is
    /// eligible after their native layout/tier resolution.
    #[cfg(feature = "flash-attn-ck")]
    pub(crate) flash_attn_ck: Option<crate::flash_attn_ck::FlashAttnCk>,
    #[cfg(feature = "flash-attn-ck")]
    pub(crate) flash_attn_ck_workspace: Option<hip_bridge::DeviceBuffer>,
    #[cfg(feature = "flash-attn-ck")]
    pub(crate) flash_attn_ck_reported_routes: std::collections::HashSet<&'static str>,

    // ── MMQ per-weight screening (#87) — extracted to MmqScreenState ──────
    pub mmq_screen: MmqScreenState,

    // ── hipGraph capture state (extracted to graph.rs) ─────────────────────
    pub graphs: crate::graph::GraphState,

    // ── rocBLAS (CDNA3 MFMA-accelerated GEMM) ─────────────────────────────
    /// Optional rocBLAS handle. `None` on non-CDNA3 archs or when
    /// librocblas.so fails to load. Engine code should always gate on
    /// `.is_some()` and fall back to the hand-rolled HFQ4 kernels otherwise.
    pub rocblas: Option<Rocblas>,

    /// FP16 shadow cache for HFQ4-G256 weights. Populated lazily on first
    /// batched prefill through the rocBLAS path: we dequantize the MQ4
    /// weight into an FP16 buffer once, then reuse for every subsequent
    /// prefill call. Key is the MQ4 device pointer (usize for Hash); value
    /// owns the GPU-side FP16 tensor. Memory is not freed until the Gpu
    /// itself drops (weights are assumed immutable for a model's lifetime).
    ///
    /// Only populated on CDNA3 when rocBLAS loaded — 4× VRAM blow-up vs MQ4
    /// so consumer cards stay on the wave32/64 hand-rolled GEMV path.
    fp16_shadow_cache: HashMap<usize, GpuTensor>,

    /// Calibration activation capture (Tier-1 collector). When `Some`, the
    /// instrumented linear dispatch arms resolve their weight buffer pointer
    /// to a tensor name via `capture_names` and invoke `capture()` with the
    /// input activation. `None` (the default) ⇒ the check is a single
    /// `is_none()` and forwards are byte-identical. The collector is held by
    /// `Arc` so the dispatch site can clone it (breaking the borrow on `self`)
    /// before calling `capture(self, …)`.
    pub active_capture: Option<Arc<dyn ActivationCapture>>,
    /// Weight-buffer-pointer → canonical tensor name, populated when calibration
    /// is armed. Lets capture fire from ANY forward path (hand or lowered,
    /// fused or not) keyed by the weight the gemv received.
    pub capture_names: HashMap<usize, String>,

    /// Native GPTQ-on-E8 Hessian collection. `None` in production (zero
    /// overhead -- a single `is_some()` branch in the MoE CPU-top-K fallback
    /// per routed expert). The `collect_e8_hessian_native` example sets this to
    /// `Some(default)` before running the calibration forward; every routed
    /// expert then accumulates its per-256-block XX^T over the RAW pre-rotation
    /// input (gate_up: post-rmsnorm x; down: silu(g)*u), keyed by the full
    /// safetensors tensor name == `hipfire-quantize::main::hessian_key`. Drained
    /// to per-(tensor,expert) `.hblk` files after the pass. See
    /// `hipfire-dispatch::pipeline::run_moe_decode_cpu_fallback` for the hook.
    pub hessian_capture: Option<HessianCapture>,
    /// True while the Qwen4 (Flash-Next) forward runs: admits the Qwen4-only
    /// arch defaults of `HIPFIRE_QWEN4_MQ6_X4_{GFX1201,TILE,REGIONS}` (U2 only
    /// on prefill chunks of >= 512 rows).  An explicit value of those
    /// variables applies everywhere, as before.
    pub qwen4_scope: bool,
}
/// Per-256-block XX^T accumulator for ONE weight tensor (one expert), keyed
/// inside [`HessianCapture`] by the full safetensors name. Byte-for-byte the
/// same accumulation + `.hblk` layout as
/// `hipfire-quantize::bin::collect_e8_hessian::BlockHessian` (duplicated here
/// because that lives in a `bin` crate the GPU layer cannot import). The
/// quantizer's `load_hessian_blocks` reads exactly this format.
#[derive(Debug)]
pub struct BlockHessianAcc {
    pub k: usize,
    pub n_blocks: usize,
    /// `n_blocks` blocks of `256*256` f64 accumulators (row-major per block).
    pub blocks: Vec<Vec<f64>>,
    pub n_rows: u64,
}

impl BlockHessianAcc {
    pub fn new(k: usize) -> Self {
        assert!(k % 256 == 0, "Hessian K={k} must be divisible by 256");
        let n_blocks = k / 256;
        BlockHessianAcc {
            k,
            n_blocks,
            blocks: (0..n_blocks).map(|_| vec![0.0f64; 256 * 256]).collect(),
            n_rows: 0,
        }
    }

    /// Accumulate one pre-rotation activation row `x[0..K]` into the per-256-block
    /// diagonal XX^T (block b += x_b x_b^T over its 256 channels).
    pub fn accumulate_row(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.k);
        for b in 0..self.n_blocks {
            let xb = &x[b * 256..b * 256 + 256];
            let acc = &mut self.blocks[b];
            for i in 0..256 {
                let xi = xb[i] as f64;
                if xi == 0.0 {
                    continue;
                }
                let row = &mut acc[i * 256..i * 256 + 256];
                for j in 0..256 {
                    row[j] += xi * xb[j] as f64;
                }
            }
        }
        self.n_rows += 1;
    }

    /// Serialize to the `.hblk` format consumed by
    /// `hipfire-quantize::main::load_hessian_blocks`:
    /// `[u32 magic=0x45384831][u32 n_blocks=K/256][u32 K][f32 ... n_blocks*256*256]`.
    pub fn write_hblk(&self, dir: &std::path::Path, tensor_name: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        // MUST byte-match hessian_key(): replace('/','_').replace('\\','_').replace("..","_").
        let key = tensor_name.replace(['/', '\\'], "_").replace("..", "_");
        let path = dir.join(format!("{key}.hblk"));
        let mut buf = Vec::with_capacity(12 + self.n_blocks * 256 * 256 * 4);
        buf.extend_from_slice(&0x45_38_48_31u32.to_le_bytes()); // "E8H1"
        buf.extend_from_slice(&(self.n_blocks as u32).to_le_bytes());
        buf.extend_from_slice(&(self.k as u32).to_le_bytes());
        for b in 0..self.n_blocks {
            for &v in &self.blocks[b] {
                buf.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &buf)
    }

    /// Mean of the per-block diagonal (sanity: should be > 0 for a frequently
    /// routed expert; 0 == never accumulated).
    pub fn mean_diag(&self) -> f64 {
        let mut s = 0.0f64;
        let mut n = 0u64;
        for b in 0..self.n_blocks {
            for i in 0..256 {
                s += self.blocks[b][i * 256 + i];
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            s / n as f64
        }
    }
}

/// Host-side accumulator for native GPTQ-on-E8 Hessian collection. Lives on
/// [`Gpu`] so the MoE CPU-top-K fallback chokepoint can reach it without
/// threading a parameter through every forward path. Pure host data.
///
/// Keyed by the FULL safetensors tensor name (== the quantizer's
/// `hessian_key` input), e.g.
/// `model.language_model.layers.7.mlp.experts.42.gate_up_proj.weight`.
#[derive(Debug, Default)]
pub struct HessianCapture {
    /// `tensor_name -> per-256-block XX^T`. Lazily sized to the tensor's K on
    /// first sighting of that (tensor,expert).
    pub entries: HashMap<String, BlockHessianAcc>,
    /// Total calibration tokens processed (collector increments once/step).
    pub n_tokens: u64,
}

impl HessianCapture {
    /// Accumulate one RAW pre-rotation activation row for `name`. `x` is the
    /// linear's pre-rotation input (gate_up: post-rmsnorm x; down: silu(g)*u);
    /// only the first `k` channels are used. Creates the entry on first sight.
    pub fn accumulate(&mut self, name: &str, x: &[f32], k: usize) {
        let acc = self
            .entries
            .entry(name.to_string())
            .or_insert_with(|| BlockHessianAcc::new(k));
        if x.len() >= k {
            acc.accumulate_row(&x[..k]);
        } else {
            let mut row = vec![0.0f32; k];
            row[..x.len()].copy_from_slice(x);
            acc.accumulate_row(&row);
        }
    }

    /// Accumulate ALL of one token's routed (tensor,expert) activation rows in
    /// PARALLEL across rayon worker threads. `items` is the token's list of
    /// `(name, x, k)` work-units (gate_up + down for each of the top-k experts).
    ///
    /// WHY THIS IS THE HOT-PATH OPTIMIZATION: the per-token capture cost is
    /// dominated by the `accumulate_row` rank-1 XX^T updates over a HUGE, COLD
    /// working set (A3B: ~16k distinct (tensor,expert) accumulators, ~30 GB of
    /// f64), which is memory-system-bound (cold-line latency + first-touch page
    /// faults), single-threaded, GPU idle, 20 host cores free. Each work-unit in
    /// `items` targets a DISTINCT (tensor,expert) accumulator (distinct experts
    /// per token, distinct gate_up/down tensors), so the accumulators are
    /// pairwise DISJOINT and can be updated concurrently — spreading the cold
    /// memory traffic + faults across cores.
    ///
    /// BIT-IDENTICAL TO SERIAL: each accumulator receives EXACTLY the same
    /// `accumulate_row(x)` call it would in the per-expert serial loop, computed
    /// by exactly ONE thread; only the (already order-independent, because the
    /// targets are disjoint) cross-accumulator iteration is parallelized. The
    /// internal float sum of every H[i,j] entry is byte-for-byte the serial
    /// result. See `accumulate_token_bit_identical_to_serial` parity test.
    pub fn accumulate_token(&mut self, items: &[(String, &[f32], usize)]) {
        use rayon::prelude::*;
        // 1) Ensure every entry exists (serial; allocation must not race the
        //    HashMap). Entries are keyed by full name; first sight sizes to k.
        for (name, _x, k) in items {
            self.entries
                .entry(name.clone())
                .or_insert_with(|| BlockHessianAcc::new(*k));
        }
        // 2) Gather a raw *mut to each target accumulator. The targets are
        //    pairwise disjoint iff the names within this token are distinct —
        //    which routing guarantees (distinct expert ids; distinct tensors).
        //    A SyncPtr wrapper lets rayon move the (provably disjoint) pointers
        //    across threads; no two closures touch the same accumulator.
        #[derive(Clone, Copy)]
        struct SyncPtr(*mut BlockHessianAcc);
        // SAFETY: each pointer addresses a distinct HashMap value (distinct keys,
        // asserted below); the map is not mutated for the duration of the
        // parallel section, so the pointees are stable and non-overlapping.
        unsafe impl Send for SyncPtr {}
        unsafe impl Sync for SyncPtr {}

        // Debug guard: catch any accidental duplicate key (would alias &mut).
        debug_assert!(
            {
                let mut ns: Vec<&str> = items.iter().map(|(n, _, _)| n.as_str()).collect();
                ns.sort_unstable();
                let before = ns.len();
                ns.dedup();
                ns.len() == before
            },
            "accumulate_token: duplicate (tensor,expert) name within one token              would alias &mut — parallel accumulate requires distinct keys"
        );

        let ptrs: Vec<(SyncPtr, &[f32], usize)> = items
            .iter()
            .map(|(name, x, k)| {
                let acc: &mut BlockHessianAcc =
                    self.entries.get_mut(name).expect("entry ensured above");
                (SyncPtr(acc as *mut BlockHessianAcc), *x, *k)
            })
            .collect();

        // 3) Parallel accumulate, one work-unit per rayon task. Each task calls
        //    the SAME `accumulate_row` as serial on its own disjoint accumulator.
        ptrs.par_iter().for_each(|&(p, x, k)| {
            // SAFETY: disjoint targets (distinct keys); no aliasing across tasks.
            let acc: &mut BlockHessianAcc = unsafe { &mut *p.0 };
            if x.len() >= k {
                acc.accumulate_row(&x[..k]);
            } else {
                let mut row = vec![0.0f32; k];
                row[..x.len()].copy_from_slice(x);
                acc.accumulate_row(&row);
            }
        });
    }
}

/// Generate `n` FWHT sign values (+1.0 / -1.0) from a simple LCG seeded with `seed`.
/// Deterministic and portable; used by both host-side codec (weight encoding) and
/// device-side init (`ensure_mq_signs` / `ensure_mq_signs_128`).
pub fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
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

impl Gpu {
    /// Diagnostic: trace multi-slot session continuation matching.
    pub fn slot_trace(&self) -> bool {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slot_trace
    }

    /// Whether the multi-slot decode step should be hipGraph-captured.
    pub fn slots_decode_graph(&self) -> bool {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slots_decode_graph
    }

    /// Whether this device is an APU with unified memory (its "VRAM" is
    /// system RAM). Queried live via
    /// `hipDeviceGetAttribute(hipDeviceAttributeIntegrated)` instead of an
    /// arch-string table: APUs and dGPUs share gfx IP across generations, so
    /// name-based classification drifts every product refresh (gfx1151 is an
    /// APU while gfx1201 with the same IP family era is not, etc).
    ///
    /// Query failure returns `true`, the conservative answer for page-cache
    /// policy: assuming UMA on a dGPU only costs load speed (eviction is the
    /// historical behavior), while assuming dGPU on a real APU keeps model
    /// pages resident next to hipMalloc staging and can OOM the load.
    // bind_thread: skip — pure device-property query; reads static device
    // info, touches no stream or per-thread HIP context.
    pub fn is_uma(&self) -> bool {
        // bind_thread: skip — pure device-property query; reads static
        // device info, touches no stream or per-thread HIP context.
        match self
            .hip
            .get_device_attribute(HIP_DEVICE_ATTRIBUTE_INTEGRATED, self.device_id)
        {
            Ok(v) => v != 0,
            Err(_) => true,
        }
    }

    /// Install a real stream if launches are still going to the null stream.
    ///
    /// HIP refuses to capture the legacy default stream, so anything that
    /// wants `begin_stream_capture` must call this first. Idempotent, and the
    /// stream stays installed afterwards: every later launch simply goes to it
    /// instead of the null stream.
    pub fn ensure_capture_stream(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        if self.active_stream.is_none() {
            self.active_stream = Some(self.hip.stream_create()?);
        }
        Ok(())
    }

    /// Begin capturing this `Gpu`'s stream into a graph.
    ///
    /// Mode 1 is `hipStreamCaptureModeThreadLocal`: only this thread is
    /// restricted, rather than mode 0's process-wide restriction. Everything
    /// this crate launches during capture must already be warm -- a kernel
    /// compile mid-capture is exactly the kind of call the mode forbids.
    pub fn begin_stream_capture(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        let stream = self.active_stream.as_ref().ok_or_else(|| {
            hip_bridge::HipError::new(0, "begin_stream_capture: no active stream")
        })?;
        self.hip.stream_begin_capture(stream, 1)
    }

    /// Close the capture started by `begin_stream_capture`.
    pub fn end_stream_capture(&mut self) -> HipResult<hip_bridge::Graph> {
        self.bind_thread()?;
        let stream = self
            .active_stream
            .as_ref()
            .ok_or_else(|| hip_bridge::HipError::new(0, "end_stream_capture: no active stream"))?;
        self.hip.stream_end_capture(stream)
    }

    /// Launch a previously instantiated graph on this `Gpu`'s stream.
    pub fn launch_graph(&mut self, exec: &hip_bridge::GraphExec) -> HipResult<()> {
        self.bind_thread()?;
        let stream = self
            .active_stream
            .as_ref()
            .ok_or_else(|| hip_bridge::HipError::new(0, "launch_graph: no active stream"))?;
        self.hip.graph_launch(exec, stream)
    }

    /// Returns the active stream ref for kernel launches (None = null stream).
    pub(crate) fn stream_ref(&self) -> Option<&hip_bridge::Stream> {
        self.active_stream.as_ref()
    }

    /// Default bound for [`Self::sync_with_deadline`]: five minutes. Long
    /// enough that a healthy prefill/decode sync never trips it, short enough
    /// that a hung GPU becomes a reportable error in one operator shift.
    pub const GPU_SYNC_DEADLINE: std::time::Duration = std::time::Duration::from_secs(300);

    /// Name of the kernel most recently launched through the dispatch funnel,
    /// if any. Used to attribute a timed-out sync to the suspect kernel.
    pub fn last_launched_kernel(&self) -> Option<&str> {
        self.last_kernel.as_deref()
    }

    /// Every kernel name currently loaded on this device, sorted.
    ///
    /// Diagnostic only: a retained-capture census uses it to name the kernels a
    /// model actually loaded, so a launch that never reached the recorder can be
    /// identified by set difference against the recorded tape instead of by
    /// guessing at call sites.
    pub fn loaded_kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.functions.keys().cloned().collect();
        names.sort();
        names
    }

    /// Build the timeout error directly (no device call). Split out so the
    /// message is unit-testable on CPU-only hosts where the blocking-sync
    /// path cannot be driven.
    pub fn deadline_exceeded(
        last_kernel: Option<&str>,
        deadline: std::time::Duration,
    ) -> hip_bridge::HipError {
        match last_kernel {
            Some(kernel) => hip_bridge::HipError::new(
                0,
                &format!(
                    "GPU sync exceeded deadline of {deadline:?}: \
                     last kernel launched on this stream was '{kernel}' — \
                     suspect a hang in '{kernel}' (stream still busy; \
                     state was NOT reset)"
                ),
            )
            .with_kernel(kernel),
            None => hip_bridge::HipError::new(
                0,
                &format!(
                    "GPU sync exceeded deadline of {deadline:?} with no kernel \
                     recorded on this stream (stream still busy; state was NOT reset)"
                ),
            ),
        }
    }

    /// Poll interval for [`Self::sync_with_deadline`]: 2 ms. Coarse enough
    /// never to spin a core, fine-grained enough for a deadline measured in
    /// seconds.
    pub(crate) const SYNC_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(2);

    /// Bounded stream sync: record a completion event on this `Gpu`'s stream
    /// (or the null stream) and poll `hipEventQuery` until it completes or
    /// `deadline` elapses.
    ///
    /// No helper thread: the query is non-blocking, so the deadline is real —
    /// on timeout this returns `Err` naming `last_launched_kernel` while the
    /// GPU work is still outstanding.
    ///
    /// Returning `Err` does NOT cancel the outstanding work. The contract is
    /// "I stopped waiting", not "the GPU stopped": the stream is still busy
    /// and nothing here resets device state, so the caller must treat the
    /// device as suspect and must not touch buffers the outstanding work may
    /// still write.
    ///
    /// Existing `sync()` callers keep their unbounded semantics; use this
    /// where a deadline is already meaningful (collective rendezvous,
    /// watchdog paths) rather than migrating every sync.
    pub fn sync_with_deadline(&self, deadline: std::time::Duration) -> HipResult<()> {
        self.bind_thread()?;
        // Timing-disabled event: a pure completion probe, no profiling state.
        let event = self
            .hip
            .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?;
        self.hip.event_record(&event, self.active_stream.as_ref())?;
        let last_kernel = self.last_kernel.clone();
        let result = Self::poll_until_ready(deadline, last_kernel.as_deref(), || {
            self.hip.event_query(&event)
        });
        // Best-effort teardown. On timeout the outstanding GPU work is NOT
        // cancelled — this call only stopped waiting. Never shadow the poll
        // result with a destroy error.
        let _ = self.hip.event_destroy(event);
        result
    }

    /// Poll `query` until it reports ready or `deadline` elapses. Query
    /// errors propagate immediately; only `hipErrorNotReady` keeps polling.
    /// Split from [`Self::sync_with_deadline`] so the timeout path is
    /// unit-testable on CPU with a stubbed query.
    pub(crate) fn poll_until_ready(
        deadline: std::time::Duration,
        last_kernel: Option<&str>,
        mut query: impl FnMut() -> HipResult<bool>,
    ) -> HipResult<()> {
        let start = std::time::Instant::now();
        loop {
            if query()? {
                return Ok(());
            }
            if start.elapsed() >= deadline {
                return Err(Self::deadline_exceeded(last_kernel, deadline));
            }
            std::thread::sleep(Self::SYNC_POLL_INTERVAL);
        }
    }

    /// Bind this `Gpu`'s device on the calling thread. Delegates to
    /// `crate::graph::bind_thread`.
    #[inline]
    pub fn bind_thread(&self) -> HipResult<()> {
        crate::graph::bind_thread(&self.hip, self.device_id)
    }

    /// `bind_thread` for `&mut self -> ()` and `Drop` contexts. Delegates to
    /// `crate::graph::bind_thread_or_warn`.
    #[inline]
    pub fn bind_thread_or_warn(&self) {
        crate::graph::bind_thread_or_warn(&self.hip, self.device_id)
    }

    /// Drive the GPU to full DPM perf level before a perf-sensitive measurement.
    ///
    /// gfx1100 (and other RDNA cards) return to a low-power DPM state when
    /// GPU utilization drops. A fresh process, or a process that just did
    /// light CPU-side setup, will find the GPU partially idling. Kernels run
    /// at reduced sclk/mclk until enough sustained load convinces the driver
    /// to ramp up. That ramp-up is slow and variable (~1-10 s observed), and
    /// its variance produces cycle-time swings like 52 ms vs 358 ms on the
    /// same bench. See `docs/methodology/perf-benchmarking.md`.
    ///
    /// This runs a tight memset + small-gemm loop for `secs` seconds to pin
    /// the GPU at high DPM before the caller's timer starts. Memset stresses
    /// mclk; the existing JITed `gemv_hfq4g256` kernel (available on any
    /// caller that has compiled a DFlash/Qwen3.5 model) stresses sclk.
    pub fn dpm_warmup(&mut self, secs: f32) -> HipResult<()> {
        self.bind_thread()?;
        // 256 MB scratch — large enough to defeat L2 and tax the memory
        // controller. GDDR6 on the 7900 XTX is 24 GB so 256 MB is trivial.
        const SCRATCH_BYTES: usize = 256 * 1024 * 1024;
        let scratch = self.hip.malloc(SCRATCH_BYTES)?;
        eprint!("warming caches...");
        let t0 = std::time::Instant::now();
        let mut n: u64 = 0;
        while t0.elapsed().as_secs_f32() < secs {
            // Rotate the fill byte so the driver/card can't short-circuit
            // repeated identical writes via any dedup or cache-match path.
            self.hip
                .memset(&scratch, (n & 0xFF) as i32, SCRATCH_BYTES)?;
            self.hip.device_synchronize()?;
            n = n.wrapping_add(1);
        }
        let elapsed = t0.elapsed().as_secs_f32();
        eprintln!(" took {elapsed:.2}s");
        // Free the 256 MB scratch — DeviceBuffer has no Drop, so scope exit
        // would otherwise leak it for the lifetime of the process.
        let _ = self.hip.free(scratch);
        Ok(())
    }

    pub fn init() -> HipResult<Self> {
        Self::init_with_device(0)
    }

    pub fn init_with_device(id: i32) -> HipResult<Self> {
        let hip = HipRuntime::load()?;
        let count = hip.device_count()?;
        if count == 0 {
            return Err(hip_bridge::HipError::new(0, "no GPU devices found"));
        }
        if id < 0 || id >= count {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("device id {id} out of range (count={count})"),
            ));
        }
        if let Ok(mode) = hipfire_config::developer_var("HIPFIRE_HIP_WAIT") {
            let mode_lc = mode.to_ascii_lowercase();
            let flags = match mode_lc.as_str() {
                "auto" => Some(0x00),
                "spin" => Some(0x01),
                "yield" => Some(0x02),
                "block" | "blocking" | "blocking_sync" => Some(0x04),
                "" => None,
                other => {
                    eprintln!(
                        "WARNING: unknown HIPFIRE_HIP_WAIT={other:?}; expected auto|spin|yield|blocking"
                    );
                    None
                }
            };
            if let Some(flags) = flags {
                hip.set_device_flags(flags)?;
                eprintln!("[hipfire] HIP wait mode: {mode_lc}");
            }
        }
        // set_device must precede try_init_rocblas — rocBLAS captures the
        // currently-bound device into its handle.
        hip.set_device(id)?;

        // HIPFIRE_TARGET_ARCH overrides the detected GPU arch for kernel
        // compilation. Used to test cross-arch family targets like
        // `gfx10-1-generic` (covers Navi 10/12/14) without per-arch JIT
        // cache fragmentation. Empty / unset preserves prior behavior.
        let detected_arch = hip.get_arch(id).unwrap_or_else(|_| "gfx1010".to_string());
        // Record the DETECTED (not compile-target-overridden) arch for
        // config-time policy that runs without a Gpu handle in hand — the
        // kv_slots memory preflight resolves its auto mode (unified-memory
        // APU vs discrete GPU) from physical topology, which a
        // HIPFIRE_TARGET_ARCH override does not change.
        crate::arch_caps::note_process_gpu_arch(&detected_arch);
        // The grid.y/grid.z ceiling is a property of the physical device, so
        // it keys off the detected arch, not the HIPFIRE_TARGET_ARCH override.
        // Set once here: this `Gpu` owns its `HipRuntime` for exactly one
        // device, so the per-runtime limit is race-free.
        hip.set_launch_grid_limit_for_arch(&detected_arch);
        let arch = hipfire_config::developer_var("HIPFIRE_TARGET_ARCH")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or(detected_arch);
        let (_, vram_total) = hip.get_vram_info().unwrap_or((0, 0));

        // Check HIP runtime version matches GPU arch requirements
        let (hip_major, hip_minor) = hip.runtime_version().unwrap_or((0, 0));
        let (min_major, min_minor) = match arch.as_str() {
            "gfx1200" | "gfx1201" => (6, 4),             // RDNA4 needs ROCm 6.4+
            "gfx1150" | "gfx1151" | "gfx1152" => (7, 2), // RDNA3.5 (Strix) needs ROCm 7.2+
            "gfx1100" | "gfx1101" | "gfx1102" => (5, 5), // RDNA3 needs ROCm 5.5+
            _ => (5, 0),
        };
        if hip_major > 0
            && (hip_major < min_major || (hip_major == min_major && hip_minor < min_minor))
        {
            eprintln!(
                "WARNING: HIP runtime {}.{} may not support {}. Minimum: {}.{}",
                hip_major, hip_minor, arch, min_major, min_minor
            );
            eprintln!("  Update your HIP runtime or kernels may fail to load.");
        }
        eprintln!(
            "GPU dev {}: {} ({:.1} GB VRAM, HIP {}.{})",
            id,
            arch,
            vram_total as f64 / 1e9,
            hip_major,
            hip_minor
        );

        let flags = Arc::new(FeatureFlags::from_active_config(&arch));
        let arch_caps = crate::arch_caps::ArchCaps::new(&arch, flags.clone());

        let compiler = KernelCompiler::new(&arch, flags.hipcc_extra_flags.clone())?;

        crate::graph::LAST_BOUND_DEVICE.with(|c| c.set(id));

        let mmq_screen = flags.mmq_screen;
        let mmq_screen_threshold = flags.mmq_screen_threshold;

        #[cfg(feature = "flash-attn-ck")]
        let flash_attn_ck = flags.flash_attn_ck_lib.as_deref().and_then(|path| {
            let runtime = match unsafe { crate::flash_attn_ck::FlashAttnCk::load(path) } {
                Ok(runtime) => runtime,
                Err(error) => {
                    eprintln!("WARNING: optional CK runtime disabled: {error}");
                    return None;
                }
            };
            let expected_arch = match arch.as_str() {
                "gfx1100" => crate::flash_attn_ck::FlashAttnCkArch::Gfx1100 as i32,
                "gfx1151" => crate::flash_attn_ck::FlashAttnCkArch::Gfx1151 as i32,
                "gfx1201" => crate::flash_attn_ck::FlashAttnCkArch::Gfx1201 as i32,
                _ => {
                    eprintln!(
                        "WARNING: optional CK runtime disabled: {arch} has no exact-arch ABI cell"
                    );
                    return None;
                }
            };
            if !runtime
                .capabilities()
                .iter()
                .any(|cell| cell.arch == expected_arch)
            {
                eprintln!(
                    "WARNING: optional CK runtime disabled: artifact has no {arch} capability"
                );
                return None;
            }
            eprintln!(
                "loaded optional CK runtime for {arch}: {} capability cell(s)",
                runtime.capabilities().len()
            );
            Some(runtime)
        });
        #[cfg(feature = "flash-attn-ck")]
        let flash_attn_ck_workspace =
            if flash_attn_ck.is_some() && flags.flash_attn_ck_workspace_bytes > 0 {
                Some(hip.malloc(flags.flash_attn_ck_workspace_bytes)?)
            } else {
                None
            };

        Ok(Self {
            hip,
            arch,
            flags,
            arch_caps,
            device_id: id,
            compiler,
            modules: HashMap::new(),
            functions: HashMap::new(),
            pool: crate::pool::GpuPool::new(),
            vmm_arenas: HashMap::new(),
            vmm_generations: HashMap::new(),
            vmm_generation_counter: 0,
            orphan_vmm_arenas: Vec::new(),
            host_mapped: HashMap::new(),
            active_stream: None,
            last_kernel: None,
            scratch: crate::scratch::ScratchState {
                mq_signs1: None,
                mq_signs2: None,
                mq_signs1_128: None,
                mq_signs2_128: None,
                mq_x_rot: None,
                mq_x_rot_fp8: None,
                mq_x_rot_fp8_bytes: 0,
                mq_x_q8: None,
                mq_x_scales: None,
                mq_rmsnorm_wavegrid_scratch: None,
                argmax_host: None,
                prerotated: None,
                gdn_pair_counters: None,
                mq_rmsnorm_awq_wavegrid_scratch: None,
                gemv_residual_tmp: None,
                paro_x_scratch: None,
                paro_fused_scratch: None,
                fp16_x_scratch: None,
                fp16_x_scratch_bytes: 0,
                qsa_select_scores: None,
                qsa_select_scores_bytes: 0,
                gdn_state_f32: None,
                gdn_state_f32_bytes: 0,
                qsa_gather_f16: None,
                qsa_gather_f16_bytes: 0,
                qsa_gather_vmm: None,
                qsa_gather_vmm_mapped: 0,
                fp16_x_source_ptr: std::ptr::null_mut(),
                fp8_x_scratch: None,
                fp8_x_scratch_bytes: 0,
                fp8_x_source_ptr: std::ptr::null_mut(),
                q8_1_mmq_x_scratch: None,
                q8_1_mmq_x_scratch_bytes: 0,
                int4_mmq_x_scratch: None,
                int4_mmq_x_scratch_bytes: 0,
                int4_mmq_generation: 0,
                int4_mmq_down_scratch: None,
                int4_mmq_down_scratch_bytes: 0,
                int4_mmq_down_generation: 0,
                int4_mmq_slab_generation: None,
                int8_mmq_x_scratch: None,
                int8_mmq_x_scratch_bytes: 0,
                int8_mmq_generation: 0,
                qwen4_moe_group_scratch: None,
                qwen4_moe_group_scratch_bytes: 0,
                qwen4_moe_down_i4_scratch: None,
                qwen4_moe_down_i4_scratch_bytes: 0,
                qwen4_moe_down_i4_generation: 0,
                mq4v2_fp8_x_scratch: None,
                mq4v2_fp8_x_scratch_bytes: 0,
                mq4v2_fp8_half_sums_scratch: None,
                mq4v2_fp8_half_sums_scratch_bytes: 0,
                mq4v2_fp8_row_scales_scratch: None,
                mq4v2_fp8_row_scales_scratch_bytes: 0,
                fp8_f2_weights: None,
                fp8_f2_weights_bytes: 0,
                ksplit_det_partials: None,
                ksplit_det_partials_bytes: 0,
                sample_partials: None,
                sample_partials_bytes: 0,
                fa2_q16_scratch: None,
                fa2_q16_scratch_bytes: 0,
                fa2_fp8_q_scratch: None,
                fa2_fp8_q_scratch_bytes: 0,
            },
            replay: crate::replay::ReplayController::from_config(),
            mq4v2_symmetric: false,
            qwen4_trunk_iu4: false,
            residual_fold_arm: None,
            residual_fold_pending: None,
            #[cfg(feature = "flash-attn-ck")]
            flash_attn_ck,
            #[cfg(feature = "flash-attn-ck")]
            flash_attn_ck_workspace,
            #[cfg(feature = "flash-attn-ck")]
            flash_attn_ck_reported_routes: std::collections::HashSet::new(),
            mmq_screen: MmqScreenState {
                cache: HashMap::new(),
                enabled: mmq_screen,
                threshold: mmq_screen_threshold,
            },
            graphs: crate::graph::GraphState {
                capture_mode: false,
                capture_blobs: Vec::new(),
                graph_exec: None,
                captured_graph: None,
                ar_forward_blobs: Vec::new(),
                ar_forward_kernel_dirty: true,
                ar_forward_replay_enabled: false,
                ar_forward_binding: 0,
                ar_graph_eligible: true,
                ar_segments: Vec::new(),
                verify: crate::graph::PerBGraphCache {
                    cache: std::collections::HashMap::new(),
                    warmed_up: std::collections::HashSet::new(),
                    capturing: None,
                    lmhead_argmax: std::collections::HashSet::new(),
                },
                replay: crate::graph::PerBGraphCache {
                    cache: std::collections::HashMap::new(),
                    warmed_up: std::collections::HashSet::new(),
                    capturing: None,
                    lmhead_argmax: std::collections::HashSet::new(),
                },
            },
            rocblas: None,
            fp16_shadow_cache: HashMap::new(),
            active_capture: None,
            capture_names: HashMap::new(),
            hessian_capture: None,
            qwen4_scope: false,
        })
        .map(|mut gpu| {
            if gpu.flags.force_blob_path {
                eprintln!("[diag] HIPFIRE_BLOB_FORCE=1: all kernel launches will use the blob path (kernelParams bypassed). Diagnostic only.");
            }
            // Auto-init rocBLAS on CDNA3 so the batched-prefill MFMA path is
            // available out of the box. No-op on consumer arches.
            gpu.try_init_rocblas();
            gpu
        })
    }

    /// Try to load rocBLAS. Safe no-op on non-CDNA3 archs (we don't use
    /// rocBLAS on RDNA — the hand-rolled kernels outperform it there).
    ///
    /// On success, sets `self.rocblas = Some(_)`; prefill dispatch paths can
    /// then route through MFMA-backed GEMM. On failure (library missing,
    /// symbol missing, handle init fail), logs once and leaves `None`.
    /// Callers always fall back to the non-rocBLAS path.
    pub fn try_init_rocblas(&mut self) {
        self.bind_thread_or_warn();
        if self.rocblas.is_some() {
            return;
        }
        let cdna3 = self.arch_caps.is_cdna3();
        let all_archs = self.flags.rocblas_all_archs;
        if !cdna3 && !all_archs {
            return;
        }
        match Rocblas::load() {
            Ok(rb) => {
                // Bind to the active stream if present; otherwise rocBLAS uses
                // the default (null) stream, which still works — just bigger
                // host-side sync cost.
                if let Some(stream) = self.active_stream.as_ref() {
                    if let Err(e) = rb.set_stream(stream) {
                        eprintln!(
                            "[rocblas] failed to bind active stream ({e}); using default stream"
                        );
                    }
                }
                eprintln!("[rocblas] loaded for {}", self.arch);
                self.rocblas = Some(rb);
            }
            Err(e) => {
                eprintln!(
                    "[rocblas] not available ({}); falling back to hand-rolled GEMMs",
                    e
                );
            }
        }
    }

    /// Dequantize a TQ2-G128 (ternary) weight [M × K] into an FP16 buffer
    /// [M × K] row-major. The FP16 buffer must be pre-allocated to M*K*2 bytes.
    ///
    /// This is the prefill route for the low-bit formats: they have no tiled
    /// GEMM of their own, so a chunk of N tokens would otherwise re-read every
    /// weight row N times through the scalar GEMV. Dequantising once into a
    /// scratch and handing the chunk to the F16 GEMM amortises that read.
    pub fn dequantize_tq2g128_to_f16(
        &mut self,
        w_packed: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — thin delegator; dequantize_lowbit_to_f16 binds.
        self.dequantize_lowbit_to_f16(
            "dequant_tq2g128_to_f16",
            kernels::DEQUANT_TQ2G128_TO_F16_SRC,
            w_packed,
            w_fp16,
            m,
            k,
        )
    }

    /// Binary sibling of [`Self::dequantize_tq2g128_to_f16`].
    pub fn dequantize_bq1g128_to_f16(
        &mut self,
        w_packed: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        // bind_thread: skip — thin delegator; dequantize_lowbit_to_f16 binds.
        self.dequantize_lowbit_to_f16(
            "dequant_bq1g128_to_f16",
            kernels::DEQUANT_BQ1G128_TO_F16_SRC,
            w_packed,
            w_fp16,
            m,
            k,
        )
    }

    /// Shared body for the two low-bit dequants. Both kernels take the same
    /// (A, W_f16, M, K) kernargs and the same [M, groups] × [32] geometry, so
    /// one body keeps them from drifting.
    fn dequantize_lowbit_to_f16(
        &mut self,
        name: &'static str,
        src: &'static str,
        w_packed: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert_eq!(k % 128, 0, "{name}: K must be a multiple of 128 (got {k})");
        self.ensure_kernel(name, src, name)?;
        let mut w_in = w_packed.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 128) as u32;
        launch_params_blob!(
            self,
            name,
            [m as u32, groups, 1],
            [32, 1, 1],
            0,
            params; w_in, w_out, mi, ki
        )
    }

    /// Dequantize an HFQ4-G256 weight [M × K] into an FP16 buffer [M × K]
    /// row-major. The FP16 buffer must be pre-allocated to M*K*2 bytes.
    ///
    /// Used as a one-shot model-load step on CDNA3 when the downstream
    /// prefill GEMM path is rocBLAS/hipBLASLt. Cost scales as O(MK) — for
    /// a 35B-A3B target at load time, ~10 GB dequantized; MI300X handles
    /// this in well under a second (the math is trivial, the launch is
    /// BW-bound at HBM3 write speed).
    pub fn dequantize_hfq4g256_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "hfq4g256 dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "hfq4g256_dequantize_to_f16",
            kernels::HFQ4G256_DEQUANTIZE_TO_F16_SRC,
            "hfq4g256_dequantize_to_f16",
        )?;
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        launch_params_blob!(
            self,
            "hfq4g256_dequantize_to_f16",
            [m as u32, groups, 1],
            [128, 1, 1],
            0,
            params; w_in, w_out, mi, ki
        )
    }

    /// Dequantize MFP4G32-Lloyd matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (32-B codebook prefix + M rows).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Grid: [M, K/256]. Block: [32]. Mirrors dequantize_hfq4g256_to_f16 shape.
    pub fn dequantize_mfp4g32_lloyd_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_lloyd dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_lloyd_to_f16",
            kernels::DEQUANTIZE_MFP4G32_LLOYD_TO_F16_SRC,
            "dequantize_mfp4g32_lloyd_to_f16",
        )?;
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        launch_params_blob!(
            self,
            "dequantize_mfp4g32_lloyd_to_f16",
            [m as u32, groups, 1],
            [32, 1, 1],
            0,
            params; w_in, w_out, mi, ki
        )
    }

    /// Dequantize an mfp4+P matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (M rows, NO prefix; byte-identical to mfp4).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Decodes the per-block scale byte as E4M3 (FP8). Grid: [M, K/256]. Block: [32].
    pub fn dequantize_mfp4g32_p_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_p dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_p_to_f16",
            kernels::DEQUANTIZE_MFP4G32_P_TO_F16_SRC,
            "dequantize_mfp4g32_p_to_f16",
        )?;
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        launch_params_blob!(
            self,
            "dequantize_mfp4g32_p_to_f16",
            [m as u32, groups, 1],
            [32, 1, 1],
            0,
            params; w_in, w_out, mi, ki
        )
    }

    /// Dequantize an mfp4-E8 matrix [M x K] to FP16 [M x K] row-major.
    /// Input `w_mq4` = full tensor bytes (M rows, NO prefix; byte-identical footprint to mfp4+P).
    /// Output `w_fp16` = FP16 row-major (M x K), in the ROTATED domain.
    /// Decodes the per-block scale byte as E4M3 (FP8) * QUANT_STEP, then E8 coords.
    /// Grid: [M, K/256]. Block: [32].
    pub fn dequantize_mfp4g32_e8_to_f16(
        &mut self,
        w_mq4: &DeviceBuffer,
        w_fp16: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            k % 256 == 0,
            "mfp4g32_e8 dequant: K must be multiple of 256 (got {k})"
        );
        self.ensure_kernel(
            "dequantize_mfp4g32_e8_to_f16",
            kernels::DEQUANTIZE_MFP4G32_E8_TO_F16_SRC,
            "dequantize_mfp4g32_e8_to_f16",
        )?;
        let mut w_in = w_mq4.as_ptr();
        let mut w_out = w_fp16.as_ptr();
        let mut mi = m as i32;
        let mut ki = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut w_in as *mut _ as *mut c_void,
            &mut w_out as *mut _ as *mut c_void,
            &mut mi as *mut _ as *mut c_void,
            &mut ki as *mut _ as *mut c_void,
        ];
        let groups = (k / 256) as u32;
        launch_params_blob!(
            self,
            "dequantize_mfp4g32_e8_to_f16",
            [m as u32, groups, 1],
            [32, 1, 1],
            0,
            params; w_in, w_out, mi, ki
        )
    }

    /// Expand qt=35 MFP4G32E8SOA rows into row-major FP16 on gfx942.
    ///
    /// Unlike [`Self::dequantize_mfp4g32_e8_to_f16`], this reads the SoA wire
    /// layout used by the frozen DeepSeek4 MQ2R dense tier. The output remains
    /// in the FWHT-rotated domain, matching the rotated activation supplied by
    /// the normal MFP4E8 projection path.
    pub fn dequantize_mfp4g32_e8_soa_to_f16_gfx942(
        &mut self,
        packed: &DeviceBuffer,
        expanded: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        assert!(
            self.arch_caps.is_gfx942(),
            "qt35 SoA staging requires gfx942"
        );
        assert!(
            k % 256 == 0,
            "qt35 SoA staging requires K%256==0, got K={k}"
        );
        const KERNEL: &str = "dequantize_mfp4g32_e8_soa_to_f16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANTIZE_MFP4G32_E8_SOA_TO_F16_GFX942_SRC,
            KERNEL,
        )?;
        let packed_ptr = packed.as_ptr();
        let expanded_ptr = expanded.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &packed_ptr as *const _ as *mut c_void,
            &expanded_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, (k / 32).div_ceil(16) as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(packed_ptr);
                blob.push_ptr(expanded_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Dense parent-checkpoint FP8 weight decode on gfx942.
    ///
    /// Expands `F8_E4M3 [M,K]` with `F8_E8M0 [ceil(M/128), ceil(K/128)]`
    /// block scales into row-major BF16 `[M,K]`. Model-lifetime staging —
    /// not a hot-path decode.
    ///
    /// Grid `[M,1,1]`, block `[256,1,1]` (one workgroup per output row).
    pub fn dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942(
        &mut self,
        w: &DeviceBuffer,
        s: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: M and K must be positive (got M={m} K={k})"
                ),
            ));
        }
        let s_rows = m.div_ceil(128);
        let s_cols = k.div_ceil(128);
        let w_bytes = m
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: M*K overflow"))?;
        let s_bytes = s_rows
            .checked_mul(s_cols)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: scale shape overflow"))?;
        let out_bytes = w_bytes
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "dequant_fp8: BF16 out size overflow"))?;
        if w.size() < w_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: w buffer too small (have {} need {w_bytes})",
                    w.size()
                ),
            ));
        }
        if s.size() < s_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: s buffer too small (have {} need {s_bytes})",
                    s.size()
                ),
            ));
        }
        if out.size() < out_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942: out buffer too small (have {} need {out_bytes})",
                    out.size()
                ),
            ));
        }
        const KERNEL: &str = "dequant_fp8_e4m3_ue8m0_blk128_to_bf16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANT_FP8_E4M3_UE8M0_BLK128_TO_BF16_GFX942_SRC,
            KERNEL,
        )?;
        let w_ptr = w.as_ptr();
        let s_ptr = s.as_ptr();
        let out_ptr = out.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(w_ptr);
                blob.push_ptr(s_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Routed-expert parent-checkpoint FP4 weight decode on gfx942.
    ///
    /// Expands packed E2M1 `I8 [M, K/2]` with per-row UE8M0 scales
    /// `F8_E8M0 [M, K/32]` into row-major BF16 `[M,K]`. Hot path — one
    /// thread owns one 32-wide K-group.
    ///
    /// Grid `[M, ceil((K/32)/256), 1]`, block `[256,1,1]`.
    /// Requires `K % 32 == 0` (group size) and therefore even `K`.
    pub fn dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942(
        &mut self,
        w: &DeviceBuffer,
        s: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942 requires gfx942",
            ));
        }
        if m == 0 || k == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: M and K must be positive (got M={m} K={k})"
                ),
            ));
        }
        if k % 32 != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: K must be a multiple of 32 (got K={k})"
                ),
            ));
        }
        let n_groups = k / 32;
        let w_bytes = m
            .checked_mul(k / 2)
            .ok_or_else(|| HipError::new(0, "dequant_fp4: packed W size overflow"))?;
        let s_bytes = m
            .checked_mul(n_groups)
            .ok_or_else(|| HipError::new(0, "dequant_fp4: scale size overflow"))?;
        let out_bytes = m
            .checked_mul(k)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "dequant_fp4: BF16 out size overflow"))?;
        if w.size() < w_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: w buffer too small (have {} need {w_bytes})",
                    w.size()
                ),
            ));
        }
        if s.size() < s_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: s buffer too small (have {} need {s_bytes})",
                    s.size()
                ),
            ));
        }
        if out.size() < out_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942: out buffer too small (have {} need {out_bytes})",
                    out.size()
                ),
            ));
        }
        const KERNEL: &str = "dequant_fp4_e2m1_ue8m0_g32_to_bf16_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::DEQUANT_FP4_E2M1_UE8M0_G32_TO_BF16_GFX942_SRC,
            KERNEL,
        )?;
        let w_ptr = w.as_ptr();
        let s_ptr = s.as_ptr();
        let out_ptr = out.as_ptr();
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let mut params: Vec<*mut c_void> = vec![
            &w_ptr as *const _ as *mut c_void,
            &s_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &m_i32 as *const _ as *mut c_void,
            &k_i32 as *const _ as *mut c_void,
        ];
        let groups_y = n_groups.div_ceil(256) as u32;
        self.launch_maybe_blob(
            KERNEL,
            [m as u32, groups_y, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(w_ptr);
                blob.push_ptr(s_ptr);
                blob.push_ptr(out_ptr);
                blob.push_i32(m_i32);
                blob.push_i32(k_i32);
                blob
            },
        )
    }

    /// Parent-checkpoint FP8 activation quant simulation (fused quant+dequant).
    ///
    /// In-place over a row-major BF16 buffer shaped `[rows, last_dim]`.
    /// Groups of `block` contiguous elements along the last dim use a single
    /// UE8M0 power-of-two scale (`fast_round_scale`) and round through OCP
    /// E4M3 RNE before writing BF16(`e4m3_to_f32(q) * s`) back.
    ///
    /// `block` must be 64 (KV/compressor non-RoPE sites) or 128 (linear
    /// boundaries). `last_dim % block == 0` is required. Fail-closed on
    /// anything else — no MQ2R / Raw fallback.
    ///
    /// Launch: grid `[rows, last_dim/block, 1]`, block `[64,1,1]` (one wave
    /// per group).
    pub fn act_quant_fp8_ue8m0_inplace_gfx942(
        &mut self,
        x: &DeviceBuffer,
        rows: usize,
        last_dim: usize,
        block: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "act_quant_fp8_ue8m0_inplace_gfx942 requires gfx942",
            ));
        }
        if block != 64 && block != 128 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: block must be 64 or 128 (got {block})"
                ),
            ));
        }
        if rows == 0 || last_dim == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: rows and last_dim must be positive (got rows={rows} last_dim={last_dim})"
                ),
            ));
        }
        if last_dim % block != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: last_dim must be a multiple of block (got last_dim={last_dim} block={block})"
                ),
            ));
        }
        let need_bytes = rows
            .checked_mul(last_dim)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| HipError::new(0, "act_quant_fp8_ue8m0_inplace_gfx942: size overflow"))?;
        if x.size() < need_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp8_ue8m0_inplace_gfx942: buffer too small (have {} need {need_bytes} for {rows}×{last_dim} BF16)",
                    x.size()
                ),
            ));
        }
        const KERNEL: &str = "act_quant_fp8_ue8m0_inplace_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::ACT_QUANT_FP8_UE8M0_INPLACE_GFX942_SRC,
            KERNEL,
        )?;
        let x_ptr = x.as_ptr();
        let rows_i32 = rows as i32;
        let last_dim_i32 = last_dim as i32;
        let block_i32 = block as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &rows_i32 as *const _ as *mut c_void,
            &last_dim_i32 as *const _ as *mut c_void,
            &block_i32 as *const _ as *mut c_void,
        ];
        let n_groups = last_dim / block;
        self.launch_maybe_blob(
            KERNEL,
            [rows as u32, n_groups as u32, 1],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_i32(rows_i32);
                blob.push_i32(last_dim_i32);
                blob.push_i32(block_i32);
                blob
            },
        )
    }

    /// Parent-checkpoint FP4 activation quant simulation (fused quant+dequant).
    ///
    /// In-place over a row-major BF16 buffer shaped `[rows, last_dim]`.
    /// Groups of 32 along the last dim; UE8M0 power-of-two scale via
    /// `fast_round_scale(amax, 1/6)`, then E2M1 RNE onto
    /// `{0,.5,1,1.5,2,3,4,6}` (sign preserved) and BF16 write-back.
    ///
    /// `last_dim % 32 == 0` is required. Fail-closed otherwise.
    ///
    /// Launch: grid `[rows, last_dim/32, 1]`, block `[32,1,1]` (one group
    /// per 32-lane cohort).
    pub fn act_quant_fp4_ue8m0_g32_inplace_gfx942(
        &mut self,
        x: &DeviceBuffer,
        rows: usize,
        last_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx942() {
            return Err(HipError::new(
                0,
                "act_quant_fp4_ue8m0_g32_inplace_gfx942 requires gfx942",
            ));
        }
        if rows == 0 || last_dim == 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: rows and last_dim must be positive (got rows={rows} last_dim={last_dim})"
                ),
            ));
        }
        if last_dim % 32 != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: last_dim must be a multiple of 32 (got last_dim={last_dim})"
                ),
            ));
        }
        let need_bytes = rows
            .checked_mul(last_dim)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| {
                HipError::new(0, "act_quant_fp4_ue8m0_g32_inplace_gfx942: size overflow")
            })?;
        if x.size() < need_bytes {
            return Err(HipError::new(
                0,
                &format!(
                    "act_quant_fp4_ue8m0_g32_inplace_gfx942: buffer too small (have {} need {need_bytes} for {rows}×{last_dim} BF16)",
                    x.size()
                ),
            ));
        }
        const KERNEL: &str = "act_quant_fp4_ue8m0_g32_inplace_gfx942";
        self.ensure_kernel(
            KERNEL,
            kernels::ACT_QUANT_FP4_UE8M0_G32_INPLACE_GFX942_SRC,
            KERNEL,
        )?;
        let x_ptr = x.as_ptr();
        let rows_i32 = rows as i32;
        let last_dim_i32 = last_dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &rows_i32 as *const _ as *mut c_void,
            &last_dim_i32 as *const _ as *mut c_void,
        ];
        let n_groups = last_dim / 32;
        self.launch_maybe_blob(
            KERNEL,
            [rows as u32, n_groups as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_i32(rows_i32);
                blob.push_i32(last_dim_i32);
                blob
            },
        )
    }

    /// D→D copy with offsets that picks async (on the active stream) when
    /// a stream is set and sync otherwise. Captured graphs require async on
    /// the captured stream — sync `hipMemcpy` errors with "would make the
    /// legacy stream depend on a capturing blocking stream" under capture
    /// mode Global. Use this helper whenever the copy might live inside
    /// a captured region.
    /// `HIPFIRE_DTOD_DUMP=1` prints `file:line` and size per D→D copy.
    ///
    /// Mirrors `HIPFIRE_MEMSET_DUMP`. Added because `__amd_rocclr_copyBuffer`
    /// costs 0.378 ms/step (19.8 calls) in the ds4 gfx1151 AR decode — a 64 KB
    /// copy taking 19.11 us against 0.33 us of actual traffic, 57x off — and
    /// there are a dozen `memcpy_dtod_auto` call sites in the ds4 forward with
    /// no way to tell which ones fire. Grep the dump by source location.
    #[track_caller]
    pub fn memcpy_dtod_at_auto(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let dump = hipfire_config::developer_bool("HIPFIRE_DTOD_DUMP", false);
        if dump {
            let loc = std::panic::Location::caller();
            eprintln!("dtod bytes={} at {}:{}", size, loc.file(), loc.line());
        }
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip
                .memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip
                .memcpy_dtod_at(dst, dst_offset, src, src_offset, size)
        }
    }

    /// D→D copy with offsets that is always host-asynchronous: uses the
    /// active stream when set, otherwise the legacy/default stream via
    /// `memcpy_dtod_async_default_at`. Prefer this on non-capture hot
    /// paths where `memcpy_dtod_at_auto` would fall back to a host-sync
    /// copy. Capture paths still need an explicit active stream (same as
    /// `memcpy_dtod_at_auto`).
    /// `HIPFIRE_DTOD_DUMP=1` prints `file:line` and size per D→D copy.
    #[track_caller]
    pub fn memcpy_dtod_at_ordered_async(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        dst_offset: usize,
        src: &hip_bridge::DeviceBuffer,
        src_offset: usize,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        static DUMP: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_var("HIPFIRE_DTOD_DUMP")
                .ok()
                .as_deref()
                == Some("1")
        });
        let dump = *DUMP;
        if dump {
            let loc = std::panic::Location::caller();
            eprintln!("dtod bytes={} at {}:{}", size, loc.file(), loc.line());
        }
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip
                .memcpy_dtod_async_at(dst, dst_offset, src, src_offset, size, stream)
        } else {
            self.hip
                .memcpy_dtod_async_default_at(dst, dst_offset, src, src_offset, size)
        }
    }

    /// D→D copy (whole buffer) that picks async on the active stream when set.
    ///
    /// `#[track_caller]` so `HIPFIRE_DTOD_DUMP` attributes the copy to the real
    /// call site rather than to this forwarder.
    #[track_caller]
    pub fn memcpy_dtod_auto(
        &self,
        dst: &hip_bridge::DeviceBuffer,
        src: &hip_bridge::DeviceBuffer,
        size: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.memcpy_dtod_at_auto(dst, 0, src, 0, size)
    }

    /// H→D copy that picks async on the active stream when capturing.
    ///
    /// During hipGraph capture (`capture_mode == true`), operations on the
    /// legacy/null stream are forbidden because they would create a blocking
    /// dependency with the capturing stream. This method routes to
    /// `memcpy_htod_async` on the active (capturing) stream when in capture
    /// mode, falling back to sync `memcpy_htod` otherwise.
    pub fn memcpy_htod_auto(&self, dst: &hip_bridge::DeviceBuffer, src: &[u8]) -> HipResult<()> {
        self.bind_thread()?;
        if self.graphs.capture_mode {
            let stream = self
                .active_stream
                .as_ref()
                .expect("capture mode requires an active stream");
            self.hip.memcpy_htod_async(dst, src, stream)
        } else {
            self.hip.memcpy_htod(dst, src)
        }
    }

    /// Blocking D→H copy of `src`'s first `dst.len()` bytes.
    ///
    /// Deliberately **not** capture-aware, unlike [`Self::memcpy_htod_auto`]: a
    /// device→host copy is a host sync point, so it can never be part of a
    /// captured graph. Reaching this during capture is a bug in whoever decided
    /// to capture a graph containing a CPU-executed step, and it is reported as
    /// an error rather than enqueued onto the capturing stream (where it would
    /// either fail the capture or silently produce a replay that returns stale
    /// activations).
    pub fn memcpy_dtoh_auto(
        &self,
        dst: &mut [u8],
        src: &hip_bridge::DeviceBuffer,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if self.graphs.capture_mode {
            return Err(hip_bridge::HipError::new(
                0,
                "memcpy_dtoh_auto during hipGraph capture: a CPU-executed step cannot be \
                 captured; the capture decision must be disabled when host-mapped weights \
                 are executed on the CPU",
            ));
        }
        self.hip.memcpy_dtoh(dst, src)
    }

    /// Raw host bytes of a host-mapped tensor ([`Self::host_located`]), for the
    /// CPU-executed offload path.
    ///
    /// `None` unless this `Gpu` owns the allocation: the buffer's own pointer is
    /// the *device-visible* alias, which `hipHostGetDevicePointer` may place
    /// anywhere relative to the `hipHostMalloc` pointer `hipHostFree` needs, so
    /// the registry is the only sound source of the host address.
    ///
    /// The returned slice borrows the tensor, not the `Gpu`: the allocation lives
    /// until the model unloads (`release_registered_host_mapped`), and callers
    /// need `&mut Gpu` for the copies around a CPU step while still holding it.
    /// `HOST_TAIL_PAD_BYTES` of slack past the logical size is included in the
    /// allocation but *not* in the returned slice.
    pub fn host_bytes<'a>(&self, tensor: &'a GpuTensor) -> Option<&'a [u8]> {
        // bind_thread: skip — pure map lookup on host memory, no device state.
        if !tensor.buf.is_host_mapped() {
            return None;
        }
        let host_ptr = *self.host_mapped.get(&(tensor.buf.as_ptr() as usize))?;
        // Safety: registered by `alloc_host_mapped_tensor` as a live
        // `hipHostMalloc` of `tensor.buf.size() + HOST_TAIL_PAD_BYTES` bytes,
        // freed exactly once on unload.
        Some(unsafe { std::slice::from_raw_parts(host_ptr as *const u8, tensor.buf.size()) })
    }

    /// Writable host address of a host-mapped tensor this `Gpu` owns (the
    /// registry's `hipHostMalloc` pointer, see [`Self::host_bytes`]), for a
    /// pinned staging buffer the host rewrites between stream-ordered copies.
    /// The caller must not write while a queued copy may still read it.
    pub(crate) fn host_mapped_ptr(&self, tensor: &GpuTensor) -> Option<*mut u8> {
        if !tensor.buf.is_host_mapped() {
            return None;
        }
        self.host_mapped
            .get(&(tensor.buf.as_ptr() as usize))
            .map(|&host| host as *mut u8)
    }

    /// Helper: launch a kernel using the blob path during graph capture,
    /// or the normal kernelParams path otherwise. The `blob_builder` closure
    /// constructs the KernargBlob; it's only called when capturing.
    pub(crate) fn launch_maybe_blob(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        self.launch_maybe_blob_bound(
            func_name,
            grid,
            block,
            shared_mem,
            params,
            ReplayLaunchBindings::NONE,
            &[],
            blob_builder,
        )
    }

    /// [`Self::launch_maybe_blob`] for a launch whose kernarg carries dynamic
    /// words (railgun design §1.6: words are declared by the composer at the
    /// launch site). The declaration reaches the railgun shadow observer only;
    /// the launch and Redline's recording are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_maybe_blob_declared(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        declared_words: &[railgun::kernel::DeclaredWord],
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        self.launch_maybe_blob_bound(
            func_name,
            grid,
            block,
            shared_mem,
            params,
            ReplayLaunchBindings::NONE,
            declared_words,
            blob_builder,
        )
    }

    /// Record a position-derived PM4 workgroup binding while retaining the
    /// recorded maximum grid for HIP, hipGraph, AQL, and capture validation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_maybe_blob_position_grid(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        axis: u8,
        addend: u32,
        divisor: u32,
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        let grid_binding = pm4_dynamic_grid_enabled().then_some(
            crate::replay::ReplayGridBinding::PositionCeilDiv {
                axis,
                addend,
                divisor,
            },
        );
        self.launch_maybe_blob_bound(
            func_name,
            grid,
            block,
            shared_mem,
            params,
            ReplayLaunchBindings {
                grid: grid_binding,
                kernargs: &[],
            },
            &[],
            blob_builder,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_maybe_blob_bound(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        params: &mut [*mut std::ffi::c_void],
        bindings: ReplayLaunchBindings<'_>,
        railgun_words: &[railgun::kernel::DeclaredWord],
        blob_builder: impl FnOnce() -> hip_bridge::KernargBlob,
    ) -> HipResult<()> {
        // Fail closed before the recorder / capture tapes see the launch: an
        // oversized grid.y/grid.z must never be recorded, captured, or clamped.
        self.hip
            .validate_launch_grid(grid)
            .map_err(|e| e.with_kernel(func_name))?;
        let record = self.replay.is_recording();
        let observe = self.replay.is_g0_observing();
        // Slice 1: drain a post-scratch-growth binding refresh before the
        // first launch that follows the growth. Growth always completes
        // before this point (callers invalidate, then grow, then launch), so
        // the probe sees post-growth addresses. One branch when idle; the
        // refresh itself runs at most once per growth event. Failure drops
        // the retained route (historic path) and the HIP launch below runs.
        if self.replay.binding_refresh_pending() {
            if let Err(reason) = self.replay.refresh_bindings_after_growth(&self.hip) {
                eprintln!("[redline] PM4 binding refresh failed ({reason}); route re-armed");
                self.replay.rearm_after_layout_growth();
            }
        }
        let result: HipResult<()> =
            if record || observe || self.graphs.capture_mode || self.flags.force_blob_path {
                let mut blob = blob_builder();
                blob.pad_to(16);
                if observe {
                    self.replay
                        .observe_g0_launch(func_name, grid, block, shared_mem, blob.as_bytes());
                }
                if record {
                    let artifact = recorded_launch_artifact(&self.compiler, func_name);
                    self.replay.record_hip_launch_typed_bound(
                        &self.hip,
                        func_name,
                        artifact,
                        grid,
                        block,
                        shared_mem,
                        blob.as_bytes(),
                        bindings.grid,
                        bindings.kernargs,
                        Some(&self.compiler),
                        railgun_words,
                    );
                }
                let func = &self.functions[func_name];
                if self.graphs.capture_mode {
                    self.graphs.capture_blobs.push(blob.into_vec());
                    let buf = self.graphs.capture_blobs.last_mut().unwrap();
                    // SAFETY: every caller's builder encodes the same argument
                    // values supplied by `params`; graph-owned storage stays live
                    // through graph instantiation and replay.
                    unsafe {
                        self.hip.launch_kernel_blob(
                            func,
                            grid,
                            block,
                            shared_mem,
                            self.active_stream.as_ref(),
                            buf.as_mut_slice(),
                        )
                    }
                } else {
                    let mut bytes = blob.into_vec();
                    // SAFETY: HIP consumes the contiguous argument bytes during
                    // this one-shot launch; `bytes` remains live across the call.
                    unsafe {
                        self.hip.launch_kernel_blob(
                            func,
                            grid,
                            block,
                            shared_mem,
                            self.active_stream.as_ref(),
                            bytes.as_mut_slice(),
                        )
                    }
                }
            } else {
                let func = &self.functions[func_name];
                // SAFETY: forwarded from the typed launch wrapper that assembled
                // `params` for this kernel signature.
                unsafe {
                    self.hip.launch_kernel(
                        func,
                        grid,
                        block,
                        shared_mem,
                        self.active_stream.as_ref(),
                        params,
                    )
                }
            };
        if result.is_ok() {
            self.last_kernel = Some(func_name.to_string());
        }
        result.map_err(|e| e.with_kernel(func_name))
    }

    /// Diagnostic oracle for Redline prefix localization: relaunch the exact
    /// captured HIP blob sequence without re-entering model dispatch logic.
    /// Inputs/state must be restored by the caller first.
    pub fn replay_recorded_hip_prefix(&self, count: usize) -> HipResult<()> {
        self.bind_thread()?;
        if count > self.replay.recorded_launches().len() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "captured HIP prefix {count} exceeds {} launches",
                    self.replay.recorded_launches().len()
                ),
            ));
        }
        for launch in self.replay.recorded_launches().iter().take(count) {
            let func = self.functions.get(&launch.kernel).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("captured HIP function {:?} is not loaded", launch.kernel),
                )
            })?;
            let mut kernarg = launch.kernarg.clone();
            // SAFETY: the bytes were captured from this exact loaded function
            // and all pointees remain owned by this Gpu/model instance.
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    launch.grid,
                    launch.block,
                    launch.shared_mem,
                    self.active_stream.as_ref(),
                    &mut kernarg,
                )?;
            }
        }
        Ok(())
    }

    /// Position-aware variant of [`Self::replay_recorded_hip_prefix`].
    ///
    /// Applies the identical binding set the retained PM4 route applies (GDN
    /// frames, synthesized position bindings and every launch's engine-declared
    /// bindings, built once by `ReplayController::retained_kernarg_bindings`), so
    /// the recorded-blob oracle and the PM4 route stay equivalent: a divergence
    /// between them is then a submission difference, never a patching one.
    pub fn replay_recorded_hip_prefix_at(&self, count: usize, position: usize) -> HipResult<()> {
        self.bind_thread()?;
        let launches = self.replay.recorded_launches();
        if count > launches.len() {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "captured HIP prefix {count} exceeds {} launches",
                    launches.len()
                ),
            ));
        }
        let bindings = self
            .replay
            .retained_kernarg_bindings(count)
            .map_err(|reason| {
                hip_bridge::HipError::new(
                    0,
                    &format!("recorded-HIP oracle kernarg bindings: {reason}"),
                )
            })?;
        for (index, launch) in launches.iter().take(count).enumerate() {
            let func = self.functions.get(&launch.kernel).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("captured HIP function {:?} is not loaded", launch.kernel),
                )
            })?;
            let mut kernarg = launch.kernarg.clone();
            crate::replay::apply_kernarg_bindings_for_dispatch(
                &mut kernarg,
                index,
                position,
                &bindings,
            )
            .map_err(|reason| hip_bridge::HipError::new(0, &reason))?;
            // SAFETY: the bytes were captured from this exact loaded function
            // and all pointees remain owned by this Gpu/model instance.
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    launch.grid,
                    launch.block,
                    launch.shared_mem,
                    self.active_stream.as_ref(),
                    &mut kernarg,
                )?;
            }
        }
        Ok(())
    }

    /// Compile and load a kernel if missing. Public variant of `ensure_kernel`
    /// for callers that need to JIT a kernel by name from outside the crate
    /// (primarily the hipGraph capture/replay path).
    pub fn ensure_kernel_public(
        &mut self,
        module_name: &str,
        source: &str,
        func_name: &str,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(module_name, source, func_name)
    }

    /// Launch a pre-loaded kernel by name using the `extra`-mode kernarg
    /// blob path. This is the only launch path that survives hipGraph
    /// capture on gfx1100 / ROCm 6.x — the traditional `kernelParams`
    /// (`void**`) path records stack pointers that dangle by the time the
    /// captured graph is replayed.
    ///
    /// Caller is responsible for:
    ///  - keeping `kernargs` alive across the life of any graph that
    ///    captured this launch (HIP records the blob pointer, not the data);
    ///  - building `kernargs` with the layout matching the kernel signature
    ///    (use `hip_bridge::KernargBlob` for correct alignment).
    pub fn launch_kernel_blob(
        &self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernargs: &mut [u8],
    ) -> HipResult<()> {
        self.bind_thread()?;
        let func = self.functions.get(func_name).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!("launch_kernel_blob: function '{func_name}' not loaded"),
            )
        })?;
        unsafe {
            self.hip
                .launch_kernel_blob(func, grid, block, shared_mem, self.stream_ref(), kernargs)
        }
        .map_err(|e| e.with_kernel(func_name))
    }
    /// Occupancy probe for a pre-loaded kernel by name (oracle metadata gate).
    pub fn occupancy_max_active_blocks(
        &self,
        func_name: &str,
        block: [u32; 3],
        shared_mem: u32,
    ) -> HipResult<i32> {
        self.bind_thread()?;
        let func = self.functions.get(func_name).ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                &format!("occupancy_max_active_blocks: function '{func_name}' not loaded"),
            )
        })?;
        let block_size = block[0] * block[1] * block[2];
        self.hip
            .occupancy_max_active_blocks(func, block_size, shared_mem as usize)
    }

    /// Launch an already-built kernarg blob through the recorder-aware funnel.
    ///
    /// Blob-shaped twin of [`Self::launch_maybe_blob`] for callers that assemble
    /// a `hip_bridge::KernargBlob` directly (the shared tensor/grouped op
    /// owners). It reaches the same recorder, captures the same exact bytes, and
    /// keeps the same HipGraph capture-blob accounting as the params-shaped
    /// funnel, so a launch cannot be captured in one tape and missed in the
    /// other. [`Self::launch_kernel_blob`] stays the raw entry for the funnel
    /// itself and for the recorded-HIP oracle, which must never re-record.
    pub fn launch_blob_recorded(
        &mut self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernargs: &mut [u8],
        bindings: ReplayLaunchBindings<'_>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // Fail closed before the recorder / capture tapes see the launch.
        self.hip
            .validate_launch_grid(grid)
            .map_err(|e| e.with_kernel(func_name))?;
        if self.replay.is_g0_observing() {
            self.replay
                .observe_g0_launch(func_name, grid, block, shared_mem, kernargs);
        }
        if self.replay.is_recording() {
            let artifact = recorded_launch_artifact(&self.compiler, func_name);
            self.replay.record_hip_launch_typed_bound(
                &self.hip,
                func_name,
                artifact,
                grid,
                block,
                shared_mem,
                kernargs,
                bindings.grid,
                bindings.kernargs,
                Some(&self.compiler),
                &[],
            );
        }
        let result = if self.graphs.capture_mode {
            // Retain the exact bytes for graph instantiation: HIP records the
            // blob pointer, so graph-owned storage must outlive the capture.
            self.graphs.capture_blobs.push(kernargs.to_vec());
            let buf = self.graphs.capture_blobs.last_mut().unwrap();
            let func = self.functions.get(func_name).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("launch_blob_recorded: function '{func_name}' not loaded"),
                )
            })?;
            // SAFETY: HIP consumes the contiguous argument bytes during this
            // one-shot launch; graph-owned storage stays live through
            // instantiation and replay.
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    grid,
                    block,
                    shared_mem,
                    self.active_stream.as_ref(),
                    buf.as_mut_slice(),
                )
            }
        } else {
            let func = self.functions.get(func_name).ok_or_else(|| {
                hip_bridge::HipError::new(
                    0,
                    &format!("launch_blob_recorded: function '{func_name}' not loaded"),
                )
            })?;
            // SAFETY: HIP consumes the contiguous argument bytes during this
            // one-shot launch; `kernargs` remains live across the call.
            unsafe {
                self.hip.launch_kernel_blob(
                    func,
                    grid,
                    block,
                    shared_mem,
                    self.stream_ref(),
                    kernargs,
                )
            }
        };
        if result.is_ok() {
            self.last_kernel = Some(func_name.to_string());
        }
        result.map_err(|e| e.with_kernel(func_name))
    }

    /// Compile and load a kernel, caching the result.
    pub(crate) fn ensure_kernel(
        &mut self,
        module_name: &str,
        source: &str,
        func_name: &str,
    ) -> HipResult<()> {
        crate::scratch::compile_and_load_kernel(
            &mut self.compiler,
            &self.hip,
            &mut self.modules,
            &mut self.functions,
            module_name,
            source,
            func_name,
        )
    }

    /// Load the admitted builder bundle into the same module/function cache as
    /// JIT kernels. The embedded image keeps runtime independent of hipfire-isa
    /// and of a working-tree-relative artifact path.
    pub(crate) fn ensure_embedded_kernel(
        &mut self,
        module_name: &str,
        image: &[u8],
        func_name: &str,
    ) -> HipResult<()> {
        if self.functions.contains_key(func_name) {
            return Ok(());
        }
        if !self.modules.contains_key(module_name) {
            self.modules
                .insert(module_name.to_owned(), self.hip.module_load_data(image)?);
        }
        let func = self
            .hip
            .module_get_function(&self.modules[module_name], func_name)?;
        self.functions.insert(func_name.to_owned(), func);
        Ok(())
    }

    /// Ensure the FP16 X scratch contains the conversion of `x`. Skips the
    /// convert kernel if `x.buf.as_ptr()` matches the last converted source.
    /// Returns the FP16 device pointer.
    /// The QSA selector's global score rows, at least `bytes`. Growth follows
    /// the other scratch slots: invalidate captured graphs and re-arm replay
    /// before the old buffer is freed. A model's first use already requests
    /// its final size (the bound comes from the arena capacity, not the
    /// position), so a retained decode tape never sees the pointer move.
    pub(crate) fn qsa_select_scores(&mut self, bytes: usize) -> HipResult<*mut c_void> {
        if crate::scratch::scratch_will_grow(
            self.scratch.qsa_select_scores_bytes,
            self.scratch.qsa_select_scores.is_some(),
            bytes,
        ) {
            self.invalidate_for_scratch_growth();
            crate::scratch::grow_scratch_slot(
                &self.hip,
                &mut self.scratch.qsa_select_scores,
                &mut self.scratch.qsa_select_scores_bytes,
                bytes,
            )?;
        }
        Ok(self.scratch.qsa_select_scores.as_ref().unwrap().as_ptr())
    }

    /// The F32 working copy of a Q8 GDN state (the chunked prefill route runs
    /// on F32 state), at least `bytes`; grows like the other scratch slots.
    pub(crate) fn gdn_state_f32_scratch(&mut self, bytes: usize) -> HipResult<*mut c_void> {
        if crate::scratch::scratch_will_grow(
            self.scratch.gdn_state_f32_bytes,
            self.scratch.gdn_state_f32.is_some(),
            bytes,
        ) {
            self.invalidate_for_scratch_growth();
            crate::scratch::grow_scratch_slot(
                &self.hip,
                &mut self.scratch.gdn_state_f32,
                &mut self.scratch.gdn_state_f32_bytes,
                bytes,
            )?;
        }
        Ok(self.scratch.gdn_state_f32.as_ref().unwrap().as_ptr())
    }

    /// Whether this Gpu backs the gathered QSA workspace with a stable-VA
    /// VMM owner: gfx1151 or gfx1201 (the arches with evidence for it), a
    /// platform the VMM KV backend is certified on (not native Windows, not
    /// WSL without its unsafe override), and a runtime answering the VMM
    /// granularity query. Everything else keeps the legacy contiguous slot.
    pub(crate) fn qsa_gather_vmm_capable(&self) -> bool {
        (self.arch_caps.is_gfx1151() || self.arch_caps.is_gfx1201())
            && hipfire_config::devices::vmm_kv_platform_refusal().is_none()
            && self.vmm_recommended_granularity().is_ok()
    }

    /// The backing in use, decided once per reservation: a live slot keeps
    /// its kind, an empty Gpu asks [`Self::qsa_gather_vmm_capable`].
    fn qsa_gather_uses_vmm(&self) -> bool {
        match (&self.scratch.qsa_gather_vmm, &self.scratch.qsa_gather_f16) {
            (Some(_), _) => true,
            (None, Some(_)) => false,
            (None, None) => self.qsa_gather_vmm_capable(),
        }
    }

    /// Reserve the gathered QSA attention's workspace for at least
    /// `reserve_bytes`. VMM ([`crate::scratch::ScratchState::qsa_gather_vmm`]):
    /// a stable virtual extent, committing no new pages; a reservation at
    /// least that large is kept, and replacing a smaller one (an address
    /// move) is refused while a capture or record is armed. Legacy
    /// ([`crate::scratch::ScratchState::qsa_gather_f16`]): the whole extent
    /// allocated, as before.
    pub(crate) fn qsa_gather_reserve(&mut self, reserve_bytes: usize) -> HipResult<()> {
        if !self.qsa_gather_uses_vmm() {
            return self.qsa_gather_legacy(reserve_bytes).map(drop);
        }
        let reserve_bytes = reserve_bytes.max(1);
        if self
            .scratch
            .qsa_gather_vmm
            .as_ref()
            .is_some_and(|workspace| workspace.byte_size() >= reserve_bytes)
        {
            return Ok(());
        }
        self.refuse_qsa_gather_growth_while_armed(reserve_bytes)?;
        if self.scratch.qsa_gather_vmm.is_some() {
            self.invalidate_for_scratch_growth();
            self.release_qsa_gather_workspace()?;
        }
        // SAFETY: launches touch only the mapped prefix; `qsa_gather_scratch`
        // maps every byte a launch reads or writes before returning its base.
        let workspace = unsafe { self.alloc_vmm_tensor(&[reserve_bytes], DType::Raw, 0, &[])? };
        self.scratch.qsa_gather_vmm_mapped = self.vmm_mapped_bytes(&workspace).unwrap_or(0);
        self.scratch.qsa_gather_vmm = Some(workspace);
        Ok(())
    }

    /// The gathered QSA workspace's base, covering at least its first
    /// `bytes`. VMM: reserved for at least `reserve_bytes`
    /// ([`Self::qsa_gather_reserve`]) and mapped in place at the stable
    /// address, rounded to the driver granularity; mapping more is refused
    /// while a capture or record is armed, so a captured graph or tape never
    /// sees the workspace change. Legacy: grown to `bytes` like the other
    /// scratch slots.
    pub(crate) fn qsa_gather_scratch(
        &mut self,
        bytes: usize,
        reserve_bytes: usize,
    ) -> HipResult<*mut c_void> {
        if !self.qsa_gather_uses_vmm() {
            return self.qsa_gather_legacy(bytes);
        }
        self.qsa_gather_reserve(reserve_bytes.max(bytes))?;
        if bytes > self.scratch.qsa_gather_vmm_mapped {
            self.refuse_qsa_gather_growth_while_armed(bytes)?;
            let mut workspace = self.scratch.qsa_gather_vmm.take().expect("reserved above");
            let grown = self.map_qsa_gather_prefix(&mut workspace, bytes);
            self.scratch.qsa_gather_vmm_mapped = self.vmm_mapped_bytes(&workspace).unwrap_or(0);
            self.scratch.qsa_gather_vmm = Some(workspace);
            grown?;
        }
        Ok(self.scratch.qsa_gather_vmm.as_ref().unwrap().buf.as_ptr())
    }

    /// The legacy slot, at least `bytes`. Grows like the other scratch slots
    /// (captured state is invalidated before the old buffer is freed); Qwen4
    /// reserves the whole context at load, so a forward never grows it.
    fn qsa_gather_legacy(&mut self, bytes: usize) -> HipResult<*mut c_void> {
        if crate::scratch::scratch_will_grow(
            self.scratch.qsa_gather_f16_bytes,
            self.scratch.qsa_gather_f16.is_some(),
            bytes,
        ) {
            self.invalidate_for_scratch_growth();
            crate::scratch::grow_scratch_slot(
                &self.hip,
                &mut self.scratch.qsa_gather_f16,
                &mut self.scratch.qsa_gather_f16_bytes,
                bytes,
            )?;
        }
        Ok(self.scratch.qsa_gather_f16.as_ref().unwrap().as_ptr())
    }

    fn map_qsa_gather_prefix(&mut self, workspace: &mut GpuTensor, bytes: usize) -> HipResult<()> {
        let mapped = self.vmm_mapped_bytes(workspace).unwrap_or(0);
        let granularity = self.vmm_granularity(workspace).unwrap_or(0);
        let grow = crate::scratch::vmm_prefix_growth(mapped, workspace.byte_size(), granularity, bytes)
            .ok_or_else(|| {
                HipError::new(
                    0,
                    &format!(
                        "QSA gather workspace cannot map {bytes} bytes of a {}-byte reservation",
                        workspace.byte_size()
                    ),
                )
            })?;
        if grow > 0 {
            self.grow_vmm_tensor(workspace, grow, &[])?;
        }
        Ok(())
    }

    fn refuse_qsa_gather_growth_while_armed(&self, bytes: usize) -> HipResult<()> {
        if self.graphs.capture_mode || self.replay.is_recording() {
            return Err(HipError::new(
                0,
                &format!(
                    "QSA gather workspace growth to {bytes} bytes refused while a capture or record is armed; map it before capture"
                ),
            ));
        }
        Ok(())
    }

    /// Release the VMM gathered QSA workspace (the legacy slot is kept for
    /// the process, as before). It is this Gpu's own scratch, not a model
    /// owner, and its address never enters captured or recorded state (the
    /// route is eager-only and growth refuses while armed); a later launch
    /// reserves afresh. A failed release is retained in the VMM pending table
    /// for [`Self::ensure_vmm_cleaned`] to retry.
    pub(crate) fn release_qsa_gather_workspace(&mut self) -> HipResult<()> {
        self.scratch.qsa_gather_vmm_mapped = 0;
        match self.scratch.qsa_gather_vmm.take() {
            Some(workspace) => self.free_tensor(workspace),
            None => Ok(()),
        }
    }

    /// Physically committed bytes of the gathered QSA attention's workspace
    /// (0 before the first reservation): the VMM mapped prefix, rounded to
    /// the driver granularity, or the legacy allocation.
    pub fn qsa_gather_scratch_bytes(&self) -> usize {
        if self.scratch.qsa_gather_vmm.is_some() {
            self.scratch.qsa_gather_vmm_mapped
        } else if self.scratch.qsa_gather_f16.is_some() {
            self.scratch.qsa_gather_f16_bytes
        } else {
            0
        }
    }

    /// Reserved bytes of the gathered QSA attention's workspace (0 before the
    /// first reservation): the VMM virtual extent, or the legacy allocation.
    pub fn qsa_gather_reserved_bytes(&self) -> usize {
        match &self.scratch.qsa_gather_vmm {
            Some(workspace) => workspace.byte_size(),
            None => self.qsa_gather_scratch_bytes(),
        }
    }

    pub(crate) fn ensure_fp16_x(
        &mut self,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        // Scratch growth under a captured graph would free a pointer the graph
        // embeds: invalidate first (no-op unless captured). See
        // `invalidate_for_scratch_growth`.
        if crate::scratch::scratch_will_grow(
            self.scratch.fp16_x_scratch_bytes,
            self.scratch.fp16_x_scratch.is_some(),
            n_elems * 2,
        ) {
            self.invalidate_for_scratch_growth();
        }
        // Split borrows so `self.replay` and `self.graphs`/`self.scratch` can be
        // borrowed simultaneously. The scratch helper will record into replay
        // when `is_recording()` and push to capture_blobs when `capture_mode`,
        // using the unified `record || capture_mode || force_blob` gate.
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        self.scratch.ensure_fp16_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            x,
            n_elems,
        )
    }

    /// Ensure the deterministic-ksplit partials scratch is at least `n_bytes`.
    pub(crate) fn ensure_ksplit_det_partials(&mut self, n_bytes: usize) -> HipResult<*mut c_void> {
        if crate::scratch::scratch_will_grow(
            self.scratch.ksplit_det_partials_bytes,
            self.scratch.ksplit_det_partials.is_some(),
            n_bytes,
        ) {
            self.invalidate_for_scratch_growth();
        }
        self.scratch.ensure_ksplit_det_partials(&self.hip, n_bytes)
    }

    /// Convert F32 to F16 without caching. Used when the same x tensor
    /// pointer is reused with different contents across layers, where
    /// pointer-keyed caching would read stale FP16.
    pub(crate) fn convert_fp16_x_uncached(
        &mut self,
        x: &GpuTensor,
        n_elems: usize,
    ) -> HipResult<*mut c_void> {
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        if crate::scratch::scratch_will_grow(
            self.scratch.fp16_x_scratch_bytes,
            self.scratch.fp16_x_scratch.is_some(),
            n_elems * 2,
        ) {
            self.invalidate_for_scratch_growth();
        }
        self.scratch.convert_fp16_x_uncached(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            x,
            n_elems,
        )
    }

    /// Ensure the FP8 (E4M3) X scratch contains the conversion of `x`
    /// (an F32 GpuTensor). Returns the FP8 device pointer. gfx12 only —
    /// uses cvt_pk_fp8_f32. Caches by `x.buf.as_ptr()` like its FP16
    /// sibling so back-to-back same-X GEMM dispatches skip reconversion.
    pub(crate) fn ensure_fp8_x(&mut self, x: &GpuTensor, n_elems: usize) -> HipResult<*mut c_void> {
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        if crate::scratch::scratch_will_grow(
            self.scratch.fp8_x_scratch_bytes,
            self.scratch.fp8_x_scratch.is_some(),
            n_elems,
        ) {
            self.invalidate_for_scratch_growth();
        }
        self.scratch.ensure_fp8_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            x,
            n_elems,
        )
    }

    /// Prepare MQ4v2 FP8 activation scratch from `x`: F32 inputs pack directly
    /// via `pack_f32_to_fp8_mq4v2_gfx12` (single F32->E4M3 rounding; see the
    /// kernel comment for the accepted last-ulp note vs the F16 path's double
    /// rounding); non-F32 inputs keep the canonical-F16 + `pack_f16_to_fp8`
    /// route. Always overwrites the dedicated X8 / half-sum / row-scale
    /// buffers. Eager-only research entry — rejects graph capture and replay
    /// recording before any conversion or launch (candidate is not on the
    /// capture ABI).
    ///
    /// Pointers in the returned [`crate::scratch::Mq4v2Fp8Prepared`] belong to
    /// this `Gpu` and remain valid only until the next call or teardown.
    pub fn prepare_mq4v2_fp8_x(
        &mut self,
        x: &GpuTensor,
        n: usize,
        k: usize,
        scale_mode: i32,
    ) -> HipResult<crate::scratch::Mq4v2Fp8Prepared> {
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(
                0,
                "prepare_mq4v2_fp8_x: eager-only (capture/replay rejected)",
            ));
        }
        // All three MQ4v2 FP8 buffers grow below (both sub-paths); invalidate
        // first if any of them will. Eager-only is already enforced above.
        {
            let (x8, half_sums, row_scales) = crate::scratch::mq4v2_fp8_needed(n, k);
            let s = &self.scratch;
            if crate::scratch::scratch_will_grow(
                s.mq4v2_fp8_x_scratch_bytes,
                s.mq4v2_fp8_x_scratch.is_some(),
                x8,
            ) || crate::scratch::scratch_will_grow(
                s.mq4v2_fp8_half_sums_scratch_bytes,
                s.mq4v2_fp8_half_sums_scratch.is_some(),
                half_sums,
            ) || crate::scratch::scratch_will_grow(
                s.mq4v2_fp8_row_scales_scratch_bytes,
                s.mq4v2_fp8_row_scales_scratch.is_some(),
                row_scales,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        if x.dtype == crate::DType::F32 {
            return self.scratch.prepare_mq4v2_fp8_x_f32(
                &self.hip,
                &mut self.compiler,
                &mut self.modules,
                &mut self.functions,
                self.active_stream.as_ref(),
                &mut self.graphs.capture_blobs,
                capture_mode,
                force_blob,
                &mut self.replay,
                x.buf.as_ptr(),
                n,
                k,
                scale_mode,
            );
        }
        let n_elems = n
            .checked_mul(k)
            .ok_or_else(|| HipError::new(0, "prepare_mq4v2_fp8_x: n*k overflow"))?;
        let x_f16 = self.ensure_fp16_x(x, n_elems)?;
        self.scratch.prepare_mq4v2_fp8_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            x_f16,
            n,
            k,
            scale_mode,
        )
    }

    /// Ensure prefill activations are quantized into a llama.cpp-style
    /// `block_q8_1_mmq` layout. The scratch is ordered by [K/128 block, batch]
    /// so a 128-column batch tile is contiguous for each K tile.
    pub fn ensure_q8_1_mmq_x(
        &mut self,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        // bind_thread: skip — delegated to scratch.rs
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        {
            let needed = crate::scratch::q8_1_mmq_x_needed(k, batch_size);
            if crate::scratch::scratch_will_grow(
                self.scratch.q8_1_mmq_x_scratch_bytes,
                self.scratch.q8_1_mmq_x_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.ensure_q8_1_mmq_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            self.device_id,
            x,
            batch_size,
            k,
        )
    }
    /// Ensure prefill activations are quantized at per-128 granularity
    /// (X128 prelude for the `HIPFIRE_GFX11_MMQ_X128` opt-in path).
    /// See `scratch.rs::ensure_q8_1_mmq_x128`.
    pub fn ensure_q8_1_mmq_x128(
        &mut self,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        // bind_thread: skip — delegated to scratch.rs
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        {
            let needed = crate::scratch::q8_1_mmq_x_needed(k, batch_size);
            if crate::scratch::scratch_will_grow(
                self.scratch.q8_1_mmq_x_scratch_bytes,
                self.scratch.q8_1_mmq_x_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.ensure_q8_1_mmq_x128(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            self.device_id,
            x,
            batch_size,
            k,
        )
    }
    /// Ensure prefill activations are quantized to int4 (`block_i4_128`) for
    /// the iu4-direct MMQ consumer (`HIPFIRE_IU4_PREFILL` path).
    /// See `scratch.rs::ensure_int4_mmq_x`.
    pub fn ensure_int4_mmq_x(
        &mut self,
        x: &GpuTensor,
        batch_size: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        // bind_thread: skip — delegated to scratch.rs
        let capture_mode = self.graphs.capture_mode;
        let force_blob = self.flags.force_blob_path;
        {
            let needed = crate::scratch::int4_mmq_x_needed(k, batch_size);
            if crate::scratch::scratch_will_grow(
                self.scratch.int4_mmq_x_scratch_bytes,
                self.scratch.int4_mmq_x_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.ensure_int4_mmq_x(
            &self.hip,
            &mut self.compiler,
            &mut self.modules,
            &mut self.functions,
            self.active_stream.as_ref(),
            &mut self.graphs.capture_blobs,
            capture_mode,
            force_blob,
            &mut self.replay,
            self.device_id,
            x,
            batch_size,
            k,
        )
    }

    /// Reserve `int4_mmq_x_scratch` for a producer-emitted IU4 sidecar (C2).
    /// Does not launch the standalone quantizer.
    pub fn reserve_int4_mmq(
        &mut self,
        k: usize,
        n: usize,
    ) -> HipResult<crate::scratch::Int4MmqReservation> {
        // Mirror `reserve_int4_mmq`'s validity gate: no growth on the error path.
        if k != 0 && n != 0 && k % 256 == 0 {
            let needed = crate::scratch::int4_mmq_reserve_needed(k, n);
            if crate::scratch::scratch_will_grow(
                self.scratch.int4_mmq_x_scratch_bytes,
                self.scratch.int4_mmq_x_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.reserve_int4_mmq(&self.hip, k, n)
    }

    /// Reserve `int4_mmq_down_scratch` for the A4-fused gate/up epilogue's
    /// down-proj sidecar. Never aliases `int4_mmq_x_scratch`.
    pub fn reserve_int4_mmq_down(
        &mut self,
        k: usize,
        n: usize,
    ) -> HipResult<crate::scratch::Int4MmqDownReservation> {
        // Mirror `reserve_int4_mmq`'s validity gate: no growth on the error path.
        if k != 0 && n != 0 && k % 256 == 0 {
            let needed = crate::scratch::int4_mmq_reserve_needed(k, n);
            if crate::scratch::scratch_will_grow(
                self.scratch.int4_mmq_down_scratch_bytes,
                self.scratch.int4_mmq_down_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.reserve_int4_mmq_down(&self.hip, k, n)
    }

    /// Re-quantize f32 activations for the gfx1201 A8 MMQ consumer.
    pub fn ensure_int8_mmq_x(&mut self, x: &GpuTensor, n: usize, k: usize) -> HipResult<*mut c_void> {
        let needed = crate::scratch::int8_mmq_x_needed(k, n);
        if crate::scratch::scratch_will_grow(
            self.scratch.int8_mmq_x_scratch_bytes,
            self.scratch.int8_mmq_x_scratch.is_some(),
            needed,
        ) {
            self.invalidate_for_scratch_growth();
        }
        self.scratch.ensure_int8_mmq_x(
            &self.hip, &mut self.compiler, &mut self.modules, &mut self.functions,
            self.active_stream.as_ref(), &mut self.graphs.capture_blobs,
            self.graphs.capture_mode, self.flags.force_blob_path, &mut self.replay,
            self.device_id, x, n, k,
        )
    }

    /// Reserve (without launching the quantizer) for a fused A8 producer.
    pub fn reserve_int8_mmq(&mut self, k: usize, n: usize) -> HipResult<crate::scratch::Int8MmqReservation> {
        if k != 0 && n != 0 && k % 256 == 0 {
            let needed = crate::scratch::int8_mmq_x_needed(k, n);
            if crate::scratch::scratch_will_grow(
                self.scratch.int8_mmq_x_scratch_bytes,
                self.scratch.int8_mmq_x_scratch.is_some(),
                needed,
            ) {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch.reserve_int8_mmq(&self.hip, k, n)
    }

    pub fn int8_mmq_prepared_ptr(
        &self, prepared: &crate::scratch::Int8MmqPrepared, k: usize, n: usize,
    ) -> HipResult<*mut c_void> {
        let (generation, ptr) = self.scratch.int8_mmq_live();
        prepared.checked_ptr(generation, ptr, k, n)
    }

    pub fn a8_prefill_active(&self, n: usize, k: usize) -> bool {
        self.flags.a8_prefill_enabled()
            && self.mq4v2_symmetric
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && n >= 64 && k > 0 && k % 256 == 0
    }

    /// True when the portable producer-sidecar route is live for this call:
    /// IU4 + gfx1100/gfx1151 + eager (no replay/capture) + the K constraint
    /// of the IU4 MMQ consumer. Every producer grid is row-parallel, and the
    /// gfx11 IU4 MMQ base entry handles partial 128-row tiles.
    pub fn iu4_producer_sidecar_active(&self, batch: usize, k: usize) -> bool {
        self.flags.iu4_producer_sidecar_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch >= 64
            && k > 0
            && k % 256 == 0
    }
    /// True when the gfx1201 slice-1 silu+quant fusion is live for this call:
    /// IU4 + exact gfx1201 + eager (no replay/capture) + K constraint of the
    /// iu4 MMQ consumer. Any batch >= 64: every producer grid is row-parallel
    /// in batch (one row per workgroup), and the gfx1201 iu4 MMQ consumer
    /// handles partial 128-row tiles natively (zero-filled slab / guarded
    /// writeback), so the old %128 multiple was grid convenience only.
    /// Default on for the IU4 route (`HIPFIRE_GFX12_SILU_QUANT_FUSED=0` opts out).
    pub fn iu4_silu_quant_fused_active(&self, batch: usize, k: usize) -> bool {
        self.flags.gfx12_silu_quant_fused_enabled()
            && !self.a8_prefill_active(batch, k)
            && self.flags.iu4_prefill_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch >= 64
            && k > 0
            && k % 256 == 0
    }
    /// True when a gfx1201 slices-2-4 producer+quant fusion is live for this
    /// call: IU4 + `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED` on exact gfx1201 +
    /// eager (no replay/capture) + K constraint of the iu4 MMQ consumer. Any
    /// batch >= 64, same row-parallel justification as
    /// `iu4_silu_quant_fused_active`; the GDN fused producer additionally
    /// requires head_dim == 128 at its callsite helper.
    pub fn iu4_producer_quant_fused_active(&self, batch: usize, k: usize) -> bool {
        self.flags.gfx12_producer_quant_fused_enabled()
            && !self.a8_prefill_active(batch, k)
            && self.flags.iu4_prefill_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch >= 64
            && k > 0
            && k % 256 == 0
    }
    /// True when a gfx11 sigmoid/gated-norm producer+quant fusion is live
    /// for this call: IU4 + `HIPFIRE_GFX11_PRODUCER_QUANT_FUSED` on
    /// gfx1100/gfx1151 + eager (no replay/capture) + K constraint of the
    /// iu4 MMQ consumer. Any batch >= 64, same row-parallel justification
    /// as `iu4_producer_quant_fused_active`; the GDN fused producer
    /// additionally requires head_dim == 128 at its callsite helper.
    pub fn iu4_gfx11_producer_quant_fused_active(&self, batch: usize, k: usize) -> bool {
        self.flags.gfx11_producer_quant_fused_enabled()
            && self.flags.iu4_prefill_enabled()
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch >= 64
            && k > 0
            && k % 256 == 0
    }
    /// True when the gfx1201 FP8-stream producer fusion is live for this
    /// call: `HIPFIRE_GFX12_FP8_STREAM=1` on exact gfx1201 + eager (no
    /// replay/capture) + K constraint of the MQ4v2 FP8 consumer (`k % 256
    /// == 0`). Any batch >= 64: the producer grid is row-parallel in batch
    /// (one workgroup per row), and the fp8 kernels mask partial N tiles
    /// in-kernel (clamped loads, `oc < N` guarded stores), so the old %64
    /// multiple was grid convenience only. The callsite additionally
    /// requires Lloyd weights and scale_mode == 1. Default OFF.
    pub fn fp8_stream_active(&self, batch: usize, k: usize) -> bool {
        self.flags.gfx12_fp8_stream_enabled()
            && !self.a8_prefill_active(batch, k)
            && self.arch == "gfx1201"
            && !self.replay.is_recording()
            && !self.graphs.capture_mode
            && batch >= 64
            && k > 0
            && k % 256 == 0
    }

    /// Validate a prepared IU4 handle against the live scratch generation.
    pub fn int4_mmq_prepared_ptr(
        &self,
        prepared: &crate::scratch::Int4MmqPrepared,
        k: usize,
        n: usize,
    ) -> HipResult<*mut c_void> {
        let (gen, ptr) = self.scratch.int4_mmq_live();
        prepared.checked_ptr(gen, ptr, k, n)
    }

    /// Validate a prepared A4-fused down-sidecar handle against the live
    /// down scratch generation.
    pub fn int4_mmq_down_prepared_ptr(
        &self,
        prepared: &crate::scratch::Int4MmqDownPrepared,
        k: usize,
        n: usize,
    ) -> HipResult<*mut c_void> {
        let (gen, ptr) = self.scratch.int4_mmq_down_live();
        prepared.checked_ptr(gen, ptr, k, n)
    }

    /// Returns the number of launches recorded by the `ReplayController`.
    /// Together with `self.graphs.capture_blobs.len()`, this must agree for
    /// any body — see the `Gpu` type-level invariant doc.
    pub fn recorded_launch_count(&self) -> usize {
        // bind_thread: skip — reads the recorded replay tape, no HIP calls.
        self.replay.recorded_launches().len()
    }

    /// Returns the number of HipGraph kernarg blobs captured.
    /// See `recorded_launch_count` and the `Gpu` invariant.
    pub fn graph_blob_count(&self) -> usize {
        // bind_thread: skip — reads a captured-graph counter, no HIP calls.
        self.graphs.capture_blobs.len()
    }

    /// Debug-only assertion that the HipGraph and Replay tapes would agree.
    /// Call after a body that was captured via both mechanisms (or after two
    /// separate captures of the same body, passing the other count).
    /// A mismatch indicates a helper bypassed the replay recorder.
    pub fn debug_assert_tape_parity(&self, other_blob_count: Option<usize>) {
        // bind_thread: skip — pure comparison of the two local tapes above;
        // no HIP calls.
        let replay_len = self.recorded_launch_count();
        let graph_len = other_blob_count.unwrap_or_else(|| self.graph_blob_count());
        debug_assert_eq!(
            replay_len, graph_len,
            "tape parity violated: replay recorded {} launches but HipGraph has {} blobs — a helper bypassed the recorder",
            replay_len, graph_len
        );
    }

    /// Screen a weight matrix for MMQ safety (#87). Runs a small synthetic
    /// comparison (batch=16): f16 WMMA vs MMQ on random activations. If any
    /// output row's max abs error exceeds `mmq_screen_threshold`, the weight
    /// is marked unsafe. Result is cached by device pointer.
    ///
    /// Eager-only, and it never touches capture state: its scratch growth
    /// invalidates captured graphs like any other eager call, and its
    /// launches are not blob-recorded. Inside a capture or PM4 recording it
    /// does not screen (sync copies, pool temps freed on return) and reports
    /// `false` without caching, so the weight takes the WMMA path there and is
    /// screened at its next eager use.
    ///
    /// Returns `true` if MMQ is safe for this weight, `false` if it should
    /// fall back to WMMA.
    pub fn mmq_screen_weight(&mut self, a_raw: &GpuTensor, m: usize, k: usize) -> bool {
        self.bind_thread_or_warn();
        let key = a_raw.buf.as_ptr() as usize;
        if let Some(&safe) = self.mmq_screen.cache.get(&key) {
            return safe;
        }
        if self.graphs.capture_mode || self.replay.is_recording() {
            return false;
        }

        let screen_batch = 16usize;
        let threshold = self.mmq_screen.threshold;

        // Generate synthetic activations on CPU
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let x_data: Vec<f32> = (0..screen_batch * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let t = (state >> 33) as f32 / (u32::MAX as f32);
                t * 4.0 - 2.0
            })
            .collect();

        let result = (|| -> HipResult<bool> {
            let x_gpu = self.upload_f32(&x_data, &[screen_batch * k])?;
            let y_wmma = self.zeros(&[screen_batch * m], DType::F32)?;
            let y_mmq = self.zeros(&[screen_batch * m], DType::F32)?;

            // Reference path: use FP16 wave64 on gfx906, WMMA otherwise
            if self.arch_caps.is_gfx906() {
                self.gemm_hfq4g256_residual_fp16_wave64(
                    a_raw,
                    &x_gpu,
                    &y_wmma,
                    m,
                    k,
                    screen_batch,
                )?;
            } else {
                self.gemm_hfq4g256_residual_wmma(a_raw, &x_gpu, &y_wmma, m, k, screen_batch)?;
            }

            // MMQ path
            let xq = self.ensure_q8_1_mmq_x(&x_gpu, screen_batch, k)?;
            if self.arch_caps.is_gfx906() {
                self.gemm_hfq4g256_residual_mmq_gfx906(a_raw, &x_gpu, &y_mmq, m, k, screen_batch)?;
            } else {
                self.gemm_hfq4g256_mmq_set_prequant(a_raw, xq, &y_mmq, m, k, screen_batch)?;
            }

            self.hip.device_synchronize()?;

            let ref_out = self.download_f32(&y_wmma)?;
            let mmq_out = self.download_f32(&y_mmq)?;

            self.free_tensor(x_gpu).ok();
            self.free_tensor(y_wmma).ok();
            self.free_tensor(y_mmq).ok();

            // Per-row max error check
            let mut worst_err = 0f32;
            for r in 0..m {
                let mut row_max = 0f32;
                for b in 0..screen_batch {
                    let idx = b * m + r;
                    let err = (ref_out[idx] - mmq_out[idx]).abs();
                    if err > row_max {
                        row_max = err;
                    }
                }
                if row_max > worst_err {
                    worst_err = row_max;
                }
            }

            let safe = worst_err <= threshold;
            Ok(safe)
        })();

        let safe = result.unwrap_or_else(|e| {
            eprintln!("  MMQ screen: error during screening ({e}), assuming unsafe");
            false
        });
        self.mmq_screen.cache.insert(key, safe);
        safe
    }

    /// Ensure an FP16 shadow of `w_mq4` (HFQ4-G256 format, [M × K]) exists in
    /// `fp16_shadow_cache`. First call allocates M*K*2 bytes on device and
    /// runs the dequantize kernel; subsequent calls return the cached pointer.
    ///
    /// Cache is keyed on the MQ4 device pointer — this assumes weights are
    /// immutable after model load (standard in this engine). If the same
    /// pointer is ever reused for a different M or K, cache would return
    /// stale data: we don't try to detect that (weights don't reshape).
    ///
    /// Returns `None` if rocBLAS is not loaded (caller should fall back to
    /// the hand-rolled GEMV path). Memory is freed when the Gpu drops.
    pub(crate) fn ensure_fp16_shadow(
        &mut self,
        w_mq4: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<Option<*mut c_void>> {
        if self.rocblas.is_none() {
            return Ok(None);
        }
        let key = w_mq4.buf.as_ptr() as usize;
        if let Some(shadow) = self.fp16_shadow_cache.get(&key) {
            return Ok(Some(shadow.buf.as_ptr()));
        }
        // Allocate + dequantize. Use alloc_tensor so the shadow follows the
        // same GpuTensor hygiene (tracked in pool if applicable).
        let fp16 = self.alloc_tensor(&[m * k], DType::F16)?;
        self.dequantize_hfq4g256_to_f16(&w_mq4.buf, &fp16.buf, m, k)?;
        let ptr = fp16.buf.as_ptr();
        self.fp16_shadow_cache.insert(key, fp16);
        Ok(Some(ptr))
    }

    /// Model-lifetime F16 shadow of a BF16 `[M × K]` weight, keyed on its
    /// device pointer (weights are immutable after load); freed with the other
    /// shadows on unload.  BF16 -> F16 rounds (and flushes values below the F16
    /// range): callers must be accuracy-gated.
    pub(crate) fn ensure_bf16_f16_shadow(
        &mut self,
        weight: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<*mut c_void> {
        let key = weight.buf.as_ptr() as usize;
        if let Some(shadow) = self.fp16_shadow_cache.get(&key) {
            return Ok(shadow.buf.as_ptr());
        }
        let n = m * k;
        let fp16 = self.alloc_tensor(&[n], DType::F16)?;
        self.ensure_kernel(
            "gemm_bf16_xf32_multirow",
            kernels::GEMM_BF16_XF32_MULTIROW_SRC,
            "convert_bf16_to_f16",
        )?;
        let ip = weight.buf.as_ptr();
        let op = fp16.buf.as_ptr();
        let nv =
            i32::try_from(n).map_err(|_| hip_bridge::HipError::new(0, "BF16 shadow too large"))?;
        let mut params = [
            &ip as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "convert_bf16_to_f16",
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(ip);
                b.push_ptr(op);
                b.push_i32(nv);
                b
            },
        )?;
        let ptr = fp16.buf.as_ptr();
        self.fp16_shadow_cache.insert(key, fp16);
        Ok(ptr)
    }

    /// Ensure a model-lifetime FP16 shadow of a qt=35 MFP4G32E8SOA matrix.
    /// This entry is deliberately separate from the HFQ4 helper so adding the
    /// DeepSeek4 CDNA path cannot change any existing Qwen/HFQ4 dequant route.
    pub(crate) fn ensure_mfp4e8_soa_fp16_shadow_gfx942(
        &mut self,
        weight: &GpuTensor,
        m: usize,
        k: usize,
    ) -> HipResult<Option<*mut c_void>> {
        if self.arch != "gfx942" || self.rocblas.is_none() {
            return Ok(None);
        }
        debug_assert_eq!(weight.dtype, DType::MFP4G32E8SOA);
        let key = weight.buf.as_ptr() as usize;
        if let Some(shadow) = self.fp16_shadow_cache.get(&key) {
            return Ok(Some(shadow.buf.as_ptr()));
        }
        let fp16 = self.alloc_tensor(&[m * k], DType::F16)?;
        self.dequantize_mfp4g32_e8_soa_to_f16_gfx942(&weight.buf, &fp16.buf, m, k)?;
        let ptr = fp16.buf.as_ptr();
        self.fp16_shadow_cache.insert(key, fp16);
        Ok(Some(ptr))
    }

    /// Whether the arch is eligible for the rocBLAS/MFMA batched-prefill
    /// path. Default: CDNA3 only (MI300-series, gfx94x). Override with
    /// `HIPFIRE_ROCBLAS_ALL_ARCHS=1` for local testing on RDNA3+ — rocBLAS
    /// runs fine there (uses WMMA backends on RDNA3, not MFMA) so this is
    /// a useful smoke-path in the absence of an MI300.
    pub(crate) fn rocblas_arch_eligible(&self) -> bool {
        let all_archs = self.flags.rocblas_all_archs;
        if all_archs {
            return self.rocblas.is_some();
        }
        self.arch_caps.is_cdna3()
    }

    /// Configurable batch threshold for MFMA dispatch. Below this we stay on
    /// the hand-rolled GEMV — rocBLAS launch overhead eats the compute win
    /// at tiny batches. Overridable via `HIPFIRE_ROCBLAS_MIN_BATCH` env var.
    ///
    /// Kill-switch: `HIPFIRE_ROCBLAS_OFF=1` forces the threshold to usize::MAX,
    /// which disables the rocBLAS path entirely for A/B benchmarking against
    /// the hand-rolled GEMV baseline.
    pub(crate) fn rocblas_min_batch(&self) -> usize {
        if self.flags.rocblas_off {
            return usize::MAX;
        }
        self.flags.rocblas_min_batch.unwrap_or(4)
    }

    /// Batched-attention tile size, from `HIPFIRE_ATTN_TILE_SIZE`.
    ///
    /// Falls back to 128 when unset, zero, or not a multiple of 32. This is the
    /// single resolver for the whole crate — three hand-mirrored copies of this
    /// logic previously caused silent device-memory corruption when one of them
    /// drifted (see docs/plans/2026-08-07-batched-attention-slots.md).
    ///
    /// Directional safety note: RAISING the tile is always safe. LOWERING it
    /// increases `max_tiles` and therefore the `partials` bytes each query row
    /// needs, which can exceed buffers sized elsewhere against the 128 default.
    pub fn attn_tile_size(&self) -> usize {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags
            .attn_tile_size
            .filter(|&t| t > 0 && t % 32 == 0)
            .unwrap_or(128)
    }

    /// Multi-slot attention flash-vs-scalar crossover override, in tokens.
    /// `None` leaves the per-arch default in place.
    pub fn slots_attn_crossover(&self) -> Option<usize> {
        // bind_thread: skip — pure flag read, touches no device state.
        self.flags.slots_attn_crossover
    }

    /// Pre-compile a batch of kernels in parallel (hipcc), then load modules + functions.
    /// Each entry is (module_name, source, func_name). Turbo kernels should have
    /// TURBO_COMMON_H already prepended in their source.
    pub fn precompile_kernels(&mut self, specs: &[(&str, &str, &str)]) -> HipResult<()> {
        self.bind_thread()?;
        let batch: Vec<(&str, &str, &str)> = specs
            .iter()
            .filter(|(_, _, func)| !self.functions.contains_key(*func))
            .copied()
            .collect();

        if batch.is_empty() {
            return Ok(());
        }

        // Parallel hipcc compilation
        self.compiler.compile_batch_for_symbols(&batch)?;

        // Now load modules + extract functions (must be sequential — GPU API calls)
        for &(module_name, source, func_name) in specs {
            if self.functions.contains_key(func_name) {
                continue;
            }
            let obj_path = self.compiler.compile_for_symbol(module_name, source, func_name)?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(module_name) {
                let module = crate::scratch::module_load_or_recompile(
                    &self.hip,
                    &mut self.compiler,
                    module_name,
                    source,
                    func_name,
                    &obj_path_str,
                )?;
                self.modules.insert(module_name.to_string(), module);
            }
            let module = &self.modules[module_name];
            let func = self.hip.module_get_function(module, func_name)?;
            self.functions.insert(func_name.to_string(), func);
        }
        Ok(())
    }

    // ── Tensor allocation ───────────────────────────────────────

    pub fn ensure_gemv_residual_tmp(&mut self, min_elems: usize) -> HipResult<&GpuTensor> {
        // bind_thread: skip — delegated to scratch.rs (takes device_id explicitly).
        {
            let needed_bytes = min_elems * 4;
            let will_grow = self
                .scratch
                .gemv_residual_tmp
                .as_ref()
                .map_or(true, |tmp| tmp.buf.size() < needed_bytes);
            if will_grow {
                self.invalidate_for_scratch_growth();
            }
        }
        self.scratch
            .ensure_gemv_residual_tmp(&self.hip, self.device_id, min_elems)
    }

    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let numel: usize = shape.iter().product();
        let byte_size = numel * dtype.size();
        let buf = self.pool.alloc(&self.hip, byte_size)?;
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Reserve a dense virtual tensor and optionally map its initial prefix.
    /// The returned tensor follows the normal kernel ABI, but its ownership is
    /// registered separately so `free_tensor` releases VMM handles instead of
    /// routing the address through `hipFree`.
    ///
    /// # Safety
    ///
    /// The returned tensor describes its full reserved shape. The caller must
    /// keep every operation within the prefix reported by `vmm_mapped_bytes`,
    /// growing the mapping before any wider access.
    pub unsafe fn alloc_vmm_tensor(
        &mut self,
        shape: &[usize],
        dtype: DType,
        initial_mapped_bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let numel = shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| HipError::new(0, "VMM tensor element count overflowed"))?;
        let byte_size = numel
            .checked_mul(dtype.size())
            .ok_or_else(|| HipError::new(0, "VMM tensor byte size overflowed"))?;
        let mut arena = VmmArena::reserve(&self.hip, self.device_id, byte_size)?;
        if hipfire_config::developer_var("HIPFIRE_OFFLOAD_DEBUG").is_ok() {
            eprintln!(
                "[offload-debug] device-vmm base=0x{:x} req={} reserved={} gran={}",
                arena.base_address(),
                byte_size,
                arena.reserved_bytes(),
                arena.granularity(),
            );
        }
        // WINDOWS FIX (2026-09-09): hipMemCreate/hipMemMap on Windows/ROCm 7.2
        // (gfx1100) maps a second, later segment onto the SAME physical pages as
        // the first (vmm_arena_smoke boundary-growth assert fails; every
        // subsequent KV growth corrupts all prior KV -> token soup). Single
        // segment maps are proven correct. So on Windows, map the FULL
        // reservation in one map_next up front instead of growing in small
        // segments; grow_vmm_tensor then becomes a no-op (already fully
        // mapped). Costs up-front VRAM for the whole reservation; correctness
        // over on-demand commit on the platform whose driver breaks growth.
        #[cfg(windows)]
        let initial_mapped_bytes = arena.reserved_bytes();
        // `hipMemMap` requires a granularity-aligned size. Current callers all pass
        // aligned lengths (the KV arenas), so this never tripped — but an arbitrary
        // tensor length fails here, which is exactly how the host path broke on a
        // real quantized code blob. Round up (the reservation is already rounded by
        // `reserve`) so the latent device-side failure cannot surprise a later caller.
        let initial_mapped_bytes = initial_mapped_bytes
            .next_multiple_of(arena.granularity())
            .min(arena.reserved_bytes());
        if initial_mapped_bytes > 0 {
            if let Err(err) = arena.map_next(&self.hip, initial_mapped_bytes, access_devices) {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        }
        self.register_vmm_arena(arena, byte_size, shape, dtype)
    }

    /// Register a mapped arena as the owner of a new VMM tensor, stamping the
    /// next owner generation. Any failure retires the arena through the
    /// orphan table, exactly like the allocation path always did.
    fn register_vmm_arena(
        &mut self,
        arena: VmmArena,
        byte_size: usize,
        shape: &[usize],
        dtype: DType,
    ) -> HipResult<GpuTensor> {
        // SAFETY: exactly one owner descriptor is created for this arena and it
        // is registered below (or the arena is released on failure).
        let buf = match unsafe { arena.owner_buffer(byte_size) } {
            Ok(buf) => buf,
            Err(err) => {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        };
        let key = buf.as_ptr() as usize;
        if self.vmm_arenas.contains_key(&key) {
            let duplicate =
                HipError::new(0, &format!("duplicate VMM tensor base address 0x{key:x}"));
            return Err(self.retain_failed_vmm_arena(arena, duplicate));
        }
        self.vmm_generation_counter += 1;
        self.vmm_generations.insert(key, self.vmm_generation_counter);
        self.vmm_arenas.insert(key, arena);
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Allocate a **host-mapped** tensor: system RAM the GPU reads directly over PCIe.
    ///
    /// This is the primitive partial GPU offload uses for spilled weights, and the
    /// mechanism is load-bearing rather than an implementation detail. The obvious
    /// alternative — a host-located VMM arena (`hipMemCreate` PINNED/Host +
    /// `hipMemMap` into device VA) — is charged against the device heap 1:1 on
    /// gfx1201: measured, a held 4 GiB host arena drops the allocatable device
    /// headroom from 15360 MB to 11264 MB, which made offload net-zero for the VRAM
    /// it exists to free. `hipHostMalloc(hipHostMallocMapped)` moves the same bytes
    /// for a measured device cost of 0 MB, and `hipHostGetDevicePointer` hands back
    /// an address the kernels dereference unchanged.
    /// `examples/host_offload_headroom.rs` asserts this property.
    ///
    /// `byte_size` is the *logical* tensor size; the allocation is padded past it
    /// because forward-path kernels overread the tensor tail (see
    /// [`HOST_TAIL_PAD_BYTES`]). The pad costs host RAM only.
    fn alloc_host_mapped_tensor(
        &mut self,
        byte_size: usize,
        shape: &[usize],
        dtype: DType,
    ) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let alloc_bytes = byte_size.saturating_add(HOST_TAIL_PAD_BYTES);
        let host_ptr = self
            .hip
            .host_malloc(alloc_bytes, hip_bridge::HIP_HOST_MALLOC_MAPPED)
            .map_err(|mut error| {
                // hip-bridge backs host allocations with GTT unless the operator
                // set HSA_USERPTR_FOR_PAGED_MEM; TTM caps GTT, not free RAM.
                error.message.push_str(&format!(
                    "; {alloc_bytes} host-mapped bytes with HSA_USERPTR_FOR_PAGED_MEM={} \
                     (0 = GTT, capped by /sys/module/ttm/parameters/pages_limit)",
                    std::env::var("HSA_USERPTR_FOR_PAGED_MEM").unwrap_or_default()
                ));
                error
            })?;
        // The device alias is what kernels are handed; the host pointer is what
        // hipHostFree needs. They coincide on large-BAR boxes but need not.
        let dev_ptr = match self.hip.host_get_device_pointer(host_ptr, 0) {
            Ok(p) if !p.is_null() => p,
            _ => host_ptr,
        };
        let key = dev_ptr as usize;
        if self.host_mapped.insert(key, host_ptr as usize).is_some() {
            self.host_mapped.remove(&key);
            let _ = self.hip.host_free(host_ptr);
            return Err(HipError::new(
                0,
                &format!("duplicate host-mapped tensor base address 0x{key:x}"),
            ));
        }
        Ok(GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_host_mapped(dev_ptr, byte_size) },
            shape: shape.to_vec(),
            dtype,
        })
    }

    /// Live `hipHostMalloc` owners. Zero after a clean unload; nonzero means a leak.
    pub fn host_mapped_count(&self) -> usize {
        // bind_thread: skip — pure length read, touches no device state.
        self.host_mapped.len()
    }

    /// Release every registered host-mapped allocation, mirroring
    /// [`Self::release_registered_vmm`]. Returns the number still registered.
    fn release_registered_host_mapped(&mut self) -> HipResult<usize> {
        let mut first_error = None;
        // Remove only on success, mirroring `release_registered_vmm`: draining
        // unconditionally would drop the sole record of a pointer whose
        // `hipHostFree` failed, turning a retryable failure into an invisible
        // leak that `host_mapped_count()` could no longer report.
        let keys: Vec<usize> = self.host_mapped.keys().copied().collect();
        for key in keys {
            let host_ptr = self.host_mapped[&key];
            match self.hip.host_free(host_ptr as *mut std::ffi::c_void) {
                Ok(()) => {
                    self.host_mapped.remove(&key);
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(self.host_mapped.len()),
        }
    }

    fn retain_failed_vmm_arena(
        &mut self,
        mut arena: VmmArena,
        operation_error: HipError,
    ) -> HipError {
        let cleanup_error = arena.release(&self.hip).err();
        if !arena.is_released() {
            self.orphan_vmm_arenas.push(arena);
        }
        match cleanup_error {
            Some(cleanup) => HipError::new(
                0,
                &format!(
                    "{operation_error}; cleanup also failed: {cleanup}; arena retained for retry"
                ),
            ),
            None => operation_error,
        }
    }

    /// Grow the mapped prefix of a VMM-backed tensor by page-aligned bytes.
    pub fn grow_vmm_tensor(
        &mut self,
        tensor: &mut GpuTensor,
        additional_bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<usize> {
        self.grow_vmm_tensor_impl(tensor, additional_bytes, access_devices, false)
    }

    /// [`Self::grow_vmm_tensor`] that maps the new bytes as physical granules
    /// (`VmmArena::map_next_granules`) so they can later be shared by a prefix
    /// capture. Returns the new mapped bytes.
    pub fn grow_vmm_tensor_granules(
        &mut self,
        tensor: &mut GpuTensor,
        additional_bytes: usize,
        access_devices: &[i32],
    ) -> HipResult<usize> {
        self.grow_vmm_tensor_impl(tensor, additional_bytes, access_devices, true)
    }

    fn grow_vmm_tensor_impl(
        &mut self,
        tensor: &mut GpuTensor,
        additional_bytes: usize,
        access_devices: &[i32],
        granules: bool,
    ) -> HipResult<usize> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        let logical_bytes = tensor.byte_size();
        let arena = self.vmm_arenas.get_mut(&key).ok_or_else(|| {
            HipError::new(
                0,
                &format!("tensor at 0x{key:x} is not a registered VMM owner"),
            )
        })?;
        if granules {
            arena.map_next_granules(&self.hip, additional_bytes, access_devices)?;
        } else {
            arena.map_next(&self.hip, additional_bytes, access_devices)?;
        }
        let mapped_bytes = arena.mapped_bytes();
        tensor.buf = unsafe { arena.owner_buffer(logical_bytes)? };
        Ok(mapped_bytes)
    }

    pub fn vmm_mapped_bytes(&self, tensor: &GpuTensor) -> Option<usize> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::mapped_bytes)
    }

    /// Return the physical allocation granularity for a registered VMM tensor.
    pub fn vmm_granularity(&self, tensor: &GpuTensor) -> Option<usize> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::granularity)
    }

    /// Monotonic owner generation recorded when `tensor` was registered
    /// (`alloc_vmm_tensor` or a prefix fork); `None` when it is not a registered owner.
    pub fn vmm_owner_generation(&self, tensor: &GpuTensor) -> Option<u64> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_generations
            .get(&(tensor.buf.as_ptr() as usize))
            .copied()
    }

    /// Reserved virtual bytes of a registered VMM owner.
    pub fn vmm_reserved_bytes(&self, tensor: &GpuTensor) -> Option<usize> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::reserved_bytes)
    }

    /// End offset of the last granule of a registered VMM owner that is shared
    /// (lease count above one). Bytes below it must never be written.
    pub fn vmm_sealed_bytes(&self, tensor: &GpuTensor) -> Option<usize> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::sealed_bytes)
    }

    /// Mapped physical granules `(id, bytes)` of a registered VMM owner in offset order.
    pub fn vmm_physical_granules(&self, tensor: &GpuTensor) -> Option<Vec<(VmmPhysicalId, usize)>> {
        // bind_thread: skip — pure map lookup, touches no device state.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .map(VmmArena::physical_granules)
    }

    /// The relocation record that re-points `old` to `new`, or `None` unless both
    /// are registered VMM owners. Extents and generation describe the NEW owner.
    pub fn vmm_resource_move(&self, old: &GpuTensor, new: &GpuTensor) -> Option<VmmResourceMove> {
        // bind_thread: skip — pure map lookup, touches no device state.
        let old_base = old.buf.as_ptr() as usize;
        let new_base = new.buf.as_ptr() as usize;
        self.vmm_arenas.get(&old_base)?;
        let new_arena = self.vmm_arenas.get(&new_base)?;
        let owner_generation = *self.vmm_generations.get(&new_base)?;
        Some(VmmResourceMove {
            old_base,
            new_base,
            reserved_bytes: new_arena.reserved_bytes(),
            mapped_bytes: new_arena.mapped_bytes(),
            owner_generation,
        })
    }

    fn refuse_vmm_prefix_op_while_armed(&self, op: &str) -> HipResult<()> {
        if self.graphs.capture_mode
            || self.graphs.verify.capturing.is_some()
            || self.graphs.replay.capturing.is_some()
            || self.replay.is_recording()
        {
            return Err(HipError::new(
                0,
                &format!("VMM prefix {op} refused while a graph capture or replay record is armed"),
            ));
        }
        Ok(())
    }

    /// Release every lease of `entries` and free the slab; returns the first error
    /// after attempting all of them.
    fn discard_vmm_prefix_parts(
        &mut self,
        entries: Vec<VmmPrefixEntry>,
        slab: Option<GpuTensor>,
    ) -> Option<HipError> {
        let mut first_error = None;
        for entry in entries {
            if let Err(err) = entry.prefix.release(&self.hip) {
                first_error.get_or_insert(err);
            }
        }
        if let Some(slab) = slab {
            if let Err(err) = self.free_tensor(slab) {
                first_error.get_or_insert(err);
            }
        }
        first_error
    }

    /// Checkpoint the leading granules and frontier of several registered VMM
    /// owners. Whole leading granules below `min(valid, writable_from)` are shared
    /// by lease (no copy); `[alias, valid)` of every arena is copied byte-exact into
    /// ONE slab on the active/default stream. On failure every lease taken and the
    /// slab are released. Refused while a graph capture or replay record is armed.
    pub fn capture_vmm_prefixes(&mut self, specs: &[VmmPrefixSpec<'_>]) -> HipResult<VmmPrefixSet> {
        self.bind_thread()?;
        self.refuse_vmm_prefix_op_while_armed("capture")?;
        let mut plans: Vec<VmmPrefixPlan> = Vec::with_capacity(specs.len());
        for (index, spec) in specs.iter().enumerate() {
            let key = spec.tensor.buf.as_ptr() as usize;
            let arena = match self.vmm_arenas.get(&key) {
                Some(arena) if spec.tensor.buf.is_vmm_owner() => arena,
                _ => {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "VMM prefix spec {index}: tensor at 0x{key:x} is not a registered VMM owner"
                        ),
                    ));
                }
            };
            if plans.iter().any(|plan| plan.key == key) {
                return Err(HipError::new(
                    0,
                    &format!("VMM prefix spec {index}: duplicate tensor at 0x{key:x}"),
                ));
            }
            let mapped = arena.mapped_bytes();
            if spec.valid_bytes > mapped {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM prefix spec {index}: valid {} exceeds mapped {mapped}",
                        spec.valid_bytes
                    ),
                ));
            }
            if spec.writable_from > spec.valid_bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM prefix spec {index}: writable_from {} exceeds valid {}",
                        spec.writable_from, spec.valid_bytes
                    ),
                ));
            }
            let limit = spec
                .valid_bytes
                .min(spec.writable_from)
                .min(arena.granule_prefix_bytes());
            let alias = vmm_alias_boundary(
                arena
                    .physical_granules()
                    .into_iter()
                    .map(|(_, bytes)| bytes),
                limit,
            );
            plans.push(VmmPrefixPlan { key, mapped, alias });
        }
        let frontier_lens: Vec<usize> = specs
            .iter()
            .zip(&plans)
            .map(|(spec, plan)| spec.valid_bytes - plan.alias)
            .collect();
        let (offsets, slab_bytes) = vmm_slab_layout(&frontier_lens)
            .ok_or_else(|| HipError::new(0, "VMM prefix slab layout overflowed"))?;
        let slab = if slab_bytes > 0 {
            Some(self.alloc_tensor(&[slab_bytes], DType::Raw)?)
        } else {
            None
        };

        let mut entries: Vec<VmmPrefixEntry> = Vec::with_capacity(specs.len());
        let mut failure = None;
        for (index, (spec, plan)) in specs.iter().zip(&plans).enumerate() {
            let shared = match self.vmm_arenas.get(&plan.key) {
                Some(arena) => arena
                    .share_prefix(plan.alias)
                    .map(|prefix| (prefix, arena.reserved_bytes())),
                None => Err(HipError::new(
                    0,
                    &format!("VMM prefix spec {index}: owner vanished during capture"),
                )),
            };
            match shared {
                Ok((prefix, reserved_bytes)) => entries.push(VmmPrefixEntry {
                    shape: spec.tensor.shape.clone(),
                    dtype: spec.tensor.dtype,
                    logical_bytes: spec.tensor.byte_size(),
                    reserved_bytes,
                    valid_bytes: spec.valid_bytes,
                    writable_from: spec.writable_from,
                    alias_bytes: plan.alias,
                    slab_offset: offsets[index],
                    prefix,
                }),
                Err(err) => {
                    failure = Some(err);
                    break;
                }
            }
        }
        if let Some(err) = failure {
            return Err(self.abort_vmm_prefix_capture(entries, slab, err));
        }

        if let Some(slab_tensor) = slab.as_ref() {
            // SAFETY: each view spans the source's mapped prefix, which stays
            // mapped for the whole call; views are never freed.
            let sources: Vec<DeviceBuffer> = specs
                .iter()
                .zip(&plans)
                .map(|(spec, plan)| unsafe {
                    DeviceBuffer::from_raw(spec.tensor.buf.as_ptr(), plan.mapped)
                })
                .collect();
            let regions: Vec<crate::tensor_ops::CopyRegion<'_>> = entries
                .iter()
                .zip(&sources)
                .filter(|(entry, _)| entry.valid_bytes > entry.alias_bytes)
                .map(|(entry, source)| crate::tensor_ops::CopyRegion {
                    dst: &slab_tensor.buf,
                    dst_offset: entry.slab_offset,
                    src: source,
                    src_offset: entry.alias_bytes,
                    bytes: entry.valid_bytes - entry.alias_bytes,
                })
                .collect();
            if let Err(err) = crate::tensor_ops::copy_regions(self, &regions) {
                drop(regions);
                return Err(self.abort_vmm_prefix_capture(entries, slab, err));
            }
        }
        Ok(VmmPrefixSet {
            entries,
            slab,
            slab_bytes,
        })
    }

    fn abort_vmm_prefix_capture(
        &mut self,
        entries: Vec<VmmPrefixEntry>,
        slab: Option<GpuTensor>,
        err: HipError,
    ) -> HipError {
        match self.discard_vmm_prefix_parts(entries, slab) {
            Some(cleanup) => HipError::new(
                err.code,
                &format!("{err}; prefix cleanup also failed: {cleanup}"),
            ),
            None => err,
        }
    }

    /// Build one fresh registered owner per entry of `prefix`: shared aliased
    /// granules, private granules covering `[alias, round_up(valid, granule))`,
    /// the slab frontier copied into `[alias, valid)` and `[valid, mapped)` of
    /// the private part zeroed. Owners come back in entry order. On any failure
    /// every owner created so far is freed; `prefix` is untouched and reusable.
    /// Refused while a graph capture or replay record is armed.
    pub fn fork_vmm_prefixes(
        &mut self,
        prefix: &VmmPrefixSet,
        access_devices: &[i32],
    ) -> HipResult<Vec<GpuTensor>> {
        self.bind_thread()?;
        self.refuse_vmm_prefix_op_while_armed("fork")?;
        let mut created: Vec<GpuTensor> = Vec::with_capacity(prefix.entries.len());
        for entry in &prefix.entries {
            match self.fork_vmm_entry(entry, access_devices) {
                Ok(tensor) => created.push(tensor),
                Err(err) => return Err(self.free_forked_vmm_tensors(created, err)),
            }
        }
        if let Err(err) = self.fill_forked_vmm_tensors(prefix, &created) {
            // Queued fill kernels may still target the owners about to unmap.
            let _ = self.hip.device_synchronize();
            return Err(self.free_forked_vmm_tensors(created, err));
        }
        Ok(created)
    }

    /// Reserve and map one fork owner (no data movement).
    fn fork_vmm_entry(
        &mut self,
        entry: &VmmPrefixEntry,
        access_devices: &[i32],
    ) -> HipResult<GpuTensor> {
        let mut arena = VmmArena::reserve(&self.hip, self.device_id, entry.logical_bytes)?;
        if arena.reserved_bytes() != entry.reserved_bytes {
            let reserved = arena.reserved_bytes();
            let err = HipError::new(
                0,
                &format!(
                    "VMM fork reserve {reserved} differs from captured reserve {}",
                    entry.reserved_bytes
                ),
            );
            return Err(self.retain_failed_vmm_arena(arena, err));
        }
        if entry.alias_bytes > 0 {
            if let Err(err) = arena.map_shared_prefix(&self.hip, &entry.prefix, access_devices) {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        }
        if entry.valid_bytes > entry.alias_bytes {
            let end = entry
                .valid_bytes
                .next_multiple_of(arena.granule_bytes())
                .min(arena.reserved_bytes());
            if let Err(err) =
                arena.map_next_granules(&self.hip, end - entry.alias_bytes, access_devices)
            {
                return Err(self.retain_failed_vmm_arena(arena, err));
            }
        }
        self.register_vmm_arena(arena, entry.logical_bytes, &entry.shape, entry.dtype)
    }

    /// Copy the slab frontiers into the fork owners and zero their private tails,
    /// ordered on the active/default stream.
    fn fill_forked_vmm_tensors(
        &mut self,
        prefix: &VmmPrefixSet,
        forks: &[GpuTensor],
    ) -> HipResult<()> {
        // SAFETY: views span each fork's mapped prefix, which stays mapped for the
        // call; they are never freed.
        let mut targets: Vec<DeviceBuffer> = Vec::with_capacity(forks.len());
        for fork in forks {
            let mapped = self.vmm_mapped_bytes(fork).ok_or_else(|| {
                HipError::new(0, "forked tensor is not a registered VMM owner")
            })?;
            targets.push(unsafe { DeviceBuffer::from_raw(fork.buf.as_ptr(), mapped) });
        }
        if let Some(slab) = prefix.slab.as_ref() {
            let regions: Vec<crate::tensor_ops::CopyRegion<'_>> = prefix
                .entries
                .iter()
                .zip(&targets)
                .filter(|(entry, _)| entry.valid_bytes > entry.alias_bytes)
                .map(|(entry, target)| crate::tensor_ops::CopyRegion {
                    dst: target,
                    dst_offset: entry.alias_bytes,
                    src: &slab.buf,
                    src_offset: entry.slab_offset,
                    bytes: entry.valid_bytes - entry.alias_bytes,
                })
                .collect();
            crate::tensor_ops::copy_regions(self, &regions)?;
        }
        for (entry, target) in prefix.entries.iter().zip(&targets) {
            let tail = target.size().saturating_sub(entry.valid_bytes);
            if tail == 0 {
                continue;
            }
            let view = target.byte_view(entry.valid_bytes, tail);
            match self.active_stream.as_ref() {
                Some(stream) => self.hip.memset_async(&view, 0, tail, stream)?,
                None => self.hip.memset(&view, 0, tail)?,
            }
        }
        Ok(())
    }

    fn free_forked_vmm_tensors(&mut self, created: Vec<GpuTensor>, err: HipError) -> HipError {
        let mut cleanup_error = None;
        for tensor in created {
            if let Err(cleanup) = self.free_tensor(tensor) {
                cleanup_error.get_or_insert(cleanup);
            }
        }
        match cleanup_error {
            Some(cleanup) => HipError::new(
                err.code,
                &format!("{err}; forked owner cleanup also failed: {cleanup}"),
            ),
            None => err,
        }
    }

    /// Release every lease of `prefix` and free its slab. Attempts everything and
    /// returns the first error.
    pub fn free_vmm_prefixes(&mut self, prefix: VmmPrefixSet) -> HipResult<()> {
        // Release must run even when the thread bind fails, or the leases leak.
        self.bind_thread_or_warn();
        let VmmPrefixSet { entries, slab, .. } = prefix;
        match self.discard_vmm_prefix_parts(entries, slab) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Re-point recorded PM4 replay state (and refuse stale hipGraphs) from old
    /// VMM owners to their replacements. Every move is validated first: both
    /// bases are live registered owners, `owner_generation` equals the NEW
    /// owner's, `mapped_bytes`/`reserved_bytes` match the NEW owner, the old
    /// owner's reserve covers the range, and no old (or new) range is duplicated
    /// or overlapping. Call while both owners are alive. The hipGraph check is
    /// side-effect free and runs first so a refusal never follows a committed
    /// replay rewrite.
    pub fn relocate_qsa_resources(
        &mut self,
        moves: &[VmmResourceMove],
    ) -> HipResult<crate::replay::BindingRefreshReport> {
        self.bind_thread()?;
        validate_vmm_move_ranges(moves).map_err(|message| HipError::new(0, &message))?;
        for (index, mv) in moves.iter().enumerate() {
            let old = self
                .vmm_arenas
                .get(&mv.old_base)
                .filter(|arena| !arena.is_released())
                .ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!(
                            "VMM move {index}: old base 0x{:x} is not a live registered owner",
                            mv.old_base
                        ),
                    )
                })?;
            if old.reserved_bytes() < mv.reserved_bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM move {index}: old reserve {} is smaller than move reserve {}",
                        old.reserved_bytes(),
                        mv.reserved_bytes
                    ),
                ));
            }
            let new = self
                .vmm_arenas
                .get(&mv.new_base)
                .filter(|arena| !arena.is_released())
                .ok_or_else(|| {
                    HipError::new(
                        0,
                        &format!(
                            "VMM move {index}: new base 0x{:x} is not a live registered owner",
                            mv.new_base
                        ),
                    )
                })?;
            let generation = self.vmm_generations.get(&mv.new_base).copied();
            if generation != Some(mv.owner_generation) {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM move {index}: owner generation {} does not match new owner {generation:?}",
                        mv.owner_generation
                    ),
                ));
            }
            if new.mapped_bytes() != mv.mapped_bytes || new.reserved_bytes() != mv.reserved_bytes {
                return Err(HipError::new(
                    0,
                    &format!(
                        "VMM move {index}: mapped/reserved {}/{} do not match new owner {}/{}",
                        mv.mapped_bytes,
                        mv.reserved_bytes,
                        new.mapped_bytes(),
                        new.reserved_bytes()
                    ),
                ));
            }
        }
        self.graphs
            .relocate_resources(&self.hip, self.device_id, moves)?;
        self.replay
            .relocate_resources(&self.hip, moves)
            .map_err(|message| HipError::new(0, &message))
    }

    /// Whether a tensor was allocated by [`Self::alloc_host_mapped_tensor`] — its pages
    /// are system RAM the kernels read over PCIe rather than the card's VRAM. Pure
    /// ownership check; touches no device state. Offload tests and logs read this to
    /// prove a spilled layer actually left VRAM rather than being merely
    /// device-pinned. Device-VMM tensors are not host-located by construction.
    pub fn host_located(&self, tensor: &GpuTensor) -> bool {
        // bind_thread: skip — pure read of the tensor's own ownership tag; no device call.
        tensor.buf.is_host_mapped()
    }

    /// The primary physical allocation handle backing a tensor's VMM arena, if any.
    /// Exposed so callers can query the handle's placement with
    /// `HipRuntime::mem_get_handle_properties` (fail-closed host-located check).
    pub fn vmm_handle(&self, tensor: &GpuTensor) -> Option<HipMemGenericAllocationHandle> {
        // bind_thread: skip — pure read of the arena registry; no device call.
        self.vmm_arenas
            .get(&(tensor.buf.as_ptr() as usize))
            .and_then(|arena| arena.primary_handle())
    }

    /// Return the driver's recommended physical mapping granularity without
    /// reserving an address range. Model-owned VMM planners use this for a
    /// dry-run admission check before mapping any cache pages.
    pub fn vmm_recommended_granularity(&self) -> HipResult<usize> {
        self.bind_thread()?;
        let prop = HipMemAllocationProp::device_pinned(self.device_id);
        self.hip
            .mem_get_allocation_granularity(&prop, HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED)
    }

    pub fn vmm_allocation_count(&self) -> usize {
        // bind_thread: skip — pure length read, touches no device state.
        self.vmm_arenas.len() + self.orphan_vmm_arenas.len()
    }

    /// Retry teardown for VMM arenas retained after an earlier cleanup error.
    /// Returns the number of arenas still pending cleanup.
    pub fn retry_vmm_cleanup(&mut self) -> HipResult<usize> {
        self.bind_thread()?;
        self.release_pending_vmm()
    }

    /// Retry pending VMM arenas and require a fully idle owner table.
    ///
    /// Success means `vmm_allocation_count() == 0`. Any remaining registration
    /// is an error so unload/load cannot claim a clean handoff.
    pub fn ensure_vmm_cleaned(&mut self) -> HipResult<()> {
        self.bind_thread()?;
        // Offloaded weights are a second owner class; without this check a leaked
        // host-mapped tensor would let unload claim a clean handoff while holding
        // pinned system RAM, which is exactly what this guard exists to prevent
        // for VMM owners.
        let live_host = self.host_mapped.len();
        if live_host != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "refusing cleanup while {live_host} live host-mapped tensor owner(s) remain; unload the active model first"
                ),
            ));
        }
        // The VMM gathered QSA workspace is this Gpu's own scratch, not a
        // model owner. Only once no host-mapped (above) and no other VMM
        // owner remains, retire it so a load whose construction failed after
        // reserving it cannot wedge this gate; a failed release lands in the
        // pending table retried below.
        let workspace = usize::from(self.scratch.qsa_gather_vmm.is_some());
        let live = self.vmm_arenas.len().saturating_sub(workspace);
        if live != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "refusing VMM cleanup while {live} live VMM tensor owner(s) remain; unload the active model first"
                ),
            ));
        }
        if workspace != 0 {
            let _ = self.release_qsa_gather_workspace();
        }
        match self.retry_vmm_cleanup() {
            Ok(0) => Ok(()),
            Ok(n) => Err(HipError::new(
                0,
                &format!("VMM teardown incomplete: {n} arena(s) still pending"),
            )),
            Err(err) => {
                let n = self.orphan_vmm_arenas.len();
                if n == 0 {
                    Err(err)
                } else {
                    Err(HipError::new(
                        err.code,
                        &format!("{err}; {n} arena(s) still pending"),
                    ))
                }
            }
        }
    }

    fn release_pending_vmm(&mut self) -> HipResult<usize> {
        let mut first_error = None;
        let mut orphan_index = 0;
        while orphan_index < self.orphan_vmm_arenas.len() {
            let (result, released) = {
                let arena = &mut self.orphan_vmm_arenas[orphan_index];
                let result = arena.release(&self.hip);
                (result, arena.is_released())
            };
            if released {
                let released_arena = self.orphan_vmm_arenas.swap_remove(orphan_index);
                debug_assert!(released_arena.is_released());
            } else {
                orphan_index += 1;
            }
            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(self.orphan_vmm_arenas.len()),
        }
    }

    fn release_registered_vmm(&mut self) -> HipResult<usize> {
        let keys: Vec<usize> = self.vmm_arenas.keys().copied().collect();
        let mut first_error = None;
        for key in keys {
            let (result, released) = {
                let arena = self.vmm_arenas.get_mut(&key).unwrap();
                let result = arena.release(&self.hip);
                (result, arena.is_released())
            };
            if released {
                self.vmm_arenas.remove(&key);
                self.vmm_generations.remove(&key);
            }
            if let Err(err) = result {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        if let Err(err) = self.release_pending_vmm() {
            if first_error.is_none() {
                first_error = Some(err);
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(self.vmm_allocation_count()),
        }
    }

    /// Upload raw bytes into a **host-mapped** tensor: pinned system RAM the
    /// kernels read directly over PCIe (zero-copy), charged to no device heap.
    ///
    /// Same allocation as [`Self::upload_raw_host`]
    /// ([`Self::alloc_host_mapped_tensor`], padded by [`HOST_TAIL_PAD_BYTES`]),
    /// but the bytes are copied on the CPU through the host pointer and the pad
    /// is zeroed. Qwen4 host-mapped routed experts load through this path.
    pub fn upload_raw_host_mapped(&mut self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        let tensor = self.alloc_host_mapped_tensor(data.len(), shape, DType::Raw)?;
        let host_ptr = self.host_mapped[&(tensor.buf.as_ptr() as usize)] as *mut u8;
        // SAFETY: `alloc_host_mapped_tensor` registered `host_ptr` as a live
        // allocation of `data.len() + HOST_TAIL_PAD_BYTES` bytes that nothing
        // else references yet.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr, data.len());
            std::ptr::write_bytes(host_ptr.add(data.len()), 0, HOST_TAIL_PAD_BYTES);
        }
        Ok(tensor)
    }

    /// Allocate a pool tensor then run `init`. On init failure the owner is
    /// returned to the pool and the original error is preserved — constructors
    /// that allocate then memset/htod must not strand the buffer when init fails
    /// (`GpuTensor` has no freeing `Drop`).
    fn alloc_then_init(
        &mut self,
        shape: &[usize],
        dtype: DType,
        init: impl FnOnce(&HipRuntime, Option<&hip_bridge::Stream>, &GpuTensor) -> HipResult<()>,
    ) -> HipResult<GpuTensor> {
        let tensor = self.alloc_tensor(shape, dtype)?;
        let init_result = {
            let stream = self.active_stream.as_ref();
            init(&self.hip, stream, &tensor)
        };
        if let Err(err) = init_result {
            let _ = self.free_tensor(tensor);
            return Err(err);
        }
        Ok(tensor)
    }

    pub fn upload_f32(&mut self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
        HTOD_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bind_thread()?;
        self.alloc_then_init(shape, DType::F32, |hip, _stream, tensor| {
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            hip.memcpy_htod(&tensor.buf, bytes)
        })
    }
    /// Upload host-side **f32 data** into a **host-located** `F32` tensor — the
    /// offload counterpart to [`Self::upload_f32`]. The physical pages are system
    /// RAM that kernels read over PCIe, so this is how an offloaded layer's weights
    /// land without pinning device memory.
    ///
    /// Mirrors [`Self::upload_f32`] except the allocation target is host: the whole
    /// tensor is mapped for `self.device_id` at reserve time (the same proven pattern
    /// `examples/vmm_tensor_smoke.rs` exercises — a PCIe H2D/D2H round-trip through the
    /// mapped VA) and the f32 words are copied in via `memcpy_htod`, which ROCm routes
    /// to the mapped VA regardless of direction. The result is a `GpuTensor` whose
    /// `.buf` is host RAM readable by kernels — exactly what an offloaded layer's weight
    /// read must dereference. Byte-for-byte identical contents to the device path; only
    /// the physical location differs.
    pub fn upload_f32_host(&mut self, data: &[f32], shape: &[usize]) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let numel = shape
            .iter()
            .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
            .ok_or_else(|| HipError::new(0, "upload_f32_host: element count overflowed"))?;
        let byte_size = numel
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| HipError::new(0, "upload_f32_host: byte size overflowed"))?;
        // Host-mapped: system RAM the kernels read over PCIe, at no device-heap
        // cost (see `alloc_host_mapped_tensor`).
        let mut tensor = self.alloc_host_mapped_tensor(byte_size, shape, DType::F32)?;
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        unsafe { self.hip.memcpy_htod(&tensor.buf, bytes)? }
        Ok(tensor)
    }

    /// Upload host-side **f16 bit patterns** straight into an `F16` tensor.
    ///
    /// The counterpart to [`Self::upload_f32`] for callers that already hold
    /// half-precision words — notably the diffusion weight streamer, which
    /// converts a checkpoint's BF16 bytes to f16 one tensor at a time and must
    /// never materialise a whole-model f32 host table. Uploading f32 and
    /// casting on the device costs 2× the PCIe/fabric traffic plus a transient
    /// f32 device allocation the size of the tensor; this path costs neither.
    ///
    /// `data` is little-endian f16 words, exactly `shape.iter().product()` of
    /// them.
    pub fn upload_f16_bits(&mut self, data: &[u16], shape: &[usize]) -> HipResult<GpuTensor> {
        HTOD_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.bind_thread()?;
        let tensor = self.alloc_tensor(shape, DType::F16)?;
        let want = tensor.numel();
        if data.len() != want {
            return Err(hip_bridge::HipError::new(
                0,
                &format!(
                    "upload_f16_bits: {} words for a {want}-element {shape:?} tensor",
                    data.len()
                ),
            ));
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(tensor)
    }

    /// Allocate an F32 tensor filled with a constant `value` (host-side fill +
    /// sync htod). Used for `-inf`-initialised buffers where a byte-memset
    /// can't express the bit pattern (e.g. the compressor `score_state`, which
    /// the reference inits to `float("-inf")` so unfilled pool slots get zero
    /// softmax weight).
    pub fn full_f32(&mut self, shape: &[usize], value: f32) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        self.alloc_then_init(shape, DType::F32, |hip, _stream, tensor| {
            let data = vec![value; tensor.numel()];
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            hip.memcpy_htod(&tensor.buf, bytes)
        })
    }

    /// In-place constant fill of an existing F32 tensor (sync htod).
    pub fn fill_f32(&mut self, tensor: &GpuTensor, value: f32) -> HipResult<()> {
        self.bind_thread()?;
        let data = vec![value; tensor.numel()];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        self.hip.memcpy_htod(&tensor.buf, bytes)?;
        Ok(())
    }

    /// Read an `F16` tensor back as raw half words — no widening, so a
    /// caller can compare device bytes bit-for-bit (the streaming weight
    /// upload's parity harness does exactly that against the f32-upload +
    /// device-cast path it replaced).
    pub fn download_f16_bits(&self, tensor: &GpuTensor) -> HipResult<Vec<u16>> {
        self.bind_thread()?;
        if tensor.dtype != DType::F16 {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("download_f16_bits: tensor is {:?}, not F16", tensor.dtype),
            ));
        }
        let numel = tensor.numel();
        let mut data = vec![0u16; numel];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 2) };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    pub fn download_f32(&self, tensor: &GpuTensor) -> HipResult<Vec<f32>> {
        self.bind_thread()?;
        let numel = tensor.numel();
        let mut data = vec![0.0f32; numel];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, numel * 4) };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)?;
        Ok(data)
    }

    /// [`Self::download_f32`] into a caller-owned buffer (resized to the
    /// tensor), so a per-row reader reuses one host allocation.
    pub fn download_f32_into(&self, tensor: &GpuTensor, out: &mut Vec<f32>) -> HipResult<()> {
        self.bind_thread()?;
        let numel = tensor.numel();
        out.resize(numel, 0.0);
        // SAFETY: `out` holds exactly `numel` f32s; any byte pattern is a valid f32.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, numel * 4) };
        self.hip.memcpy_dtoh(bytes, &tensor.buf)
    }

    /// Read a `DType::Raw` tensor back as raw bytes — no widening and no dtype
    /// reinterpretation, so a caller can compare code blobs bit-for-bit.
    ///
    /// Works for host-located tensors too: `memcpy_dtoh` reads through the
    /// mapped handle across PCIe, which is what makes the offload parity check
    /// (device upload vs host upload) a plain `Vec<u8>` equality.
    pub fn download_raw_bytes(&self, tensor: &GpuTensor) -> HipResult<Vec<u8>> {
        self.bind_thread()?;
        let n = tensor.byte_size();
        let mut data = vec![0u8; n];
        self.hip.memcpy_dtoh(&mut data, &tensor.buf)?;
        Ok(data)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: DType) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        self.alloc_then_init(shape, dtype, |hip, stream, tensor| match stream {
            Some(stream) => hip.memset_async(&tensor.buf, 0, tensor.byte_size(), stream),
            None => hip.memset(&tensor.buf, 0, tensor.byte_size()),
        })
    }

    /// Upload raw bytes to GPU (for quantized weights).
    ///
    /// Allocation is a direct `hip.malloc` (not the GpuPool). On host→device
    /// copy failure the buffer is released with `hip.free` — never
    /// [`Self::free_tensor`], which would park a never-pooled allocation on
    /// the free list. Successful owners are still typically torn down via
    /// `free_tensor`, which *does* return them into the pool; later
    /// `upload_raw` calls still malloc fresh and never reclaim those slots
    /// (the pool-backed twin lives in hipfire-runtime weight fulfillment).
    pub fn upload_raw(&self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        self.upload_raw_with_copy(data, shape, HipRuntime::memcpy_htod)
    }

    /// Upload raw bytes to a **host-located** tensor (for offloaded quantized weights).
    ///
    /// The offload counterpart to [`Self::upload_raw`]: the physical pages are system
    /// RAM accessed by kernels over PCIe, so an offloaded layer's quantized codes live
    /// in host memory instead of VRAM. Same injectable copy step as [`Self::upload_raw`];
    /// only the allocation target differs — [`Self::alloc_host_mapped_tensor`] rather
    /// than `hip.malloc` — so the bytes leave the card *and* stop being charged to it.
    /// Contents are byte-for-byte identical to the device path, so numerics are
    /// unchanged while VRAM is freed.
    pub fn upload_raw_host(&mut self, data: &[u8], shape: &[usize]) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        self.upload_raw_with_copy_host(data, shape, HipRuntime::memcpy_htod)
    }

    /// [`Self::upload_raw`] with an injectable copy step so regressions can
    /// force malloc-success / copy-failure without a production knob.
    fn upload_raw_with_copy(
        &self,
        data: &[u8],
        shape: &[usize],
        copy: impl FnOnce(&HipRuntime, &DeviceBuffer, &[u8]) -> HipResult<()>,
    ) -> HipResult<GpuTensor> {
        self.bind_thread()?;
        let buf = self.hip.malloc(data.len())?;
        if let Err(err) = copy(&self.hip, &buf, data) {
            // hip.malloc owner — hip.free only. free_tensor would pool it.
            let _ = self.hip.free(buf);
            return Err(err);
        }
        Ok(GpuTensor {
            buf,
            shape: shape.to_vec(),
            dtype: DType::Raw,
        })
    }

    /// [`Self::upload_raw_host`] with an injectable copy step so a regression can force
    /// copy-failure without a production knob, verifying the host owner is released on error.
    fn upload_raw_with_copy_host(
        &mut self,
        data: &[u8],
        shape: &[usize],
        copy: impl FnOnce(&HipRuntime, &DeviceBuffer, &[u8]) -> HipResult<()>,
    ) -> HipResult<GpuTensor> {
        // Host-mapped: system RAM the kernels read over PCIe, at no device-heap
        // cost (see `alloc_host_mapped_tensor`).
        let mut tensor = self.alloc_host_mapped_tensor(data.len(), shape, DType::Raw)?;
        if let Err(err) = copy(&self.hip, &tensor.buf, data) {
            // Host-mapped owner — free_tensor runs hipHostFree, so a failed copy
            // leaves no orphaned host page or dangling registration.
            let _ = self.free_tensor(tensor);
            return Err(err);
        }
        Ok(tensor)
    }

    /// Free a tensor. Contiguous buffers return to the pool. VMM owners run
    /// arena release once: success removes the registration; failure **retains**
    /// the arena for [`Self::retry_vmm_cleanup`] / [`Self::ensure_vmm_cleaned`]
    /// and returns the error (never a false clean free).
    pub fn free_tensor(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        if tensor.buf.is_host_mapped() {
            // Host-mapped owners are never pooled: hipHostFree takes the host
            // pointer, and an offloaded weight is static (freed at unload), so
            // there is nothing to gain from reusing the mapping.
            let host_ptr = self.host_mapped.remove(&key).ok_or_else(|| {
                HipError::new(
                    0,
                    &format!("host-mapped tensor at 0x{key:x} has no registered host pointer"),
                )
            })?;
            return self.hip.host_free(host_ptr as *mut std::ffi::c_void);
        }
        if self.vmm_arenas.contains_key(&key) {
            if !tensor.buf.is_vmm_owner() {
                return Err(HipError::new(
                    0,
                    &format!("refusing to free borrowed VMM view at 0x{key:x}"),
                ));
            }
            let mut arena = self.vmm_arenas.remove(&key).unwrap();
            self.vmm_generations.remove(&key);
            let result = arena.release(&self.hip);
            if arena.is_released() {
                result
            } else {
                // The tensor owner has been consumed, so keep its still-live
                // arena only in the pending table. Cleanup retries must never
                // touch entries in `vmm_arenas`, which are active owners.
                self.orphan_vmm_arenas.push(arena);
                match result {
                    Ok(()) => Err(HipError::new(
                        0,
                        &format!(
                            "VMM tensor at 0x{key:x} reported success but arena is still live"
                        ),
                    )),
                    Err(err) => Err(HipError::new(
                        err.code,
                        &format!("{err}; arena retained for retry"),
                    )),
                }
            }
        } else if tensor.buf.is_borrowed() || tensor.buf.is_vmm_owner() {
            Err(HipError::new(
                0,
                &format!("refusing to pool non-owning tensor at 0x{key:x}"),
            ))
        } else {
            self.pool.free(tensor.buf);
            Ok(())
        }
    }
    /// Calibration capture hook for an instrumented linear: if a collector is
    /// armed and `weight`'s buffer pointer is a known calibration target, invoke
    /// `capture(name, input)`. Zero-cost (`is_none()` + return) when no collector
    /// is armed, so non-calibration forwards are byte-identical. The collector
    /// `Arc` is cloned before the call so `self` is not aliased by `active_capture`.
    #[inline]
    pub fn maybe_capture_activation(
        &mut self,
        weight: &GpuTensor,
        input: &GpuTensor,
        n: usize,
        k: usize,
    ) {
        // bind_thread: skip — early-returns unless a capture is active, and
        // capture paths run on an already-bound forward thread.
        if self.active_capture.is_none() {
            return;
        }
        let ptr = weight.buf.as_ptr() as usize;
        let name = match self.capture_names.get(&ptr) {
            Some(nm) => nm.clone(),
            None => return,
        };
        if let Some(cap) = self.active_capture.clone() {
            cap.capture(self, &name, input, n, k);
        }
    }

    /// Calibration: `acc[c] += Σ_n x[n,c]²` (per-column sum-of-squares, the
    /// imatrix / diag(H) signal). `x` is [N, K] F32; `acc` is [K] F32, ADDED into
    /// (caller zeroes once, then accumulates across the calibration corpus).
    pub fn calib_sumsq_reduce_f32(
        &mut self,
        x: &GpuTensor,
        acc: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_sumsq_reduce_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let acc_ptr = acc.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 256u32;
        let grid = ((k as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &acc_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_sumsq_reduce_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(acc_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// Calibration: `H[i,j] += Σ_n x[n,i]·x[n,j]` (the K×K GPTQ Hessian, tiled
    /// GEMM accumulate). `x` is [N, K] F32; `H` is [K, K] F32 row-major, ADDED
    /// into (caller zeroes once, then accumulates across the calibration corpus).
    pub fn calib_hessian_outer_f32(
        &mut self,
        x: &GpuTensor,
        h: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_hessian_outer_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let h_ptr = h.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let tile = 16u32;
        let grid_x = ((k as u32) + tile - 1) / tile;
        let grid_y = ((k as u32) + tile - 1) / tile;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &h_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_hessian_outer_f32",
            [grid_x, grid_y, 1],
            [tile, tile, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(h_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `w[n] += (1/k)·Σ_c d[n,c]²` — per-row (per-token) mean-square of an
    /// output-grad block `d[n,k]`. Used by GuidedQuant calibration to turn a
    /// linear's output adjoint `∂ℓ/∂z` into a per-token Fisher weight. `w` is
    /// caller-zeroed `[n]`.
    pub fn calib_row_meansq_f32(
        &mut self,
        d: &GpuTensor,
        w: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_row_meansq_f32",
        )?;
        let d_ptr = d.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 64u32;
        let grid = ((n as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &d_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_row_meansq_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(d_ptr);
                blob.push_ptr(w_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `acc[c] += Σ_n w[n]·x[n,c]²` — the per-row-weighted column sum-of-squares,
    /// i.e. the diagonal of the weighted Hessian. `w≡1` reduces to
    /// [`Self::calib_sumsq_reduce_f32`].
    pub fn calib_sumsq_weighted_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        acc: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_sumsq_weighted_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let a_ptr = acc.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let block = 64u32;
        let grid = ((k as u32) + block - 1) / block;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &a_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_sumsq_weighted_f32",
            [grid, 1, 1],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(w_ptr);
                blob.push_ptr(a_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// `H[i,j] += Σ_n w[n]·x[n,i]·x[n,j]` — the per-row-weighted (GuidedQuant /
    /// empirical-Fisher) Hessian. `w≡1` reduces to [`Self::calib_hessian_outer_f32`].
    pub fn calib_hessian_outer_weighted_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        h: &GpuTensor,
        n: usize,
        k: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "calib_reduce",
            crate::kernels::CALIB_REDUCE_SRC,
            "calib_hessian_outer_weighted_f32",
        )?;
        let x_ptr = x.buf.as_ptr();
        let w_ptr = w.buf.as_ptr();
        let h_ptr = h.buf.as_ptr();
        let n_i = n as i32;
        let k_i = k as i32;
        let tile = 16u32;
        let grid_x = ((k as u32) + tile - 1) / tile;
        let grid_y = ((k as u32) + tile - 1) / tile;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &w_ptr as *const _ as *mut c_void,
            &h_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &k_i as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "calib_hessian_outer_weighted_f32",
            [grid_x, grid_y, 1],
            [tile, tile, 1],
            0,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(w_ptr);
                blob.push_ptr(h_ptr);
                blob.push_i32(n_i);
                blob.push_i32(k_i);
                blob
            },
        )
    }

    /// Free a newly allocated ordinary contiguous tensor immediately via HIP
    /// `free`, without returning the buffer to the reusable pool.
    ///
    /// Intended for allocation-failure cleanup of tensors that were just
    /// created (e.g. via [`Self::zeros`]) and must not race a pending async
    /// memset on the active stream, and for one-off load scratch the pool
    /// would otherwise keep. Rejects VMM owners and borrowed buffers
    /// with the same policy as [`Self::free_tensor`].
    pub fn release_tensor_immediate(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.bind_thread()?;
        let key = tensor.buf.as_ptr() as usize;
        if self.vmm_arenas.contains_key(&key) {
            if !tensor.buf.is_vmm_owner() {
                return Err(HipError::new(
                    0,
                    &format!("refusing to free borrowed VMM view at 0x{key:x}"),
                ));
            }
            return Err(HipError::new(
                0,
                &format!("refusing immediate free of VMM owner at 0x{key:x}"),
            ));
        }
        if tensor.buf.is_borrowed() || tensor.buf.is_vmm_owner() {
            return Err(HipError::new(
                0,
                &format!("refusing to free non-owning tensor at 0x{key:x}"),
            ));
        }
        // zeros() may have queued an async memset on the active stream; free
        // must not race that write.
        if let Some(stream) = self.active_stream.as_ref() {
            self.hip.stream_synchronize(stream)?;
        }
        self.hip.free(tensor.buf)
    }

    /// Host→device `upload_f32` calls since process start. A hot loop that
    /// re-uploads a constant vector every call shows up here; a loop that
    /// uploads once and caches does not.
    pub fn htod_uploads() -> usize {
        HTOD_UPLOADS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Allocation counters for the buffer pool: `(new, reused, bytes_new)`.
    /// `new` counts real `hipMalloc` calls, `reused` counts free-list hits.
    /// A hot loop that frees what it allocates keeps `new` flat and grows
    /// `reused`; a loop that leaks grows `new` on every iteration.
    pub fn pool_stats(&self) -> (usize, usize, usize) {
        (
            self.pool.total_new,
            self.pool.total_reused,
            self.pool.total_allocated,
        )
    }

    /// Drain the GPU memory pool. Actually calls hipFree on all pooled buffers.
    /// Call after model unload to return VRAM to the system.
    pub fn drain_pool(&mut self) {
        self.bind_thread_or_warn();
        self.pool.drain(&self.hip);
    }

    /// Invalidate every weight-pointer-keyed cache on the Gpu. Must be called
    /// any time a loaded model's weights are about to be freed; otherwise the
    /// next model load can allocate buffers at addresses that previously held
    /// different weights and the cache will incorrectly hit on stale entries.
    /// Affected caches:
    ///   * mmq_screen_cache: per-weight (safe, unsafe) screening verdicts (#87).
    ///   * fp16_shadow_cache: lazily-built FP16 dequant of HFQ4 weights for
    ///     the rocBLAS prefill path (CDNA3-only). Owns GpuTensors, so the
    ///     entries are released back to the pool here.
    ///   * the gathered QSA attention's VMM workspace
    ///     (`ScratchState::qsa_gather_vmm`): released, so the next model
    ///     reserves its own context and unload's VMM gate sees no Gpu-held
    ///     owner. The legacy slot (`qsa_gather_f16`) is kept, as before.
    pub fn invalidate_weight_caches(&mut self) {
        self.bind_thread_or_warn();
        self.mmq_screen.cache.clear();
        let shadows: Vec<GpuTensor> = self.fp16_shadow_cache.drain().map(|(_, t)| t).collect();
        for t in shadows {
            let _ = self.free_tensor(t);
        }
        // A failed release is retained for `ensure_vmm_cleaned` to retry
        // and report.
        let _ = self.release_qsa_gather_workspace();
    }

    /// Invalidate the pointer-keyed F16 conversion cache. Must be called
    /// between layers in batched prefill when the same activation buffer
    /// (e.g. `pb_tmp`) is reused with different contents each layer —
    /// the cache sees the same GPU pointer and skips the F32→F16
    /// conversion, silently serving stale F16 data from the previous
    /// layer.
    pub fn invalidate_fp16_cache(&mut self) {
        // bind_thread: skip — nulls a CPU-side cache pointer, no device call.
        self.scratch.fp16_x_source_ptr = std::ptr::null_mut();
    }

    /// Tear down all captured hipGraphs + their kernarg blobs. Captured
    /// graphs hold device pointers into the model's KV cache, scratch, and
    /// draft weights baked into kernarg memory by hipStreamEndCapture. Once
    /// any of those tensors are freed and the pool re-uses their buffers
    /// for the next model, replaying the captured graph would execute against
    /// either dangling or wrong-content pointers. The warmup sets would also
    /// wrongly skip the per-B / per-n_steps JIT step on the new model. Must
    /// be called from `unload_model` before the underlying tensors are
    /// returned to the pool.
    ///
    /// Affected state:
    ///   * graph_exec / captured_graph: single-slot AR forward graph.
    ///   * verify_graph_cache + verify_warmed_up + verify_capturing_b:
    ///     DFlash per-B verify-forward graphs.
    ///   * replay_graph_cache + replay_warmed_up + replay_capturing_n:
    ///     DFlash per-n_steps tape-replay graphs.
    pub fn invalidate_graph_state(&mut self) {
        self.bind_thread_or_warn();
        self.graphs.graph_destroy(&self.hip, self.device_id);
        self.graphs
            .verify_graph_destroy_all(&self.hip, self.device_id);
        self.graphs
            .replay_graph_destroy_all(&self.hip, self.device_id);
    }

    /// Typed F32 device-buffer copy. Unlike hipMemcpy D2D, this launch is
    /// visible to the retained-replay recorder and carries an explicit ABI.
    pub fn copy_f32_buffer(&mut self, dst: &GpuTensor, src: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "copy_f32_buffer",
            crate::kernels::COPY_F32_BUFFER_SRC,
            "copy_f32_buffer",
        )?;
        let dp = dst.buf.as_ptr();
        let sp = src.buf.as_ptr();
        let mut n_i32 =
            i32::try_from(n).map_err(|_| HipError::new(0, "copy_f32_buffer length exceeds i32"))?;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut n_i32 as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(dp);
            blob.push_ptr(sp);
            blob.push_i32(n_i32);
            blob
        };
        // Grid is in workgroups: one 256-thread block per 256 elements.  (It
        // used to be the element count rounded up, i.e. 256x too many blocks,
        // whose `blockIdx.x * blockDim.x` overflowed i32 past 2^31 threads.)
        let grid = (n as u32).div_ceil(256);
        self.launch_maybe_blob(
            "copy_f32_buffer",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// Copy `rows` contiguous source rows into one slot of a strided
    /// destination row. Used by DSpark hidden capture to replace B separate
    /// D2D memcpy nodes with one typed dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_f32_strided_slot_buffer(
        &mut self,
        dst: &GpuTensor,
        src: &GpuTensor,
        rows: usize,
        row_elems: usize,
        dst_row_stride: usize,
        dst_slot_offset: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "copy_f32_strided_slot_buffer",
            crate::kernels::COPY_F32_BUFFER_SRC,
            "copy_f32_strided_slot_buffer",
        )?;
        let dp = dst.buf.as_ptr();
        let sp = src.buf.as_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer rows exceed i32"))?;
        let mut row_elems_i32 = i32::try_from(row_elems)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer row length exceeds i32"))?;
        let mut stride_i32 = i32::try_from(dst_row_stride)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer stride exceeds i32"))?;
        let mut slot_i32 = i32::try_from(dst_slot_offset)
            .map_err(|_| HipError::new(0, "copy_f32_strided_slot_buffer slot exceeds i32"))?;
        let mut params: Vec<*mut c_void> = vec![
            &dp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &mut rows_i32 as *mut _ as *mut c_void,
            &mut row_elems_i32 as *mut _ as *mut c_void,
            &mut stride_i32 as *mut _ as *mut c_void,
            &mut slot_i32 as *mut _ as *mut c_void,
        ];
        let blob_builder = || {
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(dp);
            blob.push_ptr(sp);
            blob.push_i32(rows_i32);
            blob.push_i32(row_elems_i32);
            blob.push_i32(stride_i32);
            blob.push_i32(slot_i32);
            blob
        };
        let total = rows
            .checked_mul(row_elems)
            .ok_or_else(|| HipError::new(0, "copy_f32_strided_slot_buffer size overflow"))?;
        let grid = (total as u32).div_ceil(256);
        self.launch_maybe_blob(
            "copy_f32_strided_slot_buffer",
            [grid, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            blob_builder,
        )
    }

    /// 2D strided F32 row copy, one launch for `n_rows` rows:
    ///
    /// ```text
    /// dst[r * dst_row_stride + dst_col_offset + c] = src[r * src_row_stride + c]
    /// ```
    ///
    /// for `r` in `0..n_rows`, `c` in `0..len`. This is the single-launch
    /// replacement for a per-row `copy_d2d` loop — the FLUX.1 MMDiT single
    /// block's `linear2` input assemble issued 2 × 4608 = 9216 tiny D2D
    /// memcpys per block, which is launch-latency bound, not bandwidth bound.
    ///
    /// `dst_col_offset` may address a position past the first row (the kernel
    /// computes `r * dst_row_stride + dst_col_offset + c` with no assumption
    /// that the offset sits inside one row); only the resulting extent has to
    /// fit the destination buffer. A caller that derives the offset from a
    /// position passes it through `dst_col_offset_per_position` so the retained
    /// recorder can declare it and replay can re-derive it from the replay
    /// position instead of replaying a capture-position offset.
    ///
    /// A `float4` fast path is taken automatically when `len`, both row
    /// strides and `dst_col_offset` are multiples of 4 and both device
    /// pointers are 16-byte aligned; every other shape falls back to the
    /// scalar path, so correctness does not depend on the alignment.
    ///
    /// Both tensors must be F32. The full accessed range of each buffer is
    /// bounds-checked here rather than left to the kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows_strided_f32(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        n_rows: usize,
        len: usize,
        src_row_stride: usize,
        dst_row_stride: usize,
        dst_col_offset: usize,
        dst_col_offset_per_position: Option<usize>,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if src.dtype != DType::F32 || dst.dtype != DType::F32 {
            return Err(HipError::new(
                0,
                &format!(
                    "copy_rows_strided_f32: both tensors must be F32 (src {:?}, dst {:?})",
                    src.dtype, dst.dtype
                ),
            ));
        }
        if n_rows == 0 || len == 0 {
            return Ok(());
        }
        if len > src_row_stride {
            return Err(HipError::new(
                0,
                &format!(
                    "copy_rows_strided_f32: len {len} exceeds src_row_stride {src_row_stride}"
                ),
            ));
        }
        let f32_sz = DType::F32.size();
        // Last element touched, +1, in each buffer.
        let src_need = (n_rows - 1)
            .checked_mul(src_row_stride)
            .and_then(|v| v.checked_add(len))
            .and_then(|v| v.checked_mul(f32_sz))
            .ok_or_else(|| HipError::new(0, "copy_rows_strided_f32: src size overflow"))?;
        let dst_need = (n_rows - 1)
            .checked_mul(dst_row_stride)
            .and_then(|v| v.checked_add(dst_col_offset))
            .and_then(|v| v.checked_add(len))
            .and_then(|v| v.checked_mul(f32_sz))
            .ok_or_else(|| HipError::new(0, "copy_rows_strided_f32: dst size overflow"))?;
        if src.buf.size() < src_need {
            return Err(HipError::new(
                0,
                &format!(
                    "copy_rows_strided_f32: src buffer too small (have {}, need {src_need} for {n_rows}×{len} stride {src_row_stride})",
                    src.buf.size()
                ),
            ));
        }
        if dst.buf.size() < dst_need {
            return Err(HipError::new(
                0,
                &format!(
                    "copy_rows_strided_f32: dst buffer too small (have {}, need {dst_need} for {n_rows}×{len} @ col {dst_col_offset} stride {dst_row_stride})",
                    dst.buf.size()
                ),
            ));
        }

        const KERNEL: &str = "copy_rows_strided_f32";
        self.ensure_kernel(KERNEL, crate::kernels::COPY_ROWS_STRIDED_F32_SRC, KERNEL)?;

        let sp = src.buf.as_ptr();
        let dp = dst.buf.as_ptr();
        // float4 needs 16-byte alignment on both the base pointer and every
        // row/column offset it derives from it.
        let aligned = len % 4 == 0
            && src_row_stride % 4 == 0
            && dst_row_stride % 4 == 0
            && dst_col_offset % 4 == 0
            && (sp as usize) % 16 == 0
            && (dp as usize) % 16 == 0;

        let n_rows_i = i32::try_from(n_rows)
            .map_err(|_| HipError::new(0, "copy_rows_strided_f32: n_rows exceeds i32"))?;
        let len_i = i32::try_from(len)
            .map_err(|_| HipError::new(0, "copy_rows_strided_f32: len exceeds i32"))?;
        let ss_i = i32::try_from(src_row_stride)
            .map_err(|_| HipError::new(0, "copy_rows_strided_f32: src_row_stride exceeds i32"))?;
        let ds_i = i32::try_from(dst_row_stride)
            .map_err(|_| HipError::new(0, "copy_rows_strided_f32: dst_row_stride exceeds i32"))?;
        let dco_i = i32::try_from(dst_col_offset)
            .map_err(|_| HipError::new(0, "copy_rows_strided_f32: dst_col_offset exceeds i32"))?;
        let vec4_i = i32::from(aligned);

        let mut blob = hip_bridge::KernargBlob::new();
        blob.push_ptr(sp);
        blob.push_ptr(dp);
        blob.push_i32(n_rows_i);
        blob.push_i32(len_i);
        blob.push_i32(ss_i);
        blob.push_i32(ds_i);
        blob.push_i32(dco_i);
        // The offset the scalar actually landed at, not a hand-counted layout
        // constant. The kernel reads it as `i32`, so the declared product must
        // stay inside `i32` for every replay position it can see.
        let dco_offset = blob.len() - 4;
        blob.push_i32(vec4_i);
        blob.pad_to(16);
        let dco_binding = match dst_col_offset_per_position {
            None => None,
            Some(factor) => {
                let factor = u32::try_from(factor).map_err(|_| {
                    HipError::new(
                        0,
                        "copy_rows_strided_f32: per-position offset factor exceeds u32",
                    )
                })?;
                Some([crate::replay::ReplayKernargBinding::PositionMulU32 {
                    offset: dco_offset,
                    factor,
                }])
            }
        };
        let kernargs: &[crate::replay::ReplayKernargBinding] =
            dco_binding.as_ref().map_or(&[], |bindings| &bindings[..]);

        const BLOCK: u32 = 256;
        let cols = if aligned { len / 4 } else { len };
        let grid_x = (cols as u32).div_ceil(BLOCK);
        // Grid-stride on y in the kernel, so capping at the conservative
        // 65535 launch limit stays correct for any row count.
        let grid_y = (n_rows as u32).min(65535);
        let bytes = n_rows * len * f32_sz * 2; // read + write
        let timer = crate::profile::begin_timer(&self.hip, KERNEL, KERNEL, bytes);
        let result = self.launch_blob_recorded(
            KERNEL,
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            blob.as_mut_slice(),
            ReplayLaunchBindings {
                grid: None,
                kernargs,
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Drop captured graph state and retained Redline replay after a live KV
    /// layout switch so the next forward cannot replay stale K/V modes, base
    /// pointers, or kernarg blobs baked under the prior tier.
    pub fn invalidate_for_kv_mode_switch(&mut self) {
        // bind_thread: skip — invalidate_graph_state binds; replay.poison is CPU.
        self.invalidate_graph_state();
        // Retained AQL/PM4 tapes capture KV base pointers and tier-dependent
        // kernargs. Poison so no old tape can run against the new logical layout.
        if self.replay.is_enabled() || self.replay.state() != crate::replay::ReplayState::Hip {
            self.replay
                .poison("KV mode switch invalidated retained Redline replay state");
        }
    }

    /// Invalidate captured execution after a model-owned cache capacity
    /// bucket grows. The allocation change is intentional and recoverable, so
    /// retained replay is re-armed rather than placed in sticky fallback.
    pub fn invalidate_for_layout_growth(&mut self) {
        // bind_thread: skip — invalidate_graph_state binds; rearm is CPU.
        self.invalidate_graph_state();
        self.replay.rearm_after_layout_growth();
    }

    /// Drop captured execution state before a scratch slot grows. A captured
    /// hipGraph embeds the pointers live at capture time; freeing a replaced
    /// buffer under it replays freed memory (`HipError(700)`), so hipGraphs
    /// still drop wholesale here. The retained PM4 route is different: its
    /// kernarg segments re-encode from `ReplayBindings`, so the route is kept
    /// and only armed for a post-growth re-resolve (revision bump +
    /// re-encode, no re-lowering). The next launch drains the refresh; a
    /// resource that actually moved fails closed back to the historic drop
    /// path (`rearm_after_layout_growth`) and HIP runs.
    ///
    /// Called by every `Gpu` scratch wrapper that can grow, gated on
    /// [`crate::scratch::scratch_will_grow`]. Cheap no-op unless a graph has
    /// been captured in this process (see
    /// [`crate::graph::any_graph_captured`]); growth is monotonic and bounded
    /// by the largest shape ever seen, so the invalidation is rare.
    ///
    /// Never invalidates mid-capture/record: destroying graph state then
    /// would clear the `capture_blobs` the in-flight capture is filling.
    /// Growth inside a capture keeps the historical behaviour (warmup-first
    /// flows size scratch before capturing, so this is not expected).
    pub fn invalidate_for_scratch_growth(&mut self) {
        if !crate::scratch::scratch_growth_invalidates(
            crate::graph::any_graph_captured(),
            self.graphs.capture_mode,
            self.replay.is_recording(),
        ) {
            return;
        }
        self.invalidate_graph_state();
        if self.replay.prepared_pm4_route_active() {
            self.replay.arm_binding_refresh_for_scratch_growth();
        } else {
            self.replay.rearm_after_layout_growth();
        }
    }

    // ── Kernel operations ───────────────────────────────────────

    /// y = A * x (matrix-vector multiply, A is [M, K], x is [K], y is [M])

    /// y = A_q4k * x (quantized matrix-vector multiply, A stored as Q4_K on GPU)
    /// a_raw: raw Q4_K bytes on GPU, x: F32 input, y: F32 output
    /// m: number of output rows, k: number of input columns (must be multiple of 256)

    /// HFQ4-G128 GEMV: flat 4-bit with 128-weight groups.
    /// K must be multiple of 128.

    /// ParoQuant Givens rotation: apply learned pairwise rotations + channel
    /// scaling to activation vector x in-place. Called before GEMV on
    /// ParoQ4G128 weights.
    ///
    /// x: [seq_len, hidden_dim] F16 (modified in place)
    /// pairs: [krot, hidden_dim] I16
    /// theta: [krot, hidden_dim/2] F16
    /// channel_scales: [hidden_dim] F16

    /// Out-of-place Givens rotation. Reads `x_in`, writes rotated
    /// activations to `x_out`. Replaces the
    /// `copy_d2d + givens_rotate` pair used by `rotate_x_paro_for` —
    /// one graph node + one inter-node dependency removed.
    #[allow(clippy::too_many_arguments)]

    /// Fused silu(gate)*up + per-channel scale + krot rounds of Givens
    /// rotation. Single-launch replacement for the
    /// `silu_mul_f32 + givens_rotate` pair used by the ParoQuant routed
    /// gate→down hop. Same shared-memory + grid contract as
    /// `givens_rotate`, plus two additional input pointers (gate, up)
    /// and a separate output pointer.
    #[allow(clippy::too_many_arguments)]

    /// Ensure the ParoQuant activation scratch buffer is allocated (F32, sized for dim).

    /// Device-to-device copy.
    ///
    /// Routes through `memcpy_dtod_auto` so it picks `memcpy_dtod_async` on
    /// the active (capturing) stream when one is set, falling back to the sync
    /// legacy-stream path otherwise. The raw `hip.memcpy_dtod` call would
    /// deadlock hipGraph capture with "operation would make the legacy stream
    /// depend on a capturing blocking stream" (matches the H2D fix in 7790ac6a).
    ///
    /// Callers must pass `n_bytes` explicitly to state intent — the prior
    /// implicit `min(src.size(), dst.size())` silently truncated mismatched
    /// copies, which was a footgun.
    pub fn copy_d2d(&self, src: &GpuTensor, dst: &GpuTensor, n_bytes: usize) -> HipResult<()> {
        // bind_thread: skip — delegates to memcpy_dtod_auto which binds
        debug_assert!(
            n_bytes <= src.buf.size(),
            "copy_d2d: n_bytes ({n_bytes}) exceeds src.buf.size ({})",
            src.buf.size()
        );
        debug_assert!(
            n_bytes <= dst.buf.size(),
            "copy_d2d: n_bytes ({n_bytes}) exceeds dst.buf.size ({})",
            dst.buf.size()
        );
        self.memcpy_dtod_auto(&dst.buf, &src.buf, n_bytes)
    }

    /// PARO4-G128 GEMV: ParoQuant pair-rotated activation + W4 weights.
    /// K must be multiple of 128 and M must be a multiple of the AWQ pack size
    /// (8). Each block computes one packed output column (8 output rows).

    /// Residual PARO4-G128 GEMV: y += A(x) where x is pair-rotated per
    /// ParoQuant metadata. One block computes one AWQ packed output column.

    /// PARO4-G128 fused SwiGLU down projection: y += W * (silu(gate) * up).
    /// Saves the standalone `silu_mul_f32` launch and ffn_hidden global write/read.

    /// PARO4-G128T direct GEMV for tiny-M projections. This keeps the Paro
    /// rotation inside the GEMV block instead of materializing x_rot globally.

    /// Residual PARO4-G128T direct GEMV for tiny-M projections.

    /// PARO4-G128 activation pre-rotation. This materializes the ParoQuant
    /// channel-scale + pair-rotation transform once per projection so the
    /// packed GEMV does not repeat it for every 8-output pack.

    /// PARO4-G128 fused SwiGLU activation + Paro pre-rotation. This is the
    /// useful fused shape for down projection: `x_rot = rotate(silu(gate)*up)`.

    /// PARO4-G128T activation pre-rotation. Same math as PARO4-G128, but
    /// theta is stored as precomputed f16 sin/cos pairs in the payload.

    /// PARO4-G128T fused SwiGLU activation + Paro pre-rotation.

    /// PARO4-G128 GEMV over an already materialized Paro-rotated activation.

    /// Residual PARO4-G128 GEMV over an already materialized Paro-rotated
    /// activation.

    /// PARO4-G128T GEMV over an already materialized Paro-rotated activation.
    /// The payload stores qweight as [M/8, K], making the inner-loop reads
    /// contiguous for the GEMV access pattern.

    /// Residual PARO4-G128T GEMV over an already materialized Paro-rotated
    /// activation.

    /// PARO4-G128T prerotated GEMV with four output lanes per block. This
    /// duplicates qweight reads relative to the 8-lane pack but lowers
    /// accumulator/register pressure for empirical Atlas testing.

    /// Residual PARO4-G128T pack4 prerotated GEMV.

    /// PARO4-G128T prerotated GEMV with two output lanes per block. This is
    /// an Atlas probe for whether lower accumulator pressure beats duplicate
    /// qweight traffic on the residual/down hot path.

    /// Residual PARO4-G128T pack2 prerotated GEMV.

    /// PARO4-G128T prerotated GEMV with one output lane per block.

    /// Residual PARO4-G128T pack1 prerotated GEMV.

    /// PARO4-G128 rotate-once wrapper used for env-gated runtime probes.

    /// PARO4-G128 rotate-once residual wrapper used for env-gated runtime probes.

    /// PARO4-G128 fused SwiGLU rotate-once down projection.

    /// PARO4-G128T rotate-once wrapper for engine-tiled qweight payloads.

    /// PARO4-G128T rotate-once residual wrapper for engine-tiled qweight payloads.

    /// PARO4-G128T fused SwiGLU rotate-once down projection.

    /// PARO4-G128T fused gate/up decode path. Gate and up have distinct
    /// Paro rotations, so this still rotates both, but batches the two
    /// rotations and the two pack4 GEMVs into two launches instead of four.

    /// PARO4-G128T fused LA projection path. The four Paro projections have
    /// distinct rotations, so this batches four rotates and four pack4 GEMVs
    /// into two launches.
    #[allow(clippy::too_many_arguments)]
    // ═══════════════════════════════════════════════════════════════════════════
    // Batch precompilation — compile all kernels a model needs in parallel
    // ═══════════════════════════════════════════════════════════════════════════

    /// Pre-compile all kernels needed for Qwen3.5 inference with a given
    /// weight quantization and KV cache type. Runs hipcc in parallel.
    #[cfg(feature = "deltanet")]
    pub fn precompile_qwen35(
        &mut self,
        weight_quant: &str,
        kv_type: &str,
        head_dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // asym kernels #include "turbo_common.h" + "givens_common.h"; the
        // runtime dispatch path (see ensure_givens4_kernel) prepends the
        // header bodies and strips the #includes. We mirror that exactly so
        // the hash matches and the runtime re-uses our cached .hsaco.
        //
        // asym3 (SP1) additionally #includes "kv_slot_desc.h" — mirror
        // ensure_givens4_kernel's conditional handling (only sources that
        // actually contain the directive get it stripped/prepended) so the
        // other assemble_asym callers (asym2/asym4/fwht2/fwht3/fwht4), which
        // don't include it, keep producing byte-identical source to what the
        // runtime compiles for them, and only asym3's precompiled hash grows
        // to include the header.
        let assemble_asym = |body: &str| -> String {
            let needs_kv_slot_desc = body.contains("#include \"kv_slot_desc.h\"");
            let stripped = body
                .replace("#include \"turbo_common.h\"", "")
                .replace("#include \"givens_common.h\"", "")
                .replace("#include \"kv_slot_desc.h\"", "");
            if needs_kv_slot_desc {
                format!(
                    "{}\n{}\n{}\n{}",
                    kernels::TURBO_COMMON_H,
                    kernels::GIVENS_COMMON_SRC,
                    kernels::KV_SLOT_DESC_H,
                    stripped
                )
            } else {
                format!(
                    "{}\n{}\n{}",
                    kernels::TURBO_COMMON_H,
                    kernels::GIVENS_COMMON_SRC,
                    stripped
                )
            }
        };

        // Common kernels for all Qwen3.5 models (DeltaNet + FullAttn shared ops)
        let mut specs: Vec<(&str, String)> = vec![
            ("rmsnorm", kernels::RMSNORM_SRC.to_string()),
            ("add_inplace", kernels::ADD_INPLACE_SRC.to_string()),
            ("mul", kernels::MUL_SRC.to_string()),
            ("silu_mul", kernels::SILU_MUL_SRC.to_string()),
            ("sigmoid", kernels::SIGMOID_SRC.to_string()),
            ("alpha_gate", kernels::ALPHA_GATE_SRC.to_string()),
            ("conv1d_silu", kernels::CONV1D_SILU_SRC.to_string()),
            ("l2_norm", kernels::L2_NORM_SRC.to_string()),
            (
                "fused_qk_l2_norm_scale",
                kernels::FUSED_QK_L2_NORM_SCALE_SRC.to_string(),
            ),
            (
                "fused_sigmoid_alpha_gate",
                kernels::FUSED_SIGMOID_ALPHA_GATE_SRC.to_string(),
            ),
            (
                "conv1d_silu_split",
                kernels::CONV1D_SILU_SPLIT_SRC.to_string(),
            ),
            (
                "conv1d_silu_split_tree",
                kernels::CONV1D_SILU_SPLIT_TREE_SRC.to_string(),
            ),
            (
                "gated_delta_net_q8_tree",
                kernels::GATED_DELTA_NET_Q8_TREE_SRC.to_string(),
            ),
            ("sigmoid_mul", kernels::SIGMOID_MUL_SRC.to_string()),
            ("topk_logits", kernels::TOPK_LOGITS_SRC.to_string()),
            ("scale_f32", kernels::SCALE_F32_SRC.to_string()),
            ("gated_norm", kernels::GATED_NORM_SRC.to_string()),
            (
                "rope_partial_interleaved",
                kernels::ROPE_PARTIAL_INTERLEAVED_SRC.to_string(),
            ),
            // FullAttn: Q+gate deinterleave split
            ("deinterleave", kernels::DEINTERLEAVE_SRC.to_string()),
            // DeltaNet: Q/K repeat-interleave for asymmetric MQA (replaces 64+ memcpy_dtod calls per layer on 4B/9B)
            (
                "repeat_interleave_qk",
                kernels::REPEAT_INTERLEAVE_QK_SRC.to_string(),
            ),
        ];

        // Weight-format-specific GEMV
        match weight_quant {
            "hfq6" => {
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
            }
            "paro4" => {
                specs.push(("gemv_paro4g128", kernels::GEMV_PARO4G128_SRC.to_string()));
            }
            "mq6" => {
                // MQ6 = FWHT-rotated HFQ6-G256. Needs both the MQ6 GEMV and the
                // raw HFQ6 GEMV (used by a few residual paths), plus the new
                // batched MQ6 GEMM for gemma4 Promote6 tensors (v_proj/down_proj).
                specs.push(("gemv_mq6g256", kernels::GEMV_MQ6G256_SRC.to_string()));
                specs.push(("gemv_hfq6g256", kernels::GEMV_HFQ6G256_SRC.to_string()));
                specs.push(("gemm_mq6g256", kernels::GEMM_MQ6G256_SRC.to_string()));
            }
            "hfq4" => {
                let (src, module) =
                    kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
                specs.push((module, src.to_string()));
                specs.push((
                    "gemv_hfq4g256_wide",
                    kernels::GEMV_HFQ4G256_WIDE_SRC.to_string(),
                ));
                // Multi-projection fused kernels (LA 4-way, FA 3-way, FFN
                // gate+up). Cross-arch — same 4-accumulator inner loop as
                // gemv_hfq4g256.hip; precompile on every arch that uses
                // the HFQ4 weight path.
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_hoist_x32 {
                    specs.push((
                        "fused_qkvza_hfq4g256_hoist_x32_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC.to_string(),
                    ));
                } else {
                    specs.push((
                        "fused_qkvza_hfq4g256",
                        kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_2wave {
                    specs.push((
                        "fused_qkvza_hfq4g256_2wave",
                        kernels::FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_k2048 {
                    specs.push((
                        "fused_qkvza_hfq4g256_k2048_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_qkv_hfq4g256",
                    kernels::FUSED_QKV_HFQ4G256_SRC.to_string(),
                ));
                specs.push((
                    "fused_gate_up_hfq4g256",
                    kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string(),
                ));
                if self.arch_caps.is_gfx1100() {
                    specs.push((
                        "fused_gate_up_hfq4g256_stage_x32_gfx1100",
                        kernels::FUSED_GATE_UP_HFQ4G256_STAGE_X32_GFX1100_SRC.to_string(),
                    ));
                }
                // gfx906/gfx908/gfx94x wave64-native variants — cut
                // wavefront pressure in half on the hottest kernels. Wave32
                // block=[32,1,1] kernels otherwise waste the upper 32 lanes
                // of every wave slot on these wave64-native arches.
                if self.arch_caps.is_wave64_native()
                    || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64)
                {
                    // Single-token QKV paths. RDNA3 may opt into Radiowave's
                    // explicit wave64 packing experiment; native-wave64
                    // targets retain their established selection.
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_wave64_native() {
                    // Remaining single-token and batched native-wave64 paths.
                    specs.push((
                        "fused_gate_up_hfq4g256_wave64",
                        kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    // Batched (DFlash verify path — hottest).
                    specs.push((
                        "gemm_qkvza_hfq4g256_wave64",
                        kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_qkv_hfq4g256_wave64",
                        kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_wave64",
                        kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_residual_wave64",
                        kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                }
                // gfx1100 multi-row GEMV is opt-in via HIPFIRE_GEMV_ROWS={2,4,8}.
                // Empirically slower than the single-row kernel on gfx1100 at all
                // tested matrix sizes (see commit log / multi-row kernel header),
                // so we only precompile when the env var explicitly requests it.
                if self.arch_caps.is_rdna3_dgpu() && self.flags.gemv_rows.unwrap_or(1) > 1 {
                    specs.push((
                        "gemv_hfq4g256_multirow_rdna3",
                        kernels::GEMV_HFQ4G256_MULTIROW_GFX1100_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_residual_multirow_rdna3",
                        kernels::GEMV_HFQ4G256_RESIDUAL_MULTIROW_GFX1100_SRC.to_string(),
                    ));
                }
            }
            "mq4" => {
                // MQ4 = FWHT-rotated HFQ4-G256 — default format for current registry.
                // Shares the HFQ4 fused kernels (same blob, different dispatch key)
                // plus MQ-specific rotation kernels.
                let (src, module) =
                    kernels::gemv_hfq4g256_for_arch(&self.arch_caps, self.flags.rdna2_variant);
                specs.push((module, src.to_string()));
                specs.push(("gemv_mq4g256", kernels::GEMV_MQ4G256_SRC.to_string()));
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_hoist_x32 {
                    specs.push((
                        "fused_qkvza_hfq4g256_hoist_x32_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_HOIST_X32_GFX1100_SRC.to_string(),
                    ));
                } else {
                    specs.push((
                        "fused_qkvza_hfq4g256",
                        kernels::FUSED_QKVZA_HFQ4G256_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkvza_2wave {
                    specs.push((
                        "fused_qkvza_hfq4g256_2wave",
                        kernels::FUSED_QKVZA_HFQ4G256_2WAVE_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_hfq4_qkvza_k2048 {
                    specs.push((
                        "fused_qkvza_hfq4g256_k2048_gfx1100",
                        kernels::FUSED_QKVZA_HFQ4G256_K2048_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_qkv_hfq4g256",
                    kernels::FUSED_QKV_HFQ4G256_SRC.to_string(),
                ));
                specs.push((
                    "fused_gate_up_hfq4g256",
                    kernels::FUSED_GATE_UP_HFQ4G256_SRC.to_string(),
                ));
                if self.arch_caps.is_gfx1100() {
                    specs.push((
                        "fused_gate_up_hfq4g256_stage_x32_gfx1100",
                        kernels::FUSED_GATE_UP_HFQ4G256_STAGE_X32_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_rmsnorm_mq_rotate",
                    kernels::FUSED_RMSNORM_MQ_ROTATE_SRC.to_string(),
                ));
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_wavegrid {
                    specs.push((
                        "fused_rmsnorm_mq_rotate_wavegrid",
                        kernels::FUSED_RMSNORM_MQ_ROTATE_WAVEGRID_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_split {
                    specs.push((
                        "rmsnorm_reduce_gfx1100",
                        kernels::RMSNORM_REDUCE_GFX1100_SRC.to_string(),
                    ));
                    specs.push((
                        "rotate_with_rms_gfx1100",
                        kernels::ROTATE_WITH_RMS_GFX1100_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_gfx1100() && self.flags.rdna3_rmsnorm_vecsum {
                    specs.push((
                        "fused_rmsnorm_mq_rotate_vecsum",
                        kernels::FUSED_RMSNORM_MQ_ROTATE_VECSUM_GFX1100_SRC.to_string(),
                    ));
                }
                specs.push((
                    "fused_silu_mul_mq_rotate",
                    kernels::FUSED_SILU_MUL_MQ_ROTATE_SRC.to_string(),
                ));
                // gfx906/gfx908/gfx94x wave64 variants — see hfq4 branch for rationale.
                if self.arch_caps.is_wave64_native()
                    || (self.arch_caps.is_rdna3_dgpu() && self.flags.rdna3_hfq4_qkv_wave64)
                {
                    // Single-token QKV paths. RDNA3 may opt into Radiowave's
                    // explicit wave64 packing experiment; native-wave64
                    // targets retain their established selection.
                    specs.push((
                        "fused_qkvza_hfq4g256_wave64",
                        kernels::FUSED_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "fused_qkv_hfq4g256_wave64",
                        kernels::FUSED_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                }
                if self.arch_caps.is_wave64_native() {
                    // Remaining single-token and batched native-wave64 paths.
                    specs.push((
                        "fused_gate_up_hfq4g256_wave64",
                        kernels::FUSED_GATE_UP_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_WAVE64_SRC.to_string(),
                    ));
                    // Batched (DFlash verify path — hottest).
                    specs.push((
                        "gemm_qkvza_hfq4g256_wave64",
                        kernels::GEMM_QKVZA_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_qkv_hfq4g256_wave64",
                        kernels::GEMM_QKV_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_wave64",
                        kernels::GEMM_HFQ4G256_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemm_hfq4g256_residual_wave64",
                        kernels::GEMM_HFQ4G256_RESIDUAL_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_GATE_UP_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                    specs.push((
                        "gemv_hfq4g256_moe_down_indexed_batched_wave64",
                        kernels::GEMV_HFQ4G256_MOE_DOWN_INDEXED_BATCHED_WAVE64_SRC.to_string(),
                    ));
                }
            }
            "q8" => {
                specs.push(("gemv_q8_0", kernels::GEMV_Q8_0_SRC.to_string()));
            }
            _ => {}
        }

        // Embedding kernels — Q8_0 is most common, also cover HFQ4G256/G128 variants
        specs.push(("embedding_q8", kernels::EMBEDDING_Q8_SRC.to_string()));
        specs.push((
            "embedding_hfq4g256",
            kernels::EMBEDDING_HFQ4G256_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g128",
            kernels::EMBEDDING_HFQ4G128_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g128_batched",
            kernels::EMBEDDING_HFQ4G128_BATCHED_SRC.to_string(),
        ));
        specs.push((
            "embedding_hfq4g256_batched",
            kernels::EMBEDDING_HFQ4G256_BATCHED_SRC.to_string(),
        ));
        specs.push((
            "embedding_q8_batched",
            kernels::EMBEDDING_Q8_BATCHED_SRC.to_string(),
        ));

        // DeltaNet kernels
        specs.push((
            "gated_delta_net_q8",
            kernels::GATED_DELTA_NET_Q8_SRC.to_string(),
        ));

        // KV cache kernels. asym3 is the current default — always ships flash.
        // q8 is the compat path with its own flash tile+reduce for long context.
        match kv_type {
            "asym4" => {
                specs.push((
                    "kv_cache_write_asym_k_givens4",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_givens4_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS4_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym4_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym4_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM4_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht4" => {
                // Same byte layout as asym4 — just different K-rotation primitive.
                specs.push((
                    "kv_cache_write_asym_k_fwht4",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht4_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT4_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht4_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT4_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht4_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT4_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht3" => {
                // Same byte layout as asym3 (single-pass 256-element), FWHT rotation.
                specs.push((
                    "kv_cache_write_asym_k_fwht3",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht3_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT3_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht3_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT3_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht3_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT3_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "fwht2" => {
                // Same byte layout as asym2, FWHT rotation. 2-pass over 128.
                specs.push((
                    "kv_cache_write_asym_k_fwht2",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_fwht2_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_FWHT2_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_fwht2_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT2_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_fwht2_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_FWHT2_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "asym3" => {
                specs.push((
                    "kv_cache_write_asym_k_givens3",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_SRC),
                ));
                if self.arch_caps.is_gfx1100() && head_dim == 256 {
                    specs.push((
                        "kv_cache_write_asym3_q8_pair_gfx1100",
                        assemble_asym(kernels::KV_CACHE_WRITE_ASYM3_Q8_PAIR_GFX1100_SRC),
                    ));
                }
                specs.push((
                    "kv_cache_write_asym_k_givens3_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS3_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym3_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym3_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM3_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "asym2" => {
                specs.push((
                    "kv_cache_write_asym_k_givens2",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_SRC),
                ));
                specs.push((
                    "kv_cache_write_asym_k_givens2_batched",
                    assemble_asym(kernels::KV_CACHE_WRITE_ASYM_K_GIVENS2_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym2_tile",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_SRC),
                ));
                specs.push((
                    "attention_flash_asym2_tile_batched",
                    assemble_asym(kernels::ATTENTION_FLASH_ASYM2_TILE_BATCHED_SRC),
                ));
                specs.push((
                    "attention_flash_asym_reduce_batched",
                    kernels::ATTENTION_FLASH_ASYM_REDUCE_BATCHED_SRC.to_string(),
                ));
            }
            "q8" | _ => {
                specs.push((
                    "kv_cache_write_q8_0",
                    kernels::KV_CACHE_WRITE_Q8_0_SRC.to_string(),
                ));
                specs.push((
                    "attention_q8_0_kv",
                    kernels::ATTENTION_Q8_0_KV_SRC.to_string(),
                ));
                specs.push(("attention_q8_0_kv_batched", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched_masked_slots: the kernel
                    // #includes kv_slot_desc.h, but this precompile path
                    // (like the runtime hipcc compile) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::ATTENTION_Q8_0_KV_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("attention_q8_0_kv_independent_masked_windowed", {
                    // Shares ATTENTION_Q8_0_KV_BATCHED_SRC with the entry
                    // above, so it needs the identical strip-and-prepend:
                    // that source #includes kv_slot_desc.h and this
                    // precompile path has no -I to kernels/src.
                    let stripped = kernels::ATTENTION_Q8_0_KV_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("attention_q8_0_flash_prefill", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched above: the kernel now
                    // #includes kv_slot_desc.h (Task 6), but this
                    // precompile path (like the runtime hipcc compile in
                    // attention_q8_0_flash_prefill_slots) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::ATTENTION_Q8_0_FLASH_PREFILL_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("kv_cache_write_q8_0_batched", {
                    // Same header-stripping treatment as
                    // attention_q8_0_kv_batched above: the kernel now
                    // #includes kv_slot_desc.h (Task 3), but this
                    // precompile path (like the runtime hipcc compile in
                    // kv_cache_write_q8_0_batched_slots) has no -I to
                    // kernels/src, so the directive must be stripped and
                    // the header body prepended instead.
                    let stripped = kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("kv_cache_write_q8_0_independent", {
                    // Shares KV_CACHE_WRITE_Q8_0_BATCHED_SRC with the entry
                    // above, so it needs the identical strip-and-prepend:
                    // that source #includes kv_slot_desc.h and this
                    // precompile path has no -I to kernels/src. Warms the
                    // module the continuous-batch decode path JITs first.
                    let stripped = kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push(("kv_cache_write_q8_0_independent_masked", {
                    // Shares KV_CACHE_WRITE_Q8_0_BATCHED_SRC with the entries
                    // above, so it needs the identical strip-and-prepend.
                    // Warms the module forward_tick JITs first on a
                    // partially-active batch.
                    let stripped = kernels::KV_CACHE_WRITE_Q8_0_BATCHED_SRC
                        .replace("#include \"kv_slot_desc.h\"", "");
                    format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped)
                }));
                specs.push((
                    "attention_flash_q8_0_tile",
                    kernels::ATTENTION_FLASH_Q8_0_TILE_SRC.to_string(),
                ));
                specs.push((
                    "attention_flash_q8_0_reduce",
                    kernels::ATTENTION_FLASH_Q8_0_REDUCE_SRC.to_string(),
                ));
            }
        }

        // Exact parallel-sampler modules used by sample_top_p_pf. Sources are
        // the same string rewrites as runtime (see sampling.rs helpers) so the
        // compile-cache hash hits on first token instead of hipcc JIT.
        for (name, src) in crate::sampling::sample_top_p_parallel_precompile_specs() {
            specs.push((name, src));
        }

        fn precompile_symbols(name: &str) -> Vec<&str> {
            match name {
                "rmsnorm" => vec!["rmsnorm_f32"],
                "add_inplace" => vec!["add_inplace_f32"],
                "mul" => vec!["mul_f32"],
                "silu_mul" => vec!["silu_mul_f32"],
                "sigmoid" => vec!["sigmoid_f32"],
                "alpha_gate" => vec!["alpha_gate_f32"],
                "conv1d_silu" => vec!["conv1d_silu_f32"],
                "l2_norm" => vec!["l2_norm_f32"],
                "fused_qk_l2_norm_scale" => vec!["fused_qk_l2_norm_scale_f32"],
                "fused_sigmoid_alpha_gate" => vec!["fused_sigmoid_alpha_gate_f32"],
                "conv1d_silu_split" => vec!["conv1d_silu_split_f32"],
                "conv1d_silu_split_tree" => vec!["conv1d_silu_split_tree_f32"],
                "gated_delta_net_q8_tree" => vec!["gated_delta_net_q8_tree"],
                "sigmoid_mul" => vec!["sigmoid_mul_f32"],
                "topk_logits" => vec!["topk_logits_f32"],
                "scale_f32" => vec!["scale_f32"],
                "gated_norm" => vec!["gated_norm_f32"],
                "rope_partial_interleaved" => vec!["rope_partial_interleaved_f32"],
                "deinterleave" => vec!["deinterleave_f32"],
                "repeat_interleave_qk" => vec!["repeat_interleave_qk_f32"],
                "gated_delta_net_q8" => vec!["gated_delta_net_q8"],
                // MQ4 GEMV module exports both the main GEMV and the standalone
                // x rotation kernel used by the prerotated dispatch path.
                "gemv_mq4g256" => vec!["gemv_mq4g256", "mq_rotate_x"],
                // Arch-variant HFQ4 GEMV modules all expose the same symbol.
                n if n.starts_with("gemv_hfq4g256_rdna") => vec!["gemv_hfq4g256"],
                n if n.starts_with("gemv_hfq4g256_gfx") => vec!["gemv_hfq4g256"],
                "fused_qkvza_hfq4g256_k2048_gfx1100" => {
                    vec!["fused_qkvza_hfq4g256_k2048"]
                }
                // Multi-row RDNA3 modules expose three entry points per .hsaco
                "gemv_hfq4g256_multirow_rdna3" => vec![
                    "gemv_hfq4g256_multirow_r2",
                    "gemv_hfq4g256_multirow_r4",
                    "gemv_hfq4g256_multirow_r8",
                ],
                "gemv_hfq4g256_residual_multirow_rdna3" => vec![
                    "gemv_hfq4g256_residual_multirow_r2",
                    "gemv_hfq4g256_residual_multirow_r4",
                    "gemv_hfq4g256_residual_multirow_r8",
                ],
                "gemv_hfq4g256_moe_gate_up_indexed_wave64" => {
                    vec!["gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"]
                }
                "gemv_hfq4g256_moe_down_indexed_wave64" => {
                    vec!["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_wave64"]
                }
                "gemv_hfq4g256_moe_gate_up_indexed_batched_wave64" => {
                    vec!["gemv_hfq4g256_moe_gate_up_k8_indexed_batched_wave64"]
                }
                "gemv_hfq4g256_moe_down_indexed_batched_wave64" => {
                    vec!["gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched_wave64"]
                }
                "sample_top_p_parallel" => vec![
                    "sample_apply_repeat_penalty",
                    "sample_topk_partial",
                    "sample_topk_finalize",
                ],
                "sample_top_p_parallel_w64" => vec![
                    "sample_apply_repeat_penalty_w64",
                    "sample_topk_partial_w64",
                    "sample_topk_finalize_w64",
                ],
                "sample_top_p_parallel_fast21" => vec![
                    "sample_apply_repeat_penalty_fast21",
                    "sample_topk_partial_fast21",
                    "sample_topk_finalize_fast21",
                ],
                "sample_top_p_parallel_fast65" => vec![
                    "sample_apply_repeat_penalty_fast65",
                    "sample_topk_partial_fast65",
                    "sample_topk_finalize_fast65",
                ],
                other => vec![other],
            }
        }
        // Batch each module once, authenticating its first exported symbol.
        let batch: Vec<(&str, &str, &str)> = specs
            .iter()
            .map(|(name, src)| (*name, src.as_str(), precompile_symbols(name)[0]))
            .collect();
        self.compiler.compile_batch_for_symbols(&batch)?;

        // Now load all modules + functions sequentially (GPU API).
        for (name, src) in &specs {
            let func_names = precompile_symbols(name);
            // Compile and ensure the module is loaded once.
            let obj_path = self.compiler.compile_for_symbol(name, src, func_names[0])?;
            let obj_path_str = obj_path.to_str().unwrap().to_string();
            if !self.modules.contains_key(*name) {
                let module = crate::scratch::module_load_or_recompile(
                    &self.hip,
                    &mut self.compiler,
                    name,
                    src,
                    func_names[0],
                    &obj_path_str,
                )?;
                self.modules.insert(name.to_string(), module);
            }
            let module = &self.modules[*name];
            for func_name in &func_names {
                if self.functions.contains_key(*func_name) {
                    continue;
                }
                self.compiler.compile_for_symbol(name, src, func_name)?;
                let func = self.hip.module_get_function(module, func_name)?;
                self.functions.insert(func_name.to_string(), func);
            }
        }

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Kernel profiler
    // ═══════════════════════════════════════════════════════════════════════════

    /// Profile all compiled kernels: hardware caps + ISA metadata + occupancy.
    pub fn profile(
        &self,
    ) -> (
        crate::profiler::GpuCapability,
        Vec<crate::profiler::KernelProfile>,
    ) {
        self.bind_thread_or_warn();
        let vram = self.hip.get_vram_info().map(|(_, t)| t as u64).unwrap_or(0);
        let cu_hint = self
            .hip
            .get_device_attribute(
                crate::profiler::HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                0,
            )
            .ok()
            .filter(|&v| v > 0)
            .map(|v| crate::profiler::hip_mp_count_to_cu_count(&self.arch, v as u32))
            .filter(|&v| (4..=256).contains(&v));
        crate::profiler::profile_kernels_with_hint(
            &self.arch,
            vram,
            self.compiler.compiled_kernels(),
            cu_hint,
        )
    }

    /// Bulk F32 → BF16 conversion with round-to-nearest-even.
    ///
    /// Elementwise `dst[i] = (bf16)src[i]` for `i < nelems`, matching the
    /// host reference `round_to_bf16` (`crates/hipfire-arch-deepseek4/src/parent/codec.rs:92-110`).
    /// Used to stage BF16 activations for `gemm_bf16_mfma_gfx942` under
    /// `HIPFIRE_CALIB_BF16=1` so calibration GEMMs land on the CDNA3 MFMA
    /// kernel instead of the scalar F32 kernel.
    pub fn convert_f32_to_bf16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        nelems: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if nelems == 0 {
            return Ok(());
        }
        let src_need = nelems
            .checked_mul(4)
            .ok_or_else(|| HipError::new(0, "convert_f32_to_bf16: nelems*4 overflow"))?;
        let dst_need = nelems
            .checked_mul(2)
            .ok_or_else(|| HipError::new(0, "convert_f32_to_bf16: nelems*2 overflow"))?;
        if src.buf.size() < src_need {
            return Err(HipError::new(
                0,
                &format!(
                    "convert_f32_to_bf16: src buffer too small (have {} need {src_need} for nelems={nelems})",
                    src.buf.size()
                ),
            ));
        }
        if dst.buf.size() < dst_need {
            return Err(HipError::new(
                0,
                &format!(
                    "convert_f32_to_bf16: dst buffer too small (have {} need {dst_need} for nelems={nelems})",
                    dst.buf.size()
                ),
            ));
        }
        if nelems > i32::MAX as usize {
            return Err(HipError::new(
                0,
                &format!("convert_f32_to_bf16: nelems {nelems} exceeds i32::MAX"),
            ));
        }
        const KERNEL: &str = "convert_f32_to_bf16";
        const SRC: &str = include_str!("../../../kernels/src/convert_f32_to_bf16.hip");
        self.ensure_kernel(KERNEL, SRC, KERNEL)?;
        let src_ptr = src.buf.as_ptr();
        let dst_ptr = dst.buf.as_ptr();
        let nelems_i32 = nelems as i32;
        let mut params: Vec<*mut c_void> = vec![
            &src_ptr as *const _ as *mut c_void,
            &dst_ptr as *const _ as *mut c_void,
            &nelems_i32 as *const _ as *mut c_void,
        ];
        let grid = nelems.div_ceil(256) as u32;
        let bytes = src_need + dst_need;
        let timer = crate::profile::begin_timer(&self.hip, "convert", KERNEL, bytes);
        let result =
            self.launch_maybe_blob(KERNEL, [grid, 1, 1], [256, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(src_ptr);
                blob.push_ptr(dst_ptr);
                blob.push_i32(nelems_i32);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

impl Drop for Gpu {
    /// Defensive: bind owning device before any future per-field `Drop`
    /// impls call `hipFree` etc. Uses `bind_thread_or_warn` to avoid
    /// panic-in-Drop from `bind_thread`'s `debug_assert!`.
    fn drop(&mut self) {
        if std::thread::panicking()
            && self.vmm_allocation_count() == 0
            && self.host_mapped.is_empty()
        {
            return;
        }
        self.bind_thread_or_warn();
        if let Err(err) = self.release_registered_vmm() {
            eprintln!("[rdna-compute] failed to release VMM arena during Gpu drop: {err}");
        }
        if let Err(err) = self.release_registered_host_mapped() {
            eprintln!("[rdna-compute] failed to release host-mapped tensor during Gpu drop: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    /// Pins the per-row geometry the per-arch copies used to hard-code.
    #[test]
    fn row_bytes_matches_the_producer_layouts() {
        assert_eq!(DType::BF16.row_bytes(160), Some(320));
        assert_eq!(DType::F32.row_bytes(3), Some(12));
        assert_eq!(DType::Q8_0.row_bytes(160), Some(170));
        assert_eq!(DType::Q8_0.row_bytes(33), Some(68));
        assert_eq!(DType::MQ4G256V2.row_bytes(2560), Some(10 * 136));
        assert_eq!(DType::MQ4G128V2.row_bytes(640), Some(5 * 68));
        assert_eq!(DType::MQ4G128V2.row_bytes(129), Some(2 * 68));
        assert_eq!(DType::MQ6G256V2.row_bytes(2560), Some(10 * 200));
        assert_eq!(DType::ParoQ4G128.row_bytes(128), Some(72));
        assert_eq!(DType::MFP4G32E8SOA.row_bytes(2560), Some(16 + 80 + 80 * 16));
        assert_eq!(DType::MFP4G32E8.row_bytes(4096), Some(2192));
        assert_eq!(DType::MFP4G32E8SOA.row_bytes(4096), Some(2192));
        assert_eq!(DType::MFP3G32E8.row_bytes(4096), Some(1680));
        assert_eq!(DType::MFP4G32E8.row_bytes(256), Some(152));
        assert_eq!(DType::MFP4G32E8SOA.row_bytes(256), Some(160));
        assert_eq!(DType::MFP3G32E8.row_bytes(256), Some(120));
        assert_eq!(DType::Q8_0.row_bytes(0), None);
        assert_eq!(DType::Raw.row_bytes(8), None);
    }

    #[test]
    fn kernarg_values_encode_natural_alignment_layout() {
        let mut b = hip_bridge::KernargBlob::new();
        let p = 0x1122_3344_5566_7788usize as *mut c_void;
        super::KernargValue::push_to(p, &mut b);
        super::KernargValue::push_to(7i32, &mut b);
        super::KernargValue::push_to(p, &mut b);
        super::KernargValue::push_to(1.5f32, &mut b);
        let mut want = Vec::new();
        want.extend_from_slice(&(p as usize).to_ne_bytes());
        want.extend_from_slice(&7i32.to_ne_bytes());
        want.extend_from_slice(&[0; 4]);
        want.extend_from_slice(&(p as usize).to_ne_bytes());
        want.extend_from_slice(&1.5f32.to_ne_bytes());
        assert_eq!(b.as_bytes(), &want[..]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "kernarg 0")]
    fn kernarg_cross_check_rejects_params_order_mismatch() {
        let mut m = 3i32;
        let mut k = 4i32;
        let params: Vec<*mut c_void> =
            vec![&mut k as *mut _ as *mut c_void, &mut m as *mut _ as *mut c_void];
        super::debug_assert_params_match(
            &params,
            &[super::kernarg_value_bytes(&m), super::kernarg_value_bytes(&k)],
        );
    }

    use std::ffi::c_void;
    use super::gen_fwht_signs;
    use super::DType;
    use super::Gpu;
    use super::HessianCapture;
    use super::MQ2G256V2_GROUP_BYTES;
    use super::MQ3G256V2_GROUP_BYTES;
    use super::MQ5G256V2_GROUP_BYTES;
    use super::MQ6G256V2_GROUP_BYTES;

    #[test]
    fn q8hfq_row_stride_matches_legacy_formula() {
        // Hard-coded expected stride (128B-aligned) — independent oracle, NOT the
        // formula re-run. raw_row = (k/32)*2 + k, then padded up to a 128B boundary.
        assert_eq!(DType::Q8HFQ.row_stride(4096), 4352); // raw 4352, already aligned
        assert_eq!(DType::Q8HFQ.row_stride(11008), 11776); // raw 11696 → padded
        assert_eq!(DType::Q8HFQ.row_stride(5120), 5504); // raw 5440 → padded
        assert_eq!(DType::Q8HFQ.row_stride(14336), 15232); // raw 15232, already aligned
        assert_eq!(DType::Q8HFQ.row_stride(256), 384); // raw 272 → padded
        assert_eq!(DType::Q8HFQ.row_stride(96), 128); // raw 102 → padded
    }

    #[test]
    fn non_q8hfq_row_stride_is_zero() {
        for dt in [
            DType::HFQ4G256,
            DType::MQ4G256,
            DType::Q8_0,
            DType::F16,
            DType::HFP4G32,
            DType::MFP4G32,
        ] {
            assert_eq!(dt.row_stride(4096), 0, "{dt:?} must have stride 0");
        }
    }

    #[test]
    fn only_hfp4_family_requires_k_mod_256() {
        assert!(DType::HFP4G32.requires_k_mod_256());
        assert!(DType::MFP4G32.requires_k_mod_256());
        for dt in [
            DType::HFQ4G256,
            DType::MQ4G256,
            DType::Q8HFQ,
            DType::Q8_0,
            DType::F16,
            DType::F32,
        ] {
            assert!(!dt.requires_k_mod_256(), "{dt:?} must NOT require k%256");
        }
    }

    #[test]
    fn neutral_v2_requires_k_mod_256_and_awq() {
        for dt in [
            DType::MQ4G256V2,
            DType::MQ4CG256,
            DType::MQ6G256V2,
            DType::MQ5G256V2,
            DType::MQ3G256V2,
            DType::MQ2G256V2,
        ] {
            assert!(dt.requires_k_mod_256(), "{dt:?} must require K%256==0");
            assert!(dt.supports_awq_sidecar(), "{dt:?} must support AWQ sidecar");
        }
        // Legacy counterparts must NOT be confused with V2 at type level.
        assert_ne!(DType::MQ6G256V2, DType::MQ6G256);
        assert_ne!(DType::MQ5G256V2, DType::MQ5G256);
        assert_ne!(DType::MQ3G256V2, DType::MQ3G256);
        assert_ne!(DType::MQ2G256V2, DType::MQ2G256);
        // Group bytes match spec one-to-one.
        assert_eq!(MQ6G256V2_GROUP_BYTES, 200);
        assert_eq!(MQ5G256V2_GROUP_BYTES, 168);
        assert_eq!(MQ3G256V2_GROUP_BYTES, 104);
        assert_eq!(MQ2G256V2_GROUP_BYTES, 72);
    }

    /// Deterministic pseudo-random rows (no RNG crate): mix in exact zeros,
    /// signed-zero, negatives, and large/small magnitudes to exercise the
    /// zero-skip branch and fp rounding.
    fn fixed_row(k: usize, seed: u64) -> Vec<f32> {
        let mut state: u64 = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) as f64 / (u32::MAX as f64)
        };
        (0..k)
            .map(|c| {
                let r = next();
                if r < 0.20 {
                    0.0f32
                } else if r < 0.25 {
                    -0.0f32
                } else {
                    let mag = (r - 0.5) * 4.0;
                    let scale = if c % 7 == 0 { 1.0e3 } else { 1.0 };
                    (mag * scale) as f32
                }
            })
            .collect()
    }

    /// LOAD-BEARING CORRECTNESS GATE for the rayon-parallel `accumulate_token`.
    /// The parallel per-token accumulate (across disjoint (tensor,expert)
    /// accumulators) MUST be BIT-IDENTICAL to the serial per-item `accumulate`
    /// reference — a wrong parallel sum would silently produce wrong Hessians
    /// and reconfound the GPTQ-on-E8 experiment. We compare every f64 entry by
    /// raw bits (`to_bits()`), not an epsilon: each accumulator is updated by
    /// exactly one thread with the SAME ops as serial, so exact equality is the
    /// correct, strongest assertion. Multiple tokens are accumulated to verify
    /// the parallel path is also correct across repeated touches of an entry.
    #[test]
    fn accumulate_token_bit_identical_to_serial() {
        // A3B-shaped: gate_up K=2048 (8 blocks), down K=512 (2 blocks); 8 experts
        // (distinct ids) -> 16 distinct (tensor,expert) keys per token.
        let n_experts = 8usize;
        let n_tokens = 5usize;
        let k_gate = 2048usize;
        let k_down = 512usize;

        let mut par = HessianCapture::default();
        let mut ser = HessianCapture::default();

        for t in 0..n_tokens {
            // Build this token's distinct work-units.
            let mut names: Vec<String> = Vec::new();
            let mut xs: Vec<Vec<f32>> = Vec::new();
            let mut ks: Vec<usize> = Vec::new();
            for e in 0..n_experts {
                let l = 7usize;
                names.push(format!(
                    "model.language_model.layers.{l}.mlp.experts.{e}.gate_up_proj.weight"
                ));
                xs.push(fixed_row(k_gate, (t * 131 + e) as u64));
                ks.push(k_gate);
                names.push(format!(
                    "model.language_model.layers.{l}.mlp.experts.{e}.down_proj.weight"
                ));
                xs.push(fixed_row(k_down, (t * 977 + e + 1) as u64));
                ks.push(k_down);
            }
            // Serial reference: per-item accumulate in fixed order.
            for i in 0..names.len() {
                ser.accumulate(&names[i], &xs[i], ks[i]);
            }
            // Parallel path: one batched call.
            let items: Vec<(String, &[f32], usize)> = (0..names.len())
                .map(|i| (names[i].clone(), xs[i].as_slice(), ks[i]))
                .collect();
            par.accumulate_token(&items);
        }

        // Compare every accumulator bit-for-bit.
        assert_eq!(par.entries.len(), ser.entries.len());
        for (name, pacc) in &par.entries {
            let sacc = ser.entries.get(name).expect("name in serial map");
            assert_eq!(pacc.n_blocks, sacc.n_blocks, "{name} n_blocks");
            assert_eq!(pacc.n_rows, sacc.n_rows, "{name} n_rows");
            for b in 0..pacc.n_blocks {
                let pb = &pacc.blocks[b];
                let sb = &sacc.blocks[b];
                for idx in 0..pb.len() {
                    assert_eq!(
                        pb[idx].to_bits(),
                        sb[idx].to_bits(),
                        "{name} block={b} idx={idx}: parallel {} != serial {} (NOT bit-identical)",
                        pb[idx],
                        sb[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn mq_signs_128_deterministic() {
        let s1 = gen_fwht_signs(43, 128);
        let s2 = gen_fwht_signs(1043, 128);
        assert_eq!(s1.len(), 128);
        assert_eq!(s2.len(), 128);
        for x in &s1 {
            assert!(*x == 1.0 || *x == -1.0, "signs1 contains {x}");
        }
        for x in &s2 {
            assert!(*x == 1.0 || *x == -1.0, "signs2 contains {x}");
        }
        // Reproducibility
        assert_eq!(gen_fwht_signs(43, 128), s1);
        assert_eq!(gen_fwht_signs(1043, 128), s2);
        // Distinct from G256 seeds
        assert_ne!(
            gen_fwht_signs(42, 128),
            s1,
            "seed 43 should differ from seed 42"
        );
    }

    /// One device is shared by every GPU test in this module, and several assert on
    /// allocator/VRAM bookkeeping. `upload_raw_copy_failure_hip_frees_owner` demands
    /// byte-exact `hipMemGetInfo` free across its window, which a sibling test
    /// allocating or freeing VRAM in that window breaks — observed as a failure on
    /// roughly one `cargo test -p rdna-compute --lib` run in three with no source
    /// change. Holding this lock for each GPU test's duration makes the suite
    /// deterministic without weakening any assertion. Poisoning is ignored: a panic
    /// in one test must not fail the rest.
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Serialized GPU handle for a test. The guard must be *bound* for the test's
    /// duration — `let Some((mut gpu, _guard)) = try_gpu() else { … }` — because
    /// dropping it immediately would release the lock.
    fn try_gpu() -> Option<(super::Gpu, std::sync::MutexGuard<'static, ()>)> {
        let guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match super::Gpu::init() {
            Ok(gpu) => Some((gpu, guard)),
            Err(_) => None,
        }
    }

    /// The CPU-executed offload path reads spilled weight bytes through
    /// `host_bytes`, which resolves the *host* pointer from the registry — the
    /// device-visible alias `hipHostGetDevicePointer` returns need not be the
    /// address `hipHostFree` needs, so guessing costs a read of unrelated
    /// memory. Pin the identity instead: the bytes' device view and host view
    /// are the same allocation, and a device tensor has no host view at all.
    #[test]
    fn host_bytes_aliases_the_uploaded_allocation() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let payload: Vec<u8> = (0..64u8)
            .map(|i| i.wrapping_mul(3).wrapping_add(1))
            .collect();
        let t = gpu
            .upload_raw_host(&payload, &[payload.len()])
            .expect("host-mapped raw upload");
        assert!(gpu.host_located(&t), "upload_raw_host must host-locate");
        assert_eq!(
            gpu.host_bytes(&t).expect("registered host pointer"),
            &payload[..],
            "the host view must be the allocation the upload filled"
        );

        let dev = gpu.alloc_tensor(&[8], DType::F32).expect("device tensor");
        assert!(
            gpu.host_bytes(&dev).is_none(),
            "a device-resident tensor has no host view"
        );

        // The D2H helper reads through the device alias, so both views must agree.
        let mut back = vec![0u8; payload.len()];
        gpu.memcpy_dtoh_auto(&mut back, &t.buf).expect("dtoh");
        assert_eq!(back, payload);

        gpu.free_tensor(t).ok();
        gpu.free_tensor(dev).ok();
    }

    /// The row-chunked requant (and its reordered ranges) packs every row
    /// byte-identically to the one-shot whole-weight requant: unpack, rotate
    /// and pack are all row-local.
    #[test]
    fn requant_g256_chunked_rows_match_the_one_shot_requant() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (m, k) = (4101usize, 512usize);
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut q8 = Vec::with_capacity(m * k / 32 * 34);
        for _ in 0..m * k / 32 {
            // f16 scales in [2^-6, 2^-5): finite, normal.
            q8.extend_from_slice(&(0x2400u16 | (next() as u16 & 0x03ff)).to_le_bytes());
            q8.extend((0..32).map(|_| next() as u8));
        }
        let mut weight = gpu.upload_raw(&q8, &[q8.len()]).expect("upload Q8_0");
        weight.dtype = DType::Q8_0;
        for target in [DType::MQ2G256V2, DType::MQ6G256V2] {
            let row = Gpu::requant_g256_bytes(1, k, target).unwrap().1;
            let whole = gpu
                .requant_g256_chunked(&weight, m, k, target, &[(0, m)], m)
                .expect("one-shot requant");
            let whole_bytes = gpu.download_raw_bytes(&whole).expect("download");
            let ranges = [(0, 1500), (3000, m), (1500, 3000)];
            let chunked = gpu
                .requant_g256_chunked(&weight, m, k, target, &ranges, 1000)
                .expect("chunked requant");
            let chunked_bytes = gpu.download_raw_bytes(&chunked).expect("download");
            let expected: Vec<u8> = ranges
                .iter()
                .flat_map(|&(start, end)| whole_bytes[start * row..end * row].iter().copied())
                .collect();
            assert_eq!(chunked_bytes.len(), expected.len());
            assert!(chunked_bytes == expected, "{target:?}: chunked rows differ");
            let default = gpu.requant_g256(&weight, m, k, target).expect("requant");
            assert!(gpu.download_raw_bytes(&default).expect("download") == whole_bytes);
            for tensor in [whole, chunked, default] {
                gpu.free_tensor(tensor).ok();
            }
        }
        gpu.free_tensor(weight).ok();
    }

    /// A D→H copy is a host sync point and can never be part of a captured
    /// graph. Entering capture with one pending is a bug in whoever decided to
    /// capture a CPU-executed step, so it must fail loudly rather than enqueue
    /// onto the capturing stream (where replay would silently return stale
    /// activations).
    #[test]
    fn memcpy_dtoh_auto_refuses_under_capture() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let t = gpu.upload_f32(&[1.0f32; 4], &[4]).expect("upload");
        let mut back = vec![0u8; 16];
        gpu.graphs.capture_mode = true;
        let err = gpu
            .memcpy_dtoh_auto(&mut back, &t.buf)
            .expect_err("must refuse during capture");
        gpu.graphs.capture_mode = false;
        assert!(
            err.to_string().contains("capture"),
            "refusal must name capture: {err}"
        );
        gpu.memcpy_dtoh_auto(&mut back, &t.buf)
            .expect("works outside capture");
        gpu.free_tensor(t).ok();
    }

    /// A leaked host-mapped owner must make teardown refuse, the way a leaked VMM
    /// arena does. Offloaded weights are a second owner class, and without this
    /// check an orphaned one would let unload report a clean handoff while still
    /// holding pinned system RAM.
    #[test]
    fn ensure_vmm_cleaned_refuses_while_a_host_mapped_owner_is_live() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        assert_eq!(
            gpu.host_mapped_count(),
            0,
            "fresh GPU owns no host-mapped tensors"
        );
        gpu.ensure_vmm_cleaned().expect("idle GPU is clean");

        let t = gpu
            .upload_f32_host(&[1.0f32; 16], &[16])
            .expect("host-mapped upload");
        assert_eq!(gpu.host_mapped_count(), 1);
        let err = gpu
            .ensure_vmm_cleaned()
            .expect_err("must refuse while a host-mapped owner is live");
        assert!(
            err.to_string().contains("host-mapped"),
            "refusal must name the owner class: {err}"
        );

        gpu.free_tensor(t).expect("free the host-mapped owner");
        assert_eq!(gpu.host_mapped_count(), 0);
        gpu.ensure_vmm_cleaned()
            .expect("clean again after the owner is freed");
    }

    /// A copy longer than 2^31 / 256 elements used to launch one block per
    /// element; `blockIdx.x * blockDim.x` then overflowed i32 and the kernel
    /// wrote before the destination (GPU page fault at 1024-row Qwen4 PLE).
    #[test]
    fn copy_f32_buffer_handles_more_than_8m_elements() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let n = 9 * 1024 * 1024 + 7;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let src_gpu = gpu.upload_f32(&src, &[n]).expect("src upload");
        let dst_gpu = gpu.zeros(&[n + 64], super::DType::F32).expect("dst");
        let dst_view = dst_gpu.sub_offset(32, n);
        gpu.copy_f32_buffer(&dst_view, &src_gpu, n).expect("copy");
        let out = gpu.download_f32(&dst_gpu).expect("download");
        assert!(out[..32].iter().all(|v| *v == 0.0), "copy wrote before dst");
        assert!(
            out[32 + n..].iter().all(|v| *v == 0.0),
            "copy wrote past dst"
        );
        assert!(
            out[32..32 + n].iter().zip(&src).all(|(a, b)| a == b),
            "copy mismatch"
        );
    }

    #[test]
    fn ensure_vmm_cleaned_never_releases_a_live_owner() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let tensor = match unsafe { gpu.alloc_vmm_tensor(&[4096], super::DType::Raw, 0, &[]) } {
            Ok(tensor) => tensor,
            Err(_) => {
                eprintln!("skip: VMM unavailable");
                return;
            }
        };
        assert_eq!(gpu.vmm_allocation_count(), 1);

        let err = gpu
            .ensure_vmm_cleaned()
            .expect_err("load cleanup must refuse a still-owned VMM arena");
        assert!(err.to_string().contains("live VMM"), "{err}");
        assert_eq!(gpu.vmm_allocation_count(), 1);
        #[cfg(not(windows))]
        assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(0));
        #[cfg(windows)]
        {
            // Single-segment workaround: the full reservation is mapped up front,
            // so a 4096-byte reserve already reports a granularity-aligned prefix.
            let mapped = gpu
                .vmm_mapped_bytes(&tensor)
                .expect("windows full-map must report mapped bytes");
            let gran = gpu
                .vmm_granularity(&tensor)
                .expect("registered VMM tensor must expose granularity");
            assert_eq!(mapped % gran, 0, "mapped must be granularity-aligned");
            assert!(
                mapped >= 4096,
                "full-map must cover the 4096-byte reservation, got {mapped}"
            );
        }

        gpu.free_tensor(tensor).expect("free live owner");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    /// The recorded-HIP oracle is the reference Redline checks PM4 replay
    /// against, so it must re-derive every engine-declared kernarg field at its
    /// replay position, exactly as the retained PM4 plan does. A declared
    /// destination-row offset captured at position `p` and replayed at `p + k`
    /// must write row `p + k`; writing row `p` again is the stale capture value.
    #[test]
    fn recorded_hip_oracle_applies_declared_bindings_at_replay_position() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let width = 64usize;
        let rows = 8usize;
        let captured = 2usize;
        let replayed = 5usize;
        let src: Vec<f32> = (0..width).map(|i| i as f32 + 1.0).collect();
        let src_gpu = gpu.upload_f32(&src, &[width]).expect("src upload");
        let dst_gpu = gpu
            .zeros(&[rows * width], super::DType::F32)
            .expect("dst allocation");

        gpu.replay =
            crate::replay::ReplayController::new_armed(crate::replay::ReplayBackendRequest::Auto);
        gpu.replay.begin_capture().expect("open recording window");
        gpu.copy_rows_strided_f32(
            &src_gpu,
            &dst_gpu,
            1,
            width,
            width,
            width,
            captured * width,
            Some(width),
        )
        .expect("captured copy");
        gpu.replay.finish_capture().expect("close recording window");
        assert_eq!(gpu.replay.recorded_launches().len(), 1);
        assert!(
            !gpu.replay.recorded_launches()[0]
                .declared_kernarg_bindings()
                .is_empty(),
            "the copy must declare its per-position row offset"
        );

        gpu.hip
            .memset(&dst_gpu.buf, 0, rows * width * std::mem::size_of::<f32>())
            .expect("clear dst");
        gpu.replay_recorded_hip_prefix_at(1, replayed)
            .expect("oracle replay");
        let out = gpu.download_f32(&dst_gpu).expect("download");

        let row = |r: usize| &out[r * width..(r + 1) * width];
        assert!(
            row(captured).iter().all(|v| *v == 0.0),
            "oracle replayed the capture-position row {captured} (stale declared binding)"
        );
        assert_eq!(row(replayed), &src[..], "oracle missed row {replayed}");
        for r in (0..rows).filter(|r| *r != replayed) {
            assert!(row(r).iter().all(|v| *v == 0.0), "stray write in row {r}");
        }

        gpu.free_tensor(src_gpu).expect("free src");
        gpu.free_tensor(dst_gpu).expect("free dst");
    }

    #[test]
    fn vmm_fullmap_covers_unaligned_reservation() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        // 4097 is a multiple of no real VMM granularity: the pre-fix Windows
        // path (map_next with the raw byte size) rejected it even though the
        // reservation itself was valid.
        const UNALIGNED: usize = 4097;
        let mut tensor =
            match unsafe { gpu.alloc_vmm_tensor(&[UNALIGNED], super::DType::Raw, 0, &access) } {
                Ok(tensor) => tensor,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
        assert_eq!(gpu.vmm_allocation_count(), 1);
        #[cfg(not(windows))]
        {
            // Growth path: nothing is mapped up front, so grow an aligned
            // cover for the unaligned reservation.
            assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(0));
            let gran = gpu
                .vmm_granularity(&tensor)
                .expect("registered VMM tensor must expose granularity");
            let cover = UNALIGNED.div_ceil(gran) * gran;
            gpu.grow_vmm_tensor(&mut tensor, cover, &access)
                .expect("grow to cover unaligned reservation");
            assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(cover));
        }
        #[cfg(windows)]
        {
            // Single-segment path: the whole reservation is mapped up front,
            // so any further growth must fail without disturbing the mapping.
            let mapped = gpu
                .vmm_mapped_bytes(&tensor)
                .expect("windows full-map must report mapped bytes");
            let gran = gpu
                .vmm_granularity(&tensor)
                .expect("registered VMM tensor must expose granularity");
            assert_eq!(mapped % gran, 0, "mapped must be granularity-aligned");
            assert!(
                mapped >= UNALIGNED,
                "full-map must cover the {UNALIGNED}-byte reservation, got {mapped}"
            );
            let over_err = gpu
                .grow_vmm_tensor(&mut tensor, gran, &access)
                .expect_err("windows full-map must already cover the reservation");
            assert!(
                over_err.to_string().contains("exceed reserve"),
                "unexpected full-map growth error: {over_err}"
            );
            assert_eq!(gpu.vmm_mapped_bytes(&tensor), Some(mapped));
        }
        // The observable logical prefix is usable on both paths.
        let expect: Vec<u8> = (0..UNALIGNED).map(|i| (i % 251) as u8).collect();
        gpu.hip
            .memcpy_htod(&tensor.buf, &expect)
            .expect("htod unaligned prefix");
        let mut actual = vec![0u8; UNALIGNED];
        gpu.hip
            .memcpy_dtoh(&mut actual, &tensor.buf)
            .expect("dtoh unaligned prefix");
        assert_eq!(actual, expect);
        gpu.free_tensor(tensor).expect("free");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    #[test]
    fn vmm_alias_boundary_walks_granule_sizes() {
        use super::vmm_alias_boundary as alias;
        const MIB: usize = 1 << 20;
        let g = [2 * MIB, 2 * MIB, 2 * MIB];
        assert_eq!(alias(g, 0), 0);
        assert_eq!(alias(g, 1), 0);
        assert_eq!(alias(g, 2 * MIB - 1), 0);
        assert_eq!(alias(g, 2 * MIB), 2 * MIB);
        assert_eq!(alias(g, 5 * MIB), 4 * MIB);
        assert_eq!(alias(g, 6 * MIB), 6 * MIB);
        assert_eq!(alias(g, 100 * MIB), 6 * MIB, "never past the granule list");
        // A shorter final granule at the reserve end is alias-eligible whole.
        let short = [2 * MIB, MIB];
        assert_eq!(alias(short, 3 * MIB), 3 * MIB);
        assert_eq!(alias(short, 3 * MIB - 1), 2 * MIB);
        assert_eq!(alias(std::iter::empty::<usize>(), MIB), 0);
        assert_eq!(alias([0usize, 2 * MIB], 4 * MIB), 0, "zero-size granule ends the walk");
        assert_eq!(alias([usize::MAX, 2], usize::MAX), usize::MAX);
        assert_eq!(alias([usize::MAX, 2], usize::MAX - 1), 0, "overflowing sum ends the walk");
    }

    /// FP8 rows are 516 B (512 payload + 4 scale), so a row straddles the 2 MiB
    /// granule edge; the alias stops at the granule and the frontier is unaligned.
    #[test]
    fn vmm_alias_fp8_rows_straddling_a_granule_and_the_writable_guard() {
        use super::vmm_alias_boundary as alias;
        const GRANULE: usize = 2 << 20;
        const ROW: usize = 516;
        let granules = [GRANULE; 4];
        let valid = 4065 * ROW; // row 4064 spans 2_097_024..2_097_540
        assert!(valid > GRANULE && (GRANULE / ROW) * ROW < GRANULE);
        // K/V: writable_from == valid -> the first granule is aliased and
        // the 388-byte frontier (a straddling row's tail) is copied.
        let limit = valid.min(valid).min(4 * GRANULE);
        let kv_alias = alias(granules, limit);
        assert_eq!(kv_alias, GRANULE);
        assert_eq!(valid - kv_alias, 388);
        assert_eq!((valid - kv_alias) % ROW, ROW - (GRANULE - (GRANULE / ROW) * ROW));
        // Raw index with an in-place restored tail: writable_from four rows
        // short of valid falls inside granule 0, so nothing may alias.
        let writable_from = valid - 4 * ROW;
        assert!(writable_from < GRANULE);
        assert_eq!(alias(granules, valid.min(writable_from).min(4 * GRANULE)), 0);
        // granule_prefix_bytes caps the alias even when valid is larger.
        assert_eq!(alias(granules, valid.min(valid).min(GRANULE)), GRANULE);
        assert_eq!(alias(granules, 3 * GRANULE - 1), 2 * GRANULE);
    }

    #[test]
    fn vmm_slab_layout_aligns_every_entry_to_256() {
        use super::vmm_slab_layout as layout;
        assert_eq!(layout(&[]), Some((vec![], 0)));
        assert_eq!(layout(&[0]), Some((vec![0], 0)));
        assert_eq!(layout(&[1]), Some((vec![0], 256)));
        assert_eq!(layout(&[256, 257]), Some((vec![0, 256], 768)));
        let (offsets, total) = layout(&[388, 0, 1000, 256]).unwrap();
        assert_eq!(offsets, vec![0, 512, 512, 1536]);
        assert_eq!(total, 1792);
        assert!(offsets.iter().all(|offset| offset % 256 == 0));
        assert_eq!(layout(&[usize::MAX]), None);
        assert_eq!(layout(&[usize::MAX - 255, 1]), None);
    }

    fn vmm_move(old_base: usize, new_base: usize, reserved: usize) -> super::VmmResourceMove {
        super::VmmResourceMove {
            old_base,
            new_base,
            reserved_bytes: reserved,
            mapped_bytes: reserved / 2,
            owner_generation: 7,
        }
    }

    #[test]
    fn vmm_move_validation_rejects_duplicate_overlap_and_malformed_moves() {
        use super::validate_vmm_move_ranges as validate;
        let ok = [
            vmm_move(0x10_000, 0x90_000, 0x1000),
            vmm_move(0x11_000, 0x91_000, 0x1000),
            vmm_move(0x20_000, 0xa0_000, 0x2000),
        ];
        assert!(validate(&ok).is_ok(), "adjacent and disjoint ranges are fine");
        assert!(validate(&[]).is_ok());

        let duplicate = [
            vmm_move(0x10_000, 0x90_000, 0x1000),
            vmm_move(0x10_000, 0x91_000, 0x1000),
        ];
        assert!(validate(&duplicate).unwrap_err().contains("old ranges"));

        let overlap = [
            vmm_move(0x10_000, 0x90_000, 0x1000),
            vmm_move(0x10_800, 0x91_000, 0x1000),
        ];
        assert!(validate(&overlap).unwrap_err().contains("old ranges"));

        let nested = [
            vmm_move(0x10_000, 0x90_000, 0x4000),
            vmm_move(0x11_000, 0x95_000, 0x1000),
        ];
        assert!(validate(&nested).unwrap_err().contains("old ranges"));

        let new_overlap = [
            vmm_move(0x10_000, 0x90_000, 0x1000),
            vmm_move(0x20_000, 0x90_800, 0x1000),
        ];
        assert!(validate(&new_overlap).unwrap_err().contains("new ranges"));

        assert!(validate(&[vmm_move(0x10_000, 0x90_000, 0)])
            .unwrap_err()
            .contains("reserved_bytes is zero"));
        let mut bad_mapped = vmm_move(0x10_000, 0x90_000, 0x1000);
        bad_mapped.mapped_bytes = 0x1001;
        assert!(validate(&[bad_mapped]).unwrap_err().contains("exceeds reserved"));
        assert!(validate(&[vmm_move(usize::MAX - 8, 0x90_000, 0x1000)])
            .unwrap_err()
            .contains("overflows"));
    }

    /// Capture / fork / free over real VMM granules. Needs a VMM-capable device;
    /// run with `cargo test -p rdna-compute --lib -- --ignored vmm_prefix_capture_fork_hw`.
    #[test]
    #[ignore = "needs a HIP device with VMM granule support"]
    fn vmm_prefix_capture_fork_hw() {
        use hip_bridge::DeviceBuffer;
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut gpu = Gpu::init().expect("vmm_prefix_capture_fork_hw requires a HIP device");
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        const ROW: usize = 516;

        let granularity = gpu
            .vmm_recommended_granularity()
            .expect("VMM granularity query");
        let granule = (2usize << 20).max(granularity).next_multiple_of(granularity);
        let mut src = unsafe { gpu.alloc_vmm_tensor(&[4 * granule], DType::Raw, 0, &access) }
            .expect("alloc VMM source");
        let mapped = gpu
            .grow_vmm_tensor_granules(&mut src, 3 * granule, &access)
            .expect("grow three granules");
        assert_eq!(mapped, 3 * granule);
        assert_eq!(gpu.vmm_reserved_bytes(&src), Some(4 * granule));
        assert_eq!(gpu.vmm_sealed_bytes(&src), Some(0));
        let src_generation = gpu.vmm_owner_generation(&src).expect("source generation");

        let view = |tensor: &super::GpuTensor, bytes: usize| unsafe {
            DeviceBuffer::from_raw(tensor.buf.as_ptr(), bytes)
        };
        let pattern: Vec<u8> = (0..mapped).map(|i| ((i * 131 + 7) % 251) as u8).collect();
        gpu.hip
            .memcpy_htod(&view(&src, mapped), &pattern)
            .expect("fill source");

        // 2.5 granules, rounded down to whole 516-byte rows.
        let valid = (5 * granule / 2) / ROW * ROW;
        let writable_from = valid - 4 * ROW;
        let set = gpu
            .capture_vmm_prefixes(&[super::VmmPrefixSpec {
                tensor: &src,
                valid_bytes: valid,
                writable_from,
            }])
            .expect("capture prefix");
        assert_eq!(set.len(), 1);
        assert_eq!(set.valid_bytes(0), valid);
        assert_eq!(set.arena_alias_bytes(0), 2 * granule);
        assert_eq!(set.alias_bytes(), 2 * granule);
        assert_eq!(set.frontier_bytes(), valid - 2 * granule);
        assert!(set.slab_bytes() >= set.frontier_bytes());
        assert_eq!(gpu.vmm_sealed_bytes(&src), Some(2 * granule));
        let aliased: Vec<_> = set.physical_granules().collect();
        assert_eq!(aliased.len(), 2);

        let mut forks = gpu.fork_vmm_prefixes(&set, &access).expect("fork prefix");
        assert_eq!(forks.len(), 1);
        let fork = forks.pop().unwrap();
        assert_ne!(fork.buf.as_ptr(), src.buf.as_ptr(), "fork must use a fresh VA");
        assert_ne!(gpu.vmm_owner_generation(&fork), Some(src_generation));
        let fork_mapped = gpu.vmm_mapped_bytes(&fork).expect("fork mapped");
        assert_eq!(fork_mapped, 3 * granule);
        assert_eq!(gpu.vmm_reserved_bytes(&fork), Some(4 * granule));
        let src_ids = gpu.vmm_physical_granules(&src).expect("source granules");
        let fork_ids = gpu.vmm_physical_granules(&fork).expect("fork granules");
        assert_eq!(fork_ids.len(), 3);
        assert_eq!(fork_ids[..2], src_ids[..2], "aliased granules share physical ids");
        assert_eq!(fork_ids[..2], aliased[..], "set reports the aliased granules");
        assert_ne!(fork_ids[2], src_ids[2], "the frontier granule is private");
        let mv = gpu.vmm_resource_move(&src, &fork).expect("resource move");
        assert_eq!(mv.old_base, src.buf.as_ptr() as usize);
        assert_eq!(mv.new_base, fork.buf.as_ptr() as usize);
        assert_eq!(mv.reserved_bytes, 4 * granule);
        assert_eq!(mv.mapped_bytes, fork_mapped);
        assert_eq!(Some(mv.owner_generation), gpu.vmm_owner_generation(&fork));

        gpu.hip.device_synchronize().expect("sync after fork");
        let mut fork_bytes = vec![0u8; fork_mapped];
        gpu.hip
            .memcpy_dtoh(&mut fork_bytes, &view(&fork, fork_mapped))
            .expect("read fork");
        assert_eq!(fork_bytes[..valid], pattern[..valid], "fork prefix != source");
        assert!(
            fork_bytes[valid..].iter().all(|byte| *byte == 0),
            "fork bytes past valid must be zero"
        );

        // A write through the fork's private frontier never reaches the source.
        let poke = [0xEEu8; ROW];
        gpu.hip
            .memcpy_htod(&view(&fork, fork_mapped).byte_view(valid, ROW), &poke)
            .expect("write fork frontier");
        let mut src_bytes = vec![0u8; mapped];
        gpu.hip
            .memcpy_dtoh(&mut src_bytes, &view(&src, mapped))
            .expect("read source");
        assert_eq!(src_bytes, pattern, "source changed by a fork write");

        // The fork outlives the source: aliased granules stay alive through the leases.
        gpu.free_tensor(src).expect("free source");
        gpu.hip
            .memcpy_dtoh(&mut fork_bytes, &view(&fork, fork_mapped))
            .expect("re-read fork");
        assert_eq!(fork_bytes[..valid], pattern[..valid]);
        assert_eq!(fork_bytes[valid..valid + ROW], poke);

        gpu.free_tensor(fork).expect("free fork");
        gpu.free_vmm_prefixes(set).expect("free prefix set");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    /// Alloc→init failure must return the owner to the pool. Baseline is taken
    /// after a successful public warm so first-touch `total_new` sits outside the
    /// measured window; a leaked owner still forces a fresh malloc on retry.
    #[test]
    fn alloc_then_init_failure_returns_pool_owner() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let warm = gpu.zeros(&[64], DType::F32).expect("warm public zeros");
        let warm_host = gpu.download_f32(&warm).expect("download warm");
        assert_eq!(warm_host.len(), 64);
        assert!(
            warm_host.iter().all(|&x| x == 0.0),
            "public zeros must clear the buffer"
        );
        gpu.free_tensor(warm).expect("free warm");
        let fresh_allocations = gpu.pool_stats().0;

        // Real allocation, injected init only — not a pre-alloc inject.
        let err = match gpu.alloc_then_init(&[64], DType::F32, |_hip, _stream, _tensor| {
            Err(hip_bridge::HipError::new(2, "injected init failure"))
        }) {
            Err(error) => error,
            Ok(tensor) => {
                let _ = gpu.free_tensor(tensor);
                panic!("injected init failure must surface");
            }
        };
        assert!(
            err.to_string().contains("injected"),
            "unexpected error: {err}"
        );

        // Immediate public retry reuses the returned slot; content still correct.
        let ok = gpu.zeros(&[64], DType::F32).expect("zeros retry");
        let host = gpu.download_f32(&ok).expect("download retry");
        assert_eq!(host, vec![0.0f32; 64]);
        gpu.free_tensor(ok).expect("free retry");
        assert_eq!(
            gpu.pool_stats().0,
            fresh_allocations,
            "failed init leaked its pool allocation instead of rolling it back",
        );
    }

    /// malloc-success / copy-failure must `hip.free` the raw owner (not
    /// `free_tensor`/pool). Soft-skip without GPU like the other leaf tests.
    #[test]
    fn upload_raw_copy_failure_hip_frees_owner() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        // Public success path still works and is free_tensor-teardown'd
        // (pooled free domain — see upload_raw docs). Not an allocator change.
        let warm = gpu
            .upload_raw(&[7u8; 64], &[64])
            .expect("warm public upload_raw");
        assert_eq!(warm.buf.size(), 64);
        assert!(warm.buf.is_hip_allocation());
        gpu.free_tensor(warm).expect("free warm into pool");

        // Global free VRAM cannot verify a release byte-exactly on this driver.
        // Measured (`free_retention` probe, gfx1201): the owner's own bytes always
        // return exactly, but the driver intermittently charges extra overhead at
        // allocation that `hipFree` never returns — 0 or 28 MiB for a 64 MiB
        // request, up to 56 MiB at 2 GiB. Byte-exact equality therefore failed a
        // correct implementation in roughly 2 of 5 runs. Use a payload far above
        // that measured overhead (a leak of this owner drops free by 512 MiB) and
        // tolerate an absolute 64 MiB, which no observed overhead reaches.
        const PROBE_BYTES: usize = 512 * 1024 * 1024;
        const TOLERANCE_BYTES: usize = 64 * 1024 * 1024;
        /// How long to let a late reclaim land. Never-returned driver overhead is
        /// covered by `TOLERANCE_BYTES` instead; this only absorbs lag.
        const POLL_MILLIS: u64 = 500;
        let payload = vec![7u8; PROBE_BYTES];

        let (free_before, total) = gpu.hip.get_vram_info().expect("vram before");
        let pool_before = gpu.pool_stats();

        let err = match gpu.upload_raw_with_copy(&payload, &[PROBE_BYTES], |_hip, _buf, _data| {
            Err(hip_bridge::HipError::new(2, "injected raw H2D failure"))
        }) {
            Err(error) => error,
            Ok(tensor) => {
                let _ = gpu.free_tensor(tensor);
                panic!("injected raw copy failure must surface");
            }
        };
        assert!(
            err.to_string().contains("injected"),
            "unexpected error: {err}"
        );

        // Reclaim on this driver is occasionally LATE rather than lost (measured:
        // the owner's bytes return exactly in 11 of 12 samples, with the "missing"
        // 28 MiB appearing on a later pass), so a single instantaneous sample can't
        // tell "reclaimed late" from "leaked" — which is precisely the distinction
        // this test exists to make. Poll briefly, and fail only if free never
        // recovers.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(POLL_MILLIS);
        let mut free_after = gpu.hip.get_vram_info().expect("vram after").0;
        while free_after + TOLERANCE_BYTES < free_before && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(POLL_MILLIS / 10));
            free_after = gpu.hip.get_vram_info().expect("vram after").0;
        }
        assert!(
            free_after + TOLERANCE_BYTES >= free_before,
            "copy-fail must hip.free the malloc owner: {PROBE_BYTES} bytes never came back \
             (free VRAM {free_before} → {free_after} after {POLL_MILLIS}ms of polling, \
             total={total}, tolerance={TOLERANCE_BYTES})"
        );
        // hip.free path must not touch pool counters (would if free_tensor'd).
        assert_eq!(
            gpu.pool_stats(),
            pool_before,
            "copy-fail must not route the raw malloc through the pool"
        );

        // Public success still works after the failure seam.
        let ok = gpu.upload_raw(&[9u8; 64], &[64]).expect("upload_raw retry");
        assert!(ok.buf.is_hip_allocation());
        gpu.free_tensor(ok).expect("free retry");
    }

    #[test]
    fn free_tensor_unmap_failure_retains_owner_for_retry() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let chunk = {
            // Discover granularity via a throwaway arena path.
            let probe = unsafe { gpu.alloc_vmm_tensor(&[4096], super::DType::Raw, 0, &[]) };
            match probe {
                Ok(t) => {
                    let g = gpu.vmm_granularity(&t).unwrap_or(2 * 1024 * 1024);
                    let _ = gpu.free_tensor(t);
                    g
                }
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            }
        };
        let access = [gpu.device_id];
        let tensor = unsafe {
            gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access)
                .expect("alloc vmm")
        };
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 1);
        let err = gpu
            .free_tensor(tensor)
            .expect_err("unmap fault must surface");
        assert!(
            err.to_string().contains("retained") || err.to_string().contains("injected"),
            "{err}"
        );
        assert_eq!(
            gpu.vmm_allocation_count(),
            1,
            "failed free must keep the arena registered"
        );
        // No double-free: retry with faults cleared must release exactly once.
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned().expect("retry cleanup");
        assert_eq!(gpu.vmm_allocation_count(), 0);
        // Idle ensure is a no-op success (no double free).
        gpu.ensure_vmm_cleaned().expect("second ensure on idle");
    }

    #[test]
    fn free_tensor_release_failure_retains_owner_for_retry() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let tensor =
            match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) } {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Release, 1);
        let err = gpu
            .free_tensor(tensor)
            .expect_err("release fault must surface");
        assert!(
            err.to_string().contains("retained") || err.to_string().contains("injected"),
            "{err}"
        );
        assert_eq!(gpu.vmm_allocation_count(), 1);
        hip_bridge::clear_vmm_faults();
        assert_eq!(gpu.retry_vmm_cleanup().expect("retry"), 0);
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    #[test]
    fn freed_vmm_tensor_va_is_never_reserved_again() {
        // ROCm 7.14+/gfx1201 keeps translating a remapped VA to its first
        // backing, so a model reload must never get a VA an earlier model used.
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let bytes = 32 * chunk;
        let mut used: Vec<(usize, usize)> = Vec::new();
        for _ in 0..8 {
            let tensor = match unsafe {
                gpu.alloc_vmm_tensor(&[bytes], super::DType::Raw, chunk, &access)
            } {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
            let base = tensor.buf.as_ptr() as usize;
            for &(lo, hi) in &used {
                assert!(
                    base + bytes <= lo || base >= hi,
                    "VA 0x{base:x}+{bytes} overlaps released 0x{lo:x}..0x{hi:x}"
                );
            }
            used.push((base, base + bytes));
            gpu.free_tensor(tensor).expect("free");
        }
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }

    #[test]
    fn access_reset_failure_does_not_publish_live_owner() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let before = gpu.vmm_allocation_count();
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::AccessReset, 1);
        let err = match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) }
        {
            Ok(tensor) => {
                let _ = gpu.free_tensor(tensor);
                panic!("access-reset fault must fail initial map");
            }
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("injected") || err.to_string().contains("access"),
            "{err}"
        );
        hip_bridge::clear_vmm_faults();
        // Successful cleanup leaves no pending owner; if cleanup itself failed
        // the arena stays registered and ensure_vmm_cleaned must still drain it.
        if gpu.vmm_allocation_count() > before {
            gpu.ensure_vmm_cleaned()
                .expect("drain retained after access fault");
        }
        assert_eq!(gpu.vmm_allocation_count(), before);
        // Subsequent alloc works (no poisoned process state).
        let ok = unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) }
            .expect("alloc after access fault");
        let _ = gpu.free_tensor(ok).expect("free");
        assert_eq!(gpu.vmm_allocation_count(), before);
    }

    #[test]
    fn ensure_vmm_cleaned_refuses_while_pending() {
        let Some((mut gpu, _guard)) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        hip_bridge::clear_vmm_faults();
        let access = [gpu.device_id];
        let chunk = 2 * 1024 * 1024;
        let tensor =
            match unsafe { gpu.alloc_vmm_tensor(&[chunk], super::DType::Raw, chunk, &access) } {
                Ok(t) => t,
                Err(_) => {
                    eprintln!("skip: VMM unavailable");
                    return;
                }
            };
        hip_bridge::inject_vmm_fault(hip_bridge::VmmFaultKind::Unmap, 2); // free + ensure attempt
        let _ = gpu.free_tensor(tensor).expect_err("fault");
        let err = gpu
            .ensure_vmm_cleaned()
            .expect_err("must refuse while pending");
        assert!(
            err.to_string().contains("pending") || err.to_string().contains("injected"),
            "{err}"
        );
        assert!(gpu.vmm_allocation_count() >= 1);
        hip_bridge::clear_vmm_faults();
        gpu.ensure_vmm_cleaned()
            .expect("clears after faults drained");
        assert_eq!(gpu.vmm_allocation_count(), 0);
    }
    #[test]
    fn scratch_convert_predicate_matches_launch_gate() {
        // The scratch fast-path `scratch_must_convert` must be exactly
        // `is_recording || capture_mode || cached != src`, which is the
        // same predicate `Gpu::launch_maybe_blob_bound` uses for
        // `record || capture_mode || force_blob` (with force_blob=false here).
        // If these drift, a helper could skip a kernel that a recorder
        // expects, or record a kernel the live path elides.
        use crate::scratch::{scratch_must_convert, use_blob_path};
        let ptr_a: *mut std::ffi::c_void = 0x1000 as *mut _;
        let ptr_b: *mut std::ffi::c_void = 0x2000 as *mut _;

        for capture in [false, true] {
            for recording in [false, true] {
                // cached == src  -> should convert only if a recorder active
                assert_eq!(
                    scratch_must_convert(capture, recording, ptr_a, ptr_a),
                    recording || capture,
                    "cached==src capture={capture} recording={recording}"
                );
                // cached != src -> always converts regardless of recorders
                assert_eq!(
                    scratch_must_convert(capture, recording, ptr_a, ptr_b),
                    true,
                    "cached!=src capture={capture} recording={recording}"
                );
                // blob path predicate must equal launch_maybe_blob_bound's gate
                for force in [false, true] {
                    let scratch_gate = use_blob_path(recording, capture, force);
                    let dispatch_gate = recording || capture || force;
                    assert_eq!(
                        scratch_gate, dispatch_gate,
                        "use_blob_path(recording={recording}, capture={capture}, force={force})"
                    );
                }
            }
        }
    }

    #[test]
    fn rotate_helpers_share_recording_gate_with_conversions() {
        // The 7 rotation helpers (`rotate_x_mq`, `rotate_x_mq_batched`,
        // `rotate_x_mq_128`, `rotate_x_mq_awq`, `rotate_x_mq_awq_batched`,
        // `rotate_x_mq_dual_fp8`, `rotate_quantize_x_mq8`) have no
        // cached-pointer elision — they always launch via `launch_maybe_blob`,
        // whose gating is `use_blob_path(is_recording, capture_mode, force)`.
        // That is the same gate `Gpu::launch_maybe_blob_bound` uses, and the
        // same gate the conversion helpers couple to via `scratch_must_convert`
        // (with force_blob=false). If a future rotate helper adds a pointer-cache
        // skip, it must reuse `scratch_must_convert`; otherwise a divergence
        // reintroduces exactly the 64-launch gap.
        //
        // Full helper invocation (allocation, kernel compile, HIP launch)
        // cannot be exercised without a device; this GPU-free test covers the
        // only driftable logic — the predicate — which is the invariant that
        // keeps `capture_blobs.len() == recorded_launches.len()`.
        use crate::scratch::{scratch_must_convert, use_blob_path};
        let ptr_a: *mut std::ffi::c_void = 0x1000 as *mut _;
        let ptr_b: *mut std::ffi::c_void = 0x2000 as *mut _;
        for capture in [false, true] {
            for recording in [false, true] {
                for force in [false, true] {
                    // Rotate helpers with no cache: must run iff blob path is taken.
                    let rotate_must_run = use_blob_path(recording, capture, force);
                    let dispatch_gate = recording || capture || force;
                    assert_eq!(
                        rotate_must_run, dispatch_gate,
                        "rotate gate mismatch recording={recording} capture={capture} force={force}"
                    );
                    // If a rotate helper ever gains a cache, it must match the
                    // conversion helper's `scratch_must_convert`.
                    assert_eq!(
                        scratch_must_convert(capture, recording, ptr_a, ptr_a),
                        recording || capture,
                        "scratch_must_convert cached==src capture={capture} recording={recording}"
                    );
                    assert_eq!(
                        scratch_must_convert(capture, recording, ptr_a, ptr_b),
                        true,
                        "scratch_must_convert cached!=src capture={capture} recording={recording}"
                    );
                    // The two predicates coincide when cached==src and force==false:
                    // both reduce to `recording || capture`. With force==true or
                    // cached!=src they are trivially true, so they cannot diverge.
                    if !force {
                        assert_eq!(
                            scratch_must_convert(capture, recording, ptr_a, ptr_a),
                            use_blob_path(recording, capture, force),
                            "conversion vs rotate gate drift capture={capture} recording={recording} force={force}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tape_parity_accessor_starts_empty() {
        // GPU-free sanity for the invariant documented on `Gpu`: a recording
        // and a HipGraph capture of the same body must produce the same launch
        // count. Only the replay side is constructible without a device, so
        // this pins the accessor and its empty state; the real invariant is
        // enforced on hardware by comparing `capture.launches` against the
        // `[verify-graph] captured ... with N blobs` line.
        use crate::replay::{ReplayBackendRequest, ReplayController};
        let ctrl = ReplayController::new(ReplayBackendRequest::Hip);
        assert_eq!(ctrl.recorded_launches().len(), 0);
    }

    #[test]
    fn deadline_error_names_last_kernel() {
        // Constructor-level pin; the timeout path itself is driven below
        // through `poll_until_ready` with a stubbed query.
        let e = Gpu::deadline_exceeded(Some("gemv_hfq4g256"), std::time::Duration::from_secs(5));
        let s = e.to_string();
        assert!(s.contains("gemv_hfq4g256"), "names the kernel: {s}");
        assert!(s.contains("5s"), "names the deadline: {s}");
        let ctx = e.context.expect("timeout carries launch context");
        assert_eq!(ctx.kernel, "gemv_hfq4g256");
    }

    #[test]
    fn deadline_error_without_kernel_says_so() {
        let e = Gpu::deadline_exceeded(None, std::time::Duration::from_secs(5));
        let s = e.to_string();
        assert!(s.contains("no kernel"), "states nothing was recorded: {s}");
        assert!(e.context.is_none());
    }

    #[test]
    fn sync_timeout_returns_in_bounded_wall_time() {
        // The test review asked for: a query that never reports ready (a
        // hung GPU) must produce `Err` within bounded wall-clock time. The
        // old scoped-thread design would block forever here on the join;
        // the event poll returns. No GPU needed — the query is stubbed.
        let deadline = std::time::Duration::from_millis(50);
        let start = std::time::Instant::now();
        let err = Gpu::poll_until_ready(deadline, Some("hung_kernel"), || Ok(false))
            .expect_err("a never-ready query must time out");
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "deadline escaped its bound: {elapsed:?}"
        );
        assert!(
            elapsed >= deadline,
            "returned before the deadline elapsed: {elapsed:?}"
        );
        let s = err.to_string();
        assert!(s.contains("hung_kernel"), "timeout names the kernel: {s}");
    }

    #[test]
    fn poll_ready_immediately_returns_ok() {
        let result = Gpu::poll_until_ready(std::time::Duration::from_secs(5), None, || Ok(true));
        assert!(result.is_ok());
    }

    #[test]
    fn poll_propagates_query_errors() {
        // A real query failure (bad handle, lost device) is not "not ready".
        let err = Gpu::poll_until_ready(std::time::Duration::from_secs(5), Some("k"), || {
            Err(hip_bridge::HipError::new(999, "boom"))
        })
        .expect_err("query errors must propagate");
        assert!(err.to_string().contains("boom"), "{err}");
    }
}
